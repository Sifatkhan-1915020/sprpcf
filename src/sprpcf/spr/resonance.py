"""Resonance peak extraction, nonlinear curve fitting, and sampling adequacy validation."""

import math
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from ..exceptions import ResonanceDetectionError


def _lorentzian(wl: np.ndarray, y0: float, A: float, wl0: float, gamma: float, m: float) -> np.ndarray:
    """Lorentzian resonance profile with linear background baseline."""
    return y0 + (2.0 * A / math.pi) * (gamma / (4.0 * (wl - wl0)**2 + gamma**2)) + m * (wl - wl0)


def _fano(wl: np.ndarray, y0: float, A: float, wl0: float, gamma: float, q: float) -> np.ndarray:
    """Asymmetric Fano resonance profile."""
    eps = 2.0 * (wl - wl0) / np.maximum(1e-12, gamma)
    return y0 + A * ((q + eps)**2) / (1.0 + eps**2)


class ResonanceFitResult:
    """Detailed nonlinear least-squares resonance fit result with physical uncertainties."""

    def __init__(
        self,
        wavelength_nm: float,
        wavelength_err_nm: float,
        peak_loss_db_cm: float,
        peak_loss_err_db_cm: float,
        fwhm_nm: float,
        fwhm_err_nm: float,
        prominence_db_cm: float,
        r_squared: float,
        points_in_fwhm: int,
        is_adequately_sampled: bool,
        model_name: str,
        fit_params: np.ndarray,
        warning: Optional[str] = None,
    ):
        self.wavelength_nm = float(wavelength_nm)
        self.wavelength_err_nm = float(wavelength_err_nm)
        self.peak_loss_db_cm = float(peak_loss_db_cm)
        self.peak_loss_err_db_cm = float(peak_loss_err_db_cm)
        self.fwhm_nm = float(fwhm_nm)
        self.fwhm_err_nm = float(fwhm_err_nm)
        self.prominence_db_cm = float(prominence_db_cm)
        self.r_squared = float(r_squared)
        self.points_in_fwhm = int(points_in_fwhm)
        self.is_adequately_sampled = bool(is_adequately_sampled)
        self.model_name = model_name
        self.fit_params = fit_params
        self.warning = warning

    @property
    def q_factor(self) -> float:
        return self.wavelength_nm / max(1e-9, self.fwhm_nm)

    def eval_fit(self, wl_nm: np.ndarray) -> np.ndarray:
        """Evaluate the fitted continuous model across given wavelengths (in nm)."""
        if self.model_name.lower() == "lorentzian":
            return _lorentzian(wl_nm, *self.fit_params)
        elif self.model_name.lower() == "fano":
            return _fano(wl_nm, *self.fit_params)
        return np.zeros_like(wl_nm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resonance_wavelength_nm": self.wavelength_nm,
            "resonance_wavelength_err_nm": self.wavelength_err_nm,
            "peak_loss_db_cm": self.peak_loss_db_cm,
            "peak_loss_err_db_cm": self.peak_loss_err_db_cm,
            "fwhm_nm": self.fwhm_nm,
            "fwhm_err_nm": self.fwhm_err_nm,
            "q_factor": self.q_factor,
            "prominence_db_cm": self.prominence_db_cm,
            "r_squared": self.r_squared,
            "points_in_fwhm": self.points_in_fwhm,
            "is_adequately_sampled": self.is_adequately_sampled,
            "model_name": self.model_name,
            "warning": self.warning,
        }

    def __repr__(self) -> str:
        status = "Adequate" if self.is_adequately_sampled else "INADEQUATE"
        return (
            f"ResonanceFitResult(lambda={self.wavelength_nm:.2f}\u00b1{self.wavelength_err_nm:.2f} nm, "
            f"FWHM={self.fwhm_nm:.2f}\u00b1{self.fwhm_err_nm:.2f} nm, R^2={self.r_squared:.4f}, "
            f"pts_in_fwhm={self.points_in_fwhm} [{status}])"
        )


class ResonancePeak:
    """Represents a detected SPR loss resonance peak."""

    def __init__(
        self,
        wavelength_m: float,
        peak_loss_db_cm: float,
        fwhm_m: float,
        prominence_db_cm: float = 0.0,
        q_factor: Optional[float] = None,
        interpolated: bool = True,
        fit_result: Optional[ResonanceFitResult] = None,
    ):
        self.wavelength_m = float(wavelength_m)
        self.peak_loss_db_cm = float(peak_loss_db_cm)
        self.fwhm_m = float(fwhm_m)
        self.prominence_db_cm = float(prominence_db_cm)
        self.q_factor = float(q_factor) if q_factor is not None else (self.wavelength_m / max(1e-15, self.fwhm_m))
        self.interpolated = interpolated
        self.fit_result = fit_result

    @property
    def wavelength_nm(self) -> float:
        return self.wavelength_m * 1e9

    @property
    def fwhm_nm(self) -> float:
        return self.fwhm_m * 1e9

    @property
    def is_adequately_sampled(self) -> bool:
        if self.fit_result:
            return self.fit_result.is_adequately_sampled
        return False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "resonance_wavelength_nm": self.wavelength_nm,
            "peak_loss_db_cm": self.peak_loss_db_cm,
            "fwhm_nm": self.fwhm_nm,
            "q_factor": self.q_factor,
            "prominence_db_cm": self.prominence_db_cm,
            "is_adequately_sampled": self.is_adequately_sampled,
        }
        if self.fit_result:
            d["fit_result"] = self.fit_result.to_dict()
        return d

    def __repr__(self) -> str:
        return (
            f"ResonancePeak(lambda_res={self.wavelength_nm:.2f} nm, "
            f"loss={self.peak_loss_db_cm:.2f} dB/cm, FWHM={self.fwhm_nm:.2f} nm, Q={self.q_factor:.1f})"
        )


def fit_resonance_curve(
    wavelengths_nm: np.ndarray,
    loss_db_cm: np.ndarray,
    model: str = "lorentzian",
) -> Optional[ResonanceFitResult]:
    """Fit a nonlinear analytical line shape (Lorentzian or Fano) to discrete SPR simulation loss data."""
    wl = np.asarray(wavelengths_nm, dtype=float)
    loss = np.asarray(loss_db_cm, dtype=float)

    if len(wl) < 4:
        return None

    max_idx = int(np.argmax(loss))
    wl0_est = wl[max_idx]
    y0_est = float(np.min(loss))
    amp_est = float((loss[max_idx] - y0_est) * math.pi * 50.0 / 2.0)
    gamma_est = 100.0  # initial FWHM guess in nm

    if model.lower() == "lorentzian":
        p0 = [y0_est, amp_est, wl0_est, gamma_est, 0.0]
        bounds = (
            [0.0, 0.0, wl[0], 5.0, -100.0],
            [np.inf, np.inf, wl[-1], (wl[-1] - wl[0]) * 1.5, 100.0],
        )
        try:
            popt, pcov = curve_fit(_lorentzian, wl, loss, p0=p0, bounds=bounds, maxfev=4000)
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(5)
            y0, A, wl0, gamma, m = popt
            fitted_loss = _lorentzian(wl, *popt)
        except Exception:
            return None
    elif model.lower() == "fano":
        p0 = [y0_est, float(loss[max_idx] - y0_est), wl0_est, gamma_est, 1.0]
        bounds = (
            [0.0, 0.0, wl[0], 5.0, -20.0],
            [np.inf, np.inf, wl[-1], (wl[-1] - wl[0]) * 1.5, 20.0],
        )
        try:
            popt, pcov = curve_fit(_fano, wl, loss, p0=p0, bounds=bounds, maxfev=4000)
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(5)
            y0, A, wl0, gamma, q = popt
            fitted_loss = _fano(wl, *popt)
        except Exception:
            return None
    else:
        raise ValueError(f"Unsupported resonance model: {model}")

    # Compute R^2
    ss_res = np.sum((loss - fitted_loss)**2)
    ss_tot = np.sum((loss - np.mean(loss))**2)
    r2 = 1.0 - (ss_res / max(1e-12, ss_tot))

    # Evaluate discrete points inside FWHM
    half_width = gamma / 2.0
    in_fwhm_mask = (wl >= wl0 - half_width) & (wl <= wl0 + half_width)
    pts_in_fwhm = int(np.sum(in_fwhm_mask))

    # Strict sampling adequacy rule: require >= 5 discrete simulation points in the FWHM linewidth
    is_adequate = pts_in_fwhm >= 5
    warning = None
    if not is_adequate:
        warning = (
            f"Linewidth undersampled: only {pts_in_fwhm} discrete simulation point(s) inside FWHM "
            f"({wl0-half_width:.1f} to {wl0+half_width:.1f} nm). Minimum 5 points required for validated FWHM."
        )

    peak_loss = float(np.max(fitted_loss))
    prominence = peak_loss - y0

    return ResonanceFitResult(
        wavelength_nm=wl0,
        wavelength_err_nm=perr[2],
        peak_loss_db_cm=peak_loss,
        peak_loss_err_db_cm=perr[0] + perr[1],
        fwhm_nm=gamma,
        fwhm_err_nm=perr[3],
        prominence_db_cm=prominence,
        r_squared=r2,
        points_in_fwhm=pts_in_fwhm,
        is_adequately_sampled=is_adequate,
        model_name=model,
        fit_params=popt,
        warning=warning,
    )


def extract_resonance_peaks(
    wavelengths_m: np.ndarray,
    loss_db_cm: np.ndarray,
    min_prominence_db_cm: float = 0.5,
    interpolate: bool = True,
    fit_model: str = "lorentzian",
) -> List[ResonancePeak]:
    """Identify SPR resonance loss peaks using prominence-based filtering, spline interpolation, and analytical fitting."""
    wl = np.asarray(wavelengths_m, dtype=float)
    loss = np.asarray(loss_db_cm, dtype=float)

    if len(wl) < 3:
        raise ResonanceDetectionError("Need at least 3 spectral points to detect resonance.")

    # Find raw discrete peaks
    peaks_indices, properties = find_peaks(loss, prominence=min_prominence_db_cm)

    # If no prominent peak found, test if global maximum qualifies
    if len(peaks_indices) == 0:
        max_idx = int(np.argmax(loss))
        if 0 < max_idx < len(loss) - 1:
            peaks_indices = np.array([max_idx])
            properties = {"prominences": np.array([loss[max_idx] - min(loss[0], loss[-1])])}
        else:
            return []

    results: List[ResonancePeak] = []

    # High-resolution spline for continuous peak and FWHM extraction
    if interpolate and len(wl) >= 4:
        spline_wl = np.linspace(wl[0], wl[-1], max(1000, len(wl) * 100))
        spline_func = interp1d(wl, loss, kind="cubic", fill_value="extrapolate")
        spline_loss = spline_func(spline_wl)
    else:
        spline_wl = wl
        spline_loss = loss

    for p_idx in peaks_indices:
        raw_peak_wl = wl[p_idx]
        prom = float(properties["prominences"][np.where(peaks_indices == p_idx)[0][0]])

        # Continuous sub-grid peak extraction using bounded scalar optimization on spline
        if interpolate and len(wl) >= 4:
            from scipy.optimize import minimize_scalar
            w_left = float(wl[max(0, p_idx - 1)])
            w_right = float(wl[min(len(wl) - 1, p_idx + 1)])
            opt = minimize_scalar(lambda w: -float(spline_func(w)), bounds=(w_left, w_right), method="bounded")
            if opt.success:
                res_wl = float(opt.x)
                peak_val = float(spline_func(res_wl))
            else:
                res_wl = float(raw_peak_wl)
                peak_val = float(loss[p_idx])
        else:
            res_wl = float(raw_peak_wl)
            peak_val = float(loss[p_idx])

        # Compute Full Width at Half Maximum (FWHM)
        half_max = peak_val - 0.5 * prom
        crossing_indices = np.where(np.diff(np.sign(spline_loss - half_max)))[0]
        if len(crossing_indices) >= 2:
            left_wls = spline_wl[crossing_indices[crossing_indices < np.argmax(spline_loss)]]
            right_wls = spline_wl[crossing_indices[crossing_indices > np.argmax(spline_loss)]]
            wl_left = left_wls[-1] if len(left_wls) > 0 else spline_wl[0]
            wl_right = right_wls[0] if len(right_wls) > 0 else spline_wl[-1]
            fwhm_m = max(1e-12, abs(wl_right - wl_left))
        else:
            widths, _, _, _ = peak_widths(loss, [p_idx], rel_height=0.5)
            fwhm_pts = widths[0]
            d_lambda = (wl[-1] - wl[0]) / max(1, len(wl) - 1)
            fwhm_m = max(1e-12, fwhm_pts * d_lambda)

        # Perform localized analytical model fitting around the identified peak
        fit_win = max(fwhm_m * 2.5, 40e-9)
        local_mask = (wl >= raw_peak_wl - fit_win) & (wl <= raw_peak_wl + fit_win)
        if np.sum(local_mask) >= 5:
            fit_res = fit_resonance_curve(wl[local_mask] * 1e9, loss[local_mask], model=fit_model)
        else:
            fit_res = fit_resonance_curve(wl * 1e9, loss, model=fit_model)

        # If we have a successful analytical fit with R^2 > 0.8 located within the peak linewidth, prefer fitted parameters
        if fit_res and fit_res.r_squared > 0.8 and abs(fit_res.wavelength_nm * 1e-9 - raw_peak_wl) <= max(fwhm_m, 20e-9):
            res_wl = fit_res.wavelength_nm * 1e-9
            fwhm_m = fit_res.fwhm_nm * 1e-9
            peak_val = fit_res.peak_loss_db_cm
            prom = fit_res.prominence_db_cm

        res_peak = ResonancePeak(
            wavelength_m=res_wl,
            peak_loss_db_cm=peak_val,
            fwhm_m=fwhm_m,
            prominence_db_cm=prom,
            fit_result=fit_res,
        )
        results.append(res_peak)

    return results
