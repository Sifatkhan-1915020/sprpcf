"""Loss spectra, dispersion curves, and avoided-crossing visualization."""

from typing import Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
from ..studies.base import StudyResult
from ..spr.analyzer import SPRAnalyzer
from ..spr.resonance import extract_resonance_peaks, fit_resonance_curve


def plot_loss_spectrum(
    study_result_or_analyzer: Union[StudyResult, SPRAnalyzer],
    mode_idx: int = 0,
    ax: Optional[plt.Axes] = None,
    show_fit: bool = True,
    show_raw_points: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot modal confinement loss spectrum distinguishing raw simulation samples, spline, and fitted curve."""
    if isinstance(study_result_or_analyzer, SPRAnalyzer):
        analyzer = study_result_or_analyzer
    else:
        analyzer = SPRAnalyzer(study_result_or_analyzer.results)

    spectrum = analyzer.extract_loss_spectrum(mode_idx)
    wls_nm = spectrum.wavelengths_m * 1e9
    loss_db_cm = spectrum.loss_db_per_cm

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    else:
        fig = ax.figure

    # 1. Discrete Raw FEM Simulation Points
    if show_raw_points:
        ax.plot(
            wls_nm,
            loss_db_cm,
            "s",
            color="#08519c",
            markersize=6,
            label="Raw FEM Samples",
            zorder=8,
        )

    # 2. Spline interpolation line
    spline_wl = np.linspace(wls_nm[0], wls_nm[-1], 200)
    from scipy.interpolate import interp1d
    spline_fn = interp1d(wls_nm, loss_db_cm, kind="cubic", fill_value="extrapolate")
    spline_loss = spline_fn(spline_wl)
    ax.plot(spline_wl, spline_loss, ":", color="#737373", linewidth=1.2, label="Cubic Spline", zorder=4)

    # 3. Nonlinear Analytical Resonance Fit
    peaks = extract_resonance_peaks(spectrum.wavelengths_m, spectrum.loss_db_per_cm)
    if peaks:
        p = peaks[0]
        fit = p.fit_result

        if show_fit and fit and fit.r_squared > 0.75:
            fit_dense_loss = fit.eval_fit(spline_wl)
            ax.plot(
                spline_wl,
                fit_dense_loss,
                "-",
                color="#e41a1c",
                linewidth=2.0,
                label=f"{fit.model_name.capitalize()} Fit (R^2={fit.r_squared:.3f})",
                zorder=6,
            )

            # Resonance center vertical line
            ax.axvline(fit.wavelength_nm, color="#e41a1c", linestyle="--", linewidth=1.2, alpha=0.7)

            # Mark FWHM linewidth span
            half_val = fit.peak_loss_db_cm - 0.5 * fit.prominence_db_cm
            fwhm_left = fit.wavelength_nm - 0.5 * fit.fwhm_nm
            fwhm_right = fit.wavelength_nm + 0.5 * fit.fwhm_nm
            ax.hlines(
                half_val,
                fwhm_left,
                fwhm_right,
                color="#2ca25f",
                linestyle="-",
                linewidth=2.5,
                label=f"FWHM: {fit.fwhm_nm:.1f}\u00b1{fit.fwhm_err_nm:.1f} nm",
                zorder=10,
            )

            # Sampling adequacy banner
            if fit.is_adequately_sampled:
                status_txt = f"\u2713 Sampling Validated ({fit.points_in_fwhm} pts in FWHM)"
                box_color = "#e5f5e0"
                edge_color = "#31a354"
            else:
                status_txt = f"! Undersampled ({fit.points_in_fwhm} pts in FWHM, min 5 req)"
                box_color = "#fff7bc"
                edge_color = "#d95f0e"

            info_txt = (
                f"\u03bb_res = {fit.wavelength_nm:.1f}\u00b1{fit.wavelength_err_nm:.1f} nm\n"
                f"Peak Loss = {fit.peak_loss_db_cm:.1f} dB/cm\n"
                f"FWHM = {fit.fwhm_nm:.1f}\u00b1{fit.fwhm_err_nm:.1f} nm\n"
                f"Q-Factor = {fit.q_factor:.1f}\n"
                f"{status_txt}"
            )
            ax.text(
                0.03,
                0.97,
                info_txt,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor=edge_color, alpha=0.9),
                zorder=15,
            )
        else:
            # Fallback peak marker from spline
            ax.plot(p.wavelength_nm, p.peak_loss_db_cm, "ro", markersize=8, label=f"Resonance: {p.wavelength_nm:.1f} nm")

    ax.set_xlabel("Wavelength (nm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Confinement Loss (dB/cm)", fontsize=10, fontweight="bold")
    ax.set_title("SPR Confinement Loss Spectrum", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_dispersion_curves(
    study_result: StudyResult,
    num_modes_to_plot: int = 4,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot real effective refractive index curves Re(n_eff) vs wavelength with avoided crossings."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    else:
        fig = ax.figure

    wls_nm = [r.wavelength_m * 1e9 for r in study_result.results]

    mode_styles = ["b.-", "r.--", "g.-.", "m:", "c-", "y-"]

    for m_i in range(num_modes_to_plot):
        neff_reals = []
        for r in study_result.results:
            if m_i < len(r.modes):
                neff_reals.append(r.modes[m_i].n_real)
            else:
                neff_reals.append(np.nan)

        style = mode_styles[m_i % len(mode_styles)]
        ax.plot(wls_nm, neff_reals, style, linewidth=1.6, markersize=5, label=f"Mode {m_i}")

    # Annotate avoided crossings if present in metadata
    events = study_result.metadata.get("avoided_crossings", [])
    for ev in events:
        px = ev["param_val"] * 1e9 if ev["param_val"] < 1e-3 else ev["param_val"]
        ax.axvline(px, color="#756bb1", linestyle="--", alpha=0.5)
        ax.text(
            px,
            ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"Anti-crossing\n(\u0394n_min={ev['min_index_gap']:.3f})",
            fontsize=7,
            color="#54278f",
            rotation=90,
            ha="right",
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Real Effective Index Re(n_eff)", fontsize=10, fontweight="bold")
    ax.set_title("Modal Dispersion Curves & Phase Matching", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
