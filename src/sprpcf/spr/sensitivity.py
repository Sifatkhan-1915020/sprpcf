"""Sensitivity metrics for wavelength and amplitude interrogation."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.stats import linregress


class SensitivityReport:
    """Quantitative evaluation of biosensor sensitivity, linearity, and polynomial dispersion."""

    def __init__(
        self,
        analyte_indices: List[float],
        resonance_wavelengths_nm: List[float],
        amplitude_sensitivities: Optional[List[float]] = None,
    ):
        self.analyte_indices = np.asarray(analyte_indices, dtype=float)
        self.resonance_wavelengths_nm = np.asarray(resonance_wavelengths_nm, dtype=float)
        self.amplitude_sensitivities = amplitude_sensitivities or []
        self.amplitude_definition = (
            "S_A(lambda, n_a) = -(1 / alpha(lambda, n_a)) * [alpha(lambda, n_a + Delta_n_a) - alpha(lambda, n_a)] / Delta_n_a  [RIU^-1]"
        )

        # Perform linear regression
        if len(self.analyte_indices) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(
                self.analyte_indices, self.resonance_wavelengths_nm
            )
            self.linear_sensitivity_nm_per_riu = float(slope)
            self.intercept = float(intercept)
            self.r_squared = float(r_value**2)
            self.std_error = float(std_err)

            # Fit line and RMSE
            fit_vals = slope * self.analyte_indices + intercept
            self.rmse_nm = float(np.sqrt(np.mean((self.resonance_wavelengths_nm - fit_vals)**2)))

            # Point-to-point incremental sensitivities: dlambda / dna
            dn = np.diff(self.analyte_indices)
            dwl = np.diff(self.resonance_wavelengths_nm)
            self.incremental_sensitivities = (dwl / np.where(dn != 0, dn, 1e-12)).tolist()
            self.max_sensitivity_nm_per_riu = float(np.max(self.incremental_sensitivities))
            self.min_sensitivity_nm_per_riu = float(np.min(self.incremental_sensitivities))

            # Quadratic polynomial fit if >= 3 analyte points
            if len(self.analyte_indices) >= 3:
                poly_coeffs = np.polyfit(self.analyte_indices, self.resonance_wavelengths_nm, 2)
                self.polynomial_coefficients = poly_coeffs.tolist()
            else:
                self.polynomial_coefficients = [0.0, float(slope), float(intercept)]
        else:
            self.linear_sensitivity_nm_per_riu = 0.0
            self.intercept = 0.0
            self.r_squared = 1.0
            self.std_error = 0.0
            self.rmse_nm = 0.0
            self.incremental_sensitivities = []
            self.max_sensitivity_nm_per_riu = 0.0
            self.min_sensitivity_nm_per_riu = 0.0
            self.polynomial_coefficients = []

    @property
    def max_amplitude_sensitivity_riu_inv(self) -> float:
        if self.amplitude_sensitivities:
            return float(np.max(self.amplitude_sensitivities))
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linear_wavelength_sensitivity_nm_per_riu": self.linear_sensitivity_nm_per_riu,
            "max_wavelength_sensitivity_nm_per_riu": self.max_sensitivity_nm_per_riu,
            "min_wavelength_sensitivity_nm_per_riu": self.min_sensitivity_nm_per_riu,
            "r_squared": self.r_squared,
            "std_error": self.std_error,
            "rmse_nm": self.rmse_nm,
            "intercept_nm": self.intercept,
            "polynomial_coefficients": self.polynomial_coefficients,
            "max_amplitude_sensitivity_riu_inv": self.max_amplitude_sensitivity_riu_inv,
            "amplitude_sensitivity_definition": self.amplitude_definition,
            "analyte_indices": self.analyte_indices.tolist(),
            "resonance_wavelengths_nm": self.resonance_wavelengths_nm.tolist(),
            "incremental_sensitivities": self.incremental_sensitivities,
        }

    def __repr__(self) -> str:
        return (
            f"SensitivityReport(S_lambda={self.linear_sensitivity_nm_per_riu:.1f} nm/RIU, "
            f"R^2={self.r_squared:.4f}, RMSE={self.rmse_nm:.2f} nm, S_A_max={self.max_amplitude_sensitivity_riu_inv:.1f} RIU^-1)"
        )


def calculate_amplitude_sensitivity_spectrum(
    wavelengths_m: np.ndarray,
    loss_curve_1: np.ndarray,
    loss_curve_2: np.ndarray,
    delta_n_analyte: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate spectral amplitude sensitivity S_A(lambda) = - (1 / alpha(n)) * [alpha(n+dn) - alpha(n)] / dn.
    
    Definition:
        S_A(\u03bb) = -\\frac{1}{\\alpha(\\lambda, n_a)} \\frac{\\partial \\alpha}{\\partial n_a}
                    \\approx -\\frac{1}{\\alpha(\\lambda, n_a)} \\frac{\\alpha(\\lambda, n_a + \\Delta n_a) - \\alpha(\\lambda, n_a)}{\\Delta n_a}

    Returns:
        wavelengths_nm: 1D array of wavelengths in nm
        s_a_spectrum: 1D array of amplitude sensitivity in RIU^-1
        max_s_a: Maximum absolute amplitude sensitivity in RIU^-1
    """
    if abs(delta_n_analyte) < 1e-12:
        raise ValueError("delta_n_analyte cannot be zero.")

    alpha1 = np.asarray(loss_curve_1, dtype=float)
    alpha2 = np.asarray(loss_curve_2, dtype=float)

    denom = np.maximum(1e-6, alpha1) * abs(delta_n_analyte)
    s_a = np.abs(alpha2 - alpha1) / denom

    max_val = float(np.max(s_a)) if len(s_a) else 0.0
    return wavelengths_m * 1e9, s_a, max_val
