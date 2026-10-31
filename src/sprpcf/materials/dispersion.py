"""Analytical and empirical dispersion models for optical materials."""

import math
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from scipy.interpolate import interp1d
from .base import Material, OpticalConstants
from ..core.constants import C0
from ..exceptions import DispersionRangeError


class ConstantIndexMaterial(Material):
    """Material with non-dispersive constant refractive index."""

    def __init__(
        self,
        name: str,
        n: float = 1.0,
        k: float = 0.0,
        category: str = "dielectric",
        notes: str = "Constant non-dispersive index",
    ):
        super().__init__(name, category=category, min_wavelength_m=100e-9, max_wavelength_m=10000e-9, notes=notes)
        self.n = float(n)
        self.k = float(k)

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        return OpticalConstants(self.n, self.k, wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "constant",
            "name": self.name,
            "n": self.n,
            "k": self.k,
            "category": self.category,
        }


class SellmeierMaterial(Material):
    """Sellmeier dispersion model for optical glasses (e.g. silica).
    
    Formula:
        n^2(lambda) = 1 + sum_i (B_i * lambda^2) / (lambda^2 - C_i)
    where lambda is in micrometers (um), and C_i are squared resonant wavelengths in um^2.
    """

    def __init__(
        self,
        name: str,
        b_coeffs: List[float],
        c_coeffs: List[float],
        min_wavelength_m: float = 210e-9,
        max_wavelength_m: float = 3700e-9,
        temperature_k: float = 293.15,
        dn_dt: float = 1.28e-5,  # Thermo-optic coefficient (1/K)
        doi_source: str = "",
        notes: str = "",
    ):
        super().__init__(
            name,
            category="glass",
            min_wavelength_m=min_wavelength_m,
            max_wavelength_m=max_wavelength_m,
            temperature_k=temperature_k,
            doi_source=doi_source,
            notes=notes,
        )
        if len(b_coeffs) != len(c_coeffs):
            raise ValueError("b_coeffs and c_coeffs must have equal length.")
        self.b_coeffs = [float(b) for b in b_coeffs]
        self.c_coeffs = [float(c) for c in c_coeffs]
        self.dn_dt = float(dn_dt)

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        self.check_wavelength_bounds(wavelength_m)
        lam_um = wavelength_m * 1e6
        lam_sq = lam_um * lam_um

        n_sq = 1.0
        for b, c in zip(self.b_coeffs, self.c_coeffs):
            denom = lam_sq - c
            if abs(denom) < 1e-12:
                denom = 1e-12
            n_sq += (b * lam_sq) / denom

        if n_sq <= 0:
            raise DispersionRangeError(f"Sellmeier evaluated to negative n^2: {n_sq} at {lam_um} um")

        n = math.sqrt(n_sq)
        if temperature_k is not None:
            delta_t = temperature_k - self.temperature_k
            n += self.dn_dt * delta_t

        return OpticalConstants(n=n, k=0.0, wavelength_m=wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "sellmeier",
            "name": self.name,
            "b_coeffs": self.b_coeffs,
            "c_coeffs": self.c_coeffs,
            "min_wavelength_m": self.min_wavelength_m,
            "max_wavelength_m": self.max_wavelength_m,
            "temperature_k": self.temperature_k,
            "dn_dt": self.dn_dt,
            "doi_source": self.doi_source,
            "notes": self.notes,
        }


class CauchyMaterial(Material):
    """Cauchy dispersion model for transparent dielectrics and optical polymers.
    
    Formula:
        n(lambda) = A + B / lambda^2 + C / lambda^4
    where lambda is in micrometers (um).
    """

    def __init__(
        self,
        name: str,
        a: float,
        b: float,
        c: float = 0.0,
        min_wavelength_m: float = 350e-9,
        max_wavelength_m: float = 1800e-9,
        category: str = "polymer",
        doi_source: str = "",
        notes: str = "",
    ):
        super().__init__(
            name,
            category=category,
            min_wavelength_m=min_wavelength_m,
            max_wavelength_m=max_wavelength_m,
            doi_source=doi_source,
            notes=notes,
        )
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        self.check_wavelength_bounds(wavelength_m)
        lam_um = wavelength_m * 1e6
        n = self.a + self.b / (lam_um**2) + self.c / (lam_um**4)
        return OpticalConstants(n=n, k=0.0, wavelength_m=wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "cauchy",
            "name": self.name,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "min_wavelength_m": self.min_wavelength_m,
            "max_wavelength_m": self.max_wavelength_m,
            "category": self.category,
            "doi_source": self.doi_source,
        }


class DrudeLorentzMaterial(Material):
    """Drude-Lorentz multi-oscillator model for plasmonic metals.
    
    Formula:
        epsilon(omega) = eps_inf - (omega_p^2 / (omega * (omega - i * gamma_0)))
                         + sum_j (f_j * omega_p^2 / (omega_j^2 - omega^2 + i * omega * Gamma_j))
    
    References:
        - A. D. Rakic et al., Appl. Opt. 37, 5271-5283 (1998).
        - P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972).
    """

    def __init__(
        self,
        name: str,
        eps_inf: float,
        omega_p_ev: float,  # Plasma frequency in eV
        gamma_0_ev: float,  # Intraband damping in eV
        oscillators: Optional[List[Dict[str, float]]] = None,  # List of {'f': ..., 'omega': ..., 'gamma': ...} in eV
        min_wavelength_m: float = 300e-9,
        max_wavelength_m: float = 2500e-9,
        doi_source: str = "",
        notes: str = "",
    ):
        super().__init__(
            name,
            category="metal",
            min_wavelength_m=min_wavelength_m,
            max_wavelength_m=max_wavelength_m,
            doi_source=doi_source,
            notes=notes,
        )
        self.eps_inf = float(eps_inf)
        self.omega_p_ev = float(omega_p_ev)
        self.gamma_0_ev = float(gamma_0_ev)
        self.oscillators = oscillators or []

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        self.check_wavelength_bounds(wavelength_m)
        from ..core.units import photon_energy_ev
        # Photon energy in eV: hbar * omega
        energy_ev = photon_energy_ev(wavelength_m)

        # Drude intraband term
        # eps_drude = - omega_p^2 / (E * (E - i * gamma_0))
        denom_drude = energy_ev * complex(energy_ev, -self.gamma_0_ev)
        eps = complex(self.eps_inf, 0.0) - (self.omega_p_ev**2) / denom_drude

        # Interband Lorentz oscillators
        for osc in self.oscillators:
            f_j = osc.get("f", 1.0)
            w_j = osc.get("omega", 1.0)
            g_j = osc.get("gamma", 0.1)
            # denom = w_j^2 - E^2 + i * E * g_j
            denom_lor = complex(w_j**2 - energy_ev**2, energy_ev * g_j)
            eps += (f_j * (self.omega_p_ev**2)) / denom_lor

        # Convert relative permittivity epsilon to complex refractive index n_tilde = sqrt(epsilon)
        # In time-harmonic convention, k >= 0 (loss)
        import cmath
        n_tilde = cmath.sqrt(eps)
        # Ensure positive real and imaginary parts
        n_real = abs(n_tilde.real)
        k_imag = abs(n_tilde.imag)

        return OpticalConstants(n=n_real, k=k_imag, wavelength_m=wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "drude_lorentz",
            "name": self.name,
            "eps_inf": self.eps_inf,
            "omega_p_ev": self.omega_p_ev,
            "gamma_0_ev": self.gamma_0_ev,
            "oscillators": self.oscillators,
            "min_wavelength_m": self.min_wavelength_m,
            "max_wavelength_m": self.max_wavelength_m,
            "doi_source": self.doi_source,
        }


class TabulatedMaterial(Material):
    """Interpolates experimental (n, k) data points across wavelength."""

    def __init__(
        self,
        name: str,
        wavelengths_m: List[float],
        n_values: List[float],
        k_values: Optional[List[float]] = None,
        category: str = "metal",
        doi_source: str = "",
        notes: str = "",
        interpolation_kind: str = "cubic",
    ):
        min_wl = min(wavelengths_m)
        max_wl = max(wavelengths_m)
        super().__init__(
            name,
            category=category,
            min_wavelength_m=min_wl,
            max_wavelength_m=max_wl,
            doi_source=doi_source,
            notes=notes,
        )
        self.wavelengths_m = np.asarray(wavelengths_m, dtype=float)
        self.n_values = np.asarray(n_values, dtype=float)
        if k_values is None:
            self.k_values = np.zeros_like(self.n_values)
        else:
            self.k_values = np.asarray(k_values, dtype=float)

        # Sort by wavelength
        sort_idx = np.argsort(self.wavelengths_m)
        self.wavelengths_m = self.wavelengths_m[sort_idx]
        self.n_values = self.n_values[sort_idx]
        self.k_values = self.k_values[sort_idx]

        kind = interpolation_kind if len(self.wavelengths_m) >= 4 else "linear"
        self._interp_n = interp1d(self.wavelengths_m, self.n_values, kind=kind, bounds_error=False, fill_value="extrapolate")
        self._interp_k = interp1d(self.wavelengths_m, self.k_values, kind=kind, bounds_error=False, fill_value="extrapolate")

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        self.check_wavelength_bounds(wavelength_m)
        n = float(self._interp_n(wavelength_m))
        k = float(self._interp_k(wavelength_m))
        return OpticalConstants(n=max(0.0, n), k=max(0.0, k), wavelength_m=wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "tabulated",
            "name": self.name,
            "wavelengths_m": self.wavelengths_m.tolist(),
            "n_values": self.n_values.tolist(),
            "k_values": self.k_values.tolist(),
            "category": self.category,
            "doi_source": self.doi_source,
        }

