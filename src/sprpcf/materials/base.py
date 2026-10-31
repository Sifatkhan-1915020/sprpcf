"""Base classes and interfaces for optical material property modeling."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from ..exceptions import MaterialError, DispersionRangeError


class OpticalConstants:
    """Complex optical constants at a specific vacuum wavelength."""

    def __init__(self, n: float, k: float = 0.0, wavelength_m: float = 1.55e-6):
        self.n = float(n)  # Real refractive index
        self.k = float(k)  # Extinction coefficient (>= 0 for absorbing media)
        self.wavelength_m = float(wavelength_m)

    @property
    def complex_n(self) -> complex:
        """Complex refractive index n_tilde = n + i*k.
        
        Under convention exp(i*(wt + beta*z)) or exp(i*beta*z) with positive attenuation.
        """
        return complex(self.n, self.k)

    @property
    def permittivity(self) -> complex:
        """Relative complex permittivity epsilon_r = (n + i*k)^2 = (n^2 - k^2) + 2*i*n*k."""
        cn = self.complex_n
        return cn * cn

    @property
    def eps_real(self) -> float:
        return self.permittivity.real

    @property
    def eps_imag(self) -> float:
        return self.permittivity.imag

    def __repr__(self) -> str:
        k_str = f" + {self.k:.4e}i" if self.k != 0 else ""
        return f"OpticalConstants(n={self.n:.4f}{k_str} at lambda={self.wavelength_m*1e9:.1f}nm)"


class Material(ABC):
    """Abstract base class for all optical materials."""

    def __init__(
        self,
        name: str,
        category: str = "dielectric",
        min_wavelength_m: float = 200e-9,
        max_wavelength_m: float = 2500e-9,
        temperature_k: float = 293.15,
        doi_source: str = "",
        notes: str = "",
    ):
        self.name = name
        self.category = category
        self.min_wavelength_m = float(min_wavelength_m)
        self.max_wavelength_m = float(max_wavelength_m)
        self.temperature_k = float(temperature_k)
        self.doi_source = doi_source
        self.notes = notes

    @abstractmethod
    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        """Calculate complex refractive index at the specified vacuum wavelength (meters)."""
        pass

    def get_refractive_index(self, wavelength_m: float, temperature_k: Optional[float] = None) -> complex:
        """Convenience method returning complex refractive index."""
        oc = self.get_optical_constants(wavelength_m, temperature_k)
        return oc.complex_n

    def get_permittivity(self, wavelength_m: float, temperature_k: Optional[float] = None) -> complex:
        """Convenience method returning complex relative permittivity."""
        oc = self.get_optical_constants(wavelength_m, temperature_k)
        return oc.permittivity

    def check_wavelength_bounds(self, wavelength_m: float) -> None:
        """Verify that queried wavelength is within documented validity range."""
        if wavelength_m < self.min_wavelength_m or wavelength_m > self.max_wavelength_m:
            raise DispersionRangeError(
                f"Material '{self.name}' queried at {wavelength_m*1e9:.2f} nm, outside valid range "
                f"[{self.min_wavelength_m*1e9:.2f} nm, {self.max_wavelength_m*1e9:.2f} nm]. Source: {self.doi_source}"
            )

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize material configuration and provenance."""
        pass

