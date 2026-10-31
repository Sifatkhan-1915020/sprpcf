"""Biological analytes, aqueous solutions, and functional receptor layer models."""

from typing import Optional, Dict, Any
from .base import Material, OpticalConstants
from .dispersion import ConstantIndexMaterial


class WaterMaterial(Material):
    """Pure liquid water dispersion model (Hale & Querry 1973 with temperature adjustment).
    
    Reference:
        G. M. Hale and M. R. Querry, 'Optical Constants of Water in the 200-nm to 200-um Wavelength Region',
        Appl. Opt. 12, 555-563 (1973).
    """

    def __init__(self, temperature_k: float = 293.15):
        super().__init__(
            name="water",
            category="biological",
            min_wavelength_m=350e-9,
            max_wavelength_m=2000e-9,
            temperature_k=temperature_k,
            doi_source="10.1364/AO.12.000555",
            notes="Liquid water optical dispersion at room temperature.",
        )
        # Thermo-optic coefficient of water ~ -1.0e-4 / K
        self.dn_dt = -1.0e-4

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        self.check_wavelength_bounds(wavelength_m)
        lam_um = wavelength_m * 1e6
        # 4-term empirical Sellmeier fit for water around room temperature
        n_base = 1.31279 + 15.762 / (lam_um**2 + 3382.4) + 0.00561 / (lam_um**2 - 0.0168)
        k = 1.0e-5 if lam_um < 1.0 else 1.3e-4  # negligible loss in visible, slight absorption in NIR

        if temperature_k is not None:
            n_base += self.dn_dt * (temperature_k - self.temperature_k)

        return OpticalConstants(n=n_base, k=k, wavelength_m=wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "water",
            "name": self.name,
            "temperature_k": self.temperature_k,
            "doi_source": self.doi_source,
        }


class GlucoseSolutionMaterial(Material):
    """Aqueous D-glucose solution with refractive index dependent on concentration (g/L or wt%).
    
    dn/dC ~ 0.000143 RIU per g/L (~0.0014 RIU per wt%).
    Reference:
        J. S. Yeh et al., Biosensors and Bioelectronics (2003).
    """

    def __init__(self, concentration_g_per_l: float = 0.0, temperature_k: float = 293.15):
        super().__init__(
            name=f"glucose_{concentration_g_per_l}gL",
            category="biological",
            min_wavelength_m=400e-9,
            max_wavelength_m=1800e-9,
            temperature_k=temperature_k,
            doi_source="10.1016/S0956-5663(03)00125-9",
            notes=f"Glucose solution at {concentration_g_per_l} g/L.",
        )
        self.concentration_g_per_l = float(concentration_g_per_l)
        self._water = WaterMaterial(temperature_k)

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        w_oc = self._water.get_optical_constants(wavelength_m, temperature_k)
        dn = 0.000143 * self.concentration_g_per_l
        return OpticalConstants(n=w_oc.n + dn, k=w_oc.k, wavelength_m=wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "glucose_solution",
            "name": self.name,
            "concentration_g_per_l": self.concentration_g_per_l,
            "doi_source": self.doi_source,
        }


class AnalyteMaterial(Material):
    """General parametric analyte material with controllable base refractive index."""

    def __init__(self, name: str = "analyte", refractive_index: float = 1.330, notes: str = "Generic liquid analyte"):
        super().__init__(name=name, category="biological", min_wavelength_m=200e-9, max_wavelength_m=3000e-9, notes=notes)
        self.refractive_index = float(refractive_index)

    def set_index(self, n: float) -> None:
        self.refractive_index = float(n)

    def get_optical_constants(self, wavelength_m: float, temperature_k: Optional[float] = None) -> OpticalConstants:
        return OpticalConstants(n=self.refractive_index, k=0.0, wavelength_m=wavelength_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "analyte",
            "name": self.name,
            "refractive_index": self.refractive_index,
            "notes": self.notes,
        }


def create_bioreceptor_layer(name: str = "bioreceptor", thickness_nm: float = 5.0, index: float = 1.45) -> ConstantIndexMaterial:
    """Protein / antibody immobilized monolayer (e.g. BSA or IgG, typical n ~ 1.45, 2-10 nm thickness)."""
    return ConstantIndexMaterial(
        name=name,
        n=index,
        k=0.0,
        category="biological",
        notes=f"Immobilized bioreceptor monolayer (~{thickness_nm} nm, n={index}).",
    )

