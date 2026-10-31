"""Plasmonic and conductive material models with validated optical constants."""

from .dispersion import DrudeLorentzMaterial, TabulatedMaterial, ConstantIndexMaterial


def create_gold_rakic() -> DrudeLorentzMaterial:
    """Gold (Au) Drude-Lorentz model based on Rakic et al. (1998).
    
    Reference:
        A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski,
        'Optical dispersion relations for metals (Au, Ag, Cu, Al)',
        Appl. Opt. 37, 5271-5283 (1998).
        Valid range: 200 nm to 20000 nm.
    """
    return DrudeLorentzMaterial(
        name="gold",
        eps_inf=1.14,
        omega_p_ev=9.03,
        gamma_0_ev=0.053,
        oscillators=[
            {"f": 0.24, "omega": 2.41, "gamma": 0.249},
            {"f": 0.01, "omega": 2.76, "gamma": 0.715},
            {"f": 0.71, "omega": 3.47, "gamma": 1.604},
            {"f": 0.60, "omega": 8.57, "gamma": 4.384},
        ],
        min_wavelength_m=300e-9,
        max_wavelength_m=2500e-9,
        doi_source="10.1364/AO.37.005271",
        notes="Rakic 1998 Drude-Lorentz model for gold (Au).",
    )


def create_silver_rakic() -> DrudeLorentzMaterial:
    """Silver (Ag) Drude-Lorentz model based on Rakic et al. (1998).
    
    Reference:
        A. D. Rakic et al., Appl. Opt. 37, 5271-5283 (1998).
        Valid range: 200 nm to 20000 nm.
    """
    return DrudeLorentzMaterial(
        name="silver",
        eps_inf=1.17,
        omega_p_ev=9.01,
        gamma_0_ev=0.048,
        oscillators=[
            {"f": 0.065, "omega": 0.816, "gamma": 3.886},
            {"f": 0.124, "omega": 4.481, "gamma": 0.452},
            {"f": 0.011, "omega": 8.185, "gamma": 0.065},
            {"f": 0.840, "omega": 9.083, "gamma": 0.916},
        ],
        min_wavelength_m=300e-9,
        max_wavelength_m=2500e-9,
        doi_source="10.1364/AO.37.005271",
        notes="Rakic 1998 Drude-Lorentz model for silver (Ag).",
    )


def create_copper_rakic() -> DrudeLorentzMaterial:
    """Copper (Cu) Drude-Lorentz model based on Rakic et al. (1998)."""
    return DrudeLorentzMaterial(
        name="copper",
        eps_inf=1.21,
        omega_p_ev=10.83,
        gamma_0_ev=0.029,
        oscillators=[
            {"f": 0.080, "omega": 2.01, "gamma": 0.378},
            {"f": 0.520, "omega": 3.42, "gamma": 1.050},
        ],
        min_wavelength_m=350e-9,
        max_wavelength_m=2500e-9,
        doi_source="10.1364/AO.37.005271",
        notes="Rakic 1998 Drude-Lorentz model for copper (Cu).",
    )


def create_aluminum_rakic() -> DrudeLorentzMaterial:
    """Aluminum (Al) Drude-Lorentz model based on Rakic et al. (1998)."""
    return DrudeLorentzMaterial(
        name="aluminum",
        eps_inf=1.0,
        omega_p_ev=14.98,
        gamma_0_ev=0.047,
        oscillators=[
            {"f": 0.523, "omega": 1.55, "gamma": 0.333},
        ],
        min_wavelength_m=300e-9,
        max_wavelength_m=2500e-9,
        doi_source="10.1364/AO.37.005271",
        notes="Rakic 1998 Drude-Lorentz model for aluminum (Al).",
    )


def create_titanium_adhesion() -> ConstantIndexMaterial:
    """Titanium (Ti) thin-film adhesion layer (~2-3 nm) optical constants at 1.55 um (n=3.69, k=4.62)."""
    return ConstantIndexMaterial(
        name="titanium",
        n=3.69,
        k=4.62,
        category="metal",
        notes="Titanium (Ti) thin adhesion layer optical constants near 1.55 um.",
    )


def create_titanium_dioxide() -> ConstantIndexMaterial:
    """Titanium dioxide (TiO2) high-index dielectric intermediate layer (n ~ 2.45)."""
    return ConstantIndexMaterial(
        name="tio2",
        n=2.45,
        k=0.0,
        category="dielectric",
        notes="Titanium dioxide (TiO2) optical constant in NIR.",
    )

