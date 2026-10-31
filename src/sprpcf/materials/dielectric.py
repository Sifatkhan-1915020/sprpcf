"""Dielectric and glass optical material models."""

from .dispersion import SellmeierMaterial, ConstantIndexMaterial


def create_fused_silica() -> SellmeierMaterial:
    """Create standard pure fused silica material using Malitson (1965) 3-term Sellmeier model.
    
    Reference:
        I. H. Malitson, 'Interspecimen Comparison of the Refractive Index of Fused Silica',
        J. Opt. Soc. Am. 55, 1205-1209 (1965).
        Valid range: 0.21 um to 3.71 um (210 nm to 3710 nm).
    """
    # B coefficients (dimensionless)
    b_coeffs = [0.6961663, 0.4079426, 0.8974794]
    # C coefficients (um^2)
    c_coeffs = [0.0684043**2, 0.1162414**2, 9.896161**2]

    return SellmeierMaterial(
        name="silica",
        b_coeffs=b_coeffs,
        c_coeffs=c_coeffs,
        min_wavelength_m=210e-9,
        max_wavelength_m=3700e-9,
        temperature_k=293.15,
        dn_dt=1.28e-5,  # 1/K
        doi_source="10.1364/JOSA.55.001205",
        notes="Malitson 1965 3-term Sellmeier model for pure fused silica.",
    )


def create_borosilicate_glass() -> SellmeierMaterial:
    """Create BK7 / Borosilicate glass using standard Schott Sellmeier dispersion."""
    b_coeffs = [1.03961212, 0.231792344, 1.01046945]
    c_coeffs = [0.00600069867, 0.0200179144, 103.560653]
    return SellmeierMaterial(
        name="bk7",
        b_coeffs=b_coeffs,
        c_coeffs=c_coeffs,
        min_wavelength_m=300e-9,
        max_wavelength_m=2500e-9,
        doi_source="Schott Optical Glass Catalog (2018)",
        notes="Schott N-BK7 borosilicate optical glass.",
    )


def create_air() -> ConstantIndexMaterial:
    """Standard dry air at STP (n = 1.000277 ~ 1.0)."""
    return ConstantIndexMaterial(name="air", n=1.0, k=0.0, category="dielectric", notes="Standard air (n=1.0)")

