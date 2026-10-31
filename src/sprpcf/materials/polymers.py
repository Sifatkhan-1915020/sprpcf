"""Polymer material models for optical fibers and microfluidics."""

from .dispersion import CauchyMaterial, ConstantIndexMaterial


def create_pmma() -> CauchyMaterial:
    """Poly(methyl methacrylate) (PMMA) optical polymer using Cauchy model.
    
    Reference:
        N. Sultanova et al., 'Dispersion Properties of Optical Polymers',
        Acta Physica Polonica A 116, 585-587 (2009).
    """
    return CauchyMaterial(
        name="pmma",
        a=1.4800,
        b=0.00511,  # um^2
        c=0.00015,  # um^4
        min_wavelength_m=400e-9,
        max_wavelength_m=1600e-9,
        category="polymer",
        doi_source="10.12693/APhysPolA.116.585",
        notes="Sultanova 2009 Cauchy dispersion for PMMA.",
    )


def create_topas() -> CauchyMaterial:
    """TOPAS (Cyclic Olefin Copolymer) transparent polymer with low loss in THz/IR."""
    return CauchyMaterial(
        name="topas",
        a=1.5255,
        b=0.00490,
        c=0.00012,
        min_wavelength_m=400e-9,
        max_wavelength_m=1700e-9,
        category="polymer",
        doi_source="10.1364/AO.52.001923",
        notes="TOPAS COC polymer optical constants.",
    )


def create_pdms() -> ConstantIndexMaterial:
    """Polydimethylsiloxane (PDMS) elastomer commonly used in microfluidics (n ~ 1.406)."""
    return ConstantIndexMaterial(
        name="pdms",
        n=1.406,
        k=0.0,
        category="polymer",
        notes="PDMS standard cured optical index at room temperature.",
    )

