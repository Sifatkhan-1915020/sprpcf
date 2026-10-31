"""Materials and optical dispersion module for SPR-PCF Studio."""

from .base import Material, OpticalConstants
from .dispersion import (
    ConstantIndexMaterial,
    SellmeierMaterial,
    CauchyMaterial,
    DrudeLorentzMaterial,
    TabulatedMaterial,
)
from .dielectric import create_fused_silica, create_borosilicate_glass, create_air
from .metals import (
    create_gold_rakic,
    create_silver_rakic,
    create_copper_rakic,
    create_aluminum_rakic,
    create_titanium_adhesion,
    create_titanium_dioxide,
)
from .polymers import create_pmma, create_topas, create_pdms
from .biological import WaterMaterial, GlucoseSolutionMaterial, AnalyteMaterial, create_bioreceptor_layer
from .database import MaterialDatabase, materials

__all__ = [
    "Material",
    "OpticalConstants",
    "ConstantIndexMaterial",
    "SellmeierMaterial",
    "CauchyMaterial",
    "DrudeLorentzMaterial",
    "TabulatedMaterial",
    "create_fused_silica",
    "create_borosilicate_glass",
    "create_air",
    "create_gold_rakic",
    "create_silver_rakic",
    "create_copper_rakic",
    "create_aluminum_rakic",
    "create_titanium_adhesion",
    "create_titanium_dioxide",
    "create_pmma",
    "create_topas",
    "create_pdms",
    "WaterMaterial",
    "GlucoseSolutionMaterial",
    "AnalyteMaterial",
    "create_bioreceptor_layer",
    "MaterialDatabase",
    "materials",
]

