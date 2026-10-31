"""Unit tests for optical materials, dispersion equations, and database."""

import pytest
import math
from sprpcf.materials.database import materials
from sprpcf.materials.dielectric import create_fused_silica, create_air
from sprpcf.materials.metals import create_gold_rakic, create_silver_rakic
from sprpcf.materials.polymers import create_pmma
from sprpcf.materials.biological import WaterMaterial, GlucoseSolutionMaterial


def test_fused_silica_dispersion():
    silica = create_fused_silica()
    # At 1.55 um (1550 nm), standard Malitson index of fused silica is ~1.444024
    oc_1550 = silica.get_optical_constants(1.55e-6)
    assert abs(oc_1550.n - 1.4440) < 0.001
    assert oc_1550.k == 0.0  # Lossless in standard Sellmeier window

    # At 1.0 um, n ~ 1.4504
    oc_1000 = silica.get_optical_constants(1.0e-6)
    assert abs(oc_1000.n - 1.4504) < 0.002
    assert oc_1000.n > oc_1550.n  # Normal dispersion dn/dlambda < 0


def test_plasmonic_metals():
    gold = create_gold_rakic()
    silver = create_silver_rakic()

    # In NIR (1.0 um to 1.55 um), metals must have negative real permittivity (Drude plasmonic regime)
    oc_gold = gold.get_optical_constants(1.55e-6)
    eps_gold = oc_gold.permittivity
    assert eps_gold.real < -50.0  # Strongly negative
    assert eps_gold.imag > 0.0   # Positive ohmic absorption loss

    oc_silver = silver.get_optical_constants(1.55e-6)
    eps_silver = oc_silver.permittivity
    assert eps_silver.real < -50.0
    assert eps_silver.imag > 0.0


def test_polymer_cauchy():
    pmma = create_pmma()
    oc = pmma.get_optical_constants(633e-9)
    # PMMA at 633 nm has refractive index around 1.49
    assert 1.48 < oc.n < 1.51


def test_biological_materials():
    water = WaterMaterial()
    oc_water = water.get_optical_constants(600e-9)
    assert 1.33 < oc_water.n < 1.34

    # Glucose solution concentration dependence
    glucose_0 = GlucoseSolutionMaterial(concentration_g_per_l=0.0)
    glucose_100 = GlucoseSolutionMaterial(concentration_g_per_l=100.0)

    n0 = glucose_0.get_optical_constants(600e-9).n
    n100 = glucose_100.get_optical_constants(600e-9).n
    # 100 g/L glucose should increase index by ~0.0143 RIU
    delta_n = n100 - n0
    assert abs(delta_n - 0.0143) < 0.002


def test_material_database():
    assert "silica" in materials.list()
    assert "gold" in materials.list()
    assert "air" in materials.list()
    assert "water" in materials.list()

    mat = materials.get("gold")
    assert mat.name == "gold"
    assert mat.category == "metal"

    matches = materials.search("glass")
    assert len(matches) > 0

