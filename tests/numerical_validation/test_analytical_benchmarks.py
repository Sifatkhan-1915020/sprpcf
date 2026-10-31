"""Mathematical and analytical benchmarks for computational electromagnetics."""

import pytest
import math
import cmath
import numpy as np
from sprpcf.core.constants import C0, NEPER_TO_DB
from sprpcf.physics.maxwell import calculate_k0, calculate_loss_db_per_m, beta_to_neff
from sprpcf.materials.metals import create_gold_rakic
from sprpcf.materials.dielectric import create_fused_silica, create_air
from sprpcf.core.project import Project
from sprpcf.geometry.cross_section import PCFGeometry
from sprpcf.geometry.primitives import Circle
from sprpcf.solvers.fem_backend import FullVectorMaxwellSolver
from sprpcf.solvers.base import SolverConfig


def test_analytical_spp_fano_dispersion():
    """Verify analytical planar metal-dielectric Surface Plasmon Polariton (SPP) dispersion.
    
    The Fano dispersion relation for a semi-infinite planar metal/dielectric interface is:
        k_spp = k0 * sqrt((eps_m * eps_d) / (eps_m + eps_d))
        n_spp = sqrt((eps_m * eps_d) / (eps_m + eps_d))
    """
    wl = 1.0e-6  # 1000 nm
    k0 = calculate_k0(wl)
    gold = create_gold_rakic()
    water_eps = 1.333**2  # Water dielectric permittivity

    eps_m = gold.get_permittivity(wl)

    # Analytical Fano formula
    n_spp_sq = (eps_m * water_eps) / (eps_m + water_eps)
    n_spp_analytical = cmath.sqrt(n_spp_sq)

    # Physical SPP characteristics:
    # 1. Real effective index of SPP is slightly greater than adjacent dielectric index
    assert n_spp_analytical.real > 1.333
    # 2. Imaginary part represents plasmonic ohmic damping
    assert n_spp_analytical.imag > 0.0


def test_attenuation_formula_exactness():
    """Verify exact equivalence between beta_imag and n_eff_imag attenuation formulas."""
    k0 = 2 * math.pi / 1.55e-6
    n_imag = 1.25e-5

    alpha_db_m = calculate_loss_db_per_m(n_imag, k0)
    expected_alpha = (20.0 / math.log(10.0)) * k0 * n_imag

    assert abs(alpha_db_m - expected_alpha) < 1e-10
    # In dB/cm: alpha [dB/cm] = alpha [dB/m] * 0.01
    alpha_db_cm = alpha_db_m * 1e-2
    assert abs(alpha_db_cm - expected_alpha * 0.01) < 1e-10


def test_step_index_fiber_waveguide_eigenmode():
    """Verify core-guided mode in step-index fiber satisfies n_clad < n_eff < n_core."""
    proj = Project("step_index_test")
    geom = PCFGeometry(cladding_radius=6.0e-6)
    # Core with radius 1.5 um
    geom.create_core(shape="circular", radius=1.5e-6, material="silica")
    proj.set_geometry(geom)

    # High index core (n=1.46), lower index cladding (n=1.44)
    from sprpcf.materials.dispersion import ConstantIndexMaterial
    proj.assign_material("fiber_core", ConstantIndexMaterial("core_mat", n=1.46))
    proj.assign_material("cladding", ConstantIndexMaterial("clad_mat", n=1.44))

    solver = FullVectorMaxwellSolver(
        SolverConfig(
            num_modes=2,
            target_index=1.455,
            grid_points_x=60,
            grid_points_y=60,
        )
    )

    res = solver.solve(
        geometry=proj.geometry,
        domain_materials=proj.get_active_domain_materials(),
        wavelength_m=1.55e-6,
    )

    assert len(res.modes) > 0
    fund = res.fundamental_mode
    assert fund is not None

    # Analytical fundamental guided mode condition: n_clad < n_eff < n_core
    assert 1.44 < fund.n_real < 1.46

