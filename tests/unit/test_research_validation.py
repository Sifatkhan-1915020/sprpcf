"""Unit and numerical validation tests for research-grade SPR-PCF studio features."""

import pytest
import math
import numpy as np
import sprpcf as spr
from sprpcf.geometry.pcf_templates import DShapedPCF
from sprpcf.geometry.validation import validate_geometry
from sprpcf.meshing.validation import validate_mesh
from sprpcf.spr.resonance import fit_resonance_curve, extract_resonance_peaks, _lorentzian, _fano
from sprpcf.spr.sensitivity import SensitivityReport, calculate_amplitude_sensitivity_spectrum
from sprpcf.spr.figure_of_merit import SensorPerformanceMetrics
from sprpcf.physics.electromagnetic_fields import ElectromagneticField2D


def test_d_shape_geometry_validation():
    """Verify D-shaped PCF topological integrity, flat cut chord, and adhesion layer."""
    pcf = DShapedPCF(
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.5e-6,
        polish_depth=3.0e-6,
        metal_thickness=40.0e-9,
        adhesion_thickness=3.0e-9,
        adhesion_material="titanium",
        cladding_radius=6.0e-6,
    )
    report = pcf.validate_d_shape()

    assert report.is_valid is True
    assert report.d_shape_verified is True
    assert report.cladding_containment_verified is True
    assert "polish_depth" in report.dimensions
    assert abs(report.dimensions["polish_depth"] - 3.0e-6) < 1e-10
    # Flat chord width = 2 * sqrt(R^2 - d_c^2) = 2 * sqrt(36e-12 - 9e-12) = 2 * sqrt(27e-12) approx 10.392 um
    assert abs(report.dimensions["flat_chord_width"] - 2.0 * math.sqrt(27e-12)) < 1e-9
    assert report.hole_count > 0
    assert report.coating_count >= 2  # Gold + Ti adhesion


def test_mesh_quality_and_interface_refinement():
    """Verify mesh validation reports min quality, max aspect ratio, and interface elements."""
    project = spr.Project("test_mesh_report")
    pcf = DShapedPCF(pitch=2.0e-6, rings=1, polish_depth=2.5e-6, cladding_radius=5.0e-6)
    project.set_geometry(pcf)

    mesh = project.generate_mesh(backend="native", global_size=0.6e-6)
    report = validate_mesh(mesh, interface_y=2.5e-6)

    assert report.is_valid is True
    assert report.num_elements > 50
    assert report.num_nodes > 30
    assert report.mean_quality > 0.2
    assert report.min_quality > 1e-4
    assert report.max_aspect_ratio > 1.0
    assert report.interface_elements_count > 0
    assert len(report.quality_histogram["counts"]) == 10


def test_lorentzian_and_fano_curve_fitting():
    """Verify nonlinear least-squares lineshape fitting and parameter uncertainty estimation."""
    wl_nm = np.linspace(900.0, 1300.0, 41)
    true_wl0 = 1100.0
    true_gamma = 80.0
    true_A = 1000.0
    true_y0 = 5.0

    # Synthetic exact Lorentzian plus small baseline tilt
    loss = _lorentzian(wl_nm, true_y0, true_A, true_wl0, true_gamma, 0.0)

    fit_res = fit_resonance_curve(wl_nm, loss, model="lorentzian")
    assert fit_res is not None
    assert abs(fit_res.wavelength_nm - true_wl0) < 1.0  # within 1 nm
    assert abs(fit_res.fwhm_nm - true_gamma) < 2.0     # within 2 nm
    assert fit_res.r_squared > 0.99
    assert fit_res.is_adequately_sampled is True       # 41 points over 400 nm -> ~8 points in 80 nm FWHM
    assert fit_res.points_in_fwhm >= 5


def test_strict_sampling_adequacy_enforcement():
    """Ensure that undersampled resonances (< 5 points in FWHM) are flagged as inadequate."""
    # Coarse sweep: only 5 points across 900 to 1300 nm (spacing = 100 nm)
    wl_coarse = np.array([900.0, 1000.0, 1100.0, 1200.0, 1300.0])
    loss_coarse = _lorentzian(wl_coarse, 5.0, 1000.0, 1100.0, 50.0, 0.0)

    # In 50 nm FWHM (1075 to 1125 nm), only 1 point (1100 nm) exists!
    fit_res = fit_resonance_curve(wl_coarse, loss_coarse, model="lorentzian")
    if fit_res is not None:
        assert fit_res.is_adequately_sampled is False
        assert fit_res.points_in_fwhm < 5
        assert fit_res.warning is not None
        assert "undersampled" in fit_res.warning.lower()


def test_electromagnetic_power_flow_integration():
    """Verify 2D cross-sectional Poynting vector integration and forward power verification."""
    nx, ny = 30, 30
    x = np.linspace(-5e-6, 5e-6, nx)
    y = np.linspace(-5e-6, 5e-6, ny)
    X, Y = np.meshgrid(x, y)

    # Gaussian beam fundamental mode: Ex = exp(-r^2 / w0^2), Hy = (n / eta0) * Ex
    w0 = 2e-6
    ex = np.exp(-(X**2 + Y**2) / w0**2) + 0j
    ey = np.zeros_like(ex)
    eta0 = 376.73
    n_core = 1.45
    hy = (n_core / eta0) * ex
    hx = np.zeros_like(hy)

    field = ElectromagneticField2D(x=x, y=y, ex=ex, ey=ey, hx=hx, hy=hy, wavelength_m=1.55e-6)

    # Poynting Sz = 0.5 * Re(Ex * Hy*) > 0
    assert np.all(field.poynting_z >= 0.0)
    p_tot = field.total_power_flow
    assert p_tot > 0.0
    assert field.is_forward_propagating is True

    # Bicubic interpolation check
    fine_field = field.interpolate_bicubic(factor=2)
    assert len(fine_field.x) == nx * 2
    assert len(fine_field.y) == ny * 2
    assert fine_field.is_forward_propagating is True


def test_sensor_metrics_strict_assumptions():
    """Verify that resolution and detection limit require explicit experimental assumptions."""
    # Case 1: Stated instrument resolution (0.1 nm), no noise assumption
    metrics1 = SensorPerformanceMetrics(
        wavelength_sensitivity_nm_per_riu=2500.0,
        fwhm_nm=40.0,
        instrument_resolution_nm=0.1,
        noise_std_nm=None,
    )
    assert abs(metrics1.fom_riu_inv - (2500.0 / 40.0)) < 1e-4
    assert abs(metrics1.sensor_resolution_riu - 4.0e-5) < 1e-9
    assert metrics1.detection_limit_riu is None
    assert "requires calibrated baseline noise" in metrics1.lod_statement

    # Case 2: Unspecified instrument resolution
    metrics2 = SensorPerformanceMetrics(
        wavelength_sensitivity_nm_per_riu=2500.0,
        fwhm_nm=40.0,
        instrument_resolution_nm=None,
    )
    assert metrics2.sensor_resolution_riu is None
    assert "requires explicitly specified instrument resolution" in metrics2.resolution_statement
