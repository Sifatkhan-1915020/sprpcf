"""Numerical validation for SPR resonance detection, sensitivity, and FOM metrics."""

import pytest
import numpy as np
from sprpcf.spr.resonance import extract_resonance_peaks
from sprpcf.spr.sensitivity import SensitivityReport, calculate_amplitude_sensitivity_spectrum
from sprpcf.spr.figure_of_merit import SensorPerformanceMetrics
from sprpcf.solvers.mode_tracking import ModeTracker, TrackedModeBranch
from sprpcf.solvers.base import OpticalMode


def test_resonance_peak_and_fwhm_extraction():
    # Construct a synthetic Lorentzian resonance peak centered at 1200 nm
    wls_nm = np.linspace(1000, 1400, 81)
    wls_m = wls_nm * 1e-9

    center_wl = 1200.0
    gamma = 25.0  # HWHM = 25 nm, FWHM = 50 nm
    peak_loss = 80.0  # dB/cm
    loss_curve = peak_loss / (1.0 + ((wls_nm - center_wl) / gamma)**2)

    peaks = extract_resonance_peaks(wls_m, loss_curve, interpolate=True)
    assert len(peaks) == 1
    p = peaks[0]

    # Check extracted resonance wavelength
    assert abs(p.wavelength_nm - 1200.0) < 1.0
    # Check extracted peak loss
    assert abs(p.peak_loss_db_cm - 80.0) < 1.0
    # Check extracted FWHM ~ 50 nm
    assert abs(p.fwhm_nm - 50.0) < 4.0


def test_sensitivity_and_fom_calculation():
    # 3 analyte indices: 1.33, 1.34, 1.35
    analyte_ris = [1.33, 1.34, 1.35]
    # Corresponding resonance wavelengths with slope 3000 nm/RIU
    res_wls = [1000.0, 1030.0, 1060.0]

    report = SensitivityReport(analyte_indices=analyte_ris, resonance_wavelengths_nm=res_wls)

    # Linear sensitivity must be exactly 3000 nm/RIU
    assert abs(report.linear_sensitivity_nm_per_riu - 3000.0) < 1e-3
    assert abs(report.r_squared - 1.0) < 1e-6

    # Sensor performance metrics with FWHM = 30 nm
    perf = SensorPerformanceMetrics(
        wavelength_sensitivity_nm_per_riu=report.linear_sensitivity_nm_per_riu,
        fwhm_nm=30.0,
        instrument_resolution_nm=0.1,
    )

    # FOM = 3000 / 30 = 100 RIU^-1
    assert abs(perf.fom_riu_inv - 100.0) < 1e-3

    # Resolution = 0.1 / 3000 = 3.33e-5 RIU
    assert abs(perf.resolution_riu - (0.1 / 3000.0)) < 1e-8


def test_amplitude_sensitivity_spectrum():
    wls_nm = np.linspace(1000, 1200, 41)
    wls_m = wls_nm * 1e-9

    loss1 = 50.0 / (1.0 + ((wls_nm - 1100.0) / 20.0)**2)
    loss2 = 50.0 / (1.0 + ((wls_nm - 1110.0) / 20.0)**2)  # shifted by 10 nm for dn = 0.005
    dn = 0.005

    wls_out, s_a, max_sa = calculate_amplitude_sensitivity_spectrum(wls_m, loss1, loss2, dn)
    assert max_sa > 0.0
    assert len(s_a) == len(wls_nm)

