"""Unit tests for units, physical constants, and parameter management."""

import pytest
import math
from sprpcf.core.constants import C0, EPSILON_0, MU_0, ETA_0, NEPER_TO_DB
from sprpcf.core.units import (
    LengthUnit,
    wavelength_to_frequency,
    frequency_to_wavelength,
    wavelength_to_wavenumber,
    photon_energy_ev,
    attenuation_conversion,
    parse_quantity,
)
from sprpcf.core.parameters import Parameter, ParameterSet


def test_physical_constants():
    assert abs(C0 - 299792458.0) < 1e-6
    # Vacuum impedance eta_0 = sqrt(mu_0 / eps_0) ~ 376.73 Ohms
    assert abs(ETA_0 - math.sqrt(MU_0 / EPSILON_0)) < 1e-4
    # Neper to dB conversion factor ~ 8.6858896
    assert abs(NEPER_TO_DB - 20.0 / math.log(10.0)) < 1e-8


def test_unit_conversions():
    # Length conversions
    assert abs(LengthUnit.to_meters(1.55, "um") - 1.55e-6) < 1e-15
    assert abs(LengthUnit.to_meters(40.0, "nm") - 40e-9) < 1e-15
    assert abs(LengthUnit.from_meters(1.55e-6, "um") - 1.55) < 1e-12

    # Frequency and wavelength
    wl = 1.55e-6
    freq = wavelength_to_frequency(wl)
    assert abs(frequency_to_wavelength(freq) - wl) < 1e-15

    # Wavenumber k0 = 2*pi / lambda
    k0 = wavelength_to_wavenumber(wl)
    assert abs(k0 - 2 * math.pi / 1.55e-6) < 1e-6

    # Photon energy at 1.55 um ~ 0.8 eV
    e_ev = photon_energy_ev(wl)
    assert 0.79 < e_ev < 0.81

    # Attenuation unit conversions
    # 100 dB/m = 1 dB/cm = 1e-4 dB/um
    assert abs(attenuation_conversion(100.0, "db/m", "db/cm") - 1.0) < 1e-6
    assert abs(attenuation_conversion(1.0, "db/cm", "db/m") - 100.0) < 1e-6


def test_parameter_set():
    pset = ParameterSet()
    p1 = pset.create(name="pitch", value=2.0, unit="um", min_value=0.1, max_value=10.0)
    assert p1.value == 2.0
    assert abs(p1.get_si_value() - 2.0e-6) < 1e-12

    # Test bounds
    with pytest.raises(ValueError):
        p1.set_value(15.0)  # Exceeds max

    # Test change listener
    changed = []
    pset.register_change_listener(lambda name, val: changed.append((name, val)))
    pset.set("pitch", 2.5)
    assert len(changed) == 1
    assert changed[0] == ("pitch", 2.5)

