"""Unit conversion utilities for computational optics and biosensing.

Provides conversions for length, frequency, energy, attenuation, and sensitivity metrics.
"""

import math
from typing import Union, Tuple
from .constants import C0, H_PLANCK, QE, NEPER_TO_DB


class LengthUnit:
    """Length conversions relative to SI meter (m)."""
    M = 1.0
    CM = 1e-2
    MM = 1e-3
    UM = 1e-6
    NM = 1e-9
    PM = 1e-12

    @classmethod
    def to_meters(cls, value: float, unit: str) -> float:
        u = unit.lower().strip()
        factors = {
            "m": cls.M,
            "meter": cls.M,
            "meters": cls.M,
            "cm": cls.CM,
            "centimeter": cls.CM,
            "mm": cls.MM,
            "millimeter": cls.MM,
            "um": cls.UM,
            "micron": cls.UM,
            "micrometer": cls.UM,
            "microns": cls.UM,
            "nm": cls.NM,
            "nanometer": cls.NM,
            "nanometers": cls.NM,
            "pm": cls.PM,
            "picometer": cls.PM,
        }
        if u not in factors:
            raise ValueError(f"Unknown length unit: '{unit}'. Supported: {list(factors.keys())}")
        return value * factors[u]

    @classmethod
    def from_meters(cls, value_m: float, unit: str) -> float:
        u = unit.lower().strip()
        factors = {
            "m": cls.M,
            "cm": cls.CM,
            "mm": cls.MM,
            "um": cls.UM,
            "nm": cls.NM,
            "pm": cls.PM,
        }
        if u not in factors:
            raise ValueError(f"Unknown length unit: '{unit}'. Supported: {list(factors.keys())}")
        return value_m / factors[u]


def wavelength_to_frequency(wavelength_m: float) -> float:
    """Convert wavelength in meters to frequency in Hertz."""
    if wavelength_m <= 0:
        raise ValueError("Wavelength must be strictly positive.")
    return C0 / wavelength_m


def frequency_to_wavelength(frequency_hz: float) -> float:
    """Convert frequency in Hertz to wavelength in meters."""
    if frequency_hz <= 0:
        raise ValueError("Frequency must be strictly positive.")
    return C0 / frequency_hz


def wavelength_to_wavenumber(wavelength_m: float) -> float:
    """Convert vacuum wavelength in meters to wavenumber k0 = 2*pi / lambda in rad/m."""
    if wavelength_m <= 0:
        raise ValueError("Wavelength must be strictly positive.")
    return 2.0 * math.pi / wavelength_m


def photon_energy_ev(wavelength_m: float) -> float:
    """Calculate photon energy in electron-volts for a given vacuum wavelength in meters."""
    nu = wavelength_to_frequency(wavelength_m)
    return (H_PLANCK * nu) / QE


def attenuation_conversion(value: float, from_unit: str, to_unit: str) -> float:
    """Convert attenuation between units (dB/m, dB/cm, dB/mm, dB/um).
    
    1 dB/m = 0.01 dB/cm = 1e-3 dB/mm = 1e-6 dB/um.
    """
    scale_to_m = {
        "db/m": 1.0,
        "db/cm": 100.0,
        "db/mm": 1000.0,
        "db/um": 1e6,
    }
    f = from_unit.lower().replace(" ", "")
    t = to_unit.lower().replace(" ", "")
    if f not in scale_to_m or t not in scale_to_m:
        raise ValueError(f"Unsupported attenuation unit conversion from '{from_unit}' to '{to_unit}'")
    # Convert to dB/m first
    val_m = value * scale_to_m[f]
    # Convert from dB/m to target
    return val_m / scale_to_m[t]


def parse_quantity(q: Union[str, float, int], default_unit: str = "m") -> Tuple[float, str]:
    """Parse a quantity string like '40 nm', '2.0 um', '1550 nm', or return numeric with default unit."""
    if isinstance(q, (int, float)):
        return float(q), default_unit
    s = str(q).strip()
    import re
    match = re.match(r"^([+-]?[\d\.eE]+)\s*([a-zA-Z_/%]*)$", s)
    if not match:
        raise ValueError(f"Cannot parse quantity expression: '{q}'")
    val_str, unit_str = match.groups()
    val = float(val_str)
    unit = unit_str if unit_str else default_unit
    return val, unit

