"""Polarization analysis and modal birefringence."""

from typing import Tuple, Dict, Any
from .electromagnetic_fields import ElectromagneticField2D


def classify_polarization(field: ElectromagneticField2D, threshold: float = 0.75) -> Dict[str, Any]:
    """Classify mode polarization state based on transverse field fractions."""
    fx, fy = field.polarization_fractions
    if fx >= threshold:
        label = "x_polarized"
    elif fy >= threshold:
        label = "y_polarized"
    else:
        label = "hybrid"

    return {
        "fx": fx,
        "fy": fy,
        "classification": label,
        "is_x_polarized": fx >= threshold,
        "is_y_polarized": fy >= threshold,
    }


def calculate_birefringence(n_eff_x: complex, n_eff_y: complex) -> Dict[str, float]:
    """Calculate phase birefringence B = |Re(n_eff_x) - Re(n_eff_y)| and loss difference."""
    b_real = abs(n_eff_x.real - n_eff_y.real)
    b_imag = abs(n_eff_x.imag - n_eff_y.imag)
    return {
        "modal_birefringence": b_real,
        "imaginary_birefringence": b_imag,
    }

