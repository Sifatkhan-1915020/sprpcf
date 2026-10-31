"""Electromagnetic physics and field analysis module."""

from .maxwell import calculate_k0, calculate_omega, beta_to_neff, neff_to_beta, calculate_loss_db_per_m
from .electromagnetic_fields import ElectromagneticField2D
from .polarization import classify_polarization, calculate_birefringence
from .poynting import calculate_domain_power_fractions, calculate_modal_overlap

__all__ = [
    "calculate_k0",
    "calculate_omega",
    "beta_to_neff",
    "neff_to_beta",
    "calculate_loss_db_per_m",
    "ElectromagneticField2D",
    "classify_polarization",
    "calculate_birefringence",
    "calculate_domain_power_fractions",
    "calculate_modal_overlap",
]

