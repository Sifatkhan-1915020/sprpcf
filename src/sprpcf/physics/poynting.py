"""Energy flow, Poynting vector, and regional power fraction analysis."""

from typing import Dict, List, Optional
import numpy as np
from .electromagnetic_fields import ElectromagneticField2D


def calculate_domain_power_fractions(
    field: ElectromagneticField2D,
    domain_masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Calculate the fractional optical power flowing through each physical domain.
    
    Fraction = integral_{domain} Sz dA / integral_{all} Sz dA.
    """
    sz = field.poynting_z
    total_power = float(np.sum(sz))

    fractions = {}
    if abs(total_power) < 1e-25:
        # Fall back to electric field intensity if Poynting vector is zero or uncomputed
        e2 = field.e_mag_sq
        tot_e2 = float(np.sum(e2))
        for dname, mask in domain_masks.items():
            fractions[dname] = float(np.sum(e2[mask])) / max(1e-25, tot_e2)
        return fractions

    for dname, mask in domain_masks.items():
        domain_power = float(np.sum(sz[mask]))
        fractions[dname] = max(0.0, domain_power / total_power)

    return fractions


def calculate_modal_overlap(field1: ElectromagneticField2D, field2: ElectromagneticField2D) -> float:
    """Calculate the normalized spatial overlap between two optical modes.
    
    Overlap = |integral (E1 . E2*) dA|^2 / (integral |E1|^2 dA * integral |E2|^2 dA).
    """
    e1_dot_e2 = field1.ex * np.conj(field2.ex) + field1.ey * np.conj(field2.ey)
    inner_prod = np.sum(e1_dot_e2)

    norm1 = np.sum(np.abs(field1.ex)**2 + np.abs(field1.ey)**2)
    norm2 = np.sum(np.abs(field2.ex)**2 + np.abs(field2.ey)**2)

    denom = norm1 * norm2
    if denom < 1e-25:
        return 0.0

    return float(np.abs(inner_prod)**2 / denom)

