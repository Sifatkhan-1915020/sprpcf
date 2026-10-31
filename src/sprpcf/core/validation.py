"""Validation utilities for geometry, materials, boundaries, and simulation configs."""

from typing import List, Dict, Any, Optional
from ..exceptions import (
    InvalidGeometryError,
    OverlappingRegionError,
    DomainAssignmentError,
    SolverConfigurationError,
)


def validate_positive_number(val: float, name: str) -> float:
    """Ensure a number is strictly greater than zero."""
    if val <= 0:
        raise ValueError(f"{name} must be strictly positive (> 0), got {val}")
    return val


def validate_non_negative_number(val: float, name: str) -> float:
    """Ensure a number is greater than or equal to zero."""
    if val < 0:
        raise ValueError(f"{name} must be non-negative (>= 0), got {val}")
    return val


def validate_thickness(thickness_m: float, name: str = "Coating thickness") -> float:
    """Ensure thickness is physically reasonable for nanophotonic/plasmonic biosensors."""
    if thickness_m <= 0:
        raise InvalidGeometryError(f"{name} must be > 0, got {thickness_m}")
    if thickness_m < 0.1e-9:
        raise InvalidGeometryError(f"{name} {thickness_m} m is below single-atomic monolayer (~0.1 nm).")
    if thickness_m > 1e-3:
        raise InvalidGeometryError(f"{name} {thickness_m} m is excessively large (> 1 mm) for thin-film coating.")
    return thickness_m


def validate_domains_assigned(all_regions: List[str], assigned_regions: Dict[str, str]) -> None:
    """Verify that every defined physical geometric domain has a valid material assignment."""
    unassigned = [r for r in all_regions if r not in assigned_regions or not assigned_regions[r]]
    if unassigned:
        raise DomainAssignmentError(
            f"The following geometric domains have no material assigned: {unassigned}. "
            "Every domain must have an assigned material before meshing and simulation."
        )


def validate_wavelength_range(start_m: float, stop_m: float, points: int) -> None:
    """Validate spectral sweep configuration."""
    validate_positive_number(start_m, "Wavelength start")
    validate_positive_number(stop_m, "Wavelength stop")
    if start_m >= stop_m:
        raise SolverConfigurationError(f"Wavelength start ({start_m} m) must be < stop ({stop_m} m)")
    if points < 2:
        raise SolverConfigurationError(f"Sweep points must be at least 2, got {points}")

