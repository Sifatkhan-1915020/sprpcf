"""Electromagnetic boundary conditions for PCF computational domains."""

from enum import Enum
from typing import Optional, Dict, Any


class BoundaryType(str, Enum):
    PML = "pml"
    PEC = "pec"
    PMC = "pmc"
    SCATTERING = "scattering"
    CONTINUITY = "continuity"


class BoundaryCondition:
    """Represents a boundary condition applied to an outer boundary or interface."""

    def __init__(
        self,
        name: str,
        btype: BoundaryType = BoundaryType.PML,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.btype = btype
        self.parameters = parameters or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.btype.value,
            "parameters": self.parameters,
        }


class PerfectlyMatchedLayer(BoundaryCondition):
    """Cylindrical or Cartesian stretched-coordinate PML."""

    def __init__(
        self,
        name: str = "outer_pml",
        thickness_m: float = 1.5e-6,
        inner_radius_m: float = 10.0e-6,
        scaling_factor: float = 1.0,
        absorption_exponent: float = 2.0,
        coordinate_system: str = "cylindrical",  # cylindrical or cartesian
    ):
        super().__init__(
            name=name,
            btype=BoundaryType.PML,
            parameters={
                "thickness_m": float(thickness_m),
                "inner_radius_m": float(inner_radius_m),
                "scaling_factor": float(scaling_factor),
                "absorption_exponent": float(absorption_exponent),
                "coordinate_system": coordinate_system,
            },
        )
        self.thickness_m = float(thickness_m)
        self.inner_radius_m = float(inner_radius_m)
        self.scaling_factor = float(scaling_factor)
        self.absorption_exponent = float(absorption_exponent)
        self.coordinate_system = coordinate_system

    def get_stretching_factor(self, r_m: float, k0: float) -> complex:
        """Calculate complex stretching factor s(r) = 1 - i * alpha * ((r - r0) / d)^m."""
        if r_m <= self.inner_radius_m:
            return complex(1.0, 0.0)
        d = self.thickness_m
        dr = min(d, max(0.0, r_m - self.inner_radius_m))
        normalized = dr / d
        # Loss parameter proportional to scaling / (k0 * d)
        sigma = self.scaling_factor * (normalized**self.absorption_exponent)
        # s = 1 - i * sigma
        return complex(1.0, -sigma)

