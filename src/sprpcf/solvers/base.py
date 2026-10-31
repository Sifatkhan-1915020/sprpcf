"""Abstract interfaces and data structures for optical fiber mode solvers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from pydantic import BaseModel, Field
from ..physics.electromagnetic_fields import ElectromagneticField2D
from ..physics.maxwell import calculate_loss_db_per_m, calculate_k0
from ..core.units import attenuation_conversion


class OpticalMode:
    """Represents a single converged electromagnetic eigenmode of the fiber."""

    def __init__(
        self,
        mode_id: int,
        n_eff: complex,
        wavelength_m: float,
        field: Optional[ElectromagneticField2D] = None,
        residual: float = 0.0,
        polarization_fraction_x: float = 0.5,
        polarization_fraction_y: float = 0.5,
        domain_power_fractions: Optional[Dict[str, float]] = None,
    ):
        self.mode_id = int(mode_id)
        self.n_eff = complex(n_eff)
        self.wavelength_m = float(wavelength_m)
        self.field = field
        self.residual = float(residual)
        self.fx = float(polarization_fraction_x)
        self.fy = float(polarization_fraction_y)
        self.domain_power_fractions = domain_power_fractions or {}

    @property
    def n_real(self) -> float:
        """Phase index Re(n_eff)."""
        return self.n_eff.real

    @property
    def n_imag(self) -> float:
        """Attenuation index Im(n_eff) >= 0."""
        return max(0.0, self.n_eff.imag)

    @property
    def k0(self) -> float:
        return calculate_k0(self.wavelength_m)

    @property
    def beta(self) -> complex:
        """Complex propagation constant beta = n_eff * k0."""
        return self.n_eff * self.k0

    @property
    def loss_db_per_m(self) -> float:
        """Confinement/propagation loss in dB/m."""
        return calculate_loss_db_per_m(self.n_imag, self.k0)

    @property
    def loss_db_per_cm(self) -> float:
        """Confinement loss in dB/cm."""
        return attenuation_conversion(self.loss_db_per_m, "db/m", "db/cm")

    @property
    def loss_db_per_um(self) -> float:
        return attenuation_conversion(self.loss_db_per_m, "db/m", "db/um")

    @property
    def polarization_label(self) -> str:
        if self.fx >= 0.75:
            return "x_pol"
        elif self.fy >= 0.75:
            return "y_pol"
        return "hybrid"

    @property
    def is_spp_like(self) -> bool:
        """True if the mode has strong field localization in or near plasmonic metal/analyte."""
        metal_frac = sum(f for k, f in self.domain_power_fractions.items() if "metal" in k.lower() or "gold" in k.lower() or "coating" in k.lower())
        analyte_frac = sum(f for k, f in self.domain_power_fractions.items() if "analyte" in k.lower() or "channel" in k.lower())
        return metal_frac > 0.05 or (analyte_frac > 0.3 and self.loss_db_per_m > 10.0)

    @property
    def is_core_guided(self) -> bool:
        """True if the mode has dominant energy in the core."""
        core_frac = sum(f for k, f in self.domain_power_fractions.items() if "core" in k.lower())
        return core_frac > 0.25

    @property
    def is_hybrid_spr(self) -> bool:
        """True if the mode exhibits simultaneous core confinement and strong plasmonic excitation."""
        metal_frac = sum(f for k, f in self.domain_power_fractions.items() if "metal" in k.lower() or "gold" in k.lower() or "coating" in k.lower())
        core_frac = sum(f for k, f in self.domain_power_fractions.items() if "core" in k.lower())
        return core_frac > 0.10 and metal_frac > 0.01

    @property
    def mode_classification(self) -> str:
        """Physical classification of the optical eigenmode."""
        if self.is_hybrid_spr:
            return "Hybridized SPR Mode"
        elif self.is_spp_like:
            return "Plasmonic SPP Mode"
        elif self.is_core_guided:
            return "Core-Guided Mode"
        return "Cladding / Leaky Mode"

    @property
    def total_power(self) -> float:
        """Integrated cross-sectional longitudinal power flow \\iint S_z dx dy."""
        return self.field.total_power_flow if self.field is not None else 0.0

    @property
    def is_forward_propagating(self) -> bool:
        """Validate forward power flow (+z direction)."""
        return self.field.is_forward_propagating if self.field is not None else True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "n_real": self.n_real,
            "n_imag": self.n_imag,
            "loss_db_per_m": self.loss_db_per_m,
            "loss_db_per_cm": self.loss_db_per_cm,
            "wavelength_um": self.wavelength_m * 1e6,
            "polarization": self.polarization_label,
            "fx": self.fx,
            "fy": self.fy,
            "residual": self.residual,
            "power_fractions": self.domain_power_fractions,
        }

    def __repr__(self) -> str:
        return (
            f"OpticalMode(id={self.mode_id}, n_eff={self.n_real:.6f} + {self.n_imag:.4e}i, "
            f"loss={self.loss_db_per_cm:.2f} dB/cm, pol={self.polarization_label})"
        )


class SolverConfig(BaseModel):
    """Configuration options for optical mode solvers."""

    num_modes: int = 6
    target_index: Optional[float] = 1.44  # Guess for fundamental core mode
    tolerance: float = 1e-8
    max_iterations: int = 500
    polarization: str = "both"  # 'x', 'y', 'both'
    solver_backend: str = "native"  # 'native', 'ngsolve'
    grid_points_x: int = 150
    grid_points_y: int = 150
    pml_thickness: float = 1.5e-6
    pml_scaling: float = 1.0


class SolverResult:
    """Container for the output of a single mode solve."""

    def __init__(
        self,
        wavelength_m: float,
        modes: List[OpticalMode],
        execution_time: float,
        solver_name: str,
        converged: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.wavelength_m = float(wavelength_m)
        self.modes = sorted(modes, key=lambda m: m.n_real, reverse=True)
        self.execution_time = float(execution_time)
        self.solver_name = solver_name
        self.converged = converged
        self.metadata = metadata or {}

    @property
    def fundamental_mode(self) -> Optional[OpticalMode]:
        """Return fundamental core-guided mode if found, or highest index mode."""
        core_modes = [m for m in self.modes if m.is_core_guided]
        if core_modes:
            return core_modes[0]
        return self.modes[0] if self.modes else None

    def get_mode_by_id(self, mode_id: int) -> Optional[OpticalMode]:
        for m in self.modes:
            if m.mode_id == mode_id:
                return m
        return None

    def get_modes_by_polarization(self, pol: str) -> List[OpticalMode]:
        target = pol.lower().strip()
        if target.startswith("x"):
            return [m for m in self.modes if m.polarization_label == "x_pol"]
        elif target.startswith("y"):
            return [m for m in self.modes if m.polarization_label == "y_pol"]
        return self.modes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wavelength_um": self.wavelength_m * 1e6,
            "modes_count": len(self.modes),
            "execution_time_s": self.execution_time,
            "solver": self.solver_name,
            "converged": self.converged,
            "modes": [m.to_dict() for m in self.modes],
        }


class BaseModeSolver(ABC):
    """Abstract base class for PCF waveguide mode solvers."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

    @abstractmethod
    def solve(
        self,
        geometry: Any,
        domain_materials: Dict[str, Any],
        wavelength_m: float,
        mesh: Optional[Any] = None,
    ) -> SolverResult:
        """Solve for optical modes at specified wavelength."""
        pass

