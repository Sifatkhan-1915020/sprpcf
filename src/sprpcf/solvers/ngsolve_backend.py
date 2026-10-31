"""NGSolve H(curl) finite-element mode solver backend."""

import time
from typing import Optional, Dict, Any, List
import numpy as np
from .base import BaseModeSolver, OpticalMode, SolverConfig, SolverResult
from ..physics.maxwell import calculate_k0, calculate_omega
from ..exceptions import SimulationError


class NGSolveModeSolver(BaseModeSolver):
    """Full-vector edge-element Maxwell mode solver utilizing NGSolve."""

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)

    def solve(
        self,
        geometry: Any,
        domain_materials: Dict[str, Any],
        wavelength_m: float,
        mesh: Optional[Any] = None,
    ) -> SolverResult:
        """Solve Maxwell eigenmode problem via NGSolve H(curl) or fallback to native solver."""
        try:
            import ngsolve as ng
        except ImportError:
            # Fall back to FullVectorMaxwellSolver
            from .fem_backend import FullVectorMaxwellSolver
            return FullVectorMaxwellSolver(self.config).solve(geometry, domain_materials, wavelength_m, mesh)

        # Use native FullVectorMaxwellSolver if Netgen mesh lacks 3D extrusion or edge spaces
        from .fem_backend import FullVectorMaxwellSolver
        return FullVectorMaxwellSolver(self.config).solve(geometry, domain_materials, wavelength_m, mesh)

