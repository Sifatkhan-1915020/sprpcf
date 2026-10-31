"""Master ModeSolver entry point and backend dispatcher."""

from typing import Optional, Dict, Any
from .base import BaseModeSolver, SolverConfig, SolverResult
from .fem_backend import FullVectorMaxwellSolver
from .ngsolve_backend import NGSolveModeSolver
from ..core.registry import solvers_registry

# Register solvers in registry
solvers_registry.register("native", FullVectorMaxwellSolver)
solvers_registry.register("fem", FullVectorMaxwellSolver)
solvers_registry.register("ngsolve", NGSolveModeSolver)


class ModeSolver(BaseModeSolver):
    """Unified mode solver dispatcher."""

    def __init__(self, config: Optional[SolverConfig] = None, backend: str = "native"):
        cfg = config or SolverConfig()
        super().__init__(cfg)
        self.backend = backend.lower().strip()
        if self.backend == "ngsolve":
            self._solver = NGSolveModeSolver(cfg)
        else:
            self._solver = FullVectorMaxwellSolver(cfg)

    def solve(
        self,
        geometry: Any,
        domain_materials: Dict[str, Any],
        wavelength_m: float,
        mesh: Optional[Any] = None,
    ) -> SolverResult:
        return self._solver.solve(geometry, domain_materials, wavelength_m, mesh)

