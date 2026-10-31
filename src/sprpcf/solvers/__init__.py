"""Solvers module for electromagnetic mode analysis."""

from .base import OpticalMode, SolverConfig, SolverResult, BaseModeSolver
from .eigenvalue import solve_sparse_eigenproblem, compute_eigenvalue_diagnostics
from .fem_backend import FullVectorMaxwellSolver
from .ngsolve_backend import NGSolveModeSolver
from .mode_solver import ModeSolver
from .mode_tracking import ModeTracker, TrackedModeBranch
from .diagnostics import generate_solver_diagnostics

__all__ = [
    "OpticalMode",
    "SolverConfig",
    "SolverResult",
    "BaseModeSolver",
    "solve_sparse_eigenproblem",
    "compute_eigenvalue_diagnostics",
    "FullVectorMaxwellSolver",
    "NGSolveModeSolver",
    "ModeSolver",
    "ModeTracker",
    "TrackedModeBranch",
    "generate_solver_diagnostics",
]

