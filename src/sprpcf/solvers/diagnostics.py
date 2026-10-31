"""Solver performance diagnostics and residual verification."""

from typing import Dict, Any, List
from .base import SolverResult


def generate_solver_diagnostics(result: SolverResult) -> Dict[str, Any]:
    """Compile numerical convergence, timing, and residual report for a simulation."""
    res_list = [m.residual for m in result.modes]
    max_res = max(res_list) if res_list else 0.0
    mean_res = sum(res_list) / len(res_list) if res_list else 0.0

    return {
        "solver": result.solver_name,
        "wavelength_nm": result.wavelength_m * 1e9,
        "execution_time_s": result.execution_time,
        "num_modes_found": len(result.modes),
        "max_residual": max_res,
        "mean_residual": mean_res,
        "all_converged": max_res < 1e-3,
        "modes_summary": [
            {
                "id": m.mode_id,
                "n_real": m.n_real,
                "loss_db_cm": m.loss_db_per_cm,
                "pol": m.polarization_label,
                "res": m.residual,
            }
            for m in result.modes
        ],
    }

