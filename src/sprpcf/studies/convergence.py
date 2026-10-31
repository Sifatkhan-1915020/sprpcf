"""Mesh and computational convergence studies with rigorous error quantification."""

from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseStudy, StudyResult
from ..solvers.base import SolverConfig, SolverResult
from ..solvers.mode_solver import ModeSolver
from ..exceptions import MeshConvergenceError


class MeshConvergenceStudy(BaseStudy):
    """Systematically refines spatial mesh / grid resolution and tracks numerical convergence."""

    def __init__(
        self,
        grid_resolutions: Optional[List[int]] = None,  # e.g. [45, 60, 75]
        refinement_levels: Optional[List[str]] = None,  # ["coarse", "medium", "fine"]
        wavelength_m: float = 1.0e-6,
        modes: int = 2,
        tolerance_neff: float = 1e-4,
        tolerance_loss_rel: float = 0.05,
        name: str = "mesh_convergence_study",
    ):
        super().__init__(name=name)
        if grid_resolutions is not None:
            self.grid_resolutions = grid_resolutions
        elif refinement_levels is not None:
            level_map = {"coarse": 45, "medium": 60, "fine": 75, "ultra_fine": 95}
            self.grid_resolutions = [level_map.get(lvl.lower(), 60) for lvl in refinement_levels]
        else:
            self.grid_resolutions = [45, 60, 75]

        self.wavelength_m = float(wavelength_m)
        self.tolerance_neff = float(tolerance_neff)
        self.tolerance_loss_rel = float(tolerance_loss_rel)
        self.solver_config.num_modes = modes

    def run(self, project: Any) -> StudyResult:
        results = []
        n_eff_history = []
        loss_history = []
        residuals_history = []
        dof_history = []

        geom = project.geometry
        materials_map = project.get_active_domain_materials()

        for N in self.grid_resolutions:
            cfg = SolverConfig(
                num_modes=self.solver_config.num_modes,
                grid_points_x=N,
                grid_points_y=N,
                tolerance=self.solver_config.tolerance,
                solver_backend=self.solver_config.solver_backend,
            )
            solver = ModeSolver(cfg)
            res = solver.solve(
                geometry=geom,
                domain_materials=materials_map,
                wavelength_m=self.wavelength_m,
            )
            results.append(res)
            fund = res.fundamental_mode
            if fund:
                n_eff_history.append(fund.n_eff)
                loss_history.append(fund.loss_db_per_cm)
                residuals_history.append(fund.residual)
            else:
                n_eff_history.append(complex(1.0, 0.0))
                loss_history.append(0.0)
                residuals_history.append(1.0)
            dof_history.append(N * N * 2)  # Two vector transverse components Ex, Ey

        # Compute relative errors between consecutive refinement levels
        neff_diffs = []
        loss_rel_diffs = []
        for i in range(len(n_eff_history) - 1):
            dn = abs(n_eff_history[i + 1] - n_eff_history[i])
            neff_diffs.append(float(dn))

            denom_loss = max(1e-6, loss_history[i])
            d_loss_rel = abs(loss_history[i + 1] - loss_history[i]) / denom_loss
            loss_rel_diffs.append(float(d_loss_rel))

        is_converged = bool(
            neff_diffs and (neff_diffs[-1] <= self.tolerance_neff or (loss_rel_diffs and loss_rel_diffs[-1] <= self.tolerance_loss_rel))
        )

        # Tabulate convergence progression
        table = []
        for i, N in enumerate(self.grid_resolutions):
            step_info = {
                "level": f"Level_{i+1}",
                "grid_points": N,
                "dof": dof_history[i],
                "n_real": float(n_eff_history[i].real),
                "n_imag": float(n_eff_history[i].imag),
                "loss_db_per_cm": float(loss_history[i]),
                "residual": float(residuals_history[i]),
                "delta_neff": neff_diffs[i - 1] if i > 0 else 0.0,
                "rel_loss_diff": loss_rel_diffs[i - 1] if i > 0 else 0.0,
            }
            table.append(step_info)

        return StudyResult(
            study_name=self.name,
            results=results,
            parameters_swept={"grid_resolution": self.grid_resolutions},
            metadata={
                "n_eff_history": [complex(n) for n in n_eff_history],
                "loss_history": loss_history,
                "dof_history": dof_history,
                "consecutive_neff_differences": neff_diffs,
                "consecutive_loss_relative_diffs": loss_rel_diffs,
                "is_converged": is_converged,
                "tolerance_neff": self.tolerance_neff,
                "tolerance_loss_rel": self.tolerance_loss_rel,
                "convergence_table": table,
            },
        )
