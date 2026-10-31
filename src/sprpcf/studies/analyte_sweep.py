"""Analyte refractive index sweep and geometric parameter sweep studies."""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from .base import BaseStudy, StudyResult
from ..solvers.base import SolverConfig, SolverResult
from ..solvers.mode_solver import ModeSolver
from ..materials.biological import AnalyteMaterial
from ..spr.analyzer import SPRAnalyzer


class AnalyteSweepStudy(BaseStudy):
    """Sweeps analyte refractive indices to evaluate biosensor wavelength and amplitude sensitivities."""

    def __init__(
        self,
        analyte_indices: List[float],
        wavelength_range: Tuple[float, float] = (600e-9, 1600e-9),
        wavelength_points: int = 41,
        modes: int = 4,
        polarization: str = "both",
        analyte_region_name: str = "sensing_analyte_channel",
        name: str = "analyte_sweep_study",
    ):
        super().__init__(name=name)
        self.analyte_indices = sorted([float(n) for n in analyte_indices])
        self.wavelength_range = wavelength_range
        self.wavelength_points = wavelength_points
        self.modes = modes
        self.polarization = polarization
        self.analyte_region_name = analyte_region_name

        self.solver_config.num_modes = modes
        self.solver_config.polarization = polarization

    def run(self, project: Any) -> StudyResult:
        all_results: List[SolverResult] = []
        analyte_results_dict: Dict[float, List[SolverResult]] = {}

        wls = np.linspace(self.wavelength_range[0], self.wavelength_range[1], self.wavelength_points)
        solver = ModeSolver(self.solver_config, backend=self.solver_config.solver_backend)

        for na in self.analyte_indices:
            # Temporarily update analyte material in project
            analyte_mat = AnalyteMaterial(name=f"temp_analyte_{na}", refractive_index=na)
            orig_mat = project.domain_assignments.get_material_name(self.analyte_region_name)
            project.assign_material(self.analyte_region_name, analyte_mat)

            step_results = []
            for wl in wls:
                res = solver.solve(
                    geometry=project.geometry,
                    domain_materials=project.get_active_domain_materials(),
                    wavelength_m=wl,
                )
                step_results.append(res)
                all_results.append(res)

            analyte_results_dict[na] = step_results

            # Restore original assignment if applicable
            if orig_mat:
                project.assign_material(self.analyte_region_name, orig_mat)

        # Run high-level sensitivity calculation
        analyzer = SPRAnalyzer(all_results)
        sens_summary = analyzer.calculate_sensitivity_from_analyte_sweep(analyte_results_dict)

        return StudyResult(
            study_name=self.name,
            results=all_results,
            parameters_swept={"analyte_index": self.analyte_indices},
            metadata={
                "sensitivity_summary": sens_summary,
                "analyte_indices": self.analyte_indices,
            },
        )


class ParameterSweepStudy(BaseStudy):
    """Sweeps arbitrary geometric or material parameters (e.g. coating thickness, pitch)."""

    def __init__(
        self,
        parameter_name: str,
        parameter_values: List[float],
        wavelength_m: float = 1.55e-6,
        modes: int = 4,
        name: str = "parameter_sweep_study",
    ):
        super().__init__(name=name)
        self.parameter_name = parameter_name
        self.parameter_values = [float(v) for v in parameter_values]
        self.wavelength_m = float(wavelength_m)
        self.solver_config.num_modes = modes

    def run(self, project: Any) -> StudyResult:
        solver = ModeSolver(self.solver_config, backend=self.solver_config.solver_backend)
        results = []

        for val in self.parameter_values:
            # Set parameter in project
            project.parameters.set(self.parameter_name, val)
            res = solver.solve(
                geometry=project.geometry,
                domain_materials=project.get_active_domain_materials(),
                wavelength_m=self.wavelength_m,
            )
            results.append(res)

        return StudyResult(
            study_name=self.name,
            results=results,
            parameters_swept={self.parameter_name: self.parameter_values},
            metadata={"wavelength_um": self.wavelength_m * 1e6},
        )

