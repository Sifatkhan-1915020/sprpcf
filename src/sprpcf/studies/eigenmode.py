"""Eigenmode and wavelength sweep studies."""

from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
from .base import BaseStudy, StudyResult
from ..solvers.base import SolverConfig, SolverResult
from ..solvers.mode_solver import ModeSolver
from ..solvers.mode_tracking import ModeTracker
from ..core.validation import validate_wavelength_range


class EigenmodeStudy(BaseStudy):
    """Calculates guided optical modes at a single wavelength or over a wavelength range."""

    def __init__(
        self,
        wavelength: Optional[float] = None,
        wavelength_range: Optional[Tuple[float, float]] = None,
        points: int = 1,
        modes: int = 6,
        polarization: str = "both",
        name: str = "eigenmode_study",
        solver_config: Optional[SolverConfig] = None,
    ):
        super().__init__(name=name, solver_config=solver_config)
        self.modes = modes
        self.polarization = polarization
        self.solver_config.num_modes = modes
        self.solver_config.polarization = polarization

        if wavelength_range is not None:
            w_start, w_stop = wavelength_range
            validate_wavelength_range(w_start, w_stop, max(2, points))
            self.wavelengths_m = np.linspace(w_start, w_stop, points)
        elif wavelength is not None:
            self.wavelengths_m = np.array([float(wavelength)])
        else:
            self.wavelengths_m = np.array([1.55e-6])

    def run(self, project: Any) -> StudyResult:
        solver = ModeSolver(self.solver_config, backend=self.solver_config.solver_backend)
        tracker = ModeTracker()
        results: List[SolverResult] = []

        geom = project.geometry
        materials_map = project.get_active_domain_materials()

        for idx, wl in enumerate(self.wavelengths_m):
            sol_res = solver.solve(
                geometry=geom,
                domain_materials=materials_map,
                wavelength_m=wl,
            )
            # Apply mode tracking
            tracker.track_step(sol_res, wl)
            results.append(sol_res)

        return StudyResult(
            study_name=self.name,
            results=results,
            parameters_swept={"wavelength_m": self.wavelengths_m.tolist()},
            metadata={
                "modes": self.modes,
                "polarization": self.polarization,
                "avoided_crossings": tracker.detect_avoided_crossings(),
            },
        )


class WavelengthSweepStudy(EigenmodeStudy):
    """Spectral sweep specifically tailored for SPR resonance and loss curve identification."""

    def __init__(
        self,
        start_wavelength_m: float = 600e-9,
        stop_wavelength_m: float = 1600e-9,
        num_points: int = 51,
        modes: int = 6,
        polarization: str = "both",
        name: str = "wavelength_sweep",
    ):
        super().__init__(
            wavelength_range=(start_wavelength_m, stop_wavelength_m),
            points=num_points,
            modes=modes,
            polarization=polarization,
            name=name,
        )


class AdaptiveWavelengthSweepStudy(BaseStudy):
    """Adaptive coarse-to-fine spectral sweep for research-grade SPR resonance resolution.
    
    Workflow:
    1. Run initial coarse sweep across the broad search window.
    2. Detect potential plasmonic resonance intervals from loss peak and prominence.
    3. Execute dense adaptive fine sweep around the detected resonance interval.
    4. Merge and sort all spectral points for high-density resonance fitting.
    """

    def __init__(
        self,
        start_wavelength_m: float = 850e-9,
        stop_wavelength_m: float = 1350e-9,
        coarse_points: int = 11,
        fine_window_nm: float = 120.0,
        fine_points: int = 15,
        modes: int = 2,
        polarization: str = "both",
        name: str = "adaptive_wavelength_sweep",
    ):
        super().__init__(name=name)
        self.start_wavelength_m = float(start_wavelength_m)
        self.stop_wavelength_m = float(stop_wavelength_m)
        self.coarse_points = int(coarse_points)
        self.fine_window_nm = float(fine_window_nm)
        self.fine_points = int(fine_points)
        self.modes = int(modes)
        self.polarization = polarization
        self.solver_config.num_modes = self.modes
        self.solver_config.polarization = polarization

    def run(self, project: Any) -> StudyResult:
        solver = ModeSolver(self.solver_config, backend=self.solver_config.solver_backend)
        tracker = ModeTracker()
        geom = project.geometry
        materials_map = project.get_active_domain_materials()

        # --- Stage 1: Coarse Sweep ---
        coarse_wls = np.linspace(self.start_wavelength_m, self.stop_wavelength_m, self.coarse_points)
        all_results_dict: Dict[float, SolverResult] = {}
        coarse_losses: List[float] = []

        for wl in coarse_wls:
            sol_res = solver.solve(geometry=geom, domain_materials=materials_map, wavelength_m=wl)
            all_results_dict[round(float(wl) * 1e11) / 1e11] = sol_res
            fund = sol_res.fundamental_mode
            loss_val = fund.loss_db_per_cm if fund else 0.0
            coarse_losses.append(loss_val)

        # Identify candidate resonance peak from coarse scan:
        # Prioritize interior peaks with prominence to avoid boundary edge cutoffs
        from scipy.signal import find_peaks
        coarse_arr = np.array(coarse_losses)
        interior_arr = np.array(coarse_losses, copy=True)
        interior_arr[0] = -np.inf
        interior_arr[-1] = -np.inf
        p_indices, _ = find_peaks(coarse_arr, prominence=0.5)
        interior_peaks = [p for p in p_indices if 0 < p < len(coarse_arr) - 1]
        if interior_peaks:
            best_idx = max(interior_peaks, key=lambda idx: coarse_arr[idx])
        elif len(coarse_arr) > 2:
            best_idx = int(np.argmax(interior_arr))
        else:
            best_idx = int(np.argmax(coarse_arr))
        res_wl_coarse = coarse_wls[best_idx]

        # --- Stage 2: Dense Fine Sweep around resonance ---
        half_win_m = (self.fine_window_nm * 1e-9) / 2.0
        fine_start = max(self.start_wavelength_m, res_wl_coarse - half_win_m)
        fine_stop = min(self.stop_wavelength_m, res_wl_coarse + half_win_m)
        fine_wls = np.linspace(fine_start, fine_stop, self.fine_points)

        for wl in fine_wls:
            key = round(float(wl) * 1e11) / 1e11
            # Avoid redundant re-simulation of coarse points
            already_solved = any(abs(k - key) < 1e-10 for k in all_results_dict)
            if not already_solved:
                sol_res = solver.solve(geometry=geom, domain_materials=materials_map, wavelength_m=wl)
                all_results_dict[key] = sol_res

        # --- Stage 3: Merge and Mode Track ---
        sorted_keys = sorted(list(all_results_dict.keys()))
        unified_results: List[SolverResult] = []

        for k in sorted_keys:
            res = all_results_dict[k]
            tracker.track_step(res, res.wavelength_m)
            unified_results.append(res)

        return StudyResult(
            study_name=self.name,
            results=unified_results,
            parameters_swept={"wavelength_m": sorted_keys},
            metadata={
                "is_adaptive": True,
                "coarse_points": self.coarse_points,
                "fine_points": self.fine_points,
                "fine_window_nm": self.fine_window_nm,
                "detected_coarse_resonance_nm": float(res_wl_coarse * 1e9),
                "total_simulated_points": len(unified_results),
                "avoided_crossings": tracker.detect_avoided_crossings(),
            },
        )

