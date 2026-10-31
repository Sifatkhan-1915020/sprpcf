"""Modal overlap-based mode tracking across parameter sweeps with anti-crossing detection."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from .base import OpticalMode, SolverResult
from ..physics.poynting import calculate_modal_overlap
from ..exceptions import ModeTrackingError


class TrackedModeBranch:
    """Represents a continuous modal branch tracked across sweep steps."""

    def __init__(self, branch_id: int, initial_mode: OpticalMode, initial_param_val: float):
        self.branch_id = branch_id
        self.modes: List[OpticalMode] = [initial_mode]
        self.param_values: List[float] = [float(initial_param_val)]

    def append(self, mode: OpticalMode, param_val: float) -> None:
        self.modes.append(mode)
        self.param_values.append(float(param_val))

    @property
    def latest_mode(self) -> OpticalMode:
        return self.modes[-1]

    @property
    def n_eff_curve(self) -> np.ndarray:
        return np.array([m.n_eff for m in self.modes])

    @property
    def n_real_curve(self) -> np.ndarray:
        return np.array([m.n_real for m in self.modes])

    @property
    def loss_db_cm_curve(self) -> np.ndarray:
        return np.array([m.loss_db_per_cm for m in self.modes])

    @property
    def peak_loss_info(self) -> Tuple[float, float, int]:
        """Return (peak_loss_db_cm, resonance_param_val, step_index)."""
        losses = self.loss_db_cm_curve
        idx = int(np.argmax(losses))
        return float(losses[idx]), float(self.param_values[idx]), idx


class ModeTracker:
    """Tracks optical modes across sweep steps using spatial field overlap."""

    def __init__(self, min_overlap_threshold: float = 0.40):
        self.min_overlap_threshold = float(min_overlap_threshold)
        self.branches: List[TrackedModeBranch] = []

    def initialize(self, first_result: SolverResult, param_val: float) -> None:
        """Initialize branches from the first simulation step."""
        self.branches = []
        for i, mode in enumerate(first_result.modes):
            branch = TrackedModeBranch(branch_id=i, initial_mode=mode, initial_param_val=param_val)
            self.branches.append(branch)

    def track_step(self, next_result: SolverResult, param_val: float) -> List[Tuple[int, int, float]]:
        """Match modes from next_result to active branches by maximizing field overlap.
        
        Returns list of (branch_id, mode_id, overlap_value).
        """
        if not self.branches:
            self.initialize(next_result, param_val)
            return [(b.branch_id, b.latest_mode.mode_id, 1.0) for b in self.branches]

        n_branches = len(self.branches)
        n_modes = len(next_result.modes)

        # Compute overlap matrix (n_branches, n_modes)
        overlap_matrix = np.zeros((n_branches, n_modes), dtype=float)
        for b_idx, branch in enumerate(self.branches):
            prev_field = branch.latest_mode.field
            for m_idx, mode in enumerate(next_result.modes):
                if prev_field is not None and mode.field is not None:
                    overlap_matrix[b_idx, m_idx] = calculate_modal_overlap(prev_field, mode.field)
                else:
                    # Index proximity fallback if field uncomputed: 1 / (1 + 100*|dn|)
                    dn = abs(branch.latest_mode.n_real - mode.n_real)
                    overlap_matrix[b_idx, m_idx] = 1.0 / (1.0 + 100.0 * dn)

        # Greedy or Hungarian matching to assign modes to branches
        assigned_modes = set()
        matches = []

        for b_idx in range(n_branches):
            best_m_idx = -1
            best_val = -1.0
            for m_idx in range(n_modes):
                if m_idx not in assigned_modes:
                    val = overlap_matrix[b_idx, m_idx]
                    if val > best_val:
                        best_val = val
                        best_m_idx = m_idx

            if best_m_idx >= 0 and best_val >= self.min_overlap_threshold:
                matched_mode = next_result.modes[best_m_idx]
                self.branches[b_idx].append(matched_mode, param_val)
                assigned_modes.add(best_m_idx)
                matches.append((b_idx, matched_mode.mode_id, float(best_val)))
            else:
                # Mode branch disappeared or dropped below threshold; append closest index
                closest_m = min(next_result.modes, key=lambda m: abs(m.n_real - self.branches[b_idx].latest_mode.n_real))
                self.branches[b_idx].append(closest_m, param_val)
                matches.append((b_idx, closest_m.mode_id, 0.0))

        return matches

    def detect_avoided_crossings(self) -> List[Dict[str, Any]]:
        """Detect avoided mode crossings and hybridization points between branches.
        
        An avoided crossing occurs when two mode curves approach a minimum spectral gap
        accompanied by an exchange of field characteristics and loss transfer.
        """
        events = []
        n_branches = len(self.branches)
        if n_branches < 2:
            return events

        for i in range(n_branches):
            for j in range(i + 1, n_branches):
                b1, b2 = self.branches[i], self.branches[j]
                if len(b1.param_values) != len(b2.param_values) or len(b1.param_values) < 3:
                    continue

                diff_n = np.abs(b1.n_real_curve - b2.n_real_curve)
                min_idx = int(np.argmin(diff_n))
                min_gap = float(diff_n[min_idx])

                # Check if it is a local minimum (anti-crossing) rather than boundary
                if 0 < min_idx < len(diff_n) - 1:
                    if diff_n[min_idx] < diff_n[min_idx - 1] and diff_n[min_idx] < diff_n[min_idx + 1]:
                        if min_gap < 0.02:  # Close approach
                            events.append({
                                "branch_1": i,
                                "branch_2": j,
                                "param_val": b1.param_values[min_idx],
                                "min_index_gap": min_gap,
                                "step_index": min_idx,
                                "loss_branch_1_db_cm": float(b1.loss_db_cm_curve[min_idx]),
                                "loss_branch_2_db_cm": float(b2.loss_db_cm_curve[min_idx]),
                            })
        return events

