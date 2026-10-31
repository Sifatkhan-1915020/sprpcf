"""Optimization convergence history and Pareto front calculation."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class OptimizationHistory:
    """Stores full optimization trial history for inverse design."""

    def __init__(self):
        self.trials: List[Dict[str, Any]] = []

    def record_trial(self, iteration: int, parameters: Dict[str, float], objective_value: float, metrics: Dict[str, Any]) -> None:
        self.trials.append({
            "iteration": iteration,
            "parameters": dict(parameters),
            "objective_value": float(objective_value),
            "metrics": dict(metrics),
        })

    @property
    def best_trial(self) -> Optional[Dict[str, Any]]:
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t["objective_value"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluations": len(self.trials),
            "best_trial": self.best_trial,
            "trials": self.trials,
        }


def compute_pareto_front(
    objectives_matrix: np.ndarray,  # shape (N, D) where higher is better for all D
) -> np.ndarray:
    """Identify Pareto optimal non-dominated indices.
    
    A point x dominates y if x_d >= y_d for all d, and x_k > y_k for at least one k.
    Returns boolean array of shape (N,) indicating Pareto non-dominated points.
    """
    n_points = objectives_matrix.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_pareto[i]:
            continue
        # Check if i is dominated by any other active candidate j
        for j in range(n_points):
            if i != j and is_pareto[j]:
                if np.all(objectives_matrix[j] >= objectives_matrix[i]) and np.any(objectives_matrix[j] > objectives_matrix[i]):
                    is_pareto[i] = False
                    break

    return is_pareto
