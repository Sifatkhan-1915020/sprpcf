"""Global and local optimization algorithms for PCF inverse design."""

from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .history import OptimizationHistory
from .objectives import Objective


class PCFOptimizer:
    """Manages parametric optimization runs over PCF designs."""

    def __init__(
        self,
        project: Any,
        parameter_bounds: Dict[str, Tuple[float, float]],
        evaluation_fn: Callable[[Any], float],
        maximize: bool = True,
    ):
        self.project = project
        self.param_names = list(parameter_bounds.keys())
        self.bounds = [parameter_bounds[k] for k in self.param_names]
        self.evaluation_fn = evaluation_fn
        self.maximize = maximize
        self.history = OptimizationHistory()
        self._iteration = 0

    def _cost_fn(self, param_vector: np.ndarray) -> float:
        self._iteration += 1
        param_dict = {name: float(val) for name, val in zip(self.param_names, param_vector)}

        # Update project parameters
        for k, v in param_dict.items():
            self.project.parameters.set(k, v)

        # Evaluate objective
        raw_score = float(self.evaluation_fn(self.project))
        cost = -raw_score if self.maximize else raw_score

        self.history.record_trial(
            iteration=self._iteration,
            parameters=param_dict,
            objective_value=raw_score,
            metrics={"cost": cost},
        )
        return cost

    def run_differential_evolution(
        self,
        maxiter: int = 20,
        popsize: int = 5,
        tol: float = 1e-2,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Global optimization via SciPy Differential Evolution."""
        res = differential_evolution(
            func=self._cost_fn,
            bounds=self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            seed=seed,
            polish=False,
        )

        best_params = {name: float(val) for name, val in zip(self.param_names, res.x)}
        best_val = -float(res.fun) if self.maximize else float(res.fun)

        return {
            "algorithm": "differential_evolution",
            "success": res.success,
            "message": res.message,
            "best_parameters": best_params,
            "best_objective_value": best_val,
            "evaluations_count": res.nfev,
            "history": self.history.to_dict(),
        }

    def run_grid_search(self, steps_per_param: int = 3) -> Dict[str, Any]:
        """Exhaustive grid search across bounded parameter space."""
        grid_axes = [np.linspace(b[0], b[1], steps_per_param) for b in self.bounds]
        import itertools

        best_score = -1e12 if self.maximize else 1e12
        best_params = None

        for pt in itertools.product(*grid_axes):
            cost = self._cost_fn(np.array(pt))
            score = -cost if self.maximize else cost
            if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                best_score = score
                best_params = {name: float(val) for name, val in zip(self.param_names, pt)}

        return {
            "algorithm": "grid_search",
            "best_parameters": best_params,
            "best_objective_value": best_score,
            "evaluations_count": self._iteration,
            "history": self.history.to_dict(),
        }

