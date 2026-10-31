"""Optimization objective functions and constraint evaluators."""

from typing import Callable, Any, Dict, Optional


class Objective:
    """Represents a scalar or composite optimization target."""

    def __init__(
        self,
        name: str,
        evaluate_fn: Callable[[Any], float],
        maximize: bool = True,
        weight: float = 1.0,
    ):
        self.name = name
        self.evaluate_fn = evaluate_fn
        self.maximize = maximize
        self.weight = float(weight)

    def evaluate(self, result: Any) -> float:
        val = float(self.evaluate_fn(result))
        return val if self.maximize else -val


def objective_maximize_sensitivity(analyte_summary: Dict[str, Any]) -> float:
    """Objective: Maximize wavelength sensitivity (nm/RIU)."""
    return analyte_summary.get("sensitivity", {}).get("linear_wavelength_sensitivity_nm_per_riu", 0.0)


def objective_maximize_fom(analyte_summary: Dict[str, Any]) -> float:
    """Objective: Maximize Figure of Merit (FOM)."""
    return analyte_summary.get("performance", {}).get("figure_of_merit_riu_inv", 0.0)


def objective_minimize_fwhm(analyte_summary: Dict[str, Any]) -> float:
    """Objective: Minimize resonance linewidth (FWHM in nm)."""
    fwhm = analyte_summary.get("fwhm_nm_mean", 50.0)
    return -float(fwhm)

