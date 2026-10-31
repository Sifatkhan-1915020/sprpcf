"""Optimization and inverse design engine."""

from .objectives import Objective, objective_maximize_sensitivity, objective_maximize_fom, objective_minimize_fwhm
from .history import OptimizationHistory, compute_pareto_front
from .algorithms import PCFOptimizer

__all__ = [
    "Objective",
    "objective_maximize_sensitivity",
    "objective_maximize_fom",
    "objective_minimize_fwhm",
    "OptimizationHistory",
    "compute_pareto_front",
    "PCFOptimizer",
]

