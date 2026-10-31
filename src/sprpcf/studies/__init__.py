"""Simulation study framework."""

from .base import BaseStudy, StudyResult
from .eigenmode import EigenmodeStudy, WavelengthSweepStudy, AdaptiveWavelengthSweepStudy
from .analyte_sweep import AnalyteSweepStudy, ParameterSweepStudy
from .convergence import MeshConvergenceStudy
from ..core.registry import studies_registry

# Register study types
studies_registry.register("eigenmode", EigenmodeStudy)
studies_registry.register("wavelength_sweep", WavelengthSweepStudy)
studies_registry.register("adaptive_wavelength_sweep", AdaptiveWavelengthSweepStudy)
studies_registry.register("analyte_sweep", AnalyteSweepStudy)
studies_registry.register("parameter_sweep", ParameterSweepStudy)
studies_registry.register("mesh_convergence", MeshConvergenceStudy)

__all__ = [
    "BaseStudy",
    "StudyResult",
    "EigenmodeStudy",
    "WavelengthSweepStudy",
    "AdaptiveWavelengthSweepStudy",
    "AnalyteSweepStudy",
    "ParameterSweepStudy",
    "MeshConvergenceStudy",
]

