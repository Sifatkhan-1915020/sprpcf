"""SPR analysis and biosensing performance module."""

from .resonance import ResonancePeak, extract_resonance_peaks
from .loss import LossSpectrum
from .phase_matching import PhaseMatchingAnalysis
from .sensitivity import SensitivityReport, calculate_amplitude_sensitivity_spectrum
from .figure_of_merit import SensorPerformanceMetrics
from .analyzer import SPRAnalyzer

__all__ = [
    "ResonancePeak",
    "extract_resonance_peaks",
    "LossSpectrum",
    "PhaseMatchingAnalysis",
    "SensitivityReport",
    "calculate_amplitude_sensitivity_spectrum",
    "SensorPerformanceMetrics",
    "SPRAnalyzer",
]

