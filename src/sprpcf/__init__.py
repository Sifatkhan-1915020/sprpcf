"""SPR-PCF Studio: A Universal Research-Grade Python Library for SPR Biosensor Design,
Geometry Modeling, Material Engineering, Meshing, FEM Simulation, Mode Analysis,
Visualization, Optimization and Research Reporting.
"""

from .version import __version__, __author__, __license__
from .capabilities import capabilities, CapabilityStatus
from .exceptions import (
    SPRPCFError,
    GeometryError,
    InvalidGeometryError,
    MaterialError,
    MissingMaterialError,
    MeshGenerationError,
    InvalidMeshError,
    MeshConvergenceError,
    SolverConfigurationError,
    EigenvalueConvergenceError,
    ModeTrackingError,
    ResonanceDetectionError,
    SimulationError,
)

# Core
from .core.constants import C0, SPEED_OF_LIGHT, EPSILON_0, MU_0, ETA_0, NEPER_TO_DB
from .core.units import LengthUnit, wavelength_to_frequency, frequency_to_wavelength
from .core.parameters import Parameter, ParameterSet
from .core.project import Project
from .core.configuration import load_config_file, SimulationConfig

# Geometry
from .geometry.base import Region2D
from .geometry.primitives import Circle, Ellipse, Rectangle, RegularPolygon, Annulus
from .geometry.holes import Hole
from .geometry.coatings import CoatingLayer
from .geometry.channels import AnalyteChannel
from .geometry.lattices import LatticeGenerator
from .geometry.cross_section import PCFGeometry
from .geometry.pcf_templates import ConventionalPCF, DShapedPCF, DualChannelPCF, HollowCorePCF
from .geometry.validation import validate_geometry, GeometryValidationReport

# Materials
from .materials.base import Material, OpticalConstants
from .materials.dispersion import (
    ConstantIndexMaterial,
    SellmeierMaterial,
    CauchyMaterial,
    DrudeLorentzMaterial,
    TabulatedMaterial,
)
from .materials.biological import AnalyteMaterial, WaterMaterial, GlucoseSolutionMaterial
from .materials.database import materials, MaterialDatabase

# Meshing
from .meshing.base import Mesh
from .meshing.settings import MeshSettings
from .meshing.validation import validate_mesh, MeshQualityReport
from .meshing import generate_mesh

# Solvers & Physics
from .physics.electromagnetic_fields import ElectromagneticField2D
from .solvers.base import OpticalMode, SolverConfig, SolverResult
from .solvers.mode_solver import ModeSolver
from .solvers.mode_tracking import ModeTracker, TrackedModeBranch

# Studies
from .studies.base import BaseStudy, StudyResult
from .studies.eigenmode import EigenmodeStudy, WavelengthSweepStudy, AdaptiveWavelengthSweepStudy
from .studies.analyte_sweep import AnalyteSweepStudy, ParameterSweepStudy
from .studies.convergence import MeshConvergenceStudy

# SPR Analysis
from .spr.analyzer import SPRAnalyzer
from .spr.resonance import ResonancePeak, extract_resonance_peaks, fit_resonance_curve, ResonanceFitResult
from .spr.loss import LossSpectrum
from .spr.sensitivity import SensitivityReport, calculate_amplitude_sensitivity_spectrum
from .spr.figure_of_merit import SensorPerformanceMetrics

# Optimization
from .optimization.algorithms import PCFOptimizer

# Visualization
from .visualization.geometry import plot_geometry
from .visualization.mesh import plot_mesh
from .visualization.fields import plot_mode_field, plot_all_field_components
from .visualization.spectra import plot_loss_spectrum, plot_dispersion_curves
from .visualization.performance import plot_sensitivity_fit, plot_amplitude_sensitivity, plot_sensitivity_vs_ri
from .visualization.dashboard import create_research_dashboard, InteractiveScientificDashboard
from .visualization.reports import generate_html_report

from . import results

# Convenience alias
plot_mode_distribution = plot_mode_field

__all__ = [
    "__version__",
    "capabilities",
    "CapabilityStatus",
    "Project",
    "PCFGeometry",
    "ConventionalPCF",
    "DShapedPCF",
    "DualChannelPCF",
    "HollowCorePCF",
    "validate_geometry",
    "GeometryValidationReport",
    "Hole",
    "CoatingLayer",
    "AnalyteChannel",
    "LatticeGenerator",
    "Circle",
    "Ellipse",
    "Rectangle",
    "RegularPolygon",
    "Annulus",
    "Material",
    "OpticalConstants",
    "ConstantIndexMaterial",
    "SellmeierMaterial",
    "CauchyMaterial",
    "DrudeLorentzMaterial",
    "TabulatedMaterial",
    "materials",
    "MaterialDatabase",
    "Mesh",
    "MeshSettings",
    "generate_mesh",
    "OpticalMode",
    "SolverConfig",
    "SolverResult",
    "ModeSolver",
    "ModeTracker",
    "TrackedModeBranch",
    "ElectromagneticField2D",
    "BaseStudy",
    "StudyResult",
    "EigenmodeStudy",
    "WavelengthSweepStudy",
    "AdaptiveWavelengthSweepStudy",
    "AnalyteSweepStudy",
    "ParameterSweepStudy",
    "MeshConvergenceStudy",
    "SPRAnalyzer",
    "ResonancePeak",
    "extract_resonance_peaks",
    "fit_resonance_curve",
    "ResonanceFitResult",
    "LossSpectrum",
    "SensitivityReport",
    "calculate_amplitude_sensitivity_spectrum",
    "SensorPerformanceMetrics",
    "PCFOptimizer",
    "plot_geometry",
    "plot_mesh",
    "plot_mode_field",
    "plot_mode_distribution",
    "plot_all_field_components",
    "plot_loss_spectrum",
    "plot_dispersion_curves",
    "plot_sensitivity_fit",
    "plot_sensitivity_vs_ri",
    "plot_amplitude_sensitivity",
    "create_research_dashboard",
    "InteractiveScientificDashboard",
    "generate_html_report",
    "load_config_file",
    "SimulationConfig",
    "C0",
    "SPEED_OF_LIGHT",
    "EPSILON_0",
    "MU_0",
    "ETA_0",
    "NEPER_TO_DB",
]
