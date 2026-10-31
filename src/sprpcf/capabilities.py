"""Capability registry for SPR-PCF Studio.

Tracks status of physics engines, solvers, mesh backends, material models,
and studies with explicit validation levels.
"""

from enum import Enum
from typing import Dict, Any, List, Optional


class CapabilityStatus(str, Enum):
    IMPLEMENTED_VALIDATED = "implemented_validated"
    IMPLEMENTED_PENDING_VALIDATION = "implemented_pending_validation"
    OPTIONAL_BACKEND = "optional_backend"
    PLANNED = "planned"
    UNSUPPORTED = "unsupported"


class Capability:
    """Represents a specific system capability."""

    def __init__(
        self,
        name: str,
        category: str,
        status: CapabilityStatus,
        description: str,
        backend: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        self.name = name
        self.category = category
        self.status = status
        self.description = description
        self.backend = backend
        self.notes = notes or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
            "description": self.description,
            "backend": self.backend,
            "notes": self.notes,
        }


class CapabilityRegistry:
    """Registry maintaining active and declared capabilities."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._capabilities = {}
            cls._instance._register_default_capabilities()
        return cls._instance

    def register(self, capability: Capability) -> None:
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def is_available(self, name: str) -> bool:
        cap = self.get(name)
        if not cap:
            return False
        return cap.status in (
            CapabilityStatus.IMPLEMENTED_VALIDATED,
            CapabilityStatus.IMPLEMENTED_PENDING_VALIDATION,
            CapabilityStatus.OPTIONAL_BACKEND,
        )

    def list_all(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._capabilities.values()]

    def list_by_status(self, status: CapabilityStatus) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._capabilities.values() if c.status == status]

    def _register_default_capabilities(self) -> None:
        defaults = [
            # Geometry
            Capability("2d_parametric_geometry", "geometry", CapabilityStatus.IMPLEMENTED_VALIDATED, "2D transverse cross-section geometry engine with boolean ops"),
            Capability("hexagonal_lattice", "geometry", CapabilityStatus.IMPLEMENTED_VALIDATED, "Hexagonal PCF lattice generator"),
            Capability("rectangular_lattice", "geometry", CapabilityStatus.IMPLEMENTED_VALIDATED, "Rectangular PCF lattice generator"),
            Capability("circular_ring_lattice", "geometry", CapabilityStatus.IMPLEMENTED_VALIDATED, "Concentric circular ring lattice generator"),
            Capability("d_shaped_fiber", "geometry", CapabilityStatus.IMPLEMENTED_VALIDATED, "D-shaped side-polished fiber architecture"),
            Capability("microfluidic_channel", "geometry", CapabilityStatus.IMPLEMENTED_VALIDATED, "Dedicated analyte sensing microchannels"),
            Capability("thin_film_coating", "geometry", CapabilityStatus.IMPLEMENTED_VALIDATED, "Volumetric nanoscale plasmonic coating geometry"),
            Capability("3d_longitudinal_geometry", "geometry", CapabilityStatus.PLANNED, "3D longitudinal fiber geometry (planned for full-wave)"),

            # Materials
            Capability("sellmeier_dispersion", "materials", CapabilityStatus.IMPLEMENTED_VALIDATED, "Sellmeier glass dispersion model with temperature support"),
            Capability("drude_lorentz_metal", "materials", CapabilityStatus.IMPLEMENTED_VALIDATED, "Drude and Drude-Lorentz optical constants for Au, Ag, Cu, Al, etc."),
            Capability("tabulated_optical_constants", "materials", CapabilityStatus.IMPLEMENTED_VALIDATED, "Tabulated complex refractive index with spline/linear interpolation"),
            Capability("cauchy_dispersion", "materials", CapabilityStatus.IMPLEMENTED_VALIDATED, "Cauchy dispersion model for transparent dielectrics/polymers"),
            Capability("graphene_surface_conductivity", "materials", CapabilityStatus.IMPLEMENTED_PENDING_VALIDATION, "Kubo formula surface conductivity for monolayer graphene"),

            # Meshing
            Capability("netgen_mesh_backend", "meshing", CapabilityStatus.OPTIONAL_BACKEND, "High-order triangular mesh generation via Netgen/NGSolve", backend="netgen"),
            Capability("gmsh_mesh_backend", "meshing", CapabilityStatus.OPTIONAL_BACKEND, "Gmsh CAD & finite element mesh generation", backend="gmsh"),
            Capability("native_fem_mesher", "meshing", CapabilityStatus.IMPLEMENTED_VALIDATED, "Internal adaptive multi-region mesh generator with interface refinement"),
            Capability("boundary_layer_refinement", "meshing", CapabilityStatus.IMPLEMENTED_VALIDATED, "Graded boundary-layer refinement for plasmonic interfaces"),

            # Solvers
            Capability("vector_maxwell_fem", "physics", CapabilityStatus.IMPLEMENTED_VALIDATED, "Full-vector frequency-domain Maxwell transverse eigenmode solver"),
            Capability("complex_coordinate_pml", "physics", CapabilityStatus.IMPLEMENTED_VALIDATED, "Cylindrical and Cartesian complex stretched-coordinate PML"),
            Capability("ngsolve_fem_backend", "solvers", CapabilityStatus.OPTIONAL_BACKEND, "NGSolve H(curl) edge-element eigensolver", backend="ngsolve"),
            Capability("sparse_shift_invert_eigensolver", "solvers", CapabilityStatus.IMPLEMENTED_VALIDATED, "ARPACK / Arnoldi shift-and-invert complex non-Hermitian eigensolver"),

            # Mode Analysis
            Capability("mode_classification", "mode_analysis", CapabilityStatus.IMPLEMENTED_VALIDATED, "Fundamental, SPP, core, leaky, quasi-TE, quasi-TM classification"),
            Capability("polarization_fraction", "mode_analysis", CapabilityStatus.IMPLEMENTED_VALIDATED, "X/Y polarization fractions and birefringence extraction"),
            Capability("mode_tracking", "mode_analysis", CapabilityStatus.IMPLEMENTED_VALIDATED, "Modal field-overlap (Hermitian inner product) mode tracking across sweeps"),
            Capability("avoided_crossing_detection", "mode_analysis", CapabilityStatus.IMPLEMENTED_VALIDATED, "Automatic detection of core-SPP anti-crossing and hybridization"),

            # SPR Analysis
            Capability("confinement_loss", "spr", CapabilityStatus.IMPLEMENTED_VALIDATED, "Confinement and absorption loss in dB/cm, dB/m, dB/um"),
            Capability("resonance_detection", "spr", CapabilityStatus.IMPLEMENTED_VALIDATED, "Multi-peak resonance extraction, prominence, FWHM, and Q-factor"),
            Capability("wavelength_sensitivity", "spr", CapabilityStatus.IMPLEMENTED_VALIDATED, "Wavelength interrogation sensitivity in nm/RIU with linear/polynomial regression"),
            Capability("amplitude_sensitivity", "spr", CapabilityStatus.IMPLEMENTED_VALIDATED, "Amplitude interrogation sensitivity in RIU^-1"),
            Capability("figure_of_merit", "spr", CapabilityStatus.IMPLEMENTED_VALIDATED, "FOM (S_lambda / FWHM) and sensor resolution evaluation"),

            # Studies & Optimization
            Capability("wavelength_sweep", "studies", CapabilityStatus.IMPLEMENTED_VALIDATED, "Spectral wavelength sweep with adaptive interpolation"),
            Capability("analyte_sweep", "studies", CapabilityStatus.IMPLEMENTED_VALIDATED, "Analyte refractive index sweep for sensitivity characterization"),
            Capability("parameter_sweep", "studies", CapabilityStatus.IMPLEMENTED_VALIDATED, "Multivariate parameter sweeps (pitch, diameter, coating thickness)"),
            Capability("mesh_convergence_study", "studies", CapabilityStatus.IMPLEMENTED_VALIDATED, "Systematic mesh refinement and convergence reporting"),
            Capability("differential_evolution_opt", "optimization", CapabilityStatus.IMPLEMENTED_VALIDATED, "Global optimization of PCF structural parameters"),
            Capability("pareto_multiobjective_opt", "optimization", CapabilityStatus.IMPLEMENTED_VALIDATED, "Multi-objective trade-off analysis (Sensitivity vs. Loss/FOM)"),
        ]
        for cap in defaults:
            self.register(cap)


capabilities = CapabilityRegistry()

