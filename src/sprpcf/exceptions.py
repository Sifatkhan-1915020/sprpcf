"""Custom exception hierarchy for SPR-PCF Studio.

Provides descriptive scientific and numerical error diagnostics.
"""


class SPRPCFError(Exception):
    """Base exception for all SPR-PCF Studio errors."""
    pass


# --- Geometry Exceptions ---
class GeometryError(SPRPCFError):
    """Base exception for geometry construction and manipulation errors."""
    pass


class InvalidGeometryError(GeometryError):
    """Raised when geometry intersects invalidly, has zero area, or violates topological rules."""
    pass


class OverlappingRegionError(GeometryError):
    """Raised when two incompatible physical domains overlap without a defined hierarchy."""
    pass


# --- Material Exceptions ---
class MaterialError(SPRPCFError):
    """Base exception for material definition and evaluation errors."""
    pass


class MissingMaterialError(MaterialError):
    """Raised when a named material is requested but not defined in the project or database."""
    pass


class DispersionRangeError(MaterialError):
    """Raised when a material model is queried outside its validated spectral range."""
    pass


# --- Domain and Boundary Exceptions ---
class DomainAssignmentError(SPRPCFError):
    """Raised when computational regions are unassigned or assigned conflicting materials."""
    pass


class BoundaryConditionError(SPRPCFError):
    """Raised when boundary conditions are incompatible with geometry or solver."""
    pass


# --- Mesh Exceptions ---
class MeshGenerationError(SPRPCFError):
    """Raised when mesh generation fails in the active meshing backend."""
    pass


class InvalidMeshError(MeshGenerationError):
    """Raised when a generated mesh contains degenerate or inverted elements."""
    pass


class MeshConvergenceError(MeshGenerationError):
    """Raised when a mesh convergence study fails to reach user-specified tolerance."""
    pass


# --- Physics and Solver Exceptions ---
class PhysicsError(SPRPCFError):
    """Base exception for electromagnetic physics formulation errors."""
    pass


class SolverConfigurationError(PhysicsError):
    """Raised when solver parameters (tolerance, target index, shift) are invalid."""
    pass


class EigenvalueConvergenceError(PhysicsError):
    """Raised when eigensolver fails to converge within maximum iterations."""
    pass


class UnsupportedFeatureError(SPRPCFError):
    """Raised when a requested feature or backend is declared unsupported or not installed."""
    pass


# --- Mode Analysis and SPR Exceptions ---
class ModeAnalysisError(SPRPCFError):
    """Base exception for mode extraction and classification errors."""
    pass


class ModeTrackingError(ModeAnalysisError):
    """Raised when mode tracking loses mode continuity across parameter sweeps."""
    pass


class ResonanceDetectionError(SPRPCFError):
    """Raised when SPR resonance cannot be detected or is ambiguous."""
    pass


# --- Study and Simulation Exceptions ---
class InvalidStudyError(SPRPCFError):
    """Raised when study configuration is malformed or missing parameters."""
    pass


class SimulationError(SPRPCFError):
    """Raised during simulation workflow execution."""
    pass


class ResultValidationError(SPRPCFError):
    """Raised when simulation output fails scientific or numerical consistency checks."""
    pass

