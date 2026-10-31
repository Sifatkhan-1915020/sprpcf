"""Meshing framework for SPR-PCF Studio."""

from .base import Mesh
from .settings import MeshSettings
from .validation import MeshQualityReport, validate_mesh
from .native_backend import NativeMeshGenerator
from .gmsh_backend import GmshMeshGenerator
from .netgen_backend import NetgenMeshGenerator
from .io import export_mesh
from ..core.registry import meshers_registry

# Register meshers in global registry
meshers_registry.register("native", NativeMeshGenerator)
meshers_registry.register("gmsh", GmshMeshGenerator)
meshers_registry.register("netgen", NetgenMeshGenerator)


def generate_mesh(geometry, settings: MeshSettings = None, backend: str = "native") -> Mesh:
    """Convenience function to generate a mesh for a PCFGeometry."""
    s = settings or MeshSettings()
    b = backend.lower().strip()
    if b == "gmsh":
        return GmshMeshGenerator(s).generate(geometry)
    elif b == "netgen":
        return NetgenMeshGenerator(s).generate(geometry)
    else:
        return NativeMeshGenerator(s).generate(geometry)


__all__ = [
    "Mesh",
    "MeshSettings",
    "MeshQualityReport",
    "validate_mesh",
    "NativeMeshGenerator",
    "GmshMeshGenerator",
    "NetgenMeshGenerator",
    "export_mesh",
    "generate_mesh",
]

