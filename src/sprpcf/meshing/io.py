"""Mesh input/output for standard scientific file formats (MSH, VTU, VTK)."""

from pathlib import Path
from typing import Union
import numpy as np
from .base import Mesh
from ..exceptions import MeshGenerationError


def export_mesh(mesh: Mesh, filepath: Union[str, Path]) -> None:
    """Export mesh to file (e.g. .msh, .vtu, .vtk) using meshio."""
    try:
        import meshio
    except ImportError:
        raise MeshGenerationError("meshio is required for mesh file export. Install via 'pip install meshio'.")

    path = Path(filepath)
    # Convert 2D nodes to 3D for meshio (x, y, 0.0)
    n_nodes = mesh.num_nodes
    points_3d = np.zeros((n_nodes, 3), dtype=float)
    points_3d[:, :2] = mesh.nodes

    cell_type = "triangle" if mesh.elements.shape[1] == 3 else "quad"
    cells = [(cell_type, mesh.elements)]

    # Map string domain tags to integer cell data
    unique_domains = mesh.get_unique_domains()
    domain_to_int = {d: i for i, d in enumerate(unique_domains)}
    cell_tags = np.array([domain_to_int[d] for d in mesh.element_domains], dtype=np.int32)

    mesh_data = meshio.Mesh(
        points=points_3d,
        cells=cells,
        cell_data={"domain": [cell_tags]},
    )
    meshio.write(str(path), mesh_data)

