"""Gmsh mesh generation backend for SPR-PCF Studio."""

from typing import Optional, Dict, List, Tuple
import numpy as np
from .base import Mesh
from .settings import MeshSettings
from ..geometry.cross_section import PCFGeometry
from ..exceptions import MeshGenerationError


class GmshMeshGenerator:
    """Meshes 2D PCF geometries using the Gmsh Python API."""

    def __init__(self, settings: Optional[MeshSettings] = None):
        self.settings = settings or MeshSettings()

    def generate(self, geometry: PCFGeometry) -> Mesh:
        try:
            import gmsh
        except ImportError:
            raise MeshGenerationError("Gmsh Python API is not installed. Install with 'pip install gmsh'.")

        partitioned = geometry.partition()
        if not partitioned:
            raise MeshGenerationError("Cannot mesh empty PCF geometry.")

        try:
            if not gmsh.isInitialized():
                gmsh.initialize()
            gmsh.model.add("spr_pcf_model")

            # Configure Gmsh meshing algorithm
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.settings.min_size or 1e-8)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.settings.max_size or 1e-6)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 32)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D

            # Build surfaces for each partitioned region via OCC
            domain_tags = {}
            for name, reg in partitioned.items():
                geom = reg.shapely_geom
                poly = geom if geom.geom_type == "Polygon" else list(geom.geoms)[0]

                # Points for exterior
                pt_tags = []
                coords = list(poly.exterior.coords)[:-1]
                target_lc = self.settings.get_domain_target_size(name)
                for x, y in coords:
                    p_tag = gmsh.model.occ.addPoint(x, y, 0.0, target_lc)
                    pt_tags.append(p_tag)

                lines = []
                n_pts = len(pt_tags)
                for i in range(n_pts):
                    l_tag = gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % n_pts])
                    lines.append(l_tag)

                loop = gmsh.model.occ.addCurveLoop(lines)
                surf = gmsh.model.occ.addPlaneSurface([loop])
                domain_tags[surf] = name

            gmsh.model.occ.synchronize()

            # Assign Physical Groups
            for surf, name in domain_tags.items():
                gmsh.model.addPhysicalGroup(2, [surf], tag=-1, name=name)

            # Generate 2D mesh
            gmsh.model.mesh.generate(2)

            # Extract nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes_2d = node_coords.reshape(-1, 3)[:, :2]
            tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}

            # Extract triangular elements per physical group
            elements_list = []
            element_domains = []

            for surf, name in domain_tags.items():
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
                for e_type, e_nodes in zip(elem_types, elem_node_tags):
                    if e_type == 2:  # 3-node triangle
                        tris = e_nodes.reshape(-1, 3)
                        for tri in tris:
                            tri_indices = [tag_to_idx[t] for t in tri]
                            elements_list.append(tri_indices)
                            element_domains.append(name)

            return Mesh(
                nodes=nodes_2d,
                elements=np.array(elements_list, dtype=int),
                element_domains=element_domains,
                backend_name="gmsh",
            )
        except Exception as e:
            raise MeshGenerationError(f"Gmsh meshing failed: {e}")
        finally:
            try:
                gmsh.clear()
            except Exception:
                pass

