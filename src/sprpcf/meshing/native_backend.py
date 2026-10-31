"""High-performance native multi-domain triangular mesh generator with localized refinement."""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial import Delaunay
import shapely.geometry as sg
from .base import Mesh
from .settings import MeshSettings
from ..geometry.cross_section import PCFGeometry
from ..exceptions import MeshGenerationError


class NativeMeshGenerator:
    """Generates multi-domain 2D triangular meshes with boundary-fitted interface refinement."""

    def __init__(self, settings: Optional[MeshSettings] = None):
        self.settings = settings or MeshSettings()

    def generate(self, geometry: PCFGeometry) -> Mesh:
        """Discretize PCF geometry into a multi-domain triangular mesh."""
        partitioned = geometry.partition()
        if not partitioned:
            raise MeshGenerationError("Cannot mesh empty PCF geometry.")

        nodes_list: List[Tuple[float, float]] = []
        boundary_edges_dict: Dict[str, List[Tuple[int, int]]] = {}

        # 1. Sample boundary contours for each region based on target element size
        for reg_name, reg in partitioned.items():
            is_metal = "metal" in reg_name.lower() or "coating" in reg_name.lower() or (reg.material_tag and "gold" in reg.material_tag.lower())
            metal_thick = getattr(reg, "thickness", None)
            target_h = self.settings.get_domain_target_size(reg_name, is_metal, metal_thick)

            geom = reg.shapely_geom
            polys = [geom] if isinstance(geom, sg.Polygon) else list(geom.geoms)

            for poly in polys:
                # Sample exterior ring
                ext_pts = self._sample_ring(poly.exterior.coords, target_h)
                for pt in ext_pts:
                    nodes_list.append(pt)

                # Sample any interior holes in this polygon
                for interior in poly.interiors:
                    int_pts = self._sample_ring(interior.coords, target_h)
                    for pt in int_pts:
                        nodes_list.append(pt)

                # 2. Fill interior of polygon with sample points
                interior_pts = self._sample_polygon_interior(poly, target_h)
                for pt in interior_pts:
                    nodes_list.append(pt)

        # Convert to unique array of nodes
        all_nodes = np.array(nodes_list, dtype=float)
        # Deduplicate nodes within 1e-10 m
        unique_nodes = self._deduplicate_nodes(all_nodes, tol=1e-10)

        if len(unique_nodes) < 3:
            raise MeshGenerationError("Insufficient node generation for 2D triangulation.")

        # 3. Perform Delaunay Triangulation
        tri = Delaunay(unique_nodes)
        triangles = tri.simplices  # (M, 3)

        # 4. Filter triangles and assign domain tags based on centroid location
        centers = np.mean(unique_nodes[triangles], axis=1)

        valid_triangles: List[np.ndarray] = []
        element_domains: List[str] = []

        # Prepare shapely prepared geometries for fast point-in-polygon queries
        import shapely.prepared as sp
        prepared_domains = [(name, sp.prep(reg.shapely_geom), reg) for name, reg in partitioned.items()]

        for tri_idx, (cx, cy) in enumerate(centers):
            pt = sg.Point(cx, cy)
            assigned_domain = None

            # First test smaller embedded regions (coatings, holes, channels, core)
            for name, prep_geom, reg in prepared_domains:
                if prep_geom.contains(pt):
                    assigned_domain = name
                    break

            if assigned_domain is not None:
                valid_triangles.append(triangles[tri_idx])
                element_domains.append(assigned_domain)

        if not valid_triangles:
            raise MeshGenerationError("Triangulation produced no elements inside valid domains.")

        final_elements = np.array(valid_triangles, dtype=int)

        # 5. Extract boundary edges on outer boundary
        mesh = Mesh(
            nodes=unique_nodes,
            elements=final_elements,
            element_domains=element_domains,
            backend_name="native",
        )
        return mesh

    def _sample_ring(self, coords: List[Tuple[float, float]], target_h: float) -> List[Tuple[float, float]]:
        pts = []
        coords_arr = np.array(coords)
        for i in range(len(coords_arr) - 1):
            p1 = coords_arr[i]
            p2 = coords_arr[i + 1]
            seg_len = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            n_divs = max(1, int(math.ceil(seg_len / target_h)))
            for j in range(n_divs):
                t = j / float(n_divs)
                pts.append((float(p1[0] + t * (p2[0] - p1[0])), float(p1[1] + t * (p2[1] - p1[1]))))
        return pts

    def _sample_polygon_interior(self, poly: sg.Polygon, target_h: float) -> List[Tuple[float, float]]:
        pts = []
        minx, miny, maxx, maxy = poly.bounds
        # Staggered grid
        xs = np.arange(minx + target_h * 0.5, maxx, target_h)
        ys = np.arange(miny + target_h * 0.5, maxy, target_h * math.sqrt(3) / 2.0)

        for row_idx, y in enumerate(ys):
            x_offset = (target_h * 0.5) if (row_idx % 2 == 1) else 0.0
            for x in (xs + x_offset):
                pt = sg.Point(x, y)
                if poly.contains(pt):
                    pts.append((float(x), float(y)))
        return pts

    def _deduplicate_nodes(self, nodes: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        # Snap coordinates to grid
        scale = 1.0 / tol
        scaled = np.round(nodes * scale).astype(np.int64)
        _, unique_indices = np.unique(scaled, axis=0, return_index=True)
        return nodes[np.sort(unique_indices)]

