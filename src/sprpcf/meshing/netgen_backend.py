"""Netgen 2D meshing backend."""

from typing import Optional, Dict, List
import numpy as np
from .base import Mesh
from .settings import MeshSettings
from ..geometry.cross_section import PCFGeometry
from ..exceptions import MeshGenerationError


class NetgenMeshGenerator:
    """Meshes 2D PCF geometries using Netgen CSG/Spline2d or OCC."""

    def __init__(self, settings: Optional[MeshSettings] = None):
        self.settings = settings or MeshSettings()

    def generate(self, geometry: PCFGeometry) -> Mesh:
        try:
            from netgen.geom2d import SplineGeometry
        except ImportError:
            raise MeshGenerationError("Netgen is not installed. Install with 'pip install ngsolve netgen-mesher'.")

        partitioned = geometry.partition()
        if not partitioned:
            raise MeshGenerationError("Cannot mesh empty PCF geometry.")

        try:
            geo = SplineGeometry()
            # Add points and lines for each region in SplineGeometry
            domain_idx = 1
            domain_map = {}
            for name, reg in partitioned.items():
                geom = reg.shapely_geom
                poly = geom if geom.geom_type == "Polygon" else list(geom.geoms)[0]
                coords = list(poly.exterior.coords)[:-1]

                pts = [geo.AppendPoint(x, y) for x, y in coords]
                n_pts = len(pts)
                for i in range(n_pts):
                    geo.Append(["line", pts[i], pts[(i + 1) % n_pts]], leftdomain=domain_idx, rightdomain=0)
                domain_map[domain_idx] = name
                domain_idx += 1

            # Mesh with Netgen
            maxh = self.settings.global_size
            ngmesh = geo.GenerateMesh(maxh=maxh)

            # Extract points
            pts = []
            for p in ngmesh.Points():
                pts.append([p[0], p[1]])
            nodes_arr = np.array(pts)

            # Extract elements and domain tags
            elements = []
            element_domains = []
            for el in ngmesh.Elements2D():
                # Netgen indices are 1-based, convert to 0-based
                elements.append([el.points[0].nr - 1, el.points[1].nr - 1, el.points[2].nr - 1])
                mat_idx = el.index
                element_domains.append(domain_map.get(mat_idx, "domain"))

            return Mesh(
                nodes=nodes_arr,
                elements=np.array(elements, dtype=int),
                element_domains=element_domains,
                backend_name="netgen",
            )
        except Exception as e:
            # Fall back to native mesher if Netgen CSG 2D fails on complex polygons
            from .native_backend import NativeMeshGenerator
            return NativeMeshGenerator(self.settings).generate(geometry)

