"""Plasmonic and functional thin-film coating layers for SPR biosensors."""

import math
from typing import Tuple, Optional, Dict, Any, List
import shapely.geometry as sg
import shapely.affinity as sa
from .base import Region2D
from ..core.validation import validate_thickness
from ..exceptions import InvalidGeometryError


class CoatingLayer(Region2D):
    """Volumetric representation of a nanoscale plasmonic or functional thin-film layer."""

    def __init__(
        self,
        name: str,
        thickness: float,
        material_tag: str,
        coating_type: str = "planar",  # planar, circumferential, internal_hole, partial_arc
        center: Tuple[float, float] = (0.0, 0.0),
        inner_radius: Optional[float] = None,
        outer_radius: Optional[float] = None,
        start_point: Optional[Tuple[float, float]] = None,
        end_point: Optional[Tuple[float, float]] = None,
        coverage_angle_deg: float = 360.0,
        start_angle_deg: float = 0.0,
    ):
        super().__init__(name, material_tag)
        self.thickness = validate_thickness(thickness, f"Coating '{name}' thickness")
        self.coating_type = coating_type.lower()
        self.center = (float(center[0]), float(center[1]))
        self.inner_radius = float(inner_radius) if inner_radius is not None else None
        self.outer_radius = float(outer_radius) if outer_radius is not None else None
        self.start_point = start_point
        self.end_point = end_point
        self.coverage_angle_deg = float(coverage_angle_deg)
        self.start_angle_deg = float(start_angle_deg)

    def to_shapely(self) -> sg.Polygon:
        if self.coating_type == "circumferential":
            # Outer ring coating on cylinder
            if self.inner_radius is None:
                raise InvalidGeometryError(f"Circumferential coating '{self.name}' requires inner_radius.")
            r_in = self.inner_radius
            r_out = self.inner_radius + self.thickness
            pt = sg.Point(self.center[0], self.center[1])
            outer_circle = pt.buffer(r_out, resolution=128)
            inner_circle = pt.buffer(r_in, resolution=128)
            ring = outer_circle.difference(inner_circle)

            if abs(self.coverage_angle_deg - 360.0) < 1e-4:
                return ring
            else:
                # Wedge cut for partial coverage
                wedge = self._create_wedge(self.center, r_out * 1.5, self.start_angle_deg, self.coverage_angle_deg)
                return ring.intersection(wedge)

        elif self.coating_type == "internal_hole":
            # Coating inside a circular air or fluid hole
            if self.outer_radius is None:
                raise InvalidGeometryError(f"Internal hole coating '{self.name}' requires outer_radius.")
            r_out = self.outer_radius
            r_in = max(1e-12, r_out - self.thickness)
            pt = sg.Point(self.center[0], self.center[1])
            outer_circle = pt.buffer(r_out, resolution=128)
            inner_circle = pt.buffer(r_in, resolution=128)
            return outer_circle.difference(inner_circle)

        elif self.coating_type == "planar":
            # Planar coating on flat side-polished D-plane
            if self.start_point is None or self.end_point is None:
                raise InvalidGeometryError(f"Planar coating '{self.name}' requires start_point and end_point.")
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-12:
                raise InvalidGeometryError("Planar coating start and end points cannot be coincident.")
            # Normal vector pointing outward (+y or perpendicular)
            nx = -dy / length
            ny = dx / length
            t = self.thickness

            coords = [
                (x1, y1),
                (x2, y2),
                (x2 + nx * t, y2 + ny * t),
                (x1 + nx * t, y1 + ny * t),
            ]
            return sg.Polygon(coords)
        else:
            raise InvalidGeometryError(f"Unknown coating type: {self.coating_type}")

    def _create_wedge(self, center: Tuple[float, float], r: float, start_deg: float, span_deg: float) -> sg.Polygon:
        n_pts = max(16, int(span_deg / 5.0))
        angles = np.linspace(math.radians(start_deg), math.radians(start_deg + span_deg), n_pts)
        pts = [(center[0], center[1])]
        for a in angles:
            pts.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))
        pts.append((center[0], center[1]))
        return sg.Polygon(pts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "coating_layer",
            "name": self.name,
            "thickness": self.thickness,
            "material_tag": self.material_tag,
            "coating_type": self.coating_type,
            "center": self.center,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "coverage_angle_deg": self.coverage_angle_deg,
        }

