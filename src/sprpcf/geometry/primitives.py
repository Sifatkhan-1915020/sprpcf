"""Geometric primitive regions for 2D PCF modeling."""

import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa
from .base import Region2D
from ..exceptions import InvalidGeometryError


class Circle(Region2D):
    """Circular domain."""

    def __init__(self, name: str, center: Tuple[float, float], radius: float, material_tag: Optional[str] = None):
        super().__init__(name, material_tag)
        if radius <= 0:
            raise InvalidGeometryError(f"Circle '{name}' radius must be > 0, got {radius}")
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)

    def to_shapely(self) -> sg.Polygon:
        pt = sg.Point(self.center[0], self.center[1])
        return pt.buffer(self.radius, resolution=64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "circle",
            "name": self.name,
            "center": self.center,
            "radius": self.radius,
            "material_tag": self.material_tag,
        }


class Ellipse(Region2D):
    """Elliptical domain with arbitrary rotation."""

    def __init__(
        self,
        name: str,
        center: Tuple[float, float],
        semi_major: float,
        semi_minor: float,
        rotation_deg: float = 0.0,
        material_tag: Optional[str] = None,
    ):
        super().__init__(name, material_tag)
        if semi_major <= 0 or semi_minor <= 0:
            raise InvalidGeometryError(f"Ellipse axes must be > 0: ({semi_major}, {semi_minor})")
        self.center = (float(center[0]), float(center[1]))
        self.semi_major = float(semi_major)
        self.semi_minor = float(semi_minor)
        self.rotation_deg = float(rotation_deg)

    def to_shapely(self) -> sg.Polygon:
        circle = sg.Point(0.0, 0.0).buffer(1.0, resolution=64)
        scaled = sa.scale(circle, xfact=self.semi_major, yfact=self.semi_minor, origin=(0.0, 0.0))
        rotated = sa.rotate(scaled, self.rotation_deg, origin=(0.0, 0.0))
        translated = sa.translate(rotated, xoff=self.center[0], yoff=self.center[1])
        return translated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ellipse",
            "name": self.name,
            "center": self.center,
            "semi_major": self.semi_major,
            "semi_minor": self.semi_minor,
            "rotation_deg": self.rotation_deg,
            "material_tag": self.material_tag,
        }


class Rectangle(Region2D):
    """Rectangular domain defined by center, width, height, and rotation."""

    def __init__(
        self,
        name: str,
        center: Tuple[float, float],
        width: float,
        height: float,
        rotation_deg: float = 0.0,
        material_tag: Optional[str] = None,
    ):
        super().__init__(name, material_tag)
        if width <= 0 or height <= 0:
            raise InvalidGeometryError(f"Rectangle dimensions must be > 0: ({width}, {height})")
        self.center = (float(center[0]), float(center[1]))
        self.width = float(width)
        self.height = float(height)
        self.rotation_deg = float(rotation_deg)

    def to_shapely(self) -> sg.Polygon:
        hw = self.width / 2.0
        hh = self.height / 2.0
        coords = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        poly = sg.Polygon(coords)
        if abs(self.rotation_deg) > 1e-6:
            poly = sa.rotate(poly, self.rotation_deg, origin=(0.0, 0.0))
        return sa.translate(poly, xoff=self.center[0], yoff=self.center[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "rectangle",
            "name": self.name,
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "rotation_deg": self.rotation_deg,
            "material_tag": self.material_tag,
        }


class RegularPolygon(Region2D):
    """Regular polygon (triangle, square, hexagon, octagon)."""

    def __init__(
        self,
        name: str,
        center: Tuple[float, float],
        radius: float,
        num_sides: int,
        rotation_deg: float = 0.0,
        material_tag: Optional[str] = None,
    ):
        super().__init__(name, material_tag)
        if radius <= 0 or num_sides < 3:
            raise InvalidGeometryError(f"Regular polygon requires radius > 0 and num_sides >= 3: ({radius}, {num_sides})")
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)
        self.num_sides = int(num_sides)
        self.rotation_deg = float(rotation_deg)

    def to_shapely(self) -> sg.Polygon:
        angles = np.linspace(0, 2 * np.pi, self.num_sides, endpoint=False) + math.radians(self.rotation_deg)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        coords = list(zip(x, y))
        return sg.Polygon(coords)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "regular_polygon",
            "name": self.name,
            "center": self.center,
            "radius": self.radius,
            "num_sides": self.num_sides,
            "rotation_deg": self.rotation_deg,
            "material_tag": self.material_tag,
        }


class Annulus(Region2D):
    """Concentric circular ring/annulus domain."""

    def __init__(
        self,
        name: str,
        center: Tuple[float, float],
        r_inner: float,
        r_outer: float,
        material_tag: Optional[str] = None,
    ):
        super().__init__(name, material_tag)
        if r_inner < 0 or r_outer <= r_inner:
            raise InvalidGeometryError(f"Annulus requires 0 <= r_inner < r_outer: ({r_inner}, {r_outer})")
        self.center = (float(center[0]), float(center[1]))
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)

    def to_shapely(self) -> sg.Polygon:
        pt = sg.Point(self.center[0], self.center[1])
        outer = pt.buffer(self.r_outer, resolution=64)
        if self.r_inner > 1e-15:
            inner = pt.buffer(self.r_inner, resolution=64)
            return outer.difference(inner)
        return outer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "annulus",
            "name": self.name,
            "center": self.center,
            "r_inner": self.r_inner,
            "r_outer": self.r_outer,
            "material_tag": self.material_tag,
        }


class ArbitraryPolygon(Region2D):
    """User-defined closed polygon from an explicit list of (x, y) vertices."""

    def __init__(self, name: str, vertices: List[Tuple[float, float]], material_tag: Optional[str] = None):
        super().__init__(name, material_tag)
        if len(vertices) < 3:
            raise InvalidGeometryError(f"Polygon '{name}' must have at least 3 vertices.")
        self.vertices = [(float(v[0]), float(v[1])) for v in vertices]

    def to_shapely(self) -> sg.Polygon:
        return sg.Polygon(self.vertices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "arbitrary_polygon",
            "name": self.name,
            "vertices": self.vertices,
            "material_tag": self.material_tag,
        }

