"""Hole definition and manipulation for PCF microstructures."""

from typing import Tuple, Optional, Dict, Any, List
import shapely.geometry as sg
import shapely.affinity as sa
from .base import Region2D
from ..exceptions import InvalidGeometryError


class Hole(Region2D):
    """Represents an individual air or fluid hole in the PCF cladding/core."""

    def __init__(
        self,
        name: str,
        center: Tuple[float, float],
        radius: float,
        material_tag: str = "air",
        ellipticity: float = 1.0,  # a / b (1.0 = circular)
        rotation_deg: float = 0.0,
        shape: str = "circle",
    ):
        super().__init__(name, material_tag)
        if radius <= 0:
            raise InvalidGeometryError(f"Hole '{name}' radius must be > 0, got {radius}")
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)
        self.ellipticity = float(ellipticity)
        self.rotation_deg = float(rotation_deg)
        self.shape = shape.lower()

    def to_shapely(self) -> sg.Polygon:
        if self.shape == "circle" and abs(self.ellipticity - 1.0) < 1e-6:
            pt = sg.Point(self.center[0], self.center[1])
            return pt.buffer(self.radius, resolution=64)
        else:
            # Elliptical hole
            semi_major = self.radius * math_sqrt(self.ellipticity)
            semi_minor = self.radius / math_sqrt(self.ellipticity)
            circle = sg.Point(0.0, 0.0).buffer(1.0, resolution=64)
            scaled = sa.scale(circle, xfact=semi_major, yfact=semi_minor, origin=(0.0, 0.0))
            if abs(self.rotation_deg) > 1e-6:
                scaled = sa.rotate(scaled, self.rotation_deg, origin=(0.0, 0.0))
            return sa.translate(scaled, xoff=self.center[0], yoff=self.center[1])

    def move(self, dx: float, dy: float) -> "Hole":
        self.center = (self.center[0] + dx, self.center[1] + dy)
        self.invalidate_cache()
        return self

    def resize(self, new_radius: float) -> "Hole":
        if new_radius <= 0:
            raise InvalidGeometryError(f"New radius must be > 0: {new_radius}")
        self.radius = float(new_radius)
        self.invalidate_cache()
        return self

    def rotate(self, angle_deg: float) -> "Hole":
        self.rotation_deg = (self.rotation_deg + angle_deg) % 360.0
        self.invalidate_cache()
        return self

    def clone(self, new_name: str) -> "Hole":
        return Hole(
            name=new_name,
            center=self.center,
            radius=self.radius,
            material_tag=self.material_tag,
            ellipticity=self.ellipticity,
            rotation_deg=self.rotation_deg,
            shape=self.shape,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "hole",
            "name": self.name,
            "center": self.center,
            "radius": self.radius,
            "ellipticity": self.ellipticity,
            "rotation_deg": self.rotation_deg,
            "shape": self.shape,
            "material_tag": self.material_tag,
        }


def math_sqrt(val: float) -> float:
    import math
    return math.sqrt(max(1e-12, val))

