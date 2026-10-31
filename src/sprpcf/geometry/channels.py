"""Microfluidic and analyte sensing channel geometry."""

from typing import Tuple, Optional, Dict, Any
import shapely.geometry as sg
from .base import Region2D
from ..exceptions import InvalidGeometryError


class AnalyteChannel(Region2D):
    """Represents a microfluidic channel or infiltrated sensing region filled with analyte."""

    def __init__(
        self,
        name: str,
        shape: str = "circular",  # circular, rectangular, open_cut
        center: Tuple[float, float] = (0.0, 0.0),
        radius: Optional[float] = None,
        dimensions: Optional[Tuple[float, float]] = None,
        material_tag: str = "analyte",
    ):
        super().__init__(name, material_tag)
        self.shape = shape.lower()
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius) if radius is not None else None
        self.dimensions = dimensions

    def to_shapely(self) -> sg.Polygon:
        if self.shape == "circular":
            if self.radius is None or self.radius <= 0:
                raise InvalidGeometryError(f"Circular channel '{self.name}' requires radius > 0.")
            pt = sg.Point(self.center[0], self.center[1])
            return pt.buffer(self.radius, resolution=64)

        elif self.shape in ["rectangular", "rectangle"]:
            if self.dimensions is None or len(self.dimensions) != 2:
                raise InvalidGeometryError(f"Rectangular channel '{self.name}' requires (width, height).")
            w, h = self.dimensions
            hw, hh = w / 2.0, h / 2.0
            cx, cy = self.center
            coords = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
            return sg.Polygon(coords)

        elif self.shape == "open_cut":
            # Semi-infinite or top-open sensing region above polished flat
            if self.dimensions is None:
                raise InvalidGeometryError("Open cut channel requires (width, height) bounding cut.")
            w, h = self.dimensions
            cx, cy = self.center
            coords = [(cx - w / 2.0, cy), (cx + w / 2.0, cy), (cx + w / 2.0, cy + h), (cx - w / 2.0, cy + h)]
            return sg.Polygon(coords)
        else:
            raise InvalidGeometryError(f"Unsupported channel shape: {self.shape}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "analyte_channel",
            "name": self.name,
            "shape": self.shape,
            "center": self.center,
            "radius": self.radius,
            "dimensions": self.dimensions,
            "material_tag": self.material_tag,
        }

