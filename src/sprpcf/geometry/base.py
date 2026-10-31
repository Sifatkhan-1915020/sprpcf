"""Foundational geometric representation for 2D PCF cross-sections."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any, Dict
import numpy as np
import shapely.geometry as sg
from shapely.validation import make_valid


class Region2D(ABC):
    """Abstract base class for a 2D geometric domain."""

    def __init__(self, name: str, material_tag: Optional[str] = None):
        self.name = name
        self.material_tag = material_tag
        self._shapely_geom: Optional[sg.base.BaseGeometry] = None

    @abstractmethod
    def to_shapely(self) -> sg.base.BaseGeometry:
        """Return the underlying Shapely geometry representation."""
        pass

    @property
    def shapely_geom(self) -> sg.base.BaseGeometry:
        if self._shapely_geom is None:
            g = self.to_shapely()
            if not g.is_valid:
                g = make_valid(g)
            self._shapely_geom = g
        return self._shapely_geom

    def invalidate_cache(self) -> None:
        self._shapely_geom = None

    @property
    def area(self) -> float:
        return float(self.shapely_geom.area)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy)."""
        return self.shapely_geom.bounds

    @property
    def centroid(self) -> Tuple[float, float]:
        c = self.shapely_geom.centroid
        return (float(c.x), float(c.y))

    def contains_point(self, x: float, y: float) -> bool:
        pt = sg.Point(x, y)
        return bool(self.shapely_geom.contains(pt))

    def intersects(self, other: "Region2D") -> bool:
        return bool(self.shapely_geom.intersects(other.shapely_geom))

    def intersection_area(self, other: "Region2D") -> float:
        inter = self.shapely_geom.intersection(other.shapely_geom)
        return float(inter.area) if hasattr(inter, "area") else 0.0

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

