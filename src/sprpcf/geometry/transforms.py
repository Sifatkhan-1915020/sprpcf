"""Geometric transformations for 2D PCF regions."""

from typing import Tuple
import shapely.affinity as sa
from .base import Region2D
from .boolean import CompositeRegion


def translate_region(region: Region2D, dx: float, dy: float, new_name: str = "") -> Region2D:
    """Translate a region by (dx, dy)."""
    geom = sa.translate(region.shapely_geom, xoff=dx, yoff=dy)
    name = new_name or f"{region.name}_translated"
    return CompositeRegion(name, geom, material_tag=region.material_tag)


def rotate_region(region: Region2D, angle_deg: float, origin: Tuple[float, float] = (0.0, 0.0), new_name: str = "") -> Region2D:
    """Rotate a region around an origin point by angle in degrees."""
    geom = sa.rotate(region.shapely_geom, angle=angle_deg, origin=origin)
    name = new_name or f"{region.name}_rotated"
    return CompositeRegion(name, geom, material_tag=region.material_tag)


def scale_region(region: Region2D, scale_factor: float, origin: Tuple[float, float] = (0.0, 0.0), new_name: str = "") -> Region2D:
    """Uniformly scale a region by scale_factor."""
    geom = sa.scale(region.shapely_geom, xfact=scale_factor, yfact=scale_factor, origin=origin)
    name = new_name or f"{region.name}_scaled"
    return CompositeRegion(name, geom, material_tag=region.material_tag)

