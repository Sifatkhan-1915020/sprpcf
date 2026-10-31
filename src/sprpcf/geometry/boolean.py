"""Robust 2D constructive solid geometry (CSG) boolean operations and topology checks."""

from typing import List, Tuple, Optional, Dict
import shapely.geometry as sg
import shapely.ops as so
from shapely.validation import make_valid
from .base import Region2D
from ..exceptions import InvalidGeometryError, OverlappingRegionError


class CompositeRegion(Region2D):
    """Region formed by boolean combination of other regions."""

    def __init__(self, name: str, geom: sg.base.BaseGeometry, material_tag: Optional[str] = None):
        super().__init__(name, material_tag)
        if not geom.is_valid:
            geom = make_valid(geom)
        self._geom = geom

    def to_shapely(self) -> sg.base.BaseGeometry:
        return self._geom

    def to_dict(self) -> Dict:
        return {
            "type": "composite",
            "name": self.name,
            "area": self.area,
            "material_tag": self.material_tag,
        }


def boolean_difference(base: Region2D, tool: Region2D, new_name: Optional[str] = None) -> CompositeRegion:
    """Compute base minus tool (subtraction)."""
    diff = base.shapely_geom.difference(tool.shapely_geom)
    if diff.is_empty or diff.area < 1e-25:
        raise InvalidGeometryError(f"Boolean difference between '{base.name}' and '{tool.name}' resulted in an empty region.")
    name = new_name or f"{base.name}_minus_{tool.name}"
    return CompositeRegion(name, diff, material_tag=base.material_tag)


def boolean_union(region1: Region2D, region2: Region2D, new_name: Optional[str] = None) -> CompositeRegion:
    """Compute union of two regions."""
    u = region1.shapely_geom.union(region2.shapely_geom)
    name = new_name or f"{region1.name}_union_{region2.name}"
    return CompositeRegion(name, u, material_tag=region1.material_tag)


def boolean_intersection(region1: Region2D, region2: Region2D, new_name: Optional[str] = None) -> CompositeRegion:
    """Compute intersection of two regions."""
    inter = region1.shapely_geom.intersection(region2.shapely_geom)
    if inter.is_empty or inter.area < 1e-25:
        raise InvalidGeometryError(f"Intersection of '{region1.name}' and '{region2.name}' is empty.")
    name = new_name or f"{region1.name}_inter_{region2.name}"
    return CompositeRegion(name, inter, material_tag=region1.material_tag)


def partition_domains(background: Region2D, embedded_regions: List[Region2D]) -> Tuple[CompositeRegion, List[Region2D]]:
    """Partition background domain by subtracting all embedded internal regions (e.g. holes, channels, coatings).
    
    Ensures complete, non-overlapping spatial partitioning where:
    area(background_remaining) + sum(area(embedded)) == area(original_background).
    """
    current_bg = background.shapely_geom
    for reg in embedded_regions:
        inter = current_bg.intersection(reg.shapely_geom)
        if not inter.is_empty and inter.area > 1e-25:
            current_bg = current_bg.difference(reg.shapely_geom)

    if current_bg.is_empty or current_bg.area < 1e-25:
        raise InvalidGeometryError(f"Background domain '{background.name}' was completely consumed by embedded regions.")

    bg_partitioned = CompositeRegion(
        name=background.name,
        geom=current_bg,
        material_tag=background.material_tag
    )
    return bg_partitioned, embedded_regions


def detect_overlapping_regions(regions: List[Region2D], tolerance: float = 1e-18) -> List[Tuple[str, str, float]]:
    """Scan list of regions and return pairs that illegally overlap with non-negligible intersection area."""
    overlaps = []
    n = len(regions)
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = regions[i], regions[j]
            if r1.shapely_geom.intersects(r2.shapely_geom):
                inter = r1.shapely_geom.intersection(r2.shapely_geom)
                if hasattr(inter, "area") and inter.area > tolerance:
                    overlaps.append((r1.name, r2.name, float(inter.area)))
    return overlaps

