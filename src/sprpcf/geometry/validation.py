"""Geometry validation, topological integrity checking, and D-shape verification."""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import shapely.geometry as sg
from ..exceptions import InvalidGeometryError


class GeometryValidationReport:
    """Detailed report on geometric validity, dimensions, and D-cut feasibility."""

    def __init__(
        self,
        is_valid: bool,
        issues: List[str],
        warnings: List[str],
        d_shape_verified: bool,
        cladding_containment_verified: bool,
        min_feature_size_m: float,
        hole_count: int,
        coating_count: int,
        channel_count: int,
        dimensions: Dict[str, float],
    ):
        self.is_valid = is_valid
        self.issues = issues
        self.warnings = warnings
        self.d_shape_verified = d_shape_verified
        self.cladding_containment_verified = cladding_containment_verified
        self.min_feature_size_m = min_feature_size_m
        self.hole_count = hole_count
        self.coating_count = coating_count
        self.channel_count = channel_count
        self.dimensions = dimensions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "d_shape_verified": self.d_shape_verified,
            "cladding_containment_verified": self.cladding_containment_verified,
            "min_feature_size_um": self.min_feature_size_m * 1e6,
            "hole_count": self.hole_count,
            "coating_count": self.coating_count,
            "channel_count": self.channel_count,
            "dimensions": {k: v * 1e6 for k, v in self.dimensions.items()},
        }

    def __repr__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        d_status = "D-Shape Verified" if self.d_shape_verified else "Standard Cladding"
        return (
            f"GeometryValidationReport(status={status}, {d_status}, "
            f"holes={self.hole_count}, min_feature={self.min_feature_size_m*1e6:.3f} um, "
            f"issues={len(self.issues)}, warnings={len(self.warnings)})"
        )


def validate_geometry(geometry: Any) -> GeometryValidationReport:
    """Perform rigorous pre-meshing topological and dimensional validation on a PCF geometry."""
    issues: List[str] = []
    warnings: List[str] = []
    dimensions: Dict[str, float] = {}

    r_clad = getattr(geometry, "cladding_radius", 0.0)
    dimensions["cladding_radius"] = r_clad
    if r_clad <= 0:
        issues.append(f"Cladding radius must be positive, got {r_clad}")

    # Holes validation
    holes = list(geometry._holes.values()) if hasattr(geometry, "_holes") else []
    min_feature_size = r_clad

    # Check inter-hole distance and hole radius
    for i, h1 in enumerate(holes):
        r1 = getattr(h1, "radius", 0.0)
        min_feature_size = min(min_feature_size, r1 * 2.0)
        for j in range(i + 1, len(holes)):
            h2 = holes[j]
            dist = math.hypot(h1.center[0] - h2.center[0], h1.center[1] - h2.center[1])
            r2 = getattr(h2, "radius", 0.0)
            wall_thick = dist - (r1 + r2)
            if wall_thick < -1e-9:
                issues.append(f"Holes '{h1.name}' and '{h2.name}' overlap (clearance {wall_thick:.2e} m)")
            elif wall_thick < 0.05e-6:
                warnings.append(f"Extremely thin inter-hole silica bridge ({wall_thick*1e9:.1f} nm) between '{h1.name}' and '{h2.name}'")

    # Cladding containment
    clad_contained = True
    if geometry._cladding_boundary:
        clad_poly = geometry._cladding_boundary.shapely_geom
        for h in holes:
            h_poly = h.shapely_geom
            if not clad_poly.contains(h_poly):
                inter = clad_poly.intersection(h_poly)
                if hasattr(inter, "area") and inter.area < h.area * 0.999:
                    issues.append(f"Air hole '{h.name}' protrudes outside fiber cladding boundary.")
                    clad_contained = False

    # D-shape verification
    d_shape_verified = False
    is_d = hasattr(geometry, "polish_depth") or "d_shaped" in getattr(geometry, "name", "").lower()
    if is_d:
        p_depth = getattr(geometry, "polish_depth", None)
        if p_depth is not None:
            dimensions["polish_depth"] = p_depth
            if p_depth <= 0.0:
                issues.append(f"Polish depth must be positive (distance from core to flat), got {p_depth}")
            elif p_depth >= r_clad:
                issues.append(f"Polish depth {p_depth*1e6:.2f} um exceeds cladding radius {r_clad*1e6:.2f} um.")
            else:
                # Check chord geometry
                half_w = math.sqrt(r_clad**2 - p_depth**2)
                dimensions["flat_chord_width"] = 2.0 * half_w
                d_shape_verified = True

                # Verify no hole breaches the polished flat
                for h in holes:
                    cy = h.center[1]
                    hr = getattr(h, "radius", 0.0)
                    if cy + hr >= p_depth - 1e-8:
                        issues.append(
                            f"Hole '{h.name}' at y={cy*1e6:.2f} um breaches or intersects the D-polish flat (y={p_depth*1e6:.2f} um)."
                        )

                # Verify plasmonic metal coating sits on the flat
                coatings = getattr(geometry, "_coatings", {})
                for c_name, layer in coatings.items():
                    min_feature_size = min(min_feature_size, layer.thickness)
                    dimensions[f"thickness_{c_name}"] = layer.thickness
                    if layer.coating_type == "planar":
                        if layer.start_point and layer.end_point:
                            y1, y2 = layer.start_point[1], layer.end_point[1]
                            if abs(y1 - p_depth) > 1e-7 or abs(y2 - p_depth) > 1e-7:
                                warnings.append(
                                    f"Planar coating '{c_name}' does not align with polish flat at y={p_depth*1e6:.2f} um."
                                )

    # PML verification
    if geometry._pml is not None:
        dimensions["pml_thickness"] = geometry._pml_thickness
        if geometry._pml_thickness < 0.5e-6:
            warnings.append(f"PML thickness ({geometry._pml_thickness*1e6:.2f} um) may be too thin for reliable absorption.")

    is_valid = len(issues) == 0
    return GeometryValidationReport(
        is_valid=is_valid,
        issues=issues,
        warnings=warnings,
        d_shape_verified=d_shape_verified,
        cladding_containment_verified=clad_contained,
        min_feature_size_m=min_feature_size,
        hole_count=len(holes),
        coating_count=len(getattr(geometry, "_coatings", {})),
        channel_count=len(getattr(geometry, "_channels", {})),
        dimensions=dimensions,
    )

