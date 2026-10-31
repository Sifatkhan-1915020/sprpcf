"""Geometry export utilities for SVG and JSON formats."""

import json
from pathlib import Path
from typing import Dict, Any, Union
from .cross_section import PCFGeometry


def export_geometry_to_json(geometry: PCFGeometry, filepath: Union[str, Path]) -> None:
    """Export PCF geometry metadata and constituent domains to JSON."""
    path = Path(filepath)
    partitioned = geometry.partition()
    data = {
        "name": geometry.name,
        "cladding_radius": geometry.cladding_radius,
        "domains": [reg.to_dict() for reg in partitioned.values() if hasattr(reg, "to_dict")],
        "summary": geometry.get_summary(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_geometry_to_svg(geometry: PCFGeometry, filepath: Union[str, Path], scale_px_per_um: float = 20.0) -> None:
    """Export PCF cross-section geometry as a scalable vector graphic (SVG)."""
    import shapely.geometry as sg
    path = Path(filepath)
    partitioned = geometry.partition()
    r_um = (geometry.cladding_radius + geometry._pml_thickness) * 1e6
    width_px = int(2.2 * r_um * scale_px_per_um)
    height_px = width_px
    cx_px = width_px / 2.0
    cy_px = height_px / 2.0

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        '  <g id="geometry">',
    ]

    material_colors = {
        "silica": "#e0e8f5",
        "air": "#ffffff",
        "gold": "#ffd700",
        "silver": "#c0c0c0",
        "analyte": "#c8f7c5",
        "pml": "#f0f0f0",
    }

    for name, reg in partitioned.items():
        mat = (reg.material_tag or "silica").lower()
        color = material_colors.get(mat, "#d0d0d0")
        geom = reg.shapely_geom

        # Helper to convert polygon exterior coords to svg polygon string
        def poly_to_svg(p: sg.Polygon) -> str:
            coords = []
            for x_m, y_m in p.exterior.coords:
                px = cx_px + (x_m * 1e6) * scale_px_per_um
                py = cy_px - (y_m * 1e6) * scale_px_per_um  # flip y
                coords.append(f"{px:.1f},{py:.1f}")
            pts = " ".join(coords)
            return f'    <polygon points="{pts}" fill="{color}" stroke="#333333" stroke-width="0.8" opacity="0.85" />'

        if isinstance(geom, sg.Polygon):
            svg_lines.append(poly_to_svg(geom))
        elif isinstance(geom, sg.MultiPolygon):
            for part in geom.geoms:
                svg_lines.append(poly_to_svg(part))

    svg_lines.append("  </g>")
    svg_lines.append("</svg>")

    path.write_text("\n".join(svg_lines), encoding="utf-8")

