"""Scientific plotting for 2D PCF geometries with nanoscale interface insets and dimension annotations."""

from typing import Optional, Any, Dict, List
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch, Rectangle as MplRectangle
import shapely.geometry as sg
from ..geometry.validation import validate_geometry


def plot_geometry(
    geometry_or_project: Any,
    ax: Optional[plt.Axes] = None,
    show_pml: bool = True,
    show_legend: bool = True,
    show_dimensions: bool = True,
    show_inset: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot PCF cross-section geometry with physical domain coloring, PML hatching, and exact dimensions."""
    if hasattr(geometry_or_project, "geometry"):
        geom = geometry_or_project.geometry
    else:
        geom = geometry_or_project

    # Perform pre-meshing validation
    val_report = validate_geometry(geom)

    partitioned = geom.partition()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    else:
        fig = ax.figure

    material_colors = {
        "silica": "#c6dbef",
        "air": "#ffffff",
        "gold": "#ffcc00",
        "titanium": "#9e9ac8",
        "chromium": "#969696",
        "silver": "#c0c0c0",
        "copper": "#b87333",
        "analyte": "#c7e9c0",
        "pml": "#f7f7f7",
    }

    plotted_labels = set()
    metal_polys = []

    for name, reg in partitioned.items():
        is_pml = "pml" in name.lower() or reg.material_tag == "pml"
        if not show_pml and is_pml:
            continue

        mat = (reg.material_tag or "silica").lower()
        color = material_colors.get(mat, "#e5f5e0")
        label = mat.capitalize() if mat not in plotted_labels else None
        if label and not is_pml:
            plotted_labels.add(mat)

        poly_geom = reg.shapely_geom
        polys = [poly_geom] if isinstance(poly_geom, sg.Polygon) else list(poly_geom.geoms)

        for p in polys:
            coords_um = np.array(p.exterior.coords) * 1e6
            if is_pml:
                mpl_poly = MplPolygon(
                    coords_um,
                    closed=True,
                    facecolor="#f9f9f9",
                    edgecolor="#777777",
                    linestyle="--",
                    linewidth=1.0,
                    hatch="///",
                    alpha=0.6,
                    label="PML (Absorbing)" if "PML" not in plotted_labels else None,
                )
                plotted_labels.add("PML")
            elif "metal" in name.lower() or mat in ["gold", "silver", "copper"]:
                mpl_poly = MplPolygon(
                    coords_um,
                    closed=True,
                    facecolor=color,
                    edgecolor="#d95f02",
                    linewidth=1.2,
                    alpha=1.0,
                    label=label,
                    zorder=10,
                )
                metal_polys.append(coords_um)
            elif "adhesion" in name.lower() or mat in ["titanium", "chromium"]:
                mpl_poly = MplPolygon(
                    coords_um,
                    closed=True,
                    facecolor=color,
                    edgecolor="#756bb1",
                    linewidth=1.0,
                    alpha=1.0,
                    label=label,
                    zorder=9,
                )
            else:
                mpl_poly = MplPolygon(
                    coords_um,
                    closed=True,
                    facecolor=color,
                    edgecolor="#2b2b2b",
                    linewidth=0.8,
                    alpha=0.9,
                    label=label,
                    zorder=5,
                )
            ax.add_patch(mpl_poly)
            label = None  # Label only once per material

    # Highlight thin metal film with callout marker
    if metal_polys and hasattr(geom, "polish_depth"):
        p_depth_um = geom.polish_depth * 1e6
        ax.plot(
            [0],
            [p_depth_um],
            marker="v",
            color="#d95f02",
            markersize=6,
            zorder=15,
        )

    r_clad_um = geom.cladding_radius * 1e6
    pml_t_um = geom._pml_thickness * 1e6 if (show_pml and geom._pml) else 0.0
    r_total_um = (r_clad_um + pml_t_um) * 1.15

    ax.set_xlim(-r_total_um, r_total_um)
    ax.set_ylim(-r_total_um, r_total_um)
    ax.set_aspect("equal")
    ax.set_xlabel("x (\u03bcm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("y (\u03bcm)", fontsize=10, fontweight="bold")

    title = f"PCF Geometry: {geom.name}"
    if val_report.d_shape_verified:
        title += " (D-Shaped PCF)"
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Dimension text box
    if show_dimensions:
        dim_lines = [
            f"Cladding \u00d8: {2*r_clad_um:.1f} \u03bcm",
        ]
        if hasattr(geom, "pitch"):
            dim_lines.append(f"Pitch (\u039b): {geom.pitch*1e6:.2f} \u03bcm")
        if hasattr(geom, "hole_radius"):
            dim_lines.append(f"Hole \u00d8 (d): {2*geom.hole_radius*1e6:.2f} \u03bcm")
        if hasattr(geom, "polish_depth"):
            dim_lines.append(f"Polish Depth (d_c): {geom.polish_depth*1e6:.2f} \u03bcm")
        if hasattr(geom, "metal_thickness"):
            dim_lines.append(f"Metal Layer: {geom.metal_thickness*1e9:.1f} nm ({getattr(geom, 'metal_material', 'gold')})")
        if getattr(geom, "adhesion_thickness", 0.0) > 0.0:
            dim_lines.append(f"Adhesion Layer: {geom.adhesion_thickness*1e9:.1f} nm ({getattr(geom, 'adhesion_material', 'Ti')})")
        if pml_t_um > 0:
            dim_lines.append(f"PML Thickness: {pml_t_um:.2f} \u03bcm")

        val_badge = "[\u2713 Validated]" if val_report.is_valid else "[! Warnings]"
        dim_lines.append(f"Status: {val_badge}")

        info_text = "\n".join(dim_lines)
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
            zorder=20,
        )

    if show_legend:
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    # Inset zoom on plasmonic interface
    if show_inset and hasattr(geom, "polish_depth"):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width="35%", height="30%", loc="upper right")
        plot_plasmonic_interface_inset(geom, ax=inset_ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_plasmonic_interface_inset(geom: Any, ax: plt.Axes) -> None:
    """Render a high-magnification cross-section view of the nanoscale thin-film interface."""
    p_depth_um = getattr(geom, "polish_depth", 3.2e-6) * 1e6
    t_au_nm = getattr(geom, "metal_thickness", 40.0e-9) * 1e9
    t_adh_nm = getattr(geom, "adhesion_thickness", 0.0) * 1e9
    r_clad_um = getattr(geom, "cladding_radius", 6.5e-6) * 1e6

    # Inset coordinate window: x in [-1.5, 1.5] um, y in [p_depth - 0.3 um, p_depth + 0.5 um]
    y_min = p_depth_um - 0.25
    y_max = p_depth_um + 0.35
    x_min, x_max = -1.2, 1.2

    # Draw silica substrate below polish flat
    ax.axhspan(y_min, p_depth_um, facecolor="#c6dbef", edgecolor="#2b2b2b", linewidth=0.8)
    ax.text(0, p_depth_um - 0.12, "Fused Silica Cladding", ha="center", va="center", fontsize=7, fontweight="bold")

    curr_y = p_depth_um
    # Draw adhesion layer if present
    if t_adh_nm > 0:
        adh_height_um = t_adh_nm * 1e-3
        ax.axhspan(curr_y, curr_y + adh_height_um, facecolor="#9e9ac8", edgecolor="#756bb1", linewidth=0.5)
        ax.text(0, curr_y + adh_height_um / 2, f"Adhesion Ti ({t_adh_nm:.0f} nm)", ha="center", va="center", fontsize=6, color="white")
        curr_y += adh_height_um

    # Draw gold plasmonic layer
    au_height_um = max(0.02, t_au_nm * 1e-3)
    ax.axhspan(curr_y, curr_y + au_height_um, facecolor="#ffcc00", edgecolor="#d95f02", linewidth=1.0)
    ax.text(0, curr_y + au_height_um / 2, f"Gold Plasmonic Film ({t_au_nm:.0f} nm)", ha="center", va="center", fontsize=7, fontweight="bold")
    curr_y += au_height_um

    # Draw analyte channel above gold
    ax.axhspan(curr_y, y_max, facecolor="#c7e9c0", edgecolor="#238b45", linewidth=0.8)
    ax.text(0, curr_y + 0.15, "Sensing Analyte Channel (n_a)", ha="center", va="center", fontsize=7, fontweight="bold", color="#005a32")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([p_depth_um])
    ax.set_yticklabels([f"y={p_depth_um:.2f}\u03bcm"], fontsize=6)
    ax.set_title("Nanoscale Interface Zoom", fontsize=8, fontweight="bold")
