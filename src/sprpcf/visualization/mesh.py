"""Scientific visualization for finite element meshes with interface zoom and quality distribution."""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from ..meshing.base import Mesh
from ..meshing.validation import MeshQualityReport, validate_mesh


def plot_mesh(
    mesh: Mesh,
    ax: Optional[plt.Axes] = None,
    quality_map: bool = False,
    show_interface_inset: bool = False,
    interface_y: Optional[float] = None,
    show_stats: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot discretized finite element mesh with quality metrics, optional zoom inset, and statistics."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    else:
        fig = ax.figure

    nodes_um = mesh.nodes * 1e6
    tris = mesh.elements[:, :3]
    tri_coords = nodes_um[tris]

    report = validate_mesh(mesh, interface_y=interface_y)

    if quality_map:
        qualities, _, _ = report._compute_triangle_metrics(mesh)
        poly_coll = PolyCollection(
            tri_coords,
            array=qualities,
            cmap="viridis",
            edgecolors="#222222",
            linewidths=0.3,
            clim=(0.0, 1.0),
        )
        ax.add_collection(poly_coll)
        cbar = fig.colorbar(poly_coll, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Element Quality (2 r_in / r_out)", fontsize=9, fontweight="bold")
    else:
        poly_coll = PolyCollection(
            tri_coords,
            facecolors="#f2f2f2",
            edgecolors="#444444",
            linewidths=0.35,
            alpha=0.9,
        )
        ax.add_collection(poly_coll)

    min_x, min_y = np.min(nodes_um, axis=0)
    max_x, max_y = np.max(nodes_um, axis=0)
    pad = (max_x - min_x) * 0.05
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (\u03bcm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("y (\u03bcm)", fontsize=10, fontweight="bold")
    ax.set_title(f"Finite Element Mesh ({mesh.num_elements} Elements)", fontsize=11, fontweight="bold")

    # Stats text box
    if show_stats:
        stats_text = (
            f"Nodes: {mesh.num_nodes}\n"
            f"Elements: {mesh.num_elements}\n"
            f"Mean Quality: {report.mean_quality:.3f}\n"
            f"Min Quality: {report.min_quality:.3e}\n"
            f"Max Aspect Ratio: {report.max_aspect_ratio:.1f}\n"
            f"Interface Elements: {report.interface_elements_count}\n"
            f"Valid: {report.is_valid}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
            zorder=20,
        )

    # Inset zoom on interface mesh
    if show_interface_inset and interface_y is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width="38%", height="32%", loc="upper right")
        plot_mesh_interface_inset(mesh, interface_y=interface_y, ax=inset_ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_mesh_interface_inset(mesh: Mesh, interface_y: float, ax: plt.Axes) -> None:
    """Render a magnified view of the finite element mesh at the plasmonic sensing interface."""
    nodes_um = mesh.nodes * 1e6
    tris = mesh.elements[:, :3]
    tri_coords = nodes_um[tris]

    iy_um = interface_y * 1e6
    x_min, x_max = -1.5, 1.5
    y_min, y_max = iy_um - 0.4, iy_um + 0.6

    # Filter triangles that fall inside the window
    centroids = np.mean(tri_coords, axis=1)
    in_window = (
        (centroids[:, 0] >= x_min - 0.5)
        & (centroids[:, 0] <= x_max + 0.5)
        & (centroids[:, 1] >= y_min - 0.5)
        & (centroids[:, 1] <= y_max + 0.5)
    )

    sub_tris = tri_coords[in_window]
    coll = PolyCollection(
        sub_tris,
        facecolors="#e8f4f8",
        edgecolors="#1b7837",
        linewidths=0.6,
        alpha=0.9,
    )
    ax.add_collection(coll)

    # Draw interface flat reference line
    ax.axhline(iy_um, color="#d95f02", linestyle="--", linewidth=1.5, label="Au Flat")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([iy_um])
    ax.set_yticklabels([f"{iy_um:.1f}\u03bcm"], fontsize=7)
    ax.set_title("Interface Mesh Zoom", fontsize=8, fontweight="bold")
