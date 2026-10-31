"""Electromagnetic field distribution plotting with smooth interpolation and multi-component breakdown."""

from typing import Optional, Union, Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import shapely.geometry as sg
from ..physics.electromagnetic_fields import ElectromagneticField2D
from ..solvers.base import OpticalMode


def _overlay_geom_boundaries(ax: plt.Axes, geometry: Any) -> None:
    """Overlay thin contour lines of cladding, holes, and metal interfaces."""
    if geometry is None:
        return
    partitioned = geometry.partition()
    for name, reg in partitioned.items():
        if "pml" in name.lower() or reg.material_tag == "pml":
            continue
        poly_geom = reg.shapely_geom
        polys = [poly_geom] if isinstance(poly_geom, sg.Polygon) else list(poly_geom.geoms)
        for p in polys:
            coords_um = np.array(p.exterior.coords) * 1e6
            is_metal = "metal" in name.lower() or reg.material_tag in ["gold", "silver", "copper"]
            edge_color = "#ffffff" if not is_metal else "#ffff00"
            lw = 1.2 if is_metal else 0.7
            ls = "-" if not is_metal else "--"
            alpha = 0.9 if is_metal else 0.6
            poly = MplPolygon(
                coords_um,
                closed=True,
                fill=False,
                edgecolor=edge_color,
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                zorder=10,
            )
            ax.add_patch(poly)


def plot_mode_field(
    mode_or_field: Union[OpticalMode, ElectromagneticField2D],
    component: str = "E_mag",  # 'E_mag', 'E_sq', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Sz', 'phase'
    geometry: Optional[Any] = None,
    smooth: bool = True,
    interpolation_factor: int = 2,
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    cmap: str = "inferno",
    dpi: int = 300,
) -> plt.Figure:
    """Plot 2D electromagnetic field component profile with artifact-free interpolation and geometry overlay."""
    mode_obj = None
    if hasattr(mode_or_field, "field"):
        mode_obj = mode_or_field
        raw_field = mode_or_field.field
        title_suffix = (
            f"Mode {mode_or_field.mode_id} [{mode_or_field.mode_classification}]\n"
            f"n_eff = {mode_or_field.n_real:.5f} + {mode_or_field.n_imag:.2e}i | "
            f"Loss = {mode_or_field.loss_db_per_cm:.2f} dB/cm"
        )
    else:
        raw_field = mode_or_field
        title_suffix = f"\u03bb = {raw_field.wavelength_m*1e9:.1f} nm"

    if raw_field is None:
        raise ValueError("Mode does not contain field data.")

    # High-quality bicubic interpolation to avoid discrete raster artifacts
    field = raw_field.interpolate_bicubic(factor=interpolation_factor) if smooth else raw_field

    comp = component.lower().strip()
    if comp in ["e_mag", "emag", "e"]:
        data = field.e_mag
        label = "|E| (normalized)"
        cmap = "inferno"
    elif comp in ["e_sq", "e2"]:
        data = field.e_mag_sq
        label = "|E|^2 (normalized intensity)"
        cmap = "inferno"
    elif comp == "ex":
        data = np.real(field.ex)
        label = "Re(Ex)"
        cmap = "RdBu_r"
    elif comp == "ey":
        data = np.real(field.ey)
        label = "Re(Ey)"
        cmap = "RdBu_r"
    elif comp == "ez":
        data = np.real(field.ez)
        label = "Re(Ez)"
        cmap = "RdBu_r"
    elif comp == "hx":
        data = np.real(field.hx)
        label = "Re(Hx)"
        cmap = "PuOr_r"
    elif comp == "hy":
        data = np.real(field.hy)
        label = "Re(Hy)"
        cmap = "PuOr_r"
    elif comp == "hz":
        data = np.real(field.hz)
        label = "Re(Hz)"
        cmap = "PuOr_r"
    elif comp in ["sz", "poynting"]:
        data = field.poynting_z
        label = "Poynting Sz (W/m^2)"
        cmap = "plasma"
    elif comp == "phase":
        data = field.dominant_phase
        label = "Phase \u2220Et (rad)"
        cmap = "twilight"
    else:
        data = field.e_mag
        label = "|E|"

    if log_scale:
        data = np.log10(np.maximum(1e-4, data / max(1e-20, np.max(data))))
        label = "log10(norm " + label + ")"

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    else:
        fig = ax.figure

    x_um = field.x * 1e6
    y_um = field.y * 1e6
    extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

    # Symmetrical colormap limits for vector field components
    clim = None
    if cmap in ["RdBu_r", "PuOr_r"]:
        max_abs = float(np.max(np.abs(data)))
        clim = (-max_abs, max_abs)

    im = ax.imshow(
        data,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="bicubic",
        clim=clim,
    )

    # Overlay geometric boundaries
    _overlay_geom_boundaries(ax, geometry)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=9, fontweight="bold")

    # Add power validation badge
    if comp in ["sz", "poynting", "e_mag", "emag"] and mode_obj:
        p_tot = mode_obj.total_power
        fwd_status = "\u2713 Forward Flow (P > 0)" if mode_obj.is_forward_propagating else "! Reverse Flow"
        ax.text(
            0.02,
            0.03,
            f"P_tot: {p_tot:.2e} W [{fwd_status}]",
            transform=ax.transAxes,
            fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#222222", alpha=0.7),
            zorder=20,
        )

    ax.set_xlabel("x (\u03bcm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("y (\u03bcm)", fontsize=10, fontweight="bold")
    ax.set_title(f"{label}\n{title_suffix}", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_all_field_components(
    mode: OpticalMode,
    geometry: Optional[Any] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Generate a comprehensive 6-panel scientific field breakdown (|E|, Ex, Ey, Ez, Sz, Phase)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=dpi)
    fig.suptitle(
        f"Full-Vector Modal Field Solutions: Mode {mode.mode_id} at \u03bb = {mode.wavelength_m*1e9:.1f} nm\n"
        f"[{mode.mode_classification}]  n_eff = {mode.n_real:.5f} + {mode.n_imag:.2e}i  |  Loss = {mode.loss_db_per_cm:.2f} dB/cm",
        fontsize=12,
        fontweight="bold",
    )

    components = [
        ("E_mag", axes[0, 0]),
        ("Ex", axes[0, 1]),
        ("Ey", axes[0, 2]),
        ("Ez", axes[1, 0]),
        ("Sz", axes[1, 1]),
        ("phase", axes[1, 2]),
    ]

    for comp, ax in components:
        plot_mode_field(
            mode,
            component=comp,
            geometry=geometry,
            ax=ax,
            smooth=True,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
