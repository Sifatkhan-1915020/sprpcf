"""Biosensor sensitivity and performance plotting."""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def plot_sensitivity_fit(
    sensitivity_dict_or_report: Any,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot resonance wavelength shift vs analyte refractive index with linear and polynomial regression."""
    if hasattr(sensitivity_dict_or_report, "to_dict"):
        data = sensitivity_dict_or_report.to_dict()
    elif isinstance(sensitivity_dict_or_report, dict) and "sensitivity" in sensitivity_dict_or_report:
        data = sensitivity_dict_or_report["sensitivity"]
    else:
        data = sensitivity_dict_or_report

    na = np.asarray(data["analyte_indices"], dtype=float)
    res_wl = np.asarray(data["resonance_wavelengths_nm"], dtype=float)
    slope = data.get("linear_wavelength_sensitivity_nm_per_riu", 0.0)
    intercept = data.get("intercept_nm", 0.0)
    r2 = data.get("r_squared", 1.0)
    rmse = data.get("rmse_nm", 0.0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    else:
        fig = ax.figure

    # Data points
    ax.plot(na, res_wl, "bs", markersize=8, label="FEM Resonance Points", zorder=8)

    # Linear fit line
    if len(na) >= 2:
        x_dense = np.linspace(na[0] - 0.002, na[-1] + 0.002, 100)
        fit_line = slope * x_dense + intercept
        ax.plot(
            x_dense,
            fit_line,
            "r--",
            linewidth=2.0,
            label=f"Linear Fit: S_\u03bb = {slope:.1f} nm/RIU (R^2 = {r2:.4f})",
            zorder=5,
        )

        # Polynomial fit if available
        poly_coeffs = data.get("polynomial_coefficients", [])
        if len(poly_coeffs) == 3 and poly_coeffs[0] != 0.0:
            poly_fn = np.poly1d(poly_coeffs)
            ax.plot(
                x_dense,
                poly_fn(x_dense),
                "g-",
                linewidth=1.5,
                alpha=0.8,
                label=f"Quadratic: d\u03bb/dn = {2*poly_coeffs[0]*na[0]+poly_coeffs[1]:.0f} to {2*poly_coeffs[0]*na[-1]+poly_coeffs[1]:.0f} nm/RIU",
                zorder=4,
            )

    # Stats info box
    info_box = (
        f"Sensitivity: {slope:.1f} nm/RIU\n"
        f"Linearity (R^2): {r2:.4f}\n"
        f"RMSE: {rmse:.2f} nm\n"
        f"Analyte Span: {na[0]:.3f} \u2013 {na[-1]:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        info_box,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
        zorder=15,
    )

    ax.set_xlabel("Analyte Refractive Index n_a (RIU)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Resonance Wavelength \u03bb_res (nm)", fontsize=10, fontweight="bold")
    ax.set_title("Wavelength Sensitivity Calibration", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_sensitivity_vs_ri(
    sensitivity_dict_or_report: Any,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot incremental wavelength sensitivity S_lambda vs analyte refractive index."""
    if hasattr(sensitivity_dict_or_report, "to_dict"):
        data = sensitivity_dict_or_report.to_dict()
    elif isinstance(sensitivity_dict_or_report, dict) and "sensitivity" in sensitivity_dict_or_report:
        data = sensitivity_dict_or_report["sensitivity"]
    else:
        data = sensitivity_dict_or_report

    na = np.asarray(data["analyte_indices"], dtype=float)
    inc_sens = np.asarray(data.get("incremental_sensitivities", []), dtype=float)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    else:
        fig = ax.figure

    if len(inc_sens) > 0:
        # Midpoints of RI intervals
        mid_na = 0.5 * (na[:-1] + na[1:])
        ax.plot(mid_na, inc_sens, "d-", color="#d95f02", linewidth=2.0, markersize=8, label="Incremental S_\u03bb(n_a)")

    ax.set_xlabel("Analyte Refractive Index n_a (RIU)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Incremental Sensitivity (nm/RIU)", fontsize=10, fontweight="bold")
    ax.set_title("Sensitivity vs Analyte Refractive Index", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_amplitude_sensitivity(
    wavelengths_nm: np.ndarray,
    s_a_spectrum: np.ndarray,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot spectral amplitude sensitivity curve S_A(lambda)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    else:
        fig = ax.figure

    max_idx = int(np.argmax(s_a_spectrum))
    max_wl = float(wavelengths_nm[max_idx])
    max_sa = float(s_a_spectrum[max_idx])

    ax.plot(wavelengths_nm, s_a_spectrum, "m-", linewidth=2.0, label="Amplitude Sensitivity S_A")
    ax.plot(max_wl, max_sa, "ko", markersize=7, label=f"Max S_A = {max_sa:.1f} RIU^-1 @ {max_wl:.1f} nm")

    ax.set_xlabel("Wavelength (nm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Amplitude Sensitivity (RIU^-1)", fontsize=10, fontweight="bold")
    ax.set_title("Spectral Amplitude Interrogation", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
