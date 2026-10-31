"""Visualization module for geometry, mesh, fields, spectra, and reports."""

from .geometry import plot_geometry
from .mesh import plot_mesh
from .fields import plot_mode_field, plot_all_field_components
from .spectra import plot_loss_spectrum, plot_dispersion_curves
from .performance import plot_sensitivity_fit, plot_amplitude_sensitivity, plot_sensitivity_vs_ri
from .dashboard import create_research_dashboard, InteractiveScientificDashboard
from .reports import generate_html_report

__all__ = [
    "plot_geometry",
    "plot_mesh",
    "plot_mode_field",
    "plot_all_field_components",
    "plot_loss_spectrum",
    "plot_dispersion_curves",
    "plot_sensitivity_fit",
    "plot_sensitivity_vs_ri",
    "plot_amplitude_sensitivity",
    "create_research_dashboard",
    "InteractiveScientificDashboard",
    "generate_html_report",
]

