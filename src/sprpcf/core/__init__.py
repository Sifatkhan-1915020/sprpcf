"""Core module for SPR-PCF Studio."""

from .constants import (
    SPEED_OF_LIGHT,
    C0,
    VACUUM_PERMITTIVITY,
    EPSILON_0,
    VACUUM_PERMEABILITY,
    MU_0,
    VACUUM_IMPEDANCE,
    ETA_0,
    NEPER_TO_DB,
)
from .units import (
    LengthUnit,
    wavelength_to_frequency,
    frequency_to_wavelength,
    wavelength_to_wavenumber,
    photon_energy_ev,
    attenuation_conversion,
    parse_quantity,
)
from .parameters import Parameter, ParameterSet
from .validation import (
    validate_positive_number,
    validate_non_negative_number,
    validate_thickness,
    validate_domains_assigned,
    validate_wavelength_range,
)
from .registry import (
    solvers_registry,
    meshers_registry,
    materials_registry,
    dispersion_registry,
    pcf_templates_registry,
    studies_registry,
    optimizers_registry,
)
from .configuration import SimulationConfig, load_config_file
from .project import Project

__all__ = [
    "SPEED_OF_LIGHT",
    "C0",
    "VACUUM_PERMITTIVITY",
    "EPSILON_0",
    "VACUUM_PERMEABILITY",
    "MU_0",
    "VACUUM_IMPEDANCE",
    "ETA_0",
    "NEPER_TO_DB",
    "LengthUnit",
    "wavelength_to_frequency",
    "frequency_to_wavelength",
    "wavelength_to_wavenumber",
    "photon_energy_ev",
    "attenuation_conversion",
    "parse_quantity",
    "Parameter",
    "ParameterSet",
    "validate_positive_number",
    "validate_non_negative_number",
    "validate_thickness",
    "validate_domains_assigned",
    "validate_wavelength_range",
    "solvers_registry",
    "meshers_registry",
    "materials_registry",
    "dispersion_registry",
    "pcf_templates_registry",
    "studies_registry",
    "optimizers_registry",
    "SimulationConfig",
    "load_config_file",
    "Project",
]

