"""Configuration loader and schema validator for YAML and JSON simulation manifests."""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
from pydantic import BaseModel, Field, field_validator
from .units import LengthUnit, parse_quantity
from ..exceptions import SimulationError


class ProjectConfig(BaseModel):
    name: str = "SPR_PCF_Simulation"
    description: Optional[str] = "SPR-PCF biosensor simulation"
    version: Optional[str] = "0.1.0"


class LatticeConfig(BaseModel):
    type: str = "hexagonal"  # hexagonal, rectangular, circular
    pitch: Union[float, str] = 2.0e-6  # meters or string like '2.0 um'
    rings: int = 3
    hole_radius: Union[float, str] = 0.5e-6


class GeometryConfig(BaseModel):
    architecture: str = "conventional"  # conventional, d_shaped, dual_channel, exposed_core, etc.
    cladding_radius: Union[float, str] = 10.0e-6
    lattice: Optional[LatticeConfig] = None
    core_shape: str = "circular"
    core_radius: Optional[Union[float, str]] = None
    d_cut_depth: Optional[Union[float, str]] = None
    analyte_channel: Optional[Dict[str, Any]] = None


class CoatingConfig(BaseModel):
    surface: str = "sensing_surface"
    material: str = "gold"
    thickness: Union[float, str] = 40.0e-9  # e.g., '40 nm' or 40e-9
    coverage_angle_deg: Optional[float] = 360.0


class MaterialItemConfig(BaseModel):
    model: str = "sellmeier"  # sellmeier, drude, drude_lorentz, constant, tabulated
    refractive_index: Optional[Union[float, complex]] = None
    source: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class MeshConfig(BaseModel):
    backend: str = "native"  # native, netgen, gmsh
    global_size: Union[float, str] = 0.2e-6
    min_size: Optional[Union[float, str]] = None
    max_size: Optional[Union[float, str]] = None
    refine_interfaces: bool = True
    metal_layer_elements: int = 3


class StudyConfig(BaseModel):
    type: str = "wavelength_sweep"  # eigenmode, wavelength_sweep, analyte_sweep, parameter_sweep
    wavelength_start: Union[float, str] = 600.0e-9
    wavelength_stop: Union[float, str] = 1600.0e-9
    wavelength_points: int = 51
    target_modes: int = 6
    polarization: str = "both"  # x, y, both
    target_index: Optional[float] = 1.44


class BoundaryConfig(BaseModel):
    type: str = "pml"  # pml, pec, pmc, scattering
    thickness: Union[float, str] = 1.5e-6
    scaling_factor: float = 1.0


class SimulationConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    coatings: List[CoatingConfig] = Field(default_factory=list)
    materials: Dict[str, MaterialItemConfig] = Field(default_factory=dict)
    domain_assignments: Dict[str, str] = Field(default_factory=dict)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    study: StudyConfig = Field(default_factory=StudyConfig)


def parse_quantity_to_si(val: Union[float, int, str], default_unit: str = "m") -> float:
    """Safely convert quantity strings or numbers to SI floats."""
    if isinstance(val, (int, float)):
        return float(val)
    parsed_num, unit = parse_quantity(val, default_unit)
    if unit in ["m", "um", "nm", "cm", "mm"]:
        return LengthUnit.to_meters(parsed_num, unit)
    return parsed_num


def load_config_file(filepath: Union[str, Path]) -> SimulationConfig:
    """Load and validate a YAML or JSON simulation configuration file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in [".yaml", ".yml"]:
        raw_dict = yaml.safe_load(content)
    elif path.suffix.lower() == ".json":
        raw_dict = json.loads(content)
    else:
        raise ValueError(f"Unsupported file extension '{path.suffix}'. Use .yaml, .yml, or .json")

    return SimulationConfig(**raw_dict)

