"""Configuration and control parameters for mesh generation."""

from typing import Dict, Optional, Any, Union
from pydantic import BaseModel, Field


class MeshSettings(BaseModel):
    """Parameters governing global and localized mesh refinement."""

    global_size: float = 0.25e-6  # meters (~250 nm)
    min_size: Optional[float] = 0.01e-6  # meters (10 nm)
    max_size: Optional[float] = 0.8e-6   # meters (800 nm)
    growth_rate: float = 1.3
    refine_interfaces: bool = True
    metal_layer_elements: int = 4  # Minimum number of element layers across thin metal film
    core_refinement_factor: float = 0.5  # Scale factor for core element size
    pml_refinement_factor: float = 1.5   # Scale factor for PML element size
    element_order: int = 1  # 1: linear, 2: quadratic
    backend: str = "native"  # 'native', 'gmsh', 'netgen'
    local_size_overrides: Dict[str, float] = Field(default_factory=dict)  # domain_name -> size_m

    def get_domain_target_size(self, domain_name: str, is_metal: bool = False, metal_thickness_m: Optional[float] = None) -> float:
        """Compute target element size for a given domain."""
        if domain_name in self.local_size_overrides:
            return self.local_size_overrides[domain_name]

        if is_metal and metal_thickness_m is not None:
            # Thin film layer: divide thickness into target element layers
            film_size = metal_thickness_m / max(1, self.metal_layer_elements)
            return max(self.min_size or 1e-9, film_size)

        if "core" in domain_name.lower():
            return self.global_size * self.core_refinement_factor
        if "pml" in domain_name.lower():
            return self.global_size * self.pml_refinement_factor

        return self.global_size

