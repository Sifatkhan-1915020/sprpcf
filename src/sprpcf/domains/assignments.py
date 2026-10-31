"""Domain material assignment and physical tagging system."""

from typing import Dict, List, Optional, Any, Union
from ..materials.base import Material
from ..materials.database import materials
from ..core.validation import validate_domains_assigned
from ..exceptions import DomainAssignmentError, MissingMaterialError


class DomainAssignmentManager:
    """Manages mapping between named geometric regions and optical materials."""

    def __init__(self):
        self._assignments: Dict[str, str] = {}  # region_name -> material_name

    def assign(self, region_name: str, material: Union[str, Material]) -> None:
        """Assign a material to a geometric region."""
        mat_name = material.name if isinstance(material, Material) else str(material)
        # Ensure material exists in global database or is a valid material object
        if isinstance(material, Material):
            materials.register(material)
        elif mat_name not in materials.list():
            raise MissingMaterialError(f"Material '{mat_name}' is not in database. Available: {materials.list()}")

        self._assignments[region_name] = mat_name

    def get_material_name(self, region_name: str) -> Optional[str]:
        return self._assignments.get(region_name)

    def get_material(self, region_name: str) -> Material:
        mat_name = self.get_material_name(region_name)
        if not mat_name:
            raise DomainAssignmentError(f"Region '{region_name}' has no material assigned.")
        return materials.get(mat_name)

    def get_all_assignments(self) -> Dict[str, str]:
        return dict(self._assignments)

    def validate(self, region_names: List[str]) -> None:
        """Check that all required regions have been assigned materials."""
        validate_domains_assigned(region_names, self._assignments)

    def to_dict(self) -> Dict[str, str]:
        return dict(self._assignments)
