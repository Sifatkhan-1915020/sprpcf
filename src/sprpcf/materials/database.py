"""Optical material database and registry."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from .base import Material
from .dispersion import ConstantIndexMaterial, TabulatedMaterial
from .dielectric import create_fused_silica, create_borosilicate_glass, create_air
from .metals import (
    create_gold_rakic,
    create_silver_rakic,
    create_copper_rakic,
    create_aluminum_rakic,
    create_titanium_adhesion,
    create_titanium_dioxide,
)
from .polymers import create_pmma, create_topas, create_pdms
from .biological import WaterMaterial, GlucoseSolutionMaterial, AnalyteMaterial, create_bioreceptor_layer
from ..exceptions import MissingMaterialError


class MaterialDatabase:
    """Central repository of optical materials for SPR-PCF Studio."""

    def __init__(self):
        self._materials: Dict[str, Material] = {}
        self._load_standard_library()

    def register(self, material: Material, override: bool = True) -> None:
        key = material.name.lower().strip()
        if key in self._materials and not override:
            raise ValueError(f"Material '{material.name}' already registered.")
        self._materials[key] = material

    def get(self, name: str) -> Material:
        key = name.lower().strip()
        if key not in self._materials:
            raise MissingMaterialError(
                f"Material '{name}' not found in database. Registered materials: {self.list()}"
            )
        return self._materials[key]

    def remove(self, name: str) -> bool:
        key = name.lower().strip()
        if key in self._materials:
            del self._materials[key]
            return True
        return False

    def list(self) -> List[str]:
        """Return list of all registered material names."""
        return sorted(list(self._materials.keys()))

    def search(self, query: str) -> List[str]:
        """Search materials by substring query in name, category, or notes."""
        q = query.lower().strip()
        matches = []
        for name, mat in self._materials.items():
            if q in name or q in mat.category.lower() or q in mat.notes.lower():
                matches.append(name)
        return sorted(matches)

    def import_json(self, filepath: Union[str, Path]) -> None:
        """Import material definitions from a JSON file."""
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        # Parse items
        if isinstance(data, list):
            for item in data:
                self._import_item(item)
        elif isinstance(data, dict):
            for name, item in data.items():
                if "name" not in item:
                    item["name"] = name
                self._import_item(item)

    def _import_item(self, item: Dict[str, Any]) -> None:
        model = item.get("model", "constant").lower()
        name = item.get("name", "custom_material")
        if model == "constant":
            mat = ConstantIndexMaterial(
                name=name,
                n=item.get("n", 1.0),
                k=item.get("k", 0.0),
                category=item.get("category", "dielectric"),
            )
        elif model == "tabulated":
            mat = TabulatedMaterial(
                name=name,
                wavelengths_m=item["wavelengths_m"],
                n_values=item["n_values"],
                k_values=item.get("k_values"),
                category=item.get("category", "metal"),
                doi_source=item.get("doi_source", ""),
            )
        else:
            raise ValueError(f"Unsupported imported material model: {model}")
        self.register(mat)

    def export_json(self, filepath: Union[str, Path]) -> None:
        """Export all registered materials to JSON."""
        path = Path(filepath)
        export_dict = {name: mat.to_dict() for name, mat in self._materials.items()}
        path.write_text(json.dumps(export_dict, indent=2), encoding="utf-8")

    def _load_standard_library(self) -> None:
        # Dielectrics / Glasses
        self.register(create_fused_silica())
        self.register(create_borosilicate_glass())
        self.register(create_air())

        # Plasmonic metals
        self.register(create_gold_rakic())
        self.register(create_silver_rakic())
        self.register(create_copper_rakic())
        self.register(create_aluminum_rakic())
        self.register(create_titanium_adhesion())
        self.register(create_titanium_dioxide())

        # Polymers
        self.register(create_pmma())
        self.register(create_topas())
        self.register(create_pdms())

        # Biological & Analytes
        self.register(WaterMaterial())
        self.register(AnalyteMaterial("analyte", 1.330))
        self.register(AnalyteMaterial("water_analyte", 1.333))
        self.register(AnalyteMaterial("analyte_1", 1.330))
        self.register(AnalyteMaterial("analyte_2", 1.340))
        self.register(GlucoseSolutionMaterial(concentration_g_per_l=0.0))
        self.register(create_bioreceptor_layer())


# Global material database instance
materials = MaterialDatabase()

