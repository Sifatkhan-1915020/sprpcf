"""Universal component registry for SPR-PCF Studio.

Provides extension points for custom geometry shapes, material dispersion models,
meshing backends, electromagnetic solvers, studies, and optimizers.
"""

from typing import Dict, Type, Any, Optional, Callable


class ComponentRegistry:
    """Generic registry for pluggable components."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, identifier: str, item: Any, override: bool = False) -> None:
        key = identifier.lower().strip()
        if key in self._registry and not override:
            raise ValueError(f"Component '{identifier}' is already registered in registry '{self.name}'.")
        self._registry[key] = item

    def get(self, identifier: str) -> Optional[Any]:
        return self._registry.get(identifier.lower().strip())

    def create(self, identifier: str, *args, **kwargs) -> Any:
        item = self.get(identifier)
        if item is None:
            raise KeyError(f"Identifier '{identifier}' not found in registry '{self.name}'. Registered: {self.list_keys()}")
        if callable(item):
            return item(*args, **kwargs)
        return item

    def list_keys(self) -> list:
        return sorted(list(self._registry.keys()))

    def __contains__(self, identifier: str) -> bool:
        return identifier.lower().strip() in self._registry


# Global registries
solvers_registry = ComponentRegistry("solvers")
meshers_registry = ComponentRegistry("meshers")
materials_registry = ComponentRegistry("materials")
dispersion_registry = ComponentRegistry("dispersion_models")
geometry_primitives_registry = ComponentRegistry("geometry_primitives")
pcf_templates_registry = ComponentRegistry("pcf_templates")
studies_registry = ComponentRegistry("studies")
optimizers_registry = ComponentRegistry("optimizers")

