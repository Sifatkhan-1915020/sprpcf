"""Parametric management system for SPR-PCF Studio.

Supports unit-aware, bounded, categorical, and derived parameters with dependency tracking and cache invalidation.
"""

from typing import Any, Dict, Optional, Callable, Set, Union, List
from .units import LengthUnit, parse_quantity


class Parameter:
    """Represents a validated, unit-aware simulation parameter."""

    def __init__(
        self,
        name: str,
        value: Any,
        unit: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        description: str = "",
        category: str = "general",
        expression: Optional[str] = None,
    ):
        self.name = name
        self.unit = unit
        self.min_value = min_value
        self.max_value = max_value
        self.description = description
        self.category = category
        self.expression = expression
        self._value = None
        self.set_value(value)

    @property
    def value(self) -> Any:
        return self._value

    def set_value(self, val: Any) -> None:
        if isinstance(val, str) and self.unit:
            try:
                parsed_val, parsed_unit = parse_quantity(val, self.unit)
                # If length unit, convert to standard SI meter or target unit
                if self.unit in ["m", "um", "nm"]:
                    meters = LengthUnit.to_meters(parsed_val, parsed_unit)
                    val = LengthUnit.from_meters(meters, self.unit)
                else:
                    val = parsed_val
            except Exception:
                pass  # Keep as string if parsing not applicable

        if isinstance(val, (int, float)):
            val = float(val)
            if self.min_value is not None and val < self.min_value:
                raise ValueError(f"Parameter '{self.name}' value {val} is below minimum {self.min_value}")
            if self.max_value is not None and val > self.max_value:
                raise ValueError(f"Parameter '{self.name}' value {val} exceeds maximum {self.max_value}")

        self._value = val

    def get_si_value(self) -> float:
        """Return numeric value in standard SI units (meters for lengths, Hz for frequencies)."""
        if not isinstance(self._value, (int, float)):
            raise TypeError(f"Parameter '{self.name}' is non-numeric.")
        if self.unit:
            u = self.unit.lower().strip()
            if u in ["m", "um", "nm", "cm", "mm"]:
                return LengthUnit.to_meters(float(self._value), u)
        return float(self._value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self._value,
            "unit": self.unit,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
            "category": self.category,
            "expression": self.expression,
        }

    def __repr__(self) -> str:
        u = f" {self.unit}" if self.unit else ""
        return f"Parameter({self.name}={self._value}{u})"


class ParameterSet:
    """A collection of parameters with dependency tracking and cache invalidation listeners."""

    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._listeners: Set[Callable[[str, Any], None]] = set()

    def add(self, param: Parameter) -> None:
        self._parameters[param.name] = param

    def create(
        self,
        name: str,
        value: Any,
        unit: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        description: str = "",
        category: str = "general",
    ) -> Parameter:
        p = Parameter(
            name=name,
            value=value,
            unit=unit,
            min_value=min_value,
            max_value=max_value,
            description=description,
            category=category,
        )
        self.add(p)
        return p

    def get(self, name: str) -> Parameter:
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found.")
        return self._parameters[name]

    def get_value(self, name: str, default: Any = None) -> Any:
        if name in self._parameters:
            return self._parameters[name].value
        return default

    def get_si(self, name: str) -> float:
        """Get parameter value converted to SI base units."""
        return self.get(name).get_si_value()

    def set(self, name: str, value: Any) -> None:
        if name not in self._parameters:
            self.create(name=name, value=value)
        else:
            self._parameters[name].set_value(value)
        # Notify listeners to invalidate dependent caches
        self._notify_change(name, value)

    def register_change_listener(self, callback: Callable[[str, Any], None]) -> None:
        self._listeners.add(callback)

    def _notify_change(self, name: str, value: Any) -> None:
        for listener in list(self._listeners):
            listener(name, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self._parameters.items()}

    def keys(self):
        return self._parameters.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._parameters

    def __getitem__(self, name: str) -> Any:
        return self.get_value(name)

    def __setitem__(self, name: str, value: Any) -> None:
        self.set(name, value)

    def clone(self) -> "ParameterSet":
        ps = ParameterSet()
        for p in self._parameters.values():
            ps.create(
                name=p.name,
                value=p.value,
                unit=p.unit,
                min_value=p.min_value,
                max_value=p.max_value,
                description=p.description,
                category=p.category,
            )
        return ps

