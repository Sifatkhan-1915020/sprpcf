"""Base study interfaces for parametric computational optical simulations."""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, List, Optional, Union
from ..solvers.base import SolverConfig, SolverResult


class StudyResult:
    """Encapsulates the outputs of a computational study."""

    def __init__(
        self,
        study_name: str,
        results: List[SolverResult],
        parameters_swept: Dict[str, List[Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.study_name = study_name
        self.results = results
        self.parameters_swept = parameters_swept
        self.metadata = metadata or {}
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def __len__(self) -> int:
        return len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "timestamp": self.timestamp,
            "steps_count": len(self.results),
            "parameters_swept": self.parameters_swept,
            "metadata": self.metadata,
        }


class BaseStudy(ABC):
    """Abstract base class for all simulation studies."""

    def __init__(self, name: str, solver_config: Optional[SolverConfig] = None):
        self.name = name
        self.solver_config = solver_config or SolverConfig()
        self.metadata: Dict[str, Any] = {}

    def set_solver(
        self,
        backend: str = "native",
        tolerance: float = 1e-8,
        num_modes: int = 6,
        target_index: Optional[float] = 1.44,
        grid_points: int = 150,
    ) -> None:
        """Configure the underlying mode solver."""
        self.solver_config.solver_backend = backend
        self.solver_config.tolerance = tolerance
        self.solver_config.num_modes = num_modes
        self.solver_config.target_index = target_index
        self.solver_config.grid_points_x = grid_points
        self.solver_config.grid_points_y = grid_points

    @abstractmethod
    def run(self, project: Any) -> StudyResult:
        """Execute the study on the provided PCF project."""
        pass

