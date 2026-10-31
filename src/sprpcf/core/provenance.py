"""Provenance tracking and simulation fingerprinting for reproducible science."""

import datetime
import hashlib
import json
import os
import platform
import sys
from typing import Dict, Any, Optional
from ..version import __version__


def get_environment_info() -> Dict[str, Any]:
    """Capture runtime execution environment metadata."""
    return {
        "sprpcf_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
        "pid": os.getpid(),
    }


def compute_simulation_fingerprint(config_dict: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint for a simulation configuration.
    
    Used for content-based caching and dependency tracking.
    """
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class SimulationMetadata:
    """Stores full execution provenance for a simulation run."""

    def __init__(self, project_name: str, study_name: str):
        self.project_name = project_name
        self.study_name = study_name
        self.timestamp_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.timestamp_end: Optional[str] = None
        self.environment = get_environment_info()
        self.fingerprint: str = ""
        self.execution_time_seconds: float = 0.0
        self.solver_status: str = "initialized"
        self.convergence_achieved: bool = False
        self.warnings: list = []
        self.notes: Dict[str, Any] = {}

    def finish(self, success: bool, execution_time: float, status: str = "completed") -> None:
        self.timestamp_end = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.convergence_achieved = success
        self.execution_time_seconds = execution_time
        self.solver_status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "study_name": self.study_name,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "execution_time_seconds": self.execution_time_seconds,
            "solver_status": self.solver_status,
            "convergence_achieved": self.convergence_achieved,
            "fingerprint": self.fingerprint,
            "environment": self.environment,
            "warnings": self.warnings,
            "notes": self.notes,
        }

