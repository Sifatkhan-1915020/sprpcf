"""Master Project container coordinating geometry, materials, meshing, studies, and execution."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from .parameters import ParameterSet
from .provenance import SimulationMetadata
from ..geometry.cross_section import PCFGeometry
from ..domains.assignments import DomainAssignmentManager
from ..materials.base import Material
from ..materials.database import materials
from ..meshing.base import Mesh
from ..meshing.settings import MeshSettings
from ..meshing import generate_mesh
from ..studies.base import BaseStudy, StudyResult
from ..exceptions import SimulationError, DomainAssignmentError


class Project:
    """Master SPR-PCF project orchestrator."""

    def __init__(self, name: str = "SPR_PCF_Sensor"):
        self.name = name
        self.geometry: PCFGeometry = PCFGeometry(name=f"{name}_geom")
        self.parameters = ParameterSet()
        self.domain_assignments = DomainAssignmentManager()
        self.mesh: Optional[Mesh] = None
        self.studies: Dict[str, BaseStudy] = {}
        self.active_study_name: Optional[str] = None
        self.last_results: Optional[StudyResult] = None
        self.metadata = SimulationMetadata(project_name=name, study_name="")

    def set_geometry(self, geometry: PCFGeometry) -> None:
        """Assign or replace the PCF cross-section geometry."""
        self.geometry = geometry
        # Invalidate mesh if geometry changes
        self.mesh = None

    def assign_material(self, region_name: str, material: Union[str, Material]) -> None:
        """Assign an optical material to a named geometric domain."""
        self.domain_assignments.assign(region_name, material)

    def get_active_domain_materials(self) -> Dict[str, Material]:
        """Resolve materials for all partitioned domains in geometry."""
        partitioned = self.geometry.partition()
        resolved: Dict[str, Material] = {}

        for reg_name, reg in partitioned.items():
            # Check direct assignment by name
            mat_name = self.domain_assignments.get_material_name(reg_name)
            if not mat_name:
                # Check assignment by material tag (e.g. 'silica', 'gold', 'air', 'analyte')
                if reg.material_tag:
                    mat_name = self.domain_assignments.get_material_name(reg.material_tag)
                    if not mat_name and reg.material_tag in materials.list():
                        mat_name = reg.material_tag

            if not mat_name:
                # Intelligent fallbacks based on name heuristics
                if "core" in reg_name.lower():
                    mat_name = "silica"
                elif "clad" in reg_name.lower():
                    mat_name = "silica"
                elif "hole" in reg_name.lower():
                    mat_name = "air"
                elif "metal" in reg_name.lower() or "coating" in reg_name.lower():
                    mat_name = "gold"
                elif "channel" in reg_name.lower() or "analyte" in reg_name.lower():
                    mat_name = "analyte"
                elif "pml" in reg_name.lower():
                    mat_name = "silica"
                else:
                    mat_name = "silica"

            resolved[reg_name] = materials.get(mat_name)

        return resolved

    def generate_mesh(
        self,
        backend: str = "native",
        global_size: float = 0.25e-6,
        refine_interfaces: bool = True,
        **kwargs,
    ) -> Mesh:
        """Discretize the PCF geometry into a computational mesh."""
        settings = MeshSettings(
            global_size=global_size,
            refine_interfaces=refine_interfaces,
            backend=backend,
            **kwargs,
        )
        self.mesh = generate_mesh(self.geometry, settings=settings, backend=backend)
        return self.mesh

    def add_study(self, study: BaseStudy, set_active: bool = True) -> None:
        """Register a computational study."""
        self.studies[study.name] = study
        if set_active or self.active_study_name is None:
            self.active_study_name = study.name

    def run(self, study_name: Optional[str] = None) -> StudyResult:
        """Execute the active or specified study."""
        name = study_name or self.active_study_name
        if not name or name not in self.studies:
            if not self.studies:
                # Default to EigenmodeStudy if none was registered
                from ..studies.eigenmode import EigenmodeStudy
                default_study = EigenmodeStudy(name="default_eigenmode_study")
                self.add_study(default_study)
                name = default_study.name
            else:
                name = list(self.studies.keys())[0]

        active_study = self.studies[name]
        self.metadata.study_name = name

        # Execute
        results = active_study.run(self)
        self.last_results = results
        return results

    def validate(self) -> List[str]:
        """Validate topological integrity and physical assignments."""
        issues = []
        issues.extend(self.geometry.validate())
        return issues

    def get_summary(self) -> Dict[str, Any]:
        """Return project overview."""
        return {
            "name": self.name,
            "geometry": self.geometry.get_summary(),
            "materials": self.domain_assignments.to_dict(),
            "mesh": self.mesh.get_summary() if self.mesh else None,
            "studies": list(self.studies.keys()),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save project metadata and configuration to JSON."""
        path = Path(filepath)
        data = {
            "name": self.name,
            "geometry": self.geometry.name,
            "domain_assignments": self.domain_assignments.to_dict(),
            "parameters": self.parameters.to_dict(),
            "studies": list(self.studies.keys()),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Project":
        """Load project configuration from JSON."""
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        proj = cls(name=data["name"])
        for reg, mat in data.get("domain_assignments", {}).items():
            proj.assign_material(reg, mat)
        return proj

