"""Unit tests for finite element mesh generation, quality metrics, and validation."""

import pytest
import numpy as np
from sprpcf.geometry.cross_section import PCFGeometry
from sprpcf.geometry.pcf_templates import ConventionalPCF
from sprpcf.meshing.settings import MeshSettings
from sprpcf.meshing.native_backend import NativeMeshGenerator
from sprpcf.meshing.validation import validate_mesh


def test_mesh_generation_and_quality():
    geom = ConventionalPCF(
        pitch=2.0e-6,
        rings=1,
        hole_radius=0.5e-6,
        cladding_radius=5.0e-6,
    )
    settings = MeshSettings(global_size=0.6e-6)
    mesher = NativeMeshGenerator(settings)
    mesh = mesher.generate(geom)

    assert mesh.num_nodes > 10
    assert mesh.num_elements > 10
    assert len(mesh.element_domains) == mesh.num_elements

    # Check element areas are all strictly positive
    areas = mesh.element_areas
    assert np.all(areas > 0.0)

    # Validate mesh quality
    report = validate_mesh(mesh)
    assert report.is_valid
    assert report.min_quality > 1e-4
    assert report.mean_quality > 0.2
