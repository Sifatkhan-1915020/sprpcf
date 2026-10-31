"""Unit tests for PCF geometry primitives, lattices, and CSG domain partitioning."""

import pytest
import math
from sprpcf.geometry.primitives import Circle, Ellipse, Rectangle, Annulus, RegularPolygon
from sprpcf.geometry.cross_section import PCFGeometry
from sprpcf.geometry.pcf_templates import ConventionalPCF, DShapedPCF, DualChannelPCF
from sprpcf.geometry.boolean import boolean_difference, boolean_union, partition_domains


def test_geometric_primitives():
    # Circle area = pi * r^2
    r = 2.0e-6
    c = Circle("circ", (0, 0), r)
    expected_area = math.pi * r**2
    assert abs(c.area - expected_area) / expected_area < 0.01

    # Rectangle area = w * h
    rect = Rectangle("rect", (0, 0), 4e-6, 3e-6)
    assert abs(rect.area - 12e-12) / 12e-12 < 0.001

    # Annulus area = pi * (r_out^2 - r_in^2)
    ann = Annulus("ann", (0, 0), 1e-6, 3e-6)
    expected_ann = math.pi * (9e-12 - 1e-12)
    assert abs(ann.area - expected_ann) / expected_ann < 0.01


def test_boolean_operations():
    c1 = Circle("c1", (0, 0), 5e-6)
    c2 = Circle("c2", (0, 0), 2e-6)
    diff = boolean_difference(c1, c2)
    assert diff.area < c1.area
    assert abs(diff.area - (c1.area - c2.area)) / c1.area < 0.01


def test_pcf_geometry_partitioning():
    geom = PCFGeometry(cladding_radius=10.0e-6)
    holes = geom.create_lattice(
        lattice_type="hexagonal",
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.5e-6,
    )
    assert len(holes) == 18  # 6 in ring 1, 12 in ring 2

    # Partition domain
    partitioned = geom.partition()
    assert "cladding" in partitioned
    # All 18 holes must exist as distinct non-overlapping regions
    for h in holes:
        assert h.name in partitioned

    # Topological validation
    issues = geom.validate()
    assert len(issues) == 0


def test_d_shaped_template():
    d_pcf = DShapedPCF(
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.5e-6,
        polish_depth=3.0e-6,
        metal_thickness=40e-9,
        cladding_radius=8.0e-6,
    )
    partitioned = d_pcf.partition()
    assert "plasmonic_metal_film" in partitioned
    assert "sensing_analyte_channel" in partitioned

    summary = d_pcf.get_summary()
    assert summary["coating_count"] >= 1
    assert summary["channel_count"] >= 1

