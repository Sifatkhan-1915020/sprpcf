"""Geometry module for SPR-PCF Studio."""

from .base import Region2D
from .primitives import Circle, Ellipse, Rectangle, RegularPolygon, Annulus, ArbitraryPolygon
from .holes import Hole
from .coatings import CoatingLayer
from .channels import AnalyteChannel
from .lattices import LatticeGenerator
from .cross_section import PCFGeometry
from .boolean import boolean_difference, boolean_union, boolean_intersection, partition_domains
from .pcf_templates import ConventionalPCF, DShapedPCF, DualChannelPCF, HollowCorePCF
from .validation import validate_geometry, GeometryValidationReport
from .transforms import translate_region, rotate_region, scale_region
from .exporters import export_geometry_to_json, export_geometry_to_svg

__all__ = [
    "Region2D",
    "Circle",
    "Ellipse",
    "Rectangle",
    "RegularPolygon",
    "Annulus",
    "ArbitraryPolygon",
    "Hole",
    "CoatingLayer",
    "AnalyteChannel",
    "LatticeGenerator",
    "PCFGeometry",
    "boolean_difference",
    "boolean_union",
    "boolean_intersection",
    "partition_domains",
    "ConventionalPCF",
    "DShapedPCF",
    "DualChannelPCF",
    "HollowCorePCF",
    "validate_geometry",
    "GeometryValidationReport",
    "translate_region",
    "rotate_region",
    "scale_region",
    "export_geometry_to_json",
    "export_geometry_to_svg",
]

