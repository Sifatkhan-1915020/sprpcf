"""Mesh validation and quality assessment routines."""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from .base import Mesh
from ..exceptions import InvalidMeshError


class MeshQualityReport:
    """Statistical summary of mesh geometric quality and interface refinement."""

    def __init__(self, mesh: Mesh, interface_y: Optional[float] = None):
        self.num_nodes = mesh.num_nodes
        self.num_elements = mesh.num_elements
        self.domains = mesh.get_unique_domains()

        # Compute triangle qualities, aspect ratios, and edge lengths
        qualities, aspect_ratios, min_edges = self._compute_triangle_metrics(mesh)
        self.min_quality = float(np.min(qualities)) if len(qualities) else 0.0
        self.mean_quality = float(np.mean(qualities)) if len(qualities) else 0.0
        self.max_aspect_ratio = float(np.max(aspect_ratios)) if len(aspect_ratios) else 0.0
        self.mean_aspect_ratio = float(np.mean(aspect_ratios)) if len(aspect_ratios) else 0.0
        self.min_edge_length_m = float(np.min(min_edges)) if len(min_edges) else 0.0

        # Element quality distribution histogram (10 bins from 0 to 1)
        if len(qualities):
            counts, bin_edges = np.histogram(qualities, bins=10, range=(0.0, 1.0))
            self.quality_histogram = {"counts": counts.tolist(), "bin_edges": bin_edges.tolist()}
        else:
            self.quality_histogram = {"counts": [], "bin_edges": []}

        # Interface elements count (elements near interface, or in metal/analyte domains)
        interface_count = 0
        metal_analyte_domains = {"plasmonic_metal_film", "sensing_analyte_channel", "adhesion_layer", "gold", "analyte"}
        for dname in mesh.get_unique_domains():
            if any(term in dname.lower() for term in metal_analyte_domains):
                interface_count += len(mesh.get_domain_indices(dname))
        
        # If no explicit domain tags, count elements within 0.5 um of interface_y
        if interface_count == 0 and interface_y is not None:
            centroids_y = np.mean(mesh.nodes[mesh.elements[:, :3], 1], axis=1)
            interface_count = int(np.sum(np.abs(centroids_y - interface_y) <= 0.5e-6))
        elif interface_count == 0:
            # Fallback: estimate top 15% elements in y
            centroids_y = np.mean(mesh.nodes[mesh.elements[:, :3], 1], axis=1)
            p85 = np.percentile(centroids_y, 85)
            interface_count = int(np.sum(centroids_y >= p85))

        self.interface_elements_count = interface_count
        self.is_valid = bool(self.min_quality > 1e-4 and np.all(mesh.element_areas > 0.0))

    def _compute_triangle_metrics(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mesh.elements.shape[1] != 3:
            ones = np.ones(mesh.num_elements)
            return ones, ones, ones * 1e-6

        p0 = mesh.nodes[mesh.elements[:, 0]]
        p1 = mesh.nodes[mesh.elements[:, 1]]
        p2 = mesh.nodes[mesh.elements[:, 2]]

        # Edge lengths
        a = np.hypot(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1])
        b = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
        c = np.hypot(p0[:, 0] - p2[:, 0], p0[:, 1] - p2[:, 1])

        min_edge = np.minimum(np.minimum(a, b), c)

        # Semiperimeter
        s = 0.5 * (a + b + c)
        areas = mesh.element_areas
        # Inradius r_in = Area / s
        r_in = np.where(s > 1e-15, areas / s, 0.0)
        # Circumradius r_out = (a * b * c) / (4 * Area)
        denom = 4.0 * np.maximum(areas, 1e-25)
        r_out = (a * b * c) / denom

        # Normalized quality: 2 * r_in / r_out (1.0 for equilateral triangle)
        quality = np.where(r_out > 1e-15, 2.0 * r_in / r_out, 0.0)
        quality = np.clip(quality, 0.0, 1.0)

        # Aspect ratio: max(edge) / min(height)
        max_edge = np.maximum(np.maximum(a, b), c)
        aspect = max_edge / np.maximum(min_edge, 1e-20)

        return quality, aspect, min_edge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "min_quality": self.min_quality,
            "mean_quality": self.mean_quality,
            "max_aspect_ratio": self.max_aspect_ratio,
            "mean_aspect_ratio": self.mean_aspect_ratio,
            "min_edge_length_um": self.min_edge_length_m * 1e6,
            "interface_elements_count": self.interface_elements_count,
            "is_valid": self.is_valid,
            "domains": self.domains,
        }

    def __repr__(self) -> str:
        return (
            f"MeshQualityReport(elements={self.num_elements}, nodes={self.num_nodes}, "
            f"mean_q={self.mean_quality:.3f}, min_q={self.min_quality:.3e}, "
            f"max_aspect={self.max_aspect_ratio:.1f}, interface_el={self.interface_elements_count})"
        )


def validate_mesh(mesh: Mesh, min_quality_threshold: float = 1e-3, interface_y: Optional[float] = None) -> MeshQualityReport:
    """Validate mesh elements and raise InvalidMeshError on inverted or degenerate elements."""
    if mesh.num_elements == 0:
        raise InvalidMeshError("Mesh contains zero elements.")
    if mesh.num_nodes == 0:
        raise InvalidMeshError("Mesh contains zero nodes.")

    # Check for non-positive areas
    neg_areas = np.where(mesh.element_areas <= 0.0)[0]
    if len(neg_areas) > 0:
        raise InvalidMeshError(f"Mesh has {len(neg_areas)} inverted or zero-area elements!")

    report = MeshQualityReport(mesh, interface_y=interface_y)
    return report
