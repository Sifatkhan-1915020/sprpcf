"""Mesh data structure representing discretized 2D computational domains."""

from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np


class Mesh:
    """Discretized 2D finite element / cell mesh."""

    def __init__(
        self,
        nodes: np.ndarray,  # shape (N_nodes, 2) in meters
        elements: np.ndarray,  # shape (N_elements, 3) or (N_elements, 4) node indices (0-based)
        element_domains: List[str],  # domain name for each element
        boundary_edges: Optional[Dict[str, List[Tuple[int, int]]]] = None,  # boundary_name -> list of (n1, n2)
        backend_name: str = "native",
    ):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.element_domains = list(element_domains)
        self.boundary_edges = boundary_edges or {}
        self.backend_name = backend_name

        if len(self.elements) != len(self.element_domains):
            raise ValueError(f"Elements count ({len(self.elements)}) must match domain tags ({len(self.element_domains)})")

        self._element_areas: Optional[np.ndarray] = None
        self._element_centroids: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def element_centroids(self) -> np.ndarray:
        """Compute centroid (x, y) for every element."""
        if self._element_centroids is None:
            # For each element, average the node coordinates
            # nodes shape: (N, 2), elements shape: (M, K)
            self._element_centroids = np.mean(self.nodes[self.elements], axis=1)
        return self._element_centroids

    @property
    def element_areas(self) -> np.ndarray:
        """Compute area of every triangular or quadrilateral element."""
        if self._element_areas is None:
            if self.elements.shape[1] == 3:
                # Triangular elements: |(x1(y2-y3) + x2(y3-y1) + x3(y1-y2))| / 2
                p0 = self.nodes[self.elements[:, 0]]
                p1 = self.nodes[self.elements[:, 1]]
                p2 = self.nodes[self.elements[:, 2]]
                areas = 0.5 * np.abs(
                    p0[:, 0] * (p1[:, 1] - p2[:, 1])
                    + p1[:, 0] * (p2[:, 1] - p0[:, 1])
                    + p2[:, 0] * (p0[:, 1] - p1[:, 1])
                )
                self._element_areas = areas
            else:
                # Generalized polygon via shoelace
                pts = self.nodes[self.elements]  # shape (M, K, 2)
                x = pts[:, :, 0]
                y = pts[:, :, 1]
                areas = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))
                self._element_areas = areas
        return self._element_areas

    def get_domain_indices(self, domain_name: str) -> np.ndarray:
        """Return array of element indices belonging to the specified domain."""
        return np.where(np.array(self.element_domains) == domain_name)[0]

    def get_unique_domains(self) -> List[str]:
        return sorted(list(set(self.element_domains)))

    def get_summary(self) -> Dict[str, Any]:
        areas = self.element_areas
        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "backend": self.backend_name,
            "domains": self.get_unique_domains(),
            "min_element_area_m2": float(np.min(areas)) if len(areas) else 0.0,
            "max_element_area_m2": float(np.max(areas)) if len(areas) else 0.0,
            "total_area_m2": float(np.sum(areas)) if len(areas) else 0.0,
        }

