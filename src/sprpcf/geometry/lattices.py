"""Lattice generation algorithms for photonic crystal fibers."""

import math
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from .holes import Hole


class LatticeGenerator:
    """Generates coordinate positions and hole instances for various PCF lattice arrangements."""

    @staticmethod
    def hexagonal(
        pitch: float,
        rings: int,
        hole_radius: float,
        center: Tuple[float, float] = (0.0, 0.0),
        missing_holes: Optional[Set[Tuple[int, int]]] = None,
        material_tag: str = "air",
        ellipticity: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> List[Hole]:
        """Generate a hexagonal (triangular) lattice of air holes.
        
        Parameters:
            pitch: Distance between centers of adjacent holes (meters).
            rings: Number of concentric hexagonal rings (outermost ring number).
            hole_radius: Radius of the circular holes (meters).
            center: Coordinate of center defect / origin.
            missing_holes: Set of (ring_idx, pos_in_ring) or specific hole indices to omit.
            material_tag: Material tag (default 'air').
            ellipticity: Ratio of semi_major to semi_minor axis (1.0 = circular).
            rotation_deg: Global rotation of the lattice in degrees.
        """
        missing = missing_holes or set()
        holes: List[Hole] = []
        hole_idx = 0

        # Unit vectors for hexagonal 60-degree basis
        theta_rot = math.radians(rotation_deg)
        cos_r = math.cos(theta_rot)
        sin_r = math.sin(theta_rot)

        for ring in range(1, rings + 1):
            for side in range(6):
                side_angle = side * (np.pi / 3.0)
                # Starting point for this side of the hexagon
                start_x = ring * pitch * np.cos(side_angle)
                start_y = ring * pitch * np.sin(side_angle)
                # Step vector along the side of the hexagon
                next_angle = (side + 2) * (np.pi / 3.0)
                step_x = pitch * np.cos(next_angle)
                step_y = pitch * np.sin(next_angle)

                for step in range(ring):
                    if (ring, side * ring + step) in missing:
                        continue
                    px = start_x + step * step_x
                    py = start_y + step * step_y

                    # Apply rotation
                    rx = px * cos_r - py * sin_r + center[0]
                    ry = px * sin_r + py * cos_r + center[1]

                    h = Hole(
                        name=f"hole_hex_r{ring}_s{side}_p{step}_{hole_idx}",
                        center=(rx, ry),
                        radius=hole_radius,
                        material_tag=material_tag,
                        ellipticity=ellipticity,
                        rotation_deg=rotation_deg,
                    )
                    holes.append(h)
                    hole_idx += 1

        return holes

    @staticmethod
    def square(
        pitch_x: float,
        pitch_y: float,
        num_x: int,
        num_y: int,
        hole_radius: float,
        center: Tuple[float, float] = (0.0, 0.0),
        omit_center: bool = True,
        missing_coords: Optional[Set[Tuple[int, int]]] = None,
        material_tag: str = "air",
    ) -> List[Hole]:
        """Generate a square / rectangular lattice of holes."""
        missing = missing_coords or set()
        holes: List[Hole] = []
        hx = num_x // 2
        hy = num_y // 2
        hole_idx = 0

        for ix in range(-hx, hx + 1):
            for iy in range(-hy, hy + 1):
                if omit_center and ix == 0 and iy == 0:
                    continue
                if (ix, iy) in missing:
                    continue
                x = center[0] + ix * pitch_x
                y = center[1] + iy * pitch_y
                h = Hole(
                    name=f"hole_sq_{ix}_{iy}_{hole_idx}",
                    center=(x, y),
                    radius=hole_radius,
                    material_tag=material_tag,
                )
                holes.append(h)
                hole_idx += 1

        return holes

    @staticmethod
    def concentric_rings(
        ring_radii: List[float],
        holes_per_ring: List[int],
        hole_radius: float,
        center: Tuple[float, float] = (0.0, 0.0),
        phase_angles_deg: Optional[List[float]] = None,
        material_tag: str = "air",
    ) -> List[Hole]:
        """Generate circular concentric ring lattice of holes."""
        if len(ring_radii) != len(holes_per_ring):
            raise ValueError("ring_radii and holes_per_ring must have equal lengths.")
        holes: List[Hole] = []
        hole_idx = 0
        phases = phase_angles_deg or [0.0] * len(ring_radii)

        for ring_idx, (r_ring, n_holes, phase) in enumerate(zip(ring_radii, holes_per_ring, phases)):
            angles = np.linspace(0, 2 * np.pi, n_holes, endpoint=False) + math.radians(phase)
            for a_idx, ang in enumerate(angles):
                x = center[0] + r_ring * math.cos(ang)
                y = center[1] + r_ring * math.sin(ang)
                h = Hole(
                    name=f"hole_ring{ring_idx}_{a_idx}_{hole_idx}",
                    center=(x, y),
                    radius=hole_radius,
                    material_tag=material_tag,
                )
                holes.append(h)
                hole_idx += 1

        return holes

    @staticmethod
    def custom_coordinates(
        coordinates: List[Tuple[float, float]],
        radii: List[float],
        material_tag: str = "air",
        prefix: str = "custom_hole",
    ) -> List[Hole]:
        """Generate holes from an arbitrary user-defined list of coordinates and radii."""
        if len(coordinates) != len(radii):
            raise ValueError("coordinates and radii must have the same length.")
        holes = []
        for i, ((x, y), r) in enumerate(zip(coordinates, radii)):
            h = Hole(name=f"{prefix}_{i}", center=(x, y), radius=r, material_tag=material_tag)
            holes.append(h)
        return holes

