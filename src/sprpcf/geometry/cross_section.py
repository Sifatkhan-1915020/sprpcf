"""Master 2D PCF geometry container and domain partitioning engine."""

from typing import List, Dict, Tuple, Optional, Any, Set, Union
import shapely.geometry as sg
import shapely.affinity as sa
from shapely.validation import make_valid
from .base import Region2D
from .primitives import Circle, Rectangle, Annulus
from .holes import Hole
from .coatings import CoatingLayer
from .channels import AnalyteChannel
from .lattices import LatticeGenerator
from .boolean import partition_domains, detect_overlapping_regions, CompositeRegion
from ..exceptions import InvalidGeometryError, OverlappingRegionError


class PCFGeometry:
    """Master container for 2D photonic crystal fiber cross-sectional geometry."""

    def __init__(self, cladding_radius: float = 10.0e-6, name: str = "pcf_geometry"):
        self.name = name
        self.cladding_radius = float(cladding_radius)
        self.center = (0.0, 0.0)

        # Main components
        self._cladding_boundary: Optional[Region2D] = Circle("cladding", self.center, self.cladding_radius, material_tag="silica")
        self._core: Optional[Region2D] = None
        self._holes: Dict[str, Hole] = {}
        self._coatings: Dict[str, CoatingLayer] = {}
        self._channels: Dict[str, AnalyteChannel] = {}
        self._custom_regions: Dict[str, Region2D] = {}
        self._pml: Optional[Region2D] = None
        self._pml_thickness: float = 0.0

        # Caches
        self._partitioned_regions: Optional[Dict[str, Region2D]] = None

    def invalidate_cache(self) -> None:
        self._partitioned_regions = None

    # --- Cladding & Boundary ---
    def set_cladding(self, radius: float, material_tag: str = "silica") -> None:
        self.cladding_radius = float(radius)
        self._cladding_boundary = Circle("cladding", self.center, self.cladding_radius, material_tag=material_tag)
        self.invalidate_cache()

    def set_d_shape(self, polish_depth: float, material_tag: str = "silica") -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Cut the circular cladding with a side-polished D-shaped flat plane at y = polish_depth.
        
        Returns the (start_point, end_point) of the polished surface.
        """
        if abs(polish_depth) >= self.cladding_radius:
            raise InvalidGeometryError(f"Polish depth {polish_depth} must be strictly inside cladding radius {self.cladding_radius}")

        # Half-width of flat cut
        import math
        half_w = math.sqrt(self.cladding_radius**2 - polish_depth**2)
        start_pt = (-half_w, polish_depth)
        end_pt = (half_w, polish_depth)

        # Circular fiber intersected with half-space y <= polish_depth
        fiber_circle = sg.Point(0.0, 0.0).buffer(self.cladding_radius, resolution=128)
        # Bounding box for y <= polish_depth
        r = self.cladding_radius * 2.0
        cut_box = sg.Polygon([(-r, -r), (r, -r), (r, polish_depth), (-r, polish_depth)])
        d_geom = fiber_circle.intersection(cut_box)

        self._cladding_boundary = CompositeRegion("cladding_d_shaped", d_geom, material_tag=material_tag)
        self.invalidate_cache()
        return start_pt, end_pt

    # --- Lattice Generation ---
    def create_lattice(
        self,
        lattice_type: str = "hexagonal",
        pitch: float = 2.0e-6,
        rings: int = 3,
        hole_radius: float = 0.5e-6,
        missing_holes: Optional[Set[Tuple[int, int]]] = None,
        material: str = "air",
        ellipticity: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> List[Hole]:
        """Generate periodic air hole lattice and register all holes."""
        if lattice_type.lower() in ["hexagonal", "triangular"]:
            holes = LatticeGenerator.hexagonal(
                pitch=pitch,
                rings=rings,
                hole_radius=hole_radius,
                center=self.center,
                missing_holes=missing_holes,
                material_tag=material,
                ellipticity=ellipticity,
                rotation_deg=rotation_deg,
            )
        elif lattice_type.lower() in ["square", "rectangular"]:
            holes = LatticeGenerator.square(
                pitch_x=pitch,
                pitch_y=pitch,
                num_x=2 * rings + 1,
                num_y=2 * rings + 1,
                hole_radius=hole_radius,
                center=self.center,
                omit_center=True,
                material_tag=material,
            )
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

        for h in holes:
            self._holes[h.name] = h
        self.invalidate_cache()
        return holes

    # --- Core Design ---
    def create_core(
        self,
        shape: str = "circular",
        radius: float = 1.5e-6,
        center: Tuple[float, float] = (0.0, 0.0),
        material: str = "silica",
    ) -> Region2D:
        """Define the central optical core region."""
        if shape.lower() == "circular":
            core = Circle("fiber_core", center, radius, material_tag=material)
        elif shape.lower() in ["rectangular", "rectangle"]:
            core = Rectangle("fiber_core", center, radius * 2.0, radius * 2.0, material_tag=material)
        else:
            raise ValueError(f"Unsupported core shape: {shape}")
        self._core = core
        self.invalidate_cache()
        return core

    # --- Hole Manipulation ---
    def add_hole(
        self,
        center: Tuple[float, float],
        radius: float,
        shape: str = "circle",
        material: str = "air",
        ellipticity: float = 1.0,
        rotation_deg: float = 0.0,
        name: Optional[str] = None,
    ) -> Hole:
        """Add an individual hole."""
        h_name = name or f"hole_{len(self._holes)}"
        hole = Hole(
            name=h_name,
            center=center,
            radius=radius,
            material_tag=material,
            ellipticity=ellipticity,
            rotation_deg=rotation_deg,
            shape=shape,
        )
        self._holes[h_name] = hole
        self.invalidate_cache()
        return hole

    def remove_region(self, name: str) -> bool:
        """Remove a named hole, coating, channel, or custom region."""
        removed = False
        if name in self._holes:
            del self._holes[name]
            removed = True
        if name in self._coatings:
            del self._coatings[name]
            removed = True
        if name in self._channels:
            del self._channels[name]
            removed = True
        if name in self._custom_regions:
            del self._custom_regions[name]
            removed = True
        if removed:
            self.invalidate_cache()
        return removed

    # --- Coatings ---
    def add_coating(
        self,
        thickness: float,
        material: str = "gold",
        coating_type: str = "planar",
        surface: str = "sensing_surface",
        center: Tuple[float, float] = (0.0, 0.0),
        inner_radius: Optional[float] = None,
        outer_radius: Optional[float] = None,
        start_point: Optional[Tuple[float, float]] = None,
        end_point: Optional[Tuple[float, float]] = None,
        coverage_angle_deg: float = 360.0,
        name: Optional[str] = None,
    ) -> CoatingLayer:
        """Add a nanoscale plasmonic or dielectric coating."""
        c_name = name or f"coating_{material}_{len(self._coatings)}"
        layer = CoatingLayer(
            name=c_name,
            thickness=thickness,
            material_tag=material,
            coating_type=coating_type,
            center=center,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            start_point=start_point,
            end_point=end_point,
            coverage_angle_deg=coverage_angle_deg,
        )
        self._coatings[c_name] = layer
        self.invalidate_cache()
        return layer

    # --- Analyte Channels ---
    def add_analyte_channel(
        self,
        shape: str = "rectangular",
        center: Tuple[float, float] = (0.0, 0.0),
        radius: Optional[float] = None,
        dimensions: Optional[Tuple[float, float]] = None,
        material: str = "analyte",
        name: Optional[str] = None,
    ) -> AnalyteChannel:
        """Add an analyte sensing channel or fluidic microchannel."""
        ch_name = name or f"analyte_channel_{len(self._channels)}"
        channel = AnalyteChannel(
            name=ch_name,
            shape=shape,
            center=center,
            radius=radius,
            dimensions=dimensions,
            material_tag=material,
        )
        self._channels[ch_name] = channel
        self.invalidate_cache()
        return channel

    # --- PML Configuration ---
    def configure_pml(self, thickness: float, shape: str = "circular") -> Region2D:
        """Add a Perfectly Matched Layer enclosing the computational domain."""
        self._pml_thickness = float(thickness)
        if shape.lower() == "circular":
            r_in = self.cladding_radius
            r_out = self.cladding_radius + self._pml_thickness
            self._pml = Annulus("pml_domain", self.center, r_in, r_out, material_tag="pml")
        else:
            w = (self.cladding_radius + self._pml_thickness) * 2.0
            outer_rect = Rectangle("pml_outer", self.center, w, w)
            inner_rect = Rectangle("pml_inner", self.center, self.cladding_radius * 2, self.cladding_radius * 2)
            self._pml = CompositeRegion("pml_domain", outer_rect.shapely_geom.difference(inner_rect.shapely_geom), material_tag="pml")
        self.invalidate_cache()
        return self._pml

    # --- Domain Partitioning ---
    def partition(self) -> Dict[str, Region2D]:
        """Compute non-overlapping physical domains ready for meshing.
        
        Subdivides the cladding background so that holes, coatings, channels,
        and PML are preserved as independent regions.
        """
        if self._partitioned_regions is not None:
            return self._partitioned_regions

        all_embedded: List[Region2D] = []

        # 1. Add all holes
        for h in self._holes.values():
            all_embedded.append(h)

        # 2. Add coatings
        for c in self._coatings.values():
            all_embedded.append(c)

        # 3. Add analyte channels
        for ch in self._channels.values():
            all_embedded.append(ch)

        # 4. Add custom regions
        for cr in self._custom_regions.values():
            all_embedded.append(cr)

        # Background cladding
        bg = self._cladding_boundary or Circle("cladding", self.center, self.cladding_radius, material_tag="silica")

        # Partition background by subtracting embedded regions
        bg_partitioned, embedded_kept = partition_domains(bg, all_embedded)

        partitioned: Dict[str, Region2D] = {bg_partitioned.name: bg_partitioned}
        for reg in embedded_kept:
            partitioned[reg.name] = reg

        # Add core if separately assigned
        if self._core is not None:
            partitioned[self._core.name] = self._core

        # Add PML
        if self._pml is not None:
            partitioned[self._pml.name] = self._pml

        self._partitioned_regions = partitioned
        return partitioned

    def validate(self) -> List[str]:
        """Validate topological integrity and physical feasibility of geometry."""
        issues = []
        if self.cladding_radius <= 0:
            issues.append(f"Cladding radius must be positive, got {self.cladding_radius}")

        # Check hole overlaps
        holes_list = list(self._holes.values())
        overlaps = detect_overlapping_regions(holes_list)
        if overlaps:
            for n1, n2, a in overlaps:
                issues.append(f"Holes '{n1}' and '{n2}' overlap with area {a:.3e} m^2")

        # Check holes inside cladding
        clad_geom = self._cladding_boundary.shapely_geom if self._cladding_boundary else None
        if clad_geom:
            for h in holes_list:
                if not clad_geom.contains(h.shapely_geom):
                    # Check if hole breaches cladding boundary
                    inter = clad_geom.intersection(h.shapely_geom)
                    if hasattr(inter, "area") and inter.area < h.area * 0.99:
                        issues.append(f"Hole '{h.name}' extends outside cladding boundary.")

        return issues

    def get_summary(self) -> Dict[str, Any]:
        return {
            "cladding_radius_um": self.cladding_radius * 1e6,
            "hole_count": len(self._holes),
            "coating_count": len(self._coatings),
            "channel_count": len(self._channels),
            "has_pml": self._pml is not None,
            "pml_thickness_um": self._pml_thickness * 1e6,
        }

