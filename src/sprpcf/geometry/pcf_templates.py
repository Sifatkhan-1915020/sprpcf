"""Parametric PCF architecture templates."""

from typing import Optional, Tuple, Dict, Any
from .cross_section import PCFGeometry
from .validation import validate_geometry, GeometryValidationReport
from ..core.registry import pcf_templates_registry


class ConventionalPCF(PCFGeometry):
    """Standard solid-core hexagonal photonic crystal fiber."""

    def __init__(
        self,
        pitch: float = 2.0e-6,
        rings: int = 3,
        hole_radius: float = 0.5e-6,
        cladding_radius: float = 10.0e-6,
        name: str = "conventional_pcf",
    ):
        super().__init__(cladding_radius=cladding_radius, name=name)
        self.pitch = pitch
        self.rings = rings
        self.hole_radius = hole_radius

        # Create hexagonal cladding lattice around solid center core
        self.create_lattice(
            lattice_type="hexagonal",
            pitch=self.pitch,
            rings=self.rings,
            hole_radius=self.hole_radius,
            material="air",
        )
        self.create_core(shape="circular", radius=self.pitch * 0.7, material="silica")


class DShapedPCF(PCFGeometry):
    """Side-polished D-shaped SPR sensor with gold/metal thin film on the polished flat."""

    def __init__(
        self,
        pitch: float = 2.0e-6,
        rings: int = 3,
        hole_radius: float = 0.5e-6,
        polish_depth: float = 3.5e-6,  # Distance from core center (0,0) to flat cut
        metal_thickness: float = 40.0e-9,
        metal_material: str = "gold",
        adhesion_thickness: float = 0.0,  # Optional thin adhesion layer (e.g. 2-3 nm titanium)
        adhesion_material: str = "titanium",
        analyte_layer_thickness: float = 2.0e-6,
        cladding_radius: float = 10.0e-6,
        name: str = "d_shaped_spr_pcf",
    ):
        super().__init__(cladding_radius=cladding_radius, name=name)
        self.pitch = pitch
        self.rings = rings
        self.hole_radius = hole_radius
        self.polish_depth = polish_depth
        self.metal_thickness = metal_thickness
        self.metal_material = metal_material
        self.adhesion_thickness = adhesion_thickness
        self.adhesion_material = adhesion_material
        self.analyte_layer_thickness = analyte_layer_thickness

        # 1. Cut cladding with polished D-flat at y = polish_depth
        start_pt, end_pt = self.set_d_shape(polish_depth=self.polish_depth, material_tag="silica")

        # 2. Add hexagonal hole lattice (excluding holes sliced by polish cut)
        raw_holes = self.create_lattice(
            lattice_type="hexagonal",
            pitch=self.pitch,
            rings=self.rings,
            hole_radius=self.hole_radius,
            material="air",
        )

        # Remove holes whose center is too close to or above the polish cut
        holes_to_remove = []
        for h in raw_holes:
            if h.center[1] + h.radius >= self.polish_depth - 0.2e-6:
                holes_to_remove.append(h.name)
        for h_name in holes_to_remove:
            self.remove_region(h_name)

        # 3. Create optical core
        self.create_core(shape="circular", radius=self.pitch * 0.7, material="silica")

        # 4. Add optional adhesion layer directly on silica flat (e.g. 2-3 nm Ti)
        current_y = self.polish_depth
        if self.adhesion_thickness > 0.0:
            self.add_coating(
                thickness=self.adhesion_thickness,
                material=self.adhesion_material,
                coating_type="planar",
                start_point=start_pt,
                end_point=end_pt,
                name="adhesion_layer",
            )
            current_y += self.adhesion_thickness

        # 5. Add plasmonic coating on flat surface (or above adhesion layer)
        metal_start = (start_pt[0], current_y)
        metal_end = (end_pt[0], current_y)
        self.add_coating(
            thickness=self.metal_thickness,
            material=self.metal_material,
            coating_type="planar",
            start_point=metal_start,
            end_point=metal_end,
            name="plasmonic_metal_film",
        )
        current_y += self.metal_thickness

        # 6. Add sensing analyte channel directly above the metal film
        import math
        half_w = math.sqrt(self.cladding_radius**2 - self.polish_depth**2)
        analyte_cut_height = analyte_layer_thickness
        ch_center = (0.0, current_y + analyte_cut_height / 2.0)
        self.add_analyte_channel(
            shape="rectangular",
            center=ch_center,
            dimensions=(half_w * 2.0, analyte_cut_height),
            material="analyte",
            name="sensing_analyte_channel",
        )

    def validate_d_shape(self) -> GeometryValidationReport:
        """Run strict validation verifying D-shape flat, boundaries, and clearances."""
        return validate_geometry(self)


class DualChannelPCF(PCFGeometry):
    """Photonic crystal fiber with two large microfluidic sensing channels coated with metal."""

    def __init__(
        self,
        pitch: float = 2.0e-6,
        rings: int = 3,
        hole_radius: float = 0.5e-6,
        channel_radius: float = 1.2e-6,
        metal_thickness: float = 40.0e-9,
        metal_material: str = "gold",
        cladding_radius: float = 10.0e-6,
        name: str = "dual_channel_spr_pcf",
    ):
        super().__init__(cladding_radius=cladding_radius, name=name)
        self.pitch = pitch
        self.rings = rings

        # Create base lattice
        self.create_lattice(
            lattice_type="hexagonal",
            pitch=pitch,
            rings=rings,
            hole_radius=hole_radius,
            material="air",
        )
        self.create_core(shape="circular", radius=pitch * 0.7, material="silica")

        # Add two symmetric large analyte channels on x-axis
        ch1_center = (-2 * pitch, 0.0)
        ch2_center = (2 * pitch, 0.0)

        # Remove any conflicting small holes
        to_del = []
        for h in self._holes.values():
            import math
            d1 = math.hypot(h.center[0] - ch1_center[0], h.center[1] - ch1_center[1])
            d2 = math.hypot(h.center[0] - ch2_center[0], h.center[1] - ch2_center[1])
            if d1 < channel_radius + 0.3e-6 or d2 < channel_radius + 0.3e-6:
                to_del.append(h.name)
        for d in to_del:
            self.remove_region(d)

        # Internal metal coating in channel 1
        self.add_coating(
            thickness=metal_thickness,
            material=metal_material,
            coating_type="internal_hole",
            center=ch1_center,
            outer_radius=channel_radius,
            name="metal_coating_ch1",
        )
        # Inner analyte channel 1
        self.add_analyte_channel(
            shape="circular",
            center=ch1_center,
            radius=channel_radius - metal_thickness,
            material="analyte_1",
            name="channel_1",
        )

        # Internal metal coating in channel 2
        self.add_coating(
            thickness=metal_thickness,
            material=metal_material,
            coating_type="internal_hole",
            center=ch2_center,
            outer_radius=channel_radius,
            name="metal_coating_ch2",
        )
        # Inner analyte channel 2
        self.add_analyte_channel(
            shape="circular",
            center=ch2_center,
            radius=channel_radius - metal_thickness,
            material="analyte_2",
            name="channel_2",
        )


class HollowCorePCF(PCFGeometry):
    """Hollow-core photonic crystal fiber with central air/fluid defect."""

    def __init__(
        self,
        pitch: float = 2.0e-6,
        rings: int = 3,
        hole_radius: float = 0.55e-6,
        core_radius: float = 1.8e-6,
        cladding_radius: float = 10.0e-6,
        name: str = "hollow_core_pcf",
    ):
        super().__init__(cladding_radius=cladding_radius, name=name)
        self.create_lattice(
            lattice_type="hexagonal",
            pitch=pitch,
            rings=rings,
            hole_radius=hole_radius,
            material="air",
        )
        # Central hole defect acting as hollow core
        self.create_core(shape="circular", radius=core_radius, material="air")


# Register templates
pcf_templates_registry.register("conventional", ConventionalPCF)
pcf_templates_registry.register("d_shaped", DShapedPCF)
pcf_templates_registry.register("dual_channel", DualChannelPCF)
pcf_templates_registry.register("hollow_core", HollowCorePCF)

