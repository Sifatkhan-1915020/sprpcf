"""Command implementations for the SPR-PCF CLI."""

import argparse
from pathlib import Path
import json
import yaml
from ..core.configuration import load_config_file
from ..core.project import Project
from ..geometry.cross_section import PCFGeometry
from ..geometry.pcf_templates import DShapedPCF, ConventionalPCF
from ..materials.database import materials
from ..meshing import generate_mesh
from ..studies.eigenmode import EigenmodeStudy, WavelengthSweepStudy
from ..spr.analyzer import SPRAnalyzer
from ..visualization.geometry import plot_geometry
from ..visualization.mesh import plot_mesh
from ..visualization.spectra import plot_loss_spectrum
from ..results.storage import export_study_to_csv


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a template design.yaml configuration file."""
    output_path = Path(args.output or "design.yaml")
    template = {
        "project": {
            "name": "SPR_PCF_Research",
            "description": "Parametric SPR-PCF biosensor study",
        },
        "geometry": {
            "architecture": "d_shaped",
            "cladding_radius": "10.0 um",
            "lattice": {
                "type": "hexagonal",
                "pitch": "2.0 um",
                "rings": 3,
                "hole_radius": "0.5 um",
            },
            "d_cut_depth": "3.5 um",
        },
        "coatings": [
            {
                "surface": "sensing_surface",
                "material": "gold",
                "thickness": "40 nm",
            }
        ],
        "domain_assignments": {
            "cladding": "silica",
            "fiber_core": "silica",
            "plasmonic_metal_film": "gold",
            "sensing_analyte_channel": "analyte",
        },
        "study": {
            "type": "wavelength_sweep",
            "wavelength_start": "600 nm",
            "wavelength_stop": "1600 nm",
            "wavelength_points": 41,
            "target_modes": 4,
            "polarization": "both",
        },
    }
    output_path.write_text(yaml.dump(template, sort_keys=False), encoding="utf-8")
    print(f"[OK] Initialized new SPR-PCF design template at: {output_path}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a design configuration file."""
    cfg = load_config_file(args.config)
    print(f"[OK] Configuration '{args.config}' loaded and validated successfully.")
    print(f"     Project name: {cfg.project.name}")
    print(f"     Architecture: {cfg.geometry.architecture}")
    print(f"     Study type:   {cfg.study.type}")


def cmd_geometry(args: argparse.Namespace) -> None:
    """Generate and plot PCF geometry from design config."""
    cfg = load_config_file(args.config)
    proj = Project(name=cfg.project.name)
    if cfg.geometry.architecture == "d_shaped":
        geom = DShapedPCF(name=cfg.project.name)
    else:
        geom = ConventionalPCF(name=cfg.project.name)
    proj.set_geometry(geom)

    out_file = args.output or f"{cfg.project.name}_geometry.png"
    plot_geometry(proj, save_path=out_file)
    print(f"[OK] Geometry plotted and saved to: {out_file}")


def cmd_mesh(args: argparse.Namespace) -> None:
    """Generate and plot mesh from design config."""
    cfg = load_config_file(args.config)
    proj = Project(name=cfg.project.name)
    if cfg.geometry.architecture == "d_shaped":
        geom = DShapedPCF(name=cfg.project.name)
    else:
        geom = ConventionalPCF(name=cfg.project.name)
    proj.set_geometry(geom)

    mesh = proj.generate_mesh(backend=cfg.mesh.backend)
    out_file = args.output or f"{cfg.project.name}_mesh.png"
    plot_mesh(mesh, save_path=out_file)
    print(f"[OK] Mesh generated ({mesh.num_elements} elements) and saved to: {out_file}")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute simulation study defined in config."""
    cfg = load_config_file(args.config)
    proj = Project(name=cfg.project.name)
    if cfg.geometry.architecture == "d_shaped":
        geom = DShapedPCF(name=cfg.project.name)
    else:
        geom = ConventionalPCF(name=cfg.project.name)
    proj.set_geometry(geom)

    study = WavelengthSweepStudy(
        start_wavelength_m=float(cfg.study.wavelength_start) if isinstance(cfg.study.wavelength_start, (int, float)) else 800e-9,
        stop_wavelength_m=float(cfg.study.wavelength_stop) if isinstance(cfg.study.wavelength_stop, (int, float)) else 1600e-9,
        num_points=cfg.study.wavelength_points,
        modes=cfg.study.target_modes,
        polarization=cfg.study.polarization,
    )
    proj.add_study(study)

    print(f"[*] Running study '{study.name}' across {len(study.wavelengths_m)} wavelengths...")
    results = proj.run()

    out_dir = Path(args.output_dir or "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "modes.csv"
    export_study_to_csv(results, csv_file)

    analyzer = SPRAnalyzer(results.results)
    res_peak = analyzer.find_resonance(0)
    spec_plot = out_dir / "loss_spectrum.png"
    plot_loss_spectrum(analyzer, save_path=str(spec_plot))

    print(f"[OK] Simulation completed. Results exported to: {out_dir}")
    if res_peak:
        print(f"     Dominant resonance: {res_peak.wavelength_nm:.2f} nm (Loss: {res_peak.peak_loss_db_cm:.2f} dB/cm, FWHM: {res_peak.fwhm_nm:.2f} nm)")

