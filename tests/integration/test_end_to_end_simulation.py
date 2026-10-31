"""End-to-end integration tests for SPR-PCF Studio simulation workflow."""

import pytest
from pathlib import Path
import tempfile
import numpy as np

import sprpcf as spr
from sprpcf.geometry.pcf_templates import DShapedPCF
from sprpcf.studies.eigenmode import EigenmodeStudy, WavelengthSweepStudy
from sprpcf.studies.convergence import MeshConvergenceStudy
from sprpcf.spr.analyzer import SPRAnalyzer
from sprpcf.results.storage import export_study_to_csv, export_study_to_hdf5
from sprpcf.visualization.geometry import plot_geometry
from sprpcf.visualization.mesh import plot_mesh
from sprpcf.visualization.spectra import plot_loss_spectrum
from sprpcf.visualization.reports import generate_html_report


def test_complete_end_to_end_workflow():
    # 1. Create Project
    project = spr.Project("Integration_SPR_PCF_Sensor")

    # 2. Build D-shaped PCF Geometry with polished flat, gold layer, and sensing channel
    geom = DShapedPCF(
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.5e-6,
        polish_depth=3.2e-6,
        metal_thickness=40e-9,
        metal_material="gold",
        cladding_radius=6.0e-6,
    )
    # Configure PML
    geom.configure_pml(thickness=1.2e-6)
    project.set_geometry(geom)

    # 3. Assign Materials
    project.assign_material("cladding", "silica")
    project.assign_material("fiber_core", "silica")
    project.assign_material("plasmonic_metal_film", "gold")
    project.assign_material("sensing_analyte_channel", "analyte")

    # Validate geometry
    issues = project.validate()
    assert len(issues) == 0

    # 4. Generate & Validate Mesh
    mesh = project.generate_mesh(backend="native", global_size=0.5e-6)
    assert mesh.num_elements > 0

    # 5. Configure Full-Vector Wavelength Sweep Study
    study = WavelengthSweepStudy(
        start_wavelength_m=800e-9,
        stop_wavelength_m=1200e-9,
        num_points=5,
        modes=2,
        polarization="both",
    )
    study.set_solver(backend="native", grid_points=50, tolerance=1e-6)
    project.add_study(study)

    # 6. Run Simulation
    results = project.run()
    assert len(results) == 5

    # 7. Check Mode Solutions
    for r in results.results:
        assert len(r.modes) >= 1
        fund = r.fundamental_mode
        assert fund is not None
        assert 1.30 < fund.n_real < 1.50
        assert fund.loss_db_per_cm >= 0.0

    # 8. Analyze with SPRAnalyzer
    analyzer = SPRAnalyzer(results.results)
    loss_spec = analyzer.extract_loss_spectrum(mode_idx=0)
    assert len(loss_spec.loss_db_per_cm) == 5

    # 9. Mesh Convergence Study
    conv_study = MeshConvergenceStudy(grid_resolutions=[40, 50], wavelength_m=1.0e-6)
    conv_res = conv_study.run(project)
    assert len(conv_res.results) == 2

    # 10. File Exports & Reports
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "results.csv"
        export_study_to_csv(results, csv_file)
        assert csv_file.exists()

        h5_file = tmp_path / "results.h5"
        export_study_to_hdf5(results, h5_file)
        assert h5_file.exists()

        report_file = tmp_path / "report.html"
        generate_html_report(project.get_summary(), report_file)
        assert report_file.exists()

