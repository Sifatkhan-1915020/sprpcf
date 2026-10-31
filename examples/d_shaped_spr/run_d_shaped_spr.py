"""Flagship End-to-End SPR-PCF Biosensor Simulation Workflow.

Demonstrates all 23 steps required by SPR-PCF Studio:
1. Geometry creation
2. Core definition
3. Cladding holes lattice
4. Microfluidic sensing channel
5. Plasmonic thin film
6. Material assignment
7. Complex PML configuration
8. Mesh generation
9. Mesh quality validation
10. Full-vector eigenmode study
11. Multi-mode extraction
12. Wavelength sweep
13. Mode tracking across wavelengths
14. Complex effective indices calculation
15. Confinement loss evaluation
16. SPR resonance peak identification
17. Core/SPP dispersion curve plotting
18. Full-vector field plotting
19. Sensitivity calculation across analyte refractive indices
20. FWHM and Figure of Merit (FOM)
21. Mesh convergence study
22. Result storage (HDF5 & CSV)
23. Automated scientific research report generation
"""

from pathlib import Path
import sprpcf as spr

def run_flagship_workflow():
    print("=" * 70)
    print("SPR-PCF STUDIO: FLAGSHIP END-TO-END DEMONSTRATION WORKFLOW")
    print("=" * 70)

    output_dir = Path("flagship_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Project Creation
    print("\n[Step 1] Initializing Project...")
    project = spr.Project("D_Shaped_SPR_Biosensor")

    # 2-5. Parametric D-shaped PCF Geometry with Gold Layer & Sensing Channel
    print("[Steps 2-5] Constructing D-shaped PCF Geometry...")
    geometry = spr.DShapedPCF(
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.5e-6,
        polish_depth=3.2e-6,
        metal_thickness=40e-9,  # 40 nm Au film
        metal_material="gold",
        analyte_layer_thickness=1.5e-6,
        cladding_radius=6.5e-6,
    )

    # 6-7. Material Assignment & PML
    print("[Steps 6-7] Configuring PML and assigning materials...")
    geometry.configure_pml(thickness=1.2e-6)
    project.set_geometry(geometry)

    project.assign_material("cladding", "silica")
    project.assign_material("fiber_core", "silica")
    project.assign_material("plasmonic_metal_film", "gold")
    project.assign_material("sensing_analyte_channel", "analyte")

    # 8-9. Mesh Generation & Validation
    print("[Steps 8-9] Generating and validating multi-domain mesh...")
    mesh = project.generate_mesh(backend="native", global_size=0.5e-6)
    mesh_report = spr.validate_mesh(mesh)
    print(f"       Elements: {mesh.num_elements}, Nodes: {mesh.num_nodes}")
    print(f"       Mesh Quality: Mean={mesh_report.mean_quality:.3f}, Min={mesh_report.min_quality:.4f}, Valid={mesh_report.is_valid}")

    # Plot Geometry & Mesh
    spr.plot_geometry(project, save_path=str(output_dir / "01_geometry.png"))
    spr.plot_mesh(mesh, save_path=str(output_dir / "02_mesh.png"))

    # 10-15. Full-Vector Wavelength Sweep & Mode Tracking
    print("\n[Steps 10-15] Executing Full-Vector Wavelength Sweep Study...")
    study = spr.WavelengthSweepStudy(
        start_wavelength_m=800e-9,
        stop_wavelength_m=1400e-9,
        num_points=13,
        modes=4,
        polarization="both",
    )
    study.set_solver(backend="native", grid_points=60, tolerance=1e-6)
    project.add_study(study)

    results = project.run()
    print(f"       Calculated {len(results)} spectral steps.")

    # 16. SPR Resonance Peak & Linewidth Extraction
    print("\n[Step 16] Extracting SPR Resonance & Modal Confinement Loss...")
    analyzer = spr.SPRAnalyzer(results.results)
    loss_spec = analyzer.extract_loss_spectrum(mode_idx=0)
    peaks = spr.extract_resonance_peaks(loss_spec.wavelengths_m, loss_spec.loss_db_per_cm)

    if peaks:
        dom_peak = peaks[0]
        print(f"       Resonance Wavelength: {dom_peak.wavelength_nm:.2f} nm")
        print(f"       Peak Confinement Loss: {dom_peak.peak_loss_db_cm:.2f} dB/cm")
        print(f"       Linewidth (FWHM):      {dom_peak.fwhm_nm:.2f} nm")
        print(f"       Quality Factor (Q):    {dom_peak.q_factor:.1f}")

    # 17-18. Plot Loss Spectra, Dispersion, and Field Profiles
    print("\n[Steps 17-18] Generating scientific diagnostic plots...")
    spr.plot_loss_spectrum(analyzer, save_path=str(output_dir / "03_loss_spectrum.png"))
    spr.plot_dispersion_curves(results, save_path=str(output_dir / "04_dispersion_curves.png"))

    # Plot fundamental mode field distribution at resonance or middle step
    mid_res = results.results[len(results) // 2]
    if mid_res.modes:
        spr.plot_mode_field(mid_res.modes[0], component="E_mag", save_path=str(output_dir / "05_mode_field_Emag.png"))
        spr.plot_mode_field(mid_res.modes[0], component="Sz", save_path=str(output_dir / "06_mode_field_Poynting.png"))

    # 19-20. Sensitivity & FOM across Analyte Refractive Indices
    print("\n[Steps 19-20] Evaluating Sensitivity across Analyte RI (1.330 vs 1.340)...")
    analyte_results = {
        1.330: results.results,
    }
    # Second small run at n_analyte = 1.340
    project.assign_material("sensing_analyte_channel", spr.AnalyteMaterial("analyte_134", 1.340))
    study_na2 = spr.WavelengthSweepStudy(
        start_wavelength_m=800e-9,
        stop_wavelength_m=1400e-9,
        num_points=13,
        modes=2,
    )
    study_na2.set_solver(backend="native", grid_points=60, tolerance=1e-6)
    res_na2 = study_na2.run(project)
    analyte_results[1.340] = res_na2.results

    sens_summary = analyzer.calculate_sensitivity_from_analyte_sweep(analyte_results)
    sens_val = sens_summary["sensitivity"]["linear_wavelength_sensitivity_nm_per_riu"]
    fom_val = sens_summary["performance"]["figure_of_merit_riu_inv"]
    res_val = sens_summary["performance"]["sensor_resolution_riu"]

    print(f"       Wavelength Sensitivity: {sens_val:.1f} nm/RIU")
    print(f"       Figure of Merit (FOM):  {fom_val:.1f} RIU^-1")
    print(f"       Sensor Resolution:      {res_val:.2e} RIU")

    spr.plot_sensitivity_fit(sens_summary, save_path=str(output_dir / "07_sensitivity_fit.png"))

    # 21. Mesh Convergence Study
    print("\n[Step 21] Performing Mesh Convergence Study...")
    conv_study = spr.MeshConvergenceStudy(grid_resolutions=[40, 50, 60], wavelength_m=1.0e-6)
    conv_res = conv_study.run(project)
    print(f"       Convergence status: {conv_res.metadata.get('is_converged')}")

    # 22-23. Result Storage & Automated Research Report
    print("\n[Steps 22-23] Exporting CSV, HDF5, and Automated Scientific Report...")
    spr.results.storage.export_study_to_csv(results, output_dir / "simulation_results.csv")
    spr.results.storage.export_study_to_hdf5(results, output_dir / "simulation_results.h5")

    proj_summary = project.get_summary()
    proj_summary.update(sens_summary)
    spr.generate_html_report(proj_summary, output_dir / "Research_Report.html")

    print(f"\n[DONE] All 23 steps completed successfully!")
    print(f"       Results and publication plots saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    run_flagship_workflow()

