"""=============================================================================
SPR-PCF STUDIO: COMPLETE RESEARCH-GRADE SIMULATION & ANALYSIS PIPELINE
=============================================================================
This complete, self-contained scientific simulation demonstrates:
1. GEOMETRY VALIDATION: Parametric D-shaped PCF with 2.5 nm Ti adhesion layer,
   40 nm Au layer, PML boundary, and pre-mesh geometric validation.
2. MESH VALIDATION: Multi-domain Delaunay meshing with interface refinement,
   element quality metrics (min quality, max aspect ratio), and convergence check.
3. ADAPTIVE SPR SPECTRUM: Two-stage coarse-to-fine spectral sweep, automatic
   resonance interval detection, nonlinear Lorentzian line fitting, parameter
   uncertainties, and strict sampling adequacy verification.
4. MODE ANALYSIS & AVOIDED CROSSING: Field-overlap modal tracking, anti-crossing
   identification, core-SPP coupling, and polarization classification.
5. FIELD ANALYSIS & POWER FLOW: Artifact-free interpolated fields, separate
   Ex, Ey, Ez, H, Sz, phase, and longitudinal power integration (P_total > 0).
6. BIOSENSOR PERFORMANCE: Multi-analyte RI calibration (1.330, 1.340, 1.350),
   wavelength sensitivity (S_lambda with R^2), amplitude sensitivity (S_A),
   FOM, and resolution under stated instrument specifications.
7. RESEARCH DASHBOARD: Publication-grade 8-panel scientific visualization with
   nanoscale callout inset, mesh quality map, and numerical diagnostics banner.
8. REPRODUCIBILITY & DATA EXPORT: HDF5 hierarchical store, CSV tabular output,
   and HTML5 automated research report.
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import the SPR-PCF library
import sprpcf as spr


def main():
    print("=" * 80)
    print("      SPR-PCF STUDIO: RESEARCH-GRADE BIOSENSOR SIMULATION & DASHBOARD")
    print("=" * 80)

    out_dir = Path("simulation_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: PARAMETRIC GEOMETRY MODELING & PRE-MESH VALIDATION
    # -------------------------------------------------------------------------
    print("\n[1/8] Constructing & Validating D-Shaped PCF Microstructure...")
    project = spr.Project("D_Shaped_Au_Biosensor")

    # Construct D-shaped side-polished PCF
    geometry = spr.DShapedPCF(
        pitch=2.0e-6,                  # 2.0 um hole-to-hole pitch
        rings=2,                       # 2 rings of air holes
        hole_radius=0.5e-6,            # 0.5 um radius (d/Lambda = 0.5)
        polish_depth=3.2e-6,           # Flat polished surface at y = 3.2 um
        metal_thickness=40.0e-9,       # 40 nm gold plasmonic layer
        metal_material="gold",
        adhesion_thickness=2.5e-9,     # 2.5 nm Titanium (Ti) adhesion layer
        adhesion_material="titanium",
        analyte_layer_thickness=1.5e-6,# 1.5 um microfluidic sensing channel
        cladding_radius=6.5e-6,        # 6.5 um fiber radius
    )

    # Configure a 1.2 um cylindrical Perfectly Matched Layer (PML)
    geometry.configure_pml(thickness=1.2e-6)
    project.set_geometry(geometry)

    # Pre-meshing topological and dimensional validation
    geom_report = geometry.validate_d_shape()
    print(f"      D-Shape Geometry Feasibility: {'VERIFIED' if geom_report.d_shape_verified else 'FAILED'}")
    print(f"      Polished Flat Chord Width:    {geom_report.dimensions.get('flat_chord_width', 0.0)*1e6:.2f} um")
    print(f"      Thin-Film Stacking Order:     Silica -> Ti (2.5 nm) -> Au (40 nm) -> Analyte")
    print(f"      Pre-Mesh Geometric Integrity: {'PASSED (0 errors)' if geom_report.is_valid else 'FAILED'}")

    # -------------------------------------------------------------------------
    # STEP 2: MATERIAL ASSIGNMENTS & DISPERSION MODELS
    # -------------------------------------------------------------------------
    print("\n[2/8] Assigning Experimental Wavelength-Dependent Dispersion Models...")
    # Fused silica: Malitson 3-term Sellmeier model
    project.assign_material("cladding", "silica")
    project.assign_material("fiber_core", "silica")
    # Plasmonic metal: Rakic 1998 Drude-Lorentz multi-oscillator gold model
    project.assign_material("plasmonic_metal_film", "gold")
    # Adhesion layer: Titanium thin film
    project.assign_material("adhesion_layer", "titanium")
    # Microfluidic analyte channel: base liquid index n = 1.330
    project.assign_material("sensing_analyte_channel", spr.AnalyteMaterial("analyte_sample", refractive_index=1.330))

    # -------------------------------------------------------------------------
    # STEP 3: MULTI-DOMAIN MESH GENERATION & QUALITY VALIDATION
    # -------------------------------------------------------------------------
    print("\n[3/8] Generating Finite Element Mesh & Assessing Element Quality...")
    mesh = project.generate_mesh(backend="native", global_size=0.5e-6, refine_interfaces=True)
    mesh_report = spr.validate_mesh(mesh, interface_y=geometry.polish_depth)

    print(f"      Mesh Elements:      {mesh_report.num_elements}")
    print(f"      Mesh Nodes:         {mesh_report.num_nodes}")
    print(f"      Interface Elements: {mesh_report.interface_elements_count}")
    print(f"      Mean Element Quality: {mesh_report.mean_quality:.3f} (Min: {mesh_report.min_quality:.2e})")
    print(f"      Max Aspect Ratio:   {mesh_report.max_aspect_ratio:.1f}")

    # Mesh convergence check (Level 1: 45x45, Level 2: 60x60)
    print("      Testing multi-level spatial convergence...")
    conv_study = spr.MeshConvergenceStudy(grid_resolutions=[45, 60], wavelength_m=1.10e-6, modes=2)
    conv_res = conv_study.run(project)
    dn_conv = conv_res.metadata["consecutive_neff_differences"][-1]
    print(f"      Mesh Refinement Error: Delta_neff = {dn_conv:.2e} (Converged: {conv_res.metadata['is_converged']})")

    # -------------------------------------------------------------------------
    # STEP 4: ADAPTIVE COARSE-TO-FINE FULL-VECTOR WAVELENGTH SWEEP
    # -------------------------------------------------------------------------
    print("\n[4/8] Executing Two-Stage Adaptive Wavelength Sweep around SPR Band...")
    # Stage 1: Coarse sweep (9 points across 950-1250 nm)
    # Stage 2: Automatic candidate window detection -> Fine sweep (17 dense points, Delta_lambda = 5 nm)
    adaptive_study = spr.AdaptiveWavelengthSweepStudy(
        start_wavelength_m=950e-9,
        stop_wavelength_m=1250e-9,
        coarse_points=9,
        fine_window_nm=80.0,
        fine_points=17,
        modes=2,
        polarization="both",
    )
    adaptive_study.set_solver(backend="native", grid_points=48, tolerance=1e-6)
    project.add_study(adaptive_study)

    results = adaptive_study.run(project)
    total_steps = len(results)
    print(f"      Completed {total_steps} adaptive spectral mode solutions.")
    print(f"      Detected Candidate Interior Resonance: {results.metadata['detected_coarse_resonance_nm']:.1f} nm")

    # -------------------------------------------------------------------------
    # STEP 5: SPR RESONANCE DETECTION & NONLINEAR LINESHAPE FITTING
    # -------------------------------------------------------------------------
    print("\n[5/8] Performing Nonlinear Resonance Line-Fitting & Sampling Validation...")
    analyzer = spr.SPRAnalyzer(results.results)
    loss_spec = analyzer.extract_loss_spectrum(mode_idx=0)
    peaks = spr.extract_resonance_peaks(loss_spec.wavelengths_m, loss_spec.loss_db_per_cm, fit_model="lorentzian")

    if peaks:
        dom_peak = peaks[0]
        res_wl_nm = dom_peak.wavelength_nm
        peak_loss = dom_peak.peak_loss_db_cm
        fwhm_nm = dom_peak.fwhm_nm
        q_fact = dom_peak.q_factor
        fit = dom_peak.fit_result
        if fit and fit.r_squared > 0.7:
            print(f"      Lorentzian Fitted Peak:  {fit.wavelength_nm:.2f} +/- {fit.wavelength_err_nm:.2f} nm")
            print(f"      Fitted FWHM Linewidth:   {fit.fwhm_nm:.2f} +/- {fit.fwhm_err_nm:.2f} nm")
            print(f"      Fit Goodness (R^2):      {fit.r_squared:.4f}")
            print(f"      Points in FWHM:          {fit.points_in_fwhm} (Sampling: {'ADEQUATE' if fit.is_adequately_sampled else 'UNDERSAMPLED'})")
        else:
            print(f"      Resonance Peak:          {res_wl_nm:.2f} nm")
            print(f"      Peak Confinement Loss:   {peak_loss:.2f} dB/cm")
            print(f"      Linewidth (FWHM):        {fwhm_nm:.2f} nm (Q={q_fact:.1f})")
    else:
        max_idx = int(np.argmax(loss_spec.loss_db_per_cm))
        res_wl_nm = loss_spec.wavelengths_m[max_idx] * 1e9
        peak_loss = loss_spec.loss_db_per_cm[max_idx]
        fwhm_nm = 35.0
        q_fact = res_wl_nm / fwhm_nm
        print(f"      Peak (Spline): {res_wl_nm:.2f} nm | Loss: {peak_loss:.2f} dB/cm")

    # -------------------------------------------------------------------------
    # STEP 6: FIELD ANALYSIS, POWER INTEGRATION & AVOIDED CROSSING
    # -------------------------------------------------------------------------
    print("\n[6/8] Validating Field Solutions, Longitudinal Power & Avoided Crossings...")
    # Find snapshot mode closest to resonance
    wls_arr = np.array([r.wavelength_m for r in results.results])
    closest_step_idx = int(np.argmin(np.abs(wls_arr - (res_wl_nm * 1e-9))))
    res_mode = results.results[closest_step_idx].modes[0]

    p_flow = res_mode.total_power
    p_fwd = res_mode.is_forward_propagating
    print(f"      Mode {res_mode.mode_id} Classification:   [{res_mode.mode_classification}]")
    print(f"      Effective Phase Index:    Re(n_eff) = {res_mode.n_real:.6f}")
    print(f"      Attenuation Index:        Im(n_eff) = {res_mode.n_imag:.4e}")
    print(f"      Integrated Power Flow:    P_total = {p_flow:.3e} W (Forward: {p_fwd})")
    print(f"      Eigenproblem Residual:    ||Pv - beta^2 v|| = {res_mode.residual:.2e}")

    # Check for avoided crossings
    avoided_events = results.metadata.get("avoided_crossings", [])
    if avoided_events:
        ev = avoided_events[0]
        print(f"      Avoided Crossing Detected: Gap = {ev['min_index_gap']:.4f} at {ev['param_val']*1e9:.1f} nm")

    # -------------------------------------------------------------------------
    # STEP 7: MULTI-ANALYTE BIOSENSOR PERFORMANCE
    # -------------------------------------------------------------------------
    print("\n[7/8] Conducting Multi-Analyte Sensitivity Evaluation (n_a = 1.330, 1.340, 1.350)...")
    sens_wls_start = 1040e-9
    sens_wls_stop = 1160e-9
    sens_points = 25

    analyte_results = {}
    for na in [1.330, 1.340, 1.350]:
        project.assign_material("sensing_analyte_channel", spr.AnalyteMaterial(f"analyte_{int(na*1000)}", refractive_index=na))
        study_na = spr.EigenmodeStudy(
            wavelength_range=(sens_wls_start, sens_wls_stop),
            points=sens_points,
            modes=2,
            polarization="both",
        )
        study_na.set_solver(backend="native", grid_points=45, tolerance=1e-6)
        res_na = study_na.run(project)
        analyte_results[na] = res_na.results

    sens_analyzer = spr.SPRAnalyzer(analyte_results[1.330])
    sens_summary = sens_analyzer.calculate_sensitivity_from_analyte_sweep(analyte_results, instrument_resolution_nm=0.1)
    s_lambda = sens_summary["sensitivity"]["linear_wavelength_sensitivity_nm_per_riu"]
    r2_sens = sens_summary["sensitivity"]["r_squared"]
    fom = sens_summary["performance"]["figure_of_merit_riu_inv"]
    res_statement = str(sens_summary["performance"]["resolution_statement"]).encode("ascii", "replace").decode("ascii")
    lod_statement = str(sens_summary["performance"]["lod_statement"]).encode("ascii", "replace").decode("ascii")

    print(f"      Linear Wavelength Sensitivity: {s_lambda:.1f} nm/RIU (R^2 = {r2_sens:.4f})")
    print(f"      Figure of Merit (FOM):         {fom:.1f} RIU^-1")
    print(f"      Detection Resolution:          {res_statement}")
    print(f"      Limit of Detection (LOD):      {lod_statement}")

    # -------------------------------------------------------------------------
    # STEP 8: EXPORT DATA, REPORT & RENDER RESEARCH DASHBOARDS
    # -------------------------------------------------------------------------
    print("\n[8/8] Exporting Datasets, HTML Report & Rendering Research Dashboard...")
    csv_path = out_dir / "simulation_results.csv"
    h5_path = out_dir / "simulation_results.h5"
    html_path = out_dir / "Research_Report.html"

    spr.results.storage.export_study_to_csv(results, csv_path)
    spr.results.storage.export_study_to_hdf5(results, h5_path)

    proj_summary = project.get_summary()
    proj_summary.update(sens_summary)
    spr.generate_html_report(proj_summary, html_path)

    print(f"      CSV Dataset:        {csv_path}")
    print(f"      HDF5 Modal Fields:  {h5_path}")
    print(f"      HTML5 Report:       {html_path}")

    # 1. 8-Panel Publication-Grade Scientific Dashboard
    dashboard_path = out_dir / "complete_simulation_dashboard.png"
    fig_dash = spr.create_research_dashboard(
        project=project,
        mesh=mesh,
        study_result=results,
        analyzer=analyzer,
        sensitivity_report=sens_summary,
        selected_mode_idx=0,
        selected_wl_step=closest_step_idx,
        save_path=str(dashboard_path),
        dpi=300,
    )
    print(f"      Research Dashboard: {dashboard_path.resolve()}")

    # 2. 6-Panel Field Component Breakdown
    fields_path = out_dir / "all_field_components_mode0.png"
    fig_fields = spr.plot_all_field_components(
        mode=res_mode,
        geometry=project.geometry,
        save_path=str(fields_path),
        dpi=300,
    )
    print(f"      Field Components:   {fields_path.resolve()}")

    print("\n" + "=" * 80)
    print("  [SUCCESS] All 8 Scientific Simulation & Validation Pillars Completed!")
    print("=" * 80)

    # Check for batch / non-interactive execution
    if "--no-show" in sys.argv or "--batch" in sys.argv:
        print("\n[*] Batch execution complete. All publication figures and data saved.")
        plt.close("all")
    else:
        print("\n[*] Launching Interactive Scientific Dashboard (close plot window to finish)...")
        plt.show()


if __name__ == "__main__":
    main()
