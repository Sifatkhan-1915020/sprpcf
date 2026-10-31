"""Analyte Refractive Index Sweep Example.

Simulates multiple analyte refractive indices (e.g. 1.330 to 1.360), tracks resonance shift,
and computes linear sensitivity (nm/RIU), amplitude sensitivity (RIU^-1), FOM, and resolution.
"""

from pathlib import Path
import numpy as np
import sprpcf as spr

def run_analyte_sweep():
    print("=" * 65)
    print("ANALYTE SWEEP & SENSITIVITY CHARACTERIZATION")
    print("=" * 65)

    out_dir = Path("analyte_sweep_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    project = spr.Project("Analyte_Sensitivity_PCF")
    geom = spr.DShapedPCF(
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.5e-6,
        polish_depth=3.2e-6,
        metal_thickness=40e-9,
        cladding_radius=6.0e-6,
    )
    geom.configure_pml(thickness=1.2e-6)
    project.set_geometry(geom)

    project.assign_material("cladding", "silica")
    project.assign_material("fiber_core", "silica")
    project.assign_material("plasmonic_metal_film", "gold")
    project.assign_material("sensing_analyte_channel", "analyte")

    # Run Analyte Sweep Study
    analyte_ris = [1.330, 1.340, 1.350]
    study = spr.AnalyteSweepStudy(
        analyte_indices=analyte_ris,
        wavelength_range=(850e-9, 1350e-9),
        wavelength_points=9,
        modes=2,
    )
    study.set_solver(backend="native", grid_points=50, tolerance=1e-6)

    print(f"[*] Running analyte sweep over RI: {analyte_ris}...")
    results = study.run(project)

    sens_summary = results.metadata.get("sensitivity_summary", {})
    sens_info = sens_summary.get("sensitivity", {})
    perf_info = sens_summary.get("performance", {})

    s_lambda = sens_info.get("linear_wavelength_sensitivity_nm_per_riu", 0.0)
    r2 = sens_info.get("r_squared", 1.0)
    fom = perf_info.get("figure_of_merit_riu_inv", 0.0)
    res = perf_info.get("sensor_resolution_riu", 0.0)

    print("\n--- BIOSENSING SENSITIVITY METRICS ---")
    print(f"Wavelength Sensitivity: {s_lambda:.1f} nm/RIU (R^2 = {r2:.4f})")
    print(f"Figure of Merit (FOM):  {fom:.1f} RIU^-1")
    print(f"Sensor Resolution:      {res:.2e} RIU")

    # Plot Sensitivity Linearity Fit
    plot_file = out_dir / "sensitivity_fit.png"
    spr.plot_sensitivity_fit(sens_summary, save_path=str(plot_file))
    print(f"[OK] Sensitivity plot saved to: {plot_file}")


if __name__ == "__main__":
    run_analyte_sweep()

