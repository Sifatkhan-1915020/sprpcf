"""Polarization Study Example: X- vs Y-Polarization Analysis and Modal Birefringence.

Compares modal confinement, dispersion curves, and birefringence between
orthogonal quasi-TE (X-polarized) and quasi-TM (Y-polarized) modes.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sprpcf as spr

def run_polarization_comparison():
    print("=" * 65)
    print("POLARIZATION STUDY: X-POL VS Y-POL & MODAL BIREFRINGENCE")
    print("=" * 65)

    out_dir = Path("polarization_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    project = spr.Project("Polarization_PCF_Sensor")
    geom = spr.DShapedPCF(
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.5e-6,
        polish_depth=3.2e-6,
        metal_thickness=40e-9,
        metal_material="gold",
        cladding_radius=6.0e-6,
    )
    geom.configure_pml(thickness=1.2e-6)
    project.set_geometry(geom)

    project.assign_material("cladding", "silica")
    project.assign_material("fiber_core", "silica")
    project.assign_material("plasmonic_metal_film", "gold")
    project.assign_material("sensing_analyte_channel", "analyte")

    # Run spectral sweep requesting both polarizations
    study = spr.WavelengthSweepStudy(
        start_wavelength_m=900e-9,
        stop_wavelength_m=1300e-9,
        num_points=7,
        modes=4,
        polarization="both",
    )
    study.set_solver(backend="native", grid_points=50, tolerance=1e-6)
    results = study.run(project)

    wls_nm = []
    x_loss = []
    y_loss = []
    birefringence = []

    for r in results.results:
        wls_nm.append(r.wavelength_m * 1e9)
        x_modes = [m for m in r.modes if m.polarization_label == "x_pol"]
        y_modes = [m for m in r.modes if m.polarization_label == "y_pol"]

        # Track losses
        lx = x_modes[0].loss_db_per_cm if x_modes else 0.0
        ly = y_modes[0].loss_db_per_cm if y_modes else 0.0
        x_loss.append(lx)
        y_loss.append(ly)

        # Birefringence B = |n_eff_x - n_eff_y|
        if x_modes and y_modes:
            b_val = abs(x_modes[0].n_real - y_modes[0].n_real)
        else:
            b_val = 0.0
        birefringence.append(b_val)

    # Plot polarization-dependent loss comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    ax1.plot(wls_nm, x_loss, "b-o", linewidth=2.0, label="Quasi-TE (X-polarized)")
    ax1.plot(wls_nm, y_loss, "r-s", linewidth=2.0, label="Quasi-TM (Y-polarized)")
    ax1.set_xlabel("Wavelength (nm)", fontweight="bold")
    ax1.set_ylabel("Loss (dB/cm)", fontweight="bold")
    ax1.set_title("Polarization-Dependent Loss (PDL)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(wls_nm, birefringence, "g-^", linewidth=2.0, label="Modal Birefringence B")
    ax2.set_xlabel("Wavelength (nm)", fontweight="bold")
    ax2.set_ylabel("Birefringence B = |nx - ny|", fontweight="bold")
    ax2.set_title("Phase Birefringence vs Wavelength", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "polarization_comparison.png", dpi=300)
    print(f"[OK] Polarization study finished. Saved to: {out_dir / 'polarization_comparison.png'}")


if __name__ == "__main__":
    run_polarization_comparison()

