"""Conventional Hexagonal PCF and Hollow-Core Fiber Simulation Example."""

from pathlib import Path
import sprpcf as spr

def run_hexagonal_pcf_example():
    print("=" * 65)
    print("HEXAGONAL SOLID-CORE & HOLLOW-CORE PCF ANALYSIS")
    print("=" * 65)

    out_dir = Path("hexagonal_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Solid-Core Hexagonal PCF
    print("\n1. Simulating Conventional Solid-Core Hexagonal PCF...")
    project = spr.Project("Solid_Core_Hex_PCF")
    geom = spr.ConventionalPCF(
        pitch=2.0e-6,
        rings=3,
        hole_radius=0.55e-6,
        cladding_radius=8.0e-6,
    )
    geom.configure_pml(thickness=1.5e-6)
    project.set_geometry(geom)

    project.assign_material("cladding", "silica")
    project.assign_material("fiber_core", "silica")

    study = spr.EigenmodeStudy(
        wavelength=1.55e-6,
        modes=4,
        polarization="both",
    )
    study.set_solver(backend="native", grid_points=60, tolerance=1e-6)
    results = study.run(project)

    fund = results.results[0].fundamental_mode
    print(f"   Fundamental Mode at 1550 nm: n_eff = {fund.n_real:.6f} + {fund.n_imag:.2e}i")
    print(f"   Confinement Loss:            {fund.loss_db_per_cm:.4e} dB/cm")
    print(f"   Polarization:                {fund.polarization_label} (fx={fund.fx:.3f}, fy={fund.fy:.3f})")

    spr.plot_geometry(project, save_path=str(out_dir / "solid_core_geom.png"))
    spr.plot_mode_field(fund, component="E_mag", save_path=str(out_dir / "solid_core_mode.png"))

    # 2. Hollow-Core PCF
    print("\n2. Simulating Hollow-Core PCF (Air Core)...")
    hc_geom = spr.HollowCorePCF(
        pitch=2.0e-6,
        rings=2,
        hole_radius=0.55e-6,
        core_radius=1.8e-6,
        cladding_radius=7.0e-6,
    )
    hc_geom.configure_pml(thickness=1.5e-6)
    hc_proj = spr.Project("Hollow_Core_PCF")
    hc_proj.set_geometry(hc_geom)
    hc_proj.assign_material("cladding", "silica")
    hc_proj.assign_material("fiber_core", "air")

    hc_study = spr.EigenmodeStudy(wavelength=1.55e-6, modes=2)
    hc_study.set_solver(backend="native", target_index=1.0, grid_points=50)
    hc_res = hc_study.run(hc_proj)

    hc_mode = hc_res.results[0].modes[0]
    print(f"   Air-Core Mode at 1550 nm:     n_eff = {hc_mode.n_real:.6f} + {hc_mode.n_imag:.2e}i")

    spr.plot_geometry(hc_proj, save_path=str(out_dir / "hollow_core_geom.png"))
    spr.plot_mode_field(hc_mode, component="E_mag", save_path=str(out_dir / "hollow_core_mode.png"))

    print(f"\n[OK] Hexagonal PCF examples completed. Plots saved to: {out_dir}")


if __name__ == "__main__":
    run_hexagonal_pcf_example()

