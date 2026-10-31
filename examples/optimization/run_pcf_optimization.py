"""Inverse Design and Parameter Optimization Example.

Optimizes gold coating thickness and lattice pitch to maximize biosensor FOM.
"""

from pathlib import Path
import numpy as np
import sprpcf as spr

def run_optimization_example():
    print("=" * 65)
    print("PCF INVERSE DESIGN & PARAMETRIC OPTIMIZATION")
    print("=" * 65)

    out_dir = Path("optimization_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base design
    project = spr.Project("Optimized_SPR_PCF")
    project.parameters.create(name="metal_thickness", value=40e-9, min_value=25e-9, max_value=60e-9)
    project.parameters.create(name="pitch", value=2.0e-6, min_value=1.5e-6, max_value=2.5e-6)

    def evaluate_fom(proj: spr.Project) -> float:
        t_gold = proj.parameters.get_value("metal_thickness")
        p = proj.parameters.get_value("pitch")

        # Rebuild geometry with updated parameters
        geom = spr.DShapedPCF(
            pitch=p,
            rings=2,
            hole_radius=p * 0.25,
            polish_depth=p * 1.5,
            metal_thickness=t_gold,
            cladding_radius=6.0e-6,
        )
        geom.configure_pml(thickness=1.2e-6)
        proj.set_geometry(geom)

        proj.assign_material("cladding", "silica")
        proj.assign_material("fiber_core", "silica")
        proj.assign_material("plasmonic_metal_film", "gold")
        proj.assign_material("sensing_analyte_channel", "analyte")

        study = spr.WavelengthSweepStudy(
            start_wavelength_m=900e-9,
            stop_wavelength_m=1300e-9,
            num_points=7,
            modes=2,
        )
        study.set_solver(backend="native", grid_points=45, tolerance=1e-5)
        res = study.run(proj)

        analyzer = spr.SPRAnalyzer(res.results)
        peak = analyzer.find_resonance(0)
        if peak:
            # Figure of Merit proxy: Peak Loss / FWHM
            return peak.peak_loss_db_cm / max(1.0, peak.fwhm_nm)
        return 0.0

    optimizer = spr.PCFOptimizer(
        project=project,
        parameter_bounds={
            "metal_thickness": (30e-9, 50e-9),
            "pitch": (1.8e-6, 2.2e-6),
        },
        evaluation_fn=evaluate_fom,
        maximize=True,
    )

    print("[*] Running Differential Evolution optimization (popsize=3, maxiter=2)...")
    opt_res = optimizer.run_differential_evolution(maxiter=2, popsize=3)

    best_p = opt_res["best_parameters"]
    best_score = opt_res["best_objective_value"]

    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Algorithm:       {opt_res['algorithm']}")
    print(f"Evaluations:     {opt_res['evaluations_count']}")
    print(f"Best Parameters: Au thickness = {best_p['metal_thickness']*1e9:.1f} nm, Pitch = {best_p['pitch']*1e6:.2f} um")
    print(f"Best FOM Score:  {best_score:.2f}")


if __name__ == "__main__":
    run_optimization_example()

