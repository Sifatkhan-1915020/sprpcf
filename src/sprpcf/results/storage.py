"""Scientific data storage (HDF5 and NPZ) and CSV export."""

import csv
from pathlib import Path
from typing import Dict, Any, Union, List
import numpy as np
from ..studies.base import StudyResult


def export_study_to_csv(study_result: StudyResult, filepath: Union[str, Path]) -> None:
    """Export mode propagation constants, effective indices, and loss to CSV."""
    path = Path(filepath)
    fieldnames = [
        "wavelength_nm",
        "mode_id",
        "n_real",
        "n_imag",
        "loss_db_per_cm",
        "polarization",
        "fx",
        "fy",
        "residual",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in study_result.results:
            wl_nm = r.wavelength_m * 1e9
            for m in r.modes:
                writer.writerow({
                    "wavelength_nm": round(wl_nm, 4),
                    "mode_id": m.mode_id,
                    "n_real": round(m.n_real, 6),
                    "n_imag": float(f"{m.n_imag:.6e}"),
                    "loss_db_per_cm": round(m.loss_db_per_cm, 4),
                    "polarization": m.polarization_label,
                    "fx": round(m.fx, 4),
                    "fy": round(m.fy, 4),
                    "residual": float(f"{m.residual:.6e}"),
                })


def export_study_to_hdf5(study_result: StudyResult, filepath: Union[str, Path]) -> None:
    """Save full study results including electromagnetic field arrays into HDF5."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 export. Install via 'pip install h5py'.")

    path = Path(filepath)
    with h5py.File(path, "w") as f:
        f.attrs["study_name"] = study_result.study_name
        f.attrs["timestamp"] = study_result.timestamp
        f.attrs["steps_count"] = len(study_result.results)

        for step_idx, res in enumerate(study_result.results):
            grp = f.create_group(f"step_{step_idx:04d}")
            grp.attrs["wavelength_m"] = res.wavelength_m
            grp.attrs["execution_time_s"] = res.execution_time

            for m in res.modes:
                m_grp = grp.create_group(f"mode_{m.mode_id}")
                m_grp.attrs["n_real"] = m.n_real
                m_grp.attrs["n_imag"] = m.n_imag
                m_grp.attrs["loss_db_per_cm"] = m.loss_db_per_cm
                m_grp.attrs["polarization"] = m.polarization_label
                m_grp.attrs["residual"] = m.residual

                if m.field is not None:
                    m_grp.create_dataset("x", data=m.field.x)
                    m_grp.create_dataset("y", data=m.field.y)
                    m_grp.create_dataset("ex_real", data=np.real(m.field.ex))
                    m_grp.create_dataset("ex_imag", data=np.imag(m.field.ex))
                    m_grp.create_dataset("ey_real", data=np.real(m.field.ey))
                    m_grp.create_dataset("ey_imag", data=np.imag(m.field.ey))
