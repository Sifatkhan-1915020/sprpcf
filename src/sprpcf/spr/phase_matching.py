"""Phase matching and mode coupling between fiber core and surface plasmon modes."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class PhaseMatchingAnalysis:
    """Analyzes phase-matching condition Re(n_eff,core) ~ Re(n_eff,spp)."""

    def __init__(
        self,
        wavelengths_m: np.ndarray,
        core_n_real: np.ndarray,
        spp_n_real: np.ndarray,
        core_loss_db_cm: np.ndarray,
    ):
        self.wavelengths_m = np.asarray(wavelengths_m, dtype=float)
        self.core_n_real = np.asarray(core_n_real, dtype=float)
        self.spp_n_real = np.asarray(spp_n_real, dtype=float)
        self.core_loss_db_cm = np.asarray(core_loss_db_cm, dtype=float)

    def find_phase_matching_point(self) -> Dict[str, Any]:
        """Find the intersection / closest approach of the dispersion curves."""
        diff = np.abs(self.core_n_real - self.spp_n_real)
        min_idx = int(np.argmin(diff))
        match_wl = float(self.wavelengths_m[min_idx])
        min_dn = float(diff[min_idx])

        # Coupling strength indicator: inverse of minimal index gap at resonance
        coupling_strength = 1.0 / max(1e-5, min_dn)

        return {
            "phase_match_wavelength_nm": match_wl * 1e9,
            "min_index_difference": min_dn,
            "core_index_at_match": float(self.core_n_real[min_idx]),
            "spp_index_at_match": float(self.spp_n_real[min_idx]),
            "loss_at_match_db_cm": float(self.core_loss_db_cm[min_idx]),
            "coupling_strength_indicator": coupling_strength,
        }

