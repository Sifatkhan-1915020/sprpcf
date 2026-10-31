"""Confinement loss spectrum and attenuation processing."""

from typing import List, Tuple, Dict, Any, Union
import numpy as np
from ..physics.maxwell import calculate_loss_db_per_m, calculate_k0
from ..core.units import attenuation_conversion
from ..solvers.base import OpticalMode


class LossSpectrum:
    """Stores attenuation spectrum for an optical mode across wavelengths."""

    def __init__(self, wavelengths_m: np.ndarray, n_eff_array: np.ndarray):
        self.wavelengths_m = np.asarray(wavelengths_m, dtype=float)
        self.n_eff_array = np.asarray(n_eff_array, dtype=complex)
        self.k0_array = 2.0 * np.pi / self.wavelengths_m

    @property
    def loss_db_per_m(self) -> np.ndarray:
        return calculate_loss_db_per_m(np.maximum(0.0, self.n_eff_array.imag), self.k0_array)

    @property
    def loss_db_per_cm(self) -> np.ndarray:
        return self.loss_db_per_m * 1e-2

    @property
    def loss_db_per_um(self) -> np.ndarray:
        return self.loss_db_per_m * 1e-6

    def get_loss_in_unit(self, unit: str = "db/cm") -> np.ndarray:
        u = unit.lower().replace(" ", "")
        if u == "db/cm":
            return self.loss_db_per_cm
        elif u == "db/m":
            return self.loss_db_per_m
        elif u == "db/um":
            return self.loss_db_per_um
        else:
            raise ValueError(f"Unknown loss unit '{unit}'")

