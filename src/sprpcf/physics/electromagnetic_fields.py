"""Data structures and calculations for 2D electromagnetic mode field distributions."""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.ndimage import zoom


class ElectromagneticField2D:
    """Stores full-vector transverse and longitudinal field components on a 2D mesh or grid."""

    def __init__(
        self,
        x: np.ndarray,  # 1D x coordinates (meters)
        y: np.ndarray,  # 1D y coordinates (meters)
        ex: np.ndarray,  # Complex Ex component (ny, nx)
        ey: np.ndarray,  # Complex Ey component (ny, nx)
        ez: Optional[np.ndarray] = None,  # Complex Ez component (ny, nx)
        hx: Optional[np.ndarray] = None,  # Complex Hx component (ny, nx)
        hy: Optional[np.ndarray] = None,  # Complex Hy component (ny, nx)
        hz: Optional[np.ndarray] = None,  # Complex Hz component (ny, nx)
        wavelength_m: float = 1.55e-6,
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.ex = np.asarray(ex, dtype=complex)
        self.ey = np.asarray(ey, dtype=complex)
        self.ez = np.asarray(ez, dtype=complex) if ez is not None else np.zeros_like(self.ex)
        self.hx = np.asarray(hx, dtype=complex) if hx is not None else np.zeros_like(self.ex)
        self.hy = np.asarray(hy, dtype=complex) if hy is not None else np.zeros_like(self.ex)
        self.hz = np.asarray(hz, dtype=complex) if hz is not None else np.zeros_like(self.ex)
        self.wavelength_m = float(wavelength_m)

    @property
    def dx(self) -> float:
        return abs(self.x[1] - self.x[0]) if len(self.x) > 1 else 1.0

    @property
    def dy(self) -> float:
        return abs(self.y[1] - self.y[0]) if len(self.y) > 1 else 1.0

    @property
    def e_mag_sq(self) -> np.ndarray:
        """Total electric field intensity |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2."""
        return np.abs(self.ex)**2 + np.abs(self.ey)**2 + np.abs(self.ez)**2

    @property
    def e_mag(self) -> np.ndarray:
        """Total electric field magnitude |E|."""
        return np.sqrt(self.e_mag_sq)

    @property
    def e_transverse_mag_sq(self) -> np.ndarray:
        """Transverse electric field intensity |E_t|^2 = |Ex|^2 + |Ey|^2."""
        return np.abs(self.ex)**2 + np.abs(self.ey)**2

    @property
    def h_mag_sq(self) -> np.ndarray:
        """Total magnetic field intensity |H|^2 = |Hx|^2 + |Hy|^2 + |Hz|^2."""
        return np.abs(self.hx)**2 + np.abs(self.hy)**2 + np.abs(self.hz)**2

    @property
    def h_mag(self) -> np.ndarray:
        return np.sqrt(self.h_mag_sq)

    @property
    def poynting_z(self) -> np.ndarray:
        """Longitudinal time-averaged Poynting vector S_z = 0.5 * Re(Ex * Hy* - Ey * Hx*)."""
        return 0.5 * np.real(self.ex * np.conj(self.hy) - self.ey * np.conj(self.hx))

    @property
    def total_power_flow(self) -> float:
        """Cross-sectional power flow P_total = \\iint S_z dx dy in Watts."""
        sz = self.poynting_z
        return float(np.sum(sz) * self.dx * self.dy)

    @property
    def is_forward_propagating(self) -> bool:
        """Verify net positive power flow in forward +z direction."""
        return self.total_power_flow > 0.0

    @property
    def polarization_fractions(self) -> Tuple[float, float]:
        """Compute X and Y polarization power fractions (fx, fy) where fx + fy = 1."""
        int_ex2 = float(np.sum(np.abs(self.ex)**2))
        int_ey2 = float(np.sum(np.abs(self.ey)**2))
        total = int_ex2 + int_ey2
        if total < 1e-25:
            return 0.5, 0.5
        return int_ex2 / total, int_ey2 / total

    @property
    def dominant_phase(self) -> np.ndarray:
        """Phase of dominant transverse electric field component."""
        fx, fy = self.polarization_fractions
        dom_field = self.ex if fx >= fy else self.ey
        return np.angle(dom_field)

    def normalize(self) -> "ElectromagneticField2D":
        """Normalize field so that maximum |E| is 1.0."""
        max_e = np.max(self.e_mag)
        if max_e > 1e-25:
            self.ex /= max_e
            self.ey /= max_e
            self.ez /= max_e
            self.hx /= max_e
            self.hy /= max_e
            self.hz /= max_e
        return self

    def interpolate_bicubic(self, factor: int = 2) -> "ElectromagneticField2D":
        """Bicubic spline interpolation to a finer grid for continuous rendering without raster artifacts."""
        if factor <= 1:
            return self

        # Interpolate real and imaginary parts of components
        def _zoom_c(arr: np.ndarray) -> np.ndarray:
            re = zoom(arr.real, factor, order=3)
            im = zoom(arr.imag, factor, order=3)
            return re + 1j * im

        new_x = np.linspace(self.x[0], self.x[-1], len(self.x) * factor)
        new_y = np.linspace(self.y[0], self.y[-1], len(self.y) * factor)

        return ElectromagneticField2D(
            x=new_x,
            y=new_y,
            ex=_zoom_c(self.ex),
            ey=_zoom_c(self.ey),
            ez=_zoom_c(self.ez),
            hx=_zoom_c(self.hx),
            hy=_zoom_c(self.hy),
            hz=_zoom_c(self.hz),
            wavelength_m=self.wavelength_m,
        )
