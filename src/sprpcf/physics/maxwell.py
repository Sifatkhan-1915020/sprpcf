"""Maxwell governing equations and frequency-domain electromagnetic formulations.

Governing Equations:
--------------------
For a source-free, linear, non-magnetic (mu_r = 1), isotropic or diagonal medium:
    curl(curl(E)) - k0^2 * epsilon_r * E = 0

For a longitudinally invariant waveguide propagating along z:
    E(x, y, z) = E(x, y) * exp(i * beta * z)
    H(x, y, z) = H(x, y) * exp(i * beta * z)

where:
    beta = beta_real + i * beta_imag (propagation constant, with beta_imag >= 0 for attenuation)
    k0 = 2 * pi / lambda_0 (vacuum wavenumber)
    n_eff = beta / k0 = n_real + i * n_imag (complex effective refractive index)
    confinement loss = (20 / ln(10)) * k0 * Im(n_eff) = 8.6858896 * k0 * Im(n_eff) [dB/m]
"""

import math
from typing import Tuple
from ..core.constants import C0, EPSILON_0, MU_0, NEPER_TO_DB


def calculate_k0(wavelength_m: float) -> float:
    """Calculate vacuum wavenumber k0 = 2*pi / lambda in rad/m."""
    if wavelength_m <= 0:
        raise ValueError(f"Wavelength must be positive, got {wavelength_m}")
    return 2.0 * math.pi / wavelength_m


def calculate_omega(wavelength_m: float) -> float:
    """Calculate angular frequency omega = 2*pi*c / lambda in rad/s."""
    return calculate_k0(wavelength_m) * C0


def beta_to_neff(beta: complex, k0: float) -> complex:
    """Convert complex propagation constant beta to effective refractive index n_eff = beta / k0."""
    return beta / k0


def neff_to_beta(neff: complex, k0: float) -> complex:
    """Convert complex effective refractive index n_eff to propagation constant beta = n_eff * k0."""
    return neff * k0


def calculate_loss_db_per_m(n_eff_imag: float, k0: float) -> float:
    """Calculate attenuation / confinement loss in dB/m:
    alpha = (20 / ln(10)) * k0 * Im(n_eff) ~ 8.68589 * k0 * Im(n_eff).
    """
    return NEPER_TO_DB * k0 * n_eff_imag

