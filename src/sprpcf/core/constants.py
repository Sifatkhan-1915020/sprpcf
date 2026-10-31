"""Fundamental physical constants and conversion factors in SI units.

Conforms to CODATA 2018 recommendations.
"""

import math

# Speed of light in vacuum (m/s)
SPEED_OF_LIGHT: float = 299792458.0
C0: float = SPEED_OF_LIGHT

# Vacuum electric permittivity (F/m)
VACUUM_PERMITTIVITY: float = 8.8541878128e-12
EPSILON_0: float = VACUUM_PERMITTIVITY

# Vacuum magnetic permeability (H/m)
VACUUM_PERMEABILITY: float = 1.25663706212e-6
MU_0: float = VACUUM_PERMEABILITY

# Characteristic impedance of vacuum (Ohms)
VACUUM_IMPEDANCE: float = math.sqrt(MU_0 / EPSILON_0)  # ~376.730313668 Ohms
ETA_0: float = VACUUM_IMPEDANCE

# Elementary charge (C)
ELEMENTARY_CHARGE: float = 1.602176634e-19
QE: float = ELEMENTARY_CHARGE

# Planck constant (J*s)
PLANCK_CONSTANT: float = 6.62607015e-34
H_PLANCK: float = PLANCK_CONSTANT

# Reduced Planck constant (J*s)
HBAR: float = H_PLANCK / (2.0 * math.pi)

# Boltzmann constant (J/K)
BOLTZMANN_CONSTANT: float = 1.380649e-23
KB: float = BOLTZMANN_CONSTANT

# Confinement loss unit conversion: Nepers to decibels factor
# alpha [dB/m] = 20 * log10(e) * Im(beta) = (20 / ln(10)) * Im(beta)
# ~ 8.685889638065037 dB / Np
NEPER_TO_DB: float = 20.0 / math.log(10.0)
DB_FACTOR: float = NEPER_TO_DB

