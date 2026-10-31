"""Full-Vector frequency-domain Maxwell transverse eigenmode solver with stretched-coordinate PML."""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .base import BaseModeSolver, OpticalMode, SolverConfig, SolverResult
from .eigenvalue import solve_sparse_eigenproblem
from ..physics.electromagnetic_fields import ElectromagneticField2D
from ..physics.maxwell import calculate_k0, calculate_omega
from ..physics.polarization import classify_polarization
from ..physics.poynting import calculate_domain_power_fractions
from ..core.constants import C0, EPSILON_0, MU_0
from ..geometry.cross_section import PCFGeometry
from ..exceptions import SolverConfigurationError, SimulationError


class FullVectorMaxwellSolver(BaseModeSolver):
    """Rigorous full-vector transverse Maxwell frequency-domain eigenmode solver."""

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)

    def solve(
        self,
        geometry: PCFGeometry,
        domain_materials: Dict[str, Any],
        wavelength_m: float,
        mesh: Optional[Any] = None,
    ) -> SolverResult:
        """Solve for complex propagation constants beta and full vector optical modes."""
        t_start = time.time()
        k0 = calculate_k0(wavelength_m)
        omega = calculate_omega(wavelength_m)

        # 1. Setup spatial computational grid
        # Domain size: cladding radius + PML thickness
        pml_t = geometry._pml_thickness if geometry._pml is not None else self.config.pml_thickness
        r_outer = geometry.cladding_radius + pml_t

        nx = self.config.grid_points_x
        ny = self.config.grid_points_y
        x = np.linspace(-r_outer, r_outer, nx)
        y = np.linspace(-r_outer, r_outer, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        X, Y = np.meshgrid(x, y)  # shape (ny, nx)

        # 2. Sample 2D permittivity distribution and domain masks
        eps_r, domain_masks = self._sample_permittivity(geometry, domain_materials, X, Y, wavelength_m)

        # 3. Setup complex stretched-coordinate PML factors s_x, s_y
        sx, sy = self._setup_pml_factors(X, Y, geometry.cladding_radius, pml_t, self.config.pml_scaling)

        # 4. Assemble Full-Vector Maxwell sparse operator P
        P = self._assemble_full_vector_operator(eps_r, sx, sy, dx, dy, k0, nx, ny)

        # 5. Solve for modes via Shift-and-Invert
        target_n = self.config.target_index or 1.44
        target_beta_sq = (target_n * k0)**2

        k_modes = self.config.num_modes
        vals, vecs, residuals = solve_sparse_eigenproblem(
            matrix_a=P,
            k_modes=k_modes,
            sigma=target_beta_sq,
            tol=self.config.tolerance,
            max_iter=self.config.max_iterations,
        )

        # 6. Reconstruct electromagnetic fields and physical mode properties
        modes: List[OpticalMode] = []
        n_pts = nx * ny

        for idx in range(len(vals)):
            lam = vals[idx]
            v = vecs[:, idx]
            res = residuals[idx]

            # beta = sqrt(lambda) with Re(beta) > 0 (forward) and Im(beta) >= 0 (attenuation)
            import cmath
            beta = cmath.sqrt(lam)
            if beta.real < 0:
                beta = -beta
            if beta.imag < 0:
                beta = complex(beta.real, abs(beta.imag))

            neff = beta / k0

            # Extract transverse Ex and Ey fields
            ex_flat = v[:n_pts]
            ey_flat = v[n_pts:]
            ex_2d = ex_flat.reshape((ny, nx))
            ey_2d = ey_flat.reshape((ny, nx))

            # Reconstruct longitudinal Ez and full magnetic field components
            ez_2d, hx_2d, hy_2d, hz_2d = self._reconstruct_fields(
                ex_2d, ey_2d, eps_r, sx, sy, dx, dy, beta, k0, omega
            )

            field = ElectromagneticField2D(
                x=x,
                y=y,
                ex=ex_2d,
                ey=ey_2d,
                ez=ez_2d,
                hx=hx_2d,
                hy=hy_2d,
                hz=hz_2d,
                wavelength_m=wavelength_m,
            ).normalize()

            # Analyze polarization
            pol_info = classify_polarization(field)

            # Analyze domain power fractions
            flat_masks = {dname: m.reshape((ny, nx)) for dname, m in domain_masks.items()}
            power_fractions = calculate_domain_power_fractions(field, flat_masks)

            mode = OpticalMode(
                mode_id=idx,
                n_eff=neff,
                wavelength_m=wavelength_m,
                field=field,
                residual=res,
                polarization_fraction_x=pol_info["fx"],
                polarization_fraction_y=pol_info["fy"],
                domain_power_fractions=power_fractions,
            )
            modes.append(mode)

        # Filter by polarization if requested
        if self.config.polarization == "x":
            modes = [m for m in modes if m.polarization_label == "x_pol"]
        elif self.config.polarization == "y":
            modes = [m for m in modes if m.polarization_label == "y_pol"]

        t_elapsed = time.time() - t_start
        return SolverResult(
            wavelength_m=wavelength_m,
            modes=modes,
            execution_time=t_elapsed,
            solver_name="FullVectorMaxwellSolver",
            converged=True,
            metadata={"nx": nx, "ny": ny, "k0": k0},
        )

    def _sample_permittivity(
        self,
        geometry: PCFGeometry,
        domain_materials: Dict[str, Any],
        X: np.ndarray,
        Y: np.ndarray,
        wavelength_m: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute complex permittivity epsilon_r(x, y) on the grid from geometry domains."""
        ny, nx = X.shape
        eps_r = np.ones((ny, nx), dtype=complex)
        domain_masks: Dict[str, np.ndarray] = {}

        partitioned = geometry.partition()
        # Sort so that smaller embedded regions overwrite background
        sorted_regions = sorted(
            partitioned.values(),
            key=lambda r: r.area if hasattr(r, "area") else 1e10,
            reverse=True,
        )

        import shapely.geometry as sg
        import shapely.prepared as sp

        for reg in sorted_regions:
            dname = reg.name
            mat = domain_materials.get(dname) or domain_materials.get(reg.material_tag)
            if mat is None:
                continue

            eps_val = mat.get_permittivity(wavelength_m)
            # Create mask of points inside region
            prep_geom = sp.prep(reg.shapely_geom)
            # Fast vectorized/pointwise containment
            mask = np.zeros((ny, nx), dtype=bool)
            minx, miny, maxx, maxy = reg.bounds
            # Candidate box filter
            cand_y, cand_x = np.where((X >= minx) & (X <= maxx) & (Y >= miny) & (Y <= maxy))
            for iy, ix in zip(cand_y, cand_x):
                if prep_geom.contains(sg.Point(X[iy, ix], Y[iy, ix])):
                    mask[iy, ix] = True

            eps_r[mask] = eps_val
            # Ensure disjoint partition of masks: subtract smaller embedded region from larger regions
            for prev_name in domain_masks:
                domain_masks[prev_name] = domain_masks[prev_name] & (~mask)
            domain_masks[dname] = mask

        return eps_r, domain_masks

    def _setup_pml_factors(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        r_inner: float,
        d_pml: float,
        pml_scaling: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D coordinate-stretching factors s_x and s_y."""
        R = np.hypot(X, Y)
        sx = np.ones_like(X, dtype=complex)
        sy = np.ones_like(Y, dtype=complex)

        pml_mask = R > r_inner
        if np.any(pml_mask) and d_pml > 1e-12:
            dr = np.clip((R - r_inner) / d_pml, 0.0, 1.0)
            sigma = pml_scaling * (dr**2)
            # Complex stretching: s = 1 - i * sigma
            s_val = 1.0 - 1j * sigma
            sx[pml_mask] = s_val[pml_mask]
            sy[pml_mask] = s_val[pml_mask]

        return sx, sy

    def _assemble_full_vector_operator(
        self,
        eps_r: np.ndarray,
        sx: np.ndarray,
        sy: np.ndarray,
        dx: float,
        dy: float,
        k0: float,
        nx: int,
        ny: int,
    ) -> sp.csr_matrix:
        """Assemble full-vector transverse Maxwell operator P in CSR format."""
        n_pts = nx * ny

        # 1D differentiation matrices with Dirichlet zero boundary
        # D_x on (ny, nx)
        ex_1d = np.ones(nx)
        Dx_1d = sp.diags([-ex_1d[1:], ex_1d[1:]], [-1, 1], shape=(nx, nx)) / (2.0 * dx)
        Dxx_1d = sp.diags([ex_1d[1:], -2.0 * ex_1d, ex_1d[1:]], [-1, 0, 1], shape=(nx, nx)) / (dx * dx)

        ey_1d = np.ones(ny)
        Dy_1d = sp.diags([-ey_1d[1:], ey_1d[1:]], [-1, 1], shape=(ny, ny)) / (2.0 * dy)
        Dyy_1d = sp.diags([ey_1d[1:], -2.0 * ey_1d, ey_1d[1:]], [-1, 0, 1], shape=(ny, ny)) / (dy * dy)

        # Kronecker products for 2D differential operators
        Ix = sp.eye(nx)
        Iy = sp.eye(ny)

        Dx = sp.kron(Iy, Dx_1d).tocsr()
        Dy = sp.kron(Dy_1d, Ix).tocsr()
        Dxx = sp.kron(Iy, Dxx_1d).tocsr()
        Dyy = sp.kron(Dyy_1d, Ix).tocsr()

        # PML stretching
        inv_sx = sp.diags(1.0 / sx.flatten()).tocsr()
        inv_sy = sp.diags(1.0 / sy.flatten()).tocsr()

        Dx_pml = inv_sx.dot(Dx)
        Dy_pml = inv_sy.dot(Dy)
        Dxx_pml = inv_sx.dot(inv_sx.dot(Dxx))
        Dyy_pml = inv_sy.dot(inv_sy.dot(Dyy))

        # Permittivity diagonals
        eps_flat = eps_r.flatten()
        inv_eps_flat = 1.0 / eps_flat

        Eps = sp.diags(eps_flat).tocsr()
        InvEps = sp.diags(inv_eps_flat).tocsr()

        # Assemble Pxx = Dx(InvEps * Dx(Eps)) + Dyy + k0^2 * Eps
        Dx_Eps = Dx_pml.dot(Eps)
        Pxx = Dx_pml.dot(InvEps.dot(Dx_Eps)) + Dyy_pml + (k0**2) * Eps

        # Assemble Pxy = Dx(InvEps * Dy(Eps)) - Dx * Dy
        Dy_Eps = Dy_pml.dot(Eps)
        Pxy = Dx_pml.dot(InvEps.dot(Dy_Eps)) - Dx_pml.dot(Dy_pml)

        # Assemble Pyx = Dy(InvEps * Dx(Eps)) - Dy * Dx
        Pyx = Dy_pml.dot(InvEps.dot(Dx_Eps)) - Dy_pml.dot(Dx_pml)

        # Assemble Pyy = Dy(InvEps * Dy(Eps)) + Dxx + k0^2 * Eps
        Pyy = Dy_pml.dot(InvEps.dot(Dy_Eps)) + Dxx_pml + (k0**2) * Eps

        # Combine into 2x2 block matrix
        P = sp.bmat([[Pxx, Pxy], [Pyx, Pyy]], format="csr")
        return P

    def _reconstruct_fields(
        self,
        ex: np.ndarray,
        ey: np.ndarray,
        eps_r: np.ndarray,
        sx: np.ndarray,
        sy: np.ndarray,
        dx: float,
        dy: float,
        beta: complex,
        k0: float,
        omega: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct Ez, Hx, Hy, Hz via Maxwell's relations."""
        # Stretched gradient of (eps * E)
        d_eps_ex_dx = np.gradient(eps_r * ex, dx, axis=1) / sx
        d_eps_ey_dy = np.gradient(eps_r * ey, dy, axis=0) / sy

        # Ez = (i / (beta * eps_r)) * (d(eps*Ex)/dx + d(eps*Ey)/dy)
        ez = (1j / (beta * eps_r)) * (d_eps_ex_dx + d_eps_ey_dy)

        # dEz/dx and dEz/dy
        dez_dx = np.gradient(ez, dx, axis=1) / sx
        dez_dy = np.gradient(ez, dy, axis=0) / sy

        # Faraday's law with exp[j(omega*t - beta*z)]:
        # curl(E) = -j * omega * mu0 * H
        # Hx = (1 / (omega * mu0)) * (-beta * Ey + j * dEz/dy)
        hx = (1.0 / (omega * MU_0)) * (-beta * ey + 1j * dez_dy)

        # Hy = (1 / (omega * mu0)) * (beta * Ex - j * dEz/dx)
        hy = (1.0 / (omega * MU_0)) * (beta * ex - 1j * dez_dx)

        # Hz = (i / (omega * mu0)) * (dEy/dx - dEx/dy)
        dey_dx = np.gradient(ey, dx, axis=1) / sx
        dex_dy = np.gradient(ex, dy, axis=0) / sy
        hz = (1j / (omega * MU_0)) * (dey_dx - dex_dy)

        return ez, hx, hy, hz
