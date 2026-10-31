"""Shift-and-invert sparse eigensolvers and numerical residual diagnostics."""

import time
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..exceptions import EigenvalueConvergenceError


def solve_sparse_eigenproblem(
    matrix_a: sp.spmatrix,
    matrix_b: Optional[sp.spmatrix] = None,
    k_modes: int = 6,
    sigma: complex = 1.44**2,  # Shift target (beta^2)
    tol: float = 1e-8,
    max_iter: int = 500,
    which: str = "LM",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem A * v = lambda * B * v using shift-and-invert ARPACK.
    
    Returns:
        eigenvalues: 1D complex array of length k
        eigenvectors: 2D complex array of shape (N, k)
        residuals: 1D float array of relative residuals ||A*v - lambda*B*v|| / (|lambda| * ||B*v||)
    """
    n = matrix_a.shape[0]
    k_modes = min(k_modes, n - 2)
    if k_modes < 1:
        raise ValueError("Matrix dimension too small for eigenvalue solution.")

    # Convert to CSC format for fast sparse LU factorization
    a_csc = matrix_a.tocsc()
    b_csc = matrix_b.tocsc() if matrix_b is not None else None

    try:
        vals, vecs = spla.eigs(
            A=a_csc,
            k=k_modes,
            M=b_csc,
            sigma=sigma,
            which=which,
            tol=tol,
            maxiter=max_iter,
        )
    except Exception as e:
        # Fall back to largest magnitude or standard shift without M if generalized fails
        try:
            vals, vecs = spla.eigs(A=a_csc, k=k_modes, sigma=sigma, which="LM", tol=tol * 10)
        except Exception as e2:
            raise EigenvalueConvergenceError(f"ARPACK eigensolver failed to converge: {e2}")

    # Compute numerical residuals for each eigenpair
    residuals = []
    for i in range(len(vals)):
        lam = vals[i]
        v = vecs[:, i]
        av = a_csc.dot(v)
        bv = b_csc.dot(v) if b_csc is not None else v
        diff = av - lam * bv
        norm_diff = float(np.linalg.norm(diff))
        norm_denom = float(max(1e-12, abs(lam) * np.linalg.norm(bv)))
        res = norm_diff / norm_denom
        residuals.append(res)

    return vals, vecs, np.array(residuals)


def compute_eigenvalue_diagnostics(
    eigenvalues: np.ndarray,
    residuals: np.ndarray,
    k0: float,
) -> List[Dict[str, Any]]:
    """Compute physical metrics and diagnostics for raw eigenvalues (beta^2)."""
    diagnostics = []
    for i, (lam, res) in enumerate(zip(eigenvalues, residuals)):
        # beta = sqrt(lambda)
        import cmath
        beta = cmath.sqrt(lam)
        neff = beta / k0
        diagnostics.append({
            "mode_idx": i,
            "lambda": complex(lam),
            "beta": complex(beta),
            "n_real": float(neff.real),
            "n_imag": float(abs(neff.imag)),
            "residual": float(res),
            "is_converged": bool(res < 1e-3),
        })
    return diagnostics

