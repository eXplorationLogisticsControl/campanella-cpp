"""
Shared helpers of the pytest suite: module path, SPICE kernels and result accessors.

Kernels are read from $SPICE: lsk/naif0012.tls, spk/de440.bsp, pck/gm_de440.tpc, and for the
spherical-harmonics body-fixed frames pck/moon_pa_de440_200625.bpc, fk/moon_de440_250416.tf
(MOON_PA) and pck/earth_latest_high_prec.bpc (ITRF93).
"""

import os
import sys

import numpy as np
import pytest

_LOAD_DIR = "build/Release" if sys.platform == "win32" else "build/lib"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", _LOAD_DIR))

import campanella  # noqa: E402

KERNELS = ["lsk/naif0012.tls", "spk/de440.bsp", "pck/gm_de440.tpc",
           "pck/moon_pa_de440_200625.bpc", "fk/moon_de440_250416.tf",
           "pck/earth_latest_high_prec.bpc"]
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

_loaded = False


def furnsh_kernels():
    """Load KERNELS from $SPICE once per process. Without $SPICE the test is skipped, unless
    CAMPANELLA_REQUIRE_KERNELS is set (as in CI), in which case it fails."""
    global _loaded
    if _loaded:
        return
    spice_dir = os.environ.get("SPICE", "")
    if not spice_dir:
        if os.environ.get("CAMPANELLA_REQUIRE_KERNELS", "0") not in ("", "0"):
            pytest.fail("SPICE is not set, but CAMPANELLA_REQUIRE_KERNELS is")
        pytest.skip("SPICE is not set")
    for kernel in KERNELS:
        campanella.furnsh(os.path.join(spice_dir, kernel))
    _loaded = True


def final_row(res, tf):
    """Final state (and flattened STM, if any) of an IvpResult, checking that it reached tf"""
    assert res.t[-1] == tf, f"integration stopped at t = {res.t[-1]}"
    if res.stm is None:
        return np.array(res.x[-1])
    return np.concatenate((res.x[-1], res.stm[-1].ravel()))


def row_normalised_error(M, M_ref):
    """max over rows of max_j |M_ij - Mref_ij| / max_j |Mref_ij|"""
    return np.max(np.abs(M - M_ref).max(axis=1) / np.abs(M_ref).max(axis=1))


def final_state_n(res, n):
    """Final row of res.x, for n states"""
    return np.array(res.x).reshape(-1, n)[-1]


def final_stm_n(res, n):
    """Final Phi of res.stm, n x n"""
    return np.array(res.stm).reshape(-1, n, n)[-1]


def final_stm_u(res, nx, nu):
    """Final Psi of res.stm_u, nx x nu"""
    return np.array(res.stm_u).reshape(-1, nx, nu)[-1]


def jacobian_fd_n(f, x, h):
    """Central differences of f: R^n -> R^m with steps h (n), as an m x n matrix"""
    x = np.asarray(x, dtype=float)
    cols = []
    for j in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[j] += h[j]
        xm[j] -= h[j]
        cols.append((np.asarray(f(xp)) - np.asarray(f(xm))) / (xp[j] - xm[j]))
    return np.stack(cols, axis=1)


def matrix_error(M, R, rows, cols, row_scale=None, col_scale=None, abs_floor=0.0, mode="row"):
    """Normalised error of the rows x cols matrix M against the reference R, after scaling both as
    diag(1/row_scale) M diag(col_scale). None means ones.

    mode "row": max over rows of max_j |dM_ij| / max_j |R_ij|; mode "col": the same over columns.
    A row (column) whose scaled reference is at most abs_floor has nothing to normalise by, and is
    checked absolutely: it contributes 0 if its scaled difference is at most abs_floor, inf otherwise.
    Returns inf if any value is not finite, or if a size is inconsistent."""
    if mode not in ("row", "col"):
        raise ValueError(f"unknown mode {mode}")
    M, R = np.asarray(M, dtype=float), np.asarray(R, dtype=float)
    rs = np.ones(rows) if row_scale is None else np.asarray(row_scale, dtype=float)
    cs = np.ones(cols) if col_scale is None else np.asarray(col_scale, dtype=float)
    if M.size != rows * cols or R.size != rows * cols or rs.size != rows or cs.size != cols:
        return np.inf
    scale = np.outer(1.0 / rs, cs)
    M, R = M.reshape(rows, cols) * scale, R.reshape(rows, cols) * scale
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(R))):
        return np.inf
    axis = 1 if mode == "row" else 0
    diff, ref = np.abs(M - R).max(axis=axis), np.abs(R).max(axis=axis)
    zero = ref <= abs_floor
    if np.any(diff[zero] > abs_floor):
        return np.inf
    return float(np.max(diff[~zero] / ref[~zero], initial=0.0))


def solve_linear(A, B):
    """Solve A X = B with partial pivoting (LAPACK); B may be a vector or a matrix"""
    return np.linalg.solve(A, B)


def assert_state_close(x, x_ref, rtol):
    """Position and velocity errors relative to |r| and |v| of x_ref. Per-component tolerances are
    too tight for small components: rounding differences between platforms (FMA contraction on
    arm64, libm, CSPICE builds) give ~1e-13 of |v| after 20 days."""
    x, x_ref = np.asarray(x), np.asarray(x_ref)
    for sl, what in ((slice(0, 3), "position"), (slice(3, 6), "velocity")):
        err = np.linalg.norm(x[sl] - x_ref[sl])
        assert err <= rtol * np.linalg.norm(x_ref[sl]), f"{what} error {err:.3e}"
