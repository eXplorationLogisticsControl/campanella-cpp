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
