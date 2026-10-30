"""
Tests of SpiceParameters: resolution of bodies, GMs and units, and input validation
"""

import math

import numpy as np
import pytest

from helpers import DATA_DIR, campanella, furnsh_kernels

ET0 = 740463475.5117936
STATE0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
          0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
# gm_de440.tpc
GM_DE440 = {301: 4902.8001184575496, 399: 398600.43550702266, 10: 1.3271244004127942e11}
MOON_FILE = DATA_DIR + "/gggrx_1200a_sha_n10.tab"


def test_properties():
    p = campanella.SpiceParameters(ET0, ["MOON", "399", "Sun"], GMs=[4902.8, 398600.4, 1.3e11],
                                   DU=1e5, include_srp=True, srp_A=10.0, srp_m=500.0, srp_rhoS=0.2,
                                   srp_rhoD=0.1)
    assert p.et0 == ET0
    assert p.naif_ids == ["MOON", "399", "Sun"]
    assert p.naif_codes == [301, 399, 10]
    assert p.naif_frame == "J2000"
    assert p.GMs == [4902.8, 398600.4, 1.3e11]
    assert p.DU == 1e5 and p.VU == math.sqrt(4902.8 / 1e5) and p.TU == 1e5 / p.VU
    assert p.mus == [1.0, 398600.4 / 4902.8, 1.3e11 / 4902.8]
    assert p.use_canonical_scales and p.abcorr == "NONE"
    assert p.include_srp and (p.srp_A, p.srp_m, p.srp_rhoS, p.srp_rhoD) == (10.0, 500.0, 0.2, 0.1)
    assert not p.include_spherical_harmonics and p.sh_file == ""
    assert "naif_ids=['MOON', '399', 'Sun']" in repr(p) and "include_srp=True" in repr(p)
    with pytest.raises(AttributeError):
        p.DU = 2.0


def test_defaults_are_dimensional():
    p = campanella.SpiceParameters(0.0, ["EARTH"], GMs=[398600.4])
    assert (p.DU, p.TU, p.VU) == (1.0, 1.0, 1.0)
    assert p.mus == [398600.4]
    assert not p.include_srp and not p.include_spherical_harmonics


def test_gms_from_kernels_match_explicit():
    """GMs read from gm_de440.tpc give exactly the results of the same GMs passed explicitly"""
    furnsh_kernels()
    ids = ["301", "399", "10"]
    from_spice = campanella.SpiceParameters(ET0, ids)
    # the kernel-pool parser may differ from Python's float() by one ulp
    np.testing.assert_allclose(from_spice.GMs, [GM_DE440[301], GM_DE440[399], GM_DE440[10]], rtol=1e-15)
    explicit = campanella.SpiceParameters(ET0, ids, GMs=from_spice.GMs)
    a = campanella.solve_ivp_rv(from_spice, [0.0, 86400.0], STATE0)
    b = campanella.solve_ivp_rv(explicit, [0.0, 86400.0], STATE0)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.x, b.x)


def test_names_and_numeric_ids_are_equivalent():
    furnsh_kernels()
    gm = [4902.8, 398600.4, 1.3e11]
    by_name = campanella.SpiceParameters(ET0, ["MOON", "EARTH", "SUN"], GMs=gm, include_srp=True,
                                         srp_A=10.0, srp_m=500.0)
    by_code = campanella.SpiceParameters(ET0, ["301", "399", "10"], GMs=gm, include_srp=True,
                                         srp_A=10.0, srp_m=500.0)
    assert by_name.naif_codes == by_code.naif_codes
    a = campanella.solve_ivp_rv(by_name, [0.0, 86400.0], STATE0)
    b = campanella.solve_ivp_rv(by_code, [0.0, 86400.0], STATE0)
    np.testing.assert_array_equal(a.x, b.x)


def test_sun_listed_anywhere_for_srp():
    """SRP needs the Sun in naif_ids, in any position, including as the central body"""
    furnsh_kernels()
    srp = dict(include_srp=True, srp_A=10.0, srp_m=500.0)
    campanella.SpiceParameters(ET0, ["EARTH", "SUN", "MOON"], GMs=[398600.4, 1.3e11, 4902.8], **srp)
    sun = campanella.SpiceParameters(ET0, ["SUN"], GMs=[1.3e11], **srp)
    assert sun.include_srp


@pytest.mark.parametrize("args, kwargs, match", [
    ((ET0, []), {}, "naif_ids"),
    ((float("nan"), ["EARTH"]), dict(GMs=[1.0]), "et0"),
    ((ET0, ["NOT_A_BODY"]), dict(GMs=[1.0]), "unknown NAIF body"),
    ((ET0, ["EARTH", "399"]), dict(GMs=[1.0, 1.0]), "listed twice"),
    ((ET0, ["EARTH", "MOON"]), dict(GMs=[1.0]), "one entry per NAIF id"),
    ((ET0, ["EARTH"]), dict(GMs=[0.0]), "GMs"),
    ((ET0, ["EARTH", "MOON"]), dict(GMs=[1.0, -1.0]), "GMs"),
    ((ET0, ["EARTH"]), dict(GMs=[float("inf")]), "GMs"),
    ((ET0, ["EARTH"]), dict(GMs=[1.0], DU=0.0), "DU"),
    ((ET0, ["EARTH"]), dict(GMs=[1.0], DU=-5.0), "DU"),
    ((ET0, ["EARTH"]), dict(GMs=[1.0], abcorr="LT+S"), "abcorr"),
    ((ET0, ["EARTH"]), dict(GMs=[1.0], naif_frame="NOT_A_FRAME"), "naif_frame"),
    ((ET0, ["EARTH"]), dict(GMs=[1.0], naif_frame="IAU_EARTH"), "not inertial"),
    ((ET0, ["EARTH", "MOON"]), dict(GMs=[1.0, 1.0], include_srp=True, srp_A=1.0, srp_m=1.0), "Sun"),
    ((ET0, ["EARTH", "SUN"]), dict(GMs=[1.0, 1.0], include_srp=True, srp_A=1.0, srp_m=0.0), "srp_m"),
    ((ET0, ["EARTH", "SUN"]), dict(GMs=[1.0, 1.0], include_srp=True, srp_A=-1.0, srp_m=1.0), "srp_A"),
    ((ET0, ["EARTH", "SUN"]), dict(GMs=[1.0, 1.0], include_srp=True, srp_A=1.0, srp_m=1.0,
                                   srp_rhoS=-0.1), "srp_rhoS"),
    ((ET0, ["MOON"]), dict(GMs=[4902.8], sh_file=MOON_FILE, sh_nmax=10, sh_frame="MOON_PA",
                           sh_rho_min=0.0), "sh_rho_min"),
])
def test_invalid_arguments(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        campanella.SpiceParameters(*args, **kwargs)


def test_missing_gm_in_kernel_pool():
    """A body without GM in the kernel pool: clear error naming the body"""
    furnsh_kernels()
    with pytest.raises(ValueError, match=r"no GM for body \"-82\""):
        campanella.SpiceParameters(ET0, ["399", "-82"])


def test_abcorr_case_insensitive():
    assert campanella.SpiceParameters(0.0, ["EARTH"], GMs=[1.0], abcorr="none").abcorr == "none"


def test_sh_parameters():
    furnsh_kernels()
    p = campanella.SpiceParameters(ET0, ["MOON"], GMs=[4902.800582147760], sh_file=MOON_FILE,
                                   sh_nmax=8, sh_mmax=6, sh_frame="MOON_PA", sh_rho_min=0.99)
    assert p.include_spherical_harmonics
    assert (p.sh_file, p.sh_nmax, p.sh_mmax, p.sh_frame, p.sh_rho_min) == (MOON_FILE, 8, 6, "MOON_PA", 0.99)
