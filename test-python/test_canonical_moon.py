"""
Tests for the Moon-centred propagators in canonical units.

The central body is the Moon, with LU = 100000 km and TU = LU / sqrt(MU_MOON / LU),
so the scaled lunar GM is 1. Each test propagates the same case in km/s and in
canonical units, converts the canonical result back, and compares.

Scaling:
    Cartesian state  r / LU, v / VU     (VU = LU / TU)
    MEE              p / LU; f, g, h, k, L unchanged
    GM               GM_i / MU_MOON     (= GM_i TU^2 / LU^3)
    time             t / TU; et0 stays in ephemeris seconds
    STM              Phi_km = D Phi_canon D^-1, D = diag of the state scales

The integrator tolerances are absolute in the integrated variables, so abstol = 1e-12
is 1e-7 km in canonical units against 1e-12 km in km/s. Each tolerance below is about
100 times the observed difference.
"""

import os
import sys

import numpy as np

if sys.platform == "win32":
    load_dir = "build/Release"
else:
    load_dir = "build/lib"

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..", load_dir
    )
)

import campanella


def furnsh_kernels():
    spice_dir = os.environ["SPICE"]
    campanella.furnsh(os.path.join(spice_dir, "lsk", "naif0012.tls"))
    campanella.furnsh(os.path.join(spice_dir, "spk", "de440.bsp"))


# Same Moon-centred case as test_gve.py and test_propagate_NbodySRP.py
ET0 = 740463475.5117936
STATE0 = np.array([-7505.7613773288285, 44128.19419987408, -55792.327508873765,
                   0.07193326412942684, 0.018407794390508374, 0.004868437733535174])
GM_LIST = [4.902800582147760e+03, 398600.432896939, 132712197035.766]
MU_MOON = GM_LIST[0]
BODIES = ["MOON", "EARTH", "SUN"]
SRP_ARGS = dict(A=34.675, m=1850.0, rhoS=0.9, rhoD=0.0)
NO_SRP_ARGS = dict(A=0.0, m=1850.0, rhoS=0.9, rhoD=0.0)
TF = 20 * 86400.0

# Canonical scales
LU = 100000.0
VU = np.sqrt(MU_MOON / LU)
TU = LU / VU
GM_CANON = [gm / MU_MOON for gm in GM_LIST]
TF_CANON = TF / TU

D_CART = np.diag([LU, LU, LU, VU, VU, VU])
D_MEE = np.diag([LU, 1.0, 1.0, 1.0, 1.0, 1.0])
STATE0_CANON = STATE0 / np.diag(D_CART)


def final_row(traj, tf):
    """Final row of a trajectory, checking that the integration reached tf"""
    assert traj[-1][0] == tf, f"integration stopped at t = {traj[-1][0]}"
    return np.array(traj[-1][1:])


def stm_y0(x0):
    return list(x0) + list(np.eye(6).ravel())


def row_normalised_error(M, M_ref):
    """max over rows of max_j |M_ij - Mref_ij| / max_j |Mref_ij|"""
    return np.max(np.abs(M - M_ref).max(axis=1) / np.abs(M_ref).max(axis=1))


def cart_props(cls, **args):
    """Propagator in km/s and in canonical units"""
    km = cls(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0, **args)
    canon = cls(GM=GM_CANON, BODIES=BODIES, frame="J2000", LU=LU, TU=TU, **args)
    return km, canon


def mee0_pair():
    """Initial MEE in km and in canonical units"""
    return (np.array(campanella.cart2mee(list(STATE0), MU_MOON)),
            np.array(campanella.cart2mee(list(STATE0_CANON), 1.0)))


def test_canonical_scales():
    # TU ~ 5.2 days, so 20 days is about 3.8 TU
    assert np.isclose(TU, 451624.90737307334, rtol=1e-14)
    assert GM_CANON[0] == 1.0
    np.testing.assert_allclose(np.array(GM_CANON) * LU**3 / TU**2, GM_LIST, rtol=1e-14)
    # The canonical MEE differ from the km MEE only by p / LU
    x0_km, x0_canon = mee0_pair()
    np.testing.assert_allclose(D_MEE @ x0_canon, x0_km, rtol=0, atol=1e-12)


def test_nbody_canonical():
    """Observed: |dr| 3.3e-8 km, |dv| 2.5e-13 km/s"""
    furnsh_kernels()
    km, canon = cart_props(campanella.PropagatorNbody)
    xf_km = final_row(km.solve_ivp(ET0, [0.0, TF], list(STATE0)), TF)
    xf = D_CART @ final_row(canon.solve_ivp(ET0, [0.0, TF_CANON], list(STATE0_CANON)), TF_CANON)
    print(f"Nbody: |dr| = {np.linalg.norm(xf[:3] - xf_km[:3]):.3e} km, "
          f"|dv| = {np.linalg.norm(xf[3:] - xf_km[3:]):.3e} km/s")
    np.testing.assert_allclose(xf[:3], xf_km[:3], rtol=0, atol=1e-5)
    np.testing.assert_allclose(xf[3:], xf_km[3:], rtol=0, atol=1e-10)


def test_nbody_stm_canonical():
    """Observed: canonical vs km/s STM 1.5e-12, canonical STM vs FD 1.0e-8 (relative)"""
    furnsh_kernels()
    km, canon = cart_props(campanella.PropagatorNbody)
    yf_km = final_row(km.solve_ivp_stm(ET0, [0.0, TF], stm_y0(STATE0)), TF)
    yf = final_row(canon.solve_ivp_stm(ET0, [0.0, TF_CANON], stm_y0(STATE0_CANON)), TF_CANON)
    phi_km = yf_km[6:].reshape(6, 6)
    phi = yf[6:].reshape(6, 6)

    # The state part of the STM run matches the plain run
    xf = final_row(canon.solve_ivp(ET0, [0.0, TF_CANON], list(STATE0_CANON)), TF_CANON)
    np.testing.assert_allclose(yf[:6], xf, rtol=0, atol=1e-10)

    err_km = np.max(np.abs(D_CART @ phi @ np.linalg.inv(D_CART) - phi_km)) / np.max(np.abs(phi_km))
    print(f"Nbody STM, canonical vs km/s: {err_km:.3e}")
    assert err_km <= 1e-9

    h = 1e-6
    phi_fd = np.zeros((6, 6))
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = h
        xp = final_row(canon.solve_ivp(ET0, [0.0, TF_CANON], list(STATE0_CANON + dx)), TF_CANON)
        xm = final_row(canon.solve_ivp(ET0, [0.0, TF_CANON], list(STATE0_CANON - dx)), TF_CANON)
        phi_fd[:, j] = (xp - xm) / (2 * h)
    err_fd = np.max(np.abs(phi_fd - phi)) / np.max(np.abs(phi))
    print(f"Nbody STM, canonical vs FD: {err_fd:.3e}")
    assert err_fd <= 1e-6


def test_nbody_srp_canonical():
    """Observed: |dr| 4.5e-8 km, |dv| 4.2e-13 km/s; SRP moves the final state by ~100 km"""
    furnsh_kernels()
    km, canon = cart_props(campanella.PropagatorNbodySrp, **SRP_ARGS)
    xf_km = final_row(km.solve_ivp(ET0, [0.0, TF], list(STATE0)), TF)
    xf = D_CART @ final_row(canon.solve_ivp(ET0, [0.0, TF_CANON], list(STATE0_CANON)), TF_CANON)
    print(f"NbodySrp: |dr| = {np.linalg.norm(xf[:3] - xf_km[:3]):.3e} km, "
          f"|dv| = {np.linalg.norm(xf[3:] - xf_km[3:]):.3e} km/s")
    np.testing.assert_allclose(xf[:3], xf_km[:3], rtol=0, atol=1e-5)
    np.testing.assert_allclose(xf[3:], xf_km[3:], rtol=0, atol=1e-10)

    # SRP must actually act in canonical units, so a wrong SRP scaling cannot pass
    _, canon_nbody = cart_props(campanella.PropagatorNbody)
    xf_nbody = D_CART @ final_row(
        canon_nbody.solve_ivp(ET0, [0.0, TF_CANON], list(STATE0_CANON)), TF_CANON)
    assert np.linalg.norm(xf[:3] - xf_nbody[:3]) > 10.0


def _check_gve_canonical(srp):
    """Observed (both SRP settings): dp 2.4e-8 km, other MEE 6e-12;
    against Cartesian |dr| 1.5e-7 km, |dv| 1.5e-12 km/s"""
    furnsh_kernels()
    args = SRP_ARGS if srp else NO_SRP_ARGS
    gve_km, gve = cart_props(campanella.PropagatorGve, **args)
    x0_km, x0 = mee0_pair()
    xf_km = final_row(gve_km.solve_ivp(ET0, [0.0, TF], list(x0_km)), TF)
    xf = final_row(gve.solve_ivp(ET0, [0.0, TF_CANON], list(x0)), TF_CANON)

    dx = D_MEE @ xf - xf_km
    print(f"GVE srp={srp}, canonical vs km/s MEE: {dx}")
    assert abs(dx[0]) <= 1e-5
    np.testing.assert_allclose(dx[1:], 0.0, rtol=0, atol=1e-9)

    # Against the Cartesian propagator in the same canonical units
    cls = campanella.PropagatorNbodySrp if srp else campanella.PropagatorNbody
    _, cart = cart_props(cls, **(SRP_ARGS if srp else {}))
    rv_cart = D_CART @ final_row(cart.solve_ivp(ET0, [0.0, TF_CANON], list(STATE0_CANON)), TF_CANON)
    rv_gve = D_CART @ np.array(campanella.mee2cart(list(xf), 1.0))
    print(f"GVE srp={srp} vs Cartesian: |dr| = {np.linalg.norm(rv_gve[:3] - rv_cart[:3]):.3e} km, "
          f"|dv| = {np.linalg.norm(rv_gve[3:] - rv_cart[3:]):.3e} km/s")
    np.testing.assert_allclose(rv_gve[:3], rv_cart[:3], rtol=0, atol=1e-5)
    np.testing.assert_allclose(rv_gve[3:], rv_cart[3:], rtol=0, atol=1e-10)


def test_gve_canonical_srp():
    _check_gve_canonical(srp=True)


def test_gve_canonical_nbody():
    _check_gve_canonical(srp=False)


def test_gve_stm_canonical():
    """Observed: canonical vs km/s STM 3e-13 (relative), canonical STM vs FD 1.5e-7 (row-normalised)"""
    furnsh_kernels()
    gve_km, gve = cart_props(campanella.PropagatorGve, **SRP_ARGS)
    x0_km, x0 = mee0_pair()
    yf_km = final_row(gve_km.solve_ivp_stm(ET0, [0.0, TF], stm_y0(x0_km)), TF)
    yf = final_row(gve.solve_ivp_stm(ET0, [0.0, TF_CANON], stm_y0(x0)), TF_CANON)
    phi_km = yf_km[6:].reshape(6, 6)
    phi = yf[6:].reshape(6, 6)

    # Compare in canonical units, where the STM entries are O(1) to O(1e2)
    phi_from_km = np.linalg.inv(D_MEE) @ phi_km @ D_MEE
    err_km = np.max(np.abs(phi_from_km - phi)) / np.max(np.abs(phi))
    print(f"GVE STM, canonical vs km/s: {err_km:.3e}")
    assert err_km <= 1e-9

    h = 1e-7
    phi_fd = np.zeros((6, 6))
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = h
        xp = final_row(gve.solve_ivp(ET0, [0.0, TF_CANON], list(x0 + dx)), TF_CANON)
        xm = final_row(gve.solve_ivp(ET0, [0.0, TF_CANON], list(x0 - dx)), TF_CANON)
        phi_fd[:, j] = (xp - xm) / (2 * h)
    err_fd = row_normalised_error(phi, phi_fd)
    print(f"GVE STM, canonical vs FD: {err_fd:.3e}")
    assert err_fd <= 1e-5


def test_gve_jacobian_canonical():
    """Observed: 7.8e-10 (relative)"""
    furnsh_kernels()
    _, gve = cart_props(campanella.PropagatorGve, **SRP_ARGS)
    _, x0 = mee0_pair()
    _, A = gve.eval_jacobian(ET0, 0.0, list(x0))
    A = np.array(A).reshape(6, 6)
    h = 1e-7
    A_fd = np.zeros((6, 6))
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = h
        fp = np.array(gve.eval_jacobian(ET0, 0.0, list(x0 + dx))[0])
        fm = np.array(gve.eval_jacobian(ET0, 0.0, list(x0 - dx))[0])
        A_fd[:, j] = (fp - fm) / (2 * h)
    err = np.max(np.abs(A_fd - A)) / np.max(np.abs(A))
    print(f"GVE Jacobian, canonical vs FD: {err:.3e}")
    assert err <= 1e-6
