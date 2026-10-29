"""
Tests for the Gauss variational equations in modified equinoctial elements (MEE).

Implements the test plan of docs/gve_mee.md, section 7.
"""

import math
import os
import sys

import numpy as np
import pytest

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


# ----------------------------------------------------------------- common data (7.0)
ET0 = 740463475.5117936
STATE0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
          0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
GM_LIST = [4.902800582147760e+03, 398600.432896939, 132712197035.766]
MU_MOON = GM_LIST[0]
BODIES = ["MOON", "EARTH", "SUN"]
SRP_ARGS = dict(A=34.675, m=1850.0, rhoS=0.9, rhoD=0.0)
NO_SRP_ARGS = dict(A=0.0, m=1850.0, rhoS=0.9, rhoD=0.0)
TSPAN = [0.0, 20 * 86400.0]
X0_EXPECT = [5778.190022842265, -0.34651609849537257, 0.8514048990211349,
             1.9784209071365653, 0.6178109804659669, -1.184288478833202]
# Reference final MEE from a DOP853 run at rtol 3e-14 (docs/gve_mee.md 7.0); sanity check only
XF_SRP_DOP853 = [5178.768353783176, 0.88095964643724, -0.28930277538858457,
                 0.24069918804141774, -1.2806778961781318, 21.717565975939248]
XF_NBODY_DOP853 = [5177.094816314719, 0.8795907367139397, -0.29345826808033154,
                   0.23584006097182486, -1.2775753822790343, 21.712831372236334]
# C++ Cartesian N-body reference final state (examples/solve_NbodySTM.cpp)
RV_NBODY_EXPECT = [10171.343017143072, 33674.15294325228, -60877.5267967639,
                   0.013967106146233203, -0.0805131196626756, 0.0033270015062299776]

MU_EARTH = 398600.4418

# FD steps for MEE: dp = 1e-6 p, 1e-7 for the other elements
FD_REL_P = 1e-6
FD_STEP = 1e-7


def wrap(angle):
    """Wrap an angle difference to (-pi, pi]"""
    return math.atan2(math.sin(angle), math.cos(angle))


def make_gve(srp=True, bodies=BODIES, gm=GM_LIST):
    args = SRP_ARGS if srp else NO_SRP_ARGS
    return campanella.PropagatorGve(
        GM=gm, BODIES=bodies, frame="J2000", LU=1.0, TU=1.0, **args)


def final_row(traj, tf=TSPAN[1]):
    """Final row of a trajectory, checking that the integration reached tf"""
    assert traj[-1][0] == tf, f"integration stopped at t = {traj[-1][0]}"
    return np.array(traj[-1][1:])


def stm_y0(x0):
    return list(x0) + list(np.eye(6).ravel())


def fd_steps(x0):
    return np.array([FD_REL_P * x0[0], FD_STEP, FD_STEP, FD_STEP, FD_STEP, FD_STEP])


def row_normalised_error(M, M_ref):
    """max over rows of max_j |M_ij - Mref_ij| / max_j |Mref_ij|"""
    return np.max(np.abs(M - M_ref).max(axis=1) / np.abs(M_ref).max(axis=1))


def p_scaled(Phi, p0, pf):
    """diag(1/pf, 1, ..., 1) Phi diag(p0, 1, ..., 1)"""
    return np.diag([1 / pf, 1, 1, 1, 1, 1]) @ Phi @ np.diag([p0, 1, 1, 1, 1, 1])


def assert_mee_close(x, x_ref, tol, label=""):
    """|dp|/p, |df|, |dg|, |dh|, |dk| relative to max(1, |h|, |k|), |wrap(dL)|"""
    x = np.asarray(x)
    x_ref = np.asarray(x_ref)
    hk_scale = max(1.0, abs(x_ref[3]), abs(x_ref[4]))
    errs = np.array([
        abs(x[0] - x_ref[0]) / x_ref[0],
        abs(x[1] - x_ref[1]),
        abs(x[2] - x_ref[2]),
        abs(x[3] - x_ref[3]) / hk_scale,
        abs(x[4] - x_ref[4]) / hk_scale,
        abs(wrap(x[5] - x_ref[5])),
    ])
    assert np.all(errs <= tol), f"{label} MEE errors {errs} exceed {tol}"
    return errs.max()


# ----------------------------------------------------------------- 7.1 round trips
# name: (Cartesian state, mu, MEE from docs/gve_mee.md table; None for an entry given as ~0)
CART_CASES = {
    "LEO": ([-2103.7154800484846, 3643.742096113127, 5308.456731666615,
             -6.647330004445163, -3.833624329580397, 0.004603821856452162], MU_EARTH,
            [6777.993222, 3.42020143325783e-4, 9.3969262078571e-4,
             0.418653037730661, 0.241709444030852, 2.0943951023932]),
    "GTO": ([5878.595087575521, 3000.5849397194843, -439.2672278754225,
             -3.6936046189819858, 8.180536450190633, -4.859712456831758], MU_EARTH,
            [11373.99192144, 0.694841890805239, 0.225767816090336,
             -0.238651523310869, -0.086862050847443, 0.488692190558412]),
    "HEO": ([87047.82562847203, -3393.3994629546805, 173830.49101033443,
             0.04996504784397327, 0.3187217213914905, 0.09977789493723822], MU_EARTH,
            [9750.0, -0.95, 0.0, 0.0, -0.617612587860989, -0.0174532925199436]),
    "near-polar": ([-2237.5335584636528, 3877.384107889878, -5335.099292595186,
                    -2.88464292575727, 4.994628935389528, 4.925069859403147], MU_EARTH,
                   [6999.3, -0.00642787609686526, 0.00766044443118989,
                    -0.49991274115195, 0.865874267026206, 1.22173047639603]),
    "circular equatorial": ([7000.0, 0.0, 0.0, 0.0, 7.546053290107541, 0.0], MU_EARTH,
                            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "i=170": ([-55.46704301173019, -7246.552606392538, 896.5989572886389,
               -7.530175620232078, -1.4387377569063755, -0.7594926786100786], MU_EARTH,
              [7680.0, -0.0517638090205041, 0.193185165257814,
               8.08226749259943, 8.08226749259943, 3.14159265358979]),
    "hyperbolic": ([5775.472866359693, 7498.003270192943, 5354.4553621173945,
                    -6.119856157164796, 5.295429592561897, 5.2676006404103894], MU_EARTH,
                   [25000.0, 1.29903810567666, 0.75,
                    0.358440708571026, 0.0632027679053317, 1.0471975511966]),
    "Moon": (STATE0, MU_MOON, X0_EXPECT),
}

MEE_CASES = [
    [7000.0, 0.01, -0.02, 0.3, -0.2, math.pi - 1e-12],
    [7000.0, 0.01, -0.02, 0.3, -0.2, -math.pi + 1e-12],
    [7000.0, 0.01, -0.02, 0.3, -0.2, math.pi],
    [42164.0, 0.1, 0.05, 0.01, 0.02, 50.3],
]


@pytest.mark.parametrize("name", list(CART_CASES))
def test_cart_mee_cart_round_trip(name):
    rv, mu, mee_table = CART_CASES[name]
    rv = np.array(rv)
    mee = campanella.cart2mee(rv, mu)
    assert -math.pi < mee[5] <= math.pi
    rv_back = np.array(campanella.mee2cart(mee, mu))
    dr = np.linalg.norm(rv_back[:3] - rv[:3]) / np.linalg.norm(rv[:3])
    dv = np.linalg.norm(rv_back[3:] - rv[3:]) / np.linalg.norm(rv[3:])
    assert dr <= 1e-13 and dv <= 1e-13, f"{name}: dr = {dr}, dv = {dv}"

    # cart2mee against the MEE column of the table: 1e-12 relative on p, 1e-12 absolute otherwise
    np.testing.assert_allclose(mee[0], mee_table[0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(mee[1:5], mee_table[1:5], rtol=0, atol=1e-12)
    assert abs(wrap(mee[5] - mee_table[5])) <= 1e-12


def test_cart2mee_moon_reference():
    """Initial MEE of the Moon test case: 1e-13 relative on p, 1e-13 absolute on the rest"""
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    np.testing.assert_allclose(x0[0], X0_EXPECT[0], rtol=1e-13, atol=0)
    np.testing.assert_allclose(x0[1:], X0_EXPECT[1:], rtol=0, atol=1e-13)


@pytest.mark.parametrize("mee", MEE_CASES)
def test_mee_cart_mee_round_trip(mee):
    rv = campanella.mee2cart(mee, MU_EARTH)
    mee_back = campanella.cart2mee(rv, MU_EARTH)
    assert -math.pi < mee_back[5] <= math.pi
    assert_mee_close(mee_back, mee, 1e-13)
    if mee[5] == 50.3:
        # L is wrapped by cart2mee: 50.3 - 8 revolutions
        assert abs(mee_back[5] - (50.3 - 16 * math.pi)) <= 1e-13


def test_mee_singularities():
    with pytest.raises(ValueError):
        campanella.cart2mee([7000.0, 0.0, 0.0, 0.0, -7.5, 0.0], MU_EARTH)   # i = 180 deg
    with pytest.raises(ValueError):
        campanella.cart2mee([7000.0, 0.0, 0.0, 7.5, 0.0, 0.0], MU_EARTH)   # H = 0
    with pytest.raises(ValueError):
        campanella.mee2cart([0.0, 0.1, 0.0, 0.0, 0.0, 0.0], MU_EARTH)      # p = 0
    with pytest.raises(ValueError):
        campanella.mee2cart([-7000.0, 0.1, 0.0, 0.0, 0.0, 0.0], MU_EARTH)  # p < 0
    with pytest.raises(ValueError):
        campanella.mee2cart([7000.0, 1.5, 0.0, 0.0, 0.0, math.pi], MU_EARTH)  # w < 0


@pytest.mark.parametrize("name", ["LEO", "GTO", "i=170", "hyperbolic", "Moon"])
def test_mee2cart_jacobian_vs_fd(name):
    """Analytic d(r,v)/d(mee) (5.7) against central differences of mee2cart"""
    rv, mu, _ = CART_CASES[name]
    x = np.array(campanella.cart2mee(rv, mu))
    J = np.array(campanella.mee2cart_jacobian(x, mu)).reshape(6, 6)
    steps = fd_steps(x)
    J_fd = np.zeros((6, 6))
    for m in range(6):
        xp, xm = x.copy(), x.copy()
        xp[m] += steps[m]
        xm[m] -= steps[m]
        J_fd[:, m] = (np.array(campanella.mee2cart(xp, mu))
                      - np.array(campanella.mee2cart(xm, mu))) / (2 * steps[m])
    # column-normalised (columns have different units)
    err = np.max(np.abs(J - J_fd).max(axis=0) / np.abs(J_fd).max(axis=0))
    assert err <= 1e-6, f"{name}: column-normalised error {err}"


# ----------------------------------------------------------------- 7.2 two-body invariance
def test_twobody_invariance():
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    prop = campanella.PropagatorGve(
        GM=[MU_MOON], BODIES=["MOON"], frame="J2000", LU=1.0, TU=1.0, **NO_SRP_ARGS)
    traj = prop.solve_ivp(ET0, TSPAN, x0)
    assert traj[-1][0] == TSPAN[1]
    # with no perturbation, B a_rtn = 0 exactly: p, f, g, h, k never change
    for row in traj:
        assert list(row[1:6]) == list(x0[:5])

    rv_f = np.array(campanella.mee2cart(list(traj[-1][1:]), MU_MOON))
    rv_2b = np.array(campanella.PropagatorTwoBody(MU_MOON).solve_ivp(TSPAN, STATE0)[-1][1:])
    dr = np.linalg.norm(rv_f[:3] - rv_2b[:3])
    dv = np.linalg.norm(rv_f[3:] - rv_2b[3:])
    print(f"two-body: |dr| = {dr:.3e} km, |dv| = {dv:.3e} km/s")
    assert dr <= 1e-5 and dv <= 1e-10


# ----------------------------------------------------------------- 7.3 GVE vs Cartesian
def _compare_gve_cartesian(xf, rv_f):
    rv_gve = np.array(campanella.mee2cart(list(xf), MU_MOON))
    dr = np.linalg.norm(rv_gve[:3] - rv_f[:3])
    dv = np.linalg.norm(rv_gve[3:] - rv_f[3:])
    print(f"|dr| = {dr:.3e} km, |dv| = {dv:.3e} km/s")
    assert dr <= 1e-5, f"|dr| = {dr} km"
    assert dv <= 1e-10, f"|dv| = {dv} km/s"
    x_c = campanella.cart2mee(list(rv_f), MU_MOON)
    err = assert_mee_close(xf, x_c, 1e-9, "GVE vs cart2mee(Cartesian)")
    print(f"max MEE error = {err:.3e}")


def test_gve_vs_cartesian_srp():
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    xf = final_row(make_gve(srp=True).solve_ivp(ET0, TSPAN, x0))
    prop_cart = campanella.PropagatorNbodySrp(
        GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0, **SRP_ARGS)
    rv_f = final_row(prop_cart.solve_ivp(ET0, TSPAN, STATE0))
    _compare_gve_cartesian(xf, rv_f)
    # sanity check against the independent DOP853 transcription
    assert_mee_close(xf, XF_SRP_DOP853, 1e-8, "GVE vs DOP853 reference")


def test_gve_vs_cartesian_nbody():
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    xf = final_row(make_gve(srp=False).solve_ivp(ET0, TSPAN, x0))
    prop_cart = campanella.PropagatorNbody(
        GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0)
    rv_f = final_row(prop_cart.solve_ivp(ET0, TSPAN, STATE0))
    np.testing.assert_allclose(rv_f, RV_NBODY_EXPECT, rtol=1e-12)
    _compare_gve_cartesian(xf, rv_f)
    assert_mee_close(xf, XF_NBODY_DOP853, 1e-8, "GVE vs DOP853 reference")


# ----------------------------------------------------------------- 7.4 Jacobian vs FD of the RHS
def _jacobian_fd_error(prop, t, x):
    x = np.array(x, dtype=float)
    xdot, A = prop.eval_jacobian(ET0, t, list(x))
    A = np.array(A).reshape(6, 6)
    steps = fd_steps(x)
    A_fd = np.zeros((6, 6))
    for m in range(6):
        xp, xm = x.copy(), x.copy()
        xp[m] += steps[m]
        xm[m] -= steps[m]
        A_fd[:, m] = (np.array(prop.eval_jacobian(ET0, t, list(xp))[0])
                      - np.array(prop.eval_jacobian(ET0, t, list(xm))[0])) / (2 * steps[m])
    scale = np.diag([x[0], 1, 1, 1, 1, 1])
    return row_normalised_error(A @ scale, A_fd @ scale)


@pytest.mark.parametrize("t", [0.0, 5 * 86400.0])
@pytest.mark.parametrize("srp", [True, False])
def test_jacobian_vs_fd_moon(t, srp):
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    err = _jacobian_fd_error(make_gve(srp=srp), t, x0)
    print(f"Moon, t = {t}, srp = {srp}: row-normalised error {err:.3e}")
    assert err <= 1e-6


@pytest.mark.parametrize("name", ["LEO", "GTO", "i=170"])
def test_jacobian_vs_fd_earth(name):
    furnsh_kernels()
    rv, mu, _ = CART_CASES[name]
    x = campanella.cart2mee(rv, mu)
    prop = make_gve(srp=True, bodies=["EARTH", "MOON", "SUN"],
                    gm=[MU_EARTH, GM_LIST[0], GM_LIST[2]])
    err = _jacobian_fd_error(prop, 0.0, x)
    print(f"{name}: row-normalised error {err:.3e}")
    assert err <= 1e-6


def test_jacobian_vs_fd_srp_only():
    """SRP gradient: with the central body only, rows p..k of A are driven by SRP alone"""
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    prop = make_gve(srp=True, bodies=["MOON"], gm=[MU_MOON])
    err = _jacobian_fd_error(prop, 0.0, x0)
    print(f"SRP only: row-normalised error {err:.3e}")
    assert err <= 1e-6


# ----------------------------------------------------------------- 7.5 Phi vs FD of final states
def test_stm_vs_fd():
    furnsh_kernels()
    x0 = np.array(campanella.cart2mee(STATE0, MU_MOON))
    prop = make_gve(srp=True)
    yf = final_row(prop.solve_ivp_stm(ET0, TSPAN, stm_y0(x0)))
    xf, Phi = yf[:6], yf[6:].reshape(6, 6)

    # consistency of the state part with the 6-state run (different step sequence)
    xf_plain = final_row(prop.solve_ivp(ET0, TSPAN, list(x0)))
    err_state = assert_mee_close(xf, xf_plain, 1e-7, "STM run vs plain run")
    print(f"STM run vs plain run: {err_state:.3e}")

    steps = fd_steps(x0)
    Phi_fd = np.zeros((6, 6))
    for j in range(6):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += steps[j]
        xm[j] -= steps[j]
        Phi_fd[:, j] = (final_row(prop.solve_ivp(ET0, TSPAN, list(xp)))
                        - final_row(prop.solve_ivp(ET0, TSPAN, list(xm)))) / (2 * steps[j])
    err = row_normalised_error(p_scaled(Phi, x0[0], xf[0]), p_scaled(Phi_fd, x0[0], xf[0]))
    print(f"Phi vs FD: row-normalised error {err:.3e}")
    assert err <= 1e-4


# ----------------------------------------------------------------- 7.6 Phi_MEE vs Cartesian STM
def test_stm_vs_cartesian_stm():
    furnsh_kernels()
    x0 = np.array(campanella.cart2mee(STATE0, MU_MOON))
    yf = final_row(make_gve(srp=False).solve_ivp_stm(ET0, TSPAN, stm_y0(x0)))
    xf, Phi = yf[:6], yf[6:].reshape(6, 6)

    prop_cart = campanella.PropagatorNbody(
        GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0)
    yc = final_row(prop_cart.solve_ivp_stm(ET0, TSPAN, stm_y0(STATE0)))
    Phi_cart = yc[6:].reshape(6, 6)

    J0 = np.array(campanella.mee2cart_jacobian(list(x0), MU_MOON)).reshape(6, 6)
    Jf = np.array(campanella.mee2cart_jacobian(list(xf), MU_MOON)).reshape(6, 6)
    Phi_map = np.linalg.solve(Jf, Phi_cart @ J0)
    err = row_normalised_error(p_scaled(Phi, x0[0], xf[0]), p_scaled(Phi_map, x0[0], xf[0]))
    print(f"Phi_MEE vs mapped Cartesian STM: row-normalised error {err:.3e}")
    assert err <= 1e-6


# ----------------------------------------------------------------- scaled units, domain, arguments
# Earth-centred case in canonical units: LU = 384400 km, TU = 375190 s, GM in LU^3/TU^2
LU_EM = 384400.0
TU_EM = 375190.0
ET0_EM = 6.0e8
BODIES_EM = ["EARTH", "MOON", "SUN"]
GM_EM = [g * TU_EM**2 / LU_EM**3 for g in [398600.435436, 4902.800118, 132712440041.279]]


def mee_from_classical(a_km, e, i_deg, Om_deg, om_deg, L, LU=LU_EM):
    """Prograde MEE from classical elements, p in LU"""
    t = math.tan(math.radians(i_deg) / 2)
    Om, om = math.radians(Om_deg), math.radians(om_deg)
    return [a_km * (1 - e * e) / LU, e * math.cos(om + Om), e * math.sin(om + Om),
            t * math.cos(Om), t * math.sin(Om), L]


def test_gve_vs_cartesian_scaled_units():
    """GVE vs PropagatorNbodySrp with LU, TU != 1, et0 = 6e8, SRP on, over 5 days.

    Both integrators use abstol 1e-12 in scaled units, i.e. ~3.8e-7 km per step on position-like
    quantities (vs 1e-12 km when LU = 1), over ~100-200 steps: the budget is ~1e-5 km and
    ~1e-9 km/s (VU = LU/TU ~ 1 km/s). Observed ~1e-7 km and ~6e-12 km/s. SRP moves the final
    position by ~3 km, so the comparison is sensitive to the SRP scaling in LU, TU.
    """
    furnsh_kernels()
    x0 = mee_from_classical(60000.0, 0.2, 30.0, 30.0, 40.0, 0.5)
    tf = 5 * 86400.0 / TU_EM
    rv0 = campanella.mee2cart(x0, GM_EM[0])
    finals = {}
    for label, args in [("srp", SRP_ARGS), ("no srp", NO_SRP_ARGS)]:
        prop_gve = campanella.PropagatorGve(
            GM=GM_EM, BODIES=BODIES_EM, frame="J2000", LU=LU_EM, TU=TU_EM, **args)
        prop_cart = campanella.PropagatorNbodySrp(
            GM=GM_EM, BODIES=BODIES_EM, frame="J2000", LU=LU_EM, TU=TU_EM, **args)
        xf = final_row(prop_gve.solve_ivp(ET0_EM, [0.0, tf], x0), tf)
        rv_f = final_row(prop_cart.solve_ivp(ET0_EM, [0.0, tf], rv0), tf)
        rv_gve = np.array(campanella.mee2cart(list(xf), GM_EM[0]))
        dr_km = np.linalg.norm(rv_gve[:3] - rv_f[:3]) * LU_EM
        dv_kms = np.linalg.norm(rv_gve[3:] - rv_f[3:]) * LU_EM / TU_EM
        print(f"{label}: |dr| = {dr_km:.3e} km, |dv| = {dv_kms:.3e} km/s")
        assert dr_km <= 1e-5 and dv_kms <= 1e-9
        finals[label] = rv_f
    srp_effect_km = np.linalg.norm(finals["srp"][:3] - finals["no srp"][:3]) * LU_EM
    assert srp_effect_km > 1.0, f"SRP effect {srp_effect_km} km too small for this test"


def test_near_retrograde_reaches_tf():
    """i = 179.9 deg: a trial RK stage with w < 0 must shrink the step, not abort the run"""
    furnsh_kernels()
    x0 = mee_from_classical(60000.0, 0.2, 179.9, 30.0, 40.0, 0.5)
    prop = campanella.PropagatorGve(
        GM=GM_EM, BODIES=BODIES_EM, frame="J2000", A=50.0, m=10.0, rhoS=0.3, rhoD=0.2,
        LU=LU_EM, TU=TU_EM)
    tf = 30 * 86400.0 / TU_EM
    traj = prop.solve_ivp(ET0_EM, [0.0, tf], x0)
    assert traj[-1][0] == tf, f"integration stopped at t = {traj[-1][0]} after {len(traj)} rows"
    assert np.all(np.isfinite(traj[-1]))


def test_tspan_validation():
    furnsh_kernels()
    prop = make_gve(srp=True)
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    for bad_tspan in [[0.0], [0.0, 1.0, 2.0], [10.0, 0.0]]:
        with pytest.raises(ValueError):
            prop.solve_ivp(ET0, bad_tspan, x0)
        with pytest.raises(ValueError):
            prop.solve_ivp_stm(ET0, bad_tspan, stm_y0(x0))


def test_eval_jacobian_errors():
    furnsh_kernels()
    prop = make_gve(srp=True)
    with pytest.raises(ValueError):
        prop.eval_jacobian(ET0, 0.0, [7000.0, 1.5, 0.0, 0.0, 0.0, math.pi])   # w < 0
    with pytest.raises(ValueError):
        prop.eval_jacobian(ET0, 0.0, [-7000.0, 0.1, 0.0, 0.0, 0.0, 0.0])      # p < 0
    # SPICE errors are raised with NAIF's message
    bad = make_gve(srp=False, bodies=["MOON", "NOT_A_BODY"], gm=[MU_MOON, 1.0])
    with pytest.raises(RuntimeError, match="SPICE"):
        bad.eval_jacobian(ET0, 0.0, campanella.cart2mee(STATE0, MU_MOON))
    # the SPICE error state was reset: a valid call succeeds afterwards
    prop.eval_jacobian(ET0, 0.0, campanella.cart2mee(STATE0, MU_MOON))

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
