"""
Tests for the Gauss variational equations in modified equinoctial elements (MEE):
solve_ivp_mee, solve_ivpstm_mee, eval_mee and the MEE conversions.

Implements the test plan of docs/gve_mee.md, section 7.
"""

import math

import numpy as np
import pytest

from helpers import campanella, final_row, furnsh_kernels, row_normalised_error


# ----------------------------------------------------------------- common data (7.0)
ET0 = 740463475.5117936
STATE0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
          0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
GM_LIST = [4.902800582147760e+03, 398600.432896939, 132712197035.766]
MU_MOON = GM_LIST[0]
BODIES = ["MOON", "EARTH", "SUN"]
SRP = dict(include_srp=True, srp_A=34.675, srp_m=1850.0, srp_rhoS=0.9, srp_rhoD=0.0)
TSPAN = [0.0, 20 * 86400.0]
X0_EXPECT = [5778.190022842265, -0.34651609849537257, 0.8514048990211349,
             1.9784209071365653, 0.6178109804659669, -1.184288478833202]
# Reference final MEE from a DOP853 run at rtol 3e-14 (docs/gve_mee.md 7.0); sanity check only
XF_SRP_DOP853 = [5178.768353783176, 0.88095964643724, -0.28930277538858457,
                 0.24069918804141774, -1.2806778961781318, 21.717565975939248]
XF_NBODY_DOP853 = [5177.094816314719, 0.8795907367139397, -0.29345826808033154,
                   0.23584006097182486, -1.2775753822790343, 21.712831372236334]
# C++ Cartesian N-body reference final state (examples/solve_rv.cpp)
RV_NBODY_EXPECT = [10171.343017143072, 33674.15294325228, -60877.5267967639,
                   0.013967106146233203, -0.0805131196626756, 0.0033270015062299776]

MU_EARTH = 398600.4418

# FD steps for MEE: dp = 1e-6 p, 1e-7 for the other elements
FD_REL_P = 1e-6
FD_STEP = 1e-7


def wrap(angle):
    """Wrap an angle difference to (-pi, pi]"""
    return math.atan2(math.sin(angle), math.cos(angle))


def make_params(srp=True, bodies=BODIES, gm=GM_LIST):
    return campanella.SpiceParameters(ET0, bodies, GMs=gm, **(SRP if srp else {}))


def fd_steps(x0):
    return np.array([FD_REL_P * x0[0], FD_STEP, FD_STEP, FD_STEP, FD_STEP, FD_STEP])


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
    """No perturbation: p, f, g, h, k never change, and no ephemeris is needed"""
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    params = campanella.SpiceParameters(ET0, ["MOON"], GMs=[MU_MOON])
    res = campanella.solve_ivp_mee(params, TSPAN, x0)
    assert res.t[-1] == TSPAN[1]
    # with no perturbation, B a_rtn = 0 exactly
    for row in res.x:
        assert list(row[:5]) == list(x0[:5])

    rv_f = np.array(campanella.mee2cart(list(res.x[-1]), MU_MOON))
    rv_2b = final_row(campanella.solve_ivp_rv(params, TSPAN, STATE0), TSPAN[1])
    dr = np.linalg.norm(rv_f[:3] - rv_2b[:3])
    dv = np.linalg.norm(rv_f[3:] - rv_2b[3:])
    print(f"two-body: |dr| = {dr:.3e} km, |dv| = {dv:.3e} km/s")
    assert dr <= 1e-5 and dv <= 1e-10


# ----------------------------------------------------------------- 7.3 GVE vs Cartesian
def _compare_gve_cartesian(xf, rv_f):
    rv_mee = np.array(campanella.mee2cart(list(xf), MU_MOON))
    dr = np.linalg.norm(rv_mee[:3] - rv_f[:3])
    dv = np.linalg.norm(rv_mee[3:] - rv_f[3:])
    print(f"|dr| = {dr:.3e} km, |dv| = {dv:.3e} km/s")
    assert dr <= 1e-5, f"|dr| = {dr} km"
    assert dv <= 1e-10, f"|dv| = {dv} km/s"
    x_c = campanella.cart2mee(list(rv_f), MU_MOON)
    err = assert_mee_close(xf, x_c, 1e-9, "GVE vs cart2mee(Cartesian)")
    print(f"max MEE error = {err:.3e}")


def test_gve_vs_cartesian_srp():
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    params = make_params(srp=True)
    xf = final_row(campanella.solve_ivp_mee(params, TSPAN, x0), TSPAN[1])
    rv_f = final_row(campanella.solve_ivp_rv(params, TSPAN, STATE0), TSPAN[1])
    _compare_gve_cartesian(xf, rv_f)
    # sanity check against the independent DOP853 transcription
    assert_mee_close(xf, XF_SRP_DOP853, 1e-8, "GVE vs DOP853 reference")


def test_gve_vs_cartesian_nbody():
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    params = make_params(srp=False)
    xf = final_row(campanella.solve_ivp_mee(params, TSPAN, x0), TSPAN[1])
    rv_f = final_row(campanella.solve_ivp_rv(params, TSPAN, STATE0), TSPAN[1])
    np.testing.assert_allclose(rv_f, RV_NBODY_EXPECT, rtol=1e-12)
    _compare_gve_cartesian(xf, rv_f)
    assert_mee_close(xf, XF_NBODY_DOP853, 1e-8, "GVE vs DOP853 reference")


# ----------------------------------------------------------------- 7.4 Jacobian vs FD of the RHS
def _jacobian_fd_error(params, t, x):
    x = np.array(x, dtype=float)
    xdot, A = campanella.eval_mee(params, t, list(x))
    np.testing.assert_array_equal(campanella.eom_mee(params, t, list(x)), xdot)
    steps = fd_steps(x)
    A_fd = np.zeros((6, 6))
    for m in range(6):
        xp, xm = x.copy(), x.copy()
        xp[m] += steps[m]
        xm[m] -= steps[m]
        A_fd[:, m] = (campanella.eval_mee(params, t, list(xp))[0]
                      - campanella.eval_mee(params, t, list(xm))[0]) / (2 * steps[m])
    scale = np.diag([x[0], 1, 1, 1, 1, 1])
    return row_normalised_error(A @ scale, A_fd @ scale)


@pytest.mark.parametrize("t", [0.0, 5 * 86400.0])
@pytest.mark.parametrize("srp", [True, False])
def test_jacobian_vs_fd_moon(t, srp):
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    err = _jacobian_fd_error(make_params(srp=srp), t, x0)
    print(f"Moon, t = {t}, srp = {srp}: row-normalised error {err:.3e}")
    assert err <= 1e-6


@pytest.mark.parametrize("name", ["LEO", "GTO", "i=170"])
def test_jacobian_vs_fd_earth(name):
    furnsh_kernels()
    rv, mu, _ = CART_CASES[name]
    x = campanella.cart2mee(rv, mu)
    params = make_params(srp=True, bodies=["EARTH", "MOON", "SUN"],
                         gm=[MU_EARTH, GM_LIST[0], GM_LIST[2]])
    err = _jacobian_fd_error(params, 0.0, x)
    print(f"{name}: row-normalised error {err:.3e}")
    assert err <= 1e-6


def test_jacobian_vs_fd_srp_only():
    """SRP gradient: with a massless Sun, rows p..k of A are driven by SRP alone"""
    furnsh_kernels()
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    params = make_params(srp=True, bodies=["MOON", "SUN"], gm=[MU_MOON, 0.0])
    err = _jacobian_fd_error(params, 0.0, x0)
    print(f"SRP only: row-normalised error {err:.3e}")
    assert err <= 1e-6
    xdot, _ = campanella.eval_mee(params, 0.0, x0)
    assert np.all(np.abs(xdot[:5]) > 0)


# ----------------------------------------------------------------- 7.5 Phi vs FD of final states
def test_stm_vs_fd():
    furnsh_kernels()
    x0 = np.array(campanella.cart2mee(STATE0, MU_MOON))
    params = make_params(srp=True)
    yf = final_row(campanella.solve_ivpstm_mee(params, TSPAN, list(x0)), TSPAN[1])
    xf, Phi = yf[:6], yf[6:].reshape(6, 6)

    # consistency of the state part with the 6-state run (different step sequence)
    xf_plain = final_row(campanella.solve_ivp_mee(params, TSPAN, list(x0)), TSPAN[1])
    err_state = assert_mee_close(xf, xf_plain, 1e-7, "STM run vs plain run")
    print(f"STM run vs plain run: {err_state:.3e}")

    steps = fd_steps(x0)
    Phi_fd = np.zeros((6, 6))
    for j in range(6):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += steps[j]
        xm[j] -= steps[j]
        Phi_fd[:, j] = (final_row(campanella.solve_ivp_mee(params, TSPAN, list(xp)), TSPAN[1])
                        - final_row(campanella.solve_ivp_mee(params, TSPAN, list(xm)), TSPAN[1])) / (2 * steps[j])
    err = row_normalised_error(p_scaled(Phi, x0[0], xf[0]), p_scaled(Phi_fd, x0[0], xf[0]))
    print(f"Phi vs FD: row-normalised error {err:.3e}")
    assert err <= 1e-4


# ----------------------------------------------------------------- 7.6 Phi_MEE vs Cartesian STM
@pytest.mark.parametrize("srp", [False, True])
def test_stm_vs_cartesian_stm(srp):
    furnsh_kernels()
    params = make_params(srp=srp)
    x0 = np.array(campanella.cart2mee(STATE0, MU_MOON))
    yf = final_row(campanella.solve_ivpstm_mee(params, TSPAN, list(x0)), TSPAN[1])
    xf, Phi = yf[:6], yf[6:].reshape(6, 6)

    yc = final_row(campanella.solve_ivpstm_rv(params, TSPAN, STATE0), TSPAN[1])
    Phi_cart = yc[6:].reshape(6, 6)

    J0 = np.array(campanella.mee2cart_jacobian(list(x0), MU_MOON)).reshape(6, 6)
    Jf = np.array(campanella.mee2cart_jacobian(list(xf), MU_MOON)).reshape(6, 6)
    Phi_map = np.linalg.solve(Jf, Phi_cart @ J0)
    err = row_normalised_error(p_scaled(Phi, x0[0], xf[0]), p_scaled(Phi_map, x0[0], xf[0]))
    print(f"Phi_MEE vs mapped Cartesian STM (srp={srp}): row-normalised error {err:.3e}")
    assert err <= 1e-6


# ----------------------------------------------------------------- backward propagation, t_eval
def test_forward_backward_unwrapped_L():
    """Back to x0 after 20 days; L decreases continuously (not wrapped)"""
    furnsh_kernels()
    params = make_params(srp=True)
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    fwd = campanella.solve_ivpstm_mee(params, TSPAN, x0, atol=1e-13, rtol=1e-13)
    assert fwd.x[-1][5] > x0[5] + 2 * math.pi
    bwd = campanella.solve_ivpstm_mee(params, TSPAN[::-1], list(fwd.x[-1]), atol=1e-13, rtol=1e-13)
    assert bwd.t[-1] == 0.0
    assert np.all(np.diff(bwd.x[:, 5]) < 0)
    assert_mee_close(bwd.x[-1], x0, 1e-9, "forward-backward")
    assert abs(bwd.x[-1][5] - x0[5]) <= 1e-9   # same branch of L
    D = np.diag([x0[0], 1, 1, 1, 1, 1])
    prod = np.linalg.inv(D) @ bwd.stm[-1] @ fwd.stm[-1] @ D
    assert np.max(np.abs(prod - np.eye(6))) <= 1e-7


def test_t_eval_mee():
    furnsh_kernels()
    params = make_params(srp=True)
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    dense = campanella.solve_ivpstm_mee(params, TSPAN, x0)
    t_eval = np.linspace(0.0, TSPAN[1], 7)
    res = campanella.solve_ivpstm_mee(params, TSPAN, x0, t_eval=list(t_eval))
    np.testing.assert_array_equal(res.t, t_eval)
    assert_mee_close(res.x[-1], dense.x[-1], 1e-9, "t_eval vs dense")
    np.testing.assert_allclose(res.stm[-1], dense.stm[-1], rtol=1e-6, atol=1e-9 * np.max(np.abs(dense.stm[-1])))


# ----------------------------------------------------------------- scaled units, domain, arguments
# Earth-centred case in canonical units: DU = 384400 km, TU from the Earth GM (~3.77e5 s)
DU_EM = 384400.0
ET0_EM = 6.0e8
BODIES_EM = ["EARTH", "MOON", "SUN"]
GMS_EM = [398600.435436, 4902.800118, 132712440041.279]
SRP_EM = dict(include_srp=True, srp_A=34.675, srp_m=1850.0, srp_rhoS=0.9, srp_rhoD=0.0)


def mee_from_classical(a_km, e, i_deg, Om_deg, om_deg, L, DU=DU_EM):
    """Prograde MEE from classical elements, p in DU"""
    t = math.tan(math.radians(i_deg) / 2)
    Om, om = math.radians(Om_deg), math.radians(om_deg)
    return [a_km * (1 - e * e) / DU, e * math.cos(om + Om), e * math.sin(om + Om),
            t * math.cos(Om), t * math.sin(Om), L]


def test_gve_vs_cartesian_scaled_units():
    """GVE vs Cartesian with DU = 384400 km, et0 = 6e8, SRP on, over 5 days.

    Both integrators use atol 1e-12 in scaled units, i.e. ~3.8e-7 km per step on position-like
    quantities (vs 1e-12 km when DU = 1), over ~100-200 steps: the budget is ~1e-5 km and
    ~1e-9 km/s (VU ~ 1 km/s). SRP moves the final position by ~3 km, so the comparison is
    sensitive to the SRP scaling in DU, TU.
    """
    furnsh_kernels()
    x0 = mee_from_classical(60000.0, 0.2, 30.0, 30.0, 40.0, 0.5)
    finals = {}
    for label, args in [("srp", SRP_EM), ("no srp", {})]:
        params = campanella.SpiceParameters(ET0_EM, BODIES_EM, GMs=GMS_EM, DU=DU_EM, **args)
        assert params.mus[0] == 1.0
        tf = 5 * 86400.0 / params.TU
        rv0 = campanella.mee2cart(x0, params.mus[0])
        xf = final_row(campanella.solve_ivp_mee(params, [0.0, tf], x0), tf)
        rv_f = final_row(campanella.solve_ivp_rv(params, [0.0, tf], rv0), tf)
        rv_mee = np.array(campanella.mee2cart(list(xf), params.mus[0]))
        dr_km = np.linalg.norm(rv_mee[:3] - rv_f[:3]) * params.DU
        dv_kms = np.linalg.norm(rv_mee[3:] - rv_f[3:]) * params.VU
        print(f"{label}: |dr| = {dr_km:.3e} km, |dv| = {dv_kms:.3e} km/s")
        assert dr_km <= 1e-5 and dv_kms <= 1e-9
        finals[label] = rv_f * params.DU
    srp_effect_km = np.linalg.norm(finals["srp"][:3] - finals["no srp"][:3])
    assert srp_effect_km > 1.0, f"SRP effect {srp_effect_km} km too small for this test"


def test_near_retrograde_reaches_tf():
    """i = 179.9 deg: a trial RK stage with w < 0 must shrink the step, not abort the run"""
    furnsh_kernels()
    x0 = mee_from_classical(60000.0, 0.2, 179.9, 30.0, 40.0, 0.5)
    params = campanella.SpiceParameters(ET0_EM, BODIES_EM, GMs=GMS_EM, DU=DU_EM, include_srp=True,
                                        srp_A=50.0, srp_m=10.0, srp_rhoS=0.3, srp_rhoD=0.2)
    tf = 30 * 86400.0 / params.TU
    res = campanella.solve_ivp_mee(params, [0.0, tf], x0)
    assert res.t[-1] == tf, f"integration stopped at t = {res.t[-1]} after {len(res)} rows"
    assert np.all(np.isfinite(res.x[-1]))


def test_out_of_domain_raises():
    """A trajectory that stays out of the MEE domain raises ValueError, not a truncated result"""
    params = campanella.SpiceParameters(0.0, ["EARTH"], GMs=[MU_EARTH])
    with pytest.raises(ValueError, match="MEE out of domain"):
        campanella.solve_ivp_mee(params, [0.0, 60.0], [-7000.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="MEE out of domain"):
        campanella.solve_ivpstm_mee(params, [0.0, 60.0], [7000.0, 1.5, 0.0, 0.0, 0.0, math.pi])


def test_tspan_validation():
    furnsh_kernels()
    params = make_params(srp=True)
    x0 = campanella.cart2mee(STATE0, MU_MOON)
    for bad_tspan in [[0.0], [0.0, 1.0, 2.0], [0.0, float("nan")]]:
        with pytest.raises(ValueError):
            campanella.solve_ivp_mee(params, bad_tspan, x0)
        with pytest.raises(ValueError):
            campanella.solve_ivpstm_mee(params, bad_tspan, x0)


def test_eval_mee_errors():
    furnsh_kernels()
    params = make_params(srp=True)
    with pytest.raises(ValueError):
        campanella.eval_mee(params, 0.0, [7000.0, 1.5, 0.0, 0.0, 0.0, math.pi])   # w < 0
    with pytest.raises(ValueError):
        campanella.eval_mee(params, 0.0, [-7000.0, 0.1, 0.0, 0.0, 0.0, 0.0])      # p < 0
    with pytest.raises(ValueError):
        campanella.eval_mee(params, 0.0, [7000.0, 0.1, 0.0])                      # length
    # SPICE errors are raised with NAIF's message
    bad = make_params(srp=False, bodies=["MOON", "PLUTO"], gm=[MU_MOON, 869.6])
    with pytest.raises(RuntimeError, match="SPICE"):
        campanella.eval_mee(bad, 0.0, campanella.cart2mee(STATE0, MU_MOON))
    # the SPICE error state was reset: a valid call succeeds afterwards
    campanella.eval_mee(params, 0.0, campanella.cart2mee(STATE0, MU_MOON))


def test_eomstm_mee_matches_eval():
    furnsh_kernels()
    params = make_params(srp=True)
    x = campanella.cart2mee(STATE0, MU_MOON)
    xdot, A = campanella.eval_mee(params, 100.0, x)
    ydot = campanella.eomstm_mee(params, 100.0, list(x) + list(np.eye(6).ravel()))
    np.testing.assert_array_equal(ydot[:6], xdot)
    np.testing.assert_array_equal(ydot[6:].reshape(6, 6), A)
