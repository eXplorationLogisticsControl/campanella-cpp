"""
Tests of the Python bindings of the mass-augmented dynamics under a zero-order-hold control (rvm, meem):
ThrustOptions, ThrustFrame, solve_ivp[stm]_{rvm,meem}, eval_{rvm,meem}, eom[stm]_{rvm,meem}, IvpResult,
propagate_zoh_{rvm,meem} and the limit max_steps.

Errors of the sensitivities use helpers.matrix_error, with rows and columns both checked: a row metric
alone hides a weak column and vice versa. Run with -s to see the measured values.
"""

import gc
import math
import signal
import time

import numpy as np
import pytest

from helpers import campanella, furnsh_kernels, jacobian_fd_n, matrix_error, solve_linear

ET0 = 740463475.5117936
STATE0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
          0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
GM_LIST = [4.902800582147760e+03, 398600.432896939, 132712197035.766]
MU_MOON = GM_LIST[0]
BODIES = ["MOON", "EARTH", "SUN"]
MASS_UNIT = 1850.0
SRP = dict(include_srp=True, srp_A=34.675, srp_m=MASS_UNIT, srp_rhoS=0.9, srp_rhoD=0.0)
DAY = 86400.0
TF = 5 * DAY
TAU = [0.3, -0.4, 0.5]
GAMMA = math.sqrt(0.5)     # |TAU|
U = TAU + [GAMMA]
# Weak next to gravity at the start of STATE0 (mu/r^2 = 9.6e-7 km/s^2). At 2 N this orbit is driven to
# p -> 0 in about 1.6 days, where the MEE steps collapse until max_steps stops solve_ivp_meem.
THRUST_N = 0.05
ISP = 2000.0
G0 = 9.80665
C1 = THRUST_N * 1e-3 / MASS_UNIT
C2 = THRUST_N / (ISP * G0) / MASS_UNIT
HU = np.array([1e-4, 1e-4, 1e-4, 1e-2])

# Tolerances, with the largest measured value (Linux x86-64)
TOL_PROPAGATED_FD = 1e-4    # 9.6e-7, the tolerance of issue #5
TOL_EVAL_FD = 1e-6          # 9.5e-9, the tolerance of issue #5
TOL_MASS = 1e-13            # 4.0e-14, the tolerance of issue #5
TOL_POINTWISE = 1e-14       # 2.0e-15, the tolerance of issue #5
TOL_FREE_DV = 4e-14         # 3.7e-15
TOL_FREE_PSI = 6e-12        # 5.5e-13
TOL_SRP = 2e-11             # 1.7e-12


def note(what, value):
    print(f"\n  {what}: {value:.3e}", end="")


def thruster(frame=campanella.ThrustFrame.rtn, thrust_N=THRUST_N, mass_unit_kg=MASS_UNIT):
    return campanella.ThrustOptions(thrust_N=thrust_N, isp_s=ISP, mass_unit_kg=mass_unit_kg,
                                    mee_control_frame=frame)


X0_RVM = STATE0 + [1.0]
X0_MEEM = campanella.cart2mee(STATE0, MU_MOON) + [1.0]


class Dyn:
    """One of the dynamics: its functions, thrust, initial state, FD steps and scales"""

    def __init__(self, name, suffix, frame, x0):
        self.name = name
        self.solve = getattr(campanella, "solve_ivp_" + suffix)
        self.solvestm = getattr(campanella, "solve_ivpstm_" + suffix)
        self.eval = getattr(campanella, "eval_" + suffix)
        self.eom = getattr(campanella, "eom_" + suffix)
        self.eomstm = getattr(campanella, "eomstm_" + suffix)
        self.propagate = getattr(campanella, "propagate_zoh_" + suffix)
        self.thrust = thruster(frame)
        self.x0 = x0
        self.is_mee = suffix == "meem"

    def hx(self, x):
        """Central-difference steps of the state: dm = 1e-5 m, and dp = 1e-6 p for MEE"""
        if self.is_mee:
            return np.array([1e-6 * x[0]] + [1e-7] * 5 + [1e-5 * x[6]])
        return np.array([1e-2] * 3 + [1e-7] * 3 + [1e-5 * x[6]])

    def scale(self, x):
        """Scales of the state: s_p = p for MEE, |r| and |v| for Cartesian, 1 otherwise"""
        if self.is_mee:
            return np.array([x[0]] + [1.0] * 6)
        return np.array([np.linalg.norm(x[:3])] * 3 + [np.linalg.norm(x[3:6])] * 3 + [1.0])


DYNS = [Dyn("rvm", "rvm", campanella.ThrustFrame.inertial, X0_RVM),
        Dyn("meem_rtn", "meem", campanella.ThrustFrame.rtn, X0_MEEM),
        Dyn("meem_inertial", "meem", campanella.ThrustFrame.inertial, X0_MEEM)]
IDS = [d.name for d in DYNS]


@pytest.fixture(scope="module", autouse=True)
def kernels():
    furnsh_kernels()


@pytest.fixture(scope="module")
def params():
    return campanella.SpiceParameters(ET0, BODIES, GMs=GM_LIST, **SRP)


@pytest.fixture(scope="module")
def params_nosrp():
    return campanella.SpiceParameters(ET0, BODIES, GMs=GM_LIST)


@pytest.fixture(scope="module")
def two_body():
    """Central body only, no kernels needed"""
    return campanella.SpiceParameters(ET0, ["MOON"], GMs=[MU_MOON])


def both_metrics(M, R, rows, cols, row_scale, col_scale, what, tol, abs_floor=0.0):
    """Row and column errors, both required to be <= tol"""
    err_row = matrix_error(M, R, rows, cols, row_scale, col_scale, abs_floor, "row")
    err_col = matrix_error(M, R, rows, cols, row_scale, col_scale, abs_floor, "col")
    note(f"{what} row", err_row)
    note(f"{what} col", err_col)
    assert err_row <= tol, f"{what}: row error {err_row:.3e}"
    assert err_col <= tol, f"{what}: col error {err_col:.3e}"


# ---------------------------------------------------------------- shapes and views

@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_shapes_state_only(params, d):
    r = d.solve(params, d.thrust, [0.0, DAY], d.x0, U)
    n = len(r)
    assert n >= 2
    assert r.t.shape == (n,)
    assert r.x.shape == (n, 7)
    assert r.stm is None
    assert r.stm_u is None


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_shapes_with_sensitivities(params, d):
    r = d.solvestm(params, d.thrust, [0.0, DAY], d.x0, U)
    n = len(r)
    assert r.t.shape == (n,)
    assert r.x.shape == (n, 7)
    assert r.stm.shape == (n, 7, 7)
    assert r.stm_u.shape == (n, 7, 4)
    # initial conditions, which also pin down the layout
    assert np.array_equal(r.stm[0], np.eye(7))
    assert np.array_equal(r.stm_u[0], np.zeros((7, 4)))
    assert np.array_equal(r.x[0], d.x0)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_t_eval_and_zero_length_span(params, d):
    t_eval = [0.0, 0.5 * DAY, DAY]
    r = d.solvestm(params, d.thrust, [0.0, DAY], d.x0, U, t_eval=t_eval, atol=1e-11, rtol=1e-13)
    assert np.array_equal(r.t, t_eval)
    assert r.x.shape == (3, 7) and r.stm.shape == (3, 7, 7) and r.stm_u.shape == (3, 7, 4)
    r = d.solvestm(params, d.thrust, [1.0, 1.0], d.x0, U)
    assert r.x.shape == (1, 7) and r.stm.shape == (1, 7, 7) and r.stm_u.shape == (1, 7, 4)


def test_existing_shapes_unchanged(params):
    x0_mee = X0_MEEM[:6]
    for solve, x0 in ((campanella.solve_ivp_rv, STATE0), (campanella.solve_ivp_mee, x0_mee)):
        r = solve(params, [0.0, 3600.0], x0)
        assert r.t.shape == (len(r),)
        assert r.x.shape == (len(r), 6)
        assert r.stm is None
        assert r.stm_u is None
    for solve, x0 in ((campanella.solve_ivpstm_rv, STATE0), (campanella.solve_ivpstm_mee, x0_mee)):
        r = solve(params, [0.0, 3600.0], x0)
        assert r.x.shape == (len(r), 6)
        assert r.stm.shape == (len(r), 6, 6)
        assert r.stm_u is None
    for evaluate, x0 in ((campanella.eval_rv, STATE0), (campanella.eval_mee, x0_mee)):
        out = evaluate(params, 0.0, x0)
        assert len(out) == 2
        assert out[0].shape == (6,) and out[1].shape == (6, 6)
    assert campanella.eom_rv(params, 0.0, STATE0).shape == (6,)
    assert campanella.eomstm_mee(params, 0.0, x0_mee + list(np.eye(6).ravel())).shape == (42,)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_views_are_read_only_and_keep_the_result_alive(params, d):
    r = d.solvestm(params, d.thrust, [0.0, 3600.0], d.x0, U)
    views = [r.t, r.x, r.stm, r.stm_u]
    copies = [np.array(v) for v in views]
    for v in views:
        assert not v.flags.writeable
        assert not v.flags.owndata
        with pytest.raises(ValueError):
            v.flat[0] = 1.0
    del r
    gc.collect()
    for v, c in zip(views, copies):
        assert np.array_equal(v, c)


# ---------------------------------------------------------------- ThrustOptions

def test_thrust_options_attributes():
    o = campanella.ThrustOptions(thrust_N=0.5, isp_s=3000.0, mass_unit_kg=1000.0)
    assert (o.thrust_N, o.isp_s, o.mass_unit_kg) == (0.5, 3000.0, 1000.0)
    assert o.g0 == G0
    assert o.mee_control_frame == campanella.ThrustFrame.rtn
    o = campanella.ThrustOptions(thrust_N=0.5, isp_s=3000.0, mass_unit_kg=1000.0, g0=9.8,
                                 mee_control_frame=campanella.ThrustFrame.inertial)
    assert o.g0 == 9.8
    assert o.mee_control_frame == campanella.ThrustFrame.inertial
    assert "ThrustFrame.inertial" in repr(o) and "thrust_N=0.5" in repr(o)
    with pytest.raises(AttributeError):
        o.thrust_N = 1.0


def test_thrust_options_signature():
    with pytest.raises(TypeError):
        campanella.ThrustOptions()
    with pytest.raises(TypeError):
        campanella.ThrustOptions(thrust_N=0.5, isp_s=3000.0)     # no mass unit
    with pytest.raises(TypeError):
        campanella.ThrustOptions(0.5, 3000.0, 1000.0)            # keywords only
    with pytest.raises(TypeError):
        campanella.ThrustOptions(thrust_N=0.5, isp_s=3000.0, mass_unit_kg=1000.0, mee_control_frame="rtn")


@pytest.mark.parametrize("field", ["thrust_N", "isp_s", "mass_unit_kg", "g0"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_thrust_options_bad_values(params, d, field, value):
    kw = dict(thrust_N=1.0, isp_s=1000.0, mass_unit_kg=1000.0, g0=G0)
    kw[field] = value
    bad = campanella.ThrustOptions(**kw)    # checked where it is used
    y0 = list(d.x0) + list(np.eye(7).ravel()) + [0.0] * 28
    for call in (lambda: d.solve(params, bad, [0.0, 1.0], d.x0, U),
                 lambda: d.solvestm(params, bad, [0.0, 1.0], d.x0, U),
                 lambda: d.eval(params, bad, 0.0, d.x0, U),
                 lambda: d.eom(params, bad, 0.0, d.x0, U),
                 lambda: d.eomstm(params, bad, 0.0, y0, U)):
        with pytest.raises(ValueError, match=f"{field} must be finite and > 0"):
            call()


def test_thrust_constants_overflow(params):
    d = DYNS[0]
    th = campanella.ThrustOptions(thrust_N=1e308, isp_s=1e-300, mass_unit_kg=1e-300)
    with pytest.raises(ValueError, match="c1|c2"):
        d.eval(params, th, 0.0, d.x0, U)


def test_errors_name_the_python_function(params):
    d = DYNS[0]
    bad = campanella.ThrustOptions(thrust_N=-1.0, isp_s=1000.0, mass_unit_kg=1000.0)
    with pytest.raises(ValueError, match="^eom_rvm: thrust_N"):
        d.eom(params, bad, 0.0, d.x0, U)
    with pytest.raises(ValueError, match="^eom_rvm: u must have length 4"):
        d.eom(params, d.thrust, 0.0, d.x0, U[:3])


def test_controlled_parameters_is_not_bound():
    assert not hasattr(campanella, "ControlledParameters")


# ---------------------------------------------------------------- errors

@pytest.mark.parametrize("m0", [0.0, -1.0])
@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_non_positive_mass(params, d, m0):
    x = list(d.x0[:6]) + [m0]
    y = x + list(np.eye(7).ravel()) + [0.0] * 28
    for tspan in ([0.0, 60.0], [0.0, -60.0], [0.0, 0.0]):
        for solve in (d.solve, d.solvestm):
            with pytest.raises(ValueError, match="mass out of domain"):
                solve(params, d.thrust, tspan, x, U)
    with pytest.raises(ValueError, match="mass out of domain"):
        d.eval(params, d.thrust, 0.0, x, U)
    with pytest.raises(ValueError, match="mass out of domain"):
        d.eom(params, d.thrust, 0.0, x, U)
    with pytest.raises(ValueError, match="mass out of domain"):
        d.eomstm(params, d.thrust, 0.0, y, U)
    # the scratch space of params does not poison a later valid call
    assert np.all(np.isfinite(d.eom(params, d.thrust, 0.0, d.x0, U)))


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_non_finite_mass_in_eval(params, d):
    x = list(d.x0[:6])
    with pytest.raises(ValueError, match="mass out of domain"):
        d.eval(params, d.thrust, 0.0, x + [math.nan], U)
    with pytest.raises(ValueError, match="non-finite"):
        d.eval(params, d.thrust, 0.0, x + [math.inf], U)
    for m in (math.nan, math.inf):
        with pytest.raises(ValueError, match="x0 must be finite"):
            d.solve(params, d.thrust, [0.0, 1.0], x + [m], U)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_depletion(params, d):
    t_empty = 1.0 / (C2 * GAMMA)    # m0 = 1
    assert t_empty > TF
    tf = 1.5 * t_empty
    for solve in (d.solve, d.solvestm):
        with pytest.raises(ValueError, match="mass out of domain") as e:
            solve(params, d.thrust, [0.0, tf], d.x0, U)
        assert f"{t_empty:.10g}"[:8] in str(e.value)    # with the depletion time
        with pytest.raises(ValueError, match="mass out of domain"):
            solve(params, d.thrust, [0.0, tf], d.x0, U, t_eval=[tf])


def test_depletion_is_about_the_last_output_time(two_body):
    # the mass runs out after tf, or only after the last of t_eval, or not at all going backward
    d = DYNS[0]
    th = thruster(thrust_N=1000.0)
    c2 = 1000.0 / (ISP * G0) / MASS_UNIT
    t_empty = 1.0 / c2     # Gamma = 1
    u = [0.0, 0.0, 0.0, 1.0]
    r = d.solve(two_body, th, [0.0, 2 * t_empty], d.x0, u, t_eval=[0.5 * t_empty])
    assert r.x[-1, 6] == pytest.approx(0.5, rel=1e-13)
    r = d.solve(two_body, th, [0.0, -t_empty], d.x0, u, t_eval=[-t_empty])
    assert r.x[-1, 6] == pytest.approx(2.0, rel=1e-13)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_negative_gamma_and_big_tau_are_not_rejected(two_body, d):
    r = d.solve(two_body, d.thrust, [0.0, DAY], d.x0, [1.0, 0.0, 0.0, -0.1])
    assert r.x[-1, 6] == pytest.approx(1.0 + 0.1 * C2 * DAY, rel=1e-13)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_lengths(params, d):
    y = list(d.x0) + list(np.eye(7).ravel()) + [0.0] * 28
    for solve in (d.solve, d.solvestm):
        with pytest.raises(ValueError, match="x0 must have length 7"):
            solve(params, d.thrust, [0.0, 1.0], d.x0[:6], U)
        with pytest.raises(ValueError, match="x0 must have length 7"):
            solve(params, d.thrust, [0.0, 1.0], list(d.x0) + [0.0], U)
        for u in (U[:3], U + [0.0], []):
            with pytest.raises(ValueError, match="u must have length 4"):
                solve(params, d.thrust, [0.0, 1.0], d.x0, u)
        with pytest.raises(ValueError, match="u must be finite"):
            solve(params, d.thrust, [0.0, 1.0], d.x0, [0.0, 0.0, math.nan, 0.0])
    with pytest.raises(ValueError, match="x must have length 7"):
        d.eval(params, d.thrust, 0.0, d.x0[:6], U)
    with pytest.raises(ValueError, match="x must have length 7"):
        d.eom(params, d.thrust, 0.0, d.x0[:6], U)
    with pytest.raises(ValueError, match="x must have length 84"):
        d.eomstm(params, d.thrust, 0.0, d.x0, U)
    for call in (lambda u: d.eval(params, d.thrust, 0.0, d.x0, u),
                 lambda u: d.eom(params, d.thrust, 0.0, d.x0, u),
                 lambda u: d.eomstm(params, d.thrust, 0.0, y, u)):
        with pytest.raises(ValueError, match="u must have length 4"):
            call(U[:3])
        with pytest.raises(ValueError, match="u must be finite"):
            call([math.inf, 0.0, 0.0, 0.0])


def test_mee_domain(params):
    d = DYNS[1]
    x = [-1.0] + list(d.x0[1:])
    with pytest.raises(ValueError, match="MEE out of domain"):
        d.eval(params, d.thrust, 0.0, x, U)
    with pytest.raises(ValueError, match="MEE out of domain"):
        d.eom(params, d.thrust, 0.0, x, U)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_spice_error_is_runtime_error(params, d):
    t = 1e13    # outside of the ephemerides
    y = list(d.x0) + list(np.eye(7).ravel()) + [0.0] * 28
    for call in (lambda: d.eval(params, d.thrust, t, d.x0, U),
                 lambda: d.eom(params, d.thrust, t, d.x0, U),
                 lambda: d.eomstm(params, d.thrust, t, y, U),
                 lambda: d.solve(params, d.thrust, [t, t + 1.0], d.x0, U)):
        with pytest.raises(RuntimeError, match="SPICE"):
            call()


# ---------------------------------------------------------------- mass

@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_mass_closed_form(params, d):
    r = d.solvestm(params, d.thrust, [0.0, TF], d.x0, U)
    assert r.t[-1] == TF
    err = np.max(np.abs(r.x[:, 6] - (1.0 - C2 * GAMMA * r.t)) / r.x[:, 6])
    note(f"{d.name} mass closed form", err)
    assert err <= TOL_MASS
    # the mass row of Phi is e7 and that of Psi is [0, 0, 0, -c2 t]; the tau and state terms are exact
    assert np.all(r.stm[:, 6, :6] == 0.0) and np.all(r.stm[:, 6, 6] == 1.0)
    assert np.all(r.stm_u[:, 6, :3] == 0.0)
    err = np.max(np.abs(r.stm_u[1:, 6, 3] + C2 * r.t[1:]) / (C2 * r.t[1:]))
    note(f"{d.name} Psi mass row", err)
    assert err <= TOL_MASS


def test_tsiolkovsky_and_gamma_sensitivity_in_free_space():
    # A negligible central body: v(t) = v0 + (c1 tau / k) ln(m0 / m), with k = c2 Gamma, m = m0 - k t.
    # dv/dGamma = int_0^t c1 tau c2 s / m(s)^2 ds = c1 c2 tau (ln(m/m0) + m0/m - 1) / k^2. This grows as
    # c1 c2 tau t^2 / (2 m0^2) for a short t.
    free = campanella.SpiceParameters(ET0, ["MOON"], GMs=[1e-30])
    # a strong thruster, as the closed form below loses digits to cancellation for a small mass change
    thrust_N = 2.0
    th = thruster(thrust_N=thrust_N)
    c1 = thrust_N * 1e-3 / MASS_UNIT
    c2 = thrust_N / (ISP * G0) / MASS_UNIT
    x0 = [1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    tau = np.array(TAU)
    k = c2 * GAMMA
    r = campanella.solve_ivpstm_rvm(free, th, [0.0, TF], x0, U)
    m = 1.0 - k * TF
    dv = c1 * tau / k * math.log(1.0 / m)
    err = np.linalg.norm(r.x[-1, 3:6] - dv) / np.linalg.norm(dv)
    note("Tsiolkovsky dv", err)
    assert err <= TOL_FREE_DV
    assert r.x[-1, 6] == pytest.approx(m, rel=TOL_MASS)
    psi_ref = c1 * c2 * tau * (math.log(m) + 1.0 / m - 1.0) / k**2
    err = np.linalg.norm(r.stm_u[-1, 3:6, 3] - psi_ref) / np.linalg.norm(psi_ref)
    note("Psi_v,Gamma closed form", err)
    assert err <= TOL_FREE_PSI
    short = c1 * c2 * tau * TF**2 / 2.0
    assert np.linalg.norm(psi_ref - short) <= 0.05 * np.linalg.norm(short)


# ---------------------------------------------------------------- sensitivities

@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_phi_vs_fd(params, d):
    r = d.solvestm(params, d.thrust, [0.0, TF], d.x0, U)
    xf = r.x[-1]
    Phi = r.stm[-1]
    Phi_fd = jacobian_fd_n(lambda x: d.solve(params, d.thrust, [0.0, TF], x, U).x[-1], d.x0, d.hx(d.x0))
    both_metrics(Phi, Phi_fd, 7, 7, d.scale(xf), d.scale(d.x0), f"{d.name} Phi", TOL_PROPAGATED_FD)
    # the mass column comes only from the thrust and SRP
    assert np.linalg.norm(Phi[:6, 6]) > 0.0


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_psi_vs_fd(params, d):
    r = d.solvestm(params, d.thrust, [0.0, TF], d.x0, U)
    xf = r.x[-1]
    Psi = r.stm_u[-1]
    Psi_fd = jacobian_fd_n(lambda u: d.solve(params, d.thrust, [0.0, TF], d.x0, u).x[-1], U, HU)
    both_metrics(Psi, Psi_fd, 7, 4, d.scale(xf), None, f"{d.name} Psi", TOL_PROPAGATED_FD)
    # the mass row dominates the Gamma column, so that the column metric above cannot see the
    # (r, v, elements) rows of Gamma: they get a metric of their own
    err = matrix_error(Psi[:6, [3]], Psi_fd[:6, [3]], 6, 1, d.scale(xf)[:6], None, 0.0, "col")
    note(f"{d.name} Psi non-mass Gamma col", err)
    assert err <= TOL_PROPAGATED_FD
    # and that is not trivially satisfied
    assert np.all(np.abs(Psi[:6, 3]) > 0.0)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_metric_sees_a_corrupted_entry(params, d):
    # negative control of the FD comparison: it must fail if the layout of Psi were wrong
    r = d.solvestm(params, d.thrust, [0.0, TF], d.x0, U)
    Psi = r.stm_u[-1]
    Psi_fd = jacobian_fd_n(lambda u: d.solve(params, d.thrust, [0.0, TF], d.x0, u).x[-1], U, HU)
    s = d.scale(r.x[-1])
    assert matrix_error(Psi[:, [1, 0, 2, 3]], Psi_fd, 7, 4, s, None, 0.0, "row") > 1e-2
    assert matrix_error(Psi[:, [1, 0, 2, 3]], Psi_fd, 7, 4, s, None, 0.0, "col") > 1e-2


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_psi_v_gamma_nonzero_under_thrust(params_nosrp, d):
    # Psi_(1:6),Gamma vanishes without SRP and without thrust only: Gamma changes the mass, and with it the
    # thrust acceleration c1 tau / m
    with_thrust = d.solvestm(params_nosrp, d.thrust, [0.0, DAY], d.x0, U).stm_u[-1]
    without = d.solvestm(params_nosrp, d.thrust, [0.0, DAY], d.x0, [0.0, 0.0, 0.0, GAMMA]).stm_u[-1]
    assert np.max(np.abs(with_thrust[:6, 3])) > 0.0
    assert np.array_equal(without[:6, 3], np.zeros(6))


def test_psi_gamma_nonzero_with_srp_and_no_thrust(params):
    d = DYNS[0]
    Psi = d.solvestm(params, d.thrust, [0.0, DAY], d.x0, [0.0, 0.0, 0.0, GAMMA]).stm_u[-1]
    assert np.max(np.abs(Psi[3:6, 3])) > 0.0


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_eval_jacobians_vs_fd(params, d):
    for t in (0.0, TF):
        for x in (d.x0, list(d.x0[:6]) + [0.7]):
            xdot, A, B = d.eval(params, d.thrust, t, x, U)
            assert xdot.shape == (7,) and A.shape == (7, 7) and B.shape == (7, 4)
            s = d.scale(x)
            A_fd = jacobian_fd_n(lambda y: d.eval(params, d.thrust, t, y, U)[0], x, d.hx(x))
            B_fd = jacobian_fd_n(lambda u: d.eval(params, d.thrust, t, x, u)[0], U, HU)
            both_metrics(A, A_fd, 7, 7, s, s, f"{d.name} A", TOL_EVAL_FD, abs_floor=1e-300)
            both_metrics(B, B_fd, 7, 4, s, None, f"{d.name} B", TOL_EVAL_FD, abs_floor=1e-300)


# ---------------------------------------------------------------- RTN and inertial

def rtn_basis(mee):
    """C = [rhat that nhat], from mee2cart"""
    rv = np.array(campanella.mee2cart(list(mee), MU_MOON))
    rhat = rv[:3] / np.linalg.norm(rv[:3])
    nhat = np.cross(rv[:3], rv[3:])
    nhat /= np.linalg.norm(nhat)
    return np.column_stack((rhat, np.cross(nhat, rhat), nhat))


def test_inertial_and_rtn_agree_at_a_point(params):
    rtn, inertial = DYNS[1], DYNS[2]
    x = X0_MEEM
    C = rtn_basis(x[:6])
    tau_rtn = C.T @ np.array(TAU)
    xdot_i, A_i, B_i = inertial.eval(params, inertial.thrust, 0.0, x, U)
    xdot_r, A_r, B_r = rtn.eval(params, rtn.thrust, 0.0, x, list(tau_rtn) + [GAMMA])
    s = rtn.scale(x)
    err = np.max(np.abs(xdot_i - xdot_r) / np.maximum(np.abs(xdot_i), 1e-300))
    note("xdot inertial vs rtn", err)
    assert err <= TOL_POINTWISE
    err = matrix_error(B_i[:, :3], B_r[:, :3] @ C.T, 7, 3, s, None, 1e-300, "row")
    note("B inertial vs rtn C^T", err)
    assert err <= TOL_POINTWISE
    # the mass column is the same; A differs, as C^T depends on the state only in the inertial frame
    err = np.max(np.abs(A_i[:, 6] - A_r[:, 6]) / np.maximum(np.abs(A_i[:, 6]), 1e-300))
    note("A mass column inertial vs rtn", err)
    assert err <= TOL_POINTWISE
    assert not np.array_equal(A_i[:, :6], A_r[:, :6])


def test_rvm_ignores_the_frame(params):
    d = DYNS[0]
    r = d.solve(params, thruster(campanella.ThrustFrame.rtn), [0.0, DAY], d.x0, U)
    i = d.solve(params, thruster(campanella.ThrustFrame.inertial), [0.0, DAY], d.x0, U)
    assert np.array_equal(r.x, i.x)


def test_rtn_constant_tau_is_not_inertial_constant_tau(params):
    rtn, inertial = DYNS[1], DYNS[2]
    a = rtn.solve(params, rtn.thrust, [0.0, DAY], rtn.x0, U).x[-1]
    b = inertial.solve(params, inertial.thrust, [0.0, DAY], inertial.x0, U).x[-1]
    assert np.max(np.abs(a[:6] - b[:6]) / rtn.scale(a)[:6]) > 1e-6


def test_inertial_meem_propagates_as_rvm(params):
    # same inertial tau: same trajectory, states and sensitivities (through the Jacobian of mee2cart)
    rvm, meem = DYNS[0], DYNS[2]
    a = rvm.solvestm(params, rvm.thrust, [0.0, DAY], X0_RVM, U)
    b = meem.solvestm(params, meem.thrust, [0.0, DAY], X0_MEEM, U)
    xf_rvm, xf_meem = a.x[-1], b.x[-1]
    rv = np.array(campanella.mee2cart(list(xf_meem[:6]), MU_MOON))
    dr = np.linalg.norm(rv[:3] - xf_rvm[:3])
    dv = np.linalg.norm(rv[3:] - xf_rvm[3:6])
    note("rvm vs inertial meem |dr| km", dr)
    note("rvm vs inertial meem |dv| km/s", dv)
    assert dr <= 1e-5 and dv <= 1e-10
    assert xf_rvm[6] == pytest.approx(xf_meem[6], rel=TOL_MASS)

    def J_bar(mee):
        J = np.eye(7)
        J[:6, :6] = np.array(campanella.mee2cart_jacobian(list(mee), MU_MOON)).reshape(6, 6)
        return J
    J0, Jf = J_bar(X0_MEEM[:6]), J_bar(xf_meem[:6])
    Phi_map = solve_linear(Jf, a.stm[-1] @ J0)
    Psi_map = solve_linear(Jf, a.stm_u[-1])
    s0, sf = meem.scale(X0_MEEM), meem.scale(xf_meem)
    both_metrics(Phi_map, b.stm[-1], 7, 7, sf, s0, "Phi mapped", 1e-6)
    both_metrics(Psi_map, b.stm_u[-1], 7, 4, sf, None, "Psi mapped", 1e-6)


# ---------------------------------------------------------------- eom against eval

@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_eom_agrees_with_eval(params, d):
    for t in (0.0, DAY):
        xdot, A, B = d.eval(params, d.thrust, t, d.x0, U)
        out = d.eom(params, d.thrust, t, d.x0, U)
        assert out.shape == (7,)
        assert np.array_equal(out, xdot)
        # an arbitrary Phi and Psi, as dot products of the full A and B are what is checked
        rng = np.random.default_rng(1)
        Phi, Psi = rng.normal(size=(7, 7)), rng.normal(size=(7, 4))
        y = np.concatenate((d.x0, Phi.ravel(), Psi.ravel()))
        ydot = d.eomstm(params, d.thrust, t, list(y), U)
        assert ydot.shape == (84,)
        assert np.array_equal(ydot[:7], xdot)
        dPhi, dPsi = A @ Phi, A @ Psi + B
        np.testing.assert_allclose(ydot[7:56], dPhi.ravel(), rtol=0, atol=1e-14 * np.max(np.abs(dPhi)))
        np.testing.assert_allclose(ydot[56:], dPsi.ravel(), rtol=0, atol=1e-14 * np.max(np.abs(dPsi)))


def test_eom_accepts_numpy_and_leaves_params_usable(params):
    d = DYNS[0]
    a = d.eom(params, d.thrust, 0.0, np.array(d.x0), np.array(U))
    assert np.array_equal(a, d.eom(params, d.thrust, 0.0, d.x0, U))
    a[0] = 123.0    # the output is the caller's
    assert d.eom(params, d.thrust, 0.0, d.x0, U)[0] != 123.0


# ---------------------------------------------------------------- SRP and the state mass

def srp_of(d, params, params_nosrp, x):
    """SRP contribution to xdot and to the mass column of A, without thrust"""
    u0 = [0.0] * 4
    xdot_srp, A_srp, _ = d.eval(params, d.thrust, 0.0, x, u0)
    xdot_off, A_off, _ = d.eval(params_nosrp, d.thrust, 0.0, x, u0)
    return xdot_srp - xdot_off, A_srp[:, 6] - A_off[:, 6]


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_srp_scales_with_the_state_mass(params, params_nosrp, d):
    x1 = list(d.x0[:6]) + [1.0]
    x2 = list(d.x0[:6]) + [2.0]
    s1, col1 = srp_of(d, params, params_nosrp, x1)
    s2, _ = srp_of(d, params, params_nosrp, x2)
    assert np.max(np.abs(s1)) > 0.0
    k = np.flatnonzero(np.abs(s1) > 0.0)
    if d.is_mee:
        # L is dominated by n w^2, so that the SRP part of its rate is lost to cancellation
        k = k[k != 5]
    err = np.max(np.abs(s2[k] - 0.5 * s1[k]) / np.abs(s1[k]))
    note(f"{d.name} SRP halves at m = 2", err)
    assert err <= TOL_SRP
    # the mass column of A is -xdot_srp / m, at m = 1
    err = np.max(np.abs(col1[k] + s1[k]) / np.abs(s1[k]))
    note(f"{d.name} SRP mass column", err)
    assert err <= TOL_SRP


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_srp_m_is_ignored(params, d):
    other = campanella.SpiceParameters(ET0, BODIES, GMs=GM_LIST, **dict(SRP, srp_m=1.0))
    for a, b in zip(d.eval(params, d.thrust, 0.0, d.x0, U), d.eval(other, d.thrust, 0.0, d.x0, U)):
        assert np.array_equal(a, b)


def test_mass_unit_sets_the_srp_mass(params):
    # a different m_bar with the same m_bar m and c1: same acceleration in rvm
    d = DYNS[0]
    u0 = [0.0] * 4
    a = d.eval(params, thruster(mass_unit_kg=MASS_UNIT), 0.0, list(d.x0[:6]) + [1.0], u0)[0]
    b = d.eval(params, thruster(mass_unit_kg=2 * MASS_UNIT), 0.0, list(d.x0[:6]) + [0.5], u0)[0]
    np.testing.assert_allclose(a[:6], b[:6], rtol=1e-14)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_u_zero_reduces_to_the_state_only_dynamics(params, d):
    # srp_m = m_bar m: exactly the same
    evaluate = campanella.eval_mee if d.is_mee else campanella.eval_rv
    xdot6, A6 = evaluate(params, 0.0, d.x0[:6])
    xdot, A, B = d.eval(params, d.thrust, 0.0, d.x0, [0.0] * 4)
    assert np.array_equal(xdot[:6], xdot6)
    assert np.array_equal(A[:6, :6], A6)
    assert xdot[6] == 0.0
    assert B[6, 3] == pytest.approx(-C2, rel=1e-15)


# ---------------------------------------------------------------- max_steps

# Thrust that deorbits STATE0: from about 9.9 days p -> 0 without reaching p <= 0, at about 30000 steps/s.
STALL_THRUST = dict(thrust_N=0.5, isp_s=3000.0, mass_unit_kg=MASS_UNIT)
STALL_U = [0.3, -0.5, 0.4, 0.71]
STALL_TF = 20 * DAY
STALL_STEPS = 100000
# CPU time, so that a loaded machine does not fail it: 3.5 s were measured for STALL_STEPS with sensitivities
STALL_CPU_S = 60.0


@pytest.mark.parametrize("solver", ["solve_ivp_meem", "solve_ivpstm_meem"])
def test_stall_stops_with_max_steps(params_nosrp, solver):
    th = campanella.ThrustOptions(**STALL_THRUST)
    start = time.process_time()
    with pytest.raises(RuntimeError, match=f"{solver}: reached max_steps = {STALL_STEPS} at t = .*p -> 0"):
        getattr(campanella, solver)(params_nosrp, th, [0.0, STALL_TF], X0_MEEM, STALL_U, max_steps=STALL_STEPS)
    assert time.process_time() - start < STALL_CPU_S


def test_stall_propagate_zoh_names_the_segment(params_nosrp):
    th = campanella.ThrustOptions(**STALL_THRUST)
    # the same trajectory, with the stall in the second segment
    x1 = campanella.solve_ivp_meem(params_nosrp, th, [0.0, 60.0], X0_MEEM, STALL_U).x[-1]
    x_ref = np.array([X0_MEEM, x1, x1])
    with pytest.raises(RuntimeError, match=f"propagate_zoh_meem: segment 1 of 2 .*reached max_steps = {STALL_STEPS}"):
        campanella.propagate_zoh_meem(params_nosrp, th, [0.0, 60.0, STALL_TF], x_ref, [STALL_U, STALL_U],
                                      max_steps=STALL_STEPS)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_max_steps_counts_accepted_steps(two_body, d):
    args = (two_body, d.thrust, [0.0, DAY], d.x0, U)
    full = d.solve(*args)
    steps = len(full) - 1
    assert steps > 2
    for limit in (steps, steps + 1, 0):
        r = d.solve(*args, max_steps=limit)
        assert np.array_equal(r.x, full.x), limit
    with pytest.raises(RuntimeError, match=f"reached max_steps = {steps - 1} at t = "):
        d.solve(*args, max_steps=steps - 1)
    # all of tspan, not per output time
    with pytest.raises(RuntimeError, match="reached max_steps = 1 at t = "):
        d.solve(*args, t_eval=[0.25 * DAY, 0.5 * DAY, DAY], max_steps=1)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_max_steps_negative(two_body, d):
    with pytest.raises(ValueError, match="max_steps must be >= 0"):
        d.solve(two_body, d.thrust, [0.0, 0.0], d.x0, U, max_steps=-1)
    x_ref = np.array([d.x0, d.x0])
    with pytest.raises(ValueError, match="max_steps must be >= 0"):
        d.propagate(two_body, d.thrust, [0.0, DAY], x_ref, [U], max_steps=-1)


def test_max_steps_is_only_on_the_controlled_solvers(two_body):
    with pytest.raises(TypeError):
        campanella.solve_ivp_rv(two_body, [0.0, DAY], STATE0, max_steps=1)
    with pytest.raises(TypeError):
        campanella.solve_ivpstm_mee(two_body, [0.0, DAY], X0_MEEM[:6], max_steps=1)


def test_long_arc_stays_under_the_default(two_body):
    # Three years of thrust from a low lunar orbit, in km and in canonical units. Cartesian takes the most
    # steps: 1.1e6 in km, near enough to the default for this to fail if the limit is lowered much.
    x = [1837.4, 0.0, 0.0, 0.0, 1.46, 0.8]
    th = campanella.ThrustOptions(thrust_N=0.1, isp_s=3000.0, mass_unit_kg=MASS_UNIT)
    mee = campanella.cart2mee(x, MU_MOON) + [1.0]
    canonical = campanella.SpiceParameters(ET0, ["MOON"], GMs=[MU_MOON], DU=1737.4, use_canonical_scales=True)
    rvm_canon = [v / canonical.DU for v in x[:3]] + [v / canonical.VU for v in x[3:]] + [1.0]
    u_rtn, u_inertial = [0.0, 1.0, 0.0, 1.0], [0.0, 0.1, 0.05, 0.1118]
    for solve, p, x0, u in ((campanella.solve_ivp_rvm, two_body, x + [1.0], u_inertial),
                            (campanella.solve_ivp_rvm, canonical, rvm_canon, u_inertial),
                            (campanella.solve_ivp_meem, two_body, mee, u_rtn),
                            (campanella.solve_ivp_meem, canonical, [mee[0] / canonical.DU] + mee[1:], u_rtn)):
        tf = 3 * 365.25 * DAY / p.TU
        r = solve(p, th, [0.0, tf], x0, u)
        assert r.t[-1] == tf
        if solve is campanella.solve_ivp_rvm and p is two_body:
            assert len(r) > 500000


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer")
def test_signal_stops_the_integration_and_leaves_params_usable(params_nosrp):
    class Stop(Exception):
        pass

    def handler(signum, frame):
        raise Stop()

    th = campanella.ThrustOptions(**STALL_THRUST)
    previous = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        # a limit that takes far longer than the timer, so that a lost signal fails rather than hangs
        with pytest.raises(Stop):
            campanella.solve_ivp_meem(params_nosrp, th, [0.0, STALL_TF], X0_MEEM, STALL_U, max_steps=3000000)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
    assert len(campanella.solve_ivp_meem(params_nosrp, th, [0.0, 60.0], X0_MEEM, STALL_U)) > 1


# ---------------------------------------------------------------- propagate_zoh

SEG_TIMES = [0.0, TF / 3, 2 * TF / 3, TF]
SEG_U = np.array([U, [0.1, 0.5, -0.2, 0.55], [-0.4, 0.1, 0.3, 0.51]])


def chain(d, params, x0, u_flat):
    """Continuous propagation over the segments: the states at SEG_TIMES, (4, 7)"""
    u = np.reshape(u_flat, (3, 4))
    out = [np.asarray(x0, dtype=float)]
    for k in range(3):
        out.append(np.array(d.solve(params, d.thrust, SEG_TIMES[k:k + 2], out[-1], u[k]).x[-1]))
    return np.array(out)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_propagate_shapes_and_views(params, d):
    x_ref = chain(d, params, d.x0, SEG_U)
    z = d.propagate(params, d.thrust, SEG_TIMES, x_ref, SEG_U)
    assert isinstance(z, campanella.ZohLinearization)
    assert len(z) == 3
    for name, shape in (("x_end", (3, 7)), ("Phi", (3, 7, 7)), ("Psi", (3, 7, 4)), ("c", (3, 7)), ("defects", (3, 7))):
        a = getattr(z, name)
        assert a.shape == shape, name
        assert not a.flags.writeable, name
        with pytest.raises(ValueError):
            a[0, 0] = 1.0
        assert a.base is not None, name
    # a view keeps the result alive
    a = d.propagate(params, d.thrust, SEG_TIMES, x_ref, SEG_U).Phi
    gc.collect()
    assert np.all(np.isfinite(a))


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_propagate_each_segment_is_the_solver(params, d):
    # from its own row of x_ref, not from the end of the previous segment
    x_ref = chain(d, params, d.x0, SEG_U) * (1.0 + 1e-6)
    z = d.propagate(params, d.thrust, SEG_TIMES, x_ref, SEG_U)
    for k in range(3):
        r = d.solvestm(params, d.thrust, SEG_TIMES[k:k + 2], x_ref[k], SEG_U[k])
        assert np.array_equal(z.x_end[k], r.x[-1])
        assert np.array_equal(z.Phi[k], r.stm[-1])
        assert np.array_equal(z.Psi[k], r.stm_u[-1])


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_propagate_defects_and_affine_term(params, d):
    x_ref = chain(d, params, d.x0, SEG_U)
    # a known shift of the row after segment 0; it starts segment 1 as well
    x_ref[1] += d.scale(x_ref[1]) * 1e-6
    z = d.propagate(params, d.thrust, SEG_TIMES, x_ref, SEG_U)
    # x_ref_{k+1} - x_end_k: exactly, and not the opposite sign
    assert np.array_equal(z.defects, x_ref[1:] - z.x_end)
    assert np.all(z.defects[0] * d.scale(x_ref[1]) >= 0.0)
    assert np.max(np.abs(z.defects[0] / d.scale(x_ref[1]) - 1e-6)) < 1e-7
    # x_end = Phi x_ref + Psi u_ref + c
    for k in range(3):
        rebuilt = z.Phi[k] @ x_ref[k] + z.Psi[k] @ SEG_U[k] + z.c[k]
        err = np.max(np.abs(rebuilt - z.x_end[k]) / d.scale(z.x_end[k]))
        note(f"{d.name} affine identity segment {k}", err)
        assert err <= 1e-9


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_propagate_chained_sensitivities_vs_fd(params, d):
    x_ref = chain(d, params, d.x0, SEG_U)
    z = d.propagate(params, d.thrust, SEG_TIMES, x_ref, SEG_U)
    xf = x_ref[-1]
    scale_f = d.scale(xf)

    Phi = z.Phi[2] @ z.Phi[1] @ z.Phi[0]
    Phi_fd = jacobian_fd_n(lambda x0: chain(d, params, x0, SEG_U)[-1], d.x0, d.hx(d.x0))
    both_metrics(Phi, Phi_fd, 7, 7, scale_f, d.scale(d.x0), f"{d.name} chained Phi", TOL_PROPAGATED_FD)
    both_metrics(Phi[:6], Phi_fd[:6], 6, 7, scale_f[:6], d.scale(d.x0), f"{d.name} chained Phi r,v", TOL_PROPAGATED_FD)

    # d x_N / d u_k = Phi_{N-1} ... Phi_{k+1} Psi_k
    Psi = np.hstack((z.Phi[2] @ z.Phi[1] @ z.Psi[0], z.Phi[2] @ z.Psi[1], z.Psi[2]))
    Psi_fd = jacobian_fd_n(lambda u: chain(d, params, d.x0, u)[-1], SEG_U.ravel(), np.tile(HU, 3))
    both_metrics(Psi, Psi_fd, 7, 12, scale_f, None, f"{d.name} chained Psi", TOL_PROPAGATED_FD)
    # the mass row dominates the Gamma columns of the column metric above
    both_metrics(Psi[:6], Psi_fd[:6], 6, 12, scale_f[:6], None, f"{d.name} chained Psi r,v", TOL_PROPAGATED_FD)
    assert np.all(np.abs(Psi[:6, [3, 7, 11]]) > 0.0)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_propagate_backward(two_body, d):
    fwd = d.solve(two_body, d.thrust, [0.0, DAY], d.x0, U).x[-1]
    z = d.propagate(two_body, d.thrust, [DAY, 0.0], [fwd, d.x0], [U])
    err = np.max(np.abs(z.defects[0]) / d.scale(d.x0))
    note(f"{d.name} backward defect", err)
    assert err <= 1e-10


def test_propagate_array_conversions(two_body):
    d = DYNS[0]
    x_ref = np.array([d.x0, d.x0, d.x0])
    u_ref = np.array([U, U])
    times = [0.0, 100.0, 200.0]
    ref = d.propagate(two_body, d.thrust, times, x_ref, u_ref)
    others = [(x_ref.tolist(), u_ref.tolist()),
              (np.asfortranarray(x_ref), np.asfortranarray(u_ref)),
              (np.vstack((x_ref, x_ref))[::2][:3], u_ref)]
    for x, u in others:
        assert np.array_equal(d.propagate(two_body, d.thrust, times, x, u).Phi, ref.Phi)
    assert np.array_equal(d.propagate(two_body, d.thrust, np.array(times), x_ref, u_ref).x_end, ref.x_end)
    # the input is copied, and an integer array is converted
    integer = np.array([[7000, 0, 0, 0, 1, 0, 1], [7000, 0, 0, 0, 1, 0, 1]])
    z = d.propagate(two_body, d.thrust, [0.0, 1.0], integer, np.array([[0, 0, 0, 0]]))
    assert z.x_end.shape == (1, 7)


@pytest.mark.parametrize("d", DYNS, ids=IDS)
def test_propagate_errors(two_body, d):
    t = [0.0, 100.0, 200.0]
    x = np.array([d.x0, d.x0, d.x0])
    u = np.array([U, U])
    name = "propagate_zoh_" + ("meem" if d.is_mee else "rvm")
    call = lambda times, xr, ur: d.propagate(two_body, d.thrust, times, xr, ur)
    for bad in (x.ravel(), x[:, :6], x[np.newaxis], np.zeros((3, 8))):
        with pytest.raises(ValueError, match=f"{name}: x_ref must be a 2-D array with 7 columns"):
            call(t, bad, u)
    for bad in (u.ravel(), u[:, :3], u[np.newaxis]):
        with pytest.raises(ValueError, match=f"{name}: u_ref must be a 2-D array with 4 columns"):
            call(t, x, bad)
    # rows against times
    with pytest.raises(ValueError, match="x_ref must have 7 N"):
        call(t, x[:2], u)
    with pytest.raises(ValueError, match="u_ref must have 4"):
        call(t, x, u[:1])
    with pytest.raises(ValueError, match="at least 2"):
        call([0.0], x[:1], np.zeros((0, 4)))
    with pytest.raises(ValueError, match="strictly monotonic"):
        call([0.0, 200.0, 100.0], x, u)
    bad = x.copy()
    bad[2, 3] = np.nan
    with pytest.raises(ValueError, match=r"x_ref must be finite \(row 2\)"):
        call(t, bad, u)
    # the type and the segment of a failure in a solver
    bad = x.copy()
    bad[1, 6] = -0.5
    with pytest.raises(ValueError, match=f"{name}: segment 1 of 2 .*mass out of domain"):
        call(t, bad, u)
    # no output times here
    with pytest.raises(TypeError):
        d.propagate(two_body, d.thrust, t, x, u, t_eval=[0.0])
    # the last row is only compared
    last = x.copy()
    last[2, 6] = -1.0
    assert call(t, last, u).defects[1, 6] < 0.0
