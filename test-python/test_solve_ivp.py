"""
Tests of solve_ivp_rv / solve_ivpstm_rv: references, STM, backward propagation, t_eval, EOMs
"""

import numpy as np
import pytest

from helpers import assert_state_close, campanella, final_row, furnsh_kernels, row_normalised_error

ET0 = 740463475.5117936
STATE0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
          0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
TSPAN = [0.0, 20 * 86400.0]
GM_LIST = [4.902800582147760e+03, 398600.432896939, 132712197035.766]
MU_MOON = GM_LIST[0]
BODIES = ["MOON", "EARTH", "SUN"]
SRP = dict(include_srp=True, srp_A=34.675, srp_m=1850.0, srp_rhoS=0.9, srp_rhoD=0.0)

# reference final states (unchanged from the Propagator* classes)
TWOBODY_EXPECT = [-18205.698468519975, 26479.25761849232, -38615.225993310676,
                  0.01263483050054705, 0.1635690404557372, -0.19163703441809382]
RV_NBODY_EXPECT = [10171.343017143072, 33674.15294325228, -60877.5267967639,
                   0.013967106146233203, -0.0805131196626756, 0.0033270015062299776]
# with SRP (SRP dict below); confirmed by an independent RK4 + spiceypy transcription to 1.1e-5 km
RV_SRP_EXPECT = [10265.7677192918, 33657.38156810862, -60889.03206439558,
                 0.014176756803175221, -0.08047039542453092, 0.003449062130405538]
# SRP constants of the model: solar flux at 1 AU, speed of light, astronomical unit
SRP_P0, SRP_C, AU_KM = 1367.0, 299792458.0, 149597870.7


def moon_params(**kw):
    return campanella.SpiceParameters(ET0, BODIES, GMs=GM_LIST, **kw)


def stm_fd(params, tspan, x0, steps, solve=campanella.solve_ivp_rv):
    """Central differences of the final state w.r.t. x0"""
    x0 = np.array(x0, dtype=float)
    Phi = np.zeros((6, 6))
    for j in range(6):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += steps[j]
        xm[j] -= steps[j]
        Phi[:, j] = (final_row(solve(params, tspan, list(xp)), tspan[1])
                     - final_row(solve(params, tspan, list(xm)), tspan[1])) / (2 * steps[j])
    return Phi


def test_twobody_without_ephemerides():
    """One body with an explicit GM: no SPICE query, same result as the former two-body propagator"""
    params = campanella.SpiceParameters(0.0, ["MOON"], GMs=[MU_MOON])
    res = campanella.solve_ivp_rv(params, TSPAN, STATE0)
    assert res.t[0] == 0.0 and res.t[-1] == TSPAN[1]
    assert_state_close(res.x[-1], TWOBODY_EXPECT, 1e-11)
    # energy and angular momentum
    r = np.linalg.norm(res.x[:, :3], axis=1)
    energy = 0.5 * np.sum(res.x[:, 3:] ** 2, axis=1) - MU_MOON / r
    h = np.cross(res.x[:, :3], res.x[:, 3:])
    assert np.max(np.abs(energy - energy[0])) <= 1e-12 * abs(energy[0])
    assert np.max(np.linalg.norm(h - h[0], axis=1)) <= 1e-10 * np.linalg.norm(h[0])


def test_result_shapes_and_views():
    furnsh_kernels()
    res = campanella.solve_ivp_rv(moon_params(), [0.0, 86400.0], STATE0)
    n = len(res)
    assert res.t.shape == (n,) and res.x.shape == (n, 6) and res.stm is None
    assert not res.x.flags.writeable
    np.testing.assert_array_equal(res.x[0], STATE0)
    assert np.all(np.diff(res.t) > 0)
    res = campanella.solve_ivpstm_rv(moon_params(), [0.0, 86400.0], STATE0)
    assert res.stm.shape == (len(res), 6, 6)
    np.testing.assert_array_equal(res.stm[0], np.eye(6))
    # the arrays outlive the Python reference to the result
    x = campanella.solve_ivp_rv(moon_params(), [0.0, 3600.0], STATE0).x
    np.testing.assert_array_equal(x[0], STATE0)


def test_nbody_reference_and_stm():
    """Moon-centred N-body: reference final state; STM vs central differences"""
    furnsh_kernels()
    params = moon_params()
    xf = final_row(campanella.solve_ivp_rv(params, TSPAN, STATE0), TSPAN[1])
    assert_state_close(xf, RV_NBODY_EXPECT, 1e-11)

    yf = final_row(campanella.solve_ivpstm_rv(params, TSPAN, STATE0), TSPAN[1])
    assert_state_close(yf[:6], xf, 1e-10)
    Phi = yf[6:].reshape(6, 6)
    Phi_fd = stm_fd(params, TSPAN, STATE0, [1e-6] * 6)
    err = row_normalised_error(Phi, Phi_fd)
    print(f"Cartesian STM vs FD: {err:.3e}")
    assert err <= 1e-5


def test_srp_stm_vs_fd():
    """The Cartesian STM includes the SRP gradient"""
    furnsh_kernels()
    tspan = [0.0, 5 * 86400.0]
    params = moon_params(**SRP)
    yf = final_row(campanella.solve_ivpstm_rv(params, tspan, STATE0), tspan[1])
    Phi = yf[6:].reshape(6, 6)
    Phi_fd = stm_fd(params, tspan, STATE0, [1e-3] * 3 + [1e-8] * 3)
    err = row_normalised_error(Phi, Phi_fd)
    # Phi without SRP differs from the SRP STM by more than the tolerance
    Phi_nosrp = final_row(campanella.solve_ivpstm_rv(moon_params(), tspan, STATE0), tspan[1])[6:].reshape(6, 6)
    err_nosrp = row_normalised_error(Phi_nosrp, Phi_fd)
    print(f"SRP STM vs FD: {err:.3e}; STM without SRP vs FD: {err_nosrp:.3e}")
    assert err <= 1e-6
    assert err_nosrp > 10 * err


def test_srp_reference_and_effect():
    furnsh_kernels()
    xf = final_row(campanella.solve_ivp_rv(moon_params(**SRP), TSPAN, STATE0), TSPAN[1])
    assert_state_close(xf, RV_SRP_EXPECT, 1e-11)
    xf_nbody = final_row(campanella.solve_ivp_rv(moon_params(), TSPAN, STATE0), TSPAN[1])
    assert np.linalg.norm(xf[:3] - xf_nbody[:3]) > 10.0


@pytest.mark.parametrize("rhoD", [0.0, 0.3])
def test_srp_acceleration_independent(rhoD):
    """SRP acceleration against an independent evaluation with spiceypy's own CSPICE:
    a = (P0/c) Cr (A/m) (AU/|d|)^2 d/|d|, d = r - r_sun, Cr = 1 + rhoS + 5 rhoD / 3"""
    import os
    import spiceypy
    furnsh_kernels()
    spice_dir = os.environ["SPICE"]
    spiceypy.furnsh(os.path.join(spice_dir, "lsk", "naif0012.tls"))
    spiceypy.furnsh(os.path.join(spice_dir, "spk", "de440.bsp"))
    srp = dict(SRP, srp_rhoD=rhoD)
    for DU in [None, 3000.0]:
        p_srp = moon_params(**srp, DU=DU)
        p_grav = moon_params(DU=DU)
        t = 0.7 * 86400.0 / p_srp.TU
        x = np.array(STATE0) / np.array([p_srp.DU] * 3 + [p_srp.VU] * 3)
        a = (campanella.eval_rv(p_srp, t, list(x))[0] - campanella.eval_rv(p_grav, t, list(x))[0])[3:]
        a_kms2 = a * p_srp.DU / p_srp.TU ** 2
        s = np.array(spiceypy.spkpos("SUN", ET0 + 0.7 * 86400.0, "J2000", "NONE", "MOON")[0])
        d = np.array(STATE0[:3]) - s
        Cr = 1 + SRP["srp_rhoS"] + 5 * rhoD / 3
        a_ref = SRP_P0 / SRP_C * Cr * SRP["srp_A"] / SRP["srp_m"] / 1e3 * (AU_KM / np.linalg.norm(d)) ** 2 \
            * d / np.linalg.norm(d)
        np.testing.assert_allclose(a_kms2, a_ref, rtol=1e-9)


def test_heliocentric_srp():
    """Sun as the central body: SRP is radial, outward, and K / r^2 in magnitude"""
    gm_sun = 1.3271244004127942e11
    srp = dict(include_srp=True, srp_A=10.0, srp_m=500.0, srp_rhoS=0.2)
    p_srp = campanella.SpiceParameters(ET0, ["SUN"], GMs=[gm_sun], **srp)
    p_grav = campanella.SpiceParameters(ET0, ["SUN"], GMs=[gm_sun])
    x = [1.2e8, -0.5e8, 0.3e8, 10.0, 25.0, 1.0]
    a = (campanella.eval_rv(p_srp, 0.0, x)[0] - campanella.eval_rv(p_grav, 0.0, x)[0])[3:]
    r = np.array(x[:3])
    a_ref = SRP_P0 / SRP_C * 1.2 * 10.0 / 500.0 / 1e3 * (AU_KM / np.linalg.norm(r)) ** 2 * r / np.linalg.norm(r)
    # a is the difference of two ~6e-3 km/s^2 accelerations: ~1e-13 / 1e-10 of cancellation error
    np.testing.assert_allclose(a, a_ref, rtol=1e-9)


def test_canonical_vs_dimensional_jupiter():
    """DU = 3000 km with Jupiter as a fourth body: states and STM agree after rescaling"""
    furnsh_kernels()
    gm = GM_LIST + [1.2671276409999998E+08]
    bodies = ["301", "399", "10", "5"]
    km = campanella.SpiceParameters(ET0, bodies, GMs=gm)
    canon = campanella.SpiceParameters(ET0, bodies, GMs=gm, DU=3000.0)
    DU, TU, VU = canon.DU, canon.TU, canon.VU
    assert DU == 3000.0 and VU == np.sqrt(MU_MOON / DU) and TU == DU / VU
    assert canon.mus[0] == 1.0
    D = np.diag([DU] * 3 + [VU] * 3)
    x0c = np.array(STATE0) / np.diag(D)
    yf_km = final_row(campanella.solve_ivpstm_rv(km, TSPAN, STATE0), TSPAN[1])
    tf_c = TSPAN[1] / TU
    yf_c = final_row(campanella.solve_ivpstm_rv(canon, [0.0, tf_c], list(x0c)), tf_c)
    xf = D @ yf_c[:6]
    np.testing.assert_allclose(xf[:3], yf_km[:3], rtol=0, atol=1e-5)
    np.testing.assert_allclose(xf[3:], yf_km[3:6], rtol=0, atol=1e-10)
    Phi_km = yf_km[6:].reshape(6, 6)
    Phi_c = D @ yf_c[6:].reshape(6, 6) @ np.linalg.inv(D)
    err = np.max(np.abs(Phi_c - Phi_km)) / np.max(np.abs(Phi_km))
    print(f"canonical vs km/s STM: {err:.3e}")
    assert err <= 1e-8
    # the canonical STM against central differences in canonical units
    Phi_fd = stm_fd(canon, [0.0, tf_c], x0c, [1e-6] * 6)
    assert row_normalised_error(yf_c[6:].reshape(6, 6), Phi_fd) <= 1e-5


@pytest.mark.parametrize("srp", [False, True])
def test_forward_backward(srp):
    """Propagating to tf and back returns to x0; Phi(t0, tf) Phi(tf, t0) = I"""
    furnsh_kernels()
    params = moon_params(**SRP) if srp else moon_params()
    tspan = [0.0, 5 * 86400.0]
    fwd = campanella.solve_ivpstm_rv(params, tspan, STATE0, atol=1e-13, rtol=1e-13)
    bwd = campanella.solve_ivpstm_rv(params, tspan[::-1], list(fwd.x[-1]), atol=1e-13, rtol=1e-13)
    assert bwd.t[-1] == 0.0 and np.all(np.diff(bwd.t) < 0)
    np.testing.assert_allclose(bwd.x[-1][:3], STATE0[:3], rtol=0, atol=1e-6)
    np.testing.assert_allclose(bwd.x[-1][3:], STATE0[3:], rtol=0, atol=1e-11)
    # scaled so that the position and velocity blocks weigh alike
    D = np.diag([np.linalg.norm(STATE0[:3])] * 3 + [np.linalg.norm(STATE0[3:])] * 3)
    prod = np.linalg.inv(D) @ bwd.stm[-1] @ fwd.stm[-1] @ D
    assert np.max(np.abs(prod - np.eye(6))) <= 1e-9


def test_stm_symplectic_without_srp():
    """Gravity only: Phi^T J Phi = J"""
    furnsh_kernels()
    Phi = campanella.solve_ivpstm_rv(moon_params(), [0.0, 5 * 86400.0], STATE0).stm[-1]
    J = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
    err = np.max(np.abs(Phi.T @ J @ Phi - J)) / np.max(np.abs(Phi)) ** 2
    print(f"symplecticity error: {err:.3e}")
    assert err <= 1e-10


def test_t_eval():
    furnsh_kernels()
    params = moon_params(**SRP)
    dense = campanella.solve_ivp_rv(params, TSPAN, STATE0)
    t_eval = [0.0, 1234.5, 86400.0, 10 * 86400.0 + 0.25, TSPAN[1]]
    res = campanella.solve_ivp_rv(params, TSPAN, STATE0, t_eval=t_eval)
    np.testing.assert_array_equal(res.t, t_eval)
    np.testing.assert_array_equal(res.x[0], STATE0)
    np.testing.assert_allclose(res.x[-1], dense.x[-1], rtol=1e-9)
    # without t0 the first row is the first t_eval point
    res = campanella.solve_ivp_rv(params, TSPAN, STATE0, t_eval=[86400.0])
    assert len(res) == 1 and res.t[0] == 86400.0
    # backward t_eval, with the STM
    back = campanella.solve_ivpstm_rv(params, [86400.0, 0.0], list(res.x[0]), t_eval=[43200.0, 0.0])
    np.testing.assert_array_equal(back.t, [43200.0, 0.0])
    np.testing.assert_allclose(back.x[-1][:3], STATE0[:3], rtol=0, atol=1e-6)
    assert back.stm.shape == (2, 6, 6)
    back_dense = campanella.solve_ivpstm_rv(params, [86400.0, 0.0], list(res.x[0]))
    assert row_normalised_error(back.stm[-1], back_dense.stm[-1]) <= 1e-8


def test_zero_length_tspan():
    furnsh_kernels()
    res = campanella.solve_ivpstm_rv(moon_params(), [100.0, 100.0], STATE0)
    assert len(res) == 1 and res.t[0] == 100.0
    np.testing.assert_array_equal(res.x[0], STATE0)
    np.testing.assert_array_equal(res.stm[0], np.eye(6))


@pytest.mark.parametrize("kwargs", [
    dict(tspan=[0.0]), dict(tspan=[0.0, 1.0, 2.0]), dict(tspan=[0.0, float("nan")]),
    dict(x0=STATE0[:5]), dict(x0=STATE0 + [1.0]), dict(x0=[float("inf")] + STATE0[1:]),
    dict(atol=-1.0), dict(atol=0.0, rtol=0.0), dict(rtol=float("nan")),
    dict(t_eval=[0.0, 2 * TSPAN[1]]), dict(t_eval=[-1.0]), dict(t_eval=[10.0, 5.0]),
    dict(t_eval=[5.0, 5.0]),
])
def test_argument_errors(kwargs):
    furnsh_kernels()
    args = dict(tspan=TSPAN, x0=STATE0)
    args.update(kwargs)
    tspan, x0 = args.pop("tspan"), args.pop("x0")
    for solve in [campanella.solve_ivp_rv, campanella.solve_ivpstm_rv]:
        with pytest.raises(ValueError):
            solve(moon_params(), tspan, x0, **args)


def test_non_finite_raises():
    """At the center of the body the derivative is not finite: ValueError, never NaN states"""
    params = campanella.SpiceParameters(0.0, ["EARTH"], GMs=[398600.4418])
    with pytest.raises(ValueError, match="non-finite"):
        campanella.solve_ivp_rv(params, [0.0, 100.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        campanella.eval_rv(params, 0.0, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        campanella.eom_rv(params, 0.0, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def test_spice_error_raises():
    """A body without ephemeris data raises RuntimeError with NAIF's message"""
    furnsh_kernels()
    params = campanella.SpiceParameters(ET0, ["MOON", "PLUTO"], GMs=[MU_MOON, 869.6])
    with pytest.raises(RuntimeError, match="SPKINSUFFDATA"):
        campanella.solve_ivp_rv(params, [0.0, 60.0], STATE0)
    with pytest.raises(RuntimeError, match="SPKINSUFFDATA"):
        campanella.eval_rv(params, 0.0, STATE0)
    with pytest.raises(RuntimeError, match="SPKINSUFFDATA"):
        campanella.eom_rv(params, 0.0, STATE0)
    # the SPICE error state was reset
    campanella.solve_ivp_rv(moon_params(), [0.0, 60.0], STATE0)


def test_eom_and_eval_rv():
    """eom_rv, eomstm_rv and eval_rv agree; A = [[0, I], [G, 0]] matches central differences"""
    furnsh_kernels()
    params = moon_params(**SRP)
    t = 3600.0
    x = np.array(STATE0)
    xdot, A = campanella.eval_rv(params, t, list(x))
    np.testing.assert_array_equal(campanella.eom_rv(params, t, list(x)), xdot)
    y = list(x) + list(np.eye(6).ravel())
    ydot = campanella.eomstm_rv(params, t, y)
    np.testing.assert_array_equal(ydot[:6], xdot)
    np.testing.assert_array_equal(ydot[6:].reshape(6, 6), A)
    np.testing.assert_array_equal(A[:3, :3], 0.0)
    np.testing.assert_array_equal(A[:3, 3:], np.eye(3))
    np.testing.assert_array_equal(A[3:, 3:], 0.0)
    A_fd = np.zeros((6, 6))
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = 1e-2 if j < 3 else 1e-6
        A_fd[:, j] = (campanella.eom_rv(params, t, list(x + dx))
                      - campanella.eom_rv(params, t, list(x - dx))) / (2 * dx[j])
    assert np.max(np.abs(A[3:, :3] - A_fd[3:, :3])) <= 1e-8 * np.max(np.abs(A[3:, :3]))
    with pytest.raises(ValueError):
        campanella.eom_rv(params, t, list(x[:5]))
    with pytest.raises(ValueError):
        campanella.eomstm_rv(params, t, list(x))
