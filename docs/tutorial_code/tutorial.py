"""campanella tutorial: a geostationary satellite in the ephemeris model.

Run from the repository root, with the directory of the built module on PYTHONPATH and SPICE
pointing at a kernel directory that contains lsk/naif0012.tls, spk/de440.bsp, pck/gm_de440.tpc and
pck/earth_latest_high_prec.bpc:

    SPICE=/path/to/spice PYTHONPATH=build/lib uv run python docs/tutorial_code/tutorial.py

Pass --plot to also write the figures used in the documentation (needs matplotlib).
The sections between "# [name]" and "# [name-end]" are included in docs/tutorial.md.
"""

import sys

# [setup]
import os

import numpy as np

import campanella   # the module built in build/lib; that directory must be on PYTHONPATH

np.set_printoptions(precision=6, suppress=False, linewidth=100)   # how arrays are printed below

# the repository root, two levels above this script; used for the gravity-field file in section 10
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# [setup-end]

# [kernels]
spice_dir = os.environ["SPICE"]          # e.g. /path/to/spice
for kernel in ["lsk/naif0012.tls",       # leap seconds: UTC <-> TDB (optional for campanella)
               "spk/de440.bsp",          # planetary ephemeris: positions of the Moon, Sun, ...
               "pck/gm_de440.tpc"]:      # gravitational parameters (GM) consistent with DE440
    campanella.furnsh(os.path.join(spice_dir, kernel))
# [kernels-end]

# [epoch]
import spiceypy   # only for the time conversion; it has its own CSPICE and its own kernel pool

spiceypy.furnsh(os.path.join(spice_dir, "lsk/naif0012.tls"))
et0 = spiceypy.str2et("2025-01-01T00:00:00")   # ISO strings are UTC
print(f"et0 = {et0!r} s past J2000 (TDB)")
print(f"ET - UTC seconds past J2000: {et0 - 788961600.0:.6f} s")
# [epoch-end]

# [params]
params = campanella.SpiceParameters(et0, ["EARTH", "MOON", "SUN"])   # names or IDs, as strings
print(params)
print("NAIF codes:", params.naif_codes)
print("GMs [km^3/s^2]:", params.GMs)
# [params-end]

# [initial-state]
mu_earth = params.GMs[0]
a_geo = 42164.0                                   # km
v_geo = np.sqrt(mu_earth / a_geo)                 # km/s, circular speed
x0 = [a_geo, 0.0, 0.0, 0.0, v_geo, 0.0]           # equatorial (J2000), circular
period = 2*np.pi*np.sqrt(a_geo**3 / mu_earth)
print(f"v_geo = {v_geo:.9f} km/s, period = {period/3600:.6f} h")
# [initial-state-end]

# [two-body-vs-nbody]
tf = 30*86400.0                                   # 30 days, in s
t_eval = np.linspace(0.0, tf, 31)                 # one output per day

params_2b = campanella.SpiceParameters(et0, ["EARTH"], GMs=[mu_earth])   # Earth only
res_2b = campanella.solve_ivp_rv(params_2b, [0.0, tf], x0, t_eval=t_eval)
res_nb = campanella.solve_ivp_rv(params, [0.0, tf], x0, t_eval=t_eval)

dr = np.linalg.norm(res_nb.x[:, :3] - res_2b.x[:, :3], axis=1)
for day in [1, 5, 10, 20, 30]:
    print(f"day {day:2d}: |r_nbody - r_2body| = {dr[day]:8.3f} km")

# split the day-30 difference into radial, along-track and cross-track (RTN) components,
# using the unit vectors of the two-body reference orbit
r_ref, v_ref = res_2b.x[-1, :3], res_2b.x[-1, 3:]
e_R = r_ref / np.linalg.norm(r_ref)
e_N = np.cross(r_ref, v_ref) / np.linalg.norm(np.cross(r_ref, v_ref))
e_T = np.cross(e_N, e_R)
diff = res_nb.x[-1, :3] - r_ref
print(f"day 30 difference: radial {diff @ e_R:8.3f} km, along-track {diff @ e_T:8.3f} km, "
      f"cross-track {diff @ e_N:8.3f} km")
# [two-body-vs-nbody-end]

# [accelerations]
xdot_2b, _ = campanella.eval_rv(params_2b, 0.0, x0)
xdot_nb, _ = campanella.eval_rv(params, 0.0, x0)
a_kepler = np.linalg.norm(xdot_2b[3:])
a_pert = np.linalg.norm(xdot_nb[3:] - xdot_2b[3:])
print(f"|a_Earth| = {a_kepler:.6e} km/s^2, |a_Moon + a_Sun| = {a_pert:.6e} km/s^2, "
      f"ratio = {a_pert/a_kepler:.3e}")
# [accelerations-end]

# [canonical]
params_c = campanella.SpiceParameters(et0, ["EARTH", "MOON", "SUN"], DU=a_geo)
DU, TU, VU = params_c.DU, params_c.TU, params_c.VU
print(f"DU = {DU} km, TU = {TU:.6f} s, VU = {VU:.9f} km/s")
print("mus (canonical):", params_c.mus)

x0_c = np.concatenate((np.array(x0[:3]) / DU, np.array(x0[3:]) / VU))
res_c = campanella.solve_ivp_rv(params_c, [0.0, tf / TU], x0_c, t_eval=t_eval / TU)
r_c_km = res_c.x[-1, :3] * DU                     # back to km
print(f"canonical vs km/s, final position: {np.linalg.norm(r_c_km - res_nb.x[-1, :3]):.3e} km")
# [canonical-end]

# [mee]
mee0 = campanella.cart2mee(x0, mu_earth)
print("MEE at t0 [p, f, g, h, k, L]:", np.array(mee0))
res_mee = campanella.solve_ivp_mee(params, [0.0, tf], mee0, t_eval=t_eval)

rv_from_mee = campanella.mee2cart(res_mee.x[-1], mu_earth)   # numpy rows are accepted
print(f"MEE vs Cartesian, final position: "
      f"{np.linalg.norm(np.array(rv_from_mee[:3]) - res_nb.x[-1, :3]):.3e} km")

p, f, g, h, k, L = res_mee.x.T
ecc = np.hypot(f, g)
inc = np.degrees(2*np.arctan(np.hypot(h, k)))
for day in [0, 10, 20, 30]:
    print(f"day {day:2d}: p = {p[day]:.3f} km, e = {ecc[day]:.3e}, i = {inc[day]:.5f} deg, "
          f"L = {L[day]:.4f} rad")
# [mee-end]

# [stm]
t_stm = 86400.0                                   # 1 day
res_stm = campanella.solve_ivpstm_rv(params, [0.0, t_stm], x0)
Phi = res_stm.stm[-1]                             # Phi(t_stm, t0), 6x6
print("Phi(1 day, 0):")
print(Phi)

# finite-difference check: column j of Phi is d x(t) / d x0[j]
steps = [1e-3]*3 + [1e-6]*3                       # km, km/s
Phi_fd = np.zeros((6, 6))
for j in range(6):
    xp = np.array(x0); xp[j] += steps[j]
    xm = np.array(x0); xm[j] -= steps[j]
    fp = campanella.solve_ivp_rv(params, [0.0, t_stm], xp).x[-1]
    fm = campanella.solve_ivp_rv(params, [0.0, t_stm], xm).x[-1]
    Phi_fd[:, j] = (fp - fm) / (2*steps[j])

# compare in scaled units, so that the position and velocity blocks weigh alike
D = np.diag([a_geo]*3 + [v_geo]*3)
err = np.linalg.solve(D, (Phi - Phi_fd) @ D)
print(f"max |Phi - Phi_fd| (scaled) = {np.abs(err).max():.3e}")

# linear prediction of a displaced trajectory
dx0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])    # 1 km radial offset
x_ref = campanella.solve_ivp_rv(params, [0.0, t_stm], x0).x[-1]
x_new = campanella.solve_ivp_rv(params, [0.0, t_stm], np.array(x0) + dx0).x[-1]
dx_true = x_new - x_ref
dx_lin = Phi @ dx0
print("dx(1 day), nonlinear:", dx_true)
print("dx(1 day), Phi @ dx0:", dx_lin)
# [stm-end]

# [srp]
params_srp = campanella.SpiceParameters(
    et0, ["EARTH", "MOON", "SUN"],
    include_srp=True, srp_A=20.0, srp_m=1000.0,   # m^2, kg: A/m = 0.02 m^2/kg
    srp_rhoS=0.3, srp_rhoD=0.0)
res_srp = campanella.solve_ivp_mee(params_srp, [0.0, tf], mee0, t_eval=t_eval)
ecc_srp = np.hypot(res_srp.x[:, 1], res_srp.x[:, 2])
for day in [10, 20, 30]:
    print(f"day {day:2d}: e without SRP = {ecc[day]:.3e}, with SRP = {ecc_srp[day]:.3e}")
rv_srp = campanella.mee2cart(res_srp.x[-1], mu_earth)
print(f"SRP effect on the final position: "
      f"{np.linalg.norm(np.array(rv_srp[:3]) - np.array(rv_from_mee[:3])):.3f} km")
# [srp-end]

# [sh]
campanella.furnsh(os.path.join(spice_dir, "pck/earth_latest_high_prec.bpc"))   # ITRF93 orientation
egm2008 = os.path.join(REPO, "test-python", "data", "EGM2008_n10.gfc")         # degree-10 head of EGM2008

field = campanella.load_spherical_harmonics(egm2008, 2)
J2 = -np.sqrt(5.0) * field["C"][2][0]          # unnormalized J2 from the normalized C20
print(f"EGM2008: GM = {field['GM']} km^3/s^2, R = {field['R']} km, J2 = {J2:.9e}")

for nmax in [2, 4]:
    # GMs[0] (from gm_de440.tpc) must match the GM of the file within 1e-5; it does, to 1.5e-8
    params_sh = campanella.SpiceParameters(
        et0, ["EARTH", "MOON", "SUN"],
        sh_file=egm2008, sh_nmax=nmax, sh_frame="ITRF93")
    res_sh = campanella.solve_ivp_rv(params_sh, [0.0, tf], x0, t_eval=t_eval)
    dr_sh = np.linalg.norm(res_sh.x[:, :3] - res_nb.x[:, :3], axis=1)
    print(f"sh_nmax = {nmax}: |r_sh - r_pointmass| = {dr_sh[1]:7.3f} km (day 1), "
          f"{dr_sh[10]:7.3f} km (day 10), {dr_sh[30]:7.3f} km (day 30)")

# circular speed at a_geo with J2, in the equatorial plane: v^2 = mu/r (1 + 3/2 J2 (R/r)^2)
v_j2 = np.sqrt(mu_earth / a_geo * (1 + 1.5*J2*(field["R"]/a_geo)**2))
print(f"circular speed with J2 - without: {(v_j2 - v_geo)*1e6:.2f} mm/s")
# [sh-end]

if "--plot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = "#2a78d6", "#eb6834"
    days = t_eval / 86400.0
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(days, dr, color=blue, lw=2, label="Moon + Sun, vs two-body")
    ax.plot(days, dr_sh, color=orange, lw=2, label="Earth field to degree 4, vs point mass")
    ax.set_xlabel("time since et0 [days]")
    ax.set_ylabel("position difference [km]")
    ax.set_title("Deviation from a reference trajectory", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(days, inc, color=blue, lw=2)
    ax.set_xlabel("time since et0 [days]")
    ax.set_ylabel("inclination to J2000 equator [deg]")
    ax.set_title("Inclination, Earth + Moon + Sun", fontsize=10)
    for ax in axes:
        ax.grid(color="#dddddd", lw=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.savefig(os.path.join(REPO, "docs", "_static", "tutorial_geo.png"), dpi=150)
    print("wrote docs/_static/tutorial_geo.png")
