# Basic usage through python

## Overview

Every propagation has two steps:

1. Declare the dynamics in a `SpiceParameters` object: the central body, the third bodies, solar radiation pressure (SRP), the central-body gravity field, and the units.
2. Pass it to one of the four solvers:

| | Cartesian `[x, y, z, vx, vy, vz]` | Modified equinoctial elements `[p, f, g, h, k, L]` |
|---|---|---|
| state | `solve_ivp_rv` | `solve_ivp_mee` |
| state and STM | `solve_ivpstm_rv` | `solve_ivpstm_mee` |

The dynamics are the same in both coordinate sets; only the integrated variables differ.

The integration uses the explicit embedded Runge-Kutta Prince-Dormand (8, 9) method ([`gsl_odeiv2_step_rk8pd` from GNU GSL](https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_step_type.gsl_odeiv2_step_rk8pd)), with [step-size control](https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_control_y_new) given by (`gsl_odeiv2_control_y_new`, i.e. $a_y = 1$, $a_{dydt} = 0$)

$$
D_i=\epsilon_{a b s}+\epsilon_{\text {rel }}\left|y_i\right|
$$


## A first example

```python
# make sure the directory of the built module is on sys.path
import campanella

# campanella reads ephemerides from its own CSPICE: spiceypy.furnsh does not load these kernels
campanella.furnsh("/path/to/lsk/naif0012.tls")
campanella.furnsh("/path/to/spk/de440.bsp")
campanella.furnsh("/path/to/pck/gm_de440.tpc")     # only needed when GMs is not given

et0 = 740463475.5117936                  # TDB seconds past J2000
naif_ids = ["399", "301", "10"]          # Earth (central body), Moon, Sun
params = campanella.SpiceParameters(et0, naif_ids)

x0 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]   # km, km/s
res = campanella.solve_ivp_rv(params, [0.0, 3600.0], x0, atol=1e-12, rtol=1e-12)

res.t      # (N,)   times, s
res.x      # (N, 6) states
res.stm    # None here; (N, 6, 6) for solve_ivpstm_*
```

`atol`, `rtol` and `t_eval` are keyword-only and optional. The default tolerances are `atol=1e-12` and `rtol=1e-14`.


## `SpiceParameters`

```python
campanella.SpiceParameters(
    et0, naif_ids, *,
    naif_frame="J2000",            # inertial frame of integration
    GMs=None,                      # km^3/s^2, one per NAIF id; None reads "GM" from the kernel pool
    DU=None, use_canonical_scales=True,
    abcorr="NONE",
    include_srp=False, srp_A=0.0, srp_m=0.0, srp_rhoS=0.0, srp_rhoD=0.0,
    sh_file="", sh_nmax=0, sh_mmax=-1, sh_frame="", sh_rho_min=1.0,
)
```

- **Bodies.** `naif_ids` are NAIF names or numeric ids, e.g. `["MOON", "EARTH", "SUN"]` or `["301", "399", "10"]`. The first one is the central body; the others are third bodies, whose ephemerides are read from the loaded SPK kernels. A body may be listed only once.
- **Gravitational parameters.** `GMs` are always in km^3/s^2, also in canonical units. When `GMs` is omitted, each body's `GM` is read from the kernel pool, for example `gm_de440.tpc`. A body without a GM raises `ValueError`.
- **Frame.** `naif_frame` must be an inertial frame. A rotating frame such as `IAU_EARTH` raises `ValueError`.
- **Aberration correction.** `abcorr` must be `"NONE"`. The force model uses geometric positions.
- **SRP.** `include_srp=True` requires the Sun (`"10"` or `"SUN"`) among `naif_ids`. It may be the central body. `srp_A` is the area in m^2, `srp_m` the mass in kg, and `srp_rhoS` and `srp_rhoD` the reflection coefficients. SRP has no shadow model. To use SRP without the Sun's gravity, list the Sun with a GM of 0.
- **Gravity field of the central body.** `sh_file` turns it on. See [the gravity-field section](#central-body-gravity-field-spherical-harmonics).

The object is immutable after construction. Its inputs and derived quantities are read-only properties, for example `params.GMs`, `params.mus`, `params.DU`, `params.TU`, `params.VU`, `params.naif_codes` and `params.include_srp`.

With a single body, explicit `GMs`, no SRP and no gravity field, the propagation makes no ephemeris query, so no kernels need to be loaded. This is the two-body problem:

```python
params = campanella.SpiceParameters(0.0, ["EARTH"], GMs=[398600.4418])
res = campanella.solve_ivp_rv(params, [0.0, 86400.0], [7000.0, 0.0, 0.0, 0.0, 0.43, 7.5])
```


## Units

| Argument | `DU` not given, or `use_canonical_scales=False` | `DU` given |
|---|---|---|
| distance unit `DU` | 1 km | `DU` km |
| velocity unit `VU` | 1 km/s | `sqrt(GMs[0] / DU)` km/s |
| time unit `TU` | 1 s | `DU / VU` s |

`x0`, `tspan`, `t_eval` and every output are in DU, DU/TU and TU. `mus = GMs * TU^2 / DU^3` are the GMs in these units, so the central body's scaled GM is 1 when `DU` is given. `et0` is always in ephemeris seconds, and the ephemerides are queried at `et = et0 + t * TU`. SRP's `srp_A` (m^2) and `srp_m` (kg) stay physical.

A Moon-centred example where the lunar GM is 1:

```python
import numpy as np

params = campanella.SpiceParameters(et0, ["MOON", "EARTH", "SUN"], DU=100000.0)
DU, VU, TU = params.DU, params.VU, params.TU          # TU is about 5.2 days

state0_canon = np.concatenate((np.array(state0[:3]) / DU, np.array(state0[3:]) / VU))
res = campanella.solve_ivp_rv(params, [0.0, 20*86400.0 / TU], list(state0_canon))
```

To convert results back:

- Cartesian states: multiply positions by `DU` and velocities by `VU`, and times by `TU`.
- MEE: multiply only `p` by `DU`. `f, g, h, k, L` are dimensionless.
- STMs: `Phi_km = D @ Phi_canon @ inv(D)`, with `D = diag(DU, DU, DU, VU, VU, VU)` for Cartesian states and `D = diag(DU, 1, 1, 1, 1, 1)` for MEE.

`atol` applies to the scaled variables, so `atol = 1e-12` corresponds to 1e-7 km at `DU = 100000`. `test-python/test_canonical_moon.py` and `examples/solve_moon_canonical.cpp` check each solver in these units against the same run in km and seconds.


## Solvers

```python
res = campanella.solve_ivp_rv(params, tspan, x0, *, atol=1e-12, rtol=1e-14, t_eval=None)
```

- `tspan = [t0, tf]`. If `tf < t0`, the integration runs backward. If `tf == t0`, the result is the single row `x0`.
- `x0` always has 6 elements. The `solve_ivpstm_*` variants start the STM from `Phi(t0, t0) = I`.
- Without `t_eval`, every accepted step is returned, including `t0` and `tf`. With `t_eval`, only those times are returned, and they must be strictly monotonic in the direction of integration and lie within `tspan`. `t0` is included only if it is listed.
- The result has `t (N,)`, `x (N, 6)` and `stm (N, 6, 6)`, with `stm` being `None` for `solve_ivp_*`. The arrays are read-only views of the result.
- A failure raises an exception; a partial trajectory is never returned:
  - `ValueError` for bad input, for a trajectory that goes below `sh_rho_min` times the reference radius of the gravity field, or for MEE that stay out of their domain.
  - `RuntimeError` with NAIF's message for a SPICE error, for example missing ephemeris data.

Single-point evaluation of the equations of motion:

```python
xdot, A = campanella.eval_rv(params, t, x)     # A = d(xdot)/dx, (6, 6)
xdot, A = campanella.eval_mee(params, t, x)
xdot = campanella.eom_rv(params, t, x)         # also eom_mee; eomstm_rv / eomstm_mee take 42 states
```


## Modified equinoctial elements

`solve_ivp_mee` integrates the Gauss variational equations in MEE `[p, f, g, h, k, L]`. The formulas are in [gve_mee.md](gve_mee.md). `mee2cart(mee, mu)` and `cart2mee(rv, mu)` convert between MEE and Cartesian states, where `mu` is `params.mus[0]`.

```python
import numpy as np

state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
          0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
params = campanella.SpiceParameters(
    et0, ["MOON", "EARTH", "SUN"],
    GMs=[4.902800582147760e+03, 398600.432896939, 132712197035.766],
    include_srp=True, srp_A=34.675, srp_m=1850.0, srp_rhoS=0.9,
)
mu = params.mus[0]
res = campanella.solve_ivpstm_mee(params, [0.0, 20*86400.0], campanella.cart2mee(state0, mu))
rv_final = campanella.mee2cart(list(res.x[-1]), mu)
stm_final = res.stm[-1]                                     # d(mee(tf)) / d(mee(t0))
```

Notes:

- **Singularity at 180 degrees.** The elements are prograde (retrograde factor I = +1) and singular at an inclination of 180 degrees, where h and k grow without bound. `cart2mee` raises `ValueError` within about 1.4e-6 rad of 180 degrees. The solver keeps integrating near that inclination, but accuracy degrades as the orbit approaches 180 degrees or as w = 1 + f cos L + g sin L approaches 0.
- **Domain.** A trial step that leaves the domain (p <= 0 or w <= 0) makes the integrator retry with a smaller step. If the state stays out of the domain, the solver raises `ValueError`.
- **True longitude.** `cart2mee` returns `L` in (-pi, pi]. The solver does not wrap `L`: it grows by 2 pi each revolution, and decreases in backward propagation. Compare `L` values through the wrapped difference.
- **Jacobian.** `campanella.mee2cart_jacobian(mee, mu)` returns d(r, v)/d(MEE) as a flattened 6x6 matrix. Use it to map the STM between MEE and Cartesian coordinates.
- **Tolerances.** `p` is in DU and the other elements are O(1), so the absolute tolerance acts differently on `p`. For long arcs, choose `DU` close to `p`.


## Central-body gravity field (spherical harmonics)

Give `sh_file`, `sh_nmax`, `sh_mmax`, `sh_frame` and `sh_rho_min` to `SpiceParameters`. For example, use `sh_file="gggrx_1200a_sha.tab", sh_nmax=50, sh_frame="MOON_PA"` for the Moon, or an EGM2008 `.gfc` file with `sh_frame="ITRF93"` for the Earth. `GMs[0]` must match the GM of the file within 1e-5. The file formats, the SPICE frame kernels and the GM policy are in [spherical_harmonics.md](spherical_harmonics.md).


## Migrating from the `Propagator*` classes

| Before | Now |
|---|---|
| `PropagatorTwoBody(GM).solve_ivp(tspan, y0)` | `solve_ivp_rv(SpiceParameters(0.0, ["399"], GMs=[GM]), tspan, x0)` |
| `PropagatorNbody(GM, BODIES, frame, LU, TU).solve_ivp(et0, tspan, y0)` | `solve_ivp_rv(SpiceParameters(et0, BODIES, naif_frame=frame, GMs=GM_km3s2, DU=...), tspan, x0)` |
| `PropagatorNbody.solve_ivp_stm(et0, tspan, y0 + list(I))` | `solve_ivpstm_rv(params, tspan, x0)`, then `res.stm` |
| `PropagatorNbodySrp(..., A, m, rhoS, rhoD, ...)` | `SpiceParameters(..., include_srp=True, srp_A=A, srp_m=m, srp_rhoS=rhoS, srp_rhoD=rhoD)`; the Sun must be listed |
| `PropagatorGve(...).solve_ivp` / `.solve_ivp_stm` | `solve_ivp_mee` / `solve_ivpstm_mee` |
| `PropagatorGve.eval_jacobian(et0, t, x)` | `eval_mee(params, t, x)`; `eval_rv` for Cartesian states |
| `GM` in LU^3/TU^2 | `GMs` always in km^3/s^2; `params.mus` are the scaled values |
| `LU`, `TU` | `DU` (with `use_canonical_scales`); `TU` and `VU` are derived |
| `sh_file, sh_nmax, sh_mmax, frame_bf, sh_rho_min` | `sh_file, sh_nmax, sh_mmax, sh_frame, sh_rho_min` |
| `reltol`, `abstol` | `rtol`, `atol` (keyword-only), and `t_eval` |
| list of tuples `(t, x..., stm...)` | `res.t (N,)`, `res.x (N, 6)`, `res.stm (N, 6, 6)` |
| `A = 0` turns SRP off (GVE only) | `include_srp=False`, for both coordinate sets; the Cartesian STM now includes SRP |
| `tf < t0` rejected; a failed integration printed to stderr and returned a truncated trajectory | backward propagation supported; failures raise |
| `sh_accel_inertial(path, nmax, frame, frame_bf, ..., LU, TU)` | `sh_accel_inertial(path, nmax, naif_frame, sh_frame, ..., DU, TU)` |
