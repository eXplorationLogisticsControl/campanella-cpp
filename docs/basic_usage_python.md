# Basic usage through python

## Solving initial value problems

The basic structure for solving initial value problems with `campanella` is to (1) initialize a propagator object, then (2) use the `solve_ivp` method.

The integration is performed using an Explicit embedded Runge-Kutta Prince-Dormand (8, 9) method ([`gsl_odeiv2_step_rk8pd` from GNU GSL](https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_step_type.gsl_odeiv2_step_rk8pd)), with [step-size control](https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_control_y_new) given by

$$
D_i=\epsilon_{a b s} s_i+\epsilon_{\text {rel }} *\left(a_y\left|y_i\right|+a_{d y d t} h\left|y_i\right|\right)
$$


### Two-body equations of motion

```python
# make sure path to .pyd is included for importing the library!
import campanella

# N-body propagation reads ephemerides from campanella's own CSPICE.
# spiceypy.furnsh does not load those kernels.
campanella.furnsh("/path/to/lsk/naif0012.tls")
campanella.furnsh("/path/to/spk/de440.bsp")

state0 = [7000, 0.0, 0.0, 0.0, 0.43, 7.5]
tspan = [0.0, 86400.0]
GM = 398600.44
prop = campanella.PropagatorTwoBody(GM)
traj = prop.solve_ivp(
    tspan = tspan,
    y0 = state0,
)
```

The `solve_ivp` method returns a length-n list of length-7 tuples, where each tuples contains: [t,x,y,z,vx,vy,vz]:

```
traj = [
    (t0, x0, y0, z0, vx0, vy0, vz0),
    (t1, x1, y1, z1, vx1, vy1, vz1),
    ...
    (tn, xn, yn, zn, vxn, vyn, vzn)
]
```

To change settings on tolerances, we can initialize the propagator object via

```python
prop = campanella.PropagatorTwoBody(GM, reltol=1e-12, abstol=1e-11)
```

Default values for the tolerances are `reltol=1e-14` and `abstol=1e-12`. 



### N-body + SRP in modified equinoctial elements

`PropagatorGve` integrates the Gauss variational equations in modified equinoctial elements (MEE) `[p, f, g, h, k, L]`. Its perturbations are the same as `PropagatorNbodySrp`: `BODIES[0]` is the central body, the other bodies are third bodies, and SRP is added unless `A = 0`. The constructor takes the same arguments as `PropagatorNbodySrp`. The formulas are in [gve_mee.md](gve_mee.md).

`mee2cart(mee, mu)` and `cart2mee(rv, mu)` convert between MEE and Cartesian states. `mu` is the central body's `GM[0]`.

```python
import numpy as np
import campanella

campanella.furnsh("/path/to/lsk/naif0012.tls")
campanella.furnsh("/path/to/spk/de440.bsp")

et0 = 740463475.5117936
state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
          0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
GM = [4.902800582147760e+03, 398600.432896939, 132712197035.766]

prop = campanella.PropagatorGve(
    GM = GM,
    BODIES = ["MOON", "EARTH", "SUN"],
    frame = "J2000",
    A = 34.675, m = 1850, rhoS = 0.9, rhoD = 0,
    LU = 1.0, TU = 1.0,
)
mee0 = campanella.cart2mee(state0, GM[0])
traj = prop.solve_ivp(et0, [0.0, 20*86400.0], mee0)        # rows (t, p, f, g, h, k, L)
rv_final = campanella.mee2cart(list(traj[-1][1:]), GM[0])   # [x, y, z, vx, vy, vz]

# state and 6x6 STM d(mee(t))/d(mee(t0)), flattened row-major after the state
traj_stm = prop.solve_ivp_stm(et0, [0.0, 20*86400.0], list(mee0) + list(np.eye(6).flatten()))
stm_final = np.array(traj_stm[-1][7:]).reshape(6, 6)
```

Notes:

- The elements are prograde (retrograde factor I = +1). They are singular at an inclination of 180 degrees, where h and k grow without bound. `cart2mee` raises `ValueError` within about 1.4e-6 rad of 180 degrees. The propagator has no singularity check: it keeps integrating, and accuracy degrades as the orbit approaches 180 degrees or as w = 1 + f cos L + g sin L approaches 0. A trial step that leaves the domain (p <= 0 or w <= 0) only makes the integrator retry with a smaller step.
- `tspan` must be `[t0, tf]` with `tf >= t0`. Only forward propagation is supported; anything else raises `ValueError`.
- `cart2mee` returns `L` in (-pi, pi]. The propagator does not wrap `L`, so it grows by 2 pi each revolution. When you compare `L` values, compare the wrapped difference.
- `mee2cart` raises `ValueError` if `p <= 0` or `1 + f cos L + g sin L <= 0`.
- `campanella.mee2cart_jacobian(mee, mu)` returns d(r, v)/d(MEE) as a flattened 6x6 matrix. Use it to map the STM between MEE and Cartesian coordinates.
- `prop.eval_jacobian(et0, t, x)` returns `(xdot, A)`, the right-hand side and its 6x6 Jacobian (flattened) at a single state.
- `p` is in LU and the other elements are O(1), so the absolute tolerance applies differently to `p`. For long arcs, choose LU close to `p`.

### Canonical units

Every propagator takes `LU` (km) and `TU` (s). States, `tspan` and `GM` must then be given in those units, while `et0` stays in ephemeris seconds and SRP's `A` (m^2) and `m` (kg) stay physical. A common choice for a Moon-centred problem makes the lunar GM equal to 1:

```python
MU_MOON = 4.902800582147760e+03
LU = 100000.0                       # km
VU = np.sqrt(MU_MOON / LU)          # km/s
TU = LU / VU                        # s, about 5.2 days

GM_canon = [gm / MU_MOON for gm in GM]          # = gm * TU**2 / LU**3
state0_canon = np.concatenate((np.array(state0[:3]) / LU, np.array(state0[3:]) / VU))
tspan_canon = [0.0, 20*86400.0 / TU]

prop = campanella.PropagatorGve(
    GM = GM_canon, BODIES = ["MOON", "EARTH", "SUN"], frame = "J2000",
    A = 34.675, m = 1850, rhoS = 0.9, rhoD = 0,
    LU = LU, TU = TU,
)
mee0 = campanella.cart2mee(list(state0_canon), 1.0)    # p in LU
traj = prop.solve_ivp(et0, tspan_canon, mee0)
```

To convert results back:

- Cartesian states: multiply positions by `LU` and velocities by `VU`; times by `TU`.
- MEE: multiply only `p` by `LU`. `f, g, h, k, L` are dimensionless.
- STMs: `Phi_km = D @ Phi_canon @ inv(D)`, with `D = diag(LU, LU, LU, VU, VU, VU)` for Cartesian states and `D = diag(LU, 1, 1, 1, 1, 1)` for MEE.

`abstol` applies to the scaled variables, so `abstol = 1e-12` corresponds to 1e-7 km at `LU = 100000`. `test-python/test_canonical_moon.py` and `examples/solve_moon_canonical.cpp` check each Moon-centred propagator in these units against the same run in km and seconds.
