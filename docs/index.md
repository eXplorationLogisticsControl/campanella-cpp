# campanella

`campanella` is a C++ library, with Python bindings, for propagating spacecraft trajectories in a
high-fidelity ephemeris model. It integrates the equations of motion, and optionally the
state-transition matrix (STM), in either Cartesian coordinates or modified equinoctial elements (MEE),
with:

- point-mass gravity of a central body and any number of third bodies, with positions from NAIF SPICE ephemerides
- solar radiation pressure (SRP) with a cannonball model
- central-body spherical harmonics of arbitrary degree and order
- dimensional (km, s) or canonical units

The integrator is the embedded Runge-Kutta Prince-Dormand method of the GNU Scientific Library (GSL),
`rk8pd`, which GSL labels "(8, 9)": at each step it computes an 8th-order solution and an embedded
7th-order one, and uses their difference to estimate the error and choose the step size.

## A first example

```python
import campanella

campanella.furnsh("/path/to/lsk/naif0012.tls")
campanella.furnsh("/path/to/spk/de440.bsp")
campanella.furnsh("/path/to/pck/gm_de440.tpc")

et0 = 740463475.5117936                           # TDB seconds past J2000 (2023-06-19T16:16:46 UTC)
params = campanella.SpiceParameters(et0, ["399", "301", "10"])   # Earth (central), Moon, Sun
res = campanella.solve_ivp_rv(params, [0.0, 3600.0], [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
print(len(res), res.x[-1])   # km, km/s
```

```text
58 [-4.63814609e+03 -5.05222702e+03  9.28376161e-04  5.59293347e+00
 -5.22692303e+00 -9.47383772e-07]
```

NAIF ids are strings: names like `"EARTH"` or codes like `"399"`.

New users should start with the [tutorial](tutorial.md).

```{toctree}
:maxdepth: 1
:caption: Getting started

build_library
tutorial
basic_tutorials
```

```{toctree}
:maxdepth: 1
:caption: Theory

theory
```

```{toctree}
:maxdepth: 1
:caption: Reference

api_reference
bibliography
```
