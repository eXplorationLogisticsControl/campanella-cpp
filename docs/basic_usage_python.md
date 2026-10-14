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

