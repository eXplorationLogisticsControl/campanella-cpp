# Equations of motion, frames and units

This page states the equations that `campanella` integrates, as implemented. File and line
references are to the sources under `src/`. The modified equinoctial element (MEE) form of the same
dynamics is in [Gauss variational equations in MEE](gve_mee.md), and the gravity field of the central
body in [Central-body spherical harmonics](spherical_harmonics.md). The
[tutorial](tutorial.md) shows the model in use.


## Overview

A `SpiceParameters` object fixes a list of bodies $(B_0, B_1, \dots, B_{N-1})$ given by their NAIF
IDs. $B_0$ is the *central body*; $B_1, \dots, B_{N-1}$ are *third bodies*. The spacecraft state is
its position $\boldsymbol{r}$ and velocity $\boldsymbol{v}$ relative to the centre of $B_0$, expressed
in an inertial frame. The equations of motion are (`rv_evaluate_ext`, `src/eoms/eoms_rv.cpp:35-49`)

$$
\dot{\boldsymbol{r}} = \boldsymbol{v}, \qquad
\dot{\boldsymbol{v}} = -\frac{\mu_0}{r^3}\boldsymbol{r}
  + \sum_{i=1}^{N-1} \boldsymbol{a}_{i}(\boldsymbol{r}, t)
  + \boldsymbol{a}_\text{SRP}(\boldsymbol{r}, t)
  + \boldsymbol{a}_\text{SH}(\boldsymbol{r}, t),
$$

where $\mu_i$ is the gravitational parameter of $B_i$. The perturbations are added in this order by
`add_perturbations` (`src/eoms/forces.cpp:85-158`): third bodies, then solar radiation pressure (if
`include_srp`), then central-body spherical harmonics (if `sh_file` is given). None of the
accelerations depends on $\boldsymbol{v}$.

The MEE solvers use the same `add_perturbations` for the perturbing acceleration
(`src/eoms/eoms_mee.cpp`), so both coordinate sets integrate the same physics.

The model does not include: general relativity, atmospheric drag, tides, the gravity fields of the
third bodies (they are point masses), Earth albedo or infrared radiation, or eclipses.

This page describes the natural forces, which are all that `solve_ivp_rv` and `solve_ivp_mee` integrate.
Thrust and a mass state are added by the separate `rvm` and `meem` dynamics, on the page
[Mass-augmented dynamics under zero-order-hold control](zoh_mass_dynamics.md).


## Reference frames

**Integration frame.** States are expressed in `naif_frame` (default `"J2000"`). Its axes are fixed;
its origin is the centre of the central body, which is accelerated, so the frame is not inertial in
the strict sense. The third-body term below contains the corresponding *indirect* acceleration.
`SpiceParameters` rejects a frame that is not of SPICE's inertial class, e.g. `"IAU_EARTH"`
(`check_inertial_frame`, `src/spice_parameters.cpp:48-62`). Other admissible frames include
`"ECLIPJ2000"`. In short: the integration frame is *non-rotating*, and "inertial" on these pages
means exactly that.

The SPICE frame `J2000` has its $z$ axis along the Earth's mean rotation axis at the J2000 epoch and
its $x$ axis towards the mean vernal equinox of that epoch ("mean": with nutation averaged out).
NAIF treats it as equivalent to the ICRF (International Celestial Reference Frame, defined by the
positions of distant quasars) {cite:p}`naif_frames`, the frame in which the DE ephemerides are
given {cite:p}`park2021`. Because the
Earth's axis precesses, the J2000 equator differs from the true equator of date (by 0.14° in 2025).

**Body positions.** The position $\boldsymbol{s}_i$ of each third body relative to the central body
is queried at every right-hand-side evaluation with `spkgps_c(code_i, et, naif_frame, code_0)`
(`src/eoms/forces.cpp:96`). These are *geometric* positions: where the bodies are at the instant
`et`. SPICE can instead return *apparent* positions, corrected for light time (where the body was
when the light now arriving left it) and stellar aberration (the apparent shift due to the
observer's velocity); these are what an observer sees, and they are the wrong input for a force
model, where gravity is evaluated with all positions at the same instant. `abcorr` must therefore be `"NONE"` (`src/spice_parameters.cpp:89`).

**Body-fixed frames.** A body-fixed (rotating) frame enters only through the spherical-harmonics
term, whose coefficients are given in `sh_frame` (e.g. `ITRF93` for the Earth, `MOON_PA` for the
Moon). With $T(t)$ the rotation from `naif_frame` to `sh_frame` returned by `pxform_c`
(`src/perturbations/spherical_harmonics.cpp:772`),

$$
\boldsymbol{r}_\text{bf} = T\boldsymbol{r}, \qquad
\boldsymbol{a}_\text{SH} = T^\mathsf{T}\boldsymbol{a}_\text{bf}(\boldsymbol{r}_\text{bf}), \qquad
G_\text{SH} = T^\mathsf{T} G_\text{bf}\, T
$$

(`src/perturbations/spherical_harmonics.cpp:779-803`). The orientation data come from the PCK and FK
kernels listed in [spherical_harmonics.md](spherical_harmonics.md#frames).


## Time

`et0` is in ephemeris time: TDB seconds past J2000 (2000-01-01 12:00:00 TDB), the time argument of
the SPICE ephemerides {cite:p}`naif_time`. The independent variable $t$ of the integration is
measured from `et0` in time units TU, and every ephemeris and orientation query is made at

$$
\mathrm{et} = \mathrm{et}_0 + t \cdot \mathrm{TU}
$$

(`src/eoms/forces.cpp:90`). In km/s units TU = 1 s. `tspan`, `t_eval` and the output times `res.t`
are all values of $t$. Converting a calendar date in UTC to ET requires a leapseconds kernel; see the
[tutorial](tutorial.md#4-time-et-and-tdb).


## Central body

The central body is a point mass, plus the optional spherical-harmonics series of degrees 2 and above:

$$
\boldsymbol{a}_0 = -\frac{\mu_0}{r^3}\boldsymbol{r}
\qquad (\texttt{src/eoms/eoms\_rv.cpp:35}).
$$

Its gradient (`gravity_gradient_center`, `src/eoms/gravity_gradient.cpp:16-39`) is

$$
G_0 = \frac{\partial \boldsymbol{a}_0}{\partial \boldsymbol{r}}
    = \frac{\mu_0}{r^3}\left(\frac{3\,\boldsymbol{r}\boldsymbol{r}^\mathsf{T}}{r^2} - I_3\right).
$$


## Third-body point masses

The acceleration of the spacecraft relative to the central body, due to a third body $B_i$ at
$\boldsymbol{s}_i$ (relative to the central body), is the difference between the body's pull on the
spacecraft and its pull on the central body:

$$
\boldsymbol{a}_i
 = -\mu_i\left(\frac{\boldsymbol{d}_i}{\|\boldsymbol{d}_i\|^3} + \frac{\boldsymbol{s}_i}{\|\boldsymbol{s}_i\|^3}\right),
 \qquad \boldsymbol{d}_i = \boldsymbol{r} - \boldsymbol{s}_i .
$$

The first term is the *direct* term; the second, which does not depend on $\boldsymbol{r}$, is the
*indirect* term: the acceleration of the frame origin. Omitting it would describe a spacecraft
orbiting a central body that is held fixed, which is wrong by the full pull of the third body on the
central body (for an Earth orbiter, the whole $6\times10^{-6}$ km/s² of solar gravity instead of
a tidal difference of order $10^{-9}$ km/s²). The spacecraft's own pull on the bodies is neglected.

When $r \ll s_i$, the two terms nearly cancel and evaluating them directly loses digits. The code
uses Battin's equivalent form {cite:p}`battin1999` (`thirdbody_accel`,
`src/perturbations/thirdbody.cpp:29-33`):

$$
\boldsymbol{a}_i = -\frac{\mu_i}{\|\boldsymbol{d}_i\|^3}\left(\boldsymbol{r} + F(q_i)\,\boldsymbol{s}_i\right),
\qquad
q_i = \frac{\boldsymbol{r}\cdot(\boldsymbol{r} - 2\boldsymbol{s}_i)}{\boldsymbol{s}_i\cdot\boldsymbol{s}_i},
\qquad
F(q) = q\,\frac{3 + 3q + q^2}{1 + (1+q)^{3/2}} .
$$

It follows from the direct-plus-indirect form: with $\boldsymbol{d}_i = \boldsymbol{r} - \boldsymbol{s}_i$,

$$
-\mu_i\left(\frac{\boldsymbol{d}_i}{d_i^3} + \frac{\boldsymbol{s}_i}{s_i^3}\right)
= -\frac{\mu_i}{d_i^3}\left(\boldsymbol{r} + \left[\frac{d_i^3}{s_i^3} - 1\right]\boldsymbol{s}_i\right),
\qquad \frac{d_i^2}{s_i^2} = 1 + q_i ,
$$

so $F(q) = (1+q)^{3/2} - 1$. Computing this difference directly would again cancel for small $q$;
multiplying and dividing by $(1+q)^{3/2} + 1$ gives the form above, in which nothing cancels. The direct form
loses roughly $\log_{10}(s_i/r)$ significant digits. For a geostationary orbit and the Moon
($s_i/r \approx 9$) that is about one digit, and the two forms agree to about $10^{-14}$. For a low
Earth orbit and the Sun it is about four digits: at $r = 7000$ km, the direct form evaluated in double
precision has a relative error of $5\times10^{-12}$ against a 40-digit reference, the Battin form
$2\times10^{-16}$.

$\mu_i$ is `params.mus[i]` and $\boldsymbol{s}_i$ is the SPICE position divided by DU
(`src/eoms/forces.cpp:100-104`). The gradient (`gravity_gradient_third_body`,
`src/eoms/gravity_gradient.cpp:50-78`) is that of the direct term alone:

$$
G_i = \frac{\partial \boldsymbol{a}_i}{\partial \boldsymbol{r}}
    = \frac{\mu_i}{\|\boldsymbol{d}_i\|^5}\left(3\,\boldsymbol{d}_i\boldsymbol{d}_i^\mathsf{T} - \|\boldsymbol{d}_i\|^2 I_3\right).
$$

A body with $\mu_i = 0$ is allowed (only $\mu_0 > 0$ is required, `src/spice_parameters.cpp:112-117`).
It contributes no gravity, but its position is still queried; this is how to use SRP without the
Sun's gravity.


## Solar radiation pressure

`include_srp=True` adds a cannonball-type solar radiation pressure acceleration
(`src/eoms/forces.cpp:113-136`, `src/perturbations/SRP.cpp`). With

$$
\boldsymbol{d} = \boldsymbol{r} - \boldsymbol{s}_\odot, \qquad R = \|\boldsymbol{d}\|,
$$

the position of the spacecraft relative to the Sun ($\boldsymbol{s}_\odot = \boldsymbol{0}$ if the Sun
is the central body), the acceleration is

$$
\boldsymbol{a}_\text{SRP} = \frac{K}{R^3}\,\boldsymbol{d}
 = \frac{P_0}{c}\, C_r\, \frac{A}{m} \left(\frac{\mathrm{AU}}{R}\right)^2 \hat{\boldsymbol{d}},
\qquad C_r = 1 + \rho_S + \tfrac{5}{3}\rho_D,
$$

directed away from the Sun. The constants are (`src/perturbations/SRP.cpp:11-13`):

| symbol | value | meaning |
|---|---|---|
| $P_0$ | 1367 W/m² | solar flux at 1 AU |
| $c$ | 299792458 m/s | speed of light |
| AU | 149597870.7 km | astronomical unit |

and the inputs are $A$ = `srp_A` (m²), $m$ = `srp_m` (kg), $\rho_S$ = `srp_rhoS` and $\rho_D$ = `srp_rhoD`.
The coefficient $K$, in DU³/TU², is computed once in `SpiceParameters` by `srp_coefficient`
(`src/perturbations/SRP.cpp:31-35`, called at `src/spice_parameters.cpp:196`):

$$
K = \frac{P_0}{c}\, C_r\, \frac{A}{m}\left(\frac{\mathrm{AU}}{\mathrm{DU}}\right)^2
    \Big/ \left(10^3\,\frac{\mathrm{DU}}{\mathrm{TU}^2}\right),
$$

where the factor $10^3\,\mathrm{DU}/\mathrm{TU}^2$ converts m/s² to DU/TU². At 1 AU the acceleration is
$(P_0/c)\,C_r\,A/m$ m/s², about $4.56\times10^{-6}\,C_r\,A/m$ m/s² with $A/m$ in m²/kg.

The `rvm` and `meem` dynamics do not use this value: they ignore `srp_m` and recompute $K$ at every
evaluation with the state mass $\bar m\, m$ (`src/eoms/eoms_rvm.cpp:41-45`, `src/eoms/eoms_meem.cpp:55-58`;
see [the mass-augmented dynamics](zoh_mass_dynamics.md#solar-radiation-pressure)), but `srp_m` must
still be $> 0$ when `include_srp` is on (`src/spice_parameters.cpp:146-148`).

Properties of the model as implemented:

- **Direction.** Always along $\hat{\boldsymbol{d}}$, from the Sun to the spacecraft, independent of
  attitude. (Checked numerically: the SRP acceleration returned by `eval_rv` is parallel to
  $\boldsymbol{r} - \boldsymbol{s}_\odot$ and matches the formula above to $10^{-10}$.)
- **Flux scaling.** The flux scales with the inverse square of the actual Sun–spacecraft distance,
  referenced to 1 AU.
- **No shadow.** There is no eclipse model: the spacecraft is always fully lit, including inside the
  shadow of the central body or a third body.
- **Sun position.** The Sun must be one of the bodies (`src/spice_parameters.cpp:138-141`); its
  position is the one already queried for its third-body term, so SRP costs no extra ephemeris call.
  To have SRP without solar gravity, give the Sun a GM of 0.
- **No indirect term.** Radiation pressure on the central body is not modelled; only the spacecraft
  is pushed.
- **Reflectivity.** $\rho_S$ and $\rho_D$ are the fractions of incident light reflected specularly
  and diffusely; the absorbed fraction is $\rho_A = 1 - \rho_S - \rho_D$. The library's
  $C_r = 1 + \rho_S + \tfrac{5}{3}\rho_D$ **differs from the textbook flat-plate value**. For a flat
  plate facing the Sun, each absorbed or reflected photon transfers its incident momentum (the 1),
  specular reflection adds the same again ($+\rho_S$), and Lambertian diffuse re-emission adds
  $\tfrac{2}{3}$ of it ($+\tfrac{2}{3}\rho_D$), so $C_r = 1 + \rho_S + \tfrac{2}{3}\rho_D$
  {cite:p}`milani1987`. The library's coefficient $\tfrac{5}{3}$ equals this
  $1 + \tfrac{2}{3}$ counted on top of the incident momentum that is already in the leading 1. The
  two agree when $\rho_D = 0$. To reproduce a textbook result or a prescribed $C_r$, set
  `srp_rhoD = 0` and `srp_rhoS = C_r - 1`.

The gradient (`srp_gradient`, `src/perturbations/SRP.cpp:48-56`) is, with $\kappa = K/R^3$,

$$
G_\text{SRP} = \frac{\partial \boldsymbol{a}_\text{SRP}}{\partial \boldsymbol{r}}
 = \kappa\left(I_3 - \frac{3\,\boldsymbol{d}\boldsymbol{d}^\mathsf{T}}{R^2}\right),
$$

that of a repulsive inverse-square field.


## Central-body spherical harmonics

With `sh_file`, the acceleration of degrees 2 to `sh_nmax` of the central body's gravity field is
added (`src/eoms/forces.cpp:138-156`). It is evaluated in `sh_frame` with the fully normalized
V, W recursion of Cunningham {cite:p}`cunningham1970,montenbruck2000` and rotated to `naif_frame` as
described [above](#reference-frames). The series uses the GM and reference radius of the coefficient
file, scaled as $\mu_\text{SH} = \mathrm{GM}_\text{file}\,\mathrm{TU}^2/\mathrm{DU}^3$ and
$R_\text{SH} = R_\text{file}/\mathrm{DU}$ (`src/spice_parameters.cpp:183-184`). Positions below
`sh_rho_min` times the reference radius are rejected. Everything else is in
[spherical_harmonics.md](spherical_harmonics.md).


## Units and canonical scaling

Without `DU`, or with `use_canonical_scales=False`, DU = 1 km, TU = 1 s and VU = 1 km/s. With `DU`
given (`src/spice_parameters.cpp:120-133`):

$$
\mathrm{VU} = \sqrt{\mathrm{GM}_0/\mathrm{DU}}, \qquad
\mathrm{TU} = \mathrm{DU}/\mathrm{VU} = \sqrt{\mathrm{DU}^3/\mathrm{GM}_0}, \qquad
\mu_i = \mathrm{GM}_i\,\frac{\mathrm{TU}^2}{\mathrm{DU}^3} = \frac{\mathrm{GM}_i}{\mathrm{GM}_0},
$$

with $\mathrm{GM}_i$ in km³/s², so $\mu_0 = 1$. The code computes $\mu_i$ as the ratio
$\mathrm{GM}_i/\mathrm{GM}_0$ (`src/spice_parameters.cpp:132`). Every quantity inside the equations
of motion is then in DU, TU and DU/TU:

| quantity | scaling |
|---|---|
| state, `tspan`, `t_eval`, outputs | DU, DU/TU, TU |
| third-body positions | SPICE km divided by DU (`src/eoms/forces.cpp:100-102`) |
| ephemeris time | $\mathrm{et}_0 + t\,\mathrm{TU}$ |
| SRP | $K$ in DU³/TU² (see above); `srp_A` and `srp_m` stay in m² and kg (`rvm` and `meem` use $\bar m\, m$ in kg instead of `srp_m`) |
| spherical harmonics | $\mu_\text{SH}$, $R_\text{SH}$ (see above) |
| Cartesian STM | $\Phi_\text{km} = D\,\Phi\,D^{-1}$, $D = \operatorname{diag}(\mathrm{DU},\mathrm{DU},\mathrm{DU},\mathrm{VU},\mathrm{VU},\mathrm{VU})$ |
| MEE STM | the same with $D = \operatorname{diag}(\mathrm{DU}, 1, 1, 1, 1, 1)$ |

The equations are invariant under this scaling: for the same physical state, the canonical
acceleration times $\mathrm{DU}/\mathrm{TU}^2$ equals the km/s acceleration (checked to
$5\times10^{-16}$ relative). What changes is the conditioning and the meaning of `atol`, which
applies to the scaled variables.


## Variational equations and the STM

The state-transition matrix $\Phi(t, t_0) = \partial\boldsymbol{x}(t)/\partial\boldsymbol{x}(t_0)$
satisfies

$$
\dot\Phi(t, t_0) = A(t)\,\Phi(t, t_0), \qquad \Phi(t_0, t_0) = I_6,
\qquad A = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\bigg|_{\boldsymbol{x}(t),\,t} .
$$

Because no acceleration depends on the velocity, the Cartesian Jacobian is
(`src/eoms/eoms_rv.cpp:55-68`)

$$
A = \begin{pmatrix} 0_3 & I_3 \\ G & 0_3 \end{pmatrix}, \qquad
G = G_0 + \sum_{i=1}^{N-1} G_i + G_\text{SRP} + G_\text{SH},
$$

with the gradients given above, accumulated in the same order as the accelerations. The explicit
time dependence (body motion, frame rotation) does not enter $A$: $\Phi$ is a derivative with respect
to the state at fixed time. This does not mean the motion of the Moon is ignored: $A(t)$ is
evaluated along the trajectory with the bodies where they are at each $t$, so $\Phi(t, t_0)$ fully
accounts for the time-varying forces. What it does not contain is a derivative with respect to
time, or to the epoch: shifting $\mathrm{et}_0$ is not a state perturbation. `rhsstm_rv` (`src/eoms/eoms_rv.cpp:103-121`) exploits the block structure:
the top three rows of $\dot\Phi$ are the bottom three rows of $\Phi$, and the bottom rows are
$G\,\Phi_{0:3,\,:}$.

The solvers integrate 42 variables: the state followed by $\Phi$ in row-major order, starting from
the identity (`src/integrate.cpp:112-122`). GSL's step-size control applies the same `atol` and `rtol`
to all 42 components, so the STM also constrains the step size. The sensitivity solvers of the
[mass-augmented dynamics](zoh_mass_dynamics.md#sensitivities) integrate 84 components in the same way.

For MEE, $A$ is the analytic Jacobian of the Gauss variational equations, which includes the same
gradient $G$ rotated into the RTN frame; see {ref}`section 5 of gve_mee.md <gve-jacobian>`.
To map an MEE STM to Cartesian coordinates, use the Jacobian $J = \partial(\boldsymbol{r},\boldsymbol{v})/\partial\,\text{MEE}$
returned by `mee2cart_jacobian`:
$\Phi_\text{cart}(t, t_0) = J(\boldsymbol{x}(t))\,\Phi_\text{MEE}(t, t_0)\,J(\boldsymbol{x}_0)^{-1}$.


## Numerical integration

The equations are integrated with GSL's `gsl_odeiv2_step_rk8pd`, which GSL calls the explicit
embedded Runge–Kutta Prince–Dormand (8, 9) method {cite:p}`galassi2009`. It is the 13-stage pair of
{cite:t}`prince1981`: the solution is advanced with the 8th-order formula, and the local error is
estimated from the difference to an embedded 7th-order formula (an 8(7) pair; `rk8pd.c` in the GSL
sources). It is used with the standard step-size control
`gsl_odeiv2_control_y_new(atol, rtol)` (`src/integrate.cpp:127-128`). With the desired error level

$$
D_i = \epsilon_\text{abs} + \epsilon_\text{rel}\,|y_i| ,
$$

GSL rejects a step and retries with a smaller one when the local error estimate of any component
exceeds $D_i$ by more than 10%, and lengthens the next step when all errors are well below $D_i$
{cite:p}`galassi2009`.

The first trial step is 1 TU in the direction of integration (`src/integrate.cpp:168`), and GSL
adapts it from there. The defaults are `atol = 1e-12` and `rtol = 1e-14`. Outputs at `t_eval` are
obtained by stepping exactly onto each requested time, not by interpolation
(`src/integrate.cpp:186-212`).

A right-hand-side evaluation that fails is handled in one of two ways
(`src/model.h`, `src/eoms/forces.cpp:19-24`):

- a state out of the domain (MEE with $p \le 0$ or $w \le 0$, a mass that is not $> 0$ in `rvm` and `meem`,
  a position below `sh_rho_min`, or a non-finite derivative) returns `GSL_EDOM`, and GSL retries with a smaller step. If GSL stops while
  the last evaluation was out of the domain, the solver throws `std::domain_error` (`ValueError` in
  Python), e.g. "MEE out of domain" or the `sh_rho_min` message;
- a SPICE error, e.g. an epoch outside the loaded SPK, returns `GSL_EBADFUNC`, which stops the
  integration; the solver throws `std::runtime_error` (`RuntimeError`) with NAIF's message.

GSL can also give up on its own when the step size it needs becomes too small, for example for a
trajectory that falls into the central body: the evaluations near the singularity succeed, but the
error estimate cannot be met. The solver then throws `std::runtime_error` (`RuntimeError`):
"integration failed (GSL status -1: failure) near t = ... TU".

The `rvm` and `meem` solvers also stop on mass depletion, before integrating, and at `max_steps`; see
[Integration limits](zoh_mass_dynamics.md#integration-limits).

No solver returns a partial trajectory.
