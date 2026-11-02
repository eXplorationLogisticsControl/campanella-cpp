# Mass-augmented dynamics under zero-order-hold control

This page specifies the `rvm` (Cartesian + mass) and `meem` (MEE + mass) dynamics, as implemented in
`src/eoms/eoms_rvm.cpp`, `src/eoms/eoms_meem.cpp` and `src/propagate_zoh.cpp`. They add a thrust
acceleration and a mass state to the natural forces of
[Equations of motion, frames and units](dynamics_model.md), and integrate the sensitivities that a
sequential convex programming or multiple-shooting solver needs for its per-segment linearization

$$
\boldsymbol x_{k+1} \approx \Phi_k \boldsymbol x_k + \Psi_k \boldsymbol u_k + \boldsymbol c_k .
$$

The functions are listed in the [C++ API](api_cpp.md#mass-augmented-dynamics) and the
[Python API](api_python.md). `examples/solve_zoh.cpp` is a complete multi-segment arc. Free final
time, costates and event detection are not implemented.


## Control and constants

The control is held constant over one integration (zero-order hold):

$$
\boldsymbol u = [\boldsymbol\tau^\mathsf{T},\ \Gamma]^\mathsf{T} \in \mathbb{R}^4, \qquad n_u = 4 .
$$

$\boldsymbol\tau$ is the dimensionless throttle vector, so the thrust is $T_{\max}\boldsymbol\tau$, and
$\Gamma$ is the dimensionless scale of the mass flow.

:::{important}
**The dynamics treat $\Gamma$ as independent of $\boldsymbol\tau$.** The intended relation is
$\Gamma = \|\boldsymbol\tau\|_2$, and enforcing it is the job of the caller or optimizer, not of the
dynamics. Neither $\Gamma < 0$ nor $\|\boldsymbol\tau\| > \Gamma$ is rejected, and $\Psi$ holds the
partial derivatives with respect to each of the four components. If $\Gamma = \|\boldsymbol\tau\|$ is
substituted, the derivative with respect to $\boldsymbol\tau$ at $\boldsymbol\tau \ne 0$ follows from the
chain rule, $\Psi_{:,\boldsymbol\tau} + \Psi_{:,\Gamma}\,\boldsymbol\tau^\mathsf{T}/\|\boldsymbol\tau\|$.
:::

With the mass unit $\bar m$ = `mass_unit_kg`, the constants are (`make_control`,
`src/solve_ivp.cpp:51-52`):

$$
c_1 = \frac{T_{\max}\,[\mathrm{kN}]}{\bar m\ \mathrm{DU}/\mathrm{TU}^2}, \qquad
c_2 = \frac{T_{\max}\,[\mathrm{N}]}{I_{sp}\, g_0}\cdot\frac{\mathrm{TU}}{\bar m} .
$$

| Symbol | Input | Unit | Meaning |
|---|---|---|---|
| $T_{\max}$ | `thrust_N` | N (kN in $c_1$, since kN/kg = km/s²) | maximum thrust |
| $I_{sp}$ | `isp_s` | s | specific impulse |
| $g_0$ | `g0`, default 9.80665 | m/s² | standard gravity |
| $\bar m$ | `mass_unit_kg` | kg | mass unit: $m = 1$ means $\bar m$ kilograms |
| $c_1$ | | DU/TU² | thrust acceleration at $m = 1$, $\boldsymbol\tau$ of unit norm |
| $c_2$ | | mass units per TU | mass flow at $\Gamma = 1$ |

All four inputs must be finite and positive. In C++ the constants are `ControlledParameters::c1()` and
`c2()`; Python does not expose them, so compute them from the formulas above with `params.DU` and
`params.TU`.


## Equations of motion

**Cartesian + mass (`rvm`)**, $\boldsymbol x = [\boldsymbol r, \boldsymbol v, m]$, in DU, DU/TU and mass
units, with $\boldsymbol\tau$ always in `naif_frame`:

$$
\dot{\boldsymbol r} = \boldsymbol v, \qquad
\dot{\boldsymbol v} = \boldsymbol a_g(\boldsymbol r, t) + \boldsymbol a_\text{srp}(\boldsymbol r, t, m) + \frac{c_1}{m}\boldsymbol\tau, \qquad
\dot m = -c_2\Gamma ,
$$

where $\boldsymbol a_g$ collects the central body, third bodies and spherical harmonics.

**MEE + mass (`meem`)**, $\boldsymbol x = [\boldsymbol x_e, m]$, $\boldsymbol x_e = [p, f, g, h, k, L]$, with
$\boldsymbol b$, $B_G$ (the matrix $B$ of [gve_mee.md](gve_mee.md)) and $C = [\hat{\boldsymbol r}\ \hat{\boldsymbol\theta}\ \hat{\boldsymbol n}]$:

$$
\dot{\boldsymbol x}_e = \boldsymbol b(\boldsymbol x_e) + B_G(\boldsymbol x_e)\left[C(\boldsymbol x_e)^\mathsf{T}\big(\boldsymbol a_p + \boldsymbol a_{c,I}\big) + \boldsymbol a_{c,\text{rtn}}\right],
\qquad \dot m = -c_2\Gamma .
$$

$\boldsymbol a_p$ holds the perturbations, with SRP from the state mass. The thrust acceleration
$c_1\boldsymbol\tau/m$ is $\boldsymbol a_{c,\text{rtn}}$ or $\boldsymbol a_{c,I}$, as set by
`ThrustOptions::mee_control_frame`:

| `ThrustFrame` | $\boldsymbol\tau$ is in | Note |
|---|---|---|
| `rtn` (default) | the radial, transverse and normal directions of the current state | a constant $\boldsymbol\tau$ rotates in inertial space |
| `inertial` | `naif_frame` | the same control as `rvm` |

**Domain.** A mass that is not $> 0$ or not finite is out of the domain, as are the MEE domain
$p \le 0$ or $w \le 0$ in `meem`. The mass is checked first. See
{ref}`Mass depletion <zoh-depletion>` for what the solvers do.

### Solar radiation pressure

`rvm` and `meem` always use the state mass in SRP. The coefficient $K$ of
[the SRP section](dynamics_model.md#solar-radiation-pressure) is recomputed at each evaluation with
the mass $\bar m\, m$ in place of `srp_m`, so $\boldsymbol a_\text{srp} \propto 1/m$
(`src/eoms/eoms_rvm.cpp:41-45`, `src/eoms/eoms_meem.cpp:55-58`).

`srp_m` is ignored by these dynamics, but `SpiceParameters` validates it regardless, so **it must
still be $> 0$ when `include_srp` is on**. `solve_ivp_rv` and `solve_ivp_mee` keep using `srp_m`. With
$\boldsymbol u = 0$ and `srp_m` $= \bar m\, m_0$, the state-only `rvm` and `meem` solvers return the same
bits as `rv` and `mee`.


## Sensitivities

Over a segment with fixed $\boldsymbol u$, with $A = \partial\boldsymbol f/\partial\boldsymbol x$ and
$B = \partial\boldsymbol f/\partial\boldsymbol u$ along the trajectory:

$$
\dot\Phi = A\Phi,\quad \Phi(t_0) = I_7, \qquad
\dot\Psi = A\Psi + B,\quad \Psi(t_0) = 0_{7\times4} .
$$

This form never computes $\Phi^{-1}$ and needs no quadrature. It equals
$\Psi(t) = \int_{t_0}^{t}\Phi(t, s)B(s)\,ds$.

### Augmented state layout

$n_x = 7$ and $n_u = 4$, so $n_x(1 + n_x + n_u) = 84$ components, row-major (`src/eoms/augment.h`):

| Elements | Content | Element $(i, j)$ |
|---|---|---|
| `y[0..6]` | $\boldsymbol x$, mass last | |
| `y[7..55]` | $\Phi$ (7x7) | `y[7 + 7*i + j]` |
| `y[56..83]` | $\Psi$ (7x4), columns $\tau_1, \tau_2, \tau_3, \Gamma$ | `y[56 + 4*i + j]` |

Solvers split this into `stm` and `stm_u`. Python returns them as `(N, 7, 7)` and `(N, 7, 4)`.

### Jacobians: `rvm`

$$
A = \begin{bmatrix}
0_3 & I_3 & 0_{3\times1}\\
G & 0_3 & -\dfrac{1}{m}\left(\dfrac{c_1}{m}\boldsymbol\tau + \boldsymbol a_\text{srp}\right)\\
0_{1\times3} & 0_{1\times3} & 0
\end{bmatrix},
\qquad
B = \begin{bmatrix}
0_3 & 0_{3\times1}\\
\dfrac{c_1}{m}I_3 & 0_{3\times1}\\
0_{1\times3} & -c_2
\end{bmatrix}.
$$

$G$ is the gradient of the natural forces, with $G_\text{srp}$ evaluated with the state mass. The full
products $A\Phi$ and $A\Psi + B$ are used, because the block shortcut of `eomstm_rv` would drop the mass
column.

### Jacobians: `meem`

$$
A = \begin{bmatrix} A_e & \partial\dot{\boldsymbol x}_e/\partial m\\ 0_{1\times6} & 0 \end{bmatrix},
\qquad
B = \begin{bmatrix} \dfrac{c_1}{m}B_G R & 0_{6\times1}\\ 0_{1\times3} & -c_2 \end{bmatrix},
\qquad
R = \begin{cases} I_3 & \text{RTN control}\\ C^\mathsf{T} & \text{inertial control}\end{cases}
$$

The block $A_e = A_b + A_B + B_G(G_\text{rtn}D_\text{rtn} + F)$ has the structure of
{ref}`section 5 of gve_mee.md <gve-jacobian>`. The two frames differ in how thrust enters it, because
RTN components of an inertial vector change with the state, whereas those of an RTN vector do not:

| Term | RTN thrust | Inertial thrust |
|---|---|---|
| $\dot{\boldsymbol x}_e$ | yes | yes |
| $A_B$ ($B_G$ differentiated at fixed $\boldsymbol a_\text{rtn}$) | total $\boldsymbol a_\text{rtn}$, thrust included | total $\boldsymbol a_\text{rtn}$, thrust included |
| $F = (\partial C^\mathsf{T}/\partial\boldsymbol x_e)\,\boldsymbol a_I$ | **not included** | **included**: added to $\boldsymbol a_I$ before projecting |
| $G_\text{rtn}D_\text{rtn}$ | none | none |

Thrust and SRP are both proportional to $1/m$, which gives the mass column:

$$
\frac{\partial\dot{\boldsymbol x}_e}{\partial m} =
\begin{cases}
-\dfrac{1}{m}B_G\left(C^\mathsf{T}\boldsymbol a_\text{srp} + \dfrac{c_1}{m}\boldsymbol\tau\right) & \text{RTN control}\\[2ex]
-\dfrac{1}{m}B_G C^\mathsf{T}\left(\boldsymbol a_\text{srp} + \dfrac{c_1}{m}\boldsymbol\tau\right) & \text{inertial control}
\end{cases}
$$

### Exact mass rows

$A$ has a zero mass row and $B$ has only $B_{m,\Gamma} = -c_2$. So the mass row of $\Phi$ is
$[0, \dots, 0, 1]$, the mass row of $\Psi$ is $[0, 0, 0, -c_2(t - t_0)]$, and
$m(t) = m_0 - c_2\Gamma(t - t_0)$.

### Coupling of $\Gamma$ into $\boldsymbol r$, $\boldsymbol v$ and the elements

$\Gamma$ reaches the trajectory only through the mass: $\Psi_{1:6,\Gamma}$ is driven by the mass column
of $A$ times $\Psi_{m,\Gamma}$.

**$\Psi_{1:6,\Gamma} = 0$ holds only with SRP off and $\boldsymbol\tau = 0$**. It is not zero in general.
Over a short arc, the leading-order coupling in `rvm` is

$$
\Psi_{\boldsymbol v,\Gamma} \approx \frac{c_1 c_2\,\Delta t^2}{2}\,\frac{\boldsymbol\tau}{m^2}
\quad\text{(thrust)},
\qquad
\Psi_{\boldsymbol v,\Gamma} \approx \frac{c_2\,\Delta t^2}{2m}\,\boldsymbol a_\text{srp}
\quad\text{(SRP, $\boldsymbol\tau = 0$)},
$$

with $\Delta t = t - t_0$. In free space, with $k = c_2\Gamma \neq 0$ and $m = m_0 - k\Delta t$, the
thrust term has the exact form

$$
\Psi_{\boldsymbol v,\Gamma} = \frac{c_1 c_2\,\boldsymbol\tau}{k^2}\left(\ln\frac{m}{m_0} + \frac{m_0}{m} - 1\right),
$$

whose limit for $k\Delta t \to 0$ is the leading-order term. Compared with `solve_ivpstm_rvm` in free
space, at $m_0 = 1$ with $\boldsymbol\tau = [0.3, -0.5, 0.4]$ and $\Gamma = 0.71$:

| $k\Delta t/m_0$ | Exact form, relative error | Leading-order term with final $m$, relative error |
|---|---|---|
| $1.9\times10^{-4}$ | $3\times10^{-9}$ (cancellation in the formula) | $1.3\times10^{-4}$ |
| 0.024 | $1\times10^{-13}$ | 0.017 |
| 0.24 | $1\times10^{-15}$ | 0.21 |
| 0.58 | $1\times10^{-16}$ | 0.85 |

The exact form cancels for small $k\Delta t$, so use the leading-order term or a series there.


## `propagate_zoh`

`propagate_zoh_rvm` and `propagate_zoh_meem` run one integration per segment, from $t_k$ = `times[k]`
to `times[k + 1]`, so GSL never steps across a control discontinuity. Each segment starts **from its
own row of `x_ref`**, not from the end of the previous segment. With $N$ times, every output has
$N - 1$ blocks:

| Output | Block $k$ |
|---|---|
| `x_end` | $\boldsymbol x(t_{k+1})$, integrated from $\bar{\boldsymbol x}_k$ |
| `Phi`, `Psi` | $\Phi_k$, $\Psi_k$ at $t_{k+1}$ |
| `c` | $\boldsymbol c_k = \boldsymbol x(t_{k+1}) - \Phi_k\bar{\boldsymbol x}_k - \Psi_k\bar{\boldsymbol u}_k$ |
| `defects` | $\bar{\boldsymbol x}_{k+1} - \boldsymbol x(t_{k+1})$ |

Conventions:

- **Do not wrap $L$ in `x_ref`.** The solvers do not wrap $L$, so a wrapped reference gives defects of
  multiples of $2\pi$ on every segment that crosses a wrap.
- **Backward times are allowed.** `times` may decrease. Then $\bar{\boldsymbol u}_k$ is held from `times[k]`
  down to `times[k + 1]`, and $\Phi_k$, $\Psi_k$ are still the sensitivities of $\boldsymbol x$(`times[k + 1]`)
  to $\boldsymbol x$(`times[k]`) and to $\bar{\boldsymbol u}_k$. $\dot m$ keeps its sign in $t$, so with $\Gamma > 0$
  the mass grows going backward.
- **Last row.** The last row of `x_ref` enters only the last defect, so its mass need not be $> 0$. It
  must still be finite.
- **`t_eval`** must be empty. `max_steps` applies to each segment.
- **Errors name the segment.** A failed segment keeps its exception type, with
  `"<function>: segment k of n (t = [a, b] TU): "` in front of the message. `k` counts from 0, and `n`
  is the number of segments. For example:
  `propagate_zoh_rvm: segment 1 of 2 (t = [1, 2] TU): solve_ivpstm_rvm: mass out of domain (m is not > 0) at t = 1 TU`.


## Integration limits

(zoh-depletion)=
### Mass depletion

Mass is linear in time, so the `rvm` and `meem` solvers check it **before integrating**: if $m_0$ is not
$> 0$, or $m_0 - c_2\Gamma(t_\text{last} - t_0)$ is not $> 0$, they throw `std::domain_error`, with the
depletion time in the message. $t_\text{last}$ is the last output time: `tf`, or the last of `t_eval`.
The check applies whatever the thrust, and also to a zero-length `tspan`.

Every domain error (`std::domain_error`) becomes `ValueError` in Python: mass, MEE, `sh_rho_min` and
non-finite derivatives.

### `max_steps`

`IvpOptions::max_steps`, 2,000,000 by default, **applies only to the `rvm` and `meem` solvers**; the `rv`
and `mee` solvers ignore it, and their Python functions do not accept it. It exists because a step size
collapsing onto a singularity, such as $p \to 0$ when thrust deorbits an MEE trajectory, may never fail
on its own. Reaching it with time left throws `std::runtime_error`, naming the time and the step size.

- `0` disables the limit. A negative value throws `std::invalid_argument`.
- It counts accepted steps over the whole of `tspan`, with or without `t_eval`, and per segment in
  `propagate_zoh_*`.
- **It bounds steps, not time.** The time to reach it depends on the force model. Measured on one Linux
  x86-64 machine in Release, with synthetic gravity fields for the spherical-harmonics rows:

  | Forces | State only | With sensitivities | Time for 2,000,000 steps |
  |---|---|---|---|
  | Third bodies, SRP | 34,000 steps/s | 30,000 steps/s | about 1 minute |
  | Spherical harmonics, degree 50 | 6,500 steps/s | 4,300 steps/s | 5 to 8 minutes |
  | Spherical harmonics, degree 100 | 1,700 steps/s | 1,150 steps/s | 19 to 29 minutes |
  | Spherical harmonics, degree 200 | 430 steps/s | 290 steps/s | 1.3 to 1.9 hours |

- **Memory.** With sensitivities and an empty `t_eval`, each accepted step records 85 doubles, so a stall
  reaching the default limit records 1.4 GB, and peak memory can be higher while the vectors reallocate.
  Give a `t_eval`, or a lower `max_steps`, when this matters.

At default tolerances, the header records low-thrust arcs from low Earth and lunar orbit taking up to
$1.4\times10^6$ steps over 3 years, and $5.9\times10^5$ with sensitivities over 1 year, with fewer in
canonical units. A longer arc may need a higher limit.

### Interrupting

`IvpOptions::poll` is called between accepted steps of the `rvm` and `meem` solvers, before the first
step and then every 1024 steps, and may throw to stop the integration. Python sets it to check for
signals, so **`solve_ivp*_rvm`, `solve_ivp*_meem` and `propagate_zoh_*` can be interrupted with
Ctrl-C**. `solve_ivp*_rv` and `solve_ivp*_mee` cannot.


## Numerical guidance

- **Choose $\bar m$ near $m_0$**, so that $m \approx 1$. The trajectory does not depend on $\bar m$
  beyond the integration error, because $c_1/m$ and $c_2\Gamma$ give the same physical thrust and flow. However,
  $\bar m$ sets the size of $m$ and of the mass column of $\Phi$, relative to the tolerances.
- **A single `atol` and `rtol` apply to all 84 components**: the state, every entry of $\Phi$ and every
  entry of $\Psi$, whatever their units. Sensitivities therefore take part in the step control, and a
  sensitivity run can take many more steps than a state-only run, with a different step sequence. So
  the two never agree beyond the integration error.
- **Finite differences**, as in the test suite:

  | Quantity | Steps |
  |---|---|
  | $A$ from $\dot{\boldsymbol x}$ | velocity 1e-6 km/s, mass 1e-5 $m$; MEE scaled as in [gve_mee.md](gve_mee.md) section 7.4 |
  | $B$ from $\dot{\boldsymbol x}$ | 1e-2 on each control, as $\dot{\boldsymbol x}$ is affine in $\boldsymbol u$ |
  | $\Phi$ from propagated states | 1e-3 km, 1e-6 km/s, 1e-3 $m$; in MEE 1e-6 $p$, 1e-7 otherwise |
  | $\Psi$ from propagated states | $\delta\tau$ = 1e-4, $\delta\Gamma$ = 1e-2 |

  Perturb $\Gamma$ on its own, without recomputing it from $\boldsymbol\tau$. Its effect on the
  trajectory is smaller than that of $\boldsymbol\tau$ by about $c_2\Gamma\Delta t/m$, so it needs the larger step
  to rise above integration noise. `examples/solve_zoh.cpp` also tightens the tolerances
  to `atol = rtol = 1e-14` for this purpose.
- **Check the $\boldsymbol r$, $\boldsymbol v$ rows separately.** A row metric, the largest error in a row
  over the largest entry of that row, can hide errors in the mass and $\Gamma$ columns. For example, with
  $T_{\max}$ = 20 N, $I_{sp}$ = 3000 s, $\bar m$ = 1000 kg and $\Delta t$ = 400 s, the $\Gamma$ column of the
  $\boldsymbol v$ rows of $\Psi$ is $4\times10^{-5}$ of the $\boldsymbol\tau$ column: setting it to zero passes a
  row metric at 1e-4. Add a column metric, restricted to the rows of the trajectory, so that the exact mass
  row does not set the scale of the columns (test `ZohValSens.ColumnMetricCatchesDroppedGammaCoupling`).


## Tolerances of the test suite

These differ from those proposed in issue #5, for the reasons below. Unless noted, the tolerance is
about 10 times the largest value measured with `ZOH_PRINT_ERRORS` on Linux x86-64 in Release. Other
checks use the issue's tolerance.

| Check | Issue | Test | Measured | Reason |
|---|---|---|---|---|
| Ballistic $\Phi_{1:6,1:6}$ vs `solve_ivpstm_*` | 1e-12 | 4e-11 | 4.3e-12 in Release, 9.8e-12 with an empty build type | step sequences differ (84 vs 42 components) |
| Ballistic state vs `solve_ivpstm_*` | 1e-12 | 6e-12 | 6.3e-13 in Release, 9.6e-13 with an empty build type | step sequences differ; the measured value also meets the issue's tolerance, and the test keeps the factor of 10 |
| Cartesian against MEE, mass | identical | 7e-14 | 7.1e-15 | step sequences differ |
| Defects against a state-only run | 1e-10 | 4e-9 km, 7e-13 km/s, 8e-12 for elements, 1.4e-13 for mass | 4.0e-10 km, 7.6e-14 km/s, 8.9e-13, 1.5e-14 | the defects are the integration error, as the step sequences differ |
| $\Psi$ FD step | $\delta \approx$ 1e-6 | $\delta\tau$ = 1e-4, $\delta\Gamma$ = 1e-2 | | $\delta\Gamma$ = 1e-6 moves $\boldsymbol v$ by about 3e-13 km/s on the 400 s arc above, comparable with the integration error |
| Closed-form mass, long arc | 1e-13 | 1e-12 (1e-13 on short arcs) | 4.8e-15 with an empty build type | no measured need: the issue's 1e-13 would also pass |

The state-only `rvm` and `meem` solvers with $\boldsymbol u = 0$ and `srp_m` $= \bar m\,m_0$ are bit-equal to `rv` and
`mee`. A reference built with the identical sensitivity computation has defects at most 1e-10 (the
same bits are expected). The affine identity
$\Phi_k\bar{\boldsymbol x}_k + \Psi_k\bar{\boldsymbol u}_k + \boldsymbol c_k = \boldsymbol x(t_{k+1})$ holds within
8 machine epsilons of the sum of the magnitudes of its terms.
