# Python API

Every function and class of the `campanella` module, as defined in `src/bindings.cpp`. The module
is a thin pybind11 layer over the [C++ API](api_cpp.md): the semantics, units and validation are
those of the C++ functions, which are linked from each entry. This page lists what differs in
Python: argument names and defaults, accepted and returned types, array shapes, and exception
types. A tutorial-style introduction is in [Basic usage through Python](basic_usage_python.md).

Conventions:

- **Inputs.** Arguments documented as sequences (`tspan`, `x0`, `x`, `GMs`, `t_eval`, ...) accept
  any one-dimensional sequence of numbers: a list, a tuple or a 1-D NumPy array. They are copied
  into a C++ `std::vector<double>`. A 2-D array is rejected with `TypeError`.
- **Units.** DU, TU and VU are the distance, time and velocity units of the `SpiceParameters`
  (km, s and km/s unless a canonical `DU` is given); see {ref}`Units <cpp-units>`. Times `t` are in
  TU since `et0`, the reference epoch in TDB seconds past J2000.
- **Terms.** STM is the state-transition matrix $\Phi(t, t_0) = \partial\boldsymbol{x}(t)/\partial\boldsymbol{x}(t_0)$;
  MEE are the modified equinoctial elements `[p, f, g, h, k, L]` (see [GVE in MEE](gve_mee.md));
  `atol` and `rtol` are the absolute and relative error tolerances (each integrated quantity $y_i$
  is kept within $\mathrm{atol} + \mathrm{rtol}\,|y_i|$ per step); "flattened row-major" means a
  matrix stored row by row in a flat list, element `(i, j)` of a 6x6 matrix at index `6*i + j`.
- **Exceptions.** pybind11 translates the C++ exceptions: `std::invalid_argument` and
  `std::domain_error` become `ValueError`, `std::runtime_error` becomes `RuntimeError`. So
  `ValueError` means either bad input or a state outside the domain of the model (below
  `sh_rho_min * R`, MEE out of their domain, a mass that is not > 0, non-finite derivative), and
  `RuntimeError` means a SPICE error (with NAIF's message), a gravity-field file error, an integration
  failure, or `max_steps` being reached. A
  propagation that runs into the centre of a body usually raises `RuntimeError` ("integration
  failed"), not `ValueError`: see
  [Error handling](basic_usage_cpp.md#error-handling).
  Arguments of the wrong type raise `TypeError`.
- **SPICE.** Importing the module calls {ref}`init_spice <cpp-init-spice>`, so SPICE errors raise
  `RuntimeError` instead of exiting the interpreter. No kernels are loaded on import. The module
  contains its own CSPICE: kernels loaded with `spiceypy.furnsh` are not visible to it.
- **Threads.** The functions do not release the GIL, so calls from several Python threads run one
  at a time.

Contents:

- {ref}`furnsh <py-furnsh>`
- {ref}`SpiceParameters <py-spiceparameters>`
- {ref}`IvpResult <py-ivpresult>`
- {ref}`solve_ivp_rv, solve_ivp_mee, solve_ivpstm_rv, solve_ivpstm_mee <py-solvers>`
- {ref}`eval_rv, eval_mee <py-eval>`
- {ref}`eom_rv, eom_mee, eomstm_rv, eomstm_mee <py-eom>`
- {ref}`ThrustFrame <py-thrustframe>`, {ref}`ThrustOptions <py-thrustoptions>`
- {ref}`solve_ivp_rvm, solve_ivp_meem, solve_ivpstm_rvm, solve_ivpstm_meem <py-solve-zoh>`
- {ref}`eval_rvm, eval_meem <py-eval-zoh>`
- {ref}`eom_rvm, eom_meem, eomstm_rvm, eomstm_meem <py-eom-zoh>`
- {ref}`propagate_zoh_rvm, propagate_zoh_meem, ZohLinearization <py-propagate-zoh>`
- {ref}`load_spherical_harmonics <py-load-spherical-harmonics>`,
  {ref}`sh_accel_bf <py-sh-accel-bf>`, {ref}`sh_accel_inertial <py-sh-accel-inertial>`
- {ref}`mee2cart, cart2mee, mee2cart_jacobian <py-conversions>`

The C++ functions `init_spice` and `eom_error`, and `SpiceParameters::options()`, have no Python
binding; `init_spice` runs on import, and the Python `eom_*` functions raise instead of returning a
status. `ControlledParameters` has no binding either, and `EomEvalControl` is returned as a tuple. `IvpOptions` is not
a class: its fields are keyword arguments, and `poll` is set by the module to check for signals.


(py-furnsh)=
## `furnsh`

```python
campanella.furnsh(path: str) -> None
```

Loads a SPICE kernel into the module's CSPICE. See {ref}`furnsh <cpp-furnsh>`.

**Raises** `RuntimeError` with NAIF's message if the kernel cannot be loaded.


(py-spiceparameters)=
## `SpiceParameters`

```python
campanella.SpiceParameters(
    et0: float, naif_ids: list[str], *,
    naif_frame: str = "J2000",
    GMs: list[float] | None = None,
    DU: float | None = None,
    use_canonical_scales: bool = True,
    abcorr: str = "NONE",
    include_srp: bool = False,
    srp_A: float = 0.0,
    srp_m: float = 0.0,
    srp_rhoS: float = 0.0,
    srp_rhoD: float = 0.0,
    sh_file: str = "",
    sh_nmax: int = 0,
    sh_mmax: int = -1,
    sh_frame: str = "",
    sh_rho_min: float = 1.0,
)
```

:::{important}
`naif_ids` must be a list of **strings**: NAIF names such as `"EARTH"` or numeric codes written as
strings such as `"399"`. Integers are rejected:
`SpiceParameters(et0, [399, 301, 10])` raises
`TypeError: __init__(): incompatible constructor arguments`. Write `["399", "301", "10"]` instead.
The resolved integer codes are available afterwards as `params.naif_codes`.
:::

The dynamics model. `et0` and `naif_ids` are positional or keyword; every other argument is
keyword-only and maps to the field of the same name of {ref}`SpiceOptions <cpp-spiceoptions>`,
where each is described with its units and constraints. `GMs=None` is an empty `GMs` (read the GMs
from the kernel pool), and `DU=None` an empty `DU` (km and s). See
{ref}`SpiceParameters <cpp-spiceparameters>` for `et0`, `naif_ids` and the validation.

**Raises** `ValueError` for invalid input (C++ `std::invalid_argument`), `RuntimeError` for a SPICE
error or an unreadable gravity-field file.

The object is immutable. It cannot be copied with `copy.copy` or `copy.deepcopy`, or pickled
(`TypeError`), because the binding defines no copy or pickle support. No copy is needed: since the
object never changes, several variables, threads or solver calls can share the same
`SpiceParameters`. (In C++, a copy exists to give each thread its own scratch space for `eval_*` and
`eom_*`; in Python the GIL already makes those calls run one at a time.) To send a model to another
process, for example with `multiprocessing`, send its constructor arguments and construct it there.

### Properties

All read-only (assignment raises `AttributeError`).

| Property | Type | Value |
|---|---|---|
| `et0` | `float` | reference epoch, TDB seconds past J2000 |
| `naif_ids` | `list[str]` | body names as given |
| `naif_codes` | `list[int]` | resolved NAIF codes; `[0]` is the central body |
| `naif_frame` | `str` | integration frame |
| `GMs` | `list[float]` | km^3/s^2, read from the kernels if not given |
| `mus` | `list[float]` | `GMs * TU^2 / DU^3` |
| `DU`, `TU`, `VU` | `float` | units, in km, s and km/s |
| `use_canonical_scales` | `bool` | as given |
| `abcorr` | `str` | as given |
| `include_srp` | `bool` | whether SRP is on |
| `srp_A`, `srp_m`, `srp_rhoS`, `srp_rhoD` | `float` | as given |
| `include_spherical_harmonics` | `bool` | whether the gravity field is on |
| `sh_file`, `sh_frame` | `str` | as given |
| `sh_nmax`, `sh_mmax` | `int` | as given (`sh_mmax` stays -1 if so given) |
| `sh_rho_min` | `float` | as given |

The list properties return a new list on each access. `repr()` shows `et0`, `naif_ids`,
`naif_frame`, the units and the two `include_*` flags, e.g.

```text
SpiceParameters(et0=740463475.5117936, naif_ids=['399', '301', '10'], naif_frame='J2000', DU=1, TU=1, VU=1, include_srp=False, include_spherical_harmonics=False)
```


(py-ivpresult)=
## `IvpResult`

Returned by the solvers; cannot be constructed from Python. The C++ {ref}`IvpResult <cpp-ivpresult>`,
with its arrays exposed as NumPy views.

| Attribute | Type | Shape |
|---|---|---|
| `t` | `numpy.ndarray[float64]` | `(N,)`, times in TU |
| `x` | `numpy.ndarray[float64]` | `(N, 6)`, states (Cartesian in DU, DU/TU, or MEE with `p` in DU and `L` not wrapped) |
| `stm` | `numpy.ndarray[float64]` or `None` | `(N, 6, 6)`, `stm[k, i, j]` = $\partial x_i(t_k) / \partial x_j(t_0)$; `None` for `solve_ivp_*` |
| `stm_u` | `numpy.ndarray[float64]` or `None` | `(N, 7, 4)`, `stm_u[k, i, j]` = $\partial x_i(t_k) / \partial u_j$; `None` unless `solve_ivpstm_rvm` or `solve_ivpstm_meem` |

For the `rvm` and `meem` solvers, `x` is `(N, 7)` with the mass last and `stm` is `(N, 7, 7)`. The C++ fields
`nx` and `nu` are not exposed; the array shapes carry them.

`len(res)` is N. The arrays are read-only views into the result, without copies: writing to them
raises `ValueError`, and they keep the result alive after the `IvpResult` itself goes out of scope.
Use `res.x.copy()` for a writable array.


(py-solvers)=
## `solve_ivp_rv`, `solve_ivp_mee`, `solve_ivpstm_rv`, `solve_ivpstm_mee`

```python
campanella.solve_ivp_rv(params: SpiceParameters, tspan, x0, *,
                        atol: float = 1e-12, rtol: float = 1e-14,
                        t_eval=None) -> IvpResult
```

The four solvers have the same signature. `atol`, `rtol` and `t_eval` are keyword-only and form
the C++ {ref}`IvpOptions <cpp-ivpoptions>`; `t_eval=None` means every accepted step.

| Python | C++ | State | `res.stm` |
|---|---|---|---|
| `solve_ivp_rv` | {ref}`solve_ivp_rv <cpp-solve-ivp-rv>` | Cartesian | `None` |
| `solve_ivp_mee` | {ref}`solve_ivp_mee <cpp-solve-ivp-mee>` | MEE | `None` |
| `solve_ivpstm_rv` | {ref}`solve_ivpstm_rv <cpp-solve-ivpstm-rv>` | Cartesian | `(N, 6, 6)` |
| `solve_ivpstm_mee` | {ref}`solve_ivpstm_mee <cpp-solve-ivpstm-mee>` | MEE | `(N, 6, 6)` |

**Arguments**

- `params`: the dynamics model.
- `tspan`: `[t0, tf]` in TU since `et0`; `tf < t0` integrates backward, `tf == t0` returns one row.
- `x0`: initial state, 6 values (also for the STM solvers, which start from $\Phi = I$).
- `atol`, `rtol`: tolerances.
- `t_eval`: output times in TU, within `tspan` and strictly monotonic in the direction of
  integration.

**Returns** an {ref}`IvpResult <py-ivpresult>`.

**Raises** `ValueError` for invalid input or a state that leaves the domain of the model;
`RuntimeError` for a SPICE error or an integration failure. A partial trajectory is never
returned.

```python
res = campanella.solve_ivpstm_rv(params, [0.0, 3600.0], [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
                                 t_eval=numpy.linspace(0.0, 3600.0, 4))
res.t.shape, res.x.shape, res.stm.shape     # (4,), (4, 6), (4, 6, 6)
```


(py-eval)=
## `eval_rv`, `eval_mee`

```python
campanella.eval_rv(params: SpiceParameters, t: float, x) -> tuple[numpy.ndarray, numpy.ndarray]
campanella.eval_mee(params: SpiceParameters, t: float, x) -> tuple[numpy.ndarray, numpy.ndarray]
```

Time derivative and Jacobian at one time `t` (TU) and state `x` (6 values; Cartesian for
`eval_rv`, MEE for `eval_mee`). See {ref}`eval_rv <cpp-eval-rv>` and
{ref}`eval_mee <cpp-eval-mee>`.

**Returns** `(xdot, A)`: `xdot` of shape `(6,)` and `A` = $\partial\dot{\boldsymbol{x}}/\partial\boldsymbol{x}$
of shape `(6, 6)`. Both are new, writable arrays.

**Raises** `ValueError` if `x` does not have 6 values or the state is outside the domain
(`eval_mee`: $p \le 0$ or $w \le 0$; both: below `sh_rho_min * R`, non-finite derivative);
`RuntimeError` with NAIF's message on a SPICE error.


(py-eom)=
## `eom_rv`, `eom_mee`, `eomstm_rv`, `eomstm_mee`

```python
campanella.eom_rv(params: SpiceParameters, t: float, x) -> numpy.ndarray
```

The right-hand sides of the solvers, at one time `t` (TU) and state `x`. Unlike the C++
{ref}`eom_* <cpp-eom>` functions, they take `params` first, return the derivative, and raise
instead of returning a status code.

| Function | `x` | Returns |
|---|---|---|
| `eom_rv` | Cartesian state, 6 values | `xdot`, shape `(6,)` |
| `eom_mee` | MEE, 6 values | `xdot`, shape `(6,)` |
| `eomstm_rv` | Cartesian state, then $\Phi$ row-major: 42 values | shape `(42,)`: `xdot`, then $A\Phi$ row-major |
| `eomstm_mee` | MEE, then $\Phi$ row-major: 42 values | shape `(42,)` |

**Raises** `ValueError` if `x` has the wrong length (`eom_rv: x must have length 6`) or the state
is outside the domain; `RuntimeError` with NAIF's message on a SPICE error. The message has the
form `"<function>: <reason> at t = <t> TU"`, e.g.
`eom_rv: non-finite derivative (e.g. at the center of a body) at t = 2.5 TU`; the reason is the
description that the C++ {ref}`eom_error <cpp-eom-error>` would return, to which Python adds the
function name and the time.


## Mass-augmented dynamics

Cartesian + mass (`rvm`) and MEE + mass (`meem`), with a control `u = [tau_1, tau_2, tau_3, Gamma]` held
constant over the integration. See
[Mass-augmented dynamics under zero-order-hold control](zoh_mass_dynamics.md).

(py-thrustframe)=
### `ThrustFrame`

An enum with the members `campanella.ThrustFrame.rtn` and `campanella.ThrustFrame.inertial`. See
{ref}`ThrustFrame <cpp-thrustframe>`.

(py-thrustoptions)=
### `ThrustOptions`

```python
campanella.ThrustOptions(*, thrust_N: float, isp_s: float, mass_unit_kg: float,
                         g0: float = 9.80665,
                         mee_control_frame: ThrustFrame = ThrustFrame.rtn)
```

The propulsion system; see {ref}`ThrustOptions <cpp-thrustoptions>`. All arguments are keyword-only, and
the first three are required. The attributes of the same names are read-only.

The values are checked where they are used, not here: the `rvm` and `meem` functions raise `ValueError`
unless `thrust_N`, `isp_s`, `mass_unit_kg` and `g0` are finite and > 0. $c_1$ and $c_2$ are not exposed.

The dynamics treat `Gamma` as independent of `tau`; enforcing `Gamma = |tau|` is the job of the caller or
optimizer. SRP always uses the state mass: `srp_m` of `SpiceParameters` is ignored, but must still
be > 0 when `include_srp` is on.

(py-solve-zoh)=
### `solve_ivp_rvm`, `solve_ivp_meem`, `solve_ivpstm_rvm`, `solve_ivpstm_meem`

```python
campanella.solve_ivp_rvm(params: SpiceParameters, thrust: ThrustOptions, tspan, x0, u, *,
                         atol: float = 1e-12, rtol: float = 1e-14,
                         t_eval=None, max_steps: int = 2000000) -> IvpResult
```

The four solvers have the same signature; see {ref}`the C++ solvers <cpp-solve-zoh>`.

| Python | State `x0` (7 values) | `tau` in | `res.stm` | `res.stm_u` |
|---|---|---|---|---|
| `solve_ivp_rvm` | Cartesian, mass | `naif_frame` | `None` | `None` |
| `solve_ivp_meem` | MEE, mass | `thrust.mee_control_frame` | `None` | `None` |
| `solve_ivpstm_rvm` | Cartesian, mass | `naif_frame` | `(N, 7, 7)` | `(N, 7, 4)` |
| `solve_ivpstm_meem` | MEE, mass | `thrust.mee_control_frame` | `(N, 7, 7)` | `(N, 7, 4)` |

`max_steps` bounds the accepted steps; 0 disables the limit, and a negative value raises `ValueError`. It
bounds steps, not time, and only these solvers and `propagate_zoh_*` accept it. See
[Integration limits](zoh_mass_dynamics.md#integration-limits).

**Interrupting.** These solvers and `propagate_zoh_*` check for Python signals every 1024 steps, so Ctrl-C
stops them. `solve_ivp_rv`, `solve_ivp_mee`, `solve_ivpstm_rv` and `solve_ivpstm_mee` cannot be interrupted.

**Raises** `ValueError` for invalid input or a state that leaves the domain, including a mass that is
not > 0 at `t0` or at the last output time (checked before integrating); `RuntimeError` for a SPICE error,
an integration failure, or if `max_steps` is reached.

```python
thrust = campanella.ThrustOptions(thrust_N=1.0, isp_s=3000.0, mass_unit_kg=1000.0)
x0 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.0]
u = [0.6, 0.0, 0.8, 1.0]                     # Gamma = |tau|, imposed by the caller
res = campanella.solve_ivpstm_rvm(params, thrust, [0.0, 3600.0], x0, u,
                                  t_eval=numpy.linspace(0.0, 3600.0, 4))
res.x.shape, res.stm.shape, res.stm_u.shape  # (4, 7), (4, 7, 7), (4, 7, 4)
```

(py-eval-zoh)=
### `eval_rvm`, `eval_meem`

```python
campanella.eval_rvm(params, thrust, t: float, x, u) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
campanella.eval_meem(params, thrust, t: float, x, u) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
```

**Returns** `(xdot, A, B)`, of shapes `(7,)`, `(7, 7)` and `(7, 4)`: new, writable arrays. See
{ref}`eval_rvm, eval_meem <cpp-eval-zoh>`.

**Raises** `ValueError` for invalid input, a mass that is not > 0 or not finite, a state outside the domain
(`eval_meem`: $p \le 0$ or $w \le 0$; both: below `sh_rho_min * R`, non-finite derivative); `RuntimeError`
with NAIF's message on a SPICE error.

(py-eom-zoh)=
### `eom_rvm`, `eom_meem`, `eomstm_rvm`, `eomstm_meem`

```python
campanella.eom_rvm(params, thrust, t: float, x, u) -> numpy.ndarray
```

The right-hand sides of the solvers. They take `params` and `thrust` directly, and raise like
`eval_rvm` and `eval_meem` instead of returning a status.

| Function | `x` | Returns |
|---|---|---|
| `eom_rvm`, `eom_meem` | state, 7 values | `xdot`, shape `(7,)` |
| `eomstm_rvm`, `eomstm_meem` | state, then $\Phi$ (7x7) and $\Psi$ (7x4), both row-major: 84 values | shape `(84,)`, in the same order |

(py-propagate-zoh)=
### `propagate_zoh_rvm`, `propagate_zoh_meem`

```python
campanella.propagate_zoh_rvm(params, thrust, times, x_ref, u_ref, *,
                             atol: float = 1e-12, rtol: float = 1e-14,
                             max_steps: int = 2000000) -> ZohLinearization
```

Linearizes the segments between N `times`. Each segment starts from its own row of `x_ref`. See
[`propagate_zoh`](zoh_mass_dynamics.md#propagate_zoh) and {ref}`the C++ functions <cpp-propagate-zoh>`.

- `times`: boundaries in TU, a sequence of N >= 2 values, strictly monotonic. They may decrease, for a
  backward integration.
- `x_ref`: array-like of shape `(N, 7)`. Do not wrap `L`: a wrapped `L` gives defects of multiples of $2\pi$.
  The mass of the last row need not be > 0.
- `u_ref`: array-like of shape `(N - 1, 4)`.
- `max_steps`: per segment. There is no `t_eval` argument.

**Returns** a `ZohLinearization`, which cannot be constructed from Python. `len(z)` is N - 1, and its
attributes are read-only NumPy views, as in {ref}`IvpResult <py-ivpresult>`:

| Attribute | Shape | Block `k` |
|---|---|---|
| `x_end` | `(N - 1, 7)` | state at the end of segment `k` |
| `Phi` | `(N - 1, 7, 7)` | $\Phi_k$ |
| `Psi` | `(N - 1, 7, 4)` | $\Psi_k$ |
| `c` | `(N - 1, 7)` | `x_end[k] - Phi[k] @ x_ref[k] - Psi[k] @ u_ref[k]` |
| `defects` | `(N - 1, 7)` | `x_ref[k + 1] - x_end[k]` |

**Raises** `ValueError` on bad input or shapes, or if a segment leaves the domain; `RuntimeError` on a SPICE
error, an integration failure, or if `max_steps` is reached. The message of a failed segment starts with
`"<function>: segment k of n (t = [a, b] TU): "`, with `k` from 0. An exception raised by a Python signal handler
passes through unchanged.


(py-load-spherical-harmonics)=
## `load_spherical_harmonics`

```python
campanella.load_spherical_harmonics(path: str, nmax: int, mmax: int = -1) -> dict
```

Loads a gravity field file (PDS SHADR `.tab` or ICGEM `.gfc`, detected from the contents),
truncated to degree `nmax` and order `mmax` (-1: min(`nmax`, order of the file)), through the same
cache as the solvers. Python only; there is no C++ counterpart in `campanella.hpp`.

**Returns** a `dict` with the keys

| Key | Type | Value |
|---|---|---|
| `source` | `str` | `path` |
| `GM` | `float` | gravitational parameter of the file, km^3/s^2 |
| `R` | `float` | reference radius, km |
| `file_degree`, `file_order` | `int` | maximum degree and order of the file |
| `nmax`, `mmax` | `int` | truncation used |
| `tide_system` | `str` | from the file header; `""` if the file does not say |
| `C`, `S` | `list[list[float]]` | fully normalized coefficients: `C[n][m]` for `0 <= m <= n <= nmax`; zero for `m > mmax`, `C[0][0] = 1` |

**Raises** `ValueError` for `nmax < 2`, `nmax > 1200`, `nmax` above the degree of the file, or
`mmax` above `nmax` or the order of the file; `RuntimeError` for a file that cannot be opened, is
malformed, or is truncated (for example, a file whose header claims a higher degree than the
coefficients it contains raises `RuntimeError` when `nmax` exceeds the coefficients present).


(py-sh-accel-bf)=
## `sh_accel_bf`

```python
campanella.sh_accel_bf(path: str, nmax: int, r_bf, mmax: int = -1, mu: float = 0.0,
                       R: float = 0.0, rho_min: float = 1.0) -> tuple[float, list[float], list[float]]
```

Potential, acceleration and gradient of the field in the body-fixed frame. All arguments may be
given positionally. See {ref}`sh_accel_bf <cpp-sh-accel-bf>` for the arguments and units.

**Returns** `(U, a, G)`: `U` a `float`, `a` a list of 3 floats, `G` = $\partial\boldsymbol{a}/\partial\boldsymbol{r}_{bf}$
a flat list of 9 floats (3x3 row-major). Lists, not NumPy arrays.

**Raises** `ValueError` for invalid arguments or below `rho_min * R`; `RuntimeError` for file
errors.


(py-sh-accel-inertial)=
## `sh_accel_inertial`

```python
campanella.sh_accel_inertial(path: str, nmax: int, naif_frame: str, sh_frame: str, et: float, r,
                             DU: float = 1.0, TU: float = 1.0, mmax: int = -1,
                             rho_min: float = 1.0) -> tuple[list[float], list[float]]
```

Acceleration and gradient of the field in an inertial frame, as used by the equations of motion.
See {ref}`sh_accel_inertial <cpp-sh-accel-inertial>`. The kernels of `sh_frame` must be loaded with
{ref}`furnsh <py-furnsh>`.

**Returns** `(a, G)`: `a` a list of 3 floats in DU/TU^2, `G` a flat list of 9 floats in 1/TU^2
(3x3 row-major).

**Raises** `ValueError` for invalid arguments or below `rho_min * R`; `RuntimeError` for a SPICE
error or a file error.


(py-conversions)=
## `mee2cart`, `cart2mee`, `mee2cart_jacobian`

```python
campanella.mee2cart(mee, mu: float) -> list[float]
campanella.cart2mee(rv, mu: float) -> list[float]
campanella.mee2cart_jacobian(mee, mu: float) -> list[float]
```

Conversions between MEE `[p, f, g, h, k, L]` and Cartesian states `[x, y, z, vx, vy, vz]`, in the
units of `mu` (use `params.mus[0]` for the units of a model). See {ref}`mee2cart <cpp-mee2cart>`,
{ref}`cart2mee <cpp-cart2mee>` and {ref}`mee2cart_jacobian <cpp-mee2cart-jacobian>`.

**Returns** a list of 6 floats for `mee2cart` and `cart2mee` (`L` in $(-\pi, \pi]$), and a flat
list of 36 floats (6x6 row-major) for `mee2cart_jacobian`; reshape it with
`numpy.reshape(J, (6, 6))`.

**Raises** `ValueError` if the input does not have 6 values, for MEE outside their domain
(`mee2cart`, `mee2cart_jacobian`: $p \le 0$ or $w \le 0$), or for zero angular momentum or an
inclination within about 1.4e-6 rad of 180 degrees (`cart2mee`).
