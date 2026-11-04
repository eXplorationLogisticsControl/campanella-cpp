# C++ API

Every public symbol of `include/campanella.hpp`. All symbols are in the namespace `campanella`;
the header needs C++17 and includes only standard headers. The namespace `campanella::detail` holds
forward declarations of internal types and is not part of the API.

Conventions used on this page:

- **Units.** DU, TU and VU are the distance, time and velocity units of a
  {ref}`SpiceParameters <cpp-spiceparameters>` (km, s and km/s unless a canonical `DU` is given;
  see {ref}`cpp-units`). Times `t` are in TU since `et0`.
- **Matrices** are stored row-major in flat arrays: element `(i, j)` of a 6x6 matrix is at index
  `6*i + j`, of a 3x3 matrix at `3*i + j`.
- **Cartesian states** are `[x, y, z, vx, vy, vz]` relative to the central body, in the inertial
  frame `naif_frame`. **MEE** are `[p, f, g, h, k, L]`, prograde (retrograde factor $I = +1$), with
  `p` in DU and the others dimensionless (see [GVE in MEE](gve_mee.md)).
- **Terms.** NAIF ids are the integer codes of bodies (399 Earth, 301 Moon, 10 Sun), passed as
  strings (`"399"`) or names (`"EARTH"`). Epochs are ephemeris time: TDB seconds past J2000 (see
  [Converting a calendar date to et](basic_usage_cpp.md#converting-a-calendar-date-to-et)). The
  STM is the state-transition matrix
  $\Phi(t, t_0) = \partial\boldsymbol{x}(t)/\partial\boldsymbol{x}(t_0)$. `atol` and `rtol` are
  the integrator's error tolerances ({ref}`IvpOptions <cpp-ivpoptions>`). Other terms are defined
  in [Terms used on this page](basic_usage_cpp.md#terms-used-on-this-page).
- **Exceptions.** `std::invalid_argument` for bad input, `std::domain_error` for a state outside the
  domain of the model, `std::runtime_error` for SPICE errors (with NAIF's message), file errors and
  integration failures. No function exits the process on error, provided CSPICE's error action has
  been set by {ref}`init_spice <cpp-init-spice>` (done by `furnsh` and the `SpiceParameters`
  constructor). The only output is a warning on `std::cerr` from the gravity-field loader for
  unusual files (non-zero degree-1 coefficients, or rows above the order in the header).
- **Symbols.** The library also exports internal functions of the global namespace (`mee2cart`,
  `srp_coefficient`, ...). A global function of yours with the same name and parameter types
  silently replaces them; see
  [Known issue: internal functions can be replaced by yours](basic_usage_cpp.md#known-issue-internal-functions-can-be-replaced-by-yours).
- **Thread safety.** None of the functions that may call CSPICE may run concurrently; see
  [Thread safety](basic_usage_cpp.md#thread-safety).

Contents:

- SPICE: {ref}`init_spice <cpp-init-spice>`, {ref}`furnsh <cpp-furnsh>`
- Dynamics model: {ref}`SpiceOptions <cpp-spiceoptions>`, {ref}`SpiceParameters <cpp-spiceparameters>`
- Solvers: {ref}`IvpOptions <cpp-ivpoptions>`, {ref}`IvpResult <cpp-ivpresult>`,
  {ref}`solve_ivp_rv <cpp-solve-ivp-rv>`, {ref}`solve_ivp_mee <cpp-solve-ivp-mee>`,
  {ref}`solve_ivpstm_rv <cpp-solve-ivpstm-rv>`, {ref}`solve_ivpstm_mee <cpp-solve-ivpstm-mee>`
- Equations of motion: {ref}`eom_rv, eom_mee, eomstm_rv, eomstm_mee <cpp-eom>`,
  {ref}`eom_error <cpp-eom-error>`, {ref}`EomEval <cpp-eomeval>`, {ref}`eval_rv <cpp-eval-rv>`,
  {ref}`eval_mee <cpp-eval-mee>`
- Mass-augmented dynamics: {ref}`ThrustFrame <cpp-thrustframe>`, {ref}`ThrustOptions <cpp-thrustoptions>`,
  {ref}`ControlledParameters <cpp-controlledparameters>`, {ref}`EomEvalControl <cpp-eomevalcontrol>`,
  {ref}`ZohLinearization <cpp-zohlinearization>`, {ref}`solve_ivp_rvm, solve_ivp_meem, solve_ivpstm_rvm, solve_ivpstm_meem <cpp-solve-zoh>`,
  {ref}`eom_rvm, eom_meem, eomstm_rvm, eomstm_meem <cpp-eom-zoh>`, {ref}`eval_rvm, eval_meem <cpp-eval-zoh>`,
  {ref}`propagate_zoh_rvm, propagate_zoh_meem <cpp-propagate-zoh>`
- Spherical harmonics: {ref}`sh_accel_bf <cpp-sh-accel-bf>`,
  {ref}`sh_accel_inertial <cpp-sh-accel-inertial>`
- Element conversions: {ref}`mee2cart <cpp-mee2cart>`, {ref}`cart2mee <cpp-cart2mee>`,
  {ref}`mee2cart_jacobian <cpp-mee2cart-jacobian>`


## SPICE

(cpp-init-spice)=
### `init_spice`

```cpp
void init_spice();
```

Sets the error action of the library's CSPICE to `RETURN` and its error output to `NONE`
(`erract_c("SET", "RETURN")`, `errprt_c("SET", "NONE")`). With CSPICE's default action, an error
prints a traceback and exits the process; with `RETURN`, the library detects the error and throws
`std::runtime_error` with NAIF's message.

Called by {ref}`furnsh <cpp-furnsh>` and by the {ref}`SpiceParameters <cpp-spiceparameters>`
constructor. Call it yourself only before {ref}`sh_accel_inertial <cpp-sh-accel-inertial>` in a
program that has called neither. Idempotent. Affects only the CSPICE inside the library (see
[One CSPICE per process](basic_usage_cpp.md#one-cspice-per-process)).

(cpp-furnsh)=
### `furnsh`

```cpp
void furnsh(const std::string& path);
```

Loads a SPICE kernel (or meta-kernel) into the CSPICE inside the library, after calling
{ref}`init_spice <cpp-init-spice>`. Kernels loaded by another CSPICE in the same process (your own
`furnsh_c`, spiceypy) are not visible to campanella.

**Parameters**

- `path`: path of the kernel file.

**Throws** `std::runtime_error` with NAIF's message if CSPICE cannot load the file, e.g.
`SPICE(NOSUCHFILE) -- The attempt to load "..." by the routine FURNSH failed. It could not be located.`


## Dynamics model

(cpp-spiceoptions)=
### `SpiceOptions`

```cpp
struct SpiceOptions {
    std::string naif_frame = "J2000";
    std::vector<double> GMs;
    std::optional<double> DU;
    bool use_canonical_scales = true;
    std::string abcorr = "NONE";
    bool include_srp = false;
    double srp_A = 0.0;
    double srp_m = 0.0;
    double srp_rhoS = 0.0;
    double srp_rhoD = 0.0;
    std::string sh_file;
    int sh_nmax = 0;
    int sh_mmax = -1;
    std::string sh_frame;
    double sh_rho_min = 1.0;
};
```

Optional inputs of {ref}`SpiceParameters <cpp-spiceparameters>`. An aggregate: set the fields by
name, then pass the struct. The fields are validated by the `SpiceParameters` constructor, which
throws `std::invalid_argument` for an invalid value unless stated otherwise. Its messages start with
`SpiceParameters: `, except those about the gravity-field file, which start with the file path or
with `spherical harmonics: `.

`naif_frame`
: Inertial frame in which the equations are integrated, e.g. `"J2000"` or `"ECLIPJ2000"`. Must be a
  frame known to CSPICE and of the inertial class; a rotating frame such as `"IAU_EARTH"` throws.

`GMs`
: Gravitational parameters in km^3/s^2, one per entry of `naif_ids`, in the same order. Always in
  km^3/s^2, also with canonical units. Empty (default): the `GM` of each body is read from the
  kernel pool (`BODY<code>_GM`, e.g. from `gm_de440.tpc`). Throws if a body has no GM in the pool,
  if the length differs from `naif_ids`, or unless every entry is finite and >= 0 and `GMs[0] > 0`.
  A third body with GM 0 exerts no gravity (useful to keep the Sun for SRP only).

`DU`
: Distance unit in km. Empty (default): DU = TU = VU = 1 (km, s, km/s). If given, must be finite
  and > 0 (checked even when `use_canonical_scales` is `false`). See {ref}`cpp-units`.

`use_canonical_scales`
: If `false`, `DU` is ignored and the units are km and s. Default `true`.

`abcorr`
: Aberration correction of the ephemeris queries. Only `"NONE"` (case-insensitive) is accepted; the
  force model uses geometric positions. The value is stored as given.

`include_srp`
: Adds cannonball solar radiation pressure, without shadow:
  $\boldsymbol{a} = (P_0/c)\, C_r (A/m)\, (\mathrm{AU}/|\boldsymbol{d}|)^2\, \boldsymbol{d}/|\boldsymbol{d}|$,
  with $\boldsymbol{d}$ the position from the Sun to the spacecraft, $C_r = 1 + \rho_S + 5\rho_D/3$,
  $P_0 = 1367$ W/m^2, $c = 299792458$ m/s and AU = 149597870.7 km. Requires the Sun (NAIF id 10, e.g.
  `"10"` or `"SUN"`) among `naif_ids`; it may be the central body. The `srp_*` fields are validated
  only when `include_srp` is `true`.

`srp_A`
: Area of the spacecraft in m^2, finite and >= 0. Physical also with canonical units.

`srp_m`
: Mass of the spacecraft in kg, finite and > 0 (when `include_srp`). Physical also with canonical
  units. The `rvm` and `meem` dynamics ignore it and use the state mass, but it must still be > 0.

`srp_rhoS`, `srp_rhoD`
: Specular and diffuse reflection coefficients, finite and >= 0.

`sh_file`
: Gravity field of the central body: a PDS SHADR `.tab` or ICGEM `.gfc` file, detected from its
  contents. Empty (default): no gravity field, and the other `sh_*` fields are ignored. The file is
  read once and cached by path, truncation and file stamp (see
  [Central-body spherical harmonics](spherical_harmonics.md)). A malformed file throws
  `std::runtime_error`. The GM of the file must equal `GMs[0]` within a relative 1e-5.

`sh_nmax`
: Truncation degree, 2 <= `sh_nmax` <= min(degree of the file, 1200).

`sh_mmax`
: Truncation order, <= `sh_nmax` and <= the order of the file. -1 (default) means
  min(`sh_nmax`, order of the file).

`sh_frame`
: Body-fixed frame of the coefficients, e.g. `"MOON_PA"` or `"ITRF93"`. Required with `sh_file`; must
  be known to CSPICE when the constructor runs (load its frame kernels first). Its orientation data
  (e.g. a binary PCK) are needed only during the propagation.

`sh_rho_min`
: Smallest $|\boldsymbol{r}|/R$ at which the field is evaluated, in (0, 1], with R the reference
  radius of the file. Default 1. A propagation that goes below it throws `std::domain_error`.
  Values slightly below 1 (e.g. 0.99) allow low or landing arcs, but the series is not guaranteed
  to converge there.

(cpp-spiceparameters)=
### `SpiceParameters`

```cpp
class SpiceParameters {
public:
    SpiceParameters(double et0, std::vector<std::string> naif_ids, SpiceOptions options = {});
    SpiceParameters(const SpiceParameters& other);
    SpiceParameters& operator=(const SpiceParameters& other);
    ~SpiceParameters();

    double et0() const;
    const std::vector<std::string>& naif_ids() const;
    const std::vector<int>& naif_codes() const;
    const std::string& naif_frame() const;
    const std::vector<double>& GMs() const;
    const std::vector<double>& mus() const;
    double DU() const;
    double TU() const;
    double VU() const;
    bool include_srp() const;
    bool include_spherical_harmonics() const;
    const SpiceOptions& options() const;
};
```

The dynamics model: central body, third bodies, SRP, central-body gravity field, and units. Passed
to every solver and evaluation function.

#### Constructor

```cpp
SpiceParameters(double et0, std::vector<std::string> naif_ids, SpiceOptions options = {});
```

**Parameters**

- `et0`: reference epoch, in TDB seconds past J2000 (ephemeris time). Must be finite. Times of the
  solvers are measured from it: the ephemerides are queried at `et = et0 + t * TU`.
- `naif_ids`: NAIF body names or numeric ids, e.g. `{"MOON", "EARTH", "10"}`, resolved with
  `bods2c_c`. The first is the central body; the others are third bodies. At least one; no body may
  appear twice (also under different names, e.g. `"EARTH"` and `"399"`).
- `options`: see {ref}`SpiceOptions <cpp-spiceoptions>`.

**Throws**

- `std::invalid_argument` for any invalid input listed above and in
  {ref}`SpiceOptions <cpp-spiceoptions>`: non-finite `et0`, empty `naif_ids`, unknown or duplicated
  body, unknown or non-inertial `naif_frame`, missing GM, invalid `GMs` or `DU`, `abcorr` other than
  `"NONE"`, SRP without the Sun or with invalid SRP parameters, invalid gravity-field truncation,
  missing or unknown `sh_frame`, `sh_rho_min` outside (0, 1], or a GM of the file different from
  `GMs[0]`.
- `std::runtime_error` for a SPICE error, or for a gravity-field file that cannot be opened or
  parsed.

**Notes**

- Calls {ref}`init_spice <cpp-init-spice>`, and resolves the body names with CSPICE even when no
  kernel is needed. Built-in body names and inertial frames need no kernels.
- The ephemeris coverage is not checked here; an epoch outside the loaded SPK fails during the
  propagation.
- The object is immutable after construction.

#### Copy, move and destruction

A copy shares the immutable model with the original (through a `std::shared_ptr`) and gets its
own scratch space for the {ref}`eom_* <cpp-eom>` and {ref}`eval_* <cpp-eval-rv>` functions. Copy
assignment does the same. There is no move constructor or move assignment, so a move is a copy and
the source remains valid. The destructor is defined in the library, where the internal types are
complete.

#### Accessors

| Accessor | Returns |
|---|---|
| `et0()` | the reference epoch, TDB seconds past J2000 |
| `naif_ids()` | the body names as given |
| `naif_codes()` | the resolved NAIF codes; `[0]` is the central body |
| `naif_frame()` | the integration frame |
| `GMs()` | gravitational parameters in km^3/s^2, read from the kernels if not given |
| `mus()` | gravitational parameters in DU^3/TU^2, `GMs * TU^2 / DU^3`; `mus()[0]` is 1 with canonical units |
| `DU()` | distance unit, km |
| `TU()` | time unit, s |
| `VU()` | velocity unit, km/s |
| `include_srp()` | whether SRP is on |
| `include_spherical_harmonics()` | whether the central-body gravity field is on |
| `options()` | the `SpiceOptions` given, with `GMs` filled in |

The references stay valid as long as the object (or a copy sharing its model) exists.

(cpp-units)=
#### Units

| | `DU` empty, or `use_canonical_scales == false` | `DU` given and `use_canonical_scales == true` |
|---|---|---|
| distance unit `DU()` | 1 km | `DU` km |
| velocity unit `VU()` | 1 km/s | `sqrt(GMs[0] / DU)` km/s |
| time unit `TU()` | 1 s | `DU / VU` s |
| `mus()` | `GMs` | `GMs / GMs[0]` |

All states, times, tolerances and outputs of the solvers and evaluation functions are in DU, DU/TU
and TU. `GMs`, `et0`, `srp_A` and `srp_m` are always physical.


## Solvers

(cpp-ivpoptions)=
### `IvpOptions`

```cpp
struct IvpOptions {
    double atol = 1e-12;
    double rtol = 1e-14;
    std::vector<double> t_eval;
    long long max_steps = 2000000;
    std::function<void()> poll;
};
```

Integration options of the solvers.

`atol`, `rtol`
: Absolute and relative error tolerances of GSL's `gsl_odeiv2_control_y_new`: each integrated
  component $y_i$ (the states, and the STM and control-sensitivity entries in the `solve_ivpstm_*`
  solvers: all 42 or 84 components, with one `atol` and one `rtol`) is kept
  within $\epsilon_{abs} + \epsilon_{rel}|y_i|$ per step. Both finite and >= 0, not both zero. The `rvm`
  and `meem` solvers require `atol` > 0, a deliberate restriction: see
  [Numerical guidance](zoh_mass_dynamics.md#numerical-guidance). `rv` and `mee` accept 0.
  `atol` is in the units of the state (DU, DU/TU, the MEE, or the mass unit).

`t_eval`
: Output times in TU. Empty (default): every accepted step, including `tspan[0]` and `tspan[1]`.
  Otherwise each time must lie within `tspan` (inclusive), and the times must be strictly monotonic
  in the direction of integration; if `tspan[0] == tspan[1]`, only `{tspan[0]}` is allowed. The
  integrator shortens its steps to land on each time exactly; there is no interpolation.

`max_steps`
: Largest number of accepted steps of one integration, for the `rvm` and `meem` solvers only; `rv` and
  `mee` ignore it. Reaching it with time left throws `std::runtime_error`, naming the time and step
  size. Counted over the whole of `tspan`, with or without `t_eval`, and per segment in
  {ref}`propagate_zoh_* <cpp-propagate-zoh>`. 0 disables the limit, and a negative value throws
  `std::invalid_argument`. It bounds steps, not time, and with sensitivities and an empty `t_eval` a
  stall at the default limit records 1.4 GB; see
  [Integration limits](zoh_mass_dynamics.md#integration-limits).

`poll`
: Called between steps of the `rvm` and `meem` solvers, before the first and then every 1024 accepted
  steps, unless empty. May throw to stop the integration, outside any GSL call; the exception reaches
  the caller. `rv` and `mee` ignore it.

(cpp-ivpresult)=
### `IvpResult`

```cpp
struct IvpResult {
    std::vector<double> t;
    std::vector<double> x;
    std::vector<double> stm;
    std::vector<double> stm_u;
    std::size_t nx = 6;
    std::size_t nu = 0;
    std::size_t size() const { return t.size(); }
};
```

Solution of an initial value problem, with N = `size()` outputs.

`t`
: N output times, in TU.

`x`
: N x 6 states, row-major: component `i` at output `k` is `x[6*k + i]`. Cartesian in DU and DU/TU,
  or MEE with `p` in DU. `L` is not wrapped: it grows by $2\pi$ per revolution.

`stm`
: N x 36 state-transition matrices $\Phi(t_k, t_0) = \partial \boldsymbol{x}(t_k) / \partial \boldsymbol{x}_0$,
  each 6x6 row-major: element `(i, j)` at output `k` is `stm[36*k + 6*i + j]`. Empty for the
  `solve_ivp_*` solvers.

`stm_u`
: N x 28 control sensitivities $\Psi(t_k) = \partial \boldsymbol{x}(t_k) / \partial \boldsymbol{u}$, each 7x4
  row-major: element `(i, j)` at output `k` is `stm_u[28*k + 4*i + j]`. Empty unless
  `solve_ivpstm_rvm` or `solve_ivpstm_meem`.

`nx`, `nu`
: Number of states and controls: 6 and 0 by default, or 7 and 4 for the `rvm` and `meem` solvers, whose
  `x` is N x 7 with the mass last, and whose `stm` is N x 49: `stm[49*k + 7*i + j]`.

(cpp-solve-ivp-rv)=
### `solve_ivp_rv`

```cpp
IvpResult solve_ivp_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                       const std::vector<double>& x0, const IvpOptions& options = {});
```

Propagates a Cartesian state with GSL's embedded Runge-Kutta Prince-Dormand stepper
`gsl_odeiv2_step_rk8pd` (8th order; GSL labels it "(8, 9)") and the step-size control
`gsl_odeiv2_control_y_new(atol, rtol)`. The first trial step is 1 TU in the direction of
integration.

**Parameters**

- `params`: the dynamics model. Not modified; the solver allocates its own scratch space.
- `tspan`: `{t0, tf}`, in TU since `params.et0()`. `tf < t0` integrates backward; `tf == t0`
  returns one row, `x0`.
- `x0`: initial state `[x, y, z, vx, vy, vz]` at `t0`, in DU and DU/TU (6 values).
- `options`: tolerances and output times; see {ref}`IvpOptions <cpp-ivpoptions>`.

**Returns** an {ref}`IvpResult <cpp-ivpresult>` with `t` and `x`; `stm` is empty.

**Throws**

- `std::invalid_argument` if `tspan` does not have 2 finite values, `x0` does not have 6 finite
  values, or `options` are invalid (see {ref}`IvpOptions <cpp-ivpoptions>`).
- `std::domain_error` if the trajectory goes below `sh_rho_min * R`, or if the integration stops
  with the last evaluation of the equations of motion out of the domain (a non-finite derivative,
  at `t0` or when halving the step no longer changes `t`).
- `std::runtime_error` with NAIF's message on a SPICE error (e.g. insufficient ephemeris data), or
  `"integration failed (GSL status ...)"` if GSL gives up otherwise. In particular, a trajectory
  that runs into the centre of a body usually ends this way, with `std::runtime_error` rather than
  `std::domain_error`, because GSL's step-size control fails while the last evaluation succeeded (see
  [Error handling](basic_usage_cpp.md#error-handling)).

A partial trajectory is never returned. Messages start with `solve_ivp_rv: ` and give the time of
the failure in TU.

(cpp-solve-ivp-mee)=
### `solve_ivp_mee`

```cpp
IvpResult solve_ivp_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                        const std::vector<double>& x0, const IvpOptions& options = {});
```

Propagates modified equinoctial elements `[p, f, g, h, k, L]` with the Gauss variational equations.
The forces are the same as in {ref}`solve_ivp_rv <cpp-solve-ivp-rv>`.

**Parameters** as {ref}`solve_ivp_rv <cpp-solve-ivp-rv>`, with `x0` the initial MEE (6 values; `p`
in DU). `x0` is not checked against the domain before the integration starts.

**Returns** an {ref}`IvpResult <cpp-ivpresult>` with `t` and `x` (MEE, `L` not wrapped); `stm` is
empty.

**Throws** as {ref}`solve_ivp_rv <cpp-solve-ivp-rv>`. If the elements leave their domain
($p \le 0$ or $w = 1 + f\cos L + g\sin L \le 0$), the integrator retries with smaller steps; if it
cannot recover, the solver throws `std::domain_error` when the last evaluation was out of the
domain, and otherwise `std::runtime_error` ("integration failed"), as for
{ref}`solve_ivp_rv <cpp-solve-ivp-rv>`. An `x0` outside the domain throws `std::domain_error` at
`t0`.

(cpp-solve-ivpstm-rv)=
### `solve_ivpstm_rv`

```cpp
IvpResult solve_ivpstm_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                          const std::vector<double>& x0, const IvpOptions& options = {});
```

As {ref}`solve_ivp_rv <cpp-solve-ivp-rv>`, also integrating the STM with
$\dot\Phi = A\Phi$, $\Phi(t_0, t_0) = I$, where $A = \partial\dot{\boldsymbol{x}}/\partial\boldsymbol{x}$
includes the gradients of every force in the model. `x0` has 6 values.

**Returns** an {ref}`IvpResult <cpp-ivpresult>` with `t`, `x` and `stm` (N x 36).

(cpp-solve-ivpstm-mee)=
### `solve_ivpstm_mee`

```cpp
IvpResult solve_ivpstm_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                           const std::vector<double>& x0, const IvpOptions& options = {});
```

As {ref}`solve_ivp_mee <cpp-solve-ivp-mee>`, also integrating the STM of the MEE from
$\Phi(t_0, t_0) = I$. `x0` has 6 values. To map the result to Cartesian coordinates, use
{ref}`mee2cart_jacobian <cpp-mee2cart-jacobian>`:
$\Phi_{rv} = J(t)\,\Phi_{mee}\,J(t_0)^{-1}$.

**Returns** an {ref}`IvpResult <cpp-ivpresult>` with `t`, `x` and `stm` (N x 36).


## Equations of motion

(cpp-eom)=
### `eom_rv`, `eom_mee`, `eomstm_rv`, `eomstm_mee`

```cpp
int eom_rv(double t, const double x[], double dxdt[], void* params);
int eom_mee(double t, const double x[], double dxdt[], void* params);
int eomstm_rv(double t, const double x[], double dxdt[], void* params);
int eomstm_mee(double t, const double x[], double dxdt[], void* params);
```

The right-hand sides used by the solvers, with the signature of `gsl_odeiv2_system::function`, so
they can be used with your own GSL integrator (with your own GSL; see
[Equations of motion with your own integrator](basic_usage_cpp.md#equations-of-motion-with-your-own-integrator)).

**Parameters**

- `t`: time, in TU since `et0`.
- `x`: state, Cartesian (`eom_rv`, `eomstm_rv`) or MEE (`eom_mee`, `eomstm_mee`). 6 values for
  `eom_*`; 42 for `eomstm_*`: the state, then $\Phi$ row-major.
- `dxdt`: output, same length as `x`. For `eomstm_*`, the last 36 values are $A\Phi$, row-major.
- `params`: pointer to a `SpiceParameters` (non-null). Its scratch space is written.

**Returns**

- 0 (`GSL_SUCCESS`).
- 1 (`GSL_EDOM`): the state is outside the domain (below `sh_rho_min * R`; MEE with $p \le 0$ or
  $w \le 0$) or the derivative is not finite. GSL then retries with a smaller step.
- 9 (`GSL_EBADFUNC`): SPICE error. GSL stops. CSPICE's error state has been reset.

On failure, `dxdt` is unspecified, and {ref}`eom_error <cpp-eom-error>` describes the failure.
These functions do not throw. The array lengths are not checked.

(cpp-eom-error)=
### `eom_error`

```cpp
std::string eom_error(const SpiceParameters& params);
```

Describes why the last {ref}`eom_* <cpp-eom>` or {ref}`eval_* <cpp-eval-rv>` call on `params` (on
this object, not on its copies) failed, e.g. `"non-finite derivative (e.g. at the center of a
body)"`, or `"SPICE error: SPICE(SPKINSUFFDATA) -- ..."` with NAIF's message. Returns `""` if that
call succeeded or if none was made. The solvers do not use the scratch space of `params`, so they
do not affect `eom_error`. After `eom_rvm` and the like, or `eval_rvm` and `eval_meem`, pass the
`SpiceParameters` of the call (`cp.params()` for a `ControlledParameters`).

(cpp-eomeval)=
### `EomEval`

```cpp
struct EomEval {
    std::array<double, 6> xdot;
    std::array<double, 36> A;
};
```

Result of {ref}`eval_rv <cpp-eval-rv>` and {ref}`eval_mee <cpp-eval-mee>`.

`xdot`
: time derivative of the state, in the units of the state per TU.

`A`
: Jacobian $\partial\dot{\boldsymbol{x}}/\partial\boldsymbol{x}$, 6x6 row-major. For Cartesian
  states, $A = \begin{bmatrix} 0 & I \\ G & 0 \end{bmatrix}$ with $G$ the gradient of the
  acceleration with respect to position, in 1/TU^2.

(cpp-eval-rv)=
### `eval_rv`

```cpp
EomEval eval_rv(const SpiceParameters& params, double t, const std::vector<double>& x);
```

Evaluates the Cartesian equations of motion and their Jacobian at one time and state.

**Parameters**

- `params`: the dynamics model. Taken by `const&`, but its scratch space is written; see
  [Thread safety](basic_usage_cpp.md#thread-safety).
- `t`: time, in TU since `et0`.
- `x`: Cartesian state (6 values), in DU and DU/TU.

**Returns** an {ref}`EomEval <cpp-eomeval>`.

**Throws**

- `std::invalid_argument` if `x` does not have 6 values.
- `std::domain_error` below `sh_rho_min * R`, or if the derivative or the Jacobian is not finite
  (e.g. at the centre of a body).
- `std::runtime_error` with NAIF's message on a SPICE error.

(cpp-eval-mee)=
### `eval_mee`

```cpp
EomEval eval_mee(const SpiceParameters& params, double t, const std::vector<double>& x);
```

Evaluates the Gauss variational equations in MEE and their Jacobian at one time and state.
Parameters, return value and exceptions as {ref}`eval_rv <cpp-eval-rv>`, with `x` the MEE (6
values), and also `std::domain_error` if $p \le 0$ or $w = 1 + f\cos L + g\sin L \le 0$.


## Mass-augmented dynamics

Cartesian + mass (`rvm`, state `[x, y, z, vx, vy, vz, m]`) and MEE + mass (`meem`, state
`[p, f, g, h, k, L, m]`) under a control `u = [tau, Gamma]` held constant over the integration. The mass
`m` is in units of `mass_unit_kg`. The equations, Jacobians and numerical guidance are in
[Mass-augmented dynamics under zero-order-hold control](zoh_mass_dynamics.md).

(cpp-thrustframe)=
### `ThrustFrame`

```cpp
enum class ThrustFrame { rtn, inertial };
```

Frame of the throttle vector `tau` of the `meem` dynamics. `rtn`: the radial, transverse and normal
directions of the current state, so a constant `tau` rotates in inertial space. `inertial`:
`naif_frame`, as `rvm`.

(cpp-thrustoptions)=
### `ThrustOptions`

```cpp
struct ThrustOptions {
    double thrust_N = 0.0;
    double isp_s = 0.0;
    double mass_unit_kg = 0.0;
    double g0 = 9.80665;
    ThrustFrame mee_control_frame = ThrustFrame::rtn;
};
```

Propulsion system. The thrust acceleration is $(c_1/m)\boldsymbol\tau$ in DU/TU^2, and
$\dot m = -c_2\Gamma$ in mass units per TU, with
$c_1 = T_{\max}[\mathrm{kN}] / (\bar m\,\mathrm{DU}/\mathrm{TU}^2)$ and
$c_2 = T_{\max}[\mathrm{N}] / (I_{sp} g_0) \cdot \mathrm{TU}/\bar m$.

`thrust_N`, `isp_s`, `mass_unit_kg`, `g0`
: Maximum thrust in N, specific impulse in s, mass unit $\bar m$ in kg, standard gravity in m/s^2. Not
  validated here: the functions that take them throw `std::invalid_argument` unless each is finite
  and > 0, or if $c_1$ or $c_2$ overflows or underflows.

`mee_control_frame`
: Frame of `tau` for `meem`; `rvm` is always in `naif_frame`.

The dynamics treat `Gamma` as independent of `tau`. Enforcing `Gamma = |tau|` is the responsibility of
the caller or optimizer, and neither `Gamma < 0` nor `|tau| > Gamma` is rejected. SRP always uses the
state mass $\bar m\,m$; `srp_m` is ignored, but must still be > 0 if `include_srp`.

(cpp-controlledparameters)=
### `ControlledParameters`

```cpp
class ControlledParameters {
public:
    ControlledParameters(const SpiceParameters& params, const ThrustOptions& thrust,
                         const std::vector<double>& u);
    ControlledParameters(const SpiceParameters&& params, const ThrustOptions& thrust,
                         const std::vector<double>& u) = delete;

    const SpiceParameters& params() const;
    const ThrustOptions& thrust() const;
    const std::array<double, 4>& u() const;
    double c1() const;
    double c2() const;
};
```

Parameters of {ref}`eom_rvm and the like <cpp-eom-zoh>`. Keeps a pointer to `params`, which must outlive
this object, and so a temporary `SpiceParameters` does not compile. `thrust` and `u` are copied. Copies
point to the same `params`. Like `SpiceParameters`, not thread-safe: the equations of motion write the
scratch space of `params`, also through a `const ControlledParameters`.

**Throws** `std::invalid_argument` if `u` does not have 4 finite values, if `thrust_N`, `isp_s`,
`mass_unit_kg` or `g0` is not finite and > 0, if `mee_control_frame` is not a `ThrustFrame`, or if
`c1` or `c2` overflows or underflows.

(cpp-eomevalcontrol)=
### `EomEvalControl`

```cpp
struct EomEvalControl {
    std::array<double, 7> xdot;
    std::array<double, 49> A;
    std::array<double, 28> B;
};
```

Result of {ref}`eval_rvm and eval_meem <cpp-eval-zoh>`: the time derivative, and the Jacobians
$A = \partial\dot{\boldsymbol{x}}/\partial\boldsymbol{x}$ (7x7, `A[7*i + j]`, column 6 is the mass) and
$B = \partial\dot{\boldsymbol{x}}/\partial\boldsymbol{u}$ (7x4, `B[4*i + j]`, columns `tau` then `Gamma`).

(cpp-zohlinearization)=
### `ZohLinearization`

```cpp
struct ZohLinearization {
    std::vector<double> x_end;
    std::vector<double> Phi;
    std::vector<double> Psi;
    std::vector<double> c;
    std::vector<double> defects;
};
```

Result of {ref}`propagate_zoh_rvm and propagate_zoh_meem <cpp-propagate-zoh>`, for N - 1 segments, in
flat row-major arrays.

| Field | Size | Block `k` |
|---|---|---|
| `x_end` | (N - 1) x 7 | state at the end of segment `k` |
| `Phi` | (N - 1) x 49 | $\Phi_k$, 7x7 |
| `Psi` | (N - 1) x 28 | $\Psi_k$, 7x4 |
| `c` | (N - 1) x 7 | `x_end_k - Phi_k x_ref_k - Psi_k u_ref_k` |
| `defects` | (N - 1) x 7 | `x_ref_{k+1} - x_end_k` |

(cpp-solve-zoh)=
### `solve_ivp_rvm`, `solve_ivp_meem`, `solve_ivpstm_rvm`, `solve_ivpstm_meem`

```cpp
IvpResult solve_ivp_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                        const std::vector<double>& tspan, const std::vector<double>& x0,
                        const std::vector<double>& u, const IvpOptions& options = {});
```

The four solvers have the same signature. `solve_ivp_rvm` and `solve_ivpstm_rvm` take a Cartesian state
and mass, with `tau` in `naif_frame`. `solve_ivp_meem` and `solve_ivpstm_meem` take MEE and mass, with
`tau` in `thrust.mee_control_frame`. The `solve_ivpstm_*` solvers also integrate
$\dot\Phi = A\Phi$, $\Phi(t_0) = I$ and $\dot\Psi = A\Psi + B$, $\Psi(t_0) = 0$.

**Parameters**

- `params`: the dynamics model.
- `thrust`: the propulsion system.
- `tspan`: `{t0, tf}`, in TU since `et0`. `tf < t0` integrates backward; `tf == t0` returns one row.
- `x0`: initial state, 7 values, with the mass last.
- `u`: control `[tau_1, tau_2, tau_3, Gamma]`, dimensionless, held over `tspan`.
- `options`: see {ref}`IvpOptions <cpp-ivpoptions>`.

**Returns** an {ref}`IvpResult <cpp-ivpresult>` with `nx = 7` and `nu = 4`; `stm` and `stm_u` are empty for
`solve_ivp_*`.

**Throws** as {ref}`solve_ivp_rv <cpp-solve-ivp-rv>`, and as {ref}`solve_ivp_mee <cpp-solve-ivp-mee>` for
`meem`; messages start with the function name. Also:

- `std::invalid_argument` as {ref}`ControlledParameters <cpp-controlledparameters>`, or if
  `options.max_steps < 0` or `options.atol` is not > 0.
- `std::domain_error` if `m0` is not > 0, including over a zero-length `tspan`, or if the closed form
  `m0 - c2 Gamma (t - t0)` is not above `64 eps max(m0, |c2 Gamma (t - t0)|)` at the last output time (`tf`, or
  the last of `t_eval`), so that the depletion time itself is refused. This is checked before integrating,
  whatever the thrust, and the message has the depletion time. The factor is a heuristic: results within about
  1e-12 of depletion are not meaningful; see {ref}`Mass depletion <zoh-depletion>`.
- `std::runtime_error` if `options.max_steps` is reached, with the time reached.

(cpp-eom-zoh)=
### `eom_rvm`, `eom_meem`, `eomstm_rvm`, `eomstm_meem`

```cpp
int eom_rvm(double t, const double x[], double dxdt[], void* params);
int eom_meem(double t, const double x[], double dxdt[], void* params);
int eomstm_rvm(double t, const double x[], double dxdt[], void* params);
int eomstm_meem(double t, const double x[], double dxdt[], void* params);
```

The right-hand sides, with the signature of `gsl_odeiv2_system::function`. `params` points to a
{ref}`ControlledParameters <cpp-controlledparameters>`. `x` and `dxdt` must not overlap. `eom_*` take 7
states. `eomstm_*` take 84: the state, then $\Phi$ (7x7) and $\Psi$ (7x4), both row-major.

**Returns** as {ref}`eom_rv <cpp-eom>`. A mass that is not > 0 or not finite returns `GSL_EDOM`, and so does
a non-finite derivative of $\Phi$ or $\Psi$ in `eomstm_*`, unlike `eomstm_rv` and `eomstm_mee`, which check
only the state derivative and $A$.
{ref}`eom_error <cpp-eom-error>` on `cp.params()` describes the last failure.

(cpp-eval-zoh)=
### `eval_rvm`, `eval_meem`

```cpp
EomEvalControl eval_rvm(const SpiceParameters& params, const ThrustOptions& thrust, double t,
                        const std::vector<double>& x, const std::vector<double>& u);
EomEvalControl eval_meem(const SpiceParameters& params, const ThrustOptions& thrust, double t,
                         const std::vector<double>& x, const std::vector<double>& u);
```

Evaluate the equations of motion and their Jacobians at `t` (TU), `x` (7 values) and `u` (4 values). They
use the scratch space of `params`: one thread only.

**Returns** an {ref}`EomEvalControl <cpp-eomevalcontrol>`.

**Throws** `std::invalid_argument` as {ref}`ControlledParameters <cpp-controlledparameters>`, or if `x`
does not have 7 values; `std::domain_error` if `m` is not > 0 or not finite, as well as
{ref}`eval_rv <cpp-eval-rv>` and {ref}`eval_mee <cpp-eval-mee>`; `std::runtime_error` with NAIF's message
on a SPICE error.

(cpp-propagate-zoh)=
### `propagate_zoh_rvm`, `propagate_zoh_meem`

```cpp
ZohLinearization propagate_zoh_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                                   const std::vector<double>& times, const std::vector<double>& x_ref,
                                   const std::vector<double>& u_ref, const IvpOptions& options = {});
ZohLinearization propagate_zoh_meem(const SpiceParameters& params, const ThrustOptions& thrust,
                                    const std::vector<double>& times, const std::vector<double>& x_ref,
                                    const std::vector<double>& u_ref, const IvpOptions& options = {});
```

Propagates each segment of a zero-order-hold reference from its own row of `x_ref`, with one integration
per segment, and returns the linearization $\boldsymbol{x}_{k+1} = \Phi_k\boldsymbol{x}_k + \Psi_k\boldsymbol{u}_k + \boldsymbol{c}_k$.
See [`propagate_zoh`](zoh_mass_dynamics.md#propagate_zoh) for the conventions.

**Parameters**

- `times`: segment boundaries in TU (N >= 2), strictly monotonic; they may decrease, for a backward
  integration.
- `x_ref`: reference states, N x 7 row-major. Do not wrap `L`: a wrapped `L` gives defects of multiples
  of $2\pi$. The last row only enters the last defect, so its mass need not be > 0.
- `u_ref`: controls, (N - 1) x 4 row-major.
- `options`: tolerances and `max_steps`, per segment; `t_eval` must be empty. Only the end of each segment
  is recorded, so memory does not grow with the steps.

**Returns** a {ref}`ZohLinearization <cpp-zohlinearization>`.

**Throws** as {ref}`solve_ivpstm_rvm or solve_ivpstm_meem <cpp-solve-zoh>`, as `std::domain_error`,
`std::invalid_argument` or `std::runtime_error`, with
`"<function>: segment k of n (t = [a, b] TU): "` in front of the message of a failed segment (`k` from
0); `std::invalid_argument` on inconsistent sizes, non-finite or non-monotonic input, or a non-empty
`t_eval`. A class derived from these three, which only `poll` can throw, is rethrown as its base type. Any
other exception from `poll` passes through unchanged, without the segment.


## Spherical harmonics

These functions evaluate a gravity field file directly, for testing and debugging. The solvers do
not need them. The file is loaded through the same cache as `SpiceOptions::sh_file`. Both include
degrees 2 to `nmax` only; the point mass (degree 0) is not included. File and truncation errors
are as for `sh_file`: `std::runtime_error` for a file that cannot be opened or parsed,
`std::invalid_argument` for `nmax < 2`, `nmax > 1200`, `nmax` above the degree of the file, or
`mmax` above `nmax` or the order of the file.

(cpp-sh-accel-bf)=
### `sh_accel_bf`

```cpp
std::tuple<double, std::vector<double>, std::vector<double>> sh_accel_bf(
    const std::string& path, int nmax, const std::vector<double>& r_bf,
    int mmax = -1, double mu = 0.0, double R = 0.0, double rho_min = 1.0);
```

Potential, acceleration and acceleration gradient of the gravity field in the body-fixed frame.
Makes no SPICE call.

**Parameters**

- `path`: gravity field file (PDS SHADR `.tab` or ICGEM `.gfc`).
- `nmax`: truncation degree.
- `r_bf`: body-fixed position (3 values), in km, or in the unit of `R` if `R` is given.
- `mmax`: truncation order; -1 means min(`nmax`, order of the file).
- `mu`: gravitational parameter; 0 means the GM of the file, in km^3/s^2.
- `R`: reference radius, in the unit of `r_bf`; 0 means the radius of the file, in km. Give both
  `mu` and `R` (positive, in consistent units), or neither.
- `rho_min`: smallest $|\boldsymbol{r}_{bf}|/R$ evaluated, in (0, 1].

**Returns** `(U, a, G)`: the potential $U$, the acceleration $\boldsymbol{a}$ (3 values) and
$G = \partial\boldsymbol{a}/\partial\boldsymbol{r}_{bf}$ (9 values, 3x3 row-major), in the units
implied by `mu`, `R` and `r_bf` (km^2/s^2, km/s^2 and 1/s^2 by default).

**Throws**

- `std::invalid_argument` if `r_bf` does not have 3 values, if only one of `mu` and `R` is nonzero
  or either is negative, or if `rho_min` is outside (0, 1]; and for the file and truncation errors
  above.
- `std::domain_error` if $|\boldsymbol{r}_{bf}| < $ `rho_min` $\cdot R$.

(cpp-sh-accel-inertial)=
### `sh_accel_inertial`

```cpp
std::pair<std::vector<double>, std::vector<double>> sh_accel_inertial(
    const std::string& path, int nmax, const std::string& naif_frame, const std::string& sh_frame,
    double et, const std::vector<double>& r, double DU = 1.0, double TU = 1.0, int mmax = -1,
    double rho_min = 1.0);
```

Acceleration and gradient of the gravity field in an inertial frame, exactly as the equations of
motion compute them. Rotates with $T$ = `pxform_c(naif_frame, sh_frame, et)`:
$\boldsymbol{a} = T^\top \boldsymbol{a}_{bf}$, $G = T^\top G_{bf} T$. Uses the GM and the radius of
the file.

**Parameters**

- `path`: gravity field file.
- `nmax`: truncation degree.
- `naif_frame`: inertial frame of `r`, e.g. `"J2000"`.
- `sh_frame`: body-fixed frame of the gravity field, e.g. `"MOON_PA"` or `"ITRF93"`; its kernels
  must be loaded.
- `et`: epoch, in TDB seconds past J2000.
- `r`: position relative to the body (3 values), in DU.
- `DU`: distance unit in km. Default 1.
- `TU`: time unit in s. Default 1.
- `mmax`: truncation order; -1 means min(`nmax`, order of the file).
- `rho_min`: smallest $|\boldsymbol{r}|/R$ evaluated, in (0, 1].

**Returns** `(a, G)`: $\boldsymbol{a}$ in DU/TU^2 (3 values) and
$G = \partial\boldsymbol{a}/\partial\boldsymbol{r}$ in 1/TU^2 (9 values, 3x3 row-major).

**Throws**

- `std::invalid_argument` if `r` does not have 3 values or `rho_min` is outside (0, 1]; and for the
  file and truncation errors above. `DU` and `TU` are not validated.
- `std::runtime_error` with NAIF's message on a SPICE error, e.g. an unknown frame or missing
  orientation data.
- `std::domain_error` if $|\boldsymbol{r}| < $ `rho_min` $\cdot R$.

**Precondition.** This function does not call {ref}`init_spice <cpp-init-spice>`. If neither
`furnsh` nor a `SpiceParameters` constructor has run in the process, a SPICE error prints NAIF's
traceback and exits the process instead of throwing. Call `init_spice()` first in that case.


## Element conversions

None of these functions calls SPICE. `mu` is in the units of the state (e.g. `params.mus()[0]`).

(cpp-mee2cart)=
### `mee2cart`

```cpp
std::vector<double> mee2cart(const std::vector<double>& mee, double mu);
```

Converts MEE to a Cartesian state.

**Parameters**

- `mee`: `[p, f, g, h, k, L]`, prograde; `L` may be any real number.
- `mu`: gravitational parameter.

**Returns** `[x, y, z, vx, vy, vz]` (6 values).

**Throws** `std::invalid_argument` if `mee` does not have 6 values; `std::domain_error` if
$p \le 0$ or $1 + f\cos L + g\sin L \le 0$ (also for NaN).

(cpp-cart2mee)=
### `cart2mee`

```cpp
std::vector<double> cart2mee(const std::vector<double>& rv, double mu);
```

Converts a Cartesian state to MEE.

**Parameters**

- `rv`: `[x, y, z, vx, vy, vz]`.
- `mu`: gravitational parameter.

**Returns** `[p, f, g, h, k, L]` (6 values), with `L` in $(-\pi, \pi]$.

**Throws** `std::invalid_argument` if `rv` does not have 6 values; `std::domain_error` for zero
angular momentum (rectilinear motion), or an inclination within about 1.4e-6 rad of 180 degrees,
where prograde MEE are singular.

(cpp-mee2cart-jacobian)=
### `mee2cart_jacobian`

```cpp
std::vector<double> mee2cart_jacobian(const std::vector<double>& mee, double mu);
```

Jacobian $\partial(x, y, z, v_x, v_y, v_z)/\partial(p, f, g, h, k, L)$ of
{ref}`mee2cart <cpp-mee2cart>`.

**Parameters**

- `mee`: `[p, f, g, h, k, L]`.
- `mu`: gravitational parameter.

**Returns** the 6x6 Jacobian, row-major (36 values).

**Throws** `std::invalid_argument` if `mee` does not have 6 values; `std::domain_error` if
$p \le 0$ or $w \le 0$.
