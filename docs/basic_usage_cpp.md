# Basic usage through C++

This page is the C++ counterpart of [Basic usage through Python](basic_usage_python.md). The
dynamics, units, options and errors are the same in both languages; this page covers what is
specific to C++: building and linking, the option structs, the memory layout of the results, the
exception types, and thread safety. Every symbol is listed in the [C++ API reference](api_cpp.md).

Every program on this page is in `docs/basic_usage_cpp_code/`, and every output shown is its real
output. To compile and run them all, build the C++-only configuration into the directory
`build-cpp` (so the library is `build-cpp/lib/libcampanella.so`), then run the script from the
repository root:

```bash
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON
cmake --build build-cpp
SPICE=/path/to/kernels sh docs/basic_usage_cpp_code/build_and_run.sh
```

The programs read their kernels from the directory in the environment variable `SPICE`:

| Program | Kernels |
|---|---|
| `first_example.cpp`, `eval_eom.cpp`, `gsl_driver.cpp` | `spk/de440.bsp`, `pck/gm_de440.tpc` |
| `units_and_stm.cpp`, `mee.cpp` (pass `GMs`) | `spk/de440.bsp` |
| `errors.cpp` | `spk/de440.bsp`, `pck/gm_de440.tpc`, `pck/moon_pa_de440_200625.bpc`, `fk/moon_de440_250416.tf` |
| `two_cspice.cpp` | `pck/gm_de440.tpc`, `lsk/naif0012.tls` |
| `utc2et.cpp` | `lsk/naif0012.tls` |

See [Loading kernels](#loading-kernels).

(cpp-build-dirs)=
:::{note}
**Which directory is the library in?** It depends on how you configured CMake.

| Configure command | Library | Python module |
|---|---|---|
| `cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON` | `build-cpp/lib/libcampanella.so` | none |
| `cmake -B build -S .` (the default) | none | `build/lib/campanella.cpython-*.so` |
| `cmake -B build -S . -DCAMPANELLA_BUILD_CPP=ON` | `build/lib/libcampanella.so` | `build/lib/campanella.cpython-*.so` |

The library goes into `<build directory>/lib`, where `<build directory>` is whatever you passed to
`-B`. This page assumes `build-cpp`. If you built into `build`, replace `build-cpp` with `build` in
every command.
:::

## Terms used on this page

SPICE, kernel
: NASA's SPICE toolkit reads planetary data from files called *kernels*. An SPK (`.bsp`) holds
  body positions over time (the *ephemeris*), a PCK (`.tpc`, `.bpc`) holds constants such as GMs
  and body orientations, an LSK (`.tls`) holds the leap seconds, and an FK (`.tf`) defines frames.
  CSPICE is the C version of the toolkit; campanella contains its own copy.

NAIF id
: The integer code of a body, e.g. 399 for the Earth, 301 for the Moon, 10 for the Sun. campanella
  takes them as strings (`"399"`), or as names (`"EARTH"`).

Ephemeris time, `et`
: Seconds past the epoch J2000 (2000-01-01 12:00:00 TDB) in the TDB time scale, which has no leap
  seconds. All epochs in campanella are in this scale; [utc2et.cpp](#converting-a-calendar-date-to-et)
  converts a UTC date.

STM
: The state-transition matrix $\Phi(t, t_0) = \partial \boldsymbol{x}(t) / \partial \boldsymbol{x}(t_0)$:
  a 6x6 matrix whose element $(i, j)$ says how much component $i$ of the state at time $t$ changes per
  unit change of component $j$ of the initial state.

MEE
: Modified equinoctial elements `[p, f, g, h, k, L]`, a set of orbital elements without the
  singularities of the classical elements at zero eccentricity and inclination (see
  [GVE in MEE](gve_mee.md)).

Row-major
: The order in which a matrix is stored in a flat array: row by row. Element `(i, j)` of a 6x6
  matrix is at index `6*i + j`. See [Result layout](#result-layout).

`atol`, `rtol`
: The absolute and relative error tolerances of the integrator. At each step, the estimated error
  of each integrated quantity $y_i$ must stay below $\mathrm{atol} + \mathrm{rtol}\,|y_i|$. See
  [Solvers](#solvers).

DU, TU, VU
: The distance, time and velocity units of a model: km, s and km/s unless you choose canonical units.
  See [Units](#units).

GSL
: The GNU Scientific Library, whose Runge-Kutta integrator campanella uses.


## Overview

Every propagation has two steps:

1. Construct a `campanella::SpiceParameters` object, which holds the dynamics model: the central
   body, the third bodies, solar radiation pressure (SRP), the central-body gravity field, and the
   units.
2. Pass it to one of the four solvers:

| | Cartesian `[x, y, z, vx, vy, vz]` | Modified equinoctial elements `[p, f, g, h, k, L]` |
|---|---|---|
| state | `campanella::solve_ivp_rv` | `campanella::solve_ivp_mee` |
| state and STM | `campanella::solve_ivpstm_rv` | `campanella::solve_ivpstm_mee` |

All public symbols are in the namespace `campanella` and are declared in the single header
`include/campanella.hpp`.


## Building a program

### Header and library

`campanella.hpp` includes only standard headers (`<array>`, `<memory>`, `<optional>`, `<string>`,
`<tuple>`, `<vector>`, ...), so a program needs no CSPICE or GSL headers to use the library. It
needs C++17, for `std::optional`.

The shared library `libcampanella.so` (Linux) or `libcampanella.dylib` (macOS) contains GSL and
CSPICE. Build it with `-DCAMPANELLA_BUILD_CPP=ON`, as described in
[Building campanella](build_library.md).

### Plain compiler command

From the repository root, a single-file program compiles and links with

```bash
c++ -std=c++17 my_ivp.cpp \
  -I include \
  -L build-cpp/lib -lcampanella \
  -Wl,-rpath,$PWD/build-cpp/lib \
  -o my_ivp
```

- `-I include` finds `campanella.hpp`.
- `-L build-cpp/lib -lcampanella` links `libcampanella.so` at build time.
- `-Wl,-rpath,...` records the directory of the shared library in the executable, so that the
  program finds it when it starts, without setting `LD_LIBRARY_PATH`. `$PWD` makes the path
  absolute. A relative rpath such as `-Wl,-rpath,build-cpp/lib` would be resolved against the
  directory you *run* the program from, not against the executable, so the program would only
  start from the repository root, and elsewhere it fails with "error while loading shared libraries:
  libcampanella.so: cannot open shared object file". `build_and_run.sh` uses absolute paths.

### CMake: `add_subdirectory`

Another CMake project can build campanella as part of its own build:

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(CAMPANELLA_BUILD_PYTHON OFF CACHE BOOL "")
set(CAMPANELLA_BUILD_CPP ON CACHE BOOL "")
add_subdirectory(path/to/campanella-cpp campanella)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE campanella)
```

The `campanella` target carries its include directory, so `#include "campanella.hpp"` works without
further settings, and CMake sets the rpath of `my_app` in the build tree. This was tested with
CMake 3.16.3 and GCC 9.4 on Linux. Note the following:

- The CSPICE location defaults to `third_party/cspice` under the *top-level* source directory, that
  is, under your project, not under campanella. Configure therefore downloads CSPICE again into your
  project. To reuse an existing toolkit, configure with
  `-DCSPICE_DIR=path/to/campanella-cpp/third_party/cspice`.
- GSL is downloaded and built in your build directory during the first build.
- The examples `solve_twobody`, `solve_rv`, `solve_mee`, `solve_moon_canonical` and `solve_sh` are
  also built, in `<build>/bin`. The library is written to `<build>/lib`.
- In this setup, `solve_sh` looks for its default gravity field under *your* source directory
  (`test-python/data/gggrx_1200a_sha_n10.tab` relative to it) and fails with "cannot open gravity
  field file". Pass the file explicitly:
  `solve_sh path/to/campanella-cpp/test-python/data/gggrx_1200a_sha_n10.tab`.

### One CSPICE per process

On Linux, `libcampanella.so` is linked with `--exclude-libs,ALL`: the CSPICE and GSL symbols inside
it are hidden, so they do not clash with a CSPICE or GSL linked by your program. But the CSPICE
inside campanella has its own kernel pool. Kernels loaded with your own `furnsh_c` are not visible
to campanella, and kernels loaded with `campanella::furnsh` are not visible to your `furnsh_c`.
`docs/basic_usage_cpp_code/two_cspice.cpp` links NAIF's `cspice.a` next to `libcampanella.so` and
shows this:

```text
own CSPICE has GM of Earth: 1
campanella: SpiceParameters: no GM for body "EARTH" (399) in the kernel pool; load a PCK such as gm_de440.tpc, or pass GMs
own CSPICE kernels after campanella::furnsh: 1
```

The first line is after `furnsh_c("gm_de440.tpc")` in the program's own CSPICE; the last line counts
the kernels in the program's CSPICE after `campanella::furnsh("naif0012.tls")`, which did not add to
it. On macOS the library is linked without symbol hiding, so a second CSPICE in the same process may
resolve to either copy. In both cases, load every kernel the propagation needs with
`campanella::furnsh`, and do not link CSPICE (or load spiceypy, in Python) into the same process
unless you keep the two kernel pools apart deliberately.

(cpp-symbol-interposition)=
### Known issue: internal functions can be replaced by yours

The CSPICE and GSL symbols are hidden, but the library's own internal C++ functions are not. Besides
the public `campanella::` API, `libcampanella.so` exports these functions of the *global* namespace:
`mee2cart`, `cart2mee`, `mee2cart_jacobian`, `mee_equinoctial_frame`, `mee_rtn_basis`,
`thirdbody_accel`, `gravity_gradient_center`, `gravity_gradient_third_body`, `srp_coefficient`,
`srp_gradient`, `sh_accel_bf`, `sh_perturbation`, `sh_index`, `load_spherical_harmonics` and
`load_spherical_harmonics_cached` (and the members of `SphericalHarmonicsModel` and
`SphericalHarmonicsWorkspace`). They are listed by `nm -DC --defined-only build-cpp/lib/libcampanella.so`.
The library calls them through the dynamic linker. So if your program defines a global function
with the same name *and* the same parameter types, the library silently calls yours instead. No
compiler or linker error tells you. For example, with

```cpp
std::vector<double> mee2cart(const std::vector<double>&, double) { return {42, 0, 0, 0, 0, 0}; }
```

in the program, `campanella::mee2cart({7000, 0, 0, 0, 0, 0}, 398600.0)[0]` returns 42 instead of
7000. Any of the following avoids it:

- put your own functions in a namespace (good practice anyway);
- do not give a global function one of the names above with the same parameter types (a different
  signature is a different symbol and is harmless);
- compile your code with `-fvisibility=hidden`, so that your functions are not exported to the
  library. With this flag, the example above returns 7000.

This was tested on Linux with GCC 9.4; macOS was not tested. The fix belongs in the library, which
should move these functions into `campanella::detail` or build with hidden visibility.

GSL is different: the solvers use the GSL inside the library, but a program that calls the `eom_*`
functions from its own GSL integrator must link its own GSL (see
[Equations of motion with your own integrator](#equations-of-motion-with-your-own-integrator)).


## A first example

An Earth-centred orbit with the Moon and the Sun as third bodies, in km and s:

```{literalinclude} basic_usage_cpp_code/first_example.cpp
:language: cpp
```

Output:

```text
codes: 399 301 10
GMs [km^3/s^2]: 398600.435507 4902.800118 1.327124e+11
DU = 1 km, TU = 1 s, VU = 1 km/s
N = 58, t[0] = 0, t[N-1] = 3600
x(tf) = -4638.146085040 -5052.227016189 0.000928376 5.592933472298 -5.226923028429 -0.000000947384
t =      0 s  r = [  7000.000000      0.000000      0.000000] km
t =    900 s  r = [  3948.615303   5734.978218      0.000046] km
t =   1800 s  r = [ -2570.135940   6385.733298      0.000500] km
t =   2700 s  r = [ -6720.127497   1237.099508      0.001120] km
t =   3600 s  r = [ -4638.146085  -5052.227016      0.000928] km
backward: t = 0, |r - r0| = 8.654e-11 km
```


## Loading kernels

`campanella::furnsh(path)` loads a SPICE kernel into the CSPICE inside the library. It throws
`std::runtime_error` with NAIF's message if the kernel cannot be loaded.

`campanella::init_spice()` sets CSPICE's error action to `RETURN` and turns off its error printing,
so that a SPICE error becomes a C++ exception instead of a printed traceback and process exit.
`furnsh` and the `SpiceParameters` constructor call it, so you do not normally need to call it
yourself. The exception is {ref}`sh_accel_inertial <cpp-sh-accel-inertial>`, which does not call
it: call `init_spice()` (or `furnsh`) first.

Which kernels a model needs:

| Model | Kernels |
|---|---|
| a single body, `GMs` given, no SRP, no gravity field | none (no ephemeris is queried) |
| `GMs` not given | a PCK with `BODY<code>_GM` values, e.g. `gm_de440.tpc` |
| third bodies, or SRP | an SPK covering every body over the propagation, e.g. `de440.bsp` |
| gravity field | the orientation data of `sh_frame`: `earth_latest_high_prec.bpc` for `ITRF93`; `moon_pa_de440_200625.bpc` and `moon_de440_250416.tf` for `MOON_PA` (see [Central-body spherical harmonics](spherical_harmonics.md)) |

No model needs the leap-second kernel (LSK) `naif0012.tls`. The library works in ephemeris time
throughout: it passes `et = et0 + t * TU` straight to CSPICE, and reading an SPK or a binary PCK at
a given `et` involves no leap seconds. The LSK is needed only to convert a UTC calendar date into
`et`, which campanella does not do (see
[Converting a calendar date to et](#converting-a-calendar-date-to-et)). This was checked by
propagating an Earth orbit with the Moon, the Sun, SRP and an `ITRF93` gravity field, and a lunar
orbit with a `MOON_PA` gravity field, with no LSK loaded. Except for `utc2et.cpp`, the programs on
this page do not load it.

The ephemeris coverage is checked only when the ephemeris is queried, that is, during the
propagation. A `SpiceParameters` whose `et0` is outside the SPK is constructed without error.

### Converting a calendar date to et

campanella takes epochs only as ephemeris time and does not expose CSPICE's date functions, and its
own CSPICE is hidden (see [One CSPICE per process](#one-cspice-per-process)). To get `et0` for a
calendar date, use any SPICE installation once and write the number into your program:

- in Python, `spiceypy.str2et("2025-01-01T00:00:00")` after `spiceypy.furnsh("naif0012.tls")`
  (spiceypy is installed with `uv sync`);
- in C++, a separate small program linked against NAIF's CSPICE, which the build downloads into
  `third_party/cspice`. `docs/basic_usage_cpp_code/utc2et.cpp` does this; it does not use
  campanella:

```{literalinclude} basic_usage_cpp_code/utc2et.cpp
:language: cpp
:start-at: // report SPICE errors
:end-before: return 0;
:dedent: 4
```

Compile it against `cspice.a` (not `libcampanella`), and run it with dates as arguments:

```bash
c++ -std=c++17 docs/basic_usage_cpp_code/utc2et.cpp \
  -I third_party/cspice/include third_party/cspice/lib/cspice.a -lm -o utc2et
SPICE=/path/to/kernels ./utc2et 2025-01-01T00:00:00 2023-06-17T12:00:00
```

```text
2025-01-01T00:00:00 UTC -> et = 788961669.1839275 s past J2000 (TDB)
2023-06-17T12:00:00 UTC -> et = 740275269.1844999 s past J2000 (TDB)
```

The values agree with `spiceypy.str2et` to the last digit. The difference from the naive count of
SI seconds (788961600 s for 2025-01-01, i.e. calendar days times 86400 since 2000-01-01T12:00:00)
is 69.18 s: 37 s of TAI − UTC (the accumulated leap seconds) at that date, 32.184 s of TT − TAI,
and a periodic TDB − TT term below 2 ms.


## `SpiceParameters` and `SpiceOptions`

```cpp
campanella::SpiceParameters(double et0, std::vector<std::string> naif_ids,
                            campanella::SpiceOptions options = {});
```

`et0` is the reference epoch in TDB seconds past J2000. `naif_ids` are NAIF names or numeric ids,
for example `{"MOON", "EARTH", "SUN"}` or `{"301", "399", "10"}`; the first one is the central body
and the others are third bodies. The optional inputs are the fields of `campanella::SpiceOptions`.
C++17 has no designated initializers, so set the fields by name and then pass the struct:

```{literalinclude} basic_usage_cpp_code/units_and_stm.cpp
:language: cpp
:start-at: const double et0 = 740463475.5117936;
:end-at: const campanella::SpiceParameters can(et0, bodies, options);
:dedent: 8
```

This excerpt from `units_and_stm.cpp` builds two models from the same options, the second in
canonical units.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `naif_frame` | `std::string` | `"J2000"` | inertial frame of integration |
| `GMs` | `std::vector<double>` | empty | km^3/s^2, one per NAIF id; empty reads `GM` of each body from the kernel pool |
| `DU` | `std::optional<double>` | empty | distance unit in km; see [Units](#units) |
| `use_canonical_scales` | `bool` | `true` | if `false`, `DU` is ignored and the units are km and s |
| `abcorr` | `std::string` | `"NONE"` | aberration correction; only `"NONE"` (any case) is accepted |
| `include_srp` | `bool` | `false` | cannonball SRP; the Sun must be among `naif_ids` |
| `srp_A`, `srp_m` | `double` | `0.0` | area in m^2 and mass in kg |
| `srp_rhoS`, `srp_rhoD` | `double` | `0.0` | specular and diffuse reflection coefficients |
| `sh_file` | `std::string` | `""` | central-body gravity field (`.tab` or `.gfc`); empty means none |
| `sh_nmax`, `sh_mmax` | `int` | `0`, `-1` | truncation degree and order |
| `sh_frame` | `std::string` | `""` | body-fixed frame of the gravity field |
| `sh_rho_min` | `double` | `1.0` | smallest `|r|/R` at which the field is evaluated |

The constructor validates every input and resolves the bodies, the frame, the GMs and the units
once. The object is then immutable. Its accessors return the resolved values: `et0()`,
`naif_ids()`, `naif_codes()`, `naif_frame()`, `GMs()`, `mus()`, `DU()`, `TU()`, `VU()`,
`include_srp()`, `include_spherical_harmonics()`, and `options()`, which returns the inputs with
`GMs` filled in.

Copying a `SpiceParameters` is cheap: the copy shares the immutable model and gets its own scratch
space for the `eom_*` and `eval_*` functions. You can store `SpiceParameters` objects in standard
containers and return them from functions as usual. (For readers who know move semantics: the class
has no move constructor, so `std::move` makes a copy and leaves the source usable. This only costs
the allocation of a small scratch space.)

The physics of each option (third-body accelerations, SRP, the gravity field) is described in
[Basic usage through Python](basic_usage_python.md#spiceparameters) and in
[Central-body spherical harmonics](spherical_harmonics.md).


## Units

Without `DU` (or with `use_canonical_scales = false`), every state, time and output is in km, km/s
and s. With `DU` given, the units are canonical: the distance unit is `DU` km, the velocity unit is
`VU = sqrt(GMs[0] / DU)` km/s and the time unit is `TU = DU / VU` s, so the central body's scaled
gravitational parameter `mus()[0]` is 1. `GMs`, `et0`, `srp_A` and `srp_m` are always physical. The
ephemerides are queried at `et = et0 + t * TU`.

`docs/basic_usage_cpp_code/units_and_stm.cpp` propagates a Moon-centred state and its STM for 20
days, with SRP, in km and s and with `DU = 100000` km, and compares the two. The states convert back
with `DU` and `VU`, and the STM with `Phi_km = D Phi_can D^-1`,
`D = diag(DU, DU, DU, VU, VU, VU)`:

```{literalinclude} basic_usage_cpp_code/units_and_stm.cpp
:language: cpp
:start-at: // states back to km and km/s
:end-before: std::printf("max STM difference
:dedent: 8
```

Output:

```text
DU = 100000 km, TU = 451624.907373 s (5.2271 days), VU = 0.221422686 km/s
mus = 1 81.30056 27068650.8276519
accepted steps: km/s run 1158, canonical run 866
res.x.size() = 5202 = 6 N, res.stm.size() = 31212 = 36 N
max state difference: 2.165e-08 (km, or mm/s)
max STM difference (relative): 1.293e-11
Phi(tf, t0) in km and s, first row: -3.472363 -2.770202 -3.575196 1253389.422 -1849251.888 -854831.559
```

The two runs take different steps because `atol` applies to the scaled variables: `atol = 1e-12`
is 1e-12 km in the first run and 1e-7 km in the second. For MEE, only `p` scales with `DU`; the STM
of MEE converts with `D = diag(DU, 1, 1, 1, 1, 1)`.


## Solvers

```cpp
campanella::IvpResult campanella::solve_ivp_rv(const SpiceParameters& params,
                                               const std::vector<double>& tspan,
                                               const std::vector<double>& x0,
                                               const IvpOptions& options = {});
```

`solve_ivp_mee`, `solve_ivpstm_rv` and `solve_ivpstm_mee` have the same signature.

- `tspan = {t0, tf}`, in TU since `et0`. If `tf < t0` the integration runs backward. If `tf == t0`
  the result is the single row `x0` (and the identity STM).
- `x0` always has 6 elements. The `solve_ivpstm_*` solvers start the STM from `Phi(t0, t0) = I`.
- `options` is a `campanella::IvpOptions`:

| Field | Default | Meaning |
|---|---|---|
| `atol` | `1e-12` | absolute error tolerance |
| `rtol` | `1e-14` | relative error tolerance |
| `t_eval` | empty | output times in TU; empty means every accepted step |

With `t_eval` empty, the result has every accepted step, including `t0` and `tf`. With `t_eval`,
the result has exactly those times, which must be strictly monotonic in the direction of
integration and lie within `tspan`; `t0` is in the result only if it is in `t_eval`. The output
times are hit exactly (the integrator shortens the step to land on them), not interpolated.

The integrator is GSL's `rk8pd` stepper, an explicit embedded Runge-Kutta pair of Prince and
Dormand. GSL labels it "(8, 9)" and its order function returns 8; the pair of the original paper is
known as RK8(7)13M, an 8th-order solution with a 7th-order error estimate. Step-size control is
`gsl_odeiv2_control_y_new(atol, rtol)`:
each component $y_i$ is kept within $\epsilon_{abs} + \epsilon_{rel} |y_i|$ per step. In the
`solve_ivpstm_*` solvers the 36 STM components are error-controlled with the same tolerances as
the state. The first trial step is 1 TU.

The solvers take `params` by `const&` and do not write to it: each call allocates its own scratch
space. They never return a partial trajectory: any failure throws (see
[Error handling](#error-handling)).


## Result layout

```cpp
struct IvpResult {
    std::vector<double> t;     // N
    std::vector<double> x;     // N x 6, row-major
    std::vector<double> stm;   // N x 36, each 6x6 row-major; empty for solve_ivp_*
    std::size_t size() const;  // N
};
```

For output `k` (0 <= k < N):

| Quantity | Element |
|---|---|
| time | `res.t[k]` |
| component `i` of the state | `res.x[6*k + i]` |
| element `(i, j)` of `Phi(t_k, t0)` | `res.stm[36*k + 6*i + j]` |

so `&res.x[6*k]` points to the 6 contiguous values of the state at `t_k`, and `&res.stm[36*k]` to
its STM, row by row. Element `(i, j)` of the STM is the derivative of component `i` of the state
at `t_k` with respect to component `j` of `x0`. Cartesian states are in DU and DU/TU; MEE are
`[p (DU), f, g, h, k, L]`, with `L` not wrapped to $(-\pi, \pi]$: it keeps growing by $2\pi$ per
revolution (decreasing in backward propagation).

Because the data are contiguous, a linear-algebra library can use them without copying. For
example, with the [Eigen](https://eigen.tuxfamily.org) library (not a dependency of campanella),
`Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>>(res.x.data(), res.size(), 6)`
views `res.x` as an N x 6 matrix.


## Modified equinoctial elements

`solve_ivp_mee` and `solve_ivpstm_mee` integrate the Gauss variational equations in MEE, with the
same forces as the Cartesian solvers (see [GVE in MEE](gve_mee.md)). `cart2mee` and `mee2cart`
convert states, and `mee2cart_jacobian` returns $\partial(\boldsymbol{r}, \boldsymbol{v}) /
\partial(\text{MEE})$, row-major, which maps the MEE STM to the Cartesian one:
$\Phi_{rv}(t_f, t_0) = J(t_f)\, \Phi_{mee}(t_f, t_0)\, J(t_0)^{-1}$. Use `params.mus()[0]` as `mu`,
so that the conversions are in the units of the model.

```{literalinclude} basic_usage_cpp_code/mee.cpp
:language: cpp
:start-at: campanella::SpiceOptions options;
:end-before: // accepted steps of the four solvers
:dedent: 8
```

Output:

```text
mee0 = 5778.190023 -0.346516098 0.851404899 1.978420907 0.617810980 -1.184288479
accepted steps: Cartesian 1154, MEE 7960
L(tf) = 21.712831 rad
max position difference: 3.587e-09 km
max relative STM difference: 9.821e-12
```

The GMs are the set used by the test suite and `examples/`, not the DE440 values of
`gm_de440.tpc`; any consistent set works.

The MEE STM run takes almost seven times as many steps as the Cartesian one. The cause is the STM,
not the elements. `mee.cpp` also counts the accepted steps of all four solvers on the same 20-day
arc:

```text
accepted steps            ivp_rv  ivp_mee  ivpstm_rv  ivpstm_mee
km, s                        790      662       1154        7960
km, s, atol = 1e-10          651      417       1068        1421
km, s, rtol = 1e-12          465      351        713        1095
DU = p(t0)                   434      401        890         929
```

Without the STM, MEE need *fewer* steps than Cartesian coordinates (662 against 790). With the STM,
the 36 STM entries are error-controlled with the same `atol` and `rtol` as the state. In km and s
the entries of the MEE STM range from about 7e-5 to 7e5 at $t_f$ (e.g.
$\partial p/\partial g \approx 7\times10^5$ km, $\partial f/\partial p \approx -6\times10^{-4}$
km$^{-1}$). The default tolerances (`atol = 1e-12`, `rtol = 1e-14`, close to double precision)
make this run very sensitive to them: relaxing `atol` to 1e-10 or `rtol` to 1e-12 cuts the
steps by a factor of 5.6 or 7.3. Canonical units with `DU` equal to the initial `p` bring it down to 929
steps, close to the Cartesian STM run. With an STM in MEE, use canonical units or choose the
tolerances deliberately. The elements are
prograde (retrograde factor $I = +1$); `cart2mee` throws `std::domain_error` within about 1.4e-6
rad of an inclination of 180 degrees.


## Single-point evaluation

`eval_rv` and `eval_mee` return the time derivative and its Jacobian at one time (in TU) and state:

```cpp
struct EomEval {
    std::array<double, 6> xdot;    // dx/dt
    std::array<double, 36> A;      // d(xdot)/dx, 6x6 row-major
};
campanella::EomEval campanella::eval_rv(const SpiceParameters& params, double t, const std::vector<double>& x);
campanella::EomEval campanella::eval_mee(const SpiceParameters& params, double t, const std::vector<double>& x);
```

From `docs/basic_usage_cpp_code/eval_eom.cpp`, for the Earth-centred state of the first example
(r = 7000 km, v = 7.5 km/s, a little below the circular speed of 7.546 km/s, so this is the
apoapsis of an orbit with e = 0.012), with the Moon and the Sun:

```text
xdot = 0.000000 7.500000 0.000000 -8.134703450e-03 -3.228331853e-10 -1.778184664e-10
A (row-major 6x6):
   0.000e+00   0.000e+00   0.000e+00   1.000e+00   0.000e+00   0.000e+00
   0.000e+00   0.000e+00   0.000e+00   0.000e+00   1.000e+00   0.000e+00
   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   1.000e+00
   2.324e-06  -4.721e-14  -2.599e-14   0.000e+00   0.000e+00   0.000e+00
  -4.721e-14  -1.162e-06   1.287e-13   0.000e+00   0.000e+00   0.000e+00
  -2.599e-14   1.287e-13  -1.162e-06   0.000e+00   0.000e+00   0.000e+00
eom_rv status 0, xdot[3] = -8.134703450e-03
MEE xdot = -5.953e-07 -8.504e-11 9.010e-11 -1.185e-11 -0.000e+00 1.071428571e-03
```

Although `params` is passed by `const&`, `eval_*` (and `eom_*`) write the scratch space held inside
it. See [Thread safety](#thread-safety).

### Equations of motion with your own integrator

`eom_rv`, `eom_mee`, `eomstm_rv` and `eomstm_mee` have the signature of
`gsl_odeiv2_system::function`, with `params` pointing to a `SpiceParameters`. The `eomstm_*`
functions take 42 values: the state, then $\Phi$ row-major, and return $\dot\Phi = A \Phi$. They
return 0 on success, `GSL_EDOM` (1) for a state out of the domain or a non-finite derivative (GSL
then retries with a smaller step), and `GSL_EBADFUNC` (9) on a SPICE error (GSL stops). After a
failure, `campanella::eom_error(params)` gives the reason, with NAIF's message for a SPICE error;
it returns `""` after a successful call.

The GSL inside `libcampanella` is hidden, so a program that drives the integration itself needs its
own GSL headers and libraries. Two ways to get them:

- the system package, e.g. `sudo apt install libgsl-dev` on Debian/Ubuntu or `brew install gsl` on
  macOS, then link with `-lgsl -lgslcblas`;
- the copy that the campanella build already downloaded and compiled, in
  `build-cpp/_deps/gsl-build/install` (headers in `include/`, static libraries `lib/libgsl.a` and
  `lib/libgslcblas.a`). `build_and_run.sh` uses this one:

```bash
gsl=build-cpp/_deps/gsl-build/install
c++ -std=c++17 docs/basic_usage_cpp_code/gsl_driver.cpp -I include -I $gsl/include \
  -L build-cpp/lib -lcampanella -Wl,-rpath,$PWD/build-cpp/lib \
  $gsl/lib/libgsl.a $gsl/lib/libgslcblas.a -o gsl_driver
```

`docs/basic_usage_cpp_code/gsl_driver.cpp`:

```{literalinclude} basic_usage_cpp_code/gsl_driver.cpp
:language: cpp
:start-at: // not const
:end-before: const campanella::IvpResult res
:dedent: 8
```

```text
GSL driver: r(tf) = -4638.146085040 -5052.227016189 0.000928376 km
max difference to solve_ivp_rv: 0.000e+00 km
```

With the same stepper, tolerances and initial step, the result is bitwise identical to
`solve_ivp_rv`.


## Error handling

Every function reports failure by throwing, and none exits the process. The only output the library
writes is a warning on `std::cerr` from the gravity-field loader, for a file with non-zero degree-1
coefficients (which are ignored) or with rows above the order given in its header. The exception
types are:

| Exception | When |
|---|---|
| `std::invalid_argument` | bad input: wrong vector lengths, non-finite values, unknown body or frame, non-inertial `naif_frame`, a body listed twice, no GM in the kernel pool, `GMs` of the wrong length, invalid `DU`, SRP without the Sun, invalid SRP or gravity-field options, `atol`/`rtol` out of range, `t_eval` not within `tspan` or not monotonic |
| `std::domain_error` | a state outside the domain of the model: below `sh_rho_min` times the reference radius of the gravity field; MEE with `p <= 0` or `w = 1 + f cos L + g sin L <= 0` (in `eval_mee`, `mee2cart`, `mee2cart_jacobian`, and in the MEE solvers if the state stays there, but see the note below); a non-finite derivative in `eval_*`; `cart2mee` with zero angular momentum or an inclination of 180 degrees |
| `std::runtime_error` | a SPICE error, with NAIF's message (missing kernel, insufficient ephemeris coverage, unknown frame in `sh_accel_inertial`); a malformed gravity-field file; a GSL failure, for example when the step size can no longer be reduced |

`std::invalid_argument` and `std::domain_error` both derive from `std::logic_error`, and all three
from `std::exception`; catch the specific types first. Most messages start with the name of the function
and, for failures during a propagation, give the time of the failure in TU.

`docs/basic_usage_cpp_code/errors.cpp` triggers one of each (the kernel directory is shown as
`$SPICE`):

```text
missing kernel file:
  std::runtime_error: SPICE(NOSUCHFILE) -- The attempt to load "$SPICE/spk/no_such_file.bsp" by the routine FURNSH failed. It could not be located.
no GM in the kernel pool:
  std::invalid_argument: SpiceParameters: no GM for body "EARTH" (399) in the kernel pool; load a PCK such as gm_de440.tpc, or pass GMs
unknown body:
  std::invalid_argument: SpiceParameters: unknown NAIF body "PLANET_NINE"
SRP without the Sun:
  std::invalid_argument: SpiceParameters: include_srp requires the Sun ("10" or "SUN") in naif_ids
x0 of the wrong length:
  std::invalid_argument: solve_ivp_rv: x0 must have length 6
t_eval outside tspan:
  std::invalid_argument: solve_ivp_rv: t_eval must lie within tspan
epoch outside the ephemeris:
  std::runtime_error: solve_ivp_rv: SPICE error at t = 0 TU: SPICE(SPKINSUFFDATA) -- Insufficient ephemeris data has been loaded to compute the position of 301 (MOON) relative to 399 (EARTH) at the ephemeris epoch 2950 AUG 30 17:20:00.000.
MEE out of their domain:
  std::domain_error: eval_mee: MEE out of domain (p <= 0 or w = 1 + f cos(L) + g sin(L) <= 0) at t = 0 TU
mee2cart with p < 0:
  std::domain_error: mee2cart: p must be positive.
trajectory through the centre:
  std::runtime_error: solve_ivp_rv: integration failed (GSL status -1: failure) near t = 454.423114293575 TU
below sh_rho_min * R:
  std::domain_error: solve_ivp_rv: spherical harmonics: the trajectory went below sh_rho_min * R (|r|/R = 0.999999999999999 < 1, R = 1738 km) at t = 2098.35987902801 TU; the integration was stopped. For low or landing arcs, a sh_rho_min slightly below 1 (e.g. 0.99) may be used, but the series is not guaranteed to converge there.
eom_rv at the origin returned 1: non-finite derivative (e.g. at the center of a body)
after a successful call, eom_error is ""
```

Two cases deserve a note:

- **Domain errors during a propagation.** When a trial step leaves the domain, the integrator
  retries with a smaller step. A solver throws `std::domain_error` only if the last evaluation of
  the equations of motion failed: at `t0`, or when halving the step no longer changes the time.
  If instead the integrator gives up because its step-size control cannot reduce the step any
  further while the last evaluation succeeded, it throws `std::runtime_error` ("integration failed
  (GSL status -1: failure)"). For a trajectory that runs into a body, the second case is the usual
  one, as in the radial fall onto the Earth above. Handle both if a trajectory may hit a body.
  Going below `sh_rho_min * R` is detected directly and throws `std::domain_error`.
- **SPICE errors.** CSPICE's error state is reset after each error is turned into an exception, so
  the library stays usable after catching it. The SPICE error ends the integration at once; no retry
  is attempted.


## Thread safety

CSPICE is not thread-safe: its kernel pool, its file handles and its error state are global to the
process. Every call that may reach CSPICE must therefore come from one thread at a time. These are:

- `furnsh`, `init_spice` and the `SpiceParameters` constructor (which resolves the names with
  CSPICE even when no kernels are needed);
- the solvers, `eval_*` and `eom_*` whenever the model has third bodies, SRP or a gravity field;
- `sh_accel_inertial`.

Using one `SpiceParameters` per thread does not help, since the CSPICE state is shared. In addition:

- `eval_*` and `eom_*` write the scratch space inside the `SpiceParameters` they are given, even
  through a `const&`, and `eom_error` reads it. Do not call them on the same object from two
  threads. A copy has its own scratch space.
- Each solver call temporarily disables GSL's global error handler and restores the previous one
  when it returns. This global state is not synchronized between threads.
- The cache of gravity-field files is protected by a mutex.

In short: run campanella from one thread, or serialize all calls with a mutex. Parallelism across
processes (for example one process per Monte Carlo batch) is safe, since each process has its own
CSPICE.
