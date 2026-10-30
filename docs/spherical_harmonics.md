# Central-body spherical harmonics

All solvers (`solve_ivp_rv`, `solve_ivp_mee`, `solve_ivpstm_rv`, `solve_ivpstm_mee`) and `eval_rv` / `eval_mee` can add the gravity field of the central body, the first of the `naif_ids` of `SpiceParameters`, up to an arbitrary degree and order:

```
U = (GM/R) Σ_{n=2}^{nmax} Σ_{m=0}^{min(n,mmax)} (R/r)^{n+1} P̄_nm(sin φ) (C̄_nm cos mλ + S̄_nm sin mλ)
```

The field is off unless `sh_file` is given, and the existing results do not change when it is off.

## Usage

```python
import campanella

for k in ["lsk/naif0012.tls", "spk/de440.bsp",
          "pck/moon_pa_de440_200625.bpc", "fk/moon_de440_250416.tf"]:
    campanella.furnsh("/path/to/spice/" + k)

params = campanella.SpiceParameters(
    et0, ["MOON", "EARTH", "SUN"],
    GMs = [4.902800582147760e+03, 398600.432896939, 132712197035.766],
    sh_file = "/path/to/gggrx_1200a_sha.tab",   # PDS .tab or ICGEM .gfc
    sh_nmax = 50,                               # truncation degree
    sh_mmax = -1,                               # truncation order, -1 = min(sh_nmax, order of the file)
    sh_frame = "MOON_PA",                       # body-fixed frame of the coefficients
    sh_rho_min = 1.0,                           # smallest |r|/R evaluated (see below)
)
res = campanella.solve_ivp_rv(params, [0.0, 86400.0], state0)
```

In C++ they are the `sh_*` fields of `campanella::SpiceOptions` in `include/campanella.hpp`. `examples/solve_sh.cpp` propagates a 100 km lunar orbit and checks the STM against finite differences.

The STM solvers add the gradient of the SH acceleration. In `solve_ivpstm_rv` it is added to the gravity gradient. In the GVE it is added to the perturbation gradient G of [gve_mee.md](gve_mee.md) §5.5.

Test and debugging helpers:

- `campanella.load_spherical_harmonics(path, nmax, mmax=-1)` returns a dict: `GM` (km³/s²), `R` (km), `file_degree`, `file_order`, `mmax`, `tide_system`, and `C[n][m]`, `S[n][m]`.
- `campanella.sh_accel_bf(path, nmax, r_bf, mmax=-1, mu=0, R=0, rho_min=1)` returns `(U, a, G)` in the body-fixed frame. By default it uses km and s with the GM and radius of the file. Give both `mu` and `R` (in your units), or neither; giving only one raises `ValueError`.
- `campanella.sh_accel_inertial(path, nmax, naif_frame, sh_frame, et, r, DU=1, TU=1, mmax=-1, rho_min=1)` returns `(a, G)` in an inertial frame, exactly as the solvers use them.

`G` is 3×3, flattened row-major. All three exclude the point-mass term.

## Coefficient files

The format is detected from the file content:

- **PDS SHADR `.tab`** (GRAIL gravity fields). The file is SHADR if its first line (after an optional UTF-8 byte order mark) has 8 comma-separated numbers: R [km], GM [km³/s²], σGM, degree, order, normalization state, reference longitude and latitude. The coefficient lines are `n, m, C, S, σC, σS`. The loader stops reading at `n > nmax`, so a degree-50 load reads only about 160 kB of the 88 MB file.
  - GRGM1200A (recommended): https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.tab
  - GL0660B: `jggrx_0660b_sha.tab` in the same directory.
- **ICGEM `.gfc`** (EGM2008, EGM96). Any other file is read as ICGEM:
  - The file may start with free text of any length, and optionally `begin_of_head`. Everything before `begin_of_head` is ignored.
  - The header has `keyword value` lines up to `end_of_head`, then lines `gfc n m C S σC σS`. Leading blanks are ignored.
  - Header lines that are not keyword lines are skipped, for example a comment starting with "radius of ...". Numeric keywords (`earth_gravity_constant`, `radius`, `max_degree`) need a numeric value. `norm` and `tide_system` need exactly one value.
  - A keyword that appears twice in the header (after `begin_of_head`, if present) is rejected.
  - A file without `end_of_head` is rejected. GM is converted from m³/s² to km³/s² and the radius from m to km. Fortran `D` exponents are accepted. Time-variable models (`gfct`, `trnd`, `acos`, `asin`) are rejected.
  - EGM2008: https://icgem.gfz.de/getmodel/gfc/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc
  - EGM96: https://icgem.gfz.de/getmodel/gfc/971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc

Only fully normalized coefficients are accepted (SHADR normalization state 1, ICGEM `fully_normalized`; ICGEM uses that as the default when the `norm` line is missing).

The loader throws in these cases:

- `nmax < 2`, `nmax > 1200`, or `nmax` above the degree of the file
- `mmax > nmax`, or `mmax` above the order of the file
- a non-finite coefficient (NaN, inf)
- any coefficient with 2 ≤ n ≤ nmax and m ≤ min(n, mmax) is missing (for example, a truncated file)

`mmax = -1` means min(nmax, order of the file). The order is read from the SHADR header. If a `.tab` file has rows above that order, a warning is printed and the rows are not used. ICGEM has no order keyword, so for an order-limited `.gfc` file pass `mmax` explicitly; otherwise the missing rows are reported.

Degrees 0 and 1 are not evaluated. The point mass `GM[0]` is already in the EOMs, and degree 1 vanishes in a centre-of-mass frame. The loader warns if degree-1 coefficients are non-zero.

Loaded models are cached so that repeated propagations do not re-read the file. The cache key is `(path, nmax, mmax)` plus the file's size, modification time, status-change time and inode (size and modification time only on Windows), so an edited, replaced or touched-back file is re-read. A file changed within the last 2 s is loaded but not cached, because a rewrite within the timestamp granularity could keep the same key. Entries are never evicted. Each distinct key stays in memory for the lifetime of the process: about 0.3 MB at nmax = 100, and about 70 MB at nmax = 1200.

`test-python/data/` has degree-10 heads of GRGM1200A, GL0660B, EGM2008 and EGM96, in their native formats. The `.tab` files keep their CRLF line endings (`.gitattributes`).

## Frames

The coefficients are given in a body-fixed frame. At each RHS evaluation the code does the following, with `T = pxform(naif_frame, sh_frame, et)`:

- r_bf = T r
- a = Tᵀ a_bf
- G = Tᵀ G_bf T

The rotation rate does not enter: the STM needs ∂a/∂r at fixed t.

| body | `sh_frame` | kernels |
|---|---|---|
| Moon (GRAIL fields) | `MOON_PA` | `pck/moon_pa_de440_200625.bpc`, `fk/moon_de440_250416.tf` |
| Earth (EGM2008, EGM96) | `ITRF93` | `pck/earth_latest_high_prec.bpc` (or a current `earth_*_predict.bpc` for future epochs) |

GRAIL coefficients are in the lunar **principal-axes** frame. `MOON_ME` would be off by about 0.03°. Do not use `IAU_EARTH` with EGM fields, because it has no precession-nutation or polar-motion detail. `earth_latest_high_prec.bpc` is replaced daily, and its EOP revisions change the ITRF93 rotation at about 1e-10 between releases.

## GM, radius and units

The SH sum uses the GM and the reference radius of the file. The point-mass term keeps using `GMs[0]`. The coefficients are defined relative to the file's GM and radius, and they are not rescaled.

`SpiceParameters` throws `ValueError` (C++: `std::invalid_argument`) if `GMs[0]`, in km³/s², differs from the GM of the file by more than 1e-5 (relative). This catches a field for the wrong body, or GMs given in canonical units (`GMs` are always in km³/s²). Typical differences are about 1e-7 (DE440 vs GRGM1200A lunar GM).

In canonical units the field uses `mu_sh = GM_file·TU²/DU³` and `R_sh = R_file/DU`. The acceleration is then in DU/TU² and the gradient in 1/TU², with no further conversion.

Both EGM fields are tide-free. campanella has no solid-tide model, so tide-free C̄20 is the consistent choice.

## Method

The code uses the fully normalized Cunningham / Montenbruck–Gill V, W recursion, with the acceleration and the second derivatives from Cunningham's derivative identities. It is purely Cartesian, so there is no singularity at the poles.

- The acceleration needs V, W to degree nmax+1, and the gradient to nmax+2. Both come from one table.
- The recursion and derivative factors are precomputed when the file is loaded.
- Each propagation allocates its V, W workspace once, so an RHS evaluation allocates nothing.
- The trace of G vanishes term by term.

Positions with |r|/R < `sh_rho_min` are rejected. The default 1 means the reference sphere, inside which the series is not guaranteed to converge. A position at exactly `sh_rho_min`·R within rounding (a few ulp) is accepted.

- In the solvers, a trial step below the limit is retried with a smaller step. If the trajectory itself goes below the limit, the solver raises `ValueError` (C++: `std::domain_error`) with the time and |r|/R. No solver returns a truncated trajectory: other failures raise too.
- `sh_accel_bf` and `sh_accel_inertial` raise `ValueError`.
- For low or landing arcs, a value slightly below 1 (e.g. `sh_rho_min = 0.99`, about 17 km below the GRGM1200A sphere) can be used. The lunar surface dips to about 1728 km against R = 1738 km. Below R the series may diverge, more so at high degree, so treat such results with care.

Each solver call allocates its own V, W workspace, so the solvers never share it. The `eom_*` functions use the workspace of the `SpiceParameters` they are given: a `SpiceParameters` must not be passed to them from two threads at once (a copy has its own workspace). CSPICE itself is not thread-safe.

Limits:

- `nmax ≤ 1200`. Higher degrees (EGM2008 to 2190) would need sectoral rescaling.
- The cost grows as nmax². With the default build flags (no `CMAKE_BUILD_TYPE`), degree 50 adds about 75 µs to each GVE RHS + Jacobian evaluation, against about 7 µs for degree 10.

Validation is in `test-python/test_spherical_harmonics.py`:

- body-fixed values against an mpmath direct Legendre sum (agreement ~1e-16), including points on the poles
- the J2 closed form
- the acceleration against finite differences of the potential, and the gradient against finite differences of the acceleration
- inertial values with SPICE
- Cartesian and GVE STMs against finite-differenced propagation
- GVE against Cartesian propagation
- canonical units against km/s
