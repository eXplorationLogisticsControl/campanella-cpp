# Central-body spherical harmonics

All solvers (`solve_ivp_rv`, `solve_ivp_mee`, `solve_ivpstm_rv`, `solve_ivpstm_mee`) and `eval_rv` / `eval_mee` can add the gravity field of the central body, the first of the `naif_ids` of `SpiceParameters`, up to an arbitrary degree and order:

$$
U = \frac{GM}{R}\sum_{n=2}^{n_\text{max}}\ \sum_{m=0}^{\min(n,\,m_\text{max})}
\left(\frac{R}{r}\right)^{n+1} \bar P_{nm}(\sin\phi)\left(\bar C_{nm}\cos m\lambda + \bar S_{nm}\sin m\lambda\right),
$$

with $r$, $\phi$ and $\lambda$ the radius, latitude and longitude in the body-fixed frame, $\bar P_{nm}$ the fully normalized associated Legendre functions, and $\bar C_{nm}$, $\bar S_{nm}$ the fully normalized coefficients. The acceleration is $\nabla U$. The degree $n$ sets the spatial scale of a term (about half a circumference divided by $n$); $m = 0$ terms are *zonal* (they depend on latitude only), $m = n$ terms are *sectoral*.

The zonal coefficients $J_n$ of an introductory course are unnormalized: $J_n = -\sqrt{2n+1}\,\bar C_{n0}$. For example $J_2 = -\sqrt{5}\,\bar C_{20}$; EGM2008's $\bar C_{20} = -4.8417\times10^{-4}$ gives the Earth's $J_2 = 1.0826\times10^{-3}$.

Without `sh_file`, the central body is a point mass. Its place among the other forces is described in [Equations of motion, frames and units](dynamics_model.md#central-body-spherical-harmonics), and the [tutorial](tutorial.md#central-body-gravity-field) adds an Earth field to a geostationary orbit.

## Usage

A low lunar orbit with the degree-10 head of the GRAIL field GRGM1200A that ships with the
repository. Run it from the repository root, with the module on `PYTHONPATH` and `SPICE` set as in
the [tutorial](tutorial.md#3-spice-kernels):

```python
import os
import campanella

spice = os.environ["SPICE"]
for k in ["spk/de440.bsp",
          "pck/moon_pa_de440_200625.bpc", "fk/moon_de440_250416.tf"]:   # MOON_PA orientation
    campanella.furnsh(os.path.join(spice, k))

et0 = 740463475.5117936                       # 2023-06-19T16:16:46 UTC, in TDB seconds past J2000
state0 = [1838.0, 0.0, 0.0, 0.0, 0.8166, 1.4144]   # km, km/s: about 100 km above the Moon (R = 1738 km)

params = campanella.SpiceParameters(
    et0, ["MOON", "EARTH", "SUN"],
    GMs = [4.902800582147760e+03, 398600.432896939, 132712197035.766],   # legacy test GMs, not DE440
    sh_file = "test-python/data/gggrx_1200a_sha_n10.tab",   # degree-10 head of GRGM1200A, in the repository
    sh_nmax = 10,                               # truncation degree (at most 10 for this file)
    sh_mmax = -1,                               # truncation order, -1 = min(sh_nmax, order of the file)
    sh_frame = "MOON_PA",                       # body-fixed frame of the coefficients
    sh_rho_min = 1.0,                           # smallest |r|/R evaluated (see below)
)
res = campanella.solve_ivp_rv(params, [0.0, 86400.0], state0)
print(len(res), res.x[-1])
```

```text
1273 [ 3.35735317e+02  8.96897578e+02  1.57235710e+03 -1.60300258e+00
  1.65772700e-01  2.47669669e-01]
```

For degrees above 10, download the full field (`gggrx_1200a_sha.tab`, 88 MB, link under
[Coefficient files](#coefficient-files)) and pass its path with a larger `sh_nmax`, e.g. 50. The
repository's `*_n10` files keep the header of the full model, which declares degree 1200, so a
`sh_nmax` above 10 with them does not raise "nmax above the degree of the file" but
`coefficient (11, 0) missing (truncated file, or an order-limited field: pass a smaller mmax)`. For
these files, the fix is a smaller `sh_nmax`.

In C++ they are the `sh_*` fields of `campanella::SpiceOptions` in `include/campanella.hpp`. `examples/solve_sh.cpp` propagates a 100 km lunar orbit and checks the STM against finite differences.

The STM solvers add the gradient of the SH acceleration. In `solve_ivpstm_rv` it is added to the gravity gradient. In the GVE it is added to the perturbation gradient G of [gve_mee.md](gve_mee.md) §5.5.

Test and debugging helpers:

- `campanella.load_spherical_harmonics(path, nmax, mmax=-1)` returns a dict: `source` (the file path), `GM` (km³/s²), `R` (km), `file_degree`, `file_order`, `nmax`, `mmax`, `tide_system`, and `C[n][m]`, `S[n][m]`.
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
  - EGM2008 {cite:p}`pavlis2012`: https://icgem.gfz.de/getmodel/gfc/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc
  - EGM96: https://icgem.gfz.de/getmodel/gfc/971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc

Only fully normalized coefficients are accepted (SHADR normalization state 1, ICGEM `fully_normalized`; ICGEM uses that as the default when the `norm` line is missing).

The loader throws in these cases:

- `nmax < 2`, `nmax > 1200`, or `nmax` above the degree of the file
- `mmax > nmax`, or `mmax` above the order of the file
- a non-finite coefficient (NaN, inf)
- any coefficient with 2 ≤ n ≤ nmax and m ≤ min(n, mmax) is missing (for example, a truncated file)

`mmax = -1` means min(nmax, order of the file). The order is read from the SHADR header. If a `.tab` file has rows above that order, a warning is printed and the rows are not used. ICGEM has no order keyword, so for an order-limited `.gfc` file pass `mmax` explicitly; otherwise the missing rows are reported.

Degrees 0 and 1 are not evaluated. The point mass `GM[0]` is already in the EOMs, and the degree-1 terms measure the offset of the centre of mass from the origin, so they vanish when the field is expressed about the centre of mass, as these fields are. The loader warns if degree-1 coefficients are non-zero.

:::{note}
**File cache.** Each file is read once per process and reused. In detail: the cache key is `(path, nmax, mmax)` plus the file's size, modification time, status-change time and inode (size and modification time only on Windows), so an edited, replaced or touched-back file is re-read. A file changed within the last 2 s is loaded but not cached, because a rewrite within the timestamp granularity could keep the same key. Entries are never evicted. Each distinct key stays in memory for the lifetime of the process: about 0.5 MB at nmax = 100, and about 70 MB at nmax = 1200.
:::

`test-python/data/` has degree-10 heads of GRGM1200A, GL0660B, EGM2008 and EGM96, in their native formats. The `.tab` files keep their CRLF line endings (`.gitattributes`).

## Frames

The coefficients are given in a body-fixed frame. At each RHS evaluation the code does the following, with `T = pxform(naif_frame, sh_frame, et)`:

$$
\boldsymbol r_\text{bf} = T\,\boldsymbol r, \qquad
\boldsymbol a = T^\mathsf{T}\boldsymbol a_\text{bf}, \qquad
G = T^\mathsf{T} G_\text{bf}\, T
$$

The rotation rate does not enter: the STM needs ∂a/∂r at fixed t.

| body | `sh_frame` | kernels |
|---|---|---|
| Moon (GRAIL fields) | `MOON_PA` | `pck/moon_pa_de440_200625.bpc`, `fk/moon_de440_250416.tf` |
| Earth (EGM2008, EGM96) | `ITRF93` | `pck/earth_latest_high_prec.bpc` (or a current `earth_*_predict.bpc` for future epochs) |

GRAIL coefficients are in the lunar **principal-axes** frame (`MOON_PA`, axes along the Moon's principal axes of inertia). The other lunar frame, `MOON_ME` ("mean Earth/polar axis", with $x$ towards the mean direction of the Earth), differs from it by about 0.03°, roughly 1 km at the lunar surface, so it would misplace the field. Do not use `IAU_EARTH` with EGM fields: it models the pole with a low-accuracy linear precession only, without nutation or polar motion. `earth_latest_high_prec.bpc` is replaced daily with updated Earth orientation parameters (EOP: polar motion and UT1, the measured irregularities of the Earth's rotation), so results near the present epoch can change slightly between releases of the file.

## GM, radius and units

The SH sum uses the GM and the reference radius of the file. The point-mass term keeps using `GMs[0]`. The coefficients are defined relative to the file's GM and radius, and they are not rescaled.

`SpiceParameters` throws `ValueError` (C++: `std::invalid_argument`) if `GMs[0]`, in km³/s², differs from the GM of the file by more than 1e-5 (relative). This catches a field for the wrong body, or GMs given in canonical units (`GMs` are always in km³/s²). For example, the lunar GM of DE440 (`gm_de440.tpc`, 4902.80011845755 km³/s²) and of GRGM1200A (4902.8001224453 km³/s²) differ by 8e-10; the lunar GM 4902.800582147760 used in the examples of these docs (a legacy test value, see the [tutorial](tutorial.md#central-body-gravity-field)) differs from GRGM1200A by 9e-8.

In canonical units the field uses `mu_sh = GM_file·TU²/DU³` and `R_sh = R_file/DU`. The acceleration is then in DU/TU² and the gradient in 1/TU², with no further conversion.

Both EGM fields are *tide-free*: their C̄20 excludes the permanent deformation of the Earth by the Sun's and Moon's tides. A tide-free field is meant to be used together with a solid-tide model, which restores that deformation. campanella has no solid-tide model, so the consistent static field would be the *zero-tide* one, $\bar C_{20}^\text{zero-tide} = \bar C_{20}^\text{tide-free} - 4.2\times10^{-9}$ {cite:p}`petit2010` (§6.2.1). campanella uses the file as given; the difference is about $10^{-5}$ of $\bar C_{20}$. To use zero-tide values, edit C̄20 in a copy of the file.

## Method

The code uses the fully normalized Cunningham / Montenbruck–Gill V, W recursion {cite:p}`cunningham1970,montenbruck2000`, with the acceleration and the second derivatives from Cunningham's derivative identities. It is purely Cartesian, so there is no singularity at the poles.

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

- `nmax ≤ 1200`. Higher degrees (EGM2008 goes to 2190) would need a rescaling of the recursion to avoid floating-point underflow in the high-order (sectoral) terms.
- The cost of each evaluation grows as nmax². (Absolute timings depend strongly on the build type; see [Building campanella](build_library.md).)

Validation is in `test-python/test_spherical_harmonics.py`:

- body-fixed values against an mpmath direct Legendre sum (agreement ~1e-16), including points on the poles
- the J2 closed form
- the acceleration against finite differences of the potential, and the gradient against finite differences of the acceleration
- inertial values with SPICE
- Cartesian and GVE STMs against finite-differenced propagation
- GVE against Cartesian propagation
- canonical units against km/s
