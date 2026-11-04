# Tutorial: a geostationary satellite in the ephemeris model

This tutorial takes you from an empty script to a propagation with the Moon, the Sun, solar
radiation pressure and the Earth's gravity field, with a state-transition matrix (STM) checked
against finite differences. It assumes you know the two-body problem and the classical orbital
elements, and some C++ or Python. It does not assume you have used SPICE, an ephemeris model or an
STM before.

The main language is Python. The [same program in C++](#11-the-same-program-in-c) is at the end.
Both scripts are in the repository, and every number printed on this page is their real output:

- `docs/tutorial_code/tutorial.py`
- `docs/tutorial_code/tutorial.cpp`

The Python code blocks on this page are consecutive pieces of `tutorial.py`, shown in order. They
build on each other: each block uses variables defined in the blocks before it. To follow along,
paste them into one file in order, starting with the [setup block](#2-before-you-start), or run the
whole script.

The equations behind each step are on the [dynamical model](dynamics_model.md) page.


## 1. What problem does `campanella` solve?

In an introductory course, a spacecraft orbits a single point mass:

$$
\ddot{\boldsymbol{r}} = -\frac{\mu}{r^3}\boldsymbol{r} .
$$

The solution is a conic, and the orbital elements are constant. A real spacecraft also feels the
Moon, the Sun, the planets, the oblateness of the Earth and the pressure of sunlight. None of these
is large. For a geostationary satellite, the *perturbing* acceleration of the Moon and the Sun is
about $3\times10^{-5}$ of the Earth's pull. (The Sun alone pulls on the satellite about a thousand
times harder than that, about 1/37 of the Earth's pull, but it pulls on the Earth almost equally; only the difference, the *tidal* acceleration,
changes the orbit around the Earth. [Section 5](#5-declaring-the-dynamics-spiceparameters) and
[the dynamical model](dynamics_model.md#third-body-point-masses) explain this.) The effect of these
small forces accumulates, though. After a month the trajectory is
hundreds of kilometres away from the two-body prediction, as we will see below.

To model these perturbations you need to know where the Moon and the Sun are at every instant of
the integration. That is what an *ephemeris* gives: tabulated positions of solar-system bodies,
fitted to decades of observations. A propagation that reads body positions from an ephemeris is
said to use an *ephemeris model* (or "high-fidelity model"), as opposed to idealised models such as
the two-body problem or the circular restricted three-body problem.

`campanella` integrates

$$
\ddot{\boldsymbol{r}} = -\frac{\mu_0}{r^3}\boldsymbol{r}
  + \sum_{i\ge 1} \boldsymbol{a}_{\text{3b},i}(\boldsymbol{r}, t)
  + \boldsymbol{a}_{\text{SRP}}(\boldsymbol{r}, t)
  + \boldsymbol{a}_{\text{SH}}(\boldsymbol{r}, t)
$$

relative to a central body, with the third-body positions read from NASA/JPL ephemerides through
the SPICE toolkit {cite:p}`acton1996,acton2018`. It can also integrate the STM, which tells you how
a small change of the initial state changes the final state. You need the STM for differential
correction, orbit determination, and gradient-based trajectory optimisation
([section 9](#9-the-state-transition-matrix)).


## 2. Before you start

Build the Python module (and, for the last section, the C++ library) as described in
[Building campanella](build_library.md). The module is a single file in `build/lib`. The tutorial
script also uses `numpy` and `spiceypy`, which are dependencies of the project: `uv sync` installs
them into the project's virtual environment, `.venv`. Run the script with that environment, the
module directory on `PYTHONPATH`, and `SPICE` pointing at your kernels (section 3), from the
repository root:

```bash
SPICE=/path/to/spice PYTHONPATH=build/lib uv run python docs/tutorial_code/tutorial.py
```

A plain `python3` without `uv run` will not find `numpy` unless you installed it yourself.

The outputs on this page come from a build configured without `CMAKE_BUILD_TYPE`. With an optimized
(`Release`) build, the compiler may reorder floating-point operations, so numbers that measure
integration error can differ in their last digits; in a Release build of the same sources, the
only line that changed was "MEE vs Cartesian, final position", 5.664e-07 km instead of 5.748e-07 km.

The script starts with its imports and two settings. `REPO`, the repository root, is used in
section 10 to find a gravity-field file shipped with the repository:

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [setup]"
:end-before: "# [setup-end]"
```


## 3. SPICE kernels

### What a kernel is

SPICE is NASA's toolkit for space geometry, maintained by the Navigation and Ancillary Information
Facility (NAIF) at JPL {cite:p}`acton1996`. Its data come in files called *kernels*. Each kind of
kernel holds one kind of information:

| kernel | extension | content | used here |
|---|---|---|---|
| LSK (leapseconds) | `.tls` | leap seconds, to convert UTC to the ephemeris time scale | `naif0012.tls` |
| SPK (ephemeris) | `.bsp` | positions and velocities of bodies as functions of time | `de440.bsp` |
| PCK (text) | `.tpc` | body constants, such as GMs and radii | `gm_de440.tpc` |
| PCK (binary) | `.bpc` | high-accuracy orientation of a body (e.g. the Earth or the Moon) | `earth_latest_high_prec.bpc` |
| FK (frames) | `.tf` | definitions of reference frames | not needed in this tutorial |

`de440.bsp` is JPL's planetary and lunar ephemeris DE440 {cite:p}`park2021`. It covers 1550 to 2650
and is 120 MB. `gm_de440.tpc` holds the GMs that were fitted together with DE440, so the two are
consistent.

### Where to get them

All generic kernels are on NAIF's server, <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/>:

| file | URL (under `generic_kernels/`) |
|---|---|
| `naif0012.tls` | `lsk/naif0012.tls` |
| `de440.bsp` | `spk/planets/de440.bsp` |
| `gm_de440.tpc` | `pck/gm_de440.tpc` |
| `earth_latest_high_prec.bpc` | `pck/earth_latest_high_prec.bpc` |

Keep them in one directory, with one subdirectory per kernel type, and set the environment
variable `SPICE` to that directory:

```text
/path/to/spice/
├── lsk/
│   └── naif0012.tls
├── pck/
│   ├── earth_latest_high_prec.bpc
│   └── gm_de440.tpc
└── spk/
    └── de440.bsp
```

```bash
export SPICE=/path/to/spice
```

`campanella` itself does not read `SPICE`; you pass full paths to `furnsh`. The tutorial scripts,
the examples in `examples/` and the tests use the variable so that no path is hard-coded.

### Loading kernels

`furnsh` ("furnish") loads a kernel into SPICE's kernel pool:

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [kernels]"
:end-before: "# [kernels-end]"
```

Use `campanella.furnsh`, not `spiceypy.furnsh`. `campanella` contains its own copy of the CSPICE
library, with its own kernel pool. A kernel loaded through `spiceypy` is invisible to it, and vice
versa. A missing file raises `RuntimeError` with NAIF's message.

`campanella` works in ephemeris time throughout (section 4), so a propagation needs only the SPK
and, unless you pass `GMs`, the GM kernel. The leapseconds kernel is needed only to convert UTC to
ephemeris time; loading it into `campanella` is harmless.


## 4. Time: ET and TDB

SPICE, and therefore `campanella`, measures time as *ephemeris time* (ET): seconds of Barycentric
Dynamical Time (TDB) past the J2000 epoch, 2000 January 1, 12:00:00 TDB. The time scales involved
are {cite:p}`vallado2013,petit2010`:

TAI (International Atomic Time)
: a uniform time kept by atomic clocks. It has no leap seconds.

UTC (Coordinated Universal Time)
: civil time. It runs at the rate of TAI but is kept within 0.9 s of the Earth's rotation by
  inserting leap seconds. Since 2017, TAI − UTC = 37 s.

TT (Terrestrial Time)
: TT = TAI + 32.184 s, a uniform time for observers on the Earth's surface (the 32.184 s keeps it
  continuous with the older Ephemeris Time).

TDB (Barycentric Dynamical Time)
: a coordinate time for solar-system ephemerides: a linear rescaling of Barycentric Coordinate Time
  (TCB, the time coordinate of the relativistic barycentric reference system), defined so that on
  average it keeps pace with TT. Because the Earth moves through the Sun's varying gravitational
  potential, TDB and TT differ by periodic terms smaller than 2 ms.

The equations of motion need a uniform time variable: a leap second is not a physical event, and a
time scale that jumps would put the Moon in the wrong place. TDB is the uniform time in which the
JPL ephemerides are computed, so it is what SPICE uses. Together, ET is about
37 + 32.184 = 69.184 s ahead of UTC. `campanella` has no time conversion of its
own. In Python, use `spiceypy.str2et`, which needs the leapseconds kernel loaded into *spiceypy's*
pool:

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [epoch]"
:end-before: "# [epoch-end]"
```

```text
et0 = 788961669.1839275 s past J2000 (TDB)
ET - UTC seconds past J2000: 69.183928 s
```

Here 788961600 s is the calendar count from 2000-01-01T12:00:00 to 2025-01-01T00:00:00: 9131.5 days
times 86400 s, ignoring leap seconds. The 69.18 s difference is TAI − UTC (37 s, of which 5 were
added after 2000) plus 32.184 s plus the small TDB − TT term.

In C++, compute the number once and write it into the program, as `tutorial.cpp` does. To convert
dates, use NAIF's command-line utility `chronos`, which comes with the CSPICE toolkit (the build
puts it in `third_party/cspice/exe/`):

```bash
chronos -setup /path/to/spice/lsk/naif0012.tls -from utc -to et -totype seconds -time "2025-01-01T00:00:00"
```

```text
      788961669.184                                             (ET/SECONDS)
```

Or call `str2et_c` from a separate program that links NAIF's CSPICE itself, as in
[Converting a calendar date to et](basic_usage_cpp.md#converting-a-calendar-date-to-et), which gives
the program and its compile command. Do not mix it into the program that uses `campanella`:
the library has its own CSPICE with its own kernel pool (see
[One CSPICE per process](basic_usage_cpp.md#one-cspice-per-process)).

Every propagation is anchored at a reference epoch `et0`. Times you pass to the solvers (`tspan`,
`t_eval`) are measured *from* `et0`, and the ephemerides are queried at `et = et0 + t`. This keeps
the integration variable small, which matters for floating-point resolution: a double near 30 days
($2.6\times10^{6}$ s) resolves $5\times10^{-10}$ s, whereas one near $7.9\times10^{8}$ s resolves only
$1.2\times10^{-7}$ s.


## 5. Declaring the dynamics: `SpiceParameters`

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [params]"
:end-before: "# [params-end]"
```

```text
SpiceParameters(et0=788961669.1839275, naif_ids=['EARTH', 'MOON', 'SUN'], naif_frame='J2000', DU=1, TU=1, VU=1, include_srp=False, include_spherical_harmonics=False)
NAIF codes: [399, 301, 10]
GMs [km^3/s^2]: [398600.4355070226, 4902.800118457549, 132712440041.27939]
```

What this declares:

- **The central body is the first body in the list**, here the Earth. The state you integrate is
  the spacecraft's position and velocity *relative to the Earth's centre*, and the Earth's point-mass
  gravity is the main term of the equations of motion.
- **Every other body is a third body**: here the Moon and the Sun. Their positions relative to the
  Earth are read from the SPK at every evaluation of the equations of motion. The order of the third
  bodies does not matter. Bodies you do not list are ignored.
- **NAIF IDs.** Each body has an integer code; names such as `"EARTH"` are aliases. Pass either,
  but always **as strings**, in Python and C++ alike: `["EARTH", "MOON", "SUN"]` or
  `["399", "301", "10"]`. A list of integers raises `TypeError` in Python (and does not compile in
  C++). `params.naif_codes` returns the resolved integer codes. Common codes:

  | body | NAIF ID | | body | NAIF ID |
  |---|---|---|---|---|
  | Sun | 10 | | Mars barycenter | 4 |
  | Earth | 399 | | Jupiter barycenter | 5 |
  | Moon | 301 | | Earth–Moon barycenter | 3 |

  `de440.bsp` contains the Sun, the Moon, the Earth, Mercury and Venus, but Mars and the outer
  planets only as system barycenters (4 for Mars, 5 for Jupiter, not 499 or 599), so use those IDs
  for them. The GM of a barycenter is that of the whole system, planet plus moons.
- **GMs.** Without `GMs=[...]`, each body's GM is read from the kernel pool (`gm_de440.tpc`). They
  are always in km³/s², whatever units you integrate in.
- **Frame.** States are expressed in `naif_frame`, a frame whose axes do not rotate. The default,
  `"J2000"`, has its $z$ axis along the Earth's mean rotation axis at the J2000 epoch and its $x$
  axis towards the mean vernal equinox of that epoch (the direction of the Sun at the March
  equinox). "Mean" means averaged over the short nutation wobble of the axis. SPICE treats J2000 as
  the same as the ICRF (International Celestial Reference Frame), which is defined by distant
  quasars and is the frame of the DE ephemerides.

  The Earth's axis precesses, so today's equator is not the J2000 equator: at our epoch the two
  differ by 0.139°. A satellite that is equatorial in J2000 is therefore inclined by about 0.14° to
  the true equator of 2025. Operational station-keeping uses the true equator (an Earth-fixed frame
  such as `ITRF93`, section 10); this tutorial measures everything in J2000 for simplicity.

  The origin of the frame is the Earth's centre, which is accelerated by the Moon and the Sun. The
  frame is therefore non-rotating but not strictly inertial. The third-body acceleration includes
  the correction for this, the *indirect* term, which is why only the tidal difference of the
  third-body pull acts on the orbit (see [the dynamical model](dynamics_model.md#third-body-point-masses)).

The object is immutable. To change the model, build a new one; this is cheap.


## 6. Two-body versus ephemeris model

Take a circular equatorial orbit at the geostationary radius:

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [initial-state]"
:end-before: "# [initial-state-end]"
```

```text
v_geo = 3.074666260 km/s, period = 23.934325 h
```

The period is one sidereal day, 23.934 h, as it should be. Propagate it for 30 days twice: once with
the Earth alone, and once with the Earth, the Moon and the Sun. The two-body model is a
`SpiceParameters` with a single body and an explicit GM; it makes no ephemeris queries.

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [two-body-vs-nbody]"
:end-before: "# [two-body-vs-nbody-end]"
```

```text
day  1: |r_nbody - r_2body| =   12.827 km
day  5: |r_nbody - r_2body| =   69.590 km
day 10: |r_nbody - r_2body| =  139.277 km
day 20: |r_nbody - r_2body| =  278.340 km
day 30: |r_nbody - r_2body| =  410.794 km
day 30 difference: radial   -1.608 km, along-track -404.948 km, cross-track  -69.037 km
```

`solve_ivp_rv(params, tspan, x0)` returns an object with `t` (N,), `x` (N, 6) and `stm`. With
`t_eval`, the output is at those times only; without it, at every step the integrator accepted.

Compare the size of the accelerations with `eval_rv`, which evaluates the right-hand side once:

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [accelerations]"
:end-before: "# [accelerations-end]"
```

```text
|a_Earth| = 2.242096e-04 km/s^2, |a_Moon + a_Sun| = 6.312160e-09 km/s^2, ratio = 2.815e-05
```

A perturbation of $3\times10^{-5}$ moves the satellite by 410 km in a month because most of the
difference is *along the track*. The last line of the previous output splits the day-30 difference
into radial, along-track and cross-track components: the satellite is 405 km behind the two-body
satellite, 69 km out of its plane, and only 1.6 km lower.

- **Along-track.** Averaged over an orbit, the Moon's and Sun's differential pull has a small radial
  component, which acts like a slightly weaker central gravity. The satellite's mean angular rate
  differs from the two-body value by a tiny amount, and the resulting phase error grows linearly
  with time (left panel of the figure in [section 8](#8-modified-equinoctial-elements)). We will see
  the same mechanism in the STM in [section 9](#9-the-state-transition-matrix).
- **Cross-track.** The orbit plane tilts steadily (section 8), so the out-of-plane distance grows
  too: 0.097° of inclination at 42164 km is 71 km.
- **Radial.** The radius only oscillates, by a few kilometres (between −1.9 and +4.4 km when
  sampled every 30 minutes).


## 7. Units and canonical scaling

By default, positions are in km, velocities in km/s and times in s.

First, how the integrator controls its error. It adapts its step size so that the estimated error
$E_i$ of each component $y_i$ of the state, in every step, stays below

$$
D_i = \texttt{atol} + \texttt{rtol}\,|y_i| ,
$$

where `atol` is the *absolute* tolerance (in the units of $y_i$) and `rtol` the *relative* tolerance
(a pure number). The defaults are `atol=1e-12` and `rtol=1e-14`; you can pass others to every
solver. A component much larger than `atol/rtol` is controlled by `rtol`, a smaller one by `atol`.

With mixed units, a single `atol` means different things for different components: $10^{-12}$ km
for a position, but $10^{-12}$ km/s for a velocity, which is a much tighter demand relative to the
size of that component. Scaling the problem so that every component is of order one makes one
tolerance mean the same relative accuracy everywhere. It also keeps intermediate numbers away from
very large or small magnitudes. `campanella` uses *canonical units*:
give a distance unit `DU` (km), and the velocity and time units follow from the central body's GM,

$$
\mathrm{VU} = \sqrt{\mu_0/\mathrm{DU}}, \qquad \mathrm{TU} = \mathrm{DU}/\mathrm{VU},
$$

so that the central body's GM is 1 in the scaled units. For a circular orbit of radius DU, the speed
is 1 VU and the period is $2\pi$ TU.

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [canonical]"
:end-before: "# [canonical-end]"
```

```text
DU = 42164.0 km, TU = 13713.358276 s, VU = 3.074666260 km/s
mus (canonical): [1.0, 0.01230003703388119, 332946.04877304815]
canonical vs km/s, final position: 9.223e-05 km
```

Rules to remember:

- `x0`, `tspan`, `t_eval` and all outputs are in DU, DU/TU and TU. Convert them yourself.
- `et0` and `GMs` are always in seconds and km³/s². `params.mus` are the GMs in the scaled units:
  the Moon's GM is 0.0123 of the Earth's, and the Sun's 332946 times.
- The ephemerides are queried at `et = et0 + t * TU`.
- The tolerances `atol` and `rtol` apply to the scaled variables. `atol = 1e-12` means
  $10^{-12}$ km in km/s units but $4\times10^{-8}$ km at DU = 42164 km.

The two propagations agree to 0.1 m after 30 days. They are not identical, because the integrator
takes different steps with differently scaled tolerances; the difference is integration error,
not a different model. For this Cartesian GEO case both are accurate enough, so the benefit of
scaling is not visible here. It matters more when components differ by many orders of magnitude,
as the MEE in the next section do ($p \approx 4\times10^{4}$ km next to $f, g, h, k \lesssim 10^{-3}$).


## 8. Modified equinoctial elements

The Cartesian state changes quickly (a GEO satellite goes once around the Earth per day) even when
the orbit itself barely changes. Orbital elements separate the two: in the two-body problem, five of
them are constant and only the anomaly moves. Perturbations then appear as slow drifts of the
elements, which an integrator can follow with long steps.

The classical elements $(a, e, i, \Omega, \omega, \nu)$ are singular for circular orbits ($\omega$
undefined) and equatorial orbits ($\Omega$ undefined), and our geostationary orbit is both.
*Modified equinoctial elements* (MEE) {cite:p}`walker1985` avoid this by using only combinations
that stay well defined:

$$
\begin{aligned}
p &= a(1-e^2), & f &= e\cos(\omega+\Omega), & g &= e\sin(\omega+\Omega), \\
h &= \tan(i/2)\cos\Omega, & k &= \tan(i/2)\sin\Omega, & L &= \Omega+\omega+\nu .
\end{aligned}
$$

$f$ and $g$ are the components of the eccentricity vector, and $h$ and $k$ those of a vector of
length $\tan(i/2)$ along the line of nodes. When $e \to 0$ or $i \to 0$ these vectors simply shrink
to zero, with no undefined angle. The factor $\tan(i/2)$ (rather than, say, $\sin i$) makes the
mapping one-to-one for all inclinations from 0 to 180°; it diverges only at exactly 180°, the one
singularity of these elements. ("Prograde" MEE, with retrograde factor $I = +1$; the variant with
$I = -1$ moves the singularity to $i = 0$.)

The *Gauss variational equations* (GVE) give the rate of change of each element caused by a
perturbing acceleration, split into radial, along-track (transverse) and normal components. In the
two-body problem the right-hand sides are zero except for $\dot L$, so the integrator only has to
follow the slow drift caused by the perturbations. `solve_ivp_mee` integrates the GVE with exactly
the same forces as `solve_ivp_rv`; the formulas are in
[Gauss variational equations in MEE](gve_mee.md). `cart2mee` and `mee2cart` convert states; their
`mu` is the central body's GM in the units of the state (`params.mus[0]` in general, which equals
`params.GMs[0]` in km/s units). Both accept lists or numpy arrays.

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [mee]"
:end-before: "# [mee-end]"
```

```text
MEE at t0 [p, f, g, h, k, L]: [42164.     0.     0.    -0.     0.     0.]
MEE vs Cartesian, final position: 5.748e-07 km
day  0: p = 42164.000 km, e = 0.000e+00, i = 0.00000 deg, L = 0.0000 rad
day 10: p = 42164.027 km, e = 2.138e-05, i = 0.02900 deg, L = 63.0010 rad
day 20: p = 42165.387 km, e = 6.257e-05, i = 0.06161 deg, L = 126.0020 rad
day 30: p = 42164.098 km, e = 5.267e-05, i = 0.09694 deg, L = 189.0032 rad
```

Things to notice:

- The MEE and Cartesian propagations agree to better than a millimetre: the same dynamics, two sets
  of coordinates.
- The elements show what the Cartesian difference of section 6 hides. The inclination grows by about
  0.1° in this month. The lunisolar torques make the orbit normal precess about the pole of the
  *Laplace plane*, which for geostationary orbits is tilted about 7.5° from the equator towards the
  ecliptic; an orbit that starts in the equator therefore gains inclination. Averaged over a year
  the growth is 0.75–0.95° per year, depending on the 18.6-year cycle of the Moon's orbit
  {cite:p}`soop1994`, so about 0.07° per month; the rest of this month's 0.097° is the two-week
  periodic term described next. This is why geostationary satellites need north–south
  station-keeping.
- $p$ and $e$ do not drift but oscillate. Sampled once a day, $p$ rises to 42166.0 km on day 5, falls
  to 42163.7 km on day 12, and rises again to 42165.4 km on day 20, and the daily samples of $e$
  rise and fall between $6\times10^{-6}$ and $7.6\times10^{-5}$ on a similar period of about two weeks
  (sampled every 15 minutes, $e$ reaches $1.1\times10^{-4}$). These are the short-period effects of
  the Moon, whose tidal pull repeats twice per lunar month (about every 14 days) as it moves around
  its orbit. Within each day $p$ also oscillates by about a kilometre, twice per revolution. So the
  jump of $p$ between days 10 and 20 and the drop of $e$ between days 20 and 30 are physical.
- $L$ is not wrapped to $(-\pi, \pi]$: it grows by $2\pi$ per revolution (189 rad ≈ 30 revolutions).
  Compare $L$ values through $\operatorname{atan2}(\sin\Delta L, \cos\Delta L)$.
- With $h = k = 0$, `cart2mee` returns $h = -0$; the sign of a zero has no meaning.

```{figure} _static/tutorial_geo.png
:alt: Left, position differences growing roughly linearly to 410 km (Moon and Sun) and 514 km (Earth gravity field) after 30 days; right, inclination rising from 0 to 0.097 degrees.

Left: distance between the Earth–Moon–Sun trajectory and the two-body trajectory (blue), and
between the trajectory with the Earth's gravity field to degree 4 and the point-mass Earth–Moon–Sun
trajectory (orange, [section 10](#10-solar-radiation-pressure-and-the-earths-gravity-field)).
Right: inclination to the J2000 equator, from the MEE propagation. Written by
`tutorial.py --plot`.
```

The tolerances of section 7 act on each component: $p$ is about $4\times10^4$ km, so
$\texttt{rtol}\,|p| = 4\times10^{-10}$ km dominates $\texttt{atol} = 10^{-12}$ and $p$ is controlled by
`rtol`; $f, g, h, k$ are below $10^{-3}$ here and are controlled by `atol`. For long arcs, use canonical
units with DU close to $p$ so that all six elements are of order one.


## 9. The state-transition matrix

### What it is

Let $\boldsymbol{x}(t) = \boldsymbol{\varphi}(t; \boldsymbol{x}_0)$ be the state at time $t$ of the
trajectory that starts at $\boldsymbol{x}_0$ at $t_0$. If you change the initial state by a small
$\delta\boldsymbol{x}_0$, the final state changes by

$$
\delta\boldsymbol{x}(t) \approx \Phi(t, t_0)\, \delta\boldsymbol{x}_0,
\qquad
\Phi(t, t_0) = \frac{\partial \boldsymbol{x}(t)}{\partial \boldsymbol{x}_0} .
$$

$\Phi$ is the 6×6 *state-transition matrix*. Column $j$ is the sensitivity of the whole final state
to the $j$-th initial component. It is the derivative that a differential corrector, an orbit
determination filter or an optimiser needs. For example, to find the initial velocity that makes
the spacecraft reach a target position $\boldsymbol{r}^*$ at time $t$, Newton's method iterates
$\delta\boldsymbol{v}_0 = \Phi_{rv}^{-1}\,\big(\boldsymbol{r}^* - \boldsymbol{r}(t)\big)$, where
$\Phi_{rv} = \partial\boldsymbol{r}(t)/\partial\boldsymbol{v}_0$ is the top-right 3×3 block of $\Phi$.

You could compute $\Phi$ by finite differences: twelve extra propagations, with a step size that is
a compromise between truncation and round-off error. Instead, differentiate the equations of motion
$\dot{\boldsymbol{x}} = \boldsymbol{F}(\boldsymbol{x}, t)$ with respect to $\boldsymbol{x}_0$:

$$
\dot\Phi(t, t_0) = A(t)\, \Phi(t, t_0), \qquad \Phi(t_0, t_0) = I_6,
\qquad A(t) = \left.\frac{\partial \boldsymbol{F}}{\partial \boldsymbol{x}}\right|_{\boldsymbol{x}(t)} .
$$

These are the *variational equations*. The `solve_ivpstm_*` solvers integrate them together with
the state (6 + 36 = 42 equations), with an analytic Jacobian $A$. For Cartesian states,

$$
A = \begin{pmatrix} 0_3 & I_3 \\ G & 0_3 \end{pmatrix}, \qquad
G = \frac{\partial \ddot{\boldsymbol{r}}}{\partial \boldsymbol{r}},
$$

where the gravity gradient $G$ sums the contributions of every force; see
[the dynamical model](dynamics_model.md#variational-equations-and-the-stm).

### Computing and checking it

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [stm]"
:end-before: "# [stm-end]"
```

```text
Phi(1 day, 0):
[[ 1.319094e+00 -3.338770e-04  5.694908e-05  2.275631e+02  4.366340e+03  4.043225e-01]
 [-1.884884e+01  9.986694e-01  4.846420e-05 -1.136391e+01 -2.582367e+05 -5.955217e+00]
 [ 1.936578e-04 -1.908072e-05  9.997764e-01  2.676107e-01  4.009404e+00  2.339587e+02]
 [ 1.376991e-03  8.956617e-08 -1.445491e-09  1.000715e+00  1.884862e+01  5.344868e-04]
 [ 2.339659e-05 -1.208964e-06  1.157094e-09  3.687261e-04  1.320640e+00  2.681802e-05]
 [-1.035313e-07  1.233376e-09 -1.235420e-06 -5.769326e-05 -1.447808e-03  9.999345e-01]]
max |Phi - Phi_fd| (scaled) = 3.496e-07
dx(1 day), nonlinear: [ 1.314876e+00 -1.884958e+01  1.940166e-04  1.376980e-03  2.308851e-05 -1.035243e-07]
dx(1 day), Phi @ dx0: [ 1.319094e+00 -1.884884e+01  1.936578e-04  1.376991e-03  2.339659e-05 -1.035313e-07]
```

How to read this:

- **The finite-difference check.** The entries of $\Phi$ have units (km/km, km/(km/s), ...) and span
  14 orders of magnitude, so the comparison is made on $D^{-1}\Phi D$ with
  $D = \operatorname{diag}(a, a, a, v, v, v)$, i.e. in canonical units. The agreement, $3.5\times10^{-7}$
  on entries up to about 19, is the accuracy of the central differences themselves. Do this check
  whenever you change a model or write your own Jacobian.
- **The linear prediction.** After one day, a 1 km radial offset becomes a displacement of 18.85 km
  *along track* ($-y$), and $\Phi\,\delta\boldsymbol{x}_0$ predicts it to within 5 m. The
  remaining difference ($4\times10^{-3}$ km in $x$) is the second-order term that the linearisation
  neglects; it is proportional to $\|\delta\boldsymbol{x}_0\|^2$, so it becomes negligible for
  smaller offsets.
- **The two-body physics is visible.** Raising the radius by 1 km at the same speed raises the
  semi-major axis by 2 km: from vis-viva, $v^2 = \mu(2/r - 1/a)$ at fixed $v$ gives
  $\delta a/a^2 = 2\,\delta r/r^2$, so $\delta a = 2\,\delta r$ for $r = a$. That lengthens the period
  by $\delta T/T = \tfrac{3}{2}\,\delta a/a$ (Kepler's third law), so after one revolution the
  satellite is behind by $2\pi a\cdot\tfrac{3}{2}\,\delta a/a = 3\pi\,\delta a \approx 18.85$ km:
  $\Phi_{yx} = -18.85$. Likewise
  $\Phi_{y v_y} = -2.58\times10^{5}$ s: an along-track velocity error of 1 mm/s turns into 258 m of
  along-track position error per day. This is the secular along-track growth of section 6.

The MEE solvers have their own STM, $\partial\,\mathrm{MEE}(t)/\partial\,\mathrm{MEE}(t_0)$, computed
with an analytic Jacobian of the Gauss variational equations. `campanella.mee2cart_jacobian` maps it
to Cartesian coordinates: $\Phi_\text{cart} = J(\boldsymbol{x}_f)\,\Phi_\text{MEE}\,J(\boldsymbol{x}_0)^{-1}$.
In canonical units, $\Phi_\text{km} = D\,\Phi_\text{canonical}\,D^{-1}$ with
$D = \operatorname{diag}(\mathrm{DU}, \mathrm{DU}, \mathrm{DU}, \mathrm{VU}, \mathrm{VU}, \mathrm{VU})$.


## 10. Solar radiation pressure and the Earth's gravity field

### Solar radiation pressure

Sunlight carries momentum. A spacecraft that absorbs or reflects it is pushed away from the Sun.
`campanella` uses a *cannonball* model: the force is along the Sun–spacecraft line, with a magnitude
set by the area-to-mass ratio and a reflectivity coefficient,

$$
\boldsymbol{a}_\text{SRP} = \frac{P_0}{c}\, C_r\, \frac{A}{m}
\left(\frac{1\,\mathrm{AU}}{\|\boldsymbol{d}\|}\right)^2 \frac{\boldsymbol{d}}{\|\boldsymbol{d}\|},
\qquad C_r = 1 + \rho_S + \tfrac{5}{3}\rho_D,
$$

with $\boldsymbol{d}$ the position of the spacecraft relative to the Sun and $P_0 = 1367$ W/m² the
solar flux at 1 AU ([details](dynamics_model.md#solar-radiation-pressure)). $\rho_S$ (`srp_rhoS`)
and $\rho_D$ (`srp_rhoD`) are the fractions of the incident light reflected specularly (mirror-like)
and diffusely; the rest is absorbed. The Sun must be one of the bodies, because its position is
needed.

The coefficient $\tfrac{5}{3}$ on $\rho_D$ is what the library implements. It is **not** the
textbook flat-plate value: for a flat plate facing the Sun, with absorbed, specular and diffuse
fractions summing to one, $C_r = 1 + \rho_S + \tfrac{2}{3}\rho_D$ {cite:p}`milani1987`. The two
agree when $\rho_D = 0$. To reproduce a textbook or a given $C_r$, set `srp_rhoD=0` and
`srp_rhoS` $= C_r - 1$, as below.

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [srp]"
:end-before: "# [srp-end]"
```

```text
day 10: e without SRP = 2.138e-05, with SRP = 4.508e-05
day 20: e without SRP = 6.257e-05, with SRP = 1.565e-04
day 30: e without SRP = 5.267e-05, with SRP = 1.471e-04
SRP effect on the final position: 6.380 km
```

Here $\rho_S = 0.3$ and $\rho_D = 0$ simply give $C_r = 1.3$, an illustrative value within the range
of about 1.2 to 1.5 used for real satellites. With $A/m = 0.02$ m²/kg, typical of a communications
satellite with large solar arrays, the acceleration is about $1.2\times10^{-10}$ km/s², fifty times smaller than the lunisolar
perturbation. It still nearly triples the eccentricity in a month: a constant push from one direction
(the Sun moves only 30° in a month) pumps the eccentricity vector $(f, g)$ steadily.
`srp_A` is in m² and `srp_m` in kg, also in canonical units.

The model has no shadow: the spacecraft is lit even inside the Earth's shadow. For a geostationary
satellite this matters only in the eclipse seasons around the equinoxes, when it spends up to about
70 minutes per day in shadow.

### Central-body gravity field

The Earth is not a point mass. Its equatorial bulge (the $J_2$ term) is by far the largest deviation.
`campanella` can add the central body's gravity field as a spherical-harmonic series of any degree,
read from a coefficient file ([details](spherical_harmonics.md)). The coefficients are given in a
frame fixed to the rotating Earth. SPICE calls the Earth's precise body-fixed frame `ITRF93` (the
International Terrestrial Reference Frame, 1993 realisation); its orientation, including
precession, nutation and polar motion, comes from the binary PCK `earth_latest_high_prec.bpc`.

The files store *fully normalized* coefficients $\bar C_{nm}$, $\bar S_{nm}$. The zonal coefficient
you know from a course is $J_n = -\sqrt{2n+1}\,\bar C_{n0}$, so $J_2 = -\sqrt{5}\,\bar C_{20}$.
`sh_nmax` is the highest degree used: `sh_nmax=2` gives $J_2$ together with the small $C_{21}$,
$S_{21}$, $C_{22}$, $S_{22}$ terms, and each higher degree adds finer, faster-decaying detail (the
degree-$n$ terms fall off as $(R/r)^{n+1}$).

The repository contains the first ten degrees of EGM2008 {cite:p}`pavlis2012` in
`test-python/data/EGM2008_n10.gfc`; the full model is available from ICGEM (link in
[spherical_harmonics.md](spherical_harmonics.md#coefficient-files)).

```{literalinclude} tutorial_code/tutorial.py
:language: python
:start-after: "# [sh]"
:end-before: "# [sh-end]"
```

```text
EGM2008: GM = 398600.4415 km^3/s^2, R = 6378.1363 km, J2 = 1.082626174e-03
sh_nmax = 2: |r_sh - r_pointmass| =  19.398 km (day 1), 185.410 km (day 10), 498.119 km (day 30)
sh_nmax = 4: |r_sh - r_pointmass| =  19.446 km (day 1), 187.326 km (day 10), 513.619 km (day 30)
circular speed with J2 - without: 57.13 mm/s
```

The degree-2 field accounts for 498 of the 514 km; degrees 3 and 4 add 16 km. Together it is more
than the Moon and the Sun (left panel of the figure in section 8). Again the difference is mostly
along track. In the equatorial plane $J_2$ adds to the inward pull, so a circular orbit at 42164 km
needs 57 mm/s more speed than in the point-mass field. Our initial state has the point-mass speed,
which is too slow for the $J_2$ field: the satellite settles into a slightly smaller mean orbit,
with a shorter period, and drifts. With $\Phi_{yv_y} = -2.58\times10^5$ s from section 9, a 57 mm/s
speed difference predicts about 15 km of along-track drift per day, most of the observed 17 km per
day. This is not an artefact: it is how the same initial state evolves under the two force models.
If instead you want a circular orbit in each model, start each from its own circular speed. Even
then the two drift apart, because the $J_2$ orbit at the same radius has a higher angular rate; the
radius at which a $J_2$ orbit is geostationary is 0.52 km larger than the point-mass value,
$\delta r/r = \tfrac{1}{2}J_2 (R/r)^2$.

**Which Earth GM?** Several values appear in these docs. Three are the same physical quantity
expressed in different time scales. GM has units of km³/s², so its value depends on which
"second" is used. TT and TDB are defined as rescalings of the relativistic coordinate times TCG
(geocentric) and TCB (barycentric) by the factors $1 - L_G$ and $1 - L_B$, with
$L_G = 6.97\times10^{-10}$ and $L_B = 1.55\times10^{-8}$, and a GM expressed in TT- or TDB-compatible
units is scaled by the same factor. The IERS conventions {cite:p}`petit2010` give $3.986004418\times10^{14}$
m³/s² in TCG, $3.986004415\times10^{14}$ in TT, and $3.986004356\times10^{14}$ in TDB.

- 398600.4418 km³/s² (the TCG value) is used in the kernel-free two-body examples.
- 398600.4415 km³/s² (the TT value) is the GM of the EGM2008 file.
- 398600.4355 km³/s² in `gm_de440.tpc` is the GM fitted with DE440, which works in TDB; it is close
  to, but not exactly, the IERS TDB value.

A fourth set, `GMs=[4.902800582147760e+03, 398600.432896939, 132712197035.766]` (Moon, Earth, Sun),
appears in the Moon-centred examples of the other pages. It comes from the project's original test
suite, where it was used to compute reference trajectories, and is kept so that the examples
reproduce those references. It is not a DE440 set and the differences are not time-scale effects:
its Sun GM differs from DE440's by $1.8\times10^{-6}$. For new work, read the GMs from
`gm_de440.tpc` (omit `GMs`), as this tutorial does. None of these differences matters at the
level this tutorial resolves. `SpiceParameters` requires the
central body's GM and the GM of the field file to agree within $10^{-5}$, which catches a field for
the wrong body or GMs given in the wrong units.

All of these combine: `include_srp=True` and `sh_file=...` can be given together, and the STM
solvers include the gradients of every term.


## 11. The same program in C++

The C++ interface mirrors the Python one: the keyword arguments of `SpiceParameters` are the fields
of `campanella::SpiceOptions`, the solver keywords `atol`, `rtol` and `t_eval` are the fields of
`campanella::IvpOptions`, and the solvers return a `campanella::IvpResult` whose `x` and `stm` are
flat row-major vectors (6 and 36 values per output time). Errors are exceptions
(`std::invalid_argument`, `std::domain_error`, `std::runtime_error`). All options and functions are
listed in the [C++ API](api_cpp.md); see [Basic usage through C++](basic_usage_cpp.md) for linking
with CMake and for the memory layout of the results.

```{literalinclude} tutorial_code/tutorial.cpp
:language: cpp
```

Build the C++ library first with the C++-only configure into `build-cpp`, as in
[Building campanella](build_library.md):

```bash
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON
cmake --build build-cpp
```

This writes `build-cpp/lib/libcampanella.so` (`.dylib` on macOS). (Adding `-DCAMPANELLA_BUILD_CPP=ON`
to the Python configure instead puts the library in `build/lib`; then replace `build-cpp/lib` by
`build/lib` below.) Compile and run from the repository root:

```bash
c++ -std=c++17 -O2 docs/tutorial_code/tutorial.cpp -I include \
    -L build-cpp/lib -lcampanella -Wl,-rpath,$PWD/build-cpp/lib -o tutorial
SPICE=/path/to/spice ./tutorial
```

`-L` tells the linker where the library is at build time; `-Wl,-rpath,...` records where the
program looks for it at run time. `$PWD` makes that path absolute, so `./tutorial` runs from any
directory; a relative rpath would be resolved against the directory you run from. Run it from the
repository root anyway, or pass the path of the gravity-field file as its argument: by default it
opens `test-python/data/EGM2008_n10.gfc` relative to the current directory. Output:

```text
GMs [km^3/s^2]: 398600.435507023 4902.80011845755 132712440041.279
day 30: |r_nbody - r_2body| = 410.794 km (31 outputs)
DU = 42164.0 km, TU = 13713.358276 s, VU = 3.074666260 km/s
canonical vs km/s, final position: 9.223e-05 km
MEE vs Cartesian, final position: 5.748e-07 km
day 30: e = 5.267e-05, i = 0.09694 deg
Phi[1][0] = d y(1 day) / d x0 = -18.848841
max |Phi - Phi_fd| (scaled) = 3.496e-07
day 30: e with SRP = 1.471e-04
day 30: |r_sh - r_pointmass| = 513.619 km
```

The C++ program requests the same daily outputs as the Python one (`IvpOptions::t_eval`), so the
integrator takes the same steps and every number matches the Python run.


## Where to go next

- [Basic usage (Python)](basic_usage_python.md) and [Basic usage (C++)](basic_usage_cpp.md): every
  option of `SpiceParameters` and the solvers.
- [Dynamical model](dynamics_model.md): the exact equations, frames and units.
- [Gauss variational equations in MEE](gve_mee.md) and [central-body spherical harmonics](spherical_harmonics.md).
- `examples/` in the repository: a Moon-centred propagation with an STM check (`solve_rv.cpp`), MEE
  with SRP (`solve_mee.cpp`), canonical units (`solve_moon_canonical.cpp`) and a low lunar orbit with
  a GRAIL gravity field (`solve_sh.cpp`).
- For the background, {cite:t}`vallado2013` and {cite:t}`montenbruck2000` cover perturbations,
  time systems and reference frames in depth.
