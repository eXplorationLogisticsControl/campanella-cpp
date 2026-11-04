# Gauss variational equations in modified equinoctial elements (MEE), with analytic STM

The *Gauss variational equations* (GVE) describe how the orbital elements of a spacecraft change
when, besides the central body's point-mass gravity, it feels a perturbing acceleration
$\boldsymbol a_I$. Split $\boldsymbol a_I$ into a radial component $a_R$ (along the position), a
transverse component $a_T$ (in the orbit plane, perpendicular to the position, roughly along the
velocity) and a normal component $a_N$ (along the angular momentum). Each element then changes at a
rate that is linear in $(a_R, a_T, a_N)$, with coefficients that depend only on the current
elements. In the two-body problem ($\boldsymbol a_I = \boldsymbol 0$) all elements are constant
except the angle that measures the position along the orbit.

`solve_ivp_mee` and `solve_ivpstm_mee` integrate the GVE written in the modified equinoctial
elements of {cite:t}`walker1985`, which, unlike the classical elements, are not singular for
circular or equatorial orbits. The perturbing acceleration is exactly the one of the Cartesian
solvers (`add_perturbations` in `src/eoms/forces.cpp`): third bodies, solar radiation pressure and
central-body spherical harmonics, described in
[Equations of motion, frames and units](dynamics_model.md). The two coordinate sets therefore
integrate the same physics. The [tutorial](tutorial.md#8-modified-equinoctial-elements) shows the MEE
solvers in use.

This page gives:

- the elements and the conversions `mee2cart` and `cart2mee` (sections 1–2);
- the RTN frame and the GVE (sections 3–4);
- the analytic Jacobian used to propagate the STM (section 5), with thrust and mass in section 5.8;
- a summary of how the formulas were verified (section 6) and the reference data of the test suite
  (section 7).

The first NAIF id of `SpiceParameters` is the central body and the others are third bodies. In the
text below, `GM[i]` is `SpiceParameters.mus[i]`, the gravitational parameter of body $i$ in the units
of the integration.

**Indexing.** All vectors and matrices are 0-based, as in C++.
- State: `x[0..5] = [p, f, g, h, k, L]`.
- RTN acceleration: `a_rtn[0..2] = [a_R, a_T, a_N]`.
- `B[i][j]`: row `i` is the element, column `j` is the R/T/N component.

**Name clash.** The Jacobian is called **A**, as in `Phi_dot = A Phi`. The spacecraft area (`srp_A` in
`SpiceParameters`) is written **S** in this document.

---

## 1. Definitions

### 1.1 Elements

This document uses the prograde MEE of {cite:t}`walker1985`. Walker et al. define a *retrograde
factor* $I = \pm 1$: with $I = +1$ (prograde, used here and in the library) the elements are singular
only at $i = 180°$; with $I = -1$ the singularity moves to $i = 0$. In terms of the classical elements
(a, e, i, Ω, ω, ν), with ϖ = ω + Ω the *longitude of periapsis*:

| MEE | definition | range |
|---|---|---|
| p | a(1 − e²), the semi-latus rectum, in DU | p > 0 |
| f | e cos(ω + Ω) | |
| g | e sin(ω + Ω) | |
| h | tan(i/2) cos Ω | |
| k | tan(i/2) sin Ω | |
| L | Ω + ω + ν, the true longitude, in rad | any real number, integrated continuously (see 2.4) |

Inverse relations, for reference only. The implementation never needs them:
- e = √(f² + g²) and tan(i/2) = √(h² + k²)
- Ω = atan2(k, h) and ϖ ≡ ω + Ω = atan2(g, f)
- ν = L − ϖ
- a = p / (1 − e²)

### 1.2 Auxiliary quantities

The names in brackets are the variable names in the source (`src/eoms/eoms_mee.cpp`). These are the only intermediates used on this page.

| symbol | definition | note |
|---|---|---|
| cL, sL | cos L, sin L | written $c_L$, $s_L$ in the equations |
| w | 1 + f cL + g sL | {cite:t}`betts2010` writes this as q; here q is reserved for Battin's q of the third-body term |
| s² [`s2`] | 1 + h² + k² | |
| α² | h² − k² | only used in the component form (2.2b); not needed by the code, which uses the vector form |
| r | p / w | orbit radius. Requires w > 0 |
| σ [`sig`] | √(p/μ) | |
| n₀ [`n0`] | √(μ/p³) | |
| Z | h sL − k cL | |
| w_L [`wL`] | ∂w/∂L = −f sL + g cL | |
| Z_L [`ZL`] | ∂Z/∂L = h cL + k sL | |

μ = `mus[0]` in DU³/TU². The time t is in TU. The ephemeris epoch is et = et0 + t·TU, exactly as in `eom_rv`.

### 1.3 Equinoctial frame

The equinoctial unit vectors are f̂, ĝ, ŵ, expressed in the integration frame (e.g. J2000):

$$
\hat{\boldsymbol f} = \frac{1}{s^2}\begin{pmatrix} 1 - k^2 + h^2 \\ 2hk \\ -2k \end{pmatrix},\qquad
\hat{\boldsymbol g} = \frac{1}{s^2}\begin{pmatrix} 2hk \\ 1 + k^2 - h^2 \\ 2h \end{pmatrix},\qquad
\hat{\boldsymbol w} = \frac{1}{s^2}\begin{pmatrix} 2k \\ -2h \\ 1 - h^2 - k^2 \end{pmatrix}
$$

(f̂, ĝ, ŵ) is right-handed and orthonormal for all real h and k: ŵ = f̂ × ĝ. This was verified symbolically.
- f̂ and ĝ span the orbit plane.
- f̂ is the direction from which the longitudes ϖ and L are measured.

**Derivatives of the frame.** These were derived with sympy and are used in Section 5:

$$
\begin{aligned}
\frac{\partial\hat{\boldsymbol f}}{\partial h} &= \frac{2}{s^2}\,k\,\hat{\boldsymbol g}, &
\frac{\partial\hat{\boldsymbol f}}{\partial k} &= -\frac{2}{s^2}\left(h\,\hat{\boldsymbol g} + \hat{\boldsymbol w}\right),\\
\frac{\partial\hat{\boldsymbol g}}{\partial h} &= \frac{2}{s^2}\left(\hat{\boldsymbol w} - k\,\hat{\boldsymbol f}\right), &
\frac{\partial\hat{\boldsymbol g}}{\partial k} &= \frac{2}{s^2}\,h\,\hat{\boldsymbol f},\\
\frac{\partial\hat{\boldsymbol w}}{\partial h} &= -\frac{2}{s^2}\,\hat{\boldsymbol g}, &
\frac{\partial\hat{\boldsymbol w}}{\partial k} &= \frac{2}{s^2}\,\hat{\boldsymbol f}.
\end{aligned}
$$

The frame does not depend on p, f, g or L.

---

## 2. Conversions

### 2.1 mee2cart(x, μ) → (r, v)

Vector form (recommended):

$$
\begin{aligned}
\boldsymbol r &= r\left(c_L\,\hat{\boldsymbol f} + s_L\,\hat{\boldsymbol g}\right) && \text{(2.1a)}\\
\boldsymbol v &= \sqrt{\mu/p}\,\left[-(s_L + g)\,\hat{\boldsymbol f} + (c_L + f)\,\hat{\boldsymbol g}\right] && \text{(2.1b)}
\end{aligned}
$$

The equivalent component form of {cite:t}`walker1985` and {cite:t}`betts2010` was checked against (2.1) to round-off:

$$
\begin{aligned}
r_x &= \frac{r}{s^2}\left(c_L + \alpha^2 c_L + 2hk\,s_L\right), \qquad
r_y = \frac{r}{s^2}\left(s_L - \alpha^2 s_L + 2hk\,c_L\right), \qquad
r_z = \frac{2r}{s^2}\left(h\,s_L - k\,c_L\right),\\
v_x &= -\frac{1}{s^2}\sqrt{\frac{\mu}{p}}\left(s_L + \alpha^2 s_L - 2hk\,c_L + g - 2fhk + \alpha^2 g\right),\\
v_y &= -\frac{1}{s^2}\sqrt{\frac{\mu}{p}}\left(-c_L + \alpha^2 c_L + 2hk\,s_L - f + 2ghk + \alpha^2 f\right),\\
v_z &= \frac{2}{s^2}\sqrt{\frac{\mu}{p}}\left(h\,c_L + k\,s_L + fh + gk\right) \qquad \text{(2.2b)}
\end{aligned}
$$

Preconditions:
- **p > 0 and w > 0.** w ≤ 0 can only occur for hyperbolic elements beyond the asymptote, where r would be negative.
  - `mee2cart` throws `std::domain_error`.
  - The equations of motion return `GSL_EDOM`, on which GSL halves the step and retries (it would stop on `GSL_EBADFUNC`). A trial Runge–Kutta stage that leaves the domain, e.g. after a large first step near i = 180°, therefore only shrinks the step.
  - `eval_mee` throws `std::domain_error` (`ValueError` in Python); the solvers throw it if the state stays out of the domain.
- **L is not wrapped.** Any real L is accepted.

The angular momentum is H = r × v = √(μp) ŵ, so ŵ is always along H. This is why n̂ = ŵ in Section 3.

### 2.2 cart2mee(r, v, μ) → x

1. **Angular momentum.** H = r × v and |H| = ‖H‖. If |H| = 0 (rectilinear orbit), `cart2mee` throws `std::domain_error`.
2. **Semi-latus rectum.** p = |H|² / μ.
3. **h and k from ŵ = H/|H|.** From the third component of ŵ in 1.3, 1 + ŵ_z = 2/s². Combined with ŵ_x = 2k/s² and ŵ_y = −2h/s², this gives:
   $$
   k = \frac{\hat w_x}{1 + \hat w_z} = \frac{H_x}{|H| + H_z}, \qquad
   h = -\frac{\hat w_y}{1 + \hat w_z} = -\frac{H_y}{|H| + H_z}
   $$
   The denominator is |H|(1 + cos i), which vanishes at i = 180°. This is the singularity of I = +1 elements.
   - If `|H| + H_z <= 1e-12 * |H|`, `cart2mee` throws `std::domain_error` ("retrograde equatorial orbit; MEE with I=+1 singular").
   - This trips within about 1.4e-6 rad of i = 180°, where |h| and |k| would be about 1.4e6.
   - Closer to 180° than about 0.1°, accuracy degrades. At i = 179.9° (√(h² + k²) = tan(i/2) ≈ 1146; |h| = |k| ≈ 810 for Ω = 45°), the MEE→Cartesian→MEE re-trip error in h is 5e-8 absolute, or 6e-11 relative.
4. **f̂ and ĝ from (h, k)**, with the formulas in 1.3, the same as in `mee2cart`, so that f and g are consistent with it.
5. **Eccentricity vector.** e = (v × H)/μ − r/‖r‖.

   In the orbit plane, e = e(cos ϖ f̂ + sin ϖ ĝ), because ϖ is measured from f̂. Therefore:
   $$
   f = \boldsymbol e\cdot\hat{\boldsymbol f}, \qquad g = \boldsymbol e\cdot\hat{\boldsymbol g}
   $$
6. **True longitude.** r̂ = cos L f̂ + sin L ĝ, so:
   $$
   L = \operatorname{atan2}\left(\boldsymbol r\cdot\hat{\boldsymbol g},\ \boldsymbol r\cdot\hat{\boldsymbol f}\right) \in (-\pi, \pi]
   $$

### 2.3 Edge cases

| case | behaviour |
|---|---|
| e = 0 | f = g = 0. Nothing is singular. L is still defined because it does not depend on ω |
| i = 0 | h = k = 0. Then f̂ = x̂ and ĝ = ŷ, and L is the true longitude from x̂ |
| e ≥ 1 | The formulas hold for p > 0 and w > 0. Tested with e = 1.5 |
| i → 180° | Singular. See step 3 in 2.2. The equations of motion have no check of their own: near 180°, h and k grow without bound and accuracy degrades (a regression test integrates i = 179.9° for 30 days) |
| H = 0 | `cart2mee` throws |

### 2.4 Wrapping L

- **`cart2mee`** returns L ∈ (−π, π]. This is the range of `std::atan2`.
- **The integrator** treats L as a continuous, unwrapped variable. It grows by 2π each revolution.
- **The EOM** must never wrap L. Everything uses cL and sL, so wrapping is unnecessary.
- **Comparing L values** (tests, round trips): compare wrap(L₁ − L₂) = atan2(sin(L₁ − L₂), cos(L₁ − L₂)), never L₁ − L₂ directly.
- **`mee2cart`** accepts any L.

---

## 3. RTN frame

The definition is the usual one, with the unit vectors as columns of C:

$$
\hat{\boldsymbol r} = \frac{\boldsymbol r}{\|\boldsymbol r\|}, \qquad
\hat{\boldsymbol n} = \frac{\boldsymbol H}{\|\boldsymbol H\|}, \qquad
\hat{\boldsymbol\theta} = \hat{\boldsymbol n}\times\hat{\boldsymbol r}, \qquad
C = \begin{pmatrix}\hat{\boldsymbol r} & \hat{\boldsymbol\theta} & \hat{\boldsymbol n}\end{pmatrix} \ (3\times 3),
$$

$$
\boldsymbol a_\text{rtn} = C^\mathsf{T}\boldsymbol a_I
 = \begin{pmatrix}\hat{\boldsymbol r}\cdot\boldsymbol a_I & \hat{\boldsymbol\theta}\cdot\boldsymbol a_I & \hat{\boldsymbol n}\cdot\boldsymbol a_I\end{pmatrix}^\mathsf{T}
 = \begin{pmatrix} a_R & a_T & a_N\end{pmatrix}^\mathsf{T}
$$

As explicit functions of MEE, these depend only on (h, k, L). This follows from (2.1) and H ∥ ŵ:

$$
\begin{aligned}
\hat{\boldsymbol r} &= c_L\,\hat{\boldsymbol f} + s_L\,\hat{\boldsymbol g}
  = \frac{1}{s^2}\begin{pmatrix} c_L + \alpha^2 c_L + 2hk\,s_L & s_L - \alpha^2 s_L + 2hk\,c_L & 2Z\end{pmatrix}^\mathsf{T}\\
\hat{\boldsymbol\theta} &= -s_L\,\hat{\boldsymbol f} + c_L\,\hat{\boldsymbol g}
  = \frac{1}{s^2}\begin{pmatrix} -s_L - \alpha^2 s_L + 2hk\,c_L & c_L - \alpha^2 c_L - 2hk\,s_L & 2Z_L\end{pmatrix}^\mathsf{T}\\
\hat{\boldsymbol n} &= \hat{\boldsymbol w}
  = \frac{1}{s^2}\begin{pmatrix} 2k & -2h & 1 - h^2 - k^2\end{pmatrix}^\mathsf{T} \qquad \text{(3.1)}
\end{aligned}
$$

The library builds C from (h, k, L) with (3.1), not from v. This form is exact, needs no ∂v/∂x, and is what the Jacobian in Section 5 assumes.

The derivatives of the RTN basis follow from 1.3 and are written back in the RTN basis:

$$
\begin{aligned}
\frac{\partial\hat{\boldsymbol r}}{\partial h} &= \frac{2}{s^2}\left(k\,\hat{\boldsymbol\theta} + s_L\,\hat{\boldsymbol n}\right), &
\frac{\partial\hat{\boldsymbol r}}{\partial k} &= \frac{2}{s^2}\left(-h\,\hat{\boldsymbol\theta} - c_L\,\hat{\boldsymbol n}\right), &
\frac{\partial\hat{\boldsymbol r}}{\partial L} &= \hat{\boldsymbol\theta},\\
\frac{\partial\hat{\boldsymbol\theta}}{\partial h} &= \frac{2}{s^2}\left(-k\,\hat{\boldsymbol r} + c_L\,\hat{\boldsymbol n}\right), &
\frac{\partial\hat{\boldsymbol\theta}}{\partial k} &= \frac{2}{s^2}\left(h\,\hat{\boldsymbol r} + s_L\,\hat{\boldsymbol n}\right), &
\frac{\partial\hat{\boldsymbol\theta}}{\partial L} &= -\hat{\boldsymbol r},\\
\frac{\partial\hat{\boldsymbol n}}{\partial h} &= -\frac{2}{s^2}\left(s_L\,\hat{\boldsymbol r} + c_L\,\hat{\boldsymbol\theta}\right), &
\frac{\partial\hat{\boldsymbol n}}{\partial k} &= \frac{2}{s^2}\left(c_L\,\hat{\boldsymbol r} - s_L\,\hat{\boldsymbol\theta}\right), &
\frac{\partial\hat{\boldsymbol n}}{\partial L} &= \boldsymbol 0 \qquad \text{(3.2)}
\end{aligned}
$$

All derivatives with respect to p, f and g are zero.

---

## 4. Gauss variational equations

$$
\dot{\boldsymbol x} = \boldsymbol b(\boldsymbol x) + B(\boldsymbol x)\,\boldsymbol a_\text{rtn}, \qquad
\boldsymbol a_\text{rtn} = C(\boldsymbol x)^\mathsf{T}\,\boldsymbol a_I\big(\boldsymbol r(\boldsymbol x), t\big) \qquad \text{(4.1)}
$$

- a_I is the sum of the third-body accelerations, the SRP acceleration (if `include_srp`) and the central-body spherical-harmonics acceleration (if `sh_file` is given), in the inertial integration frame, evaluated at r_vec(x) and et = et0 + t·TU.
- It is exactly the acceleration that the Cartesian equations of motion add to the central term −μ r/r³ (`add_perturbations`):
  $$
  \boldsymbol a_I = \sum_{i=1}^{N-1} \texttt{thirdbody\_accel}\big(\boldsymbol r, \boldsymbol s_i(\mathrm{et}), \mathrm{GM}[i]\big)
     + \boldsymbol a_\text{srp}(\boldsymbol r, \mathrm{et}) + \boldsymbol a_\text{sh}(\boldsymbol r, \mathrm{et})
  $$

  The individual terms are given in [the dynamical model](dynamics_model.md).
- Only the position part of `mee2cart` is needed to evaluate it.

**Keplerian part.** This is the only place μ = `GM[0]` enters:

$$
\boldsymbol b = \begin{pmatrix} 0 & 0 & 0 & 0 & 0 & n_0 w^2\end{pmatrix}^\mathsf{T}, \qquad
n_0 w^2 = \sqrt{\mu p}\left(\frac{w}{p}\right)^2
$$

**Control matrix.** B(x) is 6x3. The columns are R, T, N. All entries not listed are 0.

$$
B = \sigma\begin{pmatrix}
0 & 2p/w & 0\\
s_L & \big((w+1)c_L + f\big)/w & -gZ/w\\
-c_L & \big((w+1)s_L + g\big)/w & fZ/w\\
0 & 0 & s^2 c_L/(2w)\\
0 & 0 & s^2 s_L/(2w)\\
0 & 0 & Z/w
\end{pmatrix}
$$

In the 0-based notation of the code: `B[0][1] = 2pσ/w`, `B[1][0] = σ sL`, `B[1][1] = σ((w+1)cL + f)/w`,
`B[1][2] = −σgZ/w`, `B[2][0] = −σ cL`, `B[2][1] = σ((w+1)sL + g)/w`, `B[2][2] = σfZ/w`,
`B[3][2] = σs² cL/(2w)`, `B[4][2] = σs² sL/(2w)`, `B[5][2] = σZ/w`.

**Convention and provenance.** This is {cite:t}`walker1985` with I = +1, in the form reproduced in {cite:t}`betts2010`, with q → w. It is fixed by numerical verification, not by citation.
- **Published typos.** Published transcriptions differ in the normal-acceleration terms of ḟ and ġ: the signs of B[1][2] and B[2][2], and f vs g inside B[2][2]. Walker et al. also published an erratum {cite:p}`walker1986erratum`.
- **Correct form.** ḟ's normal term contains −g Z/w and ġ's contains +f Z/w.
- **Negative controls in Section 6(b).** Flipping both signs, or writing g instead of f in B[2][2], produces relative errors of 1e-6 to 1 against the Cartesian truth. The form above agrees to 5e-28.

---

(gve-jacobian)=
## 5. Jacobian A = ∂ẋ/∂x and the STM

### 5.0 Structure

a_I depends on x only through r_vec(x) and on t through the ephemerides. Its spatial gradient is G = ∂a_I/∂r (3x3). Differentiating (4.1) gives:

$$
A = A_b + A_B + B\,\tilde A \qquad \text{(5.0)}
$$

with

$$
\begin{aligned}
(A_b)_{im} &= \frac{\partial b_i}{\partial x_m}, &&\\
(A_B)_{im} &= \sum_j \frac{\partial B_{ij}}{\partial x_m}\,(a_\text{rtn})_j && \text{($B$ differentiated at fixed $\boldsymbol a_\text{rtn}$)},\\
\tilde A_{jm} &= \frac{\partial (a_\text{rtn})_j}{\partial x_m} = (G_\text{rtn} D_\text{rtn})_{jm} + F_{jm} && (3\times 6),\\
G_\text{rtn} &= C^\mathsf{T} G\, C && \text{($3\times3$, gradient expressed in RTN)},\\
D_\text{rtn} &= C^\mathsf{T}\,\frac{\partial\boldsymbol r}{\partial\boldsymbol x} && \text{($3\times6$, position sensitivity in RTN)},\\
F &= \frac{\partial C^\mathsf{T}}{\partial\boldsymbol x}\,\boldsymbol a_I && \text{($3\times6$, rotation of the RTN basis at fixed $\boldsymbol a_I$)}.
\end{aligned}
$$

- ∂/∂t terms do not enter: the STM is ∂x(t)/∂x(t₀) and the time dependence is explicit.
- The velocity never appears.

**STM layout.** Same as `eomstm_rv`:
- y[0..5] = x.
- y[6 + 6i + j] = Φ[i][j], row-major, with Φ(t₀) = I.
- The derivative is:
  $$
  \dot y_{6 + 6i + j} = \sum_{m=0}^{5} A_{im}\, y_{6 + 6m + j}
  $$

### 5.1 A_b: the Keplerian part

Only row 5 is nonzero:

$$
(A_b)_{5,\,:} = \begin{pmatrix} -\dfrac{3}{2p}\,n_0 w^2 & 2 n_0 w\,c_L & 2 n_0 w\,s_L & 0 & 0 & 2 n_0 w\,w_L \end{pmatrix}
$$

### 5.2 ∂B/∂x: every nonzero entry

- **Derivatives with respect to p.** Every nonzero B[i][j] is σ times a p-independent factor, except B[0][1], which is proportional to p^{3/2}. So:
  $$
  \frac{\partial B_{ij}}{\partial p} = \frac{B_{ij}}{2p} \quad \text{for all } (i,j) \ne (0,1), \qquad
  \frac{\partial B_{01}}{\partial p} = \frac{3B_{01}}{2p}
  $$
- **Other derivatives.** These use ∂w/∂f = cL, ∂w/∂g = sL, ∂w/∂L = w_L, ∂Z/∂h = sL, ∂Z/∂k = −cL, ∂Z/∂L = Z_L, and ∂s²/∂h = 2h, ∂s²/∂k = 2k.
- **Omitted entries.** Any (i, j, m) not listed below, other than the ∂/∂p entries above, is exactly 0.

| B entry | ∂/∂f | ∂/∂g | ∂/∂h | ∂/∂k | ∂/∂L |
|---|---|---|---|---|---|
| B[0][1] = 2pσ/w | −2pσ cL/w² | −2pσ sL/w² | 0 | 0 | −2pσ w_L/w² |
| B[1][0] = σ sL | 0 | 0 | 0 | 0 | σ cL |
| B[1][1] = σ(cL + (cL+f)/w) | σ[1/w − (cL+f) cL/w²] | −σ (cL+f) sL/w² | 0 | 0 | σ[−sL − sL/w − (cL+f) w_L/w²] |
| B[1][2] = −σ g Z/w | σ g Z cL/w² | −σ Z/w + σ g Z sL/w² | −σ g sL/w | σ g cL/w | −σ g (Z_L/w − Z w_L/w²) |
| B[2][0] = −σ cL | 0 | 0 | 0 | 0 | σ sL |
| B[2][1] = σ(sL + (sL+g)/w) | −σ (sL+g) cL/w² | σ[1/w − (sL+g) sL/w²] | 0 | 0 | σ[cL + cL/w − (sL+g) w_L/w²] |
| B[2][2] = σ f Z/w | σ Z/w − σ f Z cL/w² | −σ f Z sL/w² | σ f sL/w | −σ f cL/w | σ f (Z_L/w − Z w_L/w²) |
| B[3][2] = σ s² cL/(2w) | −σ s² cL²/(2w²) | −σ s² cL sL/(2w²) | σ h cL/w | σ k cL/w | (σ s²/2)(−sL/w − cL w_L/w²) |
| B[4][2] = σ s² sL/(2w) | −σ s² sL cL/(2w²) | −σ s² sL²/(2w²) | σ h sL/w | σ k sL/w | (σ s²/2)(cL/w − sL w_L/w²) |
| B[5][2] = σ Z/w | −σ Z cL/w² | −σ Z sL/w² | σ sL/w | −σ cL/w | σ (Z_L/w − Z w_L/w²) |

The rewritten forms (w+1)cL + f over w = cL + (cL+f)/w, and similarly for B[2][1], are identities. They make the derivatives shorter.

**Assembly.** With a_rtn = [a_R, a_T, a_N]:

$$
(A_B)_{im} = \frac{\partial B_{i0}}{\partial x_m}\,a_R + \frac{\partial B_{i1}}{\partial x_m}\,a_T + \frac{\partial B_{i2}}{\partial x_m}\,a_N
$$

### 5.3 D_rtn = Cᵀ ∂r_vec/∂x

From r_vec = r r̂, with r = p/w and (3.2):
- ∂r/∂p = 1/w
- ∂r/∂f = −r cL/w
- ∂r/∂g = −r sL/w
- ∂r/∂L = r(f sL − g cL)/w
- ∂r/∂h = ∂r/∂k = 0

$$
D_\text{rtn} =
\begin{pmatrix}
1/w & -r c_L/w & -r s_L/w & 0 & 0 & r(f s_L - g c_L)/w\\
0 & 0 & 0 & 2rk/s^2 & -2rh/s^2 & r\\
0 & 0 & 0 & 2r s_L/s^2 & -2r c_L/s^2 & 0
\end{pmatrix}
$$

The columns are $p, f, g, h, k, L$ and the rows R, T, N.

If the inertial form is wanted, use ∂r_vec/∂x_m = C · D_rtn[:, m].

### 5.4 F = (∂Cᵀ/∂x) a_I: the rotation of the RTN basis

From (3.2):
- F[j][m] = (∂ê_j/∂x_m) · a_I, where ê_0 = r̂, ê_1 = θ̂, ê_2 = n̂.
- The columns for p, f and g are 0.

$$
F_{:,\,3:6} =
\begin{pmatrix}
\frac{2}{s^2}\left(k a_T + s_L a_N\right) & \frac{2}{s^2}\left(-h a_T - c_L a_N\right) & a_T\\
\frac{2}{s^2}\left(-k a_R + c_L a_N\right) & \frac{2}{s^2}\left(h a_R + s_L a_N\right) & -a_R\\
-\frac{2}{s^2}\left(s_L a_R + c_L a_T\right) & \frac{2}{s^2}\left(c_L a_R - s_L a_T\right) & 0
\end{pmatrix}
$$

(columns $h, k, L$).

### 5.5 G: the inertial gradient of the perturbations

$$
G = \sum_{i=1}^{N-1} G_i + G_\text{srp} + G_\text{sh}, \qquad G_\text{rtn} = C^\mathsf{T} G\, C
$$

G_sh is present only when a central-body gravity field is given (see below).

**Third body.** `thirdbody_accel` evaluates Battin's form {cite:p}`battin1999`:
- a_i = −μ_i (r + F(q_i) s_i)/‖d_i‖³, with d_i = r − s_i.
- This is algebraically identical to −μ_i (d_i/‖d_i‖³ + s_i/‖s_i‖³).
- The indirect term does not depend on r, so the exact gradient is:

$$
G_i = \frac{\mu_i}{\|\boldsymbol d_i\|^5}\left(3\,\boldsymbol d_i \boldsymbol d_i^\mathsf{T} - \|\boldsymbol d_i\|^2 I_3\right)
$$

This is exactly `gravity_gradient_third_body(rvec, svec, GM[i], GG)`:
- svec is the body position already divided by DU, as passed to `thirdbody_accel`.
- The Cartesian STM (`eomstm_rv`) uses the same gradient.

**SRP.** `add_perturbations` (`src/eoms/forces.cpp`) computes, with the SRP constants of `src/perturbations/SRP.cpp`:
- d = r − s, where s = position of the Sun (NAIF 10, which must be listed in `naif_ids`) relative to the central body, divided by DU; s = 0 if the Sun is the central body
- R = ‖d‖
- C_r = 1 + ρ_S + 5ρ_D/3
- K = (P₀/c) C_r (S/m) (AU/DU)² / (1e3 · DU/TU²), with P₀ = 1367 W/m² the solar flux at 1 AU and AU = 149597870.7 km (`srp_coefficient`, computed once in `SpiceParameters`)
- κ = K / R³
- a_srp = κ d = K d / R³

At 1 AU this is (P₀/c) C_r S/m in m/s², converted to DU/TU². K does not depend on r or t, and the exact gradient is

$$
G_\text{srp} = \frac{\partial\boldsymbol a_\text{srp}}{\partial\boldsymbol r}
 = K\left(\frac{I_3}{R^3} - \frac{3\,\boldsymbol d\boldsymbol d^\mathsf{T}}{R^5}\right)
 = \kappa\left(I_3 - \frac{3\,\boldsymbol d\boldsymbol d^\mathsf{T}}{R^2}\right)
$$

Notes:
- This is a repulsive inverse-square field: G_srp is trace-free, with the opposite sign to a point-mass gradient.
- `srp_gradient` (`src/perturbations/SRP.cpp`) takes d and κ. The Sun position is the one already queried for its third-body term, so SRP costs no extra ephemeris call.
- There is no shadow model, which matches the Cartesian model.

**Central-body spherical harmonics.** If a gravity field is given (`sh_file`, see [spherical_harmonics.md](spherical_harmonics.md)), `sh_perturbation` adds the acceleration of degrees 2..nmax to a_I. Its gradient is

$$
G_\text{sh} = T^\mathsf{T} G_\text{bf}\, T, \qquad T = \texttt{pxform}(\texttt{naif\_frame}, \texttt{sh\_frame}, \mathrm{et})
$$

where G_bf is the body-fixed gradient from the V, W recursion.
- The SH acceleration depends only on r and t, so nothing else in §5.6 changes.
- Ṫ does not enter, because the Jacobian is taken at fixed t.

### 5.6 Algorithm for one RHS + STM evaluation

1. Unpack x. Compute cL, sL, w, s², Z, w_L, Z_L, σ, n₀ and r.
   - If p ≤ 0 or w ≤ 0, return `GSL_EDOM` (recoverable: GSL retries with a smaller step; see 2.1).
2. Compute f̂, ĝ, ŵ (1.3), then r̂, θ̂, n̂ (3.1), then r_vec = r r̂.
3. Set et = et0 + t·TU. Loop over the bodies i = 1..N−1:
   - a_I += a_i, the third-body acceleration of `thirdbody_accel`
   - G += G_i
   - on a SPICE error, save NAIF's message, `reset_c()`, and return `GSL_EBADFUNC`
4. SRP, if `include_srp`: a_I += a_srp and G += G_srp. Then central-body spherical harmonics, if
   given: a_I += a_sh and G += G_sh.
5. Compute a_rtn = [r̂·a_I, θ̂·a_I, n̂·a_I] and ẋ = b + B a_rtn.
6. For the STM version only:
   - G_rtn = Cᵀ G C
   - Ã = G_rtn D_rtn + F
   - A = A_b + A_B + B Ã
   - Φ̇ = A Φ

### 5.7 Optional: J_cart←mee = ∂(r_vec, v_vec)/∂x

`mee2cart_jacobian` returns this matrix. It maps an MEE STM to Cartesian coordinates, $\Phi_\text{cart} = J(\boldsymbol x_f)\,\Phi_\text{MEE}\,J(\boldsymbol x_0)^{-1}$. Position columns:

$$
\begin{aligned}
\frac{\partial\boldsymbol r}{\partial p} &= \frac{\hat{\boldsymbol r}}{w}, &
\frac{\partial\boldsymbol r}{\partial f} &= -\frac{r c_L}{w}\,\hat{\boldsymbol r}, &
\frac{\partial\boldsymbol r}{\partial g} &= -\frac{r s_L}{w}\,\hat{\boldsymbol r},\\
\frac{\partial\boldsymbol r}{\partial h} &= \frac{2r}{s^2}\left(k\,\hat{\boldsymbol\theta} + s_L\,\hat{\boldsymbol n}\right), &
\frac{\partial\boldsymbol r}{\partial k} &= \frac{2r}{s^2}\left(-h\,\hat{\boldsymbol\theta} - c_L\,\hat{\boldsymbol n}\right), &
\frac{\partial\boldsymbol r}{\partial L} &= \frac{r(f s_L - g c_L)}{w}\,\hat{\boldsymbol r} + r\,\hat{\boldsymbol\theta}
\end{aligned}
$$

Velocity columns, with σ' = √(μ/p) and v_vec from (2.1b):

$$
\begin{aligned}
\frac{\partial\boldsymbol v}{\partial p} &= -\frac{\boldsymbol v}{2p}, \qquad
\frac{\partial\boldsymbol v}{\partial f} = \sigma'\,\hat{\boldsymbol g}, \qquad
\frac{\partial\boldsymbol v}{\partial g} = -\sigma'\,\hat{\boldsymbol f}, \qquad
\frac{\partial\boldsymbol v}{\partial L} = -\sigma'\,\hat{\boldsymbol r},\\
\frac{\partial\boldsymbol v}{\partial h} &= \frac{2\sigma'}{s^2}\Big[-k(c_L+f)\,\hat{\boldsymbol f} - k(s_L+g)\,\hat{\boldsymbol g} + (c_L+f)\,\hat{\boldsymbol w}\Big],\\
\frac{\partial\boldsymbol v}{\partial k} &= \frac{2\sigma'}{s^2}\Big[h(c_L+f)\,\hat{\boldsymbol f} + h(s_L+g)\,\hat{\boldsymbol g} + (s_L+g)\,\hat{\boldsymbol w}\Big]
\end{aligned}
$$

J_mee←cart is obtained as the inverse of this 6x6 matrix (LU solve). It is nonsingular whenever the MEE are defined.

### 5.8 Thrust and mass: the `meem` dynamics

`solve_ivp_meem` and `solve_ivpstm_meem` append the mass m and take the control u = [τ, Γ], giving a 7x7 A and a 7x4 control Jacobian. The units and the constants c₁, c₂ are in [the mass-augmented dynamics](zoh_mass_dynamics.md). This control Jacobian is written **B_u** here, because B is the matrix of (4.1). With a_c = c₁τ/m and a_srp ∝ 1/m computed from the state mass (K of §5.5 with m̄ m in place of `srp_m`, recomputed at each evaluation):

$$
\dot{\boldsymbol x} = \boldsymbol b + B\,\big(C^\mathsf{T}\boldsymbol a_I + \boldsymbol a_{c,\text{rtn}}\big), \qquad \dot m = -c_2\Gamma
$$

Here a_I includes a_srp and, under inertial control, a_c; and a_{c,rtn} is a_c under RTN control, and zero otherwise.

**A_e.** The 6x6 block keeps the decomposition (5.0), with a_rtn the total RTN acceleration in A_B. Thrust enters each term as follows (`mee_evaluate_ext`):

| Control | ẋ and A_B | F (§5.4) | G_rtn D_rtn |
|---|---|---|---|
| RTN | included | excluded: its components do not depend on x | not affected |
| inertial | included | included: added to a_I before projecting | not affected |

**Mass column of A.** The last row of A is zero. Both thrust and SRP are proportional to 1/m:

$$
\frac{\partial\dot{\boldsymbol x}}{\partial m} =
\begin{cases}
-\frac{1}{m}\,B\left(C^\mathsf{T}\boldsymbol a_\text{srp} + \boldsymbol a_c\right) & \text{RTN control}\\[1ex]
-\frac{1}{m}\,B\,C^\mathsf{T}\left(\boldsymbol a_\text{srp} + \boldsymbol a_c\right) & \text{inertial control}
\end{cases}
$$

**Control Jacobian.**

$$
B_u = \begin{pmatrix} \frac{c_1}{m}\,B\,R & 0_{6\times1}\\ 0_{1\times3} & -c_2 \end{pmatrix}, \qquad
R = \begin{cases} I_3 & \text{RTN control}\\ C^\mathsf{T} & \text{inertial control}\end{cases}
$$

The elements do not depend directly on Γ; it reaches them only through m. The sensitivities follow Φ̇ = AΦ and Ψ̇ = AΨ + B_u, with the 84-component layout of [the same page](zoh_mass_dynamics.md#augmented-state-layout).

---

## 6. Verification

The formulas of sections 1–5 were checked against an independent Python transcription, before and
alongside the C++ implementation. Checks (a)–(c) ran in 50-digit arithmetic (mpmath), with central
finite differences (FD) of step 1e-20, so that any formula error shows up as an
O(perturbation) discrepancy instead of being hidden by double-precision FD noise. The library
itself is tested by `test-python/test_mee.py` and `tests/test_mee.cpp` (section 7).

**Test states.** μ_⊕ = 398600.4418, except for the Moon case:
- LEO: a = 6778 km, e = 1e-3, i = 51.6°
- GTO: e = 0.73
- HEO: e = 0.95, ν = 179°
- near-polar: i = 89.99°
- circular equatorial: e = 0 and i = 0 exactly
- i = 170° and i = 179.9°
- hyperbolic: e = 1.5
- the Moon test state, μ = 4902.800582147760, which has i = 128.5° in J2000, e = 0.919, r_p = 3011 km and r_a = 71529 km
- MEE states with L = π − 1e-12, −π + 1e-12, exactly π, and 50.3 (8 revolutions)

**Synthetic perturbation model for (b) and (c).** These checks use the exact third-body (Battin) and SRP functional forms of the library, with fixed body positions.
- Two third bodies: μ₁ = 0.4μ at about 3.6 r, and μ₂ = 3e5 μ at about 170 r.
- SRP with |a_srp| ≈ 0.02 μ/r².
- An extra arbitrary constant acceleration was added in (b).
- The magnitudes are inflated so the gradient terms dominate. The Jacobian rows for f, g, h, k are pure perturbation terms.

| check | metric | max error |
|---|---|---|
| (a) cart→mee→cart, mp | relative ‖Δr‖/‖r‖, ‖Δv‖/‖v‖ | 5e-49 |
| (a) cart→mee→cart, float64, all cases except i = 179.9° | same | 2.9e-15 (worst: Moon state) |
| (a) cart→mee→cart, float64, i = 179.9° | same | 3.3e-14 |
| (a) mee→cart→mee, float64 | p relative; f, g, h, k relative to max(1, \|·\|); L wrapped | 5.3e-15 (6e-11 at i = 179.9°) |
| (a) L = ±π∓1e-12, π, 50.3 | L returned in (−π, π], with wrap(ΔL) | 2e-15. L = 50.3 returns 0.0345 = 50.3 − 8·2π, as specified |
| (a) i = 180° exactly | throws | yes |
| (a) vector form (2.1) vs component form (2.2b) | relative | 3.6e-16 (float64), 5e-49 (mp) |
| (b) GVE RHS vs d/dt cart2mee along ṙ = v, v̇ = −μr/r³ + a. J_mee←cart by mp-FD with L-difference wrapped | \|Δẋᵢ\| / Σⱼ\|J_ij ċ_j\| | **5.1e-28** (all others ≤ 5e-30) |
| (b) negative control: B[1][2], B[2][2] sign-flipped | same | 3e-5 … 1.27 (detected; blind only for f = g = 0) |
| (b) negative control: g instead of f in B[2][2] | same | 1.2e-6 … 1.02 (detected) |
| (c) analytic A vs mp central FD of the RHS | row-normalised on A·diag(p,1,1,1,1,1) | **2.6e-30** |
| (c) analytic J_cart←mee (5.7) vs mp FD | column-normalised | 8.6e-29 |
| (c) negative controls on the Moon state: drop G / drop A_B | same | 2.0 / 2.06 (detected) |
| (c') analytic A vs float64 FD (δp = 1e-6 p, others 1e-7) | row-normalised | 2.4e-9 |

**(d) Real ephemerides (DE440 via spiceypy), Moon test case, 20 days.** The cases with SRP were
computed with an earlier SRP flux convention; the test suite repeats the same comparisons with the
current model (7.3, 7.5).
- **Setup.** scipy DOP853, rtol = atol = 1e-13. The model is a Python transcription of the Cartesian EOM (`eom_rv`). For the N-body-only case, the Cartesian Python model reproduces the C++ reference final state in `examples/solve_rv.cpp` to 1.7e-8 km.
- **GVE vs Cartesian, with SRP.** ‖Δr‖ = 1.3e-7 km and ‖Δv‖ = 3.9e-13 km/s.
- **GVE vs Cartesian, without SRP.** ‖Δr‖ = 6.9e-8 km and ‖Δv‖ = 5.0e-13 km/s.
- **Agreement scales with the integrator tolerance** (with SRP; without SRP it is similar):

  | rtol (= atol) | ‖Δr‖ |
  |---|---|
  | 1e-11 | 6.3e-5 km |
  | 1e-12 | 5.9e-6 km |
  | 3e-14 | 5.9e-8 km |

  So the residual is integration error, not model mismatch.
- **Φ_MEE(tf) vs central FD of propagated final MEE.** Steps: δp = 1e-6 p₀, others 1e-7.
  - Row-normalised max error: 5.0e-7, on diag(1/p_f,1,…)·Φ·diag(p₀,1,…).
  - Elementwise relative error, for entries > 1e-3 of the row max: 2.6e-5.
- **Without SRP: Φ_MEE vs J_cart←mee(x_f)⁻¹ Φ_cart J_cart←mee(x₀).** Φ_cart comes from the `eomstm_rv` model. Row-normalised: 3.0e-10.
- **Conditioning.** cond(Φ_cart) = 7e12 and cond(Φ_MEE) = 1e12.

---

## 7. Test reference data

:::{note}
This section specifies the tests in `test-python/test_mee.py` and `tests/test_mee.cpp`, which cite
its subsection numbers. It is reference material for developers; using the library does not
require it. The tests read the kernels `naif0012.tls` and `de440.bsp` from `$SPICE`.
:::

### 7.0 Common data

```
et0    = 740463475.5117936
state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
           0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
GM     = [4.902800582147760e+03, 398600.432896939, 132712197035.766]   # legacy test constants, not DE440
naif_ids = ["MOON", "EARTH", "SUN"],  naif_frame = "J2000",  DU = TU = 1
S (srp_A) = 34.675, srp_m = 1850, srp_rhoS = 0.9, srp_rhoD = 0
tspan  = [0, 20*86400]
```

Reference values from the Python transcription (float64):
- **Initial MEE.** cart2mee(state0, GM[0]), expected to 1e-13 relative on p and 1e-13 absolute on the rest:
  ```
  x0 = [5778.190022842265, -0.34651609849537257, 0.8514048990211349,
        1.9784209071365653, 0.6178109804659669, -1.184288478833202]
  ```
- **Final MEE with SRP.** A sanity check, not the primary assertion. The reference is the `solve_ivp_mee` result, confirmed by an independent RK4 + spiceypy Cartesian transcription to 1.1e-5 km (4th-order convergence):
  ```
  [5178.576789928225, 0.8809219032708616, -0.28942480229551987,
   0.24055734575159027, -1.2805787538758713, 21.717427888269437]
  ```
  L travels 3.64 revolutions.
- **Final MEE without SRP:**
  ```
  [5177.094816314719, 0.8795907367139397, -0.29345826808033154,
   0.23584006097182486, -1.2775753822790343, 21.712831372236334]
  ```

### 7.1 Round trips (no SPICE needed)

Cartesian states, μ = 398600.4418 unless stated:

| name | [x, y, z, vx, vy, vz] (km, km/s) | MEE (for cross-checking cart2mee) |
|---|---|---|
| LEO | [-2103.7154800484846, 3643.742096113127, 5308.456731666615, -6.647330004445163, -3.833624329580397, 0.004603821856452162] | [6777.993222, 3.42020143325783e-4, 9.3969262078571e-4, 0.418653037730661, 0.241709444030852, 2.0943951023932] |
| GTO, e = 0.73 | [5878.595087575521, 3000.5849397194843, -439.2672278754225, -3.6936046189819858, 8.180536450190633, -4.859712456831758] | [11373.99192144, 0.694841890805239, 0.225767816090336, -0.238651523310869, -0.086862050847443, 0.488692190558412] |
| HEO, e = 0.95 | [87047.82562847203, -3393.3994629546805, 173830.49101033443, 0.04996504784397327, 0.3187217213914905, 0.09977789493723822] | [9750.0, -0.95, ≈0, ≈0, -0.617612587860989, -0.0174532925199436] |
| near-polar, i = 89.99° | [-2237.5335584636528, 3877.384107889878, -5335.099292595186, -2.88464292575727, 4.994628935389528, 4.925069859403147] | [6999.3, -0.00642787609686526, 0.00766044443118989, -0.49991274115195, 0.865874267026206, 1.22173047639603] |
| circular equatorial | [7000, 0, 0, 0, 7.546053290107541, 0] | [7000, 0, 0, 0, 0, 0] (\|f\| ≤ 2e-16) |
| i = 170° | [-55.46704301173019, -7246.552606392538, 896.5989572886389, -7.530175620232078, -1.4387377569063755, -0.7594926786100786] | [7680, -0.0517638090205041, 0.193185165257814, 8.08226749259943, 8.08226749259943, 3.14159265358979] |
| hyperbolic, e = 1.5 | [5775.472866359693, 7498.003270192943, 5354.4553621173945, -6.119856157164796, 5.295429592561897, 5.2676006404103894] | [25000, 1.29903810567666, 0.75, 0.358440708571026, 0.0632027679053317, 1.0471975511966] |
| Moon, μ = 4902.800582147760 | state0 | x0 above |

MEE states, μ = 398600.4418:
- [7000, 0.01, -0.02, 0.3, -0.2, L] for L ∈ {π − 1e-12, −π + 1e-12, π}
- [42164, 0.1, 0.05, 0.01, 0.02, 50.3]

Assertions:
- **cart→mee→cart.** ‖Δr‖/‖r‖ ≤ 1e-13 and ‖Δv‖/‖v‖ ≤ 1e-13. Observed ≤ 3e-15; i = 170° is also within this.
- **mee→cart→mee.**
  - |Δp|/p ≤ 1e-13
  - |Δf|, |Δg| ≤ 1e-13
  - |Δh|, |Δk| ≤ 1e-13·max(1, |h|, |k|)
  - |wrap(ΔL)| ≤ 1e-13
  - L_out ∈ (−π, π]
  - For L = 50.3, L_out = 50.3 − 16π to 1e-13.
- **cart2mee against the MEE column of the table:** 1e-12 relative on p, 1e-12 absolute on the others.
- **Singularity.** cart2mee([7000, 0, 0, 0, −7.5, 0], μ) (i = 180°) throws. mee2cart with p ≤ 0 or w ≤ 0 throws.

### 7.2 Two-body invariance

Use `solve_ivp_mee` with `naif_ids = ["MOON"]`, `GMs = [GM[0]]`, no SRP, x0 as above, and 20 days.
- **p, f, g, h, k are bit-identical to x0 at every output.** With no perturbation, a_I = 0 and B·a_rtn = 0 exactly. Every RK stage then leaves these five components unchanged. Assert `==`.
- **mee2cart(x(tf)) vs `solve_ivp_rv` with the central body only.** ‖Δr‖ ≤ 1e-5 km and ‖Δv‖ ≤ 1e-10 km/s. Both have integration error; the expected value is about 1e-7 km.

### 7.3 GVE vs Cartesian, final-state agreement

Use the Moon data (7.0) with the default tolerances (atol 1e-12, rtol 1e-14) in both solvers.
- **With SRP.** Propagate x0 with `solve_ivp_mee` and state0 with `solve_ivp_rv`, from the same `SpiceParameters`. Assert ‖mee2cart(x_f) − rv_f‖ ≤ 1e-5 km in position and ≤ 1e-10 km/s in velocity. Expected about 1e-7 km and 4e-13 km/s; see 6(d).
- **Without SRP.** Same comparison with `include_srp = False`. The C++ Cartesian reference final state is in `examples/solve_rv.cpp`.
- **Also assert, in MEE space:** |p_f − p_c|/p ≤ 1e-9, |Δf|, |Δg|, |Δh|, |Δk| ≤ 1e-9 and |wrap(L_f − L_c)| ≤ 1e-9, where x_c = cart2mee(rv_f).

### 7.4 Analytic Jacobian vs FD of the RHS (single epoch)

Use x0 with the Moon data at t = 0 and also at t = 5 days, plus the LEO, GTO and i = 170° MEE states with BODIES = ["EARTH", "MOON", "SUN"] (Earth GM first).
- Central FD with δp = 1e-6·p and δ = 1e-7 for f, g, h, k, L.
- Compare Ā = A·diag(p,1,1,1,1,1) row by row: max_j |Ā_ij − Ā^FD_ij| ≤ 1e-6 · max_j |Ā^FD_ij|. The observed float64 value is 2.4e-9 with inflated perturbations.
- Additionally check the SRP gradient on its own: `test_jacobian_vs_fd_srp_only` repeats the comparison above (row-normalised, the same FD steps, tolerance 1e-6) for a Moon-centred model with a massless Sun, so that rows p..k of A are driven by SRP alone.

### 7.5 Propagated Φ vs FD of final states

Use the Moon data with SRP over 20 days.
- Propagate the 42-state with Φ(0) = I.
- **Consistency check first.** The first 6 entries must match the 6-state run to ≤ 1e-7 relative. The step sequence differs; observed 1.4e-8 km in p.
- **FD.** For each column j, propagate x0 ± δ_j e_j with δp = 1e-6·p₀ (≈ 5.8e-3 km) and δ = 1e-7 for the others, then take central differences.
- **Assertion.** Form Φ̃ = diag(1/p_f,1,1,1,1,1)·Φ·diag(p₀,1,1,1,1,1). Require max_j |Φ̃_ij − Φ̃^FD_ij| ≤ 1e-4 · max_j |Φ̃^FD_ij| for each row. Observed 5e-7.
- **Why the scaling.** Φ entries span about 1e-3 (∂f/∂p) to 7e5 km (∂p/∂g), so an unscaled elementwise rtol is meaningless. The rtol of 1e-2 used in `solve_rv.cpp` would also pass but is needlessly loose.

### 7.6 Φ_MEE vs the Cartesian STM (S = 0)

Use the Moon data without SRP: `solve_ivpstm_mee` and `solve_ivpstm_rv`. With SRP the same comparison holds, since the Cartesian STM includes the SRP gradient.

$$
J_0 = J_{\text{cart}\leftarrow\text{mee}}(\boldsymbol x_0), \qquad
J_f = J_{\text{cart}\leftarrow\text{mee}}(\boldsymbol x_f), \qquad
\Phi_\text{map} = J_f^{-1}\,\Phi_\text{cart}(t_f)\,J_0
$$

$J_0$ and $J_f$ come from 5.7 (analytic) or from finite differences of `mee2cart`. Use a linear
solve, not an explicit inverse.

- Assert the same row-normalised, p-scaled metric as 7.5: ≤ 1e-6. Observed 3e-10.
- J0 and Jf may be computed by central FD of mee2cart instead (δ as in 7.4). Then use a tolerance of 1e-5.
- Take x_f from the MEE propagation. cart2mee(rv_f) differs by about 1e-12, which is negligible.

### 7.7 Integrator tolerances for the MEE state

GSL's `gsl_odeiv2_control_y_new(atol, rtol)` uses the desired error level D_i = atol + rtol·|y_i| for every component, including all 36 STM entries, and rejects a step when any error estimate exceeds D_i by more than 10% (see [Numerical integration](dynamics_model.md#numerical-integration)).

How the state scales:
- **p** is O(1e3–1e4) km, so it is controlled by rtol. With rtol = 1e-14, that is about 6e-11 km.
- **f, g, h, k** are O(1), so they are controlled by atol. This is about 1e-12, which corresponds to roughly p·1e-12 ≈ 6e-9 km of position.
- **L** grows without bound: about 22 rad after 20 days here, and about 6e2 rad after 100 revolutions. Its effective tolerance is atol + rtol·|L|, which loosens linearly with the number of revolutions. With rtol = 1e-14 this stays below 1e-11 rad for 100 revolutions, which is negligible.

Recommendations:
- **Keep the defaults** (atol 1e-12, rtol 1e-14) for tests 7.3–7.6. They are no tighter than machine precision for any component: the spacing of L ≈ 22 is 3.6e-15.
- **For long arcs,** use canonical units with DU close to p₀ (`SpiceParameters(..., DU=p0)`), so that all six elements are O(1) and a single atol means the same for all of them. The solvers take one atol and one rtol for all components.
- **Units.** `SpiceParameters` derives TU and the scaled GMs from DU, so the Cartesian and MEE solvers always see consistent units. The SRP scaling is the same in both.

---

## References

- {cite:t}`walker1985`, with the erratum {cite:p}`walker1986erratum`.
- {cite:t}`betts2010`. Low-thrust orbit-transfer example: MEE dynamics with q ≡ w.
- {cite:t}`battin1999`. The F(q) third-body form.
