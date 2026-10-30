# Gauss variational equations in modified equinoctial elements (MEE), with analytic STM

This document specifies the MEE dynamics model and its 6x6 state-transition matrix (STM). It covers:
- the conversions `mee2cart` and `cart2mee`
- the Gauss variational equations (GVE) driven by the same perturbations as `eom_rv`
- the analytic Jacobian used to propagate the STM

Every formula below was checked numerically. Section 6 has the results. The physics must match
`eom_rv` exactly: both add the perturbations of `add_perturbations` (`src/eoms/forces.cpp`) to the
central-body term. The first NAIF id (the Moon here) is the central body, the other ids are third bodies
(`thirdbody_accel`), and SRP comes from `src/perturbations/SRP.cpp`. In the text below, `GM[i]` and
`BODIES[i]` are `SpiceParameters.mus[i]` and `naif_ids[i]`.

**Indexing.** All vectors and matrices are 0-based, as in C++.
- State: `x[0..5] = [p, f, g, h, k, L]`.
- RTN acceleration: `a_rtn[0..2] = [a_R, a_T, a_N]`.
- `B[i][j]`: row `i` is the element, column `j` is the R/T/N component.

**Name clash.** The Jacobian is called **A**, as in `Phi_dot = A Phi`. The spacecraft area (`srp_A` in
`SpiceParameters`) is written **S** in this document.

---

## 1. Definitions

### 1.1 Elements

This document uses the prograde MEE of Walker, Ireland & Owens (1985), with retrograde factor I = +1.
In terms of the classical elements (a, e, i, Ω, ω, ν):

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

The names in brackets are the suggested C++ variable names. These are the only intermediates used anywhere in this document.

| symbol | definition | note |
|---|---|---|
| cL, sL | cos L, sin L | |
| w | 1 + f cL + g sL | Betts writes this as q. Here q is reserved for Battin's q in the third-body routine |
| s² [`s2`] | 1 + h² + k² | |
| α² [`alpha2`] | h² − k² | only used in the component form (2.2b) |
| r | p / w | orbit radius. Requires w > 0 |
| σ [`sig`] | √(p/μ) | |
| n₀ [`n0`] | √(μ/p³) | |
| Z | h sL − k cL | |
| w_L [`wL`] | ∂w/∂L = −f sL + g cL | |
| Z_L [`ZL`] | ∂Z/∂L = h cL + k sL | |

μ = `mus[0]` in DU³/TU². The time t is in TU. The ephemeris epoch is et = et0 + t·TU, exactly as in `eom_rv`.

### 1.3 Equinoctial frame

The equinoctial unit vectors are f̂, ĝ, ŵ, expressed in the integration frame (e.g. J2000):

```
f̂ = (1/s²) [ 1 − k² + h²,   2 h k,        −2 k        ]
ĝ = (1/s²) [ 2 h k,         1 + k² − h²,   2 h         ]
ŵ = (1/s²) [ 2 k,           −2 h,          1 − h² − k² ]
```

(f̂, ĝ, ŵ) is right-handed and orthonormal for all real h and k: ŵ = f̂ × ĝ. This was verified symbolically.
- f̂ and ĝ span the orbit plane.
- f̂ is the direction from which the longitudes ϖ and L are measured.

**Derivatives of the frame.** These were derived with sympy and are used in Section 5:

```
∂f̂/∂h = (2/s²) ( k ĝ )             ∂f̂/∂k = −(2/s²) ( h ĝ + ŵ )
∂ĝ/∂h = (2/s²) ( ŵ − k f̂ )         ∂ĝ/∂k =  (2/s²) ( h f̂ )
∂ŵ/∂h = −(2/s²) ĝ                  ∂ŵ/∂k =  (2/s²) f̂
```

The frame does not depend on p, f, g or L.

---

## 2. Conversions

### 2.1 mee2cart(x, μ) → (r, v)

Vector form (recommended):

```
r_vec = r ( cL f̂ + sL ĝ )                          (2.1a)
v_vec = √(μ/p) [ −(sL + g) f̂ + (cL + f) ĝ ]          (2.1b)
```

The equivalent component form (Walker/Betts) was checked against (2.1) to round-off:

```
r_x = (r/s²) ( cL + α² cL + 2hk sL )
r_y = (r/s²) ( sL − α² sL + 2hk cL )
r_z = (2r/s²) ( h sL − k cL )
v_x = −(1/s²)√(μ/p) ( sL + α² sL − 2hk cL + g − 2fhk + α² g )
v_y = −(1/s²)√(μ/p) ( −cL + α² cL + 2hk sL − f + 2ghk + α² f )
v_z =  (2/s²)√(μ/p) ( h cL + k sL + f h + g k )                    (2.2b)
```

Preconditions:
- **p > 0 and w > 0.** w ≤ 0 can only occur for hyperbolic elements beyond the asymptote, where r would be negative.
  - `mee2cart` throws `std::domain_error`.
  - The EOM returns `GSL_EDOM`. Do not use `GSL_EBADFUNC`: `gsl_odeiv2_evolve_apply` aborts on it (and on `GSL_EFAULT`), whereas for any other nonzero status it halves the step and retries. An out-of-domain trial RK stage, e.g. a large first step near i = 180°, therefore only shrinks the step.
  - `eval_mee` throws `std::domain_error` (`ValueError` in Python); the solvers throw it if the step cannot be shrunk any further.
- **L is not wrapped.** Any real L is accepted.

The angular momentum is H = r × v = √(μp) ŵ, so ŵ is always along H. This is why n̂ = ŵ in Section 3.

### 2.2 cart2mee(r, v, μ) → x

1. **Angular momentum.** H = r × v and |H| = ‖H‖. If |H| = 0 (rectilinear orbit), throw `std::domain_error`.
2. **Semi-latus rectum.** p = |H|² / μ.
3. **h and k from ŵ = H/|H|.** From the third component of ŵ in 1.3, 1 + ŵ_z = 2/s². Combined with ŵ_x = 2k/s² and ŵ_y = −2h/s², this gives:
   ```
   k =  ŵ_x / (1 + ŵ_z) =  H_x / (|H| + H_z)
   h = −ŵ_y / (1 + ŵ_z) = −H_y / (|H| + H_z)
   ```
   The denominator is |H|(1 + cos i), which vanishes at i = 180°. This is the singularity of I = +1 elements.
   - **Rule:** if `|H| + H_z <= 1e-12 * |H|`, throw `std::domain_error` ("retrograde equatorial orbit; MEE with I=+1 singular").
   - This trips within about 1.4e-6 rad of i = 180°, where |h| and |k| would be about 1.4e6.
   - Closer to 180° than about 0.1°, accuracy degrades. At i = 179.9° (|h| ≈ 810), the MEE→Cartesian→MEE re-trip error in h is 5e-8 absolute, or 6e-11 relative.
4. **Build f̂ and ĝ from (h, k)** with the formulas in 1.3. Do not use any other frame construction: f and g must be consistent with `mee2cart`.
5. **Eccentricity vector.** e = (v × H)/μ − r/‖r‖.

   In the orbit plane, e = e(cos ϖ f̂ + sin ϖ ĝ), because ϖ is measured from f̂. Therefore:
   ```
   f = e · f̂ ,   g = e · ĝ
   ```
6. **True longitude.** r̂ = cos L f̂ + sin L ĝ, so:
   ```
   L = atan2( r · ĝ , r · f̂ )        ∈ (−π, π]
   ```

### 2.3 Edge cases

| case | behaviour |
|---|---|
| e = 0 | f = g = 0. Nothing is singular. L is still defined because it does not depend on ω |
| i = 0 | h = k = 0. Then f̂ = x̂ and ĝ = ŷ, and L is the true longitude from x̂ |
| e ≥ 1 | The formulas hold for p > 0 and w > 0. Tested with e = 1.5 |
| i → 180° | Singular. See step 3 in 2.2 |
| H = 0 | Throw |

### 2.4 Wrapping L

- **`cart2mee`** returns L ∈ (−π, π]. This is the range of `std::atan2`.
- **The integrator** treats L as a continuous, unwrapped variable. It grows by 2π each revolution.
- **The EOM** must never wrap L. Everything uses cL and sL, so wrapping is unnecessary.
- **Comparing L values** (tests, round trips): compare wrap(L₁ − L₂) = atan2(sin(L₁ − L₂), cos(L₁ − L₂)), never L₁ − L₂ directly.
- **`mee2cart`** accepts any L.

---

## 3. RTN frame

The definition is the usual one, with the unit vectors as columns of C:

```
r̂ = r_vec / ‖r_vec‖,    n̂ = H / ‖H‖,    θ̂ = n̂ × r̂,      C = [ r̂  θ̂  n̂ ]   (3x3)
a_rtn = Cᵀ a_I  =  [ r̂·a_I ,  θ̂·a_I ,  n̂·a_I ] = [a_R, a_T, a_N]
```

As explicit functions of MEE, these depend only on (h, k, L). This follows from (2.1) and H ∥ ŵ:

```
r̂ =  cL f̂ + sL ĝ = (1/s²) [ cL + α² cL + 2hk sL ,  sL − α² sL + 2hk cL ,  2 Z   ]
θ̂ = −sL f̂ + cL ĝ = (1/s²) [ −sL − α² sL + 2hk cL , cL − α² cL − 2hk sL ,  2 Z_L ]
n̂ =  ŵ           = (1/s²) [ 2k ,  −2h ,  1 − h² − k² ]                               (3.1)
```

Do not build C from v. The MEE form is exact, needs no ∂v/∂x, and it is what the Jacobian in Section 5 assumes.

The derivatives of the RTN basis follow from 1.3 and are written back in the RTN basis:

```
∂r̂/∂h = (2/s²)(  k θ̂ + sL n̂ )      ∂r̂/∂k = (2/s²)( −h θ̂ − cL n̂ )      ∂r̂/∂L =  θ̂
∂θ̂/∂h = (2/s²)( −k r̂ + cL n̂ )      ∂θ̂/∂k = (2/s²)(  h r̂ + sL n̂ )      ∂θ̂/∂L = −r̂
∂n̂/∂h = −(2/s²)( sL r̂ + cL θ̂ )     ∂n̂/∂k = (2/s²)(  cL r̂ − sL θ̂ )     ∂n̂/∂L =  0      (3.2)
```

All derivatives with respect to p, f and g are zero.

---

## 4. Gauss variational equations

```
ẋ = b(x) + B(x) a_rtn ,      a_rtn = C(x)ᵀ a_I( r_vec(x), t )                          (4.1)
```

- a_I is the sum of all third-body accelerations and the SRP acceleration, in the inertial integration frame, evaluated at r_vec(x) and et = et0 + t·TU.
- It is exactly the acceleration that `eom_rv` adds to the central term −μ r/r³ (`add_perturbations`):
  ```
  a_I = Σ_{i=1}^{N−1} thirdbody_accel(r, s_i(et), GM[i])
        + a_srp(r, et)
  ```
- Only the position part of `mee2cart` is needed to evaluate it.

**Keplerian part.** This is the only place μ = `GM[0]` enters:

```
b = [ 0, 0, 0, 0, 0,  n₀ w² ]ᵀ ,     n₀ w² = √(μ p) (w/p)²
```

**Control matrix.** B(x) is 6x3. The columns are R, T, N. All entries not listed are 0.

```
B[0][1] = 2 p σ / w
B[1][0] =  σ sL
B[1][1] =  σ ( (w+1) cL + f ) / w
B[1][2] = −σ g Z / w
B[2][0] = −σ cL
B[2][1] =  σ ( (w+1) sL + g ) / w
B[2][2] =  σ f Z / w
B[3][2] =  σ s² cL / (2w)
B[4][2] =  σ s² sL / (2w)
B[5][2] =  σ Z / w
```

**Convention and provenance.** This is Walker et al. (1985) with I = +1, in the form reproduced in Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming* (2010), with q → w. It is fixed by numerical verification, not by citation.
- **Published typos.** Published transcriptions differ in the normal-acceleration terms of ḟ and ġ: the signs of B[1][2] and B[2][2], and f vs g inside B[2][2]. Walker et al. also published an erratum (Celest. Mech. 38, 391–392, 1986).
- **Correct form.** ḟ's normal term contains −g Z/w and ġ's contains +f Z/w.
- **Negative controls in Section 6(b).** Flipping both signs, or writing g instead of f in B[2][2], produces relative errors of 1e-6 to 1 against the Cartesian truth. The form above agrees to 5e-28.

---

## 5. Jacobian A = ∂ẋ/∂x and the STM

### 5.0 Structure

a_I depends on x only through r_vec(x) and on t through the ephemerides. Its spatial gradient is G = ∂a_I/∂r (3x3). Differentiating (4.1) gives:

```
A = A_b + A_B + B · Ã                                                        (5.0)

A_b[i][m] = ∂b_i/∂x_m
A_B[i][m] = Σ_j (∂B[i][j]/∂x_m) a_rtn[j]          (B differentiated at fixed a_rtn)
Ã[j][m]   = ∂a_rtn[j]/∂x_m = (G_rtn · D_rtn)[j][m] + F[j][m]      (3x6)

G_rtn = Cᵀ G C                (3x3, gradient expressed in RTN)
D_rtn = Cᵀ ∂r_vec/∂x          (3x6, position sensitivity in RTN)
F     = (∂Cᵀ/∂x) a_I          (3x6, rotation of the RTN basis at fixed a_I)
```

- ∂/∂t terms do not enter: the STM is ∂x(t)/∂x(t₀) and the time dependence is explicit.
- The velocity never appears.

**STM layout.** Same as `eomstm_rv`:
- y[0..5] = x.
- y[6 + 6i + j] = Φ[i][j], row-major, with Φ(t₀) = I.
- The derivative is:
  ```
  ydot[6 + 6i + j] = Σ_{m=0}^{5} A[i][m] · y[6 + 6m + j]
  ```

### 5.1 A_b: the Keplerian part

Only row 5 is nonzero:

```
A_b[5][0] = −(3/(2p)) n₀ w²
A_b[5][1] =  2 n₀ w cL
A_b[5][2] =  2 n₀ w sL
A_b[5][5] =  2 n₀ w w_L
A_b[5][3] = A_b[5][4] = 0
```

### 5.2 ∂B/∂x: every nonzero entry

- **Derivatives with respect to p.** Every nonzero B[i][j] is σ times a p-independent factor, except B[0][1], which is proportional to p^{3/2}. So:
  ```
  ∂B[i][j]/∂p = B[i][j] / (2p)      for all (i,j) ≠ (0,1)
  ∂B[0][1]/∂p = 3 B[0][1] / (2p)
  ```
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

```
A_B[i][m] = ∂B[i][0]/∂x_m · a_R + ∂B[i][1]/∂x_m · a_T + ∂B[i][2]/∂x_m · a_N
```

### 5.3 D_rtn = Cᵀ ∂r_vec/∂x

From r_vec = r r̂, with r = p/w and (3.2):
- ∂r/∂p = 1/w
- ∂r/∂f = −r cL/w
- ∂r/∂g = −r sL/w
- ∂r/∂L = r(f sL − g cL)/w
- ∂r/∂h = ∂r/∂k = 0

```
            p        f           g           h            k             L
D_rtn = [ 1/w   −r cL/w    −r sL/w        0             0       r (f sL − g cL)/w ]   (R)
        [ 0        0          0       2 r k / s²   −2 r h / s²          r          ]   (T)
        [ 0        0          0       2 r sL / s²  −2 r cL / s²         0          ]   (N)
```

If the inertial form is wanted, use ∂r_vec/∂x_m = C · D_rtn[:, m].

### 5.4 F = (∂Cᵀ/∂x) a_I: the rotation of the RTN basis

From (3.2):
- F[j][m] = (∂ê_j/∂x_m) · a_I, where ê_0 = r̂, ê_1 = θ̂, ê_2 = n̂.
- The columns for p, f and g are 0.

```
F[0][3] = (2/s²)(  k a_T + sL a_N )     F[0][4] = (2/s²)( −h a_T − cL a_N )     F[0][5] =  a_T
F[1][3] = (2/s²)( −k a_R + cL a_N )     F[1][4] = (2/s²)(  h a_R + sL a_N )     F[1][5] = −a_R
F[2][3] = −(2/s²)( sL a_R + cL a_T )    F[2][4] = (2/s²)(  cL a_R − sL a_T )    F[2][5] =  0
```

### 5.5 G: the inertial gradient of the perturbations

```
G = Σ_{i=1}^{N−1} G_i  +  G_srp  +  G_sh ,      G_rtn = Cᵀ G C
```

G_sh is present only when a central-body gravity field is given (see below).

**Third body.** `thirdbody_accel` evaluates Battin's form:
- a_i = −μ_i (r + F(q_i) s_i)/‖d_i‖³, with d_i = r − s_i.
- This is algebraically identical to −μ_i (d_i/‖d_i‖³ + s_i/‖s_i‖³).
- The indirect term does not depend on r, so the exact gradient is:

```
G_i = μ_i ( 3 d_i d_iᵀ − ‖d_i‖² I₃ ) / ‖d_i‖⁵
```

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

```
G_srp = ∂a_srp/∂r = K ( I₃/R³ − 3 d dᵀ / R⁵ ) = κ ( I₃ − 3 d dᵀ / R² )
```

Notes:
- This is a repulsive inverse-square field: G_srp is trace-free, with the opposite sign to a point-mass gradient.
- `srp_gradient` (`src/perturbations/SRP.cpp`) takes d and κ. The Sun position is the one already queried for its third-body term, so SRP costs no extra ephemeris call.
- There is no shadow model, which matches the Cartesian model.
- Before the unified interface, the model took s as the solar system barycenter and scaled the flux by (R₀/R)² with R₀ = ‖s‖ instead of (AU/R)². The SRP results of section 6 and the DOP853 reference of 7.0 were computed with that model.

**Central-body spherical harmonics.** If a gravity field is given (`sh_file`, see [spherical_harmonics.md](spherical_harmonics.md)), `sh_perturbation` adds the acceleration of degrees 2..nmax to a_I. Its gradient is

```
G_sh = Tᵀ G_bf T ,      T = pxform(naif_frame, sh_frame, et)
```

where G_bf is the body-fixed gradient from the V, W recursion.
- The SH acceleration depends only on r and t, so nothing else in §5.6 changes.
- Ṫ does not enter, because the Jacobian is taken at fixed t.

### 5.6 Algorithm for one RHS + STM evaluation

1. Unpack x. Compute cL, sL, w, s², Z, w_L, Z_L, σ, n₀ and r.
   - If p ≤ 0 or w ≤ 0, return `GSL_EDOM` (recoverable: GSL retries with a smaller step; see 2.1).
2. Compute f̂, ĝ, ŵ (1.3), then r̂, θ̂, n̂ (3.1), then r_vec = r r̂.
3. Set et = et0 + t·TU. Loop over the bodies i = 1..N−1:
   - a_I += res[0..2]
   - G += G_i
   - on a SPICE error, save NAIF's message, `reset_c()`, and return `GSL_EBADFUNC`
4. SRP, if `include_srp`:
   - a_I += a_srp
   - G += G_srp
   - Central-body spherical harmonics, if given: a_I += a_sh and G += G_sh.
5. Compute a_rtn = [r̂·a_I, θ̂·a_I, n̂·a_I] and ẋ = b + B a_rtn.
6. For the STM version only:
   - G_rtn = Cᵀ G C
   - Ã = G_rtn D_rtn + F
   - A = A_b + A_B + B Ã
   - Φ̇ = A Φ

### 5.7 Optional: J_cart←mee = ∂(r_vec, v_vec)/∂x

This Jacobian is needed by the Cartesian-STM consistency test (7.6), and is useful for library users. Position columns:

```
∂r_vec/∂p = r̂ / w
∂r_vec/∂f = −(r cL/w) r̂
∂r_vec/∂g = −(r sL/w) r̂
∂r_vec/∂h = (2r/s²)(  k θ̂ + sL n̂ )
∂r_vec/∂k = (2r/s²)( −h θ̂ − cL n̂ )
∂r_vec/∂L = (r (f sL − g cL)/w) r̂ + r θ̂
```

Velocity columns, with σ' = √(μ/p) and v_vec from (2.1b):

```
∂v_vec/∂p = −v_vec / (2p)
∂v_vec/∂f =  σ' ĝ
∂v_vec/∂g = −σ' f̂
∂v_vec/∂h = (2σ'/s²) [ −k (cL+f) f̂ − k (sL+g) ĝ + (cL+f) ŵ ]
∂v_vec/∂k = (2σ'/s²) [  h (cL+f) f̂ + h (sL+g) ĝ + (sL+g) ŵ ]
∂v_vec/∂L = −σ' r̂
```

J_mee←cart is obtained as the inverse of this 6x6 matrix (LU solve). It is nonsingular whenever the MEE are defined.

---

## 6. Numerical verification (performed)

The throwaway verification scripts (not part of the repository) were:
- `mee_lib.py`: a line-by-line transcription of Sections 1–5
- `verify.py`: checks (a)–(c)
- `negctl.py`: Jacobian negative controls
- `prop.py`: the real-ephemeris check (d)

They were run with numpy, mpmath and scipy via `uv run --project . --with mpmath --with scipy`.
- **Formula-correctness checks** (a), (b), (c) ran in **mpmath with 50 significant digits**, with central finite differences (FD) of step 1e-20. Any formula error therefore shows up as an O(perturbation) discrepancy instead of being hidden by double-precision FD noise.
- **Tolerance guidance** comes from float64 re-runs.

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

**(d) Real ephemerides (DE440 via spiceypy), Moon test case, 20 days.**
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

## 7. Test plan for the C++/Python implementation

Public API (implemented; tests in `test-python/test_mee.py` and `tests/test_*.cpp`):
- `mee2cart(mee, mu)`, `cart2mee(rv, mu)` and `mee2cart_jacobian(mee, mu)` (5.7) as free functions, bound in pybind.
- `solve_ivp_mee(params, tspan, mee0)` and `solve_ivpstm_mee(params, tspan, mee0)` (STM from Φ(t₀) = I), with the same `SpiceParameters` as `solve_ivp_rv` (`include_srp` adds SRP). `tf < t0` integrates backward.
- For test 7.4: `eval_mee(params, t, x) -> (xdot, A)`.

All tests below use the kernels `naif0012.tls` and `de440.bsp` from `$SPICE`, as the existing tests do.

### 7.0 Common data

```
et0    = 740463475.5117936
state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
           0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
GM     = [4.902800582147760e+03, 398600.432896939, 132712197035.766]
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
- **Final MEE with SRP.** Use these values only as a sanity check, not as the primary assertion. With the corrected SRP model (above), the reference is the `solve_ivp_mee` result, confirmed by an independent RK4 + spiceypy Cartesian transcription to 1.1e-5 km (4th-order convergence):
  ```
  [5178.576789928225, 0.8809219032708616, -0.28942480229551987,
   0.24055734575159027, -1.2805787538758713, 21.717427888269437]
  ```
  The DOP853 value at rtol 3e-14 with the former SRP model was `[5178.768353783176, 0.88095964643724, -0.28930277538858457, 0.24069918804141774, -1.2806778961781318, 21.717565975939248]`. L travels 3.64 revolutions.
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
- Additionally assert the SRP-only gradient: compare G_srp with FD of the SRP acceleration in r, δ = 1e-3 km, relative tolerance 1e-6 (`test_jacobian_vs_fd_srp_only` uses a massless Sun).

### 7.5 Propagated Φ vs FD of final states

Use the Moon data with SRP over 20 days.
- Propagate the 42-state with Φ(0) = I.
- **Consistency check first.** The first 6 entries must match the 6-state run to ≤ 1e-7 relative. The step sequence differs; observed 1.4e-8 km in p.
- **FD.** For each column j, propagate x0 ± δ_j e_j with δp = 1e-6·p₀ (≈ 5.8e-3 km) and δ = 1e-7 for the others, then take central differences.
- **Assertion.** Form Φ̃ = diag(1/p_f,1,1,1,1,1)·Φ·diag(p₀,1,1,1,1,1). Require max_j |Φ̃_ij − Φ̃^FD_ij| ≤ 1e-4 · max_j |Φ̃^FD_ij| for each row. Observed 5e-7.
- **Why the scaling.** Φ entries span about 1e-3 (∂f/∂p) to 7e5 km (∂p/∂g), so an unscaled elementwise rtol is meaningless. The rtol of 1e-2 used in `solve_rv.cpp` would also pass but is needlessly loose.

### 7.6 Φ_MEE vs the Cartesian STM (S = 0)

Use the Moon data without SRP: `solve_ivpstm_mee` and `solve_ivpstm_rv`. With SRP the same comparison holds, since the Cartesian STM includes the SRP gradient.

```
J0 = J_cart←mee(x0),  Jf = J_cart←mee(x_f)                      (5.7, analytic; or FD of mee2cart)
Φ_map = Jf⁻¹ · Φ_cart(tf) · J0                                  (use a linear solve, not an explicit inverse)
```

- Assert the same row-normalised, p-scaled metric as 7.5: ≤ 1e-6. Observed 3e-10.
- J0 and Jf may be computed by central FD of mee2cart instead (δ as in 7.4). Then use a tolerance of 1e-5.
- Take x_f from the MEE propagation. cart2mee(rv_f) differs by about 1e-12, which is negligible.

### 7.7 Integrator tolerances for the MEE state

GSL's `gsl_odeiv2_control_y_new(atol, rtol)` accepts a step when |err_i| ≤ atol + rtol·|y_i| for every component, including all 36 STM entries.

How the state scales:
- **p** is O(1e3–1e4) km, so it is controlled by rtol. With rtol = 1e-14, that is about 6e-11 km.
- **f, g, h, k** are O(1), so they are controlled by atol. This is about 1e-12, which corresponds to roughly p·1e-12 ≈ 6e-9 km of position.
- **L** grows without bound: about 22 rad after 20 days here, and about 6e2 rad after 100 revolutions. Its effective tolerance is atol + rtol·|L|, which loosens linearly with the number of revolutions. With rtol = 1e-14 this stays below 1e-11 rad for 100 revolutions, which is negligible.

Recommendations:
- **Keep the defaults** (atol 1e-12, rtol 1e-14) for tests 7.3–7.6. They are no tighter than machine precision for any component: the spacing of L ≈ 22 is 3.6e-15.
- **For production with long arcs,** consider `gsl_odeiv2_control_scaled_new` with `scale_abs = {1/p₀ … }`, or simply pass p in a scaled DU (e.g. DU = p₀), so that all six elements are O(1). Then a single atol is meaningful.
- **Units.** `SpiceParameters` derives TU and the scaled GMs from DU, so the Cartesian and MEE solvers always see consistent units. The SRP scaling is the same in both.

### 7.8 Things the engineer must be careful about

1. **Do not wrap L** inside the EOM or the integrator. Only `cart2mee` returns a wrapped L. Tests must compare wrap(ΔL).
2. **Build C from (h, k, L) via (3.1), not from v.** The Jacobian in Section 5 assumes this. With this choice it needs no ∂v/∂x.
3. **The B[1][2] and B[2][2] normal terms** are −σgZ/w and +σfZ/w. Do not copy them from a reference without checking against 7.3.
4. **The p-column of ∂B/∂x** uses 3/(2p) for B[0][1] and 1/(2p) for every other entry.
5. **The third-body gradient** takes the DU-scaled body position, not the raw ephemeris output in km.
6. **The SRP gradient has the repulsive sign:** κ(I − 3 d̂ d̂ᵀ).
7. **The Moon test state is retrograde with respect to J2000** (i = 128.5°, h ≈ 1.98). This is fine for I = +1. However, if a trajectory approaches i = 180°, then h, k → ∞. The EOM has no singularity check of its own: it only returns `GSL_EDOM` for p ≤ 0 or w ≤ 0, which makes GSL retry with a smaller step (2.1), and accuracy degrades as h, k grow. A regression test integrates i = 179.9° for 30 days. Document the limitation.
8. **SPICE error handling.** After every ephemeris call, `add_perturbations` saves NAIF's message, calls `reset_c()` and returns `GSL_EBADFUNC`, which stops GSL at once (any other status would make it retry with ever smaller steps). The solvers then throw `std::runtime_error` with the message.
9. **Evaluate the ephemerides once per RHS call.** et is the same for every body, and the third-body and SRP calls share it. The Jacobian reuses a_I and the same positions, so no extra SPICE calls are needed for the STM.

---

### References

- M. J. H. Walker, B. Ireland, J. Owens, "A set of modified equinoctial orbit elements", *Celestial Mechanics* 36, 409–419 (1985). Erratum: *Celestial Mechanics* 38, 391–392 (1986).
- J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed., SIAM (2010). Low-thrust orbit-transfer example: MEE dynamics with q ≡ w.
- R. H. Battin, *An Introduction to the Mathematics and Methods of Astrodynamics*, AIAA (1999). The F(q) third-body form.
