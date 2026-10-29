#ifndef EOMS_GVE_H
#define EOMS_GVE_H

// Gauss variational equations in modified equinoctial elements (MEE), see docs/gve_mee.md.
// params is a ParamsNbody*. BODIES[0] is the central body; SRP is skipped when A == 0.
// Return GSL_EDOM if p <= 0 or w <= 0 (GSL retries with a smaller step), GSL_FAILURE on SPICE error.

// 6 states: [p, f, g, h, k, L]
int eom_gve_NBodySRP(double t, const double p[], double pdot[], void* params);

// 42 states: [p, f, g, h, k, L] followed by the 6x6 STM, row-major. Phi_dot = A Phi.
int eom_stm_gve_NBodySRP(double t, const double p[], double pdot[], void* params);

// Evaluate xdot (6) and the Jacobian A = d(xdot)/dx (6x6) at a single (t, x).
// On a SPICE error, returns GSL_FAILURE and leaves the SPICE error flag set (caller resets).
int eval_gve_jacobian_NBodySRP(double t, const double x[], double xdot[], double A[6][6], void* params);

#endif // EOMS_GVE_H
