/**
 * MEE + mass equations of motion under a zero-order-hold control, with STM and control sensitivity.
 *
 * With x = [x_e, m], x_e = [p, f, g, h, k, L], the elements follow docs/gve_mee.md eq. (4.1):
 *
 *   xdot_e = b(x_e) + B_G(x_e) (C(x_e)^T (a_p(r, m, t) + a_c,I) + a_c,rtn),   mdot = -c2 Gamma
 *
 * where a_p is the perturbing acceleration with SRP using the state mass, a_srp ~ 1/m, and the thrust
 * acceleration c1 tau / m is a_c,rtn (RTN control) or a_c,I (inertial control). So:
 *
 * - A_e = A_b + A_B + B_G (G_rtn D_rtn + F) as in the state-only dynamics. A_B takes the total RTN
 *   acceleration. F = (dC^T/dx_e) a_I takes the inertial thrust, whose RTN components change with the
 *   state, but not the RTN thrust, whose components do not. G does not depend on the thrust.
 * - d(xdot_e)/dm: both a_srp and the thrust acceleration are proportional to 1/m, so that
 *     RTN:      -(1/m) B_G (C^T a_srp + c1 tau / m)
 *     inertial: -(1/m) B_G C^T (a_srp + c1 tau / m)
 * - d(xdot_e)/dtau = (c1/m) B_G (RTN) or (c1/m) B_G C^T (inertial), d(xdot_e)/dGamma = 0.
 * - dmdot/dx = 0 and dmdot/dGamma = -c2, dmdot/dtau = 0.
 */

#include "model.h"
#include "augment.h"
#include "perturbations/SRP.h"

namespace campanella {
namespace detail {

/**
 * Evaluate the MEE + mass right-hand side and, if not null, its Jacobians
 *
 * @param model dynamics model
 * @param control zero-order-hold control and its constants
 * @param ws scratch space
 * @param t time, in TU
 * @param x state [p, f, g, h, k, L, m]
 * @param xdot time derivative of x, to be computed
 * @param A 7-by-7 Jacobian d(xdot)/dx to be computed, or nullptr to skip it
 * @param B 7-by-4 Jacobian d(xdot)/du to be computed, or nullptr to skip it
 * @return RHS_SUCCESS; RHS_EDOM if m is not > 0 or is not finite (see check_mass); otherwise as
 *         mee_evaluate_ext; RHS_EDOM (ws.fault = nonfinite) if xdot, A or B is not finite
 */
int meem_evaluate(const Model& model, const ZohControl& control, Workspace& ws, double t,
                  const double x[7], double xdot[7], double A[7][7], double B[7][4]) {
    const double m = x[6];
    if (check_mass(ws, m) != RHS_SUCCESS) {
        return RHS_EDOM;
    }
    const bool inertial = (control.frame == ThrustFrame::inertial);
    const double a_c[3] = {control.c1*control.u[0]/m, control.c1*control.u[1]/m, control.c1*control.u[2]/m};

    double a_srp[3];
    double B_G[6][3];
    double C[3][3];   // C[i][j]: row i of column j, with columns rhat, that, nhat
    MeeExtra extra;
    if (model.idx_sun >= 0) {
        const SpiceOptions& o = model.options;
        extra.srp_k = srp_coefficient(o.srp_A, control.mass_unit_kg*m, o.srp_rhoS, o.srp_rhoD, model.DU, model.TU);
    }
    if (inertial) {
        extra.a_I = a_c;
    } else {
        extra.a_rtn = a_c;
    }
    extra.a_srp_I = a_srp;
    extra.B = B_G;
    extra.C = C;

    double A_e[6][6];
    const int status = mee_evaluate_ext(model, ws, t, x, xdot, A ? A_e : nullptr, &extra);
    if (status != RHS_SUCCESS) {
        return status;
    }
    xdot[6] = -control.c2*control.u[3];
    if (check_finite(ws, &xdot[6], 1) != RHS_SUCCESS) {
        return RHS_EDOM;
    }

    if (A) {
        // RTN acceleration proportional to 1/m
        double a_m[3];
        for (int j = 0; j < 3; j++) {
            a_m[j] = inertial ? C[0][j]*(a_srp[0] + a_c[0]) + C[1][j]*(a_srp[1] + a_c[1]) + C[2][j]*(a_srp[2] + a_c[2])
                              : C[0][j]*a_srp[0] + C[1][j]*a_srp[1] + C[2][j]*a_srp[2] + a_c[j];
        }
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                A[i][j] = A_e[i][j];
            }
            A[i][6] = -(B_G[i][0]*a_m[0] + B_G[i][1]*a_m[1] + B_G[i][2]*a_m[2]) / m;
        }
        for (int j = 0; j < 7; j++) {
            A[6][j] = 0.0;
        }
        if (check_finite(ws, &A[0][0], 49) != RHS_SUCCESS) {
            return RHS_EDOM;
        }
    }

    if (B) {
        const double c1_m = control.c1 / m;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 3; j++) {
                B[i][j] = c1_m * (inertial ? B_G[i][0]*C[j][0] + B_G[i][1]*C[j][1] + B_G[i][2]*C[j][2]
                                           : B_G[i][j]);
            }
            B[i][3] = 0.0;
        }
        for (int j = 0; j < 3; j++) {
            B[6][j] = 0.0;
        }
        B[6][3] = -control.c2;
        if (check_finite(ws, &B[0][0], 28) != RHS_SUCCESS) {
            return RHS_EDOM;
        }
    }
    return RHS_SUCCESS;
}


/**
 * MEE + mass equations of motion, compatible for GSL
 *
 * @param t time, in TU
 * @param x state [p, f, g, h, k, L, m]
 * @param xdot time derivative of state
 * @param params pointer to Context, with control set
 */
int rhs_meem(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    return meem_evaluate(*ctx->model, *ctx->control, *ctx->ws, t, x, xdot, nullptr, nullptr);
}


/**
 * MEE + mass equations of motion with STM and control sensitivity, compatible for GSL
 *
 * @param t time, in TU
 * @param x state, Phi and Psi, both row-major (84 elements, see augment.h)
 * @param xdot time derivative of x
 * @param params pointer to Context, with control set
 */
int rhsstm_meem(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    double A[7][7];
    double B[7][4];
    int status = meem_evaluate(*ctx->model, *ctx->control, *ctx->ws, t, x, xdot, A, B);
    if (status != RHS_SUCCESS) {
        return status;
    }
    augment_rhs(A, B, x, xdot);
    return RHS_SUCCESS;
}

}  // namespace detail
}  // namespace campanella
