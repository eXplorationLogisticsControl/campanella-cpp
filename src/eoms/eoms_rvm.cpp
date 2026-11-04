/**
 * Cartesian + mass equations of motion under a zero-order-hold control, with STM and control sensitivity.
 *
 *   rdot = v,  vdot = a_grav(r, t) + a_srp(r, t, m) + (c1 / m) tau,  mdot = -c2 Gamma
 *
 * a_srp is proportional to 1/m. Its gradient d(a_srp)/dr is computed from the coefficient of the state
 * mass, so the block G of A needs no further scaling.
 */

#include "model.h"
#include "augment.h"
#include "perturbations/SRP.h"

namespace campanella {
namespace detail {

/**
 * Cartesian + mass RHS and, if not null, its Jacobians
 *
 *   A = [[0, I, 0], [G, 0, -(a_thrust + a_srp) / m], [0, 0, 0]]
 *   B = [[0, 0], [(c1 / m) I, 0], [0, -c2]]
 *
 * @param model dynamics model
 * @param control zero-order-hold control
 * @param ws scratch space
 * @param t time, in TU
 * @param x state [x, y, z, vx, vy, vz, m], in DU, DU/TU and mass units
 * @param xdot time derivative of x, to be computed
 * @param A 7-by-7 Jacobian d(xdot)/dx to be computed, or nullptr to skip it
 * @param B 7-by-4 Jacobian d(xdot)/du to be computed, or nullptr to skip it
 * @return as rv_evaluate; also RHS_EDOM if m is not > 0 or is not finite (see check_mass), and with
 *         ws.fault = nonfinite if xdot, A or B is not finite
 */
int rvm_evaluate(const Model& model, const ZohControl& control, Workspace& ws, double t,
                 const double x[7], double xdot[7], double A[7][7], double B[7][4]) {
    const double m = x[6];
    if (check_mass(ws, m) != RHS_SUCCESS) {
        return RHS_EDOM;
    }

    double srp_k = 0.0;   // ignored if SRP is off
    if (model.idx_sun >= 0) {
        const SpiceOptions& o = model.options;
        srp_k = srp_coefficient(o.srp_A, control.mass_unit_kg*m, o.srp_rhoS, o.srp_rhoD, model.DU, model.TU);
    }

    double A6[6][6];
    double a_srp[3];
    int status = rv_evaluate_ext(model, ws, t, x, srp_k, xdot, A ? A6 : nullptr, A ? a_srp : nullptr);
    if (status != RHS_SUCCESS) {
        return status;
    }

    const double c1_m = control.c1 / m;
    const double a_thrust[3] = {c1_m*control.u[0], c1_m*control.u[1], c1_m*control.u[2]};
    for (int i = 0; i < 3; i++) {
        xdot[3 + i] += a_thrust[i];
    }
    xdot[6] = -control.c2*control.u[3];
    if (check_finite(ws, xdot, 7) != RHS_SUCCESS) {
        return RHS_EDOM;
    }

    if (A) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                A[i][j] = A6[i][j];
            }
            A[i][6] = 0.0;
        }
        for (int j = 0; j < 7; j++) {
            A[6][j] = 0.0;
        }
        for (int i = 0; i < 3; i++) {
            A[3 + i][6] = -(a_thrust[i] + a_srp[i]) / m;
        }
        if (check_finite(ws, &A[0][0], 49) != RHS_SUCCESS) {
            return RHS_EDOM;
        }
    }

    if (B) {
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 4; j++) {
                B[i][j] = 0.0;
            }
        }
        for (int i = 0; i < 3; i++) {
            B[3 + i][i] = c1_m;
        }
        B[6][3] = -control.c2;
        return check_finite(ws, &B[0][0], 28);
    }
    return RHS_SUCCESS;
}


/**
 * Cartesian + mass equations of motion, compatible for GSL
 *
 * @param t time, in TU
 * @param x state [x, y, z, vx, vy, vz, m]
 * @param xdot time derivative of state
 * @param params pointer to Context, with control set
 */
int rhs_rvm(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    return rvm_evaluate(*ctx->model, *ctx->control, *ctx->ws, t, x, xdot, nullptr, nullptr);
}


/**
 * Cartesian + mass equations of motion with STM and control sensitivity, compatible for GSL
 *
 * @param t time, in TU
 * @param x state followed by Phi and Psi, row-major (84 elements, see augment.h)
 * @param xdot time derivative of x
 * @param params pointer to Context, with control set
 */
int rhsstm_rvm(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    double A[7][7];
    double B[7][4];
    int status = rvm_evaluate(*ctx->model, *ctx->control, *ctx->ws, t, x, xdot, A, B);
    if (status != RHS_SUCCESS) {
        return status;
    }
    augment_rhs(A, B, x, xdot);
    // a non-finite Phi or Psi on input, which the evaluator does not see
    return check_finite(*ctx->ws, xdot + 7, 77);
}

}  // namespace detail
}  // namespace campanella
