/**
 * Cartesian equations of motion, with STM: central body, third bodies, SRP and
 * central-body spherical harmonics (see add_perturbations).
 */

#include <cmath>
#include "model.h"
#include "gravity_gradient.h"

namespace campanella {
namespace detail {

/**
 * Cartesian RHS and, if A is not null, its Jacobian A = [[0, I], [G, 0]]
 *
 * @param model dynamics model
 * @param ws scratch space
 * @param t time, in TU
 * @param x state [x, y, z, vx, vy, vz], in DU, DU/TU
 * @param xdot time derivative of x, to be computed
 * @param A 6-by-6 Jacobian d(xdot)/dx to be computed, or nullptr to skip it
 * @return as add_perturbations; also RHS_EDOM (ws.fault = nonfinite) if xdot or A is not finite
 */
int rv_evaluate(const Model& model, Workspace& ws, double t, const double x[6],
                double xdot[6], double A[6][6]) {
    ws.fault = Fault::none;
    const double r[3] = {x[0], x[1], x[2]};
    const double d = std::sqrt((r[0]*r[0]) + (r[1]*r[1]) + (r[2]*r[2]));
    const double d3 = d*d*d;
    const double mu = model.mus[0];

    // central body, then the perturbations
    double a[3] = {-mu*r[0] / d3, -mu*r[1] / d3, -mu*r[2] / d3};
    double G[3][3];
    if (A) {
        gravity_gradient_center(r, mu, G);
    }
    int status = add_perturbations(model, ws, t, r, a, A ? G : nullptr);
    if (status != RHS_SUCCESS) {
        return status;
    }
    xdot[0] = x[3];
    xdot[1] = x[4];
    xdot[2] = x[5];
    xdot[3] = a[0];
    xdot[4] = a[1];
    xdot[5] = a[2];
    // e.g. at the center of a body: recoverable, GSL retries with a smaller step
    if (check_finite(ws, xdot, 6) != RHS_SUCCESS) {
        return RHS_EDOM;
    }

    if (A) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                A[i][j] = 0.0;
            }
        }
        for (int i = 0; i < 3; i++) {
            A[i][3 + i] = 1.0;
            for (int j = 0; j < 3; j++) {
                A[3 + i][j] = G[i][j];
            }
        }
        return check_finite(ws, &A[0][0], 36);
    }
    return RHS_SUCCESS;
}


/**
 * Cartesian equations of motion, compatible for GSL
 *
 * @param t time, in TU
 * @param x state [x, y, z, vx, vy, vz]
 * @param xdot time derivative of state
 * @param params pointer to Context
 */
int rhs_rv(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    return rv_evaluate(*ctx->model, *ctx->ws, t, x, xdot, nullptr);
}


/**
 * Cartesian equations of motion with STM, compatible for GSL
 *
 * Phi_dot = A Phi with A = [[0, I], [G, 0]]: the top rows of Phi_dot are the bottom rows of Phi.
 *
 * @param t time, in TU
 * @param x state [x, y, z, vx, vy, vz] followed by STM, row-major (42 elements)
 * @param xdot time derivative of state and STM
 * @param params pointer to Context
 */
int rhsstm_rv(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    double A[6][6];
    int status = rv_evaluate(*ctx->model, *ctx->ws, t, x, xdot, A);
    if (status != RHS_SUCCESS) {
        return status;
    }
    const double* Phi = x + 6;
    double* Phi_dot = xdot + 6;
    for (int j = 0; j < 18; j++) {
        Phi_dot[j] = Phi[18 + j];
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            Phi_dot[18 + 6*i + j] = A[3 + i][0]*Phi[j] + A[3 + i][1]*Phi[6 + j] + A[3 + i][2]*Phi[12 + j];
        }
    }
    return RHS_SUCCESS;
}

}  // namespace detail
}  // namespace campanella
