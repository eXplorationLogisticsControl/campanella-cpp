/**
 * Gauss variational equations (GVE) in modified equinoctial elements (MEE),
 * with the perturbations of add_perturbations, and analytic Jacobian for the STM.
 *
 * Notation and equation numbers follow docs/gve_mee.md. The perturbing acceleration is
 * exactly the one eom_rv adds to the central-body term (third bodies, SRP and the optional
 * central-body spherical harmonics, docs/spherical_harmonics.md).
 */

#include <cmath>
#include "model.h"
#include "mee.h"

namespace campanella {
namespace detail {

namespace {

double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

}  // namespace

/**
 * Evaluate the GVE right-hand side and, if A is not null, its Jacobian (sec. 5.6)
 *
 * @param model dynamics model
 * @param ws scratch space
 * @param t time, in TU
 * @param x MEE state [p, f, g, h, k, L]
 * @param xdot time derivative of x, to be computed
 * @param A 6-by-6 Jacobian d(xdot)/dx to be computed, or nullptr to skip it
 * @return RHS_SUCCESS; RHS_EDOM if p <= 0 or w <= 0, below sh_rho_min * R, or if xdot or A is not finite;
 *         RHS_EBADFUNC on SPICE error. RHS_EDOM is recoverable: gsl_odeiv2_evolve_apply halves
 *         the step and retries, whereas RHS_EBADFUNC aborts the integration.
 */
int mee_evaluate(const Model& model, Workspace& ws, double t, const double x[6],
                 double xdot[6], double A[6][6]) {
    ws.fault = Fault::none;
    const double mu = model.mus[0];
    const bool compute_jacobian = (A != nullptr);

    // 1. elements and auxiliary quantities (sec. 1.2)
    const double p = x[0], f = x[1], g = x[2], h = x[3], k = x[4], L = x[5];
    if (!(p > 0)) {
        ws.fault = Fault::mee_domain;
        return RHS_EDOM;
    }
    const double cL = std::cos(L);
    const double sL = std::sin(L);
    const double w = 1 + f*cL + g*sL;
    if (!(w > 0)) {
        ws.fault = Fault::mee_domain;
        return RHS_EDOM;
    }
    const double s2 = 1 + h*h + k*k;
    const double Z = h*sL - k*cL;
    const double wL = -f*sL + g*cL;     // dw/dL
    const double ZL = h*cL + k*sL;      // dZ/dL
    const double sig = std::sqrt(p / mu);
    const double n0 = std::sqrt(mu / (p*p*p));
    const double r = p / w;

    // 2. RTN basis (3.1) and position
    double rhat[3], that[3], nhat[3];
    mee_rtn_basis(h, k, cL, sL, rhat, that, nhat);
    double rvec[3] = {r*rhat[0], r*rhat[1], r*rhat[2]};

    // 3-4. perturbing acceleration a_I and its gradient G (third bodies, SRP, SH; sec. 5.5)
    double aI[3] = {0.0, 0.0, 0.0};
    double G[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    int status = add_perturbations(model, ws, t, rvec, aI, compute_jacobian ? G : nullptr);
    if (status != RHS_SUCCESS) {
        return status;
    }

    // 5. RTN acceleration and GVE (4.1)
    const double aR = dot3(rhat, aI);
    const double aT = dot3(that, aI);
    const double aN = dot3(nhat, aI);

    double B[6][3] = {{0.0}};
    B[0][1] = 2*p*sig / w;
    B[1][0] =  sig*sL;
    B[1][1] =  sig*((w + 1)*cL + f) / w;
    B[1][2] = -sig*g*Z / w;
    B[2][0] = -sig*cL;
    B[2][1] =  sig*((w + 1)*sL + g) / w;
    B[2][2] =  sig*f*Z / w;
    B[3][2] =  sig*s2*cL / (2*w);
    B[4][2] =  sig*s2*sL / (2*w);
    B[5][2] =  sig*Z / w;

    const double b5 = n0*w*w;   // Keplerian part, b = [0, 0, 0, 0, 0, n0 w^2]
    for (int i = 0; i < 6; i++) {
        xdot[i] = B[i][0]*aR + B[i][1]*aT + B[i][2]*aN;
    }
    xdot[5] += b5;
    if (check_finite(ws, xdot, 6) != RHS_SUCCESS) {
        return RHS_EDOM;
    }

    if (!compute_jacobian) {
        return RHS_SUCCESS;
    }

    // 6. Jacobian A = A_b + A_B + B Atilde (5.0)
    for (int i = 0; i < 6; i++) {
        for (int m = 0; m < 6; m++) {
            A[i][m] = 0.0;
        }
    }

    // A_b: Keplerian part (5.1)
    A[5][0] = -(3/(2*p)) * b5;
    A[5][1] = 2*n0*w*cL;
    A[5][2] = 2*n0*w*sL;
    A[5][5] = 2*n0*w*wL;

    // dB[i][j][m] = d(B[i][j]) / d(x[m]) (5.2); entries not set are zero
    enum { IP = 0, IF = 1, IG = 2, IH = 3, IK = 4, IL = 5 };
    double dB[6][3][6] = {{{0.0}}};
    const double w2 = w*w;
    // d/dp: B[0][1] ~ p^(3/2), all others ~ p^(1/2)
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            dB[i][j][IP] = B[i][j] / (2*p);
        }
    }
    dB[0][1][IP] = 3*B[0][1] / (2*p);
    // B[0][1] = 2 p sig / w
    dB[0][1][IF] = -2*p*sig*cL / w2;
    dB[0][1][IG] = -2*p*sig*sL / w2;
    dB[0][1][IL] = -2*p*sig*wL / w2;
    // B[1][0] = sig sL
    dB[1][0][IL] = sig*cL;
    // B[1][1] = sig (cL + (cL + f)/w)
    dB[1][1][IF] = sig*(1/w - (cL + f)*cL/w2);
    dB[1][1][IG] = -sig*(cL + f)*sL/w2;
    dB[1][1][IL] = sig*(-sL - sL/w - (cL + f)*wL/w2);
    // B[1][2] = -sig g Z / w
    dB[1][2][IF] = sig*g*Z*cL/w2;
    dB[1][2][IG] = -sig*Z/w + sig*g*Z*sL/w2;
    dB[1][2][IH] = -sig*g*sL/w;
    dB[1][2][IK] = sig*g*cL/w;
    dB[1][2][IL] = -sig*g*(ZL/w - Z*wL/w2);
    // B[2][0] = -sig cL
    dB[2][0][IL] = sig*sL;
    // B[2][1] = sig (sL + (sL + g)/w)
    dB[2][1][IF] = -sig*(sL + g)*cL/w2;
    dB[2][1][IG] = sig*(1/w - (sL + g)*sL/w2);
    dB[2][1][IL] = sig*(cL + cL/w - (sL + g)*wL/w2);
    // B[2][2] = sig f Z / w
    dB[2][2][IF] = sig*Z/w - sig*f*Z*cL/w2;
    dB[2][2][IG] = -sig*f*Z*sL/w2;
    dB[2][2][IH] = sig*f*sL/w;
    dB[2][2][IK] = -sig*f*cL/w;
    dB[2][2][IL] = sig*f*(ZL/w - Z*wL/w2);
    // B[3][2] = sig s2 cL / (2w)
    dB[3][2][IF] = -sig*s2*cL*cL/(2*w2);
    dB[3][2][IG] = -sig*s2*cL*sL/(2*w2);
    dB[3][2][IH] = sig*h*cL/w;
    dB[3][2][IK] = sig*k*cL/w;
    dB[3][2][IL] = (sig*s2/2)*(-sL/w - cL*wL/w2);
    // B[4][2] = sig s2 sL / (2w)
    dB[4][2][IF] = -sig*s2*sL*cL/(2*w2);
    dB[4][2][IG] = -sig*s2*sL*sL/(2*w2);
    dB[4][2][IH] = sig*h*sL/w;
    dB[4][2][IK] = sig*k*sL/w;
    dB[4][2][IL] = (sig*s2/2)*(cL/w - sL*wL/w2);
    // B[5][2] = sig Z / w
    dB[5][2][IF] = -sig*Z*cL/w2;
    dB[5][2][IG] = -sig*Z*sL/w2;
    dB[5][2][IH] = sig*sL/w;
    dB[5][2][IK] = -sig*cL/w;
    dB[5][2][IL] = sig*(ZL/w - Z*wL/w2);

    // A_B: B differentiated at fixed a_rtn
    for (int i = 0; i < 6; i++) {
        for (int m = 0; m < 6; m++) {
            A[i][m] += dB[i][0][m]*aR + dB[i][1][m]*aT + dB[i][2][m]*aN;
        }
    }

    // G_rtn = C^T G C, with C = [rhat that nhat] (5.5)
    const double* C_cols[3] = {rhat, that, nhat};
    double GC[3][3];   // G C
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            GC[i][j] = G[i][0]*C_cols[j][0] + G[i][1]*C_cols[j][1] + G[i][2]*C_cols[j][2];
        }
    }
    double G_rtn[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            G_rtn[i][j] = C_cols[i][0]*GC[0][j] + C_cols[i][1]*GC[1][j] + C_cols[i][2]*GC[2][j];
        }
    }

    // D_rtn = C^T dr_vec/dx (5.3)
    double D_rtn[3][6] = {{0.0}};
    D_rtn[0][IP] = 1/w;
    D_rtn[0][IF] = -r*cL/w;
    D_rtn[0][IG] = -r*sL/w;
    D_rtn[0][IL] = r*(f*sL - g*cL)/w;
    D_rtn[1][IH] = 2*r*k/s2;
    D_rtn[1][IK] = -2*r*h/s2;
    D_rtn[1][IL] = r;
    D_rtn[2][IH] = 2*r*sL/s2;
    D_rtn[2][IK] = -2*r*cL/s2;

    // F = (dC^T/dx) a_I, rotation of the RTN basis (5.4)
    double F[3][6] = {{0.0}};
    F[0][IH] =  (2/s2)*( k*aT + sL*aN);
    F[0][IK] =  (2/s2)*(-h*aT - cL*aN);
    F[0][IL] =  aT;
    F[1][IH] =  (2/s2)*(-k*aR + cL*aN);
    F[1][IK] =  (2/s2)*( h*aR + sL*aN);
    F[1][IL] = -aR;
    F[2][IH] = -(2/s2)*(sL*aR + cL*aT);
    F[2][IK] =  (2/s2)*(cL*aR - sL*aT);

    // Atilde = G_rtn D_rtn + F, then A += B Atilde
    double Atilde[3][6];
    for (int j = 0; j < 3; j++) {
        for (int m = 0; m < 6; m++) {
            Atilde[j][m] = G_rtn[j][0]*D_rtn[0][m] + G_rtn[j][1]*D_rtn[1][m]
                         + G_rtn[j][2]*D_rtn[2][m] + F[j][m];
        }
    }
    for (int i = 0; i < 6; i++) {
        for (int m = 0; m < 6; m++) {
            A[i][m] += B[i][0]*Atilde[0][m] + B[i][1]*Atilde[1][m] + B[i][2]*Atilde[2][m];
        }
    }
    return check_finite(ws, &A[0][0], 36);
}



/**
 * GVE in MEE, compatible for GSL
 *
 * @param t time, in TU
 * @param x MEE state [p, f, g, h, k, L]
 * @param xdot time derivative of state
 * @param params pointer to Context
 */
int rhs_mee(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    return mee_evaluate(*ctx->model, *ctx->ws, t, x, xdot, nullptr);
}


/**
 * GVE in MEE with STM, compatible for GSL
 *
 * @param t time, in TU
 * @param x MEE state [p, f, g, h, k, L] followed by STM, row-major (42 elements)
 * @param xdot time derivative of state and STM
 * @param params pointer to Context
 */
int rhsstm_mee(double t, const double x[], double xdot[], void* params) {
    Context* ctx = static_cast<Context*>(params);
    double A[6][6];
    int status = mee_evaluate(*ctx->model, *ctx->ws, t, x, xdot, A);
    if (status != RHS_SUCCESS) {
        return status;
    }

    // derivatives of STM, row-major: Phi_dot = A Phi
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            double sum = 0.0;
            for (int m = 0; m < 6; m++) {
                sum += A[i][m] * x[6 + 6*m + j];
            }
            xdot[6 + 6*i + j] = sum;
        }
    }
    return RHS_SUCCESS;
}

}  // namespace detail
}  // namespace campanella
