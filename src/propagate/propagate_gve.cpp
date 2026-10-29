/*
Gauss variational equations in modified equinoctial elements
*/

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include "SpiceUsr.h"

#include "eoms/params.h"
#include "perturbations/spherical_harmonics.h"
#include "eoms/eoms_gve.h"
#include "propagate/propagate_gve.h"

/**
 * Propagate spacecraft MEE using Gauss variational equations with N-body and SRP perturbations
 *
 * @param et0 initial epoch, in ephemeris seconds past J2000 (regardless of TU)
 * @param tspan integration time span, [t0, tf]
 * @param state0 initial state, MEE in order [p,f,g,h,k,L]
 * @param GM list of gravitational parameters of each body; GM[0] is the central body
 * @param BODIES list of NAIF names of each body
 * @param A surface area of spacecraft; SRP is skipped if A == 0
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param LU length unit
 * @param TU time unit
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 * @param sh optional central-body gravity field (file, nmax, mmax, frame_bf); off if sh.file is empty
 * @return vector of tuples containing state history, (t,p,f,g,h,k,L); L is not wrapped
 */
std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp_gve(
    double et0,
    const std::vector<double>& tspan,
    const std::vector<double>& state0,
    std::vector<double> GM,
    std::vector<std::string> BODIES,
    std::string frame,
    double A,
    double m,
    double rhoS,
    double rhoD,
    double LU,
    double TU,
    double reltol,
    double abstol,
    const SphericalHarmonicsOptions& sh){
    const int state_dim = 6;      // number of states
    if (state0.size() != state_dim) {
        throw std::invalid_argument("Invalid state0 length. Expected length 6.");
    }
    if (tspan.size() != 2) {
        throw std::invalid_argument("Invalid tspan length. Expected [t0, tf].");
    }
    if (tspan[1] < tspan[0]) {
        throw std::invalid_argument("Invalid tspan: tf < t0. Only forward propagation is supported.");
    }

    // define input parameters for gsl_odeiv2_system 
	ParamsNbody params;
    params.et0 = et0;
    params.GM = GM;
    params.BODIES = BODIES;
    params.frame = frame;
    params.A = A;
    params.m = m;
    params.rhoS = rhoS;
    params.rhoD = rhoD;
    params.LU = LU;
    params.TU = TU;
    setup_spherical_harmonics(params, sh);   // may throw: before the GSL allocations

    const gsl_odeiv2_step_type* stepper = gsl_odeiv2_step_rk8pd;
    gsl_odeiv2_step* step = gsl_odeiv2_step_alloc(stepper, state_dim);

	// ref: https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_control_y_new
    gsl_odeiv2_control* control = gsl_odeiv2_control_y_new(abstol, reltol);
    gsl_odeiv2_evolve* evolve = gsl_odeiv2_evolve_alloc(state_dim);

    // define ODE
    gsl_odeiv2_system sys = {eom_gve_NBodySRP, nullptr, state_dim, &params};

    double t = tspan[0];
    double t1 = tspan[1];
    double h = 1.0;

    // Set initial conditions
    double y[state_dim];
    for (int i = 0; i < state_dim; ++i) {
        y[i] = state0[i];
    }

    // Initialize a vector of tuples to store the trajectory data
    std::vector<std::tuple<double, double, double, double, double, double, double>> trajectory;

    // Store the initial state at time tspan[0]
    trajectory.emplace_back(t, y[0], y[1], y[2], y[3], y[4], y[5]);

    bool sh_domain_failure = false;
    while (t < t1) {
        int status = gsl_odeiv2_evolve_apply(evolve, control, step, &sys, &t, t1, &h, y);

        if (status != GSL_SUCCESS) {
            // below sh_rho_min * R: raise instead of returning a truncated trajectory
            sh_domain_failure = sh_below_rho_min(params);
            if (!sh_domain_failure) {
                std::cerr << "Error: Integration failed." << std::endl;
            }
            break;
        }

        // Save the trajectory data for plotting
        trajectory.emplace_back(t, y[0], y[1], y[2], y[3], y[4], y[5]);
    }

    // release memory
    gsl_odeiv2_evolve_free(evolve);
    gsl_odeiv2_control_free(control);
    gsl_odeiv2_step_free(step);
    if (sh_domain_failure) {
        throw_sh_domain_error(params, t);
    }

    return trajectory;
}



/**
 * Evaluate the GVE right-hand side and its Jacobian at a single time and MEE state
 *
 * @param et0 initial epoch, in ephemeris seconds past J2000 (regardless of TU)
 * @param t time since et0, in TU
 * @param x MEE state [p,f,g,h,k,L]
 * @param GM list of gravitational parameters of each body; GM[0] is the central body
 * @param BODIES list of NAIF names of each body
 * @param A surface area of spacecraft; SRP is skipped if A == 0
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param LU length unit
 * @param TU time unit
 * @param sh optional central-body gravity field (file, nmax, mmax, frame_bf); off if sh.file is empty
 * @return pair (xdot, A) with A = d(xdot)/dx, 6x6 row-major
 */
std::pair<std::vector<double>, std::vector<double>> eval_jacobian_gve(
    double et0,
    double t,
    const std::vector<double>& x,
    const std::vector<double>& GM,
    const std::vector<std::string>& BODIES,
    const std::string& frame,
    double A,
    double m,
    double rhoS,
    double rhoD,
    double LU,
    double TU,
    const SphericalHarmonicsOptions& sh) {
    if (x.size() != 6) {
        throw std::invalid_argument("Invalid x length. Expected length 6.");
    }
    ParamsNbody params;
    params.et0 = et0;
    params.GM = GM;
    params.BODIES = BODIES;
    params.frame = frame;
    params.A = A;
    params.m = m;
    params.rhoS = rhoS;
    params.rhoD = rhoD;
    params.LU = LU;
    params.TU = TU;
    setup_spherical_harmonics(params, sh);

    // Out-of-domain states throw here; inside the integrator they are a recoverable GSL_EDOM
    const double w = 1 + x[1]*std::cos(x[5]) + x[2]*std::sin(x[5]);
    if (!(x[0] > 0)) {
        throw std::domain_error("eval_jacobian_gve: p must be positive.");
    }
    if (!(w > 0)) {
        throw std::domain_error("eval_jacobian_gve: w = 1 + f cos(L) + g sin(L) must be positive.");
    }

    std::vector<double> xdot(6, 0.0);
    double jac[6][6];
    int status = eval_gve_jacobian_NBodySRP(t, x.data(), xdot.data(), jac, &params);
    if (status != GSL_SUCCESS && sh_below_rho_min(params)) {
        throw_sh_domain_error(params, t);
    }
    if (status != GSL_SUCCESS) {
        std::string msg = "eval_jacobian_gve: evaluation failed";
        if (failed_c()) {
            // keep NAIF's message; the SPICE error flag is still set here
            SpiceChar naif_msg[1841];
            getmsg_c("LONG", sizeof(naif_msg), naif_msg);
            reset_c();
            msg += std::string(": ") + naif_msg;
        }
        throw std::runtime_error(msg);
    }
    std::vector<double> A_flat(36);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            A_flat[6*i + j] = jac[i][j];
        }
    }
    return {xdot, A_flat};
}
