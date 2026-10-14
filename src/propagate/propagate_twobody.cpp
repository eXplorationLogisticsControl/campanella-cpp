/*
Two-body equations of motion
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include "SpiceUsr.h"

#include "eoms/eoms.h"

/**
 * Propagate spacecraft state using two-body equations of motion
 *
 * @param tspan integration time span, [t0, tf]
 * @param state0 initial state, in order [x,y,z,vx,vy,vz]
 * @param GM gravitational parameter
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 * @return vector of tuples containing state history
 */
std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp_twobody(
    const std::vector<double>& tspan,
    const std::vector<double>& state0,
    double GM,
    double reltol = 1e-14,
    double abstol = 1e-12){
    const int state_dim = 6;      // number of states
    if (state0.size() != state_dim) {
        throw std::invalid_argument("Invalid state0 length. Expected length 6.");
    }

    const gsl_odeiv2_step_type* stepper = gsl_odeiv2_step_rk8pd;
    gsl_odeiv2_step* step = gsl_odeiv2_step_alloc(stepper, state_dim);

	// ref: https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_control_y_new
    gsl_odeiv2_control* control = gsl_odeiv2_control_y_new(abstol, reltol);
    gsl_odeiv2_evolve* evolve = gsl_odeiv2_evolve_alloc(state_dim);

    gsl_odeiv2_system sys = {eom_twobody, nullptr, state_dim, &GM};

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

    while (t < t1) {
        int status = gsl_odeiv2_evolve_apply(evolve, control, step, &sys, &t, t1, &h, y);

        if (status != GSL_SUCCESS) {
            std::cerr << "Error: Integration failed." << std::endl;
            break;
        }

        // Save the trajectory data for plotting
        trajectory.emplace_back(t, y[0], y[1], y[2], y[3], y[4], y[5]);
    }

    // release memory
    gsl_odeiv2_evolve_free(evolve);
    gsl_odeiv2_control_free(control);
    gsl_odeiv2_step_free(step);

    return trajectory;
}