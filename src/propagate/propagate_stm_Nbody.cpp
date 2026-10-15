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

#include "eoms/params.h"
#include "eoms/eoms_stm.h"


/**
 * Propagate spacecraft state and STM using N-body equations of motion
 *
 * @param et0 initial epoch, in ephemeris seconds past J2000 (regardless of TU)
 * @param tspan integration time span, [t0, tf]
 * @param state0 initial state, in order [x,y,z,vx,vy,vz] + list(stm(t0,t0))
 * @param GM list of gravitational parameters of each body
 * @param BODIES list of NAIF names of each body
 * @param LU length unit
 * @param TU time unit
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 * @return vector of tuples containing state history
 */
std::vector<std::tuple<double,
                       double, double, double, double, double, double,
                       double, double, double, double, double, double,
                       double, double, double, double, double, double,
                       double, double, double, double, double, double,
                       double, double, double, double, double, double,
                       double, double, double, double, double, double,
                       double, double, double, double, double, double>> solve_ivp_stm_nbody(
    double et0,
    const std::vector<double>& tspan,
    const std::vector<double>& state0,
    std::vector<double> GM,
    std::vector<std::string> BODIES,
    std::string frame,
    double LU,
    double TU,
    double reltol = 1e-14,
    double abstol = 1e-12){
    const int state_dim = 42;      // number of states

    // Check if the length of state0 is 42
    if (state0.size() != state_dim) {
        throw std::invalid_argument("Invalid state0 length. Expected length 42 (state + stm flattened).");
    }

    const gsl_odeiv2_step_type* stepper = gsl_odeiv2_step_rk8pd;
    gsl_odeiv2_step* step = gsl_odeiv2_step_alloc(stepper, state_dim);

	// ref: https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_control_y_new
    gsl_odeiv2_control* control = gsl_odeiv2_control_y_new(abstol, reltol);
    gsl_odeiv2_evolve* evolve = gsl_odeiv2_evolve_alloc(state_dim);

    // define input parameters for gsl_odeiv2_system 
	ParamsNbody params;
    params.et0 = et0;
    params.GM = GM;
    params.BODIES = BODIES;
    params.frame = frame;
    params.LU = LU;
    params.TU = TU;

    // define ODE
    gsl_odeiv2_system sys = {eom_stm_NBody, nullptr, state_dim, &params};

    double t = tspan[0];
    double t1 = tspan[1];
    double h = 1.0;

    // Set initial conditions
    double y[state_dim];
    for (int i = 0; i < state_dim; ++i) {
        y[i] = state0[i];
    }

    // Initialize a vector of tuples to store the trajectory data
    std::vector<std::tuple<double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double>> trajectory_stm;

    // Store the initial state at time tspan[0]
    trajectory_stm.emplace_back(t, y[0],  y[1],  y[2],  y[3],  y[4],  y[5], 
                               y[6],  y[7],  y[8],  y[9],  y[10], y[11], 
                               y[12], y[13], y[14], y[15], y[16], y[17], 
                               y[18], y[19], y[20], y[21], y[22], y[23],
                               y[24], y[25], y[26], y[27], y[28], y[29],
                               y[30], y[31], y[32], y[33], y[34], y[35],
                               y[36], y[37], y[38], y[39], y[40], y[41]);

    while (t < t1) {
        int status = gsl_odeiv2_evolve_apply(evolve, control, step, &sys, &t, t1, &h, y);

        if (status != GSL_SUCCESS) {
            std::cerr << "Error: Integration failed." << std::endl;
            break;
        }

        // Save the trajectory data for plotting
        trajectory_stm.emplace_back(t, y[0],  y[1],  y[2],  y[3],  y[4],  y[5], 
                                   y[6],  y[7],  y[8],  y[9],  y[10], y[11], 
                                   y[12], y[13], y[14], y[15], y[16], y[17], 
                                   y[18], y[19], y[20], y[21], y[22], y[23],
                                   y[24], y[25], y[26], y[27], y[28], y[29],
                                   y[30], y[31], y[32], y[33], y[34], y[35],
                                   y[36], y[37], y[38], y[39], y[40], y[41]);
    }

    // release memory
    gsl_odeiv2_evolve_free(evolve);
    gsl_odeiv2_control_free(control);
    gsl_odeiv2_step_free(step);

    return trajectory_stm;
}