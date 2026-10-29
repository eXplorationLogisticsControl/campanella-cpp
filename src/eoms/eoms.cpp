/**
 * Equations of motion
 */

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <gsl/gsl_errno.h>
#include <SpiceUsr.h>          // access SPICE files
#include "params.h"
#include "eoms.h"
#include "perturbations/thirdbody.h"
#include "perturbations/SRP.h"
#include "perturbations/spherical_harmonics.h"


/**
 * Two-body equations of motion compatible for GSL
 */
int eom_twobody(double t, const double y[], double dydt[], void* params) {
    double GM = *(double*)params; // Gravitational constant times Earth's mass

    double r = sqrt((y[0] * y[0]) + (y[1] * y[1]) + (y[2] * y[2]));

    dydt[0] = y[3];
    dydt[1] = y[4];
    dydt[2] = y[5];
    dydt[3] = (-GM * y[0]) / (r * r * r);
    dydt[4] = (-GM * y[1]) / (r * r * r);
    dydt[5] = (-GM * y[2]) / (r * r * r);

    return GSL_SUCCESS;
}


/**
 * Central-body spherical-harmonics perturbation for the N-body EOMs
 *
 * @param parameters N-body parameters, with parameters.sh not null
 * @param t time, in TU
 * @param rvec spacecraft position, in LU
 * @param a acceleration to be computed, in LU/TU^2
 * @param GG 3-by-3 gradient to be computed, or nullptr to skip it
 * @param reset_spice_errors if true, call reset_c() on a SPICE error
 * @return GSL_SUCCESS; GSL_FAILURE on SPICE error; GSL_EDOM below sh_rho_min * R
 */
int sh_perturbation_nbody(const ParamsNbody& parameters, double t, const double rvec[3],
                          double a[3], double GG[3][3], bool reset_spice_errors) {
    SpiceDouble et = parameters.et0 + t*parameters.TU;  // ephemeris query time
    bool ok = sh_perturbation(rvec, et, parameters.frame.c_str(), parameters.frame_bf.c_str(),
                              *parameters.sh, *parameters.sh_ws, parameters.mu_sh, parameters.R_sh,
                              a, GG, parameters.sh_rho_min);
    if (failed_c()) {
        if (reset_spice_errors) {
            reset_c();
        }
        return GSL_FAILURE;
    }
    if (!ok) {
        return GSL_EDOM;   // below sh_rho_min * R: the series may diverge
    }
    return GSL_SUCCESS;
}


/**
 * N-body equations of motion compatible for GSL
 */
int eom_NBody(double t, const double p[], double pdot[], void* params) {
    ParamsNbody* parameters = static_cast<ParamsNbody*>(params);
    // the SH domain flag must describe this call only (read by the propagators after a failure)
    if (parameters->sh) {
        parameters->sh_ws->below_rho_min = false;
    }
    double et0 = parameters->et0;
    std::vector<double> GM = parameters->GM;
    std::vector<std::string> BODIES = parameters->BODIES;
    std::string frame = parameters->frame;
    double LU = parameters->LU;
    double TU = parameters->TU;
    
    // Name of reference body and frame
    ConstSpiceChar * frame_cspice = frame.c_str();
    ConstSpiceChar * ref_body_name = BODIES[0].c_str();

    // Get position coordinates
    double x = p[0];
    double y = p[1];
    double z = p[2];
    double d = sqrt((x*x) + (y*y) + (z*z));
    double d3 = d*d*d;
    // Get velocity coordinates
    double xdot = p[3];
    double ydot = p[4];
    double zdot = p[5];
    pdot[0] = xdot;
    pdot[1] = ydot;
    pdot[2] = zdot;
    // G*Mass of reference body
    double GM_ref = GM[0];
    
    // Number of bodies including reference body equals number of elements of vector GM
    int N = GM.size();   // FIXME: not a great idea!
    
    // Initial acceleration components due to reference body
    double ax = -GM_ref*x / d3;
    double ay = -GM_ref*y / d3;
    double az = -GM_ref*z / d3;
    
    // Loop for each body after reference body
    for (int i = 1; i < N; i++ ) {
        // GM value and name of body i
        ConstSpiceChar * third_body_name = BODIES[i].c_str();
        double GM_i = GM[i];
        SpiceDouble et = et0 + t*TU;  // ephemeris query time

        // compute third-body acceleration
        std::vector<double> ai_third_body = compute_thirdbody_perturbation(
            x, y, z, GM_i, third_body_name, et, frame_cspice, ref_body_name, LU);
        ax = ax + ai_third_body[0];
        ay = ay + ai_third_body[1];
        az = az + ai_third_body[2];
        if (failed_c()) {
            reset_c();
            return GSL_FAILURE;
        }
    }

    // central-body spherical harmonics
    if (parameters->sh) {
        double rvec[3] = {x, y, z};
        double a_sh[3];
        int status = sh_perturbation_nbody(*parameters, t, rvec, a_sh, nullptr, true);
        if (status != GSL_SUCCESS) {
            return status;
        }
        ax = ax + a_sh[0];
        ay = ay + a_sh[1];
        az = az + a_sh[2];
    }
    pdot[3] = ax;
    pdot[4] = ay;
    pdot[5] = az;
    return GSL_SUCCESS;
}


/**
 * N-body equations of motion compatible for GSL
 */
int eom_NBodySRP(double t, const double p[], double pdot[], void* params) {
    ParamsNbody* parameters = static_cast<ParamsNbody*>(params);
    // the SH domain flag must describe this call only (read by the propagators after a failure)
    if (parameters->sh) {
        parameters->sh_ws->below_rho_min = false;
    }
    double et0 = parameters->et0;
    std::vector<double> GM = parameters->GM;
    std::vector<std::string> BODIES = parameters->BODIES;
    std::string frame = parameters->frame;
    double A = parameters->A;
    double m = parameters->m;
    double rhoS = parameters->rhoS;
    double rhoD = parameters->rhoD;
    double LU = parameters->LU;
    double TU = parameters->TU;
    
    // Name of reference body and frame
    ConstSpiceChar * frame_cspice = frame.c_str();
    ConstSpiceChar * ref_body_name = BODIES[0].c_str();

    // Get position coordinates
    double x = p[0];
    double y = p[1];
    double z = p[2];
    double d = sqrt((x*x) + (y*y) + (z*z));
    double d3 = d*d*d;
    // Get velocity coordinates
    double xdot = p[3];
    double ydot = p[4];
    double zdot = p[5];
    pdot[0] = xdot;
    pdot[1] = ydot;
    pdot[2] = zdot;
    // G*Mass of reference body
    double GM_ref = GM[0];
    
    // Number of bodies including reference body equals number of elements of vector GM
    int N = GM.size();   // FIXME: not a great idea!
    
    // Initial acceleration components due to reference body
    double ax = -GM_ref*x / d3;
    double ay = -GM_ref*y / d3;
    double az = -GM_ref*z / d3;
    
    // Loop for each body after reference body
    for (int i = 1; i < N; i++ ) {
        // GM value and name of body i
        ConstSpiceChar * third_body_name = BODIES[i].c_str();
        double GM_i = GM[i];
        SpiceDouble et = et0 + t*TU;  // ephemeris query time
        
        // compute third-body acceleration
        std::vector<double> ai_third_body = compute_thirdbody_perturbation(
            x, y, z, GM_i, third_body_name, et, frame_cspice, ref_body_name, LU);
        ax = ax + ai_third_body[0];
        ay = ay + ai_third_body[1];
        az = az + ai_third_body[2];
        if (failed_c()) {
            reset_c();
            return GSL_FAILURE;
        }
    }

    // compute SRP acceleration
    SpiceDouble et = et0 + t*TU;  // ephemeris query time
    std::vector<double> ai_SRP = compute_SRP(
    x, y, z, et, frame_cspice, ref_body_name, A, m, rhoS, rhoD, TU, LU);
    if (failed_c()) {
        reset_c();
        return GSL_FAILURE;
    }
    ax = ax + ai_SRP[0];
    ay = ay + ai_SRP[1];
    az = az + ai_SRP[2];

    // central-body spherical harmonics
    if (parameters->sh) {
        double rvec[3] = {x, y, z};
        double a_sh[3];
        int status = sh_perturbation_nbody(*parameters, t, rvec, a_sh, nullptr, true);
        if (status != GSL_SUCCESS) {
            return status;
        }
        ax = ax + a_sh[0];
        ay = ay + a_sh[1];
        az = az + a_sh[2];
    }

    pdot[3] = ax;
    pdot[4] = ay;
    pdot[5] = az;
    return GSL_SUCCESS;
}