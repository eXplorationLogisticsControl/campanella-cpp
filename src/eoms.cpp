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
#include "perturbations.h"
#include "SRP.h"


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
 * N-body equations of motion compatible for GSL
 */
int eom_NBody(double t, const double p[], double pdot[], void* params) {
    ParamsNbody* parameters = static_cast<ParamsNbody*>(params);
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

    pdot[3] = ax;
    pdot[4] = ay;
    pdot[5] = az;
    return GSL_SUCCESS;
}