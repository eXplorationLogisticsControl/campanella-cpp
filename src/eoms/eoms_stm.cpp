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

#include "gravity_gradient_Nbody.h"		   // for jacobian 

/**
 * N-body equations of motion with STM compatible for GSL including STM
 */
int eom_stm_NBody(double t, const double p[], double pdot[], void* params) {
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
    double position_spacecraft[3] = {x, y, z};

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
    const size_t N = GM.size();   // FIXME: not a great idea!
    
    // Initial acceleration components due to reference body
    double ax = -GM_ref*x / d3;
    double ay = -GM_ref*y / d3;
    double az = -GM_ref*z / d3;

    // storage for third-body positions
    double state_bi[3];          // FIXME : Need the first 3 to be N

    // storage for gravity gradient matrix
    double Uxx[3][3];
    gravity_gradient_center(position_spacecraft, GM_ref, Uxx);
    
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

        // store third-body position
        state_bi[0] = ai_third_body[3];
        state_bi[1] = ai_third_body[4];
        state_bi[2] = ai_third_body[5];

        // compute gravity gradient matrix
        double Uxx_i[3][3];
        gravity_gradient_third_body(position_spacecraft, state_bi, GM_i, Uxx_i);
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Uxx[j][k] += Uxx_i[j][k];
            }
        }
    }

    // central-body spherical harmonics, acceleration and gradient
    if (parameters->sh) {
        double a_sh[3];
        double Uxx_sh[3][3];
        int status = sh_perturbation_nbody(*parameters, t, position_spacecraft, a_sh, Uxx_sh, true);
        if (status != GSL_SUCCESS) {
            return status;
        }
        ax = ax + a_sh[0];
        ay = ay + a_sh[1];
        az = az + a_sh[2];
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Uxx[j][k] += Uxx_sh[j][k];
            }
        }
    }
    pdot[3] = ax;
    pdot[4] = ay;
    pdot[5] = az;

    // derivatives of STM, row-major
	pdot[6]  = p[24];
	pdot[7]  = p[25];
	pdot[8]  = p[26];
	pdot[9]  = p[27];
	pdot[10] = p[28];
	pdot[11] = p[29];

	pdot[12] = p[30];
	pdot[13] = p[31];
	pdot[14] = p[32];
	pdot[15] = p[33];
	pdot[16] = p[34];
	pdot[17] = p[35];

	pdot[18] = p[36];
	pdot[19] = p[37];
	pdot[20] = p[38];
	pdot[21] = p[39];
	pdot[22] = p[40];
	pdot[23] = p[41];

	pdot[24] = Uxx[0][0]*p[6]  + Uxx[0][1]*p[12] + Uxx[0][2]*p[18];
	pdot[25] = Uxx[0][0]*p[7]  + Uxx[0][1]*p[13] + Uxx[0][2]*p[19];
	pdot[26] = Uxx[0][0]*p[8]  + Uxx[0][1]*p[14] + Uxx[0][2]*p[20];
	pdot[27] = Uxx[0][0]*p[9]  + Uxx[0][1]*p[15] + Uxx[0][2]*p[21];
	pdot[28] = Uxx[0][0]*p[10] + Uxx[0][1]*p[16] + Uxx[0][2]*p[22];
	pdot[29] = Uxx[0][0]*p[11] + Uxx[0][1]*p[17] + Uxx[0][2]*p[23];

	pdot[30] = Uxx[1][0]*p[6]  + Uxx[1][1]*p[12] + Uxx[1][2]*p[18];
	pdot[31] = Uxx[1][0]*p[7]  + Uxx[1][1]*p[13] + Uxx[1][2]*p[19];
	pdot[32] = Uxx[1][0]*p[8]  + Uxx[1][1]*p[14] + Uxx[1][2]*p[20];
	pdot[33] = Uxx[1][0]*p[9]  + Uxx[1][1]*p[15] + Uxx[1][2]*p[21];
	pdot[34] = Uxx[1][0]*p[10] + Uxx[1][1]*p[16] + Uxx[1][2]*p[22];
	pdot[35] = Uxx[1][0]*p[11] + Uxx[1][1]*p[17] + Uxx[1][2]*p[23];
	
	pdot[36] = Uxx[2][0]*p[6]  + Uxx[2][1]*p[12] + Uxx[2][2]*p[18];
	pdot[37] = Uxx[2][0]*p[7]  + Uxx[2][1]*p[13] + Uxx[2][2]*p[19];
	pdot[38] = Uxx[2][0]*p[8]  + Uxx[2][1]*p[14] + Uxx[2][2]*p[20];
	pdot[39] = Uxx[2][0]*p[9]  + Uxx[2][1]*p[15] + Uxx[2][2]*p[21];
	pdot[40] = Uxx[2][0]*p[10] + Uxx[2][1]*p[16] + Uxx[2][2]*p[22];
	pdot[41] = Uxx[2][0]*p[11] + Uxx[2][1]*p[17] + Uxx[2][2]*p[23];

    return GSL_SUCCESS;
}
