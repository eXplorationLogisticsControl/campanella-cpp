#ifndef EOMS_H
#define EOMS_H

// Declare the two-body equations of motion for a 3D satellite orbiting the Earth
int eom_twobody(double t, const double y[], double dydt[], void* params);
int eom_NBody(double t, const double p[], double pdot[], void* params);
int eom_NBodySRP(double t, const double p[], double pdot[], void* params);

// Central-body spherical-harmonics acceleration (and gradient if GG != nullptr) for parameters.sh != nullptr.
// Returns GSL_SUCCESS, GSL_FAILURE on SPICE error (reset_c() only if reset_spice_errors), GSL_EDOM if |r| < sh_rho_min R.
struct ParamsNbody;
int sh_perturbation_nbody(const ParamsNbody& parameters, double t, const double rvec[3],
                          double a[3], double GG[3][3], bool reset_spice_errors);

#endif
