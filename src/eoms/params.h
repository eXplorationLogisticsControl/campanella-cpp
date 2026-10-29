#ifndef PARAMS_H
#define PARAMS_H

#include <memory>
#include <string>
#include <vector>
#include "perturbations/spherical_harmonics.h"

/**
 * Parameters for N-body equations of motion, compatible for GSL
 */
struct ParamsNbody {
    double et0;
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double A;
    double m;
    double rhoS;
    double rhoD;
    double LU;
    double TU;
    // central-body spherical harmonics, set by setup_spherical_harmonics; nullptr => off.
    // sh is immutable and may be shared. sh_ws is scratch space written at every RHS call: copying
    // a ParamsNbody copies the shared_ptr, so copies must not be used from different threads
    // (call setup_spherical_harmonics on each copy, or give each copy its own workspace).
    std::shared_ptr<const SphericalHarmonicsModel> sh;
    std::shared_ptr<SphericalHarmonicsWorkspace> sh_ws;
    std::string frame_bf;                                 // body-fixed frame, e.g. "MOON_PA", "ITRF93"
    double mu_sh = 0.0;                                   // GM of the file * TU^2 / LU^3
    double R_sh = 0.0;                                    // reference radius of the file / LU
    double sh_rho_min = 1.0;                              // smallest |r|/R evaluated
};

#endif // PARAMS_H
