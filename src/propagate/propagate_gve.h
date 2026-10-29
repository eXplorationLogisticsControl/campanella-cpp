#ifndef PROPAGATE_GVE_H
#define PROPAGATE_GVE_H

#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "perturbations/spherical_harmonics.h"

std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp_gve(
    double et0,
    const std::vector<double>& tspan,
    const std::vector<double>& state0,
    const std::vector<double> GM,
    const std::vector<std::string> BODIES,
    std::string frame,
    double A,
    double m,
    double rhoS,
    double rhoD,
    double LU,
    double TU,
    double reltol = 1e-14,
    double abstol = 1e-12,
    const SphericalHarmonicsOptions& sh = SphericalHarmonicsOptions());

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
    const SphericalHarmonicsOptions& sh = SphericalHarmonicsOptions());
#endif // PROPAGATE_GVE_H
