#ifndef NBODYSRP_H
#define NBODYSRP_H

#include <iostream>
#include <vector>

std::vector<std::tuple<double, double, double, double, double, double, double>> propagate_NbodySRP(
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
    double abstol = 1e-12);
#endif // NBODYSRP_H
