#ifndef NBODY_H
#define NBODY_H

#include <string>
#include <tuple>
#include <vector>

std::vector<std::tuple<double, double, double, double, double, double, double>> propagate_Nbody(
    double et0,
    const std::vector<double>& tspan,
    const std::vector<double>& state0,
    const std::vector<double> GM,
    const std::vector<std::string> BODIES,
    std::string frame,
    double LU,
    double TU,
    double reltol = 1e-14,
    double abstol = 1e-12);
    
#endif // NBODY_H

