#ifndef TWOBODY_H
#define TWOBODY_H

#include <tuple>
#include <vector>

std::vector<std::tuple<double, double, double, double, double, double, double>> propagate_twobody(
    const std::vector<double>& tspan,
    const std::vector<double>& state0,
    double GM,
    double reltol = 1e-14,
    double abstol = 1e-12);
#endif // TWOBODY_H
