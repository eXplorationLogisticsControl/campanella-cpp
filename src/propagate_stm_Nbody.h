#ifndef STM_NBODY_H
#define STM_NBODY_H

#include <vector>

std::vector<std::tuple<double,
                    double, double, double, double, double, double,
                    double, double, double, double, double, double,
                    double, double, double, double, double, double,
                    double, double, double, double, double, double,
                    double, double, double, double, double, double,
                    double, double, double, double, double, double,
                    double, double, double, double, double, double>> propagate_stm_Nbody(
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
    
#endif // STM_NBODY_H

