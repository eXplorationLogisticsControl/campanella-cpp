#include "campanella.hpp"

#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

namespace {

bool close(double got, double expected) {
    return std::abs(got - expected) <= 1e-12 * std::abs(expected) + 1e-8;
}

}  // namespace

int main() {
    const double GM = 4.902800582147760e+03;
    const std::vector<double> tspan = {0.0, 20 * 86400.0};
    const std::vector<double> y0 = {
        -7505.7613773288285, 44128.19419987408, -55792.327508873765,
        0.07193326412942684, 0.018407794390508374, 0.004868437733535174};

    campanella::PropagatorTwoBody prop(GM);
    const auto traj = prop.solve_ivp(tspan, y0);
    if (traj.empty()) {
        std::cerr << "solve_ivp returned no states\n";
        return 1;
    }

    const auto& last = traj.back();
    const double got[] = {
        std::get<0>(last), std::get<1>(last), std::get<2>(last), std::get<3>(last),
        std::get<4>(last), std::get<5>(last), std::get<6>(last)};
    const double expected[] = {
        1728000.0, -18205.698468519975, 26479.25761849232, -38615.225993310676,
        0.01263483050054705, 0.1635690404557372, -0.19163703441809382};

    std::cout << "final state:";
    bool ok = true;
    for (double value : got) {
        std::cout << " " << value;
    }
    std::cout << "\n";
    for (int i = 0; i < 7; ++i) {
        if (!close(got[i], expected[i])) {
            ok = false;
        }
    }
    if (!ok) {
        std::cerr << "final state does not match the two-body reference\n";
        return 1;
    }
    return 0;
}
