// Propagate a lunar two-body orbit for 20 days and check the final state.
// With a single body and an explicit GM, no ephemeris is queried, so no kernels are loaded.

#include "campanella.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

// Same tolerance as numpy.allclose(..., rtol=1e-12): atol is numpy's default 1e-8.
bool close(double got, double expected) {
    return std::abs(got - expected) <= 1e-12 * std::abs(expected) + 1e-8;
}

}  // namespace

int main() {
    // Lunar GM, km^3/s^2. Without DU the units are km and s. State is [x, y, z, vx, vy, vz].
    campanella::SpiceOptions options;
    options.GMs = {4.902800582147760e+03};
    const campanella::SpiceParameters params(0.0, {"MOON"}, options);

    const std::vector<double> tspan = {0.0, 20 * 86400.0};
    const std::vector<double> x0 = {
        -7505.7613773288285, 44128.19419987408, -55792.327508873765,
        0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
    // Every accepted step: res.t[k] and res.x[6k .. 6k+5]
    const campanella::IvpResult res = campanella::solve_ivp_rv(params, tspan, x0);

    // Final state recorded by test-python/test_solve_ivp.py.
    const double expected[] = {
        -18205.698468519975, 26479.25761849232, -38615.225993310676,
        0.01263483050054705, 0.1635690404557372, -0.19163703441809382};
    const double* xf = &res.x[6 * (res.size() - 1)];
    bool ok = res.t.back() == tspan[1];
    std::cout << "final state (" << res.size() << " steps):";
    for (int i = 0; i < 6; ++i) {
        std::cout << " " << xf[i];
        ok &= close(xf[i], expected[i]);
    }
    std::cout << "\n";
    if (!ok) {
        std::cerr << "final state does not match the two-body reference\n";
        return 1;
    }
    return 0;
}
