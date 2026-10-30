// Propagate a Moon-centred Cartesian state with the Earth and the Sun as third bodies, together
// with its 6x6 state-transition matrix, which is checked against central differences.
// SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool close(double got, double expected, double rtol, double atol) {
    return std::abs(got - expected) <= rtol * std::abs(expected) + atol;
}

std::string kernel_path(const char* spice_dir, const char* relative) {
    std::string path(spice_dir);
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    path += relative;
    return path;
}

}  // namespace

int main() {
    const char* spice_dir = std::getenv("SPICE");
    if (spice_dir == nullptr || spice_dir[0] == '\0') {
        std::cerr << "SPICE must point at a directory of generic SPICE kernels\n";
        return 1;
    }
    try {
        // Leap seconds, then planetary ephemeris. spiceypy.furnsh does not affect this library.
        campanella::furnsh(kernel_path(spice_dir, "lsk/naif0012.tls"));
        campanella::furnsh(kernel_path(spice_dir, "spk/de440.bsp"));

        // The first NAIF id is the central body. GMs in km^3/s^2; without DU, units are km and s.
        const double et0 = 740463475.5117936;
        campanella::SpiceOptions options;
        options.GMs = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
        const campanella::SpiceParameters params(et0, {"MOON", "EARTH", "SUN"}, options);

        const std::vector<double> tspan = {0.0, 20 * 86400.0};
        const std::vector<double> x0 = {
            -7505.7613773288285, 44128.19419987408, -55792.327508873765,
            0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
        // res.stm holds Phi(t, t0) row-major, 36 values per row of res.t
        const campanella::IvpResult res = campanella::solve_ivpstm_rv(params, tspan, x0);
        const double* xf = &res.x[6 * (res.size() - 1)];
        const double* phi = &res.stm[36 * (res.size() - 1)];

        const double expected[] = {
            10171.343017143072, 33674.15294325228, -60877.5267967639,
            0.013967106146233203, -0.0805131196626756, 0.0033270015062299776};
        bool ok = res.t.back() == tspan[1];
        std::cout << "final state:";
        for (int i = 0; i < 6; ++i) {
            std::cout << " " << xf[i];
            ok &= close(xf[i], expected[i], 1e-12, 1e-8);
        }
        std::cout << "\nSTM:\n";
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 6; ++col) {
                std::cout << " " << phi[6 * row + col];
            }
            std::cout << "\n";
        }
        if (!ok) {
            std::cerr << "final state does not match the N-body reference\n";
            return 1;
        }

        // Column col of Phi is d(final state) / d(x0[col]). rtol is loose: h is 1e-6 km or km/s.
        constexpr double h = 1e-6;
        for (int col = 0; col < 6; ++col) {
            std::vector<double> forward = x0, backward = x0;
            forward[col] += h;
            backward[col] -= h;
            const std::vector<double> xp = campanella::solve_ivp_rv(params, tspan, forward).x;
            const std::vector<double> xm = campanella::solve_ivp_rv(params, tspan, backward).x;
            for (int row = 0; row < 6; ++row) {
                const double numeric = (xp[xp.size() - 6 + row] - xm[xm.size() - 6 + row]) / (2.0 * h);
                if (!close(phi[6 * row + col], numeric, 1e-2, 1e-8)) {
                    std::cerr << "STM row " << row << " col " << col << " analytic "
                              << phi[6 * row + col] << " numeric " << numeric << "\n";
                    ok = false;
                }
            }
        }
        if (!ok) {
            std::cerr << "STM does not match the finite-difference reference\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
