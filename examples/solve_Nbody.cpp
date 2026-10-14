// Propagate the same lunar initial state under Moon, Earth, and Sun gravity.
// Body ephemerides come from CSPICE, so SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {

// Same tolerance as numpy.allclose(..., rtol=1e-12): atol is numpy's default 1e-8.
bool close(double got, double expected) {
    return std::abs(got - expected) <= 1e-12 * std::abs(expected) + 1e-8;
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
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    // et0 is ephemeris seconds past J2000. tspan is seconds since et0.
    // y0 is Moon-centered [x, y, z, vx, vy, vz] in km, km/s.
    const double et0 = 740463475.5117936;
    const double GM = 4.902800582147760e+03;
    const std::vector<double> tspan = {0.0, 20 * 86400.0};
    const std::vector<double> y0 = {
        -7505.7613773288285, 44128.19419987408, -55792.327508873765,
        0.07193326412942684, 0.018407794390508374, 0.004868437733535174};

    // GM[i] belongs to BODIES[i]. BODIES[0] is the central body.
    // LU = TU = 1 keeps the integration in km and seconds.
    campanella::PropagatorNBody prop(
        {GM, 398600.432896939, 132712197035.766},
        {"MOON", "EARTH", "SUN"},
        "J2000",
        1.0,
        1.0);
    // Each row is (t, x, y, z, vx, vy, vz).
    const auto traj = prop.solve_ivp(et0, tspan, y0);
    if (traj.empty()) {
        std::cerr << "solve_ivp returned no states\n";
        return 1;
    }

    const auto& last = traj.back();
    const double got[] = {
        std::get<0>(last), std::get<1>(last), std::get<2>(last), std::get<3>(last),
        std::get<4>(last), std::get<5>(last), std::get<6>(last)};
    // Final row recorded by test-python/test_propagate_Nbody.py.
    const double expected[] = {
        1728000.0, 10171.343017143072, 33674.15294325228, -60877.5267967639,
        0.013967106146233203, -0.0805131196626756, 0.0033270015062299776};

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
        std::cerr << "final state does not match the N-body reference\n";
        return 1;
    }
    return 0;
}
