// Propagate the Moon-centered test case in modified equinoctial elements (MEE) with the
// Gauss variational equations, and compare against the Cartesian N-body + SRP propagator.
// SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

std::string kernel_path(const char* spice_dir, const char* relative) {
    std::string path(spice_dir);
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    path += relative;
    return path;
}

template <typename Tuple, std::size_t... I>
void store_tuple(const Tuple& row, double* out, std::index_sequence<I...>) {
    ((out[I] = std::get<I>(row)), ...);
}

// Copy the first N elements of a result row into a contiguous array.
template <std::size_t N, typename Tuple>
std::array<double, N> tuple_head(const Tuple& row) {
    std::array<double, N> out{};
    store_tuple(row, out.data(), std::make_index_sequence<N>{});
    return out;
}

double norm3(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
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

    // Same Moon-centered initial state as solve_Nbody.cpp. et0 is ephemeris seconds past J2000.
    const double et0 = 740463475.5117936;
    const std::vector<double> GM = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
    const std::vector<std::string> BODIES = {"MOON", "EARTH", "SUN"};
    const std::vector<double> tspan = {0.0, 20 * 86400.0};
    const std::vector<double> state0 = {
        -7505.7613773288285, 44128.19419987408, -55792.327508873765,
        0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
    // Spacecraft area, mass and optical properties for SRP. LU = TU = 1 keeps km and seconds.
    const double A = 34.675, m = 1850.0, rhoS = 0.9, rhoD = 0.0;

    // Initial MEE [p, f, g, h, k, L]. The central body (BODIES[0]) defines mu.
    const std::vector<double> mee0 = campanella::cart2mee(state0, GM[0]);
    std::cout << std::setprecision(16);
    std::cout << "initial MEE:";
    for (double v : mee0) {
        std::cout << " " << v;
    }
    std::cout << "\n";

    campanella::PropagatorGve prop_gve(GM, BODIES, "J2000", A, m, rhoS, rhoD, 1.0, 1.0);
    campanella::PropagatorNbodySrp prop_cart(GM, BODIES, "J2000", A, m, rhoS, rhoD, 1.0, 1.0);

    const auto traj_gve = prop_gve.solve_ivp(et0, tspan, mee0);
    const auto traj_cart = prop_cart.solve_ivp(et0, tspan, state0);
    if (traj_gve.empty() || traj_cart.empty()) {
        std::cerr << "solve_ivp returned no states\n";
        return 1;
    }
    const auto mee_f = tuple_head<7>(traj_gve.back());
    const auto cart_f = tuple_head<7>(traj_cart.back());
    if (mee_f[0] != tspan[1] || cart_f[0] != tspan[1]) {
        std::cerr << "integration did not reach tf\n";
        return 1;
    }

    // Final MEE; L is not wrapped and grows by 2 pi per revolution
    std::cout << "final MEE (" << traj_gve.size() << " steps):";
    for (int i = 1; i < 7; ++i) {
        std::cout << " " << mee_f[i];
    }
    std::cout << "\n";

    const std::vector<double> rv_gve = campanella::mee2cart(
        {mee_f[1], mee_f[2], mee_f[3], mee_f[4], mee_f[5], mee_f[6]}, GM[0]);
    std::cout << "final state, GVE      :";
    for (double v : rv_gve) {
        std::cout << " " << v;
    }
    std::cout << "\nfinal state, Cartesian (" << traj_cart.size() << " steps):";
    for (int i = 1; i < 7; ++i) {
        std::cout << " " << cart_f[i];
    }
    std::cout << "\n";

    // Both propagations carry integration error; the difference is expected around 1e-7 km.
    const double dr = norm3(rv_gve[0] - cart_f[1], rv_gve[1] - cart_f[2], rv_gve[2] - cart_f[3]);
    const double dv = norm3(rv_gve[3] - cart_f[4], rv_gve[4] - cart_f[5], rv_gve[5] - cart_f[6]);
    std::cout << std::setprecision(3) << std::scientific;
    std::cout << "|dr| = " << dr << " km, |dv| = " << dv << " km/s\n";

    const double tol_r = 1e-5;    // km
    const double tol_v = 1e-10;   // km/s
    if (!(dr <= tol_r && dv <= tol_v)) {
        std::cerr << "GVE and Cartesian final states differ by more than "
                  << tol_r << " km / " << tol_v << " km/s\n";
        return 1;
    }
    return 0;
}
