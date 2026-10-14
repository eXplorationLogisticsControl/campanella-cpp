// Propagate the N-body state together with its 6x6 state-transition matrix.
// The matrix is checked against central differences of solve_ivp.
// SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
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
    const double GM = 4.902800582147760e+03;
    const std::vector<double> tspan = {0.0, 20 * 86400.0};
    const std::vector<double> state0 = {
        -7505.7613773288285, 44128.19419987408, -55792.327508873765,
        0.07193326412942684, 0.018407794390508374, 0.004868437733535174};

    // y0 is [x, y, z, vx, vy, vz] followed by Phi(t0, t0) = I, stored row-major.
    std::vector<double> y0 = state0;
    y0.resize(42, 0.0);
    for (int i = 0; i < 6; ++i) {
        y0[6 + i * 6 + i] = 1.0;
    }

    // GM[i] belongs to BODIES[i]. BODIES[0] is the central body. LU = TU = 1 keeps km and seconds.
    campanella::PropagatorNBody prop(
        {GM, 398600.432896939, 132712197035.766},
        {"MOON", "EARTH", "SUN"},
        "J2000",
        1.0,
        1.0);
    // Each row is (t, x, y, z, vx, vy, vz, Phi row-major). Phi is 36 elements, so a row has 43.
    const auto traj = prop.solve_ivp_stm(et0, tspan, y0);
    if (traj.empty()) {
        std::cerr << "solve_ivp_stm returned no states\n";
        return 1;
    }

    const auto got = tuple_head<43>(traj.back());
    // got[0..6] is the same final state as solve_Nbody.cpp. got[7..] is Phi.
    const double expected_state[] = {
        1728000.0, 10171.343017143072, 33674.15294325228, -60877.5267967639,
        0.013967106146233203, -0.0805131196626756, 0.0033270015062299776};

    std::cout << "final state:";
    bool ok = true;
    for (int i = 0; i < 7; ++i) {
        std::cout << " " << got[i];
        if (!close(got[i], expected_state[i], 1e-12, 1e-8)) {
            ok = false;
        }
    }
    std::cout << "\nSTM:\n";
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            std::cout << " " << got[7 + row * 6 + col];
        }
        std::cout << "\n";
    }
    if (!ok) {
        std::cerr << "final state does not match the N-body reference\n";
        return 1;
    }

    // Perturb one initial-state component at a time and difference the final states.
    // Column col of Phi is d(final state) / d(state0[col]). rtol is loose: the step h is 1e-6 km or km/s.
    constexpr double h = 1e-6;
    for (int col = 0; col < 6; ++col) {
        auto forward = state0;
        auto backward = state0;
        forward[col] += h;
        backward[col] -= h;
        const auto traj_f = prop.solve_ivp(et0, tspan, forward);
        const auto traj_b = prop.solve_ivp(et0, tspan, backward);
        if (traj_f.empty() || traj_b.empty()) {
            std::cerr << "finite-difference propagation returned no states\n";
            return 1;
        }
        const auto yf = tuple_head<7>(traj_f.back());
        const auto yb = tuple_head<7>(traj_b.back());
        for (int row = 0; row < 6; ++row) {
            const double numeric = (yf[row + 1] - yb[row + 1]) / (2.0 * h);
            const double analytic = got[7 + row * 6 + col];
            if (!close(analytic, numeric, 1e-2, 1e-8)) {
                std::cerr << "STM row " << row << " col " << col
                          << " analytic " << analytic << " numeric " << numeric << "\n";
                ok = false;
            }
        }
    }
    if (!ok) {
        std::cerr << "STM does not match the finite-difference reference\n";
        return 1;
    }
    return 0;
}
