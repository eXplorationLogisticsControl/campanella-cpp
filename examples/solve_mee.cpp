// Propagate the Moon-centred test case in modified equinoctial elements (MEE) with the
// Gauss variational equations, and compare against the Cartesian propagation of the same
// dynamics (third bodies and solar radiation pressure).
// SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
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

        // Moon-centred, with the Earth and the Sun; SRP needs the Sun among the NAIF ids.
        const double et0 = 740463475.5117936;
        campanella::SpiceOptions options;
        options.GMs = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
        options.include_srp = true;
        options.srp_A = 34.675;     // m^2
        options.srp_m = 1850.0;     // kg
        options.srp_rhoS = 0.9;
        const campanella::SpiceParameters params(et0, {"MOON", "EARTH", "SUN"}, options);
        const double mu = params.mus()[0];

        const std::vector<double> tspan = {0.0, 20 * 86400.0};
        const std::vector<double> x0 = {
            -7505.7613773288285, 44128.19419987408, -55792.327508873765,
            0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
        const std::vector<double> mee0 = campanella::cart2mee(x0, mu);
        std::cout << std::setprecision(16) << "initial MEE:";
        for (double v : mee0) {
            std::cout << " " << v;
        }
        std::cout << "\n";

        const campanella::IvpResult mee = campanella::solve_ivp_mee(params, tspan, mee0);
        const campanella::IvpResult cart = campanella::solve_ivp_rv(params, tspan, x0);
        const std::vector<double> mee_f(mee.x.end() - 6, mee.x.end());
        const std::vector<double> cart_f(cart.x.end() - 6, cart.x.end());

        // Final MEE; L is not wrapped and grows by 2 pi per revolution
        std::cout << "final MEE (" << mee.size() << " steps):";
        for (double v : mee_f) {
            std::cout << " " << v;
        }
        const std::vector<double> rv_mee = campanella::mee2cart(mee_f, mu);
        std::cout << "\nfinal state, MEE      :";
        for (double v : rv_mee) {
            std::cout << " " << v;
        }
        std::cout << "\nfinal state, Cartesian (" << cart.size() << " steps):";
        for (double v : cart_f) {
            std::cout << " " << v;
        }
        std::cout << "\n";

        // Both propagations carry integration error; the difference is expected around 1e-7 km.
        const double dr = norm3(rv_mee[0] - cart_f[0], rv_mee[1] - cart_f[1], rv_mee[2] - cart_f[2]);
        const double dv = norm3(rv_mee[3] - cart_f[3], rv_mee[4] - cart_f[4], rv_mee[5] - cart_f[5]);
        std::cout << std::setprecision(3) << std::scientific
                  << "|dr| = " << dr << " km, |dv| = " << dv << " km/s\n";
        const double tol_r = 1e-5;    // km
        const double tol_v = 1e-10;   // km/s
        if (!(dr <= tol_r && dv <= tol_v)) {
            std::cerr << "MEE and Cartesian final states differ by more than "
                      << tol_r << " km / " << tol_v << " km/s\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
