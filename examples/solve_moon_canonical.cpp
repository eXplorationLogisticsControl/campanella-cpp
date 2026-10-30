// Propagate the Moon-centred test case in canonical units and check each result against the
// same propagation in km and seconds.
// DU = 100000 km gives VU = sqrt(MU_MOON / DU), TU = DU / VU, so the scaled lunar GM is 1.
// SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <algorithm>
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

// Last 6 states, or the last STM, of a result
std::vector<double> last(const std::vector<double>& rows, std::size_t n) {
    return std::vector<double>(rows.end() - n, rows.end());
}

// Report a position/velocity difference in km and km/s; true if within tolerance.
bool report(const char* name, const std::vector<double>& rv, const std::vector<double>& rv_ref,
            double tol_r, double tol_v) {
    const double dr = norm3(rv[0] - rv_ref[0], rv[1] - rv_ref[1], rv[2] - rv_ref[2]);
    const double dv = norm3(rv[3] - rv_ref[3], rv[4] - rv_ref[4], rv[5] - rv_ref[5]);
    std::cout << std::setprecision(3) << std::scientific
              << name << ": |dr| = " << dr << " km, |dv| = " << dv << " km/s\n";
    return dr <= tol_r && dv <= tol_v;
}

// max |D Phi_canon D^-1 - Phi_km| / max |Phi_km|
double stm_error(const std::vector<double>& phi_canon, const std::vector<double>& phi_km,
                 const double* scale) {
    double max_diff = 0.0;
    double max_ref = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            const double scaled = scale[i] * phi_canon[6 * i + j] / scale[j];
            max_diff = std::max(max_diff, std::abs(scaled - phi_km[6 * i + j]));
            max_ref = std::max(max_ref, std::abs(phi_km[6 * i + j]));
        }
    }
    return max_diff / max_ref;
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

        // Same Moon-centred case as solve_mee.cpp. et0 stays in ephemeris seconds in both runs.
        const double et0 = 740463475.5117936;
        campanella::SpiceOptions options;
        options.GMs = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
        options.include_srp = true;
        options.srp_A = 34.675;
        options.srp_m = 1850.0;
        options.srp_rhoS = 0.9;
        const campanella::SpiceParameters km(et0, {"MOON", "EARTH", "SUN"}, options);
        options.DU = 100000.0;
        const campanella::SpiceParameters canon(et0, {"MOON", "EARTH", "SUN"}, options);
        const double DU = canon.DU(), VU = canon.VU(), TU = canon.TU();

        const double tf = 20 * 86400.0;
        const double tf_canon = tf / TU;
        const std::vector<double> x0 = {
            -7505.7613773288285, 44128.19419987408, -55792.327508873765,
            0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
        const double cart_scale[6] = {DU, DU, DU, VU, VU, VU};
        const double mee_scale[6] = {DU, 1.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<double> x0_canon(6);
        for (int i = 0; i < 6; ++i) {
            x0_canon[i] = x0[i] / cart_scale[i];
        }
        std::cout << std::setprecision(16)
                  << "DU = " << DU << " km, TU = " << TU << " s, VU = " << VU << " km/s\n"
                  << "tf = " << tf_canon << " TU\n";

        // Both runs carry integration error. atol = 1e-12 is 1e-7 km in canonical units, and the
        // observed differences are below 2e-7 km and 2e-12 km/s.
        const double tol_r = 1e-5;     // km
        const double tol_v = 1e-10;    // km/s
        const double tol_stm = 1e-9;   // relative
        bool ok = true;

        // Cartesian, state and STM
        const campanella::IvpResult rv_km = campanella::solve_ivpstm_rv(km, {0.0, tf}, x0);
        const campanella::IvpResult rv = campanella::solve_ivpstm_rv(canon, {0.0, tf_canon}, x0_canon);
        std::vector<double> rv_f = last(rv.x, 6);
        for (int i = 0; i < 6; ++i) {
            rv_f[i] *= cart_scale[i];
        }
        ok &= report("Cartesian     ", rv_f, last(rv_km.x, 6), tol_r, tol_v);
        const double rv_stm_err = stm_error(last(rv.stm, 36), last(rv_km.stm, 36), cart_scale);
        std::cout << "Cartesian STM : relative difference " << rv_stm_err << "\n";
        ok &= rv_stm_err <= tol_stm;

        // MEE, state and STM. Only p carries a length scale.
        const campanella::IvpResult mee_km = campanella::solve_ivpstm_mee(
            km, {0.0, tf}, campanella::cart2mee(x0, km.mus()[0]));
        const campanella::IvpResult mee = campanella::solve_ivpstm_mee(
            canon, {0.0, tf_canon}, campanella::cart2mee(x0_canon, canon.mus()[0]));
        std::vector<double> rv_mee = campanella::mee2cart(last(mee.x, 6), canon.mus()[0]);
        for (int i = 0; i < 6; ++i) {
            rv_mee[i] *= cart_scale[i];
        }
        ok &= report("MEE           ", rv_mee, campanella::mee2cart(last(mee_km.x, 6), km.mus()[0]),
                     tol_r, tol_v);
        ok &= report("MEE vs Cart.  ", rv_mee, rv_f, tol_r, tol_v);
        const double mee_stm_err = stm_error(last(mee.stm, 36), last(mee_km.stm, 36), mee_scale);
        std::cout << "MEE STM       : relative difference " << mee_stm_err << "\n";
        ok &= mee_stm_err <= tol_stm;

        if (!ok) {
            std::cerr << "canonical and km/s results differ by more than the tolerance\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
