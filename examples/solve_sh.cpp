// Propagate a low lunar orbit with a spherical-harmonics gravity field of the Moon.
// Usage: solve_sh [gravity_file [nmax]]
//   gravity_file defaults to test-python/data/gggrx_1200a_sha_n10.tab (degree 10, nmax = 10);
//   with the full GRGM1200A file (gggrx_1200a_sha.tab, PDS) nmax defaults to 50.
// Checks: the Cartesian STM with SH against central differences, and the MEE propagation with
// SH against the Cartesian one. SPICE must point at the kernel directory, with the MOON_PA kernels
// pck/moon_pa_de440_200625.bpc and fk/moon_de440_250416.tf.

#include "campanella.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <stdexcept>
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

// Final state of a result; throws if the integration did not reach tf
std::vector<double> final_state(const campanella::IvpResult& res, double tf) {
    if (res.t.back() != tf) {
        throw std::runtime_error("integration did not reach tf");
    }
    return std::vector<double>(res.x.end() - 6, res.x.end());
}

double norm3(const double* v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

}  // namespace

int main(int argc, char** argv) {
    const char* spice_dir = std::getenv("SPICE");
    if (spice_dir == nullptr || spice_dir[0] == '\0') {
        std::cerr << "SPICE must point at a directory of generic SPICE kernels\n";
        return 1;
    }
    try {
        // Leap seconds, planetary ephemeris, and the lunar principal-axes frame MOON_PA
        campanella::furnsh(kernel_path(spice_dir, "lsk/naif0012.tls"));
        campanella::furnsh(kernel_path(spice_dir, "spk/de440.bsp"));
        campanella::furnsh(kernel_path(spice_dir, "pck/moon_pa_de440_200625.bpc"));
        campanella::furnsh(kernel_path(spice_dir, "fk/moon_de440_250416.tf"));
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    // GRAIL coefficients are given in the principal-axes frame (MOON_PA, not MOON_ME)
    std::string sh_file = std::string(CAMPANELLA_SOURCE_DIR) + "/test-python/data/gggrx_1200a_sha_n10.tab";
    int nmax = 10;
    if (argc > 1) {
        sh_file = argv[1];
        nmax = (argc > 2) ? std::atoi(argv[2]) : 50;
    }
    const std::string frame_bf = "MOON_PA";

    // Circular orbit at 100 km altitude, i = 60 deg in J2000; period about 2 h
    const double et0 = 740463475.5117936;
    const std::vector<double> GM = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
    const std::vector<std::string> naif_ids = {"MOON", "EARTH", "SUN"};
    const double r0 = 1838.0;
    const double v0 = std::sqrt(GM[0] / r0);
    const double inc = 60.0 * std::acos(-1.0) / 180.0;
    const std::vector<double> state0 = {r0, 0.0, 0.0, 0.0, v0 * std::cos(inc), v0 * std::sin(inc)};
    const double tf = 86400.0;
    std::cout << "Gravity field: " << sh_file << ", nmax = " << nmax << ", frame " << frame_bf << "\n";

    bool ok = true;
    try {
        // GMs[0] is the point mass; the SH sum uses the GM and radius of the file
        campanella::SpiceOptions options;
        options.GMs = GM;
        const campanella::SpiceParameters params_pm(et0, naif_ids, options);
        options.sh_file = sh_file;
        options.sh_nmax = nmax;
        options.sh_frame = frame_bf;
        const campanella::SpiceParameters params(et0, naif_ids, options);

        const std::vector<double> xf = final_state(campanella::solve_ivp_rv(params, {0.0, tf}, state0), tf);
        const std::vector<double> xf_pm = final_state(campanella::solve_ivp_rv(params_pm, {0.0, tf}, state0), tf);
        double dr[3] = {xf[0] - xf_pm[0], xf[1] - xf_pm[1], xf[2] - xf_pm[2]};
        std::cout << std::setprecision(16) << "Final state with SH [km, km/s]:\n  ";
        for (int i = 0; i < 6; ++i) {
            std::cout << xf[i] << (i < 5 ? ", " : "\n");
        }
        std::cout << std::setprecision(4)
                  << "Point mass vs SH after 1 day: |dr| = " << norm3(dr) << " km\n";
        ok &= norm3(dr) > 1.0;

        // STM with SH vs central differences over 6 h
        const double tf_stm = 6 * 3600.0;
        const campanella::IvpResult stm = campanella::solve_ivpstm_rv(params, {0.0, tf_stm}, state0);
        const double* phi = &stm.stm[stm.stm.size() - 36];
        const double steps[6] = {1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6};
        const double scale[6] = {r0, r0, r0, v0, v0, v0};
        double phi_fd[36];
        for (int j = 0; j < 6; ++j) {
            std::vector<double> xp = state0, xm = state0;
            xp[j] += steps[j];
            xm[j] -= steps[j];
            const std::vector<double> fp = final_state(campanella::solve_ivp_rv(params, {0.0, tf_stm}, xp), tf_stm);
            const std::vector<double> fm = final_state(campanella::solve_ivp_rv(params, {0.0, tf_stm}, xm), tf_stm);
            for (int i = 0; i < 6; ++i) {
                phi_fd[6 * i + j] = (fp[i] - fm[i]) / (2 * steps[j]);
            }
        }
        double max_err = 0.0;
        for (int i = 0; i < 6; ++i) {
            double row_err = 0.0, row_ref = 0.0;
            for (int j = 0; j < 6; ++j) {
                // scaled STM entries, so position and velocity blocks weigh alike
                const double fd = phi_fd[6 * i + j] * scale[j] / scale[i];
                const double an = phi[6 * i + j] * scale[j] / scale[i];
                row_err = std::max(row_err, std::abs(an - fd));
                row_ref = std::max(row_ref, std::abs(fd));
            }
            max_err = std::max(max_err, row_err / row_ref);
        }
        std::cout << std::scientific << std::setprecision(3)
                  << "STM with SH vs central differences (6 h): " << max_err << " (row-normalised)\n";
        ok &= max_err <= 1e-6;

        // MEE with SH vs the Cartesian propagation
        const std::vector<double> mf = final_state(
            campanella::solve_ivp_mee(params, {0.0, tf}, campanella::cart2mee(state0, GM[0])), tf);
        const std::vector<double> rv = campanella::mee2cart(mf, GM[0]);
        double dr_mee[3] = {rv[0] - xf[0], rv[1] - xf[1], rv[2] - xf[2]};
        std::cout << "MEE vs Cartesian with SH (1 day): |dr| = " << norm3(dr_mee) << " km\n";
        ok &= norm3(dr_mee) <= 1e-5;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    if (!ok) {
        std::cerr << "a spherical-harmonics check failed\n";
        return 1;
    }
    return 0;
}
