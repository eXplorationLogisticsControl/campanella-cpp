// Propagate the Moon-centered test case in canonical units and check each result
// against the same propagator run in km and seconds.
// LU = 100000 km and TU = LU / sqrt(MU_MOON / LU), so the scaled lunar GM is 1.
// SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <algorithm>
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

// [x, y, z, vx, vy, vz] followed by Phi(t0, t0) = I, stored row-major.
std::vector<double> with_identity_stm(std::vector<double> x0) {
    x0.resize(42, 0.0);
    for (int i = 0; i < 6; ++i) {
        x0[6 + i * 6 + i] = 1.0;
    }
    return x0;
}

// Final row of a trajectory; false if the integration stopped before tf.
template <std::size_t N, typename Traj>
bool final_row(const Traj& traj, double tf, std::array<double, N>& row) {
    if (traj.empty()) {
        return false;
    }
    row = tuple_head<N>(traj.back());
    return row[0] == tf;
}

// Report a position/velocity difference in km and km/s; true if within tolerance.
bool report(const char* name, const double* rv, const double* rv_ref, double tol_r, double tol_v) {
    const double dr = norm3(rv[0] - rv_ref[0], rv[1] - rv_ref[1], rv[2] - rv_ref[2]);
    const double dv = norm3(rv[3] - rv_ref[3], rv[4] - rv_ref[4], rv[5] - rv_ref[5]);
    std::cout << std::setprecision(3) << std::scientific
              << name << ": |dr| = " << dr << " km, |dv| = " << dv << " km/s\n";
    return dr <= tol_r && dv <= tol_v;
}

// max |Phi_canon_scaled - Phi_km| / max |Phi_km|, with Phi_canon_scaled = D Phi_canon D^-1
double stm_error(const double* phi_canon, const double* phi_km, const double* scale) {
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
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    // Same Moon-centered case as solve_gve.cpp. et0 stays in ephemeris seconds in both runs.
    const double et0 = 740463475.5117936;
    const std::vector<double> GM = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
    const std::vector<std::string> BODIES = {"MOON", "EARTH", "SUN"};
    const double tf = 20 * 86400.0;
    const std::vector<double> state0 = {
        -7505.7613773288285, 44128.19419987408, -55792.327508873765,
        0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
    const double A = 34.675, m = 1850.0, rhoS = 0.9, rhoD = 0.0;

    // Canonical scales. GM_i / MU_MOON equals GM_i TU^2 / LU^3.
    const double mu_moon = GM[0];
    const double LU = 100000.0;
    const double VU = std::sqrt(mu_moon / LU);
    const double TU = LU / VU;
    const double tf_canon = tf / TU;
    std::vector<double> GM_canon;
    for (double gm : GM) {
        GM_canon.push_back(gm / mu_moon);
    }
    const double cart_scale[6] = {LU, LU, LU, VU, VU, VU};
    const double mee_scale[6] = {LU, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> state0_canon(6);
    for (int i = 0; i < 6; ++i) {
        state0_canon[i] = state0[i] / cart_scale[i];
    }
    std::cout << std::setprecision(16)
              << "LU = " << LU << " km, TU = " << TU << " s, VU = " << VU << " km/s\n"
              << "tf = " << tf_canon << " TU\n";

    // Both runs carry integration error. abstol = 1e-12 is 1e-7 km in canonical units, and the
    // observed differences are below 2e-7 km and 2e-12 km/s.
    const double tol_r = 1e-5;     // km
    const double tol_v = 1e-10;    // km/s
    const double tol_stm = 1e-9;   // relative
    bool ok = true;

    // N-body, state and STM
    campanella::PropagatorNbody nbody_km(GM, BODIES, "J2000", 1.0, 1.0);
    campanella::PropagatorNbody nbody(GM_canon, BODIES, "J2000", LU, TU);
    std::array<double, 43> nbody_stm_km{}, nbody_stm{};
    if (!final_row(nbody_km.solve_ivp_stm(et0, {0.0, tf}, with_identity_stm(state0)), tf, nbody_stm_km) ||
        !final_row(nbody.solve_ivp_stm(et0, {0.0, tf_canon}, with_identity_stm(state0_canon)),
                   tf_canon, nbody_stm)) {
        std::cerr << "N-body STM integration did not reach tf\n";
        return 1;
    }
    double rv[6];
    for (int i = 0; i < 6; ++i) {
        rv[i] = nbody_stm[1 + i] * cart_scale[i];
    }
    ok &= report("N-body        ", rv, &nbody_stm_km[1], tol_r, tol_v);
    const double nbody_stm_err = stm_error(&nbody_stm[7], &nbody_stm_km[7], cart_scale);
    std::cout << "N-body STM    : relative difference " << nbody_stm_err << "\n";
    ok &= nbody_stm_err <= tol_stm;

    // N-body + SRP
    campanella::PropagatorNbodySrp srp_km(GM, BODIES, "J2000", A, m, rhoS, rhoD, 1.0, 1.0);
    campanella::PropagatorNbodySrp srp(GM_canon, BODIES, "J2000", A, m, rhoS, rhoD, LU, TU);
    std::array<double, 7> srp_f_km{}, srp_f{};
    if (!final_row(srp_km.solve_ivp(et0, {0.0, tf}, state0), tf, srp_f_km) ||
        !final_row(srp.solve_ivp(et0, {0.0, tf_canon}, state0_canon), tf_canon, srp_f)) {
        std::cerr << "N-body + SRP integration did not reach tf\n";
        return 1;
    }
    for (int i = 0; i < 6; ++i) {
        rv[i] = srp_f[1 + i] * cart_scale[i];
    }
    ok &= report("N-body + SRP  ", rv, &srp_f_km[1], tol_r, tol_v);

    // GVE in MEE, state and STM. Only p carries a length scale.
    campanella::PropagatorGve gve_km(GM, BODIES, "J2000", A, m, rhoS, rhoD, 1.0, 1.0);
    campanella::PropagatorGve gve(GM_canon, BODIES, "J2000", A, m, rhoS, rhoD, LU, TU);
    const std::vector<double> mee0_km = campanella::cart2mee(state0, mu_moon);
    const std::vector<double> mee0 = campanella::cart2mee(state0_canon, 1.0);
    std::array<double, 43> gve_stm_km{}, gve_stm{};
    if (!final_row(gve_km.solve_ivp_stm(et0, {0.0, tf}, with_identity_stm(mee0_km)), tf, gve_stm_km) ||
        !final_row(gve.solve_ivp_stm(et0, {0.0, tf_canon}, with_identity_stm(mee0)), tf_canon, gve_stm)) {
        std::cerr << "GVE STM integration did not reach tf\n";
        return 1;
    }
    const std::vector<double> rv_gve = campanella::mee2cart(
        {gve_stm[1], gve_stm[2], gve_stm[3], gve_stm[4], gve_stm[5], gve_stm[6]}, 1.0);
    const std::vector<double> rv_gve_km = campanella::mee2cart(
        {gve_stm_km[1], gve_stm_km[2], gve_stm_km[3], gve_stm_km[4], gve_stm_km[5], gve_stm_km[6]},
        mu_moon);
    for (int i = 0; i < 6; ++i) {
        rv[i] = rv_gve[i] * cart_scale[i];
    }
    ok &= report("GVE           ", rv, rv_gve_km.data(), tol_r, tol_v);
    // The canonical GVE also agrees with the canonical Cartesian N-body + SRP run
    double rv_srp[6];
    for (int i = 0; i < 6; ++i) {
        rv_srp[i] = srp_f[1 + i] * cart_scale[i];
    }
    ok &= report("GVE vs Cart.  ", rv, rv_srp, tol_r, tol_v);
    const double gve_stm_err = stm_error(&gve_stm[7], &gve_stm_km[7], mee_scale);
    std::cout << "GVE STM       : relative difference " << gve_stm_err << "\n";
    ok &= gve_stm_err <= tol_stm;

    if (!ok) {
        std::cerr << "canonical and km/s results differ by more than the tolerance\n";
        return 1;
    }
    return 0;
}
