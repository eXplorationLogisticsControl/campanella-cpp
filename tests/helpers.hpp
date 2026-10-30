// Shared helpers of the GoogleTest suite.
//
// Kernels are read from $SPICE (lsk/naif0012.tls, spk/de440.bsp, pck/gm_de440.tpc, and the
// MOON_PA / ITRF93 frame kernels). Tests that need them call REQUIRE_KERNELS(): without $SPICE
// they are skipped, unless CAMPANELLA_REQUIRE_KERNELS is set (as in CI), in which case they fail.

#ifndef CAMPANELLA_TESTS_HELPERS_HPP
#define CAMPANELLA_TESTS_HELPERS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "campanella.hpp"

namespace campanella_test {

// Moon-centred case shared with test-python
constexpr double ET0 = 740463475.5117936;
const std::vector<double> STATE0 = {-7505.7613773288285, 44128.19419987408, -55792.327508873765,
                                    0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
const std::vector<double> GM_LIST = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
const std::vector<std::string> BODIES = {"MOON", "EARTH", "SUN"};
constexpr double MU_MOON = 4.902800582147760e+03;
constexpr double TF = 20 * 86400.0;

// Low lunar orbit, r = 1838 km, i = 60 deg
inline std::vector<double> state_llo() {
    const double r0 = 1838.0;
    const double v0 = std::sqrt(MU_MOON / r0);
    const double inc = 60.0 * std::acos(-1.0) / 180.0;
    return {r0, 0.0, 0.0, 0.0, v0 * std::cos(inc), v0 * std::sin(inc)};
}

inline std::string data_path(const std::string& name) {
    return std::string(CAMPANELLA_SOURCE_DIR) + "/test-python/data/" + name;
}

inline const char* spice_dir() {
    const char* dir = std::getenv("SPICE");
    return (dir != nullptr && dir[0] != '\0') ? dir : nullptr;
}

inline bool kernels_required() {
    const char* req = std::getenv("CAMPANELLA_REQUIRE_KERNELS");
    return req != nullptr && req[0] != '\0' && std::string(req) != "0";
}

// Whether load_kernels has loaded the kernels into this process (kclear_c must reset it)
inline bool& kernels_loaded() {
    static bool loaded = false;
    return loaded;
}

// Load the kernels once per process; false if $SPICE is not set
inline bool load_kernels() {
    bool& loaded = kernels_loaded();
    if (loaded) {
        return true;
    }
    const char* dir = spice_dir();
    if (dir == nullptr) {
        return false;
    }
    for (const char* kernel : {"lsk/naif0012.tls", "spk/de440.bsp", "pck/gm_de440.tpc",
                               "pck/moon_pa_de440_200625.bpc", "fk/moon_de440_250416.tf",
                               "pck/earth_latest_high_prec.bpc"}) {
        campanella::furnsh(std::string(dir) + "/" + kernel);
    }
    loaded = true;
    return true;
}

#define REQUIRE_KERNELS()                                                              \
    do {                                                                               \
        if (!campanella_test::load_kernels()) {                                        \
            if (campanella_test::kernels_required()) {                                 \
                FAIL() << "SPICE is not set, but CAMPANELLA_REQUIRE_KERNELS is";       \
            }                                                                          \
            GTEST_SKIP() << "SPICE is not set";                                        \
        }                                                                              \
    } while (0)

// Options with explicit GMs (km^3/s^2)
inline campanella::SpiceOptions with_gms(const std::vector<double>& gms) {
    campanella::SpiceOptions o;
    o.GMs = gms;
    return o;
}

inline void add_srp(campanella::SpiceOptions& o) {
    o.include_srp = true;
    o.srp_A = 34.675;
    o.srp_m = 1850.0;
    o.srp_rhoS = 0.9;
    o.srp_rhoD = 0.3;
}

inline void add_moon_sh(campanella::SpiceOptions& o) {
    o.sh_file = data_path("gggrx_1200a_sha_n10.tab");
    o.sh_nmax = 10;
    o.sh_frame = "MOON_PA";
}

inline std::vector<double> final_state(const campanella::IvpResult& r) {
    return std::vector<double>(r.x.end() - 6, r.x.end());
}

inline std::vector<double> final_stm(const campanella::IvpResult& r) {
    return std::vector<double>(r.stm.end() - 36, r.stm.end());
}

// max over rows of max_j |M_ij - R_ij| / max_j |R_ij|, for 6x6 row-major matrices,
// after scaling both as diag(1/s) M diag(s)
inline double row_normalised_error(const std::vector<double>& M, const std::vector<double>& R,
                                   const std::array<double, 6>& s = {1, 1, 1, 1, 1, 1}) {
    double err = 0.0;
    for (int i = 0; i < 6; i++) {
        double diff = 0.0, ref = 0.0;
        for (int j = 0; j < 6; j++) {
            const double m = M[6 * i + j] * s[j] / s[i];
            const double r = R[6 * i + j] * s[j] / s[i];
            diff = std::max(diff, std::abs(m - r));
            ref = std::max(ref, std::abs(r));
        }
        err = std::max(err, diff / ref);
    }
    return err;
}

// Central differences of f: R^6 -> R^6 with steps h, as a 6x6 row-major matrix
inline std::vector<double> jacobian_fd(const std::function<std::vector<double>(const std::vector<double>&)>& f,
                                       const std::vector<double>& x, const std::array<double, 6>& h) {
    std::vector<double> J(36);
    for (int j = 0; j < 6; j++) {
        std::vector<double> xp = x, xm = x;
        xp[j] += h[j];
        xm[j] -= h[j];
        const std::vector<double> fp = f(xp), fm = f(xm);
        for (int i = 0; i < 6; i++) {
            J[6 * i + j] = (fp[i] - fm[i]) / (2 * h[j]);
        }
    }
    return J;
}

inline double norm3_diff(const std::vector<double>& a, const std::vector<double>& b, int offset) {
    double s = 0.0;
    for (int i = 0; i < 3; i++) {
        s += (a[offset + i] - b[offset + i]) * (a[offset + i] - b[offset + i]);
    }
    return std::sqrt(s);
}

}  // namespace campanella_test

#endif  // CAMPANELLA_TESTS_HELPERS_HPP
