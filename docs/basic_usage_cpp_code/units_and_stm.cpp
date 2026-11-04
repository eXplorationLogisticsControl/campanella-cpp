// Moon-centred propagation of the state and the STM in canonical units (DU = 100000 km),
// compared with the same propagation in km and s.

#include "campanella.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

// element (i, j) of the 6x6 STM at output k of res
double stm(const campanella::IvpResult& res, std::size_t k, int i, int j) {
    return res.stm[36*k + 6*i + j];
}

}  // namespace

int main() {
    const char* spice = std::getenv("SPICE");
    if (spice == nullptr) {
        std::fprintf(stderr, "set SPICE to the kernel directory\n");
        return 1;
    }
    const std::string dir(spice);
    try {
        campanella::furnsh(dir + "/spk/de440.bsp");

        const double et0 = 740463475.5117936;
        const std::vector<std::string> bodies = {"MOON", "EARTH", "SUN"};
        campanella::SpiceOptions options;
        // km^3/s^2: the GM set of the test suite and examples/, not the DE440 values
        options.GMs = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
        options.include_srp = true;
        options.srp_A = 34.675;     // m^2
        options.srp_m = 1850.0;     // kg
        options.srp_rhoS = 0.9;

        const campanella::SpiceParameters dim(et0, bodies, options);       // km, s
        options.DU = 100000.0;                                              // km
        const campanella::SpiceParameters can(et0, bodies, options);       // DU, TU
        const double DU = can.DU(), TU = can.TU(), VU = can.VU();
        std::printf("DU = %g km, TU = %.6f s (%.4f days), VU = %.9f km/s\n", DU, TU, TU/86400.0, VU);
        std::printf("mus = %.15g %.15g %.15g\n", can.mus()[0], can.mus()[1], can.mus()[2]);

        const std::vector<double> x0_km = {
            -7505.7613773288285, 44128.19419987408, -55792.327508873765,
            0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
        std::vector<double> x0_can(6);
        for (int i = 0; i < 6; i++) {
            x0_can[i] = x0_km[i] / (i < 3 ? DU : VU);
        }
        const double tf_s = 20*86400.0;

        const campanella::IvpResult a = campanella::solve_ivpstm_rv(dim, {0.0, tf_s}, x0_km);
        const campanella::IvpResult b = campanella::solve_ivpstm_rv(can, {0.0, tf_s / TU}, x0_can);
        const std::size_t na = a.size() - 1, nb = b.size() - 1;
        std::printf("accepted steps: km/s run %zu, canonical run %zu\n", a.size() - 1, b.size() - 1);
        std::printf("res.x.size() = %zu = 6 N, res.stm.size() = %zu = 36 N\n", b.x.size(), b.stm.size());

        // states back to km and km/s
        double dx = 0.0;
        for (int i = 0; i < 6; i++) {
            const double scale = (i < 3) ? DU : VU;
            dx = std::max(dx, std::abs(b.x[6*nb + i]*scale - a.x[6*na + i]) / (i < 3 ? 1.0 : 1e-6));
        }
        std::printf("max state difference: %.3e (km, or mm/s)\n", dx);

        // Phi_km = D Phi_can D^-1, with D = diag(DU, DU, DU, VU, VU, VU)
        double dphi = 0.0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                const double si = (i < 3) ? DU : VU, sj = (j < 3) ? DU : VU;
                const double ref = stm(a, na, i, j);
                dphi = std::max(dphi, std::abs(si*stm(b, nb, i, j)/sj - ref) / std::max(1.0, std::abs(ref)));
            }
        }
        std::printf("max STM difference (relative): %.3e\n", dphi);
        std::printf("Phi(tf, t0) in km and s, first row: %.6f %.6f %.6f %.3f %.3f %.3f\n",
                    stm(a, na, 0, 0), stm(a, na, 0, 1), stm(a, na, 0, 2),
                    stm(a, na, 0, 3), stm(a, na, 0, 4), stm(a, na, 0, 5));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
