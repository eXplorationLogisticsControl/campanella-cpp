// The same Moon-centred propagation in Cartesian coordinates and in MEE, and the mapping of the
// MEE STM to Cartesian coordinates with mee2cart_jacobian.

#include "campanella.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

using Mat6 = std::array<double, 36>;   // row-major

Mat6 matmul(const double* A, const double* B) {
    Mat6 C{};
    for (int i = 0; i < 6; i++)
        for (int k = 0; k < 6; k++)
            for (int j = 0; j < 6; j++)
                C[6*i + j] += A[6*i + k]*B[6*k + j];
    return C;
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

        // the GM set of the test suite and examples/ (Moon, Earth, Sun), not the DE440 values
        campanella::SpiceOptions options;
        options.GMs = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
        const campanella::SpiceParameters params(740463475.5117936, {"MOON", "EARTH", "SUN"}, options);
        const double mu = params.mus()[0];

        const std::vector<double> rv0 = {
            -7505.7613773288285, 44128.19419987408, -55792.327508873765,
            0.07193326412942684, 0.018407794390508374, 0.004868437733535174};
        const std::vector<double> mee0 = campanella::cart2mee(rv0, mu);
        std::printf("mee0 = %.6f %.9f %.9f %.9f %.9f %.9f\n",
                    mee0[0], mee0[1], mee0[2], mee0[3], mee0[4], mee0[5]);

        const std::vector<double> tspan = {0.0, 20*86400.0};
        const campanella::IvpResult rv = campanella::solve_ivpstm_rv(params, tspan, rv0);
        const campanella::IvpResult me = campanella::solve_ivpstm_mee(params, tspan, mee0);
        const std::size_t nr = rv.size() - 1, nm = me.size() - 1;
        std::printf("accepted steps: Cartesian %zu, MEE %zu\n", rv.size() - 1, me.size() - 1);

        // final MEE back to Cartesian; L is not wrapped by the solver
        const std::vector<double> meef(me.x.begin() + 6*nm, me.x.begin() + 6*nm + 6);
        const std::vector<double> rvf = campanella::mee2cart(meef, mu);
        std::printf("L(tf) = %.6f rad\n", meef[5]);
        double dr = 0.0;
        for (int i = 0; i < 3; i++) {
            dr = std::max(dr, std::abs(rvf[i] - rv.x[6*nr + i]));
        }
        std::printf("max position difference: %.3e km\n", dr);

        // Phi_rv(tf, t0) = J(tf) Phi_mee(tf, t0) J(t0)^-1. Check instead J(tf) Phi_mee = Phi_rv J(t0).
        const std::vector<double> J0 = campanella::mee2cart_jacobian(mee0, mu);
        const std::vector<double> Jf = campanella::mee2cart_jacobian(meef, mu);
        const Mat6 lhs = matmul(Jf.data(), &me.stm[36*nm]);
        const Mat6 rhs = matmul(&rv.stm[36*nr], J0.data());
        double rel = 0.0;
        for (int i = 0; i < 36; i++) {
            rel = std::max(rel, std::abs(lhs[i] - rhs[i]) / std::max(1.0, std::abs(rhs[i])));
        }
        std::printf("max relative STM difference: %.3e\n", rel);

        // accepted steps of the four solvers, in km and s and with DU = p(t0)
        std::printf("accepted steps            ivp_rv  ivp_mee  ivpstm_rv  ivpstm_mee\n");
        auto count = [&](const char* label, const campanella::SpiceParameters& p,
                         const campanella::IvpOptions& ivp) {
            const double DU = p.DU(), VU = p.VU();
            std::vector<double> x(6);
            for (int i = 0; i < 6; i++) {
                x[i] = rv0[i] / (i < 3 ? DU : VU);
            }
            const std::vector<double> m = campanella::cart2mee(x, p.mus()[0]);
            const std::vector<double> ts = {0.0, tspan[1] / p.TU()};
            std::printf("%-24s %7zu %8zu %10zu %11zu\n", label,
                        campanella::solve_ivp_rv(p, ts, x, ivp).size() - 1,
                        campanella::solve_ivp_mee(p, ts, m, ivp).size() - 1,
                        campanella::solve_ivpstm_rv(p, ts, x, ivp).size() - 1,
                        campanella::solve_ivpstm_mee(p, ts, m, ivp).size() - 1);
        };
        campanella::IvpOptions defaults, loose_atol, loose_rtol;
        loose_atol.atol = 1e-10;
        loose_rtol.rtol = 1e-12;
        count("km, s", params, defaults);
        count("km, s, atol = 1e-10", params, loose_atol);
        count("km, s, rtol = 1e-12", params, loose_rtol);
        campanella::SpiceOptions canonical = options;
        canonical.DU = mee0[0];
        count("DU = p(t0)", campanella::SpiceParameters(params.et0(), params.naif_ids(), canonical), defaults);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
