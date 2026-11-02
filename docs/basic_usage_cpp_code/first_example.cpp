// Earth-centred propagation with the Moon and the Sun as third bodies, in km and s.
// Build: see build_and_run.sh. Run with SPICE set to the kernel directory.

#include "campanella.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

int main() {
    const char* spice = std::getenv("SPICE");
    if (spice == nullptr) {
        std::fprintf(stderr, "set SPICE to the kernel directory\n");
        return 1;
    }
    const std::string dir(spice);
    try {
        // load the kernels into the CSPICE inside libcampanella; no leap-second kernel is needed
        campanella::furnsh(dir + "/spk/de440.bsp");
        campanella::furnsh(dir + "/pck/gm_de440.tpc");   // GMs, since options.GMs is left empty

        const double et0 = 740463475.5117936;           // TDB seconds past J2000
        const campanella::SpiceParameters params(et0, {"399", "301", "10"});   // Earth, Moon, Sun

        std::printf("codes: %d %d %d\n", params.naif_codes()[0], params.naif_codes()[1],
                    params.naif_codes()[2]);
        std::printf("GMs [km^3/s^2]: %.6f %.6f %.6e\n", params.GMs()[0], params.GMs()[1],
                    params.GMs()[2]);
        std::printf("DU = %g km, TU = %g s, VU = %g km/s\n", params.DU(), params.TU(), params.VU());

        // every accepted step
        const std::vector<double> x0 = {7000.0, 0.0, 0.0, 0.0, 7.5, 0.0};   // km, km/s
        const campanella::IvpResult res = campanella::solve_ivp_rv(params, {0.0, 3600.0}, x0);
        const std::size_t N = res.size();
        const double* xf = &res.x[6*(N - 1)];            // row N-1 of the N x 6 array
        std::printf("N = %zu, t[0] = %g, t[N-1] = %g\n", N, res.t[0], res.t[N - 1]);
        std::printf("x(tf) = %.9f %.9f %.9f %.12f %.12f %.12f\n",
                    xf[0], xf[1], xf[2], xf[3], xf[4], xf[5]);

        // output at chosen times, with tighter relative tolerance than the default
        campanella::IvpOptions ivp;
        ivp.rtol = 1e-13;
        ivp.t_eval = {0.0, 900.0, 1800.0, 2700.0, 3600.0};
        const campanella::IvpResult sampled = campanella::solve_ivp_rv(params, {0.0, 3600.0}, x0, ivp);
        for (std::size_t k = 0; k < sampled.size(); k++) {
            std::printf("t = %6.0f s  r = [%13.6f %13.6f %13.6f] km\n", sampled.t[k],
                        sampled.x[6*k], sampled.x[6*k + 1], sampled.x[6*k + 2]);
        }

        // backward from the final state recovers x0
        const std::vector<double> xf_vec(xf, xf + 6);
        const campanella::IvpResult back = campanella::solve_ivp_rv(params, {3600.0, 0.0}, xf_vec);
        const double* xb = &back.x[6*(back.size() - 1)];
        std::printf("backward: t = %g, |r - r0| = %.3e km\n", back.t.back(),
                    std::abs(xb[0] - x0[0]) + std::abs(xb[1] - x0[1]) + std::abs(xb[2] - x0[2]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
