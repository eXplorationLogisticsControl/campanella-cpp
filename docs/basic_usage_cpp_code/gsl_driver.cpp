// Drive campanella::eom_rv with the user's own GSL, and compare with solve_ivp_rv.
// Needs GSL headers and libraries in addition to libcampanella (see build_and_run.sh).

#include "campanella.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

int main() {
    const char* spice = std::getenv("SPICE");
    if (spice == nullptr) {
        std::fprintf(stderr, "set SPICE to the kernel directory\n");
        return 1;
    }
    const std::string dir(spice);
    try {
        campanella::furnsh(dir + "/spk/de440.bsp");
        campanella::furnsh(dir + "/pck/gm_de440.tpc");
        // not const: eom_rv writes the scratch space of params
        campanella::SpiceParameters params(740463475.5117936, {"EARTH", "MOON", "SUN"});

        gsl_odeiv2_system sys = {campanella::eom_rv, nullptr, 6, &params};
        gsl_odeiv2_driver* d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, 1.0, 1e-12, 1e-14);
        double t = 0.0;
        double y[6] = {7000.0, 0.0, 0.0, 0.0, 7.5, 0.0};
        const int status = gsl_odeiv2_driver_apply(d, &t, 3600.0, y);
        gsl_odeiv2_driver_free(d);
        if (status != GSL_SUCCESS) {
            std::fprintf(stderr, "GSL status %d: %s\n", status, campanella::eom_error(params).c_str());
            return 1;
        }

        const campanella::IvpResult res =
            campanella::solve_ivp_rv(params, {0.0, 3600.0}, {7000.0, 0.0, 0.0, 0.0, 7.5, 0.0});
        const double* xf = &res.x[6*(res.size() - 1)];
        double diff = 0.0;
        for (int i = 0; i < 3; i++) {
            diff = std::fmax(diff, std::abs(y[i] - xf[i]));
        }
        std::printf("GSL driver: r(tf) = %.9f %.9f %.9f km\n", y[0], y[1], y[2]);
        std::printf("max difference to solve_ivp_rv: %.3e km\n", diff);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
