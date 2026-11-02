// campanella tutorial in C++: a geostationary satellite in the ephemeris model.
// The same steps as tutorial.py. Build the C++ library into build-cpp (C++-only configure, see
// docs/build_library.md), then from the repository root:
//
//   c++ -std=c++17 -O2 docs/tutorial_code/tutorial.cpp -I include -L build-cpp/lib -lcampanella -Wl,-rpath,$PWD/build-cpp/lib -o tutorial
//   SPICE=/path/to/spice ./tutorial [path/to/EGM2008.gfc]
//
// The rpath is absolute ($PWD expands when compiling), so ./tutorial runs from any directory.
// SPICE must contain lsk/naif0012.tls, spk/de440.bsp, pck/gm_de440.tpc and
// pck/earth_latest_high_prec.bpc. The gravity field defaults to the degree-10 head of EGM2008
// in test-python/data (relative to the working directory).

#include "campanella.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

double dist3(const double* a, const double* b) {
    return std::sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1])
                     + (a[2] - b[2])*(a[2] - b[2]));
}

// last row of an N x 6 result; the pointer is valid while res lives
const double* final_state(const campanella::IvpResult& res) {
    return &res.x[6*(res.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    const char* spice_env = std::getenv("SPICE");
    if (spice_env == nullptr || spice_env[0] == '\0') {
        std::fprintf(stderr, "SPICE must point at a directory of SPICE kernels\n");
        return 1;
    }
    const std::string spice_dir = std::string(spice_env) + "/";
    const std::string egm2008 = (argc > 1) ? argv[1] : "test-python/data/EGM2008_n10.gfc";

    try {
        // kernels: leap seconds, planetary ephemeris, GMs
        campanella::furnsh(spice_dir + "lsk/naif0012.tls");
        campanella::furnsh(spice_dir + "spk/de440.bsp");
        campanella::furnsh(spice_dir + "pck/gm_de440.tpc");

        // 2025-01-01T00:00:00 UTC, in TDB seconds past J2000 (from spiceypy.str2et in tutorial.py)
        const double et0 = 788961669.1839275;

        // Earth is the central body; the Moon and the Sun are third bodies. GMs from gm_de440.tpc.
        const campanella::SpiceParameters params(et0, {"EARTH", "MOON", "SUN"});
        const double mu_earth = params.GMs()[0];
        std::printf("GMs [km^3/s^2]: %.15g %.15g %.15g\n", params.GMs()[0], params.GMs()[1],
                    params.GMs()[2]);

        // circular equatorial orbit at the geostationary radius
        const double a_geo = 42164.0;
        const double v_geo = std::sqrt(mu_earth / a_geo);
        const std::vector<double> x0 = {a_geo, 0.0, 0.0, 0.0, v_geo, 0.0};
        const double tf = 30*86400.0;

        // one output per day, as t_eval in tutorial.py (and the same in TU for the canonical run)
        campanella::IvpOptions daily;
        for (int day = 0; day <= 30; ++day) {
            daily.t_eval.push_back(day*86400.0);
        }

        // two-body reference vs Earth + Moon + Sun
        campanella::SpiceOptions two_body;
        two_body.GMs = {mu_earth};
        const campanella::SpiceParameters params_2b(et0, {"EARTH"}, two_body);
        const campanella::IvpResult res_2b = campanella::solve_ivp_rv(params_2b, {0.0, tf}, x0, daily);
        const campanella::IvpResult res_nb = campanella::solve_ivp_rv(params, {0.0, tf}, x0, daily);
        std::printf("day 30: |r_nbody - r_2body| = %.3f km (%zu outputs)\n",
                    dist3(final_state(res_nb), final_state(res_2b)), res_nb.size());

        // canonical units: DU = a_geo, so the Earth's scaled GM is 1
        campanella::SpiceOptions canonical;
        canonical.DU = a_geo;
        const campanella::SpiceParameters params_c(et0, {"EARTH", "MOON", "SUN"}, canonical);
        const double DU = params_c.DU(), TU = params_c.TU(), VU = params_c.VU();
        std::printf("DU = %.1f km, TU = %.6f s, VU = %.9f km/s\n", DU, TU, VU);
        const std::vector<double> x0_c = {x0[0]/DU, x0[1]/DU, x0[2]/DU, x0[3]/VU, x0[4]/VU, x0[5]/VU};
        campanella::IvpOptions daily_c;
        for (double t : daily.t_eval) {
            daily_c.t_eval.push_back(t/TU);
        }
        const campanella::IvpResult res_c = campanella::solve_ivp_rv(params_c, {0.0, tf/TU}, x0_c, daily_c);
        const double* xc = final_state(res_c);
        const double r_c_km[3] = {xc[0]*DU, xc[1]*DU, xc[2]*DU};
        std::printf("canonical vs km/s, final position: %.3e km\n", dist3(r_c_km, final_state(res_nb)));

        // the same dynamics in modified equinoctial elements
        const std::vector<double> mee0 = campanella::cart2mee(x0, mu_earth);
        const campanella::IvpResult res_mee = campanella::solve_ivp_mee(params, {0.0, tf}, mee0, daily);
        const double* mf = final_state(res_mee);
        const std::vector<double> rv_mee = campanella::mee2cart(std::vector<double>(mf, mf + 6), mu_earth);
        std::printf("MEE vs Cartesian, final position: %.3e km\n", dist3(rv_mee.data(), final_state(res_nb)));
        std::printf("day 30: e = %.3e, i = %.5f deg\n", std::hypot(mf[1], mf[2]),
                    2*std::atan(std::hypot(mf[3], mf[4]))*180.0/std::acos(-1.0));

        // STM over one day, checked against central differences (scaled by a_geo and v_geo)
        const double t_stm = 86400.0;
        const campanella::IvpResult res_stm = campanella::solve_ivpstm_rv(params, {0.0, t_stm}, x0);
        const double* Phi = &res_stm.stm[36*(res_stm.size() - 1)];   // 6x6, row-major
        const double steps[6] = {1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6};
        const double scale[6] = {a_geo, a_geo, a_geo, v_geo, v_geo, v_geo};
        double max_err = 0.0;
        for (int j = 0; j < 6; ++j) {
            std::vector<double> xp = x0, xm = x0;
            xp[j] += steps[j];
            xm[j] -= steps[j];
            const campanella::IvpResult rp = campanella::solve_ivp_rv(params, {0.0, t_stm}, xp);
            const campanella::IvpResult rm = campanella::solve_ivp_rv(params, {0.0, t_stm}, xm);
            for (int i = 0; i < 6; ++i) {
                const double fd = (final_state(rp)[i] - final_state(rm)[i]) / (2*steps[j]);
                max_err = std::fmax(max_err, std::abs(Phi[6*i + j] - fd)*scale[j]/scale[i]);
            }
        }
        std::printf("Phi[1][0] = d y(1 day) / d x0 = %.6f\n", Phi[6*1 + 0]);
        std::printf("max |Phi - Phi_fd| (scaled) = %.3e\n", max_err);

        // solar radiation pressure: A/m = 0.02 m^2/kg
        campanella::SpiceOptions srp;
        srp.include_srp = true;
        srp.srp_A = 20.0;      // m^2
        srp.srp_m = 1000.0;    // kg
        srp.srp_rhoS = 0.3;
        const campanella::SpiceParameters params_srp(et0, {"EARTH", "MOON", "SUN"}, srp);
        const campanella::IvpResult res_srp = campanella::solve_ivp_mee(params_srp, {0.0, tf}, mee0, daily);
        const double* ms = final_state(res_srp);
        std::printf("day 30: e with SRP = %.3e\n", std::hypot(ms[1], ms[2]));

        // Earth gravity field to degree and order 4, in the ITRF93 body-fixed frame
        campanella::furnsh(spice_dir + "pck/earth_latest_high_prec.bpc");
        campanella::SpiceOptions sh;
        sh.sh_file = egm2008;
        sh.sh_nmax = 4;
        sh.sh_frame = "ITRF93";
        // GMs[0] (from gm_de440.tpc) must match the GM of the file within 1e-5; otherwise the
        // constructor throws std::invalid_argument. Here they agree to 1.5e-8.
        const campanella::SpiceParameters params_sh(et0, {"EARTH", "MOON", "SUN"}, sh);
        const campanella::IvpResult res_sh = campanella::solve_ivp_rv(params_sh, {0.0, tf}, x0, daily);
        std::printf("day 30: |r_sh - r_pointmass| = %.3f km\n", dist3(final_state(res_sh), final_state(res_nb)));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
