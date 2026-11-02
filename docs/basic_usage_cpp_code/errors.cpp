// What each kind of failure throws. Run from the repository root (for the gravity field file).

#include "campanella.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Run f and print the type and message of the exception it throws
void report(const char* what, const std::function<void()>& f) {
    try {
        f();
        std::printf("%s: no exception\n", what);
    } catch (const std::invalid_argument& ex) {
        std::printf("%s:\n  std::invalid_argument: %s\n", what, ex.what());
    } catch (const std::domain_error& ex) {
        std::printf("%s:\n  std::domain_error: %s\n", what, ex.what());
    } catch (const std::runtime_error& ex) {
        std::printf("%s:\n  std::runtime_error: %s\n", what, ex.what());
    }
}

}  // namespace

int main() {
    const char* spice = std::getenv("SPICE");
    if (spice == nullptr) {
        std::fprintf(stderr, "set SPICE to the kernel directory\n");
        return 1;
    }
    const std::string dir(spice);
    using campanella::SpiceOptions;
    using campanella::SpiceParameters;
    const double et0 = 740463475.5117936;

    report("missing kernel file", [&] { campanella::furnsh(dir + "/spk/no_such_file.bsp"); });

    campanella::furnsh(dir + "/spk/de440.bsp");

    report("no GM in the kernel pool", [&] { SpiceParameters(et0, {"EARTH", "MOON"}); });
    campanella::furnsh(dir + "/pck/gm_de440.tpc");

    report("unknown body", [&] { SpiceParameters(et0, {"EARTH", "PLANET_NINE"}); });
    report("SRP without the Sun", [&] {
        SpiceOptions o;
        o.include_srp = true;
        o.srp_A = 10.0;
        o.srp_m = 500.0;
        SpiceParameters(et0, {"EARTH", "MOON"}, o);
    });

    const SpiceParameters params(et0, {"EARTH", "MOON", "SUN"});
    const std::vector<double> x0 = {7000.0, 0.0, 0.0, 0.0, 7.5, 0.0};
    report("x0 of the wrong length", [&] { campanella::solve_ivp_rv(params, {0.0, 60.0}, {7000.0, 0.0}); });
    report("t_eval outside tspan", [&] {
        campanella::IvpOptions ivp;
        ivp.t_eval = {0.0, 120.0};
        campanella::solve_ivp_rv(params, {0.0, 60.0}, x0, ivp);
    });

    // de440.bsp covers 1550-2650; et = 3e10 s is about the year 2950
    report("epoch outside the ephemeris", [&] {
        const SpiceParameters late(3e10, {"EARTH", "MOON", "SUN"});
        campanella::solve_ivp_rv(late, {0.0, 60.0}, x0);
    });

    report("MEE out of their domain", [&] {
        campanella::eval_mee(params, 0.0, {-7000.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    });
    report("mee2cart with p < 0", [&] { campanella::mee2cart({-7000.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1.0); });

    // a radial fall onto the Earth: the acceleration becomes non-finite at the centre
    report("trajectory through the centre", [&] {
        campanella::solve_ivp_rv(params, {0.0, 3600.0}, {7000.0, 0.0, 0.0, -10.0, 0.0, 0.0});
    });

    // a lunar orbit that dips below the reference radius of the gravity field (1738 km)
    campanella::furnsh(dir + "/pck/moon_pa_de440_200625.bpc");
    campanella::furnsh(dir + "/fk/moon_de440_250416.tf");
    report("below sh_rho_min * R", [&] {
        SpiceOptions o;
        o.sh_file = "test-python/data/gggrx_1200a_sha_n10.tab";
        o.sh_nmax = 10;
        o.sh_frame = "MOON_PA";
        const SpiceParameters moon(et0, {"MOON", "EARTH"}, o);
        // periapsis at about 1700 km from the centre
        campanella::solve_ivp_rv(moon, {0.0, 86400.0}, {1900.0, 0.0, 0.0, 0.0, 1.55, 0.0});
    });

    // eom_* report failures through their return value, and eom_error gives the reason
    std::vector<double> xdot(6);
    const double origin[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int status = campanella::eom_rv(0.0, origin, xdot.data(), const_cast<SpiceParameters*>(&params));
    std::printf("eom_rv at the origin returned %d: %s\n", status, campanella::eom_error(params).c_str());
    campanella::eom_rv(0.0, x0.data(), xdot.data(), const_cast<SpiceParameters*>(&params));
    std::printf("after a successful call, eom_error is \"%s\"\n", campanella::eom_error(params).c_str());
    return 0;
}
