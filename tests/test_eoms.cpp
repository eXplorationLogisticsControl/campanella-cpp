// Equations of motion: Jacobians against central differences for every force combination,
// eom_* as GSL right-hand sides, and their return codes

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include "helpers.hpp"

using namespace campanella_test;
using campanella::SpiceOptions;
using campanella::SpiceParameters;

namespace {

struct Forces {
    const char* name;
    bool srp;
    bool sh;
};

const Forces FORCE_CASES[] = {
    {"third bodies", false, false},
    {"third bodies + SRP", true, false},
    {"third bodies + SH", false, true},
    {"third bodies + SRP + SH", true, true},
};

SpiceParameters moon_params(const Forces& f) {
    SpiceOptions o = with_gms(GM_LIST);
    if (f.srp) {
        add_srp(o);
    }
    if (f.sh) {
        add_moon_sh(o);
    }
    return SpiceParameters(ET0, BODIES, o);
}

std::vector<double> xdot_rv(const SpiceParameters& p, double t, const std::vector<double>& x) {
    const campanella::EomEval e = campanella::eval_rv(p, t, x);
    return std::vector<double>(e.xdot.begin(), e.xdot.end());
}

std::vector<double> xdot_mee(const SpiceParameters& p, double t, const std::vector<double>& x) {
    const campanella::EomEval e = campanella::eval_mee(p, t, x);
    return std::vector<double>(e.xdot.begin(), e.xdot.end());
}

// The position block G of A = [[0, I], [G, 0]] against central differences
double rv_gradient_error(const SpiceParameters& p, double t, const std::vector<double>& x, double h) {
    const campanella::EomEval e = campanella::eval_rv(p, t, x);
    const std::vector<double> A_fd = jacobian_fd(
        [&](const std::vector<double>& y) { return xdot_rv(p, t, y); }, x, {h, h, h, 1e-6, 1e-6, 1e-6});
    double diff = 0.0, ref = 0.0;
    for (int i = 3; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            diff = std::max(diff, std::abs(e.A[6 * i + j] - A_fd[6 * i + j]));
            ref = std::max(ref, std::abs(A_fd[6 * i + j]));
        }
    }
    // the kinematic rows are exact
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            EXPECT_EQ(e.A[6 * i + j], (j == i + 3) ? 1.0 : 0.0);
        }
    }
    return diff / ref;
}

double mee_jacobian_error(const SpiceParameters& p, double t, const std::vector<double>& x) {
    const campanella::EomEval e = campanella::eval_mee(p, t, x);
    const std::vector<double> A_fd = jacobian_fd(
        [&](const std::vector<double>& y) { return xdot_mee(p, t, y); }, x,
        {1e-6 * x[0], 1e-7, 1e-7, 1e-7, 1e-7, 1e-7});
    std::vector<double> A(e.A.begin(), e.A.end());
    // scale the p column so that all columns are dimensionless
    return row_normalised_error(A, A_fd, {1.0 / x[0], 1, 1, 1, 1, 1});
}

}  // namespace

TEST(Eoms, RvJacobianVsFiniteDifferences) {
    REQUIRE_KERNELS();
    for (const Forces& f : FORCE_CASES) {
        const SpiceParameters p = moon_params(f);
        // low lunar orbit (SH matters) and the 20-day case (third bodies and SRP matter)
        EXPECT_LE(rv_gradient_error(p, 3600.0, state_llo(), 1e-3), 1e-7) << f.name << ", LLO";
        EXPECT_LE(rv_gradient_error(p, 5 * 86400.0, STATE0, 1.0), 1e-7) << f.name << ", STATE0";
    }
}

TEST(Eoms, MeeJacobianVsFiniteDifferences) {
    REQUIRE_KERNELS();
    for (const Forces& f : FORCE_CASES) {
        const SpiceParameters p = moon_params(f);
        EXPECT_LE(mee_jacobian_error(p, 3600.0, campanella::cart2mee(state_llo(), MU_MOON)), 1e-6)
            << f.name << ", LLO";
        EXPECT_LE(mee_jacobian_error(p, 5 * 86400.0, campanella::cart2mee(STATE0, MU_MOON)), 1e-6)
            << f.name << ", STATE0";
    }
}

TEST(Eoms, SrpGradientAloneVsFiniteDifferences) {
    // a massless Sun: the perturbation is SRP only, so G - G_central is the SRP gradient
    REQUIRE_KERNELS();
    SpiceOptions o = with_gms({MU_MOON, 0.0});
    add_srp(o);
    const SpiceParameters p(ET0, {"MOON", "SUN"}, o);
    const SpiceParameters p_grav(ET0, {"MOON", "SUN"}, with_gms({MU_MOON, 0.0}));
    const campanella::EomEval e = campanella::eval_rv(p, 0.0, STATE0);
    const campanella::EomEval e_grav = campanella::eval_rv(p_grav, 0.0, STATE0);
    auto a_srp = [&](const std::vector<double>& x) {
        const std::vector<double> a = xdot_rv(p, 0.0, x), g = xdot_rv(p_grav, 0.0, x);
        return std::vector<double>{a[3] - g[3], a[4] - g[4], a[5] - g[5], 0, 0, 0};
    };
    const std::vector<double> J = jacobian_fd(a_srp, STATE0, {100.0, 100.0, 100.0, 1, 1, 1});
    double diff = 0.0, ref = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const double G_srp = e.A[6 * (3 + i) + j] - e_grav.A[6 * (3 + i) + j];
            diff = std::max(diff, std::abs(G_srp - J[6 * i + j]));
            ref = std::max(ref, std::abs(J[6 * i + j]));
        }
    }
    ASSERT_GT(ref, 0.0);
    EXPECT_LE(diff / ref, 1e-6);
    EXPECT_LE(mee_jacobian_error(p, 0.0, campanella::cart2mee(STATE0, MU_MOON)), 1e-6);
}

TEST(Eoms, SunCentredSrpJacobian) {
    REQUIRE_KERNELS();
    SpiceOptions o = with_gms({1.3271244004127942e11, 398600.435507});
    add_srp(o);
    const SpiceParameters p(ET0, {"SUN", "EARTH"}, o);
    const std::vector<double> x = {1.0e8, 9.0e7, 1.0e6, -18.0, 20.0, 0.5};
    EXPECT_LE(rv_gradient_error(p, 0.0, x, 1e3), 1e-7);
    EXPECT_LE(mee_jacobian_error(p, 0.0, campanella::cart2mee(x, p.mus()[0])), 1e-6);
}

TEST(Eoms, HeliocentricSrpMagnitude) {
    // the Sun as central body: a_srp = (P0/c) Cr (A/m) (AU/r)^2 r/|r|, radial and outward
    const double gm_sun = 1.3271244004127942e11;
    SpiceOptions o = with_gms({gm_sun});
    add_srp(o);
    const SpiceParameters p(ET0, {"SUN"}, o);
    const SpiceParameters p_grav(ET0, {"SUN"}, with_gms({gm_sun}));
    const std::vector<double> x = {1.2e8, -0.5e8, 0.3e8, 10.0, 25.0, 1.0};
    const campanella::EomEval e = campanella::eval_rv(p, 0.0, x), g = campanella::eval_rv(p_grav, 0.0, x);
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const double Cr = 1 + o.srp_rhoS + 5 * o.srp_rhoD / 3;
    const double au = 149597870.7;
    const double a_mag = 1367.0 / 299792458.0 * Cr * o.srp_A / o.srp_m / 1e3 * (au / r) * (au / r);
    for (int i = 0; i < 3; i++) {
        // difference of two ~6e-3 km/s^2 accelerations: cancellation error ~1e-13 / 1e-10
        EXPECT_NEAR(e.xdot[3 + i] - g.xdot[3 + i], a_mag * x[i] / r, 1e-9 * a_mag) << i;
    }
}

TEST(Eoms, EomFunctionsAgreeWithEval) {
    REQUIRE_KERNELS();
    SpiceParameters p = moon_params(FORCE_CASES[3]);
    const std::vector<double> x = state_llo();
    const std::vector<double> m = campanella::cart2mee(x, MU_MOON);
    for (int k = 0; k < 2; k++) {
        const bool mee = (k == 1);
        const std::vector<double>& y = mee ? m : x;
        const campanella::EomEval e = mee ? campanella::eval_mee(p, 100.0, y) : campanella::eval_rv(p, 100.0, y);
        double xdot[6];
        ASSERT_EQ((mee ? campanella::eom_mee : campanella::eom_rv)(100.0, y.data(), xdot, &p), GSL_SUCCESS);
        // 42 states with an arbitrary Phi: Phi_dot = A Phi
        double y42[42], ydot[42];
        for (int i = 0; i < 6; i++) {
            y42[i] = y[i];
        }
        for (int i = 0; i < 36; i++) {
            y42[6 + i] = std::sin(1.0 + i);
        }
        ASSERT_EQ((mee ? campanella::eomstm_mee : campanella::eomstm_rv)(100.0, y42, ydot, &p), GSL_SUCCESS);
        for (int i = 0; i < 6; i++) {
            EXPECT_EQ(xdot[i], e.xdot[i]);
            EXPECT_EQ(ydot[i], e.xdot[i]);
            for (int j = 0; j < 6; j++) {
                double sum = 0.0;
                for (int l = 0; l < 6; l++) {
                    sum += e.A[6 * i + l] * y42[6 + 6 * l + j];
                }
                EXPECT_NEAR(ydot[6 + 6 * i + j], sum, 1e-15 * std::abs(sum) + 1e-300) << i << "," << j;
            }
        }
    }
}

TEST(Eoms, UsableAsGslSystem) {
    // eom_rv drives GSL's own driver to the same final state as solve_ivp_rv
    REQUIRE_KERNELS();
    SpiceParameters p = moon_params(FORCE_CASES[1]);
    gsl_odeiv2_system sys = {campanella::eom_rv, nullptr, 6, &p};
    gsl_odeiv2_driver* d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, 1.0, 1e-12, 1e-14);
    double t = 0.0;
    double y[6];
    std::copy(STATE0.begin(), STATE0.end(), y);
    const int status = gsl_odeiv2_driver_apply(d, &t, 86400.0, y);
    gsl_odeiv2_driver_free(d);
    ASSERT_EQ(status, GSL_SUCCESS);
    const std::vector<double> ref = final_state(campanella::solve_ivp_rv(p, {0.0, 86400.0}, STATE0));
    for (int i = 0; i < 6; i++) {
        EXPECT_NEAR(y[i], ref[i], 1e-9 * std::abs(ref[i]) + 1e-12) << i;
    }
}

TEST(Eoms, ReturnCodes) {
    REQUIRE_KERNELS();
    SpiceParameters p = moon_params(FORCE_CASES[2]);
    double xdot[6];
    // MEE out of domain: recoverable
    const double bad_p[6] = {-7000.0, 0.1, 0.0, 0.0, 0.0, 0.0};
    const double bad_w[6] = {7000.0, 1.5, 0.0, 0.0, 0.0, std::acos(-1.0)};
    EXPECT_EQ(campanella::eom_mee(0.0, bad_p, xdot, &p), GSL_EDOM);
    EXPECT_EQ(campanella::eom_mee(0.0, bad_w, xdot, &p), GSL_EDOM);
    // below the reference radius of the field: recoverable
    const double low[6] = {0.0, 0.0, 1700.0, 1.6, 0.0, 0.0};
    EXPECT_EQ(campanella::eom_rv(0.0, low, xdot, &p), GSL_EDOM);
    EXPECT_THROW(campanella::eval_rv(p, 0.0, std::vector<double>(low, low + 6)), std::domain_error);
    // a SPICE error stops the integration, and is reported with NAIF's message by eval_*
    SpiceParameters pluto(ET0, {"MOON", "PLUTO"}, with_gms({MU_MOON, 869.6}));
    EXPECT_EQ(campanella::eom_rv(0.0, STATE0.data(), xdot, &pluto), GSL_EBADFUNC);
    // the message survives for C++ callers driving GSL themselves
    EXPECT_NE(campanella::eom_error(pluto).find("SPICE(SPKINSUFFDATA)"), std::string::npos)
        << campanella::eom_error(pluto);
    try {
        campanella::eval_rv(pluto, 0.0, STATE0);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("SPICE(SPKINSUFFDATA)"), std::string::npos) << e.what();
    }
    // SPICE was reset: a valid evaluation succeeds
    EXPECT_EQ(campanella::eom_rv(0.0, STATE0.data(), xdot, &p), GSL_SUCCESS);
    EXPECT_EQ(campanella::eom_error(p), "");
    // at the center of the body: non-finite, recoverable
    const double origin[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    SpiceParameters two_body(0.0, {"EARTH"}, with_gms({398600.4418}));
    EXPECT_EQ(campanella::eom_rv(0.0, origin, xdot, &two_body), GSL_EDOM);
    EXPECT_NE(campanella::eom_error(two_body).find("non-finite"), std::string::npos);
    EXPECT_THROW(campanella::eval_rv(two_body, 0.0, std::vector<double>(origin, origin + 6)), std::domain_error);
    EXPECT_THROW(campanella::eval_rv(p, 0.0, {1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(campanella::eval_mee(p, 0.0, {1.0, 2.0}), std::invalid_argument);
}
