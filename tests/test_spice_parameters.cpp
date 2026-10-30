// SpiceParameters: resolution, units, validation, copies; two-body without kernels

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <SpiceUsr.h>

#include "helpers.hpp"

using namespace campanella_test;
using campanella::SpiceOptions;
using campanella::SpiceParameters;

TEST(SpiceParameters, TwoBodyWithoutKernels) {
    // no kernel at all: built-in names and frames, explicit GM, no third body, SRP or SH
    kclear_c();
    kernels_loaded() = false;
    const SpiceParameters params(0.0, {"MOON"}, with_gms({MU_MOON}));
    EXPECT_EQ(params.naif_codes(), std::vector<int>{301});
    const campanella::IvpResult rv = campanella::solve_ivp_rv(params, {0.0, TF}, STATE0);
    ASSERT_EQ(rv.t.back(), TF);
    const std::vector<double> expect = {-18205.698468519975, 26479.25761849232, -38615.225993310676,
                                        0.01263483050054705, 0.1635690404557372, -0.19163703441809382};
    const std::vector<double> xf = final_state(rv);
    for (int i = 0; i < 6; i++) {
        EXPECT_NEAR(xf[i], expect[i], 1e-12 * std::abs(expect[i]));
    }
    // energy is conserved along the whole trajectory
    auto energy = [&](size_t k) {
        const double* x = &rv.x[6 * k];
        return 0.5 * (x[3] * x[3] + x[4] * x[4] + x[5] * x[5])
               - MU_MOON / std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    };
    for (size_t k = 0; k < rv.size(); k++) {
        EXPECT_NEAR(energy(k), energy(0), 1e-12 * std::abs(energy(0)));
    }
    // MEE: only L changes, and the final state agrees with the Cartesian one
    const campanella::IvpResult mee =
        campanella::solve_ivpstm_mee(params, {0.0, TF}, campanella::cart2mee(STATE0, MU_MOON));
    const std::vector<double> rv_mee = campanella::mee2cart(final_state(mee), MU_MOON);
    EXPECT_LE(norm3_diff(rv_mee, xf, 0), 1e-5);
    EXPECT_LE(norm3_diff(rv_mee, xf, 3), 1e-10);
}

TEST(SpiceParameters, DimensionalByDefault) {
    const SpiceParameters p(ET0, {"EARTH"}, with_gms({398600.4}));
    EXPECT_EQ(p.DU(), 1.0);
    EXPECT_EQ(p.TU(), 1.0);
    EXPECT_EQ(p.VU(), 1.0);
    EXPECT_EQ(p.mus(), std::vector<double>{398600.4});
    EXPECT_FALSE(p.include_srp());
    EXPECT_FALSE(p.include_spherical_harmonics());
    EXPECT_EQ(p.naif_frame(), "J2000");
    EXPECT_EQ(p.et0(), ET0);
}

TEST(SpiceParameters, CanonicalScales) {
    SpiceOptions o = with_gms(GM_LIST);
    o.DU = 1e5;
    const SpiceParameters p(ET0, BODIES, o);
    EXPECT_EQ(p.DU(), 1e5);
    EXPECT_EQ(p.VU(), std::sqrt(MU_MOON / 1e5));
    EXPECT_EQ(p.TU(), 1e5 / p.VU());
    EXPECT_NEAR(p.TU(), 451624.90737307334, 1e-14 * 451624.90737307334);
    EXPECT_EQ(p.mus()[0], 1.0);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_NEAR(p.mus()[i] * p.DU() * p.DU() * p.DU() / (p.TU() * p.TU()), GM_LIST[i], 1e-14 * GM_LIST[i]);
    }
    EXPECT_EQ(p.GMs(), GM_LIST);   // GMs stay in km^3/s^2
    // the flag wins over DU
    o.use_canonical_scales = false;
    const SpiceParameters q(ET0, BODIES, o);
    EXPECT_EQ(q.DU(), 1.0);
    EXPECT_EQ(q.TU(), 1.0);
    EXPECT_EQ(q.mus(), GM_LIST);
}

TEST(SpiceParameters, GmsFromKernelsMatchExplicit) {
    REQUIRE_KERNELS();
    const SpiceParameters spice(ET0, {"MOON", "EARTH", "SUN"});
    ASSERT_EQ(spice.GMs().size(), 3u);
    EXPECT_NEAR(spice.GMs()[0], 4902.8001184575496, 1e-15 * 4902.8);
    EXPECT_NEAR(spice.GMs()[1], 398600.43550702266, 1e-15 * 398600.4);
    EXPECT_NEAR(spice.GMs()[2], 1.3271244004127942e11, 1e-15 * 1.3e11);
    EXPECT_EQ(spice.options().GMs, spice.GMs());
    const SpiceParameters explicit_gms(ET0, {"MOON", "EARTH", "SUN"}, with_gms(spice.GMs()));
    const campanella::IvpResult a = campanella::solve_ivp_rv(spice, {0.0, 86400.0}, STATE0);
    const campanella::IvpResult b = campanella::solve_ivp_rv(explicit_gms, {0.0, 86400.0}, STATE0);
    EXPECT_EQ(a.t, b.t);
    EXPECT_EQ(a.x, b.x);
}

TEST(SpiceParameters, MissingGmInKernelPool) {
    REQUIRE_KERNELS();
    try {
        SpiceParameters(ET0, {"EARTH", "-82"});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("no GM for body \"-82\""), std::string::npos) << e.what();
    }
}

TEST(SpiceParameters, NamesAndCodesAreEquivalent) {
    REQUIRE_KERNELS();
    SpiceOptions o = with_gms(GM_LIST);
    add_srp(o);
    const SpiceParameters by_name(ET0, {"MOON", "EARTH", "SUN"}, o);
    const SpiceParameters by_code(ET0, {"301", "399", "10"}, o);
    EXPECT_EQ(by_name.naif_codes(), (std::vector<int>{301, 399, 10}));
    EXPECT_EQ(by_name.naif_codes(), by_code.naif_codes());
    EXPECT_EQ(by_name.naif_ids(), (std::vector<std::string>{"MOON", "EARTH", "SUN"}));
    EXPECT_EQ(campanella::solve_ivp_rv(by_name, {0.0, 86400.0}, STATE0).x,
              campanella::solve_ivp_rv(by_code, {0.0, 86400.0}, STATE0).x);
}

TEST(SpiceParameters, WorkspacesAreNotShared) {
    // a failure in a copy does not leave its fault in the original
    REQUIRE_KERNELS();
    SpiceOptions o = with_gms(GM_LIST);
    add_moon_sh(o);
    SpiceParameters a(ET0, BODIES, o);
    SpiceParameters b = a;
    const double low[6] = {0.0, 0.0, 1700.0, 1.6, 0.0, 0.0};
    double xdot[6];
    ASSERT_EQ(campanella::eom_rv(0.0, low, xdot, &b), 1);
    EXPECT_NE(campanella::eom_error(b).find("sh_rho_min"), std::string::npos) << campanella::eom_error(b);
    EXPECT_EQ(campanella::eom_error(a), "");
    // self-assignment and assignment from a moved object leave valid objects
    a = a;
    SpiceParameters c = std::move(b);
    EXPECT_TRUE(b.include_spherical_harmonics());
    EXPECT_EQ(campanella::eom_rv(0.0, STATE0.data(), xdot, &b), 0);
    EXPECT_EQ(campanella::eom_error(b), "");
    EXPECT_EQ(campanella::eom_rv(0.0, STATE0.data(), xdot, &a), 0);
    EXPECT_EQ(campanella::eom_rv(0.0, STATE0.data(), xdot, &c), 0);
}

TEST(SpiceParameters, CopiesAreIndependent) {
    REQUIRE_KERNELS();
    SpiceOptions o = with_gms(GM_LIST);
    add_moon_sh(o);
    const SpiceParameters a(ET0, BODIES, o);
    SpiceParameters b = a;                 // copy: shares the model, has its own workspace
    SpiceParameters c(ET0, {"EARTH"}, with_gms({398600.4}));
    c = a;                                 // copy assignment
    SpiceParameters d = std::move(c);      // move
    const std::vector<double> x = state_llo();
    double xa[6], xb[6], xd[6];
    ASSERT_EQ(campanella::eom_rv(10.0, x.data(), xa, const_cast<SpiceParameters*>(&a)), 0);
    ASSERT_EQ(campanella::eom_rv(10.0, x.data(), xb, &b), 0);
    ASSERT_EQ(campanella::eom_rv(10.0, x.data(), xd, &d), 0);
    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(xa[i], xb[i]);
        EXPECT_EQ(xa[i], xd[i]);
    }
    EXPECT_TRUE(d.include_spherical_harmonics());
    EXPECT_EQ(d.options().sh_nmax, 10);
}

namespace {

template <typename F>
void expect_invalid(F make, const std::string& fragment) {
    try {
        make();
        ADD_FAILURE() << "expected std::invalid_argument containing \"" << fragment << "\"";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

}  // namespace

TEST(SpiceParameters, Validation) {
    const std::vector<double> one = {1.0};
    expect_invalid([] { SpiceParameters(ET0, {}); }, "naif_ids");
    expect_invalid([&] { SpiceParameters(NAN, {"EARTH"}, with_gms(one)); }, "et0");
    expect_invalid([&] { SpiceParameters(ET0, {"NOT_A_BODY"}, with_gms(one)); }, "unknown NAIF body");
    expect_invalid([] { SpiceParameters(ET0, {"EARTH", "399"}, with_gms({1.0, 1.0})); }, "listed twice");
    expect_invalid([&] { SpiceParameters(ET0, {"EARTH", "MOON"}, with_gms(one)); }, "one entry per NAIF id");
    expect_invalid([] { SpiceParameters(ET0, {"EARTH"}, with_gms({0.0})); }, "GMs");
    expect_invalid([] { SpiceParameters(ET0, {"EARTH", "MOON"}, with_gms({1.0, -1.0})); }, "GMs");
    expect_invalid([] { SpiceParameters(ET0, {"EARTH"}, with_gms({INFINITY})); }, "GMs");
    expect_invalid([] { SpiceOptions o = with_gms({1.0}); o.DU = 0.0; SpiceParameters(ET0, {"EARTH"}, o); }, "DU");
    expect_invalid([] { SpiceOptions o = with_gms({1.0}); o.DU = NAN; SpiceParameters(ET0, {"EARTH"}, o); }, "DU");
    expect_invalid([] { SpiceOptions o = with_gms({1.0}); o.abcorr = "LT+S"; SpiceParameters(ET0, {"EARTH"}, o); },
                   "abcorr");
    expect_invalid([] { SpiceOptions o = with_gms({1.0}); o.naif_frame = "NOT_A_FRAME"; SpiceParameters(ET0, {"EARTH"}, o); },
                   "naif_frame");
    expect_invalid([] { SpiceOptions o = with_gms({1.0}); o.naif_frame = "IAU_EARTH"; SpiceParameters(ET0, {"EARTH"}, o); },
                   "not inertial");
    expect_invalid([] { SpiceOptions o = with_gms({1.0, 1.0}); add_srp(o); SpiceParameters(ET0, {"EARTH", "MOON"}, o); },
                   "Sun");
    expect_invalid([] { SpiceOptions o = with_gms({1.0, 1.0}); add_srp(o); o.srp_m = 0.0;
                        SpiceParameters(ET0, {"EARTH", "SUN"}, o); }, "srp_m");
    expect_invalid([] { SpiceOptions o = with_gms({1.0, 1.0}); add_srp(o); o.srp_A = -1.0;
                        SpiceParameters(ET0, {"EARTH", "SUN"}, o); }, "srp_A");
    expect_invalid([] { SpiceOptions o = with_gms({1.0, 1.0}); add_srp(o); o.srp_rhoD = NAN;
                        SpiceParameters(ET0, {"EARTH", "SUN"}, o); }, "srp_rhoS and srp_rhoD");
    expect_invalid([] { SpiceOptions o = with_gms({1.0}); o.DU = -5.0; o.use_canonical_scales = false;
                        SpiceParameters(ET0, {"EARTH"}, o); }, "DU");
}

TEST(SpiceParameters, SphericalHarmonicsValidation) {
    REQUIRE_KERNELS();
    expect_invalid([] { SpiceOptions o = with_gms({MU_MOON}); add_moon_sh(o); o.sh_nmax = 1;
                        SpiceParameters(ET0, {"MOON"}, o); }, "sh_nmax");
    expect_invalid([] { SpiceOptions o = with_gms({MU_MOON}); add_moon_sh(o); o.sh_frame = "";
                        SpiceParameters(ET0, {"MOON"}, o); }, "sh_frame");
    expect_invalid([] { SpiceOptions o = with_gms({MU_MOON}); add_moon_sh(o); o.sh_rho_min = 1.5;
                        SpiceParameters(ET0, {"MOON"}, o); }, "sh_rho_min");
    expect_invalid([] { SpiceOptions o = with_gms({MU_MOON}); add_moon_sh(o); o.sh_frame = "NOT_A_FRAME";
                        SpiceParameters(ET0, {"MOON"}, o); }, "unknown sh_frame");
    // the Moon field around the Earth: GMs[0] does not match the file
    expect_invalid([] { SpiceOptions o = with_gms({398600.4}); add_moon_sh(o); SpiceParameters(ET0, {"EARTH"}, o); },
                   "differs from the GM");
    // a missing gravity field file is a load error
    EXPECT_THROW({ SpiceOptions o = with_gms({MU_MOON}); add_moon_sh(o); o.sh_file = "/nonexistent.tab";
                   SpiceParameters(ET0, {"MOON"}, o); }, std::runtime_error);
}
