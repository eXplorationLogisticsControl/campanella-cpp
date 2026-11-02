// Mass-augmented solvers under a zero-order-hold control: constants of the thrust model, the public
// solvers forward and backward, t_eval, zero-length spans, mass faults, input validation, and the
// GSL-compatible equations of motion. Set ZOH_PRINT_ERRORS to print the measured errors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include "helpers.hpp"
#include "model.h"

using namespace campanella_test;
using campanella::ControlledParameters;
using campanella::EomEvalControl;
using campanella::IvpOptions;
using campanella::IvpResult;
using campanella::SpiceOptions;
using campanella::SpiceParameters;
using campanella::ThrustFrame;
using campanella::ThrustOptions;

// A temporary SpiceParameters would leave ControlledParameters with a dangling pointer
static_assert(!std::is_constructible<ControlledParameters, SpiceParameters&&, const ThrustOptions&,
                                     const std::vector<double>&>::value);
static_assert(!std::is_constructible<ControlledParameters, const SpiceParameters&&, const ThrustOptions&,
                                     const std::vector<double>&>::value);
// ... while an lvalue is accepted, and the thrust and u are copied, so temporaries of those are fine
static_assert(std::is_constructible<ControlledParameters, SpiceParameters&, const ThrustOptions&,
                                    const std::vector<double>&>::value);
static_assert(std::is_constructible<ControlledParameters, const SpiceParameters&, ThrustOptions&&,
                                    std::vector<double>&&>::value);

namespace {

using Solver = IvpResult (*)(const SpiceParameters&, const ThrustOptions&, const std::vector<double>&,
                             const std::vector<double>&, const std::vector<double>&, const IvpOptions&);
using Eval = EomEvalControl (*)(const SpiceParameters&, const ThrustOptions&, double,
                                const std::vector<double>&, const std::vector<double>&);
using Eom = int (*)(double, const double[], double[], void*);
using Evaluator = int (*)(const campanella::detail::Model&, const campanella::detail::ZohControl&,
                          campanella::detail::Workspace&, double, const double[7], double[7], double[7][7],
                          double[7][4]);

// One set of dynamics, through its public functions
struct Dyn {
    const char* name;
    bool mee;
    ThrustFrame frame;
    Solver solve;
    Solver solvestm;
    Eval eval;
    Eom eom;
    Eom eomstm;
    Evaluator evaluate;
};

const Dyn DYNS[] = {
    {"rvm", false, ThrustFrame::rtn, campanella::solve_ivp_rvm, campanella::solve_ivpstm_rvm, campanella::eval_rvm,
     campanella::eom_rvm, campanella::eomstm_rvm, campanella::detail::rvm_evaluate},
    {"meem RTN", true, ThrustFrame::rtn, campanella::solve_ivp_meem, campanella::solve_ivpstm_meem,
     campanella::eval_meem, campanella::eom_meem, campanella::eomstm_meem, campanella::detail::meem_evaluate},
    {"meem inertial", true, ThrustFrame::inertial, campanella::solve_ivp_meem, campanella::solve_ivpstm_meem,
     campanella::eval_meem, campanella::eom_meem, campanella::eomstm_meem, campanella::detail::meem_evaluate},
};

const std::vector<double> U = {0.3, -0.5, 0.4, 0.71};
const double MASS = 0.87;
const double INF = std::numeric_limits<double>::infinity();
const double NAN_ = std::numeric_limits<double>::quiet_NaN();

// Tolerances: about 10 times the largest value measured with ZOH_PRINT_ERRORS (in the comments), on
// Linux x86-64 in Release and with an empty build type
const double TOL_ROUNDTRIP_STATE = 1e-12;   // 6.9e-14
const double TOL_ROUNDTRIP_PHI = 2e-10;     // 2.2e-11
const double TOL_ROUNDTRIP_PSI = 2e-11;     // 1.7e-12
const double TOL_STATE_ONLY = 5e-11;        // 4.8e-12, the step sizes differ
const double TOL_TEVAL_STATE = 3e-13;       // 2.3e-14
const double TOL_TEVAL_SENS = 1e-11;        // 1.1e-12
const double TOL_GSL = 1e-13;               // 0: the driver takes the same steps as the solvers

ThrustOptions thruster(const Dyn& d) {
    ThrustOptions th;
    th.thrust_N = 0.5;
    th.isp_s = 3000.0;
    th.mass_unit_kg = 1850.0;
    th.mee_control_frame = d.frame;
    return th;
}

// A mass flow that empties the tank in about 8 hours
ThrustOptions strong_thruster(const Dyn& d) {
    ThrustOptions th = thruster(d);
    th.thrust_N = 1000.0;
    th.mass_unit_kg = 1000.0;
    return th;
}

SpiceOptions canonical(SpiceOptions o) {
    o.DU = 1737.4;
    return o;
}

// Central body only, no kernels needed
SpiceParameters two_body(bool canon) {
    const SpiceOptions o = with_gms({MU_MOON});
    return SpiceParameters(ET0, {"MOON"}, canon ? canonical(o) : o);
}

// Every force, in km and s or in canonical units
SpiceParameters all_forces(bool canon) {
    SpiceOptions o = with_gms(GM_LIST);
    add_srp(o);
    add_moon_sh(o);
    return SpiceParameters(ET0, BODIES, canon ? canonical(o) : o);
}

// A Cartesian state in km, km/s, in the units and coordinates of d, with the mass appended
std::vector<double> state_of(const Dyn& d, const SpiceParameters& p, std::vector<double> rv, double m) {
    for (int i = 0; i < 3; i++) {
        rv[i] /= p.DU();
        rv[3 + i] /= p.VU();
    }
    std::vector<double> x = d.mee ? campanella::cart2mee(rv, p.mus()[0]) : rv;
    x.push_back(m);
    return x;
}

// Scale of each state: |r|, |v| and m in Cartesian; p, 1 and m in MEE
std::vector<double> scales(const Dyn& d, const std::vector<double>& x) {
    if (d.mee) {
        return {x[0], 1.0, 1.0, 1.0, 1.0, 1.0, x[6]};
    }
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const double v = std::sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]);
    return {r, r, r, v, v, v, x[6]};
}

double scaled_max_diff(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& s) {
    double e = 0.0;
    for (size_t i = 0; i < s.size(); i++) {
        e = std::max(e, std::isfinite(a[i]) && std::isfinite(b[i]) ? std::abs(a[i] - b[i]) / s[i] : INF);
    }
    return e;
}

std::vector<double> row_of(const std::vector<double>& v, size_t row, size_t width) {
    return std::vector<double>(v.begin() + row * width, v.begin() + (row + 1) * width);
}

// C = A B, for A n x n and B n x m, row-major
std::vector<double> product(const std::vector<double>& A, const std::vector<double>& B, int n, int m) {
    std::vector<double> C(n * m, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < n; l++) {
                C[m * i + j] += A[n * i + l] * B[m * l + j];
            }
        }
    }
    return C;
}

std::vector<double> identity7() {
    std::vector<double> I(49, 0.0);
    for (int i = 0; i < 7; i++) {
        I[8 * i] = 1.0;
    }
    return I;
}

// Measured values, on request
void note(const char* what, const char* name, bool canon, double value) {
    if (std::getenv("ZOH_PRINT_ERRORS") != nullptr) {
        std::printf("%-28s %-14s %-5s %.3e\n", what, name, canon ? "canon" : "km", value);
    }
}

void expect_invalid(const std::function<void()>& f, const std::string& text, const std::string& context) {
    try {
        f();
        ADD_FAILURE() << context << ": expected std::invalid_argument with \"" << text << "\"";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << context << ": " << e.what();
    } catch (const std::exception& e) {
        ADD_FAILURE() << context << ": wrong exception: " << e.what();
    }
}

// The input is invalid for all the functions of d that take thrust and u
void expect_control_invalid(const Dyn& d, const SpiceParameters& p, const ThrustOptions& th,
                            const std::vector<double>& u, const std::string& text, const std::string& context) {
    const std::vector<double> x = state_of(d, p, state_llo(), MASS);
    const std::vector<double> tspan = {0.0, 1.0};
    const std::string where = std::string(d.name) + ", " + context;
    expect_invalid([&] { d.solve(p, th, tspan, x, u, {}); }, text, where + ", solve");
    expect_invalid([&] { d.solvestm(p, th, tspan, x, u, {}); }, text, where + ", solvestm");
    expect_invalid([&] { d.eval(p, th, 0.0, x, u); }, text, where + ", eval");
    expect_invalid([&] { ControlledParameters cp(p, th, u); }, text, where + ", ControlledParameters");
}

// The time of a message such as "... at t = 12.5 TU"; NaN if there is none
double fault_time(const std::string& msg) {
    const std::string key = " at t = ";
    const size_t pos = msg.find(key);
    return pos == std::string::npos ? NAN_ : std::strtod(msg.c_str() + pos + key.size(), nullptr);
}

}  // namespace

TEST(ThrustConstants, KnownValuesAndCanonicalScaling) {
    // Expected values worked out by hand with 40-digit decimal arithmetic, not with the library.
    // km, s: 1 N / 1000 kg = 1e-3 m/s^2 = 1e-6 km/s^2, and 1 N / (3000 s * 9.80665 m/s^2) / 1000 kg.
    const double rtol = 1e-15;
    const std::vector<double> u = {0.0, 0.0, 0.0, 0.0};
    ThrustOptions thrust;
    thrust.thrust_N = 1.0;
    thrust.isp_s = 3000.0;
    thrust.mass_unit_kg = 1000.0;
    const SpiceParameters km(ET0, {"MOON"}, with_gms({MU_MOON}));
    ASSERT_EQ(km.DU(), 1.0);
    ASSERT_EQ(km.TU(), 1.0);
    const ControlledParameters cp_km(km, thrust, u);
    EXPECT_NEAR(cp_km.c1(), 1e-6, rtol * 1e-6);
    EXPECT_NEAR(cp_km.c2(), 3.399054043259760808567e-8, rtol * 3.4e-8);

    // canonical: DU = 1737.4 km, TU = DU / sqrt(GM / DU) = 1034.254957477028... s
    SpiceOptions o = with_gms({MU_MOON});
    o.DU = 1737.4;
    const SpiceParameters canon(ET0, {"MOON"}, o);
    ASSERT_NEAR(canon.TU(), 1034.254957477028464706, rtol * 1034.0);
    const ControlledParameters cp_canon(canon, thrust, u);
    EXPECT_NEAR(cp_canon.c1(), 6.156805094197133404697e-4, rtol * 6.2e-4);
    EXPECT_NEAR(cp_canon.c2(), 3.515488494973745586604e-5, rtol * 3.5e-5);
    // c1_canon = c1_km TU^2 / DU, c2_canon = c2_km TU
    EXPECT_NEAR(cp_canon.c1(), cp_km.c1() * canon.TU() * canon.TU() / canon.DU(), rtol * 6.2e-4);
    EXPECT_NEAR(cp_canon.c2(), cp_km.c2() * canon.TU(), rtol * 3.5e-5);

    // every input enters: another propulsion system and mass unit
    thrust.thrust_N = 0.35;
    thrust.isp_s = 2150.0;
    thrust.mass_unit_kg = 1850.0;
    const ControlledParameters cp_km2(km, thrust, u);
    EXPECT_NEAR(cp_km2.c1(), 1.891891891891891891892e-7, rtol * 1.9e-7);
    EXPECT_NEAR(cp_km2.c2(), 8.972989931923944309228e-9, rtol * 9e-9);
    const ControlledParameters cp_canon2(canon, thrust, u);
    EXPECT_NEAR(cp_canon2.c1(), 1.164800963767025238727e-4, rtol * 1.2e-4);
    EXPECT_NEAR(cp_canon2.c2(), 9.280359320483803559859e-6, rtol * 9.3e-6);

    // g0 is in m/s^2 and enters c2 only
    thrust.g0 = 3.71;
    const ControlledParameters cp_g0(km, thrust, u);
    EXPECT_EQ(cp_g0.c1(), cp_km2.c1());
    EXPECT_NEAR(cp_g0.c2(), 2.371832121722424486795e-8, rtol * 2.4e-8);
}

TEST(Solve, ForwardAndBackward) {
    // Six hours of low lunar orbit with every force, there and back
    REQUIRE_KERNELS();
    for (const bool canon : {false, true}) {
        const SpiceParameters p = all_forces(canon);
        for (const Dyn& d : DYNS) {
            SCOPED_TRACE(std::string(d.name) + (canon ? ", canonical" : ", km"));
            const ThrustOptions th = thruster(d);
            const double c2 = ControlledParameters(p, th, U).c2();
            const std::vector<double> x0 = state_of(d, p, state_llo(), MASS);
            const std::vector<double> s = scales(d, x0);
            const double t0 = 3600.0 / p.TU();
            const double tf = t0 + 6 * 3600.0 / p.TU();

            const IvpResult f = d.solvestm(p, th, {t0, tf}, x0, U, {});
            ASSERT_GT(f.size(), 2u);
            EXPECT_EQ(f.t.front(), t0);
            EXPECT_EQ(f.t.back(), tf);
            EXPECT_EQ(row_of(f.x, 0, 7), x0);
            EXPECT_EQ(row_of(f.stm, 0, 49), identity7());
            EXPECT_EQ(row_of(f.stm_u, 0, 28), std::vector<double>(28, 0.0));

            const std::vector<double> xf = final_state_n(f, 7);
            const IvpResult b = d.solvestm(p, th, {tf, t0}, xf, U, {});
            ASSERT_GT(b.size(), 2u);
            EXPECT_EQ(b.t.front(), tf);
            EXPECT_EQ(b.t.back(), t0);
            EXPECT_EQ(row_of(b.x, 0, 7), xf);
            EXPECT_EQ(row_of(b.stm, 0, 49), identity7());
            EXPECT_EQ(row_of(b.stm_u, 0, 28), std::vector<double>(28, 0.0));
            for (size_t k = 1; k < f.size(); k++) {
                EXPECT_GT(f.t[k], f.t[k - 1]);
            }
            for (size_t k = 1; k < b.size(); k++) {
                EXPECT_LT(b.t[k], b.t[k - 1]);
            }

            // the mass and its rows, in each direction: the sign of Psi[m, Gamma] follows the direction
            for (const IvpResult* r : {&f, &b}) {
                const double start = r->t.front();
                const double m_start = r->x[6];
                for (size_t k = 0; k < r->size(); k++) {
                    const double dt = r->t[k] - start;
                    EXPECT_NEAR(r->x[7 * k + 6], m_start - c2 * U[3] * dt, 1e-14 * m_start);
                    for (int j = 0; j < 7; j++) {
                        EXPECT_EQ(r->stm[49 * k + 7 * 6 + j], j == 6 ? 1.0 : 0.0);
                    }
                    for (int j = 0; j < 3; j++) {
                        EXPECT_EQ(r->stm_u[28 * k + 4 * 6 + j], 0.0);
                    }
                    EXPECT_NEAR(r->stm_u[28 * k + 4 * 6 + 3], -c2 * dt, 1e-13 * c2 * std::abs(tf - t0));
                }
            }

            // back at the start: the state, Phi_b Phi_f = I and Psi_b = -Phi_b Psi_f
            const double roundtrip = scaled_max_diff(final_state_n(b, 7), x0, s);
            note("round trip, state", d.name, canon, roundtrip);
            EXPECT_LE(roundtrip, TOL_ROUNDTRIP_STATE);

            const std::vector<double> Phi_f = final_stm_n(f, 7), Psi_f = final_stm_u(f, 7, 4);
            const std::vector<double> Phi_b = final_stm_n(b, 7), Psi_b = final_stm_u(b, 7, 4);
            const double err_phi = matrix_error(product(Phi_b, Phi_f, 7, 7), identity7(), 7, 7, s, s, 0.0);
            note("round trip, Phi", d.name, canon, err_phi);
            EXPECT_LE(err_phi, TOL_ROUNDTRIP_PHI);
            std::vector<double> minus = product(Phi_b, Psi_f, 7, 4);
            for (double& v : minus) {
                v = -v;
            }
            const double err_psi = matrix_error(Psi_b, minus, 7, 4, s, {}, 0.0);
            note("round trip, Psi", d.name, canon, err_psi);
            EXPECT_LE(err_psi, TOL_ROUNDTRIP_PSI);

            // the state-only solvers, with their own step sizes, reach the same states
            const IvpResult fs = d.solve(p, th, {t0, tf}, x0, U, {});
            const double err_state = scaled_max_diff(final_state_n(fs, 7), xf, s);
            note("state only vs stm", d.name, canon, err_state);
            EXPECT_LE(err_state, TOL_STATE_ONLY);
            const IvpResult bs = d.solve(p, th, {tf, t0}, xf, U, {});
            EXPECT_EQ(bs.t.back(), t0);
            // from the end of another integration, so the error of both counts
            EXPECT_LE(scaled_max_diff(final_state_n(bs, 7), x0, s), TOL_STATE_ONLY);
        }
    }
}

TEST(Solve, TEval) {
    REQUIRE_KERNELS();
    for (const bool canon : {false, true}) {
        const SpiceParameters p = all_forces(canon);
        const double hour = 3600.0 / p.TU();
        for (const Dyn& d : DYNS) {
            SCOPED_TRACE(std::string(d.name) + (canon ? ", canonical" : ", km"));
            const ThrustOptions th = thruster(d);
            const double c2 = ControlledParameters(p, th, U).c2();
            const std::vector<double> x0 = state_of(d, p, state_llo(), MASS);
            const std::vector<double> s = scales(d, x0);
            const double tf = 3 * hour;

            for (const double dir : {1.0, -1.0}) {
                // includes t0, stops before tf, and has an output inside a step
                IvpOptions o;
                o.t_eval = {0.0, dir * 0.5 * hour, dir * 1.25 * hour, dir * 2.0 * hour};
                const IvpResult stm = d.solvestm(p, th, {0.0, dir * tf}, x0, U, o);
                const IvpResult state = d.solve(p, th, {0.0, dir * tf}, x0, U, o);
                const size_t N = o.t_eval.size();
                EXPECT_EQ(stm.t, o.t_eval);
                EXPECT_EQ(state.t, o.t_eval);
                ASSERT_EQ(stm.x.size(), N * 7);
                ASSERT_EQ(stm.stm.size(), N * 49);
                ASSERT_EQ(stm.stm_u.size(), N * 28);
                ASSERT_EQ(state.x.size(), N * 7);
                EXPECT_TRUE(state.stm.empty());
                EXPECT_TRUE(state.stm_u.empty());
                EXPECT_EQ(row_of(stm.x, 0, 7), x0);
                EXPECT_EQ(row_of(stm.stm, 0, 49), identity7());
                EXPECT_EQ(row_of(stm.stm_u, 0, 28), std::vector<double>(28, 0.0));

                for (size_t k = 1; k < N; k++) {
                    EXPECT_NEAR(stm.x[7 * k + 6], MASS - c2 * U[3] * o.t_eval[k], 1e-14 * MASS);
                    // the same as an integration that ends there
                    const IvpResult direct = d.solvestm(p, th, {0.0, o.t_eval[k]}, x0, U, {});
                    const std::vector<double> xk = row_of(stm.x, k, 7);
                    const double err_x = scaled_max_diff(xk, final_state_n(direct, 7), s);
                    const double err_phi = matrix_error(row_of(stm.stm, k, 49), final_stm_n(direct, 7), 7, 7, s, s, 0.0);
                    const double err_psi = matrix_error(row_of(stm.stm_u, k, 28), final_stm_u(direct, 7, 4), 7, 4, s, {}, 0.0);
                    note("t_eval vs direct, state", d.name, canon, err_x);
                    note("t_eval vs direct, Phi", d.name, canon, err_phi);
                    note("t_eval vs direct, Psi", d.name, canon, err_psi);
                    EXPECT_LE(err_x, TOL_TEVAL_STATE);
                    EXPECT_LE(err_phi, TOL_TEVAL_SENS);
                    EXPECT_LE(err_psi, TOL_TEVAL_SENS);
                    EXPECT_LE(scaled_max_diff(row_of(state.x, k, 7), xk, s), TOL_STATE_ONLY);
                }
            }

            // ends exactly on tf
            IvpOptions o;
            o.t_eval = {tf};
            const IvpResult one = d.solvestm(p, th, {0.0, tf}, x0, U, o);
            ASSERT_EQ(one.size(), 1u);
            EXPECT_EQ(one.t[0], tf);
            EXPECT_EQ(one.stm_u.size(), 28u);

            // the validation is the one of the existing solvers
            for (const std::vector<double>& bad : {std::vector<double>{-hour}, std::vector<double>{2 * tf},
                                                   std::vector<double>{hour, hour}, std::vector<double>{2 * hour, hour},
                                                   std::vector<double>{NAN_}}) {
                o.t_eval = bad;
                expect_invalid([&] { d.solve(p, th, {0.0, tf}, x0, U, o); }, "t_eval", d.name);
                expect_invalid([&] { d.solvestm(p, th, {0.0, tf}, x0, U, o); }, "t_eval", d.name);
            }
        }
    }
}

TEST(Solve, ZeroLengthSpan) {
    // no kernels, no integration
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            SCOPED_TRACE(std::string(d.name) + (canon ? ", canonical" : ", km"));
            const ThrustOptions th = thruster(d);
            const std::vector<double> x0 = state_of(d, p, state_llo(), MASS);
            IvpOptions at_t0;
            at_t0.t_eval = {5.0};
            for (const IvpOptions& o : {IvpOptions{}, at_t0}) {
                const IvpResult r = d.solvestm(p, th, {5.0, 5.0}, x0, U, o);
                ASSERT_EQ(r.size(), 1u);
                EXPECT_EQ(r.t[0], 5.0);
                EXPECT_EQ(r.x, x0);
                EXPECT_EQ(r.stm, identity7());
                EXPECT_EQ(r.stm_u, std::vector<double>(28, 0.0));
                EXPECT_EQ(r.nx, 7u);
                EXPECT_EQ(r.nu, 4u);

                const IvpResult state = d.solve(p, th, {5.0, 5.0}, x0, U, o);
                ASSERT_EQ(state.size(), 1u);
                EXPECT_EQ(state.x, x0);
                EXPECT_TRUE(state.stm.empty());
                EXPECT_TRUE(state.stm_u.empty());
            }
        }
    }
}

TEST(Solve, ResultMetadata) {
    // no kernels
    const SpiceParameters p = two_body(false);
    const std::vector<double> rv = state_llo();
    const std::vector<double> mee = campanella::cart2mee(rv, MU_MOON);
    const std::vector<double> tspan = {0.0, 60.0};

    // the existing solvers keep nx = 6, nu = 0, and have no control sensitivity
    for (const IvpResult& r : {campanella::solve_ivp_rv(p, tspan, rv), campanella::solve_ivp_mee(p, tspan, mee)}) {
        EXPECT_EQ(r.nx, 6u);
        EXPECT_EQ(r.nu, 0u);
        EXPECT_EQ(r.x.size(), r.size() * 6);
        EXPECT_TRUE(r.stm.empty());
        EXPECT_TRUE(r.stm_u.empty());
    }
    for (const IvpResult& r : {campanella::solve_ivpstm_rv(p, tspan, rv), campanella::solve_ivpstm_mee(p, tspan, mee)}) {
        EXPECT_EQ(r.nx, 6u);
        EXPECT_EQ(r.nu, 0u);
        EXPECT_EQ(r.x.size(), r.size() * 6);
        EXPECT_EQ(r.stm.size(), r.size() * 36);
        EXPECT_TRUE(r.stm_u.empty());
    }

    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d);
        const std::vector<double> x0 = state_of(d, p, rv, MASS);
        const IvpResult state = d.solve(p, th, tspan, x0, U, {});
        EXPECT_EQ(state.nx, 7u);
        EXPECT_EQ(state.nu, 4u);
        EXPECT_GT(state.size(), 1u);
        EXPECT_EQ(state.x.size(), state.size() * 7);
        EXPECT_TRUE(state.stm.empty());
        EXPECT_TRUE(state.stm_u.empty());

        const IvpResult stm = d.solvestm(p, th, tspan, x0, U, {});
        EXPECT_EQ(stm.nx, 7u);
        EXPECT_EQ(stm.nu, 4u);
        EXPECT_GT(stm.size(), 1u);
        EXPECT_EQ(stm.x.size(), stm.size() * 7);
        EXPECT_EQ(stm.stm.size(), stm.size() * 49);
        EXPECT_EQ(stm.stm_u.size(), stm.size() * 28);
    }
}

TEST(Mass, DepletionThrows) {
    // tau = 0, Gamma = 1, SRP off: the mass reaches zero at t* = m0 / c2 with nothing else singular.
    // The solver sees it coming; with the check removed, the state-only integration ends in the same
    // error on a stage with m <= 0, which Mass.DepletionByIntegrator keeps covered through eom_*.
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            SCOPED_TRACE(std::string(d.name) + (canon ? ", canonical" : ", km"));
            const ThrustOptions th = strong_thruster(d);
            const std::vector<double> x0 = state_of(d, p, state_llo(), MASS);
            const std::vector<double> u = {0.0, 0.0, 0.0, 1.0};
            const double t_star = MASS / ControlledParameters(p, th, u).c2();

            // up to there it is an ordinary integration
            const IvpResult before = d.solve(p, th, {0.0, 0.9 * t_star}, x0, u, {});
            EXPECT_NEAR(before.x.back(), 0.1 * MASS, 1e-13 * MASS);

            // backward, a negative Gamma empties the tank in the same time
            const std::vector<double> u_back = {0.0, 0.0, 0.0, -1.0};
            for (const double dir : {1.0, -1.0}) {
                try {
                    d.solve(p, th, {0.0, dir * 2 * t_star}, x0, dir > 0 ? u : u_back, {});
                    ADD_FAILURE() << "expected std::domain_error, dir = " << dir;
                } catch (const std::domain_error& e) {
                    const std::string msg = e.what();
                    EXPECT_NE(msg.find(d.mee ? "solve_ivp_meem" : "solve_ivp_rvm"), std::string::npos) << msg;
                    EXPECT_NE(msg.find("mass out of domain"), std::string::npos) << msg;
                    // the message has 15 digits
                    const double t = fault_time(msg);
                    note("depletion, |t - t*| / t*", d.name, canon, std::abs(t - dir * t_star) / t_star);
                    EXPECT_NEAR(t, dir * t_star, 1e-13 * t_star) << msg;
                }
            }
        }
    }
}

TEST(Mass, DepletionThrowsWithThrustAndSensitivities) {
    // The cases where the integrator alone ends in GSL_FAILURE: all are domain errors at t*. Only the
    // output times count: t_eval may stop before tf, on either side of t*.
    const SpiceParameters p = two_body(false);
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = strong_thruster(d);
        const std::vector<double> x0 = state_of(d, p, state_llo(), MASS);
        const std::vector<double> u = {0.3, -0.5, 0.4, 0.71};
        const double t_star = MASS / (ControlledParameters(p, th, u).c2() * u[3]);
        for (const Solver solve : {d.solve, d.solvestm}) {
            // not t* itself: the rounding of m0 - c2 Gamma t* decides on which side it falls
            for (const double tf : {t_star * (1 + 1e-9), 2 * t_star}) {
                try {
                    solve(p, th, {0.0, tf}, x0, u, {});
                    ADD_FAILURE() << "expected std::domain_error, tf = " << tf;
                } catch (const std::domain_error& e) {
                    EXPECT_NE(std::string(e.what()).find("mass out of domain"), std::string::npos) << e.what();
                    EXPECT_NEAR(fault_time(e.what()), t_star, 1e-13 * t_star) << e.what();
                }
            }
            IvpOptions o;
            o.t_eval = {0.0, 1.5 * t_star};
            EXPECT_THROW(solve(p, th, {0.0, 2 * t_star}, x0, u, o), std::domain_error);
            // zero flow does not run out, and a mass that grows forward does not either
            o.t_eval = {0.0};
            EXPECT_EQ(solve(p, th, {0.0, 2 * t_star}, x0, u, o).size(), 1u);
            EXPECT_NO_THROW(solve(p, th, {0.0, 60.0}, x0, {0.0, 0.0, 0.0, 0.0}, {}));
            EXPECT_NO_THROW(solve(p, th, {0.0, 60.0}, x0, {0.0, 0.0, 0.0, -1.0}, {}));
        }
    }
}

TEST(Mass, DepletionByIntegrator) {
    // A driver without the check of the solvers (state only, no thrust, no SRP): the stages with m <= 0
    // are refused, GSL shrinks the step, and stops at t* with the fault of the mass.
    SpiceParameters p = two_body(false);
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = strong_thruster(d);
        const std::vector<double> u = {0.0, 0.0, 0.0, 1.0};
        ControlledParameters cp(p, th, u);
        const double t_star = MASS / cp.c2();
        std::vector<double> y = state_of(d, p, state_llo(), MASS);
        gsl_odeiv2_system sys = {d.eom, nullptr, 7, &cp};
        gsl_odeiv2_step* step = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, 7);
        gsl_odeiv2_control* control = gsl_odeiv2_control_y_new(1e-12, 1e-14);
        gsl_odeiv2_evolve* evolve = gsl_odeiv2_evolve_alloc(7);
        gsl_error_handler_t* handler = gsl_set_error_handler_off();
        double t = 0.0, h = 1.0;
        int status = GSL_SUCCESS;
        while (status == GSL_SUCCESS && t < 2 * t_star) {
            status = gsl_odeiv2_evolve_apply(evolve, control, step, &sys, &t, 2 * t_star, &h, y.data());
        }
        gsl_set_error_handler(handler);
        gsl_odeiv2_evolve_free(evolve);
        gsl_odeiv2_control_free(control);
        gsl_odeiv2_step_free(step);
        EXPECT_EQ(status, GSL_EDOM);
        EXPECT_NE(campanella::eom_error(p).find("mass out of domain"), std::string::npos) << campanella::eom_error(p);
        note("integrator, |t - t*| / t*", d.name, false, std::abs(t - t_star) / t_star);
        EXPECT_NEAR(t, t_star, 1e-13 * t_star);
        EXPECT_GT(y[6], 0.0);
    }
}

TEST(Mass, NonPositiveInitialThrows) {
    // no kernels. Over a zero-length span nothing evaluates the RHS, so this is the solver's own check.
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            SCOPED_TRACE(std::string(d.name) + (canon ? ", canonical" : ", km"));
            const ThrustOptions th = thruster(d);
            for (const double m : {0.0, -0.0, -1.0, -1e-300}) {
                const std::vector<double> x0 = state_of(d, p, state_llo(), m);
                for (const std::vector<double>& tspan : {std::vector<double>{0.0, 1.0}, std::vector<double>{0.0, -1.0},
                                                         std::vector<double>{0.0, 0.0}}) {
                    EXPECT_THROW(d.solve(p, th, tspan, x0, U, {}), std::domain_error) << m;
                    EXPECT_THROW(d.solvestm(p, th, tspan, x0, U, {}), std::domain_error) << m;
                }
                try {
                    d.solve(p, th, {0.0, 1.0}, x0, U, {});
                } catch (const std::domain_error& e) {
                    EXPECT_NE(std::string(e.what()).find("mass out of domain"), std::string::npos) << e.what();
                    EXPECT_EQ(fault_time(e.what()), 0.0) << e.what();
                }
            }
            // a mass that is not finite is bad input, as any other element of x0
            for (const double m : {NAN_, INF, -INF}) {
                const std::vector<double> x0 = state_of(d, p, state_llo(), m);
                expect_invalid([&] { d.solve(p, th, {0.0, 1.0}, x0, U, {}); }, "x0 must be finite", d.name);
                expect_invalid([&] { d.solvestm(p, th, {0.0, 0.0}, x0, U, {}); }, "x0 must be finite", d.name);
            }
        }
    }
}

TEST(Mass, BadMassGivesTheSameFaultEverywhere) {
    // Both evaluators, at every level, with and without the Jacobians. The state is also out of the
    // domain in the second case: the mass is checked first. No kernels.
    using namespace campanella::detail;
    struct Case {
        double m;
        Fault fault;
        const char* text;
    };
    const Case cases[] = {
        {0.0, Fault::mass_domain, "mass out of domain"},   {-0.0, Fault::mass_domain, "mass out of domain"},
        {-1.0, Fault::mass_domain, "mass out of domain"},  {-INF, Fault::mass_domain, "mass out of domain"},
        {NAN_, Fault::mass_domain, "mass out of domain"},  {INF, Fault::nonfinite, "non-finite"},
    };
    SpiceParameters p = two_body(false);
    const Model& model = Access::model(p);
    Workspace& ws = Access::workspace(p);
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d);
        const ZohControl control = make_control(p, th, U, "test");
        ControlledParameters cp(p, th, U);
        const std::vector<double> good = state_of(d, p, state_llo(), 1.0);
        for (const Case& c : cases) {
            SCOPED_TRACE(c.m);
            std::vector<double> nominal = good;
            nominal[6] = c.m;
            std::vector<double> outside(7, 0.0);   // at the center of the body; p = 0 in MEE
            outside[6] = c.m;
            for (const std::vector<double>* xp : {&nominal, &outside}) {
                const std::vector<double>& x = *xp;
                double xdot[7], A[7][7], B[7][4];
                using AP = double (*)[7];
                using BP = double (*)[4];
                const struct { AP a; BP b; } outputs[] = {{nullptr, nullptr}, {A, nullptr}, {nullptr, B}, {A, B}};
                for (const auto& out : outputs) {
                    ws.fault = Fault::none;
                    EXPECT_EQ(d.evaluate(model, control, ws, 0.0, x.data(), xdot, out.a, out.b), RHS_EDOM);
                    EXPECT_EQ(ws.fault, c.fault);
                }
                EXPECT_EQ(d.eom(0.0, x.data(), xdot, &cp), GSL_EDOM);
                EXPECT_NE(campanella::eom_error(p).find(c.text), std::string::npos) << campanella::eom_error(p);
                std::vector<double> y(84, 0.0), ydot(84);
                std::copy(x.begin(), x.end(), y.begin());
                EXPECT_EQ(d.eomstm(0.0, y.data(), ydot.data(), &cp), GSL_EDOM);
                EXPECT_NE(campanella::eom_error(p).find(c.text), std::string::npos) << campanella::eom_error(p);
                try {
                    d.eval(p, th, 0.0, x, U);
                    ADD_FAILURE() << "expected std::domain_error";
                } catch (const std::domain_error& e) {
                    EXPECT_NE(std::string(e.what()).find(c.text), std::string::npos) << e.what();
                    // true of a NaN as well as of a negative mass
                    EXPECT_EQ(std::string(e.what()).find("<= 0"), std::string::npos) << e.what();
                }

                // the fault does not stick
                double good_xdot[7];
                EXPECT_EQ(d.eom(0.0, good.data(), good_xdot, &cp), GSL_SUCCESS);
                EXPECT_EQ(campanella::eom_error(p), "");
            }
        }
    }
}

TEST(Evaluators, JacobianOutputsAreIndependent) {
    // Asking for A or B does not change xdot or the other one; a null output is not written to
    REQUIRE_KERNELS();
    using namespace campanella::detail;
    const SpiceParameters p = all_forces(false);
    const Model& model = Access::model(p);
    Workspace& ws = Access::workspace(p);
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ZohControl control = make_control(p, thruster(d), U, "test");
        const std::vector<double> x = state_of(d, p, state_llo(), MASS);
        double xdot[7], A[7][7], B[7][4];
        ASSERT_EQ(d.evaluate(model, control, ws, 100.0, x.data(), xdot, A, B), RHS_SUCCESS);

        double xdot2[7], A2[7][7], B2[7][4];
        ASSERT_EQ(d.evaluate(model, control, ws, 100.0, x.data(), xdot2, nullptr, nullptr), RHS_SUCCESS);
        EXPECT_TRUE(std::equal(xdot, xdot + 7, xdot2));
        ASSERT_EQ(d.evaluate(model, control, ws, 100.0, x.data(), xdot2, A2, nullptr), RHS_SUCCESS);
        EXPECT_TRUE(std::equal(xdot, xdot + 7, xdot2));
        EXPECT_TRUE(std::equal(&A[0][0], &A[0][0] + 49, &A2[0][0]));
        ASSERT_EQ(d.evaluate(model, control, ws, 100.0, x.data(), xdot2, nullptr, B2), RHS_SUCCESS);
        EXPECT_TRUE(std::equal(xdot, xdot + 7, xdot2));
        EXPECT_TRUE(std::equal(&B[0][0], &B[0][0] + 28, &B2[0][0]));
    }
}

TEST(Invalid, X0Length) {
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            const ThrustOptions th = thruster(d);
            for (const size_t n : {size_t(0), size_t(6), size_t(8), size_t(84)}) {
                const std::vector<double> x(n, 1.0);
                expect_invalid([&] { d.solve(p, th, {0.0, 1.0}, x, U, {}); }, "x0 must have length 7", d.name);
                expect_invalid([&] { d.solvestm(p, th, {0.0, 1.0}, x, U, {}); }, "x0 must have length 7", d.name);
                expect_invalid([&] { d.eval(p, th, 0.0, x, U); }, "x must have length 7", d.name);
            }
        }
    }
}

TEST(Invalid, X0NotFinite) {
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            const ThrustOptions th = thruster(d);
            for (int i = 0; i < 7; i++) {
                for (const double bad : {NAN_, INF, -INF}) {
                    std::vector<double> x = state_of(d, p, state_llo(), MASS);
                    x[i] = bad;
                    expect_invalid([&] { d.solve(p, th, {0.0, 1.0}, x, U, {}); }, "x0 must be finite", d.name);
                    expect_invalid([&] { d.solvestm(p, th, {0.0, 1.0}, x, U, {}); }, "x0 must be finite", d.name);
                }
            }
        }
    }
}

TEST(Invalid, ULength) {
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            const ThrustOptions th = thruster(d);
            for (const size_t n : {size_t(0), size_t(3), size_t(5)}) {
                expect_control_invalid(d, p, th, std::vector<double>(n, 0.1), "u must have length 4",
                                       "length " + std::to_string(n));
            }
        }
    }
}

TEST(Invalid, UNotFinite) {
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            const ThrustOptions th = thruster(d);
            for (int i = 0; i < 4; i++) {
                for (const double bad : {NAN_, INF, -INF}) {
                    std::vector<double> u = U;
                    u[i] = bad;
                    expect_control_invalid(d, p, th, u, "u must be finite", "u[" + std::to_string(i) + "]");
                }
            }
        }
    }
}

TEST(Invalid, ThrustOptionsFields) {
    struct Field {
        const char* name;
        double ThrustOptions::*member;
    };
    const Field fields[] = {{"thrust_N", &ThrustOptions::thrust_N},
                            {"isp_s", &ThrustOptions::isp_s},
                            {"mass_unit_kg", &ThrustOptions::mass_unit_kg},
                            {"g0", &ThrustOptions::g0}};
    for (const bool canon : {false, true}) {
        const SpiceParameters p = two_body(canon);
        for (const Dyn& d : DYNS) {
            for (const Field& field : fields) {
                for (const double bad : {0.0, -0.0, -1.0, NAN_, INF, -INF}) {
                    ThrustOptions th = thruster(d);
                    th.*field.member = bad;
                    expect_control_invalid(d, p, th, U, std::string(field.name) + " must be finite and > 0", field.name);
                }
            }
            // a default ThrustOptions has no propulsion system
            expect_control_invalid(d, p, ThrustOptions{}, U, "thrust_N", "default");

            // no enumerator
            ThrustOptions th = thruster(d);
            th.mee_control_frame = static_cast<ThrustFrame>(2);
            expect_control_invalid(d, p, th, U, "mee_control_frame", "frame");

            // valid one by one, but the constants overflow or underflow
            th = thruster(d);
            th.thrust_N = 1e308;
            th.mass_unit_kg = 1e-300;
            expect_control_invalid(d, p, th, U, "c1", "overflow");
            th = thruster(d);
            th.thrust_N = 1e-300;
            th.mass_unit_kg = 1e300;
            expect_control_invalid(d, p, th, U, "c1", "underflow");
        }
    }
}

TEST(Invalid, GammaIsNotChecked) {
    // the solver of the optimal control problem owns Gamma = |tau|: neither Gamma < 0 nor |tau| > Gamma is rejected
    const SpiceParameters p = two_body(false);
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d);
        const std::vector<double> x0 = state_of(d, p, state_llo(), MASS);
        for (const std::vector<double>& u : {std::vector<double>{1.0, 1.0, 1.0, -1.0},
                                             std::vector<double>{5.0, 0.0, 0.0, 0.1}}) {
            EXPECT_NO_THROW(d.solve(p, th, {0.0, 60.0}, x0, u, {})) << d.name;
            EXPECT_NO_THROW(d.eval(p, th, 0.0, x0, u)) << d.name;
            EXPECT_NO_THROW(ControlledParameters(p, th, u)) << d.name;
        }
    }
}

TEST(Eom, AgreesWithEval) {
    // 84 states with an arbitrary Phi and Psi, with a nonzero mass row: full products of A and B
    REQUIRE_KERNELS();
    for (const bool canon : {false, true}) {
        SpiceParameters p = all_forces(canon);
        for (const Dyn& d : DYNS) {
            SCOPED_TRACE(std::string(d.name) + (canon ? ", canonical" : ", km"));
            const ThrustOptions th = thruster(d);
            ControlledParameters cp(p, th, U);
            const double t = 100.0 / p.TU();
            const std::vector<double> x = state_of(d, p, state_llo(), MASS);
            const EomEvalControl e = d.eval(p, th, t, x, U);

            double xdot[7], y[84], ydot[84];
            ASSERT_EQ(d.eom(t, x.data(), xdot, &cp), GSL_SUCCESS);
            std::copy(x.begin(), x.end(), y);
            for (int i = 7; i < 84; i++) {
                y[i] = std::sin(1.0 + i);
            }
            ASSERT_EQ(d.eomstm(t, y, ydot, &cp), GSL_SUCCESS);
            for (int i = 0; i < 7; i++) {
                EXPECT_EQ(xdot[i], e.xdot[i]) << i;
                EXPECT_EQ(ydot[i], e.xdot[i]) << i;
                // the rounding of a sum of 8 terms is below 1e-15 of the sum of their magnitudes
                for (int j = 0; j < 7; j++) {
                    double sum = 0.0, scale = 0.0;
                    for (int l = 0; l < 7; l++) {
                        sum += e.A[7 * i + l] * y[7 + 7 * l + j];
                        scale += std::abs(e.A[7 * i + l] * y[7 + 7 * l + j]);
                    }
                    EXPECT_NEAR(ydot[7 + 7 * i + j], sum, 1e-15 * scale) << "Phi " << i << "," << j;
                }
                for (int j = 0; j < 4; j++) {
                    double sum = e.B[4 * i + j], scale = std::abs(e.B[4 * i + j]);
                    for (int l = 0; l < 7; l++) {
                        sum += e.A[7 * i + l] * y[56 + 4 * l + j];
                        scale += std::abs(e.A[7 * i + l] * y[56 + 4 * l + j]);
                    }
                    EXPECT_NEAR(ydot[56 + 4 * i + j], sum, 1e-15 * scale) << "Psi " << i << "," << j;
                }
            }
            // what was compared is not trivially zero
            double mass_column = 0.0, b_tau = 0.0;
            for (int i = 0; i < 6; i++) {
                mass_column = std::max(mass_column, std::abs(e.A[7 * i + 6]));
                for (int j = 0; j < 3; j++) {
                    b_tau = std::max(b_tau, std::abs(e.B[4 * i + j]));
                }
            }
            EXPECT_GT(mass_column, 0.0);
            EXPECT_GT(b_tau, 0.0);
            EXPECT_EQ(e.B[4 * 6 + 3], -cp.c2());
        }
    }
}

TEST(Eom, UsableAsGslSystem) {
    // eom_* drive GSL's own driver to the same result as the solvers
    REQUIRE_KERNELS();
    SpiceOptions o = with_gms(GM_LIST);
    add_srp(o);
    SpiceParameters p(ET0, BODIES, o);
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d);
        ControlledParameters cp(p, th, U);
        const std::vector<double> x0 = state_of(d, p, STATE0, MASS);
        const std::vector<double> s = scales(d, x0);
        const std::vector<double> tspan = {0.0, 86400.0};

        gsl_odeiv2_system sys = {d.eom, nullptr, 7, &cp};
        gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, 1.0, 1e-12, 1e-14);
        double t = 0.0;
        std::vector<double> y = x0;
        int status = gsl_odeiv2_driver_apply(driver, &t, tspan[1], y.data());
        gsl_odeiv2_driver_free(driver);
        ASSERT_EQ(status, GSL_SUCCESS);
        const double err_state = scaled_max_diff(y, final_state_n(d.solve(p, th, tspan, x0, U, {}), 7), s);
        note("GSL driver, state", d.name, false, err_state);
        EXPECT_LE(err_state, TOL_GSL);

        gsl_odeiv2_system sys_stm = {d.eomstm, nullptr, 84, &cp};
        driver = gsl_odeiv2_driver_alloc_y_new(&sys_stm, gsl_odeiv2_step_rk8pd, 1.0, 1e-12, 1e-14);
        t = 0.0;
        std::vector<double> y84(84, 0.0);
        std::copy(x0.begin(), x0.end(), y84.begin());
        for (int i = 0; i < 7; i++) {
            y84[7 + 8 * i] = 1.0;
        }
        status = gsl_odeiv2_driver_apply(driver, &t, tspan[1], y84.data());
        gsl_odeiv2_driver_free(driver);
        ASSERT_EQ(status, GSL_SUCCESS);
        const IvpResult ref = d.solvestm(p, th, tspan, x0, U, {});
        const std::vector<double> x_gsl(y84.begin(), y84.begin() + 7), Phi(y84.begin() + 7, y84.begin() + 56),
            Psi(y84.begin() + 56, y84.end());
        const double err_stm_state = scaled_max_diff(x_gsl, final_state_n(ref, 7), s);
        const double err_phi = matrix_error(Phi, final_stm_n(ref, 7), 7, 7, s, s, 0.0);
        const double err_psi = matrix_error(Psi, final_stm_u(ref, 7, 4), 7, 4, s, {}, 0.0);
        note("GSL driver, stm state", d.name, false, err_stm_state);
        note("GSL driver, Phi", d.name, false, err_phi);
        note("GSL driver, Psi", d.name, false, err_psi);
        EXPECT_LE(err_stm_state, TOL_GSL);
        EXPECT_LE(err_phi, TOL_GSL);
        EXPECT_LE(err_psi, TOL_GSL);
        // the thrust is really there
        EXPECT_GT(scaled_max_diff(x_gsl, x0, s), 1e-3);
    }
}

TEST(Eom, SpiceErrorStopsAndIsReported) {
    // a SPICE error stops GSL, as for eom_rv, and the next valid call clears it
    REQUIRE_KERNELS();
    SpiceParameters pluto(ET0, {"MOON", "PLUTO"}, with_gms({MU_MOON, 869.6}));
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d);
        ControlledParameters cp(pluto, th, U);
        const std::vector<double> x = state_of(d, pluto, STATE0, MASS);
        double xdot[7];
        EXPECT_EQ(d.eom(0.0, x.data(), xdot, &cp), GSL_EBADFUNC);
        EXPECT_NE(campanella::eom_error(pluto).find("SPICE(SPKINSUFFDATA)"), std::string::npos)
            << campanella::eom_error(pluto);
        EXPECT_THROW(d.eval(pluto, th, 0.0, x, U), std::runtime_error);
        EXPECT_THROW(d.solve(pluto, th, {0.0, 1.0}, x, U, {}), std::runtime_error);
    }
}

// ---------------------------------------------------------------- max_steps

namespace {

// Thrust that deorbits STATE0: from about 9.9 days p -> 0 without reaching p <= 0. The steps collapse and
// no error ever comes, at about 30000 steps/s.
ThrustOptions deorbit_thruster() {
    ThrustOptions th;
    th.thrust_N = 0.5;
    th.isp_s = 3000.0;
    th.mass_unit_kg = 1850.0;
    return th;
}

double cpu_seconds() { return static_cast<double>(std::clock())/CLOCKS_PER_SEC; }

// CPU time, so that a loaded machine does not fail it: 3.5 s were measured for 100000 steps
constexpr long long STALL_STEPS = 100000;
constexpr double STALL_CPU_S = 60.0;
constexpr double STALL_TF = 20*86400.0;

std::vector<double> stall_x0() {
    std::vector<double> x = campanella::cart2mee(STATE0, GM_LIST[0]);
    x.push_back(1.0);
    return x;
}

void expect_contains(const std::string& msg, const std::string& what) {
    EXPECT_NE(msg.find(what), std::string::npos) << msg;
}

}  // namespace

TEST(MaxSteps, DefaultValueIsTheOneInTheDocumentation) {
    EXPECT_EQ(IvpOptions{}.max_steps, 2000000);
}

TEST(MaxSteps, StallThrowsInBoundedWork) {
    REQUIRE_KERNELS();
    const SpiceParameters p(ET0, {"MOON", "EARTH", "SUN"}, with_gms(GM_LIST));
    const std::vector<double> u = {0.3, -0.5, 0.4, 0.71};
    IvpOptions o;
    o.max_steps = STALL_STEPS;
    const struct { const char* name; Solver solve; } cases[] = {
        {"solve_ivp_meem", campanella::solve_ivp_meem}, {"solve_ivpstm_meem", campanella::solve_ivpstm_meem}};
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        const double start = cpu_seconds();
        try {
            c.solve(p, deorbit_thruster(), {0.0, STALL_TF}, stall_x0(), u, o);
            ADD_FAILURE() << "expected std::runtime_error";
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            expect_contains(msg, std::string(c.name) + ": reached max_steps = 100000 at t = ");
            expect_contains(msg, "p -> 0");
            expect_contains(msg, "0 disables the limit");
        }
        EXPECT_LT(cpu_seconds() - start, STALL_CPU_S);
    }
}

TEST(MaxSteps, PollRunsEvery1024Steps) {
    REQUIRE_KERNELS();
    const SpiceParameters p(ET0, {"MOON", "EARTH", "SUN"}, with_gms(GM_LIST));
    IvpOptions o;
    long long polls = 0;
    o.max_steps = 20480;
    o.poll = [&polls] { polls++; };
    EXPECT_THROW(campanella::solve_ivp_meem(p, deorbit_thruster(), {0.0, STALL_TF}, stall_x0(),
                                            {0.3, -0.5, 0.4, 0.71}, o), std::runtime_error);
    EXPECT_EQ(polls, 20);   // steps 0, 1024, ..., 19456
}

TEST(MaxSteps, CountsAcceptedStepsExactly) {
    const SpiceParameters p(ET0, {"MOON"}, with_gms({GM_LIST[0]}));
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        ThrustOptions th = thruster(d);
        th.mee_control_frame = d.frame;
        std::vector<double> x0 = d.mee ? campanella::cart2mee(STATE0, GM_LIST[0]) : STATE0;
        x0.push_back(1.0);
        const std::vector<double> tspan = {0.0, 86400.0};
        const IvpResult full = d.solve(p, th, tspan, x0, U, {});
        const long long steps = static_cast<long long>(full.size()) - 1;
        ASSERT_GT(steps, 2);
        for (const long long limit : {steps, steps + 1, 0LL}) {
            IvpOptions o;
            o.max_steps = limit;
            const IvpResult r = d.solve(p, th, tspan, x0, U, o);
            EXPECT_EQ(r.x, full.x) << limit;
        }
        IvpOptions low;
        low.max_steps = steps - 1;
        EXPECT_THROW(d.solve(p, th, tspan, x0, U, low), std::runtime_error);
        // over all of tspan, not per output time
        IvpOptions te;
        te.max_steps = 1;
        te.t_eval = {21600.0, 43200.0, 86400.0};
        EXPECT_THROW(d.solve(p, th, tspan, x0, U, te), std::runtime_error);
        // a zero-length span takes no step
        IvpOptions one;
        one.max_steps = 1;
        EXPECT_EQ(d.solve(p, th, {5.0, 5.0}, x0, U, one).size(), 1u);
    }
}

TEST(MaxSteps, NegativeIsRejected) {
    const SpiceParameters p(ET0, {"MOON"}, with_gms({GM_LIST[0]}));
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        std::vector<double> x0 = d.mee ? campanella::cart2mee(STATE0, GM_LIST[0]) : STATE0;
        x0.push_back(1.0);
        IvpOptions o;
        o.max_steps = -1;
        for (const Solver solve : {d.solve, d.solvestm}) {
            try {
                solve(p, thruster(d), {0.0, 0.0}, x0, U, o);
                ADD_FAILURE() << "expected std::invalid_argument";
            } catch (const std::invalid_argument& e) {
                expect_contains(e.what(), "max_steps must be >= 0");
            }
        }
    }
}

TEST(MaxSteps, RvAndMeeIgnoreIt) {
    const SpiceParameters p(ET0, {"MOON"}, with_gms({GM_LIST[0]}));
    const std::vector<double> m = campanella::cart2mee(STATE0, GM_LIST[0]);
    const std::vector<double> tspan = {0.0, 86400.0};
    int polls = 0;
    for (const long long limit : {1LL, -1LL}) {
        IvpOptions o;
        o.max_steps = limit;
        o.poll = [&polls] { polls++; };
        EXPECT_EQ(campanella::solve_ivp_rv(p, tspan, STATE0, o).x, campanella::solve_ivp_rv(p, tspan, STATE0).x);
        EXPECT_EQ(campanella::solve_ivp_mee(p, tspan, m, o).x, campanella::solve_ivp_mee(p, tspan, m).x);
        EXPECT_EQ(campanella::solve_ivpstm_rv(p, tspan, STATE0, o).stm, campanella::solve_ivpstm_rv(p, tspan, STATE0).stm);
        EXPECT_EQ(campanella::solve_ivpstm_mee(p, tspan, m, o).stm, campanella::solve_ivpstm_mee(p, tspan, m).stm);
    }
    EXPECT_EQ(polls, 0);
}

TEST(MaxSteps, PollCanStopTheIntegration) {
    const SpiceParameters p(ET0, {"MOON"}, with_gms({GM_LIST[0]}));
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        std::vector<double> x0 = d.mee ? campanella::cart2mee(STATE0, GM_LIST[0]) : STATE0;
        x0.push_back(1.0);
        IvpOptions o;
        o.poll = [] { throw std::logic_error("stop"); };
        EXPECT_THROW(d.solve(p, thruster(d), {0.0, 86400.0}, x0, U, o), std::logic_error);
        // a poll that returns changes nothing, and the handler of GSL and the parameters are left as they were
        IvpOptions quiet;
        quiet.poll = [] {};
        EXPECT_EQ(d.solve(p, thruster(d), {0.0, 86400.0}, x0, U, quiet).x, d.solve(p, thruster(d), {0.0, 86400.0}, x0, U, {}).x);
    }
}

TEST(MaxSteps, LongArcsStayUnderTheDefault) {
    // Three years of thrust from a low lunar orbit, in km and in canonical units. Cartesian takes the most
    // steps: 1.1e6 in km, near enough to the default for this to fail if the limit is lowered much.
    const std::vector<double> rv = {1837.4, 0.0, 0.0, 0.0, 1.46, 0.8};
    std::vector<double> rvm = rv;
    rvm.push_back(1.0);
    std::vector<double> mee = campanella::cart2mee(rv, GM_LIST[0]);
    mee.push_back(1.0);
    ThrustOptions th;
    th.thrust_N = 0.1;
    th.isp_s = 3000.0;
    th.mass_unit_kg = 1850.0;
    const std::vector<double> u_rtn = {0.0, 1.0, 0.0, 1.0};
    const std::vector<double> u_inertial = {0.0, 0.1, 0.05, 0.1118};
    const SpiceParameters km(ET0, {"MOON"}, with_gms({GM_LIST[0]}));
    SpiceOptions o = with_gms({GM_LIST[0]});
    o.DU = 1737.4;
    o.use_canonical_scales = true;
    const SpiceParameters canon(ET0, {"MOON"}, o);
    std::vector<double> mee_canon = mee;
    mee_canon[0] /= canon.DU();
    std::vector<double> rvm_canon = rvm;
    for (int i = 0; i < 3; i++) {
        rvm_canon[i] /= canon.DU();
        rvm_canon[3 + i] /= canon.VU();
    }
    const double tf = 3*365.25*86400.0;
    const IvpResult a = campanella::solve_ivp_rvm(km, th, {0.0, tf}, rvm, u_inertial);
    EXPECT_EQ(a.t.back(), tf);
    EXPECT_GT(a.size(), 500000u);
    EXPECT_EQ(campanella::solve_ivp_rvm(canon, th, {0.0, tf/canon.TU()}, rvm_canon, u_inertial).t.back(), tf/canon.TU());
    EXPECT_EQ(campanella::solve_ivp_meem(km, th, {0.0, tf}, mee, u_rtn).t.back(), tf);
    EXPECT_EQ(campanella::solve_ivp_meem(canon, th, {0.0, tf/canon.TU()}, mee_canon, u_rtn).t.back(), tf/canon.TU());
}
