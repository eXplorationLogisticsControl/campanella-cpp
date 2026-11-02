// Validation of the propagated mass-augmented ZOH dynamics, against results derived independently of the
// code: the closed-form mass and rocket equation, finite differences of the final state, quadrature,
// the semigroup property, backward propagation, Cartesian against MEE, and changes of units.
//
// Constants c1 and c2 are computed here from the physical inputs on purpose, rather than read from
// ControlledParameters. Set ZOH_PRINT_ERRORS to print every measured error with its tolerance.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "helpers.hpp"

using namespace campanella_test;
using campanella::EomEvalControl;
using campanella::IvpOptions;
using campanella::IvpResult;
using campanella::SpiceOptions;
using campanella::SpiceParameters;
using campanella::ThrustFrame;
using campanella::ThrustOptions;

namespace {

using Vec = std::vector<double>;
using Solver = IvpResult (*)(const SpiceParameters&, const ThrustOptions&, const Vec&, const Vec&, const Vec&,
                             const IvpOptions&);
using Eval = EomEvalControl (*)(const SpiceParameters&, const ThrustOptions&, double, const Vec&, const Vec&);

struct Dyn {
    const char* name;
    bool mee;
    ThrustFrame frame;
    Solver solve;
    Solver solvestm;
    Eval eval;
};

const Dyn RVM = {"rvm", false, ThrustFrame::rtn, campanella::solve_ivp_rvm, campanella::solve_ivpstm_rvm,
                 campanella::eval_rvm};
const Dyn MEEM_RTN = {"meem-rtn", true, ThrustFrame::rtn, campanella::solve_ivp_meem,
                      campanella::solve_ivpstm_meem, campanella::eval_meem};
const Dyn MEEM_INERTIAL = {"meem-inertial", true, ThrustFrame::inertial, campanella::solve_ivp_meem,
                           campanella::solve_ivpstm_meem, campanella::eval_meem};
const Dyn DYNS[] = {RVM, MEEM_RTN, MEEM_INERTIAL};

const Vec U = {0.3, -0.5, 0.4, 0.71};
const double DAY = 86400.0;
const double FLOOR = 1e-14;

// Tolerances that the issue and plan do not fix, or that they fix below what is achievable: about 10
// times the largest value measured with ZOH_PRINT_ERRORS (in the comments), on Linux x86-64 in Release.
const double TOL_BALLISTIC_STATE = 6e-12;    // 6.3e-13; issue 1e-12, the step sequences differ
const double TOL_BALLISTIC_STM = 4e-11;      // 4.3e-12; issue 1e-12, the step sequences differ
const double TOL_CROSSVAL_MASS = 7e-14;      // 7.1e-15; issue "identical", the step sequences differ
const double TOL_THRUST_COUPLING = 2e-3;     // 1.7e-4, the neglected higher-order terms
const double TOL_SRP_COUPLING = 2e-2;        // 1.6e-3, the neglected higher-order terms
const double TOL_SEMIGROUP = 2e-11;          // 1.7e-12
const double TOL_ROUNDTRIP_STATE = 1e-12;    // 7.1e-14
const double TOL_ROUNDTRIP_PHI = 7e-11;      // 6.3e-12
const double TOL_ROUNDTRIP_PSI = 1e-11;      // 9.7e-13
const double TOL_TSIOLKOVSKY_V = 6e-14;      // 5.3e-15 km/s
const double TOL_TSIOLKOVSKY_R = 3e-6;       // 2.4e-7 km, at |r| = 2.3e9 km
const double TOL_TSIOLKOVSKY_M = 3e-15;      // 2.8e-16
const double TOL_ROCKET = 3e-14;             // 2.6e-15
const double TOL_UNITS_STATE = 7e-13;        // 6.5e-14
const double TOL_UNITS_SENS = 1e-11;         // 1.0e-12
const double TOL_MASS_UNIT_STATE = 5e-13;    // 4.3e-14
const double TOL_MASS_UNIT_SENS = 1e-11;     // 1.0e-12

void check(const std::string& what, double value, double tol) {
    if (std::getenv("ZOH_PRINT_ERRORS") != nullptr) {
        std::printf("ZOHVAL %-58s %.3e  tol %.1e\n", what.c_str(), value, tol);
        std::fflush(stdout);
    }
    EXPECT_LE(value, tol) << what;
}

// The default keeps the orbit of STATE0 for 20 days: 0.5 N on 1850 kg would give a delta-v
// larger than its orbital speed, and deorbit it.
ThrustOptions thruster(const Dyn& d, double thrust_N = 0.05, double isp_s = 3000.0, double mass_unit_kg = 1850.0) {
    ThrustOptions th;
    th.thrust_N = thrust_N;
    th.isp_s = isp_s;
    th.mass_unit_kg = mass_unit_kg;
    th.mee_control_frame = d.frame;
    return th;
}

double c1_of(const ThrustOptions& th, const SpiceParameters& p) {
    return (th.thrust_N * 1e-3) / th.mass_unit_kg / (p.DU() / (p.TU() * p.TU()));
}

double c2_of(const ThrustOptions& th, const SpiceParameters& p) {
    return th.thrust_N / (th.isp_s * th.g0) / th.mass_unit_kg * p.TU();
}

SpiceParameters moon_params(bool srp, bool canon = false, double srp_m = 1850.0) {
    SpiceOptions o = with_gms(GM_LIST);
    if (srp) {
        add_srp(o);
        o.srp_m = srp_m;
    }
    if (canon) {
        o.DU = 1737.4;
    }
    return SpiceParameters(ET0, BODIES, o);
}

// A Cartesian state in km, km/s, in the units and coordinates of d, with the mass appended
Vec state_of(const Dyn& d, const SpiceParameters& p, Vec rv, double m) {
    for (int i = 0; i < 3; i++) {
        rv[i] /= p.DU();
        rv[3 + i] /= p.VU();
    }
    Vec x = d.mee ? campanella::cart2mee(rv, p.mus()[0]) : rv;
    x.push_back(m);
    return x;
}

// Scale of each state: |r|, |v|, m in Cartesian; p, 1, m in MEE (s_p = p, as gve_mee.md 7.4)
Vec scales(const Dyn& d, const Vec& x) {
    if (d.mee) {
        return {x[0], 1.0, 1.0, 1.0, 1.0, 1.0, x[6]};
    }
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const double v = std::sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]);
    return {r, r, r, v, v, v, x[6]};
}

// Largest scaled difference of two states
double state_error(const Vec& a, const Vec& b, const Vec& s) {
    double e = 0.0;
    for (size_t i = 0; i < s.size(); i++) {
        const double di = std::abs(a[i] - b[i]) / s[i];
        e = std::isfinite(di) ? std::max(e, di) : std::numeric_limits<double>::infinity();
    }
    return e;
}

// A row-major n x m product: A is n x n, B is n x m
Vec product(const Vec& A, const Vec& B, int n, int m) {
    Vec C(n * m, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < n; l++) {
                C[m * i + j] += A[n * i + l] * B[m * l + j];
            }
        }
    }
    return C;
}

Vec identity7() {
    Vec I(49, 0.0);
    for (int i = 0; i < 7; i++) {
        I[8 * i] = 1.0;
    }
    return I;
}

// Row and column metrics, both to the same tolerance
void check_matrix(const std::string& what, const Vec& M, const Vec& R, int rows, int cols, const Vec& row_scale,
                  const Vec& col_scale, double tol) {
    check(what + " row", matrix_error(M, R, rows, cols, row_scale, col_scale, FLOOR, ErrorMode::row), tol);
    check(what + " col", matrix_error(M, R, rows, cols, row_scale, col_scale, FLOOR, ErrorMode::col), tol);
}

Vec trajectory_rows(const Vec& Psi) { return Vec(Psi.begin(), Psi.begin() + 24); }

Vec fd_steps_state(const Dyn& d, const SpiceParameters& p, const Vec& x0) {
    const double dm = 1e-3 * x0[6];
    if (d.mee) {
        return {1e-6 * x0[0], 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, dm};
    }
    const double dr = 1e-3 / p.DU(), dv = 1e-6 / p.VU();
    return {dr, dr, dr, dv, dv, dv, dm};
}

Vec fd_phi(const Dyn& d, const SpiceParameters& p, const ThrustOptions& th, double tf, const Vec& x0, const Vec& u) {
    return jacobian_fd_n(
        [&](const Vec& x) { return final_state_n(d.solve(p, th, {0.0, tf}, x, u, IvpOptions{}), 7); }, x0,
        fd_steps_state(d, p, x0));
}

Vec fd_psi(const Dyn& d, const SpiceParameters& p, const ThrustOptions& th, double tf, const Vec& x0, const Vec& u) {
    return jacobian_fd_n(
        [&](const Vec& v) { return final_state_n(d.solve(p, th, {0.0, tf}, x0, v, IvpOptions{}), 7); }, u,
        {1e-4, 1e-4, 1e-4, 1e-2});
}

void check_phi_fd(const std::string& label, const Dyn& d, const SpiceParameters& p, const ThrustOptions& th,
                  double tf, const Vec& x0, const Vec& u) {
    const IvpResult r = d.solvestm(p, th, {0.0, tf}, x0, u, IvpOptions{});
    const Vec s = scales(d, x0);
    const Vec Phi = final_stm_n(r, 7), fd = fd_phi(d, p, th, tf, x0, u);
    check_matrix(label + " " + d.name + " Phi", Phi, fd, 7, 7, s, s, 1e-4);
    // The mass row dominates the mass column; without it, that column shows the thrust and SRP terms
    check_matrix(label + " " + d.name + " Phi r,v rows", Vec(Phi.begin(), Phi.begin() + 42),
                 Vec(fd.begin(), fd.begin() + 42), 6, 7, Vec(s.begin(), s.begin() + 6), s, 1e-4);
}

void check_psi_fd(const std::string& label, const Dyn& d, const SpiceParameters& p, const ThrustOptions& th,
                  double tf, const Vec& x0, const Vec& u) {
    const IvpResult r = d.solvestm(p, th, {0.0, tf}, x0, u, IvpOptions{});
    const Vec s = scales(d, x0), Psi = final_stm_u(r, 7, 4), fd = fd_psi(d, p, th, tf, x0, u);
    check_matrix(label + " " + d.name + " Psi", Psi, fd, 7, 4, s, {}, 1e-4);
    // The mass row dominates the Gamma column; without it, that column shows the coupling through m
    check_matrix(label + " " + d.name + " Psi r,v rows", trajectory_rows(Psi), trajectory_rows(fd), 6, 4,
                 Vec(s.begin(), s.begin() + 6), {}, 1e-4);
}

// A short segment of a strong thruster: c2 dt / 2 < 1e-4, so that Gamma's coupling through the mass is
// below the threshold of the row metric, yet well above the noise of the finite differences
const double SHORT_THRUST_N = 20.0;
const double SHORT_TF = 400.0;

}  // namespace

// ---- Mass ------------------------------------------------------------------------------------------

namespace {

void check_mass_samples(const std::string& label, const Dyn& d, const SpiceParameters& p, const ThrustOptions& th,
                        double t0, double tf, double m0, double tol) {
    const Vec x0 = state_of(d, p, STATE0, m0);
    const double rate = c2_of(th, p) * U[3];
    const IvpResult r = d.solve(p, th, {t0, tf}, x0, U, IvpOptions{});
    double err = 0.0;
    for (size_t k = 0; k < r.size(); k++) {
        const double expected = m0 - rate * (r.t[k] - t0);
        err = std::max(err, std::abs(r.x[7 * k + 6] - expected) / expected);
    }
    check("Mass " + label + " " + d.name, err, tol);
}

void check_mass_rows(const std::string& label, const Dyn& d, const SpiceParameters& p, const ThrustOptions& th,
                     double t0, double tf, double tol) {
    const Vec x0 = state_of(d, p, STATE0, 0.9);
    const double c2 = c2_of(th, p);
    const IvpResult r = d.solvestm(p, th, {t0, tf}, x0, U, IvpOptions{});
    ASSERT_GT(r.size(), 2u);
    // Normalised by the whole span: the rounding of t[k] - t0 makes no sense relative to a first small step
    const double span = std::abs(c2 * (tf - t0));
    double err = 0.0;
    for (size_t k = 0; k < r.size(); k++) {
        for (int j = 0; j < 7; j++) {
            ASSERT_EQ(r.stm[49 * k + 42 + j], j == 6 ? 1.0 : 0.0) << d.name << " k " << k << " j " << j;
        }
        for (int j = 0; j < 3; j++) {
            ASSERT_EQ(r.stm_u[28 * k + 24 + j], 0.0) << d.name << " k " << k << " j " << j;
        }
        err = std::max(err, std::abs(r.stm_u[28 * k + 27] + c2 * (r.t[k] - t0)) / span);
    }
    check("Psi m,Gamma " + label + " " + d.name, err, tol);
}

}  // namespace

TEST(ZohValMass, ClosedForm) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    const SpiceParameters pc = moon_params(true, true);
    for (const Dyn& d : DYNS) {
        check_mass_samples("short", d, p, thruster(d, 20.0), 1000.0, 1000.0 + 3600.0, 0.95, 1e-13);
        check_mass_samples("short backward", d, p, thruster(d, 20.0), 3600.0, 0.0, 0.5, 1e-13);
        check_mass_samples("short canonical", d, pc, thruster(d, 20.0), 0.0, 3600.0 / pc.TU(), 0.95, 1e-13);
        check_mass_samples("long", d, p, thruster(d), 0.0, TF, 1.0, 1e-12);
    }
}

TEST(ZohValMass, RowsExact) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    for (const Dyn& d : DYNS) {
        check_mass_rows("short", d, p, thruster(d, 20.0), 1000.0, 1000.0 + 3600.0, 1e-13);
        check_mass_rows("short backward", d, p, thruster(d, 20.0), 3600.0, 0.0, 1e-13);
        check_mass_rows("long", d, p, thruster(d), 0.0, TF, 1e-12);
    }
}

// ---- Gamma ------------------------------------------------------------------------------------------

TEST(ZohValGamma, IndependentWhenBallisticNoSrp) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(false);
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d);
        const Vec x0 = state_of(d, p, STATE0, 1.0);
        const IvpResult r = d.solvestm(p, th, {0.0, 5 * DAY}, x0, {0.0, 0.0, 0.0, 0.7}, IvpOptions{});
        for (size_t k = 0; k < r.size(); k++) {
            for (int i = 0; i < 6; i++) {
                ASSERT_EQ(r.stm_u[28 * k + 4 * i + 3], 0.0) << d.name << " k " << k << " i " << i;
            }
        }
        // The mass does burn, and the mass row still sees Gamma
        EXPECT_LT(r.x.back(), 1.0) << d.name;
        EXPECT_LT(r.stm_u.back(), 0.0) << d.name;
    }
}

TEST(ZohValGamma, CoupledWithThrust) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(false);
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d, SHORT_THRUST_N);
        const Vec x0 = state_of(d, p, STATE0, 1.0);
        const Vec Psi = final_stm_u(d.solvestm(p, th, {0.0, SHORT_TF}, x0, U, IvpOptions{}), 7, 4);
        double largest = 0.0;
        for (int i = 0; i < 6; i++) {
            largest = std::max(largest, std::abs(Psi[4 * i + 3]));
        }
        EXPECT_GT(largest, 0.0) << d.name;
    }
    // Over a short arc, the leading term of d(int c1 tau / m)/dGamma: c1 c2 tau dt^2 / (2 m^2).
    const ThrustOptions th = thruster(RVM, SHORT_THRUST_N);
    const double m = 0.8;
    const double c = c1_of(th, p) * c2_of(th, p) * SHORT_TF * SHORT_TF / (2 * m * m);
    const Vec Psi = final_stm_u(RVM.solvestm(p, th, {0.0, SHORT_TF}, state_of(RVM, p, STATE0, m), U, IvpOptions{}), 7, 4);
    double err = 0.0;
    for (int i = 0; i < 3; i++) {
        err = std::max(err, std::abs(Psi[4 * (3 + i) + 3] - c * U[i]) / std::abs(c * U[i]));
    }
    check("Gamma thrust coupling vs leading term rvm", err, TOL_THRUST_COUPLING);
}

TEST(ZohValGamma, CoupledWithSrp) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true), p0 = moon_params(false);
    const Vec u = {0.0, 0.0, 0.0, 0.7};
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d, SHORT_THRUST_N);
        const Vec x0 = state_of(d, p, STATE0, 1.0);
        const Vec Psi = final_stm_u(d.solvestm(p, th, {0.0, 3600.0}, x0, u, IvpOptions{}), 7, 4);
        double largest = 0.0;
        for (int i = 0; i < 6; i++) {
            largest = std::max(largest, std::abs(Psi[4 * i + 3]));
        }
        EXPECT_GT(largest, 0.0) << d.name;
    }
    // Lighter means more SRP: over a short arc, Psi_v,Gamma = a_srp c2 dt^2 / (2 m)
    const ThrustOptions th = thruster(RVM, SHORT_THRUST_N);
    const double m = 0.8, dt = 3600.0;
    const Vec x0 = state_of(RVM, p, STATE0, m);
    const EomEvalControl on = campanella::eval_rvm(p, th, 0.0, x0, u);
    const EomEvalControl off = campanella::eval_rvm(p0, th, 0.0, x0, u);
    const Vec Psi = final_stm_u(RVM.solvestm(p, th, {0.0, dt}, x0, u, IvpOptions{}), 7, 4);
    double err = 0.0, norm = 0.0;
    for (int i = 0; i < 3; i++) {
        const double expected = (on.xdot[3 + i] - off.xdot[3 + i]) * c2_of(th, p) * dt * dt / (2 * m);
        err = std::max(err, std::abs(Psi[4 * (3 + i) + 3] - expected));
        norm = std::max(norm, std::abs(expected));
    }
    ASSERT_GT(norm, 0.0);
    check("Gamma SRP coupling vs leading term rvm", err / norm, TOL_SRP_COUPLING);
}

// ---- Ballistic reduction ----------------------------------------------------------------------------

TEST(ZohValBallistic, StateOnlyBitEqual) {
    REQUIRE_KERNELS();
    const double mbar = 1850.0;
    const Vec zero(4, 0.0);
    for (double m0 : {1.0, 0.87}) {
        for (bool srp : {false, true}) {
            const SpiceParameters p = moon_params(srp, false, mbar * m0);
            for (const Dyn& d : DYNS) {
                const std::string where = std::string(d.name) + " srp " + std::to_string(srp) + " m " + std::to_string(m0);
                const Vec x0 = state_of(d, p, STATE0, m0);
                const Vec x6(x0.begin(), x0.begin() + 6);
                const IvpResult a = d.solve(p, thruster(d, 0.05, 3000.0, mbar), {0.0, TF}, x0, zero, IvpOptions{});
                const IvpResult b = d.mee ? campanella::solve_ivp_mee(p, {0.0, TF}, x6)
                                          : campanella::solve_ivp_rv(p, {0.0, TF}, x6);
                ASSERT_EQ(a.t, b.t) << where;
                for (size_t k = 0; k < a.size(); k++) {
                    for (int i = 0; i < 6; i++) {
                        ASSERT_EQ(a.x[7 * k + i], b.x[6 * k + i]) << where << " k " << k << " i " << i;
                    }
                    ASSERT_EQ(a.x[7 * k + 6], m0) << where << " k " << k;
                }
            }
        }
    }
}

TEST(ZohValBallistic, StmMatches) {
    REQUIRE_KERNELS();
    const double mbar = 1850.0, m0 = 1.0;
    const Vec zero(4, 0.0);
    for (bool srp : {false, true}) {
        const SpiceParameters p = moon_params(srp, false, mbar * m0);
        for (const Dyn& d : DYNS) {
            const std::string where = std::string(d.name) + " srp " + std::to_string(srp);
            const Vec x0 = state_of(d, p, STATE0, m0);
            const Vec x6(x0.begin(), x0.begin() + 6);
            const IvpResult a = d.solvestm(p, thruster(d, 0.05, 3000.0, mbar), {0.0, TF}, x0, zero, IvpOptions{});
            const IvpResult b = d.mee ? campanella::solve_ivpstm_mee(p, {0.0, TF}, x6)
                                      : campanella::solve_ivpstm_rv(p, {0.0, TF}, x6);
            const Vec xa = final_state_n(a, 7), xb = final_state_n(b, 6);
            const Vec s = scales(d, x0);
            const Vec s6(s.begin(), s.begin() + 6);
            check("Ballistic state " + where, state_error(xa, xb, s6), TOL_BALLISTIC_STATE);
            const Vec Pa = final_stm_n(a, 7), Pb = final_stm_n(b, 6);
            Vec Pa6(36);
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    Pa6[6 * i + j] = Pa[7 * i + j];
                }
                // The mass enters only through SRP when u = 0
                if (!srp) {
                    EXPECT_EQ(Pa[7 * i + 6], 0.0) << where << " i " << i;
                }
            }
            check_matrix("Ballistic Phi " + where, Pa6, Pb, 6, 6, s6, s6, TOL_BALLISTIC_STM);
        }
    }
}

// ---- Sensitivities vs finite differences -------------------------------------------------------------

TEST(ZohValSens, PhiVsFd) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    for (const Dyn& d : DYNS) {
        check_phi_fd("5d", d, p, thruster(d), 5 * DAY, state_of(d, p, STATE0, 0.9), U);
        check_phi_fd("short", d, p, thruster(d, SHORT_THRUST_N), SHORT_TF, state_of(d, p, STATE0, 0.9), U);
    }
}

TEST(ZohValSens, PsiVsFd) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    for (const Dyn& d : DYNS) {
        check_psi_fd("5d", d, p, thruster(d), 5 * DAY, state_of(d, p, STATE0, 0.9), U);
        check_psi_fd("short", d, p, thruster(d, SHORT_THRUST_N), SHORT_TF, state_of(d, p, STATE0, 0.9), U);
    }
}

// A Psi that drops the coupling of Gamma through the mass must fail the column metric on the short arc,
// where the row metric cannot see it: that is why both are required above
TEST(ZohValSens, ColumnMetricCatchesDroppedGammaCoupling) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d, SHORT_THRUST_N);
        const Vec x0 = state_of(d, p, STATE0, 0.9);
        const Vec fd = fd_psi(d, p, th, SHORT_TF, x0, U);
        Vec Psi = final_stm_u(d.solvestm(p, th, {0.0, SHORT_TF}, x0, U, IvpOptions{}), 7, 4);
        for (int i = 0; i < 6; i++) {
            Psi[4 * i + 3] = 0.0;
        }
        const Vec s = scales(d, x0);
        const double row = matrix_error(Psi, fd, 7, 4, s, {}, FLOOR, ErrorMode::row);
        const double col = matrix_error(trajectory_rows(Psi), trajectory_rows(fd), 6, 4,
                                        Vec(s.begin(), s.begin() + 6), {}, FLOOR, ErrorMode::col);
        check(std::string("dropped coupling, row metric blind ") + d.name, row, 1e-4);
        EXPECT_GT(col, 0.5) << d.name;
    }
}

// With tau = 0 the mass acts only through SRP: this isolates a term much smaller than the thrust's
TEST(ZohValSens, SrpMassCouplingVsFd) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    const Vec u = {0.0, 0.0, 0.0, 0.7};
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d, 0.5);
        const Vec x0 = state_of(d, p, STATE0, 0.9);
        check_phi_fd("SRP only", d, p, th, 5 * DAY, x0, u);
        check_psi_fd("SRP only", d, p, th, 5 * DAY, x0, u);
    }
}

TEST(ZohValSens, CanonicalUnitsVsFd) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true, true);
    for (const Dyn& d : DYNS) {
        const Vec x0 = state_of(d, p, STATE0, 0.9);
        check_phi_fd("canon 5d", d, p, thruster(d), 5 * DAY / p.TU(), x0, U);
        check_psi_fd("canon 5d", d, p, thruster(d), 5 * DAY / p.TU(), x0, U);
        check_psi_fd("canon short", d, p, thruster(d, SHORT_THRUST_N), SHORT_TF / p.TU(), x0, U);
    }
}

namespace {

// Psi(tf) against Phi(tf) int Phi^-1 B dt, by composite Simpson over a uniform t_eval
void check_quadrature(const std::string& label, const Dyn& d, const SpiceParameters& p, const ThrustOptions& th,
                      double tf, int intervals, const Vec& rv0) {
    IvpOptions o;
    for (int k = 0; k <= intervals; k++) {
        o.t_eval.push_back(k == intervals ? tf : tf * k / intervals);
    }
    const Vec x0 = state_of(d, p, rv0, 0.9);
    const IvpResult r = d.solvestm(p, th, {0.0, tf}, x0, U, o);
    ASSERT_EQ(r.size(), static_cast<size_t>(intervals + 1));
    Vec integral(28, 0.0);
    for (int k = 0; k <= intervals; k++) {
        const Vec xk(r.x.begin() + 7 * k, r.x.begin() + 7 * (k + 1));
        const Vec Phi(r.stm.begin() + 49 * k, r.stm.begin() + 49 * (k + 1));
        const EomEvalControl e = d.eval(p, th, r.t[k], xk, U);
        const Vec integrand = solve_linear(Phi, Vec(e.B.begin(), e.B.end()), 7, 4);
        const double w = (k == 0 || k == intervals) ? 1.0 : (k % 2 == 1 ? 4.0 : 2.0);
        for (int j = 0; j < 28; j++) {
            integral[j] += w * integrand[j];
        }
    }
    for (double& v : integral) {
        v *= tf / intervals / 3.0;
    }
    const Vec quad = product(final_stm_n(r, 7), integral, 7, 4);
    check_matrix("Quadrature " + label + " " + d.name, final_stm_u(r, 7, 4), quad, 7, 4, scales(d, x0), {}, 1e-6);
}

}  // namespace

TEST(ZohValSens, Quadrature) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    for (const Dyn& d : DYNS) {
        check_quadrature("far 1h", d, p, thruster(d, SHORT_THRUST_N), 3600.0, 200, STATE0);
        check_quadrature("LLO 2h", d, p, thruster(d, SHORT_THRUST_N), 7200.0, 400, state_llo());
    }
}

TEST(ZohValSens, Semigroup) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    const double t1 = 1.5 * DAY, t2 = 3.0 * DAY;
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d);
        const Vec x0 = state_of(d, p, STATE0, 0.9);
        const IvpResult a = d.solvestm(p, th, {0.0, t1}, x0, U, IvpOptions{});
        const IvpResult b = d.solvestm(p, th, {t1, t2}, final_state_n(a, 7), U, IvpOptions{});
        const IvpResult c = d.solvestm(p, th, {0.0, t2}, x0, U, IvpOptions{});
        const Vec Pa = final_stm_n(a, 7), Pb = final_stm_n(b, 7);
        Vec Psi = product(Pb, final_stm_u(a, 7, 4), 7, 4);
        const Vec Psi_b = final_stm_u(b, 7, 4);
        for (int j = 0; j < 28; j++) {
            Psi[j] += Psi_b[j];
        }
        const Vec s = scales(d, x0);
        check_matrix(std::string("Semigroup Psi ") + d.name, Psi, final_stm_u(c, 7, 4), 7, 4, s, {},
                     TOL_SEMIGROUP);
        check_matrix(std::string("Semigroup Phi ") + d.name, product(Pb, Pa, 7, 7), final_stm_n(c, 7), 7, 7, s, s,
                     TOL_SEMIGROUP);
    }
}

// ---- Backward ------------------------------------------------------------------------------------------

TEST(ZohValBackward, Roundtrip) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    const double tf = 5 * DAY;
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d);
        const Vec x0 = state_of(d, p, STATE0, 0.9);
        const IvpResult f = d.solvestm(p, th, {0.0, tf}, x0, U, IvpOptions{});
        const IvpResult b = d.solvestm(p, th, {tf, 0.0}, final_state_n(f, 7), U, IvpOptions{});
        const Vec s = scales(d, x0);
        check(std::string("Roundtrip state ") + d.name, state_error(final_state_n(b, 7), x0, s), TOL_ROUNDTRIP_STATE);
        const Vec Phi_b = final_stm_n(b, 7);
        check_matrix(std::string("Roundtrip Phi_b Phi_f = I ") + d.name, product(Phi_b, final_stm_n(f, 7), 7, 7),
                     identity7(), 7, 7, s, s, TOL_ROUNDTRIP_PHI);
        Vec expected = product(Phi_b, final_stm_u(f, 7, 4), 7, 4);
        for (double& v : expected) {
            v = -v;
        }
        check_matrix(std::string("Roundtrip Psi_b = -Phi_b Psi_f ") + d.name, final_stm_u(b, 7, 4), expected, 7, 4, s,
                     {}, TOL_ROUNDTRIP_PSI);
    }
}

// ---- Cartesian against MEE ---------------------------------------------------------------------------

namespace {

// blkdiag(J, 1), J = d(rv)/d(mee)
Vec jbar(const Vec& mee7, double mu) {
    const Vec J = campanella::mee2cart_jacobian(Vec(mee7.begin(), mee7.begin() + 6), mu);
    Vec Jb(49, 0.0);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            Jb[7 * i + j] = J[6 * i + j];
        }
    }
    Jb[48] = 1.0;
    return Jb;
}

}  // namespace

TEST(ZohValCrossVal, CartVsMee) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    const ThrustOptions th = thruster(MEEM_INERTIAL);
    const std::vector<Vec> controls = {U, {-0.6, 0.2, 0.1, 0.65}, {0.1, 0.1, -0.9, 0.92}, {0.0, 0.0, 0.0, 0.0}, {0.5, 0.5, 0.5, 0.87}};
    const double seg = 0.5 * DAY;

    Vec xc = state_of(RVM, p, STATE0, 0.9);
    Vec xm = state_of(MEEM_INERTIAL, p, STATE0, 0.9);
    const Vec J0 = jbar(xm, MU_MOON);
    Vec chain_c = identity7(), chain_m = identity7();
    Vec chain_u_c, chain_u_m;  // sensitivity of the current state to the first control

    for (size_t k = 0; k < controls.size(); k++) {
        const double t0 = k * seg, t1 = (k + 1) * seg;
        const Vec Jk = jbar(xm, MU_MOON);
        const IvpResult rc = RVM.solvestm(p, th, {t0, t1}, xc, controls[k], IvpOptions{});
        const IvpResult rm = MEEM_INERTIAL.solvestm(p, th, {t0, t1}, xm, controls[k], IvpOptions{});
        xc = final_state_n(rc, 7);
        xm = final_state_n(rm, 7);
        const Vec Jf = jbar(xm, MU_MOON);
        const Vec s = scales(MEEM_INERTIAL, xm);
        const std::string where = "seg " + std::to_string(k);

        const Vec Phi_c = final_stm_n(rc, 7), Psi_c = final_stm_u(rc, 7, 4);
        const Vec Phi_m = final_stm_n(rm, 7), Psi_m = final_stm_u(rm, 7, 4);
        check_matrix("CrossVal Phi " + where, Phi_m, solve_linear(Jf, product(Phi_c, Jk, 7, 7), 7, 7), 7, 7, s, s, 1e-6);
        check_matrix("CrossVal Psi " + where, Psi_m, solve_linear(Jf, Psi_c, 7, 4), 7, 4, s, {}, 1e-6);

        // Sensitivities of the whole chain, with the Jacobians at its two ends only
        chain_c = product(Phi_c, chain_c, 7, 7);
        chain_m = product(Phi_m, chain_m, 7, 7);
        chain_u_c = k == 0 ? Psi_c : product(Phi_c, chain_u_c, 7, 4);
        chain_u_m = k == 0 ? Psi_m : product(Phi_m, chain_u_m, 7, 4);

        const Vec cart = campanella::mee2cart(Vec(xm.begin(), xm.begin() + 6), MU_MOON);
        check("CrossVal dr km " + where, norm3_diff(cart, xc, 0), 1e-5);
        check("CrossVal dv km/s " + where, norm3_diff(cart, xc, 3), 1e-10);
        check("CrossVal dm " + where, std::abs(xm[6] - xc[6]), TOL_CROSSVAL_MASS);
    }
    const Vec Jf = jbar(xm, MU_MOON);
    const Vec s = scales(MEEM_INERTIAL, xm);
    check_matrix("CrossVal chained Phi", chain_m, solve_linear(Jf, product(chain_c, J0, 7, 7), 7, 7), 7, 7, s, s, 1e-6);
    check_matrix("CrossVal chained Psi(u0)", chain_u_m, solve_linear(Jf, chain_u_c, 7, 4), 7, 4, s, {}, 1e-6);
}

// ---- The rocket equation --------------------------------------------------------------------------------

namespace {

// Far from a negligible central body, the only force is the thrust. Everything below is integrated by
// hand from dv/dt = c1 tau / m, dm/dt = -c2 Gamma, with k = c2 Gamma.
const double ROCKET_T_N = 1000.0, ROCKET_ISP = 300.0, ROCKET_MBAR = 1000.0, ROCKET_TF = 2500.0;
const Vec ROCKET_X0 = {1e9, -2e9, 5e8, 1.0, -0.5, 0.2, 1.0};
const double ROCKET_GAMMA = 0.8;
const Vec ROCKET_U = {0.6 * ROCKET_GAMMA, 0.0, 0.8 * ROCKET_GAMMA, ROCKET_GAMMA};

SpiceParameters free_space() { return SpiceParameters(ET0, {"MOON"}, with_gms({1e-20})); }

ThrustOptions rocket_thruster() { return thruster(RVM, ROCKET_T_N, ROCKET_ISP, ROCKET_MBAR); }

double block_error(const std::function<double(int)>& got, const std::function<double(int)>& want, int n) {
    double diff = 0.0, ref = 0.0;
    for (int i = 0; i < n; i++) {
        diff = std::max(diff, std::abs(got(i) - want(i)));
        ref = std::max(ref, std::abs(want(i)));
    }
    return diff / ref;
}

}  // namespace

TEST(ZohValRocket, TsiolkovskyFreeSpace) {
    const SpiceParameters p = free_space();
    const IvpResult r = campanella::solve_ivp_rvm(p, rocket_thruster(), {0.0, ROCKET_TF}, ROCKET_X0, ROCKET_U, IvpOptions{});
    ASSERT_GT(r.size(), 2u);
    // Isp g0 in km/s: T [N] over mdot [kg/s] is m/s
    const double ve = ROCKET_ISP * 9.80665 * 1e-3;
    const double mdot_kg_s = ROCKET_T_N * ROCKET_GAMMA / (ROCKET_ISP * 9.80665);
    const double m0 = ROCKET_X0[6];
    const double k = mdot_kg_s / ROCKET_MBAR;
    const Vec dir = {0.6, 0.0, 0.8};
    double dv_err = 0.0, dr_err = 0.0, dm_err = 0.0;
    double dv_final = 0.0;
    for (size_t l = 0; l < r.size(); l++) {
        const double t = r.t[l];
        const double m = m0 - k * t;
        const double L = std::log(m0 / m);
        dm_err = std::max(dm_err, std::abs(r.x[7 * l + 6] - m));
        for (int i = 0; i < 3; i++) {
            const double dv = ve * dir[i] * L;
            const double dr = ve * dir[i] * (t + (m / k) * std::log(m / m0));
            dv_err = std::max(dv_err, std::abs(r.x[7 * l + 3 + i] - (ROCKET_X0[3 + i] + dv)));
            dr_err = std::max(dr_err, std::abs(r.x[7 * l + i] - (ROCKET_X0[i] + ROCKET_X0[3 + i] * t + dr)));
        }
        dv_final = ve * L;
    }
    ASSERT_GT(dv_final, 3.0);  // a large burn, so that a factor anywhere is visible
    check("Tsiolkovsky dv km/s", dv_err, TOL_TSIOLKOVSKY_V);
    check("Tsiolkovsky dr km", dr_err, TOL_TSIOLKOVSKY_R);
    check("Tsiolkovsky dm", dm_err, TOL_TSIOLKOVSKY_M);
}

TEST(ZohValRocket, ClosedFormSensitivities) {
    const SpiceParameters p = free_space();
    const IvpResult r = campanella::solve_ivpstm_rvm(p, rocket_thruster(), {0.0, ROCKET_TF}, ROCKET_X0, ROCKET_U, IvpOptions{});
    const Vec Phi = final_stm_n(r, 7), Psi = final_stm_u(r, 7, 4);

    const double t = ROCKET_TF;
    const double c1 = (ROCKET_T_N * 1e-3) / ROCKET_MBAR;
    const double c2 = ROCKET_T_N / (ROCKET_ISP * 9.80665) / ROCKET_MBAR;
    const double m0 = ROCKET_X0[6];
    const double k = c2 * ROCKET_GAMMA;
    const double m = m0 - k * t;
    const double L = std::log(m0 / m);
    const double* tau = ROCKET_U.data();

    // v - v0 = (c1 tau / k) L
    const double v_tau = c1 / k * L;
    // int_0^t ln(m0/m(s)) ds = t + (m/k) ln(m/m0)
    const double r_tau = c1 / k * (t - (m / k) * L);
    // d/dGamma of int c1 tau / (m0 - c2 Gamma s) ds = c1 c2 tau int s / m(s)^2 ds
    const double v_gamma = c1 * c2 / (k * k) * (m0 / m - 1.0 - L);
    const double r_gamma = c1 * c2 / (k * k) * ((m0 + m) / k * L - 2.0 * t);
    // d/dm0 of int c1 tau / (m0 - k s) ds = -c1 tau int ds / m(s)^2
    const double v_m = -c1 / k * (1.0 / m - 1.0 / m0);
    const double r_m = -c1 / k * (L / k - t / m0);

    check("Rocket Psi_v,tau diag", block_error([&](int i) { return Psi[4 * (3 + i) + i]; }, [&](int) { return v_tau; }, 3), TOL_ROCKET);
    check("Rocket Psi_r,tau diag", block_error([&](int i) { return Psi[4 * i + i]; }, [&](int) { return r_tau; }, 3), TOL_ROCKET);
    check("Rocket Psi_v,Gamma", block_error([&](int i) { return Psi[4 * (3 + i) + 3]; }, [&](int i) { return v_gamma * tau[i]; }, 3), TOL_ROCKET);
    check("Rocket Psi_r,Gamma", block_error([&](int i) { return Psi[4 * i + 3]; }, [&](int i) { return r_gamma * tau[i]; }, 3), TOL_ROCKET);
    check("Rocket Phi_v,m", block_error([&](int i) { return Phi[7 * (3 + i) + 6]; }, [&](int i) { return v_m * tau[i]; }, 3), TOL_ROCKET);
    check("Rocket Phi_r,m", block_error([&](int i) { return Phi[7 * i + 6]; }, [&](int i) { return r_m * tau[i]; }, 3), TOL_ROCKET);
    check("Rocket Phi_r,v", block_error([&](int i) { return Phi[7 * i + 3 + i]; }, [&](int) { return t; }, 3), TOL_ROCKET);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i != j) {
                EXPECT_NEAR(Psi[4 * i + j], 0.0, TOL_ROCKET * std::abs(r_tau)) << i << " " << j;
                EXPECT_NEAR(Psi[4 * (3 + i) + j], 0.0, TOL_ROCKET * std::abs(v_tau)) << i << " " << j;
            }
        }
    }
}

// ---- Units ------------------------------------------------------------------------------------------------

TEST(ZohValUnits, CanonicalMatchesDimensional) {
    REQUIRE_KERNELS();
    const SpiceParameters pd = moon_params(true), pc = moon_params(true, true);
    const double tf = 5 * DAY;
    ASSERT_NE(pc.DU(), 1.0);
    for (const Dyn& d : DYNS) {
        const ThrustOptions th = thruster(d);
        const Vec xd = state_of(d, pd, STATE0, 0.9), xc = state_of(d, pc, STATE0, 0.9);
        const IvpResult rd = d.solvestm(pd, th, {0.0, tf}, xd, U, IvpOptions{});
        const IvpResult rc = d.solvestm(pc, th, {0.0, tf / pc.TU()}, xc, U, IvpOptions{});
        // x_canonical = x_dimensional / D, with u dimensionless in both
        const double DU = pc.DU(), VU = pc.VU();
        const Vec D = d.mee ? Vec{DU, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0} : Vec{DU, DU, DU, VU, VU, VU, 1.0};
        const Vec fd = final_state_n(rd, 7), fc = final_state_n(rc, 7);
        Vec mapped(7);
        for (int i = 0; i < 7; i++) {
            mapped[i] = fd[i] / D[i];
        }
        const Vec s = scales(d, xc);
        check(std::string("Units state ") + d.name, state_error(fc, mapped, s), TOL_UNITS_STATE);

        const Vec Pd = final_stm_n(rd, 7), Qd = final_stm_u(rd, 7, 4);
        Vec Phi(49), Psi(28);
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                Phi[7 * i + j] = Pd[7 * i + j] * D[j] / D[i];
            }
            for (int j = 0; j < 4; j++) {
                Psi[4 * i + j] = Qd[4 * i + j] / D[i];
            }
        }
        check_matrix(std::string("Units Phi ") + d.name, final_stm_n(rc, 7), Phi, 7, 7, s, s, TOL_UNITS_SENS);
        check_matrix(std::string("Units Psi ") + d.name, final_stm_u(rc, 7, 4), Psi, 7, 4, s, {}, TOL_UNITS_SENS);
    }
}

TEST(ZohValUnits, MassUnitInvariance) {
    REQUIRE_KERNELS();
    const SpiceParameters p = moon_params(true);
    const double tf = 5 * DAY, scale = 1000.0;
    for (const Dyn& d : DYNS) {
        const Vec xa = state_of(d, p, STATE0, 1.0), xb = state_of(d, p, STATE0, scale);
        const IvpResult ra = d.solvestm(p, thruster(d, 0.05, 3000.0, scale), {0.0, tf}, xa, U, IvpOptions{});
        const IvpResult rb = d.solvestm(p, thruster(d, 0.05, 3000.0, 1.0), {0.0, tf}, xb, U, IvpOptions{});
        const Vec fa = final_state_n(ra, 7), fb = final_state_n(rb, 7);
        // m_b = scale m_a
        Vec Dm(7, 1.0);
        Dm[6] = scale;
        Vec mapped(7);
        for (int i = 0; i < 7; i++) {
            mapped[i] = fa[i] * Dm[i];
        }
        const Vec s = scales(d, xb);
        check(std::string("MassUnit state ") + d.name, state_error(fb, mapped, s), TOL_MASS_UNIT_STATE);

        const Vec Pa = final_stm_n(ra, 7), Qa = final_stm_u(ra, 7, 4);
        Vec Phi(49), Psi(28);
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                Phi[7 * i + j] = Pa[7 * i + j] * Dm[i] / Dm[j];
            }
            for (int j = 0; j < 4; j++) {
                Psi[4 * i + j] = Qa[4 * i + j] * Dm[i];
            }
        }
        // The mass column must scale: a Phi_i,m of equal value would be off by a factor of 1000
        EXPECT_NE(Pa[7 * 3 + 6], 0.0) << d.name;
        check_matrix(std::string("MassUnit Phi ") + d.name, final_stm_n(rb, 7), Phi, 7, 7, s, s, TOL_MASS_UNIT_SENS);
        check_matrix(std::string("MassUnit Psi ") + d.name, final_stm_u(rb, 7, 4), Psi, 7, 4, s, {}, TOL_MASS_UNIT_SENS);
    }
}
