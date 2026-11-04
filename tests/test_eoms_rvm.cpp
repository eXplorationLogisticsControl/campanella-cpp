// Cartesian + mass equations of motion: A and B against central differences, reduction to rv,
// SRP with the state mass, isolated mass columns, sensitivity of the error metric, and return codes.
// Set RVM_PRINT_ERRORS to print the measured errors.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>

#include "helpers.hpp"

using namespace campanella_test;
using campanella::ControlledParameters;
using campanella::EomEval;
using campanella::EomEvalControl;
using campanella::SpiceOptions;
using campanella::SpiceParameters;
using campanella::ThrustOptions;

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

// A case of a state, its time and the position step of the differences
struct Point {
    const char* name;
    double t;
    std::vector<double> x;
    double dr;
};

const double GM_SUN = 1.3271244004127942e11;
const double AU_KM = 149597870.7;
const std::vector<double> X_HELIO = {1.2e8, -0.5e8, 0.3e8, 10.0, 25.0, 1.0};

// The steps asked by the tests: dv in km/s, dm relative to m, du on a control affine in xdot
const double DV = 1e-6;
const double DM_REL = 1e-5;
const double DU_CTRL = 1e-2;

// Tolerances of the differences; see the rounding estimates next to each use
const double TOL_ROW = 1e-6;
// measured: at most 3.9e-8 (LLO with SRP); the rounding of xdot over the mass step sets it
const double TOL_COL = 4e-7;
// SRP-only mass column, where the rounding is larger. Measured: 1.4e-5, 1.2e-8, 1.4e-7
const double TOL_SRP_COL_LLO = 1e-4;
const double TOL_SRP_COL_STATE0 = 1e-7;
const double TOL_SRP_COL_HELIO = 1e-6;
// every structurally zero entry of the differences is exactly zero, so no absolute slack is needed
const double FLOOR = 0.0;

// A thrust at m = 0.87 well above the SRP acceleration
const std::vector<double> U_THRUST = {0.3, -0.5, 0.4, 0.71};
const std::vector<double> U_ZERO = {0.0, 0.0, 0.0, 0.0};
const double MASS = 0.87;

ThrustOptions thruster(double mass_unit_kg = 1850.0) {
    ThrustOptions th;
    th.thrust_N = 0.5;
    th.isp_s = 3000.0;
    th.mass_unit_kg = mass_unit_kg;
    return th;
}

SpiceParameters moon_params(const Forces& f, double srp_m = 1850.0) {
    SpiceOptions o = with_gms(GM_LIST);
    if (f.srp) {
        add_srp(o);
        o.srp_m = srp_m;
    }
    if (f.sh) {
        add_moon_sh(o);
    }
    return SpiceParameters(ET0, BODIES, o);
}

SpiceParameters helio_params(bool srp) {
    SpiceOptions o = with_gms({GM_SUN});
    if (srp) {
        add_srp(o);
    }
    return SpiceParameters(ET0, {"SUN"}, o);
}

std::vector<Point> moon_points() {
    return {{"LLO", 3600.0, state_llo(), 1e-3}, {"STATE0", 5 * 86400.0, STATE0, 1.0}};
}

std::vector<double> with_mass(std::vector<double> x, double m) {
    x.push_back(m);
    return x;
}

std::vector<double> xdot_rvm(const SpiceParameters& p, const ThrustOptions& th, double t,
                             const std::vector<double>& x, const std::vector<double>& u) {
    const EomEvalControl e = campanella::eval_rvm(p, th, t, x, u);
    return std::vector<double>(e.xdot.begin(), e.xdot.end());
}

std::vector<double> matrix_of(const EomEvalControl& e, bool B) {
    return B ? std::vector<double>(e.B.begin(), e.B.end()) : std::vector<double>(e.A.begin(), e.A.end());
}

std::vector<double> mass_column(const EomEvalControl& e) {
    std::vector<double> col(7);
    for (int i = 0; i < 7; i++) {
        col[i] = e.A[7 * i + 6];
    }
    return col;
}

// d(xdot)/dx by central differences
std::vector<double> A_fd(const SpiceParameters& p, const ThrustOptions& th, double t,
                         const std::vector<double>& x, const std::vector<double>& u, double dr) {
    return jacobian_fd_n([&](const std::vector<double>& y) { return xdot_rvm(p, th, t, y, u); }, x,
                         {dr, dr, dr, DV, DV, DV, DM_REL * x[6]});
}

// d(xdot)/du by central differences
std::vector<double> B_fd(const SpiceParameters& p, const ThrustOptions& th, double t,
                         const std::vector<double>& x, const std::vector<double>& u) {
    return jacobian_fd_n([&](const std::vector<double>& v) { return xdot_rvm(p, th, t, x, v); }, u,
                         {DU_CTRL, DU_CTRL, DU_CTRL, DU_CTRL});
}

// d(xdot)/dm alone by central differences, 7 x 1
std::vector<double> mass_column_fd(const SpiceParameters& p, const ThrustOptions& th, double t,
                                   const std::vector<double>& x, const std::vector<double>& u) {
    return jacobian_fd_n(
        [&](const std::vector<double>& m) {
            std::vector<double> y = x;
            y[6] = m[0];
            return xdot_rvm(p, th, t, y, u);
        },
        {x[6]}, {DM_REL * x[6]});
}

double error_of(const std::vector<double>& M, const std::vector<double>& R, int rows, int cols,
                ErrorMode mode) {
    return matrix_error(M, R, rows, cols, {}, {}, FLOOR, mode);
}

void report(const std::string& label, double value) {
    if (std::getenv("RVM_PRINT_ERRORS") != nullptr) {
        std::printf("    %-60s %.3e\n", label.c_str(), value);
    }
}

// (P0/c) Cr (A/m) (AU/r)^2, in km/s^2, for a mass in kg
double srp_magnitude(const SpiceOptions& o, double mass_kg, double r) {
    const double Cr = 1 + o.srp_rhoS + 5 * o.srp_rhoD / 3;
    return 1367.0 / 299792458.0 * Cr * o.srp_A / mass_kg / 1e3 * (AU_KM / r) * (AU_KM / r);
}

double norm3(const std::vector<double>& x) {
    return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

}  // namespace

TEST(Rvm, AandBVsFd) {
    REQUIRE_KERNELS();
    const ThrustOptions th = thruster();
    for (const Forces& f : FORCE_CASES) {
        const SpiceParameters p = moon_params(f);
        for (const Point& c : moon_points()) {
            const std::vector<double> x = with_mass(c.x, MASS);
            const EomEvalControl e = campanella::eval_rvm(p, th, c.t, x, U_THRUST);
            const std::vector<double> Afd = A_fd(p, th, c.t, x, U_THRUST, c.dr);
            const std::vector<double> Bfd = B_fd(p, th, c.t, x, U_THRUST);
            const std::vector<double> A = matrix_of(e, false), B = matrix_of(e, true);
            const std::string name = std::string(f.name) + ", " + c.name;
            const double a_row = error_of(A, Afd, 7, 7, ErrorMode::row);
            const double a_col = error_of(A, Afd, 7, 7, ErrorMode::col);
            const double b_row = error_of(B, Bfd, 7, 4, ErrorMode::row);
            const double b_col = error_of(B, Bfd, 7, 4, ErrorMode::col);
            report(name + ": A row", a_row);
            report(name + ": A col", a_col);
            report(name + ": B row", b_row);
            report(name + ": B col", b_col);
            EXPECT_LE(a_row, TOL_ROW) << name;
            EXPECT_LE(a_col, TOL_COL) << name;
            EXPECT_LE(b_row, TOL_ROW) << name;
            EXPECT_LE(b_col, TOL_COL) << name;
        }
    }
}

TEST(Rvm, UZeroEqualsRvExactly) {
    REQUIRE_KERNELS();
    // pairs (m_bar, m), with srp_m their product
    const double pairs[][2] = {{1850.0, 1.0}, {925.0, 2.0}, {2000.0, 0.925}, {3100.0, MASS}};
    for (const Forces& f : FORCE_CASES) {
        for (const auto& pair : pairs) {
            const double srp_m = pair[0] * pair[1];
            const SpiceParameters p = moon_params(f, srp_m);
            const ThrustOptions th = thruster(pair[0]);
            for (const Point& c : moon_points()) {
                const EomEval rv = campanella::eval_rv(p, c.t, c.x);
                const EomEvalControl e = campanella::eval_rvm(p, th, c.t, with_mass(c.x, pair[1]), U_ZERO);
                for (int i = 0; i < 6; i++) {
                    EXPECT_EQ(e.xdot[i], rv.xdot[i]) << f.name << ", " << c.name << ", " << i;
                    for (int j = 0; j < 6; j++) {
                        EXPECT_EQ(e.A[7 * i + j], rv.A[6 * i + j]) << f.name << ", " << c.name;
                    }
                }
                EXPECT_EQ(e.xdot[6], 0.0);
                for (int j = 0; j < 7; j++) {
                    EXPECT_EQ(e.A[7 * 6 + j], 0.0);
                }
                // the mass column is the SRP alone
                for (int i = 0; i < 3; i++) {
                    EXPECT_EQ(e.A[7 * i + 6], 0.0);
                    if (!f.srp) {
                        EXPECT_EQ(e.A[7 * (3 + i) + 6], 0.0);
                    }
                }
            }
        }
    }
}

TEST(Rvm, SrpMIsIgnored) {
    REQUIRE_KERNELS();
    const ThrustOptions th = thruster();
    for (const Forces& f : {FORCE_CASES[1], FORCE_CASES[3]}) {
        const SpiceParameters p1 = moon_params(f, 1850.0), p2 = moon_params(f, 12.5);
        for (const Point& c : moon_points()) {
            const std::vector<double> x = with_mass(c.x, MASS);
            const EomEvalControl e1 = campanella::eval_rvm(p1, th, c.t, x, U_THRUST);
            const EomEvalControl e2 = campanella::eval_rvm(p2, th, c.t, x, U_THRUST);
            EXPECT_EQ(std::memcmp(e1.xdot.data(), e2.xdot.data(), sizeof(double) * 7), 0) << f.name << c.name;
            EXPECT_EQ(std::memcmp(e1.A.data(), e2.A.data(), sizeof(double) * 49), 0) << f.name << c.name;
            EXPECT_EQ(std::memcmp(e1.B.data(), e2.B.data(), sizeof(double) * 28), 0) << f.name << c.name;
            // control: the state-only dynamics do see srp_m, and the state mass does matter
            EXPECT_NE(campanella::eval_rv(p1, c.t, c.x).xdot[3], campanella::eval_rv(p2, c.t, c.x).xdot[3]);
            EXPECT_NE(campanella::eval_rvm(p1, th, c.t, with_mass(c.x, 2 * MASS), U_ZERO).xdot[3],
                      campanella::eval_rvm(p1, th, c.t, x, U_ZERO).xdot[3]);
        }
    }
}

TEST(Rvm, SrpStateMassMagnitude) {
    // the Sun as central body: radial and outward, with the mass m_bar m rather than srp_m; no kernels
    const SpiceParameters p = helio_params(true), p_grav = helio_params(false);
    const SpiceOptions& o = p.options();
    ASSERT_EQ(o.srp_m, 1850.0);
    const ThrustOptions th = thruster(1000.0);
    const double r = norm3(X_HELIO);
    const double a1 = srp_magnitude(o, 1000.0, r);
    std::array<double, 3> at_1{}, at_2{};
    for (double m : {1.0, 2.0}) {
        const std::vector<double> x = with_mass(X_HELIO, m);
        const std::vector<double> e = xdot_rvm(p, th, 0.0, x, U_ZERO);
        const std::vector<double> g = xdot_rvm(p_grav, th, 0.0, x, U_ZERO);
        for (int i = 0; i < 3; i++) {
            const double a = e[3 + i] - g[3 + i];
            // difference of two ~6e-3 km/s^2 accelerations, as in Eoms.HeliocentricSrpMagnitude
            EXPECT_NEAR(a, a1 / m * X_HELIO[i] / r, 1e-9 * a1) << "m = " << m << ", " << i;
            (m == 1.0 ? at_1 : at_2)[i] = a;
        }
    }
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(at_2[i], at_1[i] / 2, 1e-9 * a1) << i;
    }
}

TEST(Rvm, SrpGradientScalesWithStateMass) {
    // Too small against gravity for the differences of Rvm.AandBVsFd: closed form, against G(m) - G_grav.
    // G_grav is ~5e-14 /s^2 and G_srp ~5e-19 /s^2: the cancellation error is ~1e-11 of G_srp.
    const SpiceParameters p = helio_params(true), p_grav = helio_params(false);
    const SpiceOptions& o = p.options();
    const ThrustOptions th = thruster();
    const double r = norm3(X_HELIO);
    for (double m : {1.0, 2.0, MASS}) {
        const std::vector<double> x = with_mass(X_HELIO, m);
        const EomEvalControl e = campanella::eval_rvm(p, th, 0.0, x, U_ZERO);
        const EomEvalControl g = campanella::eval_rvm(p_grav, th, 0.0, x, U_ZERO);
        const double kappa = srp_magnitude(o, 1850.0 * m, r) / r;
        std::vector<double> G_srp(9), G_ref(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                G_srp[3 * i + j] = e.A[7 * (3 + i) + j] - g.A[7 * (3 + i) + j];
                G_ref[3 * i + j] = kappa * ((i == j ? 1.0 : 0.0) - 3 * X_HELIO[i] * X_HELIO[j] / (r * r));
            }
        }
        const double err = matrix_error(G_srp, G_ref, 3, 3, {}, {}, 0.0, ErrorMode::col);
        report("G_srp vs closed form, m = " + std::to_string(m), err);
        EXPECT_LE(err, 1e-9) << "m = " << m;
    }
}

TEST(RvmMassColumn, SrpOnlyVsFd) {
    // tau = 0, so the column is -a_srp / m. The SRP change over dm is ~4e-15 km/s^2 and the rounding of
    // xdot ~2e-16 |a|, which is a noise floor of ~1e-4 at LLO (|a| = 1.5e-3), ~1e-7 at STATE0.
    REQUIRE_KERNELS();
    const ThrustOptions th = thruster();
    const std::vector<double> u = {0.0, 0.0, 0.0, 0.71};
    for (const Forces& f : {FORCE_CASES[1], FORCE_CASES[3]}) {
        const SpiceParameters p = moon_params(f);
        for (const Point& c : moon_points()) {
            const std::vector<double> x = with_mass(c.x, MASS);
            const std::vector<double> col = mass_column(campanella::eval_rvm(p, th, c.t, x, u));
            const std::vector<double> fd = mass_column_fd(p, th, c.t, x, u);
            const double err = error_of(col, fd, 7, 1, ErrorMode::col);
            const std::string name = std::string(f.name) + ", " + c.name;
            report("SRP-only mass column: " + name, err);
            EXPECT_LE(err, c.dr < 1.0 ? TOL_SRP_COL_LLO : TOL_SRP_COL_STATE0) << name;
            EXPECT_GT(std::abs(col[3]), 0.0) << name;
        }
    }
    const SpiceParameters p = helio_params(true);
    const std::vector<double> x = with_mass(X_HELIO, MASS);
    const std::vector<double> col = mass_column(campanella::eval_rvm(p, th, 0.0, x, u));
    const double err = error_of(col, mass_column_fd(p, th, 0.0, x, u), 7, 1, ErrorMode::col);
    report("SRP-only mass column: heliocentric", err);
    EXPECT_LE(err, TOL_SRP_COL_HELIO);
}

TEST(RvmMassColumn, ThrustOnlyVsFd) {
    // SRP off; the change over dm is ~3e-12 km/s^2, which is a noise floor of ~1e-7 at LLO
    REQUIRE_KERNELS();
    const ThrustOptions th = thruster();
    for (const Forces& f : {FORCE_CASES[0], FORCE_CASES[2]}) {
        const SpiceParameters p = moon_params(f);
        for (const Point& c : moon_points()) {
            const std::vector<double> x = with_mass(c.x, MASS);
            const std::vector<double> col = mass_column(campanella::eval_rvm(p, th, c.t, x, U_THRUST));
            const std::vector<double> fd = mass_column_fd(p, th, c.t, x, U_THRUST);
            const double err = error_of(col, fd, 7, 1, ErrorMode::col);
            const std::string name = std::string(f.name) + ", " + c.name;
            report("Thrust-only mass column: " + name, err);
            EXPECT_LE(err, TOL_COL) << name;
        }
    }
}

TEST(Rvm, MassColumnSignsAreCorrect) {
    // no kernels: Sun-centred, with and without SRP
    const SpiceParameters p = helio_params(true), p_off = helio_params(false);
    const ThrustOptions th = thruster();
    const double c1 = 0.5e-3 / 1850.0;
    const double c2 = 0.5 / (3000.0 * 9.80665) / 1850.0;
    const std::vector<double> x = with_mass(X_HELIO, MASS);

    // thrust alone: lighter is more acceleration, so the column opposes tau
    const EomEvalControl t = campanella::eval_rvm(p_off, th, 0.0, x, U_THRUST);
    for (int i = 0; i < 3; i++) {
        const double ref = -c1 * U_THRUST[i] / (MASS * MASS);
        EXPECT_NEAR(t.A[7 * (3 + i) + 6], ref, 1e-15 * std::abs(ref)) << i;
        EXPECT_LT(t.A[7 * (3 + i) + 6] * U_THRUST[i], 0.0) << i;
        EXPECT_GT(t.B[4 * (3 + i) + i], 0.0) << i;
        EXPECT_NEAR(t.B[4 * (3 + i) + i], c1 / MASS, 1e-15 * c1 / MASS) << i;
    }
    EXPECT_NEAR(t.B[4 * 6 + 3], -c2, 1e-15 * c2);
    EXPECT_NEAR(t.xdot[6], -c2 * U_THRUST[3], 1e-15 * c2);
    EXPECT_LT(t.xdot[6], 0.0);

    // SRP alone, which points away from the Sun: the column points toward it
    const EomEvalControl s = campanella::eval_rvm(p, th, 0.0, x, U_ZERO);
    const double r = norm3(X_HELIO);
    const double a = srp_magnitude(p.options(), 1850.0 * MASS, r);
    for (int i = 0; i < 3; i++) {
        const double ref = -a / MASS * X_HELIO[i] / r;
        EXPECT_NEAR(s.A[7 * (3 + i) + 6], ref, 1e-13 * std::abs(ref)) << i;
        EXPECT_LT(s.A[7 * (3 + i) + 6] * X_HELIO[i], 0.0) << i;
    }
}

TEST(RvmMetric, DetectsCorruption) {
    REQUIRE_KERNELS();
    const ThrustOptions th = thruster();
    const SpiceParameters p = moon_params(FORCE_CASES[3]);
    for (const Point& c : moon_points()) {
        const std::vector<double> x = with_mass(c.x, MASS);
        const EomEvalControl e = campanella::eval_rvm(p, th, c.t, x, U_THRUST);
        const std::vector<double> A = matrix_of(e, false), B = matrix_of(e, true);
        const std::vector<double> Afd = A_fd(p, th, c.t, x, U_THRUST, c.dr);
        const std::vector<double> Bfd = B_fd(p, th, c.t, x, U_THRUST);
        ASSERT_LE(error_of(A, Afd, 7, 7, ErrorMode::row), TOL_ROW) << c.name;
        ASSERT_LE(error_of(A, Afd, 7, 7, ErrorMode::col), TOL_COL) << c.name;

        // apply factor to the rows [first, last) of a column
        auto corrupt = [](std::vector<double> M, int cols, int col, int first, int last, double factor) {
            for (int i = first; i < last; i++) {
                M[cols * i + col] *= factor;
            }
            return M;
        };
        for (double factor : {0.0, -1.0}) {
            const std::string name = std::string(c.name) + ", factor " + std::to_string(factor);
            const std::vector<double> bad_mass = corrupt(A, 7, 6, 3, 6, factor);
            report(name + ": mass column, col mode", error_of(bad_mass, Afd, 7, 7, ErrorMode::col));
            report(name + ": mass column, row mode", error_of(bad_mass, Afd, 7, 7, ErrorMode::row));
            EXPECT_GT(error_of(bad_mass, Afd, 7, 7, ErrorMode::col), 100 * TOL_COL) << name;
            EXPECT_GT(error_of(bad_mass, Afd, 7, 7, ErrorMode::row), 100 * TOL_ROW) << name;

            const std::vector<double> bad_b = corrupt(B, 4, 1, 4, 5, factor);
            EXPECT_GT(error_of(bad_b, Bfd, 7, 4, ErrorMode::row), 100 * TOL_ROW) << name;
            EXPECT_GT(error_of(bad_b, Bfd, 7, 4, ErrorMode::col), 100 * TOL_COL) << name;

            const std::vector<double> bad_gamma = corrupt(B, 4, 3, 0, 7, factor);
            EXPECT_GT(error_of(bad_gamma, Bfd, 7, 4, ErrorMode::row), 100 * TOL_ROW) << name;
            EXPECT_GT(error_of(bad_gamma, Bfd, 7, 4, ErrorMode::col), 100 * TOL_COL) << name;
        }

        // one wrong entry of G, which must show even where the mass column dominates the row
        std::vector<double> bad_g = A;
        bad_g[7 * 4 + 1] *= 1.01;
        report(std::string(c.name) + ": G * 1.01, col mode", error_of(bad_g, Afd, 7, 7, ErrorMode::col));
        report(std::string(c.name) + ": G * 1.01, row mode", error_of(bad_g, Afd, 7, 7, ErrorMode::row));
        EXPECT_GT(error_of(bad_g, Afd, 7, 7, ErrorMode::col), 100 * TOL_COL) << c.name;

        // a structurally zero entry, in a row (of B) whose reference is entirely zero
        std::vector<double> bad_zero = B;
        bad_zero[4 * 0 + 0] = 1e-30;
        EXPECT_EQ(error_of(bad_zero, Bfd, 7, 4, ErrorMode::row), std::numeric_limits<double>::infinity());

        // non-finite values are never hidden
        std::vector<double> nan_a = A, nan_b = B;
        nan_a[7 * 3 + 6] = std::nan("");
        nan_b[4 * 6 + 3] = std::nan("");
        const double inf = std::numeric_limits<double>::infinity();
        for (ErrorMode mode : {ErrorMode::row, ErrorMode::col}) {
            EXPECT_EQ(error_of(nan_a, Afd, 7, 7, mode), inf);
            EXPECT_EQ(error_of(nan_b, Bfd, 7, 4, mode), inf);
        }
    }
}

TEST(Rvm, NonPositiveMassFaults) {
    // no kernels. The mass is checked first: at the origin the fault is still the mass.
    SpiceParameters p = helio_params(true);
    const ThrustOptions th = thruster();
    ControlledParameters cp(p, th, U_THRUST);
    double xdot[84], y[84] = {};
    const std::vector<double> origin(6, 0.0);
    const double inf = std::numeric_limits<double>::infinity();
    for (double m : {0.0, -0.0, -1.0, -inf, std::nan("")}) {
        for (const std::vector<double>& pos : {X_HELIO, origin}) {
            const std::vector<double> x = with_mass(pos, m);
            std::copy(x.begin(), x.end(), y);
            EXPECT_EQ(campanella::eom_rvm(0.0, x.data(), xdot, &cp), GSL_EDOM) << m;
            EXPECT_NE(campanella::eom_error(p).find("mass"), std::string::npos) << campanella::eom_error(p);
            EXPECT_EQ(campanella::eomstm_rvm(0.0, y, xdot, &cp), GSL_EDOM) << m;
            EXPECT_NE(campanella::eom_error(p).find("mass"), std::string::npos) << campanella::eom_error(p);
            EXPECT_THROW(campanella::eval_rvm(p, th, 0.0, x, U_THRUST), std::domain_error) << m;
        }
    }
    // the fault does not stick
    const std::vector<double> good = with_mass(X_HELIO, 1.0);
    EXPECT_EQ(campanella::eom_rvm(0.0, good.data(), xdot, &cp), GSL_SUCCESS);
    EXPECT_EQ(campanella::eom_error(p), "");
}

TEST(Rvm, NonFiniteOutputsFault) {
    // no kernels; each passes m > 0, and is recoverable
    SpiceParameters p = helio_params(true);
    const ThrustOptions th = thruster();
    ControlledParameters cp(p, th, U_ZERO);
    void* params = &cp;
    double xdot[7];
    const double inf = std::numeric_limits<double>::infinity();
    // no output would show an infinite mass; a mass so small that c1 / m overflows; the center of the Sun
    for (const std::vector<double>& x : {with_mass(X_HELIO, inf), with_mass(X_HELIO, 1e-320),
                                         with_mass(std::vector<double>(6, 0.0), 1.0)}) {
        EXPECT_EQ(campanella::eom_rvm(0.0, x.data(), xdot, params), GSL_EDOM) << x[6];
        EXPECT_NE(campanella::eom_error(p).find("non-finite"), std::string::npos) << campanella::eom_error(p);
        EXPECT_THROW(campanella::eval_rvm(p, th, 0.0, x, U_ZERO), std::domain_error) << x[6];
    }
}

TEST(Rvm, EomFunctionsAgreeWithEval) {
    // 84 states with an arbitrary Phi and Psi: Phi_dot = A Phi and Psi_dot = A Psi + B, as full products.
    // Rows of Phi and Psi for the mass are nonzero, so that the mass column of A matters.
    REQUIRE_KERNELS();
    SpiceParameters p = moon_params(FORCE_CASES[3]);
    const ThrustOptions th = thruster();
    ControlledParameters cp(p, th, U_THRUST);
    void* params = &cp;
    const std::vector<double> x = with_mass(state_llo(), MASS);
    const EomEvalControl e = campanella::eval_rvm(p, th, 100.0, x, U_THRUST);

    double xdot[7], y[84], ydot[84];
    ASSERT_EQ(campanella::eom_rvm(100.0, x.data(), xdot, params), GSL_SUCCESS);
    for (int i = 0; i < 7; i++) {
        y[i] = x[i];
    }
    for (int i = 7; i < 84; i++) {
        y[i] = std::sin(1.0 + i);
    }
    ASSERT_EQ(campanella::eomstm_rvm(100.0, y, ydot, params), GSL_SUCCESS);
    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(xdot[i], e.xdot[i]);
        EXPECT_EQ(ydot[i], e.xdot[i]);
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
}
