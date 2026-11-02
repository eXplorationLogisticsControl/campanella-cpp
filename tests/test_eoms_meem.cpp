// MEE + mass equations of motion: A and B against central differences for every force combination
// and both control frames, reduction to the state-only dynamics, the role of the thrust in F and A_B
// (docs/gve_mee.md sec. 5), the mass column, the sensitivity right-hand side and the domain faults.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "helpers.hpp"
#include "model.h"

using namespace campanella_test;
using campanella::EomEvalControl;
using campanella::SpiceOptions;
using campanella::SpiceParameters;
using campanella::ThrustFrame;
using campanella::ThrustOptions;

namespace {

constexpr double MBAR = 925.0;   // MBAR * M_STATE is the srp_m of add_srp, exactly
constexpr double M_STATE = 2.0;
constexpr double T_N = 10.0;
constexpr double ISP_S = 3000.0;
const std::vector<double> U = {0.3, -0.5, 0.7, 0.9};
const std::vector<double> U_ZERO = {0.0, 0.0, 0.0, 0.0};
const ThrustFrame FRAMES[] = {ThrustFrame::rtn, ThrustFrame::inertial};

const char* frame_name(ThrustFrame frame) {
    return frame == ThrustFrame::rtn ? "RTN" : "inertial";
}

ThrustOptions thrust_options(ThrustFrame frame) {
    ThrustOptions th;
    th.thrust_N = T_N;
    th.isp_s = ISP_S;
    th.mass_unit_kg = MBAR;
    th.mee_control_frame = frame;
    return th;
}

constexpr double C1 = (T_N * 1e-3) / MBAR;   // DU = TU = 1, kN/kg = km/s^2
const double C2 = T_N / (ISP_S * 9.80665) / MBAR;

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

SpiceParameters earth_params(const Forces& f) {
    SpiceOptions o = with_gms({GM_LIST[1], GM_LIST[0], GM_LIST[2]});
    if (f.srp) {
        add_srp(o);
    }
    if (f.sh) {
        o.sh_file = data_path("EGM2008_n10.gfc");
        o.sh_nmax = 10;
        o.sh_frame = "ITRF93";
    }
    return SpiceParameters(ET0, {"EARTH", "MOON", "SUN"}, o);
}

// A central body only: needs no kernel
SpiceParameters two_body_params() {
    return SpiceParameters(0.0, {"MOON"}, with_gms({MU_MOON}));
}

std::vector<double> with_mass(std::vector<double> x, double m = M_STATE) {
    x.push_back(m);
    return x;
}

// The state of state_llo has L = 0 and k = 0, where the normal thrust vanishes in Ldot and kdot: advanced
// to L = 1, so that no row of B_G is zero.
std::vector<double> x_llo() {
    std::vector<double> x = campanella::cart2mee(state_llo(), MU_MOON);
    x[5] = 1.0;
    return with_mass(x);
}
std::vector<double> x_moon() { return with_mass(campanella::cart2mee(STATE0, MU_MOON)); }
// GTO of docs/gve_mee.md sec. 7.1
std::vector<double> x_gto() {
    return with_mass({11373.99192144, 0.694841890805239, 0.225767816090336, -0.238651523310869,
                      -0.086862050847443, 0.488692190558412});
}
// generic elements, with h, k, f, g and sin L, cos L all away from zero; for a Moon-centred model
std::vector<double> x_generic() { return with_mass({5000.0, 0.05, -0.02, 0.3, -0.2, 1.0}); }

struct Case {
    std::string name;
    SpiceParameters params;
    std::vector<double> x;
    double t;
};

// LLO and the 20-day state around the Moon, GTO around the Earth; all are non-equatorial
std::vector<Case> cases_for(const Forces& f) {
    std::vector<Case> cases;
    cases.push_back({std::string(f.name) + ", LLO", moon_params(f), x_llo(), 3600.0});
    cases.push_back({std::string(f.name) + ", Moon", moon_params(f), x_moon(), 5 * 86400.0});
    cases.push_back({std::string(f.name) + ", GTO", earth_params(f), x_gto(), 3600.0});
    return cases;
}

std::vector<double> xdot_of(const EomEvalControl& e) {
    return std::vector<double>(e.xdot.begin(), e.xdot.end());
}
std::vector<double> A_of(const EomEvalControl& e) { return std::vector<double>(e.A.begin(), e.A.end()); }
std::vector<double> B_of(const EomEvalControl& e) { return std::vector<double>(e.B.begin(), e.B.end()); }

// Steps of docs/gve_mee.md sec. 7.4, with dm = 1e-5 m
std::vector<double> fd_steps(const std::vector<double>& x) {
    return {1e-6 * x[0], 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-5 * x[6]};
}

// Scaling s of sec. 7.4: M -> diag(1/s) M diag(s), which makes the p row and column dimensionless
std::vector<double> scale(const std::vector<double>& x) { return {x[0], 1, 1, 1, 1, 1, 1}; }

constexpr double ABS_FLOOR = 1e-30;   // the mass row is exactly zero

// The columns of C = [rhat that nhat] as functions of (h, k, L), with the component form of
// docs/gve_mee.md eq. (3.1), not the vector form of the library. C[i][j] is row i of column j.
std::array<std::array<double, 3>, 3> rtn_matrix(const std::vector<double>& x) {
    const double h = x[3], k = x[4];
    const double cL = std::cos(x[5]), sL = std::sin(x[5]);
    const double s2 = 1 + h * h + k * k;
    const double al2 = h * h - k * k;
    const double Z = h * sL - k * cL;
    const double ZL = h * cL + k * sL;
    std::array<std::array<double, 3>, 3> C;
    C[0] = {(cL + al2 * cL + 2 * h * k * sL) / s2, (-sL - al2 * sL + 2 * h * k * cL) / s2, 2 * k / s2};
    C[1] = {(sL - al2 * sL + 2 * h * k * cL) / s2, (cL - al2 * cL - 2 * h * k * sL) / s2, -2 * h / s2};
    C[2] = {2 * Z / s2, 2 * ZL / s2, (1 - h * h - k * k) / s2};
    return C;
}

// C^T v
std::vector<double> to_rtn(const std::vector<double>& x, const std::vector<double>& v) {
    const auto C = rtn_matrix(x);
    std::vector<double> out(3);
    for (int j = 0; j < 3; j++) {
        out[j] = C[0][j] * v[0] + C[1][j] * v[1] + C[2][j] * v[2];
    }
    return out;
}

// F = (dC^T/dx) a_I of docs/gve_mee.md sec. 5.4, from a_rtn = C^T a_I, as a 3x6 row-major matrix
std::vector<double> F_matrix(const std::vector<double>& x, const std::vector<double>& a_rtn) {
    const double h = x[3], k = x[4];
    const double cL = std::cos(x[5]), sL = std::sin(x[5]);
    const double s2 = 1 + h * h + k * k;
    const double aR = a_rtn[0], aT = a_rtn[1], aN = a_rtn[2];
    std::vector<double> F(18, 0.0);
    F[0 * 6 + 3] = (2 / s2) * (k * aT + sL * aN);
    F[0 * 6 + 4] = (2 / s2) * (-h * aT - cL * aN);
    F[0 * 6 + 5] = aT;
    F[1 * 6 + 3] = (2 / s2) * (-k * aR + cL * aN);
    F[1 * 6 + 4] = (2 / s2) * (h * aR + sL * aN);
    F[1 * 6 + 5] = -aR;
    F[2 * 6 + 3] = -(2 / s2) * (sL * aR + cL * aT);
    F[2 * 6 + 4] = (2 / s2) * (cL * aR - sL * aT);
    return F;
}

std::vector<double> xdot_meem(const SpiceParameters& p, ThrustFrame frame, double t, const std::vector<double>& x,
                              const std::vector<double>& u) {
    return xdot_of(campanella::eval_meem(p, thrust_options(frame), t, x, u));
}

// d(B u)/dx by central differences, 7x7 row-major. Since xdot is affine in u at fixed x, this is
// the part of A due to the control: the thrust in A_B, in F (through C^T in B) and in the mass column.
std::vector<double> d_Bu_dx_fd(const SpiceParameters& p, ThrustFrame frame, double t, const std::vector<double>& x,
                               const std::vector<double>& u) {
    return jacobian_fd_n(
        [&](const std::vector<double>& y) {
            const EomEvalControl e = campanella::eval_meem(p, thrust_options(frame), t, y, u);
            std::vector<double> Bu(7, 0.0);
            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 4; j++) {
                    Bu[i] += e.B[4 * i + j] * u[j];
                }
            }
            return Bu;
        },
        x, fd_steps(x));
}

std::vector<double> minus(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> c(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] - b[i];
    }
    return c;
}

// To report the measured values: MEEM_REPORT=1
void report(const char* what, double value) {
    if (std::getenv("MEEM_REPORT") != nullptr) {
        std::printf("MEEM_REPORT %-44s %.3e\n", what, value);
    }
}

struct Worst {
    const char* name;
    double value = 0.0;
    explicit Worst(const char* n) : name(n) {}
    double operator()(double v) {
        value = std::max(value, v);
        return v;
    }
    ~Worst() { report(name, value); }
};

}  // namespace

TEST(Meem, AandBVsFd) {
    REQUIRE_KERNELS();
    Worst worst_A_row("A row"), worst_A_col("A col"), worst_B_row("B row"), worst_B_col("B col");
    for (const Forces& f : FORCE_CASES) {
        for (const Case& c : cases_for(f)) {
            for (ThrustFrame frame : FRAMES) {
                const std::string where = c.name + ", " + frame_name(frame);
                const EomEvalControl e = campanella::eval_meem(c.params, thrust_options(frame), c.t, c.x, U);
                const std::vector<double> A_fd = jacobian_fd_n(
                    [&](const std::vector<double>& y) { return xdot_meem(c.params, frame, c.t, y, U); }, c.x,
                    fd_steps(c.x));
                // xdot is affine in u: the steps only need to be large against the rounding of xdot
                const std::vector<double> B_fd = jacobian_fd_n(
                    [&](const std::vector<double>& v) { return xdot_meem(c.params, frame, c.t, c.x, v); }, U,
                    {1e-2, 1e-2, 1e-2, 1e-2});
                const std::vector<double> s = scale(c.x);
                EXPECT_LE(worst_A_row(matrix_error(A_of(e), A_fd, 7, 7, s, s, ABS_FLOOR, ErrorMode::row)), 1e-6) << where;
                // limited by the rounding of Ldot in the columns h, k, where only perturbations and thrust
                // enter: ulp(Ldot) / (2 dh) against an entry of 1e-7 at GTO. Measured 4.2e-6.
                EXPECT_LE(worst_A_col(matrix_error(A_of(e), A_fd, 7, 7, s, s, ABS_FLOOR, ErrorMode::col)), 2e-5) << where;
                EXPECT_LE(worst_B_row(matrix_error(B_of(e), B_fd, 7, 4, s, {}, ABS_FLOOR, ErrorMode::row)), 1e-6) << where;
                EXPECT_LE(worst_B_col(matrix_error(B_of(e), B_fd, 7, 4, s, {}, ABS_FLOOR, ErrorMode::col)), 1e-6) << where;
                // the mass row is exact
                for (int j = 0; j < 7; j++) {
                    EXPECT_EQ(e.A[7 * 6 + j], 0.0);
                }
                for (int j = 0; j < 3; j++) {
                    EXPECT_EQ(e.B[4 * 6 + j], 0.0);
                }
                EXPECT_EQ(e.B[4 * 6 + 3], -C2);
                for (int i = 0; i < 6; i++) {
                    EXPECT_EQ(e.B[4 * i + 3], 0.0);
                }
                EXPECT_EQ(e.xdot[6], -C2 * U[3]);
            }
        }
    }
}

TEST(Meem, UZeroEqualsMeeExactly) {
    REQUIRE_KERNELS();
    for (const Forces& f : FORCE_CASES) {
        for (const Case& c : cases_for(f)) {
            const std::vector<double> x6(c.x.begin(), c.x.begin() + 6);
            const campanella::EomEval ref = campanella::eval_mee(c.params, c.t, x6);
            for (ThrustFrame frame : FRAMES) {
                const EomEvalControl e = campanella::eval_meem(c.params, thrust_options(frame), c.t, c.x, U_ZERO);
                for (int i = 0; i < 6; i++) {
                    EXPECT_EQ(e.xdot[i], ref.xdot[i]) << c.name << ", i = " << i;
                    for (int j = 0; j < 6; j++) {
                        EXPECT_EQ(e.A[7 * i + j], ref.A[6 * i + j]) << c.name << ", " << i << ", " << j;
                    }
                }
                // Gamma = 0
                EXPECT_EQ(e.xdot[6], 0.0);
            }
        }
    }
}

TEST(Meem, InertialMatchesRtnXdot) {
    REQUIRE_KERNELS();
    const std::vector<double> tau_I = {U[0], U[1], U[2]};
    Worst worst("InertialMatchesRtnXdot, term-relative");
    for (const Forces& f : FORCE_CASES) {
        for (const Case& c : cases_for(f)) {
            const std::vector<double> tau_rtn = to_rtn(c.x, tau_I);
            const std::vector<double> u_rtn = {tau_rtn[0], tau_rtn[1], tau_rtn[2], U[3]};
            const EomEvalControl in = campanella::eval_meem(c.params, thrust_options(ThrustFrame::inertial), c.t, c.x, U);
            const EomEvalControl rt = campanella::eval_meem(c.params, thrust_options(ThrustFrame::rtn), c.t, c.x, u_rtn);
            for (int i = 0; i < 7; i++) {
                // the largest term of this row: the thrust terms, and xdot, which has the Keplerian
                // term of L and the natural perturbations
                double term = std::abs(rt.xdot[i]);
                for (int j = 0; j < 3; j++) {
                    term = std::max(term, std::abs(rt.B[4 * i + j] * tau_rtn[j]));
                }
                const double err = std::abs(in.xdot[i] - rt.xdot[i]);
                if (term > 0.0) {
                    EXPECT_LE(worst(err / term), 1e-13) << c.name << ", i = " << i;
                } else {
                    EXPECT_EQ(err, 0.0) << c.name << ", i = " << i;
                }
            }
        }
    }
}

TEST(Meem, RtnMatrixOfTestIsTheRtnFrame) {
    // the C of this file against the definition: r/|r|, H/|H|, nhat x rhat
    for (const std::vector<double>& x : {x_llo(), x_moon(), x_generic()}) {
        const std::vector<double> rv = campanella::mee2cart({x.begin(), x.begin() + 6}, MU_MOON);
        const double r = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
        const double H[3] = {rv[1] * rv[5] - rv[2] * rv[4], rv[2] * rv[3] - rv[0] * rv[5], rv[0] * rv[4] - rv[1] * rv[3]};
        const double Hn = std::sqrt(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
        const double rhat[3] = {rv[0] / r, rv[1] / r, rv[2] / r};
        const double nhat[3] = {H[0] / Hn, H[1] / Hn, H[2] / Hn};
        const double that[3] = {nhat[1] * rhat[2] - nhat[2] * rhat[1], nhat[2] * rhat[0] - nhat[0] * rhat[2],
                                nhat[0] * rhat[1] - nhat[1] * rhat[0]};
        const auto C = rtn_matrix(x);
        for (int i = 0; i < 3; i++) {
            EXPECT_NEAR(C[i][0], rhat[i], 1e-14);
            EXPECT_NEAR(C[i][1], that[i], 1e-14);
            EXPECT_NEAR(C[i][2], nhat[i], 1e-14);
        }
    }
}

// The same thrust vector at the current state: xdot and A_B agree, so that the A_e of the two frames
// differ by the rotation of the RTN basis under the inertial thrust, B_G F(thrust).
TEST(Meem, InertialAndRtnAeDiffer) {
    REQUIRE_KERNELS();
    const std::vector<double> tau_I = {U[0], U[1], U[2]};
    Worst worst("InertialAndRtnAeDiffer");
    for (const Forces& f : FORCE_CASES) {
        for (const Case& c : cases_for(f)) {
            const std::vector<double> tau_rtn = to_rtn(c.x, tau_I);
            const std::vector<double> u_rtn = {tau_rtn[0], tau_rtn[1], tau_rtn[2], U[3]};
            const EomEvalControl in = campanella::eval_meem(c.params, thrust_options(ThrustFrame::inertial), c.t, c.x, U);
            const EomEvalControl rt = campanella::eval_meem(c.params, thrust_options(ThrustFrame::rtn), c.t, c.x, u_rtn);
            const double m = c.x[6];
            const std::vector<double> a_c = {C1 / m * tau_rtn[0], C1 / m * tau_rtn[1], C1 / m * tau_rtn[2]};
            const std::vector<double> F = F_matrix(c.x, a_c);
            // B_G F, from the B of the RTN control, B[e, tau] = (c1/m) B_G. The mass column of the
            // two frames must agree, as must the mass row.
            std::vector<double> expected(49, 0.0), diff(49);
            for (int i = 0; i < 6; i++) {
                for (int q = 0; q < 6; q++) {
                    for (int j = 0; j < 3; j++) {
                        expected[7 * i + q] += rt.B[4 * i + j] * (m / C1) * F[6 * j + q];
                    }
                }
            }
            double largest = 0.0;
            for (int i = 0; i < 49; i++) {
                diff[i] = in.A[i] - rt.A[i];
                largest = std::max(largest, std::abs(expected[i]));
            }
            ASSERT_GT(largest, 0.0) << c.name;
            const std::vector<double> s = scale(c.x);
            // relative to each row, then F has no column p, f, g, and the mass columns agree
            EXPECT_LE(worst(matrix_error(diff, expected, 7, 7, s, s, 1e-20, ErrorMode::row)), 1e-12) << c.name;
            for (int i = 0; i < 6; i++) {
                for (int q : {0, 1, 2, 6}) {
                    EXPECT_NEAR(diff[7 * i + q] * s[q] / s[i], 0.0, 1e-13 * largest) << c.name << ", " << i << ", " << q;
                }
            }
        }
    }
}

// With a_T = 0, pdot = B[0][1] a_T vanishes for every state under RTN thrust, because a_T is a fixed
// number. A thrust in F would make it a_T(x), leaving a nonzero row 0 of A.
TEST(MeemF, RtnThrustExcludedFromF) {
    const SpiceParameters p = two_body_params();
    for (const std::vector<double>& x : {x_llo(), x_moon(), x_generic()}) {
        for (const std::vector<double>& u : {std::vector<double>{1.0, 0.0, 0.0, 1.0},
                                             std::vector<double>{0.0, 0.0, 1.0, 1.0},
                                             std::vector<double>{0.6, 0.0, -0.8, 1.0}}) {
            const EomEvalControl e = campanella::eval_meem(p, thrust_options(ThrustFrame::rtn), 0.0, x, u);
            for (int j = 0; j < 7; j++) {
                EXPECT_EQ(e.A[7 * 0 + j], 0.0) << "column " << j;
            }
        }
    }
}

// With natural forces too: A(tau) - A(0) is A_B of the thrust alone, which is d(B u)/dx as B has no
// C^T in this frame.
TEST(MeemF, RtnThrustExcludedFromFWithForces) {
    REQUIRE_KERNELS();
    Worst worst("RtnThrustExcludedFromFWithForces");
    const Forces f = {"third bodies + SRP + SH", true, true};
    for (const Case& c : cases_for(f)) {
        const ThrustOptions th = thrust_options(ThrustFrame::rtn);
        const std::vector<double> diff = minus(A_of(campanella::eval_meem(c.params, th, c.t, c.x, U)),
                                               A_of(campanella::eval_meem(c.params, th, c.t, c.x, U_ZERO)));
        const std::vector<double> expected = d_Bu_dx_fd(c.params, ThrustFrame::rtn, c.t, c.x, U);
        const std::vector<double> s = scale(c.x);
        EXPECT_LE(worst(matrix_error(diff, expected, 7, 7, s, s, ABS_FLOOR, ErrorMode::row)), 1e-6) << c.name;
    }
}

// Closed forms of row 0 of A in the two-body problem, for an inertial thrust along each axis of the
// RTN frame at this state: pdot = B[0][1] (that . a), with the basis differentiated by sec. 5.4.
TEST(MeemF, InertialThrustIncludedInF) {
    const SpiceParameters p = two_body_params();
    const ThrustOptions th_in = thrust_options(ThrustFrame::inertial);
    const ThrustOptions th_rt = thrust_options(ThrustFrame::rtn);
    Worst worst("InertialThrustIncludedInF");
    for (const std::vector<double>& x : {x_llo(), x_moon(), x_generic()}) {
        const auto C = rtn_matrix(x);
        const double m = x[6], h = x[3], k = x[4];
        const double cL = std::cos(x[5]), sL = std::sin(x[5]);
        const double s2 = 1 + h * h + k * k;
        const double a = C1 / m;
        const double B01 = campanella::eval_meem(p, th_rt, 0.0, x, U_ZERO).B[4 * 0 + 1] / a;   // 2 p sig / w
        ASSERT_GT(B01, 0.0);
        const auto close = [&](double value, double ref) {
            const double scale_row = B01 * a;   // the size of the entries of this row
            EXPECT_LE(worst(std::abs(value - ref) / scale_row), 1e-13);
        };
        {   // along rhat: a_R = a
            const std::vector<double> u = {C[0][0], C[1][0], C[2][0], 1.0};
            const EomEvalControl e = campanella::eval_meem(p, th_in, 0.0, x, u);
            close(e.A[7 * 0 + 3], B01 * (2 / s2) * (-k * a));
            close(e.A[7 * 0 + 4], B01 * (2 / s2) * (h * a));
            close(e.A[7 * 0 + 5], -B01 * a);
            EXPECT_GT(std::abs(e.A[7 * 0 + 5]), 0.1 * B01 * a);
        }
        {   // along nhat: a_N = a
            const std::vector<double> u = {C[0][2], C[1][2], C[2][2], 1.0};
            const EomEvalControl e = campanella::eval_meem(p, th_in, 0.0, x, u);
            close(e.A[7 * 0 + 3], B01 * (2 / s2) * (cL * a));
            close(e.A[7 * 0 + 4], B01 * (2 / s2) * (sL * a));
            close(e.A[7 * 0 + 5], 0.0);
        }
    }
}

// A_B takes the total RTN acceleration, thrust included. In the two-body problem with a transverse
// thrust, row 0 of A_e is d(B[0][1])/dx a_T plus B[0][1] F[1][:], and F[1][:] is zero in columns p, f,
// g, and in L as a_R = 0, for both frames.
TEST(MeemAB, UsesTotalAcceleration) {
    const SpiceParameters p = two_body_params();
    Worst worst_closed("UsesTotalAcceleration, closed form");
    for (const std::vector<double>& x : {x_llo(), x_moon(), x_generic()}) {
        const auto C = rtn_matrix(x);
        const double m = x[6], pp = x[0], f = x[1], g = x[2], L = x[5];
        const double cL = std::cos(L), sL = std::sin(L);
        const double w = 1 + f * cL + g * sL;
        const double wL = -f * sL + g * cL;
        const double sig = std::sqrt(pp / MU_MOON);
        const double a = C1 / m;
        for (ThrustFrame frame : FRAMES) {
            const std::vector<double> u = (frame == ThrustFrame::rtn)
                                              ? std::vector<double>{0.0, 1.0, 0.0, 1.0}
                                              : std::vector<double>{C[0][1], C[1][1], C[2][1], 1.0};
            const EomEvalControl e = campanella::eval_meem(p, thrust_options(frame), 0.0, x, u);
            const double B01 = 2 * pp * sig / w;
            const double row[4][2] = {{e.A[7 * 0 + 0] * pp, 3 * B01 / (2 * pp) * a * pp},
                                      {e.A[7 * 0 + 1], -2 * pp * sig * cL / (w * w) * a},
                                      {e.A[7 * 0 + 2], -2 * pp * sig * sL / (w * w) * a},
                                      {e.A[7 * 0 + 5], -2 * pp * sig * wL / (w * w) * a}};
            for (const auto& entry : row) {
                EXPECT_LE(worst_closed(std::abs(entry[0] - entry[1]) / (B01 * a)), 1e-13) << frame_name(frame);
            }
            EXPECT_GT(std::abs(e.A[7 * 0 + 0]), 0.0);
        }
    }
}

// With natural forces too: A(tau) - A(0) is the part from the thrust, which a total without the thrust
// fails, and A(tau) against differences has the natural part, which a total of thrust only fails.
TEST(MeemAB, UsesTotalAccelerationWithForces) {
    REQUIRE_KERNELS();
    Worst worst("UsesTotalAccelerationWithForces");
    Worst worst_total("UsesTotalAccelerationWithForces, A vs FD");
    for (const Forces& f : FORCE_CASES) {
        for (const Case& c : cases_for(f)) {
            for (ThrustFrame frame : FRAMES) {
                const ThrustOptions th = thrust_options(frame);
                const std::vector<double> diff = minus(A_of(campanella::eval_meem(c.params, th, c.t, c.x, U)),
                                                       A_of(campanella::eval_meem(c.params, th, c.t, c.x, U_ZERO)));
                const std::vector<double> expected = d_Bu_dx_fd(c.params, frame, c.t, c.x, U);
                const std::vector<double> s = scale(c.x);
                EXPECT_LE(worst(matrix_error(diff, expected, 7, 7, s, s, ABS_FLOOR, ErrorMode::row)), 1e-6)
                    << c.name << ", " << frame_name(frame);
                // the natural part of the total, against differences: not against eval_mee, which
                // shares the code of A_B
                const std::vector<double> A_fd = jacobian_fd_n(
                    [&](const std::vector<double>& y) { return xdot_meem(c.params, frame, c.t, y, U); }, c.x,
                    fd_steps(c.x));
                EXPECT_LE(worst_total(matrix_error(A_of(campanella::eval_meem(c.params, th, c.t, c.x, U)), A_fd, 7, 7, s, s,
                                                   ABS_FLOOR, ErrorMode::row)),
                          1e-6)
                    << c.name << ", " << frame_name(frame);
            }
        }
    }
}

// SRP alone in the mass column. xdot[5] holds the Keplerian term, whose rounding is the noise of the
// central difference; the physical area to mass ratio would leave SRP within that noise for m, so the
// area is inflated, as for docs/gve_mee.md sec. 6, to compare the whole column.
TEST(Meem, MassColumnSrpOnlyVsFd) {
    REQUIRE_KERNELS();
    Worst worst_fd("MassColumnSrp, FD"), worst_on_off("MassColumnSrp, -(on - off)/m");
    SpiceOptions on = with_gms(GM_LIST);
    add_srp(on);
    const SpiceParameters physical(ET0, BODIES, on);
    on.srp_A *= 1e4;
    const SpiceParameters inflated(ET0, BODIES, on);
    const SpiceParameters off(ET0, BODIES, with_gms(GM_LIST));
    const std::vector<double> no_thrust = {0.0, 0.0, 0.0, 0.0};
    for (const std::vector<double>& x : {x_llo(), x_moon()}) {
        const std::vector<double> s = scale(x);
        const std::vector<double> s6(s.begin(), s.begin() + 6);
        for (ThrustFrame frame : FRAMES) {
            const EomEvalControl e = campanella::eval_meem(inflated, thrust_options(frame), 0.0, x, no_thrust);
            const std::vector<double> A_fd = jacobian_fd_n(
                [&](const std::vector<double>& y) { return xdot_meem(inflated, frame, 0.0, y, no_thrust); }, x,
                fd_steps(x));
            std::vector<double> col(6), col_fd(6);
            for (int i = 0; i < 6; i++) {
                col[i] = e.A[7 * i + 6];
                col_fd[i] = A_fd[7 * i + 6];
            }
            EXPECT_LE(worst_fd(matrix_error(col, col_fd, 6, 1, s6, {}, ABS_FLOOR, ErrorMode::col)), 1e-9);
            EXPECT_GT(std::abs(col[0]), 0.0);

            // a_srp ~ 1/m: the column is -(xdot with SRP - xdot without SRP) / m, at the physical size.
            const EomEvalControl e_on = campanella::eval_meem(physical, thrust_options(frame), 0.0, x, no_thrust);
            const EomEvalControl e_off = campanella::eval_meem(off, thrust_options(frame), 0.0, x, no_thrust);
            std::vector<double> col_on(6), expected(6);
            for (int i = 0; i < 6; i++) {
                col_on[i] = e_on.A[7 * i + 6];
                expected[i] = -(e_on.xdot[i] - e_off.xdot[i]) / x[6];
            }
            EXPECT_LE(worst_on_off(matrix_error(col_on, expected, 6, 1, s6, {}, ABS_FLOOR, ErrorMode::col)), 1e-9);
        }
    }
}

// Thrust alone in the mass column, by differences and by the closed form -(1/m) B[e, tau] tau
TEST(Meem, MassColumnThrustOnlyVsFd) {
    Worst worst_fd("MassColumnThrust, FD"), worst_closed("MassColumnThrust, closed form");
    const SpiceParameters p = two_body_params();
    for (const std::vector<double>& x : {x_llo(), x_moon(), x_generic()}) {
        const std::vector<double> s = scale(x);
        const std::vector<double> s6(s.begin(), s.begin() + 6);
        for (ThrustFrame frame : FRAMES) {
            const EomEvalControl e = campanella::eval_meem(p, thrust_options(frame), 0.0, x, U);
            const std::vector<double> A_fd = jacobian_fd_n(
                [&](const std::vector<double>& y) { return xdot_meem(p, frame, 0.0, y, U); }, x, fd_steps(x));
            std::vector<double> col(6), col_fd(6), closed(6);
            for (int i = 0; i < 6; i++) {
                col[i] = e.A[7 * i + 6];
                col_fd[i] = A_fd[7 * i + 6];
                for (int j = 0; j < 3; j++) {
                    closed[i] -= e.B[4 * i + j] * U[j] / x[6];
                }
            }
            EXPECT_LE(worst_fd(matrix_error(col, col_fd, 6, 1, s6, {}, ABS_FLOOR, ErrorMode::col)), 1e-8) << frame_name(frame);
            EXPECT_LE(worst_closed(matrix_error(col, closed, 6, 1, s6, {}, ABS_FLOOR, ErrorMode::col)), 1e-14) << frame_name(frame);
            EXPECT_GT(std::abs(col[0]), 0.0);
        }
    }
    // no thrust and no SRP: the mass column is zero
    for (ThrustFrame frame : FRAMES) {
        const EomEvalControl e = campanella::eval_meem(p, thrust_options(frame), 0.0, x_llo(), U_ZERO);
        for (int i = 0; i < 7; i++) {
            EXPECT_EQ(e.A[7 * i + 6], 0.0);
        }
    }
}

// SRP takes the state mass m_bar m, and srp_m only has to be valid
TEST(Meem, SrpUsesStateMassNotSrpM) {
    REQUIRE_KERNELS();
    SpiceOptions o = with_gms(GM_LIST);
    add_srp(o);
    o.srp_A *= 1e4;
    const SpiceParameters a(ET0, BODIES, o);
    o.srp_m = 111.0;
    const SpiceParameters b(ET0, BODIES, o);
    const SpiceParameters off(ET0, BODIES, with_gms(GM_LIST));
    const std::vector<double> x = x_llo();
    const std::vector<double> x4 = with_mass({x.begin(), x.begin() + 6}, 2 * x[6]);
    for (ThrustFrame frame : FRAMES) {
        const ThrustOptions th = thrust_options(frame);
        const EomEvalControl ea = campanella::eval_meem(a, th, 0.0, x, U);
        const EomEvalControl eb = campanella::eval_meem(b, th, 0.0, x, U);
        for (int i = 0; i < 7; i++) {
            EXPECT_EQ(ea.xdot[i], eb.xdot[i]);
            for (int j = 0; j < 7; j++) {
                EXPECT_EQ(ea.A[7 * i + j], eb.A[7 * i + j]);
            }
        }
        // twice the mass: half the SRP
        const std::vector<double> d2 = minus(xdot_meem(a, frame, 0.0, x, U_ZERO), xdot_meem(off, frame, 0.0, x, U_ZERO));
        const std::vector<double> d4 = minus(xdot_meem(a, frame, 0.0, x4, U_ZERO), xdot_meem(off, frame, 0.0, x4, U_ZERO));
        for (int i = 0; i < 5; i++) {
            ASSERT_NE(d2[i], 0.0);
            EXPECT_NEAR(d4[i] / d2[i], 0.5, 1e-9) << "i = " << i;
        }
        // the same product m_bar m: the same SRP
        ThrustOptions th2 = th;
        th2.mass_unit_kg = 2 * MBAR;
        const std::vector<double> x1 = with_mass({x.begin(), x.begin() + 6}, 1.0);
        const EomEvalControl e1 = campanella::eval_meem(a, th2, 0.0, x1, U_ZERO);
        const EomEvalControl e2 = campanella::eval_meem(a, th, 0.0, x, U_ZERO);
        for (int i = 0; i < 6; i++) {
            EXPECT_EQ(e1.xdot[i], e2.xdot[i]);
        }
    }
}

// B of each frame against its definition, with the C of this file
TEST(Meem, BIsBGTimesFrame) {
    const SpiceParameters p = two_body_params();
    for (const std::vector<double>& x : {x_llo(), x_moon(), x_generic()}) {
        const auto C = rtn_matrix(x);
        const EomEvalControl rt = campanella::eval_meem(p, thrust_options(ThrustFrame::rtn), 0.0, x, U);
        const EomEvalControl in = campanella::eval_meem(p, thrust_options(ThrustFrame::inertial), 0.0, x, U);
        for (int i = 0; i < 6; i++) {
            double row_max = 0.0;
            for (int j = 0; j < 3; j++) {
                row_max = std::max(row_max, std::abs(rt.B[4 * i + j]));
            }
            for (int j = 0; j < 3; j++) {
                // B_in[i][j] = sum_l B_G[i][l] (C^T)[l][j] = sum_l B_G[i][l] C[j][l]
                const double expected = rt.B[4 * i + 0] * C[j][0] + rt.B[4 * i + 1] * C[j][1] + rt.B[4 * i + 2] * C[j][2];
                EXPECT_NEAR(in.B[4 * i + j], expected, 1e-14 * row_max) << i << ", " << j;
            }
        }
    }
}

TEST(Meem, RhsAndRhsstmMatchEvaluate) {
    using namespace campanella::detail;
    const SpiceParameters p = two_body_params();
    const Model& model = Access::model(p);
    Workspace& ws = Access::workspace(p);
    for (ThrustFrame frame : FRAMES) {
        ZohControl control = make_control(p, thrust_options(frame), U, "test");
        Context ctx{&model, &ws, &control};
        const std::vector<double> x = x_generic();
        const EomEvalControl e = campanella::eval_meem(p, thrust_options(frame), 0.0, x, U);

        double xdot[7];
        ASSERT_EQ(rhs_meem(0.0, x.data(), xdot, &ctx), RHS_SUCCESS);
        for (int i = 0; i < 7; i++) {
            EXPECT_EQ(xdot[i], e.xdot[i]);
        }

        // any of A and B may be null
        double A[7][7], B[7][4], xd[7];
        ASSERT_EQ(meem_evaluate(model, control, ws, 0.0, x.data(), xd, A, nullptr), RHS_SUCCESS);
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                EXPECT_EQ(A[i][j], e.A[7 * i + j]);
            }
        }
        ASSERT_EQ(meem_evaluate(model, control, ws, 0.0, x.data(), xd, nullptr, B), RHS_SUCCESS);
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 4; j++) {
                EXPECT_EQ(B[i][j], e.B[4 * i + j]);
            }
        }

        // y = [x, Phi, Psi], row-major, with entries that identify their position
        std::vector<double> y(84), ydot(84, -1.0);
        std::copy(x.begin(), x.end(), y.begin());
        for (int q = 7; q < 84; q++) {
            y[q] = std::sin(1.0 + q);
        }
        ASSERT_EQ(rhsstm_meem(0.0, y.data(), ydot.data(), &ctx), RHS_SUCCESS);
        for (int i = 0; i < 7; i++) {
            EXPECT_EQ(ydot[i], e.xdot[i]);
            for (int j = 0; j < 7; j++) {
                double sum = 0.0;
                for (int l = 0; l < 7; l++) {
                    sum += e.A[7 * i + l] * y[7 + 7 * l + j];
                }
                EXPECT_EQ(ydot[7 + 7 * i + j], sum);
            }
            for (int j = 0; j < 4; j++) {
                double sum = e.B[4 * i + j];
                for (int l = 0; l < 7; l++) {
                    sum += e.A[7 * i + l] * y[56 + 4 * l + j];
                }
                EXPECT_EQ(ydot[56 + 4 * i + j], sum);
            }
        }
    }
}

TEST(Meem, NonPositiveMassFaults) {
    using namespace campanella::detail;
    const SpiceParameters p = two_body_params();
    const Model& model = Access::model(p);
    Workspace& ws = Access::workspace(p);
    ZohControl control = make_control(p, thrust_options(ThrustFrame::rtn), U, "test");
    Context ctx{&model, &ws, &control};
    const std::vector<double> nominal = x_generic();
    for (double m : {0.0, -0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::infinity()}) {
        std::vector<double> x = nominal;
        x[6] = m;
        double xdot[7];
        ws.fault = Fault::none;
        EXPECT_EQ(meem_evaluate(model, control, ws, 0.0, x.data(), xdot, nullptr, nullptr), RHS_EDOM) << m;
        EXPECT_EQ(ws.fault, Fault::mass_domain) << m;
        ws.fault = Fault::none;
        EXPECT_EQ(rhs_meem(0.0, x.data(), xdot, &ctx), RHS_EDOM) << m;
        EXPECT_EQ(ws.fault, Fault::mass_domain) << m;
        std::vector<double> y(84, 0.0), ydot(84);
        std::copy(x.begin(), x.end(), y.begin());
        ws.fault = Fault::none;
        EXPECT_EQ(rhsstm_meem(0.0, y.data(), ydot.data(), &ctx), RHS_EDOM) << m;
        EXPECT_EQ(ws.fault, Fault::mass_domain) << m;
        EXPECT_THROW(campanella::eval_meem(p, thrust_options(ThrustFrame::rtn), 0.0, x, U), std::domain_error) << m;
    }
    // the mass is checked first, and a later success clears the fault
    std::vector<double> x = nominal;
    x[0] = -1.0;
    x[6] = -1.0;
    double xdot[7];
    EXPECT_EQ(meem_evaluate(model, control, ws, 0.0, x.data(), xdot, nullptr, nullptr), RHS_EDOM);
    EXPECT_EQ(ws.fault, Fault::mass_domain);
    EXPECT_EQ(meem_evaluate(model, control, ws, 0.0, nominal.data(), xdot, nullptr, nullptr), RHS_SUCCESS);
    EXPECT_EQ(ws.fault, Fault::none);
}

TEST(Meem, TinyMassIsNonFinite) {
    using namespace campanella::detail;
    const SpiceParameters p = two_body_params();
    const Model& model = Access::model(p);
    Workspace& ws = Access::workspace(p);
    // no thrust: xdot is finite, while 1/m in B is not
    ZohControl control = make_control(p, thrust_options(ThrustFrame::rtn), U_ZERO, "test");
    std::vector<double> x = x_generic();
    x[6] = 1e-320;
    double xdot[7], B[7][4];
    EXPECT_EQ(meem_evaluate(model, control, ws, 0.0, x.data(), xdot, nullptr, B), RHS_EDOM);
    EXPECT_EQ(ws.fault, Fault::nonfinite);
}

TEST(Meem, ExistingDomainFaultsKept) {
    using namespace campanella::detail;
    const SpiceParameters p = two_body_params();
    const Model& model = Access::model(p);
    Workspace& ws = Access::workspace(p);
    ZohControl control = make_control(p, thrust_options(ThrustFrame::rtn), U, "test");
    Context ctx{&model, &ws, &control};
    const std::vector<std::vector<double>> bad = {
        {0.0, 0.05, -0.02, 0.3, -0.2, 1.0, 2.0},    // p = 0
        {-1.0, 0.05, -0.02, 0.3, -0.2, 1.0, 2.0},   // p < 0
        {5000.0, -1.0, 0.0, 0.3, -0.2, 0.0, 2.0},   // w = 0
        {5000.0, -2.0, 0.0, 0.3, -0.2, 0.0, 2.0},   // w < 0
    };
    for (const std::vector<double>& x : bad) {
        double xdot[7];
        double A[7][7], B[7][4];
        EXPECT_EQ(meem_evaluate(model, control, ws, 0.0, x.data(), xdot, A, B), RHS_EDOM);
        EXPECT_EQ(ws.fault, Fault::mee_domain);
        EXPECT_EQ(rhs_meem(0.0, x.data(), xdot, &ctx), RHS_EDOM);
        EXPECT_EQ(ws.fault, Fault::mee_domain);
        EXPECT_THROW(campanella::eval_meem(p, thrust_options(ThrustFrame::rtn), 0.0, x, U), std::domain_error);
    }
}

TEST(Meem, ExistingShDomainFaultKept) {
    REQUIRE_KERNELS();
    using namespace campanella::detail;
    const SpiceParameters p = moon_params({"third bodies + SH", false, true});
    ZohControl control = make_control(p, thrust_options(ThrustFrame::rtn), U, "test");
    double xdot[7];
    const double x[7] = {1000.0, 0.0, 0.0, 0.3, -0.2, 1.0, 2.0};   // below the reference radius
    EXPECT_EQ(meem_evaluate(Access::model(p), control, Access::workspace(p), 0.0, x, xdot, nullptr, nullptr), RHS_EDOM);
    EXPECT_EQ(Access::workspace(p).fault, Fault::sh_domain);
}
