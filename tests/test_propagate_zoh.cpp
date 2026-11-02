// propagate_zoh_{rvm,meem}: the linearization x_{k+1} = Phi_k x_k + Psi_k u_k + c_k of a zero-order-hold
// reference, its defects, its sensitivities against finite differences of the whole chain, and its
// failures. Set ZOH_PRINT_ERRORS to print the measured errors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "helpers.hpp"

using namespace campanella_test;
using campanella::ControlledParameters;
using campanella::IvpOptions;
using campanella::IvpResult;
using campanella::SpiceOptions;
using campanella::SpiceParameters;
using campanella::ThrustFrame;
using campanella::ThrustOptions;
using campanella::ZohLinearization;

namespace {

constexpr int NX = 7;
constexpr int NU = 4;
const double PI = std::acos(-1.0);
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
const double INF = std::numeric_limits<double>::infinity();

using Solver = IvpResult (*)(const SpiceParameters&, const ThrustOptions&, const std::vector<double>&,
                             const std::vector<double>&, const std::vector<double>&, const IvpOptions&);
using Propagate = ZohLinearization (*)(const SpiceParameters&, const ThrustOptions&, const std::vector<double>&,
                                       const std::vector<double>&, const std::vector<double>&, const IvpOptions&);

struct Dyn {
    const char* name;
    bool mee;
    ThrustFrame frame;
    Solver solve;
    Solver solvestm;
    Propagate propagate;
};

const Dyn DYNS[] = {
    {"rvm", false, ThrustFrame::rtn, campanella::solve_ivp_rvm, campanella::solve_ivpstm_rvm,
     campanella::propagate_zoh_rvm},
    {"meem RTN", true, ThrustFrame::rtn, campanella::solve_ivp_meem, campanella::solve_ivpstm_meem,
     campanella::propagate_zoh_meem},
    {"meem inertial", true, ThrustFrame::inertial, campanella::solve_ivp_meem, campanella::solve_ivpstm_meem,
     campanella::propagate_zoh_meem},
};

// Gamma is close to |tau|, and every segment has another direction
const std::vector<double> U_REF = {0.3,  -0.5, 0.4,  0.71,
                                   -0.6, 0.2,  0.1,  0.64,
                                   0.1,  0.7,  -0.3, 0.77,
                                   0.5,  0.4,  0.6,  0.88};
const double MASS = 0.87;

// An arc: Cartesian state in km, km/s, segment boundaries in s, and a propulsion system strong enough
// for the Gamma column of Psi to be well above the integration noise in the (r, v) rows
struct Arc {
    const char* name;
    std::vector<double> rv;
    std::vector<double> seconds;
    double thrust_N;
    double isp_s;
};

std::vector<Arc> arcs() {
    return {{"moon", STATE0, {0.0, 0.4 * 86400.0, 1.1 * 86400.0, 1.5 * 86400.0, 2.0 * 86400.0}, 0.5, 300.0},
            {"llo", state_llo(), {0.0, 2000.0, 5500.0, 8000.0, 11000.0}, 5.0, 1000.0}};
}

ThrustOptions thruster(const Dyn& d, const Arc& arc) {
    ThrustOptions th;
    th.thrust_N = arc.thrust_N;
    th.isp_s = arc.isp_s;
    th.mass_unit_kg = 1000.0;
    th.mee_control_frame = d.frame;
    return th;
}

SpiceOptions canonical(SpiceOptions o) {
    o.DU = 1737.4;
    return o;
}

SpiceParameters two_body() {
    return SpiceParameters(ET0, {"MOON"}, with_gms({MU_MOON}));
}

SpiceParameters all_forces(bool canon) {
    SpiceOptions o = with_gms(GM_LIST);
    add_srp(o);
    add_moon_sh(o);
    return SpiceParameters(ET0, BODIES, canon ? canonical(o) : o);
}

std::vector<double> state_of(const Dyn& d, const SpiceParameters& p, std::vector<double> rv, double m) {
    for (int i = 0; i < 3; i++) {
        rv[i] /= p.DU();
        rv[3 + i] /= p.VU();
    }
    std::vector<double> x = d.mee ? campanella::cart2mee(rv, p.mus()[0]) : rv;
    x.push_back(m);
    return x;
}

std::vector<double> times_of(const SpiceParameters& p, const std::vector<double>& seconds) {
    std::vector<double> t;
    for (const double s : seconds) {
        t.push_back(s / p.TU());
    }
    return t;
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

// The (r, v) rows of a 7 x cols row-major matrix. The mass row has entries near 1 (Phi) or c2 dt (Psi),
// which would otherwise be the reference of the mass and Gamma columns and hide their (r, v) rows.
std::vector<double> top_rows(const std::vector<double>& M, int cols) {
    return std::vector<double>(M.begin(), M.begin() + 6 * cols);
}

// States at the N times from x0, each segment starting at the end of the previous one, with solver
std::vector<double> chain(Solver solver, const SpiceParameters& p, const ThrustOptions& th,
                          const std::vector<double>& times, const std::vector<double>& x0,
                          const std::vector<double>& u_ref) {
    std::vector<double> x = x0;
    for (size_t k = 0; k + 1 < times.size(); k++) {
        const IvpResult r = solver(p, th, {times[k], times[k + 1]}, row_of(x, k, NX), row_of(u_ref, k, NU), {});
        x.insert(x.end(), r.x.end() - NX, r.x.end());
    }
    return x;
}

void note(const char* what, const std::string& name, double value) {
    if (std::getenv("ZOH_PRINT_ERRORS") != nullptr) {
        std::printf("%-34s %-26s %.3e\n", what, name.c_str(), value);
    }
}

// Every combination of arc and dynamics, in km and s and in canonical units
void for_each_case(const std::function<void(const Dyn&, const Arc&, const SpiceParameters&, const std::string&)>& f) {
    for (const bool canon : {false, true}) {
        const SpiceParameters p = all_forces(canon);
        for (const Arc& arc : arcs()) {
            for (const Dyn& d : DYNS) {
                const std::string name = std::string(d.name) + " " + arc.name + (canon ? " canon" : " km");
                SCOPED_TRACE(name);
                f(d, arc, p, name);
            }
        }
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

double fault_time(const std::string& msg) {
    const std::string key = " at t = ";
    const size_t pos = msg.find(key);
    return pos == std::string::npos ? NAN_ : std::strtod(msg.c_str() + pos + key.size(), nullptr);
}

}  // namespace

TEST(Zoh, DefectsZeroOnOwnChain) {
    // The reference is the identical computation as each segment, so the same bits are expected: this
    // shows that each block comes from the last step of its own segment, in the documented layout.
    REQUIRE_KERNELS();
    for_each_case([](const Dyn& d, const Arc& arc, const SpiceParameters& p, const std::string& name) {
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> times = times_of(p, arc.seconds);
        const std::vector<double> x_ref = chain(d.solvestm, p, th, times, state_of(d, p, arc.rv, MASS), U_REF);
        const ZohLinearization z = d.propagate(p, th, times, x_ref, U_REF, {});

        const size_t segments = times.size() - 1;
        ASSERT_EQ(z.x_end.size(), segments * NX);
        ASSERT_EQ(z.Phi.size(), segments * NX * NX);
        ASSERT_EQ(z.Psi.size(), segments * NX * NU);
        ASSERT_EQ(z.c.size(), segments * NX);
        ASSERT_EQ(z.defects.size(), segments * NX);

        double worst = 0.0;
        for (const double value : z.defects) {
            worst = std::max(worst, std::abs(value));
        }
        note("own chain, max |defect|", name, worst);
        EXPECT_LE(worst, 1e-10);

        const double c2 = ControlledParameters(p, th, row_of(U_REF, 0, NU)).c2();
        for (size_t k = 0; k < segments; k++) {
            const IvpResult r = d.solvestm(p, th, {times[k], times[k + 1]}, row_of(x_ref, k, NX), row_of(U_REF, k, NU), {});
            EXPECT_EQ(row_of(z.x_end, k, NX), final_state_n(r, NX));
            EXPECT_EQ(row_of(z.Phi, k, NX * NX), final_stm_n(r, NX));
            EXPECT_EQ(row_of(z.Psi, k, NX * NU), final_stm_u(r, NX, NU));
            // row-major blocks, against the exact mass row, in the units of the segment
            const std::vector<double> Phi = row_of(z.Phi, k, NX * NX);
            const std::vector<double> Psi = row_of(z.Psi, k, NX * NU);
            EXPECT_EQ(Phi[NX * 6 + 6], 1.0);
            EXPECT_EQ(Phi[NX * 6 + 0], 0.0);
            EXPECT_EQ(Psi[NU * 6 + 0], 0.0);
            EXPECT_NEAR(Psi[NU * 6 + 3], -c2 * (times[k + 1] - times[k]), 1e-13 * c2 * (times[k + 1] - times[k]));
        }
    });
}

TEST(Zoh, DefectsFromStateOnly) {
    // The reference is a state-only run, as an SCP iterate would be. Its step sizes differ, since the
    // sensitivities take part in the step control, so these defects are the integration error, in
    // km, km/s and mass units; MEE elements and L are dimensionless.
    // Tolerances: about 10 times the largest value measured over all the cases, on Linux x86-64 in
    // Release and with an empty build type (measured: 4.0e-10 km, 7.6e-14 km/s, 8.9e-13, 1.5e-14).
    // The plan's estimate of 1e-6 km and 1e-11 km/s was far too loose.
    REQUIRE_KERNELS();
    const double tol_km = 4e-9, tol_kms = 7e-13, tol_element = 8e-12, tol_mass = 1.4e-13;
    for_each_case([&](const Dyn& d, const Arc& arc, const SpiceParameters& p, const std::string& name) {
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> times = times_of(p, arc.seconds);
        const std::vector<double> x_ref = chain(d.solve, p, th, times, state_of(d, p, arc.rv, MASS), U_REF);
        const ZohLinearization z = d.propagate(p, th, times, x_ref, U_REF, {});
        double position = 0.0, velocity = 0.0, element = 0.0, mass = 0.0;
        for (size_t k = 0; k + 1 < times.size(); k++) {
            for (int i = 0; i < 6; i++) {
                const double value = std::abs(z.defects[NX * k + i]);
                if (d.mee) {
                    if (i == 0) {
                        position = std::max(position, value * p.DU());
                    } else {
                        element = std::max(element, value);
                    }
                } else if (i < 3) {
                    position = std::max(position, value * p.DU());
                } else {
                    velocity = std::max(velocity, value * p.VU());
                }
            }
            mass = std::max(mass, std::abs(z.defects[NX * k + 6]));
        }
        note("state-only defect, km", name, position);
        note("state-only defect, km/s", name, velocity);
        note("state-only defect, element", name, element);
        note("state-only defect, mass", name, mass);
        EXPECT_LE(position, tol_km);
        EXPECT_LE(velocity, tol_kms);
        EXPECT_LE(element, tol_element);
        EXPECT_LE(mass, tol_mass);
    });
}

TEST(Zoh, DefectSignAndOwnStartState) {
    // defects = x_ref_{k+1} - x(t_{k+1}), with each segment starting from its own row of x_ref
    const SpiceParameters p = two_body();
    const Arc arc = arcs()[1];
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> times = times_of(p, arc.seconds);
        const std::vector<double> x_ref = chain(d.solvestm, p, th, times, state_of(d, p, arc.rv, MASS), U_REF);
        const ZohLinearization z0 = d.propagate(p, th, times, x_ref, U_REF, {});

        std::vector<double> shifted = x_ref;
        const double delta = 1e-3;
        shifted[NX * 2 + 0] += delta;   // first element of the state at t_2
        const ZohLinearization z1 = d.propagate(p, th, times, shifted, U_REF, {});
        // the end of the segment before does not move, and its defect does, by +delta
        EXPECT_EQ(row_of(z1.x_end, 1, NX), row_of(z0.x_end, 1, NX));
        EXPECT_NEAR(z1.defects[NX * 1 + 0] - z0.defects[NX * 1 + 0], delta, 1e-15 * std::abs(x_ref[NX * 2 + 0]) + 1e-15);
        // the segment after starts from the shifted state
        EXPECT_NE(row_of(z1.x_end, 2, NX), row_of(z0.x_end, 2, NX));
        const IvpResult direct = d.solvestm(p, th, {times[2], times[3]}, row_of(shifted, 2, NX), row_of(U_REF, 2, NU), {});
        EXPECT_EQ(row_of(z1.x_end, 2, NX), final_state_n(direct, NX));
    }
}

TEST(Zoh, AffineIdentity) {
    // Near-tautological: c is computed from these very terms, so this only shows that c has this sign
    // and that Phi and Psi are paired with x_ref and u_ref in this layout. That Phi and Psi are the
    // right sensitivities is shown by Zoh.ChainedVsFd.
    REQUIRE_KERNELS();
    const double eps = std::numeric_limits<double>::epsilon();
    for_each_case([&](const Dyn& d, const Arc& arc, const SpiceParameters& p, const std::string& name) {
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> times = times_of(p, arc.seconds);
        const std::vector<double> x_ref = chain(d.solve, p, th, times, state_of(d, p, arc.rv, MASS), U_REF);
        const ZohLinearization z = d.propagate(p, th, times, x_ref, U_REF, {});
        double worst = 0.0;
        for (size_t k = 0; k + 1 < times.size(); k++) {
            for (int i = 0; i < NX; i++) {
                double sum = z.c[NX * k + i];
                double magnitude = std::abs(z.c[NX * k + i]);
                for (int j = 0; j < NX; j++) {
                    const double term = z.Phi[NX * NX * k + NX * i + j] * x_ref[NX * k + j];
                    sum += term;
                    magnitude += std::abs(term);
                }
                for (int j = 0; j < NU; j++) {
                    const double term = z.Psi[NX * NU * k + NU * i + j] * U_REF[NU * k + j];
                    sum += term;
                    magnitude += std::abs(term);
                }
                const double residual = std::abs(sum - z.x_end[NX * k + i]);
                worst = std::max(worst, residual / (eps * magnitude));
                EXPECT_LE(residual, 8 * eps * magnitude) << "segment " << k << ", row " << i;
            }
        }
        note("affine residual, eps sum|terms|", name, worst);
    });
}

TEST(Zoh, ChainedVsFd) {
    // Chain the segments and compare with central differences of the final state of the whole arc, with
    // respect to x0 and to each u_k, by row and by column. The (r, v) rows are also checked alone, as
    // the mass row dominates the mass and Gamma columns (see top_rows).
    REQUIRE_KERNELS();
    const double tol = 1e-4;
    const double delta_tau = 1e-4, delta_Gamma = 1e-2;
    double worst_row = 0.0, worst_col = 0.0;
    for_each_case([&](const Dyn& d, const Arc& arc, const SpiceParameters& p, const std::string& name) {
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> times = times_of(p, arc.seconds);
        const std::vector<double> x0 = state_of(d, p, arc.rv, MASS);
        const std::vector<double> x_ref = chain(d.solve, p, th, times, x0, U_REF);
        const ZohLinearization z = d.propagate(p, th, times, x_ref, U_REF, {});
        const size_t segments = times.size() - 1;

        const auto final_of = [&](const std::vector<double>& x, const std::vector<double>& u) {
            const std::vector<double> xs = chain(d.solve, p, th, times, x, u);
            return std::vector<double>(xs.end() - NX, xs.end());
        };
        const std::vector<double> s0 = scales(d, x0);
        const std::vector<double> sf = scales(d, std::vector<double>(x_ref.end() - NX, x_ref.end()));
        const double floor = 1e-14;

        const auto check = [&](const std::vector<double>& M, const std::vector<double>& R, int cols,
                               const std::vector<double>& col_scale, const std::string& what) {
            const std::vector<double> top_scale(sf.begin(), sf.begin() + 6);
            const double row = matrix_error(M, R, NX, cols, sf, col_scale, floor, ErrorMode::row);
            const double col = matrix_error(M, R, NX, cols, sf, col_scale, floor, ErrorMode::col);
            const double top_row = matrix_error(top_rows(M, cols), top_rows(R, cols), 6, cols, top_scale, col_scale,
                                                floor, ErrorMode::row);
            const double top_col = matrix_error(top_rows(M, cols), top_rows(R, cols), 6, cols, top_scale, col_scale,
                                                floor, ErrorMode::col);
            note((what + " row").c_str(), name, std::max(row, top_row));
            note((what + " col").c_str(), name, std::max(col, top_col));
            worst_row = std::max({worst_row, row, top_row});
            worst_col = std::max({worst_col, col, top_col});
            EXPECT_LE(row, tol) << what;
            EXPECT_LE(col, tol) << what;
            EXPECT_LE(top_row, tol) << what;
            EXPECT_LE(top_col, tol) << what;
        };

        // d x_f / d x_0
        std::vector<double> h_x(NX);
        for (int i = 0; i < NX; i++) {
            h_x[i] = (i == 6 ? 1e-5 : 1e-7) * s0[i];
        }
        std::vector<double> Phi_all(NX * NX, 0.0), Phi_wrong_order(NX * NX, 0.0);
        for (int i = 0; i < NX; i++) {
            Phi_all[NX * i + i] = 1.0;
            Phi_wrong_order[NX * i + i] = 1.0;
        }
        for (size_t k = 0; k < segments; k++) {
            Phi_all = product(row_of(z.Phi, k, NX * NX), Phi_all, NX, NX);
            Phi_wrong_order = product(Phi_wrong_order, row_of(z.Phi, k, NX * NX), NX, NX);
        }
        const std::vector<double> fd_x = jacobian_fd_n(
            [&](const std::vector<double>& x) { return final_of(x, U_REF); }, x0, h_x);
        check(Phi_all, fd_x, NX, s0, "Phi");
        // ... and these metrics do see a chain in the wrong order
        const double wrong = matrix_error(Phi_wrong_order, fd_x, NX, NX, sf, s0, floor, ErrorMode::row);
        note("Phi in the wrong order, row", name, wrong);
        EXPECT_GT(wrong, 100 * tol);

        // d x_f / d u_k
        for (size_t k = 0; k < segments; k++) {
            std::vector<double> sens = row_of(z.Psi, k, NX * NU);
            for (size_t l = k + 1; l < segments; l++) {
                sens = product(row_of(z.Phi, l, NX * NX), sens, NX, NU);
            }
            const std::vector<double> u_k = row_of(U_REF, k, NU);
            const std::vector<double> fd_u = jacobian_fd_n(
                [&](const std::vector<double>& u) {
                    std::vector<double> all = U_REF;
                    std::copy(u.begin(), u.end(), all.begin() + NU * k);
                    return final_of(x0, all);
                },
                u_k, {delta_tau, delta_tau, delta_tau, delta_Gamma});
            check(sens, fd_u, NU, {}, "Psi u_" + std::to_string(k));
        }
    });
    note("chained vs FD, worst row", "all", worst_row);
    note("chained vs FD, worst col", "all", worst_col);
}

TEST(Zoh, BackwardTimes) {
    // Decreasing times integrate backward: Phi_k maps x(t_k) to x(t_{k+1}) with t_{k+1} < t_k, so a
    // backward segment undoes the forward one, and its Psi is -Phi_b Psi_f. Tolerances: about 10 times
    // the measured 1.7e-14, 4.5e-11 and 1.2e-13.
    const SpiceParameters p = two_body();
    const Arc arc = arcs()[1];
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> times = times_of(p, arc.seconds);
        const std::vector<double> x_ref = chain(d.solvestm, p, th, times, state_of(d, p, arc.rv, MASS), U_REF);
        const ZohLinearization forward = d.propagate(p, th, times, x_ref, U_REF, {});

        const size_t N = times.size();
        std::vector<double> times_b, x_b, u_b;
        for (size_t k = N; k-- > 0;) {
            times_b.push_back(times[k]);
            const std::vector<double> row = row_of(x_ref, k, NX);
            x_b.insert(x_b.end(), row.begin(), row.end());
        }
        for (size_t k = N - 1; k-- > 0;) {
            const std::vector<double> row = row_of(U_REF, k, NU);
            u_b.insert(u_b.end(), row.begin(), row.end());
        }
        const ZohLinearization backward = d.propagate(p, th, times_b, x_b, u_b, {});

        const std::vector<double> s = scales(d, x_ref);
        double defect = 0.0, inverse = 0.0, psi = 0.0;
        for (size_t k = 0; k + 1 < N; k++) {
            for (int i = 0; i < NX; i++) {
                defect = std::max(defect, std::abs(backward.defects[NX * k + i]) / s[i]);
            }
            const size_t f = N - 2 - k;   // the forward segment undone by backward segment k
            const std::vector<double> Phi_b = row_of(backward.Phi, k, NX * NX);
            std::vector<double> I(NX * NX, 0.0);
            for (int i = 0; i < NX; i++) {
                I[NX * i + i] = 1.0;
            }
            inverse = std::max(inverse, matrix_error(product(Phi_b, row_of(forward.Phi, f, NX * NX), NX, NX), I, NX, NX,
                                                     {}, {}, 0.0, ErrorMode::row));
            std::vector<double> minus = product(Phi_b, row_of(forward.Psi, f, NX * NU), NX, NU);
            for (double& value : minus) {
                value = -value;
            }
            psi = std::max(psi, matrix_error(row_of(backward.Psi, k, NX * NU), minus, NX, NU, {}, {}, 1e-13,
                                             ErrorMode::row));
        }
        note("backward, scaled defect", d.name, defect);
        note("backward, Phi_b Phi_f - I", d.name, inverse);
        note("backward, Psi_b + Phi_b Psi_f", d.name, psi);
        EXPECT_LE(defect, 2e-13);
        EXPECT_LE(inverse, 5e-10);
        EXPECT_LE(psi, 1.3e-12);
    }
}

TEST(Zoh, Validation) {
    const SpiceParameters p = two_body();
    const Arc arc = arcs()[1];
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> t = {0.0, 10.0, 20.0};
        const std::vector<double> x0 = state_of(d, p, arc.rv, MASS);
        std::vector<double> x = x0;
        x.insert(x.end(), x0.begin(), x0.end());
        x.insert(x.end(), x0.begin(), x0.end());
        const std::vector<double> u = {0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1};
        const auto call = [&](const std::vector<double>& t_, const std::vector<double>& x_,
                              const std::vector<double>& u_, const IvpOptions& o = {}) {
            return [&d, &p, &th, t_, x_, u_, o] { d.propagate(p, th, t_, x_, u_, o); };
        };
        EXPECT_NO_THROW(call(t, x, u)());

        // sizes
        expect_invalid(call({}, {}, {}), "at least 2", "no times");
        expect_invalid(call({0.0}, x0, {}), "at least 2", "one time");
        expect_invalid(call(t, x0, u), "x_ref", "x_ref short");
        expect_invalid(call(t, std::vector<double>(22, 1.0), u), "x_ref", "x_ref long");
        expect_invalid(call(t, x, {0.1, 0.0, 0.0, 0.1}), "u_ref", "u_ref short");
        expect_invalid(call(t, x, std::vector<double>(9, 0.1)), "u_ref", "u_ref long");

        // times: strictly monotonic in one direction
        expect_invalid(call({0.0, 0.0, 1.0}, x, u), "strictly monotonic", "repeated first");
        expect_invalid(call({0.0, 1.0, 1.0}, x, u), "strictly monotonic", "repeated last");
        expect_invalid(call({0.0, 2.0, 1.0}, x, u), "strictly monotonic", "up then down");
        expect_invalid(call({2.0, 1.0, 3.0}, x, u), "strictly monotonic", "down then up");
        EXPECT_NO_THROW(call({20.0, 10.0, 0.0}, x, u)());

        // non-finite values, at any row, including the last row of x_ref, which no segment starts from
        for (const double bad : {NAN_, INF, -INF}) {
            for (size_t i = 0; i < t.size(); i++) {
                std::vector<double> t_bad = t;
                t_bad[i] = bad;
                expect_invalid(call(t_bad, x, u), "times must be finite", "times " + std::to_string(i));
            }
            for (const size_t i : {size_t(0), size_t(7 + 3), size_t(14 + 6)}) {
                std::vector<double> x_bad = x;
                x_bad[i] = bad;
                expect_invalid(call(t, x_bad, u), "x_ref must be finite (row " + std::to_string(i / 7) + ")",
                               "x_ref " + std::to_string(i));
            }
            for (const size_t i : {size_t(0), size_t(7)}) {
                std::vector<double> u_bad = u;
                u_bad[i] = bad;
                expect_invalid(call(t, x, u_bad), "u_ref must be finite (row " + std::to_string(i / 4) + ")",
                               "u_ref " + std::to_string(i));
            }
        }

        // output times have no meaning here
        IvpOptions with_t_eval;
        with_t_eval.t_eval = {0.0, 20.0};
        expect_invalid(call(t, x, u, with_t_eval), "t_eval must be empty", "t_eval");

        // what the solvers check is reported with its type
        ThrustOptions no_thrust = th;
        no_thrust.thrust_N = 0.0;
        expect_invalid([&] { d.propagate(p, no_thrust, t, x, u, {}); }, "thrust_N", "thrust_N");
        IvpOptions bad_tol;
        bad_tol.atol = -1.0;
        expect_invalid(call(t, x, u, bad_tol), "atol", "atol");

        // Gamma is not tied to tau, as in the solvers
        EXPECT_NO_THROW(call(t, x, {0.5, 0.5, 0.5, -0.1, 0.0, 0.0, 0.0, 0.0})());

        // the last row is only compared, so a bad mass there is a defect and not an error
        std::vector<double> x_last = x;
        x_last.back() = -1.0;
        const ZohLinearization z = d.propagate(p, th, t, x_last, u, {});
        EXPECT_EQ(z.defects[NX + 6], -1.0 - z.x_end[NX + 6]);
    }
}

TEST(Zoh, FailuresKeepTheirType) {
    // domain_error for a state out of the domain, with the segment
    const SpiceParameters p = two_body();
    const Arc arc = arcs()[1];
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> t = {0.0, 10.0, 20.0, 30.0};
        const std::vector<double> x0 = state_of(d, p, arc.rv, MASS);
        std::vector<double> x;
        for (int k = 0; k < 4; k++) {
            x.insert(x.end(), x0.begin(), x0.end());
        }
        const std::vector<double> u(12, 0.0);
        x[NX * 1 + 6] = -0.5;   // the mass at the start of segment 1
        try {
            d.propagate(p, th, t, x, u, {});
            ADD_FAILURE() << "expected std::domain_error";
        } catch (const std::domain_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find(d.mee ? "propagate_zoh_meem: segment 1 " : "propagate_zoh_rvm: segment 1 "),
                      std::string::npos) << msg;
            EXPECT_NE(msg.find("mass out of domain"), std::string::npos) << msg;
        }
        if (d.mee) {
            std::vector<double> x_p = x;
            x_p[NX * 1 + 6] = MASS;
            x_p[NX * 2 + 0] = -1.0;   // p at the start of segment 2
            try {
                d.propagate(p, th, t, x_p, u, {});
                ADD_FAILURE() << "expected std::domain_error";
            } catch (const std::domain_error& e) {
                const std::string msg = e.what();
                EXPECT_NE(msg.find("segment 2 "), std::string::npos) << msg;
                EXPECT_NE(msg.find("MEE out of domain"), std::string::npos) << msg;
            }
        }
    }
}

TEST(Zoh, SpiceFailureKeepsItsType) {
    // runtime_error with NAIF's message: de440 has no PLUTO relative to the Moon
    REQUIRE_KERNELS();
    const SpiceParameters p(ET0, {"MOON", "PLUTO"}, with_gms({MU_MOON, 869.6}));
    const Arc arc = arcs()[1];
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> x0 = state_of(d, p, arc.rv, MASS);
        std::vector<double> x;
        for (int k = 0; k < 3; k++) {
            x.insert(x.end(), x0.begin(), x0.end());
        }
        try {
            d.propagate(p, th, {0.0, 60.0, 120.0}, x, std::vector<double>(8, 0.0), {});
            ADD_FAILURE() << "expected std::runtime_error";
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("segment 0 of 2"), std::string::npos) << msg;
            EXPECT_NE(msg.find("SPICE(SPKINSUFFDATA)"), std::string::npos) << msg;
        }
    }
}

TEST(Zoh, L_NotWrapped) {
    // L is not wrapped by the solvers, and x_ref must not be either. If a caller wraps L into [0, 2 pi),
    // the segment over which L crosses a multiple of 2 pi has a defect of -2 pi n in L, n being the
    // number of crossings, which is a false discontinuity of the trajectory: an SCP or a
    // multiple-shooting solver would try to close it. Everything else stays small: the dynamics are
    // 2 pi periodic in L. Callers should wrap only after solving.
    const SpiceParameters p = two_body();
    const Arc arc = arcs()[1];
    for (const Dyn& d : DYNS) {
        if (!d.mee) {
            continue;
        }
        SCOPED_TRACE(d.name);
        const ThrustOptions th = thruster(d, arc);
        const std::vector<double> times = times_of(p, arc.seconds);
        const std::vector<double> x_ref = chain(d.solvestm, p, th, times, state_of(d, p, arc.rv, MASS), U_REF);
        const size_t N = times.size();
        ASSERT_GT(x_ref[NX * (N - 1) + 5], 2 * PI) << "the arc must cross L = 2 pi";

        const ZohLinearization own = d.propagate(p, th, times, x_ref, U_REF, {});
        EXPECT_EQ(own.x_end[NX * (N - 2) + 5], x_ref[NX * (N - 1) + 5]) << "the solver must not wrap L";
        for (const double value : own.defects) {
            EXPECT_LE(std::abs(value), 1e-10);
        }

        std::vector<double> wrapped = x_ref;
        for (size_t k = 0; k < N; k++) {
            wrapped[NX * k + 5] -= 2 * PI * std::floor(x_ref[NX * k + 5] / (2 * PI));
        }
        const ZohLinearization z = d.propagate(p, th, times, wrapped, U_REF, {});
        int crossings = 0;
        double worst_L = 0.0, worst_others = 0.0;
        for (size_t k = 0; k + 1 < N; k++) {
            const double n = std::floor(x_ref[NX * (k + 1) + 5] / (2 * PI)) - std::floor(x_ref[NX * k + 5] / (2 * PI));
            crossings += static_cast<int>(n);
            worst_L = std::max(worst_L, std::abs(z.defects[NX * k + 5] + 2 * PI * n));
            for (int i = 0; i < NX; i++) {
                if (i != 5) {
                    worst_others = std::max(worst_others, std::abs(z.defects[NX * k + i]) / (i == 0 ? x_ref[0] : 1.0));
                }
            }
        }
        note("wrapped L, |defect_L + 2 pi n|", d.name, worst_L);
        note("wrapped L, other scaled defects", d.name, worst_others);
        EXPECT_GE(crossings, 1);
        EXPECT_LE(worst_L, 3e-13);    // measured 2.3e-14
        EXPECT_LE(worst_others, 1e-14);   // measured 8.9e-16
    }
}

TEST(Zoh, MassDepletionNamesSegment) {
    // The mass runs out within the last segment, whose start is valid. The fault has the depletion time.
    const SpiceParameters p = two_body();
    for (const Dyn& d : DYNS) {
        SCOPED_TRACE(d.name);
        ThrustOptions th;
        th.thrust_N = 1000.0;
        th.isp_s = 3000.0;
        th.mass_unit_kg = 1000.0;
        th.mee_control_frame = d.frame;
        const std::vector<double> u_row = {0.0, 0.0, 0.0, 1.0};
        const double t_star = MASS / ControlledParameters(p, th, u_row).c2();
        const std::vector<double> times = {0.0, 0.3 * t_star, 0.6 * t_star, 1.2 * t_star};
        std::vector<double> u_ref;
        for (int k = 0; k < 3; k++) {
            u_ref.insert(u_ref.end(), u_row.begin(), u_row.end());
        }
        const std::vector<double> first = chain(d.solve, p, th, {times[0], times[1], times[2]},
                                                state_of(d, p, state_llo(), MASS), u_ref);
        std::vector<double> x_ref = first;
        x_ref.insert(x_ref.end(), first.end() - NX, first.end());
        try {
            d.propagate(p, th, times, x_ref, u_ref, {});
            ADD_FAILURE() << "expected std::domain_error";
        } catch (const std::domain_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("segment 2 of 3"), std::string::npos) << msg;
            EXPECT_NE(msg.find("mass out of domain"), std::string::npos) << msg;
            EXPECT_NEAR(fault_time(msg), t_star, 1e-13 * t_star) << msg;
        }
        // the same arc is fine up to the depletion
        std::vector<double> early = times;
        early[3] = 0.99 * t_star;
        EXPECT_NO_THROW(d.propagate(p, th, early, x_ref, u_ref, {}));
    }
}

TEST(Zoh, MaxStepsIsPerSegmentAndNamesIt) {
    // thrust that deorbits STATE0: from about 9.9 days p -> 0 without reaching p <= 0
    REQUIRE_KERNELS();
    const SpiceParameters p(ET0, {"MOON", "EARTH", "SUN"}, with_gms(GM_LIST));
    ThrustOptions th;
    th.thrust_N = 0.5;
    th.isp_s = 3000.0;
    th.mass_unit_kg = 1850.0;
    const std::vector<double> u1 = {0.3, -0.5, 0.4, 0.71};
    std::vector<double> x0 = campanella::cart2mee(STATE0, GM_LIST[0]);
    x0.push_back(1.0);
    const std::vector<double> x1 = final_state_n(campanella::solve_ivp_meem(p, th, {0.0, 60.0}, x0, u1), NX);
    std::vector<double> x = x0;
    x.insert(x.end(), x1.begin(), x1.end());
    x.insert(x.end(), x1.begin(), x1.end());
    std::vector<double> u = u1;
    u.insert(u.end(), u1.begin(), u1.end());

    IvpOptions o;
    o.max_steps = 100000;
    try {
        campanella::propagate_zoh_meem(p, th, {0.0, 60.0, 20*86400.0}, x, u, o);
        ADD_FAILURE() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("propagate_zoh_meem: segment 1 of 2 "), std::string::npos) << msg;
        EXPECT_NE(msg.find("solve_ivpstm_meem: reached max_steps = 100000 at t = "), std::string::npos) << msg;
    }

    // a limit that each of the segments keeps, and not all of them together
    const std::vector<double> t = {0.0, 3600.0, 7200.0};
    const ZohLinearization ref = campanella::propagate_zoh_meem(p, th, t, x, u);
    std::size_t steps[2];
    for (std::size_t k = 0; k < 2; k++) {
        steps[k] = campanella::solve_ivpstm_meem(p, th, {t[k], t[k + 1]},
                                                 std::vector<double>(x.begin() + NX * k, x.begin() + NX * (k + 1)),
                                                 u1).size() - 1;
    }
    IvpOptions each;
    each.max_steps = static_cast<long long>(std::max(steps[0], steps[1]));
    EXPECT_EQ(campanella::propagate_zoh_meem(p, th, t, x, u, each).x_end, ref.x_end);
    each.max_steps = 0;
    EXPECT_EQ(campanella::propagate_zoh_meem(p, th, t, x, u, each).x_end, ref.x_end);
    each.max_steps = -1;
    EXPECT_THROW(campanella::propagate_zoh_meem(p, th, t, x, u, each), std::invalid_argument);
}
