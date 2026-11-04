/**
 * Linearization of zero-order-hold segments, one integration per segment
 */

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "campanella.hpp"

namespace campanella {

namespace {

constexpr std::size_t NX = 7;
constexpr std::size_t NU = 4;

using Solver = IvpResult (*)(const SpiceParameters&, const ThrustOptions&, const std::vector<double>&,
                             const std::vector<double>&, const std::vector<double>&, const IvpOptions&);

[[noreturn]] void invalid(const char* who, const std::string& what) {
    throw std::invalid_argument(std::string(who) + ": " + what);
}

void check_finite(const std::vector<double>& values, std::size_t width, const char* name, const char* who) {
    for (std::size_t i = 0; i < values.size(); i++) {
        if (!std::isfinite(values[i])) {
            invalid(who, std::string(name) + " must be finite (row " + std::to_string(i/width) + ")");
        }
    }
}

// Checks what no segment sees (the last row of x_ref) or would see too late (a later row or time)
void validate(const std::vector<double>& times, const std::vector<double>& x_ref,
              const std::vector<double>& u_ref, const IvpOptions& options, const char* who) {
    const std::size_t n = times.size();
    if (n < 2) {
        invalid(who, "times must have at least 2 elements");
    }
    if (x_ref.size() != NX*n) {
        invalid(who, "x_ref must have 7 N elements, for N times");
    }
    if (u_ref.size() != NU*(n - 1)) {
        invalid(who, "u_ref must have 4 (N - 1) elements, for N times");
    }
    if (!options.t_eval.empty()) {
        invalid(who, "t_eval must be empty: each segment returns its final time only");
    }
    check_finite(times, 1, "times", who);
    const double dir = (times[1] < times[0]) ? -1.0 : 1.0;
    for (std::size_t k = 0; k + 1 < n; k++) {
        if (!(dir*(times[k + 1] - times[k]) > 0.0)) {
            invalid(who, "times must be strictly monotonic (row " + std::to_string(k + 1) + ")");
        }
    }
    check_finite(x_ref, NX, "x_ref", who);
    check_finite(u_ref, NU, "u_ref", who);
}

// Solve one segment. A domain_error, invalid_argument or runtime_error is rethrown as that base type, with
// the segment added to its message, so a class derived from them is not kept. Anything else (another
// exception thrown by options.poll) passes through unchanged, without the segment.
IvpResult solve_segment(Solver solve, const SpiceParameters& params, const ThrustOptions& thrust,
                        const std::vector<double>& times, const std::vector<double>& x_ref,
                        const std::vector<double>& u_ref, const IvpOptions& options, std::size_t k,
                        const char* who) {
    const auto where = [&](const std::exception& e) {
        std::ostringstream msg;
        msg.precision(15);
        msg << who << ": segment " << k << " of " << times.size() - 1 << " (t = [" << times[k] << ", "
            << times[k + 1] << "] TU): " << e.what();
        return msg.str();
    };
    try {
        const std::vector<double> x0(x_ref.begin() + NX*k, x_ref.begin() + NX*(k + 1));
        const std::vector<double> u(u_ref.begin() + NU*k, u_ref.begin() + NU*(k + 1));
        // only the end of the segment is used, so the steps are not recorded
        IvpOptions end_only = options;
        end_only.t_eval = {times[k + 1]};
        return solve(params, thrust, {times[k], times[k + 1]}, x0, u, end_only);
    } catch (const std::domain_error& e) {
        throw std::domain_error(where(e));
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(where(e));
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(where(e));
    }
}

// Each segment starts from its own row of x_ref, not from the end of the previous one. See the header
// for the backward case; the defects of a caller that wraps L are tested in Zoh.L_NotWrapped.
ZohLinearization propagate(Solver solve, const SpiceParameters& params, const ThrustOptions& thrust,
                           const std::vector<double>& times, const std::vector<double>& x_ref,
                           const std::vector<double>& u_ref, const IvpOptions& options, const char* who) {
    validate(times, x_ref, u_ref, options, who);
    const std::size_t segments = times.size() - 1;
    ZohLinearization out;
    out.x_end.reserve(NX*segments);
    out.Phi.reserve(NX*NX*segments);
    out.Psi.reserve(NX*NU*segments);
    out.c.reserve(NX*segments);
    out.defects.reserve(NX*segments);

    for (std::size_t k = 0; k < segments; k++) {
        const IvpResult res = solve_segment(solve, params, thrust, times, x_ref, u_ref, options, k, who);
        const double* x_end = res.x.data() + res.x.size() - NX;
        const double* Phi = res.stm.data() + res.stm.size() - NX*NX;
        const double* Psi = res.stm_u.data() + res.stm_u.size() - NX*NU;
        const double* x_k = x_ref.data() + NX*k;
        const double* x_next = x_k + NX;
        const double* u_k = u_ref.data() + NU*k;

        out.x_end.insert(out.x_end.end(), x_end, x_end + NX);
        out.Phi.insert(out.Phi.end(), Phi, Phi + NX*NX);
        out.Psi.insert(out.Psi.end(), Psi, Psi + NX*NU);
        for (std::size_t i = 0; i < NX; i++) {
            double c_i = x_end[i];
            for (std::size_t j = 0; j < NX; j++) {
                c_i -= Phi[NX*i + j]*x_k[j];
            }
            for (std::size_t j = 0; j < NU; j++) {
                c_i -= Psi[NU*i + j]*u_k[j];
            }
            out.c.push_back(c_i);
            out.defects.push_back(x_next[i] - x_end[i]);
        }
    }
    return out;
}

}  // namespace

ZohLinearization propagate_zoh_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                                   const std::vector<double>& times, const std::vector<double>& x_ref,
                                   const std::vector<double>& u_ref, const IvpOptions& options) {
    return propagate(&solve_ivpstm_rvm, params, thrust, times, x_ref, u_ref, options, "propagate_zoh_rvm");
}

ZohLinearization propagate_zoh_meem(const SpiceParameters& params, const ThrustOptions& thrust,
                                    const std::vector<double>& times, const std::vector<double>& x_ref,
                                    const std::vector<double>& u_ref, const IvpOptions& options) {
    return propagate(&solve_ivpstm_meem, params, thrust, times, x_ref, u_ref, options, "propagate_zoh_meem");
}

}  // namespace campanella
