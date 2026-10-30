/**
 * Public solvers, equations of motion and single-point evaluation
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "campanella.hpp"
#include "integrate.h"
#include "model.h"

namespace campanella {

using detail::Access;

IvpResult solve_ivp_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                       const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhs_rv, Access::model(params), false, tspan, x0, options,
                             "solve_ivp_rv");
}

IvpResult solve_ivp_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                        const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhs_mee, Access::model(params), false, tspan, x0, options,
                             "solve_ivp_mee");
}

IvpResult solve_ivpstm_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                          const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhsstm_rv, Access::model(params), true, tspan, x0, options,
                             "solve_ivpstm_rv");
}

IvpResult solve_ivpstm_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                           const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhsstm_mee, Access::model(params), true, tspan, x0, options,
                             "solve_ivpstm_mee");
}


namespace {

int call_rhs(detail::Rhs rhs, double t, const double x[], double dxdt[], void* params) {
    const SpiceParameters& p = *static_cast<const SpiceParameters*>(params);
    detail::Context ctx{&Access::model(p), &Access::workspace(p)};
    return rhs(t, x, dxdt, &ctx);
}

using Evaluate = int (*)(const detail::Model&, detail::Workspace&, double, const double[6],
                         double[6], double[6][6]);

EomEval evaluate(Evaluate f, const SpiceParameters& params, double t, const std::vector<double>& x,
                 const char* who) {
    if (x.size() != 6) {
        throw std::invalid_argument(std::string(who) + ": x must have length 6");
    }
    const detail::Model& model = Access::model(params);
    detail::Workspace& ws = Access::workspace(params);   // reused: no allocation per call
    EomEval out;
    double A[6][6];
    int status = f(model, ws, t, x.data(), out.xdot.data(), A);
    if (status != detail::RHS_SUCCESS) {
        detail::throw_fault(model, ws, status, t, who);
    }
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            out.A[6*i + j] = A[i][j];
        }
    }
    return out;
}

}  // namespace

int eom_rv(double t, const double x[], double dxdt[], void* params) {
    return call_rhs(&detail::rhs_rv, t, x, dxdt, params);
}

int eom_mee(double t, const double x[], double dxdt[], void* params) {
    return call_rhs(&detail::rhs_mee, t, x, dxdt, params);
}

int eomstm_rv(double t, const double x[], double dxdt[], void* params) {
    return call_rhs(&detail::rhsstm_rv, t, x, dxdt, params);
}

int eomstm_mee(double t, const double x[], double dxdt[], void* params) {
    return call_rhs(&detail::rhsstm_mee, t, x, dxdt, params);
}

EomEval eval_rv(const SpiceParameters& params, double t, const std::vector<double>& x) {
    return evaluate(&detail::rv_evaluate, params, t, x, "eval_rv");
}

EomEval eval_mee(const SpiceParameters& params, double t, const std::vector<double>& x) {
    return evaluate(&detail::mee_evaluate, params, t, x, "eval_mee");
}

std::string eom_error(const SpiceParameters& params) {
    const detail::Workspace& ws = Access::workspace(params);
    if (ws.fault == detail::Fault::none) {
        return "";
    }
    return detail::fault_message(Access::model(params), ws, 0, std::nan(""));
}

}  // namespace campanella
