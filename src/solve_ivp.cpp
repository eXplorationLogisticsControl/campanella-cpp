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

namespace detail {

namespace {

void check_positive(double value, const char* name, const char* who) {
    if (!(std::isfinite(value) && value > 0.0)) {
        throw std::invalid_argument(std::string(who) + ": " + name + " must be finite and > 0");
    }
}

}  // namespace

ZohControl make_control(const SpiceParameters& params, const ThrustOptions& thrust,
                        const std::vector<double>& u, const char* who) {
    check_positive(thrust.thrust_N, "thrust_N", who);
    check_positive(thrust.isp_s, "isp_s", who);
    check_positive(thrust.mass_unit_kg, "mass_unit_kg", who);
    check_positive(thrust.g0, "g0", who);
    if (thrust.mee_control_frame != ThrustFrame::rtn && thrust.mee_control_frame != ThrustFrame::inertial) {
        throw std::invalid_argument(std::string(who) + ": mee_control_frame is not a ThrustFrame");
    }
    if (u.size() != 4) {
        throw std::invalid_argument(std::string(who) + ": u must have length 4");
    }
    ZohControl control;
    for (int i = 0; i < 4; i++) {
        if (!std::isfinite(u[i])) {
            throw std::invalid_argument(std::string(who) + ": u must be finite");
        }
        control.u[i] = u[i];
    }
    const double DU = params.DU();
    const double TU = params.TU();
    control.c1 = (thrust.thrust_N*1e-3) / (thrust.mass_unit_kg*DU/(TU*TU));
    control.c2 = thrust.thrust_N / (thrust.isp_s*thrust.g0) * TU / thrust.mass_unit_kg;
    // valid one by one, the inputs may still overflow or underflow together: c1 = 0 would mean no thrust
    check_positive(control.c1, "c1 (from thrust_N, mass_unit_kg and the units)", who);
    check_positive(control.c2, "c2 (from thrust_N, isp_s, g0, mass_unit_kg and the units)", who);
    control.mass_unit_kg = thrust.mass_unit_kg;
    control.frame = thrust.mee_control_frame;
    return control;
}

}  // namespace detail

ControlledParameters::ControlledParameters(const SpiceParameters& params, const ThrustOptions& thrust,
                                           const std::vector<double>& u)
    : params_(&params), thrust_(thrust) {
    const detail::ZohControl control = detail::make_control(params, thrust, u, "ControlledParameters");
    for (int i = 0; i < 4; i++) {
        u_[i] = control.u[i];
    }
    c1_ = control.c1;
    c2_ = control.c2;
}

IvpResult solve_ivp_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                       const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhs_rv, Access::model(params), 6, 0, false, nullptr, tspan, x0, options,
                             "solve_ivp_rv");
}

IvpResult solve_ivp_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                        const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhs_mee, Access::model(params), 6, 0, false, nullptr, tspan, x0, options,
                             "solve_ivp_mee");
}

IvpResult solve_ivpstm_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                          const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhsstm_rv, Access::model(params), 6, 0, true, nullptr, tspan, x0, options,
                             "solve_ivpstm_rv");
}

IvpResult solve_ivpstm_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                           const std::vector<double>& x0, const IvpOptions& options) {
    return detail::integrate(&detail::rhsstm_mee, Access::model(params), 6, 0, true, nullptr, tspan, x0, options,
                             "solve_ivpstm_mee");
}

namespace {

IvpResult solve_zoh(detail::Rhs rhs, bool with_stm, const SpiceParameters& params,
                    const ThrustOptions& thrust, const std::vector<double>& tspan,
                    const std::vector<double>& x0, const std::vector<double>& u,
                    const IvpOptions& options, const char* who) {
    const detail::ZohControl control = detail::make_control(params, thrust, u, who);
    return detail::integrate(rhs, Access::model(params), 7, 4, with_stm, &control, tspan, x0, options, who);
}

}  // namespace

IvpResult solve_ivp_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                        const std::vector<double>& tspan, const std::vector<double>& x0,
                        const std::vector<double>& u, const IvpOptions& options) {
    return solve_zoh(&detail::rhs_rvm, false, params, thrust, tspan, x0, u, options, "solve_ivp_rvm");
}

IvpResult solve_ivp_meem(const SpiceParameters& params, const ThrustOptions& thrust,
                         const std::vector<double>& tspan, const std::vector<double>& x0,
                         const std::vector<double>& u, const IvpOptions& options) {
    return solve_zoh(&detail::rhs_meem, false, params, thrust, tspan, x0, u, options, "solve_ivp_meem");
}

IvpResult solve_ivpstm_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                           const std::vector<double>& tspan, const std::vector<double>& x0,
                           const std::vector<double>& u, const IvpOptions& options) {
    return solve_zoh(&detail::rhsstm_rvm, true, params, thrust, tspan, x0, u, options, "solve_ivpstm_rvm");
}

IvpResult solve_ivpstm_meem(const SpiceParameters& params, const ThrustOptions& thrust,
                            const std::vector<double>& tspan, const std::vector<double>& x0,
                            const std::vector<double>& u, const IvpOptions& options) {
    return solve_zoh(&detail::rhsstm_meem, true, params, thrust, tspan, x0, u, options, "solve_ivpstm_meem");
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

int call_rhs_zoh(detail::Rhs rhs, double t, const double x[], double dxdt[], void* params) {
    const ControlledParameters& cp = *static_cast<const ControlledParameters*>(params);
    detail::ZohControl control;
    control.c1 = cp.c1();
    control.c2 = cp.c2();
    control.mass_unit_kg = cp.thrust().mass_unit_kg;
    control.frame = cp.thrust().mee_control_frame;
    for (int i = 0; i < 4; i++) {
        control.u[i] = cp.u()[i];
    }
    detail::Context ctx{&Access::model(cp.params()), &Access::workspace(cp.params()), &control};
    return rhs(t, x, dxdt, &ctx);
}

using EvaluateZoh = int (*)(const detail::Model&, const detail::ZohControl&, detail::Workspace&, double,
                            const double[7], double[7], double[7][7], double[7][4]);

EomEvalControl evaluate_zoh(EvaluateZoh f, const SpiceParameters& params, const ThrustOptions& thrust,
                            double t, const std::vector<double>& x, const std::vector<double>& u,
                            const char* who) {
    const detail::ZohControl control = detail::make_control(params, thrust, u, who);
    if (x.size() != 7) {
        throw std::invalid_argument(std::string(who) + ": x must have length 7");
    }
    const detail::Model& model = Access::model(params);
    detail::Workspace& ws = Access::workspace(params);   // reused: no allocation per call
    EomEvalControl out;
    double A[7][7];
    double B[7][4];
    int status = f(model, control, ws, t, x.data(), out.xdot.data(), A, B);
    if (status != detail::RHS_SUCCESS) {
        detail::throw_fault(model, ws, status, t, who);
    }
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            out.A[7*i + j] = A[i][j];
        }
        for (int j = 0; j < 4; j++) {
            out.B[4*i + j] = B[i][j];
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

int eom_rvm(double t, const double x[], double dxdt[], void* params) {
    return call_rhs_zoh(&detail::rhs_rvm, t, x, dxdt, params);
}

int eom_meem(double t, const double x[], double dxdt[], void* params) {
    return call_rhs_zoh(&detail::rhs_meem, t, x, dxdt, params);
}

int eomstm_rvm(double t, const double x[], double dxdt[], void* params) {
    return call_rhs_zoh(&detail::rhsstm_rvm, t, x, dxdt, params);
}

int eomstm_meem(double t, const double x[], double dxdt[], void* params) {
    return call_rhs_zoh(&detail::rhsstm_meem, t, x, dxdt, params);
}

EomEvalControl eval_rvm(const SpiceParameters& params, const ThrustOptions& thrust, double t,
                        const std::vector<double>& x, const std::vector<double>& u) {
    return evaluate_zoh(&detail::rvm_evaluate, params, thrust, t, x, u, "eval_rvm");
}

EomEvalControl eval_meem(const SpiceParameters& params, const ThrustOptions& thrust, double t,
                         const std::vector<double>& x, const std::vector<double>& u) {
    return evaluate_zoh(&detail::meem_evaluate, params, thrust, t, x, u, "eval_meem");
}

std::string eom_error(const SpiceParameters& params) {
    const detail::Workspace& ws = Access::workspace(params);
    if (ws.fault == detail::Fault::none) {
        return "";
    }
    return detail::fault_message(Access::model(params), ws, 0, std::nan(""));
}

}  // namespace campanella
