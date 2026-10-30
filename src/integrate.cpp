/**
 * Integration of the equations of motion with GSL's rk8pd stepper
 */

#include <cmath>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include "integrate.h"

namespace campanella {
namespace detail {

namespace {

struct StepFree { void operator()(gsl_odeiv2_step* p) const { gsl_odeiv2_step_free(p); } };
struct ControlFree { void operator()(gsl_odeiv2_control* p) const { gsl_odeiv2_control_free(p); } };
struct EvolveFree { void operator()(gsl_odeiv2_evolve* p) const { gsl_odeiv2_evolve_free(p); } };

// GSL's default error handler aborts the process; report through return codes instead
struct GslErrorHandlerOff {
    gsl_error_handler_t* previous = gsl_set_error_handler_off();
    ~GslErrorHandlerOff() { gsl_set_error_handler(previous); }
};

[[noreturn]] void invalid(const char* who, const std::string& what) {
    throw std::invalid_argument(std::string(who) + ": " + what);
}

void validate(const std::vector<double>& tspan, const std::vector<double>& x0,
              const IvpOptions& options, const char* who) {
    if (tspan.size() != 2) {
        invalid(who, "tspan must be [t0, tf]");
    }
    if (!std::isfinite(tspan[0]) || !std::isfinite(tspan[1])) {
        invalid(who, "tspan must be finite");
    }
    if (x0.size() != 6) {
        invalid(who, "x0 must have length 6");
    }
    for (double xi : x0) {
        if (!std::isfinite(xi)) {
            invalid(who, "x0 must be finite");
        }
    }
    if (!(options.atol >= 0.0) || !(options.rtol >= 0.0) || !std::isfinite(options.atol)
        || !std::isfinite(options.rtol) || (options.atol == 0.0 && options.rtol == 0.0)) {
        invalid(who, "atol and rtol must be finite and >= 0, and not both zero");
    }
    const double t0 = tspan[0];
    const double tf = tspan[1];
    const double dir = (tf > t0) ? 1.0 : ((tf < t0) ? -1.0 : 0.0);
    const std::vector<double>& te = options.t_eval;
    for (size_t k = 0; k < te.size(); k++) {
        if (!std::isfinite(te[k]) || dir*(te[k] - t0) < 0 || dir*(tf - te[k]) < 0
            || (dir == 0.0 && te[k] != t0)) {
            invalid(who, "t_eval must lie within tspan");
        }
        if (k > 0 && !(dir*(te[k] - te[k - 1]) > 0)) {
            invalid(who, "t_eval must be strictly monotonic in the direction of integration");
        }
    }
}

}  // namespace

/**
 * Integrate the equations of motion over tspan
 *
 * @param rhs GSL-compatible RHS, called with a Context*
 * @param model dynamics model
 * @param with_stm if true, integrate 42 states with Phi(t0, t0) = I
 * @param tspan [t0, tf], in TU; tf < t0 integrates backward
 * @param x0 initial state (6)
 * @param options tolerances and output times
 * @param who name of the calling function, for the error messages
 * @return t, x and (with_stm) stm at every accepted step, or at options.t_eval
 */
IvpResult integrate(Rhs rhs, const Model& model, bool with_stm, const std::vector<double>& tspan,
                    const std::vector<double>& x0, const IvpOptions& options, const char* who) {
    validate(tspan, x0, options, who);
    const size_t dim = with_stm ? 42 : 6;
    const double t0 = tspan[0];
    const double tf = tspan[1];
    const double dir = (tf < t0) ? -1.0 : 1.0;

    double y[42];
    for (size_t i = 0; i < 6; i++) {
        y[i] = x0[i];
    }
    if (with_stm) {
        for (size_t i = 0; i < 36; i++) {
            y[6 + i] = (i % 7 == 0) ? 1.0 : 0.0;
        }
    }

    GslErrorHandlerOff handler_off;
    // ref: https://www.gnu.org/software/gsl/doc/html/ode-initval.html#c.gsl_odeiv2_control_y_new
    std::unique_ptr<gsl_odeiv2_step, StepFree> step(gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, dim));
    std::unique_ptr<gsl_odeiv2_control, ControlFree> control(
        gsl_odeiv2_control_y_new(options.atol, options.rtol));
    std::unique_ptr<gsl_odeiv2_evolve, EvolveFree> evolve(gsl_odeiv2_evolve_alloc(dim));
    if (!step || !control || !evolve) {
        throw std::bad_alloc();
    }
    Workspace ws(model);
    Context ctx{&model, &ws};
    gsl_odeiv2_system sys = {rhs, nullptr, dim, &ctx};

    IvpResult out;
    auto record = [&](double t) {
        out.t.push_back(t);
        out.x.insert(out.x.end(), y, y + 6);
        if (with_stm) {
            out.stm.insert(out.stm.end(), y + 6, y + 42);
        }
    };

    double t = t0;
    double h = dir*1.0;
    const bool every_step = options.t_eval.empty();
    // a step that ends on target sets t = target exactly, and keeps h for the next target
    auto advance = [&](double target) {
        while (dir*(target - t) > 0) {
            int status = gsl_odeiv2_evolve_apply(evolve.get(), control.get(), step.get(), &sys,
                                                 &t, target, &h, y);
            if (status != GSL_SUCCESS) {
                throw_fault(model, ws, status, t, who);
            }
            if (every_step) {
                record(t);
            }
        }
    };

    if (every_step) {
        record(t);
        advance(tf);
    } else {
        for (double te : options.t_eval) {
            advance(te);
            record(t);
        }
    }
    return out;
}


/**
 * Message for the failure described by ws.fault
 *
 * @param model dynamics model
 * @param ws scratch space of the failed call
 * @param status GSL status
 * @param t time of the failure, in TU
 */
std::string fault_message(const Model& model, const Workspace& ws, int status, double t) {
    std::ostringstream msg;
    msg.precision(15);
    std::ostringstream at;
    at.precision(15);
    if (!std::isnan(t)) {
        at << " at t = " << t << " TU";
    }
    switch (ws.fault) {
    case Fault::spice:
        msg << "SPICE error" << at.str() << ": " << ws.spice_msg;
        break;
    case Fault::sh_domain:
        msg << "spherical harmonics: the trajectory went below sh_rho_min * R (|r|/R = " << ws.sh_ws->rho
            << " < " << model.options.sh_rho_min << ", R = " << model.sh->R_km << " km)" << at.str()
            << "; the integration was stopped. For low or landing arcs, a sh_rho_min slightly below 1"
               " (e.g. 0.99) may be used, but the series is not guaranteed to converge there.";
        break;
    case Fault::mee_domain:
        msg << "MEE out of domain (p <= 0 or w = 1 + f cos(L) + g sin(L) <= 0)" << at.str();
        break;
    case Fault::nonfinite:
        msg << "non-finite derivative (e.g. at the center of a body)" << at.str();
        break;
    case Fault::none:
        msg << "integration failed (GSL status " << status << ": " << gsl_strerror(status) << ")";
        if (!std::isnan(t)) {
            msg << " near t = " << t << " TU";
        }
        break;
    }
    return msg.str();
}


/**
 * Throw the exception for the failure described by ws.fault: std::runtime_error for a SPICE error
 * or a GSL failure, std::domain_error for a state out of the domain
 *
 * @param model dynamics model
 * @param ws scratch space of the failed call
 * @param status GSL status
 * @param t time of the failure, in TU
 * @param who name of the calling function
 */
void throw_fault(const Model& model, const Workspace& ws, int status, double t, const char* who) {
    const std::string msg = std::string(who) + ": " + fault_message(model, ws, status, t);
    if (ws.fault == Fault::spice || ws.fault == Fault::none) {
        throw std::runtime_error(msg);
    }
    throw std::domain_error(msg);
}

}  // namespace detail
}  // namespace campanella
