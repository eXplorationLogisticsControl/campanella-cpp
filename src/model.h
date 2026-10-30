#ifndef CAMPANELLA_MODEL_H
#define CAMPANELLA_MODEL_H

// Internal dynamics model shared by the EOMs and the integrator. Not part of the public header.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "campanella.hpp"
#include "perturbations/spherical_harmonics.h"

namespace campanella {
namespace detail {

/**
 * Everything the EOMs need, derived once from SpiceParameters. Immutable, may be shared.
 */
struct Model {
    SpiceOptions options;                  // inputs, with GMs filled in
    double et0 = 0.0;
    std::vector<std::string> naif_ids;     // as given
    std::vector<int> codes;                // NAIF codes; [0] is the central body
    std::vector<double> mus;               // GMs * TU^2 / DU^3
    double DU = 1.0;
    double TU = 1.0;
    double VU = 1.0;
    // SRP: index of the Sun in codes (0 if it is the central body), -1 if SRP is off
    int idx_sun = -1;
    double srp_k = 0.0;                    // a_srp = srp_k d / |d|^3, d = r - r_sun, in DU^3/TU^2
    // central-body spherical harmonics; nullptr if off
    std::shared_ptr<const SphericalHarmonicsModel> sh;
    double mu_sh = 0.0;                    // GM of the file * TU^2 / DU^3
    double R_sh = 0.0;                     // reference radius of the file / DU
};

// Why the last RHS call failed
enum class Fault { none, mee_domain, sh_domain, nonfinite, spice };

/**
 * Scratch space written at every RHS call. One per integration; never shared between threads.
 */
struct Workspace {
    std::vector<double> body_pos;          // 3 per NAIF code, w.r.t. the central body, in DU
    std::optional<SphericalHarmonicsWorkspace> sh_ws;
    Fault fault = Fault::none;
    std::string spice_msg;                 // NAIF's message of the last SPICE error
    explicit Workspace(const Model& model);
};

// What GSL's void* points to inside the library
struct Context {
    const Model* model;
    Workspace* ws;
};

// Access to the internals of SpiceParameters
struct Access {
    static const Model& model(const SpiceParameters& p) { return *p.model_; }
    static Workspace& workspace(const SpiceParameters& p) { return *p.ws_; }
};

// Status codes of the RHS functions (values of gsl_errno.h, without including GSL here)
constexpr int RHS_SUCCESS = 0;    // GSL_SUCCESS
constexpr int RHS_EDOM = 1;       // GSL_EDOM: out of domain, GSL retries with a smaller step
constexpr int RHS_EBADFUNC = 9;   // GSL_EBADFUNC: SPICE error, GSL stops

/**
 * Add the perturbing accelerations (third bodies, then SRP, then spherical harmonics) at time t
 * (TU) and position r (DU) to a, and their gradient d(a)/d(r) to G if G is not null.
 * Adding (rather than overwriting) keeps the summation order of the callers.
 *
 * @return RHS_SUCCESS; RHS_EDOM below sh_rho_min * R (ws.fault = sh_domain);
 *         RHS_EBADFUNC on a SPICE error (ws.fault = spice, NAIF's message in ws.spice_msg, reset_c called)
 */
int add_perturbations(const Model& model, Workspace& ws, double t, const double r[3],
                      double a[3], double G[3][3]);

// Cartesian RHS: xdot (6) and, if A is not null, A = d(xdot)/dx. Return codes as add_perturbations.
int rv_evaluate(const Model& model, Workspace& ws, double t, const double x[6],
                double xdot[6], double A[6][6]);

// GVE in MEE: xdot (6) and, if A is not null, A = d(xdot)/dx. Also RHS_EDOM (ws.fault = mee_domain)
// if p <= 0 or w <= 0.
int mee_evaluate(const Model& model, Workspace& ws, double t, const double x[6],
                 double xdot[6], double A[6][6]);

// If any of the n values is not finite: ws.fault = nonfinite and RHS_EDOM; else RHS_SUCCESS
int check_finite(Workspace& ws, const double* v, int n);

// GSL-compatible RHS; params is a Context*. 6 states, or 42 for the rhsstm_* variants.
int rhs_rv(double t, const double x[], double xdot[], void* params);
int rhsstm_rv(double t, const double x[], double xdot[], void* params);
int rhs_mee(double t, const double x[], double xdot[], void* params);
int rhsstm_mee(double t, const double x[], double xdot[], void* params);

// Message describing ws.fault after a failure at time t (TU), with status the GSL status
std::string fault_message(const Model& model, const Workspace& ws, int status, double t);

// Throw the exception describing ws.fault (see fault_message). who names the calling function.
[[noreturn]] void throw_fault(const Model& model, const Workspace& ws, int status, double t,
                              const char* who);

// NAIF's short and long messages of the current SPICE error, e.g. "SPICE(SPKINSUFFDATA) -- ...";
// resets the SPICE error state
std::string spice_error_message();

// If SPICE failed, reset it and throw std::runtime_error with NAIF's message
void spice_throw_if_failed();

}  // namespace detail
}  // namespace campanella

#endif  // CAMPANELLA_MODEL_H
