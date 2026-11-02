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
enum class Fault { none, mee_domain, sh_domain, nonfinite, spice, mass_domain };

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

/**
 * Zero-order-hold control of the mass-augmented dynamics, with its constants in the units of Model.
 * thrust acceleration = (c1 / m) tau, mdot = -c2 Gamma
 */
struct ZohControl {
    double c1 = 0.0;                       // thrust acceleration at m = 1, in DU/TU^2
    double c2 = 0.0;                       // mass flow at Gamma = 1, in mass units per TU
    double mass_unit_kg = 0.0;             // mass unit m_bar, in kg
    ThrustFrame frame = ThrustFrame::rtn;  // frame of tau (meem only)
    double u[4] = {0.0, 0.0, 0.0, 0.0};    // tau (3), Gamma
};

// What GSL's void* points to inside the library
struct Context {
    const Model* model;
    Workspace* ws;
    const ZohControl* control = nullptr;   // required by the *_rvm and *_meem RHS, unused by the others
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
 * @param srp_k SRP coefficient to use (see Model::srp_k); ignored if SRP is off
 * @param a_srp if not null, receives the SRP acceleration alone; zero if SRP is off
 * @return RHS_SUCCESS; RHS_EDOM below sh_rho_min * R (ws.fault = sh_domain);
 *         RHS_EBADFUNC on a SPICE error (ws.fault = spice, NAIF's message in ws.spice_msg, reset_c called)
 */
int add_perturbations(const Model& model, Workspace& ws, double t, const double r[3], double srp_k,
                      double a[3], double G[3][3], double a_srp[3]);

// add_perturbations with the fixed SRP coefficient of the model
inline int add_perturbations(const Model& model, Workspace& ws, double t, const double r[3],
                             double a[3], double G[3][3]) {
    return add_perturbations(model, ws, t, r, model.srp_k, a, G, nullptr);
}

// rv_evaluate with an explicit SRP coefficient; a_srp as in add_perturbations
int rv_evaluate_ext(const Model& model, Workspace& ws, double t, const double x[6], double srp_k,
                    double xdot[6], double A[6][6], double a_srp[3]);

// Cartesian RHS: xdot (6) and, if A is not null, A = d(xdot)/dx. Return codes as add_perturbations.
int rv_evaluate(const Model& model, Workspace& ws, double t, const double x[6],
                double xdot[6], double A[6][6]);

/**
 * Inputs and outputs of mee_evaluate_ext. All pointers may be null. The outputs are valid only if
 * xdot was computed successfully; they do not need A.
 *
 * a_I enters the projection on the RTN basis, so it reaches xdot, A_B and F (docs/gve_mee.md sec. 5).
 * a_rtn is independent of the state: it reaches xdot and A_B, but not F.
 */
struct MeeExtra {
    double srp_k = 0.0;                    // SRP coefficient to use, instead of Model::srp_k
    const double* a_I = nullptr;           // in: extra inertial acceleration (3), in DU/TU^2
    const double* a_rtn = nullptr;         // in: extra RTN acceleration (3), in DU/TU^2
    double* a_srp_I = nullptr;             // out: SRP acceleration in the inertial frame (3)
    double (*B)[3] = nullptr;              // out: B_G (6x3)
    double (*C)[3] = nullptr;              // out: C = [rhat that nhat] (3x3), C[i][j] = column j, row i
};

// mee_evaluate with the additions of MeeExtra; extra == nullptr is the same as mee_evaluate
int mee_evaluate_ext(const Model& model, Workspace& ws, double t, const double x[6],
                     double xdot[6], double A[6][6], const MeeExtra* extra);

// GVE in MEE: xdot (6) and, if A is not null, A = d(xdot)/dx. Also RHS_EDOM (ws.fault = mee_domain)
// if p <= 0 or w <= 0.
int mee_evaluate(const Model& model, Workspace& ws, double t, const double x[6],
                 double xdot[6], double A[6][6]);

// If any of the n values is not finite: ws.fault = nonfinite and RHS_EDOM; else RHS_SUCCESS
int check_finite(Workspace& ws, const double* v, int n);

// Mass state of the rvm and meem dynamics: RHS_EDOM with ws.fault = mass_domain if m is not > 0 (NaN
// included), or with ws.fault = nonfinite if m = +inf, where the thrust and SRP would just vanish;
// else RHS_SUCCESS, with ws.fault left as is
int check_mass(Workspace& ws, double m);

// GSL-compatible RHS; params is a Context*. 6 states, or 42 for the rhsstm_* variants.
int rhs_rv(double t, const double x[], double xdot[], void* params);
int rhsstm_rv(double t, const double x[], double xdot[], void* params);
int rhs_mee(double t, const double x[], double xdot[], void* params);
int rhsstm_mee(double t, const double x[], double xdot[], void* params);

/**
 * Cartesian + mass RHS: xdot (7) and, if not null, A = d(xdot)/dx and B = d(xdot)/du. The SRP
 * coefficient uses the state mass, srp_coefficient(A, mass_unit_kg * m, rhoS, rhoD, DU, TU).
 *
 * @return as rv_evaluate; also RHS_EDOM if m is not > 0 or is not finite (see check_mass), and
 *         RHS_EDOM (ws.fault = nonfinite) if xdot, A or B is not finite
 */
int rvm_evaluate(const Model& model, const ZohControl& control, Workspace& ws, double t,
                 const double x[7], double xdot[7], double A[7][7], double B[7][4]);

// MEE + mass RHS, as rvm_evaluate; also RHS_EDOM (ws.fault = mee_domain) as mee_evaluate. The mass
// is checked first.
int meem_evaluate(const Model& model, const ZohControl& control, Workspace& ws, double t,
                  const double x[7], double xdot[7], double A[7][7], double B[7][4]);

// GSL-compatible RHS; params is a Context* with control set, which must stay valid during the calls.
// 7 states, or 84 for the rhsstm_* variants: the state, Phi (7x7) and Psi (7x4), both row-major
// (see eoms/augment.h). x and xdot must not overlap.
int rhs_rvm(double t, const double x[], double xdot[], void* params);
int rhsstm_rvm(double t, const double x[], double xdot[], void* params);
int rhs_meem(double t, const double x[], double xdot[], void* params);
int rhsstm_meem(double t, const double x[], double xdot[], void* params);

/**
 * Validate thrust and u (std::invalid_argument prefixed with who) and derive the constants, with
 * c1 = (T_N 1e-3) / (m_bar DU / TU^2) and c2 = T_N / (Isp g0) * TU / m_bar. T_N 1e-3 is in kN, so
 * that kN/kg = km/s^2; g0 stays in m/s^2 as T_N / (Isp g0) is in kg/s.
 */
ZohControl make_control(const SpiceParameters& params, const ThrustOptions& thrust,
                        const std::vector<double>& u, const char* who);

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
