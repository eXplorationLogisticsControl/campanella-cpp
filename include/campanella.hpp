#ifndef CAMPANELLA_HPP
#define CAMPANELLA_HPP

// Public C++ interface of campanella. Only standard headers are included here: CSPICE, GSL and
// the internal headers under src/ stay private to the library.

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <functional>
#include <vector>

namespace campanella {

namespace detail {
struct Model;
struct Workspace;
struct Access;
}  // namespace detail

// CSPICE's default error action is to print a traceback and exit the process.
// RETURN leaves the error flag set so callers can turn it into an exception.
void init_spice();

// Load a kernel into this library's CSPICE. spiceypy.furnsh does not.
// Calls init_spice. Kernels are not loaded until this is called.
void furnsh(const std::string& path);

/**
 * Optional inputs of SpiceParameters. Set the fields by name, then pass the struct.
 *
 * @param naif_frame inertial frame of integration, e.g. "J2000"
 * @param GMs gravitational parameters in km^3/s^2, one per NAIF id; empty means read "GM" of
 *        each body from the loaded kernels (e.g. gm_de440.tpc)
 * @param DU distance unit in km; with use_canonical_scales, VU = sqrt(GMs[0] / DU) and TU = DU / VU
 * @param use_canonical_scales if false, or if DU is not given, DU = TU = VU = 1 (km, s, km/s)
 * @param abcorr aberration correction of the ephemerides; only "NONE" is accepted
 * @param include_srp solar radiation pressure; requires the Sun (NAIF id 10) among the NAIF ids
 * @param srp_A area of the spacecraft, in m^2
 * @param srp_m mass of the spacecraft, in kg; the rvm and meem dynamics use the state mass instead,
 *        but it is validated all the same: must be > 0 if include_srp
 * @param srp_rhoS specular reflection coefficient
 * @param srp_rhoD diffuse reflection coefficient
 * @param sh_file central-body gravity field (PDS SHADR .tab or ICGEM .gfc); "" means none
 * @param sh_nmax truncation degree of the gravity field (>= 2 if sh_file is given)
 * @param sh_mmax truncation order; -1 means min(sh_nmax, order of the file)
 * @param sh_frame body-fixed frame of the gravity field, e.g. "MOON_PA", "ITRF93"
 * @param sh_rho_min smallest |r|/R at which the field is evaluated, in (0, 1]; below it the
 *        propagation throws std::domain_error
 */
struct SpiceOptions {
    std::string naif_frame = "J2000";
    std::vector<double> GMs;
    std::optional<double> DU;
    bool use_canonical_scales = true;
    std::string abcorr = "NONE";
    bool include_srp = false;
    double srp_A = 0.0;
    double srp_m = 0.0;
    double srp_rhoS = 0.0;
    double srp_rhoD = 0.0;
    std::string sh_file;
    int sh_nmax = 0;
    int sh_mmax = -1;
    std::string sh_frame;
    double sh_rho_min = 1.0;
};

/**
 * Dynamics model: central body, third bodies, SRP and spherical harmonics, and the units.
 * Immutable after construction. Copies share the model and are otherwise independent; a move is
 * a copy.
 *
 * Not thread-safe: CSPICE is not, and the eom_* and eval_* functions use the scratch space of the
 * object. Use one thread for SPICE-dependent calls.
 *
 * States, times and outputs are in DU, DU/TU and TU; the ephemerides are queried at
 * et = et0 + t * TU.
 *
 * @param et0 reference epoch, in TDB seconds past J2000
 * @param naif_ids NAIF names or numeric ids, e.g. {"MOON", "EARTH", "10"}; the first one is the
 *        central body, the others are third bodies
 * @param options see SpiceOptions
 * @throws std::invalid_argument on bad input, std::runtime_error with NAIF's message on a SPICE error
 */
class SpiceParameters {
public:
    SpiceParameters(double et0, std::vector<std::string> naif_ids, SpiceOptions options = {});
    SpiceParameters(const SpiceParameters& other);
    SpiceParameters& operator=(const SpiceParameters& other);
    ~SpiceParameters();

    double et0() const;
    const std::vector<std::string>& naif_ids() const;   // as given
    const std::vector<int>& naif_codes() const;         // resolved NAIF codes; [0] is the central body
    const std::string& naif_frame() const;
    const std::vector<double>& GMs() const;             // km^3/s^2
    const std::vector<double>& mus() const;             // GMs * TU^2 / DU^3
    double DU() const;                                  // km
    double TU() const;                                  // s
    double VU() const;                                  // km/s
    bool include_srp() const;
    bool include_spherical_harmonics() const;
    const SpiceOptions& options() const;                // inputs, with GMs filled in

private:
    friend struct detail::Access;
    std::shared_ptr<const detail::Model> model_;
    std::unique_ptr<detail::Workspace> ws_;             // scratch space of the eom_* functions
};

/**
 * Integration options.
 *
 * @param atol absolute error tolerance. The rvm and meem solvers require atol > 0, a deliberate restriction
 *        (see docs/zoh_mass_dynamics.md, Numerical guidance); rv and mee accept 0
 * @param rtol relative error tolerance
 * @param t_eval output times, in TU, strictly monotonic in the direction of integration and
 *        within tspan; empty means every accepted step (including tspan[0] and tspan[1])
 * @param max_steps largest number of accepted steps of one integration (rvm and meem solvers only,
 *        ignored by rv and mee): reaching it with time left throws std::runtime_error, because a
 *        step size collapsing onto a singularity (p -> 0 under thrust) may never fail by itself.
 *        Counted over the whole of tspan, with or without t_eval; per segment in propagate_zoh_*.
 *        0 disables the limit, and a negative value throws std::invalid_argument. It bounds steps, not time:
 *        with the default a stall ends in about a minute with third bodies and SRP, but takes much longer
 *        with high-degree spherical harmonics, up to hours at degree 200 (docs/zoh_mass_dynamics.md). It
 *        records 1.4 GB with sensitivities if t_eval is empty. At default tolerances, low-thrust arcs from low Earth and lunar orbit took up to
 *        1.4e6 steps over 3 years, and with sensitivities 5.9e5 over 1 year; canonical units took fewer.
 * @param poll called between steps of the rvm and meem solvers, every 1024 steps and before the
 *        first, unless empty; may throw to stop the integration, outside any GSL call.
 *        Ignored by rv and mee.
 */
struct IvpOptions {
    double atol = 1e-12;
    double rtol = 1e-14;
    std::vector<double> t_eval;
    long long max_steps = 2000000;
    std::function<void()> poll;
};

/**
 * Solution of an initial value problem.
 *
 * @param t times, in TU (N)
 * @param x states, N x nx row-major: Cartesian [DU, DU/TU], or MEE [p in DU, f, g, h, k, L]
 *        with L not wrapped; followed by the mass, in mass units, for the rvm and meem solvers
 * @param stm state-transition matrices Phi(t, t0), N x nx^2, each nx x nx row-major: element (i, j)
 *        of row k is stm[nx^2 k + nx i + j]; empty for solve_ivp_*
 * @param stm_u control sensitivities Psi(t) = dx(t)/du, N x (nx nu), each nx x nu row-major: element
 *        (i, j) of row k is stm_u[nx nu k + nu i + j]; empty unless solve_ivpstm_rvm or solve_ivpstm_meem
 * @param nx number of states: 6, or 7 for the rvm and meem solvers
 * @param nu number of controls: 0, or 4 for the rvm and meem solvers
 */
struct IvpResult {
    std::vector<double> t;
    std::vector<double> x;
    std::vector<double> stm;
    std::vector<double> stm_u;
    std::size_t nx = 6;
    std::size_t nu = 0;
    std::size_t size() const { return t.size(); }
};

/**
 * Propagate a Cartesian state [x, y, z, vx, vy, vz].
 *
 * @param params dynamics model
 * @param tspan {t0, tf}, in TU since et0; tf < t0 integrates backward, tf == t0 returns one row
 * @param x0 initial state (6)
 * @param options tolerances and output times
 * @throws std::invalid_argument on bad input; std::domain_error if the trajectory goes below
 *         sh_rho_min * R; std::runtime_error with NAIF's message on a SPICE error, or if the
 *         integration fails. A partial trajectory is never returned.
 */
IvpResult solve_ivp_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                       const std::vector<double>& x0, const IvpOptions& options = {});

/**
 * Propagate modified equinoctial elements [p, f, g, h, k, L] (prograde, I = +1) with the Gauss
 * variational equations. The dynamics are the same as solve_ivp_rv.
 * As solve_ivp_rv; also throws std::domain_error if p <= 0 or 1 + f cos L + g sin L <= 0 persists.
 */
IvpResult solve_ivp_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                        const std::vector<double>& x0, const IvpOptions& options = {});

// As solve_ivp_rv, also integrating the STM from Phi(t0, t0) = I. x0 has 6 elements.
IvpResult solve_ivpstm_rv(const SpiceParameters& params, const std::vector<double>& tspan,
                          const std::vector<double>& x0, const IvpOptions& options = {});

// As solve_ivp_mee, also integrating the STM from Phi(t0, t0) = I. x0 has 6 elements.
IvpResult solve_ivpstm_mee(const SpiceParameters& params, const std::vector<double>& tspan,
                           const std::vector<double>& x0, const IvpOptions& options = {});

/**
 * Equations of motion with the signature of gsl_odeiv2_system::function.
 * params points to a SpiceParameters, whose scratch space is written.
 * eom_* take 6 states; eomstm_* take 42: the state, then Phi row-major, with Phi_dot = A Phi.
 *
 * @return 0 (GSL_SUCCESS); GSL_EDOM (1) for a state out of the domain or a non-finite derivative
 *         (GSL retries with a smaller step); GSL_EBADFUNC (9) on a SPICE error (GSL stops).
 *         eom_error describes the last failure.
 */
int eom_rv(double t, const double x[], double dxdt[], void* params);
int eom_mee(double t, const double x[], double dxdt[], void* params);
int eomstm_rv(double t, const double x[], double dxdt[], void* params);
int eomstm_mee(double t, const double x[], double dxdt[], void* params);

// Why the last eom_* or eval_* call on params failed (with NAIF's message for a SPICE error);
// "" if it succeeded. For eom_rvm and the like, pass the SpiceParameters of the ControlledParameters.
std::string eom_error(const SpiceParameters& params);

/**
 * Time derivative and Jacobian at a single time and state.
 *
 * @param xdot d(x)/dt (6)
 * @param A d(xdot)/dx, 6x6 row-major
 */
struct EomEval {
    std::array<double, 6> xdot;
    std::array<double, 36> A;
};

// Evaluate eom_rv and its Jacobian at t (TU) and Cartesian x. Throws as solve_ivp_rv.
EomEval eval_rv(const SpiceParameters& params, double t, const std::vector<double>& x);

// Evaluate eom_mee and its Jacobian at t (TU) and MEE x. Throws std::domain_error if p <= 0 or w <= 0.
EomEval eval_mee(const SpiceParameters& params, double t, const std::vector<double>& x);

/**
 * Frame of the throttle vector tau of the meem dynamics. rtn (the default of ThrustOptions): the
 * radial, transverse and normal directions of the current state, so a constant tau rotates in
 * inertial space. inertial: naif_frame, as rvm.
 */
enum class ThrustFrame { rtn, inertial };

/**
 * Propulsion system of the mass-augmented dynamics (rvm, meem), held constant over a tspan.
 *
 * The control is u = [tau, Gamma], both dimensionless: tau (3) is the throttle vector, so that the
 * thrust is T_max tau, and Gamma scales the mass flow. The state has the mass m, in units of
 * mass_unit_kg (m_bar), so m = 1 means m_bar kilograms:
 *
 *   thrust acceleration = (c1 / m) tau, in DU/TU^2,    mdot = -c2 Gamma, in mass units per TU
 *   c1 = (T_max [kN]) / (m_bar DU / TU^2),  the thrust acceleration at m = 1, in DU/TU^2
 *   c2 = (T_max [N]) / (Isp g0) * TU / m_bar,  in mass units per TU
 *
 * The dynamics treat Gamma as independent of tau, and do not tie one to the other: the solver of the
 * optimal control problem is responsible for Gamma = |tau|. Neither Gamma < 0 nor |tau| > Gamma is
 * rejected.
 *
 * SRP uses the state mass, m_bar m. SpiceOptions::srp_m is ignored, but still required to be > 0 by
 * SpiceParameters if include_srp.
 *
 * @param thrust_N maximum thrust T_max, in N
 * @param isp_s specific impulse, in s
 * @param mass_unit_kg mass unit m_bar, in kg
 * @param g0 standard gravity, in m/s^2
 * @param mee_control_frame frame of tau for meem, RTN by default; rvm is always in naif_frame
 */
struct ThrustOptions {
    double thrust_N = 0.0;
    double isp_s = 0.0;
    double mass_unit_kg = 0.0;
    double g0 = 9.80665;
    ThrustFrame mee_control_frame = ThrustFrame::rtn;
};

/**
 * Time derivative and Jacobians of the mass-augmented dynamics at a single time, state and control.
 *
 * @param xdot d(x)/dt (7)
 * @param A d(xdot)/dx, 7x7 row-major: element (i, j) is A[7 i + j]; column 6 is the mass
 * @param B d(xdot)/du, 7x4 row-major: element (i, j) is B[4 i + j]; columns tau (3), Gamma
 */
struct EomEvalControl {
    std::array<double, 7> xdot;
    std::array<double, 49> A;
    std::array<double, 28> B;
};

/**
 * Linearization x_{k+1} = Phi_k x_k + Psi_k u_k + c_k of N - 1 zero-order-hold segments.
 *
 * @param x_end state at the end of each segment, (N - 1) x 7 row-major
 * @param Phi state-transition matrices, (N - 1) x 49, each 7x7 row-major
 * @param Psi control sensitivities, (N - 1) x 28, each 7x4 row-major
 * @param c affine terms x_end_k - Phi_k x_ref_k - Psi_k u_ref_k, (N - 1) x 7 row-major
 * @param defects x_ref_{k+1} - x_end_k, (N - 1) x 7 row-major
 */
struct ZohLinearization {
    std::vector<double> x_end;
    std::vector<double> Phi;
    std::vector<double> Psi;
    std::vector<double> c;
    std::vector<double> defects;
};

/**
 * Parameters of eom_rvm, eom_meem, eomstm_rvm and eomstm_meem: a dynamics model, a propulsion system
 * and a constant control, with the constants c1 and c2 of ThrustOptions.
 *
 * Non-owning: keeps a pointer to params, which must outlive this object, and so a temporary
 * SpiceParameters does not compile. thrust and u are copied. Copies point to the same params.
 *
 * Not thread-safe, like SpiceParameters: the eom_* functions write the scratch space of params, also
 * through a const ControlledParameters. Use one thread for all the objects sharing a params.
 *
 * @param params dynamics model
 * @param thrust propulsion system
 * @param u control [tau, Gamma] (4), dimensionless
 * @throws std::invalid_argument if u does not have 4 finite values, if thrust_N, isp_s, mass_unit_kg
 *         or g0 is not finite and > 0, if mee_control_frame is not a ThrustFrame, or if c1 or c2
 *         overflows or underflows
 */
class ControlledParameters {
public:
    ControlledParameters(const SpiceParameters& params, const ThrustOptions& thrust,
                         const std::vector<double>& u);
    ControlledParameters(const SpiceParameters&& params, const ThrustOptions& thrust,
                         const std::vector<double>& u) = delete;

    const SpiceParameters& params() const { return *params_; }
    const ThrustOptions& thrust() const { return thrust_; }
    const std::array<double, 4>& u() const { return u_; }
    double c1() const { return c1_; }                   // thrust acceleration at m = 1, DU/TU^2
    double c2() const { return c2_; }                   // mass flow at Gamma = 1, mass units per TU

private:
    const SpiceParameters* params_;
    ThrustOptions thrust_;
    std::array<double, 4> u_;
    double c1_;
    double c2_;
};

/**
 * Propagate a Cartesian state and mass [x, y, z, vx, vy, vz, m], in DU, DU/TU and mass units, with a
 * constant control: vdot = a + (c1 / m) tau with tau in naif_frame, mdot = -c2 Gamma. See ThrustOptions
 * for the units and for Gamma, and for SRP, which uses the state mass and ignores srp_m. The result
 * has nx = 7 and nu = 4. Not thread-safe (see SpiceParameters).
 *
 * @param params dynamics model
 * @param thrust propulsion system
 * @param tspan {t0, tf}, in TU since et0; tf < t0 integrates backward, tf == t0 returns one row
 * @param x0 initial state (7)
 * @param u control [tau, Gamma] (4), dimensionless, held over tspan
 * @param options tolerances and output times
 * @throws as solve_ivp_rv; also std::invalid_argument as ControlledParameters, and if x0 is not
 *         finite; std::domain_error if m0 is not > 0, including over a zero-length tspan, or if the
 *         closed form m0 - c2 Gamma (t - t0) is not above 64 eps max(m0, |c2 Gamma (t - t0)|), so that the
 *         depletion time itself is refused, at the last output time (tf, or the last of t_eval), before
 *         integrating and whatever the thrust; the message has the depletion time. The factor is a
 *         heuristic: results within about 1e-12 of depletion are not meaningful, and may be returned;
 *         std::runtime_error if options.max_steps is reached, with the time reached;
 *         std::invalid_argument if options.max_steps < 0 or options.atol is not > 0
 */
IvpResult solve_ivp_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                        const std::vector<double>& tspan, const std::vector<double>& x0,
                        const std::vector<double>& u, const IvpOptions& options = {});

// As solve_ivp_rvm, also integrating Phi from I and Psi from 0, with Phi_dot = A Phi, Psi_dot = A Psi + B.
// See IvpResult for the layout of stm (7x7) and stm_u (7x4), both row-major.
IvpResult solve_ivpstm_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                           const std::vector<double>& tspan, const std::vector<double>& x0,
                           const std::vector<double>& u, const IvpOptions& options = {});

/**
 * Propagate MEE and mass [p, f, g, h, k, L, m], in DU, mass units and otherwise dimensionless, with a
 * constant control, with tau in thrust.mee_control_frame (RTN by default). As solve_ivp_rvm; also
 * throws std::domain_error as solve_ivp_mee.
 */
IvpResult solve_ivp_meem(const SpiceParameters& params, const ThrustOptions& thrust,
                         const std::vector<double>& tspan, const std::vector<double>& x0,
                         const std::vector<double>& u, const IvpOptions& options = {});

// As solve_ivp_meem, also integrating Phi from I and Psi from 0, with Phi_dot = A Phi, Psi_dot = A Psi + B.
// See IvpResult for the layout of stm (7x7) and stm_u (7x4), both row-major.
IvpResult solve_ivpstm_meem(const SpiceParameters& params, const ThrustOptions& thrust,
                            const std::vector<double>& tspan, const std::vector<double>& x0,
                            const std::vector<double>& u, const IvpOptions& options = {});

/**
 * Mass-augmented equations of motion with the signature of gsl_odeiv2_system::function.
 * params points to a ControlledParameters, whose params has its scratch space written, so one thread
 * only. x and dxdt must not overlap. eom_* take 7 states; eomstm_* take 84: the state, then Phi (7x7)
 * and Psi (7x4), both row-major, with Phi_dot = A Phi and Psi_dot = A Psi + B.
 *
 * @return as eom_rv. A mass m that is not > 0 or not finite is out of the domain (GSL_EDOM). So is a
 *         non-finite derivative of Phi or Psi in eomstm_*, unlike eomstm_rv and eomstm_mee, which only
 *         check the state derivative and A. eom_error on cp.params() describes the last failure.
 */
int eom_rvm(double t, const double x[], double dxdt[], void* params);
int eom_meem(double t, const double x[], double dxdt[], void* params);
int eomstm_rvm(double t, const double x[], double dxdt[], void* params);
int eomstm_meem(double t, const double x[], double dxdt[], void* params);

// Evaluate eom_rvm and its Jacobians at t (TU), x (7) and u (4). Throws as solve_ivp_rvm, with
// std::domain_error if m is not > 0 or not finite. Uses the scratch space of params: one thread only.
EomEvalControl eval_rvm(const SpiceParameters& params, const ThrustOptions& thrust, double t,
                        const std::vector<double>& x, const std::vector<double>& u);

// Evaluate eom_meem and its Jacobians at t (TU), x (7) and u (4). Throws as solve_ivp_meem, with
// std::domain_error if m is not > 0 or not finite. Uses the scratch space of params: one thread only.
EomEvalControl eval_meem(const SpiceParameters& params, const ThrustOptions& thrust, double t,
                         const std::vector<double>& x, const std::vector<double>& u);

/**
 * Propagate each segment of a zero-order-hold reference from its own reference state, with one
 * integration per segment, and linearize it.
 *
 * times may decrease, for a backward integration: u_ref[k] is then held from times[k] down to
 * times[k + 1], and Phi and Psi are still the sensitivities of x(times[k + 1]) to x(times[k]) and to
 * u_ref[k]. The last row of x_ref only enters the last defect, so its mass is not required to be > 0.
 * L is not wrapped: a caller wrapping L in x_ref sees defects of multiples of 2 pi.
 *
 * @param params dynamics model
 * @param thrust propulsion system
 * @param times segment boundaries, in TU (N), strictly monotonic
 * @param x_ref reference states, N x 7 row-major
 * @param u_ref controls, (N - 1) x 4 row-major
 * @param options tolerances and max_steps, per segment; t_eval must be empty. Only the end of each
 *        segment is recorded, so memory does not grow with the steps
 * @throws as solve_ivpstm_rvm, as std::domain_error, std::invalid_argument or std::runtime_error, with
 *         "<function>: segment k of n (t = [a, b] TU): " in front of the message of a failed segment; a class
 *         derived from these, which only options.poll can throw, is rethrown as its base type. Any other
 *         exception from options.poll passes through unchanged and without the segment.
 *         std::invalid_argument on inconsistent sizes, non-finite or non-monotonic input, or a non-empty t_eval
 */
ZohLinearization propagate_zoh_rvm(const SpiceParameters& params, const ThrustOptions& thrust,
                                   const std::vector<double>& times, const std::vector<double>& x_ref,
                                   const std::vector<double>& u_ref, const IvpOptions& options = {});

// As propagate_zoh_rvm, in MEE + mass. Throws as solve_ivpstm_meem.
ZohLinearization propagate_zoh_meem(const SpiceParameters& params, const ThrustOptions& thrust,
                                    const std::vector<double>& times, const std::vector<double>& x_ref,
                                    const std::vector<double>& u_ref, const IvpOptions& options = {});

/**
 * Spherical-harmonics potential, acceleration and gradient in the body-fixed frame (degrees 2..nmax).
 *
 * @param path gravity field file (.tab or .gfc)
 * @param nmax truncation degree
 * @param r_bf body-fixed position
 * @param mmax truncation order; -1 means min(nmax, order of the file)
 * @param mu gravitational parameter; 0 means the GM of the file, in km^3/s^2
 * @param R reference radius, in the unit of r_bf; 0 means the radius of the file, in km.
 *        Give both mu and R, or neither (std::invalid_argument otherwise)
 * @param rho_min smallest |r_bf|/R evaluated, in (0, 1]
 * @return (U, a, G) with G = d(a)/d(r_bf), 3x3 row-major; throws std::domain_error if |r_bf| < rho_min R
 */
std::tuple<double, std::vector<double>, std::vector<double>> sh_accel_bf(
    const std::string& path, int nmax, const std::vector<double>& r_bf,
    int mmax = -1, double mu = 0.0, double R = 0.0, double rho_min = 1.0);

/**
 * Spherical-harmonics acceleration and gradient in an inertial frame, as used by the EOMs.
 *
 * @param path gravity field file (.tab or .gfc)
 * @param nmax truncation degree
 * @param naif_frame inertial frame of r, e.g. "J2000"
 * @param sh_frame body-fixed frame of the gravity field, e.g. "MOON_PA", "ITRF93"
 * @param et epoch, in ephemeris seconds
 * @param r position, in DU
 * @param DU distance unit, km
 * @param TU time unit, s
 * @param mmax truncation order; -1 means min(nmax, order of the file)
 * @param rho_min smallest |r|/R evaluated, in (0, 1]
 * @return (a, G): a in DU/TU^2, G = d(a)/d(r) in 1/TU^2, 3x3 row-major;
 *         throws std::runtime_error on SPICE error, std::domain_error if |r| < rho_min R
 */
std::pair<std::vector<double>, std::vector<double>> sh_accel_inertial(
    const std::string& path, int nmax, const std::string& naif_frame, const std::string& sh_frame,
    double et, const std::vector<double>& r, double DU = 1.0, double TU = 1.0, int mmax = -1,
    double rho_min = 1.0);

/**
 * Converts modified equinoctial elements (MEE) to a Cartesian state.
 *
 * @param mee [p, f, g, h, k, L], prograde (I = +1); L may be any real number
 * @param mu gravitational parameter
 * @return [x, y, z, vx, vy, vz]; throws std::domain_error if p <= 0 or 1 + f cos L + g sin L <= 0
 */
std::vector<double> mee2cart(const std::vector<double>& mee, double mu);

/**
 * Converts a Cartesian state to modified equinoctial elements (MEE).
 *
 * @param rv [x, y, z, vx, vy, vz]
 * @param mu gravitational parameter
 * @return [p, f, g, h, k, L] with L in (-pi, pi]; throws std::domain_error for zero
 *         angular momentum or inclination within ~1.4e-6 rad of 180 deg
 */
std::vector<double> cart2mee(const std::vector<double>& rv, double mu);

/**
 * Jacobian d(x, y, z, vx, vy, vz)/d(p, f, g, h, k, L) of mee2cart.
 *
 * @param mee [p, f, g, h, k, L]
 * @param mu gravitational parameter
 * @return 6x6 matrix, row-major (36 elements)
 */
std::vector<double> mee2cart_jacobian(const std::vector<double>& mee, double mu);

}  // namespace campanella

#endif  // CAMPANELLA_HPP
