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
 * @param srp_m mass of the spacecraft, in kg
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
 * @param atol absolute error tolerance
 * @param rtol relative error tolerance
 * @param t_eval output times, in TU, strictly monotonic in the direction of integration and
 *        within tspan; empty means every accepted step (including tspan[0] and tspan[1])
 */
struct IvpOptions {
    double atol = 1e-12;
    double rtol = 1e-14;
    std::vector<double> t_eval;
};

/**
 * Solution of an initial value problem.
 *
 * @param t times, in TU (N)
 * @param x states, N x 6 row-major: Cartesian [DU, DU/TU], or MEE [p in DU, f, g, h, k, L]
 *        with L not wrapped
 * @param stm state-transition matrices Phi(t, t0), N x 36, each 6x6 row-major; empty for solve_ivp_*
 */
struct IvpResult {
    std::vector<double> t;
    std::vector<double> x;
    std::vector<double> stm;
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
// "" if it succeeded
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
