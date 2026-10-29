#ifndef CAMPANELLA_HPP
#define CAMPANELLA_HPP

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "propagate/propagate_twobody.h"
#include "propagate/propagate_Nbody.h"
#include "propagate/propagate_NbodySRP.h"
#include "propagate/propagate_stm_Nbody.h"
#include "propagate/propagate_gve.h"
#include "propagate/propagate_stm_gve.h"
#include "eoms/mee.h"
#include "perturbations/spherical_harmonics.h"

namespace campanella {

// CSPICE's default error action is to print a traceback and exit the process.
// RETURN leaves the error flag set so callers can turn it into an exception.
void init_spice();

// Load a kernel into this library's CSPICE. spiceypy.furnsh does not.
// Calls init_spice. Kernels are not loaded until this is called.
void furnsh(const std::string& path);

/**
 * Propagates a spacecraft using two-body equations of motion.
 *
 * @param GM gravitational parameter
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 */
struct PropagatorTwoBody {
    double GM;
    double reltol;
    double abstol;

    PropagatorTwoBody(double GM, double reltol = 1e-14, double abstol = 1e-12)
        : GM(GM), reltol(reltol), abstol(abstol) {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_twobody(tspan, y0, GM, reltol, abstol);
    }
};

/**
 * Propagates a spacecraft using N-body equations of motion.
 *
 * @param GM list of gravitational parameters of each body
 * @param BODIES list of NAIF names of each body
 * @param frame inertial reference frame
 * @param LU length unit
 * @param TU time unit
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 * @param sh_file optional central-body gravity field (.tab or .gfc), see docs/spherical_harmonics.md
 * @param sh_nmax truncation degree of the gravity field (>= 2 if sh_file is given)
 * @param sh_mmax truncation order of the gravity field; -1 means min(sh_nmax, order of the file)
 * @param frame_bf body-fixed frame of the gravity field, e.g. "MOON_PA", "ITRF93"
 * @param sh_rho_min smallest |r|/R at which the field is evaluated, in (0, 1]; below it the
 *        propagation throws std::domain_error
 */
struct PropagatorNbody {
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double LU;
    double TU;
    double reltol;
    double abstol;
    SphericalHarmonicsOptions sh;

    PropagatorNbody(std::vector<double> GM,
                    std::vector<std::string> BODIES,
                    std::string frame,
                    double LU,
                    double TU,
                    double reltol = 1e-14,
                    double abstol = 1e-12,
                    std::string sh_file = "",
                    int sh_nmax = 0,
                    int sh_mmax = -1,
                    std::string frame_bf = "",
                    double sh_rho_min = 1.0)
        : GM(GM), BODIES(BODIES), frame(frame), LU(LU), TU(TU), reltol(reltol), abstol(abstol),
          sh{sh_file, sh_nmax, sh_mmax, frame_bf, sh_rho_min} {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_nbody(
            et0, tspan, y0, GM, BODIES, frame, LU, TU, reltol, abstol, sh);
    }

    std::vector<std::tuple<double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double>> solve_ivp_stm(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_stm_nbody(
            et0, tspan, y0, GM, BODIES, frame, LU, TU, reltol, abstol, sh);
    }
};

/**
 * Propagates a spacecraft using N-body equations of motion with solar radiation pressure.
 *
 * @param GM list of gravitational parameters of each body
 * @param BODIES list of NAIF names of each body
 * @param frame inertial reference frame
 * @param A surface area of spacecraft
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param LU length unit
 * @param TU time unit
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 * @param sh_file, sh_nmax, sh_mmax, frame_bf, sh_rho_min optional central-body gravity field, as PropagatorNbody
 */
struct PropagatorNbodySrp {
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double A;
    double m;
    double rhoS;
    double rhoD;
    double LU;
    double TU;
    double reltol;
    double abstol;
    SphericalHarmonicsOptions sh;

    PropagatorNbodySrp(std::vector<double> GM,
                       std::vector<std::string> BODIES,
                       std::string frame,
                       double A,
                       double m,
                       double rhoS,
                       double rhoD,
                       double LU,
                       double TU,
                       double reltol = 1e-14,
                       double abstol = 1e-12,
                       std::string sh_file = "",
                       int sh_nmax = 0,
                       int sh_mmax = -1,
                       std::string frame_bf = "",
                       double sh_rho_min = 1.0)
        : GM(GM), BODIES(BODIES), frame(frame), A(A), m(m), rhoS(rhoS), rhoD(rhoD),
          LU(LU), TU(TU), reltol(reltol), abstol(abstol),
          sh{sh_file, sh_nmax, sh_mmax, frame_bf, sh_rho_min} {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_nbody_srp(
            et0, tspan, y0, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, reltol, abstol, sh);
    }
};


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
 * @param frame inertial frame of r, e.g. "J2000"
 * @param frame_bf body-fixed frame of the gravity field, e.g. "MOON_PA", "ITRF93"
 * @param et epoch, in ephemeris seconds
 * @param r position, in LU
 * @param LU length unit
 * @param TU time unit
 * @param mmax truncation order; -1 means min(nmax, order of the file)
 * @param rho_min smallest |r|/R evaluated, in (0, 1]
 * @return (a, G): a in LU/TU^2, G = d(a)/d(r) in 1/TU^2, 3x3 row-major;
 *         throws std::runtime_error on SPICE error, std::domain_error if |r| < rho_min R
 */
std::pair<std::vector<double>, std::vector<double>> sh_accel_inertial(
    const std::string& path, int nmax, const std::string& frame, const std::string& frame_bf,
    double et, const std::vector<double>& r, double LU = 1.0, double TU = 1.0, int mmax = -1,
    double rho_min = 1.0);

/**
 * Converts modified equinoctial elements (MEE) to a Cartesian state.
 *
 * @param mee [p, f, g, h, k, L], prograde (I = +1); L may be any real number
 * @param mu gravitational parameter
 * @return [x, y, z, vx, vy, vz]; throws std::domain_error if p <= 0 or 1 + f cos L + g sin L <= 0
 */
inline std::vector<double> mee2cart(const std::vector<double>& mee, double mu) {
    return ::mee2cart(mee, mu);
}

/**
 * Converts a Cartesian state to modified equinoctial elements (MEE).
 *
 * @param rv [x, y, z, vx, vy, vz]
 * @param mu gravitational parameter
 * @return [p, f, g, h, k, L] with L in (-pi, pi]; throws std::domain_error for zero
 *         angular momentum or inclination within ~1.4e-6 rad of 180 deg
 */
inline std::vector<double> cart2mee(const std::vector<double>& rv, double mu) {
    return ::cart2mee(rv, mu);
}

/**
 * Jacobian d(x, y, z, vx, vy, vz)/d(p, f, g, h, k, L) of mee2cart.
 *
 * @param mee [p, f, g, h, k, L]
 * @param mu gravitational parameter
 * @return 6x6 matrix, row-major (36 elements)
 */
inline std::vector<double> mee2cart_jacobian(const std::vector<double>& mee, double mu) {
    return ::mee2cart_jacobian(mee, mu);
}

/**
 * Propagates a spacecraft in modified equinoctial elements (MEE) using the Gauss variational
 * equations, with N-body and solar radiation pressure perturbations.
 * The physics is the same as PropagatorNbodySrp; set A = 0 to remove SRP.
 * States are [p, f, g, h, k, L] (prograde, I = +1; singular as the inclination approaches 180 deg).
 *
 * @param GM list of gravitational parameters of each body; GM[0] is the central body
 * @param BODIES list of NAIF names of each body
 * @param frame inertial reference frame
 * @param A surface area of spacecraft
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param LU length unit
 * @param TU time unit
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 * @param sh_file, sh_nmax, sh_mmax, frame_bf, sh_rho_min optional central-body gravity field, as PropagatorNbody
 */
struct PropagatorGve {
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double A;
    double m;
    double rhoS;
    double rhoD;
    double LU;
    double TU;
    double reltol;
    double abstol;
    SphericalHarmonicsOptions sh;

    PropagatorGve(std::vector<double> GM,
                  std::vector<std::string> BODIES,
                  std::string frame,
                  double A,
                  double m,
                  double rhoS,
                  double rhoD,
                  double LU,
                  double TU,
                  double reltol = 1e-14,
                  double abstol = 1e-12,
                  std::string sh_file = "",
                  int sh_nmax = 0,
                  int sh_mmax = -1,
                  std::string frame_bf = "",
                  double sh_rho_min = 1.0)
        : GM(GM), BODIES(BODIES), frame(frame), A(A), m(m), rhoS(rhoS), rhoD(rhoD),
          LU(LU), TU(TU), reltol(reltol), abstol(abstol),
          sh{sh_file, sh_nmax, sh_mmax, frame_bf, sh_rho_min} {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_gve(
            et0, tspan, y0, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, reltol, abstol, sh);
    }

    std::vector<std::tuple<double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double>> solve_ivp_stm(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_stm_gve(
            et0, tspan, y0, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, reltol, abstol, sh);
    }

    /**
     * Evaluates the equations of motion and their Jacobian at a single time and state.
     *
     * @param et0 initial epoch, in ephemeris seconds past J2000
     * @param t time since et0, in TU
     * @param x MEE state [p, f, g, h, k, L]
     * @return (xdot, A) with A = d(xdot)/dx, 6x6 row-major
     *         throws std::domain_error if p <= 0 or w <= 0, std::runtime_error on SPICE error
     */
    std::pair<std::vector<double>, std::vector<double>> eval_jacobian(
        double et0,
        double t,
        const std::vector<double>& x) const {
        return ::eval_jacobian_gve(et0, t, x, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, sh);
    }
};

}  // namespace campanella

#endif  // CAMPANELLA_HPP
