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
 */
struct PropagatorNbody {
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double LU;
    double TU;
    double reltol;
    double abstol;

    PropagatorNbody(std::vector<double> GM,
                    std::vector<std::string> BODIES,
                    std::string frame,
                    double LU,
                    double TU,
                    double reltol = 1e-14,
                    double abstol = 1e-12)
        : GM(GM), BODIES(BODIES), frame(frame), LU(LU), TU(TU), reltol(reltol), abstol(abstol) {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_nbody(
            et0, tspan, y0, GM, BODIES, frame, LU, TU, reltol, abstol);
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
            et0, tspan, y0, GM, BODIES, frame, LU, TU, reltol, abstol);
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
                       double abstol = 1e-12)
        : GM(GM), BODIES(BODIES), frame(frame), A(A), m(m), rhoS(rhoS), rhoD(rhoD),
          LU(LU), TU(TU), reltol(reltol), abstol(abstol) {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_nbody_srp(
            et0, tspan, y0, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, reltol, abstol);
    }
};


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
                  double abstol = 1e-12)
        : GM(GM), BODIES(BODIES), frame(frame), A(A), m(m), rhoS(rhoS), rhoD(rhoD),
          LU(LU), TU(TU), reltol(reltol), abstol(abstol) {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return ::solve_ivp_gve(
            et0, tspan, y0, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, reltol, abstol);
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
            et0, tspan, y0, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, reltol, abstol);
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
        return ::eval_jacobian_gve(et0, t, x, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU);
    }
};

}  // namespace campanella

#endif  // CAMPANELLA_HPP
