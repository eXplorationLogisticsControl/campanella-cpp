#ifndef CAMPANELLA_HPP
#define CAMPANELLA_HPP

#include <string>
#include <tuple>
#include <vector>

#include "propagate_twobody.h"
#include "propagate_Nbody.h"
#include "propagate_NbodySRP.h"
#include "propagate_stm_Nbody.h"

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
        return ::propagate_twobody(tspan, y0, GM, reltol, abstol);
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
struct PropagatorNBody {
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double LU;
    double TU;
    double reltol;
    double abstol;

    PropagatorNBody(std::vector<double> GM,
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
        return ::propagate_Nbody(
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
        return ::propagate_stm_Nbody(
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
struct PropagatorNBodySRP {
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

    PropagatorNBodySRP(std::vector<double> GM,
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
        return ::propagate_NbodySRP(
            et0, tspan, y0, GM, BODIES, frame, A, m, rhoS, rhoD, LU, TU, reltol, abstol);
    }
};

}  // namespace campanella

#endif  // CAMPANELLA_HPP
