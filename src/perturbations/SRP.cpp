/**
 * Solar Radiation Pressure (SRP) accelerations
 */

#include <cmath>
#include "SRP.h"

namespace {

// SRP constants
const double SRP_P0 = 1367;              // solar flux at 1 AU [W/m^2]
const double SRP_C = 299792458;          // speed of light [m/s]
const double SRP_AU = 149597870.7;       // astronomical unit [km]

}  // namespace

/**
 * SRP coefficient K, such that a_SRP = K d / |d|^3 with d the position from the Sun to the spacecraft
 *
 * At 1 AU the acceleration is (P0 / c) Cr A / m, in m/s^2; K converts it to DU/TU^2 and moves the
 * 1/|d|^2 scaling into DU.
 *
 * @param A surface area of spacecraft, in m^2
 * @param m mass of spacecraft, in kg
 * @param rhoS specular reflection coefficient
 * @param rhoD diffuse reflection coefficient
 * @param DU distance unit, in km
 * @param TU time unit, in s
 * @return K, in DU^3/TU^2
 */
double srp_coefficient(double A, double m, double rhoS, double rhoD, double DU, double TU) {
    double Cr = 1 + rhoS + 5 * rhoD / 3;
    double au = SRP_AU / DU;
    return SRP_P0/SRP_C * Cr * A/m * au*au / (1e3 * DU/(TU*TU));
}


/**
 * Gradient of the SRP acceleration a = kappa d with respect to spacecraft position.
 *
 * With d = r - s, R = |d| and kappa ~ 1/R^3, the gradient is kappa (I - 3 d d^T / R^2):
 * a repulsive inverse-square field.
 *
 * @param d position of the spacecraft w.r.t. the Sun
 * @param kappa SRP factor at d
 * @param GG 3-by-3 gradient matrix to be computed
 */
void srp_gradient(const double d[3], double kappa, double GG[3][3]) {
    double R2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            GG[i][j] = kappa * (delta_ij - 3 * d[i] * d[j] / R2);
        }
    }
}
