/**
 * Solar Radiation Pressure (SRP) accelerations
 */

#include <cmath>
#include "SRP.h"

namespace {

//SRP constants
const double SRP_P0 = 1367;       // [W/m^2]
const double SRP_C = 299792458;   // [m/s]

}  // namespace

/**
 * SRP magnitude factor kappa, such that a_SRP = kappa * (r - s)
 *
 * @param R0 distance from reference body to sun, in DU
 * @param R distance from sun to spacecraft, in DU
 * @param A surface area of spacecraft, in m^2
 * @param m mass of spacecraft, in kg
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param TU time unit, in s
 * @param DU distance unit, in km
 */
double srp_legacy_kappa(double R0, double R, double A, double m, double rhoS, double rhoD,
                        double TU, double DU) {
    double P = SRP_P0*(R0/R)*(R0/R)/SRP_C;
    double Cr = 1 + rhoS + 5 * rhoD / 3;
    return P*Cr*A/R/m / (1e3 * DU/(TU*TU));
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
