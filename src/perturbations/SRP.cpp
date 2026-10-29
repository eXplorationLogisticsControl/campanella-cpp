/**
 * Solar Radiation Pressure (SRP) accelertations
*/


#include <cmath>
#include <vector>
#include <SpiceUsr.h>          // access SPICE files

namespace {

//SRP constants (should these be inputs?)
const double SRP_P0 = 1367;       // [w/m^2]  [W/LU^2]
const double SRP_C = 299792458;   //[m/s]  [LU/TU]

/**
 * SRP magnitude factor kappa, such that a_SRP = kappa * (r - s)
 *
 * @param R0 distance from reference body to sun, in LU
 * @param R distance from sun to spacecraft, in LU
 * @param A surface area of spacecraft
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param TU divisor from s to integrator's time scale
 * @param LU divisor from km to integrator's distance scale
 */
double compute_srp_magnitude(double R0, double R,
                             double A, double m, double rhoS, double rhoD,
                             double TU, double LU) {
    // SRP equations
    double P = SRP_P0*(R0/R)*(R0/R)/SRP_C;
    double Cr = 1 + rhoS + 5 * rhoD / 3;
    return P*Cr*A/R/m / (1e3 * LU/(TU*TU));
}

}  // namespace

/**
 * Compute SRP perturbation 
 * 
 * @param x spacecraft x-position 
 * @param y spacecraft y-position
 * @param z spacecraft z-position
 * @param et epoch at which ephemeris of perturbing body is queried, in ephemeris sec
 * @param frame_cspice NAIF name of frame of integration
 * @param ref_body_name NAIF name of central body of integration
 * @param A surface area of spacecraft
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param TU divisor from s to integrator's time scale
 * @param LU divisor from km to integrator's distance scale
 * @return {ax, ay, az, xs, ys, zs}: SRP acceleration, then position of the sun
 *         with respect to the reference body, divided by LU
*/
std::vector<double> compute_SRP(double x,
                                double y,
                                double z,
                                SpiceDouble et,
                                ConstSpiceChar * frame_cspice,
                                ConstSpiceChar * ref_body_name,
                                double A,
                                double m,
                                double rhoS,
                                double rhoD,
                                double TU,
                                double LU) {
    SpiceDouble ri [3];
    SpiceDouble ti;

    //A = A/LU/LU;

    //Should this be an input? And is the solar system barycenter the correct Spice name for the sun?
    ConstSpiceChar * sun;
    sun = "SOLAR SYSTEM BARYCENTER";
    
    // Position from reference body to sun
    spkpos_c(sun, et, frame_cspice, "NONE", ref_body_name, ri, &ti);

    // Position from reference body to sun, converted to length scale
    double xi = ri[0] / LU;
    double yi = ri[1] / LU;
    double zi = ri[2] / LU;
    double R0 = sqrt(xi*xi + yi*yi + zi*zi);
    // Position from Sun to spacecraft (Sun to reference body + reference body to spacecraft)
    double xt = x - xi;
    double yt = y - yi;
    double zt = z - zi;
    double R = sqrt(xt*xt + yt*yt + zt*zt);

    // SRP effects from sun on spacecraft
    double srp_magnitude = compute_srp_magnitude(R0, R, A, m, rhoS, rhoD, TU, LU);
    double ax_SRP = srp_magnitude * xt;  // P*Cr*A*xt/R/m;
    double ay_SRP = srp_magnitude * yt;  // P*Cr*A*yt/R/m;
    double az_SRP = srp_magnitude * zt;  // P*Cr*A*zt/R/m;

    // concatenate into output vector
    std::vector<double> result = {ax_SRP, ay_SRP, az_SRP, xi, yi, zi};
    return result;
}


/**
 * Gradient of the SRP acceleration of compute_SRP with respect to spacecraft position.
 *
 * With d = r - s, R = |d| and kappa = srp_magnitude (a_SRP = kappa d, kappa ~ 1/R^3),
 * the gradient is kappa (I - 3 d d^T / R^2). This is a repulsive inverse-square field.
 *
 * @param rvec The position vector of spacecraft.
 * @param svec The position vector of the sun with respect to the reference body,
 *             divided by LU (indices 3..5 returned by compute_SRP).
 * @param A surface area of spacecraft
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param TU divisor from s to integrator's time scale
 * @param LU divisor from km to integrator's distance scale
 * @param GG 3-by-3 gradient matrix to be computed.
 */
void srp_gradient(const double rvec[],
                  const double svec[],
                  double A,
                  double m,
                  double rhoS,
                  double rhoD,
                  double TU,
                  double LU,
                  double GG[3][3]) {
    double d[3] = {rvec[0] - svec[0], rvec[1] - svec[1], rvec[2] - svec[2]};
    double R0 = sqrt(svec[0]*svec[0] + svec[1]*svec[1] + svec[2]*svec[2]);
    double R2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    double R = sqrt(R2);
    double kappa = compute_srp_magnitude(R0, R, A, m, rhoS, rhoD, TU, LU);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            GG[i][j] = kappa * (delta_ij - 3 * d[i] * d[j] / R2);
        }
    }
}
