/**
 * Perurbative accelertations
*/


#include <cmath>
#include <vector>
#include <SpiceUsr.h>          // access SPICE files

/**
 * Compute third-body perturbation using Battin's F(q) function
 * 
 * @param x spacecraft x-position 
 * @param y spacecraft y-position
 * @param z spacecraft z-position
 * @param GM_i gravitational constant of perturbing body, scaled for LU, TU
 * @param third_body_name NAIF name of perturbing body
 * @param et epoch at which ephemeris of perturbing body is queried, in ephemeris sec
 * @param frame_cspice NAIF name of frame of integration
 * @param ref_body_name NAIF name of central body of integration
 * @param LU divisor from km to integrator's distance scalce
*/
std::vector<double> compute_thirdbody_perturbation(double x,
                                                   double y,
                                                   double z,
                                                   double GM_i,
                                                   ConstSpiceChar * third_body_name,
                                                   SpiceDouble et,
                                                   ConstSpiceChar * frame_cspice,
                                                   ConstSpiceChar * ref_body_name,
                                                   double LU) {
    // Compute axi, ayi, azi here
    SpiceDouble ri [3];  // position of perturbing body w.r.t. origin
    SpiceDouble ti;
    spkpos_c(third_body_name, et, frame_cspice, "NONE", ref_body_name, ri, &ti);

    // Magnitude of position, converted to length scale
    double xi = ri[0] / LU;
    double yi = ri[1] / LU;
    double zi = ri[2] / LU;
    double di = sqrt(xi*xi + yi*yi + zi*zi);

    // Position from perturbing body to spacecraft
    double xj = x - xi;
    double yj = y - yi;
    double zj = z - zi;
    double dj = sqrt(xj*xj + yj*yj + zj*zj);
    double dj3 = dj*dj*dj;

    // Gravitational effects from body i on spacecraft
    double qk = (x*(x-2*xi) + y*(y-2*yi) + z*(z-2*zi))/(di*di);
    double Fqk = qk*(3 + 3*qk + qk*qk)/(1 + sqrt(1+qk)*sqrt(1+qk)*sqrt(1+qk));
    double axi = -GM_i*(x + Fqk*xi) / dj3;  
    double ayi = -GM_i*(y + Fqk*yi) / dj3;
    double azi = -GM_i*(z + Fqk*zi) / dj3;

    // concatenate into output vector
    std::vector<double> result = {axi, ayi, azi, xi, yi, zi};
    return result;
}

