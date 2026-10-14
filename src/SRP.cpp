/**
 * Solar Radiation Pressure (SRP) accelertations
*/


#include <cmath>
#include <vector>
#include <SpiceUsr.h>          // access SPICE files

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

    //SRP constants (should these be inputs?)
    double P0 = 1367; // [w/m^2]  [W/LU^2]
    double c = 299792458; //[m/s]  [LU/TU]

    //A = A/LU/LU;

    //Should this be an input? And is the solar system barycenter the correct Spice name for the sun?
    ConstSpiceChar * sun;
    sun = "SOLAR SYSTEM BARYCENTER";
    
    // Position from reference body to sun
    spkpos_c(sun, et, frame_cspice, "NONE", ref_body_name, ri, &ti);

    // Position from Sun to reference body
    double xi = ri[0] / LU;
    double yi = ri[1] / LU;
    double zi = ri[2] / LU;
    double R0 = sqrt(xi*xi + yi*yi + zi*zi);
    // Position from Sun to spacecraft (Sun to reference body + reference body to spacecraft)
    double xt = x - xi;
    double yt = y - yi;
    double zt = z - zi;
    double R = sqrt(xt*xt + yt*yt + zt*zt);

    // SRP equations
    double P = P0*(R0/R)*(R0/R)/c;
    double Cr = 1 + rhoS + 5 * rhoD / 3;

    // SRP effects from sun on spacecraft
    double srp_magnitude = P*Cr*A/R/m / (1e3 * LU/(TU*TU));
    double ax_SRP = srp_magnitude * xt;  // P*Cr*A*xt/R/m;
    double ay_SRP = srp_magnitude * yt;  // P*Cr*A*yt/R/m;
    double az_SRP = srp_magnitude * zt;  // P*Cr*A*zt/R/m;

    // concatenate into output vector
    std::vector<double> result = {ax_SRP, ay_SRP, az_SRP};
    return result;
}
