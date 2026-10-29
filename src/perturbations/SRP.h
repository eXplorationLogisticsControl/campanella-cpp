#ifndef SRP_H
#define SRP_H

#include <vector>
#include <SpiceUsr.h>          // access SPICE files


// Returns {ax, ay, az, xs, ys, zs}: SRP acceleration and sun position w.r.t. reference body / LU
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
                                                   double LU);

// Gradient d(a_SRP)/dr of compute_SRP, given the sun position returned by compute_SRP
void srp_gradient(const double rvec[],
                  const double svec[],
                  double A,
                  double m,
                  double rhoS,
                  double rhoD,
                  double TU,
                  double LU,
                  double GG[3][3]);

#endif
