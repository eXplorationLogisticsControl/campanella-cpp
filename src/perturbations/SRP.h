#ifndef SRP_H
#define SRP_H

// Solar radiation pressure (SRP), cannonball-like model:
//   a = K d / |d|^3,  d = r - r_sun,  K = (P0 / c) Cr (A / m) AU^2,  Cr = 1 + rhoS + 5 rhoD / 3
// with P0 the solar flux at 1 AU. No shadow.

// K in DU^3/TU^2, for A in m^2, m in kg, DU in km and TU in s
double srp_coefficient(double A, double m, double rhoS, double rhoD, double DU, double TU);

// Gradient d(a_SRP)/dr = kappa (I - 3 d d^T / |d|^2), for a = kappa d (kappa = K / |d|^3)
void srp_gradient(const double d[3], double kappa, double GG[3][3]);

#endif
