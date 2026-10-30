#ifndef SRP_H
#define SRP_H

// Solar radiation pressure (SRP), cannonball-like model with a = kappa (r - s), s the Sun position.

// LEGACY (removed with the SRP fix): kappa with the flux scaled by R0 = |s|, the distance from the
// central body to the Sun, rather than 1 AU.
double srp_legacy_kappa(double R0, double R, double A, double m, double rhoS, double rhoD,
                        double TU, double DU);

// Gradient d(a_SRP)/dr = kappa (I - 3 d d^T / |d|^2), d = r - s, for a = kappa d
void srp_gradient(const double d[3], double kappa, double GG[3][3]);

#endif
