#ifndef MEE_H
#define MEE_H

#include <vector>

// Modified equinoctial elements (MEE) [p, f, g, h, k, L], prograde (I = +1).
// See docs/gve_mee.md.

// Equinoctial frame unit vectors fhat, ghat, what (sec. 1.3)
void mee_equinoctial_frame(double h, double k, double fhat[3], double ghat[3], double what[3]);

// RTN unit vectors rhat, that, nhat as functions of (h, k, L) (sec. 3, eq. 3.1)
void mee_rtn_basis(double h, double k, double cL, double sL,
                   double rhat[3], double that[3], double nhat[3]);

// Convert MEE to Cartesian state [x, y, z, vx, vy, vz]. Throws on p <= 0 or w <= 0.
std::vector<double> mee2cart(const std::vector<double>& mee, double mu);

// Convert Cartesian state to MEE. L is returned in (-pi, pi]. Throws for H = 0 or i -> 180 deg.
std::vector<double> cart2mee(const std::vector<double>& rv, double mu);

// Jacobian d(r, v)/d(mee), 6x6 row-major (sec. 5.7). Throws on p <= 0 or w <= 0.
std::vector<double> mee2cart_jacobian(const std::vector<double>& mee, double mu);

#endif // MEE_H
