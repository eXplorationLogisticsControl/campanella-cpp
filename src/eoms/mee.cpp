/**
 * Conversions between Cartesian states and modified equinoctial elements (MEE)
 *
 * MEE are [p, f, g, h, k, L] of Walker, Ireland & Owens (1985) with retrograde factor I = +1.
 * Formulas and section numbers refer to docs/gve_mee.md.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "mee.h"

namespace {

double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

void check_length(const std::vector<double>& v, const char* name) {
    if (v.size() != 6) {
        throw std::invalid_argument(std::string(name) + " must have length 6.");
    }
}

// Check p > 0 and w > 0; NaN fails both. caller names the function in the message.
void check_mee_domain(double p, double w, const char* caller) {
    if (!(p > 0)) {
        throw std::domain_error(std::string(caller) + ": p must be positive.");
    }
    if (!(w > 0)) {
        throw std::domain_error(
            std::string(caller) + ": w = 1 + f cos(L) + g sin(L) must be positive.");
    }
}

}  // namespace


/**
 * Equinoctial frame unit vectors, expressed in the integration frame (sec. 1.3)
 *
 * @param h MEE h
 * @param k MEE k
 * @param fhat direction from which longitudes are measured, in orbit plane
 * @param ghat second in-plane direction
 * @param what orbit normal, along angular momentum
 */
void mee_equinoctial_frame(double h, double k, double fhat[3], double ghat[3], double what[3]) {
    double s2 = 1 + h*h + k*k;
    fhat[0] = (1 - k*k + h*h) / s2;
    fhat[1] = 2*h*k / s2;
    fhat[2] = -2*k / s2;

    ghat[0] = 2*h*k / s2;
    ghat[1] = (1 + k*k - h*h) / s2;
    ghat[2] = 2*h / s2;

    what[0] = 2*k / s2;
    what[1] = -2*h / s2;
    what[2] = (1 - h*h - k*k) / s2;
}


/**
 * RTN unit vectors as functions of (h, k, L) (sec. 3, eq. 3.1)
 *
 * @param h MEE h
 * @param k MEE k
 * @param cL cos(L)
 * @param sL sin(L)
 * @param rhat radial unit vector
 * @param that along-track unit vector, nhat x rhat
 * @param nhat orbit normal unit vector
 */
void mee_rtn_basis(double h, double k, double cL, double sL,
                   double rhat[3], double that[3], double nhat[3]) {
    double fhat[3], ghat[3];
    mee_equinoctial_frame(h, k, fhat, ghat, nhat);
    for (int i = 0; i < 3; i++) {
        rhat[i] =  cL*fhat[i] + sL*ghat[i];
        that[i] = -sL*fhat[i] + cL*ghat[i];
    }
}


/**
 * Convert MEE to Cartesian state (sec. 2.1, vector form 2.1)
 *
 * @param mee [p, f, g, h, k, L]; L may be any real number
 * @param mu gravitational parameter
 * @return [x, y, z, vx, vy, vz]
 */
std::vector<double> mee2cart(const std::vector<double>& mee, double mu) {
    check_length(mee, "mee");
    double p = mee[0], f = mee[1], g = mee[2], h = mee[3], k = mee[4], L = mee[5];
    double cL = std::cos(L);
    double sL = std::sin(L);
    double w = 1 + f*cL + g*sL;
    check_mee_domain(p, w, "mee2cart");
    double r = p / w;
    double sqrt_mu_p = std::sqrt(mu / p);

    double fhat[3], ghat[3], what[3];
    mee_equinoctial_frame(h, k, fhat, ghat, what);

    std::vector<double> rv(6);
    for (int i = 0; i < 3; i++) {
        rv[i]     = r * (cL*fhat[i] + sL*ghat[i]);                         // (2.1a)
        rv[3 + i] = sqrt_mu_p * (-(sL + g)*fhat[i] + (cL + f)*ghat[i]);   // (2.1b)
    }
    return rv;
}


/**
 * Convert Cartesian state to MEE (sec. 2.2)
 *
 * @param rv [x, y, z, vx, vy, vz]
 * @param mu gravitational parameter
 * @return [p, f, g, h, k, L], with L in (-pi, pi]
 */
std::vector<double> cart2mee(const std::vector<double>& rv, double mu) {
    check_length(rv, "rv");
    double rvec[3] = {rv[0], rv[1], rv[2]};
    double vvec[3] = {rv[3], rv[4], rv[5]};

    // 1. angular momentum
    double H[3];
    cross3(rvec, vvec, H);
    double Hn = std::sqrt(dot3(H, H));
    if (!(Hn > 0)) {
        throw std::domain_error("cart2mee: zero angular momentum (rectilinear orbit).");
    }

    // 2. semi-latus rectum
    double p = Hn*Hn / mu;

    // 3. h, k from what = H/|H|; denominator |H|(1 + cos i) vanishes at i = 180 deg
    double den = Hn + H[2];
    if (den <= 1e-12 * Hn) {
        throw std::domain_error("cart2mee: retrograde equatorial orbit; MEE with I=+1 singular.");
    }
    double h = -H[1] / den;
    double k =  H[0] / den;

    // 4. equinoctial frame, consistent with mee2cart
    double fhat[3], ghat[3], what[3];
    mee_equinoctial_frame(h, k, fhat, ghat, what);

    // 5. eccentricity vector e = (v x H)/mu - r/|r|
    double rn = std::sqrt(dot3(rvec, rvec));
    double vxH[3];
    cross3(vvec, H, vxH);
    double evec[3];
    for (int i = 0; i < 3; i++) {
        evec[i] = vxH[i]/mu - rvec[i]/rn;
    }
    double f = dot3(evec, fhat);
    double g = dot3(evec, ghat);

    // 6. true longitude
    double L = std::atan2(dot3(rvec, ghat), dot3(rvec, fhat));

    return {p, f, g, h, k, L};
}


/**
 * Jacobian of the Cartesian state with respect to MEE, d(r, v)/d(mee) (sec. 5.7)
 *
 * @param mee [p, f, g, h, k, L]
 * @param mu gravitational parameter
 * @return 6x6 Jacobian, row-major: J[6*i + m] = d(rv[i]) / d(mee[m])
 */
std::vector<double> mee2cart_jacobian(const std::vector<double>& mee, double mu) {
    check_length(mee, "mee");
    double p = mee[0], f = mee[1], g = mee[2], h = mee[3], k = mee[4], L = mee[5];
    double cL = std::cos(L);
    double sL = std::sin(L);
    double w = 1 + f*cL + g*sL;
    check_mee_domain(p, w, "mee2cart_jacobian");
    double s2 = 1 + h*h + k*k;
    double r = p / w;
    double sqrt_mu_p = std::sqrt(mu / p);   // sigma' in sec. 5.7

    double fhat[3], ghat[3], what[3];
    mee_equinoctial_frame(h, k, fhat, ghat, what);
    double rhat[3], that[3], nhat[3];
    mee_rtn_basis(h, k, cL, sL, rhat, that, nhat);

    std::vector<double> J(36, 0.0);
    for (int i = 0; i < 3; i++) {
        double vvec_i = sqrt_mu_p * (-(sL + g)*fhat[i] + (cL + f)*ghat[i]);

        // position rows
        J[6*i + 0] = rhat[i] / w;
        J[6*i + 1] = -(r*cL/w) * rhat[i];
        J[6*i + 2] = -(r*sL/w) * rhat[i];
        J[6*i + 3] = (2*r/s2) * ( k*that[i] + sL*nhat[i]);
        J[6*i + 4] = (2*r/s2) * (-h*that[i] - cL*nhat[i]);
        J[6*i + 5] = (r*(f*sL - g*cL)/w) * rhat[i] + r*that[i];

        // velocity rows
        J[6*(3 + i) + 0] = -vvec_i / (2*p);
        J[6*(3 + i) + 1] =  sqrt_mu_p * ghat[i];
        J[6*(3 + i) + 2] = -sqrt_mu_p * fhat[i];
        J[6*(3 + i) + 3] = (2*sqrt_mu_p/s2)
            * (-k*(cL + f)*fhat[i] - k*(sL + g)*ghat[i] + (cL + f)*what[i]);
        J[6*(3 + i) + 4] = (2*sqrt_mu_p/s2)
            * ( h*(cL + f)*fhat[i] + h*(sL + g)*ghat[i] + (sL + g)*what[i]);
        J[6*(3 + i) + 5] = -sqrt_mu_p * rhat[i];
    }
    return J;
}
