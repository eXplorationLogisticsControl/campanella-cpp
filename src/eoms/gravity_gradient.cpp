/*
Gravity gradient for Nbody problem
*/

#include <cmath>
#include "gravity_gradient.h"


/**
 * Calculates the gravity gradient matrix for the center body in an N-body problem.
 *
 * @param rvec The position vector of spacecraft.
 * @param mu The gravitational constant.
 * @param GG 3-by-3 gravity gradient matrix to be computed.
 */
void gravity_gradient_center(
	const double rvec[3],
	double mu,
	double GG[3][3]
) {
	// unpack rvec and compute r
	double x = rvec[0], y = rvec[1], z = rvec[2];
	double r = std::sqrt(x*x + y*y + z*z);
	double mur3 = mu / std::pow(r, 3);
	double r2 = std::pow(r, 2);

	// compute entries of the gravity gradient matrix
	GG[0][0] = mur3 * (-1 + 3*std::pow(x,2)/r2);
	GG[0][1] = mur3 * 3*x*y/r2;
	GG[0][2] = mur3 * 3*x*z/r2;

	GG[1][0] = mur3 * 3*x*y/r2;
	GG[1][1] = mur3 * (-1 + 3*std::pow(y,2)/r2);
	GG[1][2] = mur3 * 3*y*z/r2;

	GG[2][0] = mur3 * 3*x*z/r2;
	GG[2][1] = mur3 * 3*y*z/r2;
	GG[2][2] = mur3 * (-1 + 3*std::pow(z,2)/r2);
}


/**
 * Calculates the gravity gradient matrix for the third body in an N-body problem.
 *
 * @param rvec The position vector of spacecraft.
 * @param svec The position vector of the third body with respect to the center body
 * @param mu3 The gravitational constant of the third body.
 * @param GG 3-by-3 gravity gradient matrix to be computed.
 */
void gravity_gradient_third_body(
	const double rvec[3],
	const double svec[3],
	double mu3,
	double GG[3][3]) {
	
    // Compute the difference vector dvec
    double dvec[3] = {rvec[0] - svec[0], rvec[1] - svec[1], rvec[2] - svec[2]};
    
    // Unpack dvec
    double dx = dvec[0], dy = dvec[1], dz = dvec[2];
    
    // Compute the norm of dvec
    double d = std::sqrt(dx*dx + dy*dy + dz*dz);
	double d5 = std::pow(d, 5);
    
    // Define entries of the gravity gradient matrix
    GG[0][0] = mu3 * (2*dx*dx - dy*dy - dz*dz) / d5;
    GG[0][1] = 3 * mu3 * dx * dy / d5;
    GG[0][2] = 3 * mu3 * dx * dz / d5;

    GG[1][0] = 3 * mu3 * dx * dy / d5;
    GG[1][1] = mu3 * (2*dy*dy - dx*dx - dz*dz) / d5;
    GG[1][2] = 3 * mu3 * dy * dz / d5;

    GG[2][0] = 3 * mu3 * dx * dz / d5;
    GG[2][1] = 3 * mu3 * dy * dz / d5;
    GG[2][2] = mu3 * (2*dz*dz - dx*dx - dy*dy) / d5;
}