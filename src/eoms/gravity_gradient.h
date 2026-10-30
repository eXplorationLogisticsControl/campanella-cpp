#ifndef GRAVITY_GRADIENT_H
#define GRAVITY_GRADIENT_H

// Gradient d(a)/d(r) of the point-mass acceleration of the central body
void gravity_gradient_center(
    const double rvec[3],
    double mu,
    double GG[3][3]
);

// Gradient d(a)/d(r) of the third-body acceleration (svec: third body w.r.t. the central body)
void gravity_gradient_third_body(
	const double rvec[3],
	const double svec[3],
	double mu3,
	double GG[3][3]);

#endif // GRAVITY_GRADIENT_H
