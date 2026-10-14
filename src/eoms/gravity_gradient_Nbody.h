#ifndef GRAVITY_GRADIENT_NBODY_H
#define GRAVITY_GRADIENT_NBODY_H

void gravity_gradient_center(
    double rvec[],
    double mu,
    double GG[3][3]
);

void gravity_gradient_third_body(
	double rvec[],
	double svec[],
	double mu3,
	double GG[3][3]);

#endif // GRAVITY_GRADIENT_NBODY_H