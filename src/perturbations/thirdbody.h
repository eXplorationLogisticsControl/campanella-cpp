#ifndef THIRDBODY_H
#define THIRDBODY_H

// Third-body perturbation (direct and indirect terms) with Battin's F(q) function.
//
// @param r spacecraft position w.r.t. the central body
// @param s third-body position w.r.t. the central body, in the unit of r
// @param mu gravitational parameter of the third body, in units consistent with r
// @param a acceleration, written
void thirdbody_accel(const double r[3], const double s[3], double mu, double a[3]);

#endif
