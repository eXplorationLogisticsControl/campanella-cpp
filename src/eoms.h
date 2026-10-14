#ifndef EOMS_H
#define EOMS_H

// Declare the two-body equations of motion for a 3D satellite orbiting the Earth
int eom_twobody(double t, const double y[], double dydt[], void* params);
int eom_NBody(double t, const double p[], double pdot[], void* params);
int eom_NBodySRP(double t, const double p[], double pdot[], void* params);

#endif
