/**
 * Third-body perturbation
 */

#include <cmath>
#include "thirdbody.h"

/**
 * Compute third-body perturbation using Battin's F(q) function
 *
 * @param r spacecraft position w.r.t. the central body
 * @param s third-body position w.r.t. the central body
 * @param mu gravitational parameter of the third body
 * @param a acceleration to be computed
 */
void thirdbody_accel(const double r[3], const double s[3], double mu, double a[3]) {
    const double x = r[0], y = r[1], z = r[2];
    const double xi = s[0], yi = s[1], zi = s[2];
    double di = std::sqrt(xi*xi + yi*yi + zi*zi);

    // Position from perturbing body to spacecraft
    double xj = x - xi;
    double yj = y - yi;
    double zj = z - zi;
    double dj = std::sqrt(xj*xj + yj*yj + zj*zj);
    double dj3 = dj*dj*dj;

    // Gravitational effects from body i on spacecraft
    double qk = (x*(x-2*xi) + y*(y-2*yi) + z*(z-2*zi))/(di*di);
    double Fqk = qk*(3 + 3*qk + qk*qk)/(1 + std::sqrt(1+qk)*std::sqrt(1+qk)*std::sqrt(1+qk));
    a[0] = -mu*(x + Fqk*xi) / dj3;
    a[1] = -mu*(y + Fqk*yi) / dj3;
    a[2] = -mu*(z + Fqk*zi) / dj3;
}
