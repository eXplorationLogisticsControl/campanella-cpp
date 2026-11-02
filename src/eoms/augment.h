#ifndef CAMPANELLA_AUGMENT_H
#define CAMPANELLA_AUGMENT_H

// Augmented state of the mass-augmented dynamics: y = [x (7), Phi (7x7), Psi (7x4)], 84 elements,
// with Phi and Psi row-major.

namespace campanella {
namespace detail {

/**
 * Derivatives of the sensitivities, inverse-free: Phi_dot = A Phi and Psi_dot = A Psi + B.
 * Full products: A has a mass column, so the shortcut of rhsstm_rv does not apply.
 *
 * @param A d(xdot)/dx (7x7)
 * @param B d(xdot)/du (7x4)
 * @param y augmented state (84)
 * @param ydot time derivative of y; elements 7 to 83 are computed, the state derivative is left as is
 */
inline void augment_rhs(const double A[7][7], const double B[7][4], const double y[], double ydot[]) {
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            double sum = 0.0;
            for (int l = 0; l < 7; l++) {
                sum += A[i][l] * y[7 + 7*l + j];
            }
            ydot[7 + 7*i + j] = sum;
        }
        for (int j = 0; j < 4; j++) {
            double sum = B[i][j];
            for (int l = 0; l < 7; l++) {
                sum += A[i][l] * y[56 + 4*l + j];
            }
            ydot[56 + 4*i + j] = sum;
        }
    }
}

}  // namespace detail
}  // namespace campanella

#endif  // CAMPANELLA_AUGMENT_H
