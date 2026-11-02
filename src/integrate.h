#ifndef CAMPANELLA_INTEGRATE_H
#define CAMPANELLA_INTEGRATE_H

#include <cstddef>
#include <vector>
#include "model.h"

namespace campanella {
namespace detail {

using Rhs = int (*)(double t, const double x[], double xdot[], void* params);

/**
 * Integrate rhs (nx states, or nx (1 + nx + nu) with with_stm, starting from Phi = I and Psi = 0,
 * both row-major) over tspan with GSL rk8pd.
 * Validates the inputs and throws on any failure (see throw_fault); never returns a partial result.
 *
 * @param zoh ZOH control passed to rhs through Context, or nullptr. If given, the last state is
 *        the mass, and m0 <= 0 throws std::domain_error, also over a zero-length tspan. Only then are
 *        options.max_steps and options.poll used, so that rv and mee are unaffected.
 * @param who name of the calling function, for the error messages
 */
IvpResult integrate(Rhs rhs, const Model& model, std::size_t nx, std::size_t nu, bool with_stm,
                    const ZohControl* zoh, const std::vector<double>& tspan,
                    const std::vector<double>& x0, const IvpOptions& options, const char* who);

}  // namespace detail
}  // namespace campanella

#endif  // CAMPANELLA_INTEGRATE_H
