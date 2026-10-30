#ifndef CAMPANELLA_INTEGRATE_H
#define CAMPANELLA_INTEGRATE_H

#include <vector>
#include "model.h"

namespace campanella {
namespace detail {

using Rhs = int (*)(double t, const double x[], double xdot[], void* params);

/**
 * Integrate rhs (6 states, or 42 with with_stm, starting from Phi = I) over tspan with GSL rk8pd.
 * Validates the inputs and throws on any failure (see throw_fault); never returns a partial result.
 *
 * @param who name of the calling function, for the error messages
 */
IvpResult integrate(Rhs rhs, const Model& model, bool with_stm, const std::vector<double>& tspan,
                    const std::vector<double>& x0, const IvpOptions& options, const char* who);

}  // namespace detail
}  // namespace campanella

#endif  // CAMPANELLA_INTEGRATE_H
