/**
 * Perturbing accelerations shared by all equations of motion: third bodies, SRP and
 * central-body spherical harmonics. All ephemeris queries of the EOMs are made here.
 */

#include <cmath>
#include <SpiceUsr.h>          // access SPICE files
#include "model.h"
#include "gravity_gradient.h"
#include "perturbations/thirdbody.h"
#include "perturbations/SRP.h"
#include "perturbations/spherical_harmonics.h"

namespace campanella {
namespace detail {

namespace {

// Record NAIF's message, reset SPICE and return the status that stops GSL
int capture_spice_error(Workspace& ws) {
    ws.spice_msg = spice_error_message();
    ws.fault = Fault::spice;
    return RHS_EBADFUNC;
}

void add3(double a[3], const double b[3]) {
    for (int j = 0; j < 3; j++) {
        a[j] = a[j] + b[j];
    }
}

void add33(double G[3][3], const double B[3][3]) {
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            G[j][k] += B[j][k];
        }
    }
}

}  // namespace

int check_finite(Workspace& ws, const double* v, int n) {
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(v[i])) {
            ws.fault = Fault::nonfinite;
            return RHS_EDOM;
        }
    }
    return RHS_SUCCESS;
}

/**
 * Add the perturbing accelerations, and optionally their gradient, at time t and position r
 *
 * @param model dynamics model
 * @param ws scratch space; body positions are stored in ws.body_pos
 * @param t time, in TU
 * @param r spacecraft position w.r.t. the central body, in DU
 * @param a acceleration, in DU/TU^2, added to
 * @param G gradient d(a)/d(r), in 1/TU^2, added to; nullptr to skip it
 * @return RHS_SUCCESS; RHS_EDOM below sh_rho_min * R; RHS_EBADFUNC on a SPICE error
 */
int add_perturbations(const Model& model, Workspace& ws, double t, const double r[3],
                      double a[3], double G[3][3]) {
    const size_t N = model.codes.size();
    const char* frame = model.options.naif_frame.c_str();
    const SpiceInt center = model.codes[0];
    const SpiceDouble et = model.et0 + t*model.TU;   // ephemeris query time

    // third bodies
    for (size_t i = 1; i < N; i++) {
        double* s = &ws.body_pos[3*i];
        SpiceDouble lt;
        spkgps_c(model.codes[i], et, frame, center, s, &lt);
        if (failed_c()) {
            return capture_spice_error(ws);
        }
        for (int j = 0; j < 3; j++) {
            s[j] = s[j] / model.DU;
        }
        double a_i[3];
        thirdbody_accel(r, s, model.mus[i], a_i);
        add3(a, a_i);
        if (G) {
            double G_i[3][3];
            gravity_gradient_third_body(r, s, model.mus[i], G_i);
            add33(G, G_i);
        }
    }

    // solar radiation pressure, from the Sun position of the loop above (zero if it is the center)
    if (model.idx_sun >= 0) {
        const double* s = &ws.body_pos[3*model.idx_sun];
        const double d[3] = {r[0] - s[0], r[1] - s[1], r[2] - s[2]};
        const double R = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        const double kappa = model.srp_k / (R*R*R);
        const double a_srp[3] = {kappa*d[0], kappa*d[1], kappa*d[2]};
        add3(a, a_srp);
        if (G) {
            double G_srp[3][3];
            srp_gradient(d, kappa, G_srp);
            add33(G, G_srp);
        }
    }

    // central-body spherical harmonics
    if (model.sh) {
        double a_sh[3];
        double G_sh[3][3];
        bool ok = sh_perturbation(r, et, frame, model.options.sh_frame.c_str(), *model.sh, *ws.sh_ws,
                                  model.mu_sh, model.R_sh, a_sh, G ? G_sh : nullptr,
                                  model.options.sh_rho_min);
        if (failed_c()) {
            return capture_spice_error(ws);
        }
        if (!ok) {
            ws.fault = Fault::sh_domain;
            return RHS_EDOM;   // below sh_rho_min * R: the series may diverge
        }
        add3(a, a_sh);
        if (G) {
            add33(G, G_sh);
        }
    }
    return RHS_SUCCESS;
}

}  // namespace detail
}  // namespace campanella
