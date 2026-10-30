#include "campanella.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "SpiceUsr.h"
#include "eoms/mee.h"
#include "model.h"
#include "perturbations/spherical_harmonics.h"

namespace campanella {

namespace detail {

std::string spice_error_message() {
    SpiceChar short_msg[26];
    SpiceChar long_msg[1841];
    getmsg_c("SHORT", sizeof(short_msg), short_msg);
    getmsg_c("LONG", sizeof(long_msg), long_msg);
    reset_c();
    return std::string(short_msg) + " -- " + long_msg;
}

void spice_throw_if_failed() {
    if (failed_c()) {
        throw std::runtime_error(spice_error_message());
    }
}

}  // namespace detail

using detail::spice_throw_if_failed;

namespace {

void spice_errors_return() {
    SpiceChar action[] = "RETURN";
    erract_c("SET", sizeof(action), action);
    SpiceChar report[] = "NONE";
    errprt_c("SET", sizeof(report), report);
}

}  // namespace

void init_spice() {
    spice_errors_return();
}

void furnsh(const std::string& path) {
    init_spice();
    furnsh_c(path.c_str());
    spice_throw_if_failed();
}

std::tuple<double, std::vector<double>, std::vector<double>> sh_accel_bf(
    const std::string& path, int nmax, const std::vector<double>& r_bf, int mmax, double mu, double R,
    double rho_min) {
    if (r_bf.size() != 3) {
        throw std::invalid_argument("sh_accel_bf: r_bf must have length 3");
    }
    if ((mu != 0.0) != (R != 0.0) || mu < 0.0 || R < 0.0) {
        throw std::invalid_argument("sh_accel_bf: give both mu and R (positive, same units), or neither");
    }
    if (!(rho_min > 0.0 && rho_min <= 1.0)) {
        throw std::invalid_argument("sh_accel_bf: rho_min must be in (0, 1]");
    }
    std::shared_ptr<const SphericalHarmonicsModel> model = load_spherical_harmonics_cached(path, nmax, mmax);
    SphericalHarmonicsWorkspace ws(model->nmax);
    double a[3];
    double G[3][3];
    double U;
    if (!::sh_accel_bf(*model, ws, r_bf.data(), (mu > 0) ? mu : model->GM_km3s2,
                       (R > 0) ? R : model->R_km, a, G, &U, rho_min)) {
        throw std::domain_error("sh_accel_bf: |r_bf| is below rho_min times the reference radius");
    }
    return std::make_tuple(U, std::vector<double>(a, a + 3), std::vector<double>(&G[0][0], &G[0][0] + 9));
}

std::pair<std::vector<double>, std::vector<double>> sh_accel_inertial(
    const std::string& path, int nmax, const std::string& naif_frame, const std::string& sh_frame,
    double et, const std::vector<double>& r, double DU, double TU, int mmax, double rho_min) {
    if (r.size() != 3) {
        throw std::invalid_argument("sh_accel_inertial: r must have length 3");
    }
    if (!(rho_min > 0.0 && rho_min <= 1.0)) {
        throw std::invalid_argument("sh_accel_inertial: rho_min must be in (0, 1]");
    }
    std::shared_ptr<const SphericalHarmonicsModel> model = load_spherical_harmonics_cached(path, nmax, mmax);
    SphericalHarmonicsWorkspace ws(model->nmax);
    const double mu_sh = model->GM_km3s2*TU*TU / (DU*DU*DU);
    const double R_sh = model->R_km / DU;
    double a[3];
    double G[3][3];
    bool ok = ::sh_perturbation(r.data(), et, naif_frame.c_str(), sh_frame.c_str(), *model, ws, mu_sh, R_sh, a, G, rho_min);
    spice_throw_if_failed();
    if (!ok) {
        throw std::domain_error("sh_accel_inertial: |r| is below rho_min times the reference radius");
    }
    return {std::vector<double>(a, a + 3), std::vector<double>(&G[0][0], &G[0][0] + 9)};
}

std::vector<double> mee2cart(const std::vector<double>& mee, double mu) {
    return ::mee2cart(mee, mu);
}

std::vector<double> cart2mee(const std::vector<double>& rv, double mu) {
    return ::cart2mee(rv, mu);
}

std::vector<double> mee2cart_jacobian(const std::vector<double>& mee, double mu) {
    return ::mee2cart_jacobian(mee, mu);
}

}  // namespace campanella
