// Propagate a multi-segment low-thrust arc around the Moon with a zero-order-hold control, one
// solve_ivpstm_rvm call per segment, and check the sensitivities of the last segment (Phi = dx_f/dx_0,
// Psi = dx_f/du) against central finite differences of the state-only solve_ivp_rvm.
// The control is u = [tau, Gamma], with Gamma = |tau| imposed here, as an optimal control solver
// would. The finite differences perturb tau and Gamma separately: the dynamics treat them as
// independent, and so does Psi.
// SPICE must point at the kernel directory.

#include "campanella.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

const int NX = 7;
const int NU = 4;

std::string kernel_path(const char* spice_dir, const char* relative) {
    std::string path(spice_dir);
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    path += relative;
    return path;
}

double norm3(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

std::vector<double> last_row(const std::vector<double>& data, std::size_t width) {
    return std::vector<double>(data.end() - static_cast<std::ptrdiff_t>(width), data.end());
}

// Largest |M - R|_ij over the smaller of the infinity norms of row i and column j of R: a row metric
// alone can hide an error in a small column, and the reverse.
// Infinite if anything is not finite, since max would otherwise skip a NaN.
double scaled_error(const std::vector<double>& M, const std::vector<double>& R, int rows, int cols) {
    std::vector<double> row_norm(rows, 0.0), col_norm(cols, 0.0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            const double r = std::abs(R[cols * i + j]);
            row_norm[i] = std::max(row_norm[i], r);
            col_norm[j] = std::max(col_norm[j], r);
        }
    }
    double worst = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            const double diff = std::abs(M[cols * i + j] - R[cols * i + j]);
            if (!std::isfinite(diff)) {
                return std::numeric_limits<double>::infinity();
            }
            if (diff == 0.0) {
                continue;
            }
            const double scale = std::min(row_norm[i], col_norm[j]);
            worst = std::max(worst, scale > 0.0 ? diff / scale : std::numeric_limits<double>::infinity());
        }
    }
    return worst;
}

void print_matrix(const char* name, const std::vector<double>& M, int rows, int cols) {
    std::cout << name << ":\n" << std::setprecision(6) << std::scientific;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(15) << M[cols * i + j];
        }
        std::cout << "\n";
    }
}

}  // namespace

int main() {
    const char* spice_dir = std::getenv("SPICE");
    if (spice_dir == nullptr || spice_dir[0] == '\0') {
        std::cerr << "SPICE must point at a directory of generic SPICE kernels\n";
        return 1;
    }
    try {
        campanella::furnsh(kernel_path(spice_dir, "lsk/naif0012.tls"));
        campanella::furnsh(kernel_path(spice_dir, "spk/de440.bsp"));

        // Moon-centred in canonical units, with the Earth, the Sun and SRP.
        const double et0 = 740463475.5117936;
        campanella::SpiceOptions options;
        options.GMs = {4.902800582147760e+03, 398600.432896939, 132712197035.766};
        options.DU = 8000.0;        // km
        options.include_srp = true;
        options.srp_A = 20.0;       // m^2
        options.srp_m = 1000.0;     // kg; must be > 0, but rvm uses the state mass
        options.srp_rhoS = 0.9;
        const campanella::SpiceParameters params(et0, {"MOON", "EARTH", "SUN"}, options);
        const double DU = params.DU();
        const double VU = params.VU();
        const double TU = params.TU();
        const double mu = params.mus()[0];

        campanella::ThrustOptions thrust;
        thrust.thrust_N = 1.0;
        thrust.isp_s = 3000.0;
        thrust.mass_unit_kg = 1000.0;

        // Elliptic orbit, a = 8000 km and e = 0.1 inclined by 30 deg, starting at periapsis
        const double r_p = 7200.0;   // km
        const double a_0 = 8000.0;   // km
        const double v_p = std::sqrt(options.GMs[0] * (2.0 / r_p - 1.0 / a_0));   // km/s
        const double inc = 30.0 * std::acos(-1.0) / 180.0;
        std::vector<double> x = {r_p / DU, 0.0, 0.0, 0.0, v_p * std::cos(inc) / VU,
                                 v_p * std::sin(inc) / VU, 1.0};

        // One-hour segments, with a coast; tau is along the velocity at the start of each segment
        const std::vector<double> throttle = {1.0, 1.0, 1.0, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 1.0, 1.0, 1.0};
        const double dt = 3600.0 / TU;

        // Tight enough for the small Psi entries to rise above the integration error of the differences
        campanella::IvpOptions ivp;
        ivp.atol = 1e-14;
        ivp.rtol = 1e-14;

        std::cout << "DU = " << DU << " km, TU = " << TU << " s, mu = " << mu << "\n"
                  << " seg  throttle     t [h]    |r| [km]  |v| [km/s]      a [km]     m [kg]\n";
        const auto print_row = [&](int seg, double level, double t, const std::vector<double>& s) {
            const double r = norm3(s[0], s[1], s[2]);
            const double v = norm3(s[3], s[4], s[5]);
            std::cout << std::fixed << std::setw(4) << seg << std::setprecision(1) << std::setw(10) << level
                      << std::setw(10) << t * TU / 3600.0 << std::setprecision(3) << std::setw(12) << r * DU
                      << std::setprecision(6) << std::setw(12) << v * VU << std::setprecision(3)
                      << std::setw(12) << DU / (2.0 / r - v * v / mu) << std::setw(11)
                      << s[6] * thrust.mass_unit_kg << "\n";
        };
        print_row(0, 0.0, 0.0, x);

        // March the segments, keeping the start and control of the last one
        double t = 0.0;
        double gamma_dt = 0.0;
        std::vector<double> x_k, u_k, tspan_k, Phi, Psi;
        for (std::size_t k = 0; k < throttle.size(); k++) {
            const double v = norm3(x[3], x[4], x[5]);
            const double level = throttle[k];
            const std::vector<double> tau = {level * x[3] / v, level * x[4] / v, level * x[5] / v};
            x_k = x;
            u_k = {tau[0], tau[1], tau[2], norm3(tau[0], tau[1], tau[2])};
            tspan_k = {t, t + dt};
            const campanella::IvpResult seg =
                campanella::solve_ivpstm_rvm(params, thrust, tspan_k, x_k, u_k, ivp);
            x = last_row(seg.x, NX);
            Phi = last_row(seg.stm, NX * NX);
            Psi = last_row(seg.stm_u, NX * NU);
            t = tspan_k[1];
            gamma_dt += u_k[3] * dt;
            print_row(static_cast<int>(k) + 1, level, t, x);
        }

        // The mass has a closed form, from the mass flow T / (Isp g0)
        const double c2 = thrust.thrust_N / (thrust.isp_s * thrust.g0) * TU / thrust.mass_unit_kg;
        const double dm = std::abs(x[6] - (1.0 - c2 * gamma_dt));

        // Central differences over the last segment. Steps in canonical units: the position and
        // velocity steps are small against the orbit, the mass step is 1e-5 of the mass.
        const std::vector<double> h_x = {1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-5 * x_k[6]};
        const std::vector<double> h_u = {1e-4, 1e-4, 1e-4, 1e-2};
        std::vector<double> Phi_fd(NX * NX), Psi_fd(NX * NU);
        for (int j = 0; j < NX; j++) {
            std::vector<double> plus = x_k, minus = x_k;
            plus[j] += h_x[j];
            minus[j] -= h_x[j];
            const std::vector<double> xp = last_row(campanella::solve_ivp_rvm(params, thrust, tspan_k, plus, u_k, ivp).x, NX);
            const std::vector<double> xm = last_row(campanella::solve_ivp_rvm(params, thrust, tspan_k, minus, u_k, ivp).x, NX);
            for (int i = 0; i < NX; i++) {
                Phi_fd[NX * i + j] = (xp[i] - xm[i]) / (2.0 * h_x[j]);
            }
        }
        for (int j = 0; j < NU; j++) {
            std::vector<double> plus = u_k, minus = u_k;
            plus[j] += h_u[j];
            minus[j] -= h_u[j];
            const std::vector<double> xp = last_row(campanella::solve_ivp_rvm(params, thrust, tspan_k, x_k, plus, ivp).x, NX);
            const std::vector<double> xm = last_row(campanella::solve_ivp_rvm(params, thrust, tspan_k, x_k, minus, ivp).x, NX);
            for (int i = 0; i < NX; i++) {
                Psi_fd[NU * i + j] = (xp[i] - xm[i]) / (2.0 * h_u[j]);
            }
        }

        std::cout << "\nlast segment, u = [" << std::setprecision(6) << std::fixed << u_k[0] << ", " << u_k[1]
                  << ", " << u_k[2] << ", " << u_k[3] << "]\n";
        print_matrix("Psi", Psi, NX, NU);
        print_matrix("Psi, finite differences", Psi_fd, NX, NU);
        print_matrix("Phi", Phi, NX, NX);
        print_matrix("Phi, finite differences", Phi_fd, NX, NX);

        const double err_phi = scaled_error(Phi, Phi_fd, NX, NX);
        const double err_psi = scaled_error(Psi, Psi_fd, NX, NU);
        std::cout << std::setprecision(3) << std::scientific
                  << "\nmass against closed form: " << dm
                  << "\nPhi error, scaled by row and column: " << err_phi
                  << "\nPsi error, scaled by row and column: " << err_psi << "\n";

        // About 10 times the measured errors, which are round-off in the differences
        const double tol_phi = 2e-9;
        const double tol_psi = 5e-8;
        const double tol_mass = 1e-14;
        if (!(err_phi <= tol_phi && err_psi <= tol_psi && dm <= tol_mass)) {
            std::cerr << "finite-difference check failed: tolerances are Phi " << tol_phi << ", Psi "
                      << tol_psi << ", mass " << tol_mass << "\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
