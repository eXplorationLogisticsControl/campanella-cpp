#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

// Central-body spherical-harmonics gravity: acceleration and gradient of the potential
// U = (mu/R) sum_{n=2}^{nmax} sum_{m=0}^{n} (C_nm V_nm + S_nm W_nm), fully normalized.
// See docs/spherical_harmonics.md. This header does not include SpiceUsr.h.

#include <memory>
#include <string>
#include <vector>

// Largest supported truncation degree (no sectoral rescaling, see docs/spherical_harmonics.md)
const int SH_NMAX_LIMIT = 1200;

// Triangular index of (n, m), 0 <= m <= n
inline int sh_index(int n, int m) { return n*(n + 1)/2 + m; }

/**
 * Coefficients and precomputed recursion factors. Immutable after loading, so one model
 * can be shared between propagations.
 */
struct SphericalHarmonicsModel {
    std::string source;        // file path
    std::string tide_system;   // "" if the file does not say
    double GM_km3s2 = 0.0;     // gravitational parameter of the file header, km^3/s^2
    double R_km = 0.0;         // reference radius, km
    int file_degree = 0;       // maximum degree of the file
    int file_order = 0;        // maximum order of the file (SHADR header; ICGEM: file_degree)
    int nmax = 0;              // truncation degree used, 2 <= nmax <= file_degree
    int mmax = 0;              // truncation order used, 0 <= mmax <= nmax
    int nmin = 2;              // degrees 0 (point mass, in the EOMs) and 1 (zero) are excluded
    // fully normalized coefficients, index sh_index(n, m), n <= nmax; zero for m > mmax
    std::vector<double> C, S;
    // V, W recursion factors for n <= nmax + 2 (index sh_index(n, m)), sectoral factor c_m
    std::vector<double> rec_a, rec_b, rec_c;
    // acceleration and gradient factors for n <= nmax (index sh_index(n, m))
    std::vector<double> FAz, FAp, FAm, FG0, FGpp, FGp, FGm, FGmm;
};

/**
 * V, W tables to degree nmax + 2. Allocated once per propagation, reused at every RHS call.
 */
struct SphericalHarmonicsWorkspace {
    std::vector<double> V, W;
    // set by every sh_accel_bf call: true if it refused |r|/R < rho_min, with rho = |r|/R
    bool below_rho_min = false;
    double rho = 0.0;
    explicit SphericalHarmonicsWorkspace(int nmax);
};

// Load a PDS SHADR .tab or ICGEM .gfc file (auto-detected), truncated to nmax, mmax
// (mmax < 0 => min(nmax, order of the file)).
// Throws std::runtime_error or std::invalid_argument on any problem.
SphericalHarmonicsModel load_spherical_harmonics(const std::string& path, int nmax, int mmax = -1);

// As load_spherical_harmonics, but cached by (path, nmax, mmax) and the file's size, times and
// inode, so repeated propagations do not re-read the file. Files changed within the last 2 s are
// not cached. Entries are never evicted.
std::shared_ptr<const SphericalHarmonicsModel> load_spherical_harmonics_cached(
    const std::string& path, int nmax, int mmax = -1);

// Body-fixed acceleration, gradient (optional) and potential (optional); no allocation.
bool sh_accel_bf(const SphericalHarmonicsModel& model,
                 SphericalHarmonicsWorkspace& ws,
                 const double r_bf[3],
                 double mu,
                 double R,
                 double a_bf[3],
                 double G_bf[3][3],
                 double* U,
                 double rho_min = 1.0);

// Inertial acceleration and gradient (optional), rotating with pxform_c(frame, frame_bf, et).
bool sh_perturbation(const double rvec[3],
                     double et,
                     const char* frame_cspice,
                     const char* frame_bf_cspice,
                     const SphericalHarmonicsModel& model,
                     SphericalHarmonicsWorkspace& ws,
                     double mu_sh,
                     double R_sh,
                     double a[3],
                     double GG[3][3],
                     double rho_min = 1.0);

#endif // SPHERICAL_HARMONICS_H
