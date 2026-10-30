/*
 * Python bindings (pybind11)
 */

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>       // enable automatic conversion between C++ STL types and Python types

#include "campanella.hpp"
#include "model.h"
#include "perturbations/spherical_harmonics.h"

namespace py = pybind11;
using namespace campanella;

namespace {

// Read-only NumPy view of data owned by base (no copy)
py::array_t<double> view(std::vector<py::ssize_t> shape, const std::vector<double>& data, py::handle base) {
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = sizeof(double);
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    py::array_t<double> a(shape, strides, data.data(), base);
    py::detail::array_proxy(a.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return a;
}

using Solver = IvpResult (*)(const SpiceParameters&, const std::vector<double>&,
                             const std::vector<double>&, const IvpOptions&);

// Python signature solve(params, tspan, x0, *, atol, rtol, t_eval)
void def_solver(py::module_& m, const char* name, Solver solver, const char* doc) {
    m.def(name,
          [solver](const SpiceParameters& params, const std::vector<double>& tspan,
                   const std::vector<double>& x0, double atol, double rtol,
                   std::optional<std::vector<double>> t_eval) {
              IvpOptions options;
              options.atol = atol;
              options.rtol = rtol;
              if (t_eval) {
                  options.t_eval = std::move(*t_eval);
              }
              return solver(params, tspan, x0, options);
          },
          py::arg("params"), py::arg("tspan"), py::arg("x0"), py::kw_only(),
          py::arg("atol") = IvpOptions{}.atol, py::arg("rtol") = IvpOptions{}.rtol,
          py::arg("t_eval") = py::none(), doc);
}

using Eom = int (*)(double, const double[], double[], void*);

// Python signature eom(params, t, x) -> xdot; raises like eval_* on a failure
void def_eom(py::module_& m, const char* name, Eom eom, size_t dim, const char* doc) {
    m.def(name,
          [eom, dim, name](const SpiceParameters& params, double t, const std::vector<double>& x) {
              if (x.size() != dim) {
                  throw std::invalid_argument(std::string(name) + ": x must have length " + std::to_string(dim));
              }
              py::array_t<double> xdot(std::vector<py::ssize_t>{static_cast<py::ssize_t>(dim)});
              int status = eom(t, x.data(), xdot.mutable_data(), const_cast<SpiceParameters*>(&params));
              if (status != detail::RHS_SUCCESS) {
                  detail::throw_fault(detail::Access::model(params), detail::Access::workspace(params),
                                      status, t, name);
              }
              return xdot;
          },
          py::arg("params"), py::arg("t"), py::arg("x"), doc);
}

py::tuple eval_tuple(const EomEval& e) {
    py::array_t<double> xdot(std::vector<py::ssize_t>{6}, e.xdot.data());
    py::array_t<double> A(std::vector<py::ssize_t>{6, 6}, e.A.data());
    return py::make_tuple(xdot, A);
}

}  // namespace


PYBIND11_MODULE(campanella, m) {
    m.doc() = "campanella is a c++ library for high-fidelity ephemeris model dynamics";

    // Initialize CSPICE when the module is loaded.
    // Kernels are not loaded here. Call campanella.furnsh() from Python.
    // spiceypy.furnsh() does not affect this library: campanella links its own CSPICE.
    campanella::init_spice();

    m.def("furnsh", &campanella::furnsh, py::arg("path"),
          "Load a SPICE kernel into campanella's CSPICE. spiceypy.furnsh does not.");

    const SpiceOptions d;
    py::class_<SpiceParameters>(m, "SpiceParameters")
        .def(py::init([](double et0, std::vector<std::string> naif_ids, std::string naif_frame,
                         std::optional<std::vector<double>> GMs, std::optional<double> DU,
                         bool use_canonical_scales, std::string abcorr, bool include_srp,
                         double srp_A, double srp_m, double srp_rhoS, double srp_rhoD,
                         std::string sh_file, int sh_nmax, int sh_mmax, std::string sh_frame,
                         double sh_rho_min) {
                 SpiceOptions o;
                 o.naif_frame = std::move(naif_frame);
                 if (GMs) {
                     o.GMs = std::move(*GMs);
                 }
                 o.DU = DU;
                 o.use_canonical_scales = use_canonical_scales;
                 o.abcorr = std::move(abcorr);
                 o.include_srp = include_srp;
                 o.srp_A = srp_A;
                 o.srp_m = srp_m;
                 o.srp_rhoS = srp_rhoS;
                 o.srp_rhoD = srp_rhoD;
                 o.sh_file = std::move(sh_file);
                 o.sh_nmax = sh_nmax;
                 o.sh_mmax = sh_mmax;
                 o.sh_frame = std::move(sh_frame);
                 o.sh_rho_min = sh_rho_min;
                 return SpiceParameters(et0, std::move(naif_ids), std::move(o));
             }),
             py::arg("et0"), py::arg("naif_ids"), py::kw_only(),
             py::arg("naif_frame") = d.naif_frame, py::arg("GMs") = py::none(), py::arg("DU") = py::none(),
             py::arg("use_canonical_scales") = d.use_canonical_scales, py::arg("abcorr") = d.abcorr,
             py::arg("include_srp") = d.include_srp, py::arg("srp_A") = d.srp_A, py::arg("srp_m") = d.srp_m,
             py::arg("srp_rhoS") = d.srp_rhoS, py::arg("srp_rhoD") = d.srp_rhoD,
             py::arg("sh_file") = d.sh_file, py::arg("sh_nmax") = d.sh_nmax, py::arg("sh_mmax") = d.sh_mmax,
             py::arg("sh_frame") = d.sh_frame, py::arg("sh_rho_min") = d.sh_rho_min,
             R"mydelimiter(
                Dynamics model: central body, third bodies, SRP, spherical harmonics and units.
                Immutable. States, times and outputs are in DU, DU/TU and TU; et = et0 + t * TU.

                Raises ValueError on bad input and RuntimeError with NAIF's message on a SPICE error.

                Args:
                    et0 (float): reference epoch, TDB seconds past J2000
                    naif_ids (list): NAIF names or numeric ids, e.g. ["399", "MOON", "10"];
                        the first one is the central body, the others are third bodies
                    naif_frame (str): inertial frame of integration
                    GMs (list): gravitational parameters in km^3/s^2, one per NAIF id;
                        None (default) reads "GM" from the loaded kernels (e.g. gm_de440.tpc)
                    DU (float): distance unit in km; with use_canonical_scales, VU = sqrt(GMs[0] / DU)
                        and TU = DU / VU. None (default) means DU = TU = VU = 1 (km, s, km/s)
                    use_canonical_scales (bool): if False, DU = TU = VU = 1 even if DU is given
                    abcorr (str): aberration correction; only "NONE" is accepted
                    include_srp (bool): solar radiation pressure; requires the Sun ("10") in naif_ids
                    srp_A (float): spacecraft area, m^2
                    srp_m (float): spacecraft mass, kg
                    srp_rhoS (float): specular reflection coefficient
                    srp_rhoD (float): diffuse reflection coefficient
                    sh_file (str): central-body gravity field, PDS SHADR .tab or ICGEM .gfc; "" (default) for none
                    sh_nmax (int): truncation degree of the gravity field, >= 2 if sh_file is given
                    sh_mmax (int): truncation order; -1 (default) means min(sh_nmax, order of the file)
                    sh_frame (str): body-fixed frame of the gravity field, e.g. "MOON_PA" or "ITRF93"
                    sh_rho_min (float): smallest |r|/R at which the field is evaluated, in (0, 1], default 1;
                        a propagation that goes below it raises ValueError. Values slightly below 1
                        (e.g. 0.99) allow low or landing arcs, but the series may not converge there
             )mydelimiter")
        .def_property_readonly("et0", &SpiceParameters::et0)
        .def_property_readonly("naif_ids", &SpiceParameters::naif_ids)
        .def_property_readonly("naif_codes", &SpiceParameters::naif_codes)
        .def_property_readonly("naif_frame", &SpiceParameters::naif_frame)
        .def_property_readonly("GMs", &SpiceParameters::GMs, "gravitational parameters, km^3/s^2")
        .def_property_readonly("mus", &SpiceParameters::mus, "GMs * TU^2 / DU^3")
        .def_property_readonly("DU", &SpiceParameters::DU, "distance unit, km")
        .def_property_readonly("TU", &SpiceParameters::TU, "time unit, s")
        .def_property_readonly("VU", &SpiceParameters::VU, "velocity unit, km/s")
        .def_property_readonly("use_canonical_scales",
                               [](const SpiceParameters& p) { return p.options().use_canonical_scales; })
        .def_property_readonly("abcorr", [](const SpiceParameters& p) { return p.options().abcorr; })
        .def_property_readonly("include_srp", &SpiceParameters::include_srp)
        .def_property_readonly("srp_A", [](const SpiceParameters& p) { return p.options().srp_A; })
        .def_property_readonly("srp_m", [](const SpiceParameters& p) { return p.options().srp_m; })
        .def_property_readonly("srp_rhoS", [](const SpiceParameters& p) { return p.options().srp_rhoS; })
        .def_property_readonly("srp_rhoD", [](const SpiceParameters& p) { return p.options().srp_rhoD; })
        .def_property_readonly("include_spherical_harmonics", &SpiceParameters::include_spherical_harmonics)
        .def_property_readonly("sh_file", [](const SpiceParameters& p) { return p.options().sh_file; })
        .def_property_readonly("sh_nmax", [](const SpiceParameters& p) { return p.options().sh_nmax; })
        .def_property_readonly("sh_mmax", [](const SpiceParameters& p) { return p.options().sh_mmax; })
        .def_property_readonly("sh_frame", [](const SpiceParameters& p) { return p.options().sh_frame; })
        .def_property_readonly("sh_rho_min", [](const SpiceParameters& p) { return p.options().sh_rho_min; })
        .def("__repr__", [](const SpiceParameters& p) {
            std::ostringstream s;
            s.precision(16);
            s << "SpiceParameters(et0=" << p.et0() << ", naif_ids=[";
            for (size_t i = 0; i < p.naif_ids().size(); i++) {
                s << (i ? ", " : "") << "'" << p.naif_ids()[i] << "'";
            }
            s << "], naif_frame='" << p.naif_frame() << "', DU=" << p.DU() << ", TU=" << p.TU()
              << ", VU=" << p.VU() << ", include_srp=" << (p.include_srp() ? "True" : "False")
              << ", include_spherical_harmonics=" << (p.include_spherical_harmonics() ? "True" : "False")
              << ")";
            return s.str();
        });

    py::class_<IvpResult>(m, "IvpResult",
                          "Solution of an initial value problem: t (N,), x (N, 6) and stm (N, 6, 6) or None")
        .def_property_readonly("t", [](py::object self) {
            const IvpResult& r = self.cast<const IvpResult&>();
            return view({static_cast<py::ssize_t>(r.size())}, r.t, self);
        }, "times, in TU")
        .def_property_readonly("x", [](py::object self) {
            const IvpResult& r = self.cast<const IvpResult&>();
            return view({static_cast<py::ssize_t>(r.size()), 6}, r.x, self);
        }, "states: Cartesian in DU, DU/TU, or MEE [p (DU), f, g, h, k, L] with L not wrapped")
        .def_property_readonly("stm", [](py::object self) -> py::object {
            const IvpResult& r = self.cast<const IvpResult&>();
            if (r.stm.empty()) {
                return py::none();
            }
            return view({static_cast<py::ssize_t>(r.size()), 6, 6}, r.stm, self);
        }, "state-transition matrices Phi(t, t0), or None for solve_ivp_*")
        .def("__len__", &IvpResult::size);

    const char* solve_doc = R"mydelimiter(
            Args:
                params (SpiceParameters): dynamics model
                tspan (list): [t0, tf] in TU since et0; tf < t0 integrates backward
                x0 (list): initial state (6)
                atol (float): absolute error tolerance, default 1e-12
                rtol (float): relative error tolerance, default 1e-14
                t_eval (list): output times in TU, strictly monotonic in the direction of integration
                    and within tspan; None (default) returns every accepted step

            Returns:
                (IvpResult): t (N,), x (N, 6) and stm (N, 6, 6), or None without STM

            Raises ValueError on bad input or if the trajectory leaves the domain (below sh_rho_min * R,
            or p <= 0 / w <= 0 in MEE), and RuntimeError on a SPICE error or if the integration fails.
        )mydelimiter";
    def_solver(m, "solve_ivp_rv", &solve_ivp_rv,
               (std::string("Propagate a Cartesian state [x, y, z, vx, vy, vz]\n") + solve_doc).c_str());
    def_solver(m, "solve_ivp_mee", &solve_ivp_mee,
               (std::string("Propagate MEE [p, f, g, h, k, L] with the Gauss variational equations\n")
                + solve_doc).c_str());
    def_solver(m, "solve_ivpstm_rv", &solve_ivpstm_rv,
               (std::string("Propagate a Cartesian state and its STM, from Phi(t0, t0) = I\n") + solve_doc).c_str());
    def_solver(m, "solve_ivpstm_mee", &solve_ivpstm_mee,
               (std::string("Propagate MEE and their STM, from Phi(t0, t0) = I\n") + solve_doc).c_str());

    def_eom(m, "eom_rv", &eom_rv, 6, "Cartesian equations of motion: xdot (6) at t (TU) and x (6)");
    def_eom(m, "eom_mee", &eom_mee, 6, "GVE in MEE: xdot (6) at t (TU) and x (6)");
    def_eom(m, "eomstm_rv", &eomstm_rv, 42,
            "Cartesian equations of motion with STM: xdot (42) at t (TU) and x (42: state, then Phi row-major)");
    def_eom(m, "eomstm_mee", &eomstm_mee, 42,
            "GVE in MEE with STM: xdot (42) at t (TU) and x (42: state, then Phi row-major)");

    m.def("eval_rv", [](const SpiceParameters& p, double t, const std::vector<double>& x) {
              return eval_tuple(eval_rv(p, t, x));
          }, py::arg("params"), py::arg("t"), py::arg("x"),
          R"mydelimiter(
            Evaluate the Cartesian equations of motion and their Jacobian at a single time and state

            Raises ValueError below sh_rho_min * R, and RuntimeError with NAIF's message on a SPICE error.

            Returns:
                (tuple): (xdot (6,), A (6, 6)) with A = d(xdot)/dx
          )mydelimiter");
    m.def("eval_mee", [](const SpiceParameters& p, double t, const std::vector<double>& x) {
              return eval_tuple(eval_mee(p, t, x));
          }, py::arg("params"), py::arg("t"), py::arg("x"),
          R"mydelimiter(
            Evaluate the GVE in MEE and their Jacobian at a single time and state

            Raises ValueError if p <= 0 or w <= 0, or below sh_rho_min * R, and RuntimeError with
            NAIF's message on a SPICE error.

            Returns:
                (tuple): (xdot (6,), A (6, 6)) with A = d(xdot)/dx
          )mydelimiter");

    m.def("load_spherical_harmonics",
          [](const std::string& path, int nmax, int mmax) {
              std::shared_ptr<const SphericalHarmonicsModel> model =
                  load_spherical_harmonics_cached(path, nmax, mmax);
              py::list C, S;
              for (int n = 0; n <= model->nmax; n++) {
                  py::list Cn, Sn;
                  for (int k = 0; k <= n; k++) {
                      Cn.append(model->C[sh_index(n, k)]);
                      Sn.append(model->S[sh_index(n, k)]);
                  }
                  C.append(Cn);
                  S.append(Sn);
              }
              py::dict info;
              info["source"] = model->source;
              info["GM"] = model->GM_km3s2;
              info["R"] = model->R_km;
              info["file_degree"] = model->file_degree;
              info["file_order"] = model->file_order;
              info["nmax"] = model->nmax;
              info["mmax"] = model->mmax;
              info["tide_system"] = model->tide_system;
              info["C"] = C;
              info["S"] = S;
              return info;
          },
          py::arg("path"), py::arg("nmax"), py::arg("mmax") = -1,
          R"mydelimiter(
            Load a gravity field file (PDS SHADR .tab or ICGEM .gfc, auto-detected)

            Raises RuntimeError for a malformed or truncated file, ValueError for nmax < 2,
            nmax > 1200 or nmax above the degree of the file.

            Args:
                path (str): gravity field file
                nmax (int): truncation degree
                mmax (int): truncation order; -1 means min(nmax, order of the file)

            Returns:
                (dict): GM (km^3/s^2), R (km), file_degree, file_order, nmax, mmax, tide_system, source, and the
                fully normalized coefficients C[n][m], S[n][m] for n <= nmax (zero for m > mmax; C[0][0] = 1)
          )mydelimiter");

    m.def("sh_accel_bf", &campanella::sh_accel_bf,
          py::arg("path"), py::arg("nmax"), py::arg("r_bf"), py::arg("mmax") = -1,
          py::arg("mu") = 0.0, py::arg("R") = 0.0, py::arg("rho_min") = 1.0,
          R"mydelimiter(
            Spherical-harmonics potential, acceleration and gradient in the body-fixed frame

            Degrees 2..nmax; the point mass is not included. Raises ValueError below rho_min * R.

            Args:
                path (str): gravity field file
                nmax (int): truncation degree
                r_bf (list): body-fixed position, in km unless R is given
                mmax (int): truncation order; -1 means min(nmax, order of the file)
                mu (double): gravitational parameter; 0 means the GM of the file (km^3/s^2)
                R (double): reference radius in the unit of r_bf; 0 means the radius of the file (km).
                    Give both mu and R, or neither (ValueError otherwise)
                rho_min (double): smallest |r_bf|/R evaluated, in (0, 1]

            Returns:
                (tuple): (U, a, G), with G = d(a)/d(r_bf) flattened row-major (9 elements)
          )mydelimiter");

    m.def("sh_accel_inertial", &campanella::sh_accel_inertial,
          py::arg("path"), py::arg("nmax"), py::arg("naif_frame"), py::arg("sh_frame"), py::arg("et"),
          py::arg("r"), py::arg("DU") = 1.0, py::arg("TU") = 1.0, py::arg("mmax") = -1,
          py::arg("rho_min") = 1.0,
          R"mydelimiter(
            Spherical-harmonics acceleration and gradient in an inertial frame, as used by the EOMs

            Rotates with SPICE pxform(naif_frame, sh_frame, et): a = T^T a_bf, G = T^T G_bf T.
            Needs the kernels of sh_frame (see docs/spherical_harmonics.md). Raises RuntimeError on a
            SPICE error and ValueError below rho_min * R.

            Args:
                path (str): gravity field file
                nmax (int): truncation degree
                naif_frame (str): inertial frame of r, e.g. "J2000"
                sh_frame (str): body-fixed frame of the gravity field, e.g. "MOON_PA", "ITRF93"
                et (double): epoch, in ephemeris seconds
                r (list): position, in DU
                DU (double): distance unit, km
                TU (double): time unit, s
                mmax (int): truncation order; -1 means min(nmax, order of the file)
                rho_min (double): smallest |r|/R evaluated, in (0, 1]

            Returns:
                (tuple): (a, G), a in DU/TU^2 and G = d(a)/d(r) in 1/TU^2 flattened row-major (9 elements)
          )mydelimiter");

    m.def("mee2cart", &campanella::mee2cart, py::arg("mee"), py::arg("mu"),
          R"mydelimiter(
            Convert modified equinoctial elements (MEE) to a Cartesian state

            MEE are prograde (retrograde factor I = +1). Raises ValueError if p <= 0 or
            w = 1 + f cos(L) + g sin(L) <= 0.

            Args:
                mee (list): MEE, in order [p,f,g,h,k,L]; L may be any real number
                mu (double): gravitational parameter

            Returns:
                (list): Cartesian state [x,y,z,vx,vy,vz]
          )mydelimiter");

    m.def("cart2mee", &campanella::cart2mee, py::arg("rv"), py::arg("mu"),
          R"mydelimiter(
            Convert a Cartesian state to modified equinoctial elements (MEE)

            MEE are prograde (retrograde factor I = +1). Raises ValueError for zero angular
            momentum or an inclination within about 1.4e-6 rad of 180 deg.

            Args:
                rv (list): Cartesian state [x,y,z,vx,vy,vz]
                mu (double): gravitational parameter

            Returns:
                (list): MEE [p,f,g,h,k,L], with L in (-pi, pi]
          )mydelimiter");

    m.def("mee2cart_jacobian", &campanella::mee2cart_jacobian, py::arg("mee"), py::arg("mu"),
          R"mydelimiter(
            Jacobian of mee2cart, d(x,y,z,vx,vy,vz)/d(p,f,g,h,k,L)

            Args:
                mee (list): MEE, in order [p,f,g,h,k,L]
                mu (double): gravitational parameter

            Returns:
                (list): 6x6 Jacobian, flattened row-major (36 elements)
          )mydelimiter");
}
