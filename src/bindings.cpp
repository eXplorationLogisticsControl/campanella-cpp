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

using SolverZoh = IvpResult (*)(const SpiceParameters&, const ThrustOptions&, const std::vector<double>&,
                                const std::vector<double>&, const std::vector<double>&, const IvpOptions&);

// Runs between integration steps with the GIL held: lets Ctrl-C (or any Python signal handler) stop an
// rvm or meem integration. The exception unwinds integrate() only, never a GSL frame.
void check_python_signals() {
    if (PyErr_CheckSignals() != 0) {
        throw py::error_already_set();
    }
}

// Python signature solve(params, thrust, tspan, x0, u, *, atol, rtol, t_eval, max_steps)
void def_solver_zoh(py::module_& m, const char* name, SolverZoh solver, const char* doc) {
    m.def(name,
          [solver](const SpiceParameters& params, const ThrustOptions& thrust, const std::vector<double>& tspan,
                   const std::vector<double>& x0, const std::vector<double>& u, double atol, double rtol,
                   std::optional<std::vector<double>> t_eval, long long max_steps) {
              IvpOptions options;
              options.atol = atol;
              options.rtol = rtol;
              if (t_eval) {
                  options.t_eval = std::move(*t_eval);
              }
              options.max_steps = max_steps;
              options.poll = &check_python_signals;
              return solver(params, thrust, tspan, x0, u, options);
          },
          py::arg("params"), py::arg("thrust"), py::arg("tspan"), py::arg("x0"), py::arg("u"), py::kw_only(),
          py::arg("atol") = IvpOptions{}.atol, py::arg("rtol") = IvpOptions{}.rtol,
          py::arg("t_eval") = py::none(), py::arg("max_steps") = IvpOptions{}.max_steps, doc);
}

using RefArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// A 2-D array with the given number of columns, flattened row-major. The number of rows is checked by
// propagate_zoh_*, against times.
std::vector<double> flatten_rows(const RefArray& a, py::ssize_t cols, const char* name, const char* who) {
    if (a.ndim() != 2 || a.shape(1) != cols) {
        std::ostringstream msg;
        msg << who << ": " << name << " must be a 2-D array with " << cols << " columns, got shape (";
        for (py::ssize_t i = 0; i < a.ndim(); i++) {
            msg << (i ? ", " : "") << a.shape(i);
        }
        msg << ")";
        throw std::invalid_argument(msg.str());
    }
    return std::vector<double>(a.data(), a.data() + a.size());
}

using Propagate = ZohLinearization (*)(const SpiceParameters&, const ThrustOptions&, const std::vector<double>&,
                                       const std::vector<double>&, const std::vector<double>&, const IvpOptions&);

// Python signature propagate(params, thrust, times, x_ref, u_ref, *, atol, rtol, max_steps)
void def_propagate_zoh(py::module_& m, const char* name, Propagate propagate, const char* doc) {
    m.def(name,
          [propagate, name](const SpiceParameters& params, const ThrustOptions& thrust,
                            const std::vector<double>& times, const RefArray& x_ref, const RefArray& u_ref,
                            double atol, double rtol, long long max_steps) {
              const std::vector<double> x = flatten_rows(x_ref, 7, "x_ref", name);
              const std::vector<double> u = flatten_rows(u_ref, 4, "u_ref", name);
              IvpOptions options;
              options.atol = atol;
              options.rtol = rtol;
              options.max_steps = max_steps;
              options.poll = &check_python_signals;
              return propagate(params, thrust, times, x, u, options);
          },
          py::arg("params"), py::arg("thrust"), py::arg("times"), py::arg("x_ref"), py::arg("u_ref"), py::kw_only(),
          py::arg("atol") = IvpOptions{}.atol, py::arg("rtol") = IvpOptions{}.rtol,
          py::arg("max_steps") = IvpOptions{}.max_steps, doc);
}

// Python signature eom(params, thrust, t, x, u) -> xdot; raises like eval_* on a failure.
// The ControlledParameters is local to the call, and is not exposed to Python.
void def_eom_zoh(py::module_& m, const char* name, Eom eom, size_t dim, const char* doc) {
    m.def(name,
          [eom, dim, name](const SpiceParameters& params, const ThrustOptions& thrust, double t,
                           const std::vector<double>& x, const std::vector<double>& u) {
              detail::make_control(params, thrust, u, name);   // so that errors name this function
              if (x.size() != dim) {
                  throw std::invalid_argument(std::string(name) + ": x must have length " + std::to_string(dim));
              }
              const ControlledParameters cp(params, thrust, u);
              py::array_t<double> xdot(std::vector<py::ssize_t>{static_cast<py::ssize_t>(dim)});
              int status = eom(t, x.data(), xdot.mutable_data(), const_cast<ControlledParameters*>(&cp));
              if (status != detail::RHS_SUCCESS) {
                  detail::throw_fault(detail::Access::model(params), detail::Access::workspace(params),
                                      status, t, name);
              }
              return xdot;
          },
          py::arg("params"), py::arg("thrust"), py::arg("t"), py::arg("x"), py::arg("u"), doc);
}

py::tuple eval_tuple_zoh(const EomEvalControl& e) {
    py::array_t<double> xdot(std::vector<py::ssize_t>{7}, e.xdot.data());
    py::array_t<double> A(std::vector<py::ssize_t>{7, 7}, e.A.data());
    py::array_t<double> B(std::vector<py::ssize_t>{7, 4}, e.B.data());
    return py::make_tuple(xdot, A, B);
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

    py::enum_<ThrustFrame>(m, "ThrustFrame",
                           "Frame of the throttle vector tau of the meem dynamics. rtn: radial, transverse and "
                           "normal directions of the current state, so a constant tau rotates in inertial space. "
                           "inertial: naif_frame, as rvm.")
        .value("rtn", ThrustFrame::rtn)
        .value("inertial", ThrustFrame::inertial);

    const ThrustOptions thrust_defaults;
    py::class_<ThrustOptions>(m, "ThrustOptions")
        .def(py::init([](double thrust_N, double isp_s, double mass_unit_kg, double g0, ThrustFrame mee_control_frame) {
                 ThrustOptions o;
                 o.thrust_N = thrust_N;
                 o.isp_s = isp_s;
                 o.mass_unit_kg = mass_unit_kg;
                 o.g0 = g0;
                 o.mee_control_frame = mee_control_frame;
                 return o;
             }),
             py::kw_only(), py::arg("thrust_N"), py::arg("isp_s"), py::arg("mass_unit_kg"),
             py::arg("g0") = thrust_defaults.g0, py::arg("mee_control_frame") = thrust_defaults.mee_control_frame,
             R"mydelimiter(
                Propulsion system of the mass-augmented dynamics (rvm, meem). Immutable.

                The control is u = [tau, Gamma], both dimensionless: the thrust is T_max tau, and the state
                has the mass m in units of mass_unit_kg (m_bar), with
                    thrust acceleration = (c1 / m) tau, in DU/TU^2,    mdot = -c2 Gamma, in mass units per TU
                    c1 = (T_max [kN]) / (m_bar DU / TU^2),    c2 = (T_max [N]) / (Isp g0) * TU / m_bar

                The dynamics treat Gamma as independent of tau: the solver of the optimal control problem is
                responsible for Gamma = |tau|, and neither Gamma < 0 nor |tau| > Gamma is rejected.
                SRP uses the state mass, m_bar m. srp_m of SpiceParameters is ignored, but still required
                to be > 0 if include_srp.

                The values are checked where they are used, not here: the rvm and meem functions raise
                ValueError unless thrust_N, isp_s, mass_unit_kg and g0 are finite and > 0.

                Args:
                    thrust_N (float): maximum thrust T_max, in N
                    isp_s (float): specific impulse, in s
                    mass_unit_kg (float): mass unit m_bar, in kg
                    g0 (float): standard gravity, in m/s^2
                    mee_control_frame (ThrustFrame): frame of tau for meem, RTN by default;
                        rvm is always in naif_frame
             )mydelimiter")
        .def_readonly("thrust_N", &ThrustOptions::thrust_N, "maximum thrust, N")
        .def_readonly("isp_s", &ThrustOptions::isp_s, "specific impulse, s")
        .def_readonly("mass_unit_kg", &ThrustOptions::mass_unit_kg, "mass unit, kg")
        .def_readonly("g0", &ThrustOptions::g0, "standard gravity, m/s^2")
        .def_readonly("mee_control_frame", &ThrustOptions::mee_control_frame, "frame of tau for meem")
        .def("__repr__", [](const ThrustOptions& o) {
            std::ostringstream s;
            s.precision(16);
            s << "ThrustOptions(thrust_N=" << o.thrust_N << ", isp_s=" << o.isp_s
              << ", mass_unit_kg=" << o.mass_unit_kg << ", g0=" << o.g0 << ", mee_control_frame=ThrustFrame."
              << (o.mee_control_frame == ThrustFrame::rtn ? "rtn" : "inertial") << ")";
            return s.str();
        });

    py::class_<IvpResult>(m, "IvpResult",
                          "Solution of an initial value problem: t (N,), x (N, nx), stm (N, nx, nx) or None, and "
                          "stm_u (N, nx, nu) or None, with nx = 6 and nu = 0 (rv, mee) or nx = 7 and nu = 4 (rvm, meem)")
        .def_property_readonly("t", [](py::object self) {
            const IvpResult& r = self.cast<const IvpResult&>();
            return view({static_cast<py::ssize_t>(r.size())}, r.t, self);
        }, "times, in TU")
        .def_property_readonly("x", [](py::object self) {
            const IvpResult& r = self.cast<const IvpResult&>();
            return view({static_cast<py::ssize_t>(r.size()), static_cast<py::ssize_t>(r.nx)}, r.x, self);
        }, "states: Cartesian in DU, DU/TU, or MEE [p (DU), f, g, h, k, L] with L not wrapped; "
           "followed by the mass, in mass units, for rvm and meem")
        .def_property_readonly("stm", [](py::object self) -> py::object {
            const IvpResult& r = self.cast<const IvpResult&>();
            if (r.stm.empty()) {
                return py::none();
            }
            const py::ssize_t nx = static_cast<py::ssize_t>(r.nx);
            return view({static_cast<py::ssize_t>(r.size()), nx, nx}, r.stm, self);
        }, "state-transition matrices Phi(t, t0), or None for solve_ivp_*")
        .def_property_readonly("stm_u", [](py::object self) -> py::object {
            const IvpResult& r = self.cast<const IvpResult&>();
            if (r.stm_u.empty()) {
                return py::none();
            }
            return view({static_cast<py::ssize_t>(r.size()), static_cast<py::ssize_t>(r.nx),
                         static_cast<py::ssize_t>(r.nu)}, r.stm_u, self);
        }, "control sensitivities Psi(t) = dx(t)/du, or None unless solve_ivpstm_rvm or solve_ivpstm_meem")
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

    const char* solve_zoh_doc = R"mydelimiter(
            Args:
                params (SpiceParameters): dynamics model
                thrust (ThrustOptions): propulsion system
                tspan (list): [t0, tf] in TU since et0; tf < t0 integrates backward
                x0 (list): initial state (7), with the mass last, in mass units
                u (list): control [tau_1, tau_2, tau_3, Gamma], dimensionless, held over tspan
                atol (float): absolute error tolerance, default 1e-12
                rtol (float): relative error tolerance, default 1e-14
                t_eval (list): output times in TU, strictly monotonic in the direction of integration
                    and within tspan; None (default) returns every accepted step
                max_steps (int): largest number of accepted steps, default 2000000; 0 disables the limit

            Returns:
                (IvpResult): t (N,), x (N, 7), stm (N, 7, 7) and stm_u (N, 7, 4), the last two None without
                sensitivities. Phi(t0, t0) = I, Psi(t0) = 0, Phi_dot = A Phi, Psi_dot = A Psi + B

            Raises ValueError on bad input, if m0 is not > 0 or the closed-form mass is not > 0 at the last
            output time (checked before integrating), or if the trajectory leaves the domain (below
            sh_rho_min * R, or p <= 0 / w <= 0 in MEE), and RuntimeError on a SPICE error, if the integration
            fails, or if max_steps is reached (the steps collapse as p -> 0 when thrust deorbits the trajectory).
            Python signals, such as Ctrl-C, are handled every 1024 steps.
        )mydelimiter";
    def_solver_zoh(m, "solve_ivp_rvm", &solve_ivp_rvm,
                   (std::string("Propagate a Cartesian state and mass [x, y, z, vx, vy, vz, m] with a constant "
                                "control, with tau in naif_frame\n") + solve_zoh_doc).c_str());
    def_solver_zoh(m, "solve_ivp_meem", &solve_ivp_meem,
                   (std::string("Propagate MEE and mass [p, f, g, h, k, L, m] with a constant control, "
                                "with tau in thrust.mee_control_frame\n") + solve_zoh_doc).c_str());
    def_solver_zoh(m, "solve_ivpstm_rvm", &solve_ivpstm_rvm,
                   (std::string("Propagate a Cartesian state and mass with a constant control, with their "
                                "STM and control sensitivity\n") + solve_zoh_doc).c_str());
    def_solver_zoh(m, "solve_ivpstm_meem", &solve_ivpstm_meem,
                   (std::string("Propagate MEE and mass with a constant control, with their STM and "
                                "control sensitivity\n") + solve_zoh_doc).c_str());

    py::class_<ZohLinearization>(m, "ZohLinearization",
                                 "Linearization x_{k+1} = Phi_k x_k + Psi_k u_k + c_k of N - 1 zero-order-hold "
                                 "segments. All fields are read-only arrays with N - 1 rows; len() is N - 1.")
        .def_property_readonly("x_end", [](py::object self) {
            const ZohLinearization& z = self.cast<const ZohLinearization&>();
            return view({static_cast<py::ssize_t>(z.x_end.size()/7), 7}, z.x_end, self);
        }, "state at the end of each segment (N - 1, 7)")
        .def_property_readonly("Phi", [](py::object self) {
            const ZohLinearization& z = self.cast<const ZohLinearization&>();
            return view({static_cast<py::ssize_t>(z.Phi.size()/49), 7, 7}, z.Phi, self);
        }, "state-transition matrices (N - 1, 7, 7)")
        .def_property_readonly("Psi", [](py::object self) {
            const ZohLinearization& z = self.cast<const ZohLinearization&>();
            return view({static_cast<py::ssize_t>(z.Psi.size()/28), 7, 4}, z.Psi, self);
        }, "control sensitivities (N - 1, 7, 4)")
        .def_property_readonly("c", [](py::object self) {
            const ZohLinearization& z = self.cast<const ZohLinearization&>();
            return view({static_cast<py::ssize_t>(z.c.size()/7), 7}, z.c, self);
        }, "affine terms x_end_k - Phi_k x_ref_k - Psi_k u_ref_k (N - 1, 7)")
        .def_property_readonly("defects", [](py::object self) {
            const ZohLinearization& z = self.cast<const ZohLinearization&>();
            return view({static_cast<py::ssize_t>(z.defects.size()/7), 7}, z.defects, self);
        }, "x_ref_{k+1} - x_end_k (N - 1, 7)")
        .def("__len__", [](const ZohLinearization& z) { return z.x_end.size()/7; });

    const char* propagate_zoh_doc = R"mydelimiter(
            Each segment starts from its own row of x_ref, not from the end of the previous one.

            Args:
                params (SpiceParameters): dynamics model
                thrust (ThrustOptions): propulsion system
                times (list): segment boundaries in TU (N), strictly monotonic; they may decrease, for a
                    backward integration, with u_ref[k] held from times[k] down to times[k + 1]
                x_ref (array): reference states (N, 7), with the mass last. The mass of the last row is
                    not required to be > 0, as that row only enters the last defect. L is not wrapped, so
                    a wrapped L in x_ref gives defects of multiples of 2 pi
                u_ref (array): controls [tau_1, tau_2, tau_3, Gamma] (N - 1, 4)
                atol (float): absolute error tolerance, default 1e-12
                rtol (float): relative error tolerance, default 1e-14
                max_steps (int): largest number of accepted steps per segment, default 2000000;
                    0 disables the limit

            Returns:
                (ZohLinearization): x_end, Phi, Psi, c and defects, read-only

            Raises ValueError on bad input or shapes, or if a segment leaves the domain, and RuntimeError on a
            SPICE error, if the integration fails, or if max_steps is reached. The message of a failed segment
            starts with "<function>: segment k of n (t = [a, b] TU): ". Python signals, such as Ctrl-C, are
            handled every 1024 steps.
        )mydelimiter";
    def_propagate_zoh(m, "propagate_zoh_rvm", &propagate_zoh_rvm,
                      (std::string("Linearize zero-order-hold segments in Cartesian state and mass\n")
                       + propagate_zoh_doc).c_str());
    def_propagate_zoh(m, "propagate_zoh_meem", &propagate_zoh_meem,
                      (std::string("Linearize zero-order-hold segments in MEE and mass\n")
                       + propagate_zoh_doc).c_str());

    def_eom_zoh(m, "eom_rvm", &eom_rvm, 7,
                "Cartesian + mass equations of motion: xdot (7) at t (TU), x (7) and u (4)");
    def_eom_zoh(m, "eom_meem", &eom_meem, 7, "GVE in MEE + mass: xdot (7) at t (TU), x (7) and u (4)");
    def_eom_zoh(m, "eomstm_rvm", &eomstm_rvm, 84,
                "Cartesian + mass equations of motion with sensitivities: xdot (84) at t (TU), u (4) and "
                "x (84: state, then Phi (7x7) and Psi (7x4), both row-major)");
    def_eom_zoh(m, "eomstm_meem", &eomstm_meem, 84,
                "GVE in MEE + mass with sensitivities: xdot (84) at t (TU), u (4) and "
                "x (84: state, then Phi (7x7) and Psi (7x4), both row-major)");

    m.def("eval_rvm", [](const SpiceParameters& p, const ThrustOptions& thrust, double t,
                         const std::vector<double>& x, const std::vector<double>& u) {
              return eval_tuple_zoh(eval_rvm(p, thrust, t, x, u));
          }, py::arg("params"), py::arg("thrust"), py::arg("t"), py::arg("x"), py::arg("u"),
          R"mydelimiter(
            Evaluate the Cartesian + mass equations of motion and their Jacobians at a single time, state and control

            Raises ValueError on bad input, if m is not > 0 or not finite, or below sh_rho_min * R, and
            RuntimeError with NAIF's message on a SPICE error.

            Returns:
                (tuple): (xdot (7,), A (7, 7), B (7, 4)) with A = d(xdot)/dx and B = d(xdot)/du
          )mydelimiter");
    m.def("eval_meem", [](const SpiceParameters& p, const ThrustOptions& thrust, double t,
                          const std::vector<double>& x, const std::vector<double>& u) {
              return eval_tuple_zoh(eval_meem(p, thrust, t, x, u));
          }, py::arg("params"), py::arg("thrust"), py::arg("t"), py::arg("x"), py::arg("u"),
          R"mydelimiter(
            Evaluate the GVE in MEE + mass and their Jacobians at a single time, state and control

            Raises ValueError on bad input, if m is not > 0 or not finite, if p <= 0 or w <= 0, or below
            sh_rho_min * R, and RuntimeError with NAIF's message on a SPICE error.

            Returns:
                (tuple): (xdot (7,), A (7, 7), B (7, 4)) with A = d(xdot)/dx and B = d(xdot)/du
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
