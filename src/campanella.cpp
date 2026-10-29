/*
 * campanella.cpp
 */

#include <iostream>
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // enable automatic conversion between C++ STL types and Python types
#include "SpiceUsr.h"

#include "campanella.hpp"

namespace py = pybind11;


int add_j2000(int i, int j) {
    // test SPICE
    double j2000_date = j2000_c();
    std::cout << "J2000 date: " << j2000_date << std::endl;

    // double moon_state[3];
    // double lt;
    // spkpos_c("MOON", j2000_date, "J2000", "NONE", "EARTH", moon_state, &lt);

    return i + j;
}


PYBIND11_MODULE(campanella, m) {
    m.doc() = "campanella is a c++ library for astrodynamics with python binding from SSOG";

    // Initialize CSPICE when the module is loaded.
    // Kernels are not loaded here. Call campanella.furnsh() from Python.
    // spiceypy.furnsh() does not affect this library: campanella links its own CSPICE.
    campanella::init_spice();

    m.def("add_j2000", &add_j2000,
          py::arg("i"), py::arg("j"),
          "A function that adds two numbers");

    m.def("furnsh", &campanella::furnsh, py::arg("path"),
          "Load a SPICE kernel into campanella's CSPICE. spiceypy.furnsh does not.");

    py::class_<campanella::PropagatorTwoBody>(m, "PropagatorTwoBody")
        .def(py::init<double, double, double>(),
             py::arg("GM"), py::arg("reltol") = 1e-14, py::arg("abstol") = 1e-12,
             R"mydelimiter(
                PropagatorTwoBody class

                Propagates a spacecraft using two-body equations of motion

                Args:
                    GM (double): gravitational parameter
                    reltol (double): relative error tolerance, defaults to 1e-14
                    abstol (double): absolute error tolerance, defaults to 1e-12
            )mydelimiter")

        .def("solve_ivp", &campanella::PropagatorTwoBody::solve_ivp,
             py::arg("tspan"), py::arg("y0"),
             R"mydelimiter(
                Propagate spacecraft state using two-body equations of motion

                Args:
                    tspan (list): integration time span, [t0, tf]
                    y0 (list): initial state, in order [x,y,z,vx,vy,vz]

                Returns:
                    (list): list of tuples each containing (time, x, y, z, vx, vy, vz)
             )mydelimiter");

    py::class_<campanella::PropagatorNbody>(m, "PropagatorNbody")
        .def(py::init<std::vector<double>,
                      std::vector<std::string>,
                      std::string,
                      double,
                      double,
                      double,
                      double,
                      std::string,
                      int,
                      int,
                      std::string,
                      double>(),
             py::arg("GM"), py::arg("BODIES"), py::arg("frame"), py::arg("LU"), py::arg("TU"),
             py::arg("reltol") = 1e-14, py::arg("abstol") = 1e-12,
             py::arg("sh_file") = "", py::arg("sh_nmax") = 0, py::arg("sh_mmax") = -1,
             py::arg("frame_bf") = "", py::arg("sh_rho_min") = 1.0,
             R"mydelimiter(
                PropagatorNbody class

                Propagates a spacecraft using N-body equations of motion

                Args:
                    GM (list): list of gravitational parameters, in appropriate scale
                    BODIES (list): list strings corresponding to NAIF body names
                    frame (char): name of inertial reference frame
                    LU (double): canonical length unit
                    TU (double): canonical time unit
                    reltol (double): relative error tolerance, defaults to 1e-14
                    abstol (double): absolute error tolerance, defaults to 1e-12
                    sh_file (str): optional central-body gravity field, PDS SHADR .tab or ICGEM .gfc; "" (default) turns it off
                    sh_nmax (int): truncation degree of the gravity field, >= 2 if sh_file is given
                    sh_mmax (int): truncation order of the gravity field; -1 (default) means min(sh_nmax, order of the file)
                    frame_bf (str): body-fixed frame of the gravity field, e.g. "MOON_PA" or "ITRF93"
                    sh_rho_min (double): smallest |r|/R at which the field is evaluated, in (0, 1], default 1;
                        a propagation that goes below it raises ValueError. Values slightly below 1
                        (e.g. 0.99) allow low or landing arcs, but the series may not converge there
            )mydelimiter")

        .def("solve_ivp", &campanella::PropagatorNbody::solve_ivp,
             py::arg("et0"), py::arg("tspan"), py::arg("y0"),
             R"mydelimiter(
                Propagate spacecraft state using N-body equations of motion

                Args:
                    et0 (double): initial epoch
                    tspan (list): integration time span, [t0, tf]
                    y0 (list): initial state, in order [x,y,z,vx,vy,vz]

                Returns:
                    (list): list of tuples each containing (time, x, y, z, vx, vy, vz)
             )mydelimiter")

        .def("solve_ivp_stm", &campanella::PropagatorNbody::solve_ivp_stm,
             py::arg("et0"), py::arg("tspan"), py::arg("y0"),
             R"mydelimiter(
                Propagate spacecraft state using N-body equations of motion

                Args:
                    et0 (double): initial epoch
                    tspan (list): integration time span, [t0, tf]
                    y0 (list): initial state, in order [x,y,z,vx,vy,vz] + list(STM(0,0))

                Returns:
                    (list): list of tuples each containing (time, x, y, z, vx, vy, vz) + list(STM(t,t0))
             )mydelimiter");

    py::class_<campanella::PropagatorNbodySrp>(m, "PropagatorNbodySrp")
        .def(py::init<std::vector<double>,
                      std::vector<std::string>,
                      std::string,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      std::string,
                      int,
                      int,
                      std::string,
                      double>(),
             py::arg("GM"), py::arg("BODIES"), py::arg("frame"),
             py::arg("A"), py::arg("m"), py::arg("rhoS"), py::arg("rhoD"),
             py::arg("LU"), py::arg("TU"), py::arg("reltol") = 1e-14, py::arg("abstol") = 1e-12,
             py::arg("sh_file") = "", py::arg("sh_nmax") = 0, py::arg("sh_mmax") = -1,
             py::arg("frame_bf") = "", py::arg("sh_rho_min") = 1.0,
             R"mydelimiter(
                PropagatorNbodySrp class

                Propagates a spacecraft using N-body equations of motion

                Args:
                    GM (list): list of gravitational parameters, in appropriate scale
                    BODIES (list): list strings corresponding to NAIF body names
                    frame (char): name of inertial reference frame
                    A (double): surface area of spacecraft
                    m (double): mass of spacecraft
                    rhoS (double): reflection rate of spacecraft
                    rhoD (double): scattering rate of spacecraft
                    LU (double): canonical length unit
                    TU (double): canonical time unit
                    reltol (double): relative error tolerance, defaults to 1e-14
                    abstol (double): absolute error tolerance, defaults to 1e-12
                    sh_file (str): optional central-body gravity field, PDS SHADR .tab or ICGEM .gfc; "" (default) turns it off
                    sh_nmax (int): truncation degree of the gravity field, >= 2 if sh_file is given
                    sh_mmax (int): truncation order of the gravity field; -1 (default) means min(sh_nmax, order of the file)
                    frame_bf (str): body-fixed frame of the gravity field, e.g. "MOON_PA" or "ITRF93"
                    sh_rho_min (double): smallest |r|/R at which the field is evaluated, in (0, 1], default 1;
                        a propagation that goes below it raises ValueError. Values slightly below 1
                        (e.g. 0.99) allow low or landing arcs, but the series may not converge there
            )mydelimiter")

        .def("solve_ivp", &campanella::PropagatorNbodySrp::solve_ivp,
             py::arg("et0"), py::arg("tspan"), py::arg("y0"),
             R"mydelimiter(
                Propagate spacecraft state using N-body equations of motion

                Args:
                    et0 (double): initial epoch
                    tspan (list): integration time span, [t0, tf]
                    y0 (list): initial state, in order [x,y,z,vx,vy,vz]

                Returns:
                    (list): list of tuples each containing (time, x, y, z, vx, vy, vz)
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
          py::arg("path"), py::arg("nmax"), py::arg("frame"), py::arg("frame_bf"), py::arg("et"),
          py::arg("r"), py::arg("LU") = 1.0, py::arg("TU") = 1.0, py::arg("mmax") = -1,
          py::arg("rho_min") = 1.0,
          R"mydelimiter(
            Spherical-harmonics acceleration and gradient in an inertial frame, as used by the propagators

            Rotates with SPICE pxform(frame, frame_bf, et): a = T^T a_bf, G = T^T G_bf T.
            Needs the kernels of frame_bf (see docs/spherical_harmonics.md). Raises RuntimeError on a
            SPICE error and ValueError below rho_min * R.

            Args:
                path (str): gravity field file
                nmax (int): truncation degree
                frame (str): inertial frame of r, e.g. "J2000"
                frame_bf (str): body-fixed frame of the gravity field, e.g. "MOON_PA", "ITRF93"
                et (double): epoch, in ephemeris seconds
                r (list): position, in LU
                LU (double): length unit, km
                TU (double): time unit, s
                mmax (int): truncation order; -1 means min(nmax, order of the file)
                rho_min (double): smallest |r|/R evaluated, in (0, 1]

            Returns:
                (tuple): (a, G), a in LU/TU^2 and G = d(a)/d(r) in 1/TU^2 flattened row-major (9 elements)
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

    py::class_<campanella::PropagatorGve>(m, "PropagatorGve")
        .def(py::init<std::vector<double>,
                      std::vector<std::string>,
                      std::string,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      std::string,
                      int,
                      int,
                      std::string,
                      double>(),
             py::arg("GM"), py::arg("BODIES"), py::arg("frame"),
             py::arg("A"), py::arg("m"), py::arg("rhoS"), py::arg("rhoD"),
             py::arg("LU"), py::arg("TU"), py::arg("reltol") = 1e-14, py::arg("abstol") = 1e-12,
             py::arg("sh_file") = "", py::arg("sh_nmax") = 0, py::arg("sh_mmax") = -1,
             py::arg("frame_bf") = "", py::arg("sh_rho_min") = 1.0,
             R"mydelimiter(
                PropagatorGve class

                Propagates a spacecraft in modified equinoctial elements (MEE) using the
                Gauss variational equations, with N-body and SRP perturbations.
                The dynamics are the same as PropagatorNbodySrp; set A = 0 to remove SRP.

                Args:
                    GM (list): list of gravitational parameters, in appropriate scale; GM[0] is the central body
                    BODIES (list): list strings corresponding to NAIF body names
                    frame (char): name of inertial reference frame
                    A (double): surface area of spacecraft
                    m (double): mass of spacecraft
                    rhoS (double): reflection rate of spacecraft
                    rhoD (double): scattering rate of spacecraft
                    LU (double): canonical length unit
                    TU (double): canonical time unit
                    reltol (double): relative error tolerance, defaults to 1e-14
                    abstol (double): absolute error tolerance, defaults to 1e-12
                    sh_file (str): optional central-body gravity field, PDS SHADR .tab or ICGEM .gfc; "" (default) turns it off
                    sh_nmax (int): truncation degree of the gravity field, >= 2 if sh_file is given
                    sh_mmax (int): truncation order of the gravity field; -1 (default) means min(sh_nmax, order of the file)
                    frame_bf (str): body-fixed frame of the gravity field, e.g. "MOON_PA" or "ITRF93"
                    sh_rho_min (double): smallest |r|/R at which the field is evaluated, in (0, 1], default 1;
                        a propagation that goes below it raises ValueError. Values slightly below 1
                        (e.g. 0.99) allow low or landing arcs, but the series may not converge there
            )mydelimiter")

        .def("solve_ivp", &campanella::PropagatorGve::solve_ivp,
             py::arg("et0"), py::arg("tspan"), py::arg("y0"),
             R"mydelimiter(
                Propagate spacecraft MEE using Gauss variational equations

                Args:
                    et0 (double): initial epoch
                    tspan (list): integration time span, [t0, tf]
                    y0 (list): initial state, MEE in order [p,f,g,h,k,L]

                Returns:
                    (list): list of tuples each containing (time, p, f, g, h, k, L); L is not wrapped
             )mydelimiter")

        .def("solve_ivp_stm", &campanella::PropagatorGve::solve_ivp_stm,
             py::arg("et0"), py::arg("tspan"), py::arg("y0"),
             R"mydelimiter(
                Propagate spacecraft MEE and STM using Gauss variational equations

                Args:
                    et0 (double): initial epoch
                    tspan (list): integration time span, [t0, tf]
                    y0 (list): initial state, MEE in order [p,f,g,h,k,L] + list(STM(0,0))

                Returns:
                    (list): list of tuples each containing (time, p, f, g, h, k, L) + list(STM(t,t0))
             )mydelimiter")

        .def("eval_jacobian", &campanella::PropagatorGve::eval_jacobian,
             py::arg("et0"), py::arg("t"), py::arg("x"),
             R"mydelimiter(
                Evaluate the equations of motion and their Jacobian at a single time and state

                Raises ValueError if p <= 0 or w <= 0, and RuntimeError with NAIF's message on a SPICE error.

                Args:
                    et0 (double): initial epoch
                    t (double): time since et0, in TU
                    x (list): MEE state [p,f,g,h,k,L]

                Returns:
                    (tuple): (xdot, A), with xdot of length 6 and A = d(xdot)/dx flattened row-major (36 elements)
             )mydelimiter");
}
