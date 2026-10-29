/*
 * campanella.cpp
 */

#include <iostream>
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
                      double>(),
             py::arg("GM"), py::arg("BODIES"), py::arg("frame"), py::arg("LU"), py::arg("TU"),
             py::arg("reltol") = 1e-14, py::arg("abstol") = 1e-12,
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
                      double>(),
             py::arg("GM"), py::arg("BODIES"), py::arg("frame"),
             py::arg("A"), py::arg("m"), py::arg("rhoS"), py::arg("rhoD"),
             py::arg("LU"), py::arg("TU"), py::arg("reltol") = 1e-14, py::arg("abstol") = 1e-12,
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
                      double>(),
             py::arg("GM"), py::arg("BODIES"), py::arg("frame"),
             py::arg("A"), py::arg("m"), py::arg("rhoS"), py::arg("rhoD"),
             py::arg("LU"), py::arg("TU"), py::arg("reltol") = 1e-14, py::arg("abstol") = 1e-12,
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
