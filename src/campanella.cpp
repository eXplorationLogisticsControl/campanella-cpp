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
}
