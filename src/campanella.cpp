/*
 * campanella.cpp 
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // enable automatic conversion between C++ STL types and Python types
#include <regex>
#include "SpiceUsr.h"

#include "propagate_twobody.h"
#include "propagate_Nbody.h"
#include "propagate_NbodySRP.h"
#include "propagate_stm_Nbody.h"


namespace py = pybind11;


// CSPICE's default error action is to print a traceback and exit the process.
// That kills pytest with no Python traceback. RETURN leaves the error flag set
// so callers can turn it into an exception instead.
static void spice_errors_return() {
    SpiceChar action[] = "RETURN";
    erract_c("SET", sizeof(action), action);
    SpiceChar report[] = "NONE";
    errprt_c("SET", sizeof(report), report);
}

static void spice_throw_if_failed() {
    if (!failed_c()) {
        return;
    }
    SpiceChar msg[1841];
    getmsg_c("LONG", sizeof(msg), msg);
    reset_c();
    throw std::runtime_error(msg);
}

static void furnsh_kernel(const std::string& path) {
    furnsh_c(path.c_str());
    spice_throw_if_failed();
}

// Kernels are not loaded here. Call campanella.furnsh() from Python.
// spiceypy.furnsh() does not affect this library: campanella links its own CSPICE.
static void init_spice() {
    spice_errors_return();
}


int add_j2000(int i, int j) {
    // test SPICE
    double j2000_date = j2000_c();
    std::cout << "J2000 date: " << j2000_date << std::endl;

    // double moon_state[3];
    // double lt;
    // spkpos_c("MOON", j2000_date, "J2000", "NONE", "EARTH", moon_state, &lt);

    return i + j;
}


/**
 * PropagatorTwoBody class 
 * 
 * Propagates a spacecraft using two-body equations of motion
 * 
 * @param GM gravitational parameter
 */
struct PropagatorTwoBody {
    double GM;
    double reltol;
    double abstol;
    
    PropagatorTwoBody(double GM, double reltol = 1e-14, double abstol = 1e-12) 
        : GM(GM), reltol(reltol), abstol(abstol) {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return propagate_twobody(tspan, y0, GM, reltol, abstol);
    }
};


/**
 * PropagatorNBody class 
 * 
 * Propagates a spacecraft using N-body equations of motion
 * 
 * @param GM list of gravitational parameters of each body
 * @param BODIES list of NAIF names of each body
 * @param LU length unit
 * @param TU time unit
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 */
struct PropagatorNBody {
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double LU;
    double TU;
    double reltol;
    double abstol;
    
    PropagatorNBody(std::vector<double> GM,
                    std::vector<std::string> BODIES,
                    std::string frame,
                    double LU,
                    double TU,
                    double reltol = 1e-14,
                    double abstol = 1e-12) 
        : GM(GM), BODIES(BODIES), frame(frame), LU(LU), TU(TU), reltol(reltol), abstol(abstol) {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return propagate_Nbody(
            et0,
            tspan,
            y0,
            GM,
            BODIES,
            frame,
            LU,
            TU,
            reltol,
            abstol);
    }

    std::vector<std::tuple<double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double,
                           double, double, double, double, double, double>> solve_ivp_stm(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return propagate_stm_Nbody(
            et0,
            tspan,
            y0,
            GM,
            BODIES,
            frame,
            LU,
            TU,
            reltol,
            abstol);
    }
};


/**
 * PropagatorNBodySRP class 
 * 
 * Propagates a spacecraft using N-body equations of motion
 * 
 * @param GM list of gravitational parameters of each body
 * @param BODIES list of NAIF names of each body
 * @param A surface area of spacecraft
 * @param m mass of spacecraft
 * @param rhoS reflection rate of spacecraft
 * @param rhoD scattering rate of spacecraft
 * @param LU length unit
 * @param TU time unit
 * @param reltol relative error tolerance, defaults to 1e-14
 * @param abstol absolute error tolerance, defaults to 1e-12
 */
struct PropagatorNBodySRP {
    std::vector<double> GM;
    std::vector<std::string> BODIES;
    std::string frame;
    double A;
    double m;
    double rhoS;
    double rhoD;
    double LU;
    double TU;
    double reltol;
    double abstol;
    
    PropagatorNBodySRP(std::vector<double> GM,
                    std::vector<std::string> BODIES,
                    std::string frame,
                    double A,
                    double m,
                    double rhoS,
                    double rhoD,
                    double LU,
                    double TU,
                    double reltol = 1e-14,
                    double abstol = 1e-12) 
        : GM(GM), BODIES(BODIES), frame(frame), A(A), m(m), rhoS(rhoS), rhoD(rhoD), LU(LU), TU(TU), reltol(reltol), abstol(abstol) {}

    std::vector<std::tuple<double, double, double, double, double, double, double>> solve_ivp(
        double et0,
        const std::vector<double>& tspan,
        const std::vector<double>& y0) {
        return propagate_NbodySRP(
            et0,
            tspan,
            y0,
            GM,
            BODIES,
            frame,
            A,
            m,
            rhoS,
            rhoD,
            LU,
            TU,
            reltol,
            abstol);
    }
};



PYBIND11_MODULE(campanella, m) {
    m.doc() = """campanella is a c++ library for astrodynamics with python binding from SSOG"""; // optional module docstring

    // Initialize CSPICE when the module is loaded
    init_spice();

    m.def("add_j2000", &add_j2000,
          py::arg("i"), py::arg("j"),
          "A function that adds two numbers");

    m.def("furnsh", &furnsh_kernel, py::arg("path"),
          "Load a SPICE kernel into campanella's CSPICE. spiceypy.furnsh does not.");
    
    py::class_<PropagatorTwoBody>(m, "PropagatorTwoBody")
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
            
        .def("solve_ivp", &PropagatorTwoBody::solve_ivp,
             py::arg("tspan"), py::arg("y0"),  
             R"mydelimiter(
                Propagate spacecraft state using two-body equations of motion

                Args:
                    tspan (list): integration time span, [t0, tf]
                    y0 (list): initial state, in order [x,y,z,vx,vy,vz]

                Returns:
                    (list): list of tuples each containing (time, x, y, z, vx, vy, vz)
             )mydelimiter");

    py::class_<PropagatorNBody>(m, "PropagatorNBody")
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
                PropagatorNBody class

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

        .def("solve_ivp", &PropagatorNBody::solve_ivp,
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

        .def("solve_ivp_stm", &PropagatorNBody::solve_ivp_stm,
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

    py::class_<PropagatorNBodySRP>(m, "PropagatorNBodySRP")
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
                PropagatorNBodySRP class

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

        .def("solve_ivp", &PropagatorNBodySRP::solve_ivp,
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
