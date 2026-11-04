// Single-point evaluation of the equations of motion and their Jacobian.

#include "campanella.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

int main() {
    const char* spice = std::getenv("SPICE");
    if (spice == nullptr) {
        std::fprintf(stderr, "set SPICE to the kernel directory\n");
        return 1;
    }
    const std::string dir(spice);
    try {
        campanella::furnsh(dir + "/spk/de440.bsp");
        campanella::furnsh(dir + "/pck/gm_de440.tpc");
        const campanella::SpiceParameters params(740463475.5117936, {"EARTH", "MOON", "SUN"});

        const std::vector<double> x = {7000.0, 0.0, 0.0, 0.0, 7.5, 0.0};
        const campanella::EomEval e = campanella::eval_rv(params, 0.0, x);
        std::printf("xdot = %.6f %.6f %.6f %.9e %.9e %.9e\n",
                    e.xdot[0], e.xdot[1], e.xdot[2], e.xdot[3], e.xdot[4], e.xdot[5]);
        std::printf("A (row-major 6x6):\n");
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                std::printf(" %11.3e", e.A[6*i + j]);
            }
            std::printf("\n");
        }

        // eom_rv computes the same xdot, with the GSL signature
        double xdot[6];
        const int status = campanella::eom_rv(0.0, x.data(), xdot,
                                              const_cast<campanella::SpiceParameters*>(&params));
        std::printf("eom_rv status %d, xdot[3] = %.9e\n", status, xdot[3]);

        // in MEE: p, f, g, h, k, L
        const std::vector<double> mee = campanella::cart2mee(x, params.mus()[0]);
        const campanella::EomEval m = campanella::eval_mee(params, 0.0, mee);
        std::printf("MEE xdot = %.3e %.3e %.3e %.3e %.3e %.9e\n",
                    m.xdot[0], m.xdot[1], m.xdot[2], m.xdot[3], m.xdot[4], m.xdot[5]);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
