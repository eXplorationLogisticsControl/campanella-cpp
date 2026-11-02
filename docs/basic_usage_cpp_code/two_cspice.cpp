// A program that links its own CSPICE next to libcampanella: the two CSPICE copies have separate
// kernel pools (on Linux, where the library hides its CSPICE symbols). See build_and_run.sh.

#include "campanella.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "SpiceUsr.h"

int main() {
    const char* spice = std::getenv("SPICE");
    if (spice == nullptr) {
        std::fprintf(stderr, "set SPICE to the kernel directory\n");
        return 1;
    }
    const std::string dir(spice);

    // load a GM kernel into the program's own CSPICE
    furnsh_c((dir + "/pck/gm_de440.tpc").c_str());
    std::printf("own CSPICE has GM of Earth: %d\n", static_cast<int>(bodfnd_c(399, "GM")));

    // campanella's CSPICE does not see it
    try {
        const campanella::SpiceParameters params(0.0, {"EARTH"});
        std::printf("campanella saw it\n");
    } catch (const std::exception& ex) {
        std::printf("campanella: %s\n", ex.what());
    }

    // and the program's CSPICE does not see kernels loaded with campanella::furnsh
    campanella::furnsh(dir + "/lsk/naif0012.tls");
    SpiceInt count = 0;
    ktotal_c("ALL", &count);
    std::printf("own CSPICE kernels after campanella::furnsh: %d\n", static_cast<int>(count));
    return 0;
}
