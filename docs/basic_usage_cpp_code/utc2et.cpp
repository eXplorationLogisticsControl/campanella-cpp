// Convert UTC calendar dates to ephemeris time (TDB seconds past J2000) with NAIF's CSPICE.
// A separate program: it links CSPICE itself and does not use libcampanella (see build_and_run.sh).
// Usage: SPICE=/path/to/kernels ./utc2et 2025-01-01T00:00:00 ...

#include <cstdio>
#include <cstdlib>
#include <string>

#include "SpiceUsr.h"

int main(int argc, char** argv) {
    const char* spice = std::getenv("SPICE");
    if (spice == nullptr) {
        std::fprintf(stderr, "set SPICE to the kernel directory\n");
        return 1;
    }
    // report SPICE errors through failed_c() instead of exiting
    erract_c("SET", 0, const_cast<SpiceChar*>("RETURN"));
    errprt_c("SET", 0, const_cast<SpiceChar*>("NONE"));

    // the leap-second kernel is what str2et_c needs to convert UTC
    furnsh_c((std::string(spice) + "/lsk/naif0012.tls").c_str());

    const char* dates[] = {"2025-01-01T00:00:00", "2023-06-17T12:00:00"};
    const int n = (argc > 1) ? argc - 1 : 2;
    for (int i = 0; i < n; i++) {
        const char* utc = (argc > 1) ? argv[i + 1] : dates[i];
        SpiceDouble et = 0.0;
        str2et_c(utc, &et);
        if (failed_c()) {
            SpiceChar msg[1841];
            getmsg_c("LONG", sizeof(msg), msg);
            reset_c();
            std::fprintf(stderr, "%s: %s\n", utc, msg);
            return 1;
        }
        std::printf("%s UTC -> et = %.7f s past J2000 (TDB)\n", utc, et);
    }
    return 0;
}
