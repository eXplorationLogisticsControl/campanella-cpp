#include "campanella.hpp"

#include <stdexcept>
#include <string>

#include "SpiceUsr.h"

namespace campanella {
namespace {

void spice_errors_return() {
    SpiceChar action[] = "RETURN";
    erract_c("SET", sizeof(action), action);
    SpiceChar report[] = "NONE";
    errprt_c("SET", sizeof(report), report);
}

void spice_throw_if_failed() {
    if (!failed_c()) {
        return;
    }
    SpiceChar msg[1841];
    getmsg_c("LONG", sizeof(msg), msg);
    reset_c();
    throw std::runtime_error(msg);
}

}  // namespace

void init_spice() {
    spice_errors_return();
}

void furnsh(const std::string& path) {
    init_spice();
    furnsh_c(path.c_str());
    spice_throw_if_failed();
}

}  // namespace campanella
