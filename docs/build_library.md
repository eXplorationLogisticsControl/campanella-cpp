# Building `campanella`

This page goes through building the `campanella` library. Unless stated explicitly, the process is the same on Linux and macOS. Windows has no prebuilt CSPICE package in this setup; see [Windows](#windows).


## 0. Pre-requisites

- CMake
- a C++ compiler
- Python 3.11 (or whichever interpreter you will import the module with), including development headers, when building the Python module. The shared C++ library does not need Python.


## 1. Build

From the repository root:

```bash
rm -rf build third_party
cmake -B build -S .
cmake --build build --config Release
```

`rm -rf build third_party` drops a previous configure and a CSPICE tree built for another machine. Configure then:

- downloads GNU GSL and pybind11 into `build/_deps`
- downloads the NAIF CSPICE archive for this host into `third_party/cspice` when `lib/cspice.a` is absent

| Host | CSPICE package |
|---|---|
| macOS Apple Silicon | `MacM1_OSX_clang_64bit` |
| macOS Intel | `MacIntel_OSX_AppleC_64bit` |
| Linux x86_64 | `PC_Linux_GCC_64bit` |

The Python module is `build/lib/campanella.cpython-<version>-<platform>.so` (on multi-config generators, under `build/Release`). If `.venv` exists, CMake builds for that interpreter so `uv run` can import the module. Otherwise it uses the first Python it finds. To build for a specific interpreter:

```bash
cmake -B build -S . -DPython_EXECUTABLE=$(which python3.11)
```

Three options select the products. The defaults above build only the Python module.

| Option | Default | Product |
|---|---|---|
| `CAMPANELLA_BUILD_PYTHON` | `ON` | `build/lib/campanella.cpython-….so` |
| `CAMPANELLA_BUILD_CPP` | `OFF` | `build/lib/libcampanella.dylib` (macOS) or `libcampanella.so` (Linux), and the examples |
| `CAMPANELLA_BUILD_TESTS` | `OFF` | `bin/campanella_tests` in the build directory (GoogleTest, fetched during configure) |

A C++-only configure does not search for Python or fetch pybind11:

```bash
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON
cmake --build build-cpp
```

Programs include `include/campanella.hpp`, which needs only the standard library. The shared library already contains GSL and CSPICE; do not also link CSPICE or spiceypy into that process.

```cpp
#include "campanella.hpp"

campanella::furnsh("/path/to/lsk/naif0012.tls");
campanella::furnsh("/path/to/spk/de440.bsp");

campanella::SpiceOptions options;                 // set the fields by name
options.GMs = {4.902800582147760e+03, 398600.432896939, 132712197035.766};   // km^3/s^2
options.include_srp = true;
options.srp_A = 34.675;                           // m^2
options.srp_m = 1850.0;                           // kg
const campanella::SpiceParameters params(et0, {"MOON", "EARTH", "SUN"}, options);

campanella::IvpOptions ivp;                       // atol, rtol, t_eval
ivp.rtol = 1e-12;
const campanella::IvpResult res = campanella::solve_ivpstm_rv(params, {0.0, 86400.0}, x0, ivp);
// res.t (N), res.x (N x 6, row-major), res.stm (N x 36, row-major)
```

The options, units and errors are the same as in Python; see [basic_usage_python.md](basic_usage_python.md). `campanella::eom_rv`, `eom_mee`, `eomstm_rv` and `eomstm_mee` have the signature of `gsl_odeiv2_system::function`, with a `SpiceParameters*` as `params`. They return `GSL_EDOM` for a state out of the domain (GSL retries with a smaller step) and `GSL_EBADFUNC` on a SPICE error; `campanella::eom_error(params)` then gives the reason, with NAIF's message. A `SpiceParameters` is not thread-safe: CSPICE is not, and the `eom_*` and `eval_*` functions use its scratch space.

The examples are built in `build-cpp/bin`:

| Example | |
|---|---|
| `solve_twobody` | two-body propagation; needs no kernels |
| `solve_rv` | Cartesian state and STM with the Earth and the Sun as third bodies; STM checked against finite differences |
| `solve_mee` | MEE (Gauss variational equations) with SRP, checked against the Cartesian propagation |
| `solve_moon_canonical` | canonical units (DU = 100000 km) against km and s |
| `solve_sh` | low lunar orbit with a GRAIL gravity field |

All but `solve_twobody` load kernels from the directory in `SPICE`. A one-file program links the build tree directly:

```bash
c++ -std=c++17 my_ivp.cpp \
  -I include \
  -L build-cpp/lib -lcampanella \
  -Wl,-rpath,build-cpp/lib
```

The library's install name is `@rpath/libcampanella.dylib` (or the Linux soname). `-Wl,-rpath` is the directory that contains the shared library. From another CMake project:

```cmake
set(CAMPANELLA_BUILD_PYTHON OFF CACHE BOOL "")
set(CAMPANELLA_BUILD_CPP ON CACHE BOOL "")
add_subdirectory(path/to/campanella-cpp)
target_link_libraries(my_app PRIVATE campanella)
```

`-DCAMPANELLA_BUILD_CPP=ON` with the Python option left on builds both into `build/lib`. Configure stops if all three options are off.


## 2. Tests

The C++ tests use GoogleTest and run through ctest:

```bash
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON -DCAMPANELLA_BUILD_TESTS=ON
cmake --build build-cpp
cd build-cpp && SPICE=/path/to/kernels ctest --output-on-failure
```

The test executable links the library's object files rather than `libcampanella`, so that it shares the library's CSPICE. Tests that need kernels read them from `SPICE`: `lsk/naif0012.tls`, `spk/de440.bsp`, `pck/gm_de440.tpc`, `pck/moon_pa_de440_200625.bpc`, `fk/moon_de440_250416.tf` and `pck/earth_latest_high_prec.bpc`. Without `SPICE` they are skipped. If `CAMPANELLA_REQUIRE_KERNELS=1` is set (as in CI), they fail instead. The Python tests run with `uv run pytest` after building the module.

`scripts/fetch_cspice.sh` selects the same CSPICE package from `uname` and extracts it into `third_party/cspice`. Configure does this itself when the library is missing.

If `third_party/cspice` is already filled with another operating system's archive, configuration stops. On macOS a Linux archive fails to link. Remove `third_party/cspice` and configure again, or unpack a matching toolkit and point CMake at its `cspice` directory:

```bash
cmake -B build -S . -DCSPICE_DIR=/path/to/cspice
```


## Windows

The prebuilt packages above do not include Windows. Configure stops unless `-DCSPICE_DIR` points at a CSPICE toolkit you unpacked yourself (the directory that contains `include/SpiceUsr.h` and `lib/cspice.a`). GSL is still fetched during configure. pybind11 is fetched when `CAMPANELLA_BUILD_PYTHON` is on.
