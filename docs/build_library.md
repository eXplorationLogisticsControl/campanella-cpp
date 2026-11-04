# Building `campanella`

This page goes through building the `campanella` library. Unless stated explicitly, the process is the same on Linux and macOS. Windows has no prebuilt CSPICE package in this setup; see [Windows](#windows).


## 0. Pre-requisites

- CMake
- a C++ compiler
- Python 3.11 or newer (the interpreter you will import the module with), including development headers, when building the Python module. The shared C++ library does not need Python.
- [uv](https://docs.astral.sh/uv/), to create the project's Python environment (optional, but the commands on these pages use it).
- Internet access during the first configure, which downloads GSL, pybind11 (Python build) and, if `third_party/cspice` is absent, CSPICE. The first build compiles GSL, so it takes noticeably longer than later builds.


## 1. Build

From the repository root, for the Python module:

```bash
uv sync                                  # 1. create .venv with Python and the packages of pyproject.toml
rm -rf build third_party                 # 2. drop any previous configure
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # 3. configure; finds .venv and builds for its Python
cmake --build build                               # 4. build
```

`.venv` is the project's virtual environment. Run `uv sync` before configuring, so that CMake builds
the module for that interpreter and `uv run` can import it.

`-DCMAKE_BUILD_TYPE=Release` turns on compiler optimization. Without it, the default single-config
generators (Makefiles, Ninja) compile campanella with no optimization flags. (With multi-config
generators such as Visual Studio or Xcode, choose the configuration at build time instead:
`cmake --build build --config Release`.) GSL is compiled by its own configure script, which the
current CMake file calls with `CFLAGS=-fPIC` only, so GSL itself is not optimized in either case.
The numbers printed on these pages were produced with builds configured without
`CMAKE_BUILD_TYPE`.

`rm -rf build third_party` drops a previous configure and a CSPICE tree built for another machine. Configure then:

- downloads GNU GSL and pybind11 into `build/_deps`
- downloads the NAIF CSPICE archive for this host into `third_party/cspice` when `lib/cspice.a` is absent

| Host | CSPICE package |
|---|---|
| macOS Apple Silicon | `MacM1_OSX_clang_64bit` |
| macOS Intel | `MacIntel_OSX_AppleC_64bit` |
| Linux x86_64 | `PC_Linux_GCC_64bit` |

The Python module is `build/lib/campanella.cpython-<version>-<platform>.so`, e.g. `cpython-313` for Python 3.13. (Multi-config generators such as Visual Studio and Xcode put it under `build/lib/Release`.) If `.venv` exists, CMake builds for that interpreter. Otherwise it uses the first Python it finds. To build for a specific interpreter:

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
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp
```

Programs include `include/campanella.hpp`, which needs only the standard library. The shared library already contains GSL and CSPICE; do not also link CSPICE or spiceypy into that process.

A complete first program, with its compile command and output, is
[A first example](basic_usage_cpp.md#a-first-example) (`docs/basic_usage_cpp_code/first_example.cpp`);
the [tutorial](tutorial.md#11-the-same-program-in-c) has a longer one.

The options, units and errors are the same as in Python; see [basic_usage_python.md](basic_usage_python.md) and [basic_usage_cpp.md](basic_usage_cpp.md). `campanella::eom_rv`, `eom_mee`, `eomstm_rv` and `eomstm_mee` have the signature of `gsl_odeiv2_system::function`, with a `SpiceParameters*` as `params`. They return `GSL_EDOM` for a state out of the domain (GSL retries with a smaller step) and `GSL_EBADFUNC` on a SPICE error; `campanella::eom_error(params)` then gives the reason, with NAIF's message. A `SpiceParameters` is not thread-safe: CSPICE is not, and the `eom_*` and `eval_*` functions use its scratch space. `campanella::eom_rvm`, `eom_meem`, `eomstm_rvm` and `eomstm_meem` are the same for the [mass-augmented dynamics](zoh_mass_dynamics.md), with a `ControlledParameters*` as `params`; `eom_error` then takes `cp.params()`.

The examples are built in `build-cpp/bin`:

| Example | |
|---|---|
| `solve_twobody` | two-body propagation; needs no kernels |
| `solve_rv` | Cartesian state and STM with the Earth and the Sun as third bodies; STM checked against finite differences |
| `solve_mee` | MEE (Gauss variational equations) with SRP, checked against the Cartesian propagation |
| `solve_moon_canonical` | canonical units (DU = 100000 km) against km and s |
| `solve_sh` | low lunar orbit with a GRAIL gravity field |
| `solve_zoh` | multi-segment low-thrust arc around the Moon with a zero-order-hold control and SRP from the state mass, in canonical units; $\Phi$, $\Psi$ and the mass of the last segment checked, and exits non-zero if a check fails |

All but `solve_twobody` load kernels from the directory in `SPICE`. A one-file program links the build tree directly:

```bash
c++ -std=c++17 my_ivp.cpp \
  -I include \
  -L build-cpp/lib -lcampanella \
  -Wl,-rpath,$PWD/build-cpp/lib
```

The program finds the library at run time through the rpath given at link time. Give it as an absolute path (`$PWD/...`): a relative rpath is resolved against the directory the program is run from. From another CMake project:

```cmake
set(CAMPANELLA_BUILD_PYTHON OFF CACHE BOOL "")
set(CAMPANELLA_BUILD_CPP ON CACHE BOOL "")
add_subdirectory(path/to/campanella-cpp campanella)   # the second argument is the build directory
target_link_libraries(my_app PRIVATE campanella)
```

The library is written to `<build directory>/lib`: `build-cpp/lib` with the C++-only configure above, or `build/lib` if you add `-DCAMPANELLA_BUILD_CPP=ON` to the default (Python) configure, which then builds both. The commands elsewhere in these docs assume `build-cpp`; see also the table of {ref}`build directories <cpp-build-dirs>`. Configure stops if all three options are off.


## 2. Tests

The C++ tests use GoogleTest and run through ctest:

```bash
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON -DCAMPANELLA_BUILD_TESTS=ON
cmake --build build-cpp
cd build-cpp && SPICE=/path/to/kernels ctest --output-on-failure
```

The test executable links the library's object files rather than `libcampanella`, so that it shares the library's CSPICE. Tests that need kernels read them from `SPICE`: `lsk/naif0012.tls`, `spk/de440.bsp`, `pck/gm_de440.tpc`, `pck/moon_pa_de440_200625.bpc`, `fk/moon_de440_250416.tf` and `pck/earth_latest_high_prec.bpc`. Without `SPICE` they are skipped. If `CAMPANELLA_REQUIRE_KERNELS=1` is set (as in CI), they fail instead. The Python tests run after building the module, from the repository root:

```bash
SPICE=/path/to/kernels uv run pytest
```

They put `build/lib` on the module path themselves, so `PYTHONPATH` is not needed; without `SPICE`, the tests that need kernels are skipped.

`scripts/fetch_cspice.sh` selects the same CSPICE package from `uname` and extracts it into `third_party/cspice`. Configure does this itself when the library is missing.

If `third_party/cspice` is already filled with another operating system's archive, configuration stops. On macOS a Linux archive fails to link. Remove `third_party/cspice` and configure again, or unpack a matching toolkit and point CMake at its `cspice` directory:

```bash
cmake -B build -S . -DCSPICE_DIR=/path/to/cspice
```


Once the library is built, continue with the [tutorial](tutorial.md).


## Windows

Windows is not tested. The prebuilt packages above do not include Windows, and configure stops unless `-DCSPICE_DIR` points at a CSPICE toolkit you unpacked yourself. Two obstacles remain with the current CMake file: it links `${CSPICE_DIR}/lib/cspice.a`, whereas NAIF's Windows (MSVC) toolkit ships `lib/cspice.lib`, and it builds GSL with GSL's autotools `configure` script, which needs a POSIX shell environment (e.g. MSYS2 or Cygwin) rather than a plain MSVC one. pybind11 is fetched when `CAMPANELLA_BUILD_PYTHON` is on.
