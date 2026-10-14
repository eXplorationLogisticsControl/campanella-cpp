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

Two options select the product. The defaults above build only the Python module.

| Option | Default | Product |
|---|---|---|
| `CAMPANELLA_BUILD_PYTHON` | `ON` | `build/lib/campanella.cpython-….so` |
| `CAMPANELLA_BUILD_CPP` | `OFF` | `build/lib/libcampanella.dylib` (macOS) or `libcampanella.so` (Linux) |

A C++-only configure does not search for Python or fetch pybind11:

```bash
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON
cmake --build build-cpp
```

Programs include `include/campanella.hpp`. The shared library already contains GSL and CSPICE; do not also link CSPICE or spiceypy into that process. `build-cpp/bin/solve_twobody` links the library and runs a two-body propagation. `build-cpp/bin/solve_Nbody` propagates the same initial state with `PropagatorNbody`, and `build-cpp/bin/solve_NbodySTM` also propagates the state-transition matrix. Both load `lsk/naif0012.tls` and `spk/de440.bsp` from the directory in `SPICE`. A one-file program links the build tree directly:

```bash
c++ -std=c++17 my_ivp.cpp \
  -I include -I src \
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

`-DCAMPANELLA_BUILD_CPP=ON` with the Python option left on builds both into `build/lib`. Configure stops if both options are off.

`scripts/fetch_cspice.sh` selects the same CSPICE package from `uname` and extracts it into `third_party/cspice`. Configure does this itself when the library is missing.

If `third_party/cspice` is already filled with another operating system's archive, configuration stops. On macOS a Linux archive fails to link. Remove `third_party/cspice` and configure again, or unpack a matching toolkit and point CMake at its `cspice` directory:

```bash
cmake -B build -S . -DCSPICE_DIR=/path/to/cspice
```


## Windows

The prebuilt packages above do not include Windows. Configure stops unless `-DCSPICE_DIR` points at a CSPICE toolkit you unpacked yourself (the directory that contains `include/SpiceUsr.h` and `lib/cspice.a`). GSL is still fetched during configure. pybind11 is fetched when `CAMPANELLA_BUILD_PYTHON` is on.
