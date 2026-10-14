# Campanella
campanella is a c++ library for astrodynamics with python binding from SSOG

## Directory structure

```sh
.
├── docs
│   ├── basic_usage_python.md
│   └── build_library.md
├── examples
│   ├── solve_Nbody.cpp               # N-body propagation; needs SPICE kernels
│   ├── solve_NbodySTM.cpp            # N-body state and state-transition matrix
│   └── solve_twobody.cpp             # two-body propagation
├── include
│   └── campanella.hpp                # public C++ header
├── notes
│   ├── list_cmakelists.md
│   ├── setup_linux.md
│   └── setup_windows.md
├── scripts
│   └── fetch_cspice.sh               # manual CSPICE download
├── src                               # C++ library and pybind11 bindings
│   └── ...
├── test-python                       # pytest tests for the built module
│   └── ...
├── .gitignore
├── CMakeLists.txt
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

## Installation

Install CMake and a C++ compiler. Clone this repository, then `cd` to the repository's root.

Configure downloads the host CSPICE package into `third_party/cspice` when `lib/cspice.a` is absent, and fetches GSL into the build directory. The Python build also fetches pybind11. Pass `-DCAMPANELLA_BUILD_CPP=ON` with the Python option left on to build both into one directory. Configure stops if both options are off.

### Build for cpp

```bash
rm -rf build-cpp
cmake -B build-cpp -S . -DCAMPANELLA_BUILD_PYTHON=OFF -DCAMPANELLA_BUILD_CPP=ON
cmake --build build-cpp
```

This writes `build-cpp/lib/libcampanella.dylib` on macOS, or `libcampanella.so` on Linux. Include `include/campanella.hpp` and link that library. GSL and CSPICE are already inside it, so do not also link CSPICE or spiceypy into the same process. The examples `solve_twobody`, `solve_Nbody`, and `solve_NbodySTM` are built in `build-cpp/bin` and find the library through its rpath. `solve_Nbody` and `solve_NbodySTM` load `lsk/naif0012.tls` and `spk/de440.bsp` from the directory in `SPICE`.

```bash
c++ -std=c++17 my_ivp.cpp \
  -I include -I src \
  -L build-cpp/lib -lcampanella \
  -Wl,-rpath,build-cpp/lib
```

Another CMake project can add this repository as a subdirectory:

```cmake
set(CAMPANELLA_BUILD_PYTHON OFF CACHE BOOL "")
set(CAMPANELLA_BUILD_CPP ON CACHE BOOL "")
add_subdirectory(path/to/campanella-cpp)
target_link_libraries(my_app PRIVATE campanella)
```

### Build for python

```bash
rm -rf build
cmake -B build -S .
cmake --build build --config Release
```

The Python module is written to `build/lib` (on Windows, `build/Release`). Its name follows the Python CMake found, for example `campanella.cpython-311-darwin.so` for Python 3.11 on macOS. Import it by adding that directory to `PYTHONPATH`.

`SPICE` must point at a directory of generic SPICE kernels (`lsk/naif0012.tls` and `spk/de440.bsp`). Load them with `campanella.furnsh(path)`. `spiceypy.furnsh` does not: that package links a separate CSPICE.

```bash
uv sync
uv run pytest
```

To build for a specific interpreter:

```bash
rm -rf build
cmake -B build -S . -DPython_EXECUTABLE=$(which python3.11)
cmake --build build --config Release
```

For further details see:

- building the library: [docs/build_library.md](docs/build_library.md)
- using the library via python: [docs/basic_usage_python.md](docs/basic_usage_python.md)


## Common Issues

### Python versions

- If `.venv` exists, CMake builds the extension for that interpreter. Otherwise it uses the first Python it finds, which on macOS is often a Homebrew framework rather than the `uv` environment. A module built for Python 3.13 will not load in Python 3.11. Pass `-DPython_EXECUTABLE=` to choose the interpreter.

### `SPICE(NOLOADEDFILES)` on Windows

If you get the error

```
==============================================================================

Toolkit version: N0067

SPICE(NOLOADEDFILES) --

At least one SPK file needs to be loaded by SPKLEF before beginning a search.

A traceback follows.  The name of the highest level module is first.
spkpos_c --> SPKPOS --> SPKEZP --> SPKGPS --> SPKSFS

Oh, by the way:  The SPICELIB error handling actions are USER-TAILORABLE.  You
can choose whether the Toolkit aborts or continues when errors occur, which
error messages to output, and where to send the output.  Please read the ERROR
"Required Reading" file, or see the routines ERRACT, ERRDEV, and ERRPRT.

==============================================================================
```

Load the kernels with `campanella.furnsh(path)`. `spiceypy.furnsh()` loads a different CSPICE and does not affect this library. A missing kernel raises `RuntimeError` from `furnsh` instead of exiting the process.


## Todo

### General

- [x] Figure out wrap for python (via `pybind11`)
- [ ] Figure out wrap for MEX
- [x] Integration and saving states at user-defined times (see: [https://www.gnu.org/software/gsl/doc/html/ode-initval.html#examples](https://www.gnu.org/software/gsl/doc/html/ode-initval.html#examples))


### Ballistic (natural dynamics)

- [ ] Equations of motion
  - [x] Two-body
  - [ ] CR3BP
  - [x] N-body
  - [x] N-body + SRP
  - [ ] N-body + spherical harmonics
- [ ] Equations of motion with state-transition matrix
  - [ ] Two-body
  - [ ] CR3BP
  - [x] N-body
  - [ ] N-body + SRP
  - [ ] N-body + spherical harmonics


### Optimal Control Problem

- [ ] Two-body states & costates
  - [ ] Cartesian
  - [ ] MEE
  - [ ] CR3BP

### Propagation functionality

- [ ] Epoch-referenced mode
- [ ] Event detection
