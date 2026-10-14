# Campanella
campanella is a c++ library for astrodynamics with python binding from SSOG

## Installation

1. Install CMake and a C++ compiler.

2. Clone this repository, then `cd` to the repository's root.

3. Configure and build. Configure downloads the host CSPICE package into `third_party/cspice` when `lib/cspice.a` is absent, and fetches GSL and pybind11 into the build directory.

```bash
rm -rf build third_party
cmake -B build -S .
cmake --build build --config Release
```

The Python module is written to `build/lib` (on Windows, `build/Release`). Its name follows the Python CMake found, for example `campanella.cpython-311-darwin.so` for Python 3.11 on macOS. Import it by adding that directory to `PYTHONPATH`.

4. Run tests through python. `SPICE` must point at a directory of generic SPICE kernels (`lsk/naif0012.tls` and `spk/de440.bsp`). Load them with `campanella.furnsh(path)`. `spiceypy.furnsh` does not: that package links a separate CSPICE.

```bash
uv run python tests/test_import.py
```

For further details see:

- building the library: [docs/build_library.md](docs/build_library.md)
- using the library via python: [docs/basic_usage_python.md](docs/basic_usage_python.md)

## Quickstart

To build for Python 3.11:

```bash
rm -rf build third_party
cmake -B build -S . -DPython_EXECUTABLE=$(which python3.11)
cmake --build build --config Release
```


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
