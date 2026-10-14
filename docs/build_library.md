# Building `campanella`

This page goes through building the `campanella` library. Unless stated explicitly, the process is the same on Linux and macOS. Windows has no prebuilt CSPICE package in this setup; see [Windows](#windows).


## 0. Pre-requisites

- CMake
- a C++ compiler
- Python 3.11 (or whichever interpreter you will import the module with), including development headers


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

`scripts/fetch_cspice.sh` selects the same CSPICE package from `uname` and extracts it into `third_party/cspice`. Configure does this itself when the library is missing.

If `third_party/cspice` is already filled with another operating system's archive, configuration stops. On macOS a Linux archive fails to link. Remove `third_party/cspice` and configure again, or unpack a matching toolkit and point CMake at its `cspice` directory:

```bash
cmake -B build -S . -DCSPICE_DIR=/path/to/cspice
```


## Windows

The prebuilt packages above do not include Windows. Configure stops unless `-DCSPICE_DIR` points at a CSPICE toolkit you unpacked yourself (the directory that contains `include/SpiceUsr.h` and `lib/cspice.a`). GSL and pybind11 are still fetched during configure.
