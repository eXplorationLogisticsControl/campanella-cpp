# Setting up the build on Linux and macOS

## 0. Setup `CMake`

Install CMake and a C++ compiler. Python development headers are required for the extension module.

## 1. Compile

From the repository root (where `CMakeLists.txt` is):

```bash
rm -rf build third_party
cmake -B build -S .
cmake --build build --config Release
```

`rm -rf build` removes a previous build directory. `rmdir` only works when that directory is already empty.

Configure downloads the host CSPICE package into `third_party/cspice` when `lib/cspice.a` is absent:

- Apple Silicon: `MacM1_OSX_clang_64bit`
- Intel macOS: `MacIntel_OSX_AppleC_64bit`
- Linux x86_64: `PC_Linux_GCC_64bit`

GSL and pybind11 are fetched into `build/_deps`. The Python module is written to `build/lib`. If `.venv` exists, CMake builds for that interpreter. Pass `-DPython_EXECUTABLE=` to select a different one.

If `third_party/cspice` was built for another OS, delete it and configure again, or pass `-DCSPICE_DIR` to a toolkit unpacked elsewhere. `scripts/fetch_cspice.sh` downloads the same package.
