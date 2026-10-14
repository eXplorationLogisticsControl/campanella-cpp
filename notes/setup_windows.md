# Building on Windows

The CSPICE packages this project downloads are for Linux x86_64 and macOS (Apple Silicon and Intel). Configure stops on Windows unless you unpack a matching toolkit and pass its root:

```bash
cmake -B build -S . -DCSPICE_DIR=C:\path\to\cspice
cmake --build build --config Release
```

`CSPICE_DIR` is the directory that contains `include\SpiceUsr.h` and `lib\cspice.a`. GSL and pybind11 are still downloaded during configure. The Python module is written under `build\Release`.

To select the Python interpreter:

```bash
cmake -B build -S . -DCSPICE_DIR=C:\path\to\cspice -DPython_EXECUTABLE=C:\path\to\python.exe
```

If a previous configure failed, delete the `build` directory before running CMake again.
