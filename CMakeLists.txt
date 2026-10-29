cmake_minimum_required(VERSION 3.12)
include(FetchContent)

project(campanella LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(CAMPANELLA_BUILD_PYTHON "Build the pybind11 module" ON)
option(CAMPANELLA_BUILD_CPP "Build the shared C++ library" OFF)
if(NOT CAMPANELLA_BUILD_PYTHON AND NOT CAMPANELLA_BUILD_CPP)
    message(FATAL_ERROR
        "Nothing to build. Pass -DCAMPANELLA_BUILD_PYTHON=ON and/or -DCAMPANELLA_BUILD_CPP=ON.")
endif()

# Find Python interpreter and development libraries first.
# `uv run` uses .venv. On macOS, CMake otherwise prefers a Python.framework
# (Homebrew 3.13 here) and writes campanella.cpython-313-*.so, which 3.11
# will not import. Prefer the venv's base interpreter, which has the headers
# uv's pyvenv.cfg does not expose to FindPython.
if(CAMPANELLA_BUILD_PYTHON AND NOT Python_EXECUTABLE)
    if(WIN32)
        set(_campanella_venv_python "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    else()
        set(_campanella_venv_python "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    endif()
    if(EXISTS "${_campanella_venv_python}")
        execute_process(
            COMMAND "${_campanella_venv_python}" -c "import sys; print(sys.base_prefix)"
            OUTPUT_VARIABLE _campanella_python_root
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _campanella_python_rc
        )
        if(_campanella_python_rc EQUAL 0)
            if(WIN32 AND EXISTS "${_campanella_python_root}/python.exe")
                set(_campanella_python_exe "${_campanella_python_root}/python.exe")
            elseif(EXISTS "${_campanella_python_root}/bin/python3")
                set(_campanella_python_exe "${_campanella_python_root}/bin/python3")
            endif()
            if(_campanella_python_exe)
                set(Python_EXECUTABLE "${_campanella_python_exe}" CACHE FILEPATH
                    "Interpreter matching the project .venv (override with -DPython_EXECUTABLE)")
                set(Python_ROOT_DIR "${_campanella_python_root}" CACHE PATH
                    "Prefix of the interpreter used to build the extension")
                message(STATUS "Project .venv selects Python: ${Python_EXECUTABLE}")
            endif()
        endif()
    endif()
endif()
if(CAMPANELLA_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Found Python ${Python_VERSION}")

    # Set Python version variables for pybind11
    set(PYBIND11_PYTHON_VERSION "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
    set(bindings_python_version ${PYBIND11_PYTHON_VERSION}) # Keep original variable name if used elsewhere
endif()

# Dependencies: GSL, and pybind11 when the Python module is enabled
if(CAMPANELLA_BUILD_PYTHON)
    message(STATUS "Using FetchContent for GSL and pybind11")
else()
    message(STATUS "Using FetchContent for GSL")
endif()

# --- Fetch and Build GNU GSL ---
include(ExternalProject)
set(GSL_VERSION 2.8)
set(GSL_TARBALL_SHA256 "6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190")
set(GSL_URL "https://mirrors.ocf.berkeley.edu/gnu/gsl/gsl-${GSL_VERSION}.tar.gz")

# 1. Declare GSL dependency (just URL and Hash)
FetchContent_Declare(
    GSL
    URL      ${GSL_URL}
    URL_HASH SHA256=${GSL_TARBALL_SHA256}
)

# 2. Populate GSL (download/extract if needed)
# FetchContent_MakeAvailable internally calls FetchContent_Populate
# It also sets gsl_SOURCE_DIR and gsl_BINARY_DIR
FetchContent_MakeAvailable(GSL)
FetchContent_GetProperties(GSL
  SOURCE_DIR GSL_SOURCE_DIR # Where tarball was extracted (e.g., _deps/gsl-src)
  BINARY_DIR GSL_BINARY_DIR # Where EP will build (e.g., _deps/gsl-build)
)

# Define install location within the build directory
set(GSL_INSTALL_DIR ${GSL_BINARY_DIR}/install)

# 3. Define ExternalProject build steps for GSL
# This target will be built automatically when needed
ExternalProject_Add(gsl_external_build # Unique name for the external project target
    SOURCE_DIR        ${GSL_SOURCE_DIR}
    BINARY_DIR        ${GSL_BINARY_DIR}
    # Configure GSL: install to local prefix, build static libs, ensure PIC
    CONFIGURE_COMMAND ${GSL_SOURCE_DIR}/configure --prefix=${GSL_INSTALL_DIR} --disable-shared CFLAGS=-fPIC
    BUILD_COMMAND     $(MAKE) -j # Build using make
    INSTALL_COMMAND   $(MAKE) install # Install headers/libs to the prefix
    BUILD_IN_SOURCE   FALSE
    # Prevent ExternalProject from re-downloading/patching etc.
    DOWNLOAD_COMMAND  ""
    UPDATE_COMMAND    ""
    PATCH_COMMAND     ""
    # Log build output for debugging
    LOG_CONFIGURE     TRUE
    LOG_BUILD         TRUE
    LOG_INSTALL       TRUE
)

message(STATUS "GSL source dir: ${GSL_SOURCE_DIR}")
message(STATUS "GSL binary dir: ${GSL_BINARY_DIR}")
message(STATUS "GSL install dir: ${GSL_INSTALL_DIR}")

# --- Create IMPORTED targets pointing to the *installed* GSL libs/headers ---
set(GSL_INCLUDE_DIR ${GSL_INSTALL_DIR}/include) # Headers installed here
set(GSL_LIBRARY     ${GSL_INSTALL_DIR}/lib/libgsl.a)
set(GSL_CBLAS_LIBRARY ${GSL_INSTALL_DIR}/lib/libgslcblas.a)

message(STATUS "Expecting GSL includes: ${GSL_INCLUDE_DIR}")
message(STATUS "Expecting GSL library: ${GSL_LIBRARY}")
message(STATUS "Expecting GSL CBLAS library: ${GSL_CBLAS_LIBRARY}")

# Ensure target directories exist before defining imported targets
# to prevent CMake generation error (files appear only after build).
file(MAKE_DIRECTORY ${GSL_INCLUDE_DIR})
file(MAKE_DIRECTORY ${GSL_INSTALL_DIR}/lib)

# Define GSL::gsl target
if(NOT TARGET GSL::gsl)
    add_library(GSL::gsl STATIC IMPORTED GLOBAL)
    set_target_properties(GSL::gsl PROPERTIES
        IMPORTED_LOCATION "${GSL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GSL_INCLUDE_DIR}"
    )
    # Add dependency: GSL::gsl target depends on the external build finishing
    add_dependencies(GSL::gsl gsl_external_build)
    message(STATUS "Defined IMPORTED target GSL::gsl linked to external build")
endif()

# Define GSL::gslcblas target
if(NOT TARGET GSL::gslcblas)
    add_library(GSL::gslcblas STATIC IMPORTED GLOBAL)
    set_target_properties(GSL::gslcblas PROPERTIES
        IMPORTED_LOCATION "${GSL_CBLAS_LIBRARY}"
        INTERFACE_LINK_LIBRARIES "GSL::gsl" # gslcblas depends on gsl
    )
    # Add dependency: GSL::gslcblas target depends on the external build finishing
    add_dependencies(GSL::gslcblas gsl_external_build)
    message(STATUS "Defined IMPORTED target GSL::gslcblas linked to external build")
endif()

set(GSL_TARGET GSL::gsl)
set(GSL_CBLAS_TARGET GSL::gslcblas)

# --- Fetch pybind11 ---
if(CAMPANELLA_BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1 # Specify desired tag/commit
    )
    FetchContent_MakeAvailable(pybind11)
    # pybind11::module uses -undefined dynamic_lookup on macOS. Linking Python::Python
    # as well loads a second libpython into a statically linked interpreter and
    # crashes in take_gil on import.
    set(PYBIND11_TARGET pybind11::module)
endif()

# --- CSPICE: official NAIF prebuilt archive for this host, into third_party/cspice ---
# Same selection as chebyshev-ephemeris:
#   Linux x86_64        -> PC_Linux_GCC_64bit
#   macOS Apple Silicon -> MacM1_OSX_clang_64bit
#   macOS Intel         -> MacIntel_OSX_AppleC_64bit
set(CSPICE_DIR "${CMAKE_SOURCE_DIR}/third_party/cspice" CACHE PATH "CSPICE toolkit root")

if(NOT CSPICE_URL)
    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _host_arch)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND _host_arch MATCHES "arm64|aarch64")
        set(_CSPICE_PKG "MacM1_OSX_clang_64bit")
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND _host_arch MATCHES "x86_64|amd64")
        set(_CSPICE_PKG "MacIntel_OSX_AppleC_64bit")
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND _host_arch MATCHES "x86_64|amd64")
        set(_CSPICE_PKG "PC_Linux_GCC_64bit")
    elseif(NOT EXISTS "${CSPICE_DIR}/lib/cspice.a")
        message(FATAL_ERROR
            "No prebuilt CSPICE package for ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_PROCESSOR}. "
            "Unpack a matching toolkit from https://naif.jpl.nasa.gov/naif/toolkit_C.html and pass -DCSPICE_DIR=<cspice root>.")
    endif()
    if(_CSPICE_PKG)
        set(CSPICE_URL "https://naif.jpl.nasa.gov/pub/naif/toolkit/C/${_CSPICE_PKG}/packages/cspice.tar.Z")
    endif()
endif()
if(_CSPICE_PKG)
    message(STATUS "CSPICE for host ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_PROCESSOR}: ${_CSPICE_PKG}")
endif()

set(_cspice_lib "${CSPICE_DIR}/lib/cspice.a")
if(EXISTS "${_cspice_lib}")
    file(READ "${_cspice_lib}" _cspice_hex HEX LIMIT 262144)
    string(TOLOWER "${_cspice_hex}" _cspice_hex)
    string(FIND "${_cspice_hex}" "7f454c46" _elf_at)   # ELF (Linux)
    string(FIND "${_cspice_hex}" "cffaedfe" _macho_at) # Mach-O 64-bit, little-endian (macOS)
    set(_cspice_mismatch OFF)
    set(_cspice_need "")
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        set(_cspice_need "Mach-O")
        if(_macho_at LESS 0)
            set(_cspice_mismatch ON)
        endif()
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        set(_cspice_need "ELF")
        if(_elf_at LESS 0 AND NOT _macho_at LESS 0)
            set(_cspice_mismatch ON)
        endif()
    endif()
    if(_cspice_mismatch)
        message(FATAL_ERROR
            "CSPICE library ${_cspice_lib} is not a ${_cspice_need} archive.\n"
            "This host is ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_PROCESSOR}, "
            "which needs ${_CSPICE_PKG}.\n"
            "Remove ${CSPICE_DIR} and reconfigure so CMake downloads\n  ${CSPICE_URL}\n"
            "or unpack that package and pass -DCSPICE_DIR=<cspice root> "
            "(scripts/fetch_cspice.sh selects the same package from uname).")
    endif()
else()
    message(STATUS "Fetching CSPICE from ${CSPICE_URL}")
    get_filename_component(_tp "${CSPICE_DIR}" DIRECTORY)
    file(MAKE_DIRECTORY "${_tp}")
    file(DOWNLOAD "${CSPICE_URL}" "${_tp}/cspice.tar.Z" SHOW_PROGRESS STATUS _st)
    list(GET _st 0 _st_code)
    if(NOT _st_code EQUAL 0)
        list(GET _st 1 _st_msg)
        message(FATAL_ERROR "CSPICE download failed (${_st_code}): ${_st_msg}\n${CSPICE_URL}")
    endif()
    execute_process(COMMAND tar xzf cspice.tar.Z WORKING_DIRECTORY "${_tp}" RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "CSPICE extraction failed; run scripts/fetch_cspice.sh")
    endif()
endif()

add_library(cspice STATIC IMPORTED)
set_target_properties(cspice PROPERTIES
    IMPORTED_LOCATION "${CSPICE_DIR}/lib/cspice.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CSPICE_DIR}/include")

# Location for output location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Solver sources, compiled once and linked into whichever product is enabled.
add_library(campanella_core OBJECT
    src/perturbations/spice_kernels.cpp
    src/propagate/propagate_twobody.cpp
    src/propagate/propagate_Nbody.cpp
    src/propagate/propagate_NbodySRP.cpp
    src/propagate/propagate_stm_Nbody.cpp
    src/propagate/propagate_gve.cpp
    src/propagate/propagate_stm_gve.cpp
    src/eoms/eoms.cpp
    src/eoms/eoms_stm.cpp
    src/eoms/eoms_gve.cpp
    src/eoms/mee.cpp
    src/eoms/gravity_gradient_Nbody.cpp
    src/perturbations/thirdbody.cpp
    src/perturbations/SRP.cpp)
set_target_properties(campanella_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(campanella_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(campanella_core PUBLIC
    ${GSL_TARGET}
    ${GSL_CBLAS_TARGET}
    cspice)
add_dependencies(campanella_core gsl_external_build)

if(CAMPANELLA_BUILD_CPP)
    # $<TARGET_OBJECTS> keeps this working when cmake_minimum_required is 3.12.
    # Do not also link campanella_core: that would add the same objects twice.
    add_library(campanella SHARED $<TARGET_OBJECTS:campanella_core>)
    target_include_directories(campanella PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(campanella PRIVATE
        ${GSL_TARGET}
        ${GSL_CBLAS_TARGET}
        cspice)
    add_dependencies(campanella gsl_external_build)
    set_target_properties(campanella PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    # NAIF's prebuilt cspice.a may contain text relocations. -z notext lets the
    # shared library link on Linux without rebuilding CSPICE. --exclude-libs
    # keeps those CSPICE symbols inside libcampanella.
    if(UNIX AND NOT APPLE)
        set_property(TARGET campanella APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,-z,notext -Wl,--exclude-libs,ALL")
    endif()

    add_executable(solve_twobody examples/solve_twobody.cpp)
    target_link_libraries(solve_twobody PRIVATE campanella)
    set_target_properties(solve_twobody PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:campanella>")

    add_executable(solve_Nbody examples/solve_Nbody.cpp)
    target_link_libraries(solve_Nbody PRIVATE campanella)
    set_target_properties(solve_Nbody PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:campanella>")

    add_executable(solve_NbodySTM examples/solve_NbodySTM.cpp)
    target_link_libraries(solve_NbodySTM PRIVATE campanella)
    set_target_properties(solve_NbodySTM PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:campanella>")

    add_executable(solve_gve examples/solve_gve.cpp)
    target_link_libraries(solve_gve PRIVATE campanella)
    set_target_properties(solve_gve PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:campanella>")

    add_executable(solve_moon_canonical examples/solve_moon_canonical.cpp)
    target_link_libraries(solve_moon_canonical PRIVATE campanella)
    set_target_properties(solve_moon_canonical PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:campanella>")
endif()

if(CAMPANELLA_BUILD_PYTHON)
    add_library(campanella_python MODULE
        src/campanella.cpp
        $<TARGET_OBJECTS:campanella_core>)
    set_target_properties(campanella_python PROPERTIES OUTPUT_NAME campanella)
    pybind11_extension(campanella_python)
    add_dependencies(campanella_python gsl_external_build)
    target_include_directories(campanella_python PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(campanella_python PRIVATE
        ${GSL_TARGET}
        ${GSL_CBLAS_TARGET}
        cspice
        ${PYBIND11_TARGET}
    )
endif()
