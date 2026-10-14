# List of smaller versions CMakeLists.txt

(Thining of some better solutions - maybe using submodules?)

```C
cmake_minimum_required(VERSION 3.12)
project(TwoBodyIntegration)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Look for packages
find_package(GSL REQUIRED)

# Add the executable target
add_executable(TwoBodyIntegration 
    src/twobody.cpp
    src/eoms.cpp)

# Link packages
# https://cmake.org/cmake/help/latest/command/target_link_libraries.html
target_link_libraries(TwoBodyIntegration GSL::gsl GSL::gslcblas)

# Location for output location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
```