#!/bin/sh
# Compile every program of this directory against build-cpp/lib and run it.
# Usage, from the repository root: SPICE=/path/to/kernels sh docs/basic_usage_cpp_code/build_and_run.sh
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
out="$root/build-cpp/docs_examples"
mkdir -p "$out"
for src in "$root"/docs/basic_usage_cpp_code/*.cpp; do
    name=$(basename "$src" .cpp)
    extra=""
    if [ "$name" = utc2et ]; then
        # links NAIF's CSPICE directly, not libcampanella
        cspice="$root/third_party/cspice"
        c++ -std=c++17 -Wall -Wextra -O2 "$src" -I "$cspice/include" "$cspice/lib/cspice.a" -lm -o "$out/$name"
        echo "== $name"
        (cd "$root" && "$out/$name") || true
        continue
    fi
    if [ "$name" = two_cspice ]; then
        # links NAIF's CSPICE in addition to libcampanella
        cspice="$root/third_party/cspice"
        extra="-I $cspice/include $cspice/lib/cspice.a -lm"
    fi
    if [ "$name" = gsl_driver ]; then
        # a program that drives the eom_* functions itself needs its own GSL
        gsl="$root/build-cpp/_deps/gsl-build/install"
        extra="-I $gsl/include $gsl/lib/libgsl.a $gsl/lib/libgslcblas.a"
    fi
    c++ -std=c++17 -Wall -Wextra -O2 "$src" -I "$root/include" \
        -L "$root/build-cpp/lib" -lcampanella -Wl,-rpath,"$root/build-cpp/lib" $extra -o "$out/$name"
    echo "== $name"
    (cd "$root" && "$out/$name") || true
done
