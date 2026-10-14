#!/bin/sh
# Fetch the official NAIF CSPICE archive for this host into third_party/cspice.
# Linux x86_64: PC_Linux_GCC_64bit. Apple Silicon: MacM1_OSX_clang_64bit. Intel macOS: MacIntel_OSX_AppleC_64bit.
set -e
os=$(uname -s)
arch=$(uname -m)
case "$os:$arch" in
  Darwin:arm64) pkg=MacM1_OSX_clang_64bit ;;
  Darwin:x86_64) pkg=MacIntel_OSX_AppleC_64bit ;;
  Linux:x86_64) pkg=PC_Linux_GCC_64bit ;;
  *) echo "No CSPICE package for $os $arch. Unpack a toolkit and pass -DCSPICE_DIR." >&2; exit 1 ;;
esac
cd "$(dirname "$0")/../third_party"
curl -fL "https://naif.jpl.nasa.gov/pub/naif/toolkit/C/${pkg}/packages/cspice.tar.Z" | tar xz
