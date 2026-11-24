#!/bin/sh
# Compatibility wrapper; the portable entry point detects the operating system.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec cmake "-DTCC_INSTALL_DIR=${1:-$root/dist/mac}" -P "$root/build.cmake"
