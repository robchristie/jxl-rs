#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libjxl_root="${LIBJXL_ROOT:-"$repo_root/reference/libjxl"}"
libjxl_build="${LIBJXL_BUILD:-"$libjxl_root/build-rs-oracle"}"
out="${OUT:-"$repo_root/target/reference-trace/jxl_vardct_trace"}"

mkdir -p "$(dirname "$out")"

cxx="${CXX:-c++}"
"$cxx" \
  -std=c++17 \
  -O2 \
  -DNDEBUG \
  -DJXL_ENABLE_SKCMS=0 \
  -DJXL_ENABLE_TRANSCODE_JPEG=1 \
  -I"$libjxl_root" \
  -I"$libjxl_build" \
  -I"$libjxl_root/lib/include" \
  -I"$libjxl_root/third_party/highway" \
  "$repo_root/tools/reference-trace/jxl_vardct_trace.cc" \
  "$libjxl_build/lib/libjxl-internal.a" \
  "$libjxl_build/third_party/highway/libhwy.a" \
  -pthread \
  -lm \
  -o "$out"

printf '%s\n' "$out"
