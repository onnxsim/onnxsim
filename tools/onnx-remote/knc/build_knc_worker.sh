#!/usr/bin/env bash
# Build the onnx-remote reference worker for Intel Knights Corner (k1om), statically linked.
# Needs the k1om toolchain (docs in README.md; NFCXX_K1OM_ROOT, default /opt/k1om). Output: $OUT (default build/knc/).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/.." && pwd)
k1om=${NFCXX_K1OM_ROOT:-/opt/k1om}
out=${OUT:-$PWD/build/knc}
[ -x "$k1om/bin/k1om-mpss-linux-g++" ] || { echo "build_knc_worker: no k1om g++ at $k1om/bin" >&2; exit 1; }
mkdir -p "$out"
# The transport is C++17 code, built with the GCC 5 libstdc++ of the k1om toolchain as -std=gnu++14.
cxx=("$k1om/bin/k1om-mpss-linux-g++" -std=gnu++14 -O2 -w -c -I"$src")
for f in remote_transport remote_worker remote_profile; do
  "${cxx[@]}" "$src/$f.cpp" -o "$out/$f.o"
done
"$k1om/bin/k1om-mpss-linux-g++" -static -O2 "$out/remote_worker.o" "$out/remote_transport.o" "$out/remote_profile.o" \
  -o "$out/onnx-remote-worker-k1om"
echo "$out/onnx-remote-worker-k1om"
