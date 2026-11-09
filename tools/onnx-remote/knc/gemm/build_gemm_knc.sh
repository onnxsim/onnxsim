#!/usr/bin/env bash
# Build the Knights Corner (k1om) SGEMM benchmark and the kernel microbenchmark, statically linked.
# Needs the k1om toolchain (NFCXX_K1OM_ROOT, default /opt/k1om). Outputs go to $OUT (default build/knc-gemm/).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
k1om=${NFCXX_K1OM_ROOT:-/opt/k1om}
out=${OUT:-$PWD/build/knc-gemm}
[ -x "$k1om/bin/k1om-mpss-linux-gcc" ] || { echo "build_gemm_knc: no k1om gcc at $k1om/bin" >&2; exit 1; }
mkdir -p "$out"
cc=("$k1om/bin/k1om-mpss-linux-gcc" -O3 -std=gnu99 -static -pthread -w -I"$here")
"${cc[@]}" "$here/gemm_knc.c" -o "$out/gemm_knc"
"${cc[@]}" -DKC=128 -DPG=2 -DRC=32 "$here/bench_kernel.c" -o "$out/bench_kernel"
echo "$out/gemm_knc $out/bench_kernel"
