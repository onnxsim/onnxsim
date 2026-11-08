#!/bin/bash
# Compile the PLaMo row-block layer programs (export_chunks.py --rows ...) for the v69 phone: HMX fp16 GEMMs, a 3 MB VTCM
# budget, and the HMX weight prefetch ahead turned off (HMX_PF_AHEAD=0). The prefetch ahead costs 2x on the weight pack
# (2560x2560 slice matmul: 5.3 ms with it, 2.6 ms without; the outputs are identical).
#   compile_blocks.sh ONNX_DIR OUT_DIR
# Needs TINYGRAD_ROOT, HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN (see onnx_remote/compile_v69.sh).
set -eo pipefail
src=$1; out=$2
: "${TINYGRAD_ROOT:?set TINYGRAD_ROOT to the tinygrad fork checkout}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
: "${HEXAGON_TOOLCHAIN:?set HEXAGON_TOOLCHAIN}"
export PYTHON="${PYTHON:-python3}" CC="${CC:-clang-19}"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
for onnx in "$src"/chunk_*.onnx; do
  b=$(basename "$onnx" .onnx)
  GEMM_FP16=1 HMX=1 HMX_VTCM_KB=3072 HMX_PF_AHEAD=0 \
    "$here/../onnx_remote/compile_v69.sh" "$onnx" "$out/$b.tghx" "$out/$b.manifest.json" 2>&1 | tail -1
done
