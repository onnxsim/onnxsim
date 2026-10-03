#!/bin/bash
# onnx-remote-compiler command for the Hexagon v65 target: an ONNX model -> a tinygrad-generated standalone v65 DSP program,
# packed as a `tghx-v65` artifact for onnx-remote-hexagon-worker.
#
#   onnx-remote-compiler --port 39502 --cache-dir ~/.cache/onnxsim-v65 --target hexagon-v65 --compiler-id "tinygrad-$(git -C "$TINYGRAD_ROOT" rev-parse --short HEAD)" \
#     --command '.../openpilot_v65/compile_v65.sh {input} {output} {manifest}'
#
# The model goes through the fork's compile3.py capture (DEV=DSP under qemu, v65 settings) and dsp_graph_v65: emitted, checked
# bit for bit under qemu against the JIT (COMPILE_QEMU_CHECK=0 skips that), and built into the FastRPC skel. Needs TINYGRAD_ROOT,
# HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN and CC (a v65-capable clang); DSP_THREADS (default 4) is baked into the program.
set -eo pipefail
input=$1; output=$2; manifest=$3
: "${TINYGRAD_ROOT:?}" "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
work=$(mktemp -d "${TMPDIR:-/tmp}/onnxsim-v65-XXXXXX")
trap 'rm -rf "$work"' EXIT
export MOCKDSP=1 DEV=DSP CONV_PAD_MATERIALIZE=1 DSP_V65_HW=1 DSP_THREADS="${DSP_THREADS:-4}" NOLOCALS=1 BEAM=0 CC="${CC:-clang-19}" \
  FLOAT16=0 ONNX_FP16_AS_FP32=1 BENCH_RUNS=1 DSP_ALL_INPUTS=1 ALL_OUTPUTS=1 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$TINYGRAD_ROOT/examples/openpilot:$TINYGRAD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
log="$work/compile.log"
python3 "$TINYGRAD_ROOT/examples/openpilot/compile3.py" "$input" "$work/model.pkl" > "$log" 2>&1 || { tail -30 "$log" >&2; exit 1; }
check=(--qemu); [ "${COMPILE_QEMU_CHECK:-1}" = 0 ] && check=()
python3 "$TINYGRAD_ROOT/examples/openpilot/dsp_graph_v65.py" "$work/model.pkl" "$work/g" "${check[@]}" --build \
  --onnx "$input" --artifact "$output" --manifest "$manifest" >> "$log" 2>&1 || { tail -30 "$log" >&2; exit 1; }
grep -E "^(reference|emitted|artifact)" "$log" >&2
