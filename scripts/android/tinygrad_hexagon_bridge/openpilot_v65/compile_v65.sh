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
#
# Extra KEY=VAL arguments after the three paths are exported for the capture, so they are part of the compiler command and
# therefore of its cache key. The integer path for a QDQ model (see ../README.md, "Integer path"):
#   compile_v65.sh {input} {output} {manifest} ONNX_QDQ_INT_CONV=1 ONNX_QDQ_LUT=1 TC_OPT=1
set -eo pipefail
input=$1; output=$2; manifest=$3; shift 3
for kv in "$@"; do case "$kv" in *=*) ;; *) echo "compile_v65.sh: expected KEY=VAL, got $kv" >&2; exit 2 ;; esac; done
: "${TINYGRAD_ROOT:?}" "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
work=$(mktemp -d "${TMPDIR:-/tmp}/onnxsim-v65-XXXXXX")
trap 'rm -rf "$work"' EXIT
export MOCKDSP=1 DEV=DSP CONV_PAD_MATERIALIZE=1 DSP_V65_HW=1 DSP_THREADS="${DSP_THREADS:-4}" NOLOCALS=1 BEAM=0 CC="${CC:-clang-19}" \
  FLOAT16=0 ONNX_FP16_AS_FP32=1 BENCH_RUNS=1 COMPILE3_SKIP_SELFTEST=1 DSP_ALL_INPUTS=1 ALL_OUTPUTS=2 DSP_V65_VGATHER=1 DSP_V65_PERF_VOTE=3 ONNX_QDQ_REQUANT=1 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$TINYGRAD_ROOT/examples/openpilot:$TINYGRAD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for kv in "$@"; do export "${kv?}"; done  # after the defaults, so an extra can override one
log="$work/compile.log"
t0=$(date +%s)
python3 "$TINYGRAD_ROOT/examples/openpilot/compile3.py" "$input" "$work/model.pkl" > "$log" 2>&1 || { tail -30 "$log" >&2; exit 1; }
t1=$(date +%s)
check=(--qemu); [ "${COMPILE_QEMU_CHECK:-1}" = 0 ] && check=()
python3 "$TINYGRAD_ROOT/examples/openpilot/dsp_graph_v65.py" "$work/model.pkl" "$work/g" "${check[@]}" --build \
  --onnx "$input" --artifact "$output" --manifest "$manifest" >> "$log" 2>&1 || { tail -30 "$log" >&2; exit 1; }
t2=$(date +%s)
grep -E "^(reference|emitted|artifact|built|timing)" "$log" >&2
echo "compile_v65: capture $((t1-t0)) s, export + check + build $((t2-t1)) s, total $((t2-t0)) s" >&2
