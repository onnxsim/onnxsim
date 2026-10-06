#!/bin/bash
# onnx-remote-compiler command for the Hexagon v65 target: an ONNX model -> a tinygrad-generated standalone v65 DSP program,
# packed as a `tghx-v65` artifact for onnx-remote-hexagon-worker.
#
#   onnx-remote-compiler --port 39502 --cache-dir ~/.cache/onnxsim-v65 --target hexagon-v65 --compiler-id "tinygrad-$(git -C "$TINYGRAD_ROOT" rev-parse --short HEAD)" \
#     --command '.../openpilot_v65/compile_v65.sh {input} {output} {manifest}'
#
# The model goes through the fork's compile3.py capture (DEV=DSP under qemu, v65 settings) and dsp_graph_v65: emitted, checked
# bit for bit under qemu against the JIT (COMPILE_QEMU_CHECK=0 skips that), and built into the FastRPC skel. Needs TINYGRAD_ROOT
# and CC (a v65-capable clang); DSP_THREADS (default 4) is baked into the program.
#
# Portable mode, no Qualcomm SDK: COMPILE_BUILD=0 skips the FastRPC skel (qaic + hexagon-clang + hexagon-link are the only
# things that need it), leaving capture + emit + the bit-exact qemu check. Verified end to end with HEXAGON_SDK_ROOT and
# HEXAGON_TOOLCHAIN unset: tinygrad's DSPCompiler builds every v65 kernel with $CC and links -nostdlib, and _find_libgcc()
# only adds the SDK's libgcc.a when it can find one. So "emitted program under qemu: bit-exact vs the JIT" needs nothing but
# a v65-capable clang and qemu-hexagon-static. (--artifact still needs the skel: dsp_graph_v65.pack reads the tg_graph.so that
# --build produces, so an artifact is only produced in the default build mode.)
#
# Extra KEY=VAL arguments after the three paths are exported for the capture, so they are part of the compiler command and
# therefore of its cache key. The integer path for a QDQ model (see ../README.md, "Integer path"):
#   compile_v65.sh {input} {output} {manifest} ONNX_QDQ_INT_CONV=1 ONNX_QDQ_LUT=1 TC_OPT=1
set -eo pipefail
input=$1; output=$2; manifest=$3; shift 3
for kv in "$@"; do case "$kv" in *=*) ;; *) echo "compile_v65.sh: expected KEY=VAL, got $kv" >&2; exit 2 ;; esac; done
: "${TINYGRAD_ROOT:?}"
# HEXAGON_SDK_ROOT / HEXAGON_TOOLCHAIN are only needed by --build (the FastRPC skel: qaic + hexagon-clang +
# hexagon-link). Capture, emit and the bit-exact qemu check need neither -- tinygrad's DSPCompiler compiles
# v65 with $CC and links -nostdlib, and treats the SDK's libgcc.a as a best-effort extra. So COMPILE_BUILD=0
# gives a fully portable host-side compile with no Qualcomm SDK installed at all.
: "${COMPILE_BUILD:=1}"
build_args=(--build)
artifact_args=(--artifact "$output" --manifest "$manifest")
if [ "$COMPILE_BUILD" = 0 ]; then
  build_args=(); artifact_args=()   # dsp_graph_v65.pack reads the tg_graph.so that --build makes, so no artifact without it
elif [ -z "${HEXAGON_SDK_ROOT:-}" ] || [ -z "${HEXAGON_TOOLCHAIN:-}" ]; then
  echo "compile_v65.sh: --build needs HEXAGON_SDK_ROOT and HEXAGON_TOOLCHAIN (or COMPILE_BUILD=0 to skip the skel)" >&2
  exit 2
fi
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
python3 "$TINYGRAD_ROOT/examples/openpilot/dsp_graph_v65.py" "$work/model.pkl" "$work/g" "${check[@]}" "${build_args[@]}" "${artifact_args[@]}" \
  --onnx "$input" >> "$log" 2>&1 || { tail -30 "$log" >&2; exit 1; }
t2=$(date +%s)
grep -E "^(reference|emitted|artifact|built|timing)" "$log" >&2
echo "compile_v65: capture $((t1-t0)) s, export + check + build $((t2-t1)) s, total $((t2-t0)) s" >&2
