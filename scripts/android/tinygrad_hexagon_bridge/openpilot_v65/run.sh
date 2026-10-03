#!/bin/bash
# Generate every openpilot modeld program with tinygrad as a standalone Hexagon v65 (SDM845 cDSP) program, check it bit for bit
# under qemu, build the FastRPC skel + client, and optionally run it on an attached phone. See ../README.md, "openpilot on v65".
#
#   run.sh model <name> <model.onnx>                          compile3 capture (fp32 weights, DSP_THREADS) -> export -> qemu -> build
#   run.sh warp  <name> <compile_warp.py args...>             the camera warps modeld builds (compile_warp.py in the fork)
#   run.sh phone <name> [iters] [threads] [batch] [prof]      push g_<name> to the phone and run it (holds the shared phone lock)
#
# Environment: TINYGRAD_ROOT (the onnxsim/tinygrad fork, branch openpilot-v65-graph), HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN,
# CC (a clang that targets hexagonv65; the SDK 6.x compiler starts at v68), WORK (output dir, default ./openpilot_v65_work),
# DSP_THREADS (default 4), DEVICE_SERIAL, PHONE_LOCK (default ~/.cache/android-phone/phone-run, if present).
set -eo pipefail
: "${TINYGRAD_ROOT:?set TINYGRAD_ROOT to the tinygrad fork checkout}"
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
CC="${CC:-clang-19}"; WORK="${WORK:-$PWD/openpilot_v65_work}"; DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
PHONE_LOCK="${PHONE_LOCK:-$HOME/.cache/android-phone/phone-run}"
mkdir -p "$WORK"; cd "$WORK"
ENV=(MOCKDSP=1 DEV=DSP CONV_PAD_MATERIALIZE=1 DSP_V65_HW=1 DSP_THREADS="${DSP_THREADS:-4}" NOLOCALS=1 BEAM=0 CC="$CC" HEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"
     HEXAGON_TOOLCHAIN="$HEXAGON_TOOLCHAIN" PYTHONUNBUFFERED=1 PYTHONPATH="$TINYGRAD_ROOT/examples/openpilot:$TINYGRAD_ROOT")
MODEL_ENV=(FLOAT16=0 ONNX_FP16_AS_FP32=1 BENCH_RUNS=1 DSP_ALL_INPUTS=1 ALL_OUTPUTS=1)
# heavy (qemu runs every kernel): one job at a time, memory-capped when systemd-run is available
run() { if command -v systemd-run >/dev/null; then
          systemd-run --user --wait --collect --pipe -q -p MemoryMax=24G -p MemorySwapMax=0 --working-directory="$PWD" -E PATH="$PATH" \
            $(for e in "${ENV[@]}" "${EXTRA[@]}"; do printf -- "-E %s " "$e"; done) "$@"
        else env "${ENV[@]}" "${EXTRA[@]}" "$@"; fi; }
export_graph() {  # <name> [--inputs npz]
  local name=$1; shift
  run python3 "$TINYGRAD_ROOT/examples/openpilot/dsp_graph_v65.py" "$WORK/$name.pkl" "$WORK/g_$name" "$@" --qemu --build > "g_$name.log" 2>&1 \
    || { grep -v -i warning "g_$name.log" | tail -20; exit 1; }
  grep -E "^(reference|emitted|built)" "g_$name.log"
}
cmd=$1; name=$2; shift 2
case "$cmd" in
  model)
    EXTRA=("${MODEL_ENV[@]}")
    [ -f "$name.pkl" ] || run python3 "$TINYGRAD_ROOT/examples/openpilot/compile3.py" "$1" "$WORK/$name.pkl" > "$name.capture.log" 2>&1 \
      || { tail -20 "$name.capture.log"; exit 1; }
    export_graph "$name" ;;
  warp)
    EXTRA=()
    [ -f "$name.pkl" ] || run python3 "$TINYGRAD_ROOT/examples/openpilot/compile_warp.py" "$@" --output "$WORK/$name.pkl" > "$name.capture.log" 2>&1 \
      || { tail -20 "$name.capture.log"; exit 1; }
    export_graph "$name" --inputs "$WORK/${name}_inputs.npz" ;;
  phone)
    d="$WORK/g_$name"; R=/data/local/tmp/openpilot_v65/$name
    job="adb -s $DEVICE_SERIAL shell mkdir -p $R && adb -s $DEVICE_SERIAL push -q $d/client $d/tg_graph.so $d/blob.bin $d/input.bin $d/ref.bin $R/ >/dev/null && \
adb -s $DEVICE_SERIAL shell 'cd $R && chmod 755 client && ADSP_LIBRARY_PATH=. ./client \"file:///tg_graph.so?tg_graph_skel_handle_invoke&_modver=1.0&_dom=cdsp\" . $*'"
    if [ -x "$PHONE_LOCK" ]; then PHONE_LOCK_OWNER="${PHONE_LOCK_OWNER:-openpilot-v65}" "$PHONE_LOCK" bash -c "$job"; else bash -c "$job"; fi ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
