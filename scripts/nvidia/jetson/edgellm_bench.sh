#!/bin/bash
# Benchmark TensorRT Edge-LLM engines on a Jetson Orin with clocks locked, then restore.
#
#   edgellm_bench.sh [-e EDGE_LLM_DIR] ENGINE_DIR [ENGINE_DIR ...]
#
# Runs llm_bench prefill (128, 512 tokens) and decode (128, 1024 past tokens) on each
# engine, prints time and tokens/s, and compares decode against the memory roofline given
# the engine's weight bytes. Needs sudo for nvpmodel/jetson_clocks (password prompt is the
# caller's problem; use `sudo -v` first).
#
# Locks MAXN_SUPER (nvpmodel -m 2) + jetson_clocks, then goes back to the previous nvpmodel
# mode. It deliberately does not use `jetson_clocks --store/--restore`: --store hung on
# JetPack 7.2.1 after an earlier --restore had failed. Switching nvpmodel back resets clocks.
set -euo pipefail

EDGE_LLM_DIR="${EDGE_LLM_DIR:-$HOME/TensorRT-Edge-LLM}"
while getopts "e:" opt; do
  case $opt in e) EDGE_LLM_DIR="$OPTARG" ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { echo "usage: $0 [-e EDGE_LLM_DIR] ENGINE_DIR..." >&2; exit 2; }
ENGINES=()
for e in "$@"; do ENGINES+=("$(realpath "$e")"); done

export PATH=/usr/local/cuda/bin:$PATH
# llm_bench finds the AttentionPlugin library relative to the working directory.
cd "$EDGE_LLM_DIR"
BENCH="./build/examples/llm/llm_bench"
PEAK_GBS=102.4   # Orin Nano/NX 128-bit LPDDR5 at 3199 MHz; use 204.8 on AGX Orin

prev_mode=$(sudo nvpmodel -q | awk '/NV Power Mode/ {print $NF}')
prev_id=$(sudo nvpmodel -q | tail -1)
restore() { sudo nvpmodel -m "$prev_id" || true; echo "restored power mode $prev_mode"; }
trap restore EXIT
sudo nvpmodel -m 2
sudo jetson_clocks
sleep 2
sudo nvpmodel -q | head -1

for engine in "${ENGINES[@]}"; do
  # Bytes read per decoded token = everything in the engine dir the runtime streams every
  # step: the engine plus external weight files, minus the embedding (a lookup).
  weight_bytes=$(find "$engine" -maxdepth 1 \( -name '*.engine' -o -name 'external_*.safetensors' \) \
                 -printf '%s\n' | awk '{s+=$1} END {print s}')
  echo "== $engine  (weights streamed per token: $((weight_bytes / 1000000)) MB)"
  for mode in "prefill --inputLen 128" "prefill --inputLen 512" \
              "decode --pastKVLen 128" "decode --pastKVLen 1024"; do
    out=$("$BENCH" --engineDir "$engine" --mode $mode 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    ms=$(grep -m1 "E2E Time (actual" <<<"$out" | sed 's/.*: \([0-9.]*\) ms/\1/' || true)
    tps=$(grep -m1 "Tokens/sec" <<<"$out" | sed 's/.*: //' || true)
    if [ -z "$ms" ]; then
      echo "$mode: llm_bench failed:"; grep ERROR <<<"$out" | head -3 | cut -c1-200
      continue
    fi
    line=$(printf '%-26s %9s ms  %8s tok/s' "$mode" "$ms" "$tps")
    if [[ $mode == decode* ]]; then
      past=${mode##* }
      # 36-layer, 8 KV-head, head-dim-128 fp16 KV = 147 KB/token; scale for your model.
      kv=$((past * ${KV_BYTES_PER_TOKEN:-147456}))
      ideal=$(awk -v b=$((weight_bytes + kv)) -v g=$PEAK_GBS 'BEGIN {printf "%.1f", b / (g * 1e9) * 1e3}')
      ach=$(awk -v b=$((weight_bytes + kv)) -v ms="$ms" 'BEGIN {printf "%.0f", b / (ms * 1e-3) / 1e9}')
      line="$line   ideal ${ideal} ms @${PEAK_GBS} GB/s, achieved ${ach} GB/s"
    fi
    echo "$line"
  done
done
