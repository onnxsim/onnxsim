#!/usr/bin/env bash
# Run the re-fusion survey on an NVIDIA GPU (developed against an RTX 2060).
#
#   bash bench/refusion_gpu_run.sh            # set up + run everything
#   bash bench/refusion_gpu_run.sh --quick    # tiny model, fast smoke test
#
# It creates .venv-refusion/, installs onnxruntime-gpu and builds onnxsim from
# this checkout, then writes two JSON files in the repo root:
#
#   refusion_cpu_<host>.json   -- pattern-level survey (fusion counts)
#   refusion_gpu_<host>.json   -- CUDA EP latency, fp32 + fp16
#
# Those two files are the result; everything else is progress output.
set -euo pipefail

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
VENV="$REPO_ROOT/.venv-refusion"
HOST="$(hostname -s 2>/dev/null || echo host)"

if [[ ! -d "$VENV" ]]; then
  echo "==> creating $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> installing dependencies"
pip install -q --upgrade pip
# onnxruntime-gpu ships the CUDA 12 / cuDNN 9 kernels; no system CUDA needed.
# The CPU onnxruntime package must NOT be installed alongside it.
pip uninstall -q -y onnxruntime 2>/dev/null || true
pip install -q onnxruntime-gpu onnx numpy sympy coloredlogs packaging psutil

echo "==> building onnxsim from this checkout (a few minutes the first time)"
git submodule update --init --recursive
pip install -q .

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

if [[ "$QUICK" == "1" ]]; then
  PRESET=tiny
  ITERS=20
else
  PRESET=bert-base
  ITERS=100
fi

echo "==> CPU / pattern survey (preset=$PRESET)"
# --bisect names the optimizer pass behind any fusion the simplified model lost;
# it re-runs simplify once per default pass, so it is the slow part of this run.
python bench/refusion_survey.py \
  --preset "$PRESET" \
  --bisect \
  --json "refusion_cpu_${HOST}.json"

echo "==> GPU survey on the CUDA execution provider (preset=$PRESET)"
python bench/refusion_gpu_bench.py \
  --preset "$PRESET" \
  --iters "$ITERS" \
  --json "refusion_gpu_${HOST}.json"

echo
echo "done -- share these two files:"
ls -la "refusion_cpu_${HOST}.json" "refusion_gpu_${HOST}.json"
