#!/bin/bash
# AWQ-INT4 quantize an LLM (backbone + lm_head) with TensorRT Edge-LLM on a GPU with only
# ~8 GB of memory (for example an RTX 5050).
#
#   quantize_int4_small_gpu.sh EDGE_LLM_CHECKOUT MODEL_DIR OUTPUT_DIR [PYTHON]
#
# Why: tensorrt_edgellm/quantization/quantize.py hard-codes calibration batch_size = 16 for
# int4_awq. With --lm_head_quantization int4_awq the lm_head AWQ search materialises
# batch x 512 x vocab fp32 logits (2.3 GiB for 8 samples at vocab 151936) and OOMs an 8 GB
# card regardless of --num_samples. This runs the quantizer from a scratch copy of the
# package with batch_size = 2 and leaves the checkout untouched.
set -euo pipefail
CHECKOUT=$1; MODEL=$2; OUT=$3; PY=${4:-python}
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/edgellm-quant-XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
cp -r "$CHECKOUT/tensorrt_edgellm" "$SCRATCH/"
sed -i 's/batch_size = 16 if quantization in (None, "int4_awq")/batch_size = 2 if quantization in (None, "int4_awq")/' \
  "$SCRATCH/tensorrt_edgellm/quantization/quantize.py"
grep -q 'batch_size = 2 if quantization' "$SCRATCH/tensorrt_edgellm/quantization/quantize.py" \
  || { echo "patch did not apply: quantize.py changed upstream" >&2; exit 1; }
PYTHONPATH="$SCRATCH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m tensorrt_edgellm.scripts.quantize llm \
    --model_dir "$MODEL" --output_dir "$OUT" \
    --quantization int4_awq --lm_head_quantization int4_awq --num_samples 64
