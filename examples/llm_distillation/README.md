# LLM knowledge-distillation demo (~1B-parameter student)

A standalone example, unrelated to onnxsim's own ONNX-to-ONNX simplification
pipeline: it trains a smaller "student" causal LM to mimic a larger "teacher"
model via knowledge distillation (Hinton et al. -- temperature-scaled
soft-target KL divergence, combined with the usual next-token hard-label
loss).

## What it actually does

- **Student**: a real ~1.1B-parameter decoder-only architecture, matching
  TinyLlama-1.1B's published shape (`hidden_size=2048`, 22 layers, 32
  attention heads / 4 KV heads, 32000-token vocabulary).
- **Teacher**: a wider/deeper sibling architecture (roughly Open-LLaMA-3B
  sized) used as the distillation target.
- Both are randomly initialized by default -- this demonstrates the
  distillation *mechanics* (losses, gradient flow, checkpointing) end to end,
  not a pretrained-quality result. Pass `--teacher-model-id`/
  `--student-model-id` to distill from/into real Hugging Face Hub
  checkpoints instead.
- Training data is synthetic random token ids by default, for the same
  reason: swap `run_distillation`'s batches for a real tokenized corpus for
  an actual training run.

## Install

    pip install onnxsim[transformers]

Pulls in `torch`, `transformers`, and `optimum` -- the last only used by the
optional `--export-onnx` step below.

## Usage

Fast smoke test (tiny synthetic models, 2 steps, well under a second):

    python examples/llm_distillation/distill.py --tiny

A real run -- the default teacher (~3B) + student (~1.1B) together are
several GB of fp32 parameters, so this needs real RAM/GPU:

    python examples/llm_distillation/distill.py \
        --steps 1000 --batch-size 8 --device cuda \
        --output-dir ./distilled-tinyllama-student

Distilling from/into real pretrained checkpoints instead of random init:

    python examples/llm_distillation/distill.py \
        --teacher-model-id meta-llama/Llama-3.2-3B \
        --student-model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --steps 1000 --device cuda

## Closing the loop back to onnxsim

Once you have a distilled student checkpoint, that's exactly what
`onnxsim.export_transformers_model()` (see `onnxsim/transformers_export.py`,
and the "Transformers export" section of the top-level README) is for:
export it to ONNX via `optimum` and simplify the result in one call. Pass
`--export-onnx` to do this automatically right after training, or run it
yourself:

```python
import onnxsim

onnxsim.export_transformers_model("./distilled-student-demo", "./distilled-student-demo/onnx")
```
