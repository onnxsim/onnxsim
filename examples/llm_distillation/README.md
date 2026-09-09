# LLM knowledge-distillation demo (~162M-parameter student)

A standalone example, unrelated to onnxsim's own ONNX-to-ONNX simplification
pipeline: it trains a smaller "student" causal LM to mimic a larger "teacher"
model via knowledge distillation (Hinton et al. -- temperature-scaled
soft-target KL divergence, combined with the usual next-token hard-label
loss). Both architectures are kept deliberately small so the whole thing
runs on CPU in a reasonable time -- this is a demo of the distillation
*mechanics*, not a from-scratch pretraining recipe.

## What it actually does

- **Student**: a real ~162M-parameter decoder-only architecture, matching
  the widely-used `JackFram/llama-160m` shape (`hidden_size=768`, 12 layers,
  12 attention heads, 32000-token vocabulary, no GQA) -- the same
  architecture commonly used as a speculative-decoding draft model.
- **Teacher**: a wider/deeper sibling architecture (~1.1B parameters,
  matching TinyLlama-1.1B's published shape) used as the distillation
  target.
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

A real run -- the default teacher (~1.1B) + student (~162M) together are
under 5GB of fp32 parameters, so this is feasible on CPU (slow) or a single
modest GPU:

    python examples/llm_distillation/distill.py \
        --steps 1000 --batch-size 8 --device cuda \
        --output-dir ./distilled-llama160m-student

Distilling from/into real pretrained checkpoints instead of random init:

    python examples/llm_distillation/distill.py \
        --teacher-model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --student-model-id JackFram/llama-160m \
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
