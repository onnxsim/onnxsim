# Importing GGUF weight values (`import_gguf_weights`)

## What this is

`onnxsim.import_gguf_weights(model, gguf_path)` hydrates an *existing* ONNX
graph's initializers, by name, from any GGUF file -- including a plain
third-party checkpoint like a Hugging Face GGUF export (e.g. Unsloth's Qwen3
GGUFs), which has no ONNX graph inside it at all, just weight tensors and
llama.cpp architecture metadata.

This is different in kind from `onnxsim.import_gguf`: that function requires
a GGUF file `onnxsim.export_gguf` itself produced (an embedded `model.onnx`
blob alongside the tensors -- the same self-describing-archive trick
`export_safetensors`/`import_safetensors` use), and reconstructs the whole
graph from it. A real quantized-LLM `.gguf` has no such embedded graph, so
`import_gguf` cannot open one. `import_gguf_weights` needs no embedded
model: bring your own graph for the same architecture (e.g. exported by
another tool), with initializers named to match the checkpoint's own tensor
names, and this fills in their values.

```python
import onnx
import onnxsim

model = onnx.load("qwen3_architecture.onnx")  # from some other exporter
model, skipped = onnxsim.import_gguf_weights(model, "model.gguf")
onnx.save(model, "qwen3_hydrated.onnx")
```

`skipped` lists GGUF tensors present in the file whose quantized format has
no decoder here (see "Scope" below) -- **not** tensors simply absent from
`model`'s initializers, which are silently left alone.

## GGML "K-quant" support

Real quantized checkpoints store most of their weights as one of GGML's
block-quantized formats, not plain float. This function decodes the
**K-quant** family -- `Q4_K`, `Q5_K`, `Q6_K` (256-element super-blocks, each
with its own packed 6-bit per-sub-block scale/min pair) and `Q8_0`
(32-element blocks, one fp16 scale each) -- which is what Unsloth's `*_K_M`/
`*_K_S`/`Q8_0` GGUF exports actually use for the bulk of their tensors. Every
block layout and dequantization formula is transcribed directly from GGML's
own reference implementation
(https://github.com/ggml-org/ggml -- `ggml-common.h`'s block structs,
`ggml-quants.c`'s `dequantize_row_q*` functions) and cross-checked against an
independent from-scratch re-implementation over full random blocks before
being committed (see `onnxsim/ggml_kquant.h`'s file comment).

A matched K-quant tensor's initializer has its `data_type` forced to
`FLOAT` regardless of what `model` previously declared for it -- the
decoded values are only meaningful as float32.

## Scope

Handled:
- `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` -- decoded to float32.
- Any raw (already-unquantized) GGML type (`F32`, `F16`, `BF16`, `F64`,
  `I8`/`I16`/`I32`/`I64`) -- copied through unchanged, same as `import_gguf`.

Not handled (reported in `skipped`, left untouched):
- The legacy `Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_1` family, `Q2_K`/`Q3_K`/`Q8_K`,
  and every `IQ*` importance-matrix variant -- these have real, different
  block layouts this decoder does not implement.

Only an initializer whose *name* matches a GGUF tensor is ever touched;
`import_gguf_weights` never adds new initializers or otherwise changes the
graph's structure.

## Why not reconstruct the whole model from the GGUF file?

A `.gguf` LLM checkpoint's metadata (layer count, hidden size, attention/RoPE
configuration, ...) describes a llama.cpp-runtime model, not an ONNX
computation graph -- there is no `Add`/`MatMul`/`Attention` node structure to
extract. Building one from scratch for an arbitrary architecture is a much
larger undertaking (closer to a dedicated GGUF-to-ONNX exporter) than
onnxsim's scope here: `import_gguf_weights` solves the narrower, immediately
useful problem of getting a checkpoint's *weight values* into a graph you
already have.

## Tests

`tests/test_import_gguf_weights.py` writes real, byte-accurate GGUF v3
files containing hand-encoded `Q8_0`/`Q4_K` blocks with known values
(computing each expected float independently, not by reusing the C++
decoder under test) and checks `import_gguf_weights`' decoded result against
them, plus coverage for multi-dimensional shapes (GGML's
innermost-dimension-first `ne[]` vs ONNX's outermost-first shape), the
unsupported/unmatched skip list, and raw-dtype passthrough.
