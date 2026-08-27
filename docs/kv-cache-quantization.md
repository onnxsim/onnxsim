# KV-cache quantization (`quantize_kv_cache`)

## What this is

`onnxsim.quantize_kv_cache` finds every autoregressive decoder KV-cache
stream in a model -- a graph input (`past_key`/`past_key_values.{i}.key`,
...) concatenated with this step's freshly computed key or value along the
sequence axis, feeding a graph output (`present_key`/`present.{i}.key`,
...) that a caller feeds back in as next step's `past_*` input, exactly the
shape `tools/onnx-deploy`'s own `KvCachePipeline` and
`tests/test_symexpr_kv_cache_consistency.py`'s toy model both use -- and
quantizes it to INT8, symmetric, one scale per channel (the head-dim axis),
calibrated once from representative data and shared by every cached token
for that stream's whole lifetime.

Every other quantizer in onnxsim compresses a **weight**: something
computed once, offline, before the model ever runs. A KV cache is the
opposite -- an **activation** that keeps growing for the entire lifetime of
one generation, one new key/value vector appended per decode step. That is
exactly why it is worth quantizing at all: it is the part of an LLM's
memory footprint that scales with sequence length, unlike the weights.

```
Before:
  past_key: graph input, float32 [..., seq_past, head_dim]
  new_key:  float32 [..., seq_new, head_dim]         -- this step's own K/V
  present_key = Concat(past_key, new_key, axis=seq)  -- graph output,
                and consumed by the attention math (QK^T / softmax@V)

After:
  past_key: graph input, INT8 [..., seq_past, head_dim]    -- dtype changed
  key_scale: initializer, float32 [head_dim]                 -- per-channel
  key_zero_point: initializer, INT8 [head_dim], all zero     -- symmetric
  new_key_q = QuantizeLinear(new_key, key_scale, key_zero_point, axis=-1)
  present_key = Concat(past_key, new_key_q, axis=seq)   -- INT8 graph output
  present_key_f = DequantizeLinear(present_key, key_scale, key_zero_point,
                                    axis=-1)             -- float32
  <every other consumer of the old float present_key now reads present_key_f>
```

Concatenating `past_key` (already int8) with `new_key_q` (freshly quantized
with the *same* per-channel scale) along the sequence axis is lossless with
respect to what was already stored -- the scale never changes step to step,
so there is no compounding requantization error the way there would be if
the whole growing cache were dequantized and requantized with a fresh scale
every step. Only this step's new tokens are ever quantized, so the cost per
decode step stays constant as the sequence grows, and the graph's own
`present_*` output is genuinely compressed (roughly 4x smaller than
float32) the whole way through a caller's decode loop -- not just an
internal round-trip that still stores float32 everywhere.

## Where this comes from

Two published techniques quantize the KV cache well: **KIVI** (Liu et al.,
ICML 2024, <https://arxiv.org/abs/2402.02750>) and **KVQuant** (Hooper et
al., NeurIPS 2024, <https://arxiv.org/abs/2401.18079>). Both share the same
core empirical finding: Key activations have a handful of channels with
persistently large magnitude across the *whole* sequence, so quantizing Key
**per channel** (one scale shared by every cached token) preserves far more
accuracy than quantizing it per token. `quantize_kv_cache` reproduces that
part of both papers.

What it does **not** reproduce:

- **KIVI's per-token Value quantization** (a fresh scale per cached token,
  rather than one static per-channel scale). Per-token quantization would
  need the scale itself to be a second, parallel growing KV-cache stream
  (one scale per cached row, concatenated alongside the codes every step) --
  a real increase in I/O surface this module's MVP scope skips, applying
  the same static per-channel scheme to Value too. This matches what many
  serving engines' plain "int8/fp8 KV cache" flag already does in
  production, even though it is a documented simplification relative to
  KIVI's own per-token scheme for Value specifically.
- **KIVI's residual-window bookkeeping** (the most recent `R` tokens kept
  in full precision, only finalized into low-bit once they age out of that
  window). Deciding which tokens have "aged out" and need finalizing is
  cross-step, host-side state -- not something one exported ONNX graph can
  express on its own. It belongs in
  `tools/onnx-deploy/include/onnx_deploy/kv_cache_pipeline.h` (which
  already owns exactly this kind of cross-step cache state across decode
  steps) as a follow-up, not here.
- **KVQuant's non-uniform codebook and dense-and-sparse outlier isolation**
  -- both add real complexity (a fitted, non-uniform datatype; per-vector
  outlier separation) for a further accuracy gain past plain per-channel
  INT8; not implemented here, a natural next step if calibrated INT8 alone
  turns out not to be enough for a given model.

## Scope

Handled:

- A `Concat(past, new, axis=seq)` node whose `past` operand is a float32
  graph input consumed *only* by that Concat, and whose own output is
  directly a graph output. No assumption is made about tensor names -- this
  matches `past_key`/`present_key` as well as `optimum-onnx`'s own
  `past_key_values.{i}.key`/`present.{i}.key` convention.
- Opsets >= 13 (`QuantizeLinear`/`DequantizeLinear`'s per-channel `axis`
  needs opset 13).

Left untouched (safe no-op, node passes through as-is):

- A `past` operand with any other consumer besides the Concat (declining
  rather than silently breaking that other use).
- A Concat whose axis *is* the last (channel) axis -- no distinct axis is
  left to quantize per-channel on.
- A model with no matching Concat pattern, or an opset older than 13.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("decoder_with_past_model.onnx")
quantized = onnxsim.quantize_kv_cache(model, num_samples=32)
onnx.save(quantized, "decoder_with_past_model.kv_int8.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`
(random input, no external dependency); pass real representative batches
(e.g. via `onnxsim.load_huggingface_calibration_data`) for a tighter
calibrated scale, since a per-channel scale that under-covers a real
model's activation range clips outliers on exactly the channels KIVI/
KVQuant's own finding says matter most.

A model quantized this way needs a caller that actually stores its
`past_*`/`present_*` tensors as INT8 across decode steps to see any real
memory benefit -- `tools/onnx-deploy`'s `KvCachePipeline`
(`include/onnx_deploy/kv_cache_pipeline.h`) supports this:
`detail::BorrowView` handles INT8 tensors (alongside the original fp32/
int64) and threads them through `Generate()`'s decode loop exactly like any
other cache dtype, verified end-to-end by
`tools/onnx-deploy/scripts/make_toy_int8_kv_decoder.py` and
`.github/workflows/onnx-deploy.yml` against a real, growing INT8 cache
across many steps and two different ONNX Runtime releases. Note that a real
multi-file `optimum-onnx`-style export needs `decoder_model.onnx` (the
"no past" first step) quantized consistently with `decoder_with_past_model.onnx`
for the same cache stream too, since both files share the same
`present.*`/`past_key_values.*` dtype contract; `quantize_kv_cache` only
matches graphs with a `Concat(past, new)` pattern (present in the "with
past" file, absent in the first-step file, which has no past to concat),
so quantizing both files of a real pipeline consistently is left to the
caller for now.

`tests/test_kv_cache_quantization.py` verifies this end-to-end with a
genuine two-step round trip (an empty starting cache, then feeding the
first step's own INT8 `present_key` output back in as the second step's
`past_key` input -- exactly what a real pipeline does), comparing the
dequantized cache tensor against the float model's own output.
