# Ozaki scheme on Axera NPUs (AX650N and similar): handoff note

**Status: research note, not a roadmap.** This records what a session spent
investigating -- whether the Ozaki scheme (an accurate/high-precision matrix
multiplication technique) could be targeted at Axera's low-precision NPU
hardware -- so the investigation doesn't need to be re-run from scratch. No
code in this repo implements or depends on any of this; there is no existing
Ozaki-scheme integration anywhere in onnxsim.

## What the Ozaki scheme is

Ozaki et al. (2012), "Error-free transformations of matrix multiplication",
later extended to integer/Tensor-Core hardware ("Ozaki Scheme II" and
follow-ups by Mukunoki et al.). It splits each fp32/fp64 matrix into several
lower-precision components via error-free transformations (e.g. Dekker-style
splitting), runs multiple low-precision matmuls on those components on fast
low-precision hardware (fp16/bf16 lanes, or int8 Tensor Cores in the
int8-based variant), and sums the results plus correction terms to
reconstruct a result far more accurate than a single low-precision matmul.
It is fundamentally a **kernel-level numerical algorithm for the
multiply-accumulate itself**, not a graph-level optimization -- and it is
the *inverse* goal of ordinary model quantization: quantization accepts
accuracy loss for speed/memory, Ozaki-style schemes use the same
low-precision compute paths but add split + correction so accuracy is
**not** lost.

It rides on the same low-precision hardware paths quantization uses, but as
a technique it's independent of "quantization" as ML people usually mean it.
This repo's own quantization-drop tooling (`onnxsim/accuracy.py`:
`quantize_fp16`/`quantize_bf16`, `estimate_model_quantization_drop`,
`measure_accuracy_drop`) *measures* the loss ordinary quantization causes;
it doesn't attempt to recover it the way Ozaki-style schemes do.

## Why Axera hardware specifically

`scripts/axera/` already integrates a real toolchain (Pulsar2, compiling
ONNX to `.axmodel` for the AX6xx/AX8xx NPU line, exercised against real
AX650N hardware -- see `scripts/axera/README.md`). Axera's NPU matrix unit is
described (README.md's own hardware notes) as **int8/int16-in, fp16/fp32-out**,
with confirmed **int32 accumulation followed by per-channel requantization**
on real hardware -- exactly the datapath Ozaki-style schemes are designed
to exploit and correct around. That makes Axera a plausible, concrete
target to reason about, rather than a generic "NPU" thought experiment.

## What's confirmed to work in onnxsim's favor

- `MatMul`, `Add`, `Mul`, `Cast`, `Conv` are all in `AX650_SUPPORTED_OPS`
  (`scripts/axera/pulsar2_ops.py`) and are real, hardware-verified ops.
- Pulsar2 has already been shown (in this repo's own `legalize.py`) to
  accept hand-built, semantics-preserving multi-op decompositions of
  roughly the shape an Ozaki scheme needs: `dilated_conv_to_taps` rewrites
  one dilated `Conv` into N sliced 1x1 convs summed via a chain of `Add`
  nodes, verified bit/near-bit-exact against onnxruntime and confirmed to
  build and run correctly on real AX650N hardware. So "several small
  matmuls plus elementwise correction ops" is structurally within what the
  compiler accepts.
- The problem such a scheme would solve is empirically present on this
  hardware: a 675-op quantized vocoder deployed via Pulsar2 measured only
  3.33 dB SNR despite 661/675 individual layers scoring >0.99 cosine
  similarity -- diagnosed as accumulated per-layer INT8 quantization error
  compounding as a random walk (`scripts/axera/README.md`). No attempt at
  compensated summation, Kahan summation, double-double arithmetic, or any
  Ozaki-scheme-like correction exists anywhere in this repo today (a
  repo-wide search for "Ozaki", "Kahan", "two-sum", "compensated summation",
  "double-double" returns zero hits outside this note).

## What's confirmed to block a clean implementation

An Ozaki decomposition needs each low-precision matmul to run at an exact,
known, user-chosen scale so the correction terms can reconstruct the
high-precision result. That specific control surface is currently broken or
missing on the Pulsar2/AX650N path:

1. **No FP32/high-precision override for MatMul or Conv.**
   `layer_configs.data_type: "FP32"` is only accepted for elementwise ops
   (`LeakyRelu, Sigmoid, Relu, Add, Mul, Div, Sub, Concat, Softmax`) --
   never for `MatMul` or `Conv` (`Conv` only accepts a separate
   `output_data_type: "FP32"`, which doesn't help control the matmul's own
   working precision).
2. **Pulsar2's own mixed-precision matmul feature fails to build.**
   `quant.highest_mix_precision` hits a real `TileFailException` -- the
   attention path's promoted-precision matmul tile doesn't fit the NPU's
   memory budget. This is Axera's own attempt at something adjacent to
   what an Ozaki scheme needs, and it doesn't currently work.
3. **Pre-quantized QDQ graphs (the mechanism to pin exact per-matmul
   scales) crash the compiler.** `model_type: "QuantONNX"` is meant to let
   a user supply their own `QuantizeLinear`/`DequantizeLinear` scales
   instead of Pulsar2's automatic PTQ -- exactly what an Ozaki
   implementation would need to control each split matmul's scale
   precisely. Any `MatMul` whose weight input comes through a
   `DequantizeLinear` crashes Pulsar2's own PPQ-based
   `ax_quant_graph_optimize` pass (locked in as a minimal repro in
   `tests/test_axera_quantonnx.py`).
4. **The compiler auto-quantizes and auto-fuses the whole graph.** Even if
   several separately-scaled int8 matmuls got past the parser, Pulsar2's
   own quantizer re-quantizes essentially the whole graph automatically,
   and its fusion is opaque and can re-target sequences a hand-written
   decomposition depends on staying separate (e.g. it re-fuses `Pad` into
   `Conv` even after a rewrite deliberately separates them). There's no
   guarantee a decomposition's intended per-matmul scales survive.
5. **No general-purpose custom-op escape hatch.** Unsupported ops can hard-fail
   the whole build at the frontend parse stage (confirmed for `LRN`) rather
   than falling back to CPU, and there is no documented way to inject an
   arbitrary custom kernel for a single op -- a full alternate execution
   path, not a per-op plugin, is the only shape of extensibility Pulsar2
   offers (mirroring, incidentally, onnxsim's own `ModelExecutor`
   extensibility seam discussed earlier in this investigation -- see
   `onnxsim/onnxsim.h`, `docs/dlpack-executor.md`).

## Bottom line

The arithmetic primitives an Ozaki-scheme implementation needs exist on
Axera silicon (int8-in/fp32-out MAC unit, int32 accumulate + requantize,
MatMul/Add/Mul/Cast all NPU-legal, and precedent for hand-built multi-op
decompositions being accepted by the compiler). But the toolchain currently
lacks the fine-grained, reliable per-matmul scale/precision control such a
scheme depends on to reconstruct high accuracy -- the closest official
mechanisms (`highest_mix_precision`, `QuantONNX`) both fail on real
hardware today. Implementing this would mean fighting Pulsar2's automatic
(and, per this repo's findings, currently buggy) quantization and fusion
behavior rather than being cleanly expressible through any documented API.

## If this is picked up again

Concrete next steps, in the order they'd need to be resolved:

1. File/track the `QuantONNX` + `DequantizeLinear`-fed-`MatMul` crash
   upstream with Axera (repro already exists at
   `tests/test_axera_quantonnx.py`) -- it blocks the most direct route to
   per-matmul scale control, independent of Ozaki-scheme work specifically.
2. Once (or if) QDQ-controlled MatMul scales work, prototype the smallest
   possible Ozaki-scheme building block (a single split 2-term matmul with
   one correction term) as a hand-built ONNX subgraph, following the same
   pattern `legalize.py`'s `dilated_conv_to_taps` uses, and check whether it
   survives Pulsar2's fusion/quantization passes with scales intact.
3. Measure whether the NPU's per-op overhead (this repo's `README.md` notes
   fixed per-layer overhead dominating small-model decode) makes a
   several-matmuls-per-original-matmul decomposition worth it on this
   hardware at all, versus just deploying at higher native precision
   (`W8A16` in the LLM path) where accuracy is the actual constraint.
