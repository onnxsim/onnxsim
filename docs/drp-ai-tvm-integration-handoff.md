# DRP-AI TVM (Renesas RZ/V) integration: handoff note

**Status: research/feasibility note, not a roadmap.** This records what a
session spent investigating -- whether/how onnxsim could feed models into
Renesas' DRP-AI TVM toolchain -- so the investigation doesn't need to be
re-run from scratch. No code in this repo implements or depends on any of
this; there is no existing DRP-AI TVM integration anywhere in onnxsim.

## What DRP-AI TVM is

[DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) is Renesas'
compiler stack (now folded into their "RUHMI" framework, powered by
EdgeCortix MERA) for deploying models on the DRP-AI accelerator built into
the RZ/V2L, RZ/V2M, RZ/V2MA, RZ/V2H and RZ/V2N MPU lines. It is a fork of
**Apache TVM**: an ONNX (or PyTorch/TFLite/EXIR) model is imported through
TVM's `relay.frontend.from_onnx` (wrapped as `mera2_.from_onnx()`), and
`drp.build()`/the MERA backend partitions the resulting Relay graph between
the DRP-AI accelerator and the CPU, then runs Renesas' own DRP-AI Translator
to generate the accelerator microcode. This is structurally the same role
`scripts/axera/` already plays for Axera's Pulsar2 (ONNX in, vendor NPU
binary out) -- so it's a fair question whether the same "onnxsim legalizes
the graph before the vendor compiler sees it" pattern applies here too.

The repo was cloned read-only from
`https://github.com/renesas-rz/rzv_drp-ai_tvm` (public) to read its actual
source and docs; nothing from it is vendored into onnxsim.

## Why a hardware-verified integration like `scripts/axera/` wasn't possible this session

Axera's Pulsar2 ships as a public Docker image, which is why
`scripts/axera/pulsar2_docker.py` could drive a real compile end-to-end.
DRP-AI TVM's own `Dockerfile` instead `COPY`s in, and installs, two
artifacts that are **not** fetchable from a public URL:

- The **DRP-AI Translator** installer (`DRP-AI_Translator*-Install` or
  `DRP-AI_Translator_i8*-Install`), gated behind Renesas' product pages
  (registration required).
- The **RZ/V AI SDK** (`RTK0EF01...zip`, one per board), same gating, and
  intended to be paired with a specific evaluation board (RZ/V2L/M/MA/H/N
  EVK).

DRP-AI TVM's own build then compiles these, plus its `tvm/` submodule,
locally into `mera2_r*.whl` / `tvm-*.whl` / `mera2_c*.whl` wheels -- there is
no `pip install` for the real backend. Without a Renesas account and (for
on-target validation) a physical evaluation board, the DRP-AI codegen and
translator cannot be built or exercised here, so nothing below about actual
compile success/failure on real hardware is verified -- only what's readable
from the toolchain's own source and documentation.

## What's confirmed by reading the actual pinned source

- **The ONNX frontend is genuinely stock, old Apache TVM.** `.gitmodules`
  pins `tvm` to branch `v0.8` at commit `046910a4100c5a822133ade5dfe851a1eb0ad95a`
  (frozen since ~2022), and vendors ONNX at `rel-1.10.1` (opset ceiling 15).
  Fetching `python/relay/frontend/onnx.py` from that exact commit and
  extracting `_get_convert_map()` gives the real op support surface: **189
  ONNX op types**, independent of anything a live pip-installed modern TVM
  would report (which would be misleading here -- DRP-AI TVM does not track
  upstream TVM).
- **Concrete, current gaps against onnxsim's own op vocabulary**: `GroupNormalization`,
  `Mish`, `STFT`, `DFT`, `Col2Im`, `LpNormalization`, the `Bitwise*` ops, and
  the window-function ops (`HammingWindow`/`HannWindow`/`BlackmanWindow`) are
  **not** in the v0.8 convert map -- all postdate opset 15. `GroupNormalization`
  and `Mish` in particular are common in vision/diffusion and newer YOLO
  exports, so a model using either would need to be legalized (decomposed to
  ops the v0.8 frontend does know, e.g. `InstanceNormalization`+reshape for
  `GroupNormalization`, `Softplus`+`Tanh`+`Mul` for `Mish`) before DRP-AI TVM
  can import it at all. **No such decomposition pass exists in onnxsim or its
  onnx-optimizer fork today** (checked by grep; only unrelated substring hits).
  Conversely opset-15-and-earlier ops onnxsim/onnx-optimizer already targets
  (`LayerNormalization`, `Einsum`, `Trilu`, `GridSample`, `ScatterND`,
  `Optional*`, `CastLike`, `ReduceL1`) are all present in the v0.8 map, so
  most of onnxsim's normal simplification output is already within range.
- **`docs/Error_List.md` names hard, structural backend limits** that read
  like exactly the kind of thing a graph-level legalization pass exists to
  work around (mirroring `scripts/axera/legalize.py`'s role for Pulsar2):
  - `TVMError: Only 3d or 1d input tensors are supported by drp translator`
    / the same for output tensors -- the translator's own boundary
    constraint, not TVM's.
  - `Check failed: current_node->input_size() == 2: Bias not expected to be
    merged into the convolution yet` / `Check failed: bias_add_axis == 1` /
    `Check failed: attr->axis == 1` -- the codegen assumes a specific
    Conv+Bias/BatchNorm channel-axis shape, i.e. it wants NCHW-canonical
    bias placement, not an arbitrary broadcast.
  - `Warning: Network requires memory too big` -- a hard budget on the
    DRP-AI-specific memory area, i.e. a real motivation for size-reducing
    passes (onnxsim already has several: constant folding, weight pruning,
    quantization) ahead of compile, not just after.
  - `Resize mode not supported` -- another point where onnxsim's existing
    `Resize`-related rewrites (it already has some, for other backends)
    would need auditing against whichever specific modes DRP-AI's codegen
    accepts.
- **onnxsim already has a real QDQ ingest/export path that lines up with
  DRP-AI TVM's quantized-target API.** `docs/Compile_API.md` describes
  `mera2_.from_onnx_qat(model_file, shape_dict, qat_type, ...)` for V2H/V2N
  (`QatType.from_str("PyTorch")` or `"TensorFlow"`) as the way to hand DRP-AI
  TVM a model whose `QuantizeLinear`/`DequantizeLinear` scales were already
  learned rather than PTQ-calibrated. `onnxsim/qat_interop.py` (see
  `docs/qat.md`) already does the general version of exactly this: it
  ingests a QDQ graph with learned scales, canonicalizes it back to onnxsim's
  internal representation without discarding the learned values, and can
  round-trip it back out (`export_fake_quant`). It is also deliberately
  conservative about exactly the kind of QDQ shape it won't guess at (a
  non-constant scale, a mismatched per-axis length) -- which is the same
  category of thing that crashed Axera's own QDQ-fed-MatMul path per
  `docs/ozaki-scheme-axera-handoff.md`. This is the most concrete, already-built
  piece of leverage onnxsim has toward this integration, but it has not been
  exercised against a real DRP-AI TVM QAT import in this session (no board,
  no translator) -- so "lines up with the documented API" is as far as this
  note can confirm it.

## What's confirmed to block a clean implementation

1. **No way to verify any of the above end-to-end in this environment.**
   Everything above is read from source/docs, not exercised through the real
   translator/codegen. In particular, whether a legalized graph that clears
   the documented op/shape/axis constraints actually compiles and runs
   correctly on silicon is unverified -- the Axera investigation found real,
   surprising toolchain bugs (crashing QDQ-fed MatMul, broken
   `highest_mix_precision`) that only turned up by actually running the
   compiler; DRP-AI TVM likely has its own such surprises that no amount of
   source-reading will surface.
2. **The frozen `v0.8` TVM pin is a moving target in the wrong direction.**
   Any legalization work has to target this specific 2022-era op/opset
   ceiling, not current upstream TVM and not onnxsim's own onnx fork's
   opset target -- the two will keep drifting apart as onnxsim's own default
   opset advances, unlike Axera's Pulsar2 (a maintained, versioned product
   onnxsim can just re-check against a new Docker tag).
3. **The codegen errors in `Error_List.md` are mostly "contact Renesas
   Electronics"**, i.e. undocumented internal assumptions rather than a
   published constraint list -- `legalize.py`-style rewrites for the ones
   that *are* documented (the 1D/3D-only boundary, the bias-axis-1
   assumption) would still need real-hardware validation to know they're
   sufficient, not just necessary.
4. **The DRP-AI Pre-processing Runtime** (a separate compile module for
   camera/image preprocessing pipelines, `docs/PreRuntime.md`) is a distinct
   DSL/toolchain outside the ONNX graph entirely -- out of scope for
   anything onnxsim does, and not investigated here.

## Bottom line

Structurally this is a good fit for onnxsim's existing role (the same
"legalize an ONNX graph for a vendor NPU compiler's real, narrower op/shape
support" job `scripts/axera/` already does), and onnxsim already has two
concrete pieces of leverage confirmed by reading the pinned source: most of
its own simplification output already falls inside DRP-AI TVM's v0.8-era op
set, and `onnxsim/qat_interop.py`'s QDQ round-trip lines up with the
documented `from_onnx_qat` API for the quantized (V2H/V2N) targets. But
unlike Axera's Pulsar2, nothing here could be run: the toolchain is gated
behind a Renesas account and (for real validation) a physical evaluation
board, so every finding above is "confirmed by source/docs," never
"confirmed by compiling." That gap matters concretely -- the Axera
investigation's most important findings (a crashing QDQ path, a broken
mixed-precision feature) were only discoverable by actually running the
compiler.

## If this is picked up again

Concrete next steps, in the order they'd need to be resolved:

1. Get a Renesas account, download the DRP-AI Translator + one board's RZ/V
   AI SDK (RZ/V2L is the least specialized target -- no quantization
   required, unlike V2H/V2N), and build the Docker image from DRP-AI TVM's
   own `Dockerfile`. This alone would upgrade every "confirmed by
   source/docs" finding above to "confirmed by compiling."
2. Once buildable, write `scripts/drp_ai/op_coverage.py` mirroring
   `scripts/axera/op_coverage.py`: compile a batch of onnxsim's own test
   fixtures (pre- and post-`simplify()`) through `mera2_.from_onnx()` and
   record which fail at the frontend-import stage (op not in the v0.8
   convert map -- fixable by a legalization pass) versus the codegen stage
   (an `Error_List.md`-cataloged backend limit -- needs the graph reshaped
   around it, or is a genuine toolchain bug to report upstream).
3. Prototype decompositions for the confirmed-missing ops that matter most
   in practice -- `GroupNormalization` and `Mish` first, both common in
   models onnxsim already sees elsewhere in this repo's fixtures/examples --
   as an onnx-optimizer pass, the same shape as this repo's existing
   opset-downgrade rewrites.
4. Once a QDQ model is available, exercise `onnxsim/qat_interop.py`'s
   `export_fake_quant`/`quantize_static_keeping_qdq_scales` round trip
   against real `mera2_.from_onnx_qat()` calls on V2H/V2N to see whether the
   scales/zero-points it produces are accepted, or whether DRP-AI TVM's own
   quantizer has the kind of undocumented shape assumptions that broke
   Pulsar2's `QuantONNX` path.
5. Measure whether the "3d or 1d input/output tensors only" translator
   boundary constraint actually bites on realistic models (most vision
   models are 4D NCHW at the graph boundary) -- if it does, that is the
   single highest-value legalization pass to write, since it would otherwise
   hard-block compilation rather than just degrade a decomposition's
   performance.
