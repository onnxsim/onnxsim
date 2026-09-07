# Delivering QAT in onnxsim: a design note

**Status: design note, not an implementation.** It answers "how *could* we
deliver quantization-aware training as an onnxsim feature, and can the
training math run on WebGPU or an NPU?" and records the shape of the work so
the question doesn't have to be re-derived. `docs/nncf-comparison-future-work.md`
currently lists QAT as out of scope ("QAT needs a training loop with a
framework-native model"); this note revisits that on the narrower reading
below, and leaves the broader one (task loss, labels, epochs) exactly as
out of scope as it was.

## The thing to notice first: most of QAT is already here

onnxsim does not have "no training". It has, today, six passes that are
gradient descent with a straight-through estimator, hand-derived, in plain
numpy:

| Module | What it optimizes by SGD/Adam | Objective |
| --- | --- | --- |
| `adaround.py` | per-element floor/ceil relaxation (rectified sigmoid) | one layer's output MSE |
| `adaquant.py` | rounding **+ activation scale/zero-point** (STE, LSQ-style) | one layer's output MSE |
| `brecq.py` | rounding, backpropagated through a **multi-layer block** | the block's final output MSE |
| `flexround.py` | a multiplicative rounding reparametrization | layer output MSE |
| `autoround.py` | rounding + clip ratio, LSQ-style scale gradient | layer output MSE |
| `omniquant.py` | learnable clip ratio | layer output MSE |

What separates these from QAT is not the machinery, it is four axes:

1. **What is free.** They optimize *which integer a weight rounds to* (and
   sometimes the scale). QAT lets the underlying float weight itself move.
2. **Scope.** One layer, or one hand-identified block (`brecq.py`'s
   `block_input_name`/`block_output_name`). QAT trains the whole model.
3. **Objective.** Local reconstruction against the float model's own
   activations. QAT uses a task loss on labelled data.
4. **Budget.** Hundreds of Adam steps on a handful of calibration batches,
   on CPU, in numpy. QAT is orders of magnitude more compute.

Axes 1-2 and 4 are engineering. Axis 3 is the architectural one: a task loss
means labels, a dataset API, a metric, and a training-loop lifecycle onnxsim
has never had. So split the feature there.

## Three deliverables, only two of which we should build

### A. QAT interop (graph-only, no training)

Consume and produce QAT artifacts without running any training:

- **Ingest.** A model exported from PyTorch/TF QAT arrives as a QDQ graph
  whose scales and zero-points were *learned*. Today's calibration-based
  entry points (`calibration.py`'s `calibrate` / `quantize_static`) would
  re-derive them from min/max observation and throw the learned values away.
  A `keep_existing_qdq_scales` path — detect QDQ pairs already present,
  canonicalize them, and let the rest of the pipeline (folding, QDQ→QOperator,
  `passes/`) run without re-calibrating — is pure graph rewriting and fits
  onnxsim's architecture exactly.
- **Egress.** Emit a fake-quant (QDQ) model whose scale/zero-point
  initializers are marked as the tensors an external trainer should make
  learnable, so a user can round-trip: onnxsim picks the scheme and the
  per-layer bit widths (`mixed_precision.py`, `precision_estimator.py`), the
  user trains in their framework, onnxsim re-imports the trained scales via
  the ingest path.

This is the cheapest real QAT value we can ship and it costs no new paradigm.

### B. Graph-native QAT = block-wise, label-free fine-tuning

The honest name for what onnxsim can do itself: **knowledge-distillation QAT**.
The float model is the teacher; the quantized model is the student; the loss
is reconstruction against the teacher's own activations — exactly the
objective `brecq.py` already optimizes, extended along axes 1, 2 and 4:

- weights themselves free (fp32 master weights, fake-quantized in the
  forward with an STE), not only their rounding bit;
- learnable step size for weights *and* activations (LSQ / LSQ+ / PACT) —
  `adaquant.py` already does this for activation scale/zero-point, so the
  gradients exist in-tree;
- block discovery that walks through normalization/activation nodes instead
  of `brecq.py`'s strict linear-chain restriction, then a sliding window over
  blocks, ending with an optional end-to-end pass on the whole graph;
- real data via `calibration.load_huggingface_calibration_data`, many epochs
  over it rather than one probe capture.

No labels, no task loss, no metric API — the teacher supplies the target.
That keeps the lifecycle onnxsim already has (model in, model out) and still
delivers the thing users actually want from QAT: recovering accuracy that
PTQ cannot, especially at 4 bits and below with activation quantization.

### C. Task-loss QAT

Out of scope, unchanged. If it is ever wanted, the seam is a caller-supplied
callback returning dL/d(model output) per batch — B's machinery would then
carry it — but the dataset/label/metric surface is a different project.

## Making it run on WebGPU and NPUs

The constraint that shapes the implementation: **the training step must be
expressible as an ordinary ONNX inference graph.**

That is achievable precisely *because* this codebase hand-derives its
gradients instead of using autodiff. A hand-derived backward pass is just
dataflow — `MatMul`, `Transpose`, `Mul`, `Sub`, `ReduceSum`, `Sigmoid`,
`Clip`, `Where`, `Sqrt` — with no autograd tape and no `Gradient` op. So one
Adam step over one block can be emitted as a single **step graph**:

```
inputs :  X (activations), Y* (teacher output), W, s, zp, m, v, t, lr
outputs:  W', s', zp', m', v', loss
```

a pure function, optimizer state carried in and out as tensors. Run it N
times and you have trained a block — using nothing but an inference runtime.

Every accelerator this project already reaches then comes for free, through
boundaries that exist today:

- **Python.** `backend.run_model(..., providers=[...])` (`onnxsim/backend.py`)
  already takes an ordered execution-provider list and already validates it.
  CUDA, DirectML, and the NPU-class EPs each vendor harness in `scripts/`
  already exercises — QNN (`scripts/qualcomm`), Core ML/ANE (`scripts/apple`),
  OpenVINO (`scripts/intel`), MIGraphX (`scripts/amd`), Axera pulsar2
  (`scripts/axera`) — take the same step graph unmodified.
- **Browser.** The WASM build already delegates model evaluation to
  onnxruntime-web through `JsModelExecutor` (`docs/wasm_ort_web.md`), and
  `scripts/convertmodel/ort_executor.mjs`'s `makeOrtRunner(ort, { providers })`
  already selects `webgpu`, `webnn-gpu` or `webnn-npu` with a `wasm` fallback
  (`docs/webnn.md`). A step graph is just another model to run there.

So the accelerator story is not a second backend to write; it is the existing
`ModelExecutor`/`providers` boundary, applied to a graph we already know how
to build. Three things it does *not* give for free:

- **Readback dominates if ignored.** A naive loop copies every parameter and
  every Adam moment host↔device per step. Parameters must stay resident:
  `IOBinding` on the Python side, and on the web side onnxruntime-web's
  GPU-buffer tensors with `preferredOutputLocation: "gpu-buffer"` (ORT-web
  ≥1.17; the converter page is on 1.27), feeding each step's outputs straight
  back in as the next step's inputs. Only the loss comes back to the host.
- **NPUs are inference silicon, and it shows.** WebNN, QNN, Core ML and the
  Axera path compile a *fixed* graph, prefer fp16/int8, and often reject
  fp32 accumulation or dynamic shapes. Adam in fp16 is not stable. The
  realistic split is therefore: **NPU runs the forward passes** — the frozen
  teacher, and the student forward, which is where the FLOPs are, static
  shape, fp16-friendly — while **the gradient and optimizer step runs on
  WebGPU or CPU** with fp32 master weights. That is ordinary mixed-precision
  training practice, and it means "QAT on an NPU" should be advertised as
  NPU-accelerated, not NPU-resident.
- **Determinism goes away.** GPU/NPU reductions reassociate;
  `tests/test_constant_fold_determinism.py`'s standard cannot hold for a
  trained result. CI stays on CPU with exact assertions; accelerator paths
  get tolerance-based tests and stay opt-in.

Rejected alternatives, recorded so they aren't re-proposed: **ORT on-device
training artifacts** (authoring them needs PyTorch, and the training EP
coverage is far narrower than the inference EPs — it would *lose* WebGPU and
NPU reach, not gain it); **a torch dependency** (violates this project's
no-framework rule and reaches neither WebGPU nor an NPU from the wheel);
**hand-written WGSL kernels** (re-implements what onnxruntime-web's WebGPU EP
already does, and would serve the browser only).

## Staging

Each stage is independently shippable and independently useful.

0. **Provider plumbing.** `apply_adaround`/`apply_adaquant`/`apply_brecq`
   already accept `providers=`, but only for the teacher's activation capture
   — the optimization itself is CPU numpy. Meanwhile
   `scripts/convertmodel/quantize_calibration.mjs` hard-codes
   `executionProviders: ["wasm"]` and ignores the page's own EP dropdown.
   Fixing that one line puts browser calibration on WebGPU/WebNN today, and
   is the smallest end-to-end proof that the accelerator path works.
1. **`qat_graph.py`: the step-graph builder/runner.** Emit forward + backward
   + Adam as one ONNX model; run it through `backend` with `providers=`.
   Prove it by porting one existing pass (`adaround.py` is the smallest) onto
   it and showing bit-comparable CPU results plus a working non-CPU provider.
   This is the load-bearing piece; everything else is application.
2. **`qat.py` / `apply_qat()`.** Free weights + LSQ/LSQ+ scales + block-wise
   teacher distillation over `load_huggingface_calibration_data`, a
   `QuantizationConfig` flag to reach it from `quantize()`, a doc, and tests
   in the established style (parser-built models, a *measured* improvement
   over RTN/AdaRound on a scenario engineered to make them differ, scoped as
   honestly as `brecq.py` scopes its own).
3. **Browser QAT panel.** A "fine-tune" panel in the converter page: data
   from `hf_datasets.mjs`, execution from `ort_executor.mjs` on WebGPU, a
   loss curve, and `quantize_metrics.mjs` for the before/after. Client-side
   QAT with no server is something none of the comparable tools offer.
4. **QAT interop (A).** Ingest externally-trained QDQ scales; emit the
   fake-quant model for external trainers.

## Costs and risks

- **Op coverage.** The backward graph's ops must all be implemented by the
  target EP. On WebGPU that is very likely; on WebNN/NPU it is exactly why
  the forward/backward split above exists. Needs measuring per EP before any
  claim is made in the README.
- **A second code path.** This is the objection `nncf-comparison-future-work.md`
  raises, and it is real — mitigated by the fact that stage 1's step-graph
  builder *replaces* numpy inner loops in existing passes rather than sitting
  beside them, and speeds them up on the way.
- **No new dependency.** Nothing above adds a runtime dependency: onnxruntime
  stays optional, onnxruntime-web is already on the converter page, numpy is
  already there.
- **Expectation management.** Label-free distillation QAT is not
  paper-QAT accuracy. It should be documented as what it is — the strongest
  thing reachable without labels — the same way `brecq.py`'s docstring refuses
  to claim its paper's reported numbers.
