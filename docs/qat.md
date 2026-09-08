# Delivering QAT in onnxsim: a design note

**Status: design note, with stages 0-2 and 4 implemented.**
`onnxsim/qat_graph.py`, `onnxsim/graph_grad.py`, `onnxsim/qat.py`
(`apply_qat`, `apply_qat_all_blocks`), `onnxsim/qat_interop.py` and the
converter page's calibration-provider picker have all landed; the browser
fine-tuning panel (stage 3) has not (see "Staging" below). This note answers "how *could* we deliver quantization-aware training as an onnxsim
feature, and can the training math run on WebGPU or an NPU?" and records the
shape of the work so the question doesn't have to be re-derived. `docs/nncf-comparison-future-work.md`
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

### A. QAT interop (graph-only, no training) -- built, see stage 4

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
  gradients exist in-tree, and `apply_qat` now reuses them directly rather
  than re-deriving them (`learn_scales` for weights,
  `learn_activation_scales` for activations);
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
  every Adam moment host-to-device per step. Parameters must stay resident.
  The Python half of this is done: `backend.Runner.bind_loop` uploads the
  constants once, allocates the state on the session's device and
  ping-pongs it between two buffers (binding one buffer as both a step's
  input and its output is unsafe -- onnxruntime may write an output before
  it has finished reading its inputs), so only scalars go up and only the
  loss comes down. On CPU that measures as a wash, which is the expected
  result and not a disappointment: with the CPU provider the "device" is
  host memory. The web half remains -- onnxruntime-web's GPU-buffer tensors
  with `preferredOutputLocation: "gpu-buffer"` (ORT-web >=1.17; the
  converter page is on 1.27).
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

0. **Provider plumbing -- done.** The converter page's browser-side
   calibration (`scripts/convertmodel/quantize_calibration.mjs`) used to
   hard-code `executionProviders: ["wasm"]`; it now takes the providers the
   Quantize panel's own **calibration execution provider** picker selects
   (`providersForEp` from `webnn.mjs`, so every accelerated choice keeps its
   WASM fallback), and asks for onnxruntime-web's "all" bundle when a WebNN
   device is chosen. WASM stays the default: an accelerated provider can
   compute in a different precision, which moves the observed min/max, so
   calibrating on one is opt-in. Unit-tested browser-free in
   `scripts/convertmodel/test/quantize_calibration.test.mjs`.
1. **`qat_graph.py`: the step-graph builder/runner -- done.** `GraphBuilder`
   assembles a hand-derived gradient as ONNX nodes, `adam_update` appends one
   Adam step, `make_step_graph` wraps the result into a pure
   `(constants, state, per-step scalars) -> (next state, loss)` function, and
   `run_step_graph` runs it N times through a single `onnxsim.backend.Runner`
   -- one session for the whole loop, on whichever `providers=` the caller
   names. `onnxsim.adaround` is the first caller ported onto it:
   `apply_adaround(..., step_providers=[...])` runs the optimization as a step
   graph instead of in host numpy. `tests/test_qat_graph.py` checks that it
   agrees with the numpy loop it replaces (>95% of elements pick the identical
   floor/ceil decision; the rest is float32-vs-float64 boundary rounding, and
   the reconstruction error lands within 10%) and that both step graphs stay
   inside `qat_graph.EP_FRIENDLY_OPS`, the operator set the accelerator
   backends actually implement.

   Since then: `adaquant` is ported onto it too, which is the harder case and
   the evidence the machinery generalizes -- it optimizes three parameter
   groups at once (rounding relaxation, activation scale, activation
   zero-point), so the step graph carries nine state tensors and three
   `adam_update` calls. It tracks its numpy loop tighter than adaround's port
   does: identical weight codes on five of six seeds, identical integer
   zero-point on all six. `autoround` followed, so all three rounding passes
   now take `step_providers=`. And the state now stays on the device between
   steps (`backend.Runner.bind_loop`), which is what makes the accelerator
   path worth taking rather than merely possible.
2. **`qat.py` / `apply_qat()` -- done**, and it needed one thing this note
   did not anticipate. Block-wise teacher distillation over an arbitrary
   topology means differentiating an arbitrary slice of the graph, which no
   pass here could do: every gradient in the repo is hand-derived for one
   fixed shape, which is exactly why `brecq.py` is capped at a linear chain.
   So `graph_grad.py` came first -- `build_backward` walks a forward slice in
   reverse and emits the gradient as ordinary ONNX nodes, 20 op rules, each
   checked against central finite differences, its emission pinned to
   `qat_graph.EP_FRIENDLY_OPS`. It is a rule table and a reverse walk, not an
   autograd framework: the ONNX graph is already the tape.

   On top of it, `apply_qat` trains the fp32 weights themselves (a
   straight-through estimator through the fake-quant, so an element can
   migrate several codes from where round-to-nearest put it -- the
   restriction every rounding pass here inherits), optionally the per-block
   scales LSQ-style, against the float block's own output.

   Measured honestly, and not a uniform win: on a two-Linear-plus-`Relu`
   block, which `brecq` returns byte-identical because it cannot see that
   topology at all, reconstruction error falls 16.0 -> 6.5. Against
   `apply_adaround` on a *single* layer, where the objective is identical
   and only the parametrization differs, freeing the weight wins on
   low-rank calibration activations (RTN 5.96 / AdaRound 3.07 / QAT 1.78)
   and **loses** on full-rank ones (28.5 / 14.6 / 22.8) -- a well-determined
   reconstruction problem has its optimum within one quantization step of
   round-to-nearest, so floor/ceil is all the freedom worth having there.
   Both directions are asserted in `tests/test_qat.py`.

   The walk over blocks landed since: `discover_qat_blocks` plans the whole
   model without the caller naming a tensor, and `apply_qat_all_blocks`
   trains the plan in order. Boundary discovery is a liveness argument
   rather than a pattern match -- cut wherever exactly one tensor is live
   across a gap, since a slice bounded that way is self-contained -- so
   residual connections *place* the boundaries instead of defeating them,
   and a span containing an op `graph_grad` cannot differentiate becomes a
   gap between blocks rather than a failure of the model. The walk is
   sequential by default: each block's target is the teacher's output but
   its input is recaptured from the student after the previous blocks were
   tuned, which differs from what `adaround`/`brecq` do (capture once, never
   re-run the student) and measured better on every seed tried (whole-model
   output error on a 4-layer chain: 400.3 round-to-nearest, 206.2
   capture-once, 168.1 sequential). Note that sequential's *per-block*
   losses read higher, because a dirtier input is a harder reconstruction
   problem -- the block-local loss is not the quantity that matters.

   Minibatching landed since (`batch_size=`), and the design question it
   forced is worth recording: `Runner.bind_loop` uploads constants once
   precisely to avoid per-step transfers, which is exactly what a batch that
   changes every step breaks. The resolution keeps the whole calibration set
   resident and feeds a per-step int64 index that a `Gather` turns into that
   step's rows, so per-step traffic stays at a few scalars -- at the cost of
   `Gather` in `EP_FRIENDLY_OPS` (justified there: WebNN specifies it, unlike
   `Round`) and of the set still having to fit in one static tensor. So it
   buys convergence rate and stochastic-gradient behaviour, not a
   larger-than-memory dataset: at equal epochs it roughly halves the error at
   a tight budget (11.41 -> 6.33) and washes out once full batch has
   converged.

   Activation quantization landed too, as `learn_activation_scales=True`.
   It could not be a feature added on top of the weight-only scheme, because
   a `quantize_weight_only_int4` model has no activation quantizer anywhere
   to train -- so the flag *selects* `quantize_static`'s QDQ scheme (uint8
   asymmetric activations, per-channel INT8 weights) and trains its
   quantizers jointly with its weights, refusing the wrong pairing loudly in
   both directions. The gradients are adaquant's, reused rather than
   re-derived: the quantize-dequantize chain is emitted stage by stage and
   handed to `graph_grad`, with the straight-through estimator made
   *structural* -- `round(r)` written as `r + (round(r) - r)`, the residual's
   nodes left out of the differentiated list -- so the ordinary rules
   reproduce adaquant's closed form, which a test checks against numpy. No
   `EP_FRIENDLY_OPS` addition was needed; the chain is `Exp`/`Div`/`Add`/
   `Sub`/`Mul` plus the existing clip and round helpers.

   Measured the same way as everything else here, and it lands the same
   shape: on a block whose range was calibrated from one outlier (~30x too
   wide) it takes the whole-model error from 4.08 (weights alone) to 2.54,
   38% better; on the same block calibrated on representative data it is a
   small regression at every learning rate tried. It fixes a quantizer whose
   range is wrong rather than improving one that is right.

   The `QuantizationConfig` flag landed as well: `quantize(config)` chains
   `apply_qat_all_blocks` the way it already chained AWQ/GPTQ/GPTAQ, for
   both schemes. Its position in that pipeline is forced at both ends rather
   than chosen -- after the correction passes, because QAT warm-starts from
   whatever codes the model carries; before `double_quant`, because that
   pass moves each scale out of an initializer and into a nested
   `DequantizeLinear`, which is the shape the layer finder requires, so the
   other order makes QAT a silent no-op. `learn_activation_scales` is
   derived from the scheme rather than exposed, so the API cannot express
   the pairing `apply_qat` refuses.

   The end-to-end pass is closed, and it needed no code: **the whole graph is
   already a legal block.** Naming the graph's own input and output builds one
   step graph over every node and takes the loss against the float model's
   final output -- the end-to-end objective exactly, and the only slice with
   no teacher-forcing approximation left, since its externals are the graph's
   own inputs. The premise of the original item, that a block is necessarily
   smaller than the model, was simply wrong.

   So the question was only whether to recommend it, and the measurements say
   no. At an equal step budget on an eight-stage stack it fits the calibration
   set 12-26% better -- it must, that being the quantity it minimizes where
   the walk only approximates it -- and generalizes 1-12% *worse* to held-out
   rows, on all eight seeds. That is BRECQ's own argument for the block being
   the right unit. It also costs ~6x the wall clock and 3.4x the peak RSS
   (105 MB against 31 MB on a one-million-parameter stack), linear in depth
   where the walk is flat, and it is all-or-nothing on operator coverage: one
   node without a gradient rule refuses the whole model where the walk makes
   it a gap. A depth sweep puts the crossover around four stages; 64x the data
   narrows the held-out gap without closing it.

   Real data via `load_huggingface_calibration_data` needed no code either:
   the loader returns `List[Dict[str, ndarray]]`, which is exactly the
   `Sequence[Tensors]` `apply_qat` already takes, and `_capture` concatenates
   across batches along axis 0 rather than using only the first. The thing
   worth pinning was that *many* batches are genuinely used -- a pass that
   quietly trained on `calibration_data[0]` would look identical from the
   outside, training and improving the model while silently using a fraction
   of the data the caller paid to download -- so a test asserts the step
   graph's teacher constant carries every row. That closes this stage.
3. **Browser QAT panel.** A "fine-tune" panel in the converter page: data
   from `hf_datasets.mjs`, execution from `ort_executor.mjs` on WebGPU, a
   loss curve, and `quantize_metrics.mjs` for the before/after. Client-side
   QAT with no server is something none of the comparable tools offer.
4. **QAT interop (A) -- done.** `onnxsim/qat_interop.py`.
   `quantize_static_keeping_qdq_scales` reads the parameters a QAT export
   already carries, canonicalizes back to float only the pairs onnxsim can
   re-emit, calibrates *only* the tensors with no usable annotation (a fully
   annotated export needs no calibration data at all), and writes the
   learned values back bit-exactly -- re-deriving a weight's integer codes
   from its learned scale, since swapping the scale alone would change what
   the weight dequantizes to. `export_fake_quant` closes the round trip,
   naming the initializers a trainer should make learnable (returned, and
   also stamped into `metadata_props`, since a round trip goes through a
   saved file -- but non-load-bearing, because ingest re-derives everything
   structurally). Refusal is a documented table and a refused pair is left
   in the graph exactly as authored.

   Why this matters more than it sounds: a learned scale is usually
   deliberately *tighter* than the observed min/max, clipping outliers to
   spend the codes where the values are. Re-deriving it, which is what
   `quantize_static` would silently do, is a regression rather than a wash.

## Using what has landed

```python
import onnxsim

# The optimization loop itself now runs as an ONNX step graph on these
# providers rather than in host numpy. `providers=` (unchanged) still selects
# where the float model's calibration activations are captured.
tuned = onnxsim.apply_adaround(
    float_model,
    quantized_model,
    calibration_data=batches,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    step_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

Omitting `step_providers` keeps the existing float64 numpy loop, which is what
CI runs: it is exact and reproducible, and a non-CPU provider is neither.
`apply_adaquant` and `apply_autoround` take the same argument. In the browser, the Quantize
panel's **calibration execution provider** picker does the equivalent for
calibration's own forward passes (WebGPU, or WebNN's GPU/NPU device types).

Block-wise QAT itself, over a block the caller names by its input and output
tensor:

```python
tuned = onnxsim.apply_qat(
    float_model,
    quantized_model,           # quantize_weight_only_int4's output
    block_input_name="hidden",
    block_output_name="block_out",
    calibration_data=batches,
    learn_scales=True,         # LSQ per-block scales alongside the weights
    step_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

To train the activation quantizers as well, pass a `quantize_static` model
instead -- the flag selects that scheme, it does not add to the weight-only
one:

```python
tuned = onnxsim.apply_qat(
    float_model,
    quantized_model,           # quantize_static's output, not int4's
    block_input_name="hidden",
    block_output_name="block_out",
    calibration_data=batches,
    learn_activation_scales=True,
    activation_learning_rate=1e-2,
)
```

There is no numpy alternative there -- the step graph *is* the
implementation -- so `step_providers=None` simply means CPU. Anything between
the two named tensors that `graph_grad` can differentiate is fair game; an op
it cannot is refused up front, before any calibration runs.

To put a *new* algorithm on a step graph: build its gradient with
`qat_graph.GraphBuilder`, close the loop with `qat_graph.adam_update` and
`make_step_graph`, and run it with `run_step_graph`.
`adaround._build_rounding_step_graph` is the worked example, and it is short
for the reason this whole approach works -- a hand-derived gradient is just
arithmetic.

## Costs and risks

- **Op coverage -- now measurable, and measured on one backend.**
  `scripts/convertmodel/test/step_graph_ep.test.mjs` runs real step graphs
  (committed fixtures emitted by the actual builders, and a generator that
  refuses to run unless their union is exactly `EP_FRIENDLY_OPS`) under one
  execution provider at a time, with no fallback in the provider list so a
  fallback cannot be mistaken for support. All 21 members verified on
  `wasm`. WebGPU and the WebNN device types are *not* verified: ORT-web
  refuses them in headless Node, so the test skips them loudly and says so,
  and `ORT_REQUIRE` turns that skip into a hard failure on a host that has
  them. The highest-risk unverified item is the minibatch index, the one
  non-float step-graph input -- WebNN is on record rejecting int64.

  Note also what the allowlist does *not* cover: `apply_qat` copies the
  block's own forward nodes into the step graph verbatim, so a block
  containing a `Relu` or a `Softmax` puts those in the graph too. The set
  constrains what onnxsim emits, not what the block contains.
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
