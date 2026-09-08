# Delivering QAT in onnxsim: a design note

**Status: design note, all five stages implemented.**
`onnxsim/qat_graph.py`, `onnxsim/graph_grad.py`, `onnxsim/qat.py`
(`apply_qat`, `apply_qat_all_blocks`, and the `apply_block_finetune` pair the
same machinery becomes with the quantizer removed), `onnxsim/qat_interop.py`,
and the converter page's calibration-provider picker and fine-tuning panel
have all landed. This note answers "how *could* we deliver quantization-aware
training as an onnxsim feature, and can the training math run on WebGPU or an
NPU?" and records the shape of the work so the question doesn't have to be
re-derived. `docs/nncf-comparison-future-work.md`
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
   reverse and emits the gradient as ordinary ONNX nodes, 21 op rules, each
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
3. **Browser QAT panel -- done**, as `scripts/convertmodel/qat_ui.mjs` over
   `qat_finetune.mjs` and `qat_blocks.mjs`, driving the three wasm bindings
   `interface.cpp` exposes. Two things in the sketch above did not survive
   contact with the loop, and both are worth recording.

   **`ort_executor.mjs` is the wrong runner for this.** Its `makeOrtRunner`
   creates a session per call and passes tensors positionally as one packed
   blob -- exactly right for the constant-folding trampoline it was written
   for, and exactly wrong for a training loop, which needs one session held
   across every step and binds inputs by name. The panel opens an
   onnxruntime-web session directly, as `quantize_calibration.mjs` does, on
   whatever `providersForEp` returns.

   **The minibatch row stream cannot be bit-compatible with Python.**
   `qat_graph.minibatch_indices` draws each epoch's permutation from
   `np.random.default_rng([seed, epoch])` -- numpy's PCG64, which a browser
   cannot regenerate. The port keeps every property the docstring argues for
   (a permutation per epoch from `(seed, epoch)` alone, a straddling batch
   completed from the front of the next epoch, the whole stream a pure
   function of `t`) but not the sequence, so the same seed picks different,
   equally valid batches. This is the one deliberate gap in Python/browser
   parity, and it is confined to *which rows* a step sees: the emitted step
   graph stays byte-identical, which is what `qat_parity_fixtures.txt` pins.

   The panel captures every block's activations from the float model, which
   is the capture-once walk rather than `apply_qat_all_blocks`' sequential
   one -- the weaker of the two by this note's own measurements (206.2
   against 168.1 above). `QatCapture` carries no "recapture this from the
   student" flag, so doing better needs a change to the plan rather than to
   the panel; it is flagged in `qat_ui.mjs` rather than quietly left out.
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

## What is left when the quantizer is taken out

Nothing in stage 2 is specific to quantization except one node. A block, the
teacher's activation at its output, a mean squared reconstruction error, a
backward pass emitted by `graph_grad`, an Adam step graph, a liveness-cut
block plan and a sequential walk over it -- quantization enters at exactly one
place, the fake-quantizer sitting between the master weight and the block's
own node. Remove that and the same loop trains the block's own float weights
against the teacher: ordinary block-wise, label-free distillation, with no
labels to supply and no loss to choose. That is `apply_block_finetune` and
`apply_block_finetune_all_blocks` -- `apply_qat(..., fake_quant=False)` under
a name that says what it does.

The one substantive difference beside the missing quantizer is where the
trainable layers come from. QAT matches the student's quantized layers against
the teacher's float ones and seeds each master weight from the *teacher*,
because the student's weight is a lossy encoding of it. Fine-tuning scans the
*student* for plain MatMul/Gemms whose weight is a 2-D fp32 initializer and
starts from the student's own values -- they are the starting point precisely
because they are not the teacher's. The block's untrained constants (a
LayerNorm's scale and bias, a Gemm's `C`) come from the student for the same
reason: substituting the teacher's would train the block to compensate for a
substitution the deployed model does not make.

**The two models have to differ, or there is nothing to learn.** The loss is
the student block's output against the reference's, so a student that *is* the
reference starts at zero loss and stays there (measured: exactly `0.0` at the
first step and the last). This is for a model something else already changed
and the change cost accuracy -- a pruned model, a model whose weights were
quantized and dequantized back to fp32 or rewritten by one of the rounding
passes, a model already tuned once and being tuned further against the
original.

It is **not** a way to train on new data or a new task. The objective is
"reproduce what the reference produced", which by construction cannot exceed
the reference: deliverable C above is unchanged, and this is not a back door
into it. The two scale flags are refused rather than ignored for the same
reason -- `learn_scales` and `learn_activation_scales` each name a parameter
of a quantizer, and this is the mode with no quantizer in it, so asking for
them raises. Ignoring them would be the worse failure: a caller who asked to
learn scales and got a model whose scales are untouched has no way to tell
that from a run where learning them did not help.

Operator coverage is the same constraint it is under QAT, and it is still the
binding one on a real model: `graph_grad.SUPPORTED_OPS` is 21 rules
(`LayerNormalization` among them since this branch), so a normalization in the
middle of a block is now more nodes in the slice rather than a boundary
between blocks -- but the whole default ONNX domain is 202 operators.

### Against `apply_pruning_finetune`, which came first

`onnxsim/finetune.py`'s `apply_pruning_finetune` solves the same kind of
problem for pruning and should usually be tried first. It fits each layer
*individually* and in *closed form*: one ridge regression, one linear solve
per layer, no iteration, no learning rate -- and an exactness argument this
has no equivalent of, since pruning never changes the value of a channel that
survives, so a pruned layer's own input activation *is* the original model's,
restricted to the surviving channels. Where it applies it is strictly better.

What it cannot do is what a block buys. A layer pruned on both its input and
its output channels at once falls outside its channel-correspondence
reconstruction and it declines to touch it. And a per-layer least-squares fit
cannot let two layers with a nonlinearity between them trade error off against
each other, because that objective is not a linear least-squares problem at
all. `apply_block_finetune` is the general, slower, weaker-guarantee
alternative for those cases -- and, unlike a numpy solve, it is a step graph,
so `step_providers=` reaches the same accelerators the rest of this note is
about. It is a fallback, not a replacement.

### Preserving a pruned model's zeros

By default the optimizer updates every element of a trained weight, so on a
magnitude-pruned (unstructured) model it *fills the zeros back in* -- measured
on a two-layer block at 50% sparsity, 128 zeros per weight before and **0
after**, while the block's loss fell 0.140 -> 5.3e-06. Every signal a caller
looks at says that run went well, which is what makes it dangerous rather than
merely wrong.

`preserve_sparsity=True` zeroes the weight gradient wherever the master weight
started at zero. One `Mul` per trained layer is enough to hold those elements
at zero *exactly*, not approximately: with a gradient of 0 every step, Adam's
`m` and `v` stay 0, its step is `lr * 0 / (sqrt(0) + eps)` -- exactly 0 -- and
the parameter never moves. No clean-up pass, and no drift in between. It is
opt-in because "this element is zero" and "this element was pruned away" are
the same bit pattern, and only the caller knows which they meant. Structured
pruning needs nothing here: the channel is gone from the tensor rather than
zeroed inside it.

It costs what it must. Half the free parameters are pinned, so the block
cannot fit the reference as closely -- on the model above, held-out error
0.241 before, **0.191** with the zeros kept, against 0.000 for the
unconstrained run that returns a dense model. The unconstrained number is
lower and is not the model that was asked for.

**Under `apply_qat` the same flag applies, and there the failure it prevents
is intermittent rather than total** -- which is worse. A weight must drift half
a quantization step before `round(w / s)` reports anything but 0, so at the
default learning rate of 1e-4 a short run moves no code off zero and the
problem is invisible. It is invisible, not absent: on the same weight at 300
iterations, the count of pruned zeros returning as nonzero codes was 6 at lr
1e-3, 38 at 1e-2 and 177 at 5e-2. A sparsity that erodes with the learning
rate and the budget is a worse way to lose it than losing it outright. With
the flag on it is 0 at every rate.

### Measured, and the measurement is the point

The honest headline is not the loss curve. On a `MatMul`/`Relu`/`MatMul` block
(16x16 fp32 weights, N(0, 0.15) noise on the second weight, 300 iterations at
lr 2e-2, full batch), the *training* loss falls by roughly 1700-5200x -- e.g.
0.2597 -> 0.000154 -- while the end-to-end output error, measured against the
reference on 8 held-out input batches the tuning never saw, only falls to
0.64-0.78 of the student's. The block-local objective is fitted almost
exactly and generalizes far less well, which is the same warning
`apply_qat_all_blocks` already carries about its per-block losses: the loss
`losses=` records is not the quantity that matters.

What dominates is how much calibration data there is. Same model and budget,
varying only the number of calibration rows; mean held-out error ratio against
the untuned student over 4 seeds (lower is better):

| calibration rows | held-out error ratio |
| --- | --- |
| 16 | 0.70 |
| 64 | 0.06 |
| 256 | 0.004 |
| 1024 | 0.002 |

16 rows is a fit with as many rows as each weight has input channels, so it
interpolates the calibration set and leaves most of the error standing
anywhere else; by 64 rows it recovers ~94% of the error and by 256 rows ~99.6%.
The same cliff shows up in `apply_block_finetune_all_blocks` over a 5-layer MLP
(3 blocks discovered, all trained, 300 iterations each): a mean held-out ratio
of 0.46 at 16 rows against 0.015 at 256, over 3 seeds. The rows-versus-channels
reading is intuition rather than a bound -- there is a nonlinearity in the
middle of the block and two weights are trained jointly -- but the direction
was the same everywhere it was measured.

And `num_iterations` interacts with it, in opposite directions either side of
the cliff. Held-out error ratio, 4 seeds, same model and learning rate:

| rows | 25 iters | 50 | 100 | 300 |
| --- | --- | --- | --- | --- |
| 16 | 0.60-0.70 | 0.59-0.70 | 0.59-0.72 | 0.62-0.76 |
| 256 | 0.30-0.34 | 0.14-0.18 | 0.03-0.07 | 0.000-0.002 |

With enough rows, longer is monotonically better -- 300 iterations won on
every seed. With too few, it is monotonically *worse* past a knee somewhere
around 25-100: the training loss keeps falling three or four orders of
magnitude while the held-out error climbs, which is textbook overfitting and
is the same thing the table above measures from the other side.

So `num_iterations` is the regularizer here, and the default of 1000 is aimed
at the well-fed case. But note the sizes before reaching for it: stopping
early at 16 rows buys perhaps 3-9% relative, where going to 256 rows buys
99.8%. Early stopping is worth doing and is not a substitute for data, which
is why no regularization parameter was added -- one would look like a fix for
a problem it barely moves.

Two things follow, and they belong in any use of this. First, `num_samples`
defaults to 8 *batches* of random data, so the row count is 8 times the
model's own batch dimension: for a model with a small batch dimension that is
on the wrong side of the cliff above, and real data
(`load_huggingface_calibration_data`) is the fix rather than more iterations.
Second, a falling loss is not evidence that the model got better. Measure on
held-out inputs.

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

And the same loop with no quantizer in it, walking a whole model one block at
a time -- the reference here is not a float *teacher* of a quantized student
but simply the model as it was before something changed it:

```python
tuned, results = onnxsim.apply_block_finetune_all_blocks(
    original_model,            # the reference, before pruning/requantizing
    changed_model,             # the one being tuned back towards it
    calibration_data=batches,  # rows matter more than iterations; see above
    num_iterations=300,
    learning_rate=2e-2,
)
for r in results:
    print(r.block.output_name, r.trained, r.skipped_reason, r.final_loss)
```

`apply_block_finetune` is the same thing over one caller-named block, and
refuses loudly where the walk skips and records a reason -- the split
`apply_qat`/`apply_qat_all_blocks` already draw, for the same reason.

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
