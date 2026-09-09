// The driving half of LoRA/QLoRA fine-tuning in the browser, for the
// converter page's "Fine-tune (LoRA)" panel (lora_ui.mjs).
//
// onnxsim/lora_entry.h splits the feature in two on purpose, and for the same
// reason qat_entry.h does (see qat_finetune.mjs's own top comment): building
// the step graph -- and, for LoRA, injecting the adapter in the first place --
// is graph surgery and lives in wasm (onnxsim_lora_inject /
// _build_step_graph / _write_back / _release_plan, see interface.cpp), while
// *running* the step graph is an ordinary inference loop and belongs wherever
// the inference runtime is -- here, on onnxruntime-web. lora_entry.h's own
// "intended browser flow" comment is the contract this file implements:
//
//   1. inject a trainable low-rank branch into the base model, getting back
//      the injected model and the LoraAdapter it produced;
//   2. build the step graph for one block (ordinarily the whole model --
//      InjectLora already applies globally in one pass, so there is no
//      per-layer boundary to discover the way QAT's blocks need one), getting
//      back its state ping-pong map, the activations to capture, the
//      per-step scalars and the initial state tensors (the adapter's
//      *current* A/B values, so a partially trained adapter resumes rather
//      than restarts);
//   3. capture those activations -- the block's own externals from the
//      *injected* model, and the reconstruction target from a *reference*
//      model, which is why this is two model sessions rather than QAT's one
//      (see captureLoraActivations below) -- bind them plus the initial
//      state, and run the step graph `n` times, feeding the per-step scalars
//      and carrying each state output back into its state input;
//   4. write the final state back into the injected model, and release the
//      plan.
//
// **Reused rather than reimplemented.** `adamBiasCorrections`,
// `minibatchIndices`, `bindCaptures`, `finalStateForWriteBack`,
// `runStepLoop`, `captureActivations`, `buildCalibrationRows` and
// `renderLossCurve` are imported straight from qat_finetune.mjs. None of the
// seven does anything QAT-specific: the bias-correction formula, the seeded
// minibatch stream, the input/output binding conventions, the state
// ping-pong loop, running a model over calibration rows to capture named
// tensors, and the loss-curve SVG are the same mechanics LoraStepPlan asks
// for -- LoraStepPlan's own top comment in lora_entry.h says as much
// ("Mirrors qat_entry.h's QatStepPlan, minus everything that exists there
// only for the two quantized schemes"). Re-implementing them here would be a
// second copy of the same minibatch wraparound arithmetic and xorshift
// permutation to keep in sync by hand; importing them keeps the one copy
// that qat_finetune.test.mjs already exercises authoritative for both
// panels. What genuinely differs -- LoRA's own scalar schedule (one learning
// rate, not three), the injection step, dual-model capture, and the plan's
// extra `parameters` field -- is written fresh below.
//
// The same two contract details qat_finetune.mjs's own top comment flags are
// load-bearing here too: a capture binds to its `input`, never its `source`;
// and `stepGraph` and every `initialState.data` (plus, for LoRA,
// `model` from the injection step) are views over wasm buffers the next call
// reuses, copied out immediately.

import { loadOrt } from "./inference_browser.mjs";
import { discoverLoraBlocks } from "./lora_blocks.mjs";
import { primaryGraphInput, readGraph } from "./qat_blocks.mjs";
import {
  adamBiasCorrections,
  bindCaptures,
  buildCalibrationRows,
  captureActivations,
  copyPlan,
  finalStateForWriteBack,
  minibatchIndices,
  renderLossCurve,
  runStepLoop,
} from "./qat_finetune.mjs";

// Re-exported so a page (or a test) that imports only lora_finetune.mjs has
// the whole surface it needs without also importing qat_finetune.mjs by
// name -- see this file's own top comment on why these seven are shared
// rather than duplicated.
export {
  adamBiasCorrections,
  bindCaptures,
  buildCalibrationRows,
  finalStateForWriteBack,
  minibatchIndices,
  renderLossCurve,
  runStepLoop,
};

// The value for every scalar LoraStepPlan.scalars asks for, at step `t` of
// `numSteps` -- qat_finetune.mjs's own stepScalars, narrowed to the one rate
// LoRA trains (LoraStepPlan's own field comment: "lora__lr", plus Adam's two
// bias-correction factors, and nothing else -- no scale rates, since LoRA has
// no quantizer scales to learn). The learning rate decays linearly to zero
// over the run by default, mirroring train_lora's own `lr_decay`. An unknown
// scalar name throws for the same reason it does there: a step graph input
// left unfed is an opaque onnxruntime error at run time.
export function loraStepScalars(scalarNames, t, options = {}) {
  const { numSteps = 1, learningRate = 1e-3, lrDecay = true } = options;
  const decay = lrDecay && numSteps > 0 ? 1 - t / numSteps : 1;
  const rates = {
    lora__lr: learningRate * decay,
    ...adamBiasCorrections(t),
  };
  const values = {};
  for (const name of scalarNames) {
    if (!(name in rates)) {
      throw new Error(`the step graph asks for an unknown per-step scalar '${name}'`);
    }
    values[name] = rates[name];
  }
  return values;
}

// Capture the activations a LoRA step graph binds, out of *two* models --
// unlike QAT, where every capture (the block's externals and the teacher
// alike) comes from the same float model, because there the float model
// produces both. LoRA's block externals are the *injected* model's own
// activations (train_lora's own `qat._capture(model, ...)`, where `model` is
// the model being trained) and its reconstruction target is a *reference*
// model's activation at the same block output (train_lora's
// `qat._capture(reference_model, [block_output_name], ...)`)  -- two
// different models in the general case, so this runs captureActivations
// once per model and merges the results, rather than once as QAT does.
//
// For the recommended default -- the whole graph as one block -- every
// non-teacher capture is a graph input of the injected model, so
// captureActivations' own optimization (a source that is a graph input is
// read straight from `rowFeeds`, never asked of a model) means the injected
// model is not actually run at all in that case; only the reference model is,
// for the one teacher capture. That only stops being true for a
// caller-narrowed sub-block, where a non-teacher capture can be a genuine
// intermediate activation.
export async function captureLoraActivations(
  ort,
  runtime,
  injectedBytes,
  referenceBytes,
  captures,
  rowFeeds,
  options = {},
) {
  const teacherCaptures = captures.filter((c) => c.teacher);
  const blockCaptures = captures.filter((c) => !c.teacher);
  const [block, teacher] = await Promise.all([
    blockCaptures.length
      ? captureActivations(ort, runtime, injectedBytes, blockCaptures, rowFeeds, options)
      : {},
    teacherCaptures.length
      ? captureActivations(ort, runtime, referenceBytes, teacherCaptures, rowFeeds, options)
      : {},
  ]);
  return { ...block, ...teacher };
}

// Copy a LoRA plan out of the wasm heap -- qat_finetune.mjs's copyPlan plus
// `parameters` (LoraStepPlan's own field, absent from QatStepPlan). Every
// wasm-view-sensitive field copyPlan already handles (stepGraph,
// initialState.data) needs no LoRA-specific treatment; `parameters` is a
// plain array of strings embind already materializes on the JS side (no view
// into wasm memory to alias), so spreading it is enough.
export function copyLoraPlan(built) {
  return {
    ...copyPlan(built),
    parameters: [...(built.parameters || [])],
  };
}

// Injects a trainable low-rank adapter into `baseBytes` -- onnxsim_lora_inject,
// copied out of the wasm heap immediately (its `model` field is a view over a
// buffer the next wasm call reuses, exactly like every other model-returning
// binding here).
//
// `injectOptions` is the InjectLoraOptions object the binding takes
// ({ rank, hasAlpha, alpha, targetOpTypes, restrictTargetNames, targetNames,
// seed }); see onnxsim_lora_inject's own doc comment in interface.cpp for
// what each field means and inject_lora's "meaningful only when hasAlpha is
// set" note on `alpha`.
//
// Returns { bytes, adapter }. `adapter` is the LoraAdapter InjectLora
// produced, as a plain array -- pass it to buildLoraStepPlan unchanged to
// train every injected branch, or filter it first to train only some.
export function injectLoraAdapter(runtime, baseBytes, injectOptions = {}) {
  const injected = runtime.onnxsim_lora_inject(baseBytes, injectOptions);
  if (!injected) {
    throw new Error(
      "could not inject a LoRA adapter (see the log/console for the refusal)",
    );
  }
  return {
    bytes: new Uint8Array(injected.model).slice(),
    adapter: injected.adapter.map((t) => ({ ...t })),
  };
}

// The injected model's own graph input/output names, for the default "train
// the whole model as one block" flow lora_entry.h's own top comment
// describes and onnxsim.train_lora's own API already treats as ordinary
// (naming the graph's own input/output trains the whole graph, exactly as it
// does for onnxsim.apply_qat's block argument). `blockInput` is picked with
// qat_blocks.mjs's own primaryGraphInput heuristic rather than the first
// declared graph input, since a real model's other inputs are commonly a
// mask or position-id tensor a liveness argument has no business treating as
// the block boundary; `blockOutput` is the first declared graph output,
// which covers every single-output model (the common case) and lets a
// multi-output one be named explicitly instead.
//
// Reused from qat_blocks.mjs for the same reason the training-loop pieces
// above are: reading a model's own graph topology is not QAT-specific logic.
export function wholeGraphBlock(injectedBytes) {
  const graph = readGraph(injectedBytes);
  const input = primaryGraphInput(graph);
  const output = graph.outputs[0] || null;
  if (!input || !output) {
    throw new Error(
      "could not determine the injected model's own graph input/output -- " +
        "name a block input/output tensor explicitly",
    );
  }
  return { input, output };
}

// Builds the step graph for one block -- onnxsim_lora_build_step_graph,
// copied out of the wasm heap immediately via copyLoraPlan.
//
// `adapter` is the LoraAdapter array injectLoraAdapter returned (or a
// caller-filtered subset of it). `options` is the LoraOptions object the
// binding takes ({ batchSize, batchSeed, shuffle }).
export function buildLoraStepPlan(
  runtime,
  injectedBytes,
  adapter,
  blockInput,
  blockOutput,
  numRows,
  options = {},
) {
  const built = runtime.onnxsim_lora_build_step_graph(
    injectedBytes,
    adapter,
    blockInput,
    blockOutput,
    numRows,
    options,
  );
  if (!built) {
    // BuildLoraStepGraph refuses loudly -- an unclosed block, a node with no
    // gradient rule (named), an adapter with no targets -- and the binding
    // turns that into null with the reason on the console, which the page
    // mirrors into its log.
    throw new Error(
      `could not build a LoRA step graph for ${blockInput} → ${blockOutput} ` +
        "(see the log/console for the refusal)",
    );
  }
  return copyLoraPlan(built);
}

// Train one already-injected model's block: build its step graph, run the
// loop, write the trained state back, release the plan. The piece
// trainLoraAdapter's single-block call and trainLoraAdapterAllBlocks' loop
// (over lora_blocks.mjs's own proposal) both reduce to -- injection happens
// once, before either, since InjectLora is a whole-model pass with nothing
// block-specific about it.
//
// `injectedBytes`/`adapter` are what injectLoraAdapter (or a previous call to
// this function, chained -- see trainLoraAdapterAllBlocks) returned.
// `adapter` may be the whole array or a caller-narrowed subset
// (lora_blocks.mjs's own `targetOutputs` filtered against it): only the
// targets passed are trained, exactly as onnxsim_lora_build_step_graph's own
// contract says.
//
// The one training mode implemented is reference-model distillation
// (`train_lora(..., reference_model=...)`'s browser equivalent): the block's
// reconstruction target is captured from `referenceBytes` at `blockOutput`,
// on the same calibration rows fed to the injected model. Caller-supplied
// `target_data` (train_lora's other, mutually exclusive mode -- the loss
// target handed in directly rather than read off a reference model, i.e.
// genuine supervised fine-tuning against labels) is not implemented here;
// see trainLoraAdapter's own doc comment for why.
//
// Returns { bytes, losses }. `bytes` is the injected model with this block's
// adapter tensors written back -- feed it into the next block's own call as
// its `injectedBytes` to keep a multi-block run's writes visible to later
// blocks, the same "recapture from the student after previous blocks were
// tuned" sequencing docs/qat.md measures as better than capturing once. The
// plan is always released, including on a failure: nothing expires on its
// own, and a page training block after block would otherwise keep every step
// graph it ever built.
export async function trainLoraBlock(runtime, {
  injectedBytes,
  adapter,
  referenceBytes,
  blockInput,
  blockOutput,
  rowFeeds,
  options = {},
  numSteps = 200,
  rates = {},
  providers = ["wasm"],
  needWebnn = false,
  ortLoader = loadOrt,
  log = () => {},
  onStep = () => {},
}) {
  const numRows = rowFeeds.length;
  const plan = buildLoraStepPlan(
    runtime,
    injectedBytes,
    adapter,
    blockInput,
    blockOutput,
    numRows,
    options,
  );

  try {
    const ort = await ortLoader(needWebnn ? "all" : "default");
    const captured = await captureLoraActivations(
      ort,
      runtime,
      injectedBytes,
      referenceBytes,
      plan.captures,
      rowFeeds,
      { providers, log },
    );
    const captureFeeds = bindCaptures(plan.captures, captured);
    for (const [name, value] of Object.entries(captureFeeds)) {
      captureFeeds[name] = new ort.Tensor("float32", value.data, value.dims);
    }

    const session = await ort.InferenceSession.create(plan.stepGraph, {
      executionProviders: providers,
    });
    const initialState = {};
    for (const tensor of plan.initialState) {
      // A fresh copy, so the byte offset is 0 and the Float32Array view is
      // aligned. Every state tensor a step graph carries is float32.
      const bytes = tensor.data.slice();
      initialState[tensor.name] = new ort.Tensor(
        "float32",
        new Float32Array(bytes.buffer),
        tensor.dims,
      );
    }

    const { state, losses } = await runStepLoop({
      state: plan.state,
      scalars: plan.scalars,
      loss: plan.loss,
      initialState,
      captures: captureFeeds,
      rowIndex: plan.rowIndexInput
        ? {
            input: plan.rowIndexInput,
            size: plan.rowIndexSize,
            numRows: plan.numRows,
            seed: options.batchSeed || 0,
            shuffle: options.shuffle !== false,
          }
        : null,
      numSteps,
      rates,
      scalarValues: (t) => loraStepScalars(plan.scalars, t, { ...rates, numSteps }),
      runStep: (feeds) => session.run(feeds),
      makeScalar: (value) => new ort.Tensor("float32", Float32Array.of(value), []),
      makeRowIndex: (rows) => new ort.Tensor("int64", rows, [rows.length]),
      onStep,
    });

    const written = runtime.onnxsim_lora_write_back(
      injectedBytes,
      plan.planHandle,
      finalStateForWriteBack(plan.state, state),
    );
    if (!written) {
      throw new Error("writing the trained adapter back failed (see the log/console)");
    }
    if (typeof session.release === "function") await session.release();
    return {
      bytes: new Uint8Array(written).slice(),
      losses,
      plan,
    };
  } finally {
    runtime.onnxsim_lora_release_plan(plan.planHandle);
  }
}

// Train one LoRA adapter end to end, over one block: inject, then
// trainLoraBlock once. This is lora_entry.h's whole "intended browser flow"
// as one call, the way qat_finetune.mjs's fineTuneBlock is
// BuildQatStepGraph's -- see trainLoraBlock's own doc comment for the
// training-mode/argument notes that still apply here unchanged, and
// trainLoraAdapterAllBlocks below for training more than one block in a run.
//
// `baseBytes` is the model to inject into; `referenceBytes` defaults to
// `baseBytes` itself (the ordinary case: the adapter should reproduce the
// *same* model's own behaviour on rows the base model was not exactly
// evaluated on post-injection -- injection alone is a numeric no-op, since
// `B` starts at zero, so at step 0 the injected model already agrees with
// the reference exactly and the loss starts there). A caller doing real
// distillation against a different (larger, or differently fine-tuned) model
// passes that model's bytes instead.
//
// `blockInput`/`blockOutput` default to the injected model's own graph
// input/output (wholeGraphBlock) -- train the whole model as one block,
// InjectLora's own scope.
//
// Returns { bytes, losses, adapter, plan } -- `adapter` is every target
// InjectLora produced (trainLoraBlock's own return has no such field, since
// a caller driving several blocks already has the adapter array from
// injectLoraAdapter and would only be handed the same thing back unchanged).
export async function trainLoraAdapter(runtime, {
  baseBytes,
  referenceBytes = null,
  targetData = null,
  injectOptions = {},
  blockInput = null,
  blockOutput = null,
  rowFeeds,
  options = {},
  numSteps = 200,
  rates = {},
  providers = ["wasm"],
  needWebnn = false,
  ortLoader = loadOrt,
  log = () => {},
  onStep = () => {},
}) {
  if (targetData != null) {
    throw new Error(
      "target_data training is not implemented in the browser panel yet -- " +
        "pass referenceBytes for reference-model distillation instead " +
        "(see trainLoraBlock's own doc comment)",
    );
  }
  const injected = injectLoraAdapter(runtime, baseBytes, injectOptions);
  const injectedBytes = injected.bytes;

  let blockIn = blockInput;
  let blockOut = blockOutput;
  if (!blockIn || !blockOut) {
    const whole = wholeGraphBlock(injectedBytes);
    blockIn = blockIn || whole.input;
    blockOut = blockOut || whole.output;
  }

  const result = await trainLoraBlock(runtime, {
    injectedBytes,
    adapter: injected.adapter,
    referenceBytes: referenceBytes || baseBytes,
    blockInput: blockIn,
    blockOutput: blockOut,
    rowFeeds,
    options,
    numSteps,
    rates,
    providers,
    needWebnn,
    ortLoader,
    log,
    onStep,
  });
  return { ...result, adapter: injected.adapter };
}

// Train every block lora_blocks.mjs's discoverLoraBlocks proposes, in graph
// order -- trainLoraAdapter lifted from one block to a model whose injected
// adapters are spread too widely (or through enough undifferentiable ops)
// for the whole-model default to reach in one call.
//
// Each block's write-back is visible to the next: `injectedBytes` is
// threaded through the loop rather than re-read from the original
// injection, the sequential walk apply_qat_all_blocks/docs/qat.md both
// measure as better than capturing every block once from the same
// unmodified model.
//
// A block onnxsim_lora_build_step_graph refuses (an op with no gradient
// rule that discovery's own DIFFERENTIABLE_OPS pre-filter missed, say) is
// skipped rather than failing the run -- discoverBlocks' own gap-not-failure
// principle, one level up: a proposal that turns out untrainable costs that
// block, not the page's whole result. `onBlockStart`/`onBlockDone` let the
// caller report progress per block, in addition to `onStep`'s per-iteration
// calls within one.
//
// Returns { bytes, losses, adapter, results }. `losses` concatenates every
// trained block's own loss trace in order; `results` is one entry per
// proposed block ({ block, trained, skippedReason, losses }), mirroring
// apply_qat_all_blocks' own QATBlockResult so a caller can report exactly
// what happened to each proposal.
export async function trainLoraAdapterAllBlocks(runtime, {
  baseBytes,
  referenceBytes = null,
  targetData = null,
  injectOptions = {},
  maxTargetsPerBlock = 2,
  rowFeeds,
  options = {},
  numSteps = 200,
  rates = {},
  providers = ["wasm"],
  needWebnn = false,
  ortLoader = loadOrt,
  log = () => {},
  onStep = () => {},
  onBlockStart = () => {},
  onBlockDone = () => {},
}) {
  if (targetData != null) {
    throw new Error(
      "target_data training is not implemented in the browser panel yet -- " +
        "pass referenceBytes for reference-model distillation instead " +
        "(see trainLoraBlock's own doc comment)",
    );
  }
  const injected = injectLoraAdapter(runtime, baseBytes, injectOptions);
  let injectedBytes = injected.bytes;
  const teacherBytes = referenceBytes || baseBytes;

  const proposals = discoverLoraBlocks(injectedBytes, injected.adapter, { maxTargetsPerBlock });
  const allLosses = [];
  const results = [];
  for (let i = 0; i < proposals.length; i++) {
    const proposal = proposals[i];
    onBlockStart(i, proposals.length, proposal);
    const blockAdapter = injected.adapter.filter((t) => proposal.targetOutputs.includes(t.nodeOutput));
    try {
      const result = await trainLoraBlock(runtime, {
        injectedBytes,
        adapter: blockAdapter,
        referenceBytes: teacherBytes,
        blockInput: proposal.input,
        blockOutput: proposal.output,
        rowFeeds,
        options,
        numSteps,
        rates,
        providers,
        needWebnn,
        ortLoader,
        log,
        onStep: (t, loss) => onStep(i, proposals.length, t, loss),
      });
      injectedBytes = result.bytes;
      allLosses.push(...result.losses);
      results.push({ block: proposal, trained: true, skippedReason: null, losses: result.losses });
    } catch (blockErr) {
      results.push({
        block: proposal,
        trained: false,
        skippedReason: blockErr.message,
        losses: [],
      });
    }
    onBlockDone(i, proposals.length, results[results.length - 1]);
  }

  return { bytes: injectedBytes, losses: allLosses, adapter: injected.adapter, results };
}
