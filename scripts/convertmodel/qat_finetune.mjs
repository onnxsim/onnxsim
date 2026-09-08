// The driving half of block-wise QAT in the browser, for the converter page's
// "Fine-tune (QAT)" panel (qat_ui.mjs).
//
// onnxsim/qat_entry.h splits the feature in two on purpose: building the step
// graph is graph surgery and lives in wasm
// (onnxsim_qat_build_step_graph / _write_back / _release_plan, see
// interface.cpp), while *running* it is an ordinary inference loop and belongs
// wherever the inference runtime is -- here, on onnxruntime-web. That header's
// "intended browser flow" comment is the contract this file implements:
//
//   1. build the step graph for one block, getting back its state ping-pong
//      map, the activations to capture, the per-step scalars and the initial
//      state tensors;
//   2. capture those activations from the *float* model, bind them plus the
//      initial state, and run the step graph `n` times, feeding the per-step
//      scalars and carrying each state output back into its state input;
//   3. write the final state back into the quantized model, and release the
//      plan.
//
// Everything that decides *what* is fed -- the scalar schedule, the minibatch
// row stream, which capture binds to which input, the state ping-pong -- is a
// pure function here, taking injected fakes, so the loop's mechanics are
// testable under Node with no browser and no wasm (test/qat_finetune.test.mjs),
// exactly the way quantize_calibration.mjs is. Only captureActivations() and
// fineTuneBlock() touch onnxruntime-web, and they take their ORT entry points
// injected too.
//
// Two contract details from interface.cpp are load-bearing and easy to get
// wrong, so they are enforced rather than remembered:
//   - a capture binds to its `input`, never its `source`. The two differ
//     whenever the step graph renames a tensor (under a minibatch the whole
//     set is bound to a resident `qat__all_<name>` table and the block reads
//     gathered rows out of it), and binding by `source` would train on
//     uninitialized memory rather than fail.
//   - `stepGraph` and every `initialState.data` are views over wasm buffers
//     the next call reuses. They are copied out here, immediately, before any
//     other binding runs.

import { loadOrt, makeDummyInputs } from "./inference_browser.mjs";
import { createSampleContext } from "./sample_inputs.mjs";

// Adam's decay rates, as onnxsim/qat_graph.py and qat_graph_builder.cpp fix
// them. The optimizer itself lives in the step graph; only the two
// bias-correction factors are host-side.
export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;

// Adam's two bias-correction factors at (0-based) step `t`, named the way the
// step graph's inputs are: qat_graph.adam_bias_corrections and the C++
// AdamBiasCorrections, which this must agree with element for element.
//
// They are fed as scalars rather than derived inside the graph because that is
// two host-side floats per step against the state a Pow over a step counter
// would need -- see qat_graph_builder.h's note on AdamUpdate.
export function adamBiasCorrections(t) {
  return {
    m_correction: 1 / (1 - Math.pow(ADAM_BETA1, t + 1)),
    v_correction: 1 / (1 - Math.pow(ADAM_BETA2, t + 1)),
  };
}

// The value for every scalar the plan asks for, at step `t` of `numSteps`.
//
// The learning rates decay linearly to zero over the run (apply_qat's
// `lr_decay`, on by default there and here); the bias corrections do not
// depend on the schedule at all. An unknown scalar name throws rather than
// being skipped: a step graph input left unfed is an opaque onnxruntime error
// at run time, and a new scalar appearing in the plan is exactly the kind of
// contract change that should stop the loop and name itself.
export function stepScalars(scalarNames, t, options = {}) {
  const {
    numSteps = 1,
    learningRate = 1e-4,
    scaleLearningRate = 1e-5,
    activationLearningRate = 1e-2,
    lrDecay = true,
  } = options;
  const decay = lrDecay && numSteps > 0 ? 1 - t / numSteps : 1;
  const rates = {
    qat__lr: learningRate * decay,
    qat__lr_scale: scaleLearningRate * decay,
    qat__lr_act: activationLearningRate * decay,
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

// A seeded permutation of [0, n), drawn from (seed, epoch) alone.
//
// The xorshift32 is input_fill.mjs's, for the same reason it is used there:
// the page has no seeded RNG of its own and this one is four lines. It
// deliberately does *not* reproduce qat_graph.minibatch_indices' permutation,
// which comes from numpy's PCG64 via np.random.default_rng([seed, epoch]) --
// nothing in a browser can. What matters is the property, not the sequence:
// a permutation drawn from (seed, epoch) rather than advanced from a running
// generator is what keeps the row stream a pure function of the step index, so
// a run is reproducible and a loop stopped at step N and resumed there sees
// the same batches. The one thing this changes against Python is that the same
// seed picks different (equally valid) batches in the two, which is why the
// panel reports its seed.
function permutation(n, seed, epoch) {
  // imul, not `*`: the products exceed 2^53 for a large seed and would lose
  // low bits -- the only bits a hash of (seed, epoch) has to offer.
  let s = (Math.imul(seed | 0, 0x9e3779b1) ^ Math.imul(epoch | 0, 0x85ebca6b) ^ 0x2545f491) | 0;
  if (s === 0) s = 0x2545f491;
  const next = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return s >>> 0;
  };
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = next() % (i + 1);
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  return order;
}

// The row indices step `t` should train on, as a pure function of `t` --
// qat_graph.minibatch_indices, ported. Returns (t) => BigInt64Array, the dtype
// the step graph's row-index input declares (it is the only non-float input a
// step graph ever has).
//
// The stream is an endless concatenation of per-epoch permutations chopped
// into fixed-size chunks: step `t` gets stream positions [t*B, (t+1)*B), and
// position `p` is row perm[p / N][p % N]. A batch that would run off the end
// of an epoch is *completed from the front of the next* rather than being
// short, because a step graph's shapes are static -- that is the property that
// lets it compile for an NPU -- so a ragged final batch would need a second
// graph. Dropping the tail would silently discard rows, and systematically the
// same ones without shuffling; padding by repeat would quietly reweight one.
// Wrapping keeps every row's visit count equal and every batch full.
export function minibatchIndices(numRows, batchSize, { seed = 0, shuffle = true } = {}) {
  if (numRows < 1) throw new Error(`numRows must be at least 1, got ${numRows}`);
  if (batchSize < 1) throw new Error(`batchSize must be at least 1, got ${batchSize}`);
  // A batch straddling an epoch boundary asks for two permutations, so the
  // last two are kept rather than redrawn twice per step.
  const cache = new Map();
  const epochOrder = (epoch) => {
    let order = cache.get(epoch);
    if (!order) {
      order = shuffle
        ? permutation(numRows, seed, epoch)
        : Array.from({ length: numRows }, (_, i) => i);
      for (const stale of [...cache.keys()]) if (stale < epoch - 1) cache.delete(stale);
      cache.set(epoch, order);
    }
    return order;
  };
  return (t) => {
    const rows = new BigInt64Array(batchSize);
    let filled = 0;
    while (filled < batchSize) {
      const position = t * batchSize + filled;
      const epoch = Math.floor(position / numRows);
      const offset = position % numRows;
      const take = Math.min(numRows - offset, batchSize - filled);
      const order = epochOrder(epoch);
      for (let i = 0; i < take; i++) rows[filled + i] = BigInt(order[offset + i]);
      filled += take;
    }
    return rows;
  };
}

// Bind the captured activations to the step graph, keyed by each capture's
// `input` and looked up by its `source`.
//
// This is the one place the two names are allowed to differ, and the reason
// this is a function rather than a loop at the call site: `captured` is keyed
// by the tensor read out of the float graph, the feeds are keyed by the step
// graph's own input, and under a minibatch those are never the same name.
export function bindCaptures(captures, captured) {
  const feeds = {};
  for (const capture of captures) {
    const value = captured[capture.source];
    if (value === undefined) {
      throw new Error(`no captured activation for '${capture.source}'`);
    }
    feeds[capture.input] = value;
  }
  return feeds;
}

// The loop's final state, as onnxsim_qat_write_back takes it: keyed by state
// *input* name, each value a { dims, data } pair.
//
// The re-key the binding's doc comment describes -- from each state pair's
// `output` back onto its `input` -- has already happened, once, in
// runStepLoop's ping-pong, which is where it has to happen anyway for the loop
// to carry state at all. So this only checks that every state input the plan
// declares actually has a value (a plan whose state the loop never ran leaves
// the model silently untrained) and narrows each tensor to the two fields the
// binding reads, dropping the loss along with everything else.
export function finalStateForWriteBack(state, finalState) {
  const final = {};
  for (const { input } of state) {
    const tensor = finalState[input];
    if (tensor === undefined) {
      throw new Error(`the loop's final state has no value for '${input}'`);
    }
    // { dims, data } -- which is exactly an onnxruntime-web output tensor's
    // own shape, so this is a narrowing rather than a conversion.
    final[input] = { dims: [...tensor.dims], data: tensor.data };
  }
  return final;
}

// Run the step graph `numSteps` times, carrying the state.
//
// `runStep(feeds)` runs one step and returns its outputs as
// { name: { dims, data } } -- an onnxruntime-web session.run in the browser, a
// fake in the tests. The captures and the state tensors are objects of the
// runtime's own kind and are passed through untouched, so a state output can
// be fed straight back as the next step's input with no copy: onnxruntime
// allocates a fresh output tensor per run, so the two buffers ping-pong on
// their own (the aliasing hazard docs/qat.md records is about *pre-allocated*
// output buffers under IOBinding, which this path does not use).
//
// Returns { state, losses }: the final state keyed by state input, and the
// per-step loss, which is what the panel plots.
export async function runStepLoop({
  state,
  scalars = [],
  loss = "",
  initialState,
  captures = {},
  rowIndex = null,
  numSteps,
  rates = {},
  runStep,
  // The schedule is policy, the loop is mechanism: both of these are the
  // defaults rather than the definitions, so a caller can drive a step graph
  // whose scalars are named differently or replay a Python run's own batches
  // instead of drawing new ones. The panel passes neither.
  scalarValues = null,
  makeScalar = (value) => ({ dims: [], data: Float32Array.of(value) }),
  makeRowIndex = (rows) => ({ dims: [rows.length], data: rows }),
  onStep = () => {},
}) {
  let current = { ...initialState };
  const losses = [];
  const rows = rowIndex
    ? rowIndex.rows || minibatchIndices(rowIndex.numRows, rowIndex.size, rowIndex)
    : null;
  for (let t = 0; t < numSteps; t++) {
    const feeds = { ...captures };
    for (const { input } of state) feeds[input] = current[input];
    const values = scalarValues
      ? scalarValues(t)
      : stepScalars(scalars, t, { ...rates, numSteps });
    for (const name of scalars) feeds[name] = makeScalar(values[name]);
    if (rows) feeds[rowIndex.input] = makeRowIndex(rows(t));

    const outputs = await runStep(feeds, t);
    const next = {};
    for (const { input, output } of state) {
      if (outputs[output] === undefined) {
        throw new Error(`the step graph produced no '${output}' to carry back into '${input}'`);
      }
      next[input] = outputs[output];
    }
    current = next;
    if (loss && outputs[loss]) losses.push(Number(outputs[loss].data[0]));
    await onStep(t, losses.length ? losses[losses.length - 1] : null);
  }
  return { state: current, losses };
}

// Capture the activations a step graph binds, out of the float model.
//
// The names come from the plan's `captures` (their `source` field): the
// block's own externals plus the teacher, the float model's output for this
// block. A source that is a graph *input* is not asked of the model at all --
// its value is the feed we are about to send -- and only the rest are exposed
// as extra graph outputs, with onnxsim_add_graph_outputs, exactly as
// quantize_calibration.mjs does for calibration.
//
// `rowFeeds` is one feeds object per calibration row; the model is run once
// per row and the results concatenated along axis 0. That is what qat.py's
// own _capture does across calibration batches, and running row by row rather
// than as one batch keeps this working on a model whose batch axis is fixed at
// 1 -- most exported models.
//
// Returns { [source]: { dims, data } } sized exactly as the plan declared,
// which is checked here: a mismatch means the model was fed something other
// than what the step graph was built for, and it is far cheaper to say so than
// to let onnxruntime report a shape error about a `qat__`-prefixed tensor.
export async function captureActivations(ort, runtime, floatBytes, captures, rowFeeds, options = {}) {
  const { providers = ["wasm"], log = () => {} } = options;
  const wanted = new Map();
  for (const capture of captures) {
    if (!wanted.has(capture.source)) wanted.set(capture.source, capture.dims);
  }

  const session = await ort.InferenceSession.create(floatBytes, { executionProviders: providers });
  const graphInputs = new Set(session.inputNames);
  const extra = [...wanted.keys()].filter((name) => !graphInputs.has(name));

  let runner = session;
  if (extra.length) {
    const augmented = runtime.onnxsim_add_graph_outputs(floatBytes, extra);
    if (!augmented) {
      throw new Error("failed to expose the block's activations (add_graph_outputs)");
    }
    // Copy out of the wasm heap right away -- the view dies with the next wasm
    // call, and onnxruntime-web needs its own stable copy anyway.
    const augmentedBytes = new Uint8Array(augmented).slice();
    runner = await ort.InferenceSession.create(augmentedBytes, { executionProviders: providers });
  }

  const parts = new Map([...wanted.keys()].map((name) => [name, []]));
  try {
    for (let i = 0; i < rowFeeds.length; i++) {
      log(`capturing activations: row ${i + 1}/${rowFeeds.length}…`);
      const outputs = await runner.run(rowFeeds[i]);
      for (const name of wanted.keys()) {
        const tensor = graphInputs.has(name) ? rowFeeds[i][name] : outputs[name];
        if (!tensor || !tensor.data) {
          throw new Error(`the float model produced no value for '${name}'`);
        }
        parts.get(name).push(tensor.data);
      }
    }
  } finally {
    // These are per block and a whole-model run makes one pair each time
    // round, so they are released rather than left to the collector.
    for (const s of new Set([session, runner])) {
      if (typeof s.release === "function") await s.release();
    }
  }

  const captured = {};
  for (const [name, dims] of wanted) {
    const total = dims.reduce((a, b) => a * b, 1);
    const rows = parts.get(name);
    const perRow = rows.length ? rows[0].length : 0;
    if (perRow * rows.length !== total) {
      throw new Error(
        `'${name}': captured ${rows.length} row(s) of ${perRow} value(s), but the step ` +
          `graph declares ${dims.join("×")} (${total}) -- the calibration rows do not match ` +
          `the shape the block was built for`,
      );
    }
    const data = new Float32Array(total);
    rows.forEach((row, i) => data.set(row, i * perRow));
    captured[name] = { dims: [...dims], data };
  }
  return captured;
}

// Build one feeds object per calibration row.
//
// Each row gets its own sample context, so the "sample data" fill fetches a
// *fresh* random Hugging Face row per calibration row (hf_datasets.mjs's
// fetchSample* never memoize a picked row) rather than repeating one sample
// numRows times -- which would make every row of the reconstruction target
// identical and the whole calibration set worth one row. The synthetic fills
// are seeded and therefore identical row to row by construction; that is what
// they are, and the panel says so.
export async function buildCalibrationRows(ort, session, numRows, options = {}) {
  const { fill = "sample", modelName = null, log = () => {} } = options;
  const rows = [];
  for (let i = 0; i < numRows; i++) {
    const ctx = fill === "sample" ? createSampleContext() : null;
    rows.push(await makeDummyInputs(ort, session, 1, fill, modelName, ctx, log));
  }
  return rows;
}

// Copy a plan out of the wasm heap.
//
// Every buffer onnxsim_qat_build_step_graph hands back -- the step graph's
// bytes and each initial-state tensor's data -- is a view the next wasm call
// reuses, so this runs before anything else can call in. It also turns the
// embind arrays into plain ones, which is what the rest of this file expects.
export function copyPlan(built) {
  return {
    planHandle: built.planHandle,
    stepGraph: new Uint8Array(built.stepGraph).slice(),
    state: built.state.map((s) => ({ input: s.input, output: s.output })),
    scalars: [...built.scalars],
    loss: built.loss,
    captures: built.captures.map((c) => ({
      input: c.input,
      source: c.source,
      dims: [...c.dims],
      teacher: c.teacher,
    })),
    initialState: built.initialState.map((t) => ({
      name: t.name,
      dtype: t.dtype,
      dims: [...t.dims],
      data: new Uint8Array(t.data).slice(),
    })),
    rowIndexInput: built.rowIndexInput,
    rowIndexSize: built.rowIndexSize,
    numRows: built.numRows,
  };
}

// Fine-tune one block and return the updated quantized model.
//
// `rowFeeds` are the calibration rows (buildCalibrationRows); `blockInput` /
// `blockOutput` name the block, `options` is the QatOptions object the binding
// takes ({ learnScales, learnActivationScales, fakeQuant, preserveSparsity,
// batchSize, batchSeed, shuffle }), and `rates` the scalar schedule
// (stepScalars' options).
//
// Returns { bytes, losses, plan }. The plan is always released, including on a
// failure: nothing expires on its own, and a page training block after block
// would otherwise keep every step graph it ever built.
export async function fineTuneBlock(runtime, {
  floatBytes,
  quantBytes,
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
  const built = runtime.onnxsim_qat_build_step_graph(
    floatBytes,
    quantBytes,
    blockInput,
    blockOutput,
    numRows,
    options,
  );
  if (!built) {
    // BuildQatStepGraph refuses loudly -- an unclosed block, a node with no
    // gradient rule (named), a block with no layer of the requested scheme --
    // and the binding turns that into null with the reason on the console,
    // which the page mirrors into its log.
    throw new Error(
      `could not build a step graph for ${blockInput} → ${blockOutput} ` +
        "(see the log/console for the refusal)",
    );
  }
  const plan = copyPlan(built);

  try {
    const ort = await ortLoader(needWebnn ? "all" : "default");
    const captured = await captureActivations(ort, runtime, floatBytes, plan.captures, rowFeeds, {
      providers,
      log,
    });
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
      runStep: (feeds) => session.run(feeds),
      makeScalar: (value) => new ort.Tensor("float32", Float32Array.of(value), []),
      makeRowIndex: (rows) => new ort.Tensor("int64", rows, [rows.length]),
      onStep,
    });

    const written = runtime.onnxsim_qat_write_back(
      quantBytes,
      plan.planHandle,
      finalStateForWriteBack(plan.state, state),
    );
    if (!written) {
      throw new Error("writing the trained state back failed (see the log/console)");
    }
    if (typeof session.release === "function") await session.release();
    return { bytes: new Uint8Array(written).slice(), losses, plan };
  } finally {
    runtime.onnxsim_qat_release_plan(plan.planHandle);
  }
}

// The loss curve, as a standalone SVG string the caller assigns to an
// element's innerHTML -- the same shape as quantize_metrics.mjs's
// renderQuantizationQuality, and pure for the same reason: it is testable
// without a DOM.
//
// Plotted on a log scale, because a reconstruction loss falls by orders of
// magnitude over a run and a linear axis shows one initial spike and then a
// flat line at zero. Every interpolated value is our own computed number.
export function renderLossCurve(losses, { width = 420, height = 120 } = {}) {
  if (!losses || losses.length === 0) return "";
  const pad = 4;
  const finite = losses.filter((v) => Number.isFinite(v) && v > 0);
  const lo = finite.length ? Math.log10(Math.min(...finite)) : 0;
  const hi = finite.length ? Math.log10(Math.max(...finite)) : 1;
  const span = hi - lo || 1;
  const x = (i) => pad + (i / Math.max(1, losses.length - 1)) * (width - 2 * pad);
  const y = (v) => {
    if (!Number.isFinite(v) || v <= 0) return height - pad;
    return pad + (1 - (Math.log10(v) - lo) / span) * (height - 2 * pad);
  };
  const points = losses.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const first = losses[0];
  const last = losses[losses.length - 1];
  const change = first > 0 ? `${((1 - last / first) * 100).toFixed(1)}% lower` : "";
  return (
    `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" ` +
    `role="img" aria-label="block reconstruction loss per step" ` +
    `style="max-width: 100%; border: 1px solid var(--border); border-radius: var(--radius);">` +
    `<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="${points}"/>` +
    `</svg>` +
    `<p class="tool-note" style="margin-top: 0.2em;">Block reconstruction loss (log scale) over ` +
    `${losses.length} step(s): ${first.toPrecision(4)} → ${last.toPrecision(4)}` +
    `${change ? `, ${change}` : ""}.</p>`
  );
}
