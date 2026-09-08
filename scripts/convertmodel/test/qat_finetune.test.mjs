// Unit test for the QAT panel's training loop (qat_finetune.mjs).
//
// No DOM, no onnxruntime-web and no wasm. The loop's mechanics are pure
// functions taking injected fakes for exactly this reason -- the same shape
// quantize_calibration.test.mjs exercises calibration in -- because the parts
// that can silently go wrong here are not the parts that need a browser:
// binding a capture to its `source` instead of its `input` trains on
// uninitialized memory rather than failing, a state output not carried back
// makes every step start over from the initial weights, and a wrong
// bias-correction factor makes the browser train a model differently from the
// Python with nothing to say so.
//
// The Adam bias corrections are checked against values printed from
// onnxsim.qat_graph.adam_bias_corrections itself, exactly rather than within a
// tolerance: both sides compute 1/(1 - beta**(t+1)) in IEEE754 double from the
// same two constants, so anything but bit-equality is a real divergence.
//
// Usage:
//   node test/qat_finetune.test.mjs

import assert from "node:assert/strict";

// qat_finetune.mjs pulls onnxruntime-web's loader and makeDummyInputs from
// inference_browser.mjs, which wires the inference panel at import time; a
// getElementById that finds nothing makes that wiring bail out immediately
// (`if (!btn) return`), the same stub quantize_calibration.test.mjs uses.
globalThis.document = { getElementById: () => null, querySelector: () => null };
globalThis.window = { addEventListener: () => {} };
const {
  adamBiasCorrections,
  bindCaptures,
  captureActivations,
  copyPlan,
  finalStateForWriteBack,
  fineTuneBlock,
  minibatchIndices,
  renderLossCurve,
  runStepLoop,
  stepScalars,
} = await import("../qat_finetune.mjs");

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

const tensor = (dims, data) => ({ dims, data: Float32Array.from(data) });

// ---------------------------------------------------------------------------
// The scalars fed fresh every step.

await check("Adam's bias corrections match qat_graph.adam_bias_corrections", async () => {
  // Printed from onnxsim.qat_graph.adam_bias_corrections(t) with repr(), i.e.
  // the exact doubles. The C++ AdamBiasCorrections narrows the same doubles to
  // float32, so all three implementations agree before the narrowing.
  const reference = {
    0: [10.000000000000002, 999.9999999999991],
    1: [5.263157894736843, 500.250125062538],
    7: [1.7558251562653666, 125.4381565776313],
    99: [1.0000265621044142, 10.503335278386363],
  };
  for (const [t, [m, v]] of Object.entries(reference)) {
    const got = adamBiasCorrections(Number(t));
    assert.equal(got.m_correction, m, `m_correction at t=${t}`);
    assert.equal(got.v_correction, v, `v_correction at t=${t}`);
  }
});

await check("only the scalars the plan names are produced", async () => {
  // apply_qat's own set: the weight learning rate plus the two corrections,
  // with the scale rates present only when those scales are trained.
  const weightsOnly = stepScalars(["qat__lr", "m_correction", "v_correction"], 0, {
    numSteps: 10,
    learningRate: 1e-3,
  });
  assert.deepEqual(Object.keys(weightsOnly).sort(), [
    "m_correction",
    "qat__lr",
    "v_correction",
  ]);
  const withScales = stepScalars(
    ["qat__lr", "m_correction", "v_correction", "qat__lr_scale", "qat__lr_act"],
    0,
    { numSteps: 10, scaleLearningRate: 5e-5, activationLearningRate: 0.02 },
  );
  assert.equal(withScales.qat__lr_scale, 5e-5);
  assert.equal(withScales.qat__lr_act, 0.02);
});

await check("the learning rate decays linearly and can be turned off", async () => {
  const at = (t, lrDecay) =>
    stepScalars(["qat__lr"], t, { numSteps: 4, learningRate: 1, lrDecay }).qat__lr;
  assert.deepEqual([at(0, true), at(1, true), at(2, true), at(3, true)], [1, 0.75, 0.5, 0.25]);
  assert.deepEqual([at(0, false), at(3, false)], [1, 1]);
});

await check("an unknown scalar throws instead of being left unfed", async () => {
  assert.throws(() => stepScalars(["qat__lr", "qat__lr_bias"], 0, {}), /unknown per-step scalar/);
});

// ---------------------------------------------------------------------------
// The minibatch row stream.

const rowsOf = (fn, t) => Array.from(fn(t), Number);

await check("every row is visited once per epoch", async () => {
  const rows = minibatchIndices(6, 3, { seed: 1 });
  const epoch = [...rowsOf(rows, 0), ...rowsOf(rows, 1)];
  assert.deepEqual([...epoch].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5]);
});

await check("a batch that would run off the end wraps rather than being short", async () => {
  // Static shapes are what let a step graph compile for an NPU, so every batch
  // is full: the straddling one is completed from the front of the next epoch.
  const rows = minibatchIndices(5, 3, { seed: 2 });
  for (const t of [0, 1, 2, 3]) assert.equal(rows(t).length, 3);
  const first = [...rowsOf(rows, 0), ...rowsOf(rows, 1)].slice(0, 5);
  assert.deepEqual([...first].sort((a, b) => a - b), [0, 1, 2, 3, 4]);
});

await check("the stream is a pure function of the step index", async () => {
  const a = minibatchIndices(7, 2, { seed: 3 });
  const b = minibatchIndices(7, 2, { seed: 3 });
  // Out of order on purpose: a loop stopped at step N and resumed there must
  // see the same batch, which a running generator would not give.
  assert.deepEqual(rowsOf(a, 5), rowsOf(b, 5));
  assert.deepEqual(rowsOf(a, 0), rowsOf(b, 0));
  assert.deepEqual(rowsOf(a, 5), rowsOf(b, 5));
  const other = minibatchIndices(7, 2, { seed: 4 });
  const differs = [0, 1, 2, 3].some((t) => rowsOf(a, t).join() !== rowsOf(other, t).join());
  assert.ok(differs, "a different seed should draw different batches");
});

await check("shuffle:false walks the rows in order", async () => {
  const rows = minibatchIndices(4, 2, { shuffle: false });
  assert.deepEqual(rowsOf(rows, 0), [0, 1]);
  assert.deepEqual(rowsOf(rows, 1), [2, 3]);
  assert.deepEqual(rowsOf(rows, 2), [0, 1]);
});

await check("the row index is int64, the one non-float input a step graph has", async () => {
  assert.ok(minibatchIndices(4, 2, {})(0) instanceof BigInt64Array);
  assert.throws(() => minibatchIndices(0, 2, {}), /numRows/);
  assert.throws(() => minibatchIndices(4, 0, {}), /batchSize/);
});

// ---------------------------------------------------------------------------
// Binding.

await check("a capture binds to its input, never its source", async () => {
  // Under a minibatch the whole set is bound to a resident qat__all_<name>
  // table and the block reads gathered rows out of it, so the two names
  // differ; binding by `source` would train on uninitialized memory.
  const captures = [
    { input: "qat__all_X", source: "X", dims: [4, 2], teacher: false },
    { input: "qat__teacher_all", source: "Y", dims: [4, 3], teacher: true },
  ];
  const feeds = bindCaptures(captures, { X: "x-values", Y: "y-values" });
  assert.deepEqual(feeds, { qat__all_X: "x-values", qat__teacher_all: "y-values" });
  assert.throws(() => bindCaptures(captures, { X: "x-values" }), /no captured activation for 'Y'/);
});

await check("the write-back state is keyed by state input and carries only dims/data", async () => {
  const state = [
    { input: "qat__w0", output: "qat__w0_next" },
    { input: "qat__mw0", output: "qat__mw0_next" },
  ];
  const final = finalStateForWriteBack(state, {
    qat__w0: tensor([2], [1, 2]),
    qat__mw0: tensor([2], [3, 4]),
  });
  assert.deepEqual(Object.keys(final), ["qat__w0", "qat__mw0"]);
  assert.deepEqual([...final.qat__w0.data], [1, 2]);
  assert.deepEqual(final.qat__w0.dims, [2]);
  assert.throws(() => finalStateForWriteBack(state, {}), /no value for 'qat__w0'/);
});

// ---------------------------------------------------------------------------
// The loop.

// A step graph stand-in: each state output is its input plus the learning
// rate, and the loss halves each step, so what came back can be told from what
// went in.
function fakeStepGraph(state, { loss = "qat__loss" } = {}) {
  const seen = [];
  return {
    seen,
    async runStep(feeds, t) {
      seen.push(feeds);
      const outputs = { [loss]: tensor([], [1 / 2 ** t]) };
      for (const { input, output } of state) {
        const lr = feeds.qat__lr.data[0];
        outputs[output] = tensor(feeds[input].dims, [...feeds[input].data].map((v) => v + lr));
      }
      return outputs;
    },
  };
}

await check("each state output is carried into the next step's input", async () => {
  const state = [{ input: "qat__w0", output: "qat__w0_next" }];
  const graph = fakeStepGraph(state);
  const { state: final, losses } = await runStepLoop({
    state,
    scalars: ["qat__lr", "m_correction", "v_correction"],
    loss: "qat__loss",
    initialState: { qat__w0: tensor([2], [0, 0]) },
    captures: { qat__X: tensor([2, 2], [1, 2, 3, 4]) },
    numSteps: 3,
    rates: { learningRate: 1, lrDecay: false },
    runStep: graph.runStep,
  });
  // Three steps of +1 each, i.e. the state really made it round the loop.
  assert.deepEqual([...final.qat__w0.data], [3, 3]);
  assert.deepEqual(losses, [1, 0.5, 0.25]);
  // The step after the first was handed the previous step's *output*.
  assert.deepEqual([...graph.seen[1].qat__w0.data], [1, 1]);
  assert.deepEqual([...graph.seen[2].qat__w0.data], [2, 2]);
});

await check("captures are bound once and every scalar is fed every step", async () => {
  const state = [{ input: "qat__w0", output: "qat__w0_next" }];
  const graph = fakeStepGraph(state);
  const capture = tensor([1], [7]);
  await runStepLoop({
    state,
    scalars: ["qat__lr", "m_correction", "v_correction"],
    loss: "qat__loss",
    initialState: { qat__w0: tensor([1], [0]) },
    captures: { qat__X: capture },
    numSteps: 4,
    rates: { learningRate: 0.5, lrDecay: true },
    runStep: graph.runStep,
  });
  for (let t = 0; t < 4; t++) {
    const feeds = graph.seen[t];
    assert.equal(feeds.qat__X, capture, "the capture is the same tensor every step");
    assert.equal(feeds.qat__lr.data[0], 0.5 * (1 - t / 4));
    // fround because the graph's scalar inputs are float32: this is the same
    // narrowing the C++ AdamBiasCorrections does, applied once, at the point
    // the value is fed rather than while it is computed.
    assert.equal(feeds.m_correction.data[0], Math.fround(adamBiasCorrections(t).m_correction));
    assert.equal(feeds.v_correction.data[0], Math.fround(adamBiasCorrections(t).v_correction));
    assert.deepEqual(feeds.qat__lr.dims, [], "the scalars are rank-0");
  }
});

await check("a minibatch feeds its row index and nothing else changes", async () => {
  const state = [{ input: "qat__w0", output: "qat__w0_next" }];
  const graph = fakeStepGraph(state);
  await runStepLoop({
    state,
    scalars: ["qat__lr", "m_correction", "v_correction"],
    loss: "qat__loss",
    initialState: { qat__w0: tensor([1], [0]) },
    captures: {},
    rowIndex: { input: "qat__rows", size: 2, numRows: 4, seed: 0, shuffle: false },
    numSteps: 2,
    rates: { learningRate: 1, lrDecay: false },
    runStep: graph.runStep,
  });
  assert.deepEqual(Array.from(graph.seen[0].qat__rows.data, Number), [0, 1]);
  assert.deepEqual(Array.from(graph.seen[1].qat__rows.data, Number), [2, 3]);
  assert.deepEqual(graph.seen[0].qat__rows.dims, [2]);
});

await check("a missing state output stops the loop instead of restarting it", async () => {
  const state = [{ input: "qat__w0", output: "qat__w0_next" }];
  await assert.rejects(
    runStepLoop({
      state,
      scalars: ["qat__lr", "m_correction", "v_correction"],
      initialState: { qat__w0: tensor([1], [0]) },
      numSteps: 1,
      runStep: async () => ({}),
    }),
    /produced no 'qat__w0_next'/,
  );
});

// ---------------------------------------------------------------------------
// Capturing the teacher's activations.

// An ORT stand-in: one session per created model, reporting `inputNames` and
// returning `outputs` (per run index) from run().
function fakeOrt(inputNames, outputsPerRun, seen = { created: [] }) {
  let run = 0;
  return {
    seen,
    Tensor: class {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    },
    InferenceSession: {
      async create(bytes, options) {
        seen.created.push({ bytes, options });
        return {
          inputNames,
          outputNames: [],
          async run(feeds) {
            seen.lastFeeds = feeds;
            return outputsPerRun[Math.min(run++, outputsPerRun.length - 1)];
          },
        };
      },
    },
  };
}

const fakeRuntime = (extra = {}) => ({
  added: [],
  onnxsim_add_graph_outputs(bytes, names) {
    this.added.push([...names]);
    return new Uint8Array([9, 9, 9]).buffer;
  },
  ...extra,
});

await check("captures concatenate the rows along axis 0", async () => {
  const captures = [
    { input: "qat__X", source: "X", dims: [3, 2], teacher: false },
    { input: "qat__teacher", source: "Y", dims: [3, 1], teacher: true },
  ];
  const runtime = fakeRuntime();
  const ort = fakeOrt(["X"], [
    { Y: tensor([1, 1], [10]) },
    { Y: tensor([1, 1], [20]) },
    { Y: tensor([1, 1], [30]) },
  ]);
  const rowFeeds = [
    { X: tensor([1, 2], [1, 2]) },
    { X: tensor([1, 2], [3, 4]) },
    { X: tensor([1, 2], [5, 6]) },
  ];
  const captured = await captureActivations(ort, runtime, new Uint8Array([1]), captures, rowFeeds);
  // 'X' is a graph input, so its value is the feed we already have and it is
  // never asked of the model; only 'Y' needs exposing as an extra output.
  assert.deepEqual(runtime.added, [["Y"]]);
  assert.deepEqual([...captured.X.data], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(captured.X.dims, [3, 2]);
  assert.deepEqual([...captured.Y.data], [10, 20, 30]);
});

await check("no extra outputs are added when every source is a graph input", async () => {
  const runtime = fakeRuntime();
  const ort = fakeOrt(["X"], [{}]);
  await captureActivations(
    ort,
    runtime,
    new Uint8Array([1]),
    [{ input: "qat__X", source: "X", dims: [1, 2] }],
    [{ X: tensor([1, 2], [1, 2]) }],
  );
  assert.deepEqual(runtime.added, [], "add_graph_outputs should not have been called");
  assert.equal(ort.seen.created.length, 1, "one session, not two");
});

await check("rows that do not fill the declared shape are refused", async () => {
  const runtime = fakeRuntime();
  const ort = fakeOrt(["X"], [{}]);
  await assert.rejects(
    captureActivations(
      ort,
      runtime,
      new Uint8Array([1]),
      [{ input: "qat__X", source: "X", dims: [4, 2] }],
      [{ X: tensor([1, 2], [1, 2]) }],
    ),
    /do not match the shape the block was built for/,
  );
});

// ---------------------------------------------------------------------------
// The whole per-block flow against a fake wasm runtime.

function builtPlan({ rowIndexInput = "", rowIndexSize = 0 } = {}) {
  return {
    planHandle: 7,
    // Views over wasm buffers in the real thing; plain arrays here, since what
    // matters is that copyPlan takes its own copy of them.
    stepGraph: new Uint8Array([1, 2, 3]),
    state: [{ input: "qat__w0", output: "qat__w0_next" }],
    scalars: ["qat__lr", "m_correction", "v_correction"],
    loss: "qat__loss",
    captures: [
      { input: "qat__X", source: "X", dims: [2, 2], teacher: false },
      { input: "qat__teacher", source: "Y", dims: [2, 1], teacher: true },
    ],
    initialState: [
      { name: "qat__w0", dtype: 1, dims: [2], data: new Uint8Array(new Float32Array([1, 2]).buffer) },
    ],
    rowIndexInput,
    rowIndexSize,
    numRows: 2,
  };
}

function qatRuntime(built, calls = []) {
  return {
    calls,
    onnxsim_add_graph_outputs: () => new Uint8Array([9]).buffer,
    onnxsim_qat_build_step_graph(...args) {
      calls.push(["build", args]);
      return built;
    },
    onnxsim_qat_write_back(bytes, handle, finalState) {
      calls.push(["write_back", handle, finalState]);
      return new Uint8Array([4, 5, 6]).buffer;
    },
    onnxsim_qat_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };
}

await check("copyPlan copies every wasm view out", async () => {
  const built = builtPlan();
  const plan = copyPlan(built);
  // The real buffers are reused by the next wasm call; overwriting them here
  // stands in for that.
  built.stepGraph.fill(0);
  built.initialState[0].data.fill(0);
  assert.deepEqual([...plan.stepGraph], [1, 2, 3]);
  assert.deepEqual([...new Float32Array(plan.initialState[0].data.buffer)], [1, 2]);
});

await check("a block trains, writes back and releases its plan", async () => {
  const runtime = qatRuntime(builtPlan());
  const ort = fakeOrt(["X"], [{ Y: tensor([1, 1], [1]) }, { Y: tensor([1, 1], [2]) }]);
  // The step session is the third created (float, augmented, step graph); its
  // run() must therefore answer with the step graph's own outputs.
  const stepOutputs = {
    qat__w0_next: tensor([2], [9, 9]),
    qat__loss: tensor([], [0.25]),
  };
  ort.InferenceSession.create = async (bytes, options) => {
    ort.seen.created.push({ bytes, options });
    return {
      inputNames: ["X"],
      outputNames: [],
      async run(feeds) {
        ort.seen.lastFeeds = feeds;
        if (feeds.qat__w0) return stepOutputs;
        return { Y: tensor([1, 1], [1]) };
      },
    };
  };

  const result = await fineTuneBlock(runtime, {
    floatBytes: new Uint8Array([1]),
    quantBytes: new Uint8Array([2]),
    blockInput: "X",
    blockOutput: "Y",
    rowFeeds: [{ X: tensor([1, 2], [1, 2]) }, { X: tensor([1, 2], [3, 4]) }],
    numSteps: 2,
    rates: { learningRate: 1e-3 },
    providers: ["webgpu", "wasm"],
    ortLoader: async () => ort,
  });

  assert.deepEqual([...result.bytes], [4, 5, 6]);
  assert.deepEqual(result.losses, [0.25, 0.25]);
  const kinds = runtime.calls.map((c) => c[0]);
  assert.deepEqual(kinds, ["build", "write_back", "release"]);
  // num_rows is the row count the captures will carry, and the options object
  // reaches the binding unchanged.
  assert.equal(runtime.calls[0][1][4], 2);
  // The write-back is keyed by state *input*, not by the output that produced
  // the value.
  assert.deepEqual(Object.keys(runtime.calls[1][2]), ["qat__w0"]);
  assert.deepEqual([...runtime.calls[1][2].qat__w0.data], [9, 9]);
  assert.equal(runtime.calls[2][1], 7, "the plan is released by handle");
  // The panel's execution provider choice reaches every session, training loop
  // included -- the whole point of running this in the browser at all.
  for (const created of ort.seen.created) {
    assert.deepEqual(created.options.executionProviders, ["webgpu", "wasm"]);
  }
});

await check("a plan is released even when the loop fails", async () => {
  const runtime = qatRuntime(builtPlan());
  const ort = fakeOrt(["X"], [{}]);
  await assert.rejects(
    fineTuneBlock(runtime, {
      floatBytes: new Uint8Array([1]),
      quantBytes: new Uint8Array([2]),
      blockInput: "X",
      blockOutput: "Y",
      // One row where the plan declares two: the capture check fires.
      rowFeeds: [{ X: tensor([1, 2], [1, 2]) }],
      numSteps: 1,
      ortLoader: async () => ort,
    }),
  );
  assert.deepEqual(
    runtime.calls.map((c) => c[0]),
    ["build", "release"],
    "nothing expires on its own -- a failed block must still drop its step graph",
  );
});

await check("a refused block reports the refusal rather than returning null", async () => {
  const runtime = qatRuntime(null);
  await assert.rejects(
    fineTuneBlock(runtime, {
      floatBytes: new Uint8Array([1]),
      quantBytes: new Uint8Array([2]),
      blockInput: "X",
      blockOutput: "X",
      rowFeeds: [{ X: tensor([1, 2], [1, 2]) }],
      ortLoader: async () => fakeOrt(["X"], [{}]),
    }),
    /could not build a step graph for X → X/,
  );
  assert.deepEqual(runtime.calls.map((c) => c[0]), ["build"], "there is no plan to release");
});

// ---------------------------------------------------------------------------

await check("the loss curve is an SVG with one point per step", async () => {
  assert.equal(renderLossCurve([]), "");
  const svg = renderLossCurve([100, 10, 1]);
  assert.match(svg, /<polyline/);
  assert.equal(svg.match(/\d+\.\d+,\d+\.\d+/g).length, 3);
  assert.match(svg, /99\.0% lower/);
  // A zero or negative loss has no log; it must not produce NaN coordinates.
  assert.ok(!renderLossCurve([1, 0]).includes("NaN"));
});

console.log(`\nqat_finetune: ${passed} checks passed`);
