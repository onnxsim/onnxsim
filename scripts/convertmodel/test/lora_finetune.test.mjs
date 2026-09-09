// Unit test for the LoRA panel's training loop (lora_finetune.mjs).
//
// No DOM, no onnxruntime-web and no wasm -- the same shape as
// qat_finetune.test.mjs, and for the same reason: the loop's mechanics are
// pure functions taking injected fakes. Most of what lora_finetune.mjs
// exports is re-exported straight from qat_finetune.mjs (minibatchIndices,
// bindCaptures, finalStateForWriteBack, runStepLoop, ...) and is already
// covered there; this file checks that the re-export is the very same
// function (so a future refactor cannot silently fork the two panels) and
// then covers what is genuinely new here: the LoRA-only scalar schedule, the
// dual-model capture merge (block externals from the injected model, the
// teacher from a reference model -- unlike QAT, which captures both from one
// float model), the injection step, block-boundary defaulting, and the
// four-call training flow (inject, build, loop, write-back+release).
//
// Usage:
//   node test/lora_finetune.test.mjs

import assert from "node:assert/strict";

// lora_finetune.mjs re-exports pieces of qat_finetune.mjs, which pulls in
// onnxruntime-web's loader and makeDummyInputs from inference_browser.mjs --
// the same DOM stub qat_finetune.test.mjs uses.
globalThis.document = { getElementById: () => null, querySelector: () => null };
globalThis.window = { addEventListener: () => {} };

const {
  adamBiasCorrections,
  bindCaptures,
  buildLoraStepPlan,
  captureLoraActivations,
  copyLoraPlan,
  finalStateForWriteBack,
  injectLoraAdapter,
  loraStepScalars,
  minibatchIndices,
  renderLossCurve,
  runStepLoop,
  trainLoraAdapter,
  trainLoraAdapterAllBlocks,
  trainLoraBlock,
  wholeGraphBlock,
} = await import("../lora_finetune.mjs");
const qat = await import("../qat_finetune.mjs");

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

const tensor = (dims, data) => ({ dims, data: Float32Array.from(data) });

// ---------------------------------------------------------------------------
// Reuse, not reimplementation: the shared pieces must be the identical
// function object qat_finetune.mjs exports, not a look-alike copy that could
// drift from what qat_finetune.test.mjs actually exercises.

await check("the pieces shared with QAT are re-exported, not re-implemented", async () => {
  assert.equal(adamBiasCorrections, qat.adamBiasCorrections);
  assert.equal(bindCaptures, qat.bindCaptures);
  assert.equal(finalStateForWriteBack, qat.finalStateForWriteBack);
  assert.equal(minibatchIndices, qat.minibatchIndices);
  assert.equal(renderLossCurve, qat.renderLossCurve);
  assert.equal(runStepLoop, qat.runStepLoop);
});

// ---------------------------------------------------------------------------
// LoRA's own scalar schedule: one learning rate ("lora__lr"), not three.

await check("only lora__lr and the two Adam corrections are ever produced", async () => {
  const values = loraStepScalars(["lora__lr", "m_correction", "v_correction"], 0, {
    numSteps: 10,
    learningRate: 1e-3,
  });
  assert.deepEqual(Object.keys(values).sort(), ["lora__lr", "m_correction", "v_correction"]);
});

await check("the learning rate decays linearly and can be turned off", async () => {
  const at = (t, lrDecay) =>
    loraStepScalars(["lora__lr"], t, { numSteps: 4, learningRate: 1, lrDecay }).lora__lr;
  assert.deepEqual([at(0, true), at(1, true), at(2, true), at(3, true)], [1, 0.75, 0.5, 0.25]);
  assert.deepEqual([at(0, false), at(3, false)], [1, 1]);
});

await check("the bias corrections agree with QAT's (same Adam, same formula)", async () => {
  for (const t of [0, 1, 7, 99]) {
    const values = loraStepScalars(["m_correction", "v_correction"], t, {});
    const reference = adamBiasCorrections(t);
    assert.equal(values.m_correction, reference.m_correction);
    assert.equal(values.v_correction, reference.v_correction);
  }
});

await check("QAT's own scalar names (weight-scale rates) are unknown here", async () => {
  // LoRA has no scales to learn, so a step graph asking for qat__lr_scale
  // would be a contract mismatch this must not paper over.
  assert.throws(
    () => loraStepScalars(["lora__lr", "qat__lr_scale"], 0, {}),
    /unknown per-step scalar 'qat__lr_scale'/,
  );
});

// ---------------------------------------------------------------------------
// Dual-model capture: block externals from the injected model, the teacher
// from the reference model.

function fakeOrt(sessionsByBytesKey) {
  const seen = { created: [] };
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
        const key = bytes[0]; // the tests below tag each model's bytes by its first byte
        const spec = sessionsByBytesKey[key];
        let run = 0;
        return {
          inputNames: spec.inputNames,
          outputNames: [],
          async run(feeds) {
            seen.lastFeeds = feeds;
            return spec.outputsPerRun[Math.min(run++, spec.outputsPerRun.length - 1)];
          },
        };
      },
    },
  };
}

await check("block externals come from the injected model, the teacher from the reference", async () => {
  const injectedBytes = new Uint8Array([1]);
  const referenceBytes = new Uint8Array([2]);
  const captures = [
    { input: "lora__X", source: "X", dims: [2, 2], teacher: false },
    { input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true },
  ];
  const runtime = {
    added: [],
    onnxsim_add_graph_outputs(bytes, names) {
      this.added.push([bytes[0], [...names]]);
      // Tags the augmented bytes with the same key byte as the model they
      // came from, so fakeOrt's create() (below) resolves them to the same
      // spec -- the real binding preserves everything about the model
      // except the added outputs.
      return new Uint8Array([bytes[0]]).buffer;
    },
  };
  const ort = fakeOrt({
    1: { inputNames: ["X"], outputsPerRun: [{}] }, // injected model: X is a graph input
    2: { inputNames: [], outputsPerRun: [{ Y: tensor([1, 1], [42]) }] }, // reference: Y is an output
  });
  const rowFeeds = [{ X: tensor([1, 2], [1, 2]) }, { X: tensor([1, 2], [3, 4]) }];
  const captured = await captureLoraActivations(
    ort,
    runtime,
    injectedBytes,
    referenceBytes,
    captures,
    rowFeeds,
  );
  assert.deepEqual([...captured.X.data], [1, 2, 3, 4]);
  assert.deepEqual([...captured.Y.data], [42, 42]);
  // X is a graph input of the injected model, so no extra output was needed
  // for it; Y had to be exposed on the *reference* model (bytes[0] === 2).
  assert.deepEqual(runtime.added, [[2, ["Y"]]]);
});

await check("a block with no teacher capture never touches the reference model", async () => {
  const runtime = { added: [], onnxsim_add_graph_outputs: () => new Uint8Array([9]).buffer };
  const ort = fakeOrt({ 1: { inputNames: ["X"], outputsPerRun: [{}] } });
  const captured = await captureLoraActivations(
    ort,
    runtime,
    new Uint8Array([1]),
    new Uint8Array([2]),
    [{ input: "lora__X", source: "X", dims: [1, 2], teacher: false }],
    [{ X: tensor([1, 2], [5, 6]) }],
  );
  assert.deepEqual([...captured.X.data], [5, 6]);
  assert.equal(ort.seen.created.length, 1, "only the injected model's session was created");
});

// ---------------------------------------------------------------------------
// target_data mode: the teacher capture is stacked directly out of
// caller-supplied rows, no reference model touched at all.

await check("target_data stacks the teacher capture directly, without a reference model", async () => {
  const injectedBytes = new Uint8Array([1]);
  const captures = [
    { input: "lora__X", source: "X", dims: [2, 2], teacher: false },
    { input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true },
  ];
  const runtime = {
    onnxsim_add_graph_outputs() {
      throw new Error("must not be called -- target_data mode never runs the reference model");
    },
  };
  const ort = fakeOrt({ 1: { inputNames: ["X"], outputsPerRun: [{}] } });
  const rowFeeds = [{ X: tensor([1, 2], [1, 2]) }, { X: tensor([1, 2], [3, 4]) }];
  const captured = await captureLoraActivations(
    ort,
    runtime,
    injectedBytes,
    null, // no reference bytes at all -- target_data does not need them
    captures,
    rowFeeds,
    [Float32Array.of(10), Float32Array.of(20)],
  );
  assert.deepEqual([...captured.X.data], [1, 2, 3, 4]);
  assert.deepEqual([...captured.Y.data], [10, 20]);
  assert.equal(ort.seen.created.length, 1, "only the injected model's session was created");
});

await check("target_data with the wrong number of rows is refused", async () => {
  await assert.rejects(
    captureLoraActivations(
      fakeOrt({}),
      {},
      new Uint8Array([1]),
      null,
      [{ input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true }],
      [{ X: tensor([1, 1], [1]) }, { X: tensor([1, 1], [2]) }],
      [Float32Array.of(1)], // one target row, but two calibration rows
    ),
    /target_data has 1 row\(s\) but 2 calibration row\(s\)/,
  );
});

await check("a target_data row of the wrong length is refused", async () => {
  await assert.rejects(
    captureLoraActivations(
      fakeOrt({}),
      {},
      new Uint8Array([1]),
      null,
      [{ input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true }],
      [{ X: tensor([1, 1], [1]) }, { X: tensor([1, 1], [2]) }],
      [Float32Array.of(1), Float32Array.of(2, 3)], // row 1 has 2 values, not 1
    ),
    /target_data row 1: 'Y' expected 1 value\(s\).*got 2/,
  );
});

// ---------------------------------------------------------------------------
// Plan copying: LoraStepPlan carries `parameters`, which QatStepPlan does not.

function builtLoraPlan({ rowIndexInput = "", rowIndexSize = 0 } = {}) {
  return {
    planHandle: 3,
    stepGraph: new Uint8Array([1, 2, 3]),
    state: [
      { input: "w.lora_A", output: "w.lora_A_next" },
      { input: "lora__m_w.lora_A", output: "lora__m_w.lora_A_next" },
      { input: "lora__v_w.lora_A", output: "lora__v_w.lora_A_next" },
    ],
    scalars: ["lora__lr", "m_correction", "v_correction"],
    loss: "lora__loss",
    captures: [
      { input: "lora__X", source: "X", dims: [2, 2], teacher: false },
      { input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true },
    ],
    initialState: [
      {
        name: "w.lora_A",
        dtype: 1,
        dims: [2],
        data: new Uint8Array(new Float32Array([0.1, 0.2]).buffer),
      },
    ],
    rowIndexInput,
    rowIndexSize,
    numRows: 2,
    parameters: ["w.lora_A", "w.lora_B"],
  };
}

await check("copyLoraPlan copies every wasm view out, and carries parameters", async () => {
  const built = builtLoraPlan();
  const plan = copyLoraPlan(built);
  built.stepGraph.fill(0);
  built.initialState[0].data.fill(0);
  built.parameters.push("tampered");
  assert.deepEqual([...plan.stepGraph], [1, 2, 3]);
  assert.deepEqual(
    [...new Float32Array(plan.initialState[0].data.buffer)],
    [...Float32Array.of(0.1, 0.2)],
  );
  assert.deepEqual(plan.parameters, ["w.lora_A", "w.lora_B"]);
});

// ---------------------------------------------------------------------------
// Injection.

await check("injectLoraAdapter copies the model bytes out and returns the adapter", async () => {
  const model = new Uint8Array([7, 7, 7]);
  const adapter = [
    {
      weightName: "w",
      nodeOutput: "y",
      opType: "MatMul",
      loraAName: "w.lora_A",
      loraBName: "w.lora_B",
      rank: 8,
      hasAlpha: false,
      alpha: 0,
    },
  ];
  const calls = [];
  const runtime = {
    onnxsim_lora_inject(bytes, options) {
      calls.push([bytes, options]);
      return { model, adapter };
    },
  };
  const result = injectLoraAdapter(runtime, new Uint8Array([1]), { rank: 8 });
  model.fill(0); // the wasm buffer being reused by the next call
  assert.deepEqual([...result.bytes], [7, 7, 7]);
  assert.deepEqual(result.adapter, adapter);
  assert.deepEqual(calls[0][1], { rank: 8 });
});

await check("a refused injection reports the refusal rather than returning null", async () => {
  const runtime = { onnxsim_lora_inject: () => null };
  assert.throws(
    () => injectLoraAdapter(runtime, new Uint8Array([1]), {}),
    /could not inject a LoRA adapter/,
  );
});

// ---------------------------------------------------------------------------
// Block-boundary defaulting: the whole model, named by its own graph
// input/output -- lora_entry.h's own recommended default.

// A minimal encoder for just the fields readGraph/primaryGraphInput read --
// the same one qat_blocks.test.mjs uses (ModelProto.graph = 7; GraphProto
// input = 11, output = 12; ValueInfoProto.name = 1).
function varint(n) {
  const out = [];
  let v = n;
  do {
    let byte = v & 0x7f;
    v >>>= 7;
    if (v) byte |= 0x80;
    out.push(byte);
  } while (v);
  return out;
}
function bytesField(field, bytes) {
  return [...varint(field * 8 + 2), ...varint(bytes.length), ...bytes];
}
const utf8 = new TextEncoder();
const strField = (field, s) => bytesField(field, [...utf8.encode(s)]);
// NodeProto: input = 1, output = 2, name = 3, op_type = 4; TensorProto.name
// = 8 -- the extra fields makeModel below carries for
// trainLoraAdapterAllBlocks' own tests further down, which need real nodes
// for discoverLoraBlocks to find real cuts in (wholeGraphBlock's own tests
// above only ever read the graph's input/output list).
function makeNode(opType, inputs, outputs) {
  return [
    ...inputs.flatMap((n) => strField(1, n)),
    ...outputs.flatMap((n) => strField(2, n)),
    ...strField(3, outputs[0] || opType),
    ...strField(4, opType),
  ];
}
function makeModel({ nodes = [], initializers = [], inputs = [], outputs = [] }) {
  const graph = [
    ...nodes.flatMap((n) => bytesField(1, n)),
    ...initializers.flatMap((n) => bytesField(5, strField(8, n))),
    ...inputs.flatMap((n) => bytesField(11, strField(1, n))),
    ...outputs.flatMap((n) => bytesField(12, strField(1, n))),
  ];
  return new Uint8Array(bytesField(7, graph));
}

await check("wholeGraphBlock names the model's own graph input and output", async () => {
  const model = makeModel({ inputs: ["X"], outputs: ["Y"] });
  assert.deepEqual(wholeGraphBlock(model), { input: "X", output: "Y" });
});

await check("wholeGraphBlock throws rather than guessing on a graph with neither", async () => {
  assert.throws(() => wholeGraphBlock(makeModel({})), /could not determine/);
});

// ---------------------------------------------------------------------------
// The whole flow: inject, build, loop, write back, release.

// `session.run(feeds)` -- like onnxruntime-web's own, and like
// qat_finetune.mjs's own runStep wrapper -- never receives the step index,
// so this reads the step from how many calls it has already seen rather
// than taking `t` as a parameter (a fake that took `t` would pass here and
// then be exercising a signature trainLoraAdapter never actually calls).
function fakeStepGraph(state, { loss = "lora__loss" } = {}) {
  const seen = [];
  return {
    seen,
    async runStep(feeds) {
      const t = seen.length;
      seen.push(feeds);
      const outputs = { [loss]: tensor([], [1 / 2 ** t]) };
      for (const { input, output } of state) {
        const lr = feeds.lora__lr ? feeds.lora__lr.data[0] : 0;
        outputs[output] = tensor(feeds[input].dims, [...feeds[input].data].map((v) => v + lr));
      }
      return outputs;
    },
  };
}

await check("training an adapter injects, builds, loops, writes back and releases", async () => {
  const injectedModel = new Uint8Array([1]);
  const adapter = [
    {
      weightName: "w",
      nodeOutput: "y",
      opType: "MatMul",
      loraAName: "w.lora_A",
      loraBName: "w.lora_B",
      rank: 4,
      hasAlpha: false,
      alpha: 0,
    },
  ];
  const built = {
    planHandle: 11,
    stepGraph: new Uint8Array([1, 2, 3]),
    state: [{ input: "w.lora_A", output: "w.lora_A_next" }],
    scalars: ["lora__lr", "m_correction", "v_correction"],
    loss: "lora__loss",
    captures: [
      { input: "lora__X", source: "X", dims: [2, 2], teacher: false },
      { input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true },
    ],
    initialState: [
      { name: "w.lora_A", dtype: 1, dims: [2], data: new Uint8Array(new Float32Array([0, 0]).buffer) },
    ],
    rowIndexInput: "",
    rowIndexSize: 0,
    numRows: 2,
    parameters: ["w.lora_A"],
  };
  const calls = [];
  const runtime = {
    calls,
    onnxsim_add_graph_outputs: () => new Uint8Array([9]).buffer,
    onnxsim_lora_inject(bytes, options) {
      calls.push(["inject", bytes, options]);
      return { model: injectedModel, adapter };
    },
    onnxsim_lora_build_step_graph(...args) {
      calls.push(["build", args]);
      return built;
    },
    onnxsim_lora_write_back(bytes, handle, finalState) {
      calls.push(["write_back", handle, finalState]);
      return new Uint8Array([4, 5, 6]);
    },
    onnxsim_lora_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };

  const graph = fakeStepGraph(built.state);
  const ort = fakeOrt({
    1: { inputNames: ["X"], outputsPerRun: [{}] }, // the injected model: X is a graph input
  });
  ort.InferenceSession.create = async (bytes, options) => {
    ort.seen.created.push({ bytes, options });
    if (bytes[0] === 1 && bytes.length === 1) {
      return { inputNames: ["X"], outputNames: [], async run() { return {}; } };
    }
    if (bytes.length === 3) {
      // the step graph itself
      return { inputNames: [], outputNames: [], run: graph.runStep };
    }
    // the reference model (defaults to injectedModel's own bytes here)
    return {
      inputNames: [],
      outputNames: [],
      async run() {
        return { Y: tensor([1, 1], [9]) };
      },
    };
  };

  const result = await trainLoraAdapter(runtime, {
    baseBytes: new Uint8Array([1]),
    blockInput: "X",
    blockOutput: "Y",
    rowFeeds: [{ X: tensor([1, 2], [1, 2]) }, { X: tensor([1, 2], [3, 4]) }],
    numSteps: 2,
    rates: { learningRate: 1e-3 },
    providers: ["webgpu", "wasm"],
    ortLoader: async () => ort,
  });

  assert.deepEqual([...result.bytes], [4, 5, 6]);
  assert.deepEqual(result.losses, [1, 0.5]);
  // injectLoraAdapter copies each target object out (defensively, the same
  // way it copies the model bytes), so this is the same *data*, not the same
  // array/object identity onnxsim_lora_inject itself returned.
  assert.deepEqual(result.adapter, adapter);
  const kinds = calls.map((c) => c[0]);
  assert.deepEqual(kinds, ["inject", "build", "write_back", "release"]);
  // num_rows is the calibration row count; the block names default to the
  // ones passed in explicitly here.
  assert.deepEqual(calls[1][1].slice(2, 5), ["X", "Y", 2]);
  // The write-back is keyed by state *input*, not by the output that produced
  // the value.
  assert.deepEqual(Object.keys(calls[2][2]), ["w.lora_A"]);
  assert.equal(calls[3][1], 11, "the plan is released by handle");
});

await check("a plan is released even when the loop fails", async () => {
  const injectedModel = new Uint8Array([1]);
  const adapter = [{ weightName: "w", loraAName: "w.lora_A", loraBName: "w.lora_B", rank: 4 }];
  const built = {
    planHandle: 5,
    stepGraph: new Uint8Array([1]),
    state: [],
    scalars: [],
    loss: "",
    captures: [{ input: "lora__X", source: "X", dims: [4, 2], teacher: false }],
    initialState: [],
    rowIndexInput: "",
    rowIndexSize: 0,
    numRows: 1,
    parameters: ["w.lora_A"],
  };
  const calls = [];
  const runtime = {
    calls,
    onnxsim_add_graph_outputs: () => new Uint8Array([9]).buffer,
    onnxsim_lora_inject: () => ({ model: injectedModel, adapter }),
    onnxsim_lora_build_step_graph(...args) {
      calls.push(["build", args]);
      return built;
    },
    onnxsim_lora_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };
  const ort = fakeOrt({ 1: { inputNames: ["X"], outputsPerRun: [{}] } });
  await assert.rejects(
    trainLoraAdapter(runtime, {
      baseBytes: new Uint8Array([1]),
      blockInput: "X",
      blockOutput: "Y",
      // One row where the plan declares four: captureActivations' own shape
      // check fires.
      rowFeeds: [{ X: tensor([1, 2], [1, 2]) }],
      numSteps: 1,
      ortLoader: async () => ort,
    }),
  );
  assert.deepEqual(
    calls.map((c) => c[0]),
    ["build", "release"],
    "nothing expires on its own -- a failed run must still drop its step graph",
  );
});

// ---------------------------------------------------------------------------
// target_data mode: trainLoraBlock's own XOR (exactly one of
// referenceBytes/targetData) and, once satisfied, a run that never creates a
// reference-model session and trains against the supplied values.

await check("trainLoraBlock refuses both referenceBytes and targetData", async () => {
  await assert.rejects(
    trainLoraBlock(
      {},
      {
        injectedBytes: new Uint8Array([1]),
        adapter: [],
        referenceBytes: new Uint8Array([2]),
        targetData: [Float32Array.of(1)],
        blockInput: "X",
        blockOutput: "Y",
        rowFeeds: [{ X: tensor([1, 1], [1]) }],
      },
    ),
    /exactly one of referenceBytes or targetData/,
  );
});

await check("trainLoraBlock refuses neither referenceBytes nor targetData", async () => {
  await assert.rejects(
    trainLoraBlock(
      {},
      {
        injectedBytes: new Uint8Array([1]),
        adapter: [],
        blockInput: "X",
        blockOutput: "Y",
        rowFeeds: [{ X: tensor([1, 1], [1]) }],
      },
    ),
    /exactly one of referenceBytes or targetData/,
  );
});

await check("trainLoraBlock trains against caller-supplied targetData, with no reference model", async () => {
  const injectedBytes = new Uint8Array([1]);
  const built = {
    planHandle: 41,
    stepGraph: new Uint8Array([1, 2, 3]),
    state: [{ input: "w.lora_A", output: "w.lora_A_next" }],
    scalars: ["lora__lr", "m_correction", "v_correction"],
    loss: "lora__loss",
    captures: [
      { input: "lora__X", source: "X", dims: [2, 2], teacher: false },
      { input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true },
    ],
    initialState: [
      { name: "w.lora_A", dtype: 1, dims: [2], data: new Uint8Array(new Float32Array([0, 0]).buffer) },
    ],
    rowIndexInput: "",
    rowIndexSize: 0,
    numRows: 2,
    parameters: ["w.lora_A"],
  };
  const calls = [];
  const runtime = {
    onnxsim_lora_build_step_graph(...args) {
      calls.push(["build", args]);
      return built;
    },
    onnxsim_lora_write_back(bytes, handle, finalState) {
      calls.push(["write_back", handle, finalState]);
      return new Uint8Array([9, 9]);
    },
    onnxsim_lora_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };
  const graph = fakeStepGraph(built.state);
  const ort = fakeOrt({ 1: { inputNames: ["X"], outputsPerRun: [{}] } });
  ort.InferenceSession.create = async (bytes, options) => {
    ort.seen.created.push({ bytes, options });
    if (bytes.length === 3) return { inputNames: [], outputNames: [], run: graph.runStep };
    // Only ever the injected model -- X is a graph input so no augmented
    // session is needed, and no reference-model bytes exist to be asked for.
    return { inputNames: ["X"], outputNames: [], async run() { return {}; } };
  };

  const result = await trainLoraBlock(runtime, {
    injectedBytes,
    adapter: [{ weightName: "w", loraAName: "w.lora_A", loraBName: "w.lora_B", rank: 4 }],
    targetData: [Float32Array.of(5), Float32Array.of(7)],
    blockInput: "X",
    blockOutput: "Y",
    rowFeeds: [{ X: tensor([1, 2], [1, 2]) }, { X: tensor([1, 2], [3, 4]) }],
    numSteps: 2,
    rates: { learningRate: 1e-3 },
    ortLoader: async () => ort,
  });

  assert.deepEqual([...result.bytes], [9, 9]);
  assert.deepEqual(result.losses, [1, 0.5]);
  // Exactly two sessions ever created: the injected model (for its input
  // names) and the step graph -- never a third, reference-model session.
  assert.equal(ort.seen.created.length, 2);
  // The teacher feed the step graph actually trained against is the stacked
  // targetData, not anything read off a model.
  assert.deepEqual([...graph.seen[0].lora__teacher.data], [5, 7]);
  assert.deepEqual(calls.map((c) => c[0]), ["build", "write_back", "release"]);
});

// ---------------------------------------------------------------------------
// target_data mode at trainLoraAdapter's own layer: the XOR fires here too
// (not only once execution reaches trainLoraBlock), and reaching
// trainLoraBlock in this mode never exercises the referenceBytes/baseBytes
// fallback.

await check("trainLoraAdapter refuses both referenceBytes and targetData", async () => {
  await assert.rejects(
    trainLoraAdapter(
      {},
      {
        baseBytes: new Uint8Array([1]),
        referenceBytes: new Uint8Array([2]),
        targetData: [Float32Array.of(1)],
        blockInput: "X",
        blockOutput: "Y",
        rowFeeds: [{ X: tensor([1, 1], [1]) }],
      },
    ),
    /at most one of referenceBytes or targetData/,
  );
});

await check("trainLoraAdapter's targetData mode reaches trainLoraBlock without the baseBytes fallback", async () => {
  const injectedModel = new Uint8Array([1]);
  const adapter = [
    {
      weightName: "w",
      nodeOutput: "y",
      opType: "MatMul",
      loraAName: "w.lora_A",
      loraBName: "w.lora_B",
      rank: 4,
      hasAlpha: false,
      alpha: 0,
    },
  ];
  const built = {
    planHandle: 12,
    stepGraph: new Uint8Array([1, 2, 3]),
    state: [{ input: "w.lora_A", output: "w.lora_A_next" }],
    scalars: ["lora__lr", "m_correction", "v_correction"],
    loss: "lora__loss",
    captures: [
      { input: "lora__X", source: "X", dims: [2, 2], teacher: false },
      { input: "lora__teacher", source: "Y", dims: [2, 1], teacher: true },
    ],
    initialState: [
      { name: "w.lora_A", dtype: 1, dims: [2], data: new Uint8Array(new Float32Array([0, 0]).buffer) },
    ],
    rowIndexInput: "",
    rowIndexSize: 0,
    numRows: 2,
    parameters: ["w.lora_A"],
  };
  const calls = [];
  const runtime = {
    onnxsim_lora_inject(bytes, options) {
      calls.push(["inject", bytes, options]);
      return { model: injectedModel, adapter };
    },
    onnxsim_lora_build_step_graph(...args) {
      calls.push(["build", args]);
      return built;
    },
    onnxsim_lora_write_back(bytes, handle, finalState) {
      calls.push(["write_back", handle, finalState]);
      return new Uint8Array([4, 5, 6]);
    },
    onnxsim_lora_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };
  const graph = fakeStepGraph(built.state);
  const ort = fakeOrt({ 1: { inputNames: ["X"], outputsPerRun: [{}] } });
  ort.InferenceSession.create = async (bytes, options) => {
    ort.seen.created.push({ bytes, options });
    if (bytes.length === 3) return { inputNames: [], outputNames: [], run: graph.runStep };
    return { inputNames: ["X"], outputNames: [], async run() { return {}; } };
  };

  const result = await trainLoraAdapter(runtime, {
    baseBytes: new Uint8Array([1]),
    targetData: [Float32Array.of(5), Float32Array.of(7)],
    blockInput: "X",
    blockOutput: "Y",
    rowFeeds: [{ X: tensor([1, 2], [1, 2]) }, { X: tensor([1, 2], [3, 4]) }],
    numSteps: 1,
    ortLoader: async () => ort,
  });

  assert.deepEqual([...result.bytes], [4, 5, 6]);
  // Only the injected model and the step graph were ever asked for a
  // session -- baseBytes was never separately opened as a reference model.
  assert.equal(ort.seen.created.length, 2);
  assert.deepEqual([...graph.seen[0].lora__teacher.data], [5, 7]);
});

// ---------------------------------------------------------------------------
// buildLoraStepPlan on its own -- the refusal path a caller sees without
// going through the whole trainLoraAdapter flow.

await check("a refused block reports the refusal rather than returning null", async () => {
  const runtime = { onnxsim_lora_build_step_graph: () => null };
  assert.throws(
    () => buildLoraStepPlan(runtime, new Uint8Array([1]), [], "X", "X", 2, {}),
    /could not build a LoRA step graph for X → X/,
  );
});

// ---------------------------------------------------------------------------
// trainLoraAdapterAllBlocks: the multi-block loop over lora_blocks.mjs's own
// discoverLoraBlocks. Reuses makeNode/makeModel above -- discoverLoraBlocks
// needs a real graph to find real cuts in, unlike every `built` step-graph
// fixture in this section (those are the wasm binding's own return value,
// entirely under this test's control regardless of what the graph actually
// contains).

// X -> MatMul -> h (adapter #1) -> MatMul -> Y (adapter #2), the same shape
// lora_blocks.test.mjs's own INJECTED_CHAIN uses -- discoverLoraBlocks finds
// two per-target blocks here at maxTargetsPerBlock: 1.
const TWO_BLOCK_MODEL = makeModel({
  nodes: [makeNode("MatMul", ["X", "W1"], ["h"]), makeNode("MatMul", ["h", "W2"], ["Y"])],
  initializers: ["W1", "W2"],
  inputs: ["X"],
  outputs: ["Y"],
});

await check("trainLoraAdapterAllBlocks trains every discovered block in order, chaining writes", async () => {
  const adapter = [
    { weightName: "w1", nodeOutput: "h", loraAName: "w1.lora_A", loraBName: "w1.lora_B", rank: 4 },
    { weightName: "w2", nodeOutput: "Y", loraAName: "w2.lora_A", loraBName: "w2.lora_B", rank: 4 },
  ];
  // Every block's own capture is defined to read "X" (a real graph input of
  // TWO_BLOCK_MODEL) so captureActivations' graph-input fast path resolves
  // it straight from rowFeeds with no session run needed -- the fixture is
  // free to be this simple because a `built` object is the wasm binding's
  // own mocked return value, not something derived from the real graph
  // (discoverLoraBlocks, the one piece that reads the real graph, is
  // exercised for real above it).
  const makeBuilt = (handle, state) => ({
    planHandle: handle,
    stepGraph: new Uint8Array([9, handle]),
    state,
    scalars: ["lora__lr", "m_correction", "v_correction"],
    loss: "lora__loss",
    captures: [
      { input: "lora__X", source: "X", dims: [1, 1], teacher: false },
      { input: "lora__teacher", source: "Y", dims: [1, 1], teacher: true },
    ],
    initialState: [
      { name: state[0].input, dtype: 1, dims: [1], data: new Uint8Array(new Float32Array([0]).buffer) },
    ],
    rowIndexInput: "",
    rowIndexSize: 0,
    numRows: 1,
    parameters: [state[0].input],
  });
  const built1 = makeBuilt(21, [{ input: "w1.lora_A", output: "w1.lora_A_next" }]);
  const built2 = makeBuilt(22, [{ input: "w2.lora_A", output: "w2.lora_A_next" }]);

  const calls = [];
  let injectCallCount = 0;
  const runtime = {
    onnxsim_add_graph_outputs: () => new Uint8Array([0]).buffer,
    onnxsim_lora_inject(bytes, options) {
      injectCallCount += 1;
      calls.push(["inject", options]);
      return { model: TWO_BLOCK_MODEL, adapter };
    },
    onnxsim_lora_build_step_graph(bytes, adapterArg, blockInput, blockOutput, numRows, options) {
      calls.push(["build", blockInput, blockOutput, adapterArg.map((t) => t.nodeOutput)]);
      if (blockInput === "X" && blockOutput === "h") return built1;
      if (blockInput === "h" && blockOutput === "Y") return built2;
      return null;
    },
    onnxsim_lora_write_back(bytes, handle) {
      calls.push(["write_back", handle]);
      // Tagged so the *next* block's build call is provably fed this
      // block's own write-back rather than the original injected bytes --
      // the "recapture from the student after previous blocks were tuned"
      // sequencing this function's own doc comment promises.
      return new Uint8Array([100, handle]);
    },
    onnxsim_lora_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };

  const graph1 = fakeStepGraph(built1.state);
  const graph2 = fakeStepGraph(built2.state);
  const ort = fakeOrt({});
  // Keyed by shape rather than by a single byte: the step graphs ([9,
  // handle]) and write_back's own tagged output ([100, handle]) are both
  // two bytes long, and 21/22 collide across the two purposes if only
  // bytes[1] is compared -- see the block below.
  ort.InferenceSession.create = async (bytes) => {
    if (bytes.length === 2 && bytes[0] === 9) {
      return bytes[1] === 21
        ? { inputNames: [], outputNames: [], run: graph1.runStep }
        : { inputNames: [], outputNames: [], run: graph2.runStep };
    }
    if (bytes.length === 1) {
      // the reference model (baseBytes itself), queried once per block for
      // its teacher capture
      return { inputNames: [], outputNames: [], async run() { return { Y: tensor([1, 1], [9]) }; } };
    }
    // the injected model -- TWO_BLOCK_MODEL's own bytes for block 1, or
    // write_back's [100, handle] tag standing in for it for block 2 onward.
    // "X" is a real graph input of TWO_BLOCK_MODEL and every `built` fixture
    // above names it as the block-external capture's source regardless of
    // which block, so this always hits captureActivations' own graph-input
    // fast path -- no augmented session, no model actually run.
    return { inputNames: ["X"], outputNames: [], async run() { return {}; } };
  };

  const result = await trainLoraAdapterAllBlocks(runtime, {
    baseBytes: new Uint8Array([1]),
    maxTargetsPerBlock: 1,
    rowFeeds: [{ X: tensor([1, 1], [1]) }],
    numSteps: 1,
    rates: { learningRate: 1e-3 },
    ortLoader: async () => ort,
  });

  assert.equal(injectCallCount, 1, "InjectLora runs once for the whole run, not once per block");
  const builds = calls.filter((c) => c[0] === "build");
  assert.deepEqual(
    builds.map((c) => c.slice(1)),
    [
      ["X", "h", ["h"]], // block 1 trains only its own target
      ["h", "Y", ["Y"]], // block 2 trains only its own
    ],
  );
  // Block 2's build call is proof the loop threaded block 1's write-back
  // forward: onnxsim_lora_write_back tagged its output [100, 21], and that
  // is exactly the `bytes` build_step_graph's own mock never inspects but
  // this assertion does, by construction of the call order above -- the
  // stronger, more direct check is that write_back and release both fire
  // twice, once per block, in build/write_back/release order each time.
  assert.deepEqual(
    calls.map((c) => c[0]),
    ["inject", "build", "write_back", "release", "build", "write_back", "release"],
  );
  assert.equal(result.results.length, 2);
  assert.ok(result.results.every((r) => r.trained));
  assert.equal(result.losses.length, 2, "one loss per block, one step each");
  assert.deepEqual([...result.bytes], [100, 22], "the final bytes are the last block's write-back");
});

await check("trainLoraAdapterAllBlocks skips a block build_step_graph refuses and continues", async () => {
  const adapter = [
    { weightName: "w1", nodeOutput: "h", loraAName: "w1.lora_A", loraBName: "w1.lora_B", rank: 4 },
    { weightName: "w2", nodeOutput: "Y", loraAName: "w2.lora_A", loraBName: "w2.lora_B", rank: 4 },
  ];
  const built2 = {
    planHandle: 22,
    stepGraph: new Uint8Array([9, 22]),
    state: [{ input: "w2.lora_A", output: "w2.lora_A_next" }],
    scalars: ["lora__lr", "m_correction", "v_correction"],
    loss: "lora__loss",
    captures: [
      { input: "lora__X", source: "X", dims: [1, 1], teacher: false },
      { input: "lora__teacher", source: "Y", dims: [1, 1], teacher: true },
    ],
    initialState: [
      { name: "w2.lora_A", dtype: 1, dims: [1], data: new Uint8Array(new Float32Array([0]).buffer) },
    ],
    rowIndexInput: "",
    rowIndexSize: 0,
    numRows: 1,
    parameters: ["w2.lora_A"],
  };
  const calls = [];
  const runtime = {
    onnxsim_add_graph_outputs: () => new Uint8Array([0]).buffer,
    onnxsim_lora_inject: () => ({ model: TWO_BLOCK_MODEL, adapter }),
    onnxsim_lora_build_step_graph(bytes, adapterArg, blockInput, blockOutput) {
      calls.push(["build", blockInput, blockOutput]);
      if (blockInput === "X" && blockOutput === "h") return null; // refused
      if (blockInput === "h" && blockOutput === "Y") return built2;
      return null;
    },
    onnxsim_lora_write_back: () => new Uint8Array([100, 22]),
    onnxsim_lora_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };

  const graph2 = fakeStepGraph(built2.state);
  const ort = fakeOrt({});
  ort.InferenceSession.create = async (bytes) => {
    if (bytes.length === 2 && bytes[0] === 9 && bytes[1] === 22) {
      return { inputNames: [], outputNames: [], run: graph2.runStep };
    }
    if (bytes.length === 1) {
      return { inputNames: [], outputNames: [], async run() { return { Y: tensor([1, 1], [9]) }; } };
    }
    // the injected model (TWO_BLOCK_MODEL itself -- block 1 is refused
    // before it ever writes back, so block 2 still sees the original bytes)
    return { inputNames: ["X"], outputNames: [], async run() { return {}; } };
  };

  const result = await trainLoraAdapterAllBlocks(runtime, {
    baseBytes: new Uint8Array([1]),
    maxTargetsPerBlock: 1,
    rowFeeds: [{ X: tensor([1, 1], [1]) }],
    numSteps: 1,
    ortLoader: async () => ort,
  });

  assert.equal(result.results.length, 2);
  assert.equal(result.results[0].trained, false);
  assert.match(result.results[0].skippedReason, /could not build a LoRA step graph for X → h/);
  assert.equal(result.results[1].trained, true);
  assert.equal(result.losses.length, 1, "only the trained block contributes a loss");
  // The refused block never reached write_back/release; the trained one did.
  assert.deepEqual(calls.map((c) => c[0]), ["build", "build", "release"]);
});

await check("trainLoraAdapterAllBlocks refuses both referenceBytes and targetData", async () => {
  await assert.rejects(
    trainLoraAdapterAllBlocks(
      {},
      {
        baseBytes: new Uint8Array([1]),
        referenceBytes: new Uint8Array([2]),
        targetData: [Float32Array.of(1)],
        rowFeeds: [{ X: tensor([1, 1], [1]) }],
      },
    ),
    /at most one of referenceBytes or targetData/,
  );
});

await check("trainLoraAdapterAllBlocks' targetData mode reaches trainLoraBlock, no reference model", async () => {
  // One MatMul, one adapter target -- discoverLoraBlocks proposes exactly
  // one block, so a single targetData array (one row) legitimately matches
  // that one block's own reconstruction target.
  const ONE_BLOCK_MODEL = makeModel({
    nodes: [makeNode("MatMul", ["X", "W1"], ["Y"])],
    initializers: ["W1"],
    inputs: ["X"],
    outputs: ["Y"],
  });
  const adapter = [{ weightName: "w1", nodeOutput: "Y", loraAName: "w1.lora_A", loraBName: "w1.lora_B", rank: 4 }];
  const built = {
    planHandle: 51,
    stepGraph: new Uint8Array([9, 9]),
    state: [{ input: "w1.lora_A", output: "w1.lora_A_next" }],
    scalars: ["lora__lr", "m_correction", "v_correction"],
    loss: "lora__loss",
    captures: [
      { input: "lora__X", source: "X", dims: [1, 1], teacher: false },
      { input: "lora__teacher", source: "Y", dims: [1, 1], teacher: true },
    ],
    initialState: [
      { name: "w1.lora_A", dtype: 1, dims: [1], data: new Uint8Array(new Float32Array([0]).buffer) },
    ],
    rowIndexInput: "",
    rowIndexSize: 0,
    numRows: 1,
    parameters: ["w1.lora_A"],
  };
  const calls = [];
  const runtime = {
    onnxsim_lora_inject: () => ({ model: ONE_BLOCK_MODEL, adapter }),
    onnxsim_lora_build_step_graph(bytes, adapterArg, blockInput, blockOutput) {
      calls.push(["build", blockInput, blockOutput]);
      return built;
    },
    onnxsim_lora_write_back(bytes, handle, finalState) {
      calls.push(["write_back", handle, finalState]);
      return new Uint8Array([100]);
    },
    onnxsim_lora_release_plan(handle) {
      calls.push(["release", handle]);
      return true;
    },
  };
  const graph = fakeStepGraph(built.state);
  const ort = fakeOrt({});
  ort.InferenceSession.create = async (bytes) => {
    ort.seen.created.push({ bytes });
    if (bytes.length === 2 && bytes[0] === 9) return { inputNames: [], outputNames: [], run: graph.runStep };
    // the injected model -- X is a real graph input, read straight from
    // rowFeeds, so this is only ever asked for its input names.
    return { inputNames: ["X"], outputNames: [], async run() { return {}; } };
  };

  const result = await trainLoraAdapterAllBlocks(runtime, {
    baseBytes: new Uint8Array([1]),
    maxTargetsPerBlock: 1,
    targetData: [Float32Array.of(3)],
    rowFeeds: [{ X: tensor([1, 1], [2]) }],
    numSteps: 1,
    ortLoader: async () => ort,
  });

  assert.equal(result.results.length, 1);
  assert.ok(result.results[0].trained);
  // Exactly two sessions: the injected model and the step graph -- no
  // reference-model session, since there is no reference bytes anywhere in
  // this call.
  assert.equal(ort.seen.created.length, 2);
  assert.deepEqual([...graph.seen[0].lora__teacher.data], [3]);
  assert.deepEqual(calls.map((c) => c[0]), ["build", "write_back", "release"]);
});

console.log(`\nlora_finetune: ${passed} checks passed`);
