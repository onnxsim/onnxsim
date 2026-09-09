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
function makeModel({ inputs = [], outputs = [] }) {
  const graph = [
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

await check("targetData is refused rather than silently ignored", async () => {
  await assert.rejects(
    trainLoraAdapter({}, {
      baseBytes: new Uint8Array([1]),
      targetData: [tensor([1, 1], [1])],
      rowFeeds: [{ X: tensor([1, 2], [1, 2]) }],
    }),
    /target_data training is not implemented/,
  );
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

console.log(`\nlora_finetune: ${passed} checks passed`);
