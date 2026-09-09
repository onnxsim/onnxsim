// Does the LoRA step-graph builder actually cross into JavaScript?
//
// onnxsim/lora_entry.h's InjectLora / BuildLoraStepGraph / WriteBackLoraState
// are the three entry points that make LoRA/QLoRA training usable without
// Python -- the first turns a base model into an injected one plus a
// LoraAdapter, the second turns (injected model, adapter, block boundary)
// into one optimizer step expressed as a plain inference graph, the third
// folds a finished loop's state back into the injected model. Everything in
// between -- bind the captures, run the step graph n times, carry each state
// output back to its state input -- is the page's job (lora_finetune.mjs),
// because the page already has an inference runtime and running a graph in a
// loop is that runtime's. This test covers the wasm ends, mirroring
// qat_step_graph.test.mjs's own shape and level of coverage exactly (see
// that file's own top comment for the fuller rationale, which applies here
// unchanged); the differences below are LoRA's own.
//
// What it checks is that the *contract* the returned object states is true
// of the graph it returns: every state input, every capture, every per-step
// scalar and the minibatch row index are really graph inputs; every state
// "next" output and the loss are really graph outputs; each initial-state
// tensor carries as many float32 bytes as its own dims say; `parameters`
// names real adapter initializers. The numerics are *not* checked here --
// tests/test_lora.py owns those, against torch.autograd (see lora_entry.h's
// own top comment on why no LoRA-specific parity fixture is added at the
// primitive level either).
//
// Needs the wasm module built and staged next to the page
// (scripts/convertmodel/onnxsim.js + onnxsim.wasm), exactly as
// qat_step_graph.test.mjs does. Without it the test SKIPS.
// LORA_REQUIRE_WASM=1 turns that skip into a failure for a job that does
// build it.
//
// Usage:
//   node test/lora_step_graph.test.mjs
//   LORA_REQUIRE_WASM=1 node test/lora_step_graph.test.mjs   # skip => failure

import assert from "node:assert/strict";
import { copyFileSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { readShapes } from "../shapes.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const MODULE_JS = join(HERE, "..", "onnxsim.js");
const MODULE_WASM = join(HERE, "..", "onnxsim.wasm");

// How many calibration rows the loop will bind.
const NUM_ROWS = 4;
// MatMul weight [K, N]. Any 2-D float32 initializer is eligible for LoRA
// (unlike quantize_weight_only_int4's K-multiple-of-32 requirement, so this
// can stay small).
const K = 4;
const N = 2;
const RANK = 2;

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// ---------------------------------------------------------------------------
// Loading the module under Node -- identical to qat_step_graph.test.mjs's own
// loadRuntime; see that function's comment for why the .js is staged as
// .cjs.
function loadRuntime() {
  const dir = mkdtempSync(join(tmpdir(), "onnxsim-lora-"));
  const cjs = join(dir, "onnxsim.cjs");
  copyFileSync(MODULE_JS, cjs);
  const require = createRequire(import.meta.url);
  const createOnnxsim = require(cjs);
  return {
    dir,
    module: createOnnxsim({
      locateFile: (path) => (path.endsWith(".wasm") ? MODULE_WASM : join(HERE, "..", path)),
      print: (s) => console.log("    | " + s),
      printErr: (s) => console.log("    ! " + s),
    }),
  };
}

// ---------------------------------------------------------------------------
// The fixture: one MatMul, whose weight is exactly what InjectLora targets by
// default (a 2-D float32 initializer at input[1]).
function baseModelText() {
  const weights = [];
  for (let i = 0; i < K * N; i++) {
    weights.push((Math.sin(i * 0.7) * 0.5).toFixed(6));
  }
  return `
<
  ir_version: 10,
  opset_import: ["" : 21]
>
lora_fixture (float[${NUM_ROWS}, ${K}] X) => (float[${NUM_ROWS}, ${N}] Y)
<float[${K}, ${N}] W = {${weights.join(", ")}}>
{
  Y = MatMul(X, W)
}
`;
}

function copyBytes(view) {
  return new Uint8Array(view).slice();
}

function names(valueInfos) {
  return valueInfos.map((v) => v.name);
}

// ---------------------------------------------------------------------------

async function main() {
  if (!existsSync(MODULE_JS) || !existsSync(MODULE_WASM)) {
    const why =
      `scripts/convertmodel/onnxsim.{js,wasm} not found -- build the wasm module ` +
      `(./build_wasm.sh) and copy it next to the page to run this test`;
    if (process.env.LORA_REQUIRE_WASM) {
      console.error("FAIL:", why);
      process.exit(1);
    }
    console.log("  skip - " + why);
    return;
  }

  const loaded = loadRuntime();
  let runtime;
  try {
    runtime = await loaded.module;
  } finally {
    rmSync(loaded.dir, { recursive: true, force: true });
  }

  // --- the base model, and injection -----------------------------------------
  const parsed = runtime.onnxsim_parse_graph(baseModelText());
  assert.equal(parsed.error, undefined, `fixture did not parse: ${parsed.error}`);
  const baseModel = copyBytes(parsed.model);

  const injected = runtime.onnxsim_lora_inject(baseModel, { rank: RANK });
  assert.ok(injected, "onnxsim_lora_inject returned null");
  const injectedModel = copyBytes(injected.model);
  const adapter = injected.adapter.map((t) => ({ ...t }));

  await check("injection adds exactly one target, at the fixture's own MatMul", () => {
    assert.equal(adapter.length, 1);
    assert.equal(adapter[0].weightName, "W");
    assert.equal(adapter[0].opType, "MatMul");
    assert.equal(adapter[0].rank, RANK);
    assert.equal(adapter[0].hasAlpha, false);
  });

  await check("the injected model still declares X and Y -- no interface change", () => {
    const before = readShapes(baseModel);
    const after = readShapes(injectedModel);
    assert.deepEqual(names(after.inputs), names(before.inputs));
    assert.deepEqual(names(after.outputs), names(before.outputs));
  });

  // --- build the step graph, over the whole model as one block ---------------
  const built = runtime.onnxsim_lora_build_step_graph(
    injectedModel,
    adapter,
    "X",
    "Y",
    NUM_ROWS,
    {},
  );
  assert.ok(built, "onnxsim_lora_build_step_graph returned null (see the log above)");

  const stepGraph = copyBytes(built.stepGraph);
  const state = built.state.map((s) => ({ input: s.input, output: s.output }));
  const scalars = [...built.scalars];
  const captures = built.captures.map((c) => ({
    input: c.input,
    source: c.source,
    dims: [...c.dims],
    teacher: c.teacher,
  }));
  const initialState = built.initialState.map((t) => ({
    name: t.name,
    dtype: t.dtype,
    dims: [...t.dims],
    data: copyBytes(t.data),
  }));
  const parameters = [...built.parameters];
  const planHandle = built.planHandle;

  const shapes = readShapes(stepGraph);
  const graphInputs = new Set(names(shapes.inputs));
  const graphOutputs = new Set(names(shapes.outputs));

  await check("the step graph parses and has inputs and outputs", () => {
    assert.ok(stepGraph.length > 0);
    assert.ok(graphInputs.size > 0, "no graph inputs -- the bytes did not parse as a model");
    assert.ok(graphOutputs.size > 0);
  });

  await check("every state pair names a real input and a real output", () => {
    // One weight (lora_A) plus its two Adam moments, times two adapter
    // tensors (lora_A and lora_B) -- six state entries for one target.
    assert.equal(state.length, 6, `unexpected state count: ${state.length}`);
    for (const { input, output } of state) {
      assert.ok(graphInputs.has(input), `state input '${input}' is not a graph input`);
      assert.ok(graphOutputs.has(output), `state output '${output}' is not a graph output`);
      assert.notEqual(input, output, "a state input and its next value must be distinct");
    }
    const inputs = state.map((s) => s.input);
    assert.equal(new Set(inputs).size, inputs.length, "a state input appears twice");
  });

  await check("parameters names the adapter's own lora_A/lora_B initializers", () => {
    assert.deepEqual(parameters.sort(), [adapter[0].loraAName, adapter[0].loraBName].sort());
    const stateInputs = new Set(state.map((s) => s.input));
    for (const name of parameters) {
      assert.ok(stateInputs.has(name), `parameter '${name}' has no matching state input`);
    }
  });

  await check("every per-step scalar is a graph input", () => {
    // LoraStepPlan's own set: "lora__lr" plus Adam's two bias-correction
    // factors -- no scale rates, unlike QAT.
    assert.deepEqual(
      [...scalars].sort(),
      ["lora__lr", "m_correction", "v_correction"].sort(),
    );
    for (const name of scalars) {
      assert.ok(graphInputs.has(name), `scalar '${name}' is not a graph input`);
    }
  });

  await check("the loss is a graph output", () => {
    assert.ok(built.loss, "no loss reported");
    assert.ok(graphOutputs.has(built.loss), `loss '${built.loss}' is not a graph output`);
  });

  await check("the captures are bindable and exactly one is the teacher", () => {
    assert.ok(captures.length > 0);
    for (const capture of captures) {
      assert.ok(
        graphInputs.has(capture.input),
        `capture '${capture.input}' is not a graph input`,
      );
      assert.ok(capture.source.length > 0, "a capture with no source tensor to read");
      assert.equal(capture.dims[0], NUM_ROWS, `capture '${capture.input}' has ${capture.dims}`);
    }
    const teachers = captures.filter((c) => c.teacher);
    assert.equal(teachers.length, 1, "exactly one capture is the reconstruction target");
    assert.equal(teachers[0].source, "Y");
  });

  await check("full batch: no minibatch row index", () => {
    assert.equal(built.rowIndexInput, "");
    assert.equal(built.rowIndexSize, 0);
    assert.equal(built.numRows, NUM_ROWS);
  });

  await check("each initial-state tensor is float32 and its own size", () => {
    assert.ok(initialState.length > 0);
    const stateInputs = new Set(state.map((s) => s.input));
    for (const tensor of initialState) {
      assert.ok(
        stateInputs.has(tensor.name),
        `initial state '${tensor.name}' matches no state input`,
      );
      assert.equal(tensor.dtype, 1, "TensorProto.FLOAT");
      const count = tensor.dims.reduce((a, b) => a * b, 1);
      assert.equal(
        tensor.data.length,
        count * 4,
        `'${tensor.name}': ${tensor.data.length} bytes for dims ${tensor.dims}`,
      );
    }
  });

  await check("lora_B's initial state is exactly zero -- injection is a numeric no-op", () => {
    // inject_lora's own contract (lora_entry.h's top comment): B always
    // starts at zero, so the injected branch adds nothing until trained.
    const b = initialState.find((t) => t.name === adapter[0].loraBName);
    assert.ok(b, "no initial state for lora_B");
    const values = new Float32Array(b.data.buffer, b.data.byteOffset, b.data.length / 4);
    for (const v of values) assert.equal(v, 0);
  });

  // --- write the state back ---------------------------------------------------
  await check("the trained state writes back into the injected model", () => {
    const finalState = {};
    for (const tensor of initialState) {
      finalState[tensor.name] = {
        dims: tensor.dims,
        data: new Float32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length / 4),
      };
    }
    const written = runtime.onnxsim_lora_write_back(injectedModel, planHandle, finalState);
    assert.ok(written, "onnxsim_lora_write_back returned null (see the log above)");
    const out = copyBytes(written);
    const before = readShapes(injectedModel);
    const after = readShapes(out);
    assert.deepEqual(names(after.inputs), names(before.inputs));
    assert.deepEqual(names(after.outputs), names(before.outputs));
  });

  await check("a released plan is gone, and a write-back against it fails", () => {
    assert.equal(runtime.onnxsim_lora_release_plan(planHandle), true);
    assert.equal(runtime.onnxsim_lora_release_plan(planHandle), false);
    assert.equal(runtime.onnxsim_lora_write_back(injectedModel, planHandle, {}), null);
  });

  // --- the minibatch variant ---------------------------------------------------
  await check("batchSize adds an int64 row index and resident capture tables", () => {
    const batched = runtime.onnxsim_lora_build_step_graph(
      injectedModel,
      adapter,
      "X",
      "Y",
      NUM_ROWS,
      { batchSize: 2, batchSeed: 1, shuffle: true },
    );
    assert.ok(batched, "onnxsim_lora_build_step_graph returned null for a minibatch");
    const rowIndex = batched.rowIndexInput;
    const rowIndexSize = batched.rowIndexSize;
    const batchedCaptures = batched.captures.map((c) => ({
      input: c.input,
      source: c.source,
      dims: [...c.dims],
    }));
    const batchedInputs = new Set(names(readShapes(copyBytes(batched.stepGraph)).inputs));
    runtime.onnxsim_lora_release_plan(batched.planHandle);

    assert.ok(rowIndex, "a minibatch run needs a row-index input");
    assert.equal(rowIndexSize, 2);
    assert.ok(batchedInputs.has(rowIndex), `'${rowIndex}' is not a graph input`);
    for (const capture of batchedCaptures) {
      assert.ok(batchedInputs.has(capture.input), `capture '${capture.input}' is not an input`);
      assert.notEqual(capture.input, capture.source);
      assert.equal(capture.dims[0], NUM_ROWS);
    }
  });

  // --- refusals -----------------------------------------------------------------
  await check("a block with nothing in it is refused, not silently empty", () => {
    // X -> X is not a block at all: nothing between the two names.
    assert.equal(
      runtime.onnxsim_lora_build_step_graph(injectedModel, adapter, "X", "X", NUM_ROWS, {}),
      null,
    );
  });

  await check("an adapter with no targets is refused", () => {
    assert.equal(
      runtime.onnxsim_lora_build_step_graph(injectedModel, [], "X", "Y", NUM_ROWS, {}),
      null,
    );
  });

  await check("a model that will not parse is refused", () => {
    const junk = new Uint8Array([1, 2, 3]);
    assert.equal(runtime.onnxsim_lora_inject(junk, {}), null);
    assert.equal(
      runtime.onnxsim_lora_build_step_graph(junk, adapter, "X", "Y", NUM_ROWS, {}),
      null,
    );
  });

  await check("restrictTargetNames:true with an empty targetNames injects nothing", () => {
    // The one place InjectLoraOptions distinguishes "absent" from "empty" --
    // see InjectLoraOptionsFromVal's own comment in interface.cpp.
    const nothing = runtime.onnxsim_lora_inject(baseModel, {
      restrictTargetNames: true,
      targetNames: [],
    });
    assert.ok(nothing, "onnxsim_lora_inject returned null for an empty restriction");
    assert.equal(nothing.adapter.length, 0);
  });

  console.log(`\nlora step graph: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
