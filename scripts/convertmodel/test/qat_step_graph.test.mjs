// Does the QAT step-graph builder actually cross into JavaScript?
//
// onnxsim/qat_entry.h's BuildQatStepGraph / WriteBackQatState are the two
// entry points that make block-wise QAT usable without Python: the first turns
// (float model, quantized model, block boundary) into one optimizer step
// expressed as a plain inference graph, the second folds a finished loop's
// state back into the quantized model. Everything in between -- bind the
// captures, run the step graph n times, carry each state output back to its
// state input -- is the page's job, because the page already has an inference
// runtime (onnxruntime-web) and running a graph in a loop is that runtime's
// job. This test covers the two ends, which are the parts that live in wasm.
//
// What it checks is that the *contract* the returned object states is true of
// the graph it returns: every state input, every capture, every per-step
// scalar and the minibatch row index are really graph inputs; every state
// "next" output and the loss are really graph outputs; each initial-state
// tensor carries as many float32 bytes as its own dims say. A caller that
// drives the loop from that object cannot then discover a name that isn't
// there. The numerics are *not* checked here -- qat_graph_parity_test and the
// python QAT tests own that, against fixtures, and duplicating it against a
// hand-written expectation here would only be a second, weaker claim.
//
// The models are built in-process rather than committed: onnxsim_parse_graph
// turns the text form below into the float model and
// onnxsim_quantize_weight_only_int4 -- calibration-free, so no data and no ORT
// -- makes the quantized one. So there is no .onnx fixture to regenerate, and
// nothing to un-ignore in .gitignore.
//
// Needs the wasm module built and staged next to the page
// (scripts/convertmodel/onnxsim.js + onnxsim.wasm, e.g. `./build_wasm.sh` then
// `cp build-wasm-node-OFF/onnxsim.* scripts/convertmodel`, which is what
// .github/workflows/static.yml does). Without it the test SKIPS, since the
// convertmodel test job builds no wasm; QAT_REQUIRE_WASM=1 turns that skip
// into a failure for a job that does build it.
//
// Usage:
//   node test/qat_step_graph.test.mjs
//   QAT_REQUIRE_WASM=1 node test/qat_step_graph.test.mjs   # skip => failure

import assert from "node:assert/strict";
import { copyFileSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { fields, decode } from "../macs.mjs";
import { readShapes } from "../shapes.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const MODULE_JS = join(HERE, "..", "onnxsim.js");
const MODULE_WASM = join(HERE, "..", "onnxsim.wasm");

// How many calibration rows the loop will bind. The fixture's own batch axis
// is fixed at this same number, so the fixture says one thing whether the
// builder takes the row count from `num_rows` or from the model's own shapes.
const NUM_ROWS = 4;
// MatMul weight [K, N]. K must be a multiple of 32 for
// quantize_weight_only_int4 to touch the layer at all (see QuantizeWeightOnlyInt4
// in onnxsim/onnxsim.h) -- a block with no quantized layer in it is an error
// from BuildQatStepGraph, not a no-op, so the fixture has to earn its layer.
const K = 32;
const N = 2;

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// ---------------------------------------------------------------------------
// Loading the module under Node.
//
// onnxsim.js is Emscripten's MODULARIZE output (EXPORT_NAME=create_onnxsim, see
// CMakeLists.txt), i.e. CommonJS -- but this package is "type": "module", so
// Node would parse a `.js` here as ESM and find nothing exported. The npm
// package hits the same wall and answers it by staging the file as
// `onnxsim.cjs` (scripts/stage_npm_package.sh); this does the same thing for
// the duration of the test, in a temp directory, with `locateFile` pointing
// the loader back at the real onnxsim.wasm next to the page.
function loadRuntime() {
  const dir = mkdtempSync(join(tmpdir(), "onnxsim-qat-"));
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
// The fixture: one MatMul, whose weight is exactly what INT4 weight-only
// quantization takes and therefore exactly what QAT then trains.
function floatModelText() {
  // Deterministic, sign-varying, and not so uniform that every 32-element
  // block gets the same scale. Spelled into the text because it is small;
  // anything larger would belong in a committed initializer instead.
  const weights = [];
  for (let i = 0; i < K * N; i++) {
    weights.push((Math.sin(i * 0.7) * 0.5).toFixed(6));
  }
  return `
<
  ir_version: 10,
  opset_import: ["" : 21]
>
qat_fixture (float[${NUM_ROWS}, ${K}] X) => (float[${NUM_ROWS}, ${N}] Y)
<float[${K}, ${N}] W = {${weights.join(", ")}}>
{
  Y = MatMul(X, W)
}
`;
}

// Every binding that returns model bytes returns a view into the wasm heap
// that the next call invalidates -- copy before calling back in, exactly as
// quantize_calibration.mjs and quantize_ui.mjs do.
function copyBytes(view) {
  return new Uint8Array(view).slice();
}

function names(valueInfos) {
  return valueInfos.map((v) => v.name);
}

// The op types of a model's nodes, read straight out of the bytes with the
// same dependency-free wire reader shapes.mjs uses (ModelProto.graph = 7,
// GraphProto.node = 1, NodeProto.op_type = 4). A substring search over the
// model bytes would do for a name as long as "DequantizeLinear", but not for
// the three-letter op names below -- "Abs" would match three bytes of a
// weight as happily as an op type.
function opTypes(modelBytes) {
  const buf = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  const ops = new Set();
  for (const model of fields(buf)) {
    if (model.field !== 7 || model.wire !== 2) continue;
    for (const graph of fields(model.bytes)) {
      if (graph.field !== 1 || graph.wire !== 2) continue;
      for (const node of fields(graph.bytes)) {
        if (node.field === 4 && node.wire === 2) ops.add(decode(node.bytes));
      }
    }
  }
  return ops;
}

// ---------------------------------------------------------------------------

async function main() {
  if (!existsSync(MODULE_JS) || !existsSync(MODULE_WASM)) {
    const why =
      `scripts/convertmodel/onnxsim.{js,wasm} not found -- build the wasm module ` +
      `(./build_wasm.sh) and copy it next to the page to run this test`;
    if (process.env.QAT_REQUIRE_WASM) {
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
    // The loader has read the file by now; the wasm itself is loaded from the
    // page's own directory via locateFile, so the temp copy is disposable.
    rmSync(loaded.dir, { recursive: true, force: true });
  }

  // --- the two models -------------------------------------------------------
  const parsed = runtime.onnxsim_parse_graph(floatModelText());
  assert.equal(parsed.error, undefined, `fixture did not parse: ${parsed.error}`);
  const floatModel = copyBytes(parsed.model);

  const quantizedView = runtime.onnxsim_quantize_weight_only_int4(floatModel);
  assert.ok(quantizedView, "onnxsim_quantize_weight_only_int4 returned null");
  const quantized = copyBytes(quantizedView);
  await check("the fixture's MatMul really was INT4-quantized", () => {
    // A DequantizeLinear replaced the float weight, so the op type is now in
    // the model's bytes and was not before. Cheaper and less brittle here than
    // decoding the whole graph: what matters is only that the block has a
    // quantized layer for QAT to train.
    const text = Buffer.from(quantized).toString("latin1");
    assert.ok(text.includes("DequantizeLinear"), "no DequantizeLinear in the quantized model");
    assert.ok(!Buffer.from(floatModel).toString("latin1").includes("DequantizeLinear"));
  });

  // --- build the step graph -------------------------------------------------
  const built = runtime.onnxsim_qat_build_step_graph(
    floatModel,
    quantized,
    "X",
    "Y",
    NUM_ROWS,
    {},
  );
  assert.ok(built, "onnxsim_qat_build_step_graph returned null (see the log above)");

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
    assert.ok(state.length > 0, "a trained block has state; this one reported none");
    for (const { input, output } of state) {
      assert.ok(graphInputs.has(input), `state input '${input}' is not a graph input`);
      assert.ok(graphOutputs.has(output), `state output '${output}' is not a graph output`);
      assert.notEqual(input, output, "a state input and its next value must be distinct");
    }
    const inputs = state.map((s) => s.input);
    assert.equal(new Set(inputs).size, inputs.length, "a state input appears twice");
  });

  await check("every per-step scalar is a graph input", () => {
    // qat.py's own set: the learning rate plus Adam's two bias-correction
    // factors, with the scale/activation rates added only when they are trained
    // (they are not here, so exactly three).
    assert.equal(scalars.length, 3, `unexpected scalars: ${scalars.join(", ")}`);
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
      // NUM_ROWS rows of whatever the tensor is: this is the shape the caller
      // has to produce from the float model, so a wrong leading dim here is a
      // caller that binds the wrong buffer.
      assert.equal(capture.dims[0], NUM_ROWS, `capture '${capture.input}' has ${capture.dims}`);
    }
    const teachers = captures.filter((c) => c.teacher);
    assert.equal(teachers.length, 1, "exactly one capture is the reconstruction target");
    // The block's output is what the teacher is captured from.
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
      assert.equal(tensor.dtype, 1, "TensorProto.FLOAT"); // onnx.TensorProto.DataType.FLOAT
      const count = tensor.dims.reduce((a, b) => a * b, 1);
      assert.equal(
        tensor.data.length,
        count * 4,
        `'${tensor.name}': ${tensor.data.length} bytes for dims ${tensor.dims}`,
      );
    }
  });

  // --- write the state back -------------------------------------------------
  //
  // Writing the *initial* state back is a legitimate round trip: it is the
  // untrained state, so what comes out is the model a zero-step run produces.
  // What is under test is the marshaling -- that the loop's tensors, in the
  // shape onnxruntime-web hands them back ({ dims, data }), reach
  // WriteBackQatState under the plan the build returned.
  await check("the trained state writes back into the quantized model", () => {
    const finalState = {};
    for (const tensor of initialState) {
      finalState[tensor.name] = {
        dims: tensor.dims,
        // A fresh copy, so byteOffset is 0 and the Float32Array view is aligned.
        data: new Float32Array(tensor.data.buffer),
      };
    }
    const written = runtime.onnxsim_qat_write_back(quantized, planHandle, finalState);
    assert.ok(written, "onnxsim_qat_write_back returned null (see the log above)");
    const out = copyBytes(written);
    // The write-back rewrites this block's initializers and nothing else, so
    // the model's own interface is untouched.
    const before = readShapes(quantized);
    const after = readShapes(out);
    assert.deepEqual(names(after.inputs), names(before.inputs));
    assert.deepEqual(names(after.outputs), names(before.outputs));
  });

  await check("a released plan is gone, and a write-back against it fails", () => {
    assert.equal(runtime.onnxsim_qat_release_plan(planHandle), true);
    assert.equal(runtime.onnxsim_qat_release_plan(planHandle), false);
    assert.equal(runtime.onnxsim_qat_write_back(quantized, planHandle, {}), null);
  });

  // --- the minibatch variant ------------------------------------------------
  await check("batchSize adds an int64 row index and resident capture tables", () => {
    const batched = runtime.onnxsim_qat_build_step_graph(
      floatModel,
      quantized,
      "X",
      "Y",
      NUM_ROWS,
      { batchSize: 2, batchSeed: 1, shuffle: true },
    );
    assert.ok(batched, "onnxsim_qat_build_step_graph returned null for a minibatch");
    const rowIndex = batched.rowIndexInput;
    const rowIndexSize = batched.rowIndexSize;
    const batchedCaptures = batched.captures.map((c) => ({
      input: c.input,
      source: c.source,
      dims: [...c.dims],
    }));
    const batchedInputs = new Set(names(readShapes(copyBytes(batched.stepGraph)).inputs));
    runtime.onnxsim_qat_release_plan(batched.planHandle);

    assert.ok(rowIndex, "a minibatch run needs a row-index input");
    assert.equal(rowIndexSize, 2);
    assert.ok(batchedInputs.has(rowIndex), `'${rowIndex}' is not a graph input`);
    for (const capture of batchedCaptures) {
      // The whole calibration set stays resident under a renamed input and the
      // step gathers its rows out of it -- so the capture's step-graph input is
      // NOT its source tensor's name here, and it still holds all NUM_ROWS rows.
      assert.ok(batchedInputs.has(capture.input), `capture '${capture.input}' is not an input`);
      assert.notEqual(capture.input, capture.source);
      assert.equal(capture.dims[0], NUM_ROWS);
    }
  });

  // --- the fine-tuning variant ----------------------------------------------
  //
  // fakeQuant:false takes the quantizer out of the middle and trains the
  // *second* model's own float weights, so the student here is the float model
  // rather than the quantized one -- the quantized model's MatMul reads a
  // DequantizeLinear's output, not a stored weight, and would have nothing to
  // fine-tune. Both models being the same one makes for a zero loss and
  // nothing to learn, which is fine: what is under test is the graph the
  // builder returns, and the numerics belong to the python tests.
  await check("fakeQuant:false builds a step graph with no fake-quant in it", () => {
    const tuned = runtime.onnxsim_qat_build_step_graph(
      floatModel,
      floatModel,
      "X",
      "Y",
      NUM_ROWS,
      { fakeQuant: false },
    );
    assert.ok(tuned, "onnxsim_qat_build_step_graph returned null for fakeQuant:false");
    const graph = copyBytes(tuned.stepGraph);
    const tunedState = tuned.state.map((s) => ({ input: s.input, output: s.output }));
    const tunedScalars = [...tuned.scalars];
    const tunedInputs = new Set(names(readShapes(graph).inputs));
    runtime.onnxsim_qat_release_plan(tuned.planHandle);

    assert.equal(tunedState.length, 3, "one weight plus Adam's two moments");
    assert.equal(tunedScalars.length, 3, "no scale learning rate to feed");
    for (const { input } of tunedState) {
      assert.ok(tunedInputs.has(input), `state input '${input}' is not a graph input`);
    }
    // The weight fake-quant is Abs/Sign (GraphBuilder's RoundToNearest) and
    // Clip; none of the three has any other reason to be in this graph, so
    // their absence is what "the quantizer is gone" looks like from out here.
    const ops = opTypes(graph);
    assert.ok(ops.size > 0, "no nodes -- the step graph bytes did not parse");
    for (const op of ["Abs", "Sign", "Clip", "Round"]) {
      assert.ok(!ops.has(op), `the fine-tuning step graph contains a ${op}`);
    }
    assert.ok(ops.has("MatMul"), "the block's own node is still there");
  });

  await check("fakeQuant:false refuses the two scale flags rather than ignoring them", () => {
    // Both name a parameter of a quantizer, and this is the mode with no
    // quantizer in it. BuildQatStepGraph throws; the binding turns that into
    // null with the reason on the console.
    assert.equal(
      runtime.onnxsim_qat_build_step_graph(floatModel, floatModel, "X", "Y", NUM_ROWS, {
        fakeQuant: false,
        learnScales: true,
      }),
      null,
    );
  });

  // --- refusals -------------------------------------------------------------
  await check("a block with no quantized layer is refused, not silently empty", () => {
    // X -> X is not a block at all: nothing between the two names.
    assert.equal(
      runtime.onnxsim_qat_build_step_graph(floatModel, quantized, "X", "X", NUM_ROWS, {}),
      null,
    );
  });

  await check("a model that will not parse is refused", () => {
    const junk = new Uint8Array([1, 2, 3]);
    assert.equal(
      runtime.onnxsim_qat_build_step_graph(junk, quantized, "X", "Y", NUM_ROWS, {}),
      null,
    );
  });

  console.log(`\nqat step graph: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
