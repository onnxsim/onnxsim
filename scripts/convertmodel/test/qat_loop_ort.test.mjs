// Does the QAT panel's own training loop reproduce Python's trajectory?
//
// qat_finetune.test.mjs checks the loop's mechanics against fakes; this checks
// them against the real thing. The fixtures are the step graphs onnxsim's own
// builders emit (test/step_graphs.json + the four .onnx files beside it, made
// by make_step_graph_fixtures.py) together with the per-step losses
// onnxruntime's CPU provider produced for the same feeds in Python. Driving
// them through qat_finetune.mjs's runStepLoop on onnxruntime-web and landing
// on the same losses is what says the browser half of the flow -- bind the
// captures once, feed the scalars fresh, carry every state output back into
// its state input -- agrees with qat_graph.run_step_graph rather than merely
// running without error.
//
// It is a strong check precisely because getting the loop subtly wrong does
// not throw. A state output not carried back makes every step start from the
// initial weights again; a constant rebound per step, or a scalar fed one step
// late, changes the trajectory and nothing else. All three produce finite,
// plausible, wrong losses -- and all three would move these numbers.
//
// This is the *loop*, not the panel: no wasm module is involved, so nothing
// here covers onnxsim_qat_build_step_graph or _write_back (qat_step_graph.test.mjs
// owns those, and needs a built onnxsim.wasm). It runs on onnxruntime-web's
// wasm backend, the only one a headless Node container has -- step_graph_ep.test.mjs
// is where the accelerated providers are attempted and reported.
//
// Usage:
//   node test/qat_loop_ort.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

globalThis.document = { getElementById: () => null, querySelector: () => null };
globalThis.window = { addEventListener: () => {} };
const { runStepLoop } = await import("../qat_finetune.mjs");

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = join(HERE, "..", "node_modules", "onnxruntime-web", "dist") + "/";
const MANIFEST = JSON.parse(readFileSync(join(HERE, "step_graphs.json"), "utf8"));

// onnxruntime-web's CPU kernels and onnxruntime's are the same C++ code, so the
// two trajectories agree to float32 round-off. Same tolerance and same reason
// as step_graph_ep.test.mjs, which compares against these very numbers.
const RTOL = 1e-3;
const ATOL = 1e-9;

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

const ort = await import("onnxruntime-web").then((m) => {
  const mod = m.default ?? m;
  // Local wasm artifacts, single-threaded, no worker proxy: the offline
  // configuration every onnxruntime-web test here uses.
  mod.env.wasm.wasmPaths = DIST;
  mod.env.wasm.numThreads = 1;
  mod.env.wasm.proxy = false;
  return mod;
});

const floatTensor = (spec) => new ort.Tensor("float32", Float32Array.from(spec.data), spec.dims);

// Drive one fixture through the panel's own runStepLoop.
//
// The mapping is the whole point of the test, so it is spelled out rather than
// helper-ed away: a fixture's `constants` are what the panel binds as captures
// (bound once, for the life of the loop), its `state` map is the ping-pong the
// plan's `state` pairs describe, and its recorded `scalars`/`perStep` stand in
// for the schedule the panel computes -- injected here so what is under test
// is the loop and not stepScalars, which qat_finetune.test.mjs pins against
// qat_graph.adam_bias_corrections separately.
async function runFixture(graph) {
  const model = new Uint8Array(readFileSync(join(HERE, graph.file)));
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "disabled",
  });

  const captures = {};
  for (const [name, spec] of Object.entries(graph.constants)) captures[name] = floatTensor(spec);
  const initialState = {};
  const state = [];
  for (const [name, spec] of Object.entries(graph.state)) {
    initialState[name] = floatTensor(spec);
    state.push({ input: name, output: spec.output });
  }
  const perStep = graph.perStep || [];
  const rowInput = perStep.length ? Object.keys(perStep[0])[0] : null;

  const result = await runStepLoop({
    state,
    scalars: Object.keys(graph.scalars[0]),
    loss: graph.loss,
    initialState,
    captures,
    rowIndex: rowInput
      ? {
          input: rowInput,
          // The fixture's own row draw, replayed rather than redrawn: the
          // browser's minibatchIndices cannot reproduce numpy's PCG64
          // permutation, and what is being compared here is the loop.
          rows: (t) => BigInt64Array.from(perStep[t][rowInput].data.map(BigInt)),
        }
      : null,
    numSteps: graph.scalars.length,
    scalarValues: (t) => graph.scalars[t],
    runStep: (feeds) => session.run(feeds),
    makeScalar: (value) => new ort.Tensor("float32", Float32Array.from([value]), []),
    makeRowIndex: (rows) => new ort.Tensor("int64", rows, [rows.length]),
  });
  if (typeof session.release === "function") await session.release();
  return result;
}

for (const graph of MANIFEST.graphs) {
  await check(`${graph.name}: the loop tracks onnxruntime's Python trajectory`, async () => {
    const { state, losses } = await runFixture(graph);
    assert.equal(losses.length, graph.referenceLosses.length);
    losses.forEach((got, t) => {
      const want = graph.referenceLosses[t];
      assert.ok(
        Math.abs(got - want) <= ATOL + RTOL * Math.abs(want),
        `${graph.name} step ${t}: loss ${got} against Python's ${want}`,
      );
    });
    // The loss moving is what says the state really went round the loop: a
    // loop that re-fed the initial state every step would report the first
    // step's loss four times.
    assert.notEqual(losses[0], losses[losses.length - 1]);
    // Every state input the plan declares comes back, ready for the write-back.
    assert.deepEqual(
      Object.keys(state).sort(),
      Object.keys(graph.state).sort(),
      "the final state must cover every state input",
    );
  });
}

// The negative control for the check above: with the state dropped between
// steps -- exactly the bug that does not throw -- the trajectory is flat and
// visibly different, so the agreement above is evidence and not a coincidence.
await check("a loop that does not carry its state diverges", async () => {
  const graph = MANIFEST.graphs.find((g) => g.name === "qat_backward");
  const model = new Uint8Array(readFileSync(join(HERE, graph.file)));
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "disabled",
  });
  const feeds = {};
  for (const [name, spec] of Object.entries(graph.constants)) feeds[name] = floatTensor(spec);
  for (const [name, spec] of Object.entries(graph.state)) feeds[name] = floatTensor(spec);
  const stuck = [];
  for (let t = 0; t < graph.scalars.length; t++) {
    for (const [name, value] of Object.entries(graph.scalars[t])) {
      feeds[name] = new ort.Tensor("float32", Float32Array.from([value]), []);
    }
    const out = await session.run(feeds);
    stuck.push(Number(out[graph.loss].data[0]));
  }
  if (typeof session.release === "function") await session.release();
  assert.equal(stuck[0], graph.referenceLosses[0], "step 0 is the same either way");
  assert.ok(
    Math.abs(stuck[3] - graph.referenceLosses[3]) > RTOL * Math.abs(graph.referenceLosses[3]),
    "a stuck loop should not reach Python's step-3 loss",
  );
});

console.log(`\nqat loop (onnxruntime-web): ${passed} checks passed`);
