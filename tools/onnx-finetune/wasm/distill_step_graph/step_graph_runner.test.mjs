// Does the graph_grad-based distillation step graph actually train when run
// through the *official* onnxruntime-web package -- no custom WASM build,
// no onnxruntime.training?
//
// Skips (does not fail) if onnxruntime-web is not installed here, matching
// this repo's existing convention for a wasm-module-dependent test (see
// lora_step_graph.test.mjs's own top comment) -- `npm install` in this
// directory to run it for real.
//
// Usage:
//   npm install && npm test
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { StepGraphSession, labelsToOnehot, loadInitialState, parseManifest } from "./step_graph_runner.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPTS = join(HERE, "..", "..", "scripts");

let ort;
try {
  ort = await import("onnxruntime-web");
} catch {
  ort = null;
}

function runPython(script, args) {
  execFileSync("python3", [join(SCRIPTS, script), ...args], { stdio: "inherit" });
}

// A tiny xorshift PRNG so the batch is reproducible without pulling in a
// dependency just for this test.
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

test("step graph trains on plain onnxruntime-web (no training API), across varying batch sizes", { skip: ort === null && "onnxruntime-web not installed -- npm install to run this test" }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "onnx-finetune-distill-"));
  try {
    const teacherPath = join(dir, "teacher.onnx");
    const studentPath = join(dir, "student.onnx");
    const stepPath = join(dir, "step.onnx");
    const inputDim = 8;
    const numClasses = 4;

    runPython("make_toy_classifier.py", [
      "-o", teacherPath, "--input-dim", String(inputDim), "--hidden-dim", "32",
      "--num-classes", String(numClasses), "--seed", "1",
    ]);
    runPython("make_toy_classifier.py", [
      "-o", studentPath, "--input-dim", String(inputDim), "--hidden-dim", "8",
      "--num-classes", String(numClasses), "--seed", "2",
    ]);
    // No --batch-size: the step graph's batch dimension is a dim_param,
    // decided per step below, not fixed when the graph is built.
    runPython("generate_distillation_step_graph.py", [studentPath, "-o", stepPath]);

    const manifest = parseManifest(readFileSync(`${stepPath}.manifest.txt`, "utf8"));
    assert.equal(manifest.inputShape[0], "batch");
    assert.equal(manifest.numClasses, numClasses);

    const stepGraphBytes = new Uint8Array(readFileSync(stepPath));
    const initialStateBytes = readFileSync(`${stepPath}.initial_state.bin`);
    let state = loadInitialState(
      manifest,
      initialStateBytes.buffer.slice(initialStateBytes.byteOffset, initialStateBytes.byteOffset + initialStateBytes.byteLength),
    );

    const session = await StepGraphSession.create(ort, stepGraphBytes);

    // A frozen, untrained-teacher-esque forward pass would need a second
    // onnxruntime-web session over teacher.onnx; not worth it for this
    // smoke test -- a fixed, deterministic "teacher logits" pool exercises
    // exactly the same step-graph code path (the step graph does not care
    // where teacher_logits came from) with far less test setup.
    const rng = makeRng(42);
    const poolSize = 64;
    const inputPool = Float32Array.from({ length: poolSize * inputDim }, () => rng() * 2 - 1);
    const teacherLogitsPool = Float32Array.from({ length: poolSize * numClasses }, () => rng() * 2 - 1);
    const labelsPool = Array.from({ length: poolSize }, () => Math.floor(rng() * numClasses));

    // Cycles through several sizes, including ones that don't evenly divide
    // poolSize -- proof one compiled step graph really does take an
    // arbitrary batch size, the same thing test_distillation_graph_grad.py's
    // test_step_graph_trains_on_plain_onnxruntime checks on the Python side.
    const batchSizes = [32, 8, 64, 1, 17];

    const losses = [];
    for (let t = 0; t < 50; ++t) {
      const batchSize = batchSizes[t % batchSizes.length];
      const indices = Array.from({ length: batchSize }, () => Math.floor(rng() * poolSize));
      const batchInput = Float32Array.from(
        { length: batchSize * inputDim },
        (_, i) => inputPool[indices[Math.floor(i / inputDim)] * inputDim + (i % inputDim)],
      );
      const teacherLogits = Float32Array.from(
        { length: batchSize * numClasses },
        (_, i) => teacherLogitsPool[indices[Math.floor(i / numClasses)] * numClasses + (i % numClasses)],
      );
      const labels = indices.map((i) => labelsPool[i]);

      const { loss, state: nextState } = await session.step(
        manifest, state, batchInput, teacherLogits, labels, 0.05, t,
      );
      assert.ok(Number.isFinite(loss), `loss is finite at step ${t} (batch size ${batchSize})`);
      losses.push(loss);
      state = nextState;
    }

    assert.ok(losses[losses.length - 1] < losses[0], `loss should decrease: ${losses[0]} -> ${losses[losses.length - 1]}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("labelsToOnehot matches the Python reference shape/values", () => {
  const onehot = labelsToOnehot([0, 2, 1], 4);
  assert.deepEqual(Array.from(onehot), [
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
  ]);
});
