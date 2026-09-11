// Does a federated round's LoRA step graph actually train when run through
// the *official* onnxruntime-web package -- no custom WASM build, no
// onnxruntime.training -- and does its exported state round-trip through
// the Python-side FedAvg aggregator (../../scripts/federated_lora_aggregate.py,
// itself a thin wrapper over onnxsim.federated -- see tests/test_federated.py
// for that module's own math being exercised in-process) once it leaves the
// browser?
//
// Skips (does not fail) if onnxruntime-web is not installed here, matching
// ../distill_step_graph/step_graph_runner.test.mjs's own convention --
// `npm install` in this directory to run it for real. The Python-side
// pieces (the generator and the aggregator) need onnxsim's own C++
// extension built (see the repo's CLAUDE.md) -- this test does not attempt
// to build it, only to run it.
//
// Usage:
//   npm install && npm test
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { StepGraphSession, adamBiasCorrections, exportTrainedState, loadInitialState, parseManifest } from "./step_graph_runner.mjs";

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

// A tiny xorshift PRNG so each simulated client's local data is reproducible
// without pulling in a dependency just for this test.
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

async function trainOneClient(ort, stepGraphBytes, initialStateBytes, manifest, seed, numSteps) {
  const state = loadInitialState(
    manifest,
    initialStateBytes.buffer.slice(initialStateBytes.byteOffset, initialStateBytes.byteOffset + initialStateBytes.byteLength),
  );
  const session = await StepGraphSession.create(ort, stepGraphBytes);

  // One fixed local batch/target for the whole local run -- this client's
  // own private data, standing in for whatever it would really train on.
  // Different clients get different (seed-keyed) data, the non-IID case
  // FedAvg is meant to handle rather than two shards of one shared set.
  const rng = makeRng(seed);
  const inputSize = manifest.inputShape.reduce((a, b) => a * b, 1);
  const targetSize = manifest.teacherShape.reduce((a, b) => a * b, 1);
  const batchInput = Float32Array.from({ length: inputSize }, () => rng() * 2 - 1);
  const target = Float32Array.from({ length: targetSize }, () => rng() * 2 - 1);

  let current = state;
  const losses = [];
  for (let t = 0; t < numSteps; ++t) {
    const { loss, state: nextState } = await session.step(manifest, current, batchInput, target, 1e-2, t);
    assert.ok(Number.isFinite(loss), `loss is finite at local step ${t}`);
    losses.push(loss);
    current = nextState;
  }
  return { losses, trainedState: current };
}

test("federated LoRA step graph trains locally on plain onnxruntime-web and its exported state feeds the Python FedAvg aggregator", { skip: ort === null && "onnxruntime-web not installed -- npm install to run this test" }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "onnx-finetune-federated-lora-"));
  const modelPath = join(dir, "toy.onnx");
  const stepPath = join(dir, "step.onnx");

  runPython("make_toy_model.py", [
    "-o", modelPath, "--input-dim", "6", "--hidden-dim", "8", "--output-dim", "6",
  ]);
  runPython("generate_federated_lora_step_graph.py", [
    modelPath, "-o", stepPath, "--batch-size", "4", "--rank", "2",
  ]);

  const manifest = parseManifest(readFileSync(`${stepPath}.manifest.txt`, "utf8"));
  assert.deepEqual(manifest.inputShape, [4, 6]);
  assert.deepEqual(manifest.teacherShape, [4, 6]);
  assert.ok(manifest.weights.length > 0, "at least one LoRA adapter was injected");

  const stepGraphBytes = new Uint8Array(readFileSync(stepPath));
  const initialStateBytes = readFileSync(`${stepPath}.initial_state.bin`);

  // Two independent clients, same broadcast starting state, different local
  // data -- exactly onnxsim.federated.run_federated_round's contract on the
  // Python side, here exercised with a real browser-side runtime instead of
  // an in-process train_lora call.
  const clientA = await trainOneClient(ort, stepGraphBytes, initialStateBytes, manifest, 1, 150);
  const clientB = await trainOneClient(ort, stepGraphBytes, initialStateBytes, manifest, 2, 150);

  for (const { losses } of [clientA, clientB]) {
    assert.ok(losses[losses.length - 1] < 0.3 * losses[0], `local loss should drop sharply: ${losses[0]} -> ${losses[losses.length - 1]}`);
  }

  const exportedA = exportTrainedState(manifest, clientA.trainedState);
  const exportedB = exportTrainedState(manifest, clientB.trainedState);
  const expectedLength = manifest.weights.reduce((n, { shape }) => n + shape.reduce((a, b) => a * b, 1), 0);
  assert.equal(exportedA.length, expectedLength);
  assert.equal(exportedB.length, expectedLength);
  // Two clients that trained on different local data must not export
  // identical adapters -- otherwise this "test" would pass even if step()
  // silently ignored batchInput/target.
  assert.notDeepEqual(Array.from(exportedA), Array.from(exportedB));

  const clientAPath = join(dir, "client_a.bin");
  const clientBPath = join(dir, "client_b.bin");
  writeFileSync(clientAPath, Buffer.from(exportedA.buffer, exportedA.byteOffset, exportedA.byteLength));
  writeFileSync(clientBPath, Buffer.from(exportedB.buffer, exportedB.byteOffset, exportedB.byteLength));

  const roundOutputPath = join(dir, "round1.base_model.onnx");
  runPython("federated_lora_aggregate.py", [
    "--manifest", `${stepPath}.manifest.txt`,
    "--base-model", `${stepPath}.base_model.onnx`,
    "--client", clientAPath, "--client", clientBPath,
    "--weight", "1", "--weight", "1",
    "-o", roundOutputPath,
  ]);

  assert.ok(statSync(roundOutputPath).size > 0, "the aggregator wrote a next-round base model");
  assert.ok(statSync(`${roundOutputPath}.initial_state.bin`).size > 0, "the aggregator wrote a next-round initial state");
});

test("adamBiasCorrections matches the Python reference at a few steps", () => {
  const t0 = adamBiasCorrections(0);
  assert.ok(Math.abs(t0.mCorrection - 1 / (1 - 0.9)) < 1e-9);
  assert.ok(Math.abs(t0.vCorrection - 1 / (1 - 0.999)) < 1e-9);
});

test("parseManifest/loadInitialState/exportTrainedState round-trip a synthetic manifest", () => {
  const manifest = parseManifest(
    "input_name X\ninput_shape 2 3\nteacher_name Y\nteacher_shape 2 3\nlr_name lr\nloss_name loss\n" +
    "state A a_out 2 2\nweight A 2 2\n",
  );
  const initial = new Float32Array([1, 2, 3, 4]);
  const state = loadInitialState(manifest, initial.buffer);
  assert.deepEqual(Array.from(state.A), [1, 2, 3, 4]);
  const exported = exportTrainedState(manifest, state);
  assert.deepEqual(Array.from(exported), [1, 2, 3, 4]);
});
