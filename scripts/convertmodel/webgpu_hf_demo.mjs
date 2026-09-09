// Shared driving logic for the two WebGPU + Hugging Face training demos --
// used by both the interactive "WebGPU training demo" panel
// (webgpu_demo_ui.mjs) and the CI checks
// (test/webgpu_hf_demo.test.mjs, test/webgpu_cifar10_pretrain.test.mjs),
// the same three-way split qat_finetune.mjs/qat_ui.mjs/qat_loop_ort.test.mjs
// already establish: one shared module holding the actual training loop, a
// thin DOM-coupled caller for the interactive page, and a thin
// Playwright-driven caller for CI.
//
// Deliberately takes an already-loaded `ort` and already-fetched
// `modelBytes`/`manifest` rather than loading them itself: the interactive
// page loads onnxruntime-web from a CDN (inference_browser.mjs's loadOrt)
// and fetches the step graph fixture from this page's own relative
// ./test/*.onnx path, while the CI tests load a local node_modules bundle
// through a throwaway static server -- this module doesn't need to know
// which, matching qat_finetune.mjs's fineTuneBlock(runtime, {floatBytes,
// quantBytes, ...}) taking bytes rather than fetching them.
//
// executionProviders: ["webgpu"] is a hard requirement in both demos below,
// same as their own top comments already explain in the CI test files: no
// wasm/CPU fallback in the list, so a host without WebGPU fails session
// creation outright with a clear error, rather than silently training
// somewhere else and calling it a WebGPU demo.

import { fetchSampleImageBytes, fetchCifar10Batch } from "./hf_datasets.mjs";
import { normalizePixels } from "./sample_inputs.mjs";

// Decodes `bytes` (JPEG/PNG) and resizes+grayscales it to a flat
// Float32Array of length side*side, via the same canvas primitives
// sample_inputs.mjs's own (unexported) image sample-data path uses for the
// "Run inference" panel, just always grayscale/square since both demos'
// step graphs declare a fixed flattened single-channel input.
async function decodeGraySquare(bytes, side) {
  const blob = new Blob([bytes]);
  const bitmap = await createImageBitmap(blob);
  const canvas =
    typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(side, side) : document.createElement("canvas");
  canvas.width = side;
  canvas.height = side;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, side, side);
  const rgba = ctx.getImageData(0, 0, side, side).data;
  return normalizePixels(rgba, 1, side, side);
}

// Runs the step loop for `manifest.scalars.length` steps against fixed
// `constants`, threading `manifest.state` through exactly as
// qat_graph.run_step_graph's feed-per-step path does. `onStep(t, loss)`, if
// given, is called after each step -- the interactive panel uses it to show
// live progress; nothing here needs the callback's return value.
async function runStepLoop(ort, session, manifest, constants, onStep) {
  let state = {};
  for (const [name, spec] of Object.entries(manifest.state)) {
    state[name] = new ort.Tensor("float32", Float32Array.from(spec.data), spec.dims);
  }
  const losses = [];
  let lastOutputs = null;
  for (let t = 0; t < manifest.scalars.length; t++) {
    const feeds = { ...constants, ...state };
    for (const [name, value] of Object.entries(manifest.scalars[t])) {
      feeds[name] = new ort.Tensor("float32", Float32Array.from([value]), []);
    }
    const out = await session.run(feeds);
    const next = {};
    for (const [name, spec] of Object.entries(manifest.state)) {
      next[name] = out[spec.output];
    }
    state = next;
    lastOutputs = out;
    const loss = Number(out[manifest.loss].data[0]);
    losses.push(loss);
    if (onStep) onStep(t, loss);
  }
  return { losses, lastOutputs };
}

// The "one real photo -> fixed target" demo: fetches a fresh photo from
// frgfm/imagenette every call, fits it toward a zero target for
// manifest.scalars.length steps. See test/make_step_graph_fixtures.py's
// build_qat_hf_demo for why the target is arbitrary and what this does and
// does not prove.
//
// Returns { losses, imageLabel }.
export async function runPhotoDemo({ ort, modelBytes, manifest, onStep }) {
  const { bytes, label } = await fetchSampleImageBytes();
  const side = Math.round(Math.sqrt(manifest.inputDim));
  const x = await decodeGraySquare(bytes, side);

  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "disabled",
    logSeverityLevel: 0,
    logVerbosityLevel: 4,
  });
  const constants = {
    x: new ort.Tensor("float32", x, [1, manifest.inputDim]),
    teacher: new ort.Tensor("float32", new Float32Array([0]), [1, 1]),
  };
  const { losses } = await runStepLoop(ort, session, manifest, constants, onStep);
  await session.release?.();
  return { losses, imageLabel: label };
}

// The CIFAR-10 pretraining demo: fetches a fixed batch of manifest.numSamples
// real, labeled photos from uoft-cs/cifar10 once, then trains on that same
// batch for manifest.scalars.length steps -- see
// test/make_step_graph_fixtures.py's build_qat_cifar10_pretrain for the
// architecture this drives.
//
// Returns { losses, correct, total, labelNames, predictions }, where
// `predictions`/`correct` are the argmax-vs-true-label classification result
// from the *final* step's own output, not a running metric.
export async function runCifar10PretrainDemo({ ort, modelBytes, manifest, onStep }) {
  const samples = await fetchCifar10Batch(manifest.numSamples);
  const side = Math.round(Math.sqrt(manifest.inputDim));
  const x = new Float32Array(manifest.numSamples * manifest.inputDim);
  const teacher = new Float32Array(manifest.numSamples * manifest.numClasses);
  const labels = [];
  for (let i = 0; i < samples.length; i++) {
    const flat = await decodeGraySquare(samples[i].bytes, side);
    x.set(flat, i * manifest.inputDim);
    teacher[i * manifest.numClasses + samples[i].label] = 1.0;
    labels.push(samples[i].label);
  }

  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "disabled",
    logSeverityLevel: 0,
    logVerbosityLevel: 4,
  });
  const constants = {
    x: new ort.Tensor("float32", x, [manifest.numSamples, manifest.inputDim]),
    teacher: new ort.Tensor("float32", teacher, [manifest.numSamples, manifest.numClasses]),
  };
  const { losses, lastOutputs } = await runStepLoop(ort, session, manifest, constants, onStep);
  await session.release?.();

  const y = lastOutputs[manifest.outputName].data;
  const predictions = [];
  for (let i = 0; i < manifest.numSamples; i++) {
    let best = 0;
    for (let c = 1; c < manifest.numClasses; c++) {
      if (y[i * manifest.numClasses + c] > y[i * manifest.numClasses + best]) best = c;
    }
    predictions.push(best);
  }
  const correct = predictions.filter((p, i) => p === labels[i]).length;

  return { losses, correct, total: manifest.numSamples, labelNames: samples.map((s) => s.labelName), predictions };
}
