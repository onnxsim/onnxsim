// Browser-side calibration for onnxsim's calibration-based quantization
// methods (Static / QOperator, see quantize_ui.mjs): runs the model over
// synthetic random inputs through onnxruntime-web and records each candidate
// tensor's observed (min, max) -- the same shape of work
// onnxsim.calibration.calibrate() does in Python (onnxsim/calibration.py),
// driven entirely in the browser instead: no Python, no server round trip.
//
// The WASM module itself never runs inference for this step: onnxsim's own
// constant-folding executor only evaluates throwaway sub-models, not a full
// forward pass with arbitrary extra outputs. Instead, runtime.onnxsim_add_graph_outputs
// (see interface.cpp) exposes each candidate tensor as an extra graph output
// -- a bare ValueInfoProto with no type/shape, exactly like calibrate()'s own
// onnx.ValueInfoProto(name=name) -- and onnxruntime-web (already loaded on
// this page for the "Run inference" panel) actually executes the augmented
// model to observe them.
//
// Those forward passes are the expensive part of calibration (numSamples full
// runs of the model), so they are not pinned to WASM: the caller passes the
// execution providers the Quantize panel's own EP picker selected, the same
// providersForEp() list the inference panel uses (webnn.mjs), which always
// ends in "wasm" so an unavailable WebGPU/WebNN device just falls back. WASM
// stays the default -- an accelerated provider can compute in a different
// precision (fp16 on a WebNN NPU device), which moves the observed min/max a
// little, so choosing one is opt-in rather than something calibration does
// behind the user's back.

import { loadOrt, makeDummyInputs } from "./inference_browser.mjs";

// Calibrates `tensorNames` (from runtime.onnxsim_list_quantizable_activations
// / _list_qoperator_quantizable_outputs) over `numSamples` random input
// batches. Returns { names, flat }: `names` is the list of tensor names
// actually observed (a tensor with no data in any run, e.g. from a
// zero-sized dim, is dropped), and `flat` is a plain array of
// [min0, max0, min1, max1, ...] in the same order -- exactly the shape
// runtime.onnxsim_quantize_static / _quantize_qoperator expect.
// `options` carries the execution-provider choice ({ providers, needWebnn } as
// returned by providersForEp) and, for tests, injectable onnxruntime-web
// entry points.
export async function calibrateRanges(runtime, modelBuf, tensorNames, numSamples = 8, log = () => {}, options = {}) {
  const {
    providers = ["wasm"],
    needWebnn = false,
    ortLoader = loadOrt,
    makeInputs = makeDummyInputs,
  } = options;
  if (!tensorNames || tensorNames.length === 0) {
    return { names: [], flat: [] };
  }
  const augmented = runtime.onnxsim_add_graph_outputs(modelBuf, tensorNames);
  if (!augmented) {
    throw new Error("failed to prepare the model for calibration (add_graph_outputs)");
  }
  // Copy out of the wasm heap right away -- the view is invalidated by the
  // next wasm call, and onnxruntime-web needs its own stable copy anyway.
  const augmentedBytes = new Uint8Array(augmented).slice();

  // The WebNN providers only exist in onnxruntime-web's "all" bundle, the
  // same variant split the inference panel makes.
  const ort = await ortLoader(needWebnn ? "all" : "default");
  const sess = await ort.InferenceSession.create(augmentedBytes, {
    executionProviders: providers,
  });
  const wanted = new Set(tensorNames);
  const ranges = new Map();
  for (let i = 0; i < numSamples; i++) {
    log(`calibrating: sample ${i + 1}/${numSamples}…`);
    // "random" (a seeded xorshift32, see input_fill.mjs) rather than
    // "zeros"/"sample": a fixed input would collapse every tensor's observed
    // range to a single point, and "sample" would need network fetches this
    // background calibration step shouldn't depend on.
    const feeds = await makeInputs(ort, sess, 1, "random", null, null, log);
    const outputs = await sess.run(feeds);
    for (const name of sess.outputNames) {
      if (!wanted.has(name)) continue;
      const data = outputs[name] && outputs[name].data;
      if (!data || data.length === 0) continue;
      let mn = Infinity;
      let mx = -Infinity;
      for (let j = 0; j < data.length; j++) {
        const v = Number(data[j]);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const prev = ranges.get(name);
      ranges.set(name, prev ? [Math.min(prev[0], mn), Math.max(prev[1], mx)] : [mn, mx]);
    }
  }
  const names = Array.from(ranges.keys());
  const flat = [];
  for (const name of names) {
    const [mn, mx] = ranges.get(name);
    flat.push(mn, mx);
  }
  return { names, flat };
}
