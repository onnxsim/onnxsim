// Unit test for the converter page's browser-side calibration
// (quantize_calibration.mjs). No DOM and no real onnxruntime-web: calibrateRanges
// takes injectable ORT entry points (`ortLoader`, `makeInputs`) for exactly this,
// so the range accumulation and the execution-provider plumbing -- the Quantize
// panel's EP picker reaching the session that actually runs the model -- are
// testable under Node.
//
// Usage:
//   node test/quantize_calibration.test.mjs

import assert from "node:assert/strict";
import { providersForEp } from "../webnn.mjs";

// quantize_calibration.mjs pulls onnxruntime-web's loader from
// inference_browser.mjs, which wires the inference panel at import time. A
// getElementById that finds nothing makes that wiring bail out immediately
// (`if (!btn) return`), the same globalThis stub versions.test.mjs uses; the
// import has to follow the stub, hence the dynamic import.
globalThis.document = { getElementById: () => null, querySelector: () => null };
globalThis.window = { addEventListener: () => {} };
const { calibrateRanges } = await import("../quantize_calibration.mjs");

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// A runtime whose onnxsim_add_graph_outputs just records what it was asked for
// and hands back some bytes (the real one returns the augmented model).
function fakeRuntime(calls = []) {
  return {
    calls,
    onnxsim_add_graph_outputs(modelBuf, names) {
      calls.push({ modelBuf, names: [...names] });
      return new Uint8Array([1, 2, 3]).buffer;
    },
  };
}

// `samples` is one {tensor: values} object per run; the session reports every
// tensor named across them as an output, the way an augmented model would.
function fakeOrt(samples, seen = {}) {
  const outputNames = [...new Set(samples.flatMap((s) => Object.keys(s)))];
  let i = 0;
  return {
    seen,
    InferenceSession: {
      async create(bytes, options) {
        seen.bytes = bytes;
        seen.options = options;
        return {
          outputNames,
          inputMetadata: [],
          async run() {
            const sample = samples[Math.min(i, samples.length - 1)];
            i += 1;
            return Object.fromEntries(
              Object.entries(sample).map(([k, v]) => [k, { data: v }]),
            );
          },
        };
      },
    },
  };
}

function options(ort, extra = {}) {
  return {
    ortLoader: async (variant) => {
      ort.seen.variant = variant;
      return ort;
    },
    makeInputs: async () => ({}),
    ...extra,
  };
}

// Nothing to calibrate: no session is created at all (loading onnxruntime-web
// for an empty tensor list would be a pointless multi-megabyte fetch).
await check("no candidate tensors short-circuits", async () => {
  const runtime = fakeRuntime();
  const ort = fakeOrt([]);
  const result = await calibrateRanges(runtime, new ArrayBuffer(0), [], 4, () => {}, options(ort));
  assert.deepEqual(result, { names: [], flat: [] });
  assert.equal(ort.seen.options, undefined);
  assert.equal(runtime.calls.length, 0);
});

// Each tensor's range is the min/max across every sample, and `flat` is
// [min, max, ...] in the returned `names` order -- the shape
// runtime.onnxsim_quantize_static expects.
await check("ranges accumulate across samples", async () => {
  const ort = fakeOrt([
    { a: [0, 1, 2], b: [-5, 5] },
    { a: [-3, 7], b: [-1, 1] },
  ]);
  const { names, flat } = await calibrateRanges(
    fakeRuntime(), new ArrayBuffer(0), ["a", "b"], 2, () => {}, options(ort),
  );
  assert.deepEqual(names, ["a", "b"]);
  assert.deepEqual(flat, [-3, 7, -5, 5]);
});

// A tensor that never produced data (e.g. a zero-sized dim) is dropped rather
// than reported with an empty/infinite range.
await check("tensors with no data are dropped", async () => {
  const ort = fakeOrt([{ a: [1, 2], empty: [] }]);
  const { names, flat } = await calibrateRanges(
    fakeRuntime(), new ArrayBuffer(0), ["a", "empty"], 1, () => {}, options(ort),
  );
  assert.deepEqual(names, ["a"]);
  assert.deepEqual(flat, [1, 2]);
});

// Only the requested tensors are observed, even though an augmented model's
// session also reports the model's own original outputs.
await check("outputs outside the requested set are ignored", async () => {
  const ort = fakeOrt([{ a: [1, 2], Y: [100, 200] }]);
  const { names, flat } = await calibrateRanges(
    fakeRuntime(), new ArrayBuffer(0), ["a"], 1, () => {}, options(ort),
  );
  assert.deepEqual(names, ["a"]);
  assert.deepEqual(flat, [1, 2]);
});

// The panel's EP choice reaches the session, and the WebNN choices ask for the
// onnxruntime-web bundle that actually carries the WebNN backend.
await check("execution providers and bundle variant are forwarded", async () => {
  for (const ep of ["wasm", "webgpu", "webnn-npu"]) {
    const { providers, needWebnn } = providersForEp(ep);
    const ort = fakeOrt([{ a: [1, 2] }]);
    await calibrateRanges(
      fakeRuntime(), new ArrayBuffer(0), ["a"], 1, () => {},
      options(ort, { providers, needWebnn }),
    );
    assert.deepEqual(ort.seen.options.executionProviders, providers);
    assert.equal(ort.seen.variant, needWebnn ? "all" : "default");
    // Every accelerated choice keeps a WASM fallback, so an unavailable
    // device degrades instead of failing the whole quantization.
    assert.equal(providers[providers.length - 1], "wasm");
  }
});

// Default (no options): WASM only, and the default bundle -- calibrating on an
// accelerator stays opt-in.
await check("defaults to WASM", async () => {
  const ort = fakeOrt([{ a: [1, 2] }]);
  await calibrateRanges(fakeRuntime(), new ArrayBuffer(0), ["a"], 1, () => {}, options(ort));
  assert.deepEqual(ort.seen.options.executionProviders, ["wasm"]);
  assert.equal(ort.seen.variant, "default");
});

console.log(`\nquantize_calibration: ${passed} checks passed`);
