// Unit test for the converter page's browser-side bias-correction
// measurement (bias_correction_calibration.mjs). No DOM and no real
// onnxruntime-web: measureBiasCorrections takes injectable ORT entry points
// (`ortLoader`, `makeInputs`), the same way quantize_calibration.mjs's own
// calibrateRanges does (see quantize_calibration.test.mjs), so the
// accumulation, grid-fitting, and held-out validation logic are testable
// under Node without a browser or a compiled onnxsim.wasm.
//
// Usage:
//   node test/bias_correction_calibration.test.mjs

import assert from "node:assert/strict";

// bias_correction_calibration.mjs pulls onnxruntime-web's loader from
// inference_browser.mjs, which wires the inference panel at import time --
// same stub-then-dynamic-import as quantize_calibration.test.mjs.
globalThis.document = { getElementById: () => null, querySelector: () => null };
globalThis.window = { addEventListener: () => {} };
const { measureBiasCorrections, blockAverage, bilinearUpsample } = await import(
  "../bias_correction_calibration.mjs"
);

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// A runtime whose list/add_graph_outputs are driven by fixed test data
// instead of a real onnxsim.wasm.
function fakeRuntime(candidates, addGraphOutputsCalls = []) {
  return {
    addGraphOutputsCalls,
    onnxsim_list_correctable_outputs(floatBuf, modifiedBuf) {
      return candidates;
    },
    onnxsim_add_graph_outputs(modelBuf, names) {
      addGraphOutputsCalls.push({ modelBuf, names: [...names] });
      return new Uint8Array([1, 2, 3]).buffer;
    },
  };
}

// One fake onnxruntime-web that serves a float session and a modified
// session from two parallel per-batch tensor lists (`floatBatches`/
// `modifiedBatches`, one `{dims, data}` per batch) -- the float and modified
// sessions always see the *same* batch index, since makeInputs (called once
// per batch, before both sessions run) is what advances it.
function fakeOrt(name, floatBatches, modifiedBatches) {
  let batchIndex = -1;
  const makeSession = (batches) => ({
    outputNames: [name],
    async run() {
      const b = batches[Math.min(batchIndex, batches.length - 1)];
      return { [name]: { dims: b.dims, data: b.data } };
    },
  });
  const created = [];
  return {
    created,
    advance: () => {
      batchIndex += 1;
    },
    InferenceSession: {
      async create(bytes) {
        // First create() call is for the float model, second for modified --
        // matches measureBiasCorrections' own call order.
        const sess = makeSession(created.length === 0 ? floatBatches : modifiedBatches);
        created.push(sess);
        return sess;
      },
    },
  };
}

function options(ort, extra = {}) {
  return {
    ortLoader: async () => ort,
    makeInputs: async () => {
      ort.advance();
      return {};
    },
    ...extra,
  };
}

await check("no candidates short-circuits (no sessions created)", async () => {
  const calls = [];
  const runtime = fakeRuntime([], calls);
  const result = await measureBiasCorrections(runtime, new ArrayBuffer(0), new ArrayBuffer(0), {});
  assert.deepEqual(result, { corrections: [], candidates: [] });
  assert.equal(calls.length, 0);
});

await check("plain per-channel correction recovers a known injected bias", async () => {
  // dims [N=1, C=2, H=2, W=2]; modified = float - [0.5, -0.3] per channel
  // (broadcast over the 4 spatial positions), identical every sample -- the
  // mean error is exactly the injected bias.
  const inject = [0.5, -0.3];
  const floatData = [1, 2, 3, 4, 5, 6, 7, 8]; // channel0: 1..4, channel1: 5..8
  const modifiedData = floatData.map((v, i) => v - inject[Math.floor(i / 4)]);
  const dims = [1, 2, 2, 2];
  const batches = Array.from({ length: 4 }, () => ({ dims, data: floatData.slice() }));
  const modifiedBatches = Array.from({ length: 4 }, () => ({ dims, data: modifiedData.slice() }));

  const runtime = fakeRuntime([{ name: "y", axis: 1, spatial: true }]);
  const ort = fakeOrt("y", batches, modifiedBatches);

  const { corrections } = await measureBiasCorrections(runtime, new ArrayBuffer(0), new ArrayBuffer(0), {
    spatial: false,
    numSamples: 4,
    ...options(ort),
  });

  assert.equal(corrections.length, 1);
  assert.deepEqual(corrections[0].shape, [1, 2, 1, 1]);
  for (let i = 0; i < 2; i++) {
    assert.ok(
      Math.abs(corrections[0].data[i] - inject[i]) < 1e-9,
      `channel ${i}: expected ${inject[i]}, got ${corrections[0].data[i]}`,
    );
  }
});

await check("plain correction below the noise floor is skipped", async () => {
  const dims = [1, 1, 1, 1];
  const floatBatches = [{ dims, data: [1] }];
  const modifiedBatches = [{ dims, data: [1 - 1e-9] }]; // far below correctionThreshold
  const runtime = fakeRuntime([{ name: "y", axis: 1, spatial: true }]);
  const ort = fakeOrt("y", floatBatches, modifiedBatches);

  const { corrections } = await measureBiasCorrections(runtime, new ArrayBuffer(0), new ArrayBuffer(0), {
    spatial: false,
    numSamples: 1,
    ...options(ort),
  });
  assert.deepEqual(corrections, []);
});

await check("spatial correction is accepted when it generalizes to held-out data", async () => {
  // A per-position pattern shared identically between the fit and
  // validation samples -- gridSize (8) clamps to H=W=2, so blockAverage is
  // exact per-pixel and the correction should drive validation error to ~0.
  const dims = [1, 1, 2, 2];
  const pattern = [1, -1, 2, -2]; // float - modified, per position
  const floatData = [10, 20, 30, 40];
  const modifiedData = floatData.map((v, i) => v - pattern[i]);
  const allBatches = Array.from({ length: 10 }, () => ({ dims, data: floatData.slice() }));
  const allModifiedBatches = Array.from({ length: 10 }, () => ({ dims, data: modifiedData.slice() }));

  const runtime = fakeRuntime([{ name: "y", axis: 1, spatial: true }]);
  const ort = fakeOrt("y", allBatches, allModifiedBatches);

  const { corrections } = await measureBiasCorrections(runtime, new ArrayBuffer(0), new ArrayBuffer(0), {
    spatial: true,
    numSamples: 10,
    validationFraction: 0.3,
    gridSize: 8,
    ...options(ort),
  });

  assert.equal(corrections.length, 1);
  assert.deepEqual(corrections[0].shape, [1, 1, 2, 2]);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(corrections[0].data[i] - pattern[i]) < 1e-9);
  }
});

await check("spatial correction is rejected when it doesn't generalize", async () => {
  // Fit samples see +1 everywhere; validation samples see -1 everywhere (the
  // opposite sign) -- applying the fitted (+1) correction on validation data
  // doubles the error instead of cancelling it, so it must be rejected.
  const dims = [1, 1, 2, 2];
  const floatData = [10, 20, 30, 40];
  const fitBatches = Array.from({ length: 4 }, () => ({
    dims,
    data: floatData.map((v) => v - 1),
  })).map((b) => ({ dims, data: b.data }));
  const valBatches = Array.from({ length: 4 }, () => ({
    dims,
    data: floatData.map((v) => v + 1),
  }));
  const modifiedBatches = [...fitBatches, ...valBatches];
  const floatBatches = Array.from({ length: 8 }, () => ({ dims, data: floatData.slice() }));

  const runtime = fakeRuntime([{ name: "y", axis: 1, spatial: true }]);
  const ort = fakeOrt("y", floatBatches, modifiedBatches);

  const { corrections } = await measureBiasCorrections(runtime, new ArrayBuffer(0), new ArrayBuffer(0), {
    spatial: true,
    numSamples: 8,
    validationFraction: 0.5, // 4 fit, 4 validation -- matches the split above
    gridSize: 8,
    ...options(ort),
  });
  assert.deepEqual(corrections, []);
});

await check("blockAverage/bilinearUpsample round-trip is a no-op at full resolution", () => {
  const flat = new Float64Array([1, 2, 3, 4]);
  const coarse = blockAverage(flat, 1, 2, 2, 2, 2);
  assert.deepEqual(Array.from(coarse), [1, 2, 3, 4]);
  const smooth = bilinearUpsample(coarse, 1, 2, 2, 2, 2);
  assert.deepEqual(Array.from(smooth), [1, 2, 3, 4]);
});

console.log(`\nbias_correction_calibration: ${passed} checks passed`);
