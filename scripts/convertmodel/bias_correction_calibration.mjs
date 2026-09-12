// Browser-side measurement for onnxsim's bias-correction accuracy-recovery
// tools (see onnxsim/bias_correction.py's own module docstring for the
// technique and when it does or doesn't help): runs the float model and a
// "modified" model (e.g. quantized elsewhere, or with a Resize node's mode/
// coordinate_transformation_mode swapped for a deployment target's
// accelerator) on the same synthetic calibration data through
// onnxruntime-web, and for every eligible Conv/Gemm/MatMul/Resize output
// (from runtime.onnxsim_list_correctable_outputs) measures a correction:
//
//  - plain (correct_bias): a single per-channel mean shift
//    (float_output - modified_output, averaged over every other axis),
//    kept whenever its magnitude clears a tiny noise floor -- no held-out
//    check, matching correct_bias's own unconditional behavior.
//  - spatial (correct_spatial_bias, only for axis-1/NCHW-shaped candidates):
//    a coarse per-channel grid of position-wise mean error, smoothed to
//    full resolution, fit on part of the calibration data and only kept if
//    it measurably reduces error on the rest -- since it can otherwise fit
//    noise and make things worse (see bias_correction.py's own docstring).
//
// Only the measurement lives here; runtime.onnxsim_apply_bias_corrections
// (interface.cpp/bias_correction_entry.h) does the actual graph edit. The
// WASM module itself never runs inference for this step -- same split as
// quantize_calibration.mjs's own calibrateRanges: runtime.onnxsim_add_graph_outputs
// exposes each candidate as an extra graph output (a bare ValueInfoProto,
// exactly like calibrate()'s own onnx.ValueInfoProto(name=name)), and
// onnxruntime-web (already loaded on this page for the "Run inference"
// panel) executes the augmented models to observe them.

import { loadOrt, makeDummyInputs } from "./inference_browser.mjs";

// Downsamples a channel-major [c, h, w] Float64Array to [c, gh, gw] by
// averaging each of its (possibly unevenly sized) rectangular blocks --
// ports bias_correction.py's own _block_average exactly (same block
// boundaries: block i spans rows/cols round(i*n/g) .. round((i+1)*n/g)).
// numpy.array_split(arange(n), g) boundaries: the first (n % g) blocks get
// ceil(n/g) elements, the rest get floor(n/g) -- NOT evenly-rounded
// boundaries (round(i*n/g)), which gives a different partition whenever g
// doesn't evenly divide n.
function arraySplitBoundaries(n, g) {
  const q = Math.floor(n / g);
  const r = n % g;
  const starts = [0];
  for (let i = 0; i < g; i++) starts.push(starts[i] + q + (i < r ? 1 : 0));
  return starts;
}

export function blockAverage(flat, c, h, w, gh, gw) {
  const rowStarts = arraySplitBoundaries(h, gh);
  const colStarts = arraySplitBoundaries(w, gw);
  const coarse = new Float64Array(c * gh * gw);
  for (let ch = 0; ch < c; ch++) {
    const chBase = ch * h * w;
    for (let gy = 0; gy < gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        let sum = 0;
        let count = 0;
        for (let y = rowStarts[gy]; y < rowStarts[gy + 1]; y++) {
          const rowBase = chBase + y * w;
          for (let x = colStarts[gx]; x < colStarts[gx + 1]; x++) {
            sum += flat[rowBase + x];
            count++;
          }
        }
        coarse[ch * gh * gw + gy * gw + gx] = count > 0 ? sum / count : 0;
      }
    }
  }
  return coarse;
}

// Upsamples a channel-major [c, gh, gw] Float64Array to [c, h, w] by
// bilinear interpolation between grid-cell centers (clamp-to-edge past the
// outermost centers) -- ports bias_correction.py's own _bilinear_upsample.
export function bilinearUpsample(coarse, c, gh, gw, h, w) {
  if (gh === h && gw === w) return coarse;
  const out = new Float64Array(c * h * w);
  const coord = (o, n, g) => Math.min(Math.max(((o + 0.5) * g) / n - 0.5, 0), g - 1);
  for (let ch = 0; ch < c; ch++) {
    const gridBase = ch * gh * gw;
    const outBase = ch * h * w;
    for (let oy = 0; oy < h; oy++) {
      const cy = coord(oy, h, gh);
      const y0 = Math.floor(cy);
      const y1 = Math.min(y0 + 1, gh - 1);
      const wy = cy - y0;
      for (let ox = 0; ox < w; ox++) {
        const cx = coord(ox, w, gw);
        const x0 = Math.floor(cx);
        const x1 = Math.min(x0 + 1, gw - 1);
        const wx = cx - x0;
        const top = coarse[gridBase + y0 * gw + x0] * (1 - wx) + coarse[gridBase + y0 * gw + x1] * wx;
        const bottom = coarse[gridBase + y1 * gw + x0] * (1 - wx) + coarse[gridBase + y1 * gw + x1] * wx;
        out[outBase + oy * w + ox] = top * (1 - wy) + bottom * wy;
      }
    }
  }
  return out;
}

// Accumulates one batch's (float - modified) error into `entry` for
// candidate `c`. `entry.sum` is per-channel (length C) for a plain
// correction, or per-position (length C*H*W, channel-major) for a spatial
// one; `entry.count` is how many elements each sum slot has actually
// summed, so the final mean is always sum/count regardless of shape.
function accumulateCandidate(c, floatTensor, modifiedTensor, entry, spatial) {
  const dims = floatTensor.dims;
  if (!dims || dims.length === 0) return;
  if (
    !modifiedTensor.dims ||
    dims.length !== modifiedTensor.dims.length ||
    dims.some((d, i) => d !== modifiedTensor.dims[i])
  ) {
    return; // shape mismatch -- not a candidate this batch can measure
  }
  const f = floatTensor.data;
  const m = modifiedTensor.data;

  if (spatial) {
    if (dims.length !== 4) return; // only NCHW-shaped candidates have a grid to fit
    const [n, ch, h, w] = dims;
    const chw = ch * h * w;
    if (!entry.sum) entry.sum = new Float64Array(chw);
    entry.dims = dims;
    for (let b = 0; b < n; b++) {
      const base = b * chw;
      for (let k = 0; k < chw; k++) entry.sum[k] += Number(f[base + k]) - Number(m[base + k]);
    }
    entry.count += n;
    return;
  }

  const axis = c.axis >= 0 ? c.axis : dims.length + c.axis;
  const chCount = dims[axis];
  const chStride = dims.slice(axis + 1).reduce((a, b) => a * b, 1);
  const outer = dims.slice(0, axis).reduce((a, b) => a * b, 1);
  if (!entry.sum) entry.sum = new Float64Array(chCount);
  entry.dims = dims;
  for (let o = 0; o < outer; o++) {
    for (let ch = 0; ch < chCount; ch++) {
      const base = (o * chCount + ch) * chStride;
      for (let s = 0; s < chStride; s++) entry.sum[ch] += Number(f[base + s]) - Number(m[base + s]);
    }
  }
  entry.count += outer * chStride;
}

// Sum of squared (float - (modified [+ correction])) over one batch, for
// validating a spatial correction actually helps. `correction` may be null
// (the "before" measurement).
function sumSquaredError(floatTensor, modifiedTensor, correction) {
  const f = floatTensor.data;
  const m = modifiedTensor.data;
  const n = Math.min(f.length, m.length);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const corrected = correction ? Number(m[i]) + correction[i % correction.length] : Number(m[i]);
    const d = Number(f[i]) - corrected;
    total += d * d;
  }
  return total;
}

// Measures bias corrections for every eligible Conv/Gemm/MatMul/Resize
// output shared between `floatBuf` and `modifiedBuf` (raw model bytes).
// Returns `{ corrections, candidates }`: `candidates` is
// runtime.onnxsim_list_correctable_outputs' own result (for reporting), and
// `corrections` is the accepted-only array ready for
// runtime.onnxsim_apply_bias_corrections -- `{ name, shape, data }` per
// entry, `data` a plain Array<number> (row-major, matching `shape`).
export async function measureBiasCorrections(runtime, floatBuf, modifiedBuf, options = {}) {
  const {
    spatial = false,
    numSamples = spatial ? 16 : 8,
    gridSize = 8,
    validationFraction = 0.3,
    correctionThreshold = 1e-6,
    providers = ["wasm"],
    ortLoader = loadOrt,
    makeInputs = makeDummyInputs,
    log = () => {},
  } = options;

  const candidates = runtime.onnxsim_list_correctable_outputs(floatBuf, modifiedBuf);
  if (!candidates || candidates.length === 0) {
    return { corrections: [], candidates: [] };
  }
  const names = candidates.map((c) => c.name);

  const floatAugmented = runtime.onnxsim_add_graph_outputs(floatBuf, names);
  const modifiedAugmented = runtime.onnxsim_add_graph_outputs(modifiedBuf, names);
  if (!floatAugmented || !modifiedAugmented) {
    throw new Error("failed to prepare models for bias-correction calibration (add_graph_outputs)");
  }
  const floatBytes = new Uint8Array(floatAugmented).slice();
  const modifiedBytes = new Uint8Array(modifiedAugmented).slice();

  const ort = await ortLoader("default");
  const floatSess = await ort.InferenceSession.create(floatBytes, { executionProviders: providers });
  const modifiedSess = await ort.InferenceSession.create(modifiedBytes, { executionProviders: providers });

  const numVal = spatial
    ? Math.min(Math.max(1, Math.round(numSamples * validationFraction)), numSamples - 1)
    : 0;
  const numFit = numSamples - numVal;

  const state = new Map(candidates.map((c) => [c.name, { sum: null, count: 0, dims: null }]));

  async function runBatch() {
    const feeds = await makeInputs(ort, floatSess, 1, "random", null, null, log);
    const [floatOut, modifiedOut] = await Promise.all([floatSess.run(feeds), modifiedSess.run(feeds)]);
    return { floatOut, modifiedOut };
  }

  for (let i = 0; i < numFit; i++) {
    log(`bias correction: fit sample ${i + 1}/${numFit}…`);
    const { floatOut, modifiedOut } = await runBatch();
    for (const c of candidates) {
      const f = floatOut[c.name];
      const m = modifiedOut[c.name];
      if (!f || !m) continue;
      accumulateCandidate(c, f, m, state.get(c.name), spatial && c.spatial);
    }
  }

  // Build each candidate's (still unvalidated) correction.
  const fitted = new Map(); // name -> { data: Float64Array (matches entry.dims' broadcast), shape }
  for (const c of candidates) {
    const entry = state.get(c.name);
    if (!entry.sum || entry.count === 0) continue;
    const dims = entry.dims;
    const mean = entry.sum.map((v) => v / entry.count);

    if (spatial && c.spatial && dims.length === 4) {
      const [, ch, h, w] = dims;
      const gh = Math.min(gridSize, h);
      const gw = Math.min(gridSize, w);
      const coarse = blockAverage(mean, ch, h, w, gh, gw);
      const smooth = bilinearUpsample(coarse, ch, gh, gw, h, w);
      fitted.set(c.name, { data: smooth, shape: [1, ch, h, w] });
    } else {
      if (mean.reduce((m2, v) => Math.max(m2, Math.abs(v)), 0) <= correctionThreshold) {
        continue; // no measurable bias -- correct_bias's own no-op threshold
      }
      const axis = c.axis >= 0 ? c.axis : dims.length + c.axis;
      const shape = dims.map(() => 1);
      shape[axis] = dims[axis];
      fitted.set(c.name, { data: mean, shape });
    }
  }

  if (!spatial || numVal === 0) {
    const corrections = [];
    for (const [name, { data, shape }] of fitted) {
      corrections.push({ name, shape, data: Array.from(data) });
    }
    return { corrections, candidates };
  }

  // Validate each spatial correction on held-out samples: only keep it if
  // it measurably reduces squared error versus leaving modified_model alone.
  const sqBefore = new Map(Array.from(fitted.keys(), (name) => [name, 0]));
  const sqAfter = new Map(Array.from(fitted.keys(), (name) => [name, 0]));
  for (let i = 0; i < numVal; i++) {
    log(`bias correction: validation sample ${i + 1}/${numVal}…`);
    const { floatOut, modifiedOut } = await runBatch();
    for (const [name, { data }] of fitted) {
      const f = floatOut[name];
      const m = modifiedOut[name];
      if (!f || !m) continue;
      sqBefore.set(name, sqBefore.get(name) + sumSquaredError(f, m, null));
      sqAfter.set(name, sqAfter.get(name) + sumSquaredError(f, m, data));
    }
  }

  const corrections = [];
  for (const [name, { data, shape }] of fitted) {
    if (sqAfter.get(name) < sqBefore.get(name)) {
      corrections.push({ name, shape, data: Array.from(data) });
    }
  }
  return { corrections, candidates };
}
