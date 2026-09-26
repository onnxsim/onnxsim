// Unit tests for the onnxsim npm package's public API (index.mjs).
//
// Needs the wasm module already built and staged (onnxsim.cjs / onnxsim.wasm)
// -- see scripts/build_npm_package.sh, or .github/workflows/static.yml, which
// runs this right after its own build via `npm test`.
//
// simplify() runs with constantFolding disabled here so the test never needs
// onnxruntime-web's wasm assets located on disk (see docs/wasm_ort_web.md):
// this exercises the embind marshaling, shape inference, and onnx-optimizer
// passes, not the onnxruntime-web constant-folding bridge -- that path is
// covered by the convertmodel demo's own inference test
// (scripts/convertmodel/test/inference.test.mjs), which configures
// onnxruntime-web's wasmPaths explicitly.
//
// Usage:
//   cd npm/onnxsim && npm install && npm test

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  applyAttentionHeadPruning,
  applyDoubleQuantization,
  applyEmbeddingVocabMagnitudePruning,
  applyEmbeddingVocabPruning,
  applyGgufQ4_0,
  applyGgufQ5_0,
  applyGgufQ8_0,
  applyGgufQ2K,
  applyGgufQ3K,
  applyGgufQ4K,
  applyGgufQ5K,
  applyIq4Nl,
  applyLeptoquant,
  quantizeWeightOnlyNf4,
  quantizeWeightOnlyIf4,
  quantizeWeightOnlyNvfp4,
  applyDeepseekFp8,
  applyKmeansQuantization,
  applyHqq,
  applyIbertGelu,
  applyIbertSoftmax,
  applyAdpq,
  applyIcquant,
  applyOlive,
  applyAqlm,
  applyDropByDrop,
  applyLoBcq,
  applyQuipSharp,
  applyAttentionQuantization,
  applyZeroquant,
  applyIntactkv,
  applyKbvqMoe,
  quantizeWeightOnlyLlmFp4,
  applyQoq,
  applyDsq,
  applyDaq,
  applyLowRankCompensation,
  quantizeEmbeddingBinary,
  applyMoeExpertChannelPruning,
  applyOutlierSuppression,
  applyOutlierSuppressionPlus,
  applyLlmInt8,
  applySpqr,
  quantizeWeightOnlyPbLlm,
  quantizeWeightOnlySqueezeLlm,
  applyBillm,
  quantizeKvCache,
  applyOwq,
  applyGear,
  applyRotateKv,
  applyGptq,
  applyAdaround,
  applyQronos,
  applyTesseraq,
  applyAwq,
  applyQuarot,
  applyQuarotGptq,
  applyGptvq,
  applySmoothQuant,
  applyStructuredPruning,
  applyWandaPruning,
  crossLayerEqualize,
  pruneMagnitude,
  quantizeAttentionDynamic,
  quantizeDynamicMatMulIntegerToFloat,
  quantizeWeightOnlyInt8Block,
  quantizeWeightOnlyInt16,
  quantizeWeightOnlyMatMulNbits,
  quantizeWeightOnlyMxfp4,
  setModelExecutorRunner,
  simplify,
  versions,
} from "../index.mjs";
import { makeOrtRunner } from "../ort_executor.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = join(HERE, "..", "..", "..", "scripts", "convertmodel", "test", "model.onnx");
// GPTQ needs a float model plus its INT4-quantized counterpart. Both are
// checked in, generated deterministically by quantizing the float one:
//   python3 -c "import onnx, numpy as np
//   from onnx import parser
//   import onnxsim
//   rng = np.random.default_rng(0)
//   W = (rng.standard_normal((32, 8)) * 0.5).astype(np.float32)
//   m = parser.parse_model('''
//   <ir_version: 10, opset_import: [\"\": 21]>
//   g (float[batch,32] X) => (float[batch,8] Y) { Y = MatMul(X, W) }
//   ''')
//   m.graph.initializer.extend([onnx.numpy_helper.from_array(W, 'W')])
//   onnx.save(m, 'model_gptq.onnx')
//   onnx.save(onnxsim.quantize_weight_only_int4(m), 'model_gptq_int4.onnx')"
const FIXTURE_GPTQ = join(HERE, "..", "..", "..", "scripts", "convertmodel", "test", "model_gptq.onnx");
const FIXTURE_GPTQ_INT4 = join(HERE, "..", "..", "..", "scripts", "convertmodel", "test", "model_gptq_int4.onnx");

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// onnxsim.cjs is Emscripten's minified, single-physical-line output; an
// uncaught throw/rejection reaching Node's default handler makes it dump the
// *entire* file as "source context" instead of a useful message (it treats
// the whole file as one "line"). Catch everything here and print just the
// error's message/stack so a real failure stays readable.
try {
  await check("simplify() returns a non-empty model and no trace by default", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    const { model, trace } = await simplify(input, { constantFolding: false });
    assert.ok(model instanceof Uint8Array);
    assert.ok(model.length > 0);
    assert.equal(trace, "");
  });

  await check("simplify() accepts an ArrayBuffer too", async () => {
    const input = readFileSync(FIXTURE);
    const { model } = await simplify(
      input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength),
      { constantFolding: false },
    );
    assert.ok(model instanceof Uint8Array);
    assert.ok(model.length > 0);
  });

  await check("simplify() rejects a non-bytes input", async () => {
    await assert.rejects(() => simplify("not bytes"), TypeError);
  });

  await check("simplify() rejects bytes that aren't a valid model", async () => {
    await assert.rejects(() => simplify(new Uint8Array([1, 2, 3])));
  });

  await check("versions() reports onnxsim and onnx-optimizer version strings", async () => {
    const v = await versions();
    assert.equal(typeof v.onnxsim, "string");
    assert.ok(v.onnxsim.length > 0);
    assert.equal(typeof v.onnx_optimizer, "string");
    assert.ok(v.onnx_optimizer.length > 0);
  });

  await check("custom executor hook runs constant folding through WASM", async () => {
    const ort = await import("onnxruntime-web");
    const dist = join(HERE, "..", "node_modules", "onnxruntime-web", "dist") + "/";
    ort.env.wasm.wasmPaths = dist;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    const ortRunner = makeOrtRunner(ort);
    let calls = 0;
    await setModelExecutorRunner(async (...args) => {
      calls += 1;
      return ortRunner(...args);
    });
    const input = new Uint8Array(readFileSync(FIXTURE));
    const { model } = await simplify(input, { constantFolding: true });
    assert.ok(model.length > 0);
    assert.ok(calls > 0, "constant folding did not call the custom executor");
  });

  // Data-free C++ passes: the fixture (MatMul+Add+Relu, X:[N,4], W:[4,3])
  // exercises them without needing onnxruntime-web, same as the simplify()
  // checks above.
  await check("data-free quantize passes return non-empty models", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    for (const fn of [
      crossLayerEqualize,
      quantizeDynamicMatMulIntegerToFloat,
      quantizeAttentionDynamic,
      quantizeWeightOnlyInt16,
      quantizeWeightOnlyInt8Block,
      quantizeWeightOnlyMxfp4,
      quantizeWeightOnlyMatMulNbits,
      applyDoubleQuantization,
      applyIq4Nl,
      applyGgufQ4_0,
      applyGgufQ5_0,
      applyGgufQ8_0,
      applyGgufQ2K,
      applyGgufQ3K,
      applyGgufQ4K,
      applyGgufQ5K,
      applyLeptoquant,
      quantizeWeightOnlyNf4,
      quantizeWeightOnlyIf4,
      quantizeWeightOnlyNvfp4,
      applyDeepseekFp8,
      applyKmeansQuantization,
      applyHqq,
      applyIbertGelu,
      applyIbertSoftmax,
      applyAdpq,
      applyIcquant,
      applyOlive,
      applyAqlm,
      applyDropByDrop,
      applyLoBcq,
      applyQuipSharp,
      // FIXTURE has no decomposed-attention/MoE subgraph, so
      // applyAttentionQuantization/applyKbvqMoe match nothing and the model
      // comes back unchanged -- same "binding round-trips" point as
      // applyQuarot's own inclusion in the pruning sweep below.
      applyAttentionQuantization,
      applyZeroquant,
      applyIntactkv,
      applyKbvqMoe,
      quantizeWeightOnlyLlmFp4,
      applyQoq,
      // FIXTURE has no SwiGLU/GLU-style MLP block, so applyDsq matches
      // nothing and the model comes back unchanged -- same
      // "binding round-trips" point as applyKbvqMoe's own inclusion above.
      applyDsq,
      // FIXTURE's own output last dim (3) isn't a multiple of 8, so
      // quantizeEmbeddingBinary declines and the model comes back
      // unchanged -- same "binding round-trips" point as applyDsq's own
      // inclusion above.
      quantizeEmbeddingBinary,
    ]) {
      const out = await fn(input);
      assert.ok(out instanceof Uint8Array, fn.name);
      assert.ok(out.length > 0, fn.name);
    }
  });

  await check("applyDaq quantizes a fine-tuned checkpoint against its base", async () => {
    const baseModel = new Uint8Array(readFileSync(FIXTURE));
    const out = await applyDaq(baseModel, baseModel, { metric: "cosine" });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyLowRankCompensation round-trips a two-model call", async () => {
    const floatModel = new Uint8Array(readFileSync(FIXTURE));
    // No DequantizeLinear(INT4, ...)-fed layer in FIXTURE, so no candidate
    // matches -- the point here is the binding round-trips, same as the
    // applyDaq test above.
    const out = await applyLowRankCompensation(floatModel, floatModel, { rank: 8 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("data-free pruning passes return non-empty models", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    for (const [fn, options] of [
      [pruneMagnitude, { sparsity: 0.5 }],
      [applyStructuredPruning, { sparsity: 0.5, importanceNorm: "l2" }],
      [applyAttentionHeadPruning, { sparsity: 0.5 }],
      [applyMoeExpertChannelPruning, { sparsity: 0.5 }],
      // K=4 is not divisible by 32, so QuaRot matches nothing and the model
      // comes back unchanged -- the point is the binding round-trips.
      [applyQuarot, { seed: 0, blockSize: 32, epsilon: 1e-6 }],
    ]) {
      const out = await fn(input, options);
      assert.ok(out instanceof Uint8Array, fn.name);
      assert.ok(out.length > 0, fn.name);
    }
  });

  await check("embedding vocab pruning declines a model with no vocab chain", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    const r = await applyEmbeddingVocabPruning(input, { dropTokenIds: [1, 2] });
    assert.equal(r.matched, false);
    assert.ok(r.model instanceof Uint8Array && r.model.length > 0);
    const r2 = await applyEmbeddingVocabMagnitudePruning(input, { sparsity: 0.5 });
    assert.equal(r2.matched, false);
    assert.ok(r2.model instanceof Uint8Array && r2.model.length > 0);
  });

  await check("applyWandaPruning runs on synthetic onnxruntime-web calibration data", async () => {
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applyWandaPruning(input, [batch(), batch()], { sparsity: 0.5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyOutlierSuppression declines a model with no LayerNorm", async () => {
    // The shared fixture has no LayerNormalization node, so this is a
    // no-op round-trip -- the point is the binding (including calibration
    // crossing) works, not that it migrates anything.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applyOutlierSuppression(input, [batch(), batch()], { alpha: 0.5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyLlmInt8 declines a pre-opset-18 model", async () => {
    // The shared fixture is opset 13 and LLM.int8() needs opset >= 18
    // (ReduceMax axes-as-input), so this is a no-op round-trip -- the
    // point is the binding (including calibration crossing) works. Real
    // decompositions are covered by the Python parity tests and were
    // verified bit-identical through this same binding on an opset-18
    // model during development.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 8.0, 0.0]), [1, 4]),
    });
    const out = await applyLlmInt8(input, [batch(), batch()], { outlierThreshold: 6.0 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applySpqr declines a pre-opset-21 model", async () => {
    // The shared fixture is opset 13 and SpQR needs opset >= 21 (INT4
    // tensor type, DequantizeLinear's block_size attribute), so this is a
    // no-op round-trip -- the point is the binding (including calibration
    // crossing) works. Real quantization is covered by the Python parity
    // tests.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applySpqr(input, [batch(), batch()], { blockSize: 16 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("quantizeWeightOnlyPbLlm runs on synthetic calibration data", async () => {
    // No block-size/opset gate on this technique (ordinary Cast/Mul,
    // opset 11+), so the shared K=4 fixture is a genuine candidate -- the
    // point is the binding (including calibration crossing) works, not a
    // tight numeric check (that's the Python parity tests' own job).
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await quantizeWeightOnlyPbLlm(input, [batch(), batch()], { salientRatio: 0.15 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("quantizeWeightOnlySqueezeLlm declines a non-block-divisible model", async () => {
    // The shared fixture's K=4 is below the default block_size=32, so
    // this is a no-op round-trip -- the point is the binding (including
    // calibration crossing) works. Real quantization is covered by the
    // Python parity tests.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await quantizeWeightOnlySqueezeLlm(input, [batch(), batch()], { blockSize: 32 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyBillm runs on synthetic calibration data", async () => {
    // No block-size divisibility requirement (any K works, chunked into
    // min(blockSize, remaining)-wide blocks), so the shared K=4 fixture is
    // a genuine candidate -- the point is the binding (including
    // calibration crossing) works, not a tight numeric check.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applyBillm(input, [batch(), batch()], { blockSize: 128 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("quantizeKvCache round-trips on a model with no KV-cache pattern", async () => {
    // The shared fixture is a plain MatMul with no Concat(past, new,
    // axis=seq) KV-cache stream, so this is a no-op round-trip -- the
    // point is the binding (including calibration crossing and the
    // optional valueOutputNames array) works. Real quantization is
    // covered by the Python parity tests.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await quantizeKvCache(input, [batch(), batch()]);
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyGear round-trips on a model with no KV-cache pattern", async () => {
    // Same no-KV-cache-pattern reasoning as quantizeKvCache above -- the
    // point is the binding works, not a tight numeric check.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applyGear(input, [batch(), batch()], { rank: 4 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyRotateKv round-trips on a model with no KV-cache/attention pattern", async () => {
    // Same no-matching-pattern reasoning as quantizeKvCache above -- the
    // point is the binding works, not a tight numeric check.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applyRotateKv(input, [batch(), batch()]);
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyOwq restores salient columns on synthetic calibration data", async () => {
    // Same dedicated fixtures as the GPTQ check above: a K=32 float MatMul
    // plus its quantize_weight_only_int4 output, so the layer is a real
    // OWQ candidate.
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const quantModel = new Uint8Array(readFileSync(FIXTURE_GPTQ_INT4));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyOwq(floatModel, quantModel, [batch(), batch()], { outlierFraction: 0.1 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyGptq optimizes INT4 codes on synthetic calibration data", async () => {
    // Dedicated fixtures (see the generation note below): a K=32 float
    // MatMul plus its quantize_weight_only_int4 output, so the layer is
    // a real GPTQ candidate (the shared K=4 fixture is below the int4
    // block size and would decline).
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const quantModel = new Uint8Array(readFileSync(FIXTURE_GPTQ_INT4));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyGptq(floatModel, quantModel, [batch(), batch()], {});
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyAdaround optimizes rounding on synthetic calibration data", async () => {
    // Same dedicated fixtures as the GPTQ check above; a small iteration
    // count keeps this check fast, the point being the two-model binding
    // (including the beta_start/beta_end tuple split) round-trips, not
    // full convergence.
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const quantModel = new Uint8Array(readFileSync(FIXTURE_GPTQ_INT4));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyAdaround(floatModel, quantModel, [batch(), batch()], {
      numIterations: 20,
    });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyQronos corrects codes on synthetic calibration data", async () => {
    // Same dedicated fixtures as the GPTQ check above -- a single-layer
    // model has no upstream-quantized predecessor, so this also
    // exercises Qronos's own exact GPTQ reduction, but the point here is
    // just that the two-model binding (including the per-layer re-probe
    // against the progressively-corrected working model) round-trips.
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const quantModel = new Uint8Array(readFileSync(FIXTURE_GPTQ_INT4));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyQronos(floatModel, quantModel, [batch(), batch()], {});
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyTesseraq optimizes rounding on synthetic calibration data", async () => {
    // Same dedicated fixtures as the GPTQ/Qronos checks above; a small
    // iteration count keeps this check fast, the point being the
    // two-model binding (including the beta_start/beta_end tuple split)
    // round-trips, not full convergence.
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const quantModel = new Uint8Array(readFileSync(FIXTURE_GPTQ_INT4));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyTesseraq(floatModel, quantModel, [batch(), batch()], {
      numIterations: 20,
      parRounds: 2,
    });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyAwq searches scales on synthetic calibration data", async () => {
    // Same dedicated fixtures as the GPTQ check above.
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const quantModel = new Uint8Array(readFileSync(FIXTURE_GPTQ_INT4));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyAwq(floatModel, quantModel, [batch(), batch()], { numAlphaSteps: 5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyQuarotGptq rotates and quantizes on synthetic calibration data", async () => {
    // Same dedicated float fixture as the GPTQ/AWQ checks above (K=32,
    // divisible by the default block size) -- unlike those, this pass
    // takes only the float model: it derives its own rotation and
    // quantizes from scratch, like applyQuarot's own single-model binding.
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyQuarotGptq(floatModel, [batch(), batch()], { seed: 0 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyGptvq quantizes weight groups on synthetic calibration data", async () => {
    // Same dedicated float fixture as the GPTQ/AWQ/QuarotGptq checks
    // above (K=32) -- like applyQuarotGptq, this pass takes only the
    // float model: it fits its own codebook and quantizes from scratch.
    const ort = await import("onnxruntime-web");
    const floatModel = new Uint8Array(readFileSync(FIXTURE_GPTQ));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array(32).map((_, i) => ((i * 37) % 11) - 5), [1, 32]),
    });
    const out = await applyGptvq(floatModel, [batch(), batch()], { seed: 0, numCentroids: 16 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applySmoothQuant migrates scales on synthetic calibration data", async () => {
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 4.0, 0.0]), [1, 4]),
    });
    const out = await applySmoothQuant(input, [batch(), batch()], { alpha: 0.5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyOutlierSuppressionPlus migrates scales on synthetic calibration data", async () => {
    // The shared MatMul fixture migrates for real here (unlike the
    // LayerNorm-gated check above): the point is the Sub/Mul/Add rewrite
    // round-trips through the binding, not just a decline.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 4.0, 8.0]), [1, 4]),
    });
    const out = await applyOutlierSuppressionPlus(input, [batch(), batch()], { alpha: 0.5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > input.length);
  });

  console.log(`PASS: ${passed} checks`);
} catch (err) {
  console.error(`FAIL after ${passed} checks:`, (err && err.stack) || err);
  process.exitCode = 1;
}
