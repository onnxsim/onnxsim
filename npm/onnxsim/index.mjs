// Node.js wrapper around onnxsim's WebAssembly build.
//
// The WASM module has a hookable ModelExecutor bridge. This package currently
// ships the ORT-web variant (see docs/wasm_ort_web.md): the module delegates
// constant folding to onnxruntime-web, which is an ordinary npm dependency
// rather than a second copy of ONNX Runtime compiled into the module.
// onnxsim.cjs / onnxsim.wasm / ort_executor.mjs are build
// artifacts staged in by scripts/stage_npm_package.sh (see that script and
// .github/workflows/static.yml) — they are not checked into git.

import createOnnxsim from "./onnxsim.cjs";
import { makeOrtRunner } from "./ort_executor.mjs";

// Matches onnxsim's Python CLI default (DEFAULT_TENSOR_SIZE_THRESHOLDHOLD in
// onnxsim/onnx_simplifier.py): skip folding a constant larger than this many
// bytes, trading a smaller "no large tensor" optimization for a smaller
// model / faster fold.
const DEFAULT_TENSOR_SIZE_THRESHOLD = 1.5 * 1024 ** 3;

let runtimePromise;

function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      let runtime;
      try {
        runtime = await createOnnxsim({
          print: (str) => console.log(str),
          printErr: (str) => console.error(str),
        });
      } catch (err) {
        // onnxsim.cjs is Emscripten's minified, single-physical-line output;
        // an uncaught throw from inside it makes Node's default formatter
        // dump the *entire* file as "source context" instead of a useful
        // message. Catch and rethrow a short, readable error instead.
        const detail = (err && err.message) || err;
        throw new Error(`onnxsim: failed to initialize the wasm module: ${detail}`, { cause: err });
      }
      if (
        typeof runtime.onnxsim_needs_ort_web === "function" &&
        runtime.onnxsim_needs_ort_web()
      ) {
        const ort = await import("onnxruntime-web");
        const runner = makeOrtRunner(ort.default ?? ort);
        runtime.onnxsimModelExecutorRun = runner;
        runtime.onnxsimOrtWebRun = runner;
      }
      return runtime;
    })();
  }
  return runtimePromise;
}

/**
 * Install a custom runner for constant-folding subgraphs.
 *
 * The runner receives one serialized subgraph and a batched tensor payload:
 * `(modelBytes, inputsData, inputsMeta) => Promise<{data, meta}>`.
 * The metadata layout is documented in docs/wasm_ort_web.md. This can target
 * a remote executor, WebGPU, or another browser/embedded runtime.
 */
export async function setModelExecutorRunner(runner) {
  if (typeof runner !== "function") {
    throw new TypeError("onnxsim: model executor runner must be a function");
  }
  const runtime = await getRuntime();
  runtime.onnxsimModelExecutorRun = runner;
  // Keep the old property populated for generated modules from older builds.
  runtime.onnxsimOrtWebRun = runner;
}

function toBytes(model) {
  if (model instanceof Uint8Array) return model;
  if (model instanceof ArrayBuffer) return new Uint8Array(model);
  throw new TypeError(
    "onnxsim: model must be a Uint8Array or ArrayBuffer of serialized ONNX bytes",
  );
}

/**
 * Simplify a serialized ONNX model.
 *
 * @param {Uint8Array|ArrayBuffer} model - serialized `onnx.ModelProto` bytes.
 * @param {object} [options]
 * @param {string[]} [options.skipOptimizers] - onnx-optimizer pass names to skip.
 * @param {boolean} [options.constantFolding=true]
 * @param {boolean} [options.shapeInference=true]
 * @param {number} [options.tensorSizeThreshold] - skip folding constants
 *   larger than this many bytes (default 1.5 GiB, matching the Python CLI).
 * @param {number} [options.targetOpsetVersion=-1] - <=0 keeps the model's opset.
 * @param {boolean} [options.profile=false] - also return a Chrome trace JSON.
 * @param {boolean} [options.annotateModelInfo=false] - bake MAC/FLOP counts
 *   into the model's metadata_props.
 * @param {boolean} [options.graphDiff=false] - print a detailed node/value-level
 *   before/after diff (which nodes/values were removed, added, or changed) via
 *   console.log, in addition to the op-count summary always printed.
 * @returns {Promise<{model: Uint8Array, trace: string}>}
 */
export async function simplify(model, options = {}) {
  const bytes = toBytes(model);
  const runtime = await getRuntime();
  const {
    skipOptimizers = [],
    constantFolding = true,
    shapeInference = true,
    tensorSizeThreshold = DEFAULT_TENSOR_SIZE_THRESHOLD,
    targetOpsetVersion = -1,
    profile = false,
    annotateModelInfo = false,
    graphDiff = false,
  } = options;

  let result = runtime.onnxsimplify_export(
    bytes,
    skipOptimizers,
    constantFolding,
    shapeInference,
    tensorSizeThreshold,
    targetOpsetVersion,
    profile,
    annotateModelInfo,
    graphDiff,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: simplify failed (see stderr output for details)");
  }
  return { model: new Uint8Array(result.model), trace: result.trace || "" };
}

/** onnxsim / onnx-optimizer version strings baked into this build. */
export async function versions() {
  const runtime = await getRuntime();
  return runtime.onnxsim_versions();
}

// ---------------------------------------------------------------------------
// Data-free quantization / pruning passes (C++ ports, no calibration data).
//
// Each takes serialized `onnx.ModelProto` bytes and returns the rewritten
// model bytes. Signatures mirror the Python `*_cpp` wrappers of the same
// name (see onnxsim/onnx_simplifier.py): options objects carry the same
// fields and defaults, and an absent/undefined optional (n/m, token-id
// lists, input names) means "not given", exactly like the C++ std::nullopt
// it feeds.

async function callModelPass(exportName, model, args = []) {
  const bytes = toBytes(model);
  const runtime = await getRuntime();
  const fn = runtime[exportName];
  if (typeof fn !== "function") {
    throw new Error(`onnxsim: this build has no export '${exportName}' (rebuild the wasm module?)`);
  }
  let result = fn(bytes, ...args);
  // Calibration-driven passes run the model through onnxruntime-web and come
  // back as a Promise in this build; data-free passes return synchronously.
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error(`onnxsim: ${exportName} failed (see stderr output for details)`);
  }
  return new Uint8Array(result);
}

/** Data-free cross-layer equalization preprocessing (no calibration data). */
export async function crossLayerEqualize(model) {
  return callModelPass("onnxsim_cross_layer_equalize", model);
}

/** Dynamic INT8 quantization with a single MatMulIntegerToFloat contrib op. */
export async function quantizeDynamicMatMulIntegerToFloat(model) {
  return callModelPass("onnxsim_quantize_dynamic_matmul_integer_to_float", model);
}

/** Dynamic INT8 quantization of an existing fused Attention node. */
export async function quantizeAttentionDynamic(model) {
  return callModelPass("onnxsim_quantize_attention_dynamic", model);
}

/** INT16 weight-only quantization (per output channel, symmetric). */
export async function quantizeWeightOnlyInt16(model) {
  return callModelPass("onnxsim_quantize_weight_only_int16", model);
}

/** Block-wise INT8 weight-only quantization (block size 32). */
export async function quantizeWeightOnlyInt8Block(model) {
  return callModelPass("onnxsim_quantize_weight_only_int8_block", model);
}

/** OCP Microscaling MXFP4 weight-only quantization. */
export async function quantizeWeightOnlyMxfp4(model) {
  return callModelPass("onnxsim_quantize_weight_only_mxfp4", model);
}

/** Weight-only quantization into ORT's com.microsoft::MatMulNBits op. */
export async function quantizeWeightOnlyMatMulNbits(model) {
  return callModelPass("onnxsim_quantize_weight_only_matmul_nbits", model);
}

/** QLoRA-style double quantization of existing DequantizeLinear scales. */
export async function applyDoubleQuantization(model) {
  return callModelPass("onnxsim_apply_double_quantization", model);
}

/** Any-precision LLM quantization (e.g. 4-bit in an 8-bit-compatible layout). */
export async function applyAnyPrecisionLlm(model, { bits = 4, maxBits = 8, blockSize = 32 } = {}) {
  return callModelPass("onnxsim_apply_any_precision_llm", model, [bits, maxBits, blockSize]);
}

/** QuaRot rotation preprocessing plus INT4 round-to-nearest quantization. */
export async function applyQuarot(model, { seed = 0, blockSize = 32, epsilon = 1e-6 } = {}) {
  return callModelPass("onnxsim_apply_quarot", model, [seed, blockSize, epsilon]);
}

/** IQ4_NL non-uniform 16-entry codebook weight-only quantization. */
export async function applyIq4Nl(model) {
  return callModelPass("onnxsim_apply_iq4_nl", model);
}

/** GGUF Q4_0 weight-only quantization. */
export async function applyGgufQ4_0(model) {
  return callModelPass("onnxsim_apply_gguf_q4_0", model);
}

/** GGUF Q4_1 weight-only quantization. */
export async function applyGgufQ4_1(model) {
  return callModelPass("onnxsim_apply_gguf_q4_1", model);
}

/** GGUF Q5_0 weight-only quantization. */
export async function applyGgufQ5_0(model) {
  return callModelPass("onnxsim_apply_gguf_q5_0", model);
}

/** GGUF Q5_1 weight-only quantization. */
export async function applyGgufQ5_1(model) {
  return callModelPass("onnxsim_apply_gguf_q5_1", model);
}

/** GGUF Q8_0 weight-only quantization. */
export async function applyGgufQ8_0(model) {
  return callModelPass("onnxsim_apply_gguf_q8_0", model);
}

/** GGUF Q2_K weight-only quantization. */
export async function applyGgufQ2K(model) {
  return callModelPass("onnxsim_apply_gguf_q2_k", model);
}

/** GGUF Q3_K weight-only quantization. */
export async function applyGgufQ3K(model) {
  return callModelPass("onnxsim_apply_gguf_q3_k", model);
}

/** GGUF Q4_K weight-only quantization. */
export async function applyGgufQ4K(model) {
  return callModelPass("onnxsim_apply_gguf_q4_k", model);
}

/** GGUF Q5_K weight-only quantization. */
export async function applyGgufQ5K(model) {
  return callModelPass("onnxsim_apply_gguf_q5_k", model);
}

/** GGUF ternary weight-only quantization. */
export async function applyGgufTernary(model) {
  return callModelPass("onnxsim_apply_gguf_ternary", model);
}

/** FP6 LLM weight-only quantization. */
export async function applyFp6Llm(model) {
  return callModelPass("onnxsim_apply_fp6_llm", model);
}

/** GGUF Q6_K weight-only quantization. */
export async function applyGgufQ6K(model) {
  return callModelPass("onnxsim_apply_gguf_q6_k", model);
}

/** AngelSlim's LeptoQuant outlier-aware block FP8 weight quantization. */
export async function applyLeptoquant(model) {
  return callModelPass("onnxsim_apply_leptoquant", model);
}

/** bitsandbytes' NF4 weight-only quantization. */
export async function quantizeWeightOnlyNf4(model) {
  return callModelPass("onnxsim_quantize_weight_only_nf4", model);
}

/** IF4 (per-block INT4/FP4 choice) weight-only quantization. */
export async function quantizeWeightOnlyIf4(model) {
  return callModelPass("onnxsim_quantize_weight_only_if4", model);
}

/** NVIDIA's NVFP4 weight-only quantization. */
export async function quantizeWeightOnlyNvfp4(model) {
  return callModelPass("onnxsim_quantize_weight_only_nvfp4", model);
}

/** DeepSeek-V3-style block FP8 weight quantization (weight side only). */
export async function applyDeepseekFp8(model) {
  return callModelPass("onnxsim_apply_deepseek_fp8", model);
}

/** K-means per-layer codebook weight quantization. */
export async function applyKmeansQuantization(model) {
  return callModelPass("onnxsim_apply_kmeans_quantization", model);
}

/** HQQ (Half-Quadratic Quantization) weight-only quantization. */
export async function applyHqq(model) {
  return callModelPass("onnxsim_apply_hqq", model);
}

/**
 * I-BERT's own i-GELU polynomial approximation of Erf. Unlike every
 * other data-free pass above, this is not a weight quantizer -- it
 * replaces every standalone Erf node with a closed-form approximation.
 */
export async function applyIbertGelu(model) {
  return callModelPass("onnxsim_apply_ibert_gelu", model);
}

/**
 * I-BERT's own integer-friendly Softmax exp-approximation. Also not a
 * weight quantizer -- replaces every standalone Softmax node. Needs
 * opset 18+.
 */
export async function applyIbertSoftmax(model) {
  return callModelPass("onnxsim_apply_ibert_softmax", model);
}

/** AdpQ: calibration-free salient/non-salient weight quantization. */
export async function applyAdpq(model) {
  return callModelPass("onnxsim_apply_adpq", model);
}

/** ICQuant: single-outlier-per-block weight quantization. */
export async function applyIcquant(model) {
  return callModelPass("onnxsim_apply_icquant", model);
}

/** OliVe: outlier-victim pair weight quantization. */
export async function applyOlive(model) {
  return callModelPass("onnxsim_apply_olive", model);
}

/** AQLM: additive/residual multi-codebook weight quantization. */
export async function applyAqlm(model) {
  return callModelPass("onnxsim_apply_aqlm", model);
}

/** Drop-by-Drop: additive multi-bitwidth codebook weight quantization. */
export async function applyDropByDrop(model) {
  return callModelPass("onnxsim_apply_drop_by_drop", model);
}

/** LO-BCQ: block-clustered codebook weight quantization. */
export async function applyLoBcq(model) {
  return callModelPass("onnxsim_apply_lo_bcq", model);
}

/** QuIP#: rotation-based incoherence processing + E8-lattice vector quantization. */
export async function applyQuipSharp(model) {
  return callModelPass("onnxsim_apply_quip_sharp", model);
}

/**
 * Attention computation quantization: per-token dynamic INT8 for the
 * decomposed attention subgraph's own Q/K/V operands, fixed-scale UINT8 for
 * the Softmax output. Unlike every other data-free pass above, this is not
 * a weight quantizer -- none of the four quantized tensors is a constant
 * weight, so this builds new quantize/dequantize graph nodes instead of
 * replacing an initializer. Needs opset 18+.
 */
export async function applyAttentionQuantization(model) {
  return callModelPass("onnxsim_apply_attention_quantization", model);
}

/**
 * ZeroQuant: group-wise INT8 weight quantization paired with per-token
 * dynamic INT8 activation quantization, executed as a real int8 x int8
 * MatMulInteger (unlike every other per-token-dynamic-INT8 pass in this
 * module, which simulates precision loss with an immediate float
 * round-trip instead). Needs opset 18+.
 */
export async function applyZeroquant(model, { blockSize = 32, epsilon = 1e-12 } = {}) {
  return callModelPass("onnxsim_apply_zeroquant", model, [blockSize, epsilon]);
}

/**
 * IntactKV: splits a KV-cache stream's own fixed-length leading pivot
 * prefix into its own always-exact stream, so a following KV-cache
 * quantizer can leave those tokens untouched. A companion pass, not a
 * quantizer -- real graph input/output surgery, not a weight replacement.
 */
export async function applyIntactkv(model) {
  return callModelPass("onnxsim_apply_intactkv", model);
}

/**
 * KBVQ-MoE: KLT-shared-basis plus per-expert bias-corrected vector
 * quantization for a com.microsoft::MoE router group's own experts.
 */
export async function applyKbvqMoe(model) {
  return callModelPass("onnxsim_apply_kbvq_moe", model);
}

/**
 * LLM-FP4: searches, per matched layer's weight, an exponent/mantissa FP4
 * format split (E1M2/E2M1/E3M0) and a per-block real-valued clip-ratio
 * scale minimizing reconstruction MSE.
 */
export async function quantizeWeightOnlyLlmFp4(model) {
  return callModelPass("onnxsim_quantize_weight_only_llm_fp4", model);
}

/**
 * QServe's QoQ quantization: progressive (INT8-then-INT4) block-wise
 * weight quantization, folded into one combined per-group scale.
 */
export async function applyQoq(model) {
  return callModelPass("onnxsim_apply_qoq", model);
}

/**
 * D2Quant's Dual-Scale Quantizer (DSQ): a per-input-channel auxiliary
 * scale for SwiGLU/GLU down-projection weights, absorbed into the paired
 * up-projection's own raw weight.
 */
export async function applyDsq(model) {
  return callModelPass("onnxsim_apply_dsq", model);
}

/**
 * DAQ (Delta-Aware Quantization): unlike every other data-free pass
 * above, this needs TWO models (a base and a fine-tuned checkpoint,
 * matched by node output name), so it does not go through
 * callModelPass's own single-model convenience wrapper.
 */
export async function applyDaq(baseModel, postTrainedModel, { metric = "cosine", skipNames } = {}) {
  const baseBytes = toBytes(baseModel);
  const postBytes = toBytes(postTrainedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_daq;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_daq' (rebuild the wasm module?)");
  }
  let result = fn(baseBytes, postBytes, metric, skipNames);
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_daq failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * Low-Rank Compensation (LoRC): unlike every other data-free pass above,
 * this needs TWO models (a float model and its own INT4-quantized
 * counterpart, matched by node output name), so it does not go through
 * callModelPass's own single-model convenience wrapper.
 */
export async function applyLowRankCompensation(floatModel, quantizedModel, { rank = 8 } = {}) {
  const floatBytes = toBytes(floatModel);
  const quantizedBytes = toBytes(quantizedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_low_rank_compensation;
  if (typeof fn !== "function") {
    throw new Error(
      "onnxsim: this build has no export 'onnxsim_apply_low_rank_compensation' (rebuild the wasm module?)",
    );
  }
  let result = fn(floatBytes, quantizedBytes, rank);
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_low_rank_compensation failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * Embedding-output binarization: data-free, single-model, targets a whole
 * graph OUTPUT declaration rather than a matched node. Binarizes the
 * resolved FLOAT output (sign-thresholded) and packs 8 consecutive
 * elements MSB-first into one uint8 byte. `outputName` omitted requires
 * the graph to have exactly one FLOAT output.
 */
export async function quantizeEmbeddingBinary(model, { outputName } = {}) {
  return callModelPass("onnxsim_quantize_embedding_binary", model, [outputName ?? null]);
}

/**
 * Magnitude pruning (unstructured, or N:M when both `n` and `m` are given).
 * `globalSparsity` pools every layer into one whole-model ranking.
 */
export async function pruneMagnitude(model, { sparsity = 0.5, n, m, globalSparsity = false } = {}) {
  return callModelPass("onnxsim_prune_magnitude", model, [sparsity, n, m, globalSparsity]);
}

/** Structured (channel) pruning of MatMul/Gemm/Conv chains. */
export async function applyStructuredPruning(
  model,
  { sparsity = 0.5, importanceNorm = "l2", globalSparsity = false } = {},
) {
  return callModelPass("onnxsim_apply_structured_pruning", model, [
    sparsity,
    importanceNorm,
    globalSparsity,
  ]);
}

/** Attention-head (or KV-group) pruning of fused self-attention blocks. */
export async function applyAttentionHeadPruning(model, { sparsity = 0.5, importanceNorm = "l2" } = {}) {
  return callModelPass("onnxsim_apply_attention_head_pruning", model, [sparsity, importanceNorm]);
}

/** MoE expert-intermediate-channel (`inter_size`) pruning. */
export async function applyMoeExpertChannelPruning(model, { sparsity = 0.5 } = {}) {
  return callModelPass("onnxsim_apply_moe_expert_channel_pruning", model, [sparsity]);
}

/** QMoE (quantized-weight) expert-intermediate-channel pruning. */
export async function applyQmoeExpertChannelPruning(model, { sparsity = 0.5 } = {}) {
  return callModelPass("onnxsim_apply_qmoe_expert_channel_pruning", model, [sparsity]);
}

/**
 * Embedding vocabulary pruning down to an explicit keep/drop set.
 * Unlike every other pass here the returned model does NOT accept the
 * original token ids -- remap them with the returned `keptTokenIds`
 * (`idMap` is `{keptTokenIds[i]: i}`).
 */
export async function applyEmbeddingVocabPruning(
  model,
  { keepTokenIds, dropTokenIds, inputName } = {},
) {
  const bytes = toBytes(model);
  const runtime = await getRuntime();
  let result = runtime.onnxsim_apply_embedding_vocab_pruning(
    bytes,
    keepTokenIds,
    dropTokenIds,
    inputName,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_embedding_vocab_pruning failed (see stderr output for details)");
  }
  return {
    model: new Uint8Array(result.model),
    matched: !!result.matched,
    keptTokenIds: Array.from(result.keptTokenIds || []),
    lmHeadPruned: !!result.lmHeadPruned,
  };
}

/** Importance-ranked embedding vocabulary pruning (lowest-L2-norm rows go). */
export async function applyEmbeddingVocabMagnitudePruning(
  model,
  { sparsity = 0.5, protectTokenIds, inputName } = {},
) {
  const bytes = toBytes(model);
  const runtime = await getRuntime();
  let result = runtime.onnxsim_apply_embedding_vocab_magnitude_pruning(
    bytes,
    sparsity,
    protectTokenIds,
    inputName,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error(
      "onnxsim: onnxsim_apply_embedding_vocab_magnitude_pruning failed (see stderr output for details)",
    );
  }
  return {
    model: new Uint8Array(result.model),
    matched: !!result.matched,
    keptTokenIds: Array.from(result.keptTokenIds || []),
    lmHeadPruned: !!result.lmHeadPruned,
  };
}

// ---------------------------------------------------------------------------
// Calibration-driven passes (Wanda/SparseGPT/imatrix families, MoE
// whole-expert, transformer-block depth pruning).
//
// `calibration` is one batch or an array of batches; each batch maps a graph
// input name to a tensor given either as an onnxruntime-web Tensor
// (`{ type: "float32"|"int64"|..., dims, data }` -- run the model over sample
// inputs with onnxruntime-web and hand its input/output tensors straight in)
// or in canonical form (`{ dtype, dims, data }` with `dtype` the ONNX
// TensorProto.DataType enum value). Tensor bytes are raw little-endian
// element data: a TypedArray's own bytes cross with no copy, so
// `new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength)`
// (or the TypedArray itself) is already the right `data`.
//
// These run probe sub-models through onnxruntime-web, so they need it
// loaded (like simplify with constantFolding) and always return a Promise.

// onnxruntime-web tensor type string -> ONNX TensorProto.DataType enum value.
const ORT_TYPE_TO_ONNX_DTYPE = {
  float32: 1,
  uint8: 2,
  int8: 3,
  uint16: 4,
  int16: 5,
  int32: 6,
  int64: 7,
  bool: 9,
  float16: 10,
  float64: 11,
  uint64: 13,
};

function toRawBytes(data, what) {
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  throw new TypeError(
    `onnxsim: calibration tensor '${what}' data must be a TypedArray/Uint8Array of raw little-endian bytes`,
  );
}

function normalizeCalibrationTensor(t, name) {
  if (t && typeof t.type === "string" && Array.isArray(t.dims) && t.data) {
    const dtype = ORT_TYPE_TO_ONNX_DTYPE[t.type];
    if (!dtype) {
      throw new TypeError(`onnxsim: unsupported onnxruntime-web tensor type '${t.type}' for input '${name}'`);
    }
    return { dtype, dims: [...t.dims], data: toRawBytes(t.data, name) };
  }
  if (t && typeof t.dtype === "number" && Array.isArray(t.dims) && t.data) {
    return { dtype: t.dtype, dims: [...t.dims], data: toRawBytes(t.data, name) };
  }
  throw new TypeError(
    `onnxsim: calibration tensor '${name}' must be an onnxruntime-web Tensor or { dtype, dims, data }`,
  );
}

function normalizeCalibrationBatches(calibration) {
  if (!calibration) return [];
  const batches = Array.isArray(calibration) ? calibration : [calibration];
  return batches.map((batch) => {
    const entries = batch instanceof Map ? batch.entries() : Object.entries(batch);
    const out = {};
    for (const [name, t] of entries) {
      out[name] = normalizeCalibrationTensor(t, name);
    }
    return out;
  });
}

async function callCalibratedPass(exportName, model, calibration, args = []) {
  return callModelPass(exportName, model, [normalizeCalibrationBatches(calibration), ...args]);
}

/** Calibration-driven (Wanda-style) structured channel pruning. */
export async function applyStructuredWandaPruning(
  model,
  calibration,
  { sparsity = 0.5, epsilon = 1e-8, importanceNorm = "l2", globalSparsity = false } = {},
) {
  return callCalibratedPass("onnxsim_apply_structured_wanda_pruning", model, calibration, [
    sparsity,
    epsilon,
    importanceNorm,
    globalSparsity,
  ]);
}

/** Calibration-driven (Wanda-style) attention-head pruning. */
export async function applyAttentionHeadWandaPruning(
  model,
  calibration,
  { sparsity = 0.5, epsilon = 1e-8, importanceNorm = "l2" } = {},
) {
  return callCalibratedPass("onnxsim_apply_attention_head_wanda_pruning", model, calibration, [
    sparsity,
    epsilon,
    importanceNorm,
  ]);
}

/** SparseGPT unstructured/N:M pruning (Hessian-error-compensated). */
export async function applySparsegptPruning(
  model,
  calibration,
  { sparsity = 0.5, n, m, percdamp = 0.01, procBlockSize = 128 } = {},
) {
  return callCalibratedPass("onnxsim_apply_sparsegpt_pruning", model, calibration, [
    sparsity,
    n,
    m,
    percdamp,
    procBlockSize,
  ]);
}

/** Wanda unstructured/N:M pruning (|W| * activation-norm importance). */
export async function applyWandaPruning(
  model,
  calibration,
  { sparsity = 0.5, n, m, epsilon = 1e-8, globalSparsity = false } = {},
) {
  return callCalibratedPass("onnxsim_apply_wanda_pruning", model, calibration, [
    sparsity,
    n,
    m,
    epsilon,
    globalSparsity,
  ]);
}

/** MoE whole-expert pruning (drops lowest router-weight experts). */
export async function applyMoeWholeExpertPruning(model, calibration, { sparsity = 0.5 } = {}) {
  return callCalibratedPass("onnxsim_apply_moe_whole_expert_pruning", model, calibration, [sparsity]);
}

/** QMoE whole-expert pruning (quantized-weight counterpart). */
export async function applyQmoeWholeExpertPruning(model, calibration, { sparsity = 0.5 } = {}) {
  return callCalibratedPass("onnxsim_apply_qmoe_whole_expert_pruning", model, calibration, [sparsity]);
}

/** Transformer-block (depth) pruning by block-influence similarity. */
export async function applyTransformerBlockPruning(
  model,
  calibration,
  { sparsity = 0.5, numBlocksToDrop } = {},
) {
  return callCalibratedPass("onnxsim_apply_transformer_block_pruning", model, calibration, [
    sparsity,
    numBlocksToDrop,
  ]);
}

/** llama.cpp imatrix INT4 quantization with importance-weighted scales. */
export async function applyImatrixQuantization(
  model,
  calibration,
  { blockSize = 32, numScaleCandidates = 41, scaleLo = 0.4, scaleHi = 1.6, skipNames } = {},
) {
  return callCalibratedPass("onnxsim_apply_imatrix_quantization", model, calibration, [
    blockSize,
    numScaleCandidates,
    scaleLo,
    scaleHi,
    skipNames,
  ]);
}

/**
 * Outlier Suppression Gamma Migration (lossless pre-conditioning ahead
 * of a W8A8 quantizer): folds the migration scale into matched
 * LayerNormalization gamma/bias and compensates downstream MatMul/Gemm
 * weights. Adds zero nodes. Returns a float model.
 */
export async function applyOutlierSuppression(
  model,
  calibration,
  { alpha = 0.5, epsilon = 1e-5 } = {},
) {
  return callCalibratedPass("onnxsim_apply_outlier_suppression", model, calibration, [
    alpha,
    epsilon,
  ]);
}

/**
 * LLM.int8() outlier/float32 + vector-wise INT8 decomposition: outlier
 * channels stay float32, the rest go through MatMulInteger (per-row
 * activation scales, per-output-channel weight scales, uint8
 * activation). Output tensor names are preserved.
 */
export async function applyLlmInt8(
  model,
  calibration,
  { outlierThreshold = 6.0, epsilon = 1e-8 } = {},
) {
  return callCalibratedPass("onnxsim_apply_llm_int8", model, calibration, [
    outlierThreshold,
    epsilon,
  ]);
}

/**
 * SpQR outlier-aware block-wise INT4 quantization: per-element outliers
 * (by Hessian-diagonal-weighted sensitivity) are excluded from their own
 * block's scale and stored as an exact sparse correction.
 */
export async function applySpqr(
  model,
  calibration,
  { blockSize = 16, outlierFraction = 0.01 } = {},
) {
  return callCalibratedPass("onnxsim_apply_spqr", model, calibration, [
    blockSize,
    outlierFraction,
  ]);
}

/**
 * PB-LLM structured mixed-precision binarizer: the `salientRatio`
 * fraction of input channels with the highest Hessian-diagonal-weighted
 * magnitude stay INT8, every other channel is binarized to ~1 bit/element.
 */
export async function quantizeWeightOnlyPbLlm(
  model,
  calibration,
  { salientRatio = 0.15 } = {},
) {
  return callCalibratedPass("onnxsim_quantize_weight_only_pb_llm", model, calibration, [
    salientRatio,
  ]);
}

/**
 * SqueezeLLM sensitivity-weighted per-group codebook (a real
 * GatherND-based graph rewrite, not folded to a single initializer) plus
 * a dense-and-sparse outlier correction.
 */
export async function quantizeWeightOnlySqueezeLlm(
  model,
  calibration,
  { blockSize = 32, bits = 4, outlierFraction = 0.0045, numKmeansIterations = 20 } = {},
) {
  return callCalibratedPass("onnxsim_quantize_weight_only_squeezellm", model, calibration, [
    blockSize,
    bits,
    outlierFraction,
    numKmeansIterations,
  ]);
}

/**
 * BiLLM ~1-bit-average weight binarizer: Hessian-guided salient-column
 * selection, a two-level binary residual approximation for salient
 * columns, plain flat binary for the rest, and OBC-style forward error
 * compensation (reuses GPTQ's own Cholesky-factored-inverse-Hessian
 * mechanism).
 */
export async function applyBillm(
  model,
  calibration,
  { blockSize = 128, percdamp = 0.01, maxSalientSearch = 30 } = {},
) {
  return callCalibratedPass("onnxsim_apply_billm", model, calibration, [
    blockSize,
    percdamp,
    maxSalientSearch,
  ]);
}

/**
 * KV-cache quantization (KIVI/KVQuant): per-channel static INT8 for a
 * matched `Concat(past, new, axis=seq)` stream's own Key-style values,
 * per-token dynamic INT8 (data-free) for Value-style ones (present output
 * name contains ".value", or listed explicitly in `valueOutputNames`).
 */
export async function quantizeKvCache(
  model,
  calibration,
  { valueOutputNames } = {},
) {
  return callCalibratedPass("onnxsim_quantize_kv_cache", model, calibration, [
    valueOutputNames,
  ]);
}

/**
 * OWQ: rescues the top `outlierFraction` OBS-salient columns of an
 * already-`quantizeWeightOnlyInt4`-quantized layer back to exact float32
 * precision via an additive Gather/MatMul/Add correction, reusing GPTQ's
 * own Cholesky-factored-inverse-Hessian mechanism. Takes the float model
 * and its quantized counterpart, like `applyGptq`. Returns the corrected
 * quantized model bytes.
 */
export async function applyOwq(
  floatModel,
  quantizedModel,
  calibration,
  { outlierFraction = 0.01, percdamp = 0.01 } = {},
) {
  const floatBytes = toBytes(floatModel);
  const quantBytes = toBytes(quantizedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_owq;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_owq' (rebuild the wasm module?)");
  }
  let result = fn(
    floatBytes,
    quantBytes,
    normalizeCalibrationBatches(calibration),
    outlierFraction,
    percdamp,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_owq failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * GEAR: low-rank-plus-sparse residual compensation layered on top of
 * `quantizeKvCache`'s own static per-channel INT8 base quantization,
 * applied only to a freshly-produced KV-cache token.
 */
export async function applyGear(
  model,
  calibration,
  { rank = 4, outlierFraction = 0.05 } = {},
) {
  return callCalibratedPass("onnxsim_apply_gear", model, calibration, [
    rank,
    outlierFraction,
  ]);
}

/**
 * RotateKV: fits a per-stream orthogonal rotation from a matched KV-cache
 * stream's own calibration-activation covariance and applies it to both
 * the stream's fresh Key and its compensating Query, exact by
 * construction for any orthogonal rotation.
 */
export async function applyRotateKv(model, calibration) {
  return callCalibratedPass("onnxsim_apply_rotatekv", model, calibration, []);
}

/**
 * SmoothQuant migration (lossless pre-conditioning ahead of a W8A8
 * quantizer): rescales matched weight columns by `s` and inserts a `Mul`
 * dividing the activation by `s`. Returns a float model.
 */
export async function applySmoothQuant(
  model,
  calibration,
  { alpha = 0.5, epsilon = 1e-5 } = {},
) {
  return callCalibratedPass("onnxsim_apply_smoothquant", model, calibration, [
    alpha,
    epsilon,
  ]);
}

/**
 * Outlier Suppression+ shifting and scaling (lossless pre-conditioning
 * ahead of a W8A8 quantizer): recenters activation channels with a `Sub`,
 * rescales with a `Mul`, restores the shift's contribution with an output
 * `Add`. Returns a float model.
 */
export async function applyOutlierSuppressionPlus(
  model,
  calibration,
  { alpha = 0.5, epsilon = 1e-5 } = {},
) {
  return callCalibratedPass("onnxsim_apply_outlier_suppression_plus", model, calibration, [
    alpha,
    epsilon,
  ]);
}

/**
 * GPTQ sequential, Hessian-compensated INT4 rounding: takes the float
 * model and its `quantize_weight_only_int4`-quantized counterpart,
 * reuses the quantized model's own per-block scales, and rewrites its
 * INT4 codes. Returns the optimized quantized model bytes.
 */
export async function applyGptq(
  floatModel,
  quantizedModel,
  calibration,
  { percdamp = 0.01, procBlockSize = 128 } = {},
) {
  const floatBytes = toBytes(floatModel);
  const quantBytes = toBytes(quantizedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_gptq;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_gptq' (rebuild the wasm module?)");
  }
  let result = fn(floatBytes, quantBytes, normalizeCalibrationBatches(calibration), percdamp, procBlockSize);
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_gptq failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * AdaRound: a rectified-sigmoid relaxation of each weight element's
 * floor/ceil rounding decision, optimized by a hand-rolled Adam loop to
 * minimize a layer's own reconstruction error against real calibration
 * activations. Takes the float model and its
 * `quantize_weight_only_int4`-quantized counterpart, like `applyGptq`.
 * Returns the optimized quantized model bytes.
 */
export async function applyAdaround(
  floatModel,
  quantizedModel,
  calibration,
  {
    numIterations = 300,
    learningRate = 0.1,
    regParam = 0.01,
    warmStart = 0.2,
    betaStart = 20.0,
    betaEnd = 2.0,
  } = {},
) {
  const floatBytes = toBytes(floatModel);
  const quantBytes = toBytes(quantizedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_adaround;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_adaround' (rebuild the wasm module?)");
  }
  let result = fn(
    floatBytes,
    quantBytes,
    normalizeCalibrationBatches(calibration),
    numIterations,
    learningRate,
    regParam,
    warmStart,
    betaStart,
    betaEnd,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_adaround failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * Qronos: a sequential, whole-model generalization of `applyGptq` that
 * additionally accounts for the error already baked into a layer's
 * activations because upstream layers were quantized first, not just
 * this layer's own rounding -- processes layers in the float model's
 * own node order, re-probing the progressively-corrected quantized
 * model before each subsequent layer. Returns the optimized quantized
 * model bytes.
 */
export async function applyQronos(
  floatModel,
  quantizedModel,
  calibration,
  { percdamp = 0.01, procBlockSize = 128 } = {},
) {
  const floatBytes = toBytes(floatModel);
  const quantBytes = toBytes(quantizedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_qronos;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_qronos' (rebuild the wasm module?)");
  }
  let result = fn(floatBytes, quantBytes, normalizeCalibrationBatches(calibration), percdamp, procBlockSize);
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_qronos failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * TesseraQ: "Progressive Adaptive Rounding" (PAR) -- an AdaRound-style
 * rectified-sigmoid rounding relaxation, optimized by a hand-rolled Adam
 * loop jointly with each weight block's own dequantization scale (in
 * log-space), with a coarse-to-fine element-by-element hardening
 * schedule across `parRounds` rounds. Takes the float model and its
 * `quantize_weight_only_int4`-quantized counterpart, like `applyGptq`.
 * Returns the optimized quantized model bytes.
 */
export async function applyTesseraq(
  floatModel,
  quantizedModel,
  calibration,
  {
    numBits = 4,
    numIterations = 400,
    parRounds = 4,
    learningRate = 0.1,
    scaleLearningRate = 0.01,
    regParam = 0.01,
    warmStart = 0.2,
    betaStart = 20.0,
    betaEnd = 2.0,
  } = {},
) {
  const floatBytes = toBytes(floatModel);
  const quantBytes = toBytes(quantizedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_tesseraq;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_tesseraq' (rebuild the wasm module?)");
  }
  let result = fn(
    floatBytes,
    quantBytes,
    normalizeCalibrationBatches(calibration),
    numBits,
    numIterations,
    parRounds,
    learningRate,
    scaleLearningRate,
    regParam,
    warmStart,
    betaStart,
    betaEnd,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_tesseraq failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * AWQ grid-searched per-channel weight rescaling: takes the float model
 * and its `quantize_weight_only_int4`-quantized counterpart, reuses the
 * quantized model's structure, and rewrites improved layers (INT4
 * weight/scale plus a compensating `Mul`). Returns the optimized
 * quantized model bytes.
 */
export async function applyAwq(
  floatModel,
  quantizedModel,
  calibration,
  { numAlphaSteps = 20 } = {},
) {
  const floatBytes = toBytes(floatModel);
  const quantBytes = toBytes(quantizedModel);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_awq;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_awq' (rebuild the wasm module?)");
  }
  let result = fn(floatBytes, quantBytes, normalizeCalibrationBatches(calibration), numAlphaSteps);
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_awq failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

/**
 * GPTVQ: a genuine combination of `applyGptq`'s own sequential,
 * Hessian-compensated correction with a k-means-fit vector codebook --
 * small groups of consecutive input-channel columns of every matched
 * MatMul/vanilla-Gemm node's constant 2-D FLOAT32 weight are jointly
 * quantized against the codebook, then each group's resulting per-column
 * residual is propagated into every not-yet-quantized column exactly like
 * `applyGptq`'s own per-column correction. Rewires only the matched
 * node's weight input (Gather+Reshape[+Transpose]); the node itself,
 * including any bias, is left otherwise unchanged. Returns the
 * quantized model bytes.
 */
export async function applyGptvq(
  model,
  calibration,
  { seed = 0, vectorDim = 2, numCentroids = 256, numIterations = 10, percdamp = 0.01, skipNames } = {},
) {
  return callCalibratedPass("onnxsim_apply_gptvq", model, calibration, [
    seed,
    vectorDim,
    numCentroids,
    numIterations,
    percdamp,
    skipNames,
  ]);
}

/**
 * QuaRot+GPTQ: the real QuaRot paper's optional, tighter weight quantizer --
 * identical to `applyQuarot` (same per-layer random rotation, same
 * data-free per-token INT4 activation quantization) except the weight is
 * quantized via `applyGptq`'s own Hessian-compensated column algorithm,
 * evaluated in the rotated activation space, instead of round-to-nearest.
 * Unlike `applyGptq`/`applyAwq`, takes a single model (this pass derives
 * its own rotation and quantizes from scratch, like `applyQuarot`).
 * Returns the rotated-and-quantized model bytes.
 */
export async function applyQuarotGptq(
  model,
  calibration,
  { seed = 0, blockSize = 32, percdamp = 0.01, procBlockSize = 128, epsilon = 1e-12 } = {},
) {
  const bytes = toBytes(model);
  const runtime = await getRuntime();
  const fn = runtime.onnxsim_apply_quarot_gptq;
  if (typeof fn !== "function") {
    throw new Error("onnxsim: this build has no export 'onnxsim_apply_quarot_gptq' (rebuild the wasm module?)");
  }
  let result = fn(
    bytes,
    normalizeCalibrationBatches(calibration),
    seed,
    blockSize,
    percdamp,
    procBlockSize,
    epsilon,
  );
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (!result) {
    throw new Error("onnxsim: onnxsim_apply_quarot_gptq failed (see stderr output for details)");
  }
  return new Uint8Array(result);
}

export default {
  simplify,
  versions,
  crossLayerEqualize,
  quantizeDynamicMatMulIntegerToFloat,
  quantizeAttentionDynamic,
  quantizeWeightOnlyInt16,
  quantizeWeightOnlyInt8Block,
  quantizeWeightOnlyMxfp4,
  quantizeWeightOnlyMatMulNbits,
  applyDoubleQuantization,
  applyAnyPrecisionLlm,
  applyQuarot,
  applyIq4Nl,
  applyGgufQ4_0,
  applyGgufQ4_1,
  applyGgufQ5_0,
  applyGgufQ5_1,
  applyGgufQ8_0,
  applyGgufQ2K,
  applyGgufQ3K,
  applyGgufQ4K,
  applyGgufQ5K,
  applyGgufTernary,
  applyFp6Llm,
  applyGgufQ6K,
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
  pruneMagnitude,
  applyStructuredPruning,
  applyAttentionHeadPruning,
  applyMoeExpertChannelPruning,
  applyQmoeExpertChannelPruning,
  applyEmbeddingVocabPruning,
  applyEmbeddingVocabMagnitudePruning,
  applyStructuredWandaPruning,
  applyAttentionHeadWandaPruning,
  applySparsegptPruning,
  applyWandaPruning,
  applyMoeWholeExpertPruning,
  applyQmoeWholeExpertPruning,
  applyTransformerBlockPruning,
  applyImatrixQuantization,
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
  applyQuarotGptq,
  applyGptvq,
  applySmoothQuant,
};
