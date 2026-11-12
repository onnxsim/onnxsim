#pragma once

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"
// Pulls in EmbeddingVocabPruningResult (needed below by
// ApplyEmbeddingVocabPruning/ApplyEmbeddingVocabMagnitudePruning's own
// declarations) -- unlike every other entry point in this header, which
// duplicates a bare `onnx::ModelProto`-returning prototype verbatim rather
// than including its own home header, sharing a single struct definition
// (rather than a byte-for-byte duplicate struct body in each header) avoids
// two independently-edited copies of the same type ever drifting apart.
// finetune_entry.h/lora_entry.h are always built (see
// ONNXSIM_NON_CORE_FEATURE_SOURCES in CMakeLists.txt -- QAT/LoRA/fine-tuning
// are not part of what ONNXSIM_NON_CORE_FEATURES controls), so their own
// includes stay outside the guarded block below.
#include "finetune_entry.h"
#include "lora_entry.h"

#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
#include "adaquant_entry.h"
#include "adaround_entry.h"
#include "affinequant_entry.h"
#include "autoround_entry.h"
#include "awq_entry.h"
#include "billm_entry.h"
#include "brecq_entry.h"
#include "bwa_ptq_entry.h"
#include "dac_entry.h"
#include "daq_entry.h"
#include "duquant_entry.h"
#include "easyquant_entry.h"
#include "embedding_quantization_entry.h"
#include "flexround_entry.h"
#include "foem_entry.h"
#include "fptq_entry.h"
#include "gear_entry.h"
#include "gptaq_entry.h"
#include "gptq_entry.h"
#include "gptvq_entry.h"
#include "imatrix_quant_entry.h"
#include "kv_cache_quantization_entry.h"
#include "llm_fp4_activation_entry.h"
#include "llm_int8_entry.h"
#include "low_rank_compensation_entry.h"
#include "lqer_entry.h"
#include "mixed_precision_entry.h"
#include "moequant_entry.h"
#include "norm_tweaking_entry.h"
#include "omniquant_entry.h"
#include "outlier_suppression_entry.h"
#include "outlier_suppression_plus_entry.h"
#include "owq_entry.h"
#include "paroquant_entry.h"
#include "pb_llm_entry.h"
#include "ptq4vit_entry.h"
#include "qronos_entry.h"
#include "quantease_entry.h"
#include "quarot_gptq_entry.h"
#include "rotatekv_entry.h"
#include "rptq_entry.h"
#include "slim_llm_entry.h"
#include "smooth_attention_entry.h"
#include "smoothquant_entry.h"
#include "spinquant_entry.h"
#include "spqr_entry.h"
#include "squeezellm_entry.h"
#include "structured_pruning_entry.h"
#include "svdquant_entry.h"
#include "tesseraq_entry.h"

#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// RAII owner for a DLManagedTensor: releasing it invokes the tensor's own
// DLPack deleter exactly once (per the DLPack contract), which frees whatever
// the producer attached -- a borrowed-buffer no-op, an Ort::Value, a host
// allocation, etc. Move-only.
struct DLManagedTensorDeleter {
  void operator()(DLManagedTensor* t) const {
    if (t != nullptr && t->deleter != nullptr) {
      t->deleter(t);
    }
  }
};
using DLManagedTensorPtr =
    std::unique_ptr<DLManagedTensor, DLManagedTensorDeleter>;

// The constant-folding executor boundary. onnxsim runs each fold group by
// building a throwaway sub-model and asking an executor to evaluate it. Tensors
// cross this boundary as DLPack DLManagedTensors rather than onnx::TensorProto,
// so an executor can borrow onnxsim's buffers (and hand its results back)
// without a protobuf serialize/parse round trip. This is also the seam an
// embedder implements to plug in its own ONNX runtime (see the C ABI executor
// callback in capi/onnxsim_c_api.h, and docs/dlpack-executor.md).
struct ModelExecutor {
  virtual ~ModelExecutor() = default;

  // Evaluate `model`, whose graph inputs are fed by `inputs` (positional, i.e.
  // inputs[i] feeds model.graph().input(i)), and return one tensor per graph
  // output (positional, matching model.graph().output()).
  //
  // Ownership: `inputs` are BORROWED for the duration of the call -- the
  // executor must not retain them past return. Each returned DLManagedTensorPtr
  // is freshly owned by the caller. Tensors are CPU, contiguous, and in host
  // byte order (raw_data's little-endian layout is converted at the DLPack
  // boundary -- see dlpack_bridge.h).
  //
  // public for pybind11 / nanobind trampolines
  virtual std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const = 0;
};

// A user-supplied whole-graph rewriter. When one is passed to ``Simplify`` it
// is run inside the simplification fixed point, letting Python code (for
// example an
// ``onnxscript.rewriter`` rule set) rewrite the model between the optimizer and
// constant-folding rounds so a rewrite can unlock further simplification and
// vice versa. Passing ``nullptr`` (the default) leaves simplification behaviour
// exactly as before.
struct GraphRewriter {
  virtual ~GraphRewriter() = default;

  // Rewrite ``model`` in place. Returns ``true`` if the model was changed and
  // ``false`` if the rewriter left it untouched -- in the latter case ``model``
  // is not modified, so callers can skip re-copying it. Being able to report
  // "nothing changed" lets a rewriter that matched no rule (for example an
  // ``onnxscript.rewriter`` rule set whose patterns did not fire) avoid parsing
  // and copying a fresh, identical ModelProto back on every fixed-point round.
  // public it for pybind11
  virtual bool _Run(onnx::ModelProto& model) const = 0;
};

void InitEnv();

#ifdef ONNXSIM_HAS_ORT
// Returns the built-in model executor backed by ONNX Runtime. Only available
// when onnxsim is built with the built-in ONNX Runtime.
std::shared_ptr<const ModelExecutor> GetBuiltinModelExecutor();
#endif

#ifdef ONNXSIM_HAS_XNNPACK
// Returns a model executor backed by Google's XNNPACK (see
// onnxsim/xnnpack_executor.h and docs/dlpack-executor.md). Only available
// when onnxsim is built with ONNXSIM_BUILTIN_XNNPACK. Unlike
// GetBuiltinModelExecutor, this executor supports only a small, explicit
// subset of ops (see onnxsim/onnx_to_xnnpack_subgraph.h) -- Run() throws
// std::runtime_error for anything else, so it is meant as an alternative,
// explicitly-opted-into backend (e.g. for testing XNNPACK embeddability),
// not a general-purpose drop-in replacement for the ORT-backed executor.
std::shared_ptr<const ModelExecutor> GetXnnpackModelExecutor();
#endif

// ``target_opset_version``, when set, converts the model to that opset version
// of the default ONNX domain (using onnx's version converter) before
// simplifying, so the simplifier can clean up any redundant nodes the
// conversion introduces. std::nullopt leaves the opset version unchanged.
// ``initializers_as_constants`` (default true) controls whether graph
// initializers are treated as constant tensors during simplification. With the
// default, initializers are constants: constant folding materializes nodes that
// depend only on them, and the onnx optimizer's value-baking passes (e.g.
// fuse_bn_into_conv) may fold them. When set to false, initializers are treated
// as non-constant, so nodes rooted only at initializers are left in the graph
// and their weights survive simplification as tunable tensors; ``Constant``
// nodes are still treated as constants either way.
// ``include_inline_functions`` (default false) inlines the model's local
// (model-defined) functions into the main graph before simplifying, via onnx's
// inliner. This flattens function calls into plain ops so the optimizer, shape
// inference and constant folding can see through them; schema-defined
// (built-in) functions are left alone. With the default the model's functions
// are left untouched.
// ``mutable_initializer`` (default true, i.e. skip) additionally folds an
// initializer that also appears as a graph input, like any other constant,
// when set to false -- see RemoveInitializerFromInput's own comment.
// ``overwrite_input_shapes``, when set, overwrites the named graph inputs'
// shape dims with the given values (a non-positive entry keeps the original,
// possibly dynamic, dimension). ``unused_output``, when set, drops the named
// graph outputs before simplification so dead-end elimination cleans up
// nodes that only fed them. Both throw std::runtime_error if a name does
// not match an existing graph input/output.
// ``extra_optimizers``, when set, runs the named onnx-optimizer passes in
// addition to the default fuse/elimination set -- the counterpart to
// ``skip_optimizers``. This is how a pass registered as ``PassType::Other``
// (excluded from the default set because it is a graph-shape rewrite rather
// than a pure node reduction or fusion, e.g. a defusion that trades a
// backend-specific op for a more portable but larger equivalent) gets
// opted into, without changing what runs by default for every other caller.
// Has no effect when ``skip_optimizers`` is ``std::nullopt`` (which disables
// optimization entirely). An unknown pass name throws (surfaced from
// onnx-optimizer's own pass registry lookup) rather than being silently
// ignored, since -- unlike a typo in ``skip_optimizers``, which just means
// nothing new is skipped -- a typo here means the caller's requested pass
// silently never runs.
onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version = std::nullopt,
    const GraphRewriter* rewriter = nullptr,
    bool initializers_as_constants = true,
    bool include_inline_functions = false, bool mutable_initializer = true,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes = std::nullopt,
    const std::optional<std::vector<std::string>>& unused_output = std::nullopt,
    const std::optional<std::vector<std::string>>& extra_optimizers =
        std::nullopt);

// Same as ``Simplify`` above, except ``model`` is taken by mutable reference
// and its initializers' raw tensor bytes are moved out (via the same
// move-based ModelProto -> Graph -> ModelProto round trip already used
// internally for shape inference / constant folding's own resident Graph)
// instead of deep-copied into the working copy the fixed point runs on. For
// a model whose weights dominate its size, this roughly halves
// ``Simplify``'s own peak memory (see
// docs/bench/RESULTS_synthetic_decoder_oom.md for measurements and
// docs/bench/TODO_large_decoder_submodule_oom.md for the original report this
// traces back to).
//
// Only call this when ``model`` is about to be discarded or overwritten by
// the caller -- afterward its initializers are left with empty raw data
// (structure otherwise intact: shapes, names, node list, doc strings, ...).
// ``SimplifyPath`` uses this for exactly that reason: its own ``model`` is
// immediately overwritten by the result and never read again beforehand.
onnx::ModelProto SimplifyConsumeInput(
    const ModelExecutor& executor, onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version = std::nullopt,
    const GraphRewriter* rewriter = nullptr,
    bool initializers_as_constants = true,
    bool include_inline_functions = false, bool mutable_initializer = true,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes = std::nullopt,
    const std::optional<std::vector<std::string>>& unused_output = std::nullopt,
    const std::optional<std::vector<std::string>>& extra_optimizers =
        std::nullopt);

// Debugging helpers: run a *single* one of the transforms that ``Simplify``
// otherwise drives to a fixed point, once, on a copy of ``model``, and return
// the result. They let a caller inspect the isolated effect of a step (e.g. the
// WASM converter's "run a single feature" panel) instead of the whole
// fixed-point simplification. The input model is never mutated.
//
// ``InferShapesOnce`` runs ONNX shape inference (populates value_info / output
// types). ``PropagateDataOnce`` runs onnxsim's partial-shape / data-propagation
// pass, which rewrites nodes whose output value became statically known into
// ``Constant`` nodes. ``FoldConstantOnce`` runs the same partial-shape pass and
// then one constant-folding round through ``executor`` (so it needs a model
// executor, exactly like ``Simplify``); ``tensor_size_threshold`` caps the size
// of tensors that folding may materialize, matching ``Simplify``'s parameter.
onnx::ModelProto InferShapesOnce(const onnx::ModelProto& model);
onnx::ModelProto PropagateDataOnce(const onnx::ModelProto& model);
onnx::ModelProto FoldConstantOnce(const ModelExecutor& executor,
                                  const onnx::ModelProto& model,
                                  size_t tensor_size_threshold,
                                  bool initializers_as_constants = true);

// Cross-Layer Equalization (CLE) -- the data-free weight-equalization
// preprocessing technique from "Data-Free Quantization Through Weight
// Equalization and Bias Correction" (Nagel et al., 2019), also shipped as
// part of Qualcomm's AIMET toolkit. This is NOT a quantization scheme --
// no Quantize/DequantizeLinear node is ever introduced, and every value
// this produces is bit-for-bit the same computation as the input model, just
// reparameterized -- it is meant to run *before* a quantize_* function, to
// make the per-tensor or per-channel quantization that follows more
// accurate.
//
// For every pair of adjacent Conv layers Conv1 -> [activation] -> Conv2
// where the activation (if any) is positive-homogeneous of degree 1 --
// f(a*x) = a*f(x) for every a > 0, true of Relu/PRelu/LeakyRelu and
// trivially true of "no activation at all" -- and both convs have `group`
// == 1, rescales each shared channel c by S[c] = sqrt(r1[c] / r2[c])
// (r1[c]/r2[c] being Conv1's/Conv2's own per-channel weight range):
// Conv1's weight/bias for channel c divided by S[c], Conv2's weight for
// channel c multiplied by S[c]. This makes the two layers' per-channel
// weight ranges identical (the most balanced a fixed pair can be), without
// changing the composed function at all -- the activation's positive
// homogeneity is exactly what lets S[c] and 1/S[c] cancel across it. See
// ``passes/cross_layer_equalization.h`` for the full derivation, the
// documented scope limitations (Conv only, no ConvTranspose or Gemm/MatMul,
// FLOAT32 only), and why one call already equalizes a whole chain of layers
// (not just one adjacent pair) via onnxsim's fixed-point pass driver.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this one rewrite,
// repeated to a fixed point, to a copy of ``model`` (which is left
// untouched) and returns the result.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto CrossLayerEqualize(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Dynamically quantizes every MatMul, and every "vanilla" Gemm (transA=0,
// alpha=1, beta=1), whose weight is a constant 2-D float32 tensor: the weight
// is quantized to INT8 ahead of time (per output channel, symmetric, from its
// static values -- no calibration data needed), while the activation is
// quantized to uint8 in the graph itself via ``DynamicQuantizeLinear``, which
// computes its own scale/zero-point from each run's actual input range. This
// mirrors the "dynamic quantization" scheme ONNX Runtime's
// ``quantize_dynamic`` applies to MatMul/Gemm. See
// ``passes/dynamic_quantize_matmul.h`` for the rewrite itself.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this one rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match (dynamic or non-2-D weights, non-default Gemm
// attributes, non-float32 operands, an opset older than 11) are left as-is.
onnx::ModelProto QuantizeDynamic(const onnx::ModelProto& model);

// Same rewrite as ``QuantizeDynamic`` -- same matching rules, same weight
// quantization, same runtime ``DynamicQuantizeLinear`` activation
// quantization -- but the dequantize step is a single ONNX Runtime
// "com.microsoft" contrib op, ``MatMulIntegerToFloat``, instead of
// ``QuantizeDynamic``'s three-to-four separate standard-ONNX nodes
// (``MatMulInteger`` + ``Cast`` + two ``Mul``s + an optional ``Add``):
// ``MatMulIntegerToFloat``'s own schema dequantizes and adds an optional
// bias directly, so this needs only ``DynamicQuantizeLinear`` plus the one
// contrib op. This adds "com.microsoft" (version 1) to the model's opset
// imports the first time it rewrites a node -- the one respect in which the
// result is less portable than ``QuantizeDynamic``'s pure-standard-ONNX
// output. See ``passes/dynamic_quantize_matmul_integer_to_float.h`` for the
// rewrite itself.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this one rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match (dynamic or non-2-D weights, non-default Gemm
// attributes, non-float32 operands, an opset older than 11) are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeDynamicMatMulIntegerToFloat(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Dynamically quantizes an existing "com.microsoft" ``Attention`` node (see
// ``passes/fuse_attention.h`` -- this does not fuse attention itself, it
// expects one to already be present) into ``Attention``'s quantized
// counterpart, ``QAttention``: the merged Q/K/V weight is quantized to INT8
// ahead of time (per output channel, symmetric, from its static values --
// no calibration data needed), while the activation is quantized to uint8 in
// the graph itself via ``DynamicQuantizeLinear``, mirroring
// ``QuantizeDynamic``'s own scheme. See ``passes/dynamic_quantize_attention.h``
// for the rewrite itself, including why an uneven ``qkv_hidden_sizes`` split
// (V's hidden size differing from Q/K's, which plain ``Attention`` allows) is
// declined.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding,
// fuse_attention, or any other pass -- it applies exactly this one rewrite,
// once, to a copy of ``model`` (which is left untouched) and returns the
// result. Call ``Simplify`` first to produce ``Attention`` nodes to quantize
// if the input model doesn't already have any. Nodes that do not match (no
// ``Attention`` node, a non-constant or non-2-D weight, a non-float32
// operand, an opset older than 11, or an uneven ``qkv_hidden_sizes`` split)
// are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeAttentionDynamic(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Dynamically quantizes every MatMul/"vanilla" Gemm whose constant weight is
// *structurally ternary* -- every element of every output column is one of
// {-s, 0, +s} for that column's own scale ``s``, the representation BitNet
// b1.58 (https://github.com/microsoft/BitNet) and similar ternary-weight
// models use internally, which a generic ONNX export still stores as a dense
// float32 initializer. Detected nodes get exactly the ``QuantizeDynamic``
// rewrite (``DynamicQuantizeLinear`` + ``MatMulInteger`` + dequantize), except
// the weight's INT8 encoding is a lossless {-1, 0, 1} code instead of a
// rounded approximation of its full range. Nodes whose weight is not
// structurally ternary are left untouched by this call -- combine with
// ``QuantizeDynamic`` (which fires on any constant float32 weight, ternary or
// not) if a model mixes ternary and ordinary layers and both should be
// quantized. See ``passes/dynamic_quantize_ternary_matmul.h`` for the rewrite
// itself and the rewrite's relationship to ``QuantizeDynamic``.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this one rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeTernary(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Weight-only quantizes every MatMul, every "vanilla" Gemm (transA=0,
// alpha=1, beta=1), and every Conv, whose weight is a constant float32
// tensor (2-D for MatMul/Gemm, rank >= 3 for Conv): the weight is quantized
// to INT8 ahead of time (per output channel, symmetric, from its static
// values -- same as ``QuantizeDynamic``/``QuantizeStatic``), inserting a
// single ``DequantizeLinear`` in its place. Unlike both of those, the
// activation is never touched -- no ``DynamicQuantizeLinear``, no
// QuantizeLinear/DequantizeLinear pair, no calibration data of any kind --
// so this only shrinks the model's weight storage; it does not change
// activation precision or add any runtime quantize/dequantize cost on the
// activation path. This mirrors the "weight-only quantization" scheme most
// real-world weight-heavy ONNX deployments (large linear/embedding layers in
// transformer-style decoders, for example) actually ship, as opposed to full
// activation quantization. See ``passes/weight_only_quantize_matmul.h`` and
// ``passes/weight_only_quantize_conv.h`` for the rewrites themselves.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly these rewrites, once
// each, to a copy of ``model`` (which is left untouched) and returns the
// result. Nodes that do not match (dynamic or unsupported-rank weights,
// non-default Gemm attributes, non-float32 operands, an opset older than 13)
// are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeWeightOnly(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Block-wise INT4 weight-only quantizes every MatMul, every "vanilla" Gemm
// (transA=0, alpha=1, beta=1), and every Conv, whose weight is a constant
// float32 tensor whose flattened reduction size (K for MatMul/Gemm --
// transposed [N, K] or not; Cin/groups * prod(kernel dims) for Conv) is
// evenly divisible by 32: the weight is quantized to INT4 (values in
// [-7, 7]) with a separate symmetric scale per 32-element block of that
// reduction, per output channel, inserting a single
// ``DequantizeLinear(axis=..., block_size=32)`` in its place (Conv's weight
// is flattened to 2-D for this, then a ``Reshape`` restores its original
// shape -- see ``passes/weight_only_quantize_int4_conv.h``). Like
// ``QuantizeWeightOnly``, the activation is never touched -- no calibration
// data, no runtime quantize/dequantize cost on the activation path -- but at
// roughly half the storage for a comparable accuracy cost, since block-local
// scales absorb most of what a single wider INT8 per-channel range would
// otherwise lose. Uses ONNX opset 21's INT4 tensor type and
// DequantizeLinear's `block_size` attribute (standard ONNX, not a contrib
// op). See ``passes/weight_only_quantize_int4_matmul.h`` and
// ``passes/weight_only_quantize_int4_conv.h`` for the rewrites themselves.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly these rewrites, once
// each, to a copy of ``model`` (which is left untouched) and returns the
// result. Nodes that do not match (dynamic or unsupported-rank weights, a
// reduction size not divisible by 32, non-default Gemm attributes,
// non-float32 operands, an opset older than 21) are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeWeightOnlyInt4(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Weight-only quantizes every MatMul, and every "vanilla" Gemm (transA=0,
// alpha=1, beta=1), whose weight is a constant 2-D float32 tensor, to ONNX
// Runtime's ``com.microsoft::MatMulNBits`` contrib op -- a *vendor-specific*
// counterpart to ``QuantizeWeightOnlyInt4``: same INT4, same 32-element
// block-wise scale, but packed into ORT's own single fused op (the format
// ORT's own GenAI/quantization tooling emits for LLM/ASR weight
// compression) instead of ``QuantizeWeightOnlyInt4``'s portable, standard
// ONNX opset-21 INT4-tensor-plus-``DequantizeLinear`` pair. Smaller and
// faster on ONNX Runtime specifically, at the cost of needing ORT (or
// another runtime implementing this contrib op) to run at all -- unlike
// every other ``Quantize*`` function here, the result does not load on an
// arbitrary conformant ONNX runtime. See
// ``passes/weight_only_quantize_matmul_nbits.h`` for the rewrite itself,
// including the exact bit-packing format.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this one rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match (dynamic or non-2-D weights, non-default Gemm
// attributes, non-float32 operands) are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeWeightOnlyMatMulNBits(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// INT16 weight-only quantizes every MatMul, every "vanilla" Gemm (transA=0,
// alpha=1, beta=1), and every Conv, whose weight is a constant float32
// tensor: the weight is quantized to INT16 (per output channel, symmetric,
// scale = max(|w|) / 32767) with a single ``DequantizeLinear(axis=...)`` in
// its place. Like ``QuantizeWeightOnly``, the activation is never touched --
// no calibration data, no runtime quantize/dequantize cost on the activation
// path -- and this uses the exact same per-channel scheme, just with INT16's
// ~8x finer step (1/32767 relative) instead of INT8's 1/127. That extra
// resolution matters specifically for channels with a few extreme-outlier
// weights, where INT8's coarser step would leave the channel's *typical*
// (median-magnitude) weight rounding to within one quantization step of zero
// -- effectively lost; ``estimate_quantization_precision`` (see
// ``precision_estimator.py``'s Python-side ``max_outlier_ratio`` check) flags
// exactly this case and recommends INT16 as one fix. The tradeoff: INT16 is
// only ~2x smaller than float32 (INT8 is ~4x), so this is meant for the
// specific outlier-heavy weights ``QuantizeWeightOnly``'s INT8 handles
// poorly, not as a blanket replacement for it. Uses ONNX opset 21's INT16
// QuantizeLinear/DequantizeLinear type support (standard ONNX, not a contrib
// op). See ``passes/weight_only_quantize_int16_matmul.h`` and
// ``passes/weight_only_quantize_int16_conv.h`` for the rewrites themselves.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly these rewrites, once
// each, to a copy of ``model`` (which is left untouched) and returns the
// result. Nodes that do not match (dynamic or unsupported-rank weights,
// non-default Gemm attributes, non-float32 operands, an opset older than 21)
// are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeWeightOnlyInt16(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Block-wise INT8 weight-only quantizes every MatMul, every "vanilla" Gemm
// (transA=0, alpha=1, beta=1), and every Conv, whose weight is a constant
// float32 tensor whose flattened reduction size (K for MatMul/Gemm --
// transposed [N, K] or not; Cin/groups * prod(kernel dims) for Conv) is
// evenly divisible by 32: the weight is quantized to INT8 (values in
// [-127, 127]) with a separate symmetric scale per 32-element block of that
// reduction, per output channel, inserting a single
// ``DequantizeLinear(axis=..., block_size=32)`` in its place (Conv's weight
// is flattened to 2-D for this, then a ``Reshape`` restores its original
// shape -- see ``passes/weight_only_quantize_int8_block_conv.h``). Sits
// between ``QuantizeWeightOnly``'s single per-channel INT8 scale (coarser,
// no block overhead) and ``QuantizeWeightOnlyInt4``'s block-wise INT4
// (finer blocks, but only 15 representable codes per block): the same
// storage as ``QuantizeWeightOnly`` (INT8 codes are still 1 byte each; only
// the scale tensor grows, from one float per channel to one float per
// (block, channel) pair) with resolution closer to a per-block scheme.
// Like ``QuantizeWeightOnly``, the activation is never touched -- no
// calibration data, no runtime quantize/dequantize cost on the activation
// path. Uses ONNX opset 21's ``DequantizeLinear`` `block_size` attribute
// (standard ONNX, not a contrib op) -- the same opset floor as
// ``QuantizeWeightOnlyInt4``, even though plain INT8 itself needs only
// opset 13. See ``passes/weight_only_quantize_int8_block_matmul.h`` and
// ``passes/weight_only_quantize_int8_block_conv.h`` for the rewrites
// themselves.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly these rewrites, once
// each, to a copy of ``model`` (which is left untouched) and returns the
// result. Nodes that do not match (dynamic or unsupported-rank weights, a
// reduction size not divisible by 32, non-default Gemm attributes,
// non-float32 operands, an opset older than 21) are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeWeightOnlyInt8Block(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// OCP Microscaling MXFP4 weight-only quantizes every MatMul and every
// "vanilla" Gemm (transA=0, alpha=1, beta=1) whose weight is a constant
// float32 tensor whose reduction dimension K is evenly divisible by 32
// (the OCP MX spec's own canonical block size). Unlike every other
// ``QuantizeWeightOnly*`` scheme, MXFP4's per-block scale is constrained to
// a pure power of two, and its 4-bit codes follow a fixed, non-uniform
// (E2M1 floating-point) codebook rather than an ordinary affine range --
// ONNX has no native MX tensor type, so the rewrite builds the
// dequantization out of ordinary opset-11+ ops (``Gather`` a codebook,
// ``Mul`` by the per-block scale) instead of a single ``DequantizeLinear``.
// The weight is quantized from its own static values only -- no calibration
// data, and (like every ``QuantizeWeightOnly*`` scheme) the activation is
// never touched. See ``passes/weight_only_quantize_mxfp4_matmul.h`` and
// ``passes/quantize_mxfp4_common.h`` for the rewrite itself and the
// format's own definition.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this rewrite, once, to
// a copy of ``model`` (which is left untouched) and returns the result. A
// layer with a non-constant, non-2-D weight, or a reduction dimension not
// divisible by 32, is left untouched; a model with no matching layer is
// returned unchanged.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeWeightOnlyMXFP4(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Applies QLoRA-style double quantization (Dettmers et al., 2023, Section
// 3.2) to every ``DequantizeLinear`` node already present in ``model`` whose
// scale input is a constant float32 tensor with at least 64 values (a
// per-block or per-channel scale -- a single scalar per-tensor scale isn't
// worth the overhead of a second quantizer around it). Unlike every
// ``Quantize*`` scheme above, this has no "live weight" of its own to
// quantize: it is a second pass over an *already-quantized* model, and
// composes with any of them (or any other model containing
// ``DequantizeLinear`` nodes) unchanged. See ``passes/double_quantization.h``
// for the exact rewrite.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this rewrite, to every
// matching node, to a copy of ``model`` (which is left untouched) and
// returns the result. A scale that is not a constant initializer, not
// float32, or too small, is left untouched; a model with no matching node is
// returned unchanged.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyDoubleQuantization(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Magnitude pruning (Han et al., 2015) -- the data-free unstructured
// pruning baseline. Zeros the least-magnitude entries of every
// MatMul/vanilla-Gemm layer's constant 2-D FLOAT/FLOAT16/BFLOAT16 weight,
// every Conv layer's constant 4-D FLOAT/FLOAT16/BFLOAT16 weight (ordinary,
// depthwise, and general grouped Conv alike), and every
// ``com.microsoft::Attention`` node's constant 2-D FLOAT/FLOAT16/BFLOAT16
// merged QKV weight, independently per output row/filter: within each row,
// keeps the max(1, round(cols * (1 - sparsity))) highest-magnitude entries
// and zeros the rest. Full parity with the pure-Python
// ``apply_magnitude_pruning`` -- see ``passes/magnitude_pruning.h`` for the
// exact rewrite.
//
// ``n``/``m`` (both ``std::nullopt``, or both given together: N:M
// semi-structured pruning, ``0 < n <= m``) mirror pruning.py's own identical
// parameters and validation (``_validate_pattern``) exactly: keeps the ``n``
// highest-magnitude entries per group of ``m`` columns instead of using
// ``sparsity``. ``global_sparsity`` pools every matched layer's own ``|W|``
// entries into one ranking across the WHOLE model (every graph, including
// nested If/Loop/Scan/BeamSearch-family subgraphs) and picks a single
// keep-count from ``sparsity``'s fraction of that pooled total -- mirrors
// pruning.py's own ``apply_magnitude_pruning`` ``global_sparsity`` mode
// exactly (including its "no per-row floor" property). Incompatible with
// ``n``/``m``: throws ``std::invalid_argument`` when both are given.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this rewrite, to every
// matching layer, to a copy of ``model`` (which is left untouched) and
// returns the result. ``sparsity`` must be in [0, 1) when ``n``/``m`` are not
// given; throws ``std::invalid_argument`` otherwise. A layer with a
// non-constant, non-2-D (MatMul/Gemm/Attention), or non-4-D (Conv) weight is
// left untouched.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto PruneMagnitude(const onnx::ModelProto& model, double sparsity,
                                const std::optional<int64_t>& n = std::nullopt,
                                const std::optional<int64_t>& m = std::nullopt,
                                bool global_sparsity = false);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Any-Precision LLM (Park et al., 2024, ICML 2024, "Any-Precision LLM:
// Low-Cost Deployment of Multiple, Different-Sized LLMs") -- C++ port of
// any_precision_llm.py's own apply_any_precision_llm. Weight-only quantizes
// every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight to
// ``bits`` bits per element, per (output channel, ``block_size``-element
// K-block), via a nested bit-plane code built once to ``max_bits`` (repeated
// within-bin bisection at each bin's own current min/max midpoint) and
// truncated down to ``bits`` by a plain integer right-shift -- see
// ``passes/any_precision_llm.h`` for the exact rewrite and rationale. Every
// element is replaced by its own quantize-dequantize (per-bin-mean
// reconstruction) round trip; the result stays float32 (same shape/dtype as
// the original weight) -- this is a compute-only rewrite, not a compressed
// storage format (see that header's own scope note on why: no ONNX tensor
// type below INT4 exists to store 3/5/6/7-bit codes natively).
//
// Throws ``std::invalid_argument`` if ``max_bits < 1`` or ``bits`` is not in
// ``[1, max_bits]``. Unlike ``Simplify``, this does not run shape inference,
// constant folding or any other simplification pass. A layer with a
// non-constant, non-2-D weight is left untouched.
//
// ACCEPTED, PERMANENT DIVERGENCE from the pure-Python
// ``apply_any_precision_llm`` (``any_precision_llm.py``): floating-point
// summation/iteration order differs (this port groups bin members via an
// ``std::unordered_map``, not numpy's own reduction order), so results can
// differ in the last ULP or two -- the same "independently correct, not
// required to be bit-for-bit identical" contract ``ApplyQuarot``/
// ``apply_quarot_cpp`` already established for their own pair.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyAnyPrecisionLlm(const onnx::ModelProto& model,
                                      int64_t bits, int64_t max_bits,
                                      int64_t block_size);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// QuaRot (Ashkboos et al., 2024) rotation preprocessing plus INT4
// round-to-nearest quantization of *both* the weight and the activation of
// every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight whose
// reduction dimension K is divisible by 32. Rotating the whole residual
// stream by a random orthogonal matrix removes activation outliers the same
// way :func:`quantize_weight_only_int4`-style block quantization already
// tolerates weight outliers, letting both MatMul operands drop to INT4 with
// no calibration data at all -- see ``passes/quarot.h`` for the exact
// rewrite and rationale, and its scope note on the per-layer (not fused
// cross-layer) rotation this port applies.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this rewrite, to every
// matching layer, to a copy of ``model`` (which is left untouched) and
// returns the result. A layer with a non-constant, non-2-D weight, or a
// reduction dimension not divisible by ``block_size``, is left untouched; a
// model with no matching layer, or an opset older than 21, is returned
// unchanged. ``seed`` derives a fresh, deterministic random rotation per
// matched layer. ``block_size`` is the number of reduction-dimension (K)
// elements sharing one weight quantization scale, matching
// ``quantize_weight_only_int4``'s own default. ``epsilon`` floors a token's
// own max-abs rotated-activation value before it is used as a quantization
// scale, avoiding a divide-by-zero on an all-zero token.
//
// ACCEPTED, PERMANENT DIVERGENCE from the pure-Python ``apply_quarot``
// (``quarot.py``): the same ``seed`` does NOT produce the same rotation (or
// output) on both sides. This C++ port builds its random orthogonal
// rotation via Gram-Schmidt with an independent per-node RNG derivation;
// the Python port uses a sign-corrected QR decomposition sequenced through
// a single ``numpy.random.Generator``. Both constructions are independently
// Haar-uniform (mathematically valid, just different) -- see
// ``passes/random_orthogonal.h`` and ``passes/quarot.h`` for the full
// investigation and evidence. This is intentional and will not be changed;
// ``ApplyQuarot``/``apply_quarot_cpp`` and ``apply_quarot`` are two
// independently-correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyQuarot(const onnx::ModelProto& model, uint64_t seed,
                             int64_t block_size, float epsilon);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's IQ4_NL -- C++ port of iq4_nl.py's own
// apply_iq4_nl_quantization. Weight-only quantizes every MatMul/vanilla-Gemm
// layer with a constant 2-D float32 weight into a fixed, 16-entry
// non-uniform ("non-linear") codebook: every 32 consecutive elements of the
// weight's own flattened storage share one scale (max(|block|) /
// max(|codebook|)), and each element snaps to whichever codebook entry
// (times that scale) is closest -- see ``passes/iq4_nl.h`` for the exact
// rewrite, and iq4_nl.py's own docstring for the full rationale and,
// importantly, this format's own codebook provenance (this repo could not
// find or verify llama.cpp's real IQ4_NL codebook anywhere in-tree, so it
// ships its own computationally-derived, honestly-documented non-uniform
// codebook instead -- not a transcription of llama.cpp's own table).
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass. A layer with a non-constant, non-2-D
// weight is left untouched; this port does not support ``Conv`` weights the
// way ``apply_iq4_nl_quantization``'s Python side optionally does.
//
// ACCEPTED, PERMANENT DIVERGENCE from the pure-Python
// ``apply_iq4_nl_quantization`` (``iq4_nl.py``): not required to be
// bit-for-bit identical (see ``passes/iq4_nl.h``'s own note on why this
// port is nonetheless expected to track the Python port unusually closely
// among this repo's *_cpp ports, having no accumulation or
// iterative-refinement step at all) -- ``ApplyIQ4NL``/
// ``apply_iq4_nl_quantization_cpp`` and ``apply_iq4_nl_quantization`` are
// two independently-correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyIQ4NL(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's legacy GGUF "Q4_0"/"Q4_1" block formats -- C++ port of
// gguf_legacy_quant.py's own apply_gguf_q4_0_quantization/
// apply_gguf_q4_1_quantization. Weight-only quantizes every MatMul/
// vanilla-Gemm layer with a constant 2-D float32 weight, one plain
// 32-element block at a time over the weight's own flattened storage (no
// super-block/sub-block requantization like Q4_K, no fixed codebook like
// IQ4_NL): Q4_0 is symmetric with no separate min
// (``dequant = (code - 8) * d``, code in [0, 15]); Q4_1 is asymmetric with
// an explicit per-block min (``dequant = code * d + m``, code in [0, 15]).
// See ``passes/gguf_legacy_quant.h`` for the exact rewrite, and
// gguf_legacy_quant.py's own docstring for the full rationale and this
// format's own encoder-provenance honesty note (the dequantization formula
// is transcribed from this repo's own verified
// ``onnxsim/ggml_legacy_quant.h`` decoder; the encoder's own choice of
// d/m is an ordinary, honestly-scoped min/max fit, not a verified
// reproduction of llama.cpp's own encoder).
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass. A layer with a non-constant, non-2-D
// weight is left untouched; this port does not support ``Conv`` weights the
// way ``apply_gguf_q4_0_quantization``/``apply_gguf_q4_1_quantization``'s
// Python side optionally does.
//
// ACCEPTED, PERMANENT DIVERGENCE from the pure-Python
// ``apply_gguf_q4_0_quantization``/``apply_gguf_q4_1_quantization``
// (``gguf_legacy_quant.py``): not required to be bit-for-bit identical (see
// ``passes/gguf_legacy_quant.h``'s own note on why this port is
// nonetheless expected to track the Python port unusually closely, having
// no accumulation or iterative-refinement step at all) --
// ``ApplyGgufQ4_0``/``ApplyGgufQ4_1`` and their ``_cpp`` Python wrappers
// and ``apply_gguf_q4_0_quantization``/``apply_gguf_q4_1_quantization`` are
// independently-correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ4_0(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ4_1(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's legacy GGUF Q5_0/Q5_1 block formats -- C++ port of
// gguf_legacy_quant_5bit.py's own apply_gguf_q5_0_quantization/
// apply_gguf_q5_1_quantization, the same scheme as
// ApplyGgufQ4_0/ApplyGgufQ4_1 above, one bit wider (code in [0, 31]). See
// passes/gguf_legacy_quant_5bit.h for the exact reconstruction formulas
// and the same ACCEPTED, PERMANENT DIVERGENCE note (with s/Q4/Q5/) --
// ApplyGgufQ5_0/ApplyGgufQ5_1 and their _cpp Python wrappers and
// apply_gguf_q5_0_quantization/apply_gguf_q5_1_quantization are
// independently-correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ5_0(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ5_1(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's GGUF Q8_0 block format -- C++ port of gguf_q8_0.py's own
// apply_gguf_q8_0_quantization: a plain 32-element block, one fp16 scale,
// signed 8-bit code with no bias/min (`dequant = code * d`). See
// passes/gguf_q8_0.h for the exact reconstruction formula and the same
// ACCEPTED, PERMANENT DIVERGENCE note (with s/Q4/Q8/) --
// ApplyGgufQ8_0 and its _cpp Python wrapper and
// apply_gguf_q8_0_quantization are independently-correct,
// non-interchangeable entry points, not aliases. Unlike the Python side,
// this port does not add an include_conv option (only a constant 2-D
// weight on MatMul/vanilla-Gemm is matched), matching gguf_q6_k.h's own
// established scope decision.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ8_0(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's GGUF Q2_K K-quant format -- C++ port of gguf_q2_k.py's own
// apply_gguf_q2_k_quantization: a 256-element super-block split into 16
// sub-blocks of 16, each with its own asymmetric affine (scale, min)
// pair re-quantized to 4-bit codes relative to one shared super-block
// (d, dmin) reference pair, times a 2-bit element code
// (`dequant = d*sc_j*q - dmin*m_j`). See passes/gguf_q2_k.h for the
// exact reconstruction formula and the same ACCEPTED, PERMANENT
// DIVERGENCE note (with s/Q6/Q2/) -- ApplyGgufQ2K and its _cpp Python
// wrapper and apply_gguf_q2_k_quantization are independently-correct,
// non-interchangeable entry points, not aliases. Unlike the Python side,
// this port does not add an include_conv option (only a constant 2-D
// weight on MatMul/vanilla-Gemm is matched), matching gguf_q6_k.h's own
// established scope decision.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ2K(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's GGUF Q3_K K-quant format -- C++ port of gguf_q3_k.py's own
// apply_gguf_q3_k_quantization: a 256-element super-block split into 16
// sub-blocks of 16, each with its own 6-bit unsigned scale code sc_j
// ([0, 63], restricted by this encoder to a non-negative offset,
// sc_j - 32 in [0, 31]) times one shared float16 super-block scale
// d_all, times the format's own asymmetric 3-bit element code q
// ([-4, 3]) -- `dequant = d_all * (sc_j - 32) * q`. See
// passes/gguf_q3_k.h for the exact reconstruction formula and the same
// ACCEPTED, PERMANENT DIVERGENCE note (with s/Q6/Q3/) -- ApplyGgufQ3K and
// its _cpp Python wrapper and apply_gguf_q3_k_quantization are
// independently-correct, non-interchangeable entry points, not aliases.
// Unlike the Python side, this port does not add an include_conv option
// (only a constant 2-D weight on MatMul/vanilla-Gemm is matched),
// matching gguf_q6_k.h's own established scope decision.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ3K(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's GGUF Q4_K K-quant format -- C++ port of onnxsim.gguf_kquant's
// own apply_gguf_q4_k_quantization: a 256-element super-block split into
// 8 sub-blocks of 32, each with its own asymmetric affine (scale, min)
// pair re-quantized to 6-bit codes relative to one shared super-block
// (d, dmin) reference pair, times a 4-bit element code
// (`dequant = d*sc_j*q - dmin*m_j`). See passes/gguf_q4_k.h for the
// exact reconstruction formula and the same ACCEPTED, PERMANENT
// DIVERGENCE note (with s/Q2/Q4/) -- ApplyGgufQ4K and its _cpp Python
// wrapper and apply_gguf_q4_k_quantization are independently-correct,
// non-interchangeable entry points, not aliases. Unlike the Python side,
// this port does not add an include_conv option (only a constant 2-D
// weight on MatMul/vanilla-Gemm is matched), matching gguf_q6_k.h's own
// established scope decision.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ4K(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's GGUF Q5_K K-quant format -- C++ port of onnxsim.gguf_q5_k's
// own apply_gguf_q5_k_quantization: identical to ApplyGgufQ4K above
// except the element code is 5 bits ([0, 31]) rather than Q4_K's 4. See
// passes/gguf_q5_k.h for the exact reconstruction formula and the same
// ACCEPTED, PERMANENT DIVERGENCE note (with s/Q4/Q5/) -- ApplyGgufQ5K and
// its _cpp Python wrapper and apply_gguf_q5_k_quantization are
// independently-correct, non-interchangeable entry points, not aliases.
// Unlike the Python side, this port does not add an include_conv option
// (only a constant 2-D weight on MatMul/vanilla-Gemm is matched),
// matching gguf_q6_k.h's own established scope decision.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ5K(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// BitNet b1.58's published absmean ternary weight quantization (Ma et al.,
// 2024, "The Era of 1-bit LLMs"), as shipped by llama.cpp's GGUF
// TQ1_0/TQ2_0 tensor types -- C++ port of gguf_ternary_quant.py's own
// apply_gguf_ternary_quantization. Weight-only quantizes every MatMul/
// vanilla-Gemm layer with a constant 2-D float32 weight, one 256-element
// block at a time over the weight's own flattened storage: every element
// is restricted to one of {-1, 0, +1} times one shared per-block scale
// ``d = mean(|block|)`` (the paper's own published rule, round-tripped
// through float16 to match llama.cpp's own storage). See
// ``passes/gguf_ternary_quant.h`` for the exact rewrite, and
// gguf_ternary_quant.py's own docstring for the full rationale and this
// format's own honesty note (this port represents the format as a plain
// float32 quantize-dequantize round trip, not llama.cpp's own literal
// bit-packed layout -- no ONNX tensor type below INT4 exists either way).
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass. A layer with a non-constant, non-2-D
// weight is left untouched; this port does not support ``Conv`` weights the
// way ``apply_gguf_ternary_quantization``'s Python side optionally does.
//
// ACCEPTED, PERMANENT DIVERGENCE from the pure-Python
// ``apply_gguf_ternary_quantization`` (``gguf_ternary_quant.py``): not
// required to be bit-for-bit identical (see
// ``passes/gguf_ternary_quant.h``'s own note on why this port is
// nonetheless expected to track the Python port unusually closely, having
// no accumulation or iterative-refinement step at all) --
// ``ApplyGgufTernaryQuant``/its ``_cpp`` Python wrapper and
// ``apply_gguf_ternary_quantization`` are independently-correct,
// non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufTernaryQuant(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// FP6-LLM (Xia et al., 2024, "FP6-LLM: Efficiently Serving Large Language
// Models Through FP6-Centric Algorithm-System Co-Design") -- C++ port of
// fp6_llm.py's own apply_fp6_llm_quantization. Weight-only quantizes every
// MatMul/vanilla-Gemm layer with a constant 2-D float32 weight, one
// 64-element block at a time over the weight's own flattened storage:
// every block is rescaled by its own ``max(|block|)`` to fill FP6 E3M2's
// (1 sign bit, 3 exponent bits, 2 mantissa bits) narrow representable
// range, cast to the nearest representable value, and rescaled back. See
// ``passes/fp6_llm.h`` for the exact rewrite -- including how its
// 64-entry codebook is built from the format's own exponent/mantissa
// arithmetic definition rather than any bit-level codec -- and
// fp6_llm.py's own docstring for the full rationale.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass. A layer with a non-constant, non-2-D
// weight is left untouched; this port only implements the paper's primary
// E3M2 format and does not support ``Conv`` weights the way
// ``apply_fp6_llm_quantization``'s Python side optionally does.
//
// ACCEPTED, PERMANENT DIVERGENCE from the pure-Python
// ``apply_fp6_llm_quantization`` (``fp6_llm.py``): not required to be
// bit-for-bit identical (see ``passes/fp6_llm.h``'s own note on why this
// port is nonetheless expected to track the Python port unusually
// closely, having no accumulation or iterative-refinement step at all) --
// ``ApplyFp6Llm``/its ``_cpp`` Python wrapper and
// ``apply_fp6_llm_quantization`` are independently-correct,
// non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyFp6Llm(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// llama.cpp's GGUF Q6_K K-quant format -- C++ port of gguf_q6_k.py's own
// apply_gguf_q6_k_quantization. Weight-only quantizes every MatMul/
// vanilla-Gemm layer with a constant 2-D float32 weight, one 256-element
// super-block at a time over the weight's own flattened storage: each
// super-block is split into 16 sub-blocks of 16 elements, each sub-block
// sharing one 8-bit scale code multiplied by one shared float16
// super-block scale, times a symmetric 6-bit element code. See
// ``passes/gguf_q6_k.h`` for the exact rewrite, and gguf_q6_k.py's own
// docstring for the full rationale and this format's own
// encoder-provenance honesty note (the dequantization formula is
// transcribed from this repo's own verified ``onnxsim/ggml_kquant.h``
// decoder; the encoder's own choice of sub-block/super-block scale is an
// honestly-scoped fit, not a verified reproduction of llama.cpp's own
// encoder). Unlike Q4_K/Q5_K (which this repo has no C++ port for at
// all, due to their packed asymmetric 6-bit (scale, min) sub-block
// codes), Q6_K's own reconstruction has no packed-bitfield complexity to
// get wrong, so this port ships alongside its Python module.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass. A layer with a non-constant, non-2-D
// weight is left untouched; this port does not support ``Conv`` weights the
// way ``apply_gguf_q6_k_quantization``'s Python side optionally does.
//
// ACCEPTED, PERMANENT DIVERGENCE from the pure-Python
// ``apply_gguf_q6_k_quantization`` (``gguf_q6_k.py``): not required to be
// bit-for-bit identical (see ``passes/gguf_q6_k.h``'s own note on why
// this port is nonetheless expected to track the Python port unusually
// closely, having no accumulation or iterative-refinement step at all) --
// ``ApplyGgufQ6K``/its ``_cpp`` Python wrapper and
// ``apply_gguf_q6_k_quantization`` are independently-correct,
// non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyGgufQ6K(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// AngelSlim's LeptoQuant -- C++ port of leptoquant.py's own
// apply_leptoquant: DeepSeek-V3-style 128x128-block FP8 E4M3 weight
// quantization, refined by a 5-point outlier-fraction grid search per
// tile. See passes/leptoquant.h for the exact reconstruction formula.
// ACCEPTED, PERMANENT DIVERGENCE from apply_leptoquant: this scheme's own
// grid search is closed-form/deterministic, so this port is expected to
// track the Python port's own float64 numpy implementation closely, up
// to floating-point summation-order/quantile-interpolation differences.
// ApplyLeptoquant and its _cpp Python wrapper and apply_leptoquant remain
// independently-correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyLeptoquant(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// bitsandbytes' NF4 (NormalFloat 4-bit) weight-only quantization -- C++
// port of nf4.py's own quantize_weight_only_nf4: a fixed, 16-value,
// zero-symmetric non-uniform codebook, one scale per 64-element
// (output-channel, K-block) group. See passes/nf4.h for the exact
// reconstruction formula. Unlike quantize_weight_only_nf4, this port
// folds the round trip directly into a replacement float32 initializer
// rather than building it out of Gather/Reshape/Mul graph nodes.
// ACCEPTED, PERMANENT DIVERGENCE from quantize_weight_only_nf4: no
// accumulation step, so this port is expected to track the Python port's
// own float64 numpy implementation closely, up to floating-point
// summation-order/argmin tie-breaking differences. ApplyNF4 and its _cpp
// Python wrapper and quantize_weight_only_nf4 remain independently-
// correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyNF4(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// IF4 (Adaptive Block-Scaled Data Types) -- C++ port of
// if4_quantization.py's own quantize_weight_only_if4: per
// (output-channel, 16-element K-block), tries both a plain signed INT4
// grid and MXFP4's own E2M1 codebook and keeps whichever reconstructs
// that block with lower MSE. See passes/if4_quantization.h for the exact
// reconstruction formula. Unlike quantize_weight_only_if4, this port
// folds the round trip directly into a replacement float32 initializer
// rather than building it out of Cast/Gather/Reshape/Mul graph nodes.
// ACCEPTED, PERMANENT DIVERGENCE from quantize_weight_only_if4: no
// accumulation step, so this port is expected to track the Python port's
// own float64 numpy implementation closely, up to floating-point
// summation-order differences. ApplyIF4 and its _cpp Python wrapper and
// quantize_weight_only_if4 remain independently-correct, non-
// interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyIF4(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// NVIDIA's NVFP4 weight-only quantization -- C++ port of
// nvfp4_quantization.py's own quantize_weight_only_nvfp4: shares OCP
// MXFP4's E2M1 element codebook but replaces its power-of-two-only block
// scale with a two-level (per-tensor global scale, E4M3-rounded
// per-block scale) rule. See passes/nvfp4_quantization.h for the exact
// reconstruction formula; the graph shape (Cast/Gather/Reshape/Mul) is
// identical to weight_only_quantize_mxfp4_matmul.h's own, matching this
// repo's own established representation for a real, hardware-meaningful
// microscaling format. ACCEPTED, PERMANENT DIVERGENCE from
// quantize_weight_only_nvfp4: no accumulation step, so this port is
// expected to track the Python port's own float64 numpy implementation
// closely, up to floating-point summation-order differences.
// ApplyNVFP4Quantization and its _cpp Python wrapper and
// quantize_weight_only_nvfp4 remain independently-correct, non-
// interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyNVFP4Quantization(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// DeepSeek-V3-style fine-grained block FP8 weight quantization -- C++
// port of the *weight* half of deepseek_fp8.py's own apply_deepseek_fp8:
// one real FLOAT8E4M3FN round trip per 128x128 output-channel x
// input-feature tile. See passes/deepseek_fp8.h for the exact
// reconstruction formula and its own scope-narrowing note (the
// activation-quantization/W8A8 half of apply_deepseek_fp8 is out of
// scope for this port entirely). ApplyDeepSeekFp8 and its _cpp Python
// wrapper and apply_deepseek_fp8 remain independently-correct, non-
// interchangeable entry points, not aliases -- though unlike most
// sibling *_cpp ports here, both directions are a real, fully-specified
// FP8 cast with no encoder ambiguity, so this port is expected to track
// deepseek_fp8.py's own quantize_dequantize_block_fp8 unusually closely.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyDeepSeekFp8(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// K-means per-layer codebook weight quantization (Han et al., 2015,
// "Deep Compression") -- C++ port of kmeans_quantization.py's own
// quantize_weight_only_kmeans: a 16-centroid codebook fit per layer via
// Lloyd's algorithm, directly to that layer's own weight values. See
// passes/kmeans_quantization.h for the exact algorithm and its own
// documented narrow-edge-case divergence (the too-few-distinct-
// percentiles initialization fallback). Unlike quantize_weight_only_
// kmeans, this port folds the round trip directly into a replacement
// float32 initializer rather than building it out of Cast/Gather graph
// nodes. ApplyKMeansQuantization and its _cpp Python wrapper and
// quantize_weight_only_kmeans remain independently-correct, non-
// interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyKMeansQuantization(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// HQQ -- Half-Quadratic Quantization (Badri & Shaji, 2023) -- C++ port
// of hqq.py's own quantize_weight_only_int4_hqq: an asymmetric affine
// INT4 quantizer whose zero-point is refined per (output-channel,
// 32-element K-block) group by a bounded (10-step) Iteratively
// Reweighted Least Squares (IRLS) fit. See passes/hqq.h for the exact
// reconstruction formula. Unlike quantize_weight_only_int4_hqq, this
// port folds the round trip directly into a replacement float32
// initializer rather than building a real DequantizeLinear node with
// packed UINT4 codes. ACCEPTED, PERMANENT DIVERGENCE from
// quantize_weight_only_int4_hqq: IRLS here is a deterministic
// fixed-point iteration with no RNG, so this port is expected to track
// the Python port's own float64 numpy implementation closely, up to
// floating-point summation-order differences. ApplyHQQ and its _cpp
// Python wrapper and quantize_weight_only_int4_hqq remain independently-
// correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyHQQ(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// I-BERT's own i-GELU polynomial approximation of Erf -- C++ port of
// ibert_gelu.py's own apply_ibert_gelu. Unlike every weight-only port
// above, this is a nonlinear-activation rewrite: it matches any
// standalone, single-input/single-output Erf node anywhere in the graph
// (no MatMul/weight involvement at all) and rebuilds its output from a
// fixed Abs/Clip/Add/Mul/Sign sequence. See passes/ibert_gelu.h for the
// exact formula and constants. Both sides build the exact same five ONNX
// ops from the exact same three float32 constants, so this port is
// expected to be numerically identical to apply_ibert_gelu up to
// onnxruntime's own float32 evaluation of the resulting graph.
// ApplyIBertGelu and its _cpp Python wrapper and apply_ibert_gelu remain
// independently-correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyIBertGelu(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// I-BERT's own integer-friendly Softmax exp-approximation -- C++ port of
// ibert_softmax.py's own apply_ibert_softmax. Also a nonlinear-activation
// rewrite, not a weight quantizer: matches any standalone Softmax node
// and rebuilds its exp-and-normalize computation out of ordinary
// opset-18+ ops (ReduceMax/Sub/Neg/Div/Floor/Add/Mul/Pow/ReduceSum/Div).
// See passes/ibert_softmax.h for the exact formula, constants, and its
// own documented scope narrowing (the final normalization uses a plain
// Div, not the paper's own integer-only iterative reciprocal -- already
// ibert_softmax.py's own scope, not one this port adds). A model whose
// opset is below 18 is left completely untouched. ApplyIBertSoftmax and
// its _cpp Python wrapper and apply_ibert_softmax remain independently-
// correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyIBertSoftmax(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// AdpQ (Ghaffari et al., 2024) -- C++ port of adpq.py's own
// quantize_weight_only_adpq: a calibration-free salient/non-salient
// weight split per (output-channel, 128-element K-group), decided
// purely from a weight's own values via a median/MAD-based adaptive
// threshold borrowed from Adaptive LASSO. See passes/adpq.h for the
// exact formula. Unlike quantize_weight_only_adpq, this port folds the
// round trip directly into a replacement float32 initializer rather
// than building a real DequantizeLinear+ScatterND+Add graph rewrite.
// ACCEPTED, PERMANENT DIVERGENCE from quantize_weight_only_adpq: no
// accumulation step, so this port is expected to track the Python
// port's own float64 numpy implementation closely, up to floating-point
// median-tie-breaking and summation-order differences. ApplyADPQ and its
// _cpp Python wrapper and quantize_weight_only_adpq remain independently-
// correct, non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyADPQ(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// ICQuant (Li, Hanna, Fragouli, Diggavi, 2025) -- C++ port of
// icquant.py's own quantize_weight_only_icquant: per (output-channel,
// 32-element K-block), the single largest-magnitude element is excluded
// from the block's own scale computation and reconstructed exactly; the
// rest quantize to a symmetric 7-level-per-side grid. See
// passes/icquant.h for the exact formula and its own note that
// icquant.py's "combinadic" index encoding is a pure storage detail with
// no effect on the reconstructed values, so this port skips it entirely.
// Unlike quantize_weight_only_icquant, this port folds the round trip
// directly into a replacement float32 initializer rather than building a
// real INT4/DequantizeLinear/ScatterND/MatMul/Add graph rewrite.
// ACCEPTED, PERMANENT DIVERGENCE from quantize_weight_only_icquant: no
// accumulation step, so this port is expected to track the Python port's
// own float64 numpy implementation closely, up to floating-point
// summation-order differences and up to how ties are broken among
// equal-magnitude outlier candidates. ApplyICQuant and its _cpp Python
// wrapper and quantize_weight_only_icquant remain independently-correct,
// non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyICQuant(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// OliVe -- Outlier-Victim Pair quantization (Guo et al., ISCA 2023) --
// C++ port of olive.py's own quantize_weight_only_olive: per
// (output-channel, 32-element K-block), adjacent element pairs with
// exactly one outlier member become an OVP pair (the outlier gets a
// wider code, its neighbor a narrower "victim" code at the same total
// bit budget as two ordinary elements); pairs with zero or two outliers
// fall back to ordinary quantization. See passes/olive.h for the exact
// formula. Unlike quantize_weight_only_olive, this port folds the round
// trip directly into a replacement float32 initializer rather than
// building a real DequantizeLinear x2 + Cast + Where + MatMul[+Add]
// graph rewrite. ACCEPTED, PERMANENT DIVERGENCE from
// quantize_weight_only_olive: no accumulation step, so this port is
// expected to track the Python port's own float64 numpy implementation
// closely, up to floating-point summation-order/median-tie-breaking
// differences. ApplyOlive and its _cpp Python wrapper and
// quantize_weight_only_olive remain independently-correct, non-
// interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyOlive(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// AQLM -- Additive Quantization for Language Models (Egiazarian et al.,
// 2024) -- C++ port of aqlm.py's own quantize_weight_only_aqlm: each
// (output-channel, 8-element K-block) group is reconstructed as the sum
// of 2 codebook lookups, each codebook fit via greedy residual Lloyd's-
// algorithm k-means. See passes/aqlm.h for the exact algorithm and its
// own documented divergence. Unlike quantize_weight_only_aqlm, this port
// folds the reconstruction directly into a replacement float32
// initializer rather than building a Gather+Add-chain graph rewrite.
// ACCEPTED, PERMANENT DIVERGENCE from quantize_weight_only_aqlm: unlike
// kmeans_quantization.h's own scalar percentile-based initialization
// (which matches its Python counterpart exactly outside a narrow edge
// case), this port's own k-means initialization is deterministically
// magnitude-sorted rather than aqlm.py's own seeded-random sample --
// reproducing numpy's own PCG64 bitstream in C++ was judged not worth
// it, so this is a genuinely different (though algorithmically
// identical) fit, not merely a floating-point-order divergence.
// ApplyAQLM and its _cpp Python wrapper and quantize_weight_only_aqlm
// remain independently-correct, non-interchangeable entry points, not
// aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyAQLM(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Drop-by-Drop (Babaoglu, Chen, Khisti, 2026) -- C++ port of
// drop_by_drop.py's own quantize_weight_only_drop_by_drop: each
// (output-channel, 8-element K-block) group is reconstructed by 4
// additive residual codebook stages, each an *importance-weighted*
// Lloyd's-algorithm k-means fit (weighted by the group's own fixed
// original-magnitude RMS), unlike AQLM's own unweighted fit. See
// passes/drop_by_drop.h for the exact algorithm and its own documented
// divergence. Unlike quantize_weight_only_drop_by_drop, this port folds
// the reconstruction directly into a replacement float32 initializer
// rather than building a Gather/Add-chain graph rewrite. ACCEPTED,
// PERMANENT DIVERGENCE from quantize_weight_only_drop_by_drop: the same
// deterministic-magnitude-sorted-init-instead-of-seeded-random-sample
// divergence ApplyAQLM's own doc comment above already explains in full,
// applied to each of this port's own 4 weighted k-means fits. ApplyDropByDrop
// and its _cpp Python wrapper and quantize_weight_only_drop_by_drop
// remain independently-correct, non-interchangeable entry points, not
// aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyDropByDrop(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// LO-BCQ -- Block Clustered Quantization (Elangovan, Sakr, Raghunathan,
// Khailany, 2025) -- C++ port of the weight-side half of lo_bcq.py's own
// quantize_weight_only_lo_bcq: blocks of the reduction dimension are
// clustered by their own [mean, std] feature vector into 4 groups (via
// multi-dimensional Lloyd's k-means), then each cluster fits its own
// small 1-D codebook from only its own currently-assigned blocks, in an
// outer loop alternating cluster-codebook refit and block-reassignment.
// See passes/lo_bcq.h for the exact algorithm and its own two documented
// divergences. Unlike quantize_weight_only_lo_bcq, this port folds the
// reconstruction directly into a replacement float32 initializer rather
// than building a Gather/GatherElements/Reshape graph rewrite. ACCEPTED,
// PERMANENT DIVERGENCE from quantize_weight_only_lo_bcq, two-fold: see
// passes/lo_bcq.h's own top-of-file comment for the full rationale (a
// deterministic percentile-derived-centroid 1-D fit, matching
// kmeans_quantization.h's own precedent; and a deterministic
// feature-norm-sorted block-clustering init, matching ApplyAQLM's own
// precedent). ApplyLoBcq and its _cpp Python wrapper and
// quantize_weight_only_lo_bcq remain independently-correct,
// non-interchangeable entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyLoBcq(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// QuIP# (Tseng et al., 2024) -- C++ port of quip_sharp.py's own
// apply_quip_sharp: conjugates the weight by a pair of random orthogonal
// matrices (Wtilde = V @ W @ U, incoherence processing) so its entries
// look i.i.d. Gaussian, then jointly quantizes each 8-element group to
// the nearest E8 lattice point (Conway & Sloane, 1982). See
// passes/quip_sharp.h for the exact algorithm and its own documented
// divergence. Unlike quantize_weight_only_quip_sharp's own graph
// rewrite (which keeps U, V and the packed INT4 lattice codes as
// explicit initializers and new MatMul nodes), this port folds the
// entire rotate/quantize/rotate-back sandwich into a single replacement
// weight initializer -- exact, not an approximation, since U/V are
// square and only sandwich the weight (see passes/quip_sharp.h's own
// top-of-file comment for the associativity argument). ACCEPTED,
// PERMANENT DIVERGENCE from apply_quip_sharp: the random orthogonal
// matrices are generated via passes/random_orthogonal.h's Gram-Schmidt
// construction (the same precedent ApplyQuarot already establishes for
// this codebase), not quip_sharp.py's own QR-with-sign-correction one --
// both are independently Haar-uniform; cross-language bit parity is
// explicitly not a goal. ApplyQuipSharp and its _cpp Python wrapper and
// apply_quip_sharp remain independently-correct, non-interchangeable
// entry points, not aliases.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyQuipSharp(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Attention computation quantization -- C++ port of
// attention_quantization.py's own apply_attention_quantization: quantizes
// the decomposed attention subgraph's own Q/K/V operands (data-free,
// per-token dynamic INT8, scale = max(|x|, axis=-1)/127) and the Softmax
// output itself (a fixed UINT8 scale of 1/255, since a Softmax output's
// range is guaranteed [0, 1]). Unlike every weight-only *_cpp port in this
// repo, this is not a fold-to-initializer pass -- none of the four
// quantized tensors is a constant weight, so this builds real new
// quantize/dequantize graph nodes instead, the same shape
// ApplyIBertSoftmax/ApplyQuarot already establish. See
// passes/attention_quantization.h for the exact match/rewrite and its own
// documented FLOAT-dtype scope narrowing. No RNG/fitting step anywhere in
// this technique, so this port is expected to track
// apply_attention_quantization closely, up to ordinary floating-point
// summation-order differences.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyAttentionQuantization(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// ZeroQuant (Yao et al., 2022) -- C++ port of zeroquant.py's own
// apply_zeroquant: pairs this repo's own existing group-wise INT8 weight
// quantization (quantize_weight_only_int8_block) with per-token dynamic
// INT8 activation quantization, and -- unlike every other per-token-
// dynamic-INT8 use in this repo, which immediately dequantizes back to
// float32 -- feeds the result into a genuine int8 x int8 MatMulInteger.
// Cannot fold to a single replacement weight (the activation's own
// per-token scale is a runtime value, the same reason ApplyQuarot can't
// fold either); builds real new graph nodes instead, node for node
// matching zeroquant.py's own Shape/Gather/Concat/Reshape flatten prelude
// and Split/MatMulInteger/Cast/Mul/Sum construction. See passes/zeroquant.h
// for the exact node sequence and its own "Why grouped MatMulInteger"
// rationale. `block_size` (elements per weight-quantization group along K,
// default 32) and `epsilon` (scale floor, default 1e-12) mirror
// apply_zeroquant's own identically-named/defaulted parameters exactly. No
// RNG/fitting step anywhere in this technique, so this port is expected to
// track apply_zeroquant closely, up to ordinary floating-point
// summation-order differences.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyZeroQuant(const onnx::ModelProto& model,
                                int64_t block_size, float epsilon);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// IntactKV (Liu et al., 2024) -- C++ port of intactkv.py's own
// apply_intactkv: not a quantizer, but a *companion* pass that splits a
// KV-cache stream's own fixed-length leading "pivot" prefix
// (attention-sink tokens, disproportionately sensitive to quantization
// error) out into its own always-exact stream, so a following KV-cache
// quantizer can leave the pivots untouched forever and quantize only the
// remaining, still-growing "rest" of the cache. Matches a
// `Concat(past, new, axis=seq)` KV-cache stream (past: a float32 graph
// input consumed by nothing else, output: a graph output) and rewrites it
// into a new fixed-size `*_pivot` graph input/output plus an ordinary
// `*_rest` stream, reconstructed back under the ORIGINAL present_* name via
// a new Concat node -- real graph input/output surgery, not a
// fold-to-initializer or a same-shape node rewrite. See passes/intactkv.h
// for the exact match/rewrite. No RNG/fitting step anywhere in this
// technique (a closed-form structural transformation), so this port is
// expected to track apply_intactkv exactly. This port hardcodes
// intactkv.py's own default `num_pivot_tokens=4` rather than exposing it as
// a parameter.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyIntactKv(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// KBVQ-MoE (Xu et al., 2026) -- C++ port of kbvq_moe.py's own
// apply_kbvq_moe: fits a KLT (PCA) basis shared across a `com.microsoft::
// MoE` router group's own experts (a closed-form economy SVD of the
// centered [E, D] expert-weight stack, the same Jacobi-SVD technique
// low_rank_compensation_entry.cpp already establishes for this codebase --
// see passes/kbvq_moe.h's own "SVD IMPLEMENTATION NOTE" for why this port
// keeps its own header-only copy rather than sharing that file's), then
// vector-quantizes each expert's own residual against that shared basis
// with an ordinary per-expert k-means codebook (reusing
// kmeans_quantization.h's own QuantizeDequantizeKMeans directly -- see
// passes/kbvq_moe.h's own top-of-file comment for why its bits=4/
// kmeans_iters=20 defaults already match that function's own hardcoded
// constants exactly). Only FLOAT32 fc1_experts_weights/fc2_experts_weights
// are quantized (FLOAT16/BFLOAT16 left untouched, matching kbvq_moe.py's
// own scope exactly); node/shape matching mirrors pruning.py's own
// _match_moe_producer, reused unmodified by kbvq_moe.py itself. ACCEPTED,
// PERMANENT DIVERGENCE from apply_kbvq_moe: none beyond what
// kmeans_quantization.h's own top-of-file comment already documents for
// QuantizeDequantizeKMeans's own deterministic percentile-init scheme (vs.
// kbvq_moe.py's own seeded-random-sample fallback in the rare
// too-few-distinct-percentiles edge case) -- the KLT/SVD piece itself has
// no RNG at all. This port hardcodes kbvq_moe.py's own defaults (rank=4,
// bits=4, kmeans_iters=20) rather than exposing them as parameters.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyKbvqMoe(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// LLM-FP4 (Liu et al., 2023, EMNLP) -- C++ port of llm_fp4.py's own
// quantize_weight_only_llm_fp4 (weight-only half only; that module's own
// apply_llm_fp4_activation_quantization[_per_tensor] are separate,
// out-of-scope activation-quantization passes): a standard sign/exponent/
// mantissa FP4 format whose per-block scale is an ordinary real-valued
// float (not restricted to a power of two, unlike MXFP4) and whose
// exponent/mantissa bit split is itself searched (E1M2, E2M1, E3M0) per
// tensor, picking whichever (format, per-block scale) combination
// minimizes reconstruction MSE. Data-free: both choices are fit directly to
// each weight's own values by exhaustive grid search. This port hardcodes
// llm_fp4.py's own defaults (block_size=32, formats=(e1m2, e2m1, e3m0),
// num_scale_candidates=17, min_clip_ratio=0.5) rather than exposing them as
// parameters, and omits its own skip_names parameter. ACCEPTED NUMERICAL
// SCOPE: a closed-form grid search with no RNG, so this port is expected to
// track the Python reference closely, up to floating-point summation-order
// differences and first-occurrence tie-breaking (matching numpy's own
// argmin/comparison convention).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeWeightOnlyLlmFp4(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// QServe's QoQ quantization (Lin et al., 2024, MLSys 2025) -- C++ port of
// qoq.py's own quantize_weight_only_qoq (the module's primary contribution;
// that module's own apply_smooth_attention is a separate, calibration-
// driven KV-cache-side technique, out of scope here): progressive
// (INT8-then-INT4) block-wise weight quantization, first to a protective
// per-output-channel INT8 grid, then that already-INT8-quantized tensor
// down to INT4 per (channel, block) group -- folded into one combined
// per-group scale so the graph only needs a single DequantizeLinear. This
// port hardcodes qoq.py's own defaults (block_size=32, int8_clip_max=119).
// ACCEPTED, PERMANENT DIVERGENCE: none -- a closed-form, deterministic
// two-stage scheme with no RNG, expected to track the Python reference
// closely.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyQoq(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// D2Quant's Dual-Scale Quantizer (DSQ) (Yan et al., 2026) -- C++ port of
// d2quant.py's own apply_dsq (that module's own apply_dac is a separate,
// calibration-driven technique, out of scope here): a weight-side fix for
// down-projection matrices in a SwiGLU/GLU-style MLP block. Derives a
// per-input-channel auxiliary scale for the down-projection (fit by
// alternating: quantize W/s with an ordinary blockwise INT4 quantizer, then
// re-solve s in closed form as a per-column weighted least squares against
// the quantized reconstruction) and folds its reciprocal into the paired
// up-projection's own raw weight -- "absorbable," no new runtime op beyond
// the down-projection's own DequantizeLinear. Matches a down-projection
// MatMul/vanilla-Gemm whose activation input is produced by a two-operand
// elementwise Mul, at least one of whose operands is directly the output of
// another MatMul/vanilla-Gemm (the up-projection); both matched weights'
// own consumer counts and shapes are checked, mirroring apply_dsq's own
// matching exactly. This port hardcodes d2quant.py's own defaults
// (block_size=32, num_iterations=15). ACCEPTED, PERMANENT DIVERGENCE: none
// -- a closed-form, deterministic alternating fit with no RNG, expected to
// track the Python reference closely.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyDsq(const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// DAQ (Delta-Aware Quantization) -- C++ port of daq.py's own apply_daq,
// declared in daq_entry.h (included above) rather than duplicated here.
// Unlike every other port in this file, DAQ is data-free but still takes
// two full ModelProto arguments (a base and a fine-tuned checkpoint,
// matched by node output name) -- the same two-model correspondence
// assumption ApplyGptq/ApplyQronos make, minus their ModelExecutor/
// calibration_data. See daq_entry.h for the full rationale, the exact
// coarse-to-fine FP8 scale search, and its own ACCEPTED, PERMANENT
// DIVERGENCE note.

// Low-Rank Compensation (LoRC), from ZeroQuant-V2 (Yao et al., 2023) --
// C++ port of low_rank_compensation.py's own apply_low_rank_compensation,
// declared in low_rank_compensation_entry.h (included above) rather than
// duplicated here, mirroring how ApplyDAQ is documented in this file
// instead of daq_entry.h. Like DAQ, this is a data-free, two-model port
// (a float model and its own already-INT4-quantized counterpart, matched
// by node output name, the same correspondence ApplyGptq/ApplyQronos/DAQ
// all make). Unlike every other data-free *_cpp port in this file (which
// folds its correction into a replacement weight initializer), LoRC's
// own correction is *additive to a matched layer's output*: it computes
// the quantization error matrix's best rank-r approximation (Eckart-Young
// theorem, via a hand-rolled Jacobi SVD -- no linear-algebra library is
// linked into this codebase) and injects it as two new small MatMul
// nodes plus an Add. See low_rank_compensation_entry.h/.cpp for the full
// rationale and their own ACCEPTED, PERMANENT DIVERGENCE note on the SVD
// choice.

// llama.cpp's "importance matrix" (imatrix) -- C++ port of
// imatrix_quant.py's own apply_imatrix_quantization, declared in
// imatrix_quant_entry.h (included above) rather than duplicated here,
// mirroring how ApplyWandaPruning/ApplySparseGptPruning (structured_
// pruning_entry.h, also included above) are documented in their own home
// header instead of this one. See imatrix_quant.py's own module docstring
// for the technique, imatrix_quant_entry.h for this port's own scope, and
// that header's own top comment for why this calibration-driven pass
// follows ApplyWandaPruning's protobuf-level shape rather than
// ApplyQuarot's data-free PredicateBasedPass one.

// Outlier Suppression (Wei et al., 2022) Gamma Migration -- C++ port of
// outlier_suppression.py's own apply_outlier_suppression, declared in
// outlier_suppression_entry.h (included above) rather than duplicated
// here, mirroring how ApplyImatrixQuantization (imatrix_quant_entry.h,
// also included above) is documented in its own home header instead of
// this one. See outlier_suppression.py's own module docstring for the
// technique and outlier_suppression_entry.h for this port's own scope
// (including why this calibration-driven pass follows
// ApplyImatrixQuantization's own protobuf-level shape).

// Outlier Suppression+ (Wei et al., 2023) shifting and scaling -- C++
// port of outlier_suppression_plus.py's own
// apply_outlier_suppression_plus, declared in
// outlier_suppression_plus_entry.h (included above) rather than
// duplicated here, mirroring how ApplyOutlierSuppression
// (outlier_suppression_entry.h, also included above) is documented in
// its own home header instead of this one. See
// outlier_suppression_plus.py's own module docstring for the technique
// and outlier_suppression_plus_entry.h for this port's own scope.

// SmoothQuant (Xiao et al., 2022) migration -- C++ port of
// smoothquant.py's own apply_smoothquant, declared in
// smoothquant_entry.h (included above) rather than duplicated here,
// mirroring how ApplyImatrixQuantization (imatrix_quant_entry.h, also
// included above) is documented in its own home header instead of this
// one. See smoothquant.py's own module docstring for the technique and
// smoothquant_entry.h for this port's own scope (including why this
// calibration-driven pass follows ApplyImatrixQuantization's own
// protobuf-level shape).

// LLM.int8() (Dettmers et al., 2022) outlier/INT8 decomposition -- C++
// port of llm_int8.py's own apply_llm_int8, declared in llm_int8_entry.h
// (included above) rather than duplicated here, mirroring how
// ApplyOutlierSuppressionPlus (outlier_suppression_plus_entry.h, also
// included above) is documented in its own home header instead of this
// one. See llm_int8.py's own module docstring for the technique and
// llm_int8_entry.h for this port's own scope.

// EasyQuant (Wu, Judd, Isaev, Micikevicius, 2020) coordinate-descent
// W8A8 scale search -- C++ port of easyquant.py's own apply_easyquant,
// declared in easyquant_entry.h (included above) rather than duplicated
// here, mirroring how ApplyLlmInt8 (llm_int8_entry.h, also included
// above) is documented in its own home header instead of this one. See
// easyquant.py's own module docstring for the technique and
// easyquant_entry.h for this port's own scope.

// FPTQ (Li, Zhang, Li, Yao, Zhang, Chu, Sun, Du and Xie, 2023)
// logarithmic-equalization migration -- C++ port of fptq.py's own
// apply_fptq, declared in fptq_entry.h (included above) rather than
// duplicated here, mirroring how ApplySmoothQuant (smoothquant_entry.h,
// also included above) is documented in its own home header instead of
// this one. See fptq.py's own module docstring for the technique and
// fptq_entry.h for this port's own scope.

// GPTQ (Frantar et al., 2022) sequential Hessian-compensated rounding --
// C++ port of gptq.py's own apply_gptq, declared in gptq_entry.h
// (included above) rather than duplicated here, mirroring how
// ApplyLlmInt8 (llm_int8_entry.h, also included above) is documented in
// its own home header instead of this one. See gptq.py's own module
// docstring for the technique and gptq_entry.h for this port's own scope
// (including its accepted numerical scope).

// GPTAQ (Li, Yin, Lee, Xiao, Panda, 2025) asymmetric-calibration
// correction layered on top of ApplyGptq -- C++ port of gptaq.py's own
// apply_gptaq, declared in gptaq_entry.h (included above) rather than
// duplicated here, mirroring how ApplyGptq (gptq_entry.h, also included
// above) is documented in its own home header instead of this one. See
// gptaq.py's own module docstring for the technique's first-principles
// derivation and gptaq_entry.h for this port's own scope (including its
// accepted numerical scope, shared with ApplyGptq's).

// AWQ (Lin et al., 2023) grid-searched per-channel rescaling -- C++ port
// of awq.py's own apply_awq, declared in awq_entry.h (included above)
// rather than duplicated here, mirroring how ApplyGptq (gptq_entry.h,
// also included above) is documented in its own home header instead of
// this one. See awq.py's own module docstring for the technique and
// awq_entry.h for this port's own scope.

// Qronos: a sequential, whole-model generalization of ApplyGptq that
// additionally accounts for the error already baked into a layer's
// activations because upstream layers were quantized first (not just
// this layer's own rounding error) -- processes layers in the float
// model's own node order, re-probing the progressively-corrected
// quantized model before each subsequent layer. C++ port of
// qronos.py's own apply_qronos, declared in qronos_entry.h (included
// above) rather than duplicated here, mirroring how ApplyAwq
// (awq_entry.h, also included above) is documented in its own home
// header instead of this one. See qronos.py's own module docstring for
// the technique and qronos_entry.h for this port's own scope (including
// its accepted numerical scope, shared with ApplyGptq's).

// AdaRound: Nagel et al. 2020's rectified-sigmoid relaxation of each
// weight element's floor/ceil rounding decision, optimized by a
// hand-rolled Adam loop to minimize a layer's own reconstruction error
// against real calibration activations. C++ port of adaround.py's own
// apply_adaround, declared in adaround_entry.h (included above) rather
// than duplicated here, mirroring how ApplyQronos (qronos_entry.h, also
// included above) is documented in its own home header instead of this
// one. See adaround.py's own module docstring for the technique and
// adaround_entry.h for this port's own scope -- including its accepted
// numerical scope (same class as ApplyTesseraq's own, below: an
// iterative Adam optimization, not a closed-form computation, so
// cross-language floating-point agreement is measured empirically
// (tests/test_adaround_cpp.py) rather than assumed).

// TesseraQ: "Progressive Adaptive Rounding" (PAR) -- ApplyAdaRound-style
// rectified-sigmoid rounding relaxation, but optimized by a hand-rolled
// Adam loop jointly with each weight block's own dequantization scale
// (in log-space), with a coarse-to-fine element-by-element hardening
// schedule across a handful of rounds instead of a single monolithic
// anneal. C++ port of tesseraq.py's own apply_tesseraq, declared in
// tesseraq_entry.h (included above) rather than duplicated here,
// mirroring how ApplyAdaround (adaround_entry.h, also included above)
// is documented in its own home header instead of this one. See
// tesseraq.py's own module docstring for the technique and
// tesseraq_entry.h for this port's own scope -- including its accepted
// numerical scope, which is NOT the same as every closed-form port's own
// (ApplyGptq/ApplyAwq/ApplyQronos/ApplyGptvq's correction half): this is
// an iterative Adam optimization, not a single closed-form computation,
// so cross-language floating-point agreement is measured empirically
// (tests/test_tesseraq_cpp.py) rather than assumed from the algorithm's
// own structure.

// QuaRot+GPTQ (Ashkboos et al., 2024): ApplyQuarot's own per-layer random
// rotation and data-free activation quantization, but with the weight
// quantized via ApplyGptq's own Hessian-based column algorithm (evaluated
// in the rotated activation space) instead of round-to-nearest -- C++ port
// of quarot.py's own apply_quarot_gptq, declared in quarot_gptq_entry.h
// (included above) rather than duplicated here, mirroring how ApplyAwq
// (awq_entry.h, also included above) is documented in its own home header
// instead of this one. See quarot.py's own module docstring for the
// technique and quarot_gptq_entry.h for this port's own scope (including
// its accepted numerical scope and its own permanent RNG divergence from
// the Python reference, shared with ApplyQuarot's).

// GPTVQ (Van Baalen et al., 2024): a genuine combination of ApplyGptq's
// own sequential, Hessian-compensated correction with a k-means-fit
// vector codebook (like onnxsim.aqlm's own single shared codebook) --
// small groups of consecutive input-channel columns are jointly
// quantized against the codebook, then each group's resulting per-column
// residual is propagated into every not-yet-quantized column exactly
// like ApplyGptq's own per-column correction -- C++ port of gptvq.py's
// own quantize_weight_only_gptvq, declared in gptvq_entry.h (included
// above) rather than duplicated here, mirroring how ApplyQuarotGptq
// (quarot_gptq_entry.h, also included above) is documented in its own
// home header instead of this one. See gptvq.py's own module docstring
// for the technique and gptvq_entry.h for this port's own scope
// (including its accepted numerical scope and its own permanent RNG
// divergence from the Python reference for the k-means codebook fit).

// OmniQuant (Shao et al., 2023): grid-searched Learnable Weight Clipping
// (LWC) plus a closed-form-shift/grid-searched-scale Learnable Equivalent
// Transformation (LET), applied to every quantize_weight_only_int4-
// quantized MatMul/Gemm layer shared (by node output name) between a float
// model and its quantized counterpart -- C++ port of omniquant.py's own
// apply_omniquant, declared in omniquant_entry.h (included above) rather
// than duplicated here, mirroring how ApplyGptvq (gptvq_entry.h, also
// included above) is documented in its own home header instead of this
// one. See omniquant.py's own module docstring for the technique (and why
// this is a bounded grid search rather than the paper's own gradient
// descent) and omniquant_entry.h for this port's own scope.

// AffineQuant (Ma et al., 2024, ICLR): OmniQuant's own LWC plus a
// block-diagonal (not fully dense) Learnable Equivalent Transformation --
// a per-block orthogonal rotation on top of OmniQuant's own diagonal
// scale/shift, searched against the same reconstruction-error objective --
// C++ port of affinequant.py's own apply_affinequant, built directly on
// ApplyOmniquant's own machinery (omniquant_entry.h, also included above),
// declared in affinequant_entry.h (included above) rather than duplicated
// here. See affinequant.py's own module docstring for the technique and
// affinequant_entry.h for this port's own scope (including its accepted
// eigendecomposition-algorithm divergence for the block rotation).

// BRECQ (Li et al., 2021, ICLR): jointly optimizes every
// quantize_weight_only_int4-quantized MatMul/Gemm layer inside a
// caller-delimited block (a linear chain plus an optional trailing
// residual Add) against the block's own final output reconstruction
// error, Fisher-diagonal weighted -- extending ApplyAdaround's own
// rectified-sigmoid relaxation and hand-rolled Adam loop to a jointly
// optimized block of layers instead of one layer at a time -- C++ port of
// brecq.py's own apply_brecq, declared in brecq_entry.h (included above)
// rather than duplicated here, mirroring how ApplyAdaround
// (adaround_entry.h, also included above) is documented in its own home
// header instead of this one. See brecq.py's own module docstring for the
// technique (and what it simplifies relative to the paper) and
// brecq_entry.h for this port's own scope -- including its accepted
// numerical scope (same class as ApplyAdaround's own: an iterative Adam
// optimization, not a closed-form computation).

// Structured (channel) pruning: removes whole output channels from
// MatMul/vanilla-Gemm and Conv layers -- real structural pruning (smaller
// weight tensors, smaller matmuls on any runtime), as opposed to
// ``PruneMagnitude``'s value-only zeroing. For every MatMul/vanilla-Gemm or
// 2-D Conv "producer" node whose output feeds, through zero or more
// shape-preserving elementwise ops (an activation, or -- MatMul/Gemm only --
// an Add/Mul against a constant per-channel bias/scale, or -- Conv only -- a
// depthwise Conv hop) with no other consumer anywhere along that path, into
// exactly one downstream "consumer" of the same family: ranks the
// producer's output channels by L2 norm of their own weight row/filter,
// drops the lowest-``sparsity``-fraction of them, and removes the
// corresponding rows/columns from the producer's weight (and bias, if
// constant) and every intermediate per-channel constant, and the matching
// columns/rows from the consumer's weight. A general grouped Conv (neither
// ``group=1`` nor fully depthwise) is matched too, as a producer and/or
// consumer, ranking/pruning each of its ``group`` channel blocks
// independently. The gated-FFN SwiGLU/GeGLU pattern is matched too -- two
// producers combined by a ``Mul`` (or ONNX opset-28+'s native ``SwiGLU``
// node) feeding one consumer, both pruned to the same combined-importance-
// ranked channel indices. A Conv or MatMul/Gemm residual (skip-connection)
// chain is matched too -- a channel-preserving ``Add(a, b)`` where both
// operands are non-constant forces whichever real producer(s) feed ``a``/
// ``b`` to agree on one shared channel-index set, resolved via a backward
// walk plus union-find grouping across such merge points that also covers a
// whole chain of such merges transitively sharing one spine channel count
// (a lone residual connection, or a linear stack of ``Add``-only merges).
// Once a group's shared channel-index set is established, it can also fan
// out *forward* to more than one independent ordinary consumer -- so a real
// multi-block ResNet/transformer stage's shared "post-block" tensor, read by
// both the next block's own first Conv/MatMul *and*, unchanged, that
// block's own ``Add``, is now reached rather than declined; a general
// grouped Conv may take part in this merge too, as a producer, the primary
// consumer, and/or an extra fan-out branch, as long as every one of those
// that is grouped shares the exact same ``group`` count. For MatMul/Gemm
// specifically, a fused
// ``com.microsoft::SkipLayerNormalization``/``SkipSimplifiedLayerNormalization``
// node -- what onnxruntime's transformer optimizer collapses a bare
// residual ``Add`` plus the following LayerNorm into, and so what a
// fully-optimized transformer's own residual connections typically look
// like -- is recognized as an eligible merge point too, its own
// ``gamma``/``beta``/``bias`` constants riding along as a per-channel
// affine hop on the resolved chain; a Conv residual chain only ever sees a
// bare ``Add`` (there is no Conv analogue of that fused op). A fused
// ``com.microsoft::BiasGelu``/``FastGelu`` node (a bias-add fused into the
// following Gelu-family activation) is recognized as a per-channel hop too,
// MatMul/Gemm chains only, and ``com.microsoft::QuickGelu`` is a plain
// unary pass-through hop everywhere a unary activation is already allowed.
//
// A ``Concat``-merged skip connection (the U-Net-style encoder/decoder
// merge) is matched too, for both MatMul/Gemm (last-axis ``Concat`` only)
// and Conv (channel-axis ``Concat``): unlike ``Add``, a ``Concat``'s
// branches are structurally independent -- each owns a fixed, disjoint
// offset range of the merged channel range -- so each branch is ranked and
// pruned entirely on its own, by the same L2-norm/combined-importance
// criterion as any other chain here; only the shared downstream consumer's
// weight needs new slicing, at each branch's own fixed offset. A branch may
// itself resolve through a gated (SwiGLU/GeGLU) combine or a whole
// Add/SkipLayerNormalization residual group; a branch that fans out
// elsewhere, or would need to cross another ``Concat`` or a fused
// self-attention op boundary, declines the *entire* group, never partially
// pruned. See ``structured_pruning_entry.cpp`` for the exact algorithm.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly this rewrite, to every
// matching chain, to a copy of ``model`` (which is left untouched) and
// returns the result. ``sparsity`` must be in [0, 1); throws
// ``std::invalid_argument`` otherwise. Anything not matching the exact
// topology above (branching, a non-constant bias, a consumer whose
// reduction dimension doesn't line up, ...) is left completely untouched.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Attention-head pruning: removes whole attention heads -- or, for
// grouped-query attention, whole KV groups -- from every matched
// ``com.microsoft::Attention``, ``com.microsoft::GroupQueryAttention``, or
// plain ``ai.onnx::Attention`` node whose output feeds, optionally through a
// single shape-preserving ``Reshape``, exactly one downstream MatMul/
// vanilla-Gemm's reduction dimension (the output projection) -- the
// attention analogue of ``ApplyStructuredPruning``, at head (or KV-group)
// instead of single-channel granularity.
//
// For each matched plain ``com.microsoft::Attention`` block (a single merged
// QKV weight/bias): ranks every head by the combined Frobenius norm of its
// own Q, K, and V weight columns, drops the lowest-``sparsity``-fraction of
// heads (at least one head is always kept), and removes the corresponding
// column blocks from the merged QKV weight (and bias, if present),
// decrementing ``num_heads``/``qkv_hidden_sizes`` accordingly, and the
// matching row block from the output projection's weight.
//
// For each matched ``GroupQueryAttention`` or plain ``ai.onnx::Attention``
// block (separate, un-merged Q/K/V producers): ranks every *KV group* (a KV
// head and the ``num_heads / kv_num_heads`` query heads the kernel maps to
// it) by the combined Frobenius norm of that group's own Q+K+V weight
// block, drops the lowest-``sparsity``-fraction of groups (at least one
// group is always kept), and removes the corresponding column blocks from
// all three producers (and their biases, if present) together with the
// matching row block from the output projection's weight, decrementing the
// query head count and ``kv_num_heads`` by the number of groups dropped --
// so their ratio (query heads per KV head) is unchanged. An individual
// query head is never dropped on its own: only a whole group, since neither
// kernel has a way to keep a KV head alive for some, but not all, of the
// query heads that shared it.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass. ``sparsity`` must be in [0, 1); throws
// ``std::invalid_argument`` otherwise. Anything not matching that exact
// topology (a non-constant weight, a packed-QKV GroupQueryAttention node, a
// GroupQueryAttention/plain ai.onnx Attention node with a non-empty constant
// past-KV-cache or attention-mask input, an ai.onnx Attention node with
// differing Q/K/V head sizes or without explicit
// ``q_num_heads``/``kv_num_heads`` attributes, a consumer whose reduction
// dimension doesn't line up, ...) is left completely untouched. The
// calibration-driven Wanda upgrade of this same matching/ranking machinery
// (mirroring the pure-Python ``onnxsim.apply_attention_head_wanda_pruning``)
// is ``ApplyAttentionHeadWandaPruning`` in structured_pruning_entry.h --
// declared there rather than duplicated here, alongside
// ``ApplyStructuredWandaPruning``, since both take a ``ModelExecutor&``
// (only forward-declared in this header, see this header's own top comment)
// and this header's own duplicated-prototype convention is otherwise
// reserved for the plain, executor-free entry points.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyAttentionHeadPruning(const onnx::ModelProto& model,
                                           double sparsity);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// MoE expert-intermediate-channel pruning: removes intermediate
// (``inter_size``) channels from every expert of a matched
// ``com.microsoft::MoE`` node at once -- real structural pruning (smaller
// ``fc1_experts_weights``/``fc2_experts_weights``, smaller per-expert
// matmuls on any runtime), data-free.
//
// Ranks every ``inter_size`` index by combined (root-sum-square) L2 norm of
// ``fc1_experts_weights``' own row (across every expert and ``hidden_size``
// at once) and ``fc2_experts_weights``' own column (same reduction), plus
// ``fc1_experts_bias``'s own entry when present, drops the lowest-
// ``sparsity``-fraction of indices (at least one is always kept), and
// removes the matching row from ``fc1_experts_weights``/
// ``fc1_experts_bias`` and column from ``fc2_experts_weights``, identically
// across every expert -- ``num_experts``, ``k``, and every node attribute
// are untouched, since pruning ``inter_size`` changes no other tensor's
// shape anywhere in the graph, including the node's own output.
//
// A node with ``fc3_experts_weights`` present, a ``swiglu``/unrecognized
// ``activation_type``, a non-constant or tied/shared weight, or any other
// shape this pass doesn't recognize is left completely untouched. This is
// the C++ port of ``onnxsim.apply_moe_expert_channel_pruning`` --
// whole-expert pruning (shrinking ``num_experts`` itself, which needs
// runtime calibration data this build has no ONNX Runtime linked in to
// provide) is a deliberately separate, NOT-ported feature. See
// ``structured_pruning_entry.cpp``'s own "MoE (com.microsoft::MoE)
// expert-intermediate-channel pruning" section comment for the full scope
// and safety argument.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass. ``sparsity`` must be in [0, 1); throws
// ``std::invalid_argument`` otherwise.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyMoeExpertChannelPruning(const onnx::ModelProto& model,
                                              double sparsity);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// QMoE expert-channel pruning: removes intermediate (``inter_size``)
// channels from every expert of a matched ``com.microsoft::QMoE`` node at
// once -- the quantized-weight counterpart of ``ApplyStructuredPruning``,
// targeting ``QMoE``'s own packed ``uint8`` ``fc1_experts_weights``/
// ``fc2_experts_weights`` (plus their ``scales``/``zero_points``/
// ``global_scale`` operands, co-sliced in lockstep) instead of plain float
// weights. See ``structured_pruning_entry.cpp``'s own "QMoE (com.microsoft,
// quantized-weight Mixture-of-Experts) expert-channel structured pruning"
// section comment for the exact matched topology.
//
// Supports ``quant_type='int'`` (``expert_weight_bits`` in {2, 4, 8}, with
// no ``block_size`` -- whole-row per-channel scale -- or a groupwise
// ``block_size``) and ``quant_type='nvfp4'`` (E2M1-packed weights,
// ``float8e4m3fn`` per-block scales, a required per-expert ``float32``
// global scale, ``block_size`` fixed at 16); ``'fp4'``, ``'fp8'``, and
// ``'wfp4afp8'`` remain out of scope, as does ``fc3_experts_weights``,
// ``router_weights``, a ``swiglu``/unrecognized ``activation_type``, and a
// CUTLASS-prepacked (``weights_prepacked`` outside {-1, 0}) weight layout.
//
// Ranks every ``inter_size`` index by combined (root-sum-square) L2 norm of
// ``fc1_experts_weights``'/``fc2_experts_weights``' own DEQUANTIZED row/
// column (never written back -- the actual rewrite always slices the
// existing packed codes/scales/zero_points in place, re-packing a sub-byte-
// packed axis rather than ever re-quantizing a sliced float weight from
// scratch) plus ``fc1_experts_bias``'s own entry when present, drops the
// lowest-``sparsity``-fraction of indices (at least one always kept,
// floored to a multiple of ``8 / expert_weight_bits`` -- or, with
// ``block_size`` set, to whole ``block_size``-sized groups, since
// ``fc2_experts_weights``' own quantization blocks group along
// ``inter_size`` and a value can't be dropped out of a shared-scale group
// without re-quantizing it), and removes the matching row from ``fc1``'s
// own weight/scales/bias/zero_points and column from ``fc2``'s own weight
// (plus, only when ``block_size`` is set, ``fc2``'s own scales/
// zero_points too), identically across every expert. ``num_experts``, `k`,
// and every node attribute are untouched.
//
// Unlike pruning.py's own ``apply_qmoe_expert_channel_pruning``, this port
// only admits FLOAT32 (not FLOAT16/BFLOAT16) ``fc1``/``fc2`` scales and
// bias, matching this codebase's C++-port scope decision for MatMulNBits
// above; and does not include the complementary whole-expert-removal pass
// (``onnxsim.apply_qmoe_whole_expert_pruning``), which needs runtime
// calibration data (an ONNX Runtime inference session observing router
// activations) this C++ port has no ONNX Runtime linked into at all.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass. ``sparsity`` must be in [0, 1); throws
// ``std::invalid_argument`` otherwise. Anything not matching the exact
// topology above is left completely untouched.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto ApplyQMoEExpertChannelPruning(const onnx::ModelProto& model,
                                               double sparsity);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Embedding vocabulary pruning: shrinks a matched token-embedding table's
// vocabulary axis (a plain ``Gather``'s ``data`` input feeding a graph
// input's token-id tensor, plus, where a tied or confidently-auto-
// identified untied ``lm_head`` exists, its own vocab-logits projection
// too) down to a caller-supplied, explicit keep-set -- the C++ port of
// pruning.py's own ``apply_embedding_vocab_pruning``. See
// structured_pruning_entry.h's own ``EmbeddingVocabPruningResult``/
// ``ApplyEmbeddingVocabPruning`` doc comments for the full contract
// (**unlike every other pass in this header, the pruned model this
// returns does not accept the original model's own token ids -- see
// those doc comments**), and structured_pruning_entry.cpp's own
// "Embedding vocabulary pruning" section comment for the exact matched
// topology, scope, and the deliberately-narrower-than-pruning.py
// restrictions this C++ port makes (plain ``Gather`` producer only, plain
// ``MatMul``/``Gemm`` ``lm_head`` only, FLOAT32-only tensors).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
EmbeddingVocabPruningResult ApplyEmbeddingVocabPruning(
    const onnx::ModelProto& model,
    const std::optional<std::vector<int64_t>>& keep_token_ids,
    const std::optional<std::vector<int64_t>>& drop_token_ids,
    const std::optional<std::string>& input_name);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// The importance-ranked variant -- see structured_pruning_entry.h's own
// ``ApplyEmbeddingVocabMagnitudePruning`` doc comment, the C++ port of
// pruning.py's own ``apply_embedding_vocab_magnitude_pruning``.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
EmbeddingVocabPruningResult ApplyEmbeddingVocabMagnitudePruning(
    const onnx::ModelProto& model, double sparsity,
    const std::optional<std::vector<int64_t>>& protect_token_ids,
    const std::optional<std::string>& input_name);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the activation tensor names that ``QuantizeStatic`` could quantize in
// ``model`` -- the first input of every MatMul, every "vanilla" Gemm
// (transA=0, alpha=1, beta=1), and every Conv, whose weight is a constant
// float32 tensor (2-D for MatMul/Gemm, rank >= 3 -- [Cout, Cin/groups, k...]
// -- for Conv) and whose activation is float32 -- so a caller can calibrate
// exactly (and only) the tensors that matter, by running the model over
// representative data and recording each listed tensor's observed (min,
// max). Names are deduplicated and given in no particular order; the
// opset-version check ``QuantizeStatic``'s passes apply is not repeated
// here, since calibrating a tensor that then turns out to be unusable
// (opset < 13) is harmless -- the pass simply ignores its entry in
// ``activation_ranges``.
std::vector<std::string> ListQuantizableActivations(
    const onnx::ModelProto& model);

// Statically (calibration-based) quantizes every MatMul, every "vanilla"
// Gemm (transA=0, alpha=1, beta=1), and every Conv, whose weight is a
// constant float32 tensor (2-D for MatMul/Gemm, rank >= 3 for Conv) and
// whose activation's tensor name is a key of ``activation_ranges``: the
// weight is quantized to INT8 ahead of time (per output channel, symmetric,
// from its static values -- same as ``QuantizeDynamic``), and the activation
// is quantized to uint8 using a *fixed* (scale, zero_point) derived from its
// calibrated (min, max) range in ``activation_ranges`` (see
// ``ListQuantizableActivations`` to discover which tensors to calibrate).
// The rewrite inserts a QuantizeLinear/DequantizeLinear pair around each
// quantized tensor (the "QDQ" format) rather than replacing the
// MatMul/Gemm/Conv itself, so the graph still computes in float32 -- a
// QDQ-aware runtime fuses the pattern into a true integer kernel at load
// time. See ``passes/static_quantize_matmul.h`` and
// ``passes/static_quantize_conv.h`` for the rewrites themselves, and
// ``QuantizeDynamic`` for the no-calibration alternative (MatMul/Gemm only).
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly these rewrites, once
// each, to a copy of ``model`` (which is left untouched) and returns the
// result. Nodes that do not match, or whose activation has no entry in
// ``activation_ranges``, are left as-is.
onnx::ModelProto QuantizeStatic(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Same as ``QuantizeStatic``, but a "W8A16" scheme: the weight stays INT8
// (identical per-output-channel symmetric scheme), while the activation is
// quantized to UINT16 instead of UINT8 -- an 8x finer calibrated affine step
// (1/65535 relative vs UINT8's 1/255). Useful for activations a QDQ round
// trip is unusually sensitive to (e.g. post-softmax attention scores, or a
// tensor whose calibrated range is wide relative to its typical value),
// without giving up INT8's weight compression the way widening the weight
// too would. Uses the same ``activation_ranges`` shape and
// ``ListQuantizableActivations`` to discover candidate tensors as
// ``QuantizeStatic``. See ``passes/static_quantize_int16_matmul.h`` and
// ``passes/static_quantize_int16_conv.h`` for the rewrites themselves.
// Needs opset >= 21 (UINT16 QuantizeLinear/DequantizeLinear support),
// unlike ``QuantizeStatic``'s UINT8 scheme, which only needs opset 13.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly these rewrites, once
// each, to a copy of ``model`` (which is left untouched) and returns the
// result. Nodes that do not match, whose activation has no entry in
// ``activation_ranges``, or whose opset is older than 21, are left as-is.
onnx::ModelProto QuantizeStaticInt16(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Lists the *output* tensor names that ``QuantizeQOperator`` could quantize
// in ``model``, on top of the input tensor names ``ListQuantizableActivations``
// already reports -- one entry per MatMul/"vanilla" Gemm/Conv whose weight
// qualifies (same shape ``ListQuantizableActivations`` checks), and, for a
// Conv with a bias, whose bias is also a constant float32 ``[Cout]`` tensor
// (``QLinearConv`` needs its bias pre-quantized to a static INT32 tensor;
// see ``passes/qoperator_quantize_conv.h``). QOperator format's
// ``QLinearMatMul``/``QLinearConv`` compute directly in int8 with no float
// intermediate, so they need a calibrated range for the node's *output* too,
// unlike QDQ format (``QuantizeStatic``), whose DequantizeLinear can leave
// the result in float. A caller preparing to call ``QuantizeQOperator``
// should calibrate the union of this and ``ListQuantizableActivations``.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorQuantizableOutputs(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every MatMul, every "vanilla"
// Gemm (transA=0, alpha=1, beta=1), and every Conv, whose weight is a
// constant float32 tensor (2-D for MatMul/Gemm, rank >= 3 for Conv), whose
// activation's tensor name *and* whose own output's tensor name are both
// keys of ``activation_ranges``: the weight is quantized to INT8 ahead of
// time (per output channel, symmetric, from its static values -- same as
// ``QuantizeStatic``), and both the activation and the output are quantized
// to uint8 using *fixed* (scale, zero_point) pairs derived from their
// calibrated (min, max) ranges (see ``ListQuantizableActivations`` and
// ``ListQOperatorQuantizableOutputs`` to discover which tensors to
// calibrate). Unlike ``QuantizeStatic``'s QDQ format, this replaces the
// MatMul/Gemm/Conv itself with ``QLinearMatMul``/``QLinearConv`` -- ONNX's
// directly-quantized ops (the "QOperator" format) -- so the graph computes
// in true int8 with no float MatMul/Conv left in it. A Conv's bias (if any)
// is quantized to INT32 ahead of time too (``QLinearConv``'s own bias input
// convention), which is why a Conv with a non-constant bias is left alone
// (see ``ListQOperatorQuantizableOutputs``). See
// ``passes/qoperator_quantize_matmul.h``/``passes/qoperator_quantize_conv.h``
// for the rewrites themselves and their doc comments for the QDQ-vs-QOperator
// tradeoff.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding or
// any other simplification pass -- it applies exactly these rewrites, once
// each, to a copy of ``model`` (which is left untouched) and returns the
// result. Nodes that do not match, or whose activation/output has no entry
// in ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperator(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the tensor names ``QuantizeQOperatorElementwise`` could quantize in
// ``model``: for every Add/Mul node with exactly 2 float32 inputs, neither a
// constant, one entry per operand plus one for the node's own output (three
// entries per qualifying node, since QLinearAdd/QLinearMul -- unlike
// QLinearMatMul/QLinearConv -- have no "weight" operand pre-quantized from
// its own static values; both operands are treated as calibrated
// activations). A caller preparing to call ``QuantizeQOperatorElementwise``
// should calibrate this list (there is no separate
// ``ListQuantizableActivations``-style overlap to union it with, since
// MatMul/Conv activation names and Add/Mul operand names never coincide).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorElementwiseQuantizableTensors(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every elementwise Add/Mul node
// whose two inputs are both non-constant float32 tensors, and whose two
// input names *and* whose own output name are all keys of
// ``activation_ranges``, into ONNX Runtime's "com.microsoft" contrib ops
// ``QLinearAdd``/``QLinearMul`` -- the elementwise, "QOperator"-format
// analogue of ``QuantizeQOperator``'s ``QLinearMatMul``/``QLinearConv``
// rewrite. Unlike every other ``Quantize*`` entry point in this header,
// QLinearAdd/QLinearMul are not standard ONNX ops -- they are ONNX Runtime
// contrib ops, so the emitted model needs a "com.microsoft"-aware runtime
// (ONNX Runtime itself, or another runtime importing the same contrib
// schemas) to execute; this function adds "com.microsoft" (version 1) to
// the model's opset imports the first time it rewrites a node. See
// ``ListQOperatorElementwiseQuantizableTensors`` to discover which tensors
// to calibrate, and ``passes/qoperator_quantize_elementwise.h`` for the
// rewrite itself and its doc comment on why a constant operand is left
// alone.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match, or whose operands/output have no entry in
// ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperatorElementwise(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the tensor names ``QuantizeQOperatorActivation`` could quantize in
// ``model``: for every standalone Sigmoid or LeakyRelu node with exactly 1
// float32 input, both the input's and the node's own output's tensor names
// (two entries per qualifying node).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorActivationQuantizableTensors(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every standalone Sigmoid or
// LeakyRelu node whose input is float32, and whose input name *and* whose
// own output name are both keys of ``activation_ranges``, into ONNX
// Runtime's "com.microsoft" contrib ops ``QLinearSigmoid``/
// ``QLinearLeakyRelu`` -- the unary-activation analogue of
// ``QuantizeQOperatorElementwise``'s ``QLinearAdd``/``QLinearMul`` rewrite
// (see that function's doc comment for why these are contrib, not standard,
// ONNX ops, and why the output needs a calibrated range on top of the
// input's). LeakyRelu's ``alpha`` attribute is carried over unchanged. This
// function adds "com.microsoft" (version 1) to the model's opset imports the
// first time it rewrites a node. See
// ``ListQOperatorActivationQuantizableTensors`` to discover which tensors to
// calibrate, and ``passes/qoperator_quantize_activation.h`` for the rewrite
// itself.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match, or whose input/output have no entry in
// ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperatorActivation(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the tensor names ``QuantizeQOperatorConcat`` could quantize in
// ``model``: for every Concat node whose inputs are all non-constant float32
// tensors, one entry per input plus one for the node's own output.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorConcatQuantizableTensors(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every Concat node whose inputs
// are all non-constant float32 tensors, and whose every input name *and*
// whose own output name are all keys of ``activation_ranges``, into ONNX
// Runtime's "com.microsoft" contrib op ``QLinearConcat`` -- the variadic
// analogue of ``QuantizeQOperatorElementwise``'s ``QLinearAdd``/
// ``QLinearMul`` rewrite (see that function's doc comment for why these are
// contrib, not standard, ONNX ops, and why every operand needs a calibrated
// range on top of the output's). This function adds "com.microsoft"
// (version 1) to the model's opset imports the first time it rewrites a
// node. See ``ListQOperatorConcatQuantizableTensors`` to discover which
// tensors to calibrate, and ``passes/qoperator_quantize_concat.h`` for the
// rewrite itself and its doc comment on why a constant operand is left
// alone.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match, or whose operands/output have no entry in
// ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperatorConcat(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the tensor names ``QuantizeQOperatorSoftmax`` could quantize in
// ``model``: for every standalone Softmax node with exactly 1 float32 input
// and a resolvable default-domain opset import, both the input's and the
// node's own output's tensor names (two entries per qualifying node).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorSoftmaxQuantizableTensors(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every standalone Softmax node
// whose input is float32, whose input name *and* whose own output name are
// both keys of ``activation_ranges``, and whose model has a resolvable
// default-domain ("" / "ai.onnx") opset import, into ONNX Runtime's
// "com.microsoft" contrib op ``QLinearSoftmax`` -- the reduction-axis
// analogue of ``QuantizeQOperatorActivation``'s ``QLinearSigmoid``/
// ``QLinearLeakyRelu`` rewrite (see that function's doc comment for why
// these are contrib, not standard, ONNX ops, and why the output needs a
// calibrated range on top of the input's). The ``axis`` attribute is carried
// over unchanged (defaulting to -1 when absent); the model's own
// default-domain opset version is threaded through as ``QLinearSoftmax``'s
// required ``opset`` attribute, so the rewritten node reproduces standard
// ONNX Softmax's exact axis semantics for that opset (pre-13 flattened
// reduction vs. 13+ in-place per-axis reduction) rather than guessing one.
// This function adds "com.microsoft" (version 1) to the model's opset
// imports the first time it rewrites a node. See
// ``ListQOperatorSoftmaxQuantizableTensors`` to discover which tensors to
// calibrate, and ``passes/qoperator_quantize_softmax.h`` for the rewrite
// itself.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match, or whose input/output have no entry in
// ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperatorSoftmax(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the tensor names ``QuantizeQOperatorPool`` could quantize in
// ``model``: for every standalone AveragePool/GlobalAveragePool node with
// exactly 1 float32 input and (for AveragePool) no ``dilations`` attribute,
// both the input's and the node's own output's tensor names (two entries
// per qualifying node).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorPoolQuantizableTensors(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every standalone AveragePool or
// GlobalAveragePool node whose input is float32, whose input name *and*
// whose own output name are both keys of ``activation_ranges``, into ONNX
// Runtime's "com.microsoft" contrib ops ``QLinearAveragePool``/
// ``QLinearGlobalAveragePool`` -- the pooling analogue of
// ``QuantizeQOperatorActivation``'s ``QLinearSigmoid``/``QLinearLeakyRelu``
// rewrite (see that function's doc comment for why these are contrib, not
// standard, ONNX ops, and why the output needs a calibrated range on top of
// the input's). Every attribute the original AveragePool node has
// (kernel_shape, pads, strides, ceil_mode, count_include_pad, auto_pad) is
// carried over unchanged; both ops additionally get a ``channels_last``
// attribute set to 0 (onnxsim only ever produces NCHW-layout graphs). An
// AveragePool node with a ``dilations`` attribute (standard ONNX opset 19+)
// is left untouched -- ONNX Runtime's QLinearAveragePool kernel does not
// accept that attribute. This function adds "com.microsoft" (version 1) to
// the model's opset imports the first time it rewrites a node. See
// ``ListQOperatorPoolQuantizableTensors`` to discover which tensors to
// calibrate, and ``passes/qoperator_quantize_pool.h`` for the rewrite
// itself.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match, or whose input/output have no entry in
// ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperatorPool(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the tensor names ``QuantizeQOperatorWhere`` could quantize in
// ``model``: for every ``Where`` node whose two data operands (inputs 1
// and 2) are both non-constant float32 tensors, the operands' names plus
// the node's own output name (three entries per qualifying node), mirroring
// ``ListQOperatorElementwiseQuantizableTensors``'s convention (no "weight"
// role to distinguish, since neither operand is pre-quantized from its own
// static values).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorWhereQuantizableTensors(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every ``Where`` node whose two
// data operands are both non-constant float32 tensors, and whose two
// operand names *and* whose own output name are all keys of
// ``activation_ranges``, into ONNX Runtime's "com.microsoft" contrib op
// ``QLinearWhere`` -- the ternary-select analogue of
// ``QuantizeQOperatorElementwise``'s ``QLinearAdd``/``QLinearMul`` rewrite
// (see that function's doc comment for why these are contrib, not
// standard, ONNX ops, and why every operand needs a calibrated range on top
// of the output's). The boolean condition operand is never quantized --
// ``QLinearWhere``'s schema passes it straight through as `tensor(bool)`.
// This function adds "com.microsoft" (version 1) to the model's opset
// imports the first time it rewrites a node. See
// ``ListQOperatorWhereQuantizableTensors`` to discover which tensors to
// calibrate, and ``passes/qoperator_quantize_where.h`` for the rewrite
// itself and its doc comment on why a constant operand is left alone.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match, or whose operands/output have no entry in
// ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperatorWhere(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Lists the tensor names ``QuantizeQOperatorGemm`` could quantize in
// ``model``: for every ``Gemm`` node whose weight B is a constant 2-D
// float32 tensor (and, if present, whose bias C is a constant 1-D float32
// tensor of length N with beta == 1), the activation A's name plus the
// node's own output name (two entries per qualifying node -- B and C are
// quantized from their own static values, not calibrated, the same
// "weight" role ``ListQuantizableActivations`` already treats Gemm/MatMul's
// weight as elsewhere).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
std::vector<std::string> ListQOperatorGemmQuantizableTensors(
    const onnx::ModelProto& model);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Statically (calibration-based) quantizes every ``Gemm`` node whose weight
// B is a constant 2-D float32 tensor, whose bias C (if present) is a
// constant 1-D float32 tensor of length N with beta == 1, and whose
// activation A's name *and* whose own output name are both keys of
// ``activation_ranges``, into ONNX Runtime's "com.microsoft" contrib op
// ``QGemm`` -- the fully-general analogue of ``QuantizeQOperator``'s
// ``QLinearMatMul`` rewrite, which only handles "vanilla" Gemm (transA=0,
// alpha=1) because ``QLinearMatMul`` has no transpose/scale attributes of
// its own. ``QGemm`` keeps ``transA``/``transB``/``alpha`` as attributes,
// so this function handles any transA, transB, or alpha value
// ``QuantizeQOperator`` cannot. B is quantized per output channel (INT8,
// symmetric) in its own storage layout (no forced transpose); C, when
// present, is quantized into INT32 with zero_point 0 and a per-column
// scale of ``alpha * a_scale * b_scale[n]`` -- QGemm's own documented bias
// convention -- and accumulated directly in the quantized compute, unlike
// ``QuantizeQOperator``'s vanilla-Gemm handling, which adds the bias back
// in float after dequantizing. This function adds "com.microsoft" (version
// 1) to the model's opset imports the first time it rewrites a node. See
// ``ListQOperatorGemmQuantizableTensors`` to discover which tensors to
// calibrate, and ``passes/qoperator_quantize_gemm.h`` for the rewrite
// itself and its doc comment on the scope of C shapes/beta values handled.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this rewrite, once,
// to a copy of ``model`` (which is left untouched) and returns the result.
// Nodes that do not match, or whose activation/output have no entry in
// ``activation_ranges``, are left as-is.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeQOperatorGemm(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Converts every float32 weight (and, by default, every internal activation)
// in ``model`` to float16 -- a different kind of "quantization" from every
// other ``Quantize*`` function here: float16 is still a floating-point
// format (just narrower than float32), so unlike the INT8/INT4 schemes there
// is no scale, no zero-point, and no calibration data needed at all -- every
// float32 value is simply rounded to its nearest representable float16
// value (values outside float16's finite range are clamped rather than
// rounded to an infinity). See ``passes/quantize_fp16.h`` for the rewrite
// itself.
//
// When ``keep_io_types`` is true (the default), the graph's own external
// input/output types stay float32 -- a ``Cast`` is inserted right after
// each float32 graph input and right before each float32 graph output, so
// the model's public interface is unchanged and only its internal weights
// and compute switch to float16. With ``keep_io_types`` false, graph
// inputs/outputs are redeclared float16 directly instead (no casts).
//
// No node's op_type or attributes are touched, and there is no per-op
// float16-support check: an ordinary feedforward graph ends up computing
// end-to-end in float16 as a side effect of every value along the way now
// being float16-typed, since almost every ONNX op propagates its input
// dtype to its output dtype. A model containing an op with no float16
// kernel in the runtime it is deployed on will fail at *execution* time,
// not at conversion time here -- the same limitation every other
// float32-to-float16 model converter has.
//
// Unlike ``Simplify``, this does not run shape inference, constant folding
// or any other simplification pass -- it applies exactly this one rewrite,
// once, to a copy of ``model`` (which is left untouched) and returns the
// result. Only the top-level graph is converted; nodes inside control-flow
// subgraphs (If/Loop/Scan bodies) are left as-is, and an initializer whose
// name is also a graph input (the rarely-used ONNX "optional input with a
// default value" convention) is left alone entirely.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeFp16(const onnx::ModelProto& model,
                              bool keep_io_types = true);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Converts a model's float32 weights and (by default) internal activations
// to bfloat16. The same kind of calibration-free, whole-graph "quantization"
// as ``QuantizeFp16``, just to a different narrow floating-point format:
// bfloat16 keeps float32's full 8-bit exponent range and narrows only the
// mantissa (7 bits instead of float32's 23), so no clamping is needed --
// every finite float32 value maps to a finite bfloat16 value. See
// ``passes/quantize_bf16.h`` for the rewrite itself; ``keep_io_types``, scope,
// and every other semantic exactly mirror ``QuantizeFp16`` above.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeBf16(const onnx::ModelProto& model,
                              bool keep_io_types = true);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

// Converts a model's float32 weights and (by default) internal activations
// to an 8-bit floating-point format -- the same kind of calibration-free,
// whole-graph "quantization" as ``QuantizeFp16``/``QuantizeBf16``, just to a
// much narrower floating-point format. ``format`` selects which one:
// ``"e4m3"`` (the default -- E4M3FN, 4 exponent bits / 3 mantissa bits, max
// finite magnitude 448, typically used for weights) or ``"e5m2"`` (5
// exponent bits / 2 mantissa bits, max finite magnitude 57344, a dynamic
// range similar to float16, typically used for gradients). Both are
// converted with saturation: a value whose magnitude exceeds the target
// format's max finite value (including +-Inf itself) is clamped to it
// rather than mapped to an infinity/NaN. See ``passes/quantize_fp8.h`` for
// the rewrite itself, including why the FNUZ variants of these formats are
// not offered; ``keep_io_types``, scope, and every other semantic exactly
// mirror ``QuantizeFp16`` above. Casting to/from these types needs opset >=
// 19. Throws ``std::invalid_argument`` if ``format`` is not ``"e4m3"`` or
// ``"e5m2"``.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
onnx::ModelProto QuantizeFp8(const onnx::ModelProto& model,
                             const std::string& format = "e4m3",
                             bool keep_io_types = true);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

void SimplifyPath(
    const ModelExecutor& executor, const std::string& in_path,
    const std::string& out_path,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version = std::nullopt,
    const GraphRewriter* rewriter = nullptr,
    bool initializers_as_constants = true,
    bool include_inline_functions = false, bool mutable_initializer = true,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes = std::nullopt,
    const std::optional<std::vector<std::string>>& unused_output = std::nullopt,
    const std::optional<std::vector<std::string>>& extra_optimizers =
        std::nullopt);
