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
#include "structured_pruning_entry.h"

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
// ``Simplify``'s own peak memory (see bench/RESULTS_synthetic_decoder_oom.md
// for measurements and bench/TODO_large_decoder_submodule_oom.md for the
// original report this traces back to).
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
onnx::ModelProto CrossLayerEqualize(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeDynamicMatMulIntegerToFloat(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeAttentionDynamic(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeTernary(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeWeightOnly(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeWeightOnlyInt4(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeWeightOnlyMatMulNBits(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeWeightOnlyInt16(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeWeightOnlyInt8Block(const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeWeightOnlyMXFP4(const onnx::ModelProto& model);

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
onnx::ModelProto ApplyDoubleQuantization(const onnx::ModelProto& model);

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
onnx::ModelProto PruneMagnitude(const onnx::ModelProto& model, double sparsity,
                                const std::optional<int64_t>& n = std::nullopt,
                                const std::optional<int64_t>& m = std::nullopt,
                                bool global_sparsity = false);

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
onnx::ModelProto ApplyAnyPrecisionLlm(const onnx::ModelProto& model,
                                      int64_t bits, int64_t max_bits,
                                      int64_t block_size);

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
onnx::ModelProto ApplyQuarot(const onnx::ModelProto& model, uint64_t seed,
                             int64_t block_size, float epsilon);

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
onnx::ModelProto ApplyIQ4NL(const onnx::ModelProto& model);

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
onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity);

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
onnx::ModelProto ApplyAttentionHeadPruning(const onnx::ModelProto& model,
                                           double sparsity);

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
onnx::ModelProto ApplyMoeExpertChannelPruning(const onnx::ModelProto& model,
                                              double sparsity);

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
onnx::ModelProto ApplyQMoEExpertChannelPruning(const onnx::ModelProto& model,
                                               double sparsity);

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
EmbeddingVocabPruningResult ApplyEmbeddingVocabPruning(
    const onnx::ModelProto& model,
    const std::optional<std::vector<int64_t>>& keep_token_ids,
    const std::optional<std::vector<int64_t>>& drop_token_ids,
    const std::optional<std::string>& input_name);

// The importance-ranked variant -- see structured_pruning_entry.h's own
// ``ApplyEmbeddingVocabMagnitudePruning`` doc comment, the C++ port of
// pruning.py's own ``apply_embedding_vocab_magnitude_pruning``.
EmbeddingVocabPruningResult ApplyEmbeddingVocabMagnitudePruning(
    const onnx::ModelProto& model, double sparsity,
    const std::optional<std::vector<int64_t>>& protect_token_ids,
    const std::optional<std::string>& input_name);

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
std::vector<std::string> ListQOperatorQuantizableOutputs(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperator(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

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
std::vector<std::string> ListQOperatorElementwiseQuantizableTensors(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperatorElementwise(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Lists the tensor names ``QuantizeQOperatorActivation`` could quantize in
// ``model``: for every standalone Sigmoid or LeakyRelu node with exactly 1
// float32 input, both the input's and the node's own output's tensor names
// (two entries per qualifying node).
std::vector<std::string> ListQOperatorActivationQuantizableTensors(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperatorActivation(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Lists the tensor names ``QuantizeQOperatorConcat`` could quantize in
// ``model``: for every Concat node whose inputs are all non-constant float32
// tensors, one entry per input plus one for the node's own output.
std::vector<std::string> ListQOperatorConcatQuantizableTensors(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperatorConcat(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Lists the tensor names ``QuantizeQOperatorSoftmax`` could quantize in
// ``model``: for every standalone Softmax node with exactly 1 float32 input
// and a resolvable default-domain opset import, both the input's and the
// node's own output's tensor names (two entries per qualifying node).
std::vector<std::string> ListQOperatorSoftmaxQuantizableTensors(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperatorSoftmax(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Lists the tensor names ``QuantizeQOperatorPool`` could quantize in
// ``model``: for every standalone AveragePool/GlobalAveragePool node with
// exactly 1 float32 input and (for AveragePool) no ``dilations`` attribute,
// both the input's and the node's own output's tensor names (two entries
// per qualifying node).
std::vector<std::string> ListQOperatorPoolQuantizableTensors(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperatorPool(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Lists the tensor names ``QuantizeQOperatorWhere`` could quantize in
// ``model``: for every ``Where`` node whose two data operands (inputs 1
// and 2) are both non-constant float32 tensors, the operands' names plus
// the node's own output name (three entries per qualifying node), mirroring
// ``ListQOperatorElementwiseQuantizableTensors``'s convention (no "weight"
// role to distinguish, since neither operand is pre-quantized from its own
// static values).
std::vector<std::string> ListQOperatorWhereQuantizableTensors(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperatorWhere(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

// Lists the tensor names ``QuantizeQOperatorGemm`` could quantize in
// ``model``: for every ``Gemm`` node whose weight B is a constant 2-D
// float32 tensor (and, if present, whose bias C is a constant 1-D float32
// tensor of length N with beta == 1), the activation A's name plus the
// node's own output name (two entries per qualifying node -- B and C are
// quantized from their own static values, not calibrated, the same
// "weight" role ``ListQuantizableActivations`` already treats Gemm/MatMul's
// weight as elsewhere).
std::vector<std::string> ListQOperatorGemmQuantizableTensors(
    const onnx::ModelProto& model);

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
onnx::ModelProto QuantizeQOperatorGemm(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

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
onnx::ModelProto QuantizeFp16(const onnx::ModelProto& model,
                              bool keep_io_types = true);

// Converts a model's float32 weights and (by default) internal activations
// to bfloat16. The same kind of calibration-free, whole-graph "quantization"
// as ``QuantizeFp16``, just to a different narrow floating-point format:
// bfloat16 keeps float32's full 8-bit exponent range and narrows only the
// mantissa (7 bits instead of float32's 23), so no clamping is needed --
// every finite float32 value maps to a finite bfloat16 value. See
// ``passes/quantize_bf16.h`` for the rewrite itself; ``keep_io_types``, scope,
// and every other semantic exactly mirror ``QuantizeFp16`` above.
onnx::ModelProto QuantizeBf16(const onnx::ModelProto& model,
                              bool keep_io_types = true);

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
onnx::ModelProto QuantizeFp8(const onnx::ModelProto& model,
                             const std::string& format = "e4m3",
                             bool keep_io_types = true);

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
