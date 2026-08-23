#pragma once

#include <onnx/onnx_pb.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"

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
    const std::optional<std::vector<std::string>>& unused_output =
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
    const std::optional<std::vector<std::string>>& unused_output =
        std::nullopt);
