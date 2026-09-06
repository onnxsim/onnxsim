// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Decomposes mmdeploy/mmcv's custom `MMCVMultiScaleDeformableAttention` op --
// the deformable-attention op mmdeploy emits when exporting BEVFormer,
// Deformable DETR, and similar detection-transformer models -- into a
// subgraph built from standard ONNX ops (`Split`, `Reshape`, `Transpose`,
// `Gather`, `Concat`, `Mul`, `Sub`, `ReduceSum`, and one `GridSample` per
// feature-map level). There is no ONNX Runtime kernel for this custom op, so
// without this rewrite onnxsim cannot even load/shape-infer/constant-fold a
// graph containing it, let alone simplify one; after this rewrite the graph
// runs on any backend with a `GridSample` kernel (which `GridSample` itself
// covers, or which `rewrite_gridsample_to_gather` can further lower for
// backends -- e.g. TensorRT plugin-less export pipelines -- with no
// `GridSample` kernel either; the two passes compose in either simplify()
// call by opting into both).
//
// This is a pure graph-shape rewrite (a couple dozen ops replacing one custom
// node, not a node-count reduction), so it is `PassType::Other` and never
// runs by default. Opt in with
// `extra_optimizers=["rewrite_msdeformattn_to_gridsample"]` (Python) or
// `--enable-optimization rewrite_msdeformattn_to_gridsample` (CLI).
//
// Op spec matched (the predicate declines outside this):
//  - `node->kind() == "MMCVMultiScaleDeformableAttention"`, domain `""`
//    (empty/default -- some exporters emit it there) or `"mmdeploy"`. Any
//    other domain is left alone (a same-named op from an unrelated vendor).
//  - Exactly 5 inputs, 1 output, in mmdeploy's fixed order:
//      0. `value`             FLOAT (bs, num_keys, M, D)
//      1. `spatial_shapes`    INT64 (L, 2), row `l` = `(H_l, W_l)`
//      2. `level_start_index` INT64 (L,) -- unused by this rewrite (see
//         below), checked only as a sanity/shape predicate since real
//         mmdeploy exports always provide it.
//      3. `sampling_locations` FLOAT (bs, num_queries, M, L, P, 2), values in
//         `[0, 1]` normalized coordinates (NOT `[-1, 1]` -- converting that is
//         part of the algorithm, see step 0 below).
//      4. `attention_weights`  FLOAT (bs, num_queries, M, L, P)
//    `M` (num_heads), `D` (embed_dims_per_head), `L` (num_levels) and `P`
//    (num_points) must all be statically known from the above shapes *and*
//    agree with each other (`value`'s M/D against `sampling_locations`'/
//    `attention_weights`'s M; `spatial_shapes`'s L against
//    `sampling_locations`'/`attention_weights`'s L; `sampling_locations`'s P
//    against `attention_weights`'s P) -- `L` in particular must be static
//    because it determines the number of times this pass unrolls its
//    per-level subgraph and the number of `Split` outputs. `bs`, `num_keys`
//    and `num_queries` may all be fully dynamic/symbolic: every place they
//    are needed is read from `Shape(value)` / `Shape(sampling_locations)` at
//    runtime, or handled by `Reshape`'s own `-1`-infers-this-axis mechanism,
//    never assumed static. The *values* inside `spatial_shapes` (the actual
//    `H_l`/`W_l`) are likewise read at runtime via `Gather`, never assumed
//    constant-foldable -- mirroring `rewrite_gridsample_to_gather`'s own
//    stance on `H`/`W`.
//  - The `im2col_step` attribute (a CUDA-kernel batching knob with no effect
//    on output values) is ignored entirely -- not even read.
//  - Requires opset >= 16 (the version `GridSample`, which this rewrite
//    emits, itself requires; this is a defensive check mirroring
//    `rewrite_gridsample_to_gather`'s own, since a node using this custom op
//    could in principle appear in a graph declaring any default-domain
//    opset).
//
// Derivation, verified against mmcv's pure-PyTorch fallback
// (`multi_scale_deformable_attn_pytorch`, the ground-truth semantics this
// custom op implements -- reproduced independently in NumPy in this pass's test
// file and checked against the *simplified* graph's actual output, since no
// ONNX kernel exists to compare against the original custom node):
//
//   def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
//                                            sampling_locations,
//                                            attention_weights):
//     bs, _, num_heads, embed_dims = value.shape
//     _, num_queries, num_heads, num_levels, num_points, _ = \
//         sampling_locations.shape
//     value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
//                               dim=1)
//     sampling_grids = 2 * sampling_locations - 1
//     sampling_value_list = []
//     for level, (H_, W_) in enumerate(value_spatial_shapes):
//       value_l_ = value_list[level].flatten(2).transpose(1, 2) \
//                      .reshape(bs * num_heads, embed_dims, H_, W_)
//       sampling_grid_l_ = sampling_grids[:, :, :, level] \
//                              .transpose(1, 2).flatten(0, 1)
//       sampling_value_l_ = F.grid_sample(
//           value_l_, sampling_grid_l_, mode='bilinear',
//           padding_mode='zeros', align_corners=False)
//       sampling_value_list.append(sampling_value_l_)
//     attention_weights = attention_weights.transpose(1, 2).reshape(
//         bs * num_heads, 1, num_queries, num_levels * num_points)
//     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
//               * attention_weights).sum(-1).view(bs, num_heads * embed_dims,
//                                                  num_queries)
//     return output.transpose(1, 2).contiguous()
//
// 0. `sampling_grids = 2 * sampling_locations - 1` converts `[0,1]`-normalized
//    coordinates to the `[-1,1]` convention `GridSample` itself expects.
//    Built once, up front, over the full `(bs,num_queries,M,L,P,2)` tensor --
//    it is sliced per level afterward, exactly mirroring the reference's own
//    `sampling_grids[:, :, :, level]` indexing.
// 1. Splitting `value` into one chunk per level needs each chunk's flattened
//    pixel count `H_l * W_l`, read from `spatial_shapes` at runtime (never
//    assumed constant): `col0 = Gather(spatial_shapes, [0], axis=1)`,
//    `col1 = Gather(spatial_shapes, [1], axis=1)` (each `(L,1)`, a *vector*
//    index of length 1 so the gathered axis survives as size 1, unlike a
//    scalar index which would drop it), `split_sizes =
//    Reshape(Mul(col0, col1), [-1])` -> `(L,)`. `Split(value, split_sizes,
//    axis=1)` then produces exactly `L` outputs, each `(bs, H_l*W_l, M, D)`
//    with `H_l*W_l` dynamic -- `Split` only needs the *number* of outputs
//    fixed at graph-build time (it is, `L` is static), not their sizes.
//    `level_start_index` is deliberately not used for this -- deriving chunk
//    boundaries directly from `spatial_shapes` via `Split` is equivalent
//    (mmdeploy always exports `level_start_index` as the running sum of
//    exactly these products) and more robust, since it does not depend on
//    that input's actual runtime values being consistent with
//    `spatial_shapes`'s.
// 2. Per level `l` (a plain C++ loop unrolled at pass-build time -- valid
//    since `L` is static):
//    a. Reshape `value_splits[l]` `(bs, H_l*W_l, M, D)` -> `(bs, H_l*W_l,
//       M*D)` (merging the trailing two axes, both statically sized on the
//       `M*D` side) -> `Transpose [0,2,1]` -> `(bs, M*D, H_l*W_l)` -> Reshape
//       to `(bs*M, D, H_l, W_l)`.
//
//       Every Reshape's target shape in this pass is built explicitly as a
//       runtime `Concat` of one-element int64 tensors (`bs`/`num_queries`
//       read fresh off `Shape(value)`/`Shape(sampling_locations)`, `H_l`/
//       `W_l` read off `spatial_shapes`, static dims as int64 literals, and
//       `-1` for the one axis that should be inferred from the total element
//       count) -- deliberately never `Reshape`'s own "target dim `0` means
//       copy the input's dimension at this same position" shortcut. That
//       shortcut is only safe when a reshape's leading axes are literally
//       unchanged in both value *and position*; several reshapes in this
//       pass (this one included -- `bs*M` in position 0 is a genuinely new,
//       computed quantity, not position 0 of the input, which was plain
//       `bs`) change what occupies a given axis position, where blindly
//       reusing `0` would silently copy the *wrong* input dimension. Building
//       every target shape the same explicit way throughout, rather than
//       mixing in the shortcut only where it happens to be safe, removes the
//       need to re-derive per reshape which case applies.
//    b. `level_grid = Gather(sampling_grids, l, axis=3)` (scalar index ->
//       drops axis 3) -> `(bs, num_queries, M, P, 2)` -> `Transpose
//       [0,2,1,3,4]` -> `(bs, M, num_queries, P, 2)` -> Reshape to `(bs*M,
//       num_queries, P, 2)` (`num_queries` read fresh off
//       `Shape(sampling_locations)`, per the explicit-shape policy above --
//       this is precisely the case where the position-0-copies-input-dim
//       shortcut would silently pick up `M`, the input's actual dimension at
//       output position 1, instead of `num_queries`).
//    c. `level_sampled = GridSample(value_reshaped_l, level_grid_reshaped,
//       mode="linear", padding_mode="zeros", align_corners=0)` -> `(bs*M, D,
//       num_queries, P)`. This is exactly mmcv's own `F.grid_sample(...,
//       mode='bilinear', padding_mode='zeros', align_corners=False)` call --
//       emitted as a real `GridSample` node rather than decomposed further
//       (that is `rewrite_gridsample_to_gather`'s job; the two passes
//       compose when both are requested). Deliberately the only place this
//       pass does real numeric work -- everything else is pure reshaping,
//       which is why this pass is much smaller than
//       `rewrite_gridsample_to_gather` (tens of nodes, not hundreds).
//    d. `Unsqueeze(level_sampled, axis=3)` -> `(bs*M, D, num_queries, 1, P)`,
//       collected across levels -- the size-1 axis is where the `L` levels
//       get stacked next, mirroring the reference's `torch.stack(...,
//       dim=-2)`.
// 3. `Concat(axis=3, level_outputs)` -> `(bs*M, D, num_queries, L, P)` ->
//    Reshape merging the trailing `(L, P)` -> `(bs*M, D, num_queries, L*P)`,
//    matching the reference's `.flatten(-2)`.
// 4. `attention_weights` `(bs, num_queries, M, L, P)` -> `Transpose
//    [0,2,1,3,4]` -> `(bs, M, num_queries, L, P)` -> Reshape to `(bs*M, 1,
//    num_queries, L*P)`, matching the reference's own transpose+reshape of
//    `attention_weights` (the explicit leading `1` axis is what lets the
//    subsequent `Mul` broadcast one attention weight across all `D`
//    channels).
// 5. `weighted = Mul(concat_flat, attn_reshaped)` (broadcasts the size-1 `D`
//    axis of `attn_reshaped`) -> `ReduceSum(weighted, axes=[3], keepdims=0)`
//    -> `(bs*M, D, num_queries)`, matching the reference's final `*
//    ...).sum(-1)`.
//    (`ReduceSum`'s `axes` is passed as the op's second *input*, not an
//    attribute -- required since opset 13, and this pass already requires
//    opset >= 16.)
// 6. Reshape `(bs*M, D, num_queries)` -> `(bs, M, D, num_queries)` -> Reshape
//    merging `(M, D)` -> `(bs, M*D, num_queries)` -> `Transpose [0,2,1]` ->
//    `(bs, num_queries, M*D)`. This is the final output, matching the
//    reference's `.view(bs, num_heads*embed_dims, num_queries)` followed by
//    `.transpose(1, 2)` and replacing the original node's own output (its
//    declared sizes, if any, are propagated the same way
//    `rewrite_gridsample_to_gather` does).

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Small node-construction helper bound to one
// `MMCVMultiScaleDeformableAttention` rewrite: every node it creates is
// inserted immediately before `anchor` (the custom op node itself), and
// scalar/vector int64 and float constants are cached so literals shared across
// levels (`D`, `M*D`, `P`, `L*P`, `-1`, ...) reuse one initializer instead of a
// fresh one per level.
struct MSDeformAttnToGridSampleBuilder {
  Graph& graph;
  Node* anchor;

  std::unordered_map<uint32_t, Value*> float_cache;
  std::unordered_map<int64_t, Value*> i64_scalar_cache;
  std::unordered_map<int64_t, Value*> i64_vec1_cache;

  Value* ConstF(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    auto it = float_cache.find(bits);
    if (it != float_cache.end()) {
      return it->second;
    }
    Tensor t;
    t.elem_type() = TensorProto_DataType_FLOAT;
    t.floats().push_back(v);
    Value* val = graph.addInitializerAndCreateValue(std::move(t));
    float_cache.emplace(bits, val);
    return val;
  }

  // Rank-0 (scalar) int64 constant -- an index that, used with Gather, drops
  // the gathered axis from the output.
  Value* ConstI64Scalar(int64_t v) {
    auto it = i64_scalar_cache.find(v);
    if (it != i64_scalar_cache.end()) {
      return it->second;
    }
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.int64s().push_back(v);
    Value* val = graph.addInitializerAndCreateValue(std::move(t));
    i64_scalar_cache.emplace(v, val);
    return val;
  }

  // Rank-1, single-element int64 tensor -- used both as Gather's index (kept,
  // not dropped, so the gathered axis survives as size 1) and as one segment
  // of a runtime-built Reshape/Concat target shape.
  Value* ConstI64Vec1(int64_t v) {
    auto it = i64_vec1_cache.find(v);
    if (it != i64_vec1_cache.end()) {
      return it->second;
    }
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(1);
    t.int64s().push_back(v);
    Value* val = graph.addInitializerAndCreateValue(std::move(t));
    i64_vec1_cache.emplace(v, val);
    return val;
  }

  Value* BinOp(Symbol op, Value* a, Value* b, int32_t elem_type) {
    Node* n = graph.create(op, 1);
    n->addInput(a);
    n->addInput(b);
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
  }

  Value* Add(Value* a, Value* b) { return BinOp(kAdd, a, b, a->elemType()); }
  Value* Sub(Value* a, Value* b) { return BinOp(kSub, a, b, a->elemType()); }
  Value* Mul(Value* a, Value* b) { return BinOp(kMul, a, b, a->elemType()); }

  Value* Shape(Value* a) {
    Node* n = graph.create(Symbol("Shape"), 1);
    n->addInput(a);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  // General Gather: pass a scalar index (ConstI64Scalar) to drop `axis` from
  // the output, or a length-1 vector index (ConstI64Vec1) to keep it as a
  // size-1 axis.
  Value* Gather(Value* data, Value* indices, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // The `idx`-th entry of the 1-D int64 tensor `vec` (e.g. a `Shape(...)`
  // result, or a gathered `spatial_shapes` row), as a rank-1 single-element
  // int64 tensor ready to feed into a `Concat`-built Reshape target shape.
  Value* DimAt(Value* vec, int64_t idx) {
    return Unsqueeze(Gather(vec, ConstI64Scalar(idx), 0), 0);
  }

  Value* Reshape(Value* data, Value* shape) {
    Node* n = graph.create(kReshape, 1);
    n->addInput(data);
    n->addInput(shape);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* Transpose(Value* a, std::vector<int64_t> perm) {
    Node* n = graph.create(kTranspose, 1);
    n->addInput(a);
    n->is_(kperm, std::move(perm));
    n->insertBefore(anchor);
    n->output()->setElemType(a->elemType());
    return n->output();
  }

  Value* Unsqueeze(Value* a, int64_t axis) {
    Node* n = graph.create(kUnsqueeze, 1);
    n->addInput(a);
    n->addInput(ConstI64Vec1(axis));
    n->insertBefore(anchor);
    n->output()->setElemType(a->elemType());
    return n->output();
  }

  Value* Concat(int64_t axis, const std::vector<Value*>& inputs) {
    Node* n = graph.create(kConcat, 1);
    for (Value* v : inputs) {
      n->addInput(v);
    }
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(inputs[0]->elemType());
    return n->output();
  }

  // `split_sizes`: 1-D int64 tensor of length `num_outputs`, read at runtime
  // (never assumed constant-foldable). `num_outputs` itself (here, `L`) must
  // be known at pass-build time -- that is all `Split` needs statically.
  std::vector<Value*> Split(Value* data, Value* split_sizes, int64_t axis,
                            int64_t num_outputs) {
    Node* n = graph.create(Symbol("Split"), static_cast<size_t>(num_outputs));
    n->addInput(data);
    n->addInput(split_sizes);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    std::vector<Value*> outs;
    outs.reserve(static_cast<size_t>(num_outputs));
    for (int64_t i = 0; i < num_outputs; ++i) {
      Value* o = n->outputs()[static_cast<size_t>(i)];
      o->setElemType(data->elemType());
      outs.push_back(o);
    }
    return outs;
  }

  Value* GridSampleOp(Value* X, Value* grid) {
    Node* n = graph.create(Symbol("GridSample"), 1);
    n->addInput(X);
    n->addInput(grid);
    n->s_(kmode, "linear");
    n->s_(Symbol("padding_mode"), "zeros");
    n->i_(Symbol("align_corners"), int64_t(0));
    n->insertBefore(anchor);
    n->output()->setElemType(X->elemType());
    return n->output();
  }

  // `axes`: 1-D int64 tensor (opset >= 13's input form, not the pre-13
  // attribute form -- this pass already requires opset >= 16).
  Value* ReduceSum(Value* data, Value* axes, bool keepdims) {
    Node* n = graph.create(Symbol("ReduceSum"), 1);
    n->addInput(data);
    n->addInput(axes);
    n->i_(Symbol("keepdims"), keepdims ? int64_t(1) : int64_t(0));
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }
};

struct RewriteMSDeformAttnToGridSample final : public PredicateBasedPass {
  explicit RewriteMSDeformAttnToGridSample()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_msdeformattn_to_gridsample";
  }

  // Validates the fixed 5-input/1-output shape/dtype contract and extracts
  // the statically-required dims (M, D, L, P), cross-checked for consistency
  // across the inputs that carry them. Shared by patternMatchPredicate and
  // runTransform so the two can never disagree about what was matched.
  static bool ExtractDims(Node* node, int64_t& M, int64_t& D, int64_t& L,
                          int64_t& P) {
    if (node->inputs().size() != 5 || node->outputs().size() != 1) {
      return false;
    }
    Value* value = node->input(0);
    Value* spatial_shapes = node->input(1);
    Value* level_start_index = node->input(2);
    Value* sampling_locations = node->input(3);
    Value* attention_weights = node->input(4);

    if (value->elemType() != TensorProto_DataType_FLOAT ||
        spatial_shapes->elemType() != TensorProto_DataType_INT64 ||
        level_start_index->elemType() != TensorProto_DataType_INT64 ||
        sampling_locations->elemType() != TensorProto_DataType_FLOAT ||
        attention_weights->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }

    if (!value->has_sizes() || value->sizes().size() != 4) {
      return false;
    }
    if (!spatial_shapes->has_sizes() || spatial_shapes->sizes().size() != 2) {
      return false;
    }
    if (!level_start_index->has_sizes() ||
        level_start_index->sizes().size() != 1) {
      return false;
    }
    if (!sampling_locations->has_sizes() ||
        sampling_locations->sizes().size() != 6) {
      return false;
    }
    if (!attention_weights->has_sizes() ||
        attention_weights->sizes().size() != 5) {
      return false;
    }

    const auto& vsz = value->sizes();
    const auto& ssz = spatial_shapes->sizes();
    const auto& lssz = level_start_index->sizes();
    const auto& lsz = sampling_locations->sizes();
    const auto& asz = attention_weights->sizes();

    // M, D: from `value`. bs, num_keys (vsz[0], vsz[1]) may be dynamic.
    if (!vsz[2].is_int || !vsz[3].is_int) {
      return false;
    }
    M = vsz[2].dim;
    D = vsz[3].dim;

    // L: from `spatial_shapes` (only the shape, never the runtime H_l/W_l
    // values, need be static). Its trailing axis is structurally 2
    // ((H_l, W_l) pairs) -- check it only when statically known.
    if (!ssz[0].is_int) {
      return false;
    }
    L = ssz[0].dim;
    if (ssz[1].is_int && ssz[1].dim != 2) {
      return false;
    }

    // level_start_index: sanity check only (unused by the rewrite itself,
    // see this file's header comment) -- rank 1, and its length must agree
    // with L when statically known.
    if (lssz[0].is_int && lssz[0].dim != L) {
      return false;
    }

    // M, L, P (and the trailing (x,y)-pair axis) from `sampling_locations`;
    // bs, num_queries (lsz[0], lsz[1]) may be dynamic.
    if (!lsz[2].is_int || !lsz[3].is_int || !lsz[4].is_int) {
      return false;
    }
    if (lsz[2].dim != M || lsz[3].dim != L) {
      return false;
    }
    P = lsz[4].dim;
    if (lsz[5].is_int && lsz[5].dim != 2) {
      return false;
    }

    // M, L, P from `attention_weights`, cross-checked against the above.
    if (!asz[2].is_int || !asz[3].is_int || !asz[4].is_int) {
      return false;
    }
    if (asz[2].dim != M || asz[3].dim != L || asz[4].dim != P) {
      return false;
    }

    return M > 0 && D > 0 && L > 0 && P > 0;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("MMCVMultiScaleDeformableAttention")) {
      return false;
    }
    // mmdeploy exports this op under domain "" or "mmdeploy" depending on
    // export path; a same-named op in any other (vendor/plugin) domain is
    // left alone.
    const std::string domain = node->has_domain() ? node->domain() : "";
    if (!domain.empty() && domain != "mmdeploy") {
      return false;
    }

    int64_t M, D, L, P;
    if (!ExtractDims(node, M, D, L, P)) {
      return false;
    }

    // GridSample (emitted by this rewrite) itself requires opset >= 16; this
    // is a defensive check, matching rewrite_gridsample_to_gather's own
    // stance, not something a well-formed model using this custom op could
    // otherwise violate.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 16;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    int64_t M, D, L, P;
    if (!ExtractDims(node, M, D, L, P)) {
      return false;
    }

    Value* value = node->input(0);
    Value* spatial_shapes = node->input(1);
    Value* sampling_locations = node->input(3);
    Value* attention_weights = node->input(4);

    MSDeformAttnToGridSampleBuilder b{graph, node};

    // bs, num_queries as runtime int64 vec1s -- read fresh off Shape(...) of
    // an original input, never assumed static (see header comment).
    Value* shape_value = b.Shape(value);  // (bs, num_keys, M, D)
    Value* bs_v = b.DimAt(shape_value, 0);
    Value* shape_loc = b.Shape(sampling_locations);  // (bs, nq, M, L, P, 2)
    Value* nq_v = b.DimAt(shape_loc, 1);

    Value* D_vec = b.ConstI64Vec1(D);
    Value* M_vec = b.ConstI64Vec1(M);
    Value* MD_vec = b.ConstI64Vec1(M * D);
    Value* P_vec = b.ConstI64Vec1(P);
    Value* LP_vec = b.ConstI64Vec1(L * P);
    Value* one_vec = b.ConstI64Vec1(1);
    Value* two_vec = b.ConstI64Vec1(2);
    Value* neg1_vec = b.ConstI64Vec1(-1);

    // split_sizes[l] = H_l * W_l, read at runtime from spatial_shapes -- see
    // header comment step 1. Vector (length-1) indices keep axis 1 as size 1
    // so the two columns can be multiplied elementwise before flattening to
    // (L,).
    Value* col0 = b.Gather(spatial_shapes, b.ConstI64Vec1(0), 1);  // (L,1)
    Value* col1 = b.Gather(spatial_shapes, b.ConstI64Vec1(1), 1);  // (L,1)
    Value* split_sizes = b.Reshape(b.Mul(col0, col1), neg1_vec);   // (L,)
    std::vector<Value*> value_splits = b.Split(value, split_sizes, 1, L);

    // sampling_grids = 2 * sampling_locations - 1, converting [0,1] ->
    // [-1,1] once over the full tensor (header comment step 0).
    Value* sampling_grids =
        b.Sub(b.Mul(sampling_locations, b.ConstF(2.0f)), b.ConstF(1.0f));

    std::vector<Value*> level_outputs;
    level_outputs.reserve(static_cast<size_t>(L));
    for (int64_t l = 0; l < L; ++l) {
      Value* row_l = b.Gather(spatial_shapes, b.ConstI64Scalar(l), 0);  // (2,)
      Value* H_l = b.DimAt(row_l, 0);
      Value* W_l = b.DimAt(row_l, 1);

      // (bs, H_l*W_l, M, D) -> (bs, H_l*W_l, M*D) -> (bs, M*D, H_l*W_l) ->
      // (bs*M, D, H_l, W_l).
      Value* reshaped1 = b.Reshape(value_splits[static_cast<size_t>(l)],
                                   b.Concat(0, {bs_v, neg1_vec, MD_vec}));
      Value* transposed1 = b.Transpose(reshaped1, {0, 2, 1});
      Value* value_reshaped_l =
          b.Reshape(transposed1, b.Concat(0, {neg1_vec, D_vec, H_l, W_l}));

      // (bs, nq, M, P, 2) -> (bs, M, nq, P, 2) -> (bs*M, nq, P, 2).
      Value* level_grid = b.Gather(sampling_grids, b.ConstI64Scalar(l), 3);
      Value* transposed_grid = b.Transpose(level_grid, {0, 2, 1, 3, 4});
      Value* level_grid_reshaped = b.Reshape(
          transposed_grid, b.Concat(0, {neg1_vec, nq_v, P_vec, two_vec}));

      // (bs*M, D, nq, P) -> (bs*M, D, nq, 1, P).
      Value* level_sampled =
          b.GridSampleOp(value_reshaped_l, level_grid_reshaped);
      level_outputs.push_back(b.Unsqueeze(level_sampled, 3));
    }

    // (bs*M, D, nq, L, P) -> (bs*M, D, nq, L*P).
    Value* concat_result = b.Concat(3, level_outputs);
    Value* concat_flat =
        b.Reshape(concat_result, b.Concat(0, {neg1_vec, D_vec, nq_v, LP_vec}));

    // (bs, nq, M, L, P) -> (bs, M, nq, L, P) -> (bs*M, 1, nq, L*P).
    Value* attn_transposed = b.Transpose(attention_weights, {0, 2, 1, 3, 4});
    Value* attn_reshaped = b.Reshape(
        attn_transposed, b.Concat(0, {neg1_vec, one_vec, nq_v, LP_vec}));

    // (bs*M, D, nq, L*P) -> (bs*M, D, nq) after summing out L*P.
    Value* weighted = b.Mul(concat_flat, attn_reshaped);
    Value* reduced = b.ReduceSum(weighted, b.ConstI64Vec1(3), false);

    // (bs*M, D, nq) -> (bs, M, D, nq) -> (bs, M*D, nq) -> (bs, nq, M*D).
    Value* reshaped_bmdnq =
        b.Reshape(reduced, b.Concat(0, {bs_v, M_vec, D_vec, nq_v}));
    Value* reshaped_bmdnq2 =
        b.Reshape(reshaped_bmdnq, b.Concat(0, {bs_v, MD_vec, nq_v}));
    Value* final_out = b.Transpose(reshaped_bmdnq2, {0, 2, 1});

    if (!node->output()->sizes().empty()) {
      final_out->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), final_out);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
