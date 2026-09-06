// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Decomposes mmdeploy's custom `TRTBatchedNMS` op (the TensorRT-plugin
// multiclass-batched-NMS post-processing op mmdeploy emits at the end of
// most 2-D and BEV/3-D detection pipelines, e.g. `mmdeploy.mmcv.ops.nms`)
// into a subgraph built purely from standard ONNX ops centered on
// `NonMaxSuppression`. There is no ONNX Runtime kernel (nor a reference-
// evaluator kernel) for `TRTBatchedNMS` -- it is a closed TensorRT plugin --
// so today onnxsim cannot even load, let alone simplify, a graph containing
// it. This is a pure graph-shape rewrite (one node becomes several dozen),
// so it is `PassType::Other` and never runs by default. Opt in with
// `extra_optimizers=["rewrite_trt_batched_nms"]` (Python) or
// `--enable-optimization rewrite_trt_batched_nms` (CLI).
//
// KNOWN LIMITATION -- read before trusting bit-exactness: `TRTBatchedNMS`'s
// exact numerical behavior (greedy-NMS tie-breaking when two boxes have
// identical scores, floating-point rounding inside the plugin, the precise
// interaction of `topK`/`keepTopK` with duplicate scores at the cutoff) is
// not fully documented outside NVIDIA's closed TensorRT plugin source (see
// `plugin/batchedNMSPlugin` / `efficientNMSPlugin` in the public TensorRT OSS
// repo for the closest available description). This rewrite instead
// implements a standard, well-specified "per-class greedy NMS, then global
// per-batch top-K merge" algorithm (spelled out step by step below) and
// aims only for producing the same SET of kept detections (by box + class +
// score, up to `keepTopK` and padding) as that well-specified algorithm --
// NOT bit-identical output, and NOT identical tie-breaking order against the
// real plugin. Treat any exact-order or exact-tie match against a real
// TensorRT-plugin-produced reference as coincidental.
//
// Op spec (as emitted by mmdeploy; see `mmdeploy.mmcv.ops.nms`):
//   Inputs (2, fixed order):
//     0. `boxes`:  FLOAT (N, num_boxes, num_classes_or_1, 4), corner format.
//     1. `scores`: FLOAT (N, num_boxes, num_classes).
//   Attributes (all static):
//     background_label_id (int, default -1): a class index to exclude
//       entirely from consideration if >= 0.
//     num_classes (int): authoritative static class count.
//     topK (int): max boxes retained per class going into NMS.
//     keepTopK (int): max final detections per batch item after merging
//       across classes, sorted by score descending.
//     scoreThreshold (float): scores below this are discarded pre-NMS.
//     iouThreshold (float): standard greedy-NMS IoU threshold.
//     isNormalized (int/bool, optional): irrelevant here -- see below.
//     clipBoxes (int/bool, optional, default false): if true, clip boxes to
//       [0,1] before NMS.
//   Outputs (4, fixed order):
//     0. `num_detections`: (N, 1), INT32 or INT64 (matches the node's own
//        declared output type; INT32 if undeclared -- see below).
//     1. `nmsed_boxes`:   FLOAT (N, keepTopK, 4), zero-padded past
//        `num_detections[n]`.
//     2. `nmsed_scores`:  FLOAT (N, keepTopK), zero-padded past
//        `num_detections[n]`.
//     3. `nmsed_classes`: FLOAT (N, keepTopK), padded with -1.0 (chosen over
//        0.0 so padding is distinguishable from real class 0) past
//        `num_detections[n]`.
//
// Box-coordinate convention: this pass never interprets box coordinates
// itself -- it only threads `boxes` through `NonMaxSuppression`'s own IoU
// computation and gathers/pads the results. `NonMaxSuppression` is invoked
// with `center_point_box=0` (corner format, `y1<y2,x1<x2` OR `x1<x2,y1<y2`
// -- whichever layout the exporter already guarantees), matching what
// mmdeploy's own ONNXRuntime-backend export path does for the equivalent
// non-TRT graph. Whatever convention the input `boxes` actually use is
// simply passed through unchanged.
//
// Scope (the predicate declines outside this):
//  - `boxes`' 3rd dim (`num_classes_or_1`) must be statically known and
//    exactly 1 -- i.e. boxes shared/class-agnostic across classes (the
//    overwhelmingly common case: mmdetection's typical class-agnostic bbox
//    head). ONNX `NonMaxSuppression` itself has no notion of per-class
//    boxes, so a `num_classes_or_1 > 1` graph is out of scope for this pass
//    entirely (declined, left untouched).
//  - `N` (batch size) must be statically known: `runTransform` unrolls a
//    per-batch-item C++ loop at pass-build time to perform the final
//    per-batch top-K merge (step 6 below). A dynamic-batch version would
//    need an ONNX `Loop` op, out of scope for this pass.
//  - `num_boxes` and `num_classes` (scores' other two dims) may be dynamic;
//    only `num_classes`, when `background_label_id >= 0`, must be knowable
//    (statically from `scores`' own shape, else from the `num_classes`
//    attribute -- see step 3) to size the background-exclusion mask
//    constant.
//  - `topK`, `keepTopK` must both be present and positive.
//  - Requires opset >= 13 (the floor set by this rewrite's Squeeze/
//    Unsqueeze, which use the opset-13+ axes-as-input form; every other op
//    used -- NonMaxSuppression opset 10, TopK's tensor `K` input opset 10 /
//    `sorted` attribute opset 11, GatherND opset 11, Pad's input-based
//    `pads`/`constant_value` opset 11, Compress unchanged since opset 9 --
//    needs no more than that).
//
// Derivation / decomposition, per original `TRTBatchedNMS` node:
//
// 1. `boxes_sq = Squeeze(boxes, axes=[2])` -> (N, num_boxes, 4). If
//    `clipBoxes` is present and true, `boxes_sq = Clip(boxes_sq, 0, 1)`
//    immediately after (assumption: `clipBoxes` implies coordinates are
//    already normalized to [0,1] before clipping -- `isNormalized` itself is
//    read nowhere, since this pass never otherwise interprets coordinates).
//    Both NMS's own IoU computation and the final gathered output boxes use
//    this (possibly clipped) tensor, so the two stay consistent.
// 2. `scores_t = Transpose(scores, perm=[0,2,1])` -> (N, num_classes,
//    num_boxes) -- the layout ONNX `NonMaxSuppression` wants (class axis
//    before box axis).
// 3. If `background_label_id >= 0`: rather than threading an exclusion
//    through a per-class loop (unnecessary -- see step 4), build a
//    `(1, num_classes, 1)` additive mask constant that is `-1e9` at
//    `background_label_id` and `0` everywhere else, and
//    `scores_t = Add(scores_t, mask)`. This pushes that one class's scores
//    so far below any realistic `scoreThreshold` that `NonMaxSuppression`
//    (step 4) never selects it, without needing a separate per-class branch.
//    `num_classes` for sizing this mask is taken from `scores`' own static
//    class dim when known (always exactly matches the runtime tensor), else
//    from the `num_classes` attribute (assumed to match the actual runtime
//    class count -- undefined behavior, silently wrong broadcasting, if it
//    doesn't; this is the spec's own stated contract for the attribute).
// 4. `NonMaxSuppression(boxes_sq, scores_t, max_output_boxes_per_class=
//    Const(topK), iou_threshold=Const(iouThreshold),
//    score_threshold=Const(scoreThreshold), center_point_box=0)` ->
//    `selected_indices`, shape `(num_selected, 3)`, INT64, each row
//    `[batch_index, class_index, box_index]`. This single call already
//    performs step-1-of-the-reference-algorithm's "per class, greedy NMS,
//    keep at most topK" for every `(batch, class)` pair at once -- ONNX
//    `NonMaxSuppression` is inherently per-batch-per-class internally, so no
//    separate per-class C++-unrolled loop is needed, only the per-batch-item
//    loop in step 6 (merging across classes).
// 5. Split `selected_indices` into its three columns via `Slice`
//    (`b_idx`, `c_idx`, `box_idx`, each `(num_selected, 1)`), then:
//      `boxes_sel  = GatherND(boxes_sq,  Concat([b_idx, box_idx], axis=1))`
//        -> (num_selected, 4)
//      `scores_sel = GatherND(scores_t,  Concat([b_idx, c_idx, box_idx],
//        axis=1))` -> (num_selected,) (gathered from the pre-mask-`Add`
//        `scores_t`'s numeric values is equivalent here: the background
//        class, if any, is never among `selected_indices` in the first
//        place, so its masked-vs-unmasked value never matters)
//      `classes_sel = Cast(c_idx, FLOAT)` -> (num_selected, 1)
// 6. Per batch item `n` in `0..N-1` (a static, C++-unrolled loop -- see
//    Scope above):
//    a. `mask_n = Squeeze(Equal(b_idx, Const(n)), axes=[1])` -> bool
//       `(num_selected,)`; `Compress` each of `scores_sel`/`boxes_sel`/
//       `classes_sel` along axis 0 by `mask_n` -> this batch item's
//       variable-length `scores_n`/`boxes_n`/`classes_n`.
//    b. Pad each up to at least `keepTopK` rows before `TopK` (so
//       `TopK(k=keepTopK)` is always safe regardless of how many survived):
//       `count_n = Shape(scores_n)[0]`; `pad_amount = Max(keepTopK -
//       count_n, 0)`; `Pad(..., pads=[[0],[pad_amount]], constant_value=
//       -1e9)` for `scores_n` (rank 1), `constant_value=0` for `boxes_n`
//       (rank 2, padding axis 0 only), `constant_value=-1.0` for `classes_n`
//       (rank 2, same). All three pad by the same `pad_amount`, so row `i`
//       stays aligned across the three padded tensors.
//    c. `values, indices = TopK(scores_n_padded, k=keepTopK, axis=0,
//       largest=1, sorted=1)`. `values` is already this batch item's final
//       (sorted-descending, sentinel-tail) score row; gather `boxes_n_padded`
//       and `classes_n_padded` at `indices` (axis 0) for the matching
//       boxes/classes rows.
//    d. A row is "real" (not a sentinel pad row) iff its score is still
//       above roughly half the sentinel value: `valid = Greater(values,
//       -5e8)` (any real score, even a very permissive negative
//       `scoreThreshold`, is assumed > -5e8 -- documented assumption, not
//       enforced). `num_detections[n] = ReduceSum(Cast(valid, out_dtype),
//       axis=0, keepdims=1)`. `nmsed_scores[n] = Where(valid, values, 0.0)`
//       (turns the sentinel tail into the spec's zero-padding;
//       `nmsed_boxes[n]` is already zero past `count_n` courtesy of the
//       zero-fill `Pad` in 6b, and `nmsed_classes[n]` is already `-1.0`
//       there for the same reason -- no extra `Where` needed for either).
//    e. `Unsqueeze` each of this batch item's four results on axis 0 (adding
//       the batch axis back) and `Concat` all `N` batch items' chunks along
//       axis 0 for each of the four final outputs.
//
// Multi-output replacement: `TRTBatchedNMS` has 4 outputs (unlike most
// rewrites in this codebase, which replace a single-output node), so this
// pass calls `tryReplacingAllUsesWith` once per output index rather than
// once for the whole node, bailing out (declining the rewrite, leaving the
// original node alone) if any single one fails -- see
// `tryReplacingAllUsesWith(Value*, Value*)` in `onnxoptimizer/pass.h`
// (unlike its `Node*, Node*` overload, there is no built-in "replace all 4
// outputs atomically" helper, since our four new outputs are the tails of
// four independently-built chains rather than one new node's outputs).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Small node-construction helper bound to one `TRTBatchedNMS` rewrite: every
// node it creates is inserted immediately before `anchor` (the
// `TRTBatchedNMS` node itself), and scalar float/int64 constants are cached
// so the many repeated literals (0, 1, axis indices, ...) across the
// per-batch-item loop share one initializer apiece.
struct TRTBatchedNMSBuilder {
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

  Value* ConstFVec(const std::vector<float>& v) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_FLOAT;
    t.sizes().push_back(static_cast<int64_t>(v.size()));
    for (float f : v) {
      t.floats().push_back(f);
    }
    return graph.addInitializerAndCreateValue(std::move(t));
  }

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

  // Rank-1, single-element int64 tensor -- what the opset-13+ axes-as-input
  // form of Squeeze/Unsqueeze wants, and also reused for Slice's
  // starts/ends/axes/steps inputs (each rank-1 length-1 here, since this
  // pass only ever slices/axis-indexes one element/axis at a time).
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

  Value* UnOp(Symbol op, Value* a, int32_t elem_type) {
    Node* n = graph.create(op, 1);
    n->addInput(a);
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
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
  Value* Max(Value* a, Value* b) {
    return BinOp(Symbol("Max"), a, b, a->elemType());
  }
  Value* Equal(Value* a, Value* b) {
    return BinOp(Symbol("Equal"), a, b, TensorProto_DataType_BOOL);
  }
  Value* Greater(Value* a, Value* b) {
    return BinOp(kGreater, a, b, TensorProto_DataType_BOOL);
  }

  Value* Where(Value* cond, Value* a, Value* b) {
    Node* n = graph.create(Symbol("Where"), 1);
    n->addInput(cond);
    n->addInput(a);
    n->addInput(b);
    n->insertBefore(anchor);
    n->output()->setElemType(a->elemType());
    return n->output();
  }

  Value* Clip(Value* a, Value* lo, Value* hi) {
    Node* n = graph.create(Symbol("Clip"), 1);
    n->addInput(a);
    n->addInput(lo);
    n->addInput(hi);
    n->insertBefore(anchor);
    n->output()->setElemType(a->elemType());
    return n->output();
  }

  Value* CastTo(Value* a, int32_t to) {
    Node* n = graph.create(kCast, 1);
    n->addInput(a);
    n->i_(kto, static_cast<int64_t>(to));
    n->insertBefore(anchor);
    n->output()->setElemType(to);
    return n->output();
  }

  Value* Shape(Value* a) {
    return UnOp(Symbol("Shape"), a, TensorProto_DataType_INT64);
  }

  // General Gather: `indices` may be rank-0 (drops `axis`) or rank-N (splices
  // in `indices`' shape in place of `axis`) -- both are handled uniformly by
  // ONNX `Gather`'s own semantics, so this one helper covers every use in
  // this pass (extracting a single scalar off a Shape() result, or gathering
  // whole rows by a `TopK`-produced index vector).
  Value* Gather(Value* data, Value* indices, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* GatherND(Value* data, Value* indices) {
    Node* n = graph.create(Symbol("GatherND"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // `data`'s columns `[col, col+1)` along axis 1, i.e. `data[:, col:col+1]`
  // -- used to split NonMaxSuppression's `(num_selected, 3)` selected-index
  // rows into their three `(num_selected, 1)` columns.
  Value* SliceCol(Value* data, int64_t col) {
    Node* n = graph.create(kSlice, 1);
    n->addInput(data);
    n->addInput(ConstI64Vec1(col));
    n->addInput(ConstI64Vec1(col + 1));
    n->addInput(ConstI64Vec1(1));  // axes = [1]
    n->addInput(ConstI64Vec1(1));  // steps = [1]
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

  Value* Squeeze(Value* a, int64_t axis) {
    Node* n = graph.create(kSqueeze, 1);
    n->addInput(a);
    n->addInput(ConstI64Vec1(axis));
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

  Value* ReduceSum(Value* data, int64_t axis, bool keepdims,
                   int32_t out_type) {
    Node* n = graph.create(kReduceSum, 1);
    n->addInput(data);
    n->addInput(ConstI64Vec1(axis));
    n->i_(kkeepdims, keepdims ? 1 : 0);
    n->insertBefore(anchor);
    n->output()->setElemType(out_type);
    return n->output();
  }

  // opset-11+ input-based Pad, mode="constant" (the default -- left unset).
  Value* Pad(Value* data, Value* pads, Value* constant_value) {
    Node* n = graph.create(kPad, 1);
    n->addInput(data);
    n->addInput(pads);
    n->addInput(constant_value);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* Compress(Value* data, Value* condition, int64_t axis) {
    Node* n = graph.create(Symbol("Compress"), 1);
    n->addInput(data);
    n->addInput(condition);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // Returns {values, indices}; `data` must be rank-1 here (this pass only
  // ever calls TopK on a single batch item's already-flat score vector).
  std::pair<Value*, Value*> TopK(Value* data, int64_t k, int64_t axis) {
    Node* n = graph.create(Symbol("TopK"), 2);
    n->addInput(data);
    n->addInput(ConstI64Vec1(k));
    n->i_(kaxis, axis);
    n->i_(Symbol("largest"), 1);
    n->i_(Symbol("sorted"), 1);
    n->insertBefore(anchor);
    n->outputs()[0]->setElemType(data->elemType());
    n->outputs()[1]->setElemType(TensorProto_DataType_INT64);
    return {n->outputs()[0], n->outputs()[1]};
  }

  Value* NMS(Value* boxes, Value* scores, Value* max_boxes_per_class,
            Value* iou_threshold, Value* score_threshold) {
    Node* n = graph.create(Symbol("NonMaxSuppression"), 1);
    n->addInput(boxes);
    n->addInput(scores);
    n->addInput(max_boxes_per_class);
    n->addInput(iou_threshold);
    n->addInput(score_threshold);
    n->i_(Symbol("center_point_box"), 0);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }
};

struct RewriteTRTBatchedNMS final : public PredicateBasedPass {
  explicit RewriteTRTBatchedNMS()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_trt_batched_nms";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("TRTBatchedNMS")) {
      return false;
    }
    // Match domain "" (unset) and "mmdeploy"; decline anything else (e.g. a
    // differently-shaped same-named op from some other exporter/plugin).
    if (node->has_domain() && !node->domain().empty() &&
        node->domain() != "mmdeploy") {
      return false;
    }
    if (node->inputs().size() != 2 || node->outputs().size() != 4) {
      return false;
    }
    Value* boxes = node->input(0);
    Value* scores = node->input(1);
    if (boxes->elemType() != TensorProto_DataType_FLOAT ||
        scores->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (!boxes->has_sizes() || boxes->sizes().size() != 4) {
      return false;
    }
    if (!scores->has_sizes() || scores->sizes().size() != 3) {
      return false;
    }
    // Scope: class-agnostic boxes only (num_classes_or_1 == 1, statically
    // known) -- see this file's header comment.
    const Dimension& cls_or_1 = boxes->sizes()[2];
    if (!cls_or_1.is_int || cls_or_1.dim != 1) {
      return false;
    }
    // Scope: N (batch size) must be statically known -- this pass unrolls a
    // per-batch-item C++ loop at pass-build time.
    const Dimension& N_dim = boxes->sizes()[0];
    if (!N_dim.is_int || N_dim.dim <= 0) {
      return false;
    }
    const int64_t topK =
        GetValueFromAttrWithDefault<int64_t>(node, Symbol("topK"), int64_t(-1));
    const int64_t keepTopK = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("keepTopK"), int64_t(-1));
    if (topK <= 0 || keepTopK <= 0) {
      return false;
    }
    const int64_t background_label_id = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("background_label_id"), int64_t(-1));
    if (background_label_id >= 0) {
      // Need *some* statically-known positive class count to size the
      // background-exclusion mask constant -- either scores' own shape, or
      // (failing that) the num_classes attribute.
      const Dimension& cls_dim = scores->sizes()[2];
      const bool have_static_num_classes = cls_dim.is_int && cls_dim.dim > 0;
      const int64_t attr_num_classes = GetValueFromAttrWithDefault<int64_t>(
          node, Symbol("num_classes"), int64_t(-1));
      if (!have_static_num_classes && attr_num_classes <= 0) {
        return false;
      }
    }
    // Floor set by this rewrite's own opset-13+ Squeeze/Unsqueeze
    // (axes-as-input) usage -- see this file's header comment.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 13;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* boxes = node->input(0);
    Value* scores = node->input(1);

    const int64_t N = boxes->sizes()[0].dim;
    const int64_t topK =
        GetValueFromAttrWithDefault<int64_t>(node, Symbol("topK"), int64_t(-1));
    const int64_t keepTopK = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("keepTopK"), int64_t(-1));
    const float scoreThreshold = GetValueFromAttrWithDefault<float>(
        node, Symbol("scoreThreshold"), 0.0f);
    const float iouThreshold = GetValueFromAttrWithDefault<float>(
        node, Symbol("iouThreshold"), 0.45f);
    const int64_t background_label_id = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("background_label_id"), int64_t(-1));
    const bool clipBoxes = GetValueFromAttrWithDefault<int64_t>(
                              node, Symbol("clipBoxes"), int64_t(0)) != 0;

    TRTBatchedNMSBuilder b{graph, node};

    // 1. Drop the size-1 class dim off boxes, optionally clip to [0,1].
    Value* boxes_sq = b.Squeeze(boxes, 2);  // (N, num_boxes, 4)
    if (clipBoxes) {
      boxes_sq = b.Clip(boxes_sq, b.ConstF(0.0f), b.ConstF(1.0f));
    }

    // 2. (N, num_boxes, num_classes) -> (N, num_classes, num_boxes).
    Value* scores_t = b.Transpose(scores, {0, 2, 1});

    // 3. Exclude background_label_id (if any) by pushing its scores far
    // below any realistic scoreThreshold, rather than a separate per-class
    // branch -- see this file's header comment.
    if (background_label_id >= 0) {
      const Dimension& cls_dim = scores->sizes()[2];
      const int64_t num_classes =
          (cls_dim.is_int && cls_dim.dim > 0)
              ? cls_dim.dim
              : GetValueFromAttrWithDefault<int64_t>(
                    node, Symbol("num_classes"), int64_t(1));
      std::vector<float> mask(static_cast<size_t>(num_classes), 0.0f);
      if (background_label_id < num_classes) {
        mask[static_cast<size_t>(background_label_id)] = -1e9f;
      }
      Value* mask_1d = b.ConstFVec(mask);              // (num_classes,)
      Value* mask_3d = b.Unsqueeze(mask_1d, 0);          // (1, num_classes)
      mask_3d = b.Unsqueeze(mask_3d, 2);                 // (1, num_classes, 1)
      scores_t = b.Add(scores_t, mask_3d);
    }

    // 4. One NonMaxSuppression call handles every (batch, class) pair.
    Value* selected_indices =
        b.NMS(boxes_sq, scores_t, b.ConstI64Scalar(topK), b.ConstF(iouThreshold),
             b.ConstF(scoreThreshold));  // (num_selected, 3) int64

    // 5. Split into columns and gather the actual box/score/class values.
    Value* b_idx = b.SliceCol(selected_indices, 0);    // (num_selected, 1)
    Value* c_idx = b.SliceCol(selected_indices, 1);
    Value* box_idx = b.SliceCol(selected_indices, 2);

    Value* boxes_sel =
        b.GatherND(boxes_sq, b.Concat(1, {b_idx, box_idx}));  // (S, 4)
    Value* scores_sel =
        b.GatherND(scores_t, b.Concat(1, {b_idx, c_idx, box_idx}));  // (S,)
    Value* classes_sel = b.CastTo(c_idx, TensorProto_DataType_FLOAT);  // (S,1)

    // Output dtype for num_detections: match the node's own declared type,
    // defaulting to INT32 when undeclared (both are seen in real
    // mmdeploy/TensorRT exports; INT32 is this pass's documented default).
    int32_t num_det_dtype = node->outputs()[0]->elemType();
    if (num_det_dtype == TensorProto_DataType_UNDEFINED) {
      num_det_dtype = TensorProto_DataType_INT32;
    }

    const float kSentinelScore = -1e9f;
    const float kSentinelCheck = -5e8f;  // see header comment, step 6d.

    std::vector<Value*> boxes_chunks, scores_chunks, classes_chunks,
        numdet_chunks;
    boxes_chunks.reserve(static_cast<size_t>(N));
    scores_chunks.reserve(static_cast<size_t>(N));
    classes_chunks.reserve(static_cast<size_t>(N));
    numdet_chunks.reserve(static_cast<size_t>(N));

    // 6. Per batch item: filter by b_idx == n, pad, TopK-merge across
    // classes, and reassemble the padded (1, keepTopK, ...) chunk.
    for (int64_t n = 0; n < N; ++n) {
      Value* mask_n = b.Squeeze(b.Equal(b_idx, b.ConstI64Scalar(n)), 1);

      Value* scores_n = b.Compress(scores_sel, mask_n, 0);   // (count_n,)
      Value* boxes_n = b.Compress(boxes_sel, mask_n, 0);     // (count_n, 4)
      Value* classes_n = b.Compress(classes_sel, mask_n, 0); // (count_n, 1)

      // pad_amount = max(keepTopK - count_n, 0), computed at runtime since
      // count_n is dynamic.
      Value* count_n =
          b.Gather(b.Shape(scores_n), b.ConstI64Scalar(0), 0);  // scalar
      Value* pad_amount =
          b.Max(b.Sub(b.ConstI64Scalar(keepTopK), count_n), b.ConstI64Scalar(0));
      Value* pad_amount_vec = b.Unsqueeze(pad_amount, 0);  // (1,)

      Value* pads_1d =
          b.Concat(0, {b.ConstI64Vec1(0), pad_amount_vec});  // (2,)
      Value* pads_2d = b.Concat(
          0, {b.ConstI64Vec1(0), b.ConstI64Vec1(0), pad_amount_vec,
              b.ConstI64Vec1(0)});  // (4,)

      Value* scores_n_pad = b.Pad(scores_n, pads_1d, b.ConstF(kSentinelScore));
      Value* boxes_n_pad = b.Pad(boxes_n, pads_2d, b.ConstF(0.0f));
      Value* classes_n_pad = b.Pad(classes_n, pads_2d, b.ConstF(-1.0f));

      auto [values, indices] = b.TopK(scores_n_pad, keepTopK, 0);
      Value* boxes_n_final = b.Gather(boxes_n_pad, indices, 0);      // (K,4)
      Value* classes_n_final_2d = b.Gather(classes_n_pad, indices, 0);  // (K,1)
      Value* classes_n_final = b.Squeeze(classes_n_final_2d, 1);      // (K,)

      Value* valid = b.Greater(values, b.ConstF(kSentinelCheck));
      Value* scores_n_final = b.Where(valid, values, b.ConstF(0.0f));
      Value* numdet_n =
          b.ReduceSum(b.CastTo(valid, num_det_dtype), 0, true, num_det_dtype);

      boxes_chunks.push_back(b.Unsqueeze(boxes_n_final, 0));    // (1,K,4)
      scores_chunks.push_back(b.Unsqueeze(scores_n_final, 0));  // (1,K)
      classes_chunks.push_back(b.Unsqueeze(classes_n_final, 0));// (1,K)
      numdet_chunks.push_back(b.Unsqueeze(numdet_n, 0));        // (1,1)
    }

    Value* final_boxes = b.Concat(0, boxes_chunks);
    Value* final_scores = b.Concat(0, scores_chunks);
    Value* final_classes = b.Concat(0, classes_chunks);
    Value* final_numdet = b.Concat(0, numdet_chunks);

    // N and keepTopK are both static (predicate-enforced / attribute), so
    // every final output shape is fully static regardless of num_boxes.
    final_numdet->setSizes({Dimension{N}, Dimension{1}});
    final_boxes->setSizes({Dimension{N}, Dimension{keepTopK}, Dimension{4}});
    final_scores->setSizes({Dimension{N}, Dimension{keepTopK}});
    final_classes->setSizes({Dimension{N}, Dimension{keepTopK}});

    Value* new_outputs[4] = {final_numdet, final_boxes, final_scores,
                             final_classes};
    for (int i = 0; i < 4; ++i) {
      if (!tryReplacingAllUsesWith(node->outputs()[i], new_outputs[i])) {
        return false;
      }
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
