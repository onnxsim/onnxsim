// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Decomposes `bev_pool_v2` -- the LSS-style (Lift-Splat-Shoot) camera-to-BEV
// voxel/feature pooling op at the core of BEVDet's and BEVFusion's view
// transform -- into a subgraph built purely from standard ONNX ops centered
// on opset-16+ `ScatterND(reduction="add")`. `bev_pool_v2` ships as a
// bespoke CUDA op / TensorRT plugin in BEVDet's own deployment tooling (not
// mainline mmdeploy); it has no ONNX Runtime kernel and no `onnx` reference-
// evaluator kernel, so today onnxsim cannot even load, let alone simplify, a
// graph containing it. This is a pure graph-shape rewrite (one node becomes
// a couple dozen), so it is `PassType::Other` and never runs by default.
// Opt in with `extra_optimizers=["rewrite_bev_pool_to_scatter"]` (Python) or
// `--enable-optimization rewrite_bev_pool_to_scatter` (CLI).
//
// NAMING UNCERTAINTY -- read before assuming this matches a real export:
// unlike this codebase's other mmdeploy/mmcv op-decomposition passes (whose
// op_type/domain/attribute contracts were read directly off known mmcv/
// mmdeploy source), `bev_pool_v2` is not part of mmdeploy's own op set --
// it is BEVDet's own bespoke plugin, exported via a less standardized path.
// The exact `op_type`/`domain` string a real export uses is genuinely
// unconfirmed here; `"bev_pool_v2"` (BEVDet's own Python function/CUDA-op
// name) in domain `""` or `"mmdeploy"` is this pass's best-effort guess.
// That guess lives in exactly two places -- `kBevPoolOpTypeCandidates` and
// `kBevPoolDomainCandidates` below -- specifically so it is trivial to widen
// (add another candidate string) once a real export is available to check
// against, without touching anything else in this file. The core
// gather/gather/mul/scatter-add algorithm below has real, verifiable ground
// truth (it is independently NumPy-checked in this pass's test file) and is
// the part of this pass worth trusting; the op_type/domain/attribute-name
// matching is the part that may need adjusting later.
//
// Op semantics (see also the module docstring of this pass's test file):
// given per-camera depth-bin probabilities `depth` and per-pixel camera
// features `feat`, plus three precomputed index arrays mapping each valid
// (camera, depth-bin, pixel) triple to a flat depth index, a flat feature
// index, and a flat target BEV-grid index, the op computes, for every valid
// triple, `contribution = depth_value * feature_vector`, then scatter-adds
// (sums) each contribution into its target BEV grid cell. No learned
// parameters, no convolution -- gather + multiply + grouped-sum.
//
//   Inputs (5 required, 2 optional trailing -- see below):
//     0. `depth`: FLOAT (B, N, D, H, W) -- per-pixel, per-depth-bin
//        probability.
//     1. `feat`:  FLOAT (B, N, H, W, C) -- per-pixel camera feature vector.
//     2. `ranks_depth`: INT32 or INT64 (num_valid,) -- flat index into
//        `depth[b].reshape(N*D*H*W)`, IDENTICAL for every batch item `b`
//        (fixed camera geometry, independent of the input images).
//     3. `ranks_feat`: INT32 or INT64 (num_valid,) -- flat index into
//        `feat[b].reshape(N*H*W)` (no `D` factor -- `feat` has no depth-bin
//        axis), likewise identical across `b`.
//     4. `ranks_bev`: INT32 or INT64 (num_valid,) -- flat target index into
//        the `bev_z*bev_h*bev_w` output grid, likewise identical across `b`.
//     5,6. `interval_starts`, `interval_lengths` (optional): a segment-
//        grouping optimization for BEVDet's real CUDA kernel's internal
//        parallelization strategy. This rewrite accepts them (if present)
//        but never reads either one -- `ScatterND(reduction="add")` performs
//        the exact same grouped-sum without needing that grouping metadata
//        at all, so there is nothing for them to contribute here.
//   Output: FLOAT, either (B, C, Z, H, W) if `bev_z > 1` or the node's own
//     declared output is rank 5, or (B, C, H, W) otherwise.
//
// The output BEV grid shape (`bev_z`/`bev_h`/`bev_w`) must be available
// somehow; `DetermineBevGridShape` below tries, in order:
//   1. The node's own declared output shape, if its rank is 4 or 5 and the
//      spatial/level dims (skipping the batch and channel axes) are all
//      statically known -- the most reliable source when present, since it
//      is exactly what shape inference already propagated for this node.
//   2. `bev_h`/`bev_w` int attributes (both required, both > 0), with
//      `bev_z` an optional int attribute defaulting to 1.
//   3. A single `bev_feat_shape` or `output_shape` int-list attribute,
//      either `[H, W]` (`bev_z` implied 1) or `[Z, H, W]`.
// Whichever source succeeds also decides the output rank (4 vs. 5): the
// node's own declared output rank when known, else 4 iff the resolved
// `bev_z == 1`. If none of the three succeeds, the predicate declines --
// there is no way to size the `ScatterND` target without it.
//
// Scope (the predicate declines outside this):
//  - `depth`, `feat` must be FLOAT and exactly rank 5. `ranks_depth`/
//    `ranks_feat`/`ranks_bev` must be INT32 or INT64 and exactly rank 1.
//  - The BEV grid shape must be derivable per `DetermineBevGridShape` above.
//  - Requires opset >= 16, for `ScatterND`'s `reduction` attribute (the
//    whole point of this decomposition -- see below). This is a hard
//    requirement, unlike this codebase's other opset-gated rewrites' merely
//    defensive checks: opset < 16 has no `reduction="add"` at all, so there
//    is no equivalent construction available below that opset.
//  - Unlike `rewrite_trt_batched_nms` (which unrolls a per-batch-item C++
//    loop and therefore requires a statically known batch size `N`), this
//    pass needs NO input dimension -- batch size, camera count, depth bins,
//    image height/width, or channel count -- statically known. See the
//    derivation below for why: the batch axis is carried through as an
//    explicit per-row index into a combined `ScatterND` `indices` tensor
//    rather than folded into one flat axis via a per-batch-item constant
//    offset, so nothing here depends on how many batch items there are.
//
// Derivation, per original `bev_pool_v2` node:
//
// 1. `depth_flat = Reshape(depth, [0, -1])` -> `(B, N*D*H*W)`. `0` is
//    ONNX `Reshape`'s "copy this axis from the input" sentinel (default
//    `allowzero=0`), so this needs no runtime `Shape` query for `B` and no
//    static `N`/`D`/`H`/`W` -- `-1` infers the flattened axis from the
//    input's actual (possibly all-dynamic) total size.
// 2. `feat_flat = Reshape(feat, Concat([0, -1, C]))` -> `(B, N*H*W, C)`,
//    where `C = Gather(Shape(feat), [4], axis=0)` is read at runtime (`feat`
//    is always exactly rank 5, so index `4` is always valid) -- `-1` can
//    only infer one axis, and axis 1 (`N*H*W`) already claims that role, so
//    `C` must be supplied explicitly rather than inferred.
// 3. `depth_gathered = Gather(depth_flat, ranks_depth, axis=1)` ->
//    `(B, num_valid)`; `feat_gathered = Gather(feat_flat, ranks_feat,
//    axis=1)` -> `(B, num_valid, C)`. ONNX `Gather`'s output shape is
//    `data.shape[:axis] + indices.shape + data.shape[axis+1:]` -- since
//    `ranks_depth`/`ranks_feat` are themselves rank 1 and, per the op's own
//    contract, identical for every batch item, gathering the SAME index
//    vector along axis 1 of a `(B, ...)` tensor automatically broadcasts
//    across the batch axis with no batch-specific bookkeeping at all. This
//    is the step that makes a per-batch C++ unrolled loop unnecessary.
// 4. `contrib = Mul(Unsqueeze(depth_gathered, axis=2), feat_gathered)` ->
//    `(B, num_valid, C)`, broadcasting the scalar-per-entry depth value
//    against its feature vector.
// 5. The `ScatterND` target: `zeros_flat = ConstantOfShape(Concat([B, G,
//    C]))` -> `(B, G, C)` all zeros, `G = bev_z*bev_h*bev_w` (the
//    compile-time int from `DetermineBevGridShape`; `B` and `C` are the
//    same runtime-read values from steps 1-2). `ConstantOfShape`'s default
//    fill value (when its optional `value` attribute is omitted, as here) is
//    `0` of type FLOAT32 -- exactly right, since `depth`/`feat` are
//    required FLOAT above.
// 6. Per-entry `ScatterND` index = `[batch_index, ranks_bev[i]]`: `Range(0,
//    B_scalar, 1)` -> `(B,)`, reshaped to `(B,1,1)` and `Expand`ed to
//    `(B, num_valid, 1)`; `ranks_bev` (cast to INT64 if not already)
//    reshaped to `(1, num_valid, 1)` and `Expand`ed the same way; `Concat`
//    on the last axis -> `(B, num_valid, 2)`. This explicit per-row batch
//    index -- rather than offsetting `ranks_depth`/`ranks_feat`/`ranks_bev`
//    by a per-batch-item constant into one giant flat axis, the other
//    approach this pass could have taken -- is what removes the need for a
//    statically known batch size.
// 7. `scattered = ScatterND(zeros_flat, indices, contrib,
//    reduction="add")` -> `(B, G, C)`. This one native op performs the
//    entire grouped-sum / scatter-add this op needs -- no manual segment-sum
//    bookkeeping, and no use for the (deliberately unread) optional
//    `interval_starts`/`interval_lengths` inputs.
// 8. Un-flatten `G` back into `(Z, H, W)` (channel-last) via `Reshape(
//    scattered, [0, Z, H, W, -1])` (rank 5) or `[0, H, W, -1]` (rank 4;
//    `-1` infers `C` either way, `0` again means "copy `B` from the input"),
//    then `Transpose` to move `C` to axis 1 -- `[0, 4, 1, 2, 3]` (rank 5) or
//    `[0, 3, 1, 2]` (rank 4) -- matching the op's documented
//    `(B, C, Z, H, W)` / `(B, C, H, W)` output layout. This replaces the
//    original node's own output; its declared sizes, if any, are propagated
//    onto the new output the same way this codebase's other rewrites do.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// See this file's "NAMING UNCERTAINTY" header comment: these are the only
// two places `bev_pool_v2`'s op_type/domain guess lives, specifically so
// they are trivial to widen once a real export is available to check
// against.
constexpr const char* kBevPoolOpTypeCandidates[] = {"bev_pool_v2"};
constexpr const char* kBevPoolDomainCandidates[] = {"", "mmdeploy"};

// Resolves the output BEV grid shape (Z, H, W) and the intended output rank
// (4 or 5) for one `bev_pool_v2` node -- see this file's header comment for
// the three sources tried, in order. Shared by the predicate and the
// transform so the two never disagree about which source won.
inline bool DetermineBevGridShape(Node* node, int64_t& bev_z, int64_t& bev_h,
                                  int64_t& bev_w, int& out_rank) {
  Value* out = node->output();
  if (out->has_sizes()) {
    const auto& sizes = out->sizes();
    if (sizes.size() == 5) {
      const Dimension& zd = sizes[2];
      const Dimension& hd = sizes[3];
      const Dimension& wd = sizes[4];
      if (zd.is_int && zd.dim > 0 && hd.is_int && hd.dim > 0 && wd.is_int &&
          wd.dim > 0) {
        bev_z = zd.dim;
        bev_h = hd.dim;
        bev_w = wd.dim;
        out_rank = 5;
        return true;
      }
    } else if (sizes.size() == 4) {
      const Dimension& hd = sizes[2];
      const Dimension& wd = sizes[3];
      if (hd.is_int && hd.dim > 0 && wd.is_int && wd.dim > 0) {
        bev_z = 1;
        bev_h = hd.dim;
        bev_w = wd.dim;
        out_rank = 4;
        return true;
      }
    }
  }

  const int64_t attr_h =
      GetValueFromAttrWithDefault<int64_t>(node, Symbol("bev_h"), int64_t(-1));
  const int64_t attr_w =
      GetValueFromAttrWithDefault<int64_t>(node, Symbol("bev_w"), int64_t(-1));
  if (attr_h > 0 && attr_w > 0) {
    const int64_t attr_z =
        GetValueFromAttrWithDefault<int64_t>(node, Symbol("bev_z"), int64_t(1));
    if (attr_z <= 0) {
      return false;
    }
    bev_z = attr_z;
    bev_h = attr_h;
    bev_w = attr_w;
    if (out->has_sizes() &&
        (out->sizes().size() == 4 || out->sizes().size() == 5)) {
      out_rank = static_cast<int>(out->sizes().size());
    } else {
      out_rank = (attr_z == 1) ? 4 : 5;
    }
    return true;
  }

  for (const char* attr_name : {"bev_feat_shape", "output_shape"}) {
    std::vector<int64_t> shape_attr;
    if (!GetValueFromAttr(node, Symbol(attr_name), shape_attr)) {
      continue;
    }
    if (shape_attr.size() == 2 && shape_attr[0] > 0 && shape_attr[1] > 0) {
      bev_z = 1;
      bev_h = shape_attr[0];
      bev_w = shape_attr[1];
      out_rank = (out->has_sizes() && out->sizes().size() == 5) ? 5 : 4;
      return true;
    }
    if (shape_attr.size() == 3 && shape_attr[0] > 0 && shape_attr[1] > 0 &&
        shape_attr[2] > 0) {
      bev_z = shape_attr[0];
      bev_h = shape_attr[1];
      bev_w = shape_attr[2];
      out_rank = (out->has_sizes() && out->sizes().size() == 4) ? 4 : 5;
      return true;
    }
  }
  return false;
}

// Small node-construction helper bound to one `bev_pool_v2` rewrite: every
// node it creates is inserted immediately before `anchor` (the
// `bev_pool_v2` node itself); scalar int64 constants are cached so the
// handful of repeated literals (0, 1, ...) share one initializer apiece.
struct BevPoolToScatterBuilder {
  Graph& graph;
  Node* anchor;

  std::unordered_map<int64_t, Value*> i64_scalar_cache;

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

  // Rank-1 int64 tensor holding exactly `v` -- the Reshape/Expand-target
  // and Concat-piece literals this pass builds (e.g. `[0, -1]`, `[G]`,
  // `[0, Z, H, W, -1]`) are each distinct enough that caching them isn't
  // worthwhile, unlike the scalar cache above.
  Value* ConstI64Vec(const std::vector<int64_t>& v) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(static_cast<int64_t>(v.size()));
    for (int64_t x : v) {
      t.int64s().push_back(x);
    }
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  Value* UnOp(Symbol op, Value* a, int32_t elem_type) {
    Node* n = graph.create(op, 1);
    n->addInput(a);
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
  }

  Value* Mul(Value* a, Value* b) {
    Node* n = graph.create(kMul, 1);
    n->addInput(a);
    n->addInput(b);
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

  Value* CastToI64IfNeeded(Value* a) {
    if (a->elemType() == TensorProto_DataType_INT64) {
      return a;
    }
    return CastTo(a, TensorProto_DataType_INT64);
  }

  Value* Shape(Value* a) {
    return UnOp(Symbol("Shape"), a, TensorProto_DataType_INT64);
  }

  // General Gather: `indices` may be rank-0 (drops `axis`) or rank-N
  // (splices `indices`' shape in place of `axis`), covering every use in
  // this pass uniformly -- slicing a single dim off a `Shape()` result, or
  // gathering whole `(..., axis, ...)` slices by a shared rank-1 index
  // vector.
  Value* Gather(Value* data, Value* indices, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* GatherScalar(Value* data, int64_t idx, int64_t axis) {
    return Gather(data, ConstI64Scalar(idx), axis);
  }

  Value* Reshape(Value* data, Value* shape) {
    Node* n = graph.create(kReshape, 1);
    n->addInput(data);
    n->addInput(shape);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* Expand(Value* data, Value* shape) {
    Node* n = graph.create(kExpand, 1);
    n->addInput(data);
    n->addInput(shape);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* Unsqueeze(Value* a, int64_t axis) {
    Node* n = graph.create(kUnsqueeze, 1);
    n->addInput(a);
    n->addInput(ConstI64Vec({axis}));
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

  Value* Transpose(Value* a, std::vector<int64_t> perm) {
    Node* n = graph.create(kTranspose, 1);
    n->addInput(a);
    n->is_(kperm, std::move(perm));
    n->insertBefore(anchor);
    n->output()->setElemType(a->elemType());
    return n->output();
  }

  // `start`/`limit`/`delta` must each be rank-0 (scalar) int64 tensors, per
  // ONNX `Range`'s own schema.
  Value* Range(Value* start, Value* limit, Value* delta) {
    Node* n = graph.create(Symbol("Range"), 1);
    n->addInput(start);
    n->addInput(limit);
    n->addInput(delta);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  // All-zeros tensor of runtime shape `shape_1d` (a rank-1 int64 tensor).
  // Relies on `ConstantOfShape`'s documented default `value` (0, typed
  // FLOAT32) since no explicit `value` attribute is set -- correct here
  // because this pass requires `depth`/`feat` (and therefore `elem_type`)
  // to be FLOAT.
  Value* ZerosLike(Value* shape_1d, int32_t elem_type) {
    Node* n = graph.create(Symbol("ConstantOfShape"), 1);
    n->addInput(shape_1d);
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
  }

  Value* ScatterNDAdd(Value* data, Value* indices, Value* updates) {
    Node* n = graph.create(Symbol("ScatterND"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->addInput(updates);
    n->s_(Symbol("reduction"), "add");
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }
};

struct RewriteBevPoolToScatter final : public PredicateBasedPass {
  explicit RewriteBevPoolToScatter()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_bev_pool_to_scatter";
  }

  bool patternMatchPredicate(Node* node) override {
    bool op_type_match = false;
    for (const char* op_type : kBevPoolOpTypeCandidates) {
      if (node->kind() == Symbol(op_type)) {
        op_type_match = true;
        break;
      }
    }
    if (!op_type_match) {
      return false;
    }
    if (node->has_domain() && !node->domain().empty()) {
      bool domain_match = false;
      for (const char* domain : kBevPoolDomainCandidates) {
        if (node->domain() == domain) {
          domain_match = true;
          break;
        }
      }
      if (!domain_match) {
        return false;
      }
    }
    const size_t num_inputs = node->inputs().size();
    if (num_inputs < 5 || num_inputs > 7 || node->outputs().size() != 1) {
      return false;
    }

    Value* depth = node->input(0);
    Value* feat = node->input(1);
    if (depth->elemType() != TensorProto_DataType_FLOAT ||
        feat->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (!depth->has_sizes() || depth->sizes().size() != 5) {
      return false;
    }
    if (!feat->has_sizes() || feat->sizes().size() != 5) {
      return false;
    }

    for (int i = 2; i <= 4; ++i) {
      Value* ranks = node->input(i);
      const int32_t et = ranks->elemType();
      if (et != TensorProto_DataType_INT32 &&
          et != TensorProto_DataType_INT64) {
        return false;
      }
      if (!ranks->has_sizes() || ranks->sizes().size() != 1) {
        return false;
      }
    }

    int64_t bev_z = 0, bev_h = 0, bev_w = 0;
    int out_rank = 0;
    if (!DetermineBevGridShape(node, bev_z, bev_h, bev_w, out_rank)) {
      return false;
    }

    // Hard requirement, not a defensive check -- see this file's header
    // comment: opset < 16 has no ScatterND `reduction="add"` at all.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset >= 16;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* depth = node->input(0);
    Value* feat = node->input(1);
    Value* ranks_depth = node->input(2);
    Value* ranks_feat = node->input(3);
    Value* ranks_bev = node->input(4);
    // Inputs 5/6 (interval_starts/interval_lengths), when present, are
    // intentionally never read -- see this file's header comment.

    int64_t Z = 0, H = 0, W = 0;
    int out_rank = 0;
    const bool determined = DetermineBevGridShape(node, Z, H, W, out_rank);
    ONNX_ASSERT(determined);  // patternMatchPredicate already verified this.
    const int64_t G = Z * H * W;

    BevPoolToScatterBuilder b{graph, node};

    Value* rd = b.CastToI64IfNeeded(ranks_depth);
    Value* rf = b.CastToI64IfNeeded(ranks_feat);
    Value* rb = b.CastToI64IfNeeded(ranks_bev);

    // 1-2. Flatten depth's (N,D,H,W) and feat's (N,H,W) axes; keep the
    // batch axis ("0" = copy from input) and, for feat, read C explicitly
    // at runtime (the sole "-1" already infers N*H*W).
    Value* depth_flat = b.Reshape(depth, b.ConstI64Vec({0, -1}));

    Value* feat_shape = b.Shape(feat);
    Value* c_vec = b.Gather(feat_shape, b.ConstI64Vec({4}), 0);  // (1,)=[C]
    Value* feat_reshape_shape =
        b.Concat(0, {b.ConstI64Vec({0}), b.ConstI64Vec({-1}), c_vec});
    Value* feat_flat = b.Reshape(feat, feat_reshape_shape);

    // 3-4. Shared (batch-independent) index vectors broadcast across the
    // batch axis automatically via ONNX Gather's own semantics.
    Value* depth_gathered = b.Gather(depth_flat, rd, 1);  // (B, num_valid)
    Value* feat_gathered = b.Gather(feat_flat, rf, 1);    // (B, num_valid, C)
    Value* depth_gathered_u = b.Unsqueeze(depth_gathered, 2);
    Value* contrib = b.Mul(depth_gathered_u, feat_gathered);

    // 5. All-zeros ScatterND target (B, G, C).
    Value* b_vec = b.Gather(b.Shape(depth), b.ConstI64Vec({0}), 0);  // [B]
    Value* out_flat_shape = b.Concat(0, {b_vec, b.ConstI64Vec({G}), c_vec});
    Value* zeros_flat = b.ZerosLike(out_flat_shape, depth->elemType());

    // 6. Per-entry ScatterND index = [batch_index, ranks_bev[i]].
    Value* num_valid_vec = b.Gather(b.Shape(rb), b.ConstI64Vec({0}), 0);
    Value* bcast_shape =
        b.Concat(0, {b_vec, num_valid_vec, b.ConstI64Vec({1})});

    Value* b_scalar = b.GatherScalar(b.Shape(depth), 0, 0);
    Value* batch_range =
        b.Range(b.ConstI64Scalar(0), b_scalar, b.ConstI64Scalar(1));
    Value* batch_3d = b.Reshape(batch_range, b.ConstI64Vec({-1, 1, 1}));
    Value* batch_idx = b.Expand(batch_3d, bcast_shape);

    Value* target_3d = b.Reshape(rb, b.ConstI64Vec({1, -1, 1}));
    Value* target_idx = b.Expand(target_3d, bcast_shape);

    Value* scatter_indices = b.Concat(2, {batch_idx, target_idx});

    // 7. The scatter-add itself.
    Value* scattered = b.ScatterNDAdd(zeros_flat, scatter_indices, contrib);

    // 8. Un-flatten G back into (Z,H,W) and move C to axis 1.
    Value* final_out;
    if (out_rank == 5) {
      Value* mid = b.Reshape(scattered, b.ConstI64Vec({0, Z, H, W, -1}));
      final_out = b.Transpose(mid, {0, 4, 1, 2, 3});
    } else {
      Value* mid = b.Reshape(scattered, b.ConstI64Vec({0, H, W, -1}));
      final_out = b.Transpose(mid, {0, 3, 1, 2});
    }

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
