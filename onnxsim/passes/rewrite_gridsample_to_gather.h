// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Decomposes a 2-D `GridSample` node into an equivalent subgraph built purely
// from `Shape`/`Gather`/`GatherND`/`Cast`/arithmetic ops (`Mul`, `Add`, `Sub`,
// `Div`, `Floor`, `Round`, `Abs`, `Clip`, comparisons, `Where`, `Transpose`,
// `Concat`, `Unsqueeze`) -- the standard trick TensorRT-oriented export
// pipelines (mmdeploy, BEVFormer/LoFTR-style ONNX exports, ...) use to run
// grid sampling on backends that lower straight to native TensorRT layers but
// have no `GridSample` plugin available. Every op this pass emits maps
// directly to a native TensorRT layer; `GridSample` itself does not.
//
// This is a pure graph-shape rewrite -- not a node-count reduction (quite the
// opposite: one node becomes several dozen) -- so it is `PassType::Other` and
// never runs by default. Opt in with
// `extra_optimizers=["rewrite_gridsample_to_gather"]` (Python) or
// `--enable-optimization rewrite_gridsample_to_gather` (CLI).
//
// Scope (the predicate declines outside this):
//  - Only 2-D `GridSample`: `X` rank 4 `(N,C,H,W)`, `grid` rank 4
//    `(N,Hout,Wout,2)`. `N`, `C`, `H`, `W`, `Hout`, `Wout` may all be dynamic
//    (symbolic) -- `H`/`W` are read off `Shape(X)` at runtime, never assumed
//    static.
//  - Only `X` of element type FLOAT. `GridSample`'s own schema allows any
//    tensor type for `X` (dequantizing to float internally and requantizing
//    at the end for integer types); this pass does not attempt that and
//    leaves such nodes alone.
//  - Only `mode` `"linear"` or `"nearest"`; `"cubic"` is left alone.
//    GridSample-20 renamed two of the three modes -- opset 16-19 spell them
//    `"bilinear"`/`"nearest"`/`"bicubic"` (default `"bilinear"`), opset 20+
//    `"linear"`/`"nearest"`/`"cubic"` (default `"linear"`) -- with no change
//    to the sampling algorithm, so both spellings of each mode are accepted
//    and normalized to the opset-20 name before anything else looks at it.
//    (Whichever spelling a node uses, the emitted subgraph is the same:
//    `GridSample` disappears, so there is no spelling to preserve.)
//  - All three `padding_mode` values (`"zeros"` default, `"border"`,
//    `"reflection"`) and both `align_corners` values (0 default, 1) are
//    handled.
//
// Derivation, verified against the ONNX reference implementation
// (`onnx/reference/ops/op_grid_sample.py`, see `_gs_denormalize`,
// `_gs_reflect`, `_pixel_at_array`, and `GridSample._run`'s own use of them):
//
// 1. `grid`'s last axis (size 2) holds `(x, y)` -- reversed from tensor axis
//    order `(H, W)`. `gx = Gather(grid, 0, axis=3)`, `gy = Gather(grid, 1,
//    axis=3)`; a scalar (rank-0) index drops the gathered axis, so this
//    reaches shape `(N,Hout,Wout)` directly with no separate Squeeze.
// 2. Denormalize to pixel-space floats (`_gs_denormalize`), using `W` for
//    `gx`->`x` and `H` for `gy`->`y`:
//      align_corners=1: coord = (g + 1) / 2 * (dim - 1)
//      align_corners=0: coord = ((g + 1) * dim - 1) / 2
//    `dim` (`H` or `W`) is read at runtime via `Shape(X)` + `Gather` + `Cast`
//    to float, never assumed static.
// 3. mode="nearest": a single corner per output location, `Round(coord)`
//    (ties-to-even, matching `np.rint` -- exactly what the reference's own
//    "PyTorch rounds to nearest even" comment calls for) cast to int64.
//    mode="linear": four corners `x0=Floor(x)`, `x1=x0+1`, `y0=Floor(y)`,
//    `y1=y0+1`, weighted by `wx0=1-(x-x0)`, `wx1=x-x0` (and the `y` pair the
//    same way), corner weight = `wx*wy`.
// 4. Per corner, turning a raw (possibly far out-of-range) float coordinate
//    into a valid gather index:
//      padding_mode="zeros": validity is computed from the *raw*,
//        unclamped coordinate (`0 <= v <= dim-1`) -- this matches the
//        reference exactly: `_pixel_at_array`'s "zeros" branch never adjusts
//        `i` at all, it only decides 0-vs-lookup from the untouched index.
//        The coordinate is then clamped (`Clip(v, 0, dim-1)`) purely so the
//        `GatherND` below never reads out of bounds; the corner's gathered
//        value is zeroed afterward by multiplying by the validity mask.
//      padding_mode="border": `Clip(v, 0, dim-1)`, no masking.
//      padding_mode="reflection": the closed-form triangular-wave fold
//        `reflect(v, lo, hi) = lo + (rng - |mod(v - lo, 2*rng) - rng|)`,
//        `rng = hi - lo`, `mod` a true (non-negative) floor-based modulus
//        (`w - T*Floor(w/T)`, T = 2*rng) -- *not* ONNX `Mod` with `fmod=1`,
//        which the spec requires for float operands and which follows the
//        *dividend's* sign (C `fmod` semantics), not the non-negative
//        remainder this fold needs.
//
//        The reflection bounds `(lo, hi)` are themselves `align_corners`-
//        dependent -- this is the one place it is easy to get subtly wrong
//        by assuming the `[0, dim-1]` bounds mentioned in the padding_mode
//        attribute's own doc comment apply universally. Reading
//        `_prepare_border` (which `_gs_reflect`'s callers use for their
//        `(x_min, x_max)`) shows otherwise: `align_corners=1` uses
//        `(0, dim-1)` (so `rng = dim-1`, matching the doc comment's example),
//        but `align_corners=0` uses `(-0.5, dim-0.5)` (`rng = dim`) -- the
//        half-pixel-wider "corner points, not center points" convention
//        `_gs_denormalize` itself uses for that same flag. Reflecting with
//        the wrong (always-`[0,dim-1]`) bounds silently mismatches PyTorch's
//        `align_corners=0` reflection padding.
//
//        This pass reflects each of the (up to four) integer corner
//        candidates independently, rather than the raw fractional coordinate
//        once up front the way `GridSample._run`'s outer loop structures it
//        (reflect the coordinate, *then* floor/round it for "nearest", or
//        floor it for "linear"'s `x0`/`x1` -- itself followed by a *second*,
//        per-corner reflect inside `_pixel_at_array` for whichever corner
//        still lands outside `[0, dim-1]` after the first). Hand-checked
//        against the reference algorithm across in-bounds, single-reflection
//        and multi-reflection (far out-of-range) cases: because the
//        reflection bounds sit exactly a half-pixel outside `[0, dim-1]`
//        (align_corners=0) or exactly on it (align_corners=1), flooring and
//        reflecting commute up to the same +1 corner adjustment either way,
//        and the two orderings always agree. Reflecting per corner directly
//        -- using the interpolation weight from the *unreflected* coordinate
//        throughout, exactly as the reference's own weight computation does
//        -- is both simpler to build (no extra "reflect-then-floor" step)
//        and, importantly, still exactly correct.
//
//        `rng == 0` (only reachable with align_corners=1 and dim==1 -- a
//        literal single-pixel row/column, dynamic `dim` so not statically
//        excludable) would divide by zero in the fold above; guarded via
//        `Equal`/`Where` to substitute a safe non-zero divisor and then force
//        the result to plain index 0 (the only valid index when dim==1)
//        rather than let the fold produce NaN/Inf.
//
//        A defensive `Clip` into `[0, dim-1]` follows the fold's `Round`
//        before the final `Cast` to int64, guarding against float roundoff
//        in the `Div`/`Floor`/`Mul`/`Sub` chain nudging an exact-integer
//        result a hair outside range (which would otherwise make the
//        `GatherND` below read out of bounds).
// 5. Pixels are gathered via `GatherND(Transpose(X, [0,2,3,1]), idx,
//    batch_dims=1)`: `X` is transposed once to `(N,H,W,C)` (channels-last, so
//    `GatherND`'s per-batch index lines up with axes `H,W`), and each
//    corner's index tensor is `Concat([Unsqueeze(iy,-1), Unsqueeze(ix,-1)],
//    axis=-1)` -- `(y,x)` order, matching data axis order `H,W` (the
//    opposite of `grid`'s own `(x,y)` order) -- giving `(N,Hout,Wout,2)` and
//    a `(N,Hout,Wout,C)` gather result. The "zeros" validity mask, when
//    present, multiplies this result (broadcast over `C` via an `Unsqueeze`
//    on the mask's trailing axis).
// 6. mode="nearest": that single masked/gathered `(N,Hout,Wout,C)` tensor is
//    the answer. mode="linear": each of the four corners' gathered value is
//    scaled by its (mask-adjusted, for "zeros") scalar weight (broadcast the
//    same way) and the four are summed.
//    Either way, `Transpose(..., [0,3,1,2])` back to `(N,C,Hout,Wout)`
//    replaces the original `GridSample` node's output.

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

// Small node-construction helper bound to one `GridSample` rewrite: every
// node it creates is inserted immediately before `anchor` (the `GridSample`
// node itself), and scalar float/int64 constants are cached so denormalize,
// the per-axis reflect setup and all (up to four) corners sharing the same
// literal (0.0, 1.0, 2.0, -0.5, ...) reuse one initializer instead of a fresh
// one each time.
struct GridSampleToGatherBuilder {
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

  // Rank-1, single-element int64 tensor -- the axes input Unsqueeze wants
  // (opset >= 13 form; always applicable here since GridSample itself
  // requires opset >= 16).
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
  Value* Mul(Value* a, Value* b) { return BinOp(kMul, a, b, a->elemType()); }
  Value* Div(Value* a, Value* b) { return BinOp(kDiv, a, b, a->elemType()); }
  Value* Floor(Value* a) { return UnOp(Symbol("Floor"), a, a->elemType()); }
  Value* Round(Value* a) { return UnOp(Symbol("Round"), a, a->elemType()); }
  Value* Abs(Value* a) { return UnOp(Symbol("Abs"), a, a->elemType()); }

  Value* Equal(Value* a, Value* b) {
    return BinOp(Symbol("Equal"), a, b, TensorProto_DataType_BOOL);
  }
  Value* GreaterOrEqual(Value* a, Value* b) {
    return BinOp(Symbol("GreaterOrEqual"), a, b, TensorProto_DataType_BOOL);
  }
  Value* LessOrEqual(Value* a, Value* b) {
    return BinOp(Symbol("LessOrEqual"), a, b, TensorProto_DataType_BOOL);
  }
  Value* And(Value* a, Value* b) {
    return BinOp(Symbol("And"), a, b, TensorProto_DataType_BOOL);
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

  // `indices` must be a scalar (rank-0) Value -- see ConstI64Scalar -- so the
  // gathered axis is dropped from the output rather than left as a size-1
  // dim.
  Value* GatherScalar(Value* data, Value* indices, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* GatherND(Value* data, Value* indices, int64_t batch_dims) {
    Node* n = graph.create(Symbol("GatherND"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(Symbol("batch_dims"), batch_dims);
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

  // (g + 1) / 2 * (dim - 1)                       [align_corners = 1]
  // ((g + 1) * dim - 1) / 2                        [align_corners = 0]
  Value* Denormalize(Value* g, bool align_corners, Value* dim_f,
                     Value* dim_minus1_f) {
    Value* g_plus_1 = Add(g, ConstF(1.0f));
    if (align_corners) {
      return Mul(Div(g_plus_1, ConstF(2.0f)), dim_minus1_f);
    }
    return Div(Sub(Mul(g_plus_1, dim_f), ConstF(1.0f)), ConstF(2.0f));
  }

  // Per-axis constants padding_mode="reflection" needs, shared by every
  // corner reflected against this axis (both of a linear corner pair, or the
  // single nearest-mode corner). `lo == nullptr` means align_corners=1's
  // `lo == 0` exactly -- callers skip the (otherwise no-op) add/subtract
  // rather than materialize a zero constant for it.
  struct ReflectConsts {
    Value* lo;
    Value* rng_safe;
    Value* T_safe;
    Value* is_degenerate;
  };

  ReflectConsts MakeReflectConsts(bool align_corners, Value* dim_f,
                                  Value* dim_minus1_f) {
    // align_corners=1: bounds (0, dim-1), rng = dim-1.
    // align_corners=0: bounds (-0.5, dim-0.5), rng = dim.
    // (See this file's header comment -- these bounds, not a uniform
    // [0, dim-1], are what the ONNX reference's _prepare_border actually
    // uses for reflection.)
    Value* rng = align_corners ? dim_minus1_f : dim_f;
    Value* is_degenerate = Equal(rng, ConstF(0.0f));
    Value* rng_safe = Where(is_degenerate, ConstF(1.0f), rng);
    Value* T_safe = Mul(ConstF(2.0f), rng_safe);
    Value* lo = align_corners ? nullptr : ConstF(-0.5f);
    return ReflectConsts{lo, rng_safe, T_safe, is_degenerate};
  }

  // reflect(v) = lo + (rng - |mod(v - lo, 2*rng) - rng|), `mod` a
  // non-negative floor-based modulus -- see this file's header comment for
  // why plain ONNX `Mod` (fmod=1, required for float operands) does not fit.
  // Returns an int64 index, defensively clipped into [0, dim-1] first (see
  // header comment) to guard the GatherND downstream against float roundoff.
  Value* ReflectToIndex(Value* v, const ReflectConsts& rc,
                        Value* dim_minus1_f) {
    Value* w = rc.lo ? Sub(v, rc.lo) : v;
    Value* q = Floor(Div(w, rc.T_safe));
    Value* w_mod = Sub(w, Mul(rc.T_safe, q));
    Value* folded = Sub(rc.rng_safe, Abs(Sub(w_mod, rc.rng_safe)));
    Value* reflected = rc.lo ? Add(folded, rc.lo) : folded;
    Value* reflected_safe = Where(rc.is_degenerate, ConstF(0.0f), reflected);
    Value* rounded = Round(reflected_safe);
    Value* clamped = Clip(rounded, ConstF(0.0f), dim_minus1_f);
    return CastTo(clamped, TensorProto_DataType_INT64);
  }

  // Turns one raw (possibly out-of-range) float pixel coordinate `v` into a
  // valid int64 gather index for the given padding_mode. For "zeros", the
  // caller separately derives the validity mask from `v` itself (via
  // InRange) before calling this -- what this returns is only the
  // GatherND-safe clamped index, matching this file's header comment.
  Value* BuildIndex(Value* v, Value* dim_minus1_f,
                    const std::string& padding_mode,
                    const ReflectConsts* reflect) {
    if (padding_mode == "reflection") {
      return ReflectToIndex(v, *reflect, dim_minus1_f);
    }
    return CastTo(Clip(v, ConstF(0.0f), dim_minus1_f),
                  TensorProto_DataType_INT64);
  }

  // 0 <= v <= dim_minus1, as a bool tensor -- padding_mode="zeros"' validity
  // check, always evaluated against the *raw*, unclamped coordinate.
  Value* InRange(Value* v, Value* dim_minus1_f) {
    return And(GreaterOrEqual(v, ConstF(0.0f)), LessOrEqual(v, dim_minus1_f));
  }

  // `ix`/`iy`: (N,Hout,Wout) int64 pixel coordinates (already clamped/
  // reflected into range). `xt`: X transposed to (N,H,W,C). Returns the
  // (N,Hout,Wout,C) gathered pixel values.
  Value* GatherPixel(Value* xt, Value* ix, Value* iy) {
    Value* iy_u = Unsqueeze(iy, 3);
    Value* ix_u = Unsqueeze(ix, 3);
    // (y, x) order -- matches xt's own H,W axis order, the opposite of
    // grid's own (x, y) last-axis order.
    Value* idx = Concat(3, {iy_u, ix_u});
    return GatherND(xt, idx, 1);
  }

  // Broadcasts `per_pixel` ((N,Hout,Wout)) over `data`'s trailing channel
  // axis ((N,Hout,Wout,C)) and multiplies.
  Value* MulBroadcastLastAxis(Value* data, Value* per_pixel) {
    return Mul(data, Unsqueeze(per_pixel, 3));
  }
};

// Reads `mode` off a `GridSample` node in the opset-20 spelling, whichever
// spelling the node itself uses. GridSample-20 renamed "bilinear" -> "linear"
// and "bicubic" -> "cubic" (and its default with them) without touching the
// sampling algorithm, so an opset-16..19 node means exactly what the
// same-named opset-20 one does. An absent attribute is "linear" under either
// spelling, so the default needs no opset check of its own.
inline std::string GetNormalizedGridSampleMode(Node* node) {
  const std::string mode =
      GetValueFromAttrWithDefault<std::string>(node, kmode, "linear");
  if (mode == "bilinear") {
    return "linear";
  }
  if (mode == "bicubic") {
    return "cubic";
  }
  return mode;
}

struct RewriteGridSampleToGather final : public PredicateBasedPass {
  explicit RewriteGridSampleToGather()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_gridsample_to_gather";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("GridSample")) {
      return false;
    }
    // Leave a same-named op in a non-ai.onnx domain (e.g. a vendor/plugin
    // "GridSample") alone.
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    if (node->inputs().size() != 2 || node->outputs().size() != 1) {
      return false;
    }
    Value* X = node->input(0);
    Value* grid = node->input(1);
    if (!X->has_sizes() || X->sizes().size() != 4) {
      return false;
    }
    if (!grid->has_sizes() || grid->sizes().size() != 4) {
      return false;
    }
    const Dimension& grid_last = grid->sizes()[3];
    if (grid_last.is_int && grid_last.dim != 2) {
      return false;
    }
    // GridSample's schema allows any tensor type for X (dequantizing to
    // float and requantizing internally for integer types); this rewrite's
    // arithmetic is float-only, so decline anything else.
    if (X->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const std::string mode = GetNormalizedGridSampleMode(node);
    if (mode != "linear" && mode != "nearest") {
      return false;
    }
    const std::string padding_mode = GetValueFromAttrWithDefault<std::string>(
        node, Symbol("padding_mode"), "zeros");
    if (padding_mode != "zeros" && padding_mode != "border" &&
        padding_mode != "reflection") {
      return false;
    }
    // GridSample itself requires opset >= 16; this is a defensive check, not
    // load-bearing (the node could not otherwise exist), matching the style
    // of this codebase's other opset-gated rewrites.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 16;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* X = node->input(0);
    Value* grid = node->input(1);
    const std::string mode = GetNormalizedGridSampleMode(node);
    const std::string padding_mode = GetValueFromAttrWithDefault<std::string>(
        node, Symbol("padding_mode"), "zeros");
    const bool align_corners =
        GetValueFromAttrWithDefault<int64_t>(node, Symbol("align_corners"),
                                             int64_t(0)) != 0;

    GridSampleToGatherBuilder b{graph, node};

    // H, W as runtime float scalars, read off Shape(X) -- never assumed
    // static.
    Value* shape_x = b.Shape(X);
    Value* H_f = b.CastTo(b.GatherScalar(shape_x, b.ConstI64Scalar(2), 0),
                          TensorProto_DataType_FLOAT);
    Value* W_f = b.CastTo(b.GatherScalar(shape_x, b.ConstI64Scalar(3), 0),
                          TensorProto_DataType_FLOAT);
    Value* Hm1_f = b.Sub(H_f, b.ConstF(1.0f));
    Value* Wm1_f = b.Sub(W_f, b.ConstF(1.0f));

    // grid's last axis is (x, y) -- reversed from X's own (H, W) axis order.
    Value* gx = b.GatherScalar(grid, b.ConstI64Scalar(0), 3);
    Value* gy = b.GatherScalar(grid, b.ConstI64Scalar(1), 3);

    Value* coordx = b.Denormalize(gx, align_corners, W_f, Wm1_f);
    Value* coordy = b.Denormalize(gy, align_corners, H_f, Hm1_f);

    GridSampleToGatherBuilder::ReflectConsts reflect_w{nullptr, nullptr,
                                                       nullptr, nullptr};
    GridSampleToGatherBuilder::ReflectConsts reflect_h{nullptr, nullptr,
                                                       nullptr, nullptr};
    if (padding_mode == "reflection") {
      reflect_w = b.MakeReflectConsts(align_corners, W_f, Wm1_f);
      reflect_h = b.MakeReflectConsts(align_corners, H_f, Hm1_f);
    }

    Value* xt = b.Transpose(X, {0, 2, 3, 1});  // (N,C,H,W) -> (N,H,W,C)

    Value* result;  // (N,Hout,Wout,C)
    if (mode == "nearest") {
      Value* xr = b.Round(coordx);
      Value* yr = b.Round(coordy);
      Value* ix = b.BuildIndex(xr, Wm1_f, padding_mode, &reflect_w);
      Value* iy = b.BuildIndex(yr, Hm1_f, padding_mode, &reflect_h);
      Value* gathered = b.GatherPixel(xt, ix, iy);
      if (padding_mode == "zeros") {
        Value* valid = b.And(b.InRange(xr, Wm1_f), b.InRange(yr, Hm1_f));
        Value* mask_f = b.CastTo(valid, TensorProto_DataType_FLOAT);
        gathered = b.MulBroadcastLastAxis(gathered, mask_f);
      }
      result = gathered;
    } else {
      Value* one = b.ConstF(1.0f);
      Value* x0f = b.Floor(coordx);
      Value* x1f = b.Add(x0f, one);
      Value* y0f = b.Floor(coordy);
      Value* y1f = b.Add(y0f, one);
      Value* wx1 = b.Sub(coordx, x0f);
      Value* wx0 = b.Sub(one, wx1);
      Value* wy1 = b.Sub(coordy, y0f);
      Value* wy0 = b.Sub(one, wy1);

      Value* ix0 = b.BuildIndex(x0f, Wm1_f, padding_mode, &reflect_w);
      Value* ix1 = b.BuildIndex(x1f, Wm1_f, padding_mode, &reflect_w);
      Value* iy0 = b.BuildIndex(y0f, Hm1_f, padding_mode, &reflect_h);
      Value* iy1 = b.BuildIndex(y1f, Hm1_f, padding_mode, &reflect_h);

      Value *valid_x0 = nullptr, *valid_x1 = nullptr, *valid_y0 = nullptr,
            *valid_y1 = nullptr;
      if (padding_mode == "zeros") {
        valid_x0 = b.InRange(x0f, Wm1_f);
        valid_x1 = b.InRange(x1f, Wm1_f);
        valid_y0 = b.InRange(y0f, Hm1_f);
        valid_y1 = b.InRange(y1f, Hm1_f);
      }

      auto corner = [&](Value* ix, Value* iy, Value* wx, Value* wy, Value* vx,
                        Value* vy) -> Value* {
        Value* gathered = b.GatherPixel(xt, ix, iy);
        Value* weight = b.Mul(wx, wy);
        if (padding_mode == "zeros") {
          Value* mask_f = b.CastTo(b.And(vx, vy), TensorProto_DataType_FLOAT);
          weight = b.Mul(weight, mask_f);
        }
        return b.MulBroadcastLastAxis(gathered, weight);
      };

      Value* c00 = corner(ix0, iy0, wx0, wy0, valid_x0, valid_y0);
      Value* c10 = corner(ix1, iy0, wx1, wy0, valid_x1, valid_y0);
      Value* c01 = corner(ix0, iy1, wx0, wy1, valid_x0, valid_y1);
      Value* c11 = corner(ix1, iy1, wx1, wy1, valid_x1, valid_y1);
      result = b.Add(b.Add(c00, c10), b.Add(c01, c11));
    }

    Value* final_out = b.Transpose(
        result, {0, 3, 1, 2});  // (N,Hout,Wout,C) -> (N,C,Hout,Wout)
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
