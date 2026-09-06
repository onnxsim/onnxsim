// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Decomposes mmdeploy's custom `TRTBatchedRotatedNMS` op -- the rotated-box
// sibling of `TRTBatchedNMS` (see `rewrite_trt_batched_nms.h`), used at the
// end of BEV/3-D detection heads (CenterPoint, PointPillars-style) whose
// boxes carry a yaw/rotation angle -- into a subgraph built purely from
// standard ONNX ops. Unlike the axis-aligned op, there is no ONNX primitive
// this rewrite can lean on for either half of the job:
//
//   1. ONNX `NonMaxSuppression` computes IoU internally assuming
//      axis-aligned boxes. Rotated-box IoU has no ONNX primitive at all, so
//      this pass computes it from scratch via Sutherland-Hodgman polygon
//      clipping (derived in detail below).
//   2. `NonMaxSuppression` also performs greedy NMS's sequential
//      keep-or-suppress decision process internally. With no primitive to
//      lean on for that either, this pass reimplements greedy suppression
//      itself as a `keepTopK`-unrolled (never data-dependent-`Loop`)
//      sequence of "pick the current best box, suppress everything too
//      close to it, repeat" steps (derived below).
//
// This is a pure graph-shape rewrite -- one node becomes a large fixed
// subgraph -- so it is `PassType::Other` and never runs by default. Opt in
// with `extra_optimizers=["rewrite_trt_batched_rotated_nms"]` (Python) or
// `--enable-optimization rewrite_trt_batched_rotated_nms` (CLI).
//
// ASSUMPTION FLAGGED PROMINENTLY -- box format: each box is the 5-tuple
// `(cx, cy, w, h, theta)` -- center-x, center-y, width, height, rotation
// angle in radians -- the standard OpenCV-style rotated-rect convention that
// mmcv's `box_iou_rotated`/`nms_rotated` (the CPU/CUDA ops this plugin
// mirrors) use. This pass does NOT independently confirm that mmdeploy's
// actual TensorRT-plugin export always uses this exact ordering/sign
// convention for `TRTBatchedRotatedNMS` specifically (as opposed to the
// mmcv ops it wraps) -- treat this as a documented assumption, not a
// verified fact, and re-check against your own export if results look
// rotated/mirrored/off.
//
// Op spec (mirrors `TRTBatchedNMS` exactly except box format -- see
// `rewrite_trt_batched_nms.h`'s own header comment for the non-geometry
// half of this contract):
//   Inputs (2, fixed order):
//     0. `boxes`:  FLOAT (N, num_boxes, num_classes_or_1, 5), `(cx,cy,w,h,
//        theta)` per box.
//     1. `scores`: FLOAT (N, num_boxes, num_classes).
//   Attributes (all static): `background_label_id` (int, default -1),
//     `num_classes` (int, informational only -- see Scope below),
//     `topK` (int), `keepTopK` (int), `scoreThreshold` (float),
//     `iouThreshold` (float). (`isNormalized`/`clipBoxes`, if present, are
//     not read -- axis-aligned "clip corners to [0,1]" has no meaningful
//     rotated-box analogue, so this pass does not attempt one.)
//   Outputs (4, fixed order): `num_detections` (N,1), `nmsed_boxes`
//     (N,keepTopK,5), `nmsed_scores` (N,keepTopK), `nmsed_classes`
//     (N,keepTopK) -- identical padding convention to `TRTBatchedNMS`
//     (zero-padded boxes/scores, -1.0-padded classes, past
//     `num_detections[n]`).
//
// KNOWN LIMITATION, same spirit as `rewrite_trt_batched_nms.h`'s own: this
// rewrite implements a standard, well-specified "per-class greedy rotated
// NMS, then global per-batch top-K merge" algorithm and aims only for the
// same SET of kept detections as that algorithm -- not bit-exact tie order
// against the closed TensorRT plugin. Near-identical-score or
// near-identical-IoU ties are exactly the cases where even this
// specification's own order is ambiguous; do not rely on exact tie-break
// order matching any particular reference.
//
// ===========================================================================
// Algorithm part 1: rotated IoU via Sutherland-Hodgman polygon clipping
// ===========================================================================
//
// Box corners: for `(cx,cy,w,h,theta)`, with `dx=w/2, dy=h/2`, the 4 local
// corners `(dx,dy),(-dx,dy),(-dx,-dy),(dx,-dy)` (this order traces the
// rectangle counter-clockwise in standard (x-right, y-up) axes -- verified
// via the shoelace formula on the un-rotated rectangle giving a positive
// sum, i.e. `+wh`, which is what CCW orientation implies; this matters
// below, where "inside" a clip edge is defined relative to CCW winding),
// each rotated by theta (`x'=x*cos(theta)-y*sin(theta)`, `y'=x*sin(theta)+
// y*cos(theta)`, a proper/orientation-preserving rotation) then translated
// by `(cx,cy)`.
//
// Computing the intersection of two convex quadrilaterals A (subject,
// varies per candidate box) and B (clip, the one already-picked box this
// NMS iteration is testing suppression against) via Sutherland-Hodgman:
// clip A's polygon successively against each of B's 4 edges (each edge
// extended to an infinite half-plane, "inside" = the side implied by B's
// own CCW winding, i.e. `cross(edge_dir, point - edge_start) >= 0`).
// Clipping a convex m-gon by one half-plane yields a convex polygon of at
// most `m+1` vertices (standard result: a half-plane crosses a convex
// polygon's boundary at exactly 0 or 2 points, so at most one contiguous
// "outside" run is removed and replaced by exactly 2 new boundary
// intersection points net +1 in the worst case of removing exactly 1
// vertex). Starting at 4 vertices and clipping against 4 edges in sequence,
// the true vertex count can grow at most 4 -> 5 -> 6 -> 7 -> 8, so an
// 8-slot buffer, carried through unchanged across all 4 (always exactly 4,
// known at pass-build time -- unrolled, not a data-dependent `Loop`) clip
// steps, is always sufficient.
//
// Making one clip step a FIXED-SHAPE, mask-free tensor computation (the
// genuinely novel part -- ordinary Sutherland-Hodgman appends a
// *variable*-length list per step, which has no direct fixed-tensor-shape
// equivalent) uses two tricks together:
//
//  (a) Padding-by-repeating-the-last-vertex. The buffer is always exactly 8
//      (x,y) slots; unused slots repeat the polygon's own last real vertex
//      rather than holding zeros. A run of repeated identical points forms
//      degenerate zero-length "edges" between them, which (i) never
//      register as a clip-edge transition (`inside(P)==inside(P)` trivially
//      for `P` paired with itself, so the per-edge test below never fires
//      for these), and (ii) contribute exactly 0 to the shoelace-formula
//      area sum in part 4 (`x_i*y_{i+1} - x_{i+1}*y_i` for `(i,i+1)=(P,P)`
//      is `x*y - x*y = 0`). So padding never needs separate masking --
//      anywhere downstream that treats the buffer as "a closed polygon,
//      possibly with repeated vertices" gets the right answer for free.
//      This is why the trick specifically requires repeating the *last
//      valid* vertex (not zeros, and not repeating some arbitrary other
//      vertex) -- repeating any point that is not already adjacent in the
//      real polygon's boundary would insert a spurious detour/spike.
//
//  (b) Compaction via a per-output-slot masked reduction with a
//      last-valid-vertex fallback, entirely vectorized over the 8 buffer
//      slots and the (up to 16) per-step emission candidates as ordinary
//      tensor axes -- no data-dependent indexing, and no C++-level
//      per-slot/per-candidate unrolling (only the num_boxes leading axis,
//      handled by ordinary broadcasting, and the 4-clip-steps loop are
//      unrolled at all). Per input buffer slot `k` (paired with its cyclic
//      successor `k+1`, defining edge `k` of the *subject* polygon A),
//      classic Sutherland-Hodgman emits, in order: `[]` if both endpoints
//      are outside the clip half-plane; `[E]` if both inside; `[exit
//      point]` if only the start `S` is inside; `[entry point, E]` if only
//      the end `E` is inside. Every one of these 4 cases is exactly 2
//      "candidate" slots -- `(candA, flagA)=(intersection, inside(S) XOR
//      inside(E))` and `(candB, flagB)=(E, inside(E) AND NOT degenerate(k))`
//      -- with `flagA`/`flagB` marking which candidates are real emissions
//      (verify, for a non-degenerate edge: both inside -> flagA=0,flagB=1
//      -> just `[E]`; only S inside -> flagA=1,flagB=0 -> just
//      `[intersection]`; only E inside -> flagA=1,flagB=1 -> `[intersection,
//      E]`, in that order; neither inside -> both 0 -> `[]`). Stacking
//      `(candA,flagA)` then `(candB,flagB)` per slot and flattening (a
//      reshape, not a real op cost) gives one ordered 16-candidate list per
//      box, preserving true traversal order. An inclusive prefix sum
//      (`CumSum`) of the flags over that axis gives each real candidate its
//      1-based output rank; for output slot `j` (0-indexed, target rank
//      `j+1`), the candidate (if any) with `cum==j+1 & flag` is that slot's
//      value (a `Where`+`ReduceSum` over the 16-candidate axis, since at
//      most one candidate can match); slots beyond the real count (rank >
//      total valid, itself just the final cumulative sum) fall back to
//      repeating the *last* real candidate (`cum==total_valid & flag`),
//      continuing the same last-vertex-repeat convention from (a) into the
//      *output* of each clip step, so the next step's input is already in
//      the same padding-safe shape.
//
//      The `degenerate(k)` guard on `flagB` (`Sx==Ex AND Sy==Ey` for edge
//      `k`) is not optional, and is the single most important correctness
//      detail in this whole scheme -- an earlier version of this pass
//      omitted it and produced silently wrong IoUs (verified: off by
//      several hundredths on rotated-partial-overlap test cases) whenever a
//      clip step's *output* padding run was longer than 1 slot. The subtle
//      failure mode: `flagA` self-excludes every degenerate edge for free
//      (`inside(S)==inside(E)` trivially when `S==E`, so the `XOR` is always
//      0), but `flagB=inside(E)` alone does not -- a *run* of `r` identical
//      padding-repeat slots forms `r` separate degenerate edges in this
//      buffer's cyclic pairing, and if the repeated point happens to be
//      inside the current clip half-plane (routine -- it is a genuine
//      vertex of the actual polygon), *every one* of those `r` edges
//      independently re-emits `flagB=1` for the *same* point. That inflates
//      the real-emission count by up to `r` extra (fully redundant)
//      "keep" votes -- not the intended 0 -- which both breaks the `+1`-
//      per-step growth bound this file's earlier bullet relies on (an
//      overflowing `total_valid` silently truncates real vertices out of
//      the fixed 8-slot output) and, independent of overflow, duplicates a
//      vertex into consecutive-but-not-adjacent output positions, corrupting
//      the ordering the next step's edge-pairing depends on. Excluding
//      every degenerate edge from `flagB` as well makes a padding run of
//      any length (1 to 4 repeated slots, depending on how much of the
//      buffer is unused) contribute exactly 0 emissions regardless of how
//      many repeated copies it spans -- restoring the invariant that only
//      the polygon's genuine `m` real edges (`m<=7` entering any of this
//      pass's 4 steps) can ever emit, so the classical `m -> m+1` bound
//      this file's earlier bullet derived applies to the *edge count*
//      exactly as intended, and the 8-slot output buffer this produces is
//      always sufficient -- no candidate is ever lost to truncation for any
//      of the 4 steps.
//
// Degenerate `rng==0`-style guard for the line-intersection denominator
// (`cross(edge_dir, S-B_i) - cross(edge_dir, E-B_i)`, zero exactly when `S`,
// `E` and the clip line are parallel): guarded via `Equal`/`Where` the same
// way `rewrite_gridsample_to_gather.h`'s reflection fold guards its own
// `rng==0` division -- substitute a safe nonzero divisor. The resulting
// intersection value is only ever *read* through the `flagA`-gated
// `Where`/`ReduceSum` compaction above, so a bogus value on an unused,
// flagged-invalid candidate never reaches the output regardless (ONNX
// `Where` selects, it does not multiply-and-sum, so a NaN/Inf on the
// not-taken branch cannot poison the selected result the way a masked
// *multiply* could).
//
// Area (shoelace formula, `0.5*|sum_i(x_i*y_{i+1}-x_{i+1}*y_i)|`, wrap-
// around `i+1`) is computed identically for box A's own 4 corners, box B's
// own 4 corners, and the (up to 8, padding-safe per above) clipped
// intersection polygon -- one shared helper, parameterized only by the
// buffer's slot count (4 or 8).
//
// IoU: `intersection_area / (area_A + area_B - intersection_area)`, with
// the denominator epsilon-guarded (`< 1e-9` treated as "effectively zero"
// -- e.g. two degenerate zero-area boxes, or floating-point noise around an
// exact-zero-overlap case) to a fallback IoU of 0, guarded the same
// `Equal`/`Where`-then-`Where` two-step way as `rewrite_gridsample_to_
// gather.h`'s `rng==0` case -- never a raw division that could produce
// NaN/Inf. (A degenerate zero-*area* box -- `w==0` or `h==0` -- can, as a
// side effect of the "inside" half-plane test being defined by *strict*
// `>=` against a zero-length clip edge, make the clip step degenerate to
// "everything passes" for that edge, which can make the *computed*
// intersection area come out larger than the true geometric intersection.
// This is harmless here: if box B is degenerate, `area_B==0`, and the
// denominator `area_A + 0 - intersection_area` also collapses toward 0 in
// exactly this case, so the epsilon guard's "fall back to IoU=0" fires
// regardless of the intermediate intersection-area quirk.)
//
// ===========================================================================
// Algorithm part 2: greedy NMS without `NonMaxSuppression` or `Loop`
// ===========================================================================
//
// Per (batch, class) group (boxes are shared across classes -- see Scope
// below -- so only the score vector differs per class; box corners/areas
// for all `num_boxes` candidates are therefore computed once per batch item
// and reused across every class and iteration), greedy suppression is
// `perClassK = min(topK, keepTopK)` **unrolled** iterations (never a
// data-dependent `Loop`): (a) `ArgMax` the current working score vector
// (initialized to each class's own scores, with sub-`scoreThreshold` and
// already-suppressed entries at `-1e9`) to pick the next box; (b) compute
// that picked box's rotated IoU against *every* candidate box at once (the
// `num_boxes` axis is an ordinary broadcastable tensor axis throughout this
// whole file -- never unrolled -- so `num_boxes` itself may be dynamic,
// unlike `keepTopK`/`topK`/`N`/`num_classes`, which this pass's C++ *does*
// unroll and therefore must be static, see Scope); (c) suppress: any box
// (including the picked one itself, via an explicit index-equality check
// rather than relying on self-IoU rounding to exactly 1.0) with IoU over
// `iouThreshold` gets its working score forced to `-1e9` for all later
// iterations. Capping the per-class unroll count at `min(topK, keepTopK)`
// rather than `topK` alone is a deliberate restructuring: since the
// eventual per-batch output keeps at most `keepTopK` detections total
// (across *all* classes), no class can ever usefully contribute more than
// `keepTopK` survivors to that merge regardless of how large `topK` is, so
// capping at `min(topK, keepTopK)` is exactly as correct as capping at
// `topK` while keeping the unrolled graph bounded by `keepTopK` even when
// `topK` is configured very large (a common real setting, e.g. `topK=1000,
// keepTopK=200`).
//
// Because boxes are class-agnostic but scores are per-class, this pass
// **runs the `perClassK`-unrolled greedy loop once per class (excluding
// `background_label_id`), then merges** the `num_classes_active *
// perClassK` per-class candidates into the final `keepTopK` via one `TopK`
// call per batch item -- chosen over any alternative restructuring because
// it is the direct, unmodified translation of `TRTBatchedNMS`'s own
// documented "per-class greedy NMS, then global per-batch top-K merge"
// algorithm (see `rewrite_trt_batched_nms.h`), just with `NonMaxSuppression`
// itself replaced by this file's own unrolled greedy loop wherever that
// pass used it. `background_label_id` (if `>= 0`) is handled by simply not
// emitting that class's subtree at all (a compile-time skip in the C++
// per-class loop), rather than threading an exclusion mask through
// -- simpler here than `rewrite_trt_batched_nms.h`'s additive-mask trick
// precisely because this pass already unrolls per class in C++, so there is
// no shared per-batch NMS call whose input needs uniform masking. Because
// `num_classes_active * perClassK` is a compile-time constant, the
// pre-`TopK` padding this merge needs (when there are fewer than
// `keepTopK` total candidates) is likewise a static `Pad` with a
// compile-time-known amount -- unlike `rewrite_trt_batched_nms.h`'s
// analogous step, which must compute its pad amount at runtime because its
// per-batch candidate count depends on `NonMaxSuppression`'s dynamic
// output.
//
// Scope (the predicate declines outside this; deliberately narrower than
// `rewrite_trt_batched_nms.h` -- see the module comment above for why this
// pass has no primitive to lean on for either the IoU or the suppression
// decision, making a from-scratch, fully-unrolled construction the only
// option, and correctness on a narrower scope is preferred over breadth on
// a shakier one):
//  - `boxes`' 3rd dim (`num_classes_or_1`) must be statically known and
//    exactly 1 -- identical reasoning to `rewrite_trt_batched_nms.h`: boxes
//    shared/class-agnostic across classes is both the overwhelmingly common
//    case and the only one a single shared per-box corner/area computation
//    (reused across classes) can serve.
//  - `N` (batch size) must be statically known -- this pass unrolls a
//    per-batch-item C++ loop, exactly like `rewrite_trt_batched_nms.h`.
//  - `num_classes` (scores' 3rd dim) must ALSO be statically known --
//    stricter than `rewrite_trt_batched_nms.h`, which only needed this
//    when `background_label_id >= 0` (there, one shared `NonMaxSuppression`
//    call handled every class internally). Here, since this pass runs its
//    own unrolled per-class greedy loop instead, the number of classes to
//    unroll must always be known at pass-build time. The `num_classes`
//    *attribute* is never read for this (nor for anything else) -- the
//    tensor's own static shape is the only source of truth used.
//  - `num_boxes` may be dynamic (see Algorithm part 2 above: every op that
//    touches the box-index axis is an ordinary broadcasting tensor op, not
//    a C++-unrolled one).
//  - `topK`, `keepTopK` must both be present and positive.
//  - Total unrolled greedy-NMS iteration count, `N * num_classes_active *
//    min(topK, keepTopK)` (`num_classes_active` = `num_classes` minus 1 if
//    `background_label_id` is a valid in-range class, else `num_classes`),
//    must not exceed `kMaxTotalGreedyIterations` (2000, defined below).
//    Each such iteration emits on the order of several hundred ONNX nodes
//    (four Sutherland-Hodgman clip steps, each itself several dozen ops --
//    see Algorithm part 1), so an unbounded iteration count risks
//    generating graphs of an utterly impractical size (millions of nodes)
//    for pathological/very-large configurations. This cap is a pragmatic,
//    adjustable safety valve, not a fundamental algorithmic limit --
//    lower it if pass-build/compile time is still a problem at 2000, or
//    raise it if profiling shows more headroom; it exists purely to fail
//    fast and predictably (decline the rewrite, leave the node alone)
//    rather than let a single simplify() call silently attempt to build an
//    unreasonably large graph. Real BEV-head configurations (small
//    `num_classes`, `keepTopK` in the low hundreds, `N` of a few) are
//    comfortably within this bound; extremely large `keepTopK`/`num_
//    classes` combinations are not, and should be addressed by reducing
//    those settings at export time rather than by this pass.
//  - Requires opset >= 13: this rewrite's own Squeeze/Unsqueeze (axes-as-
//    input) and Slice (starts/ends/axes/steps-as-input, opset >= 10) need
//    opset >= 13; `GreaterOrEqual`/`LessOrEqual` (opset >= 12) and `CumSum`
//    (opset >= 11) are both already covered once opset >= 13 is required.

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

// See this file's header comment ("Total unrolled greedy-NMS iteration
// count") for what this bounds and why.
constexpr int64_t kMaxTotalGreedyIterations = 2000;

// Small node-construction helper bound to one `TRTBatchedRotatedNMS`
// rewrite: every node it creates is inserted immediately before `anchor`
// (the `TRTBatchedRotatedNMS` node itself), and scalar float/int64
// constants are cached so the many repeated literals across the per-batch/
// per-class/per-iteration loops share one initializer apiece.
struct TRTBatchedRotatedNMSBuilder {
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

  // Rank-1 float tensor, shape (n,). Used for the `[1..8]` output-slot rank
  // constant and per-class padding-class-id vectors.
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

  // Rank-1, single-element int64 tensor -- the opset-13+ axes-as-input form
  // of Squeeze/Unsqueeze, and Slice's starts/ends/axes/steps inputs (each
  // used one element/axis at a time in this pass).
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

  // Rank-1 int64 tensor of arbitrary length -- Reshape's shape input and
  // Pad's pads input (both need more than one element at once here).
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
  Value* Max(Value* a, Value* b) {
    return BinOp(Symbol("Max"), a, b, a->elemType());
  }
  Value* Abs(Value* a) { return UnOp(Symbol("Abs"), a, a->elemType()); }
  Value* Cos(Value* a) { return UnOp(Symbol("Cos"), a, a->elemType()); }
  Value* Sin(Value* a) { return UnOp(Symbol("Sin"), a, a->elemType()); }

  Value* Equal(Value* a, Value* b) {
    return BinOp(Symbol("Equal"), a, b, TensorProto_DataType_BOOL);
  }
  Value* Greater(Value* a, Value* b) {
    return BinOp(kGreater, a, b, TensorProto_DataType_BOOL);
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
  Value* Or(Value* a, Value* b) {
    return BinOp(Symbol("Or"), a, b, TensorProto_DataType_BOOL);
  }
  Value* Not(Value* a) {
    return UnOp(Symbol("Not"), a, TensorProto_DataType_BOOL);
  }
  Value* Xor(Value* a, Value* b) {
    return BinOp(Symbol("Xor"), a, b, TensorProto_DataType_BOOL);
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

  // General Gather: `indices` may be rank-0 (drops `axis`) or rank-N
  // (splices `indices`' shape in place of `axis`).
  Value* Gather(Value* data, Value* indices, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // `data[start:end:step]` along `axis` (opset-10+ input-based Slice).
  Value* Slice1D(Value* data, int64_t start, int64_t end, int64_t axis,
                 int64_t step) {
    Node* n = graph.create(kSlice, 1);
    n->addInput(data);
    n->addInput(ConstI64Vec1(start));
    n->addInput(ConstI64Vec1(end));
    n->addInput(ConstI64Vec1(axis));
    n->addInput(ConstI64Vec1(step));
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
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

  Value* Reshape(Value* a, const std::vector<int64_t>& shape) {
    Node* n = graph.create(Symbol("Reshape"), 1);
    n->addInput(a);
    n->addInput(ConstI64Vec(shape));
    n->insertBefore(anchor);
    n->output()->setElemType(a->elemType());
    return n->output();
  }

  Value* ReduceSum(Value* data, int64_t axis, bool keepdims, int32_t out_type) {
    Node* n = graph.create(kReduceSum, 1);
    n->addInput(data);
    n->addInput(ConstI64Vec1(axis));
    n->i_(kkeepdims, keepdims ? 1 : 0);
    n->insertBefore(anchor);
    n->output()->setElemType(out_type);
    return n->output();
  }

  // Inclusive prefix sum along `axis` (opset-11+ CumSum, exclusive=0
  // default, reverse=0 default -- exactly what compaction needs: slot k's
  // cumulative count of real emissions at-and-before k).
  Value* CumSum(Value* data, int64_t axis) {
    Node* n = graph.create(Symbol("CumSum"), 1);
    n->addInput(data);
    n->addInput(ConstI64Scalar(axis));
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // 1-D `[start, limit)` range stepping by `delta` (opset-11+ Range).
  Value* Range(Value* start, Value* limit, Value* delta) {
    Node* n = graph.create(Symbol("Range"), 1);
    n->addInput(start);
    n->addInput(limit);
    n->addInput(delta);
    n->insertBefore(anchor);
    n->output()->setElemType(start->elemType());
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

  Value* ArgMax(Value* data, int64_t axis) {
    Node* n = graph.create(kArgMax, 1);
    n->addInput(data);
    n->i_(kaxis, axis);
    n->i_(kkeepdims, 0);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  // Returns {values, indices}; `data` must be rank-1 here (this pass only
  // ever calls TopK on a single batch item's already-flat, pooled score
  // vector).
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

  // -------------------------------------------------------------------
  // Rotated-box geometry (see this file's header comment, Algorithm 1).
  // -------------------------------------------------------------------

  // `(cx,cy,w,h,theta)`, each of shape S (either scalar `()` for a single
  // box or `(num_boxes,)` for all candidates), -> 4 corners' `(X,Y)`, each
  // of shape `S + (4,)`. `concat_axis` is `rank(S)` (0 or 1 in this file):
  // the corners are unsqueezed onto and concatenated along a brand-new
  // trailing axis.
  std::pair<Value*, Value*> Corners(Value* cx, Value* cy, Value* w, Value* h,
                                    Value* theta, int64_t concat_axis) {
    Value* half_w = Mul(w, ConstF(0.5f));
    Value* half_h = Mul(h, ConstF(0.5f));
    Value* cos_t = Cos(theta);
    Value* sin_t = Sin(theta);
    static const float kSx[4] = {1.0f, -1.0f, -1.0f, 1.0f};
    static const float kSy[4] = {1.0f, 1.0f, -1.0f, -1.0f};
    std::vector<Value*> xs, ys;
    xs.reserve(4);
    ys.reserve(4);
    for (int k = 0; k < 4; ++k) {
      Value* lx = Mul(half_w, ConstF(kSx[k]));
      Value* ly = Mul(half_h, ConstF(kSy[k]));
      Value* rx = Sub(Mul(lx, cos_t), Mul(ly, sin_t));
      Value* ry = Add(Mul(lx, sin_t), Mul(ly, cos_t));
      xs.push_back(Unsqueeze(Add(rx, cx), concat_axis));
      ys.push_back(Unsqueeze(Add(ry, cy), concat_axis));
    }
    return {Concat(concat_axis, xs), Concat(concat_axis, ys)};
  }

  // Shoelace-formula area of a (possibly padding-repeated -- see header
  // comment) closed polygon buffer `(X,Y)` of static slot count `size`
  // along `axis` (4 for a plain box, 8 for a clip-step buffer).
  Value* ShoelaceArea(Value* X, Value* Y, int64_t size, int64_t axis) {
    Value* Xn =
        Concat(axis, {Slice1D(X, 1, size, axis, 1), Slice1D(X, 0, 1, axis, 1)});
    Value* Yn =
        Concat(axis, {Slice1D(Y, 1, size, axis, 1), Slice1D(Y, 0, 1, axis, 1)});
    Value* cross = Sub(Mul(X, Yn), Mul(Xn, Y));
    Value* s = ReduceSum(cross, axis, false, X->elemType());
    return Mul(Abs(s), ConstF(0.5f));
  }

  // One Sutherland-Hodgman clip step: clips the `(num_boxes,8)` subject
  // buffer `(X,Y)` against the clip polygon's directed edge `(Bxi,Byi) ->
  // (Bxj,Byj)` (both scalars -- one edge of the single already-picked box
  // this NMS iteration tests against). Returns the new `(num_boxes,8)`
  // buffer -- see header comment for the candidate-generation-then-
  // compaction derivation.
  std::pair<Value*, Value*> ClipStep(Value* X, Value* Y, Value* Bxi, Value* Byi,
                                     Value* Bxj, Value* Byj) {
    constexpr int64_t kBuf = 8;
    constexpr int64_t kCand = 2 * kBuf;

    Value* dx_edge = Sub(Bxj, Bxi);
    Value* dy_edge = Sub(Byj, Byi);

    Value* Xn =
        Concat(1, {Slice1D(X, 1, kBuf, 1, 1), Slice1D(X, 0, 1, 1, 1)});  // "E"
    Value* Yn = Concat(1, {Slice1D(Y, 1, kBuf, 1, 1), Slice1D(Y, 0, 1, 1, 1)});

    // cross(edge_dir, point - B_i) >= 0 <=> point on B's interior side.
    Value* crossS = Sub(Mul(dx_edge, Sub(Y, Byi)), Mul(dy_edge, Sub(X, Bxi)));
    Value* crossE = Sub(Mul(dx_edge, Sub(Yn, Byi)), Mul(dy_edge, Sub(Xn, Bxi)));
    Value* insideS = GreaterOrEqual(crossS, ConstF(0.0f));
    Value* insideE = GreaterOrEqual(crossE, ConstF(0.0f));

    // A zero-length (S==E) edge -- always true of a padding-repeat slot,
    // see header comment part (a) -- must never independently re-vote to
    // "keep E": naively letting every one of the (possibly several)
    // repeated-vertex slots each re-emit the same already-counted point
    // inflates the real-emission count past this step's tight capacity
    // bound (each repeat is a SEPARATE edge in this buffer's cyclic
    // pairing, so without this guard a run of k padding repeats could
    // independently contribute up to k extra "keep" emissions of the same
    // point, not the intended 0). `insideS==insideE` trivially whenever
    // S==E, so `flagA` (the XOR) already self-excludes degenerate edges
    // with no extra guard needed; only `flagB` needs one.
    Value* is_degenerate_edge = And(Equal(X, Xn), Equal(Y, Yn));
    Value* flagA_b = Xor(insideS, insideE);  // real intersection candidate
    Value* flagB_b =
        And(insideE, Not(is_degenerate_edge));  // real "keep E" candidate

    // Line-intersection parameter t along S->E where cross(...)==0:
    // crossS + t*(crossE-crossS) = 0 => t = crossS/(crossS-crossE).
    Value* denom = Sub(crossS, crossE);
    Value* denom_safe = Where(Equal(denom, ConstF(0.0f)), ConstF(1.0f), denom);
    Value* t = Div(crossS, denom_safe);
    Value* interX = Add(X, Mul(t, Sub(Xn, X)));
    Value* interY = Add(Y, Mul(t, Sub(Yn, Y)));

    // Interleave [interX,Xn] (resp. Y, flag) per slot into one ordered
    // 16-candidate axis: stack as (num_boxes,8,2) then flatten the last two
    // dims -- row-major flattening of `(candA_k, candB_k)` pairs is exactly
    // the desired `[candA_0,candB_0,candA_1,candB_1,...]` order.
    auto interleave = [&](Value* a, Value* b) {
      Value* stacked = Concat(2, {Unsqueeze(a, 2), Unsqueeze(b, 2)});
      return Reshape(stacked, {-1, kCand});
    };
    Value* candX = interleave(interX, Xn);
    Value* candY = interleave(interY, Yn);
    Value* flag = interleave(CastTo(flagA_b, TensorProto_DataType_FLOAT),
                             CastTo(flagB_b, TensorProto_DataType_FLOAT));

    Value* cum = CumSum(flag, 1);                               // (nb,16)
    Value* total_valid = Slice1D(cum, kCand - 1, kCand, 1, 1);  // (nb,1)

    // rank_targets = [1..8], reused both as (1,8) (for is_real) and, via one
    // more Unsqueeze, (1,1,8) (for the per-candidate eligibility compare).
    std::vector<float> ranks(kBuf);
    for (int64_t j = 0; j < kBuf; ++j) {
      ranks[static_cast<size_t>(j)] = static_cast<float>(j + 1);
    }
    Value* rank_18 = Unsqueeze(ConstFVec(ranks), 0);  // (1,8)
    Value* rank_118 = Unsqueeze(rank_18, 1);          // (1,1,8)

    Value* cum_u = Unsqueeze(cum, 2);    // (nb,16,1)
    Value* flag_u = Unsqueeze(flag, 2);  // (nb,16,1)
    Value* eligible = And(Equal(cum_u, rank_118),
                          Greater(flag_u, ConstF(0.5f)));  // (nb,16,8)

    Value* zero = ConstF(0.0f);
    Value* outX_real = ReduceSum(Where(eligible, Unsqueeze(candX, 2), zero), 1,
                                 false, X->elemType());
    Value* outY_real = ReduceSum(Where(eligible, Unsqueeze(candY, 2), zero), 1,
                                 false, Y->elemType());

    Value* is_real = LessOrEqual(rank_18, total_valid);  // (nb,8)

    Value* eligible_last =
        And(Equal(cum, total_valid), Greater(flag, ConstF(0.5f)));  // (nb,16)
    Value* lastX =
        ReduceSum(Where(eligible_last, candX, zero), 1, true, X->elemType());
    Value* lastY =
        ReduceSum(Where(eligible_last, candY, zero), 1, true, Y->elemType());

    return {Where(is_real, outX_real, lastX), Where(is_real, outY_real, lastY)};
  }

  // Rotated IoU of the single box `(BX,BY)` (shape (4,) each) against every
  // one of `(AX,AY,areaA)`'s `num_boxes` candidates (shape (num_boxes,4)/
  // (num_boxes,)) at once.
  Value* RotatedIoU(Value* AX, Value* AY, Value* areaA, Value* BX, Value* BY,
                    Value* areaB) {
    // Pad A's 4 real corners to the 8-slot buffer by repeating corner 3 --
    // see header comment part (a).
    Value* lastX = Slice1D(AX, 3, 4, 1, 1);  // (num_boxes,1)
    Value* lastY = Slice1D(AY, 3, 4, 1, 1);
    Value* bufX = Concat(1, {AX, lastX, lastX, lastX, lastX});
    Value* bufY = Concat(1, {AY, lastY, lastY, lastY, lastY});
    for (int m = 0; m < 4; ++m) {
      Value* Bxi = Gather(BX, ConstI64Scalar(m), 0);
      Value* Byi = Gather(BY, ConstI64Scalar(m), 0);
      Value* Bxj = Gather(BX, ConstI64Scalar((m + 1) % 4), 0);
      Value* Byj = Gather(BY, ConstI64Scalar((m + 1) % 4), 0);
      auto next = ClipStep(bufX, bufY, Bxi, Byi, Bxj, Byj);
      bufX = next.first;
      bufY = next.second;
    }
    Value* interArea = ShoelaceArea(bufX, bufY, 8, 1);  // (num_boxes,)
    Value* denom = Sub(Add(areaA, areaB), interArea);
    Value* is_degenerate = Less(denom, ConstF(1e-9f));
    Value* denom_safe = Where(is_degenerate, ConstF(1.0f), denom);
    Value* iou = Div(interArea, denom_safe);
    return Where(is_degenerate, ConstF(0.0f), iou);
  }

  Value* Less(Value* a, Value* b) {
    return BinOp(Symbol("Less"), a, b, TensorProto_DataType_BOOL);
  }
};

struct RewriteTRTBatchedRotatedNMS final : public PredicateBasedPass {
  explicit RewriteTRTBatchedRotatedNMS()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_trt_batched_rotated_nms";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("TRTBatchedRotatedNMS")) {
      return false;
    }
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
    const Dimension& box_last = boxes->sizes()[3];
    if (box_last.is_int && box_last.dim != 5) {
      return false;
    }
    // Scope: class-agnostic boxes only -- see header comment.
    const Dimension& cls_or_1 = boxes->sizes()[2];
    if (!cls_or_1.is_int || cls_or_1.dim != 1) {
      return false;
    }
    // Scope: N (batch size) must be statically known.
    const Dimension& N_dim = boxes->sizes()[0];
    if (!N_dim.is_int || N_dim.dim <= 0) {
      return false;
    }
    // Scope: num_classes must ALSO be statically known (stricter than
    // rewrite_trt_batched_nms.h -- see header comment).
    const Dimension& cls_dim = scores->sizes()[2];
    if (!cls_dim.is_int || cls_dim.dim <= 0) {
      return false;
    }
    const int64_t topK =
        GetValueFromAttrWithDefault<int64_t>(node, Symbol("topK"), int64_t(-1));
    const int64_t keepTopK = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("keepTopK"), int64_t(-1));
    if (topK <= 0 || keepTopK <= 0) {
      return false;
    }
    const int64_t N = N_dim.dim;
    const int64_t num_classes = cls_dim.dim;
    const int64_t background_label_id = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("background_label_id"), int64_t(-1));
    const int64_t num_classes_active =
        (background_label_id >= 0 && background_label_id < num_classes)
            ? num_classes - 1
            : num_classes;
    // background_label_id excluding the only class (or an already-empty
    // num_classes) would leave nothing to pool from -- not a real scenario
    // this op is meant for, decline rather than build a Concat of zero
    // inputs.
    if (num_classes_active <= 0) {
      return false;
    }
    const int64_t per_class_k = std::min(topK, keepTopK);
    // Scope: total unrolled greedy-NMS iteration count -- see header
    // comment ("Total unrolled greedy-NMS iteration count").
    if (N * num_classes_active * per_class_k > kMaxTotalGreedyIterations) {
      return false;
    }
    // Floor set by this rewrite's own Squeeze/Unsqueeze (axes-as-input,
    // opset >= 13), which also covers GreaterOrEqual/LessOrEqual (opset
    // >= 12) and CumSum (opset >= 11) -- see header comment.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 13;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* boxes = node->input(0);
    Value* scores = node->input(1);

    const int64_t N = boxes->sizes()[0].dim;
    const int64_t num_classes = scores->sizes()[2].dim;
    const int64_t topK =
        GetValueFromAttrWithDefault<int64_t>(node, Symbol("topK"), int64_t(-1));
    const int64_t keepTopK = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("keepTopK"), int64_t(-1));
    const float scoreThreshold = GetValueFromAttrWithDefault<float>(
        node, Symbol("scoreThreshold"), 0.0f);
    const float iouThreshold =
        GetValueFromAttrWithDefault<float>(node, Symbol("iouThreshold"), 0.45f);
    const int64_t background_label_id = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("background_label_id"), int64_t(-1));
    const int64_t per_class_k = std::min(topK, keepTopK);

    TRTBatchedRotatedNMSBuilder b{graph, node};

    // Drop the size-1 class dim: (N,num_boxes,1,5) -> (N,num_boxes,5). Plain
    // Squeeze(axis=2) needs no static num_boxes at all, unlike a Reshape.
    Node* squeeze_node = graph.create(kSqueeze, 1);
    squeeze_node->addInput(boxes);
    squeeze_node->addInput(b.ConstI64Vec1(2));
    squeeze_node->insertBefore(node);
    squeeze_node->output()->setElemType(TensorProto_DataType_FLOAT);
    Value* boxes_sq = squeeze_node->output();

    const float kSentinelScore = -1e9f;
    const float kSentinelCheck = -5e8f;

    std::vector<Value*> boxes_chunks, scores_chunks, classes_chunks,
        numdet_chunks;
    boxes_chunks.reserve(static_cast<size_t>(N));
    scores_chunks.reserve(static_cast<size_t>(N));
    classes_chunks.reserve(static_cast<size_t>(N));
    numdet_chunks.reserve(static_cast<size_t>(N));

    int32_t num_det_dtype = node->outputs()[0]->elemType();
    if (num_det_dtype == TensorProto_DataType_UNDEFINED) {
      num_det_dtype = TensorProto_DataType_INT32;
    }

    for (int64_t n = 0; n < N; ++n) {
      Value* boxes_n = b.Gather(boxes_sq, b.ConstI64Scalar(n), 0);  // (nb,5)
      Value* scores_n = b.Gather(scores, b.ConstI64Scalar(n), 0);   // (nb,C)

      // Box corners/areas computed once per batch item, reused across every
      // class and every greedy iteration below.
      Value* cx = b.Gather(boxes_n, b.ConstI64Scalar(0), 1);
      Value* cy = b.Gather(boxes_n, b.ConstI64Scalar(1), 1);
      Value* w = b.Gather(boxes_n, b.ConstI64Scalar(2), 1);
      Value* h = b.Gather(boxes_n, b.ConstI64Scalar(3), 1);
      Value* theta = b.Gather(boxes_n, b.ConstI64Scalar(4), 1);
      auto AXY = b.Corners(cx, cy, w, h, theta, 1);
      Value* AX = AXY.first;
      Value* AY = AXY.second;
      Value* areaA = b.ShoelaceArea(AX, AY, 4, 1);  // (num_boxes,)

      Value* box_range =
          b.Range(b.ConstI64Scalar(0),
                  b.Gather(b.Shape(boxes_n), b.ConstI64Scalar(0), 0),
                  b.ConstI64Scalar(1));  // (num_boxes,) int64

      std::vector<Value*> pooled_scores, pooled_boxes, pooled_classes;

      for (int64_t c = 0; c < num_classes; ++c) {
        if (c == background_label_id) {
          continue;
        }
        Value* scores_nc = b.Gather(scores_n, b.ConstI64Scalar(c), 1);  // (nb,)
        Value* working = b.Where(b.Greater(scores_nc, b.ConstF(scoreThreshold)),
                                 scores_nc, b.ConstF(kSentinelScore));

        std::vector<Value*> kept_scores, kept_boxes;
        for (int64_t k = 0; k < per_class_k; ++k) {
          Value* idx_k = b.ArgMax(working, 0);           // scalar
          Value* score_k = b.Gather(working, idx_k, 0);  // scalar
          Value* box_k = b.Gather(boxes_n, idx_k, 0);    // (5,)

          Value* bcx = b.Gather(box_k, b.ConstI64Scalar(0), 0);
          Value* bcy = b.Gather(box_k, b.ConstI64Scalar(1), 0);
          Value* bw = b.Gather(box_k, b.ConstI64Scalar(2), 0);
          Value* bh = b.Gather(box_k, b.ConstI64Scalar(3), 0);
          Value* btheta = b.Gather(box_k, b.ConstI64Scalar(4), 0);
          auto BXY = b.Corners(bcx, bcy, bw, bh, btheta, 0);
          Value* areaB = b.ShoelaceArea(BXY.first, BXY.second, 4, 0);

          Value* iou =
              b.RotatedIoU(AX, AY, areaA, BXY.first, BXY.second, areaB);
          Value* suppress = b.Greater(iou, b.ConstF(iouThreshold));
          Value* self_mask = b.Equal(box_range, idx_k);
          Value* suppress_final = b.Or(suppress, self_mask);
          working = b.Where(suppress_final, b.ConstF(kSentinelScore), working);

          kept_scores.push_back(b.Unsqueeze(score_k, 0));  // (1,)
          kept_boxes.push_back(b.Unsqueeze(box_k, 0));     // (1,5)
        }

        Value* class_scores = b.Concat(0, kept_scores);  // (per_class_k,)
        Value* class_boxes = b.Concat(0, kept_boxes);    // (per_class_k,5)
        std::vector<float> class_id_vec(static_cast<size_t>(per_class_k),
                                        static_cast<float>(c));
        Value* class_classes = b.ConstFVec(class_id_vec);

        pooled_scores.push_back(class_scores);
        pooled_boxes.push_back(class_boxes);
        pooled_classes.push_back(class_classes);
      }

      Value* all_scores = b.Concat(0, pooled_scores);
      Value* all_boxes = b.Concat(0, pooled_boxes);
      Value* all_classes = b.Concat(0, pooled_classes);

      const int64_t total_pooled =
          static_cast<int64_t>(pooled_scores.size()) * per_class_k;
      if (total_pooled < keepTopK) {
        const int64_t pad_amount = keepTopK - total_pooled;
        all_scores = b.Pad(all_scores, b.ConstI64Vec({0, pad_amount}),
                           b.ConstF(kSentinelScore));
        all_boxes = b.Pad(all_boxes, b.ConstI64Vec({0, 0, pad_amount, 0}),
                          b.ConstF(0.0f));
        all_classes =
            b.Pad(all_classes, b.ConstI64Vec({0, pad_amount}), b.ConstF(-1.0f));
      }

      auto topk = b.TopK(all_scores, keepTopK, 0);
      Value* values = topk.first;
      Value* indices = topk.second;
      Value* boxes_final = b.Gather(all_boxes, indices, 0);      // (K,5)
      Value* classes_final = b.Gather(all_classes, indices, 0);  // (K,)

      Value* valid = b.Greater(values, b.ConstF(kSentinelCheck));
      Value* scores_final = b.Where(valid, values, b.ConstF(0.0f));
      Value* classes_out = b.Where(valid, classes_final, b.ConstF(-1.0f));
      Value* boxes_out =
          b.Where(b.Unsqueeze(valid, 1), boxes_final, b.ConstF(0.0f));
      Value* numdet =
          b.ReduceSum(b.CastTo(valid, num_det_dtype), 0, true, num_det_dtype);

      boxes_chunks.push_back(b.Unsqueeze(boxes_out, 0));      // (1,K,5)
      scores_chunks.push_back(b.Unsqueeze(scores_final, 0));  // (1,K)
      classes_chunks.push_back(b.Unsqueeze(classes_out, 0));  // (1,K)
      numdet_chunks.push_back(b.Unsqueeze(numdet, 0));        // (1,1)
    }

    Value* final_boxes = b.Concat(0, boxes_chunks);
    Value* final_scores = b.Concat(0, scores_chunks);
    Value* final_classes = b.Concat(0, classes_chunks);
    Value* final_numdet = b.Concat(0, numdet_chunks);

    final_numdet->setSizes({Dimension{N}, Dimension{1}});
    final_boxes->setSizes({Dimension{N}, Dimension{keepTopK}, Dimension{5}});
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
