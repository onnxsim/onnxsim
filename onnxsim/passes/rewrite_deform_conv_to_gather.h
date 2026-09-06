// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Decomposes mmdeploy/mmcv's custom deformable-convolution ops --
// `MMCVDeformConv2d` (DCNv1, unmodulated) and `MMCVModulatedDeformConv2d`
// (DCNv2, modulated) -- into a subgraph built purely from standard ONNX ops
// (`Shape`, `Range`, `Gather`, `GatherND`, `Slice`, `Cast`, `Reshape`,
// `Transpose`, `Concat`, `Unsqueeze`, `MatMul`, and elementwise arithmetic /
// comparison ops). Neither custom op has an ONNX Runtime kernel, so today
// onnxsim cannot even load a graph containing one through the normal
// simplify() pipeline's equivalence check (it has nothing to execute the
// original node with) -- this pass exists so such graphs can be simplified
// at all, by first turning the custom op into ops onnxsim (and every other
// ONNX consumer) already understands.
//
// This is a pure graph-shape rewrite -- not a node-count reduction (quite the
// opposite: one node becomes dozens to low hundreds, depending on kernel size
// and deform_groups) -- so it is `PassType::Other` and never runs by
// default. Opt in with `extra_optimizers=["rewrite_deform_conv_to_gather"]`
// (Python) or `--enable-optimization rewrite_deform_conv_to_gather` (CLI).
//
// Algorithm (modulated deformable convolution v2 -- Zhu et al.,
// "Deformable ConvNets v2", the same algorithm mmcv's CUDA kernel and
// torchvision's `deform_conv2d` both implement; DCNv1/`MMCVDeformConv2d` is
// just this with an implicit all-ones mask):
//
//   out[n, cout, ho, wo] =
//       bias[cout] (if present, else 0) +
//       sum over (cin_local in [0, Cin/groups), i in [0,kh), j in [0,kw)) of
//           weight[cout, cin_local, i, j] *
//           sampled[n, g*(Cin/groups) + cin_local, i, j, ho, wo]
//
// where `g = cout // (Cout/groups)` is cout's conv-group, and, writing
// `k = i*kw + j` for the kernel-tap index and `cin_global =
// g*(Cin/groups) + cin_local` for the actual input channel a tap reads:
//
//   dg = cin_global // (Cin/deform_groups)      -- deform-group owning this
//                                                   channel's offset/mask
//   dy = offset[n, dg*2*kh*kw + 2*k,     ho, wo]
//   dx = offset[n, dg*2*kh*kw + 2*k + 1, ho, wo]
//   y  = ho*sh - ph + i*dh + dy   -- absolute pixel-space coordinate,
//   x  = wo*sw - pw + j*dw + dx      NOT normalized to [-1, 1]
//   m  = mask[n, dg*kh*kw + k, ho, wo]   if modulated, else 1
//   sampled[...] = m * bilinear_sample_with_zero_padding(input[n,cin_global],
//   y, x)
//
// `bilinear_sample_with_zero_padding` is standard 4-corner bilinear
// interpolation where any out-of-range corner contributes 0 -- exactly
// GridSample's `padding_mode="zeros"` rule (see
// `rewrite_gridsample_to_gather.h`'s header comment, whose "GatherND-based
// bilinear sampling with a zeros-padding validity mask" trick this pass
// reuses per kernel-tap), just without that pass's `[-1,1]`-normalize /
// denormalize step: deformable conv's `offset` is already in absolute pixel
// units, so the sample coordinate is built directly from `ho`/`wo` and the
// conv's own `stride`/`padding`/`dilation`, with no denormalize.
//
// Channel-layout convention (mmdeploy/mmcv's ONNX export, matching the
// PyTorch reference `deform_conv2d`/`modulated_deform_conv2d` C++/CUDA
// kernels):
//   - `offset` has shape `(N, deform_groups*2*kh*kw, Hout, Wout)`. For tap
//     `k = i*kw + j` and deform-group `dg`, channels
//     `dg*2*kh*kw + 2*k` / `+ 2*k + 1` hold `(dy, dx)` for that tap/group at
//     every output location -- deform-group-major, then tap-major, then the
//     `(dy,dx)` pair.
//   - `mask` (modulated only) has shape `(N, deform_groups*kh*kw, Hout,
//     Wout)`, one scalar per `(dg, k)` per output location.
//   - `weight` has shape `(Cout, Cin/groups, kh, kw)` -- the usual ONNX
//     `Conv` weight layout.
//   - `Hout`/`Wout` are read directly off `offset`'s *runtime* shape
//     (`Shape(offset)`) -- this pass never recomputes them from
//     stride/padding/dilation arithmetic, since the exporter already sized
//     `offset` correctly for whatever convolution arithmetic (including any
//     asymmetric/`SAME`-style padding) produced it.
//
// Scope (the predicate declines outside this):
//  - Node kind `MMCVDeformConv2d` or `MMCVModulatedDeformConv2d`, domain `""`
//    or `"mmdeploy"` (mmdeploy exports under either depending on export
//    settings) -- any other domain is left alone (e.g. a same-named op from
//    an unrelated vendor).
//  - `MMCVDeformConv2d` (DCNv1): 3 inputs `(input, offset, weight)`, or 4
//    with an appended `bias` -- no `mask` input; treated as if `mask` were
//    all-ones. `MMCVModulatedDeformConv2d` (DCNv2): 4 inputs `(input,
//    offset, mask, weight)`, or 5 with an appended `bias`.
//  - `input`, `offset`, `mask` (when present) and `weight` (and `bias`, when
//    present) must all be FLOAT.
//  - `input` rank 4 `(N,Cin,H,W)`; `Cin` (the declared channel dim) must be
//    statically known. `N`, `H`, `W` may be dynamic -- `H`/`W` are read off
//    `Shape(input)` at runtime, never assumed static.
//  - `weight` rank 4 `(Cout, Cin/groups, kh, kw)`, all four dims statically
//    known (they are baked into the weight tensor's declared shape).
//  - `offset` rank 4, `mask` (if present) rank 4; their `Hout`/`Wout` (and
//    `N`) may be dynamic -- read off `Shape(offset)` at runtime. Their
//    channel dim, when statically known, is cross-checked against
//    `deform_groups*2*kh*kw` / `deform_groups*kh*kw` and the node is declined
//    on a mismatch (a sign the model's `deform_groups` doesn't actually match
//    its `offset`/`mask` shape).
//  - `bias`, when present, rank 1; if its length is statically known it must
//    equal `Cout`.
//  - `stride`, `padding`, `dilation` (2 ints each), `groups`, `deform_groups`
//    (or the synonym attribute name `deformable_groups`, tried second) are
//    read directly as static graph attributes -- these are never dynamic on
//    this op.
//  - `Cin % deform_groups == 0` (so `Cin/deform_groups` -- the per-deform-
//    group channel count -- is a compile-time integer); declined otherwise.
//  - **`groups == 1` only.** Supporting `groups > 1` in full generality means
//    additionally slicing `weight` (and the assembled per-tap samples) into
//    `groups` independent chunks along the output/input channel axes and
//    running the matmul contraction once per conv-group (see this pass's
//    sibling `rewrite_gridsample_to_gather.h`'s own precedent for "real
//    design freedom, prioritize correctness" scope calls) -- doable, but a
//    meaningfully larger and more error-prone rewrite for a conv-group
//    configuration that is rare in the mmdetection DCNv2 models this pass
//    targets (they overwhelmingly use `groups=1` with `deform_groups` in
//    `{1,2,4}` for the deformable-offset partitioning instead). This pass
//    scopes down to `groups == 1` and *declines* (rather than silently
//    mis-computing) anything else -- see
//    `tests/test_deform_conv_to_gather.py`'s
//    `test_declines_groups_greater_than_one` for a regression test proving the
//    decline actually fires. When `groups == 1`, `Cin/groups == Cin`, so
//    `weight`'s own declared `Cin/groups` dim must equal `Cin`; a mismatch (a
//    malformed model) is also declined defensively.
//  - Requires opset (`""` domain) `>= 13`: this pass leans on `Range`
//    (dynamic scalar `start`/`limit` inputs, opset >= 11), `GatherND`'s
//    `batch_dims` attribute (opset >= 12), and `Unsqueeze`'s axes-as-input
//    form (opset >= 13) -- the highest of the three sets the floor. A custom
//    op like `MMCVDeformConv2d` carries no opset requirement of its own the
//    way `GridSample` does, so unlike `rewrite_gridsample_to_gather.h`'s
//    defensive-only version of this check, this one can actually fire.
//
// Decomposition, per deform-group `dg` (`deform_groups` is a small static
// loop bound, unrolled at pass-build time, same as `kh*kw`):
//
// 1. Slice `input`'s channel axis to this deform-group's `Cin/deform_groups`
//    channels (`Slice(input, dg*Cin_dg, (dg+1)*Cin_dg, axis=1)`) and
//    transpose to channels-last (`(N,H,W,Cin_dg)`) once -- reused by every
//    tap of this `dg` below, exactly mirroring `rewrite_gridsample_to_gather
//    .h`'s single up-front `Transpose(X, [0,2,3,1])`.
// 2. Base pixel coordinates, built once (shared by every `dg`/tap, since they
//    depend only on `Hout`/`Wout`/`stride`/`padding`, not on `dg`/`i`/`j`):
//    `Range(0, Hout, 1)` and `Range(0, Wout, 1)` (int64, cast to float) give
//    the per-axis output-index iota; `row*sh - ph` reshaped to `(Hout,1)` and
//    `col*sw - pw` reshaped to `(1,Wout)` broadcast-add against `offset`'s
//    own `(N,Hout,Wout)` `dy`/`dx` slices to place every output location's
//    unshifted sample coordinate in one shot (numpy/ONNX broadcasting: a
//    `(Hout,1)` and a `(1,Wout)` operand both broadcast cleanly against a
//    trailing `(Hout,Wout)`).
// 3. Per tap `(i,j)` (`k = i*kw+j`): `y = base_y + i*dh + dy`, `x = base_x +
//    j*dw + dx` (`dy`/`dx` gathered from `offset`'s `dg`/`k` channels via a
//    scalar-index `Gather` on axis 1, which drops that axis directly to
//    `(N,Hout,Wout)` -- no separate `Squeeze` needed, same trick
//    `rewrite_gridsample_to_gather.h` uses for `grid`'s `x`/`y` split).
// 4. 4-corner bilinear sample with zeros padding, identical in structure to
//    `rewrite_gridsample_to_gather.h`'s `mode="linear"`, `padding_mode=
//    "zeros"` case (see that file's header comment steps 3-5) but with `y`/
//    `x` used directly as pixel coordinates -- no denormalize, since they
//    already are pixel coordinates. Each corner's validity is `0 <= v <=
//    dim-1` on the *raw* coordinate (matching the ONNX reference's zeros
//    semantics), the index fed to `GatherND` is separately clamped into
//    range so the gather itself never reads out of bounds, and the (masked)
//    corner weights `wx*wy` sum to the tap's sampled `(N,Hout,Wout,Cin_dg)`
//    value. If modulated, that value is then multiplied by the tap's mask
//    scalar (gathered from `mask`'s `dg`/`k` channel the same way as
//    `offset`).
// 5. The `kh*kw` per-tap `(N,Hout,Wout,Cin_dg)` tensors are each
//    `Unsqueeze`d at a new axis 3 and `Concat`enated there, giving
//    `(N,Hout,Wout,kh*kw,Cin_dg)`; a `Transpose` swapping the last two axes
//    (`(N,Hout,Wout,Cin_dg,kh*kw)`) followed by a `Reshape` merging them
//    (`(N,Hout,Wout,Cin_dg*kh*kw)`, using ONNX `Reshape`'s `0`-means-
//    "copy-input-dim" convention for the three leading, possibly-dynamic
//    axes) puts this `dg`'s taps in `(cin_local, i, j)` order -- matching how
//    `weight`'s own `(Cin/groups, kh, kw)` axes flatten. Because each `dg`
//    owns a *contiguous* slice of the global `Cin` range (`dg*Cin_dg
//    .. (dg+1)*Cin_dg`), concatenating these per-`dg` blocks along the same
//    trailing axis, in `dg` order, reproduces exactly the flattened
//    `(Cin*kh*kw)` order `weight`'s `(Cin, kh, kw)` -> `Cin*kh*kw` reshape
//    would use -- no extra reordering step needed.
// 6. `weight` (`(Cout, Cin, kh, kw)`, recall `groups == 1` so `Cin/groups ==
//    Cin`) is reshaped to `(Cout, Cin*kh*kw)` and transposed to
//    `(Cin*kh*kw, Cout)`. `MatMul` against step 5's `(N,Hout,Wout,Cin*kh*kw)`
//    result -- relying on `MatMul`'s numpy-style broadcasting (a 4-D times a
//    2-D operand treats the trailing two axes of each as the matrix to
//    multiply and broadcasts everything before that, so no explicit
//    flatten-then-unflatten of the `(Hout,Wout)` axes into a single axis is
//    needed) -- gives the `(N,Hout,Wout,Cout)` convolution output directly.
// 7. `bias`, if present, broadcast-adds along the trailing `Cout` axis (no
//    reshape needed -- it already lines up with `MatMul`'s output). A final
//    `Transpose([0,3,1,2])` gives the `(N,Cout,Hout,Wout)` result that
//    replaces the original node's output.

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

// Small node-construction helper bound to one deform-conv rewrite: every
// node it creates is inserted immediately before `anchor` (the
// `MMCV(Modulated)DeformConv2d` node itself). Mirrors
// `GridSampleToGatherBuilder` in `rewrite_gridsample_to_gather.h` (same
// constant-caching approach), extended with the handful of ops (`Range`,
// `Slice`, `MatMul`, `Reshape`, an arbitrary-length int64 vector constant)
// that pass didn't need.
struct DeformConvToGatherBuilder {
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

  // Rank-1, single-element int64 tensor -- what Unsqueeze's axes input
  // (opset >= 13 form) and Slice's starts/ends/axes inputs (each sliced axis
  // handled one call at a time here) want. Cached by value: the same
  // constant `1` used as e.g. both a Slice `axes` entry and an unrelated
  // Slice `start` is one shared initializer -- harmless, ONNX values are
  // pure data, not tagged by the semantic role a caller puts them in.
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

  // Arbitrary-length int64 vector constant (e.g. a Reshape shape input).
  // Not cached -- these vary enough per call site that a cache would rarely
  // hit, unlike the scalar/1-element caches above.
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
  Value* Floor(Value* a) { return UnOp(Symbol("Floor"), a, a->elemType()); }

  Value* GreaterOrEqual(Value* a, Value* b) {
    return BinOp(Symbol("GreaterOrEqual"), a, b, TensorProto_DataType_BOOL);
  }
  Value* LessOrEqual(Value* a, Value* b) {
    return BinOp(Symbol("LessOrEqual"), a, b, TensorProto_DataType_BOOL);
  }
  Value* And(Value* a, Value* b) {
    return BinOp(Symbol("And"), a, b, TensorProto_DataType_BOOL);
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

  // Single-axis, single-range slice: `data[..., start:end, ...]` along
  // `axis`. `start`/`end` follow ordinary Slice semantics (end exclusive,
  // both may be negative -- not used that way here, callers only ever pass
  // non-negative compile-time-known bounds).
  Value* Slice(Value* data, int64_t start, int64_t end, int64_t axis) {
    Node* n = graph.create(kSlice, 1);
    n->addInput(data);
    n->addInput(ConstI64Vec1(start));
    n->addInput(ConstI64Vec1(end));
    n->addInput(ConstI64Vec1(axis));
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

  // Fixed-shape reshape: `shape` entries follow ONNX Reshape's own
  // conventions (a literal dim, or `0` to copy the corresponding input dim
  // through unchanged -- used here for the leading, possibly-dynamic
  // `N`/`Hout`/`Wout` axes).
  Value* Reshape(Value* data, std::vector<int64_t> shape) {
    Node* n = graph.create(kReshape, 1);
    n->addInput(data);
    n->addInput(ConstI64Vec(shape));
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* MatMul(Value* a, Value* b) {
    return BinOp(kMatMul, a, b, a->elemType());
  }

  // 1-D range, `[start, limit)` stepping by `delta` -- `start`/`limit`/
  // `delta` must be scalar (rank-0) Values of the same type (int64 here,
  // the only type this pass needs it for).
  Value* Range(Value* start, Value* limit, Value* delta) {
    Node* n = graph.create(Symbol("Range"), 1);
    n->addInput(start);
    n->addInput(limit);
    n->addInput(delta);
    n->insertBefore(anchor);
    n->output()->setElemType(start->elemType());
    return n->output();
  }

  // Turns one raw (possibly out-of-range) float pixel coordinate into a
  // GatherND-safe int64 index by clamping into `[0, dim-1]` -- deformable
  // conv only ever needs "zeros" padding (there is no padding_mode choice on
  // this op), so unlike `rewrite_gridsample_to_gather.h`'s `BuildIndex` this
  // never needs the reflection/border branches.
  Value* ClampIndex(Value* v, Value* dim_minus1_f) {
    return CastTo(Clip(v, ConstF(0.0f), dim_minus1_f),
                  TensorProto_DataType_INT64);
  }

  // 0 <= v <= dim_minus1, as a bool tensor -- validity against the *raw*,
  // unclamped coordinate (zeros-padding semantics).
  Value* InRange(Value* v, Value* dim_minus1_f) {
    return And(GreaterOrEqual(v, ConstF(0.0f)), LessOrEqual(v, dim_minus1_f));
  }

  // `ix`/`iy`: (N,Hout,Wout) int64 pixel coordinates (already clamped into
  // range). `xt`: a channel slice of `input`, transposed to (N,H,W,C).
  // Returns the (N,Hout,Wout,C) gathered pixel values.
  Value* GatherPixel(Value* xt, Value* ix, Value* iy) {
    Value* iy_u = Unsqueeze(iy, 3);
    Value* ix_u = Unsqueeze(ix, 3);
    // (y, x) order -- matches xt's own H,W axis order.
    Value* idx = Concat(3, {iy_u, ix_u});
    return GatherND(xt, idx, 1);
  }

  // Broadcasts `per_pixel` ((N,Hout,Wout)) over `data`'s trailing channel
  // axis ((N,Hout,Wout,C)) and multiplies.
  Value* MulBroadcastLastAxis(Value* data, Value* per_pixel) {
    return Mul(data, Unsqueeze(per_pixel, 3));
  }
};

struct RewriteDeformConvToGather final : public PredicateBasedPass {
  explicit RewriteDeformConvToGather()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_deform_conv_to_gather";
  }

  // Shared by patternMatchPredicate and runTransform: pulls out everything
  // the rewrite needs to know, or reports why it can't fire. Keeping this
  // one place avoids the predicate and the transform silently drifting apart
  // on what "matches" means.
  struct MatchInfo {
    bool modulated = false;
    Value* X = nullptr;
    Value* offset = nullptr;
    Value* mask = nullptr;  // null unless modulated
    Value* weight = nullptr;
    Value* bias = nullptr;  // null if absent
    int64_t Cout = 0, Cin = 0, Cin_g = 0, kh = 0, kw = 0;
    int64_t sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1;
    int64_t groups = 1, deform_groups = 1;
  };

  static bool TryMatch(Node* node, MatchInfo& m) {
    const bool is_dcnv1 = node->kind() == Symbol("MMCVDeformConv2d");
    const bool is_dcnv2 = node->kind() == Symbol("MMCVModulatedDeformConv2d");
    if (!is_dcnv1 && !is_dcnv2) {
      return false;
    }
    const std::string domain = node->has_domain() ? node->domain() : "";
    if (domain != "" && domain != "mmdeploy") {
      return false;
    }
    if (node->outputs().size() != 1) {
      return false;
    }
    m.modulated = is_dcnv2;

    const size_t n_inputs = node->inputs().size();
    if (is_dcnv1) {
      // (input, offset, weight[, bias]).
      if (n_inputs != 3 && n_inputs != 4) {
        return false;
      }
      m.X = node->input(0);
      m.offset = node->input(1);
      m.weight = node->input(2);
      m.bias = n_inputs == 4 ? node->input(3) : nullptr;
    } else {
      // (input, offset, mask, weight[, bias]).
      if (n_inputs != 4 && n_inputs != 5) {
        return false;
      }
      m.X = node->input(0);
      m.offset = node->input(1);
      m.mask = node->input(2);
      m.weight = node->input(3);
      m.bias = n_inputs == 5 ? node->input(4) : nullptr;
    }

    for (Value* v : {m.X, m.offset, m.mask, m.weight, m.bias}) {
      if (v != nullptr && v->elemType() != TensorProto_DataType_FLOAT) {
        return false;
      }
    }

    if (!m.X->has_sizes() || m.X->sizes().size() != 4) {
      return false;
    }
    if (!m.offset->has_sizes() || m.offset->sizes().size() != 4) {
      return false;
    }
    if (m.mask != nullptr &&
        (!m.mask->has_sizes() || m.mask->sizes().size() != 4)) {
      return false;
    }
    if (!m.weight->has_sizes() || m.weight->sizes().size() != 4) {
      return false;
    }
    if (m.bias != nullptr &&
        (!m.bias->has_sizes() || m.bias->sizes().size() != 1)) {
      return false;
    }

    const auto& wsizes = m.weight->sizes();
    for (const auto& d : wsizes) {
      if (!d.is_int) {
        return false;
      }
    }
    m.Cout = wsizes[0].dim;
    m.Cin_g = wsizes[1].dim;
    m.kh = wsizes[2].dim;
    m.kw = wsizes[3].dim;

    const Dimension& cin_dim = m.X->sizes()[1];
    if (!cin_dim.is_int) {
      return false;
    }
    m.Cin = cin_dim.dim;

    if (m.bias != nullptr) {
      const Dimension& bd = m.bias->sizes()[0];
      if (bd.is_int && bd.dim != m.Cout) {
        return false;
      }
    }

    std::vector<int64_t> stride =
        GetValueFromAttrWithDefault<std::vector<int64_t>>(
            node, Symbol("stride"), {1, 1});
    std::vector<int64_t> padding =
        GetValueFromAttrWithDefault<std::vector<int64_t>>(
            node, Symbol("padding"), {0, 0});
    std::vector<int64_t> dilation =
        GetValueFromAttrWithDefault<std::vector<int64_t>>(
            node, Symbol("dilation"), {1, 1});
    if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
      return false;
    }
    m.sh = stride[0];
    m.sw = stride[1];
    m.ph = padding[0];
    m.pw = padding[1];
    m.dh = dilation[0];
    m.dw = dilation[1];

    m.groups = GetValueFromAttrWithDefault<int64_t>(node, Symbol("groups"), 1);
    m.deform_groups =
        GetValueFromAttrWithDefault<int64_t>(node, Symbol("deform_groups"), 0);
    if (m.deform_groups == 0) {
      m.deform_groups = GetValueFromAttrWithDefault<int64_t>(
          node, Symbol("deformable_groups"), 1);
    }

    // Scope limits -- see this file's header comment.
    if (m.groups != 1) {
      return false;
    }
    if (m.Cin_g != m.Cin) {
      // groups == 1 implies Cin/groups == Cin; a mismatch means the model's
      // declared weight shape and input channel count disagree.
      return false;
    }
    if (m.deform_groups <= 0 || m.Cin % m.deform_groups != 0) {
      return false;
    }

    // Cross-check offset/mask channel counts against deform_groups*kh*kw
    // when statically known.
    const Dimension& off_c = m.offset->sizes()[1];
    if (off_c.is_int && off_c.dim != m.deform_groups * 2 * m.kh * m.kw) {
      return false;
    }
    if (m.mask != nullptr) {
      const Dimension& mask_c = m.mask->sizes()[1];
      if (mask_c.is_int && mask_c.dim != m.deform_groups * m.kh * m.kw) {
        return false;
      }
    }

    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    MatchInfo m;
    if (!TryMatch(node, m)) {
      return false;
    }
    // Requires Range (opset >= 11), GatherND's batch_dims (opset >= 12) and
    // Unsqueeze's axes-as-input form (opset >= 13) -- see header comment.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 13;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    MatchInfo m;
    if (!TryMatch(node, m)) {
      return false;
    }

    const int64_t Cin_dg = m.Cin / m.deform_groups;
    const int64_t khkw = m.kh * m.kw;

    DeformConvToGatherBuilder b{graph, node};

    // H, W as runtime float scalars, read off Shape(input) -- never assumed
    // static.
    Value* shape_x = b.Shape(m.X);
    Value* H_f = b.CastTo(b.GatherScalar(shape_x, b.ConstI64Scalar(2), 0),
                          TensorProto_DataType_FLOAT);
    Value* W_f = b.CastTo(b.GatherScalar(shape_x, b.ConstI64Scalar(3), 0),
                          TensorProto_DataType_FLOAT);
    Value* Hm1_f = b.Sub(H_f, b.ConstF(1.0f));
    Value* Wm1_f = b.Sub(W_f, b.ConstF(1.0f));

    // Hout, Wout: runtime int64 scalars off Shape(offset) -- the exporter
    // already sized offset correctly, so this pass never recomputes them
    // from stride/padding/dilation.
    Value* shape_off = b.Shape(m.offset);
    Value* Hout_i64 = b.GatherScalar(shape_off, b.ConstI64Scalar(2), 0);
    Value* Wout_i64 = b.GatherScalar(shape_off, b.ConstI64Scalar(3), 0);

    // Per-axis output-index iota, and the base (pre-offset, pre-tap-shift)
    // sample coordinate each derives -- shared by every deform-group/tap.
    Value* row_i64 =
        b.Range(b.ConstI64Scalar(0), Hout_i64, b.ConstI64Scalar(1));
    Value* col_i64 =
        b.Range(b.ConstI64Scalar(0), Wout_i64, b.ConstI64Scalar(1));
    Value* row_f = b.CastTo(row_i64, TensorProto_DataType_FLOAT);
    Value* col_f = b.CastTo(col_i64, TensorProto_DataType_FLOAT);
    // (Hout,) -> (Hout,1) so it broadcasts against a trailing (Hout,Wout).
    Value* base_y =
        b.Unsqueeze(b.Sub(b.Mul(row_f, b.ConstF(static_cast<float>(m.sh))),
                          b.ConstF(static_cast<float>(m.ph))),
                    1);
    // (Wout,) -> (1,Wout).
    Value* base_x =
        b.Unsqueeze(b.Sub(b.Mul(col_f, b.ConstF(static_cast<float>(m.sw))),
                          b.ConstF(static_cast<float>(m.pw))),
                    0);

    Value* one = b.ConstF(1.0f);

    std::vector<Value*> dg_blocks;
    dg_blocks.reserve(static_cast<size_t>(m.deform_groups));

    for (int64_t dg = 0; dg < m.deform_groups; ++dg) {
      Value* x_slice = b.Slice(m.X, dg * Cin_dg, (dg + 1) * Cin_dg, 1);
      Value* xt_dg = b.Transpose(x_slice, {0, 2, 3, 1});  // (N,H,W,Cin_dg)

      std::vector<Value*> taps;
      taps.reserve(static_cast<size_t>(khkw));

      for (int64_t i = 0; i < m.kh; ++i) {
        for (int64_t j = 0; j < m.kw; ++j) {
          const int64_t k = i * m.kw + j;
          const int64_t off_dy_ch = dg * 2 * khkw + 2 * k;
          const int64_t off_dx_ch = off_dy_ch + 1;

          Value* dy = b.GatherScalar(m.offset, b.ConstI64Scalar(off_dy_ch),
                                     1);  // (N,Hout,Wout)
          Value* dx = b.GatherScalar(m.offset, b.ConstI64Scalar(off_dx_ch), 1);

          Value* y =
              b.Add(b.Add(base_y, b.ConstF(static_cast<float>(i * m.dh))), dy);
          Value* x =
              b.Add(b.Add(base_x, b.ConstF(static_cast<float>(j * m.dw))), dx);

          Value* x0f = b.Floor(x);
          Value* x1f = b.Add(x0f, one);
          Value* y0f = b.Floor(y);
          Value* y1f = b.Add(y0f, one);
          Value* wx1 = b.Sub(x, x0f);
          Value* wx0 = b.Sub(one, wx1);
          Value* wy1 = b.Sub(y, y0f);
          Value* wy0 = b.Sub(one, wy1);

          Value* ix0 = b.ClampIndex(x0f, Wm1_f);
          Value* ix1 = b.ClampIndex(x1f, Wm1_f);
          Value* iy0 = b.ClampIndex(y0f, Hm1_f);
          Value* iy1 = b.ClampIndex(y1f, Hm1_f);

          Value* valid_x0 = b.InRange(x0f, Wm1_f);
          Value* valid_x1 = b.InRange(x1f, Wm1_f);
          Value* valid_y0 = b.InRange(y0f, Hm1_f);
          Value* valid_y1 = b.InRange(y1f, Hm1_f);

          auto corner = [&](Value* ix, Value* iy, Value* wx, Value* wy,
                            Value* vx, Value* vy) -> Value* {
            Value* gathered = b.GatherPixel(xt_dg, ix, iy);
            Value* weight_v = b.Mul(wx, wy);
            Value* mask_f = b.CastTo(b.And(vx, vy), TensorProto_DataType_FLOAT);
            weight_v = b.Mul(weight_v, mask_f);
            return b.MulBroadcastLastAxis(gathered, weight_v);
          };

          Value* c00 = corner(ix0, iy0, wx0, wy0, valid_x0, valid_y0);
          Value* c10 = corner(ix1, iy0, wx1, wy0, valid_x1, valid_y0);
          Value* c01 = corner(ix0, iy1, wx0, wy1, valid_x0, valid_y1);
          Value* c11 = corner(ix1, iy1, wx1, wy1, valid_x1, valid_y1);
          Value* sampled =
              b.Add(b.Add(c00, c10), b.Add(c01, c11));  // (N,Hout,Wout,Cin_dg)

          if (m.modulated) {
            const int64_t mask_ch = dg * khkw + k;
            Value* mask_v =
                b.GatherScalar(m.mask, b.ConstI64Scalar(mask_ch), 1);
            sampled = b.MulBroadcastLastAxis(sampled, mask_v);
          }

          taps.push_back(b.Unsqueeze(sampled, 3));  // (N,Hout,Wout,1,Cin_dg)
        }
      }

      Value* stacked = b.Concat(3, taps);  // (N,Hout,Wout,kh*kw,Cin_dg)
      Value* transposed =
          b.Transpose(stacked, {0, 1, 2, 4, 3});  // (N,Hout,Wout,Cin_dg,kh*kw)
      Value* reshaped = b.Reshape(transposed, {0, 0, 0, Cin_dg * khkw});
      dg_blocks.push_back(reshaped);
    }

    Value* unfolded =
        dg_blocks.size() == 1 ? dg_blocks[0] : b.Concat(3, dg_blocks);
    // (N,Hout,Wout,Cin*kh*kw)

    Value* weight_flat =
        b.Reshape(m.weight, {m.Cout, m.Cin * khkw});     // (Cout, Cin*kh*kw)
    Value* weight_T = b.Transpose(weight_flat, {1, 0});  // (Cin*kh*kw, Cout)

    Value* result = b.MatMul(unfolded, weight_T);  // (N,Hout,Wout,Cout)
    if (m.bias != nullptr) {
      result = b.Add(result, m.bias);
    }

    Value* final_out = b.Transpose(result, {0, 3, 1, 2});  // (N,Cout,Hout,Wout)
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
