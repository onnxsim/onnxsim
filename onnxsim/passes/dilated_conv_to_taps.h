// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// A widely dilated 1-D convolution becomes one 1x1 convolution per tap,
// summed -- a port of scripts/axera/legalize.py's dilated_conv_to_taps rule,
// kept there for the compiler that motivated it (Pulsar2, which the vendor
// script's own docstring notes stores a dilated convolution internally as
// one weight block per tap already -- this rewrite moves the graph towards
// what the compiler does internally rather than away from it), but the
// rewrite itself carries no vendor-specific formula: `y[t] = sum_j w[:, :,
// j] . xp[t + j*d]` is the definition of a dilated convolution, so slicing
// the (explicitly) padded input at each tap offset and convolving with a
// kernel of one is exactly the same function, for any backend without
// dilated-conv support.
//
// Scoped identically to the vendor rule: only a plain (`group == 1`,
// `strides == [1]`), single-spatial-axis (1-D) convolution with a constant
// weight and a statically known output length qualifies -- see
// `patternMatchPredicate`. `min_dilation` is fixed at 2 (every call site in
// this project uses the vendor rule's own default; a Conv with dilation 1 is
// already what every backend supports, so there would be nothing to gain
// rewriting it).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct DilatedConvToTaps final : public PredicateBasedPass {
  explicit DilatedConvToTaps()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "dilated_conv_to_taps"; }

  static constexpr int64_t kMinDilation = 2;

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kConv) {
      return false;
    }
    const size_t num_inputs = node->inputs().size();
    if (num_inputs != 2 && num_inputs != 3) {
      return false;
    }
    if (node->hasAttribute(kgroup) && node->i(kgroup) != 1) {
      return false;
    }
    if (node->hasAttribute(kstrides)) {
      const auto& strides = node->is(kstrides);
      if (strides.size() != 1 || strides[0] != 1) {
        return false;
      }
    }
    if (!node->hasAttribute(kdilations)) {
      return false;
    }
    const auto& dil = node->is(kdilations);
    if (dil.size() != 1 || dil[0] < kMinDilation) {
      return false;
    }
    const Tensor* w = FetchConstantTensor(node->input(1));
    if (w == nullptr || w->elem_type() != TensorProto_DataType_FLOAT ||
        w->sizes().size() != 3) {
      return false;
    }
    Value* out = node->output();
    if (!out->has_sizes() || out->sizes().size() != 3) {
      return false;
    }
    const Dimension& length_dim = out->sizes()[2];
    return length_dim.is_int && length_dim.dim > 0;
  }

  static Value* ConstI64Vec1(Graph& graph, int64_t v) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(1);
    t.int64s().push_back(v);
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  // `w[:, :, j:j+1]` -- the j-th tap's [Cout, Cin, 1] slice out of a
  // [Cout, Cin, taps] weight, read flat and row-major (ReadFloatTensorFlat
  // already normalizes raw_data vs. a typed field into this layout).
  static Tensor TapWeight(const std::vector<float>& flat, int64_t cout,
                          int64_t cin, int64_t taps, int64_t j) {
    Tensor out;
    out.elem_type() = TensorProto_DataType_FLOAT;
    out.sizes() = {cout, cin, 1};
    std::vector<float> data(static_cast<size_t>(cout * cin));
    for (int64_t co = 0; co < cout; ++co) {
      for (int64_t ci = 0; ci < cin; ++ci) {
        data[static_cast<size_t>(co * cin + ci)] =
            flat[static_cast<size_t>((co * cin + ci) * taps + j)];
      }
    }
    out.set_raw_data(WriteRawDataLittleEndian(data));
    // No name set: addInitializerAndCreateValue(Tensor&&) auto-assigns a
    // fresh, guaranteed-unique one (getNextUniqueName()) when empty, rather
    // than risk a collision from two dilated Convs in the same graph
    // sharing a `stem`.
    return out;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const auto& dil = node->is(kdilations);
    const int64_t d = dil[0];
    const Tensor* w_t = FetchConstantTensor(node->input(1));
    if (w_t == nullptr) {
      return false;  // predicate already checked this; defensive only.
    }
    const int64_t cout = w_t->sizes()[0];
    const int64_t cin = w_t->sizes()[1];
    const int64_t taps = w_t->sizes()[2];
    const std::vector<float> w_flat = ReadFloatTensorFlat(*w_t);
    const int64_t length = node->output()->sizes()[2].dim;

    std::vector<int64_t> pads = {0, 0};
    if (node->hasAttribute(kpads)) {
      const auto& p = node->is(kpads);
      if (p.size() == 2) {
        pads = {p[0], p[1]};
      }
    }

    // Pad(x, [0, 0, pads[0], 0, 0, pads[1]]) -- ONNX Pad's 2*rank layout
    // (begin per axis, then end per axis) for this rank-3 [N, C, L] input,
    // padding only the trailing spatial axis.
    Tensor pads_t;
    pads_t.elem_type() = TensorProto_DataType_INT64;
    pads_t.sizes().push_back(6);
    pads_t.int64s() = {0, 0, pads[0], 0, 0, pads[1]};
    Node* pad = graph.create(kPad, 1);
    pad->addInput(node->input(0));
    pad->addInput(graph.addInitializerAndCreateValue(std::move(pads_t)));
    pad->s_(kmode, "constant");
    pad->insertBefore(node);
    pad->output()->setElemType(node->input(0)->elemType());

    std::vector<Value*> partials;
    partials.reserve(static_cast<size_t>(taps));
    for (int64_t j = 0; j < taps; ++j) {
      Tensor tap_w = TapWeight(w_flat, cout, cin, taps, j);
      Value* tap_w_v = graph.addInitializerAndCreateValue(std::move(tap_w));

      Node* slice = graph.create(kSlice, 1);
      slice->addInput(pad->output());
      slice->addInput(ConstI64Vec1(graph, j * d));
      slice->addInput(ConstI64Vec1(graph, j * d + length));
      slice->addInput(ConstI64Vec1(graph, 2));
      slice->insertBefore(node);
      slice->output()->setElemType(node->input(0)->elemType());

      Node* tap_conv = graph.create(kConv, 1);
      tap_conv->addInput(slice->output());
      tap_conv->addInput(tap_w_v);
      if (j == 0 && node->inputs().size() > 2) {
        tap_conv->addInput(node->input(2));
      }
      tap_conv->is_(kkernel_shape, std::vector<int64_t>{1});
      tap_conv->is_(kpads, std::vector<int64_t>{0, 0});
      tap_conv->is_(kdilations, std::vector<int64_t>{1});
      tap_conv->is_(kstrides, std::vector<int64_t>{1});
      tap_conv->insertBefore(node);
      tap_conv->output()->setElemType(node->output()->elemType());
      tap_conv->output()->setSizes(node->output()->sizes());

      partials.push_back(tap_conv->output());
    }

    Value* acc = partials[0];
    Node* last_add = nullptr;
    for (size_t j = 1; j < partials.size(); ++j) {
      Node* add = graph.create(kAdd, 1);
      add->addInput(acc);
      add->addInput(partials[j]);
      add->insertBefore(node);
      add->output()->setElemType(node->output()->elemType());
      add->output()->setSizes(node->output()->sizes());
      acc = add->output();
      last_add = add;
    }

    // taps == 1 (dilation with a single-tap kernel, e.g. a 1x1 conv that
    // still declared a dilation) never happens in practice for a rewrite
    // gated on kMinDilation, but stay correct: with no Add at all, the sole
    // tap convolution's own node is what replaces `node`.
    Node* replacement = last_add != nullptr ? last_add : partials[0]->node();
    if (!tryReplacingAllUsesWith(node, replacement)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
