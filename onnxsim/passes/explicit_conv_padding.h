// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// A `Conv` with asymmetric explicit `pads` (the per-axis begin/end amounts
// differ) gets a `Pad` node hoisted in front of it carrying that padding,
// and the convolution's own `pads` become all-zero. Semantics are
// unchanged: zero-padding explicitly and then convolving with no padding is
// what an asymmetric `pads` attribute already means, just spelled as two
// ops instead of one -- a way past a backend whose fused convolution kernel
// only accepts symmetric padding (see `scripts/axera/legalize.py`'s Python
// counterpart of this rule for the concrete case that motivated it: a
// causal convolution with all its padding on one side).
//
// This is a different fix from `explicit_auto_pad.h`: that pass turns a
// *symbolic* `auto_pad` mode (`SAME_UPPER`/`SAME_LOWER`/`VALID`) into
// explicit `pads`, computed from the operator spec's formula. This pass
// starts from `pads` that are *already* explicit and only acts when they
// are asymmetric -- a `Conv` with `auto_pad == "NOTSET"` and symmetric
// explicit `pads` (the common case) is left alone by both passes, and a
// `Conv` with a symbolic `auto_pad` needs `explicit_auto_pad` to run first
// before this pass has an explicit `pads` attribute to inspect at all.
//
// Only `Conv`'s spatial `pads` are handled -- `AveragePool`/`MaxPool` are
// not, since their own asymmetric-padding requirement hasn't come up in a
// real target and pooling additionally needs the padding to affect the
// output's counted denominator (`count_include_pad`), which a hoisted `Pad`
// node does not replicate.
//
// This is a pure graph-shape rewrite -- not a node-count reduction -- so it
// is `PassType::Other` and never runs by default. Opt in with
// `extra_optimizers=["explicit_conv_padding"]` (Python) or
// `--enable-optimization explicit_conv_padding` (CLI).

#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct ExplicitConvPadding final : public PredicateBasedPass {
  explicit ExplicitConvPadding()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "explicit_conv_padding"; }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kConv || !node->hasAttribute(kpads)) {
      return false;
    }
    const std::vector<int64_t>& pads = node->is(kpads);
    if (pads.empty() || pads.size() % 2 != 0) {
      return false;
    }
    const size_t half = pads.size() / 2;
    for (size_t i = 0; i < half; ++i) {
      if (pads[i] != pads[half + i]) {
        return true;
      }
    }
    return false;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const std::vector<int64_t> pads = node->is(kpads);
    const size_t half = pads.size() / 2;

    Tensor pads_init;
    pads_init.elem_type() = TensorProto_DataType_INT64;
    pads_init.sizes().push_back(static_cast<int64_t>(2 * (half + 2)));
    std::vector<int64_t>& data = pads_init.int64s();
    data.push_back(0);
    data.push_back(0);
    for (size_t i = 0; i < half; ++i) {
      data.push_back(pads[i]);
    }
    data.push_back(0);
    data.push_back(0);
    for (size_t i = 0; i < half; ++i) {
      data.push_back(pads[half + i]);
    }
    Value* pads_v = graph.addInitializerAndCreateValue(pads_init);

    Node* pad = graph.create(kPad, 1);
    pad->addInput(node->input(0));
    pad->addInput(pads_v);
    pad->s_(kmode, std::string("constant"));
    pad->output()->setElemType(node->input(0)->elemType());
    pad->insertBefore(node);

    node->replaceInput(0, pad->output());
    node->is_(kpads, std::vector<int64_t>(pads.size(), 0));
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
