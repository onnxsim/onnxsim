// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes a `Gemm` node whose weight B is
// a constant 2-D float32 tensor into ONNX Runtime's "com.microsoft" contrib
// op QGemm -- the fully-general analogue of qoperator_quantize_matmul.h's
// QLinearMatMul rewrite, which only handles "vanilla" Gemm (transA=0,
// alpha=1) because QLinearMatMul itself has no transpose/scale attributes
// of its own. QGemm keeps `transA`/`transB`/`alpha` as attributes, so this
// pass handles the general case QLinearMatMul cannot -- any transA, transB,
// or alpha -- without needing to pre-transpose the weight or reject
// alpha != 1.
//
// Before:
//   Y = Gemm(A, B, C, transA=ta, transB=tb, alpha=al, beta=1.0)
//       -- A: runtime float32; B: constant 2-D float32; C: constant 1-D
//          float32 of length N (optional)
// After:
//   Aq = QuantizeLinear(A, As, Azp)                       -- As/Azp: CALIBRATED
//   Yq = QGemm(Aq, As, Azp, Bq, Bs, Bzp, Cq, Ys, Yzp,
//              transA=ta, transB=tb, alpha=al)            -- true int8 compute
//   Y  = DequantizeLinear(Yq, Ys, Yzp)                    -- Ys/Yzp: CALIBRATED
//
// B is quantized per output channel (INT8, symmetric) *in its own storage
// layout* -- QGemm keeps `transB` as its own attribute, so unlike
// QLinearMatMul there is no need to physically transpose B into a fixed
// [K, N] layout first (see QuantizeWeightPerChannelInPlace).
//
// C (when present) must be quantized ahead of time into INT32 with
// zero_point = 0 and a per-column scale of `alpha * a_scale * b_scale[n]`
// (QGemm's own documented convention for its optional bias input) --
// unlike qoperator_quantize_matmul.h's vanilla-Gemm handling, which adds
// the bias back in float *after* dequantizing (QLinearMatMul has no bias
// input at all), QGemm accumulates the bias directly in the quantized
// compute. Only a 1-D C of exactly N elements (the common, unambiguous
// per-column-bias case) is handled; any other C shape is left alone. QGemm
// has no `beta` attribute of its own (its documented bias-scale formula
// implicitly assumes beta=1), so a Gemm with a non-default beta (and a
// bias) is left untouched, same restriction quantize_matmul_common.h's
// MatchMatMulLike already applies for the same underlying reason.
//
// A rewritten node is only reachable through ONNX Runtime's "com.microsoft"
// contrib domain, so this pass also adds that domain (version 1) to the
// model's opset imports the first time it fires, if not already present.
//
// A node is only rewritten when A's name and the node's own output's name
// both have a calibrated range (set via StaticQuantizationCalibrationRanges(),
// see QuantizeQOperatorGemm in onnxsim.h).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/quantize_matmul_common.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Reads a 1-D float32 constant tensor into a flat host-byte-order buffer,
// regardless of whether it is stored as raw bytes or a typed float array
// (mirrors quantize_matmul_common.h's ReadFloatMatrix, but for rank 1).
inline std::vector<float> ReadFloatVector1D(const Tensor& t) {
  const int64_t numel = t.sizes()[0];
  if (t.is_raw_data()) {
    return ReadRawDataHostOrder<float>(t.data<float>(), numel);
  }
  return t.floats();
}

// The pieces of a Gemm node this pass cares about, plus whatever
// QuantizeWeightPerChannelInPlace/GemmQuantizableInfo below determine about
// its shape and quantizability.
struct GemmQuantizableInfo {
  Value* a = nullptr;
  const Tensor* b_t = nullptr;
  const Tensor* c_t = nullptr;  // nullptr if Gemm has no C input
  int64_t trans_a = 0;
  int64_t trans_b = 0;
  double alpha = 1.0;
  int64_t n = 0;  // B's output-channel dimension
};

inline bool MatchGemmQuantizable(Node* n, GemmQuantizableInfo& info) {
  if (n->kind() != kGemm) {
    return false;
  }
  const size_t num_inputs = n->inputs().size();
  if (num_inputs != 2 && num_inputs != 3) {
    return false;
  }
  info.trans_a = GetValueFromAttrWithDefault(n, ktransA, int64_t(0));
  info.trans_b = GetValueFromAttrWithDefault(n, ktransB, int64_t(0));
  info.alpha = GetValueFromAttrWithDefault(n, kalpha, 1.0);
  const double beta = GetValueFromAttrWithDefault(n, kbeta, 1.0);

  info.a = n->inputs()[0];
  if (info.a->elemType() != TensorProto_DataType_FLOAT) {
    return false;
  }
  const Tensor* b_t = FetchConstantTensor(n->inputs()[1]);
  if (b_t == nullptr || b_t->elem_type() != TensorProto_DataType_FLOAT ||
      b_t->sizes().size() != 2) {
    return false;
  }
  info.b_t = b_t;
  info.n = info.trans_b != 0 ? b_t->sizes()[0] : b_t->sizes()[1];

  info.c_t = nullptr;
  if (num_inputs == 3) {
    if (beta != 1.0) {
      return false;  // QGemm's bias-scale formula assumes beta = 1.
    }
    const Tensor* c_t = FetchConstantTensor(n->inputs()[2]);
    if (c_t == nullptr || c_t->elem_type() != TensorProto_DataType_FLOAT ||
        c_t->sizes().size() != 1 || c_t->sizes()[0] != info.n) {
      return false;  // Only the common 1-D-of-length-N bias is handled.
    }
    info.c_t = c_t;
  }
  return true;
}

struct QOperatorQuantizeGemm final : public PredicateBasedPass {
  explicit QOperatorQuantizeGemm()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "qoperator_quantize_gemm"; }

  bool patternMatchPredicate(Node* n) override {
    GemmQuantizableInfo info;
    if (!MatchGemmQuantizable(n, info)) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    return ranges.count(info.a->uniqueName()) != 0 &&
           ranges.count(n->output()->uniqueName()) != 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    GemmQuantizableInfo info;
    if (!MatchGemmQuantizable(n, info)) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto a_range_it = ranges.find(info.a->uniqueName());
    const auto y_range_it = ranges.find(n->output()->uniqueName());
    if (a_range_it == ranges.end() || y_range_it == ranges.end()) {
      return false;
    }

    float a_scale_f = 1.0f;
    int32_t a_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        a_range_it->second.first, a_range_it->second.second, a_scale_f, a_zp_i);
    float y_scale_f = 1.0f;
    int32_t y_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        y_range_it->second.first, y_range_it->second.second, y_scale_f, y_zp_i);

    // B is quantized per output channel, in its own storage layout --
    // QGemm keeps transB as its own attribute, so no forced transpose.
    const int64_t channel_axis = info.trans_b != 0 ? 0 : 1;
    Tensor b_q;
    Tensor b_scale;
    QuantizeWeightPerChannelInPlace(*info.b_t, channel_axis, b_q, b_scale);
    Tensor b_zp;
    b_zp.elem_type() = TensorProto_DataType_INT8;
    b_zp.sizes() = b_scale.sizes();
    b_zp.int32s().assign(b_scale.floats().size(), 0);

    Tensor a_scale_t;
    a_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    a_scale_t.floats() = {a_scale_f};
    Tensor a_zp_t;
    a_zp_t.elem_type() = TensorProto_DataType_UINT8;
    a_zp_t.int32s() = {a_zp_i};
    Value* a_scale_v = graph.addInitializerAndCreateValue(a_scale_t);
    Value* a_zp_v = graph.addInitializerAndCreateValue(a_zp_t);

    Tensor y_scale_t;
    y_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    y_scale_t.floats() = {y_scale_f};
    Tensor y_zp_t;
    y_zp_t.elem_type() = TensorProto_DataType_UINT8;
    y_zp_t.int32s() = {y_zp_i};
    Value* y_scale_v = graph.addInitializerAndCreateValue(y_scale_t);
    Value* y_zp_v = graph.addInitializerAndCreateValue(y_zp_t);

    Value* b_q_v = graph.addInitializerAndCreateValue(b_q);
    Value* b_scale_v = graph.addInitializerAndCreateValue(b_scale);
    Value* b_zp_v = graph.addInitializerAndCreateValue(b_zp);

    // Aq = QuantizeLinear(A, As, Azp)
    Node* aq = graph.create(Symbol("QuantizeLinear"), 1);
    aq->addInput(info.a);
    aq->addInput(a_scale_v);
    aq->addInput(a_zp_v);
    aq->insertBefore(n);
    aq->output()->setElemType(TensorProto_DataType_UINT8);

    // Yq = QGemm(Aq, As, Azp, Bq, Bs, Bzp, [Cq,] Ys, Yzp, transA, transB,
    //            alpha), a "com.microsoft" contrib op.
    Node* qgemm = graph.create(Symbol("QGemm"), 1);
    qgemm->addInput(aq->output());
    qgemm->addInput(a_scale_v);
    qgemm->addInput(a_zp_v);
    qgemm->addInput(b_q_v);
    qgemm->addInput(b_scale_v);
    qgemm->addInput(b_zp_v);
    if (info.c_t != nullptr) {
      const std::vector<float>& b_scales = b_scale.floats();
      const std::vector<float> c_vals = ReadFloatVector1D(*info.c_t);
      Tensor c_q;
      c_q.elem_type() = TensorProto_DataType_INT32;
      c_q.sizes() = {info.n};
      c_q.int32s().resize(static_cast<size_t>(info.n));
      for (int64_t i = 0; i < info.n; ++i) {
        const double bias_scale =
            info.alpha * static_cast<double>(a_scale_f) *
            static_cast<double>(b_scales[static_cast<size_t>(i)]);
        const double q =
            bias_scale != 0.0
                ? std::round(c_vals[static_cast<size_t>(i)] / bias_scale)
                : 0.0;
        const double clamped = std::clamp(
            q, static_cast<double>(std::numeric_limits<int32_t>::min()),
            static_cast<double>(std::numeric_limits<int32_t>::max()));
        c_q.int32s()[static_cast<size_t>(i)] = static_cast<int32_t>(clamped);
      }
      Value* c_q_v = graph.addInitializerAndCreateValue(c_q);
      qgemm->addInput(c_q_v);
    } else {
      // No C input: represent the omitted middle-position optional as an
      // empty-string input, the standard ONNX convention for skipping a
      // non-trailing optional -- the same kUndefined-placeholder pattern
      // onnxoptimizer's own split.h pass uses for the same purpose.
      Node* undef = graph.create(kUndefined, 1);
      undef->insertBefore(n);
      undef->output()->setUniqueName("");
      qgemm->addInput(undef->output());
    }
    qgemm->addInput(y_scale_v);
    qgemm->addInput(y_zp_v);
    qgemm->i_(ktransA, info.trans_a);
    qgemm->i_(ktransB, info.trans_b);
    qgemm->f_(kalpha, static_cast<float>(info.alpha));
    qgemm->setDomain("com.microsoft");
    qgemm->insertBefore(n);
    qgemm->output()->setElemType(TensorProto_DataType_UINT8);

    // Y = DequantizeLinear(Yq, Ys, Yzp)
    Node* dq = graph.create(Symbol("DequantizeLinear"), 1);
    dq->addInput(qgemm->output());
    dq->addInput(y_scale_v);
    dq->addInput(y_zp_v);
    dq->insertBefore(n);
    dq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (n->output()->sizes().size() > 0) {
      dq->output()->setSizes(n->output()->sizes());
    }

    bool has_ms_domain = false;
    for (const OpSetID& opset : graph.opset_versions_mutable()) {
      if (opset.domain() == "com.microsoft") {
        has_ms_domain = true;
        break;
      }
    }
    if (!has_ms_domain) {
      graph.opset_versions_mutable().emplace_back("com.microsoft", 1);
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(n->output(), dq->output());
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
