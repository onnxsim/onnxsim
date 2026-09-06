// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See imatrix_quant_entry.h for the full rationale (including why this
// follows structured_pruning_entry.h's own protobuf-level,
// calibration-driven shape rather than apply_quarot_cpp's data-free
// PredicateBasedPass one) and onnxsim/imatrix_quant.py for the technique
// this ports.

#include "imatrix_quant_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"
#include "passes/imatrix_quant.h"

namespace {

// --- MatMul/vanilla-Gemm matching, protobuf level --------------------------
//
// Transcribed from structured_pruning_entry.cpp's own MatchMatMulLikeRaw
// (same file this module's own calibration shape is modeled on) -- mirrors
// onnxsim.llm_int8._match_matmul_like exactly: a MatMul, or a Gemm with
// transA=0, alpha=1 and (when it has a bias) beta=1. This pass never reads
// or rewrites a bias, so unlike MatchMatMulLikeRaw's own 2-3 input Gemm
// acceptance, the bias itself is simply ignored (node.input(2), if any, is
// left completely untouched).
struct MatMulLikeMatch {
  std::string x_name;
  std::string w_name;
  bool weight_transposed;
};

std::optional<MatMulLikeMatch> MatchMatMulLike(const onnx::NodeProto& node) {
  if (node.op_type() == "MatMul") {
    if (node.input_size() != 2) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), false};
  }
  if (node.op_type() == "Gemm") {
    const int num_inputs = node.input_size();
    if (num_inputs != 2 && num_inputs != 3) {
      return std::nullopt;
    }
    bool has_trans_a = false, has_alpha = false, has_beta = false;
    int64_t trans_a = 0, trans_b = 0;
    double alpha = 1.0, beta = 1.0;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "transA") {
        trans_a = attr.i();
        has_trans_a = true;
      } else if (attr.name() == "alpha") {
        alpha = attr.f();
        has_alpha = true;
      } else if (attr.name() == "beta") {
        beta = attr.f();
        has_beta = true;
      } else if (attr.name() == "transB") {
        trans_b = attr.i();
      }
    }
    if (has_trans_a && trans_a != 0) {
      return std::nullopt;
    }
    if (has_alpha && alpha != 1.0) {
      return std::nullopt;
    }
    if (num_inputs == 3 && has_beta && beta != 1.0) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), trans_b != 0};
  }
  return std::nullopt;
}

// --- Tensor <-> flat float buffer, protobuf level ---------------------------
//
// Transcribed verbatim from structured_pruning_entry.cpp's own
// ReadFloatTensor/SetFloatTensorData (FLOAT32 only -- this pass, like its
// own Python reference onnxsim.imatrix_quant, never widens to FLOAT16/
// BFLOAT16), reusing dlpack_bridge.h's already-verified
// kRawDataIsHostOrder/SwapElementBytes for the raw_data little-endian
// convention every onnx::TensorProto.raw_data() must hold.

std::vector<float> ReadFloatTensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(out.data(), t.raw_data().data(), out.size() * sizeof(float));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                        out.size() * sizeof(float),
                                        sizeof(float));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = t.float_data(static_cast<int>(i));
    }
  }
  return out;
}

void SetFloatInitializer(onnx::TensorProto* t, const std::string& name,
                         const std::vector<int64_t>& dims,
                         const std::vector<float>& data) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(data.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

// --- Calibration: per-input-channel mean-square activation ("imatrix") ----
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as
// structured_pruning_entry.cpp's own WandaCalibrationStats (see
// imatrix_quant_entry.h's own top comment for why this pass follows that
// file's precedent rather than a PredicateBasedPass), simplified to this
// pass' own narrower needs: FLOAT32 only (matching
// onnxsim.imatrix_quant.compute_activation_importance's own unconditional
// float64 upcast, which never widens to FLOAT16/BFLOAT16 either), always
// the last axis (every candidate here is a plain MatMul/vanilla-Gemm 2-D
// weight's own activation, never Conv/Attention), and returns the mean
// square directly (WandaCalibrationStats' own sum-of-squares-over-count,
// WITHOUT the final sqrt WandaCalibrationStats applies to turn that into an
// L2-norm/RMS) -- exactly onnxsim.imatrix_quant.compute_activation_
// importance's own `sum_sq[name] / count[name]`.
std::unordered_map<std::string, std::vector<double>>
ComputeMeanSquareImportance(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  std::unordered_map<std::string, std::vector<double>> result;
  if (probe_names.empty()) {
    return result;
  }

  onnx::ModelProto probe_model = model;
  std::unordered_set<std::string> existing_outputs;
  for (const auto& o : probe_model.graph().output()) {
    existing_outputs.insert(o.name());
  }
  for (const auto& name : probe_names) {
    if (existing_outputs.insert(name).second) {
      probe_model.mutable_graph()->add_output()->set_name(name);
    }
  }

  std::unordered_map<std::string, size_t> output_index;
  for (int i = 0; i < probe_model.graph().output_size(); ++i) {
    output_index.emplace(probe_model.graph().output(i).name(),
                         static_cast<size_t>(i));
  }
  const auto& graph_inputs = probe_model.graph().input();

  std::unordered_map<std::string, std::vector<double>> sum_sq;
  std::unordered_map<std::string, int64_t> count;

  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> input_dls;
    std::vector<const DLManagedTensor*> input_ptrs;
    input_dls.reserve(static_cast<size_t>(graph_inputs.size()));
    input_ptrs.reserve(static_cast<size_t>(graph_inputs.size()));
    for (const auto& gi : graph_inputs) {
      auto it = batch.find(gi.name());
      if (it == batch.end()) {
        throw std::invalid_argument(
            "ApplyImatrixQuantization: calibration batch is missing "
            "required graph input '" +
            gi.name() + "'");
      }
      input_dls.emplace_back(
          onnxsim::dlpack::FromTensorProtoBorrowing(it->second));
      input_ptrs.push_back(input_dls.back().get());
    }

    std::vector<DLManagedTensorPtr> outputs =
        executor.Run(probe_model, input_ptrs);

    for (const auto& name : probe_names) {
      auto oit = output_index.find(name);
      if (oit == output_index.end() || oit->second >= outputs.size()) {
        continue;  // Defensive -- every probe name was added as an output
                   // above.
      }
      const DLTensor& dl = outputs[oit->second]->dl_tensor;
      onnx::TensorProto tp = onnxsim::dlpack::ToTensorProto(dl);
      if (tp.data_type() != onnx::TensorProto::FLOAT) {
        continue;
      }
      const int64_t ndim = tp.dims_size();
      if (ndim == 0) {
        continue;
      }
      const int64_t channel_dim = tp.dims(static_cast<int>(ndim - 1));
      if (channel_dim <= 0) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      auto& acc = sum_sq[name];
      if (acc.empty()) {
        acc.assign(static_cast<size_t>(channel_dim), 0.0);
      }
      // Row-major contiguous (ToTensorProto's own output always is), channel
      // is the last axis, so flat position `flat`'s channel index is simply
      // `flat % channel_dim` -- mirrors
      // compute_activation_importance's own
      // `x.reshape(-1, x.shape[-1])`.
      const int64_t total = static_cast<int64_t>(data.size());
      for (int64_t flat = 0; flat < total; ++flat) {
        const int64_t c = flat % channel_dim;
        const double v = static_cast<double>(data[static_cast<size_t>(flat)]);
        acc[static_cast<size_t>(c)] += v * v;
      }
      count[name] += total / channel_dim;
    }
  }

  result.reserve(sum_sq.size());
  for (auto& [name, acc] : sum_sq) {
    const int64_t cnt = std::max<int64_t>(count[name], 1);
    for (double& v : acc) {
      v /= static_cast<double>(cnt);
    }
    result.emplace(name, std::move(acc));
  }
  return result;
}

}  // namespace

onnx::ModelProto ApplyImatrixQuantization(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size, int64_t num_scale_candidates, double scale_lo,
    double scale_hi, const std::unordered_set<std::string>& skip_names) {
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_map<std::string, int> init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }

  struct Candidate {
    int node_index;
    std::string x_name;
    std::string w_name;
    bool weight_transposed;
  };
  std::vector<Candidate> candidates;
  for (int i = 0; i < graph->node_size(); ++i) {
    auto m = MatchMatMulLike(graph->node(i));
    if (!m) {
      continue;
    }
    if (skip_names.count(m->w_name) != 0) {
      continue;
    }
    auto it = init_index.find(m->w_name);
    if (it == init_index.end()) {
      continue;
    }
    const onnx::TensorProto& w_init = graph->initializer(it->second);
    if (w_init.data_type() != onnx::TensorProto::FLOAT ||
        w_init.dims_size() != 2) {
      continue;
    }
    candidates.push_back({i, m->x_name, m->w_name, m->weight_transposed});
  }
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  const std::unordered_map<std::string, std::vector<double>> importance =
      ComputeMeanSquareImportance(executor, out, probe_names, calibration_data);

  // Mirrors onnxsim.bias_correction._all_names -- every name already in use
  // anywhere in the graph, so a freshly minted initializer name can never
  // collide.
  std::unordered_set<std::string> taken_names;
  for (const auto& t : graph->initializer()) {
    taken_names.insert(t.name());
  }
  for (const auto& vi : graph->input()) {
    taken_names.insert(vi.name());
  }
  for (const auto& vi : graph->output()) {
    taken_names.insert(vi.name());
  }
  for (const auto& vi : graph->value_info()) {
    taken_names.insert(vi.name());
  }
  for (const auto& n : graph->node()) {
    if (!n.name().empty()) {
      taken_names.insert(n.name());
    }
    for (const auto& s : n.input()) {
      taken_names.insert(s);
    }
    for (const auto& s : n.output()) {
      taken_names.insert(s);
    }
  }
  auto unique_name = [&](const std::string& base) {
    std::string name = base;
    int i = 0;
    while (taken_names.count(name) != 0) {
      ++i;
      name = base + "_" + std::to_string(i);
    }
    taken_names.insert(name);
    return name;
  };

  std::unordered_map<std::string, std::string> quantized_names;
  for (const auto& c : candidates) {
    auto imp_it = importance.find(c.x_name);
    if (imp_it == importance.end()) {
      continue;
    }

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows =
        c.weight_transposed ? dim0 : dim1;                // output channels
    const int64_t k = c.weight_transposed ? dim1 : dim0;  // reduction dim
    if (block_size <= 0 || k % block_size != 0 ||
        imp_it->second.size() != static_cast<size_t>(k)) {
      continue;
    }

    auto qn_it = quantized_names.find(c.w_name);
    std::string new_name;
    if (qn_it != quantized_names.end()) {
      new_name = qn_it->second;
    } else {
      const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
      std::vector<float> w_nk(static_cast<size_t>(n_rows * k));
      for (int64_t i0 = 0; i0 < dim0; ++i0) {
        for (int64_t j0 = 0; j0 < dim1; ++j0) {
          const float v = flat[static_cast<size_t>(i0 * dim1 + j0)];
          if (c.weight_transposed) {
            w_nk[static_cast<size_t>(i0 * k + j0)] = v;  // already [N, K]
          } else {
            w_nk[static_cast<size_t>(j0 * k + i0)] = v;  // [K, N] -> [N, K]
          }
        }
      }

      const std::vector<float> w_quant_nk =
          onnxsim_passes::QuantizeDequantizeInt4Imatrix(
              w_nk, n_rows, k, imp_it->second, block_size, num_scale_candidates,
              scale_lo, scale_hi);

      std::vector<float> w_quant(static_cast<size_t>(dim0 * dim1));
      for (int64_t i0 = 0; i0 < dim0; ++i0) {
        for (int64_t j0 = 0; j0 < dim1; ++j0) {
          w_quant[static_cast<size_t>(i0 * dim1 + j0)] =
              c.weight_transposed
                  ? w_quant_nk[static_cast<size_t>(i0 * k + j0)]
                  : w_quant_nk[static_cast<size_t>(j0 * k + i0)];
        }
      }

      new_name = unique_name(c.w_name + "_imatrix");
      SetFloatInitializer(graph->add_initializer(), new_name, {dim0, dim1},
                          w_quant);
      quantized_names.emplace(c.w_name, new_name);
    }
    graph->mutable_node(c.node_index)->set_input(1, new_name);
  }

  return out;
}
