/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The C++ port of onnxsim/bias_correction.py's graph-surgery half -- see
 * bias_correction_entry.h for what is ported, what stays in JS, and why.
 * None of the technique (what a correction means, when it does or doesn't
 * help) is repeated here -- see bias_correction.py's own module docstring
 * for that; this only moves already-measured numbers into the graph.
 */

#include "bias_correction_entry.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

namespace {

// Transcribed from bias_correction.py's own _all_names: every name a fresh
// node/tensor name must avoid colliding with.
std::set<std::string> AllNames(const onnx::GraphProto& graph) {
  std::set<std::string> names;
  for (const auto& t : graph.initializer()) {
    names.insert(t.name());
  }
  for (const auto& vi : graph.input()) {
    names.insert(vi.name());
  }
  for (const auto& vi : graph.output()) {
    names.insert(vi.name());
  }
  for (const auto& vi : graph.value_info()) {
    names.insert(vi.name());
  }
  for (const auto& n : graph.node()) {
    if (!n.name().empty()) {
      names.insert(n.name());
    }
    for (const auto& x : n.input()) {
      names.insert(x);
    }
    for (const auto& x : n.output()) {
      names.insert(x);
    }
  }
  return names;
}

// Transcribed from bias_correction.py's own _unique_name.
std::string UniqueName(const std::string& base, std::set<std::string>& taken) {
  std::string name = base;
  int suffix = 0;
  while (taken.count(name)) {
    name = base + "_" + std::to_string(++suffix);
  }
  taken.insert(name);
  return name;
}

// op_type -> channel axis, exactly bias_correction.py's own
// _CORRECTABLE_OPS (see that module for the NCHW-vs-trailing-feature-dim
// rationale).
const std::unordered_map<std::string, int64_t>& CorrectableOps() {
  static const std::unordered_map<std::string, int64_t> ops = {
      {"Conv", 1},
      {"Gemm", -1},
      {"MatMul", -1},
      {"Resize", 1},
  };
  return ops;
}

}  // namespace

onnx::ModelProto ApplyBiasCorrections(
    const onnx::ModelProto& model,
    const std::vector<BiasCorrectionEntry>& corrections) {
  onnx::ModelProto corrected = model;
  std::set<std::string> taken_names = AllNames(corrected.graph());
  auto* nodes = corrected.mutable_graph()->mutable_node();

  for (const auto& entry : corrections) {
    int producer_idx = -1;
    int output_index = -1;
    for (int idx = 0; idx < nodes->size(); ++idx) {
      const onnx::NodeProto& node = nodes->Get(idx);
      for (int oi = 0; oi < node.output_size(); ++oi) {
        if (node.output(oi) == entry.output_name) {
          producer_idx = idx;
          output_index = oi;
          break;
        }
      }
      if (producer_idx >= 0) {
        break;
      }
    }
    if (producer_idx < 0) {
      continue;  // not present in this model (e.g. measured elsewhere) -- skip
    }

    const std::string pre_correction_name =
        UniqueName(entry.output_name + "_bias_correction_pre", taken_names);
    nodes->Mutable(producer_idx)->set_output(output_index, pre_correction_name);

    const std::string scale_name =
        UniqueName(entry.output_name + "_bias_correction", taken_names);
    onnx::TensorProto* tensor = corrected.mutable_graph()->add_initializer();
    tensor->set_name(scale_name);
    tensor->set_data_type(onnx::TensorProto::FLOAT);
    for (int64_t dim : entry.shape) {
      tensor->add_dims(dim);
    }
    tensor->mutable_float_data()->Add(entry.data.begin(), entry.data.end());

    const std::string add_name =
        UniqueName(entry.output_name + "_bias_correction_add", taken_names);
    onnx::NodeProto* add_node = corrected.mutable_graph()->add_node();
    add_node->set_op_type("Add");
    add_node->set_name(add_name);
    add_node->add_input(pre_correction_name);
    add_node->add_input(scale_name);
    add_node->add_output(entry.output_name);

    // The Add node was just appended at the end; walk it back down to right
    // after its producer via adjacent swaps, preserving every other node's
    // relative order -- matches bias_correction.py's own
    // node.insert(producer_idx + 1, add_node) placement. ONNX doesn't
    // require topological node order, but onnxsim itself always leaves the
    // graph in one, and this keeps that true for any downstream tool that
    // assumes it.
    for (int i = nodes->size() - 1; i > producer_idx + 1; --i) {
      nodes->SwapElements(i, i - 1);
    }
  }

  return corrected;
}

std::vector<CorrectableCandidate> ListCorrectableOutputs(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& modified_model) {
  std::unordered_set<std::string> modified_outputs;
  for (const auto& n : modified_model.graph().node()) {
    for (const auto& out : n.output()) {
      modified_outputs.insert(out);
    }
  }

  std::vector<CorrectableCandidate> candidates;
  const auto& correctable_ops = CorrectableOps();
  for (const auto& n : float_model.graph().node()) {
    auto it = correctable_ops.find(n.op_type());
    if (it == correctable_ops.end()) {
      continue;
    }
    if (n.output_size() == 0 ||
        modified_outputs.find(n.output(0)) == modified_outputs.end()) {
      continue;
    }
    candidates.push_back({n.output(0), it->second, it->second == 1});
  }
  return candidates;
}
