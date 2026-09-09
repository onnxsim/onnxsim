/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The emitter half of onnxsim/qat_graph.py, in C++. See qat_graph_builder.h
 * for what a step graph is and why only the emitter crossed over; see
 * qat_graph.py for the reasoning behind the operator set, the composed
 * rounding, and Adam's shape. Neither is repeated here.
 *
 * What *is* here, because it exists only on this side: every place where
 * reproducing the Python's bytes needed C++ to be written against the grain.
 * Two of those recur, and both are silent when got wrong -- the emitted graph
 * still runs, it just stops being the graph the Python emits:
 *
 *   1. Python evaluates a call's arguments left to right, so
 *      `self.op("Clip", [a, self.const(low), self.const(high)])` names both
 *      constants before it names the Clip. C++ leaves argument evaluation
 *      order unspecified, so every such nesting is flattened into sequenced
 *      locals below. The names are what differ if it is not, and names are
 *      what the parity test compares.
 *   2. qat_graph.py's ADAM_BETA1/ADAM_BETA2 are Python floats, i.e. doubles,
 *      and numpy narrows to float32 only at the initializer. `1.0 - ADAM_BETA1`
 *      is therefore a double subtraction rounded once; doing it in float32
 *      instead rounds twice and lands on a different float (see kBeta1Double).
 */
#include "qat_graph_builder.h"

#include <onnx/inliner/inliner.h>
#include <onnx/onnx_pb.h>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// ONNX's raw_data is little-endian on every host, so these serialize by
// shifting bytes out of the integer representation rather than memcpy'ing the
// host's own layout -- correct on a big-endian host, not merely untested there.
void AppendLittleEndian(std::string& out, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
}

void AppendLittleEndian(std::string& out, int64_t value) {
  const uint64_t bits = static_cast<uint64_t>(value);
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
}

std::string Lowered(const std::string& s) {
  std::string out = s;
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

onnx::AttributeProto IntAttr(const std::string& name, int64_t value) {
  onnx::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(value);
  return attr;
}

onnx::AttributeProto IntsAttr(const std::string& name,
                              const std::vector<int64_t>& values) {
  onnx::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(onnx::AttributeProto::INTS);
  for (int64_t v : values) attr.add_ints(v);
  return attr;
}

// onnx.helper.make_tensor_value_info: the shape field is written even for a
// rank-0 value, which is what distinguishes "scalar" from "rank unknown".
void SetValueInfo(onnx::ValueInfoProto* vi, const std::string& name,
                  int32_t elem_type, const std::vector<int64_t>& dims) {
  vi->set_name(name);
  onnx::TypeProto::Tensor* tensor = vi->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(elem_type);
  onnx::TensorShapeProto* shape = tensor->mutable_shape();
  for (int64_t d : dims) shape->add_dim()->set_dim_value(d);
}

// The double-valued betas. qat_graph.py spells ADAM_BETA1 as `0.9`, a double,
// and `1.0 - ADAM_BETA1` is a double subtraction that numpy narrows once, to
// 0.1f. Subtracting the *float* kAdamBeta1 from 1.0f is exact (Sterbenz) but
// exact about the wrong number: it yields 0.10000002f, four bytes that differ
// from the Python's initializer. The static_asserts tie these back to the
// header's constants so the two spellings cannot drift apart.
constexpr double kBeta1Double = 0.9;
constexpr double kBeta2Double = 0.999;
static_assert(static_cast<float>(kBeta1Double) == kAdamBeta1,
              "kAdamBeta1 must be the float32 narrowing of 0.9");
static_assert(static_cast<float>(kBeta2Double) == kAdamBeta2,
              "kAdamBeta2 must be the float32 narrowing of 0.999");

}  // namespace

const std::set<std::string>& EpFriendlyOps() {
  static const std::set<std::string> kOps = {
      "Abs",        "Add",       "Cast",    "Clip", "Div",  "Exp", "Gather",
      "Greater",    "Less",      "MatMul",  "Mul",  "Neg",  "Pow", "Reshape",
      "ReduceMean", "ReduceSum", "Sigmoid", "Sign", "Sqrt", "Sub", "Transpose"};
  return kOps;
}

std::string GraphBuilder::Name(const std::string& hint) {
  ++counter_;
  return prefix_ + hint + "_" + std::to_string(counter_);
}

std::string GraphBuilder::Const(float value, const std::string& hint) {
  const std::string name = Name(hint);
  onnx::TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(onnx::TensorProto::FLOAT);
  // No dims: a numpy scalar has shape (), and numpy_helper.from_array copies
  // that empty shape through.
  std::string raw;
  AppendLittleEndian(raw, value);
  tensor.set_raw_data(std::move(raw));
  initializer_.push_back(std::move(tensor));
  return name;
}

std::string GraphBuilder::Const(const std::vector<float>& values,
                                const std::vector<int64_t>& dims,
                                const std::string& hint) {
  const std::string name = Name(hint);
  onnx::TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) tensor.add_dims(d);
  std::string raw;
  raw.reserve(values.size() * sizeof(float));
  for (float v : values) AppendLittleEndian(raw, v);
  tensor.set_raw_data(std::move(raw));
  initializer_.push_back(std::move(tensor));
  return name;
}

std::string GraphBuilder::ConstInt64(const std::vector<int64_t>& values,
                                     const std::string& hint) {
  const std::string name = Name(hint);
  onnx::TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(onnx::TensorProto::INT64);
  tensor.add_dims(static_cast<int64_t>(values.size()));
  std::string raw;
  raw.reserve(values.size() * sizeof(int64_t));
  for (int64_t v : values) AppendLittleEndian(raw, v);
  tensor.set_raw_data(std::move(raw));
  initializer_.push_back(std::move(tensor));
  return name;
}

std::string GraphBuilder::Op(const std::string& op_type,
                             const std::vector<std::string>& inputs,
                             const std::string& hint) {
  return Op(op_type, inputs, {}, hint);
}

std::string GraphBuilder::Op(const std::string& op_type,
                             const std::vector<std::string>& inputs,
                             const std::vector<onnx::AttributeProto>& attrs,
                             const std::string& hint) {
  const std::string out = Name(hint.empty() ? Lowered(op_type) : hint);
  OpInto(op_type, inputs, out, attrs);
  return out;
}

void GraphBuilder::OpInto(const std::string& op_type,
                          const std::vector<std::string>& inputs,
                          const std::string& output,
                          const std::vector<onnx::AttributeProto>& attrs) {
  onnx::NodeProto node;
  node.set_op_type(op_type);
  for (const std::string& in : inputs) node.add_input(in);
  node.add_output(output);
  for (const onnx::AttributeProto& attr : attrs) *node.add_attribute() = attr;
  // Node names are left empty, as onnx.helper.make_node leaves them when no
  // name= is passed; a generated node name would be one more thing the two
  // emitters would have to agree on.
  nodes_.push_back(std::move(node));
}

std::vector<std::string> GraphBuilder::Call(
    const onnx::FunctionProto& fn, const std::vector<std::string>& inputs) {
  const auto key = std::make_pair(fn.domain(), fn.name());
  if (function_ids_.insert(key).second) {
    functions_.push_back(fn);
  }
  std::vector<std::string> outs;
  outs.reserve(static_cast<size_t>(fn.output_size()));
  onnx::NodeProto node;
  node.set_op_type(fn.name());
  node.set_domain(fn.domain());
  for (const std::string& in : inputs) node.add_input(in);
  for (int i = 0; i < fn.output_size(); ++i) {
    const std::string out = Name(Lowered(fn.name()));
    node.add_output(out);
    outs.push_back(out);
  }
  nodes_.push_back(std::move(node));
  return outs;
}

std::string GraphBuilder::Add(const std::string& a, const std::string& b) {
  return Op("Add", {a, b});
}

std::string GraphBuilder::Sub(const std::string& a, const std::string& b) {
  return Op("Sub", {a, b});
}

std::string GraphBuilder::Mul(const std::string& a, const std::string& b) {
  return Op("Mul", {a, b});
}

std::string GraphBuilder::Div(const std::string& a, const std::string& b) {
  return Op("Div", {a, b});
}

std::string GraphBuilder::MatMul(const std::string& a, const std::string& b) {
  return Op("MatMul", {a, b});
}

std::string GraphBuilder::Transpose(const std::string& a) {
  return Op("Transpose", {a});
}

std::string GraphBuilder::Transpose(const std::string& a,
                                    const std::vector<int64_t>& perm) {
  return Op("Transpose", {a}, {IntsAttr("perm", perm)});
}

std::string GraphBuilder::Sqrt(const std::string& a) { return Op("Sqrt", {a}); }

std::string GraphBuilder::Sigmoid(const std::string& a) {
  return Op("Sigmoid", {a});
}

std::string GraphBuilder::Clip(const std::string& a, float low, float high) {
  // Sequenced, not nested: see this file's header comment (1).
  const std::string lo = Const(low);
  const std::string hi = Const(high);
  return Op("Clip", {a, lo, hi});
}

std::string GraphBuilder::GreaterMask(const std::string& a, float threshold) {
  const std::string bound = Const(threshold);
  const std::string gt = Op("Greater", {a, bound});
  return Op("Cast", {gt}, {IntAttr("to", onnx::TensorProto::FLOAT)});
}

std::string GraphBuilder::LessMask(const std::string& a, float threshold) {
  const std::string bound = Const(threshold);
  const std::string lt = Op("Less", {a, bound});
  return Op("Cast", {lt}, {IntAttr("to", onnx::TensorProto::FLOAT)});
}

std::string GraphBuilder::RoundToNearest(const std::string& a) {
  const std::string absolute = Op("Abs", {a});
  const std::string half = Const(0.5f);
  const std::string magnitude = Add(absolute, half);
  const std::string truncated =
      Op("Cast", {magnitude}, {IntAttr("to", onnx::TensorProto::INT32)});
  const std::string sign = Op("Sign", {a});
  const std::string rounded =
      Op("Cast", {truncated}, {IntAttr("to", onnx::TensorProto::FLOAT)});
  return Mul(sign, rounded);
}

std::string GraphBuilder::MeanSquare(const std::string& a) {
  const std::string square = Mul(a, a);
  // No axes: ReduceMean is at version 13 in opset 17, where axes is still an
  // attribute, and omitting it reduces over every axis -- which is the whole
  // tensor's mean, i.e. the scalar a loss is.
  return Op("ReduceMean", {square}, {IntAttr("keepdims", 0)});
}

std::string GraphBuilder::GatherRows(const std::string& table,
                                     const std::string& index) {
  return Op("Gather", {table, index}, {IntAttr("axis", 0)}, "rows");
}

void GraphBuilder::GatherRowsInto(const std::string& table,
                                  const std::string& index,
                                  const std::string& output) {
  // Consumes no name, exactly as gather_rows(out=...) does not call self.name.
  OpInto("Gather", {table, index}, output, {IntAttr("axis", 0)});
}

AdamOutputs AdamUpdate(GraphBuilder& b, const std::string& param,
                       const std::string& grad, const std::string& m,
                       const std::string& v, const std::string& lr,
                       const std::string& m_correction,
                       const std::string& v_correction, float eps) {
  const std::string beta1 = b.Const(kAdamBeta1);
  const std::string beta2 = b.Const(kAdamBeta2);
  const std::string one_minus_beta1 =
      b.Const(static_cast<float>(1.0 - kBeta1Double));
  const std::string one_minus_beta2 =
      b.Const(static_cast<float>(1.0 - kBeta2Double));

  // Every nesting in adam_update is flattened here for the argument-order
  // reason above; the left-to-right order of these locals *is* the Python's.
  const std::string decayed_m = b.Mul(beta1, m);
  const std::string scaled_grad = b.Mul(one_minus_beta1, grad);
  const std::string m_next = b.Add(decayed_m, scaled_grad);

  const std::string decayed_v = b.Mul(beta2, v);
  const std::string grad_squared = b.Mul(grad, grad);
  const std::string scaled_grad_squared = b.Mul(one_minus_beta2, grad_squared);
  const std::string v_next = b.Add(decayed_v, scaled_grad_squared);

  const std::string m_hat = b.Mul(m_next, m_correction);
  const std::string v_hat = b.Mul(v_next, v_correction);

  const std::string numerator = b.Mul(lr, m_hat);
  const std::string root = b.Sqrt(v_hat);
  const std::string epsilon = b.Const(eps);
  const std::string denominator = b.Add(root, epsilon);
  const std::string step = b.Div(numerator, denominator);

  return AdamOutputs{b.Sub(param, step), m_next, v_next};
}

std::pair<float, float> AdamBiasCorrections(int64_t t) {
  // In double, then narrowed once -- the Python returns Python floats and the
  // narrowing happens where they are fed. Computing in float32 throughout
  // would round the pow, the subtraction and the division separately.
  const double exponent = static_cast<double>(t) + 1.0;
  return {static_cast<float>(1.0 / (1.0 - std::pow(kBeta1Double, exponent))),
          static_cast<float>(1.0 / (1.0 - std::pow(kBeta2Double, exponent)))};
}

StepGraph MakeStepGraph(const GraphBuilder& b, const StepGraphSpec& spec) {
  StepGraph result;
  onnx::GraphProto* graph = result.model.mutable_graph();
  graph->set_name(spec.graph_name);

  // Input order is part of the contract, not an implementation detail: the
  // Python builds these from four dicts in this order and a positional binding
  // (ORT's IOBinding, and the WASM trampoline) reads them positionally.
  for (const StepGraphSpec::NamedShape& c : spec.constants) {
    SetValueInfo(graph->add_input(), c.name, c.elem_type, c.dims);
  }
  for (const StepGraphSpec::StateEntry& s : spec.state) {
    SetValueInfo(graph->add_input(), s.input, onnx::TensorProto::FLOAT, s.dims);
  }
  for (const std::string& s : spec.scalars) {
    SetValueInfo(graph->add_input(), s, onnx::TensorProto::FLOAT, {});
  }
  for (const StepGraphSpec::PerStepEntry& p : spec.per_step) {
    SetValueInfo(graph->add_input(), p.name, p.elem_type, p.dims);
  }

  for (const StepGraphSpec::StateEntry& s : spec.state) {
    SetValueInfo(graph->add_output(), s.next_output, onnx::TensorProto::FLOAT,
                 s.dims);
    result.state.emplace_back(s.input, s.next_output);
  }
  if (!spec.loss_output.empty()) {
    SetValueInfo(graph->add_output(), spec.loss_output,
                 onnx::TensorProto::FLOAT, {});
  }

  for (const onnx::NodeProto& node : b.nodes()) *graph->add_node() = node;
  for (const onnx::TensorProto& init : b.initializer()) {
    *graph->add_initializer() = init;
  }

  onnx::OperatorSetIdProto* opset = result.model.add_opset_import();
  opset->set_domain("");
  opset->set_version(kStepGraphOpset);
  // One opset_import per distinct function domain, at a fixed private
  // version this repo controls entirely -- unrelated to kStepGraphOpset,
  // which is what the function *bodies* were authored against internally
  // (each FunctionProto carries its own opset_import for that). Mirrors
  // qat_graph.py's make_step_graph exactly.
  std::set<std::string> function_domains;
  for (const onnx::FunctionProto& fn : b.functions()) {
    *result.model.add_functions() = fn;
    if (function_domains.insert(fn.domain()).second) {
      onnx::OperatorSetIdProto* fn_opset = result.model.add_opset_import();
      fn_opset->set_domain(fn.domain());
      fn_opset->set_version(1);
    }
  }
  result.model.set_ir_version(kStepGraphIrVersion);
  // No producer_name: onnx.helper.make_model sets none either, and an
  // initialized-vs-absent field is a byte-level difference.
  result.loss_name = spec.loss_output;
  if (!b.functions().empty()) {
    // Expand every call site before this model reaches a runtime: no
    // execution provider needs to know about the private grad domain, and
    // the result composes with the rest of this file exactly like a
    // hand-written rule's nodes always have.
    onnx::inliner::InlineLocalFunctions(result.model);
  }
  return result;
}
