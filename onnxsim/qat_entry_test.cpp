/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises qat_entry.{h,cpp} -- the C++ port of the graph-building half of
 * onnxsim/qat.py.
 *
 * tests/test_qat.py checks the Python the way a training pass has to be
 * checked: it runs the loop and measures that the reconstruction error falls.
 * None of that is reproducible here, because nothing in this build evaluates
 * an ONNX graph (the wheel does not compile ONNX Runtime -- see CLAUDE.md --
 * and the WASM build hands evaluation to onnxruntime-web at run time). So
 * this covers the half that *is* checkable without an evaluator, and covers
 * it exactly: that the emitted step graph is a legal ONNX model, that the
 * plan's four hand-offs to the caller (captures, state, scalars, layers) name
 * the right tensors with the right shapes, that the write-back is the exact
 * inverse of the warm start, and that every refusal fires with the message
 * that tells a human which of the two schemes they aimed at.
 *
 * Node-order parity with qat.py is *not* asserted here -- it cannot be
 * without the Python. It is onnxsim/qat_parity_fixtures.txt's job, and the
 * emission order in qat_entry.cpp is a transcription written for it.
 *
 * Plain asserts and a failure counter, like qat_graph_builder_test.cpp and
 * graph_grad_test.cpp -- this repository vendors no gtest.
 */
#include "qat_entry.h"

#include <onnx/onnx_pb.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "graph_grad.h"
#include "onnx/checker.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void CheckEqual(const std::string& got, const std::string& want,
                const std::string& what) {
  Check(got == want, what + " (got \"" + got + "\", want \"" + want + "\")");
}

void CheckEqual(int64_t got, int64_t want, const std::string& what) {
  Check(got == want, what + " (got " + std::to_string(got) + ", want " +
                         std::to_string(want) + ")");
}

// `body` must throw `E` with a message containing `fragment`. Both halves
// matter: the type is what a caller switches on, and the message is what
// tells a human whether they named the wrong block or aimed at the wrong
// quantization scheme.
template <typename E>
void CheckThrows(const std::function<void()>& body, const std::string& fragment,
                 const std::string& what) {
  try {
    body();
  } catch (const E& error) {
    const std::string message = error.what();
    Check(message.find(fragment) != std::string::npos,
          what + " (message was: " + message + ")");
    return;
  } catch (const std::exception& error) {
    Check(false, what + " -- threw the wrong type: " + error.what());
    return;
  }
  Check(false, what + " -- nothing was thrown");
}

// ---------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------

void AppendFloat(std::string& out, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
}

float DecodeFloat(const char* bytes) {
  uint32_t bits = 0;
  for (int i = 0; i < 4; ++i) {
    bits |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[i]))
            << (8 * i);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<float> FloatsOf(const onnx::TensorProto& t) {
  std::vector<float> out;
  for (size_t i = 0; i + 4 <= t.raw_data().size(); i += 4) {
    out.push_back(DecodeFloat(t.raw_data().data() + i));
  }
  return out;
}

onnx::TensorProto FloatTensor(const std::string& name,
                              const std::vector<int64_t>& dims,
                              const std::vector<float>& values) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) t.add_dims(d);
  std::string raw;
  for (float v : values) AppendFloat(raw, v);
  t.set_raw_data(std::move(raw));
  return t;
}

onnx::TensorProto RawTensor(const std::string& name, int32_t data_type,
                            const std::vector<int64_t>& dims,
                            const std::string& raw) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(data_type);
  for (int64_t d : dims) t.add_dims(d);
  t.set_raw_data(raw);
  return t;
}

onnx::NodeProto MakeNode(
    const std::string& op_type, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::pair<std::string, int64_t>>& int_attrs = {}) {
  onnx::NodeProto node;
  node.set_op_type(op_type);
  for (const std::string& in : inputs) node.add_input(in);
  for (const std::string& out : outputs) node.add_output(out);
  for (const auto& attr : int_attrs) {
    onnx::AttributeProto* a = node.add_attribute();
    a->set_name(attr.first);
    a->set_type(onnx::AttributeProto::INT);
    a->set_i(attr.second);
  }
  return node;
}

// A value info whose leading dimension is symbolic, so the tests exercise
// BuildQatStepGraph's own pinning of it to num_rows rather than reading a
// batch size the model already spelled out.
void AddBatchedInput(onnx::GraphProto* graph, const std::string& name,
                     int64_t width) {
  onnx::ValueInfoProto* vi = graph->add_input();
  vi->set_name(name);
  onnx::TypeProto::Tensor* tensor = vi->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(onnx::TensorProto::FLOAT);
  onnx::TensorShapeProto* shape = tensor->mutable_shape();
  shape->add_dim()->set_dim_param("batch");
  shape->add_dim()->set_dim_value(width);
}

void AddOutput(onnx::GraphProto* graph, const std::string& name,
               int64_t width) {
  onnx::ValueInfoProto* vi = graph->add_output();
  vi->set_name(name);
  onnx::TypeProto::Tensor* tensor = vi->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(onnx::TensorProto::FLOAT);
  onnx::TensorShapeProto* shape = tensor->mutable_shape();
  shape->add_dim()->set_dim_param("batch");
  shape->add_dim()->set_dim_value(width);
}

void Finish(onnx::ModelProto* model) {
  onnx::OperatorSetIdProto* opset = model->add_opset_import();
  opset->set_domain("");
  opset->set_version(21);  // INT4 needs opset 21.
  model->set_ir_version(10);
}

constexpr int64_t kK = 4;
constexpr int64_t kN = 3;
constexpr int64_t kBlock = 2;
constexpr int64_t kRows = 5;

// W / 0.1 is {0.3, -1.8, 3.2, 4.4, -7.1, 0.6, 5.7, -3.3, 9.1, -0.4, 1.2, 2.6}:
// no exact ties (so the half-away rounding is not what is under test) and one
// element past +7 (so the clip to the INT4 grid is).
const std::vector<float>& FloatWeight() {
  static const std::vector<float> w = {0.03f,  -0.18f, 0.32f, 0.44f,
                                       -0.71f, 0.06f,  0.57f, -0.33f,
                                       0.91f,  -0.04f, 0.12f, 0.26f};
  return w;
}

const std::vector<int8_t>& ExpectedCodes() {
  static const std::vector<int8_t> codes = {0, -2, 3, 4, -7, 1,
                                            6, -3, 7, 0, 1,  3};
  return codes;
}

// X -> MatMul(X, W) -> Y, the teacher for every block below.
onnx::ModelProto FloatModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("float");
  AddBatchedInput(graph, "X", kK);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("MatMul", {"X", "W"}, {"Y"});
  *graph->add_initializer() = FloatTensor("W", {kK, kN}, FloatWeight());
  Finish(&model);
  return model;
}

// quantize_weight_only_int4's shape: Y = MatMul(X, DequantizeLinear(Wq, Ws)),
// blocked along the reduction axis.
onnx::ModelProto Int4QuantizedModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("int4");
  AddBatchedInput(graph, "X", kK);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("DequantizeLinear", {"Wq", "Ws"}, {"Wdq"},
                                {{"axis", 0}, {"block_size", kBlock}});
  *graph->add_node() = MakeNode("MatMul", {"X", "Wdq"}, {"Y"});
  // The codes' own values are irrelevant to planning (only dtype and dims are
  // read) and are overwritten wholesale by WriteBackQatState.
  *graph->add_initializer() =
      RawTensor("Wq", onnx::TensorProto::INT4, {kK, kN}, std::string(6, '\0'));
  *graph->add_initializer() =
      FloatTensor("Ws", {kK / kBlock, kN}, std::vector<float>(6, 0.1f));
  Finish(&model);
  return model;
}

// quantize_static's shape: a uint8 affine activation QDQ pair and a
// per-output-channel symmetric INT8 weight.
onnx::ModelProto StaticQdqQuantizedModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("static");
  AddBatchedInput(graph, "X", kK);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("QuantizeLinear", {"X", "Xs", "Xzp"}, {"Xq"});
  *graph->add_node() =
      MakeNode("DequantizeLinear", {"Xq", "Xs", "Xzp"}, {"Xdq"});
  *graph->add_node() =
      MakeNode("DequantizeLinear", {"Wq", "Ws"}, {"Wdq"}, {{"axis", 1}});
  *graph->add_node() = MakeNode("MatMul", {"Xdq", "Wdq"}, {"Y"});
  *graph->add_initializer() =
      RawTensor("Wq", onnx::TensorProto::INT8, {kK, kN}, std::string(12, '\0'));
  *graph->add_initializer() =
      FloatTensor("Ws", {kN}, std::vector<float>(kN, 0.01f));
  *graph->add_initializer() = FloatTensor("Xs", {}, {0.25f});
  *graph->add_initializer() =
      RawTensor("Xzp", onnx::TensorProto::UINT8, {}, std::string(1, '\x80'));
  Finish(&model);
  return model;
}

// A deeper block: two quantized MatMuls with a Relu between them and a
// residual arriving sideways, so the slice has two trained layers and two
// external tensors. The hidden width is 4 so that both weights have an even
// number of codes, which is what INT4's two-to-a-byte packing assumes.
onnx::ModelProto DeepFloatModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("deep_float");
  AddBatchedInput(graph, "X", kK);
  AddBatchedInput(graph, "R", kN);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("MatMul", {"X", "W1"}, {"H"});
  *graph->add_node() = MakeNode("Relu", {"H"}, {"A"});
  *graph->add_node() = MakeNode("MatMul", {"A", "W2"}, {"B"});
  *graph->add_node() = MakeNode("Add", {"B", "R"}, {"Y"});
  *graph->add_initializer() =
      FloatTensor("W1", {kK, kK}, std::vector<float>(16, 0.21f));
  *graph->add_initializer() = FloatTensor("W2", {kK, kN}, FloatWeight());
  Finish(&model);
  return model;
}

onnx::ModelProto DeepInt4QuantizedModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("deep_int4");
  AddBatchedInput(graph, "X", kK);
  AddBatchedInput(graph, "R", kN);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("DequantizeLinear", {"W1q", "W1s"}, {"W1dq"},
                                {{"axis", 0}, {"block_size", kBlock}});
  *graph->add_node() = MakeNode("MatMul", {"X", "W1dq"}, {"H"});
  *graph->add_node() = MakeNode("Relu", {"H"}, {"A"});
  *graph->add_node() = MakeNode("DequantizeLinear", {"W2q", "W2s"}, {"W2dq"},
                                {{"axis", 0}, {"block_size", kK}});
  *graph->add_node() = MakeNode("MatMul", {"A", "W2dq"}, {"B"});
  *graph->add_node() = MakeNode("Add", {"B", "R"}, {"Y"});
  *graph->add_initializer() =
      RawTensor("W1q", onnx::TensorProto::INT4, {kK, kK}, std::string(8, '\0'));
  *graph->add_initializer() =
      FloatTensor("W1s", {kK / kBlock, kK}, std::vector<float>(8, 0.1f));
  *graph->add_initializer() =
      RawTensor("W2q", onnx::TensorProto::INT4, {kK, kN}, std::string(6, '\0'));
  *graph->add_initializer() =
      FloatTensor("W2s", {1, kN}, std::vector<float>(kN, 0.05f));
  Finish(&model);
  return model;
}

// The fine-tuning pair -- QatOptions::fake_quant off. Nothing here is
// quantized: this scheme trains the *student's* own float weights against the
// teacher's activation, so the two models being different weights over one
// topology is the whole thing there is to learn. Two MatMuls make the block
// two trained layers; the LayerNormalization between them carries the
// untrained constants (its scale and bias) whose source is the one
// substantive decision this scheme makes, so the two models give them
// different values too.
onnx::ModelProto FineTuneModel(float weight, float norm) {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("finetune");
  AddBatchedInput(graph, "X", kK);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("MatMul", {"X", "W1"}, {"H"});
  *graph->add_node() = MakeNode("LayerNormalization", {"H", "LnS", "LnB"},
                                {"Nrm"}, {{"axis", -1}});
  *graph->add_node() = MakeNode("MatMul", {"Nrm", "W2"}, {"Y"});
  *graph->add_initializer() =
      FloatTensor("W1", {kK, kK}, std::vector<float>(kK * kK, weight));
  *graph->add_initializer() =
      FloatTensor("LnS", {kK}, std::vector<float>(kK, norm));
  *graph->add_initializer() =
      FloatTensor("LnB", {kK}, std::vector<float>(kK, -norm));
  *graph->add_initializer() =
      FloatTensor("W2", {kK, kN}, std::vector<float>(kK * kN, weight));
  Finish(&model);
  return model;
}

// The teacher's weights, which fine-tuning must never read: it trains the
// student, and re-seeding from the teacher would throw away whatever change
// (a pruning, a simplification, an earlier tuning) made the two differ.
constexpr float kTeacherWeight = 0.25f;
constexpr float kTeacherNorm = 1.5f;
constexpr float kStudentWeight = 0.5f;
constexpr float kStudentNorm = 2.5f;

std::set<std::string> InputNames(const onnx::ModelProto& model) {
  std::set<std::string> names;
  for (const onnx::ValueInfoProto& v : model.graph().input()) {
    names.insert(v.name());
  }
  return names;
}

std::set<std::string> OpTypes(const onnx::ModelProto& model) {
  std::set<std::string> types;
  for (const onnx::NodeProto& node : model.graph().node()) {
    types.insert(node.op_type());
  }
  return types;
}

const onnx::TensorProto* FindInitializer(const onnx::ModelProto& model,
                                         const std::string& name) {
  for (const onnx::TensorProto& t : model.graph().initializer()) {
    if (t.name() == name) return &t;
  }
  return nullptr;
}

std::map<std::string, onnx::TensorProto> AsStateMap(
    const std::vector<onnx::TensorProto>& tensors) {
  std::map<std::string, onnx::TensorProto> state;
  for (const onnx::TensorProto& t : tensors) state[t.name()] = t;
  return state;
}

void CheckModel(const onnx::ModelProto& model, const std::string& what) {
  try {
    onnx::checker::check_model(model);
  } catch (const std::exception& error) {
    Check(false, what + " -- onnx::checker rejected it: " + error.what());
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// If this fails, the browser would hand onnxruntime-web a graph it refuses to
// load and QAT would be unreachable there entirely -- the one failure that
// makes every other property in this file moot.
void AnInt4MatMulBlockProducesAStepGraphTheCheckerAccepts() {
  const QatStepPlan plan = BuildQatStepGraph(FloatModel(), Int4QuantizedModel(),
                                             "X", "Y", kRows, QatOptions());
  CheckModel(plan.step_graph, "the INT4 step graph");
  CheckEqual(plan.step_graph.graph().name(), "onnxsim_qat_step",
             "the step graph carries qat.py's own graph name");
  CheckEqual(static_cast<int64_t>(plan.layers.size()), 1,
             "the block's one INT4 MatMul is the one trained layer");
  CheckEqual(plan.layers[0].codes_initializer, "Wq",
             "the trained layer writes back into the quantized model's codes");
  CheckEqual(plan.layers[0].block_size, kBlock,
             "the block size comes from the DequantizeLinear attribute");
  CheckEqual(plan.layers[0].block_axis, 0,
             "the blocked axis comes from the DequantizeLinear attribute");
  Check(plan.layers[0].packed_int4,
        "INT4 codes are flagged as packed two to a byte");
  Check(!plan.loss_name.empty(), "the step graph reports a loss");
  CheckEqual(plan.num_rows, kRows, "the plan carries the bound row count");
  Check(plan.row_index_input.empty(),
        "a full-batch run has no minibatch row index");

  // Three state tensors per layer with learn_scales off: the master weight
  // and Adam's two moments.
  CheckEqual(static_cast<int64_t>(plan.state.size()), 3,
             "an untrained-scale layer carries exactly w, m and v");
  CheckEqual(plan.state[0].first, "qat__w0", "the master weight is state 0");
  CheckEqual(plan.state[1].first, "qat__mw0", "Adam's first moment is state 1");
  CheckEqual(plan.state[2].first, "qat__vw0",
             "Adam's second moment is state 2");
  CheckEqual(static_cast<int64_t>(plan.scalars.size()), 3,
             "a weight-only run feeds one learning rate and two corrections");
  CheckEqual(plan.scalars[0], "qat__lr", "the weight learning rate is first");

  const std::set<std::string> inputs = InputNames(plan.step_graph);
  for (const std::string& name :
       {"X", "qat__teacher", "qat__w0", "qat__mw0", "qat__vw0", "qat__lr",
        "m_correction", "v_correction"}) {
    Check(inputs.count(name) != 0, "the step graph declares the input " + name);
  }
}

// The float weight is a computed value in the step graph rather than an
// initializer -- that substitution is what makes the block's own nodes
// copyable verbatim. If it regressed, the graph would train a constant.
void TheTrainedWeightBecomesAComputedTensorRatherThanAnInitializer() {
  const QatStepPlan plan = BuildQatStepGraph(FloatModel(), Int4QuantizedModel(),
                                             "X", "Y", kRows, QatOptions());
  Check(FindInitializer(plan.step_graph, "W") == nullptr,
        "the trained weight is not an initializer of the step graph");
  bool produced = false;
  for (const onnx::NodeProto& node : plan.step_graph.graph().node()) {
    for (const std::string& out : node.output()) {
      if (out == "W") produced = true;
    }
  }
  Check(produced,
        "the fake-quant writes the weight under the block's own name");
}

// Every op the step graph carries has to run wherever the caller runs it. The
// block's own nodes are outside qat.py's allowlist by design, but this block
// is a MatMul, so the whole graph must land inside it -- a Round or a Where
// creeping in would silently exclude the WebNN/NPU backends this exists for.
void EveryOpTheStepGraphEmitsIsEpFriendly() {
  const QatStepPlan plan = BuildQatStepGraph(FloatModel(), Int4QuantizedModel(),
                                             "X", "Y", kRows, QatOptions());
  for (const std::string& op : OpTypes(plan.step_graph)) {
    Check(EpFriendlyOps().count(op) != 0,
          "the step graph emits " + op + ", which is not EP-friendly");
  }
}

// The captures are the caller's whole contract with the float model: get
// these tensors, bind them here. A wrong name or a wrong shape trains on
// uninitialized memory rather than failing.
void CapturesNameTheBlocksExternalTensorsAndTheTeacher() {
  const QatStepPlan plan = BuildQatStepGraph(FloatModel(), Int4QuantizedModel(),
                                             "X", "Y", kRows, QatOptions());
  CheckEqual(static_cast<int64_t>(plan.captures.size()), 2,
             "the block reads one external tensor plus its teacher");
  CheckEqual(plan.captures[0].source_tensor, "X",
             "the block's own input is captured from the float model");
  CheckEqual(plan.captures[0].step_graph_input, "X",
             "a full-batch run binds it under its own name");
  Check(!plan.captures[0].is_teacher, "the block input is not the teacher");
  Check(plan.captures[0].dims == std::vector<int64_t>({kRows, kK}),
        "the captured input is num_rows rows of the block input's width");
  CheckEqual(plan.captures[1].source_tensor, "Y",
             "the teacher is the float model's own output for this block");
  CheckEqual(plan.captures[1].step_graph_input, "qat__teacher",
             "the teacher is bound under the private name the loss reads");
  Check(plan.captures[1].is_teacher, "the reconstruction target is flagged");
  Check(plan.captures[1].dims == std::vector<int64_t>({kRows, kN}),
        "the teacher has the block output's shape at num_rows rows");
}

// Step 0 of the loop has to reproduce round-to-nearest exactly, so the master
// weight starts at the *float* model's weight and the moments start at zero.
// Seeding from the quantized weight instead would make every later step an
// improvement on an arbitrary re-initialization rather than on the shipped
// model.
void InitialStateSeedsTheMasterWeightFromTheFloatModelAndZeroesTheMoments() {
  const QatStepPlan plan = BuildQatStepGraph(FloatModel(), Int4QuantizedModel(),
                                             "X", "Y", kRows, QatOptions());
  const std::map<std::string, onnx::TensorProto> state =
      AsStateMap(plan.initial_state);
  CheckEqual(static_cast<int64_t>(plan.initial_state.size()), 3,
             "one initial value per state input");
  const auto w = state.find("qat__w0");
  Check(w != state.end(), "the master weight is seeded");
  if (w != state.end()) {
    Check(FloatsOf(w->second) == FloatWeight(),
          "the master weight starts at the float model's own weight");
    Check(std::vector<int64_t>(w->second.dims().begin(),
                               w->second.dims().end()) ==
              std::vector<int64_t>({kK, kN}),
          "the master weight keeps the weight's storage layout");
  }
  for (const std::string& moment : {"qat__mw0", "qat__vw0"}) {
    const auto it = state.find(moment);
    Check(it != state.end(), "Adam's moment " + moment + " is seeded");
    if (it == state.end()) continue;
    bool all_zero = true;
    for (float v : FloatsOf(it->second)) {
      if (v != 0.0f) all_zero = false;
    }
    Check(all_zero, "Adam's moment " + moment + " starts at zero");
  }
}

// Feeding the warm start straight back must reproduce round-to-nearest of the
// float weights, packed low nibble first. That is the round trip the whole
// design rests on: it is what makes "step 0 == the shipped model" true, and a
// packing or rounding slip here would ship a model whose loss nobody measured.
void WriteBackQatStateReproducesRoundToNearestFromTheInitialState() {
  const onnx::ModelProto quantized = Int4QuantizedModel();
  const QatStepPlan plan =
      BuildQatStepGraph(FloatModel(), quantized, "X", "Y", kRows, QatOptions());
  const onnx::ModelProto tuned =
      WriteBackQatState(quantized, plan, AsStateMap(plan.initial_state));

  const onnx::TensorProto* codes = FindInitializer(tuned, "Wq");
  Check(codes != nullptr, "the codes initializer survives the write-back");
  if (codes == nullptr) return;
  CheckEqual(static_cast<int64_t>(codes->data_type()),
             static_cast<int64_t>(onnx::TensorProto::INT4),
             "the codes keep their INT4 dtype");
  std::string expected;
  for (size_t i = 0; i + 1 < ExpectedCodes().size(); i += 2) {
    const unsigned lo = static_cast<unsigned>(ExpectedCodes()[i]) & 0xFu;
    const unsigned hi = static_cast<unsigned>(ExpectedCodes()[i + 1]) & 0xFu;
    expected.push_back(static_cast<char>(lo | (hi << 4)));
  }
  Check(codes->raw_data() == expected,
        "the written-back codes are round-to-nearest of the float weights, "
        "packed two to a byte");

  // learn_scales was off, so the scale initializer must come through
  // byte-identical -- the weight-only path's own guarantee.
  const onnx::TensorProto* before = FindInitializer(quantized, "Ws");
  const onnx::TensorProto* after = FindInitializer(tuned, "Ws");
  Check(before != nullptr && after != nullptr, "the scale initializer exists");
  if (before != nullptr && after != nullptr) {
    Check(before->SerializeAsString() == after->SerializeAsString(),
          "an untrained scale is left byte-identical");
  }
}

// With learn_scales the scale stops being a baked-in constant and becomes
// three more state tensors plus its own learning rate. If the scalar were
// missing the loop would feed the graph an input it does not have.
void LearnScalesAddsTheScaleStateAndItsOwnLearningRate() {
  QatOptions options;
  options.learn_scales = true;
  const QatStepPlan plan = BuildQatStepGraph(FloatModel(), Int4QuantizedModel(),
                                             "X", "Y", kRows, options);
  CheckModel(plan.step_graph, "the learn_scales step graph");
  CheckEqual(static_cast<int64_t>(plan.state.size()), 6,
             "w, m, v plus the scale and its own two moments");
  CheckEqual(plan.state[3].first, "qat__s0", "the scale is state 3");
  CheckEqual(static_cast<int64_t>(plan.scalars.size()), 4,
             "the scale's learning rate joins the per-step scalars");
  CheckEqual(plan.scalars[3], "qat__lr_scale",
             "the scale's learning rate is fed last");
  CheckEqual(plan.layers[0].weight_scale_state_input, "qat__s0",
             "the write-back reads the trained scale out of the loop state");
  CheckEqual(plan.layers[0].weight_scale_initializer, "Ws",
             "the trained scale writes back into the model's own scale");

  // Feeding the warm start back must leave the scale where calibration put
  // it: the same round trip the codes get.
  const onnx::ModelProto quantized = Int4QuantizedModel();
  const onnx::ModelProto tuned =
      WriteBackQatState(quantized, plan, AsStateMap(plan.initial_state));
  const onnx::TensorProto* after = FindInitializer(tuned, "Ws");
  Check(after != nullptr, "the scale initializer survives the write-back");
  if (after != nullptr) {
    const std::vector<float> values = FloatsOf(*after);
    bool unchanged = values.size() == 6;
    for (float v : values) {
      if (v != 0.1f) unchanged = false;
    }
    Check(unchanged, "a scale fed back unchanged is written back unchanged");
  }
}

// The whole calibration set stays resident and the step gathers its own rows
// out of it. If the captures still named the block's own tensors the caller
// would bind batch-sized buffers to num_rows-sized inputs.
void AMinibatchedBlockGathersItsRowsOutOfResidentTables() {
  QatOptions options;
  options.batch_size = 2;
  const QatStepPlan plan = BuildQatStepGraph(FloatModel(), Int4QuantizedModel(),
                                             "X", "Y", kRows, options);
  CheckModel(plan.step_graph, "the minibatched step graph");
  CheckEqual(plan.row_index_input, "qat__rows",
             "the row index is the documented per-step input");
  CheckEqual(plan.row_index_size, 2, "the row index holds batch_size rows");
  CheckEqual(plan.captures[0].step_graph_input, "qat__all_X",
             "the block's input is bound to its resident table instead");
  CheckEqual(plan.captures[0].source_tensor, "X",
             "the table is still filled from the float model's own tensor");
  Check(plan.captures[0].dims == std::vector<int64_t>({kRows, kK}),
        "the table holds the whole set, not one batch");
  CheckEqual(plan.captures[1].step_graph_input, "qat__teacher_all",
             "the teacher gets its own resident table");
  Check(OpTypes(plan.step_graph).count("Gather") != 0,
        "the step graph gathers its rows rather than being rebuilt per step");

  // A batch at least as large as the set *is* the full-batch objective, so it
  // takes the full-batch path rather than wrapping the index stream around
  // and quietly reweighting the repeated rows.
  QatOptions full = options;
  full.batch_size = kRows;
  const QatStepPlan whole = BuildQatStepGraph(
      FloatModel(), Int4QuantizedModel(), "X", "Y", kRows, full);
  Check(whole.row_index_input.empty(),
        "a batch covering every row takes the full-batch path");
}

// The QDQ scheme's layers train their activation quantizer's (scale,
// zero_point) alongside the weight. Both must reach the loop as state and
// both must reach the model on write-back; a quantizer trained but never
// written back would be pure wasted compute.
void TrainingActivationQuantizersPlansBothOfTheirParameters() {
  QatOptions options;
  options.learn_activation_scales = true;
  const QatStepPlan plan = BuildQatStepGraph(
      FloatModel(), StaticQdqQuantizedModel(), "X", "Y", kRows, options);
  CheckModel(plan.step_graph, "the activation-training step graph");
  CheckEqual(static_cast<int64_t>(plan.state.size()), 9,
             "w, m, v plus the quantizer's two parameters and four moments");
  CheckEqual(plan.layers[0].log_act_scale_state_input, "qat__as0",
             "the activation scale is carried in log space");
  CheckEqual(plan.layers[0].act_zero_point_state_input, "qat__az0",
             "the zero-point is carried as a continuous state tensor");
  CheckEqual(plan.layers[0].act_scale_initializer, "Xs",
             "it writes back into quantize_static's own scale initializer");
  CheckEqual(plan.layers[0].act_zero_point_initializer, "Xzp",
             "it writes back into quantize_static's own zero-point");
  CheckEqual(plan.layers[0].block_size, kK,
             "a per-output-channel scale is a block spanning the reduction");
  CheckEqual(static_cast<int64_t>(plan.layers[0].code_max), 127,
             "the INT8 grid is used for quantize_static's weights");
  CheckEqual(plan.scalars[3], "qat__lr_act",
             "the activation learning rate joins the per-step scalars");
  Check(OpTypes(plan.step_graph).count("Exp") != 0,
        "the scale is read as exp(log_scale) inside the differentiated chain");

  // The warm start is what calibration chose, so writing it straight back
  // must reproduce the shipped quantizer exactly.
  const onnx::ModelProto quantized = StaticQdqQuantizedModel();
  const onnx::ModelProto tuned =
      WriteBackQatState(quantized, plan, AsStateMap(plan.initial_state));
  const onnx::TensorProto* scale = FindInitializer(tuned, "Xs");
  Check(scale != nullptr && !FloatsOf(*scale).empty(),
        "the activation scale survives the write-back");
  if (scale != nullptr && !FloatsOf(*scale).empty()) {
    Check(std::abs(FloatsOf(*scale)[0] - 0.25f) < 1e-6f,
          "exp(log(s)) returns the calibrated activation scale");
  }
  const onnx::TensorProto* zp = FindInitializer(tuned, "Xzp");
  Check(zp != nullptr && zp->raw_data().size() == 1,
        "the zero-point survives the write-back as one uint8");
  if (zp != nullptr && zp->raw_data().size() == 1) {
    CheckEqual(
        static_cast<int64_t>(static_cast<unsigned char>(zp->raw_data()[0])),
        128, "the calibrated zero-point is written back unchanged");
  }
}

// A block is not one layer: two trained layers must be numbered in the order
// the finder found them, and a tensor entering the block sideways must be
// captured too. If the residual were missed the block would read an undefined
// tensor; if the externals were not sorted the caller would bind them to the
// wrong inputs, since qat.py orders the step graph's constants that way.
void ADeeperBlockTrainsEveryLayerAndCapturesTheResidual() {
  const QatStepPlan plan =
      BuildQatStepGraph(DeepFloatModel(), DeepInt4QuantizedModel(), "X", "Y",
                        kRows, QatOptions());
  CheckModel(plan.step_graph, "the two-layer step graph");
  CheckEqual(static_cast<int64_t>(plan.layers.size()), 2,
             "both quantized MatMuls in the block are trained");
  CheckEqual(plan.layers[0].codes_initializer, "W1q",
             "the first layer is the one the finder saw first");
  CheckEqual(plan.layers[1].codes_initializer, "W2q",
             "the second layer follows it");
  CheckEqual(plan.layers[0].weight_state_input, "qat__w0",
             "the layers are numbered from zero in that order");
  CheckEqual(plan.layers[1].weight_state_input, "qat__w1",
             "the second layer gets the next index");
  CheckEqual(static_cast<int64_t>(plan.state.size()), 6,
             "three state tensors per trained layer");

  CheckEqual(static_cast<int64_t>(plan.captures.size()), 3,
             "the block input, the sideways residual and the teacher");
  CheckEqual(plan.captures[0].source_tensor, "R",
             "the externals are captured in sorted order, R first");
  CheckEqual(plan.captures[1].source_tensor, "X",
             "then the block input itself");
  Check(plan.captures[2].is_teacher, "the teacher is captured last");
  Check(plan.captures[0].dims == std::vector<int64_t>({kRows, kN}),
        "the residual is captured at the block output's width");

  // A per-output-channel scale (one block spanning the whole reduction) and a
  // 32-element-style blocked scale both have to work in one graph.
  CheckEqual(plan.layers[0].block_size, kBlock, "the first scale is blocked");
  CheckEqual(plan.layers[1].block_size, kK,
             "the second scale spans the whole reduction axis");
  // Not asserted here: that every op is EP-friendly. The block's own Relu is
  // copied in verbatim and the allowlist deliberately says nothing about the
  // block -- see EveryOpTheStepGraphEmitsIsEpFriendly, which uses a block
  // whose ops happen to be inside it.
}

// fake_quant off is the same loop with the quantizer taken out of the middle:
// the block reads the master weight itself, so there is nothing to
// fake-quantize and nothing for a straight-through estimator to pass through.
// The operators that make up the weight fake-quant are therefore the ones that
// must be *absent* -- a port that emitted them anyway would round the trained
// weights to a grid nobody asked for and train on quietly.
void AFloatBlockFineTunesWithNoQuantizerInTheStepGraph() {
  QatOptions options;
  options.fake_quant = false;
  const QatStepPlan plan = BuildQatStepGraph(
      FineTuneModel(kTeacherWeight, kTeacherNorm),
      FineTuneModel(kStudentWeight, kStudentNorm), "X", "Y", kRows, options);
  CheckModel(plan.step_graph, "the fine-tuning step graph");
  CheckEqual(static_cast<int64_t>(plan.layers.size()), 2,
             "both of the student's float MatMuls are trained");
  CheckEqual(plan.layers[0].codes_initializer, "W1",
             "a fine-tuned layer writes back into the weight itself");
  CheckEqual(plan.layers[1].codes_initializer, "W2",
             "the second layer follows it in graph order");
  Check(!plan.layers[0].fake_quant,
        "the write-back is told there is no quantizer to invert");
  CheckEqual(plan.layers[0].block_size, 0,
             "the unread block size is the value that breaks rather than the "
             "one that looks plausible");
  CheckEqual(plan.layers[0].weight_scale_initializer, "",
             "there is no scale initializer to write back into");

  // Three state tensors per layer and one learning rate, exactly as the
  // weight-only quantized path: the optimizer half is untouched.
  CheckEqual(static_cast<int64_t>(plan.state.size()), 6,
             "w, m and v for each of the two trained weights");
  CheckEqual(plan.state[0].first, "qat__w0", "the master weight is state 0");
  CheckEqual(plan.layers[1].weight_state_input, "qat__w1",
             "the layers are numbered from zero in the order they were found");
  CheckEqual(static_cast<int64_t>(plan.scalars.size()), 3,
             "one learning rate and Adam's two bias corrections");
  CheckEqual(static_cast<int64_t>(plan.captures.size()), 2,
             "the block's input and its teacher");
  CheckEqual(plan.captures[0].source_tensor, "X",
             "the block input is captured from the float model");
  CheckEqual(plan.captures[1].step_graph_input, "qat__teacher",
             "the teacher is bound under the private name the loss reads");
  Check(plan.captures[1].is_teacher, "the reconstruction target is flagged");

  const std::set<std::string> ops = OpTypes(plan.step_graph);
  for (const std::string& op : {"Sign", "Abs", "Clip", "Round"}) {
    Check(ops.count(op) == 0,
          "the fine-tuning step graph emits " + op +
              ", which only a weight fake-quant would need");
  }

  // The block's node reads the master weight by *name*: the substitution is a
  // renamed input rather than an Identity, since a node whose whole job is to
  // copy a tensor is a node an execution provider would have to implement for
  // no reason.
  Check(FindInitializer(plan.step_graph, "W1") == nullptr,
        "the trained weight is not an initializer of the step graph");
  bool reads_master = false;
  for (const onnx::NodeProto& node : plan.step_graph.graph().node()) {
    if (node.op_type() == "MatMul" && node.input_size() > 1 &&
        node.input(1) == "qat__w0" && node.output(0) == "H") {
      reads_master = true;
    }
  }
  Check(reads_master, "the block's own MatMul reads the master weight");
  Check(ops.count("Identity") == 0,
        "the substitution renames an input rather than emitting an Identity");
}

// Which model the block's *untrained* constants come from is a semantic
// decision, not a detail. Under QAT it is the teacher, whose weights the
// student encodes; under fine-tuning it is the student, because the student is
// a different model and quietly substituting the teacher's constants into it
// would train the block to compensate for a substitution the deployed model
// does not make.
void UntrainedBlockConstantsComeFromTheStudentWhenFineTuning() {
  QatOptions options;
  options.fake_quant = false;
  const QatStepPlan plan = BuildQatStepGraph(
      FineTuneModel(kTeacherWeight, kTeacherNorm),
      FineTuneModel(kStudentWeight, kStudentNorm), "X", "Y", kRows, options);
  for (const auto& entry :
       {std::make_pair(std::string("LnS"), kStudentNorm),
        std::make_pair(std::string("LnB"), -kStudentNorm)}) {
    const onnx::TensorProto* constant =
        FindInitializer(plan.step_graph, entry.first);
    Check(constant != nullptr,
          "the LayerNorm's " + entry.first + " is a step-graph initializer");
    if (constant == nullptr) continue;
    const std::vector<float> values = FloatsOf(*constant);
    Check(values == std::vector<float>(kK, entry.second),
          "the LayerNorm's " + entry.first +
              " is the student's own, not the teacher's");
  }

  // And the master weight is seeded from the student too, for the same
  // reason: its weights are the starting point precisely by not being the
  // teacher's.
  const std::map<std::string, onnx::TensorProto> state =
      AsStateMap(plan.initial_state);
  const auto w = state.find("qat__w0");
  Check(w != state.end(), "the master weight is seeded");
  if (w != state.end()) {
    Check(FloatsOf(w->second) == std::vector<float>(kK * kK, kStudentWeight),
          "fine-tuning starts from the student's own weight");
  }
}

// With no quantizer between the master weight and what the model stores, the
// trained tensor *is* the stored tensor: the write-back is the identity that
// the two quantized schemes' rounding and clipping stand in for.
void FineTunedWeightsAreWrittenBackAsFloatUnderTheirOwnName() {
  QatOptions options;
  options.fake_quant = false;
  const onnx::ModelProto student = FineTuneModel(kStudentWeight, kStudentNorm);
  const QatStepPlan plan =
      BuildQatStepGraph(FineTuneModel(kTeacherWeight, kTeacherNorm), student,
                        "X", "Y", kRows, options);

  // A "trained" state that is not the warm start, so the write-back is
  // visible: the whole point is that these values reach the model unrounded.
  std::map<std::string, onnx::TensorProto> final_state =
      AsStateMap(plan.initial_state);
  final_state["qat__w0"] =
      FloatTensor("qat__w0", {kK, kK}, std::vector<float>(kK * kK, 0.1234f));
  const onnx::ModelProto tuned = WriteBackQatState(student, plan, final_state);

  const onnx::TensorProto* w1 = FindInitializer(tuned, "W1");
  Check(w1 != nullptr, "the weight initializer survives the write-back");
  if (w1 != nullptr) {
    CheckEqual(static_cast<int64_t>(w1->data_type()),
               static_cast<int64_t>(onnx::TensorProto::FLOAT),
               "a fine-tuned weight is written back as fp32");
    Check(std::vector<int64_t>(w1->dims().begin(), w1->dims().end()) ==
              std::vector<int64_t>({kK, kK}),
          "it keeps the weight's storage layout");
    Check(FloatsOf(*w1) == std::vector<float>(kK * kK, 0.1234f),
          "the trained tensor is stored verbatim -- no rounding, no grid");
  }
  // The second layer was fed its warm start, so it comes back as it went in.
  const onnx::TensorProto* w2 = FindInitializer(tuned, "W2");
  Check(w2 != nullptr &&
            FloatsOf(*w2) == std::vector<float>(kK * kN, kStudentWeight),
        "an untrained-away weight is written back unchanged");
  // Nothing else in the model was touched: there is no scale and no code
  // array for this scheme to rewrite.
  const onnx::TensorProto* norm_before = FindInitializer(student, "LnS");
  const onnx::TensorProto* norm_after = FindInitializer(tuned, "LnS");
  Check(norm_before != nullptr && norm_after != nullptr &&
            norm_before->SerializeAsString() == norm_after->SerializeAsString(),
        "the block's untrained constants are left byte-identical");
}

// The two scale flags each name a parameter of a quantizer, and fake_quant off
// is the mode with no quantizer in it. Ignoring them would be the worse
// failure of the two available: a caller who asked to learn scales and got a
// model whose scales are exactly as they were has no way to tell that from a
// run in which learning them did not help.
void TheScaleFlagsAreRefusedRatherThanIgnoredWithoutFakeQuant() {
  const onnx::ModelProto teacher = FineTuneModel(kTeacherWeight, kTeacherNorm);
  const onnx::ModelProto student = FineTuneModel(kStudentWeight, kStudentNorm);
  CheckThrows<std::invalid_argument>(
      [&] {
        QatOptions options;
        options.fake_quant = false;
        options.learn_scales = true;
        BuildQatStepGraph(teacher, student, "X", "Y", kRows, options);
      },
      "learn_scales cannot be used with fake_quant=False",
      "learn_scales without a fake-quant is refused");
  CheckThrows<std::invalid_argument>(
      [&] {
        QatOptions options;
        options.fake_quant = false;
        options.learn_activation_scales = true;
        BuildQatStepGraph(teacher, student, "X", "Y", kRows, options);
      },
      "learn_activation_scales cannot be used with fake_quant=False",
      "learn_activation_scales without a fake-quant is refused");
  CheckThrows<std::invalid_argument>(
      [&] {
        QatOptions options;
        options.fake_quant = false;
        options.learn_scales = true;
        options.learn_activation_scales = true;
        BuildQatStepGraph(teacher, student, "X", "Y", kRows, options);
      },
      "learn_scales and learn_activation_scales cannot be used",
      "both flags at once are named together");
}

// A block with nothing to fine-tune is refused in terms of *this* scheme: the
// caller's model has no MatMul/Gemm whose weight is a stored 2-D fp32 tensor,
// which is a different problem from having no quantized layer and reads as
// one.
void ABlockWithNoFloatWeightToFineTuneIsRefusedInThoseTerms() {
  QatOptions options;
  options.fake_quant = false;
  CheckThrows<std::invalid_argument>(
      [&] {
        // The INT4 student's MatMul reads a DequantizeLinear's output, not an
        // initializer, so there is no stored weight for the optimizer to hold.
        BuildQatStepGraph(FloatModel(), Int4QuantizedModel(), "X", "Y", kRows,
                          options);
      },
      "contains no MatMul/Gemm with a 2-D fp32 weight initializer to fine-tune",
      "a block with no stored float weight is refused in the scheme's terms");
}

// A node with no gradient rule must be refused by op type, before any of the
// expensive work -- and the message must name both the offender and the
// supported set, or the caller has to go read graph_grad to find out what to
// do about it.
void ABlockContainingAnUndifferentiableOpIsRefusedByOpType() {
  onnx::ModelProto float_model = FloatModel();
  // Sin has no VJP rule in graph_grad, and sits inside the slice.
  float_model.mutable_graph()->clear_node();
  *float_model.mutable_graph()->add_node() = MakeNode("Sin", {"X"}, {"Xs"});
  *float_model.mutable_graph()->add_node() =
      MakeNode("MatMul", {"Xs", "W"}, {"Y"});
  onnx::ModelProto quantized = Int4QuantizedModel();
  quantized.mutable_graph()->clear_node();
  *quantized.mutable_graph()->add_node() = MakeNode("Sin", {"X"}, {"Xs"});
  *quantized.mutable_graph()->add_node() =
      MakeNode("DequantizeLinear", {"Wq", "Ws"}, {"Wdq"},
               {{"axis", 0}, {"block_size", kBlock}});
  *quantized.mutable_graph()->add_node() =
      MakeNode("MatMul", {"Xs", "Wdq"}, {"Y"});

  CheckThrows<UnsupportedOpError>(
      [&] {
        BuildQatStepGraph(float_model, quantized, "X", "Y", kRows,
                          QatOptions());
      },
      "'Sin'", "an undifferentiable op is refused by name");
  CheckThrows<std::invalid_argument>(
      [&] {
        BuildQatStepGraph(float_model, quantized, "X", "Y", kRows,
                          QatOptions());
      },
      "Choose block boundaries that exclude those nodes",
      "the refusal says what to do about it");
}

// A block with nothing of the targeted scheme in it is an error, never a
// silently unchanged model -- apply_qat's own contract.
void ABlockWithNoQuantizedLayerIsRefused() {
  onnx::ModelProto unquantized = FloatModel();  // no DequantizeLinear at all
  CheckThrows<std::invalid_argument>(
      [&] {
        BuildQatStepGraph(FloatModel(), unquantized, "X", "Y", kRows,
                          QatOptions());
      },
      "contains no quantize_weight_only_int4-quantized MatMul/Gemm layer",
      "a block with no INT4 layer is refused");
}

// The two scheme mismatches are the refusals worth getting right: a caller
// who aimed the wrong flag at their model has made a scheme error, not a
// boundary error, and would otherwise go looking at their tensor names.
void EachSchemeMismatchIsNamedRatherThanReportedAsABoundaryError() {
  CheckThrows<std::invalid_argument>(
      [&] {
        QatOptions options;
        options.learn_activation_scales = true;
        BuildQatStepGraph(FloatModel(), Int4QuantizedModel(), "X", "Y", kRows,
                          options);
      },
      "a weight-only model has no activation quantizer anywhere in it to "
      "train",
      "activation training over a weight-only model names the scheme error");

  CheckThrows<std::invalid_argument>(
      [&] {
        BuildQatStepGraph(FloatModel(), StaticQdqQuantizedModel(), "X", "Y",
                          kRows, QatOptions());
      },
      "match onnxsim.quantize_static's QDQ scheme instead",
      "a weight-only run over a QDQ model names the other scheme");
}

// The block boundary has to mean what a caller expects: an output no node
// produces is not a block end, and an input that reaches nothing is not a
// block start.
void ABlockThatIsNotAClosedSliceIsRefused() {
  CheckThrows<std::invalid_argument>(
      [&] {
        BuildQatStepGraph(FloatModel(), Int4QuantizedModel(), "X", "X", kRows,
                          QatOptions());
      },
      "is not produced by any node in the float graph",
      "a block ending at a graph input is refused");
  CheckThrows<std::invalid_argument>(
      [&] {
        BuildQatStepGraph(FloatModel(), Int4QuantizedModel(), "Y", "Y", kRows,
                          QatOptions());
      },
      "no nodes lie between",
      "a block whose input is downstream of its output is refused");
}

}  // namespace

int main() {
  AnInt4MatMulBlockProducesAStepGraphTheCheckerAccepts();
  TheTrainedWeightBecomesAComputedTensorRatherThanAnInitializer();
  EveryOpTheStepGraphEmitsIsEpFriendly();
  CapturesNameTheBlocksExternalTensorsAndTheTeacher();
  InitialStateSeedsTheMasterWeightFromTheFloatModelAndZeroesTheMoments();
  WriteBackQatStateReproducesRoundToNearestFromTheInitialState();
  LearnScalesAddsTheScaleStateAndItsOwnLearningRate();
  AMinibatchedBlockGathersItsRowsOutOfResidentTables();
  TrainingActivationQuantizersPlansBothOfTheirParameters();
  ADeeperBlockTrainsEveryLayerAndCapturesTheResidual();
  AFloatBlockFineTunesWithNoQuantizerInTheStepGraph();
  UntrainedBlockConstantsComeFromTheStudentWhenFineTuning();
  FineTunedWeightsAreWrittenBackAsFloatUnderTheirOwnName();
  TheScaleFlagsAreRefusedRatherThanIgnoredWithoutFakeQuant();
  ABlockWithNoFloatWeightToFineTuneIsRefusedInThoseTerms();
  ABlockContainingAnUndifferentiableOpIsRefusedByOpType();
  ABlockWithNoQuantizedLayerIsRefused();
  EachSchemeMismatchIsNamedRatherThanReportedAsABoundaryError();
  ABlockThatIsNotAClosedSliceIsRefused();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d qat_entry check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all qat_entry tests passed\n");
  return 0;
}
