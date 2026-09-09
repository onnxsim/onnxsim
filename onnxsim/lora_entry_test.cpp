/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises lora_entry.{h,cpp} -- the C++ port of onnxsim/lora.py.
 *
 * tests/test_lora.py checks the Python the way training has to be checked:
 * it runs torch.autograd against the emitted backward graph and measures a
 * training loss falling. Nothing in this build evaluates an ONNX graph (see
 * qat_entry_test.cpp's own comment on why -- the wheel does not compile
 * ONNX Runtime, and the WASM build hands evaluation to onnxruntime-web at
 * run time), so this file covers what *is* checkable without a runtime:
 * that InjectLora's graph surgery produces a checker-valid model with the
 * right shapes/op-types and the base weight untouched; that
 * BuildLoraStepGraph's emitted step graph is checker-valid with the right
 * captures/state/scalars/initial-state contract; that FoldFrozenPrefixes'
 * own small constant evaluator computes the right numbers for a NF4-shaped
 * dequant chain (checked directly, by decoding the folded initializer this
 * produces -- a data check, not a graph-execution one); and that every
 * refusal fires with a message naming what went wrong.
 *
 * Plain asserts and a failure counter, like qat_entry_test.cpp and the
 * other *_test.cpp files here -- this repository vendors no gtest.
 */
#include "lora_entry.h"

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

void CheckNear(float got, float want, const std::string& what) {
  Check(std::fabs(got - want) < 1e-4f, what + " (got " + std::to_string(got) +
                                           ", want " + std::to_string(want) +
                                           ")");
}

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

onnx::TensorProto Int64Tensor(const std::string& name,
                              const std::vector<int64_t>& dims,
                              const std::vector<int64_t>& values) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(onnx::TensorProto::INT64);
  for (int64_t d : dims) t.add_dims(d);
  std::string raw;
  for (int64_t v : values) {
    uint64_t bits = static_cast<uint64_t>(v);
    for (int i = 0; i < 8; ++i)
      raw.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
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

onnx::NodeProto MakeIntsNode(const std::string& op_type,
                             const std::vector<std::string>& inputs,
                             const std::vector<std::string>& outputs,
                             const std::string& attr_name,
                             const std::vector<int64_t>& ints) {
  onnx::NodeProto node = MakeNode(op_type, inputs, outputs);
  onnx::AttributeProto* a = node.add_attribute();
  a->set_name(attr_name);
  a->set_type(onnx::AttributeProto::INTS);
  for (int64_t v : ints) a->add_ints(v);
  return node;
}

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

void Finish(onnx::ModelProto* model, int64_t opset = 17) {
  onnx::OperatorSetIdProto* opset_import = model->add_opset_import();
  opset_import->set_domain("");
  opset_import->set_version(opset);
  model->set_ir_version(10);
}

constexpr int64_t kK = 4;
constexpr int64_t kN = 3;

const std::vector<float>& FloatWeight() {
  static const std::vector<float> w = {0.03f,  -0.18f, 0.32f, 0.44f,
                                       -0.71f, 0.06f,  0.57f, -0.33f,
                                       0.91f,  -0.04f, 0.12f, 0.26f};
  return w;
}

// X -> MatMul(X, W) -> Y.
onnx::ModelProto MatMulModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("matmul");
  AddBatchedInput(graph, "X", kK);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("MatMul", {"X", "W"}, {"Y"});
  *graph->add_initializer() = FloatTensor("W", {kK, kN}, FloatWeight());
  Finish(&model);
  return model;
}

// X -> Gemm<transA=1>(X, Wg) -> Y, Wg stored [K, N] since transB defaults to 0.
onnx::ModelProto GemmTransAModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("gemm");
  onnx::ValueInfoProto* x = graph->add_input();
  x->set_name("X");
  onnx::TypeProto::Tensor* xt = x->mutable_type()->mutable_tensor_type();
  xt->set_elem_type(onnx::TensorProto::FLOAT);
  xt->mutable_shape()->add_dim()->set_dim_value(kK);
  xt->mutable_shape()->add_dim()->set_dim_param("batch");
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("Gemm", {"X", "Wg"}, {"Y"}, {{"transA", 1}});
  *graph->add_initializer() = FloatTensor("Wg", {kK, kN}, FloatWeight());
  Finish(&model);
  return model;
}

// Xc[1,3,4,4] -> Conv<kernel_shape=[1,1]>(Xc, Wc[5,3,1,1]) -> Yc[1,5,4,4].
onnx::ModelProto Conv1x1Model() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("conv");
  onnx::ValueInfoProto* xc = graph->add_input();
  xc->set_name("Xc");
  onnx::TypeProto::Tensor* xct = xc->mutable_type()->mutable_tensor_type();
  xct->set_elem_type(onnx::TensorProto::FLOAT);
  for (int64_t d : {1, 3, 4, 4})
    xct->mutable_shape()->add_dim()->set_dim_value(d);
  onnx::ValueInfoProto* yc = graph->add_output();
  yc->set_name("Yc");
  onnx::TypeProto::Tensor* yct = yc->mutable_type()->mutable_tensor_type();
  yct->set_elem_type(onnx::TensorProto::FLOAT);
  for (int64_t d : {1, 5, 4, 4})
    yct->mutable_shape()->add_dim()->set_dim_value(d);
  *graph->add_node() =
      MakeIntsNode("Conv", {"Xc", "Wc"}, {"Yc"}, "kernel_shape", {1, 1});
  std::vector<float> wc(5 * 3, 0.1f);
  *graph->add_initializer() = FloatTensor("Wc", {5, 3, 1, 1}, wc);
  Finish(&model);
  return model;
}

// Xc[1,3,4,4] -> Conv<kernel_shape=[3,3]>(Xc, Wc[5,3,3,3]) -> Yc[1,5,4,4].
// Not 1x1, so ineligible for injection.
onnx::ModelProto Conv3x3Model() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("conv3x3");
  onnx::ValueInfoProto* xc = graph->add_input();
  xc->set_name("Xc");
  onnx::TypeProto::Tensor* xct = xc->mutable_type()->mutable_tensor_type();
  xct->set_elem_type(onnx::TensorProto::FLOAT);
  for (int64_t d : {1, 3, 4, 4})
    xct->mutable_shape()->add_dim()->set_dim_value(d);
  onnx::ValueInfoProto* yc = graph->add_output();
  yc->set_name("Yc");
  onnx::TypeProto::Tensor* yct = yc->mutable_type()->mutable_tensor_type();
  yct->set_elem_type(onnx::TensorProto::FLOAT);
  for (int64_t d : {1, 5, 4, 4})
    yct->mutable_shape()->add_dim()->set_dim_value(d);
  onnx::NodeProto conv =
      MakeIntsNode("Conv", {"Xc", "Wc"}, {"Yc"}, "kernel_shape", {3, 3});
  onnx::AttributeProto* pads = conv.add_attribute();
  pads->set_name("pads");
  pads->set_type(onnx::AttributeProto::INTS);
  for (int i = 0; i < 4; ++i) pads->add_ints(1);
  *graph->add_node() = conv;
  std::vector<float> wc(5 * 3 * 3 * 3, 0.01f);
  *graph->add_initializer() = FloatTensor("Wc", {5, 3, 3, 3}, wc);
  Finish(&model);
  return model;
}

// X -> Elu(X) -> X2 -> MatMul(X2, W) -> Y. Elu has no graph_grad rule, so a
// block spanning it is refused by RefuseUnsupported once folding leaves it
// in place (it reads a genuine external, X, so it is never foldable).
onnx::ModelProto UndifferentiableModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("undiff");
  AddBatchedInput(graph, "X", kK);
  AddOutput(graph, "Y", kN);
  *graph->add_node() = MakeNode("Elu", {"X"}, {"X2"});
  *graph->add_node() = MakeNode("MatMul", {"X2", "W"}, {"Y"});
  *graph->add_initializer() = FloatTensor("W", {kK, kN}, FloatWeight());
  Finish(&model);
  return model;
}

// A hand-built stand-in for what onnxsim.lora.apply_qlora's composition
// produces (inject_lora, then onnxsim.nf4.quantize_weight_only_nf4 on the
// base weight): a MatMul reading a NF4-shaped dequant chain's output
// (Cast -> Gather -> Reshape -> Reshape -> Mul -> Reshape, exactly
// onnxsim/nf4.py's own recipe) added to a LoRA branch reading A/B directly.
// apply_qlora itself is not ported to C++ -- see lora_entry.h's top comment
// on why -- so this builds the shape that composition leaves behind
// directly, to exercise FoldFrozenPrefixes against it.
//
// codebook = [0, 10, 20, 30]; Wq (uint8, [2,2]) = codes {0,1,2,3}; scale =
// 0.1. So dq_out = codebook[Wq] * scale = {0,10,20,30} * 0.1 = {0,1,2,3},
// reshaped back to [2, 2] -- a small, hand-checkable value this test
// decodes directly out of the emitted step graph's own initializers.
onnx::ModelProto QloraLikeModel() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("qlora_like");
  AddBatchedInput(graph, "X", 2);
  AddOutput(graph, "Y", 2);

  *graph->add_node() = MakeNode("Cast", {"Wq"}, {"codes_i64"},
                                {{"to", onnx::TensorProto::INT64}});
  *graph->add_node() = MakeNode("Gather", {"codebook", "codes_i64"},
                                {"gathered"}, {{"axis", 0}});
  *graph->add_node() =
      MakeNode("Reshape", {"gathered", "blocked_shape"}, {"reshaped"});
  *graph->add_node() =
      MakeNode("Reshape", {"scale", "scale_shape"}, {"scale_reshaped"});
  *graph->add_node() =
      MakeNode("Mul", {"reshaped", "scale_reshaped"}, {"scaled"});
  *graph->add_node() =
      MakeNode("Reshape", {"scaled", "orig_shape"}, {"dq_out"});
  *graph->add_node() = MakeNode("MatMul", {"X", "dq_out"}, {"base_out"});
  *graph->add_node() = MakeNode("MatMul", {"X", "A"}, {"a_out"});
  *graph->add_node() = MakeNode("MatMul", {"a_out", "B"}, {"ab_out"});
  *graph->add_node() = MakeNode("Add", {"base_out", "ab_out"}, {"Y"});

  const std::string wq_bytes = {static_cast<char>(0), static_cast<char>(1),
                                static_cast<char>(2), static_cast<char>(3)};
  *graph->add_initializer() =
      RawTensor("Wq", onnx::TensorProto::UINT8, {2, 2}, wq_bytes);
  *graph->add_initializer() =
      FloatTensor("codebook", {4}, {0.0f, 10.0f, 20.0f, 30.0f});
  *graph->add_initializer() = Int64Tensor("blocked_shape", {2}, {2, 2});
  *graph->add_initializer() = FloatTensor("scale", {1}, {0.1f});
  *graph->add_initializer() = Int64Tensor("scale_shape", {2}, {1, 1});
  *graph->add_initializer() = Int64Tensor("orig_shape", {2}, {2, 2});
  *graph->add_initializer() =
      FloatTensor("A", {2, 2}, {0.5f, -0.5f, 0.25f, -0.25f});
  *graph->add_initializer() =
      FloatTensor("B", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
  Finish(&model);
  return model;
}

LoraAdapter QloraLikeAdapter() {
  LoraTarget t;
  t.weight_name = "W";  // not read by BuildLoraStepGraph/WriteBackLoraState
  t.node_output = "Y";
  t.op_type = "MatMul";
  t.lora_a_name = "A";
  t.lora_b_name = "B";
  t.rank = 2;
  t.has_alpha = false;
  LoraAdapter adapter;
  adapter.targets.push_back(t);
  return adapter;
}

// ---------------------------------------------------------------------------
// InjectLora
// ---------------------------------------------------------------------------

void InjectingAMatMulAddsATrainableBranchAndLeavesTheBaseWeightUntouched() {
  const onnx::ModelProto model = MatMulModel();
  InjectLoraOptions options;
  options.rank = 2;
  options.seed = 1;
  const LoraInjectionResult result = InjectLora(model, options);
  onnx::checker::check_model(result.model);

  CheckEqual(static_cast<int64_t>(result.adapter.targets.size()), 1,
             "one MatMul is eligible");
  const LoraTarget& target = result.adapter.targets[0];
  CheckEqual(target.weight_name, "W", "the target names the base weight");
  CheckEqual(target.op_type, "MatMul", "the target's op type");
  CheckEqual(target.rank, 2, "the target's rank");
  Check(!target.has_alpha, "no alpha was asked for");
  Check(!target.lora_a_name.empty() && !target.lora_b_name.empty(),
        "both adapter tensor names are non-empty");
  Check(target.lora_a_name != target.lora_b_name, "A and B get distinct names");

  const onnx::TensorProto* w_after = nullptr;
  const onnx::TensorProto* a = nullptr;
  const onnx::TensorProto* b = nullptr;
  for (const onnx::TensorProto& t : result.model.graph().initializer()) {
    if (t.name() == "W") w_after = &t;
    if (t.name() == target.lora_a_name) a = &t;
    if (t.name() == target.lora_b_name) b = &t;
  }
  Check(w_after != nullptr, "W is still an initializer");
  Check(FloatsOf(*w_after) == FloatWeight(), "W is byte-for-byte unchanged");

  Check(a != nullptr, "A was added");
  CheckEqual(static_cast<int64_t>(a->dims_size()), 2, "A's rank");
  CheckEqual(a->dims(0), kK, "A's rows == K");
  CheckEqual(a->dims(1), 2, "A's cols == rank");

  Check(b != nullptr, "B was added");
  CheckEqual(static_cast<int64_t>(b->dims_size()), 2, "B's rank");
  CheckEqual(b->dims(0), 2, "B's rows == rank");
  CheckEqual(b->dims(1), kN, "B's cols == N");
  for (float v : FloatsOf(*b)) CheckNear(v, 0.0f, "B starts at zero");

  bool found_add = false;
  bool found_base_matmul = false;
  for (const onnx::NodeProto& node : result.model.graph().node()) {
    if (node.op_type() == "Add" && node.output_size() == 1 &&
        node.output(0) == "Y") {
      found_add = true;
    }
    if (node.op_type() == "MatMul" && node.input_size() == 2 &&
        node.input(0) == "X" && node.input(1) == "W") {
      found_base_matmul = true;
      Check(node.output(0) != "Y",
            "the base MatMul's output was renamed away from Y");
    }
  }
  Check(found_add, "a closing Add produces Y");
  Check(found_base_matmul, "the base MatMul (X, W) is still present");
}

void InjectingAGemmWithTransAAddsATransposeAheadOfTheBranch() {
  const onnx::ModelProto model = GemmTransAModel();
  InjectLoraOptions options;
  options.rank = 2;
  options.seed = 0;
  const LoraInjectionResult result = InjectLora(model, options);
  onnx::checker::check_model(result.model);

  CheckEqual(static_cast<int64_t>(result.adapter.targets.size()), 1,
             "one Gemm is eligible");
  const LoraTarget& target = result.adapter.targets[0];
  CheckEqual(target.op_type, "Gemm", "the target's op type");

  bool found_transpose = false;
  for (const onnx::NodeProto& node : result.model.graph().node()) {
    if (node.op_type() == "Transpose") found_transpose = true;
  }
  Check(found_transpose, "transA=1 gets an extra branch-input Transpose");

  // transA=1, transB=0 (default): the branch computes in [K, N] layout, so
  // A is [K, rank] = [4, 2] even though X itself is stored [K, batch].
  for (const onnx::TensorProto& t : result.model.graph().initializer()) {
    if (t.name() == target.lora_a_name) {
      CheckEqual(t.dims(0), kK, "A's rows == K under transA=1");
      CheckEqual(t.dims(1), 2, "A's cols == rank");
    }
  }
}

void InjectingA1x1ConvUsesTheConvsOwnStridesOnTheFirstBranchConvOnly() {
  const onnx::ModelProto model = Conv1x1Model();
  InjectLoraOptions options;
  options.rank = 2;
  options.seed = 0;
  const LoraInjectionResult result = InjectLora(model, options);
  onnx::checker::check_model(result.model);

  CheckEqual(static_cast<int64_t>(result.adapter.targets.size()), 1,
             "the 1x1 Conv is eligible");
  const LoraTarget& target = result.adapter.targets[0];
  CheckEqual(target.op_type, "Conv", "the target's op type");

  const onnx::TensorProto* a = nullptr;
  const onnx::TensorProto* b = nullptr;
  for (const onnx::TensorProto& t : result.model.graph().initializer()) {
    if (t.name() == target.lora_a_name) a = &t;
    if (t.name() == target.lora_b_name) b = &t;
  }
  Check(a != nullptr && a->dims_size() == 4, "A is rank 4");
  if (a != nullptr && a->dims_size() == 4) {
    CheckEqual(a->dims(0), 2, "A's out_ch == rank");
    CheckEqual(a->dims(1), 3, "A's in_ch matches Wc's in_ch");
    CheckEqual(a->dims(2), 1, "A's kernel height");
    CheckEqual(a->dims(3), 1, "A's kernel width");
  }
  Check(b != nullptr && b->dims_size() == 4, "B is rank 4");
  if (b != nullptr && b->dims_size() == 4) {
    CheckEqual(b->dims(0), 5, "B's out_ch matches Wc's out_ch");
    CheckEqual(b->dims(1), 2, "B's in_ch == rank");
  }

  int conv_count = 0;
  for (const onnx::NodeProto& node : result.model.graph().node()) {
    if (node.op_type() != "Conv") continue;
    ++conv_count;
    bool has_strides = false;
    for (const onnx::AttributeProto& attr : node.attribute()) {
      if (attr.name() == "strides") has_strides = true;
    }
    if (node.input(1) == target.lora_b_name) {
      Check(!has_strides,
            "the second branch conv carries no explicit strides/pads/"
            "dilations, matching lora.py's own plain 1x1/stride-1 conv");
    }
  }
  CheckEqual(static_cast<int64_t>(conv_count), 3,
             "base conv + two branch convs");
}

void ANon1x1ConvIsNotInjected() {
  const onnx::ModelProto model = Conv3x3Model();
  const LoraInjectionResult result = InjectLora(model);
  CheckEqual(static_cast<int64_t>(result.adapter.targets.size()), 0,
             "a 3x3 Conv is not a candidate for a low-rank branch");
}

void RestrictTargetNamesLimitsInjectionToTheNamedWeights() {
  const onnx::ModelProto model = MatMulModel();
  InjectLoraOptions options;
  options.restrict_target_names = true;
  options.target_names = {"SomethingElse"};
  const LoraInjectionResult miss = InjectLora(model, options);
  CheckEqual(static_cast<int64_t>(miss.adapter.targets.size()), 0,
             "target_names excluding W injects nothing");

  options.target_names = {"W"};
  const LoraInjectionResult hit = InjectLora(model, options);
  CheckEqual(static_cast<int64_t>(hit.adapter.targets.size()), 1,
             "target_names including W injects it");
}

void AlphaAddsAScaleInitializerAndAMulNode() {
  const onnx::ModelProto model = MatMulModel();
  InjectLoraOptions options;
  options.rank = 2;
  options.has_alpha = true;
  options.alpha = 4.0f;
  const LoraInjectionResult result = InjectLora(model, options);
  onnx::checker::check_model(result.model);

  bool found_mul = false;
  float scale_value = 0.0f;
  for (const onnx::NodeProto& node : result.model.graph().node()) {
    if (node.op_type() != "Mul") continue;
    for (const onnx::TensorProto& t : result.model.graph().initializer()) {
      if (t.name() == node.input(1) || t.name() == node.input(0)) {
        const std::vector<float> values = FloatsOf(t);
        if (values.size() == 1) {
          found_mul = true;
          scale_value = values[0];
        }
      }
    }
  }
  Check(found_mul, "alpha adds a Mul-by-scale node ahead of the closing Add");
  CheckNear(scale_value, 4.0f / 2.0f, "the scale is alpha / rank");
}

// ---------------------------------------------------------------------------
// BuildLoraStepGraph
// ---------------------------------------------------------------------------

void AnInjectedMatMulProducesAStepGraphTheCheckerAccepts() {
  const onnx::ModelProto model = MatMulModel();
  InjectLoraOptions options;
  options.rank = 2;
  options.seed = 0;
  const LoraInjectionResult injected = InjectLora(model, options);
  const LoraTarget& target = injected.adapter.targets[0];

  const LoraStepPlan plan =
      BuildLoraStepGraph(injected.model, injected.adapter, "X", "Y", 5);
  onnx::checker::check_model(plan.step_graph);

  CheckEqual(static_cast<int64_t>(plan.parameters.size()), 2,
             "one adapter -> two trained tensors (A, B)");
  Check(plan.parameters[0] == target.lora_a_name &&
            plan.parameters[1] == target.lora_b_name,
        "parameters lists A then B, the injection order");

  CheckEqual(static_cast<int64_t>(plan.state.size()), 6,
             "3 state entries per trained tensor (weight, m, v) x 2 tensors");
  CheckEqual(static_cast<int64_t>(plan.initial_state.size()), 6,
             "one initial-state tensor per state entry");

  CheckEqual(plan.scalars.size(), size_t{3}, "lr, m_correction, v_correction");
  CheckEqual(plan.scalars[0], "lora__lr", "the learning-rate scalar's name");
  Check(!plan.loss_name.empty(), "the plan reports a loss output");

  CheckEqual(static_cast<int64_t>(plan.captures.size()), 2,
             "X and the teacher are captured");
  bool found_x = false, found_teacher = false;
  for (const LoraCapture& c : plan.captures) {
    if (c.source_tensor == "X" && !c.is_teacher) {
      found_x = true;
      CheckEqual(static_cast<int64_t>(c.dims.size()), 2, "X's rank");
      CheckEqual(c.dims[0], 5, "X's captured row count is num_rows");
      CheckEqual(c.dims[1], kK, "X's width");
    }
    if (c.source_tensor == "Y" && c.is_teacher) {
      found_teacher = true;
      CheckEqual(c.dims[0], 5, "the teacher's captured row count");
      CheckEqual(c.dims[1], kN, "the teacher's width");
    }
  }
  Check(found_x, "X is captured");
  Check(found_teacher, "Y (the block output) is captured as the teacher");

  CheckEqual(plan.row_index_input, "", "no minibatch was asked for");
  CheckEqual(plan.num_rows, 5, "num_rows is carried through");

  // graph.input == constants (X, teacher) + state (6) + scalars (3).
  CheckEqual(static_cast<int64_t>(plan.step_graph.graph().input_size()), 11,
             "the step graph's declared inputs");
  // graph.output == state next (6) + loss (1).
  CheckEqual(static_cast<int64_t>(plan.step_graph.graph().output_size()), 7,
             "the step graph's declared outputs");
}

void InitialStateSeedsTheAdapterFromItsCurrentValueAndZeroesTheMoments() {
  const onnx::ModelProto model = MatMulModel();
  InjectLoraOptions options;
  options.rank = 2;
  options.seed = 3;
  const LoraInjectionResult injected = InjectLora(model, options);
  const LoraTarget& target = injected.adapter.targets[0];

  const onnx::TensorProto* a_before = nullptr;
  for (const onnx::TensorProto& t : injected.model.graph().initializer()) {
    if (t.name() == target.lora_a_name) a_before = &t;
  }
  Check(a_before != nullptr, "A exists in the injected model");

  const LoraStepPlan plan =
      BuildLoraStepGraph(injected.model, injected.adapter, "X", "Y", 4);

  std::map<std::string, const onnx::TensorProto*> initial;
  for (const onnx::TensorProto& t : plan.initial_state) initial[t.name()] = &t;

  Check(initial.count(target.lora_a_name) != 0, "A has an initial-state entry");
  Check(initial.count(target.lora_b_name) != 0, "B has an initial-state entry");
  Check(initial.count("lora__m_" + target.lora_a_name) != 0,
        "A's first Adam moment has an initial-state entry");
  Check(initial.count("lora__v_" + target.lora_a_name) != 0,
        "A's second Adam moment has an initial-state entry");

  if (a_before != nullptr && initial.count(target.lora_a_name) != 0) {
    Check(FloatsOf(*a_before) == FloatsOf(*initial[target.lora_a_name]),
          "A's initial state matches its current (injected) value exactly");
  }
  for (float v : FloatsOf(*initial[target.lora_b_name])) {
    CheckNear(v, 0.0f, "B's initial state is still zero");
  }
  for (float v : FloatsOf(*initial["lora__m_" + target.lora_a_name])) {
    CheckNear(v, 0.0f, "Adam's first moment starts at zero");
  }
  for (float v : FloatsOf(*initial["lora__v_" + target.lora_b_name])) {
    CheckNear(v, 0.0f, "Adam's second moment starts at zero");
  }
}

void EveryOpTheStepGraphEmitsIsEpFriendly() {
  const onnx::ModelProto model = MatMulModel();
  const LoraInjectionResult injected = InjectLora(model);
  const LoraStepPlan plan =
      BuildLoraStepGraph(injected.model, injected.adapter, "X", "Y", 4);
  for (const onnx::NodeProto& node : plan.step_graph.graph().node()) {
    Check(EpFriendlyOps().count(node.op_type()) != 0,
          "node " + node.op_type() + " (-> " +
              (node.output_size() ? node.output(0) : "?") + ") is EP-friendly");
  }
}

void AMinibatchedBlockGathersItsRowsOutOfResidentTables() {
  const onnx::ModelProto model = MatMulModel();
  const LoraInjectionResult injected = InjectLora(model);

  LoraOptions options;
  options.batch_size = 2;
  const LoraStepPlan plan = BuildLoraStepGraph(injected.model, injected.adapter,
                                               "X", "Y", 5, options);
  onnx::checker::check_model(plan.step_graph);

  Check(!plan.row_index_input.empty(), "a minibatch row-index input exists");
  CheckEqual(plan.row_index_size, 2, "the row index carries batch_size rows");

  bool found_gather = false;
  for (const onnx::NodeProto& node : plan.step_graph.graph().node()) {
    if (node.op_type() == "Gather" && node.input(1) == plan.row_index_input) {
      found_gather = true;
    }
  }
  Check(found_gather,
        "a Gather selects this step's rows out of a resident table");

  bool found_table = false;
  for (const LoraCapture& c : plan.captures) {
    if (c.source_tensor == "X") {
      Check(c.step_graph_input != "X",
            "the captured table binds under a different name than the block "
            "reads (lora__all_X, not X)");
      found_table = true;
      CheckEqual(c.dims[0], 5, "the resident table holds every captured row");
    }
  }
  Check(found_table, "X is still captured under minibatching");
}

void ABlockContainingAnUndifferentiableOpIsRefusedByOpType() {
  const onnx::ModelProto model = UndifferentiableModel();
  const LoraInjectionResult injected = InjectLora(model);
  CheckThrows<UnsupportedOpError>(
      [&]() {
        BuildLoraStepGraph(injected.model, injected.adapter, "X", "Y", 4);
      },
      "Elu", "an Elu in the block is refused by name");
}

void AnAdapterWithNoTargetsIsRefused() {
  const onnx::ModelProto model = MatMulModel();
  const LoraAdapter empty;
  CheckThrows<std::invalid_argument>(
      [&]() { BuildLoraStepGraph(model, empty, "X", "Y", 4); },
      "no injected targets",
      "an adapter with no targets is refused before anything else runs");
}

void ANonPositiveNumRowsIsRefused() {
  const onnx::ModelProto model = MatMulModel();
  const LoraInjectionResult injected = InjectLora(model);
  CheckThrows<std::invalid_argument>(
      [&]() {
        BuildLoraStepGraph(injected.model, injected.adapter, "X", "Y", 0);
      },
      "num_rows", "num_rows must be at least 1");
}

// ---------------------------------------------------------------------------
// QLoRA composition: SliceBlock's own external-capture path, and
// FoldFrozenPrefixes/EvalNode directly (see BuildLoraStepGraph's own
// comment, and LoraFoldForTesting's, for why the latter needs its own
// entry point to be exercised at all).
// ---------------------------------------------------------------------------

// BuildLoraStepGraph's own comment records why this is the outcome: the NF4
// dequant chain (Cast included, which graph_grad cannot differentiate)
// never depends on the block's own input, so SliceBlock can never place it
// in the differentiated slice -- it is captured as an ordinary
// block-external constant instead, exactly like the block's own input is.
// This is what actually keeps a QLoRA-quantized base weight trainable
// end to end through this header, checked directly against the identical
// composition in lora.py (see that comment).
void TheFrozenNf4DequantChainIsCapturedRatherThanFolded() {
  const onnx::ModelProto model = QloraLikeModel();
  const LoraAdapter adapter = QloraLikeAdapter();

  const LoraStepPlan plan = BuildLoraStepGraph(model, adapter, "X", "Y", 3);
  onnx::checker::check_model(plan.step_graph);

  for (const onnx::NodeProto& node : plan.step_graph.graph().node()) {
    Check(node.op_type() != "Cast",
          "the NF4 chain's Cast (no graph_grad rule) never entered the "
          "differentiated slice in the first place");
  }

  bool found_a_branch = false, found_b_branch = false;
  for (const onnx::NodeProto& node : plan.step_graph.graph().node()) {
    if (node.op_type() == "MatMul" && node.input_size() == 2 &&
        node.input(1) == "A") {
      found_a_branch = true;
    }
    if (node.op_type() == "MatMul" && node.input_size() == 2 &&
        node.input(1) == "B") {
      found_b_branch = true;
    }
  }
  Check(found_a_branch,
        "the LoRA branch's MatMul(X, A) is still present verbatim");
  Check(found_b_branch,
        "the LoRA branch's MatMul(a_out, B) is still present verbatim");

  bool found_capture = false;
  bool found_input = false;
  for (const LoraCapture& c : plan.captures) {
    if (c.source_tensor != "dq_out") continue;
    found_capture = true;
    CheckEqual(c.step_graph_input, "dq_out",
               "dq_out is captured under its own name, not renamed");
    Check(!c.is_teacher, "dq_out is not the reconstruction target");
    CheckEqual(static_cast<int64_t>(c.dims.size()), 2, "dq_out's rank");
    if (c.dims.size() == 2) {
      CheckEqual(c.dims[0], 2, "dq_out's first dim, from orig_shape");
      CheckEqual(c.dims[1], 2, "dq_out's second dim, from orig_shape");
    }
  }
  Check(found_capture,
        "the frozen dequant chain's output is a capture the caller must bind "
        "-- by running the model once, since it is batch-independent -- "
        "exactly like the block's own input");

  for (const onnx::ValueInfoProto& input : plan.step_graph.graph().input()) {
    if (input.name() == "dq_out") found_input = true;
  }
  Check(found_input,
        "dq_out is declared as a step-graph *input*, not baked in as an "
        "initializer -- BuildLoraStepGraph never ran the model, so it has no "
        "value to bake in");
}

// EvalNode's own small arithmetic, exercised directly (see
// LoraFoldForTesting's own comment for why BuildLoraStepGraph's public API
// cannot reach this). Feeds it exactly the NF4-shaped dequant chain
// QloraLikeModel wires up, plus the two branch MatMuls a real
// apply_qlora + inject_lora composition would leave downstream of it, and
// checks both that only the dequant chain folds and that the folded
// value is the arithmetic QloraLikeModel's own comment predicts:
// codebook[Wq] * scale = {0,10,20,30} * 0.1 = {0,1,2,3}.
void LoraFoldForTestingComputesTheNf4ChainsArithmeticCorrectly() {
  const onnx::ModelProto model = QloraLikeModel();
  std::vector<onnx::NodeProto> nodes(model.graph().node().begin(),
                                     model.graph().node().end());
  Check(nodes.size() == 10, "QloraLikeModel's own node count (sanity check)");

  const LoraFoldResult folded = LoraFoldForTesting(nodes, model, {"A", "B"});

  CheckEqual(static_cast<int64_t>(folded.kept_nodes.size()), 4,
             "only the 4 nodes reading X, A or B survive the fold");
  for (const onnx::NodeProto& node : folded.kept_nodes) {
    Check(node.op_type() == "MatMul" || node.op_type() == "Add",
          "a kept node (" + node.op_type() + ") reads something non-constant");
  }

  const onnx::TensorProto* dq_out = nullptr;
  for (const onnx::TensorProto& t : folded.extra_initializers) {
    if (t.name() == "dq_out") dq_out = &t;
  }
  Check(dq_out != nullptr, "dq_out was folded into a plain initializer");
  if (dq_out != nullptr) {
    CheckEqual(static_cast<int64_t>(dq_out->data_type()),
               static_cast<int64_t>(onnx::TensorProto::FLOAT),
               "the folded tensor's dtype follows the codebook's (FLOAT)");
    const std::vector<float> values = FloatsOf(*dq_out);
    const std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f};
    Check(values.size() == expected.size(),
          "the folded tensor's element count");
    for (size_t i = 0; i < std::min(values.size(), expected.size()); ++i) {
      CheckNear(values[i], expected[i],
                "folded value " + std::to_string(i) +
                    " (codebook[Wq] * scale, computed by EvalNode's Cast/"
                    "Gather/Reshape/Mul)");
    }
  }
}

// The invariant FoldFrozenPrefixes exists to preserve, isolated from the
// NF4 example above: a node whose inputs *look* entirely constant (every
// one of them is a plain initializer) must still not be folded when one of
// those initializers is itself a training target -- it is state the step
// graph updates every step, not a value FoldFrozenPrefixes gets to
// evaluate once and bake in.
void LoraFoldForTestingNeverFoldsAnAdapterTensorEvenWhenEveryInputLooksConstant() {
  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("protects_targets");
  *graph->add_node() = MakeNode("Mul", {"A", "C"}, {"AC"});
  *graph->add_initializer() = FloatTensor("A", {2}, {1.0f, 2.0f});
  *graph->add_initializer() = FloatTensor("C", {2}, {3.0f, 4.0f});
  Finish(&model);

  const std::vector<onnx::NodeProto> nodes(model.graph().node().begin(),
                                           model.graph().node().end());
  const LoraFoldResult folded = LoraFoldForTesting(nodes, model, {"A"});

  CheckEqual(static_cast<int64_t>(folded.kept_nodes.size()), 1,
             "the Mul(A, C) node is not folded, because A is a target");
  CheckEqual(static_cast<int64_t>(folded.extra_initializers.size()), 0,
             "nothing was foldable, so nothing was materialized");
}

// ---------------------------------------------------------------------------
// WriteBackLoraState
// ---------------------------------------------------------------------------

void WriteBackReplacesOnlyTheAdapterTensorsByteForByte() {
  const onnx::ModelProto model = MatMulModel();
  InjectLoraOptions options;
  options.rank = 2;
  options.seed = 5;
  const LoraInjectionResult injected = InjectLora(model, options);
  const LoraTarget& target = injected.adapter.targets[0];

  const LoraStepPlan plan =
      BuildLoraStepGraph(injected.model, injected.adapter, "X", "Y", 4);

  std::map<std::string, onnx::TensorProto> final_state;
  for (const onnx::TensorProto& t : plan.initial_state) {
    final_state[t.name()] = t;  // most are untouched below
  }
  const std::vector<float> new_a(static_cast<size_t>(kK * 2), 9.5f);
  const std::vector<float> new_b(static_cast<size_t>(2 * kN), -3.25f);
  final_state[target.lora_a_name] =
      FloatTensor(target.lora_a_name, {kK, 2}, new_a);
  final_state[target.lora_b_name] =
      FloatTensor(target.lora_b_name, {2, kN}, new_b);

  const onnx::ModelProto tuned =
      WriteBackLoraState(injected.model, plan, final_state);
  onnx::checker::check_model(tuned);

  const onnx::TensorProto* w = nullptr;
  const onnx::TensorProto* a = nullptr;
  const onnx::TensorProto* b = nullptr;
  for (const onnx::TensorProto& t : tuned.graph().initializer()) {
    if (t.name() == "W") w = &t;
    if (t.name() == target.lora_a_name) a = &t;
    if (t.name() == target.lora_b_name) b = &t;
  }
  Check(w != nullptr && FloatsOf(*w) == FloatWeight(),
        "the base weight is untouched by the write-back");
  Check(a != nullptr && FloatsOf(*a) == new_a,
        "A is overwritten with its trained value");
  Check(b != nullptr && FloatsOf(*b) == new_b,
        "B is overwritten with its trained value");
}

void WriteBackRefusesAMissingStateTensor() {
  const onnx::ModelProto model = MatMulModel();
  const LoraInjectionResult injected = InjectLora(model);
  const LoraStepPlan plan =
      BuildLoraStepGraph(injected.model, injected.adapter, "X", "Y", 4);
  const std::map<std::string, onnx::TensorProto> empty_state;
  CheckThrows<std::invalid_argument>(
      [&]() { WriteBackLoraState(injected.model, plan, empty_state); },
      "missing", "a missing state tensor is refused by name");
}

}  // namespace

int main() {
  InjectingAMatMulAddsATrainableBranchAndLeavesTheBaseWeightUntouched();
  InjectingAGemmWithTransAAddsATransposeAheadOfTheBranch();
  InjectingA1x1ConvUsesTheConvsOwnStridesOnTheFirstBranchConvOnly();
  ANon1x1ConvIsNotInjected();
  RestrictTargetNamesLimitsInjectionToTheNamedWeights();
  AlphaAddsAScaleInitializerAndAMulNode();

  AnInjectedMatMulProducesAStepGraphTheCheckerAccepts();
  InitialStateSeedsTheAdapterFromItsCurrentValueAndZeroesTheMoments();
  EveryOpTheStepGraphEmitsIsEpFriendly();
  AMinibatchedBlockGathersItsRowsOutOfResidentTables();
  ABlockContainingAnUndifferentiableOpIsRefusedByOpType();
  AnAdapterWithNoTargetsIsRefused();
  ANonPositiveNumRowsIsRefused();

  TheFrozenNf4DequantChainIsCapturedRatherThanFolded();
  LoraFoldForTestingComputesTheNf4ChainsArithmeticCorrectly();
  LoraFoldForTestingNeverFoldsAnAdapterTensorEvenWhenEveryInputLooksConstant();

  WriteBackReplacesOnlyTheAdapterTensorsByteForByte();
  WriteBackRefusesAMissingStateTensor();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d lora_entry check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all lora_entry tests passed\n");
  return 0;
}
