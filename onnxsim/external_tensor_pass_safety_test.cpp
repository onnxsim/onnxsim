/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Regression tests for onnx-optimizer's pass_util.h/cse_util.h EXTERNAL-
 * tensor safety fixes (IsConstantTensor/FetchConstantTensor and
 * CSETensorHash/CSETensorCompare): a tensor whose data_location is EXTERNAL
 * and carries no raw_data (onnxsim's PoolExternalData leaves a tensor this
 * way when it is too large to hydrate eagerly -- see
 * tensor_pool_bridge.h's LoadModelPooled/kSimplifyPathHydrateThresholdBytes)
 * must never be silently misread as empty/zero data by a value-baking or
 * CSE pass.
 *
 * Each test here builds a model where the naive (pre-fix) behavior would
 * have produced a WRONG result -- not just a missed optimization -- and
 * checks the fixed behavior instead: the value-dependent transform is
 * skipped (or the pass throws a catchable error), never silently
 * miscomputed. Uses onnx-optimizer's OptimizeFixed directly (no
 * ModelExecutor needed -- these are pure graph-rewrite passes, not
 * onnxsim's own constant folding).
 */
#include <onnx/onnx_pb.h>

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "custom_optimizer_passes.h"
#include "onnxoptimizer/optimize.h"

using namespace ONNX_NAMESPACE;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

TensorProto MakeExternalTensor(const std::string& name,
                               TensorProto::DataType dtype,
                               const std::vector<int64_t>& dims) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(dtype);
  for (int64_t d : dims) t.add_dims(d);
  t.set_data_location(TensorProto::EXTERNAL);
  auto* e = t.add_external_data();
  e->set_key("location");
  e->set_value(name + ".data");  // never actually opened by these tests
  return t;
}

TensorProto MakeRawTensor(const std::string& name, TensorProto::DataType dtype,
                          const std::vector<int64_t>& dims,
                          const std::string& raw) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(dtype);
  for (int64_t d : dims) t.add_dims(d);
  t.set_raw_data(raw);
  return t;
}

void AddFloatOutput(GraphProto* graph, const std::string& name) {
  auto* out = graph->add_output();
  out->set_name(name);
  out->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
}

ModelProto NewModel() {
  ModelProto model;
  model.set_ir_version(9);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  model.mutable_graph()->set_name("t");
  return model;
}

void TestFuseBNIntoConvSkipsExternalParams() {
  ModelProto model = NewModel();
  auto* graph = model.mutable_graph();

  auto* x_in = graph->add_input();
  x_in->set_name("X");
  auto* x_type = x_in->mutable_type()->mutable_tensor_type();
  x_type->set_elem_type(TensorProto::FLOAT);
  auto* x_shape = x_type->mutable_shape();
  x_shape->add_dim()->set_dim_value(1);
  x_shape->add_dim()->set_dim_value(3);
  x_shape->add_dim()->set_dim_value(8);
  x_shape->add_dim()->set_dim_value(8);

  *graph->add_initializer() =
      MakeRawTensor("W", TensorProto::FLOAT, {4, 3, 3, 3},
                    std::string(4 * 3 * 3 * 3 * 4, '\x01'));
  auto* conv = graph->add_node();
  conv->set_op_type("Conv");
  conv->add_input("X");
  conv->add_input("W");
  conv->add_output("Y");

  // BN params are EXTERNAL -- too large to have hydrated eagerly.
  *graph->add_initializer() =
      MakeExternalTensor("scale", TensorProto::FLOAT, {4});
  *graph->add_initializer() =
      MakeExternalTensor("bias", TensorProto::FLOAT, {4});
  *graph->add_initializer() =
      MakeExternalTensor("mean", TensorProto::FLOAT, {4});
  *graph->add_initializer() =
      MakeExternalTensor("var", TensorProto::FLOAT, {4});
  auto* bn = graph->add_node();
  bn->set_op_type("BatchNormalization");
  bn->add_input("Y");
  bn->add_input("scale");
  bn->add_input("bias");
  bn->add_input("mean");
  bn->add_input("var");
  bn->add_output("Z");

  AddFloatOutput(graph, "Z");

  onnxsim::RegisterCustomOptimizerPasses();
  ModelProto result;
  bool threw = false;
  try {
    result = optimization::OptimizeFixed(model, {"fuse_bn_into_conv"});
  } catch (const std::exception& e) {
    threw = true;
    std::fprintf(stderr, "  (unexpected exception: %s)\n", e.what());
  }
  Check(!threw, "fuse_bn_into_conv does not throw/crash on EXTERNAL BN params");
  if (threw) return;

  bool bn_still_present = false;
  for (const auto& n : result.graph().node()) {
    if (n.op_type() == "BatchNormalization") bn_still_present = true;
  }
  Check(bn_still_present,
        "fuse_bn_into_conv safely SKIPS fusing when BN params are EXTERNAL "
        "(if it fused using empty/zero data instead, this would silently "
        "compute a wrong Conv weight)");
}

// Positive control for TestFuseBNIntoConvSkipsExternalParams: the identical
// pattern, but with ordinary in-memory (raw_data) BN params instead of
// EXTERNAL ones, must still fuse normally. Guards against the EXTERNAL-
// safety fix above being accidentally over-broad (e.g. rejecting every
// tensor instead of just EXTERNAL ones).
void TestFuseBNIntoConvStillFusesOrdinaryParams() {
  ModelProto model = NewModel();
  auto* graph = model.mutable_graph();

  auto* x_in = graph->add_input();
  x_in->set_name("X");
  auto* x_type = x_in->mutable_type()->mutable_tensor_type();
  x_type->set_elem_type(TensorProto::FLOAT);
  auto* x_shape = x_type->mutable_shape();
  x_shape->add_dim()->set_dim_value(1);
  x_shape->add_dim()->set_dim_value(3);
  x_shape->add_dim()->set_dim_value(8);
  x_shape->add_dim()->set_dim_value(8);

  *graph->add_initializer() =
      MakeRawTensor("W", TensorProto::FLOAT, {4, 3, 3, 3},
                    std::string(4 * 3 * 3 * 3 * 4, '\x01'));
  auto* conv = graph->add_node();
  conv->set_op_type("Conv");
  conv->add_input("X");
  conv->add_input("W");
  conv->add_output("Y");

  auto ones = [](int64_t n) { return std::string(n * 4, '\x00'); };
  *graph->add_initializer() =
      MakeRawTensor("scale", TensorProto::FLOAT, {4}, ones(4));
  *graph->add_initializer() =
      MakeRawTensor("bias", TensorProto::FLOAT, {4}, ones(4));
  *graph->add_initializer() =
      MakeRawTensor("mean", TensorProto::FLOAT, {4}, ones(4));
  *graph->add_initializer() =
      MakeRawTensor("var", TensorProto::FLOAT, {4}, ones(4));
  auto* bn = graph->add_node();
  bn->set_op_type("BatchNormalization");
  bn->add_input("Y");
  bn->add_input("scale");
  bn->add_input("bias");
  bn->add_input("mean");
  bn->add_input("var");
  bn->add_output("Z");

  AddFloatOutput(graph, "Z");

  onnxsim::RegisterCustomOptimizerPasses();
  ModelProto result;
  bool threw = false;
  try {
    result = optimization::OptimizeFixed(model, {"fuse_bn_into_conv"});
  } catch (const std::exception& e) {
    threw = true;
    std::fprintf(stderr, "  (unexpected exception: %s)\n", e.what());
  }
  Check(!threw, "fuse_bn_into_conv does not throw on ordinary BN params");
  if (threw) return;

  bool bn_still_present = false;
  for (const auto& n : result.graph().node()) {
    if (n.op_type() == "BatchNormalization") bn_still_present = true;
  }
  Check(!bn_still_present,
        "fuse_bn_into_conv DOES fuse away BatchNormalization when its "
        "params are ordinary, locally-available tensors -- the EXTERNAL "
        "safety check must not be overly broad");
}

void TestEliminateDuplicateInitializerDoesNotMergeExternalTensors() {
  ModelProto model = NewModel();
  auto* graph = model.mutable_graph();

  // Two EXTERNAL initializers, same shape/dtype -- if their bytes could be
  // read they might genuinely differ; without reading them there is no way
  // to know, so they must not be merged.
  *graph->add_initializer() = MakeExternalTensor("a", TensorProto::FLOAT, {4});
  *graph->add_initializer() = MakeExternalTensor("b", TensorProto::FLOAT, {4});

  auto* add = graph->add_node();
  add->set_op_type("Add");
  add->add_input("a");
  add->add_input("b");
  add->add_output("y");
  AddFloatOutput(graph, "y");

  ModelProto result;
  bool threw = false;
  try {
    result =
        optimization::OptimizeFixed(model, {"eliminate_duplicate_initializer"});
  } catch (const std::exception& e) {
    threw = true;
    std::fprintf(stderr, "  (unexpected exception: %s)\n", e.what());
  }
  Check(!threw,
        "eliminate_duplicate_initializer does not throw/crash on EXTERNAL "
        "initializers");
  if (threw) return;
  Check(result.graph().initializer_size() == 2,
        "eliminate_duplicate_initializer does NOT merge two distinct "
        "EXTERNAL initializers just because their (unreadable) content "
        "might coincidentally match");
}

void TestEliminateCommonSubexpressionDoesNotMergeExternalConstants() {
  ModelProto model = NewModel();
  auto* graph = model.mutable_graph();

  for (int i = 0; i < 2; ++i) {
    const std::string ci = "c" + std::to_string(i);
    const std::string yi = "y" + std::to_string(i);
    auto* c = graph->add_node();
    c->set_op_type("Constant");
    c->add_output(ci);
    auto* attr = c->add_attribute();
    attr->set_name("value");
    attr->set_type(AttributeProto::TENSOR);
    *attr->mutable_t() =
        MakeExternalTensor("cval" + std::to_string(i), TensorProto::FLOAT, {4});

    auto* id = graph->add_node();
    id->set_op_type("Identity");
    id->add_input(ci);
    id->add_output(yi);
    AddFloatOutput(graph, yi);
  }

  ModelProto result;
  bool threw = false;
  try {
    result =
        optimization::OptimizeFixed(model, {"eliminate_common_subexpression"});
  } catch (const std::exception& e) {
    threw = true;
    std::fprintf(stderr, "  (unexpected exception: %s)\n", e.what());
  }
  Check(!threw,
        "eliminate_common_subexpression does not throw/crash on EXTERNAL "
        "Constant values");
  if (threw) return;
  int constant_count = 0;
  for (const auto& n : result.graph().node()) {
    if (n.op_type() == "Constant") ++constant_count;
  }
  Check(constant_count == 2,
        "eliminate_common_subexpression does NOT merge two distinct "
        "EXTERNAL Constant nodes just because their (unreadable) content "
        "might coincidentally match");
}

void TestEliminateIfWithConstCondSkipsExternalCond() {
  ModelProto model = NewModel();
  auto* graph = model.mutable_graph();

  *graph->add_initializer() = MakeExternalTensor("cond", TensorProto::BOOL, {});

  GraphProto then_g;
  then_g.set_name("then");
  auto* then_c = then_g.add_node();
  then_c->set_op_type("Constant");
  then_c->add_output("then_out");
  auto* then_attr = then_c->add_attribute();
  then_attr->set_name("value");
  then_attr->set_type(AttributeProto::TENSOR);
  *then_attr->mutable_t() =
      MakeRawTensor("tv", TensorProto::FLOAT, {1}, std::string(4, '\x01'));
  AddFloatOutput(&then_g, "then_out");

  GraphProto else_g;
  else_g.set_name("else");
  auto* else_c = else_g.add_node();
  else_c->set_op_type("Constant");
  else_c->add_output("else_out");
  auto* else_attr = else_c->add_attribute();
  else_attr->set_name("value");
  else_attr->set_type(AttributeProto::TENSOR);
  *else_attr->mutable_t() =
      MakeRawTensor("ev", TensorProto::FLOAT, {1}, std::string(4, '\x02'));
  AddFloatOutput(&else_g, "else_out");

  auto* if_node = graph->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("z");
  auto* then_attr_n = if_node->add_attribute();
  then_attr_n->set_name("then_branch");
  then_attr_n->set_type(AttributeProto::GRAPH);
  *then_attr_n->mutable_g() = then_g;
  auto* else_attr_n = if_node->add_attribute();
  else_attr_n->set_name("else_branch");
  else_attr_n->set_type(AttributeProto::GRAPH);
  *else_attr_n->mutable_g() = else_g;

  AddFloatOutput(graph, "z");

  ModelProto result;
  bool threw = false;
  try {
    result =
        optimization::OptimizeFixed(model, {"eliminate_if_with_const_cond"});
  } catch (const std::exception& e) {
    threw = true;
    std::fprintf(stderr, "  (unexpected exception: %s)\n", e.what());
  }
  Check(!threw,
        "eliminate_if_with_const_cond does not crash (UB) on an EXTERNAL "
        "cond tensor");
  if (threw) return;
  bool if_still_present = false;
  for (const auto& n : result.graph().node()) {
    if (n.op_type() == "If") if_still_present = true;
  }
  Check(if_still_present,
        "eliminate_if_with_const_cond safely SKIPS inlining when cond is "
        "EXTERNAL (reading past a null tensor pointer would be UB, not just "
        "a missed optimization)");
}

void TestFuseQKVSkipsExternalWeights() {
  ModelProto model = NewModel();
  auto* graph = model.mutable_graph();

  auto* x_in = graph->add_input();
  x_in->set_name("X");
  x_in->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::FLOAT);

  *graph->add_initializer() =
      MakeExternalTensor("Wq", TensorProto::FLOAT, {8, 8});
  *graph->add_initializer() =
      MakeExternalTensor("Wk", TensorProto::FLOAT, {8, 8});
  *graph->add_initializer() =
      MakeExternalTensor("Wv", TensorProto::FLOAT, {8, 8});

  for (const std::string& w : {"Wq", "Wk", "Wv"}) {
    auto* mm = graph->add_node();
    mm->set_op_type("MatMul");
    mm->add_input("X");
    mm->add_input(w);
    mm->add_output(w + "_out");
    AddFloatOutput(graph, w + "_out");
  }

  ModelProto result;
  bool threw = false;
  try {
    result = optimization::OptimizeFixed(model, {"fuse_qkv"});
  } catch (const std::exception& e) {
    threw = true;
    std::fprintf(stderr, "  (unexpected exception: %s)\n", e.what());
  }
  Check(!threw,
        "fuse_qkv does not crash (null deref) when Q/K/V weights are "
        "EXTERNAL");
}

}  // namespace

int main() {
  TestFuseBNIntoConvSkipsExternalParams();
  TestFuseBNIntoConvStillFusesOrdinaryParams();
  TestEliminateDuplicateInitializerDoesNotMergeExternalTensors();
  TestEliminateCommonSubexpressionDoesNotMergeExternalConstants();
  TestEliminateIfWithConstCondSkipsExternalCond();
  TestFuseQKVSkipsExternalWeights();

  if (g_failures == 0) {
    std::printf("external_tensor_pass_safety_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "external_tensor_pass_safety_test: %d failure(s)\n",
               g_failures);
  return 1;
}
