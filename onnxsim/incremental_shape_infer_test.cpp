// Exercises IncrementalShapeInferer against real ONNX op schemas: every case
// is checked both by running the driver and by cross-checking its result
// against the exact, unmodified onnx::shape_inference::InferShapes on the
// same starting model, so a divergence here means the fast path computed a
// different answer than the real ONNX driver would have -- not just "some
// answer".
#include "incremental_shape_infer.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"

namespace {

using onnx::AttributeProto;
using onnx::GraphProto;
using onnx::ModelProto;
using onnx::NodeProto;
using onnx::TensorProto;
using onnx::TypeProto;
using onnx::ValueInfoProto;

TypeProto TensorType(TensorProto::DataType dtype, std::vector<int64_t> dims) {
  TypeProto t;
  auto* tt = t.mutable_tensor_type();
  tt->set_elem_type(dtype);
  auto* shape = tt->mutable_shape();
  for (int64_t d : dims) shape->add_dim()->set_dim_value(d);
  return t;
}

ValueInfoProto ValueInfo(const std::string& name, const TypeProto& type) {
  ValueInfoProto vi;
  vi.set_name(name);
  *vi.mutable_type() = type;
  return vi;
}

// A graph output declared with no type at all (has_type() == false), so its
// type must come entirely from inference. ValueInfo(name, TypeProto()) is
// *not* equivalent: calling mutable_type() with an otherwise-default
// TypeProto still flips has_type() to true (message-field presence is
// tracked even in proto3), which both the driver and the real onnx
// implementation.cc treat as "already known" and never overwrite once
// inference fails to produce a value -- so a node whose inference is
// expected to fail would misleadingly still report FindType() != nullptr.
ValueInfoProto UntypedValueInfo(const std::string& name) {
  ValueInfoProto vi;
  vi.set_name(name);
  return vi;
}

NodeProto Node(std::string op_type, std::vector<std::string> inputs,
               std::vector<std::string> outputs) {
  NodeProto n;
  n.set_op_type(std::move(op_type));
  for (auto& i : inputs) n.add_input(i);
  for (auto& o : outputs) n.add_output(o);
  return n;
}

AttributeProto IntsAttr(const std::string& name, std::vector<int64_t> vals) {
  AttributeProto a;
  a.set_name(name);
  a.set_type(AttributeProto::INTS);
  for (int64_t v : vals) a.add_ints(v);
  return a;
}

AttributeProto TensorAttr(const std::string& name, const TensorProto& t) {
  AttributeProto a;
  a.set_name(name);
  a.set_type(AttributeProto::TENSOR);
  *a.mutable_t() = t;
  return a;
}

TensorProto Int64Tensor(const std::string& name, std::vector<int64_t> vals) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(TensorProto::INT64);
  t.add_dims(static_cast<int64_t>(vals.size()));
  for (int64_t v : vals) t.add_int64_data(v);
  return t;
}

ModelProto BaseModel() {
  ModelProto m;
  m.set_ir_version(8);
  auto* opset = m.add_opset_import();
  opset->set_domain("");
  opset->set_version(17);
  return m;
}

const TypeProto* FindType(const ModelProto& m, const std::string& name) {
  for (const auto& vi : m.graph().value_info())
    if (vi.name() == name) return &vi.type();
  for (const auto& vi : m.graph().output())
    if (vi.name() == name && vi.has_type()) return &vi.type();
  for (const auto& vi : m.graph().input())
    if (vi.name() == name && vi.has_type()) return &vi.type();
  return nullptr;
}

bool TypesEqual(const ModelProto& a, const ModelProto& b,
                const std::string& name) {
  const TypeProto* ta = FindType(a, name);
  const TypeProto* tb = FindType(b, name);
  if (!ta || !tb) return ta == tb;  // both-absent counts as equal
  return ta->SerializeAsString() == tb->SerializeAsString();
}

int g_failures = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::cerr << "CHECK failed at " << __FILE__ << ":" << __LINE__ << ": " \
                << #cond << "\n";                                            \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

// A [1,3,8,8] input through Conv(W: [4,3,3,3]) -> Relu, output untyped.
ModelProto ConvReluModel() {
  ModelProto m = BaseModel();
  GraphProto* g = m.mutable_graph();
  *g->add_input() =
      ValueInfo("X", TensorType(TensorProto::FLOAT, {1, 3, 8, 8}));
  *g->add_initializer() = [] {
    TensorProto w;
    w.set_name("W");
    w.set_data_type(TensorProto::FLOAT);
    for (int64_t d : {4, 3, 3, 3}) w.add_dims(d);
    for (int i = 0; i < 4 * 3 * 3 * 3; ++i) w.add_float_data(0.f);
    return w;
  }();
  *g->add_node() = Node("Conv", {"X", "W"}, {"conv_out"});
  *g->add_node() = Node("Relu", {"conv_out"}, {"relu_out"});
  *g->add_output() = UntypedValueInfo("relu_out");
  return m;
}

void TestConvReluMatchesReference() {
  ModelProto incremental = ConvReluModel();
  ModelProto reference = ConvReluModel();

  onnxsim::IncrementalShapeInferer inferer;
  inferer.Run(incremental);
  onnx::shape_inference::InferShapes(reference);

  CHECK(FindType(incremental, "conv_out") != nullptr);
  CHECK(TypesEqual(incremental, reference, "conv_out"));
  CHECK(TypesEqual(incremental, reference, "relu_out"));
  // Conv's output channel count comes from W's first dim (4); output spatial
  // dims are 8x8 with a 3x3 kernel, stride 1, no padding -> 6x6.
  const TypeProto* conv_out = FindType(incremental, "conv_out");
  CHECK(conv_out->tensor_type().shape().dim(0).dim_value() == 1);
  CHECK(conv_out->tensor_type().shape().dim(1).dim_value() == 4);
  CHECK(conv_out->tensor_type().shape().dim(2).dim_value() == 6);
  CHECK(conv_out->tensor_type().shape().dim(3).dim_value() == 6);
}

// Reshape's target shape comes from a Constant node's tensor value: only
// resolvable if the driver tracks Constant-produced values the way onnx's
// own driver does (ProcessConstantNode).
void TestReshapeViaConstant() {
  ModelProto incremental = BaseModel();
  GraphProto* g = incremental.mutable_graph();
  *g->add_input() = ValueInfo("X", TensorType(TensorProto::FLOAT, {2, 3, 4}));
  {
    NodeProto n = Node("Constant", {}, {"shape_const"});
    *n.add_attribute() = TensorAttr("value", Int64Tensor("", {2, 12}));
    *g->add_node() = n;
  }
  *g->add_node() = Node("Reshape", {"X", "shape_const"}, {"y"});
  *g->add_output() = UntypedValueInfo("y");

  ModelProto reference = incremental;
  onnxsim::IncrementalShapeInferer inferer;
  inferer.Run(incremental);
  onnx::shape_inference::InferShapes(reference);

  CHECK(TypesEqual(incremental, reference, "y"));
  const TypeProto* y = FindType(incremental, "y");
  CHECK(y != nullptr);
  if (y) {
    CHECK(y->tensor_type().shape().dim_size() == 2);
    CHECK(y->tensor_type().shape().dim(0).dim_value() == 2);
    CHECK(y->tensor_type().shape().dim(1).dim_value() == 12);
  }
}

// Same, but the shape tensor is a graph initializer instead of a Constant
// node's output (the other source InferenceContextImpl reads input data
// from).
void TestReshapeViaInitializer() {
  ModelProto incremental = BaseModel();
  GraphProto* g = incremental.mutable_graph();
  *g->add_input() = ValueInfo("X", TensorType(TensorProto::FLOAT, {2, 3, 4}));
  *g->add_initializer() = Int64Tensor("shape_init", {6, 4});
  *g->add_node() = Node("Reshape", {"X", "shape_init"}, {"y"});
  *g->add_output() = UntypedValueInfo("y");

  ModelProto reference = incremental;
  onnxsim::IncrementalShapeInferer inferer;
  inferer.Run(incremental);
  onnx::shape_inference::InferShapes(reference);

  CHECK(TypesEqual(incremental, reference, "y"));
  const TypeProto* y = FindType(incremental, "y");
  CHECK(y != nullptr);
  if (y) {
    CHECK(y->tensor_type().shape().dim(0).dim_value() == 6);
    CHECK(y->tensor_type().shape().dim(1).dim_value() == 4);
  }
}

// A Conv whose weight initializer is larger than kMaxCachedValueBytes must
// still infer correctly (falls back to direct, uncached inference for that
// node -- see incremental_shape_infer.cpp) and, unlike the small-weight
// ConvReluModel case, must not appear in the cache at all: caching a large
// value the op's inference function never actually reads would only add
// serialization cost for nothing, and skipping it costs no more than the
// bypassed node's usual (cheap, type-only) recomputation.
void TestLargeWeightBypassesCache() {
  ModelProto m = BaseModel();
  GraphProto* g = m.mutable_graph();
  *g->add_input() =
      ValueInfo("X", TensorType(TensorProto::FLOAT, {1, 3, 40, 40}));
  *g->add_initializer() = [] {
    TensorProto w;
    w.set_name("W");
    w.set_data_type(TensorProto::FLOAT);
    for (int64_t d : {8, 3, 20, 20})
      w.add_dims(d);  // 8*3*20*20*4B ~= 38KB, well over 4KB
    for (int i = 0; i < 8 * 3 * 20 * 20; ++i) w.add_float_data(0.f);
    return w;
  }();
  *g->add_node() = Node("Conv", {"X", "W"}, {"conv_out"});
  *g->add_node() = Node("Relu", {"conv_out"}, {"relu_out"});
  *g->add_output() = UntypedValueInfo("relu_out");

  ModelProto reference = m;
  onnxsim::IncrementalShapeInferer inferer;
  inferer.Run(m);
  onnx::shape_inference::InferShapes(reference);

  CHECK(TypesEqual(m, reference, "conv_out"));
  CHECK(TypesEqual(m, reference, "relu_out"));
  CHECK(inferer.CacheSize() ==
        1);  // Relu only; Conv's large weight bypassed caching
}

// An op with no registered schema: both the driver and the real onnx
// inference must leave its output (and anything downstream of it) without an
// actually-inferred type, rather than throwing -- matching onnx's own
// error_mode==0 leniency. "Without an actually-inferred type" is checked via
// value_case(), not has_type()/FindType()!=nullptr: onnx's own driver (and
// this one, matching it -- see undefined_value_types_by_name above) flips
// has_type() to true on every declared-but-unresolved input/output as a side
// effect of registering it, even though its content stays empty.
void TestUnsupportedOpLeavesTypeUnset() {
  ModelProto incremental = BaseModel();
  GraphProto* g = incremental.mutable_graph();
  *g->add_input() = ValueInfo("X", TensorType(TensorProto::FLOAT, {2, 3}));
  *g->add_node() = Node("TotallyMadeUpOp", {"X"}, {"mystery"});
  *g->add_node() = Node("Relu", {"mystery"}, {"y"});
  *g->add_output() = UntypedValueInfo("y");

  ModelProto reference = incremental;
  onnxsim::IncrementalShapeInferer inferer;
  inferer.Run(incremental);
  onnx::shape_inference::InferShapes(reference);

  CHECK(FindType(incremental, "mystery") == nullptr);
  CHECK(FindType(reference, "mystery") == nullptr);
  const TypeProto* y = FindType(incremental, "y");
  CHECK(y != nullptr && y->value_case() == TypeProto::VALUE_NOT_SET);
  CHECK(TypesEqual(incremental, reference, "y"));
}

// Re-running the same (unmodified) graph through one IncrementalShapeInferer
// instance must not grow its cache -- every node signature was already seen
// -- while a graph containing one genuinely different node (different Conv
// strides) grows it by two new entries: the strided Conv itself, and Relu
// again under a *new* signature (its own attributes are unchanged, but its
// input's type -- conv_out's now-smaller spatial shape -- is part of the
// cache key too, correctly, since Relu's own output shape depends on it).
// Either way the changed node's shape must come out correct, not the stale
// cached one.
void TestCacheReuseAndInvalidationBySignature() {
  onnxsim::IncrementalShapeInferer inferer;

  ModelProto round1 = ConvReluModel();
  inferer.Run(round1);
  const size_t size_after_round1 = inferer.CacheSize();
  CHECK(size_after_round1 == 2);  // Conv, Relu

  ModelProto round2 = ConvReluModel();  // structurally identical to round1
  inferer.Run(round2);
  CHECK(inferer.CacheSize() == size_after_round1);
  CHECK(TypesEqual(round1, round2, "conv_out"));
  CHECK(TypesEqual(round1, round2, "relu_out"));

  ModelProto round3 = ConvReluModel();
  for (NodeProto& n : *round3.mutable_graph()->mutable_node()) {
    if (n.op_type() == "Conv") *n.add_attribute() = IntsAttr("strides", {2, 2});
  }
  inferer.Run(round3);
  CHECK(inferer.CacheSize() ==
        size_after_round1 + 2);  // strided Conv + Relu-on-new-input-shape

  ModelProto reference3 = ConvReluModel();
  for (NodeProto& n : *reference3.mutable_graph()->mutable_node()) {
    if (n.op_type() == "Conv") *n.add_attribute() = IntsAttr("strides", {2, 2});
  }
  onnx::shape_inference::InferShapes(reference3);
  CHECK(TypesEqual(round3, reference3, "conv_out"));
  const TypeProto* conv_out = FindType(round3, "conv_out");
  CHECK(conv_out != nullptr);
  if (conv_out) {
    // stride 2, 3x3 kernel, no padding, 8x8 input -> floor((8-3)/2)+1 = 3.
    CHECK(conv_out->tensor_type().shape().dim(2).dim_value() == 3);
    CHECK(conv_out->tensor_type().shape().dim(3).dim_value() == 3);
  }
}

}  // namespace

int main() {
  TestConvReluMatchesReference();
  TestLargeWeightBypassesCache();
  TestReshapeViaConstant();
  TestReshapeViaInitializer();
  TestUnsupportedOpLeavesTypeUnset();
  TestCacheReuseAndInvalidationBySignature();

  if (g_failures == 0) {
    std::cout << "incremental_shape_infer_test: all checks passed\n";
    return 0;
  }
  std::cerr << "incremental_shape_infer_test: " << g_failures
            << " check(s) failed\n";
  return 1;
}
