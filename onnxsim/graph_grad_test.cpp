/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises graph_grad.{h,cpp} -- the C++ port of onnxsim/graph_grad.py's
 * reverse-mode differentiator.
 *
 * tests/test_graph_grad.py checks the Python rules the way gradients are
 * really checked: 104 finite-difference comparisons against a float64
 * re-evaluation of the forward. That is not reproducible here, because
 * nothing in this build evaluates an ONNX graph -- the wheel does not compile
 * ONNX Runtime (see CLAUDE.md) and the WASM build hands evaluation to
 * onnxruntime-web at run time. So this test covers the half that *is*
 * checkable without an evaluator, and covers it exactly: the rule table's
 * membership, the refusals, the operator allowlist the emitted graph must
 * stay inside, and the structural shape of the things the Python docstrings
 * single out as easiest to get subtly wrong -- undoing a broadcast,
 * accumulating a tensor read more than once, and the one rule whose emission
 * order is long enough to drift unnoticed (LayerNormalization).
 *
 * The numerical half is not duplicated and not claimed: it lives in
 * tests/test_graph_grad.py, and the C++ rules are transcriptions of the
 * Python ones written to emit the same nodes in the same order.
 *
 * Plain asserts and a failure counter, like sym_expr_test.cpp and
 * precision_estimator_test.cpp -- this repository vendors no gtest, and a
 * test that needed one could not be built here at all.
 */
#include "graph_grad.h"

#include <onnx/onnx_pb.h>

#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "qat_graph_builder.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// `body` must throw `E` with a message containing `fragment`. Both halves
// matter: the type is what a caller switches on, and the message is what
// tells a human *which* node of their block is the problem.
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

using Shapes = std::map<std::string, std::vector<int64_t>>;

onnx::AttributeProto IntAttr(const std::string& name, int64_t value) {
  onnx::AttributeProto attribute;
  attribute.set_name(name);
  attribute.set_type(onnx::AttributeProto::INT);
  attribute.set_i(value);
  return attribute;
}

onnx::AttributeProto IntsAttr(const std::string& name,
                              const std::vector<int64_t>& values) {
  onnx::AttributeProto attribute;
  attribute.set_name(name);
  attribute.set_type(onnx::AttributeProto::INTS);
  for (int64_t value : values) attribute.add_ints(value);
  return attribute;
}

onnx::AttributeProto FloatAttr(const std::string& name, float value) {
  onnx::AttributeProto attribute;
  attribute.set_name(name);
  attribute.set_type(onnx::AttributeProto::FLOAT);
  attribute.set_f(value);
  return attribute;
}

onnx::AttributeProto StrAttr(const std::string& name,
                             const std::string& value) {
  onnx::AttributeProto attribute;
  attribute.set_name(name);
  attribute.set_type(onnx::AttributeProto::STRING);
  attribute.set_s(value);
  return attribute;
}

onnx::NodeProto Node(const std::string& op_type,
                     const std::vector<std::string>& inputs,
                     const std::vector<std::string>& outputs,
                     const std::vector<onnx::AttributeProto>& attrs = {}) {
  onnx::NodeProto node;
  node.set_op_type(op_type);
  for (const std::string& input : inputs) node.add_input(input);
  for (const std::string& output : outputs) node.add_output(output);
  for (const onnx::AttributeProto& attr : attrs) *node.add_attribute() = attr;
  return node;
}

std::vector<std::string> OpTypes(const GraphBuilder& b) {
  std::vector<std::string> types;
  for (const onnx::NodeProto& node : b.nodes()) types.push_back(node.op_type());
  return types;
}

std::set<std::string> OpTypeSet(const GraphBuilder& b) {
  const std::vector<std::string> types = OpTypes(b);
  return std::set<std::string>(types.begin(), types.end());
}

// Every op type `b` emitted, with any templated rule's function call
// resolved via MakeStepGraph's own inlining -- otherwise a call node's
// op_type (the function's own name, e.g. "GradAdd") would show up here
// instead of the ops it actually expands to, which is what an allowlist
// check is about. One state entry per `grads` target is enough to give
// MakeStepGraph something to close the graph on; the values it computes for
// those outputs are never read here.
std::set<std::string> InlinedOpTypeSet(
    const GraphBuilder& b, const std::map<std::string, std::string>& grads,
    const Shapes& shapes) {
  StepGraphSpec spec;
  for (const auto& [target, grad_name] : grads) {
    spec.state.push_back({/*input=*/target + "_unused", shapes.at(target),
                          /*next_output=*/grad_name});
  }
  const StepGraph step = MakeStepGraph(b, spec);
  std::set<std::string> types;
  for (const onnx::NodeProto& node : step.model.graph().node()) {
    types.insert(node.op_type());
  }
  return types;
}

std::string Join(const std::set<std::string>& values) {
  std::string out;
  for (const std::string& value : values) {
    if (!out.empty()) out += ", ";
    out += value;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The five forward slices the allowlist test differentiates. Between them
// they reach every rule in the table; individually they are small enough that
// a failure names a rule rather than a graph.
// ---------------------------------------------------------------------------

// Broadcasting elementwise arithmetic, plus a tensor read twice. Includes an
// Add -- the one rule this slice would otherwise never exercise, and (since
// GradAddTemplated) the one source of an inlined Identity node, not just an
// Add: see BackwardOps()'s own comment for why.
std::vector<onnx::NodeProto> ElementwiseSlice() {
  return {
      Node("Sub", {"A", "B"}, {"T0"}),
      Node("Div", {"T0", "C"}, {"T1"}),
      Node("Add", {"T1", "C"}, {"T2"}),
      Node("Mul", {"T2", "T2"}, {"Y"}),
  };
}

Shapes ElementwiseShapes() {
  return {{"A", {4, 3}},  {"B", {3}},     {"C", {4, 3}}, {"T0", {4, 3}},
          {"T1", {4, 3}}, {"T2", {4, 3}}, {"Y", {4, 3}}};
}

// A matmul, a rectifier, a softmax and a clip -- the activation half of a
// quantization-reconstruction block.
std::vector<onnx::NodeProto> AttentionishSlice() {
  return {
      Node("MatMul", {"X", "W"}, {"H"}),
      Node("Relu", {"H"}, {"R"}),
      Node("Softmax", {"R"}, {"S"}, {IntAttr("axis", -1)}),
      Node("Clip", {"S", "lo", "hi"}, {"Y"}),
  };
}

Shapes AttentionishShapes() {
  return {{"X", {2, 3}}, {"W", {3, 4}}, {"H", {2, 4}},
          {"R", {2, 4}}, {"S", {2, 4}}, {"Y", {2, 4}}};
}

// The GELU-shaped transcendentals, ending in a rank change.
std::vector<onnx::NodeProto> TranscendentalSlice() {
  return {
      Node("Erf", {"A"}, {"E"}),
      Node("Sqrt", {"E"}, {"Q"}),
      Node("Reshape", {"Q", "shp"}, {"Y"}),
  };
}

Shapes TranscendentalShapes() {
  return {{"A", {2, 3}}, {"E", {2, 3}}, {"Q", {2, 3}}, {"Y", {6}}};
}

// A Gemm with a bias, the saturating activations, an alias, and both
// reductions.
std::vector<onnx::NodeProto> GemmSlice() {
  return {
      Node("Gemm", {"A", "W", "Cb"}, {"G1"},
           {FloatAttr("alpha", 2.0f), FloatAttr("beta", 0.5f)}),
      Node("Sigmoid", {"G1"}, {"P"}),
      Node("Tanh", {"P"}, {"T"}),
      Node("Identity", {"T"}, {"I"}),
      Node("Neg", {"I"}, {"N"}),
      Node("Exp", {"N"}, {"E"}),
      Node("Transpose", {"E"}, {"Tr"}, {IntsAttr("perm", {1, 0})}),
      Node("ReduceMean", {"Tr", "ax0"}, {"M"}, {IntAttr("keepdims", 0)}),
      Node("ReduceSum", {"M", "ax1"}, {"Y"}, {IntAttr("keepdims", 1)}),
  };
}

Shapes GemmShapes() {
  return {{"A", {2, 3}}, {"W", {3, 4}},  {"Cb", {4}},   {"G1", {2, 4}},
          {"P", {2, 4}}, {"T", {2, 4}},  {"I", {2, 4}}, {"N", {2, 4}},
          {"E", {2, 4}}, {"Tr", {4, 2}}, {"M", {4}},    {"Y", {1}}};
}

// A fused LayerNorm with both parameters -- the only rule that reduces over
// the normalized axes, and so the only source of ReduceMean and Sqrt in the
// emitted backward.
std::vector<onnx::NodeProto> LayerNormSlice() {
  return {Node("LayerNormalization", {"X", "S", "Bn"}, {"Y"})};
}

Shapes LayerNormShapes() {
  return {{"X", {2, 3, 4}}, {"S", {4}}, {"Bn", {4}}, {"Y", {2, 3, 4}}};
}

// A fused BatchNormalization in inference mode -- mean/var are fixed
// per-channel inputs here rather than reductions of X, so this is the only
// slice whose Neg comes from a reduction result rather than from GradSub.
std::vector<onnx::NodeProto> BatchNormSlice() {
  return {Node("BatchNormalization", {"X", "S", "Bn", "Mn", "Vr"}, {"Y"})};
}

Shapes BatchNormShapes() {
  return {{"X", {2, 3, 4, 4}}, {"S", {3}},  {"Bn", {3}},
          {"Mn", {3}},         {"Vr", {3}}, {"Y", {2, 3, 4, 4}}};
}

// InstanceNormalization -- mean/var computed from X itself like LayerNorm,
// but over the spatial axes only, with a per-channel scale/B that (like
// BatchNormalization's) needs reshaping before it broadcasts against X.
std::vector<onnx::NodeProto> InstanceNormSlice() {
  return {Node("InstanceNormalization", {"X", "S", "Bn"}, {"Y"})};
}

Shapes InstanceNormShapes() {
  return {{"X", {2, 3, 4, 4}}, {"S", {3}}, {"Bn", {3}}, {"Y", {2, 3, 4, 4}}};
}

// A convolution with a bias -- the rule with a weight tensor to
// differentiate, and (along with MaxPool/AveragePool below) one whose
// emission depends on arithmetic (the index tables) rather than only on the
// shapes.
std::vector<onnx::NodeProto> ConvSlice() {
  return {Node("Conv", {"X", "Wc", "Bc"}, {"Y"})};
}

Shapes ConvShapes() {
  return {{"X", {1, 2, 4, 4}},
          {"Wc", {3, 2, 3, 3}},
          {"Bc", {3}},
          {"Y", {1, 3, 2, 2}}};
}

// A MaxPool feeding an AveragePool -- the two rules that share Conv's
// im2col/col2im index tables without a weight tensor, and (MaxPool) the one
// rule other than Clip that emits Greater/Less/Cast.
std::vector<onnx::NodeProto> PoolSlice() {
  return {
      Node("MaxPool", {"X"}, {"M"},
           {IntsAttr("kernel_shape", {2, 2}), IntsAttr("strides", {2, 2})}),
      Node("AveragePool", {"M"}, {"Y"}, {IntsAttr("kernel_shape", {1, 1})}),
  };
}

Shapes PoolShapes() {
  return {{"X", {1, 1, 4, 4}}, {"M", {1, 1, 2, 2}}, {"Y", {1, 1, 2, 2}}};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// If this fails, the two differentiators disagree about which ops they will
// touch at all: a caller's block discovery (which is told to test against
// SupportedOps rather than to catch refusals) would admit a block in the
// browser that the Python refuses, or the reverse.
void TheSupportedOpsAreExactlyThePythonRuleTable() {
  const std::set<std::string> expected = {"Add",
                                          "AveragePool",
                                          "BatchNormalization",
                                          "Clip",
                                          "Conv",
                                          "Div",
                                          "Erf",
                                          "Exp",
                                          "Gather",
                                          "Gemm",
                                          "Identity",
                                          "InstanceNormalization",
                                          "LayerNormalization",
                                          "MatMul",
                                          "MaxPool",
                                          "Mul",
                                          "Neg",
                                          "ReduceMean",
                                          "ReduceSum",
                                          "Relu",
                                          "Reshape",
                                          "Sigmoid",
                                          "Softmax",
                                          "Sqrt",
                                          "Sub",
                                          "Tanh",
                                          "Transpose"};
  Check(SupportedOps() == expected,
        "SupportedOps() should equal graph_grad.py's _RULES keys, got {" +
            Join(SupportedOps()) + "}");
}

// If this fails, a backward graph could reach an operator no WebGPU or NPU
// execution provider implements -- numerically perfect and useless for the
// thing this port exists for, since the emitted nodes go into the same graph
// as the forward and the optimizer step.
void TheBackwardOpsAreThePythonAllowlistAndSitInsideEpFriendlyOps() {
  const std::set<std::string> expected = {
      "Add",       "Cast",    "Div",    "Exp", "Gather",   "Greater",
      "Identity",  "Less",    "MatMul", "Mul", "Neg",      "ReduceMean",
      "ReduceSum", "Reshape", "Sqrt",   "Sub", "Transpose"};
  Check(BackwardOps() == expected,
        "BackwardOps() should equal graph_grad.py's BACKWARD_OPS, got {" +
            Join(BackwardOps()) + "}");
  for (const std::string& op : BackwardOps()) {
    Check(EpFriendlyOps().count(op) != 0,
          "BackwardOps() member '" + op + "' is outside EpFriendlyOps()");
  }
}

// If this fails, some rule has started emitting an op outside the allowlist
// (the forward direction), or the allowlist has grown a member no rule can
// produce (the reverse). Both are checked, exactly as the Python test does.
void TheEmittedBackwardStaysInsideTheOperatorAllowlist() {
  struct Slice {
    std::vector<onnx::NodeProto> nodes;
    Shapes shapes;
    std::vector<std::string> targets;
  };
  const std::vector<Slice> slices = {
      {ElementwiseSlice(), ElementwiseShapes(), {"A", "B", "C"}},
      {AttentionishSlice(), AttentionishShapes(), {"X", "W"}},
      {TranscendentalSlice(), TranscendentalShapes(), {"A"}},
      {GemmSlice(), GemmShapes(), {"A", "W", "Cb"}},
      {LayerNormSlice(), LayerNormShapes(), {"X", "S", "Bn"}},
      {BatchNormSlice(), BatchNormShapes(), {"X", "S", "Bn", "Mn", "Vr"}},
      {InstanceNormSlice(), InstanceNormShapes(), {"X", "S", "Bn"}},
      {ConvSlice(), ConvShapes(), {"X", "Wc", "Bc"}},
      {PoolSlice(), PoolShapes(), {"X"}},
  };

  std::set<std::string> emitted;
  for (const Slice& slice : slices) {
    GraphBuilder b("bw_");
    const std::map<std::string, std::string> grads = BuildBackward(
        b, slice.nodes, slice.shapes, {{"Y", "dY"}}, slice.targets);
    const std::set<std::string> types =
        InlinedOpTypeSet(b, grads, slice.shapes);
    for (const std::string& op : types) {
      Check(BackwardOps().count(op) != 0,
            "backward graph reached outside the allowlist: " + op);
    }
    emitted.insert(types.begin(), types.end());
  }
  Check(emitted == BackwardOps(),
        "the seven slices together should emit every allowlisted op and no "
        "other; got {" +
            Join(emitted) + "}");
}

// If this fails, a slice containing an op with no rule would be
// differentiated anyway -- as a zero or a straight-through approximation --
// and the caller would get a quietly wrong gradient instead of a refusal it
// can act on.
void AnOpWithNoRuleIsRefusedByName() {
  const std::vector<onnx::NodeProto> nodes = {Node("Sin", {"A"}, {"Y"})};
  const Shapes shapes = {{"A", {3, 4}}, {"Y", {3, 4}}};
  GraphBuilder b;
  CheckThrows<UnsupportedOpError>(
      [&] { BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A"}); }, "Sin",
      "an op with no rule should be refused, naming the op");
}

// If this fails, whether a block is in scope would depend on which tensor the
// caller happened to seed -- so the same block would be accepted or refused
// depending on the loss, which is not a property a caller can reason about.
void AnUnsupportedOpIsRefusedEvenWhenNoGradientReachesIt() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("Sin", {"A"}, {"S"}),
      Node("Relu", {"B"}, {"Y"}),
  };
  const Shapes shapes = {{"A", {3}}, {"S", {3}}, {"B", {3}}, {"Y", {3}}};
  GraphBuilder b;
  CheckThrows<UnsupportedOpError>(
      [&] { BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"B"}); }, "Sin",
      "an unreachable unsupported op should still be refused");
}

// If this fails, a MatMul whose 1-D operand ONNX promotes and then squeezes
// back out would be differentiated as if the promotion never happened, giving
// a gradient of the wrong rank.
void AMatMulWithA1DOperandIsRefused() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("MatMul", {"A", "W"}, {"Y"})};
  const Shapes shapes = {{"A", {3}}, {"W", {3, 4}}, {"Y", {4}}};
  GraphBuilder b;
  CheckThrows<UnsupportedOpError>(
      [&] { BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"W"}); }, "1-D",
      "a MatMul with a 1-D operand should be refused");
}

// If this fails, a Reduce* whose axes moved to a tensor input at opset 13
// would have them *guessed* from the shapes when more than one guess fits --
// [3, 3] -> [3] can be either axis, and the two disagree about where the
// gradient goes.
void AmbiguousReducedAxesAreRefusedRatherThanGuessed() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("ReduceSum", {"A", "ax"}, {"Y"}, {IntAttr("keepdims", 0)})};
  const Shapes shapes = {{"A", {3, 3}}, {"Y", {3}}};
  GraphBuilder b;
  CheckThrows<UnsupportedOpError>(
      [&] { BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A"}); },
      "ambiguous", "ambiguous reduced axes should be refused");
}

// If this fails, the pre-opset-13 spelling (axes as an attribute) would go
// down the shape-recovery path it does not need, and would be refused for
// ambiguity on shapes where the attribute says exactly which axis it was.
void ReducedAxesComeFromTheAttributeWhenThereIsOne() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("ReduceSum", {"A"}, {"Y"},
           {IntsAttr("axes", {0}), IntAttr("keepdims", 0)})};
  const Shapes shapes = {{"A", {3, 3}}, {"Y", {3}}};
  GraphBuilder b;
  const std::map<std::string, std::string> grads =
      BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A"});
  Check(grads.size() == 1 && grads.count("A") == 1,
        "the attribute spelling should differentiate where the tensor "
        "spelling is ambiguous");
  // Reshape [3] back to the keepdims shape [1, 3], then broadcast by a
  // multiply -- never an Expand, which is outside the allowlist.
  const std::vector<std::string> expected = {"Reshape", "Mul"};
  Check(OpTypes(b) == expected,
        "a keepdims=0 reduction's gradient should reshape then broadcast by "
        "a multiply");
}

// If this fails, a [4, 3] gradient would land on a [3] parameter: ONNX
// carries the mismatch into the optimizer, which broadcasts again and takes
// four times the intended step. The Python module docstring calls this out as
// the one subtlety worth naming, so it is checked on its own here.
//
// BuildBackwardWithHandWrittenRules rather than plain BuildBackward: the
// broadcast-undoing this pins (ctx.ReduceTo) is identical code in both the
// hand-written and the templated "Add" rule -- only the identity-gradient
// core ahead of it differs, a Call node the templated rule adds and the
// hand-written one does not -- so pinning against the hand-written rule
// keeps this test about broadcasting, not about which rule produced `g`.
void ABroadcastGradientIsSummedBackToTheOperandShape() {
  const std::vector<onnx::NodeProto> nodes = {Node("Add", {"A", "B"}, {"Y"})};
  {
    // B is rank 1: the leading axis was replicated away entirely, so it is
    // summed with keepdims and then reshaped back down to [3].
    const Shapes shapes = {{"A", {4, 3}}, {"B", {3}}, {"Y", {4, 3}}};
    GraphBuilder b;
    BuildBackwardWithHandWrittenRules(b, nodes, shapes, {{"Y", "dY"}},
                                      {"A", "B"});
    const std::vector<std::string> expected = {"ReduceSum", "Reshape"};
    Check(OpTypes(b) == expected,
          "reducing a [4, 3] gradient to [3] should be ReduceSum then "
          "Reshape");
    Check(b.nodes().size() == 2 && b.nodes()[0].attribute_size() == 1 &&
              b.nodes()[0].attribute(0).name() == "keepdims" &&
              b.nodes()[0].attribute(0).i() == 1,
          "the un-broadcasting ReduceSum must keep dims -- ONNX cannot drop "
          "some axes and keep others in one node");
  }
  {
    // B is [1, 3]: the summed shape already *is* the target, so the Reshape
    // is not emitted. Emitting it anyway would be harmless numerically and
    // would still break name-counter parity with the Python.
    const Shapes shapes = {{"A", {4, 3}}, {"B", {1, 3}}, {"Y", {4, 3}}};
    GraphBuilder b;
    BuildBackwardWithHandWrittenRules(b, nodes, shapes, {{"Y", "dY"}},
                                      {"A", "B"});
    const std::vector<std::string> expected = {"ReduceSum"};
    Check(OpTypes(b) == expected,
          "reducing a [4, 3] gradient to [1, 3] needs no Reshape");
  }
  {
    // Same shapes on both sides: no node at all, and the gradient of each
    // operand is the seed itself.
    const Shapes shapes = {{"A", {4, 3}}, {"B", {4, 3}}, {"Y", {4, 3}}};
    GraphBuilder b;
    const std::map<std::string, std::string> grads =
        BuildBackwardWithHandWrittenRules(b, nodes, shapes, {{"Y", "dY"}},
                                          {"A", "B"});
    Check(b.nodes().empty(), "an un-broadcast Add should emit no nodes");
    Check(grads.at("A") == "dY" && grads.at("B") == "dY",
          "an un-broadcast Add's gradients are the seed itself");
  }
}

// If this fails, the C++ LayerNorm rule has drifted from
// _grad_layer_normalization in graph_grad.py -- and because a rule's nodes
// are numbered by the builder's counter in emission order, drifting by a
// *reordering* alone is enough to make every subsequent tensor in a
// browser-built step graph carry a different name from the Python's. So the
// whole emission is pinned here, node for node and in order, rather than just
// its op set.
void TheLayerNormRuleEmitsTheSameNodesInTheSameOrderAsThePython() {
  {
    GraphBuilder b;
    const std::map<std::string, std::string> grads =
        BuildBackward(b, LayerNormSlice(), LayerNormShapes(), {{"Y", "dY"}},
                      {"X", "S", "Bn"});
    // mu, xc, xc^2, var, var+eps, sqrt, inv, xhat, then gs, mean(gs),
    // gs*xhat, mean(gs*xhat), the two mean terms subtracted off and dx;
    // finally g*xhat and g summed back down to the parameters' own shapes.
    const std::vector<std::string> expected = {
        "ReduceMean", "Sub",       "Mul",    "ReduceMean", "Add", "Sqrt",
        "Div",        "Mul",       "Mul",    "ReduceMean", "Mul", "ReduceMean",
        "Sub",        "Mul",       "Sub",    "Mul",        "Mul", "ReduceSum",
        "Reshape",    "ReduceSum", "Reshape"};
    Check(OpTypes(b) == expected,
          "the LayerNorm rule should emit graph_grad.py's nodes in its order");
    Check(b.nodes().size() == expected.size() &&
              grads.at("X") == b.nodes()[15].output(0),
          "dx should be the Mul(inv, ...) that closes the dx expression");
    Check(grads.at("Bn") == b.nodes().back().output(0),
          "the bias gradient should be the last thing emitted");
    // The 1.0 of the reciprocal and epsilon are initializers, not nodes, and
    // they advance the same counter as the nodes do -- so they are checked
    // here too rather than only implied by the node list. The other four are
    // the axes/shape operands of the two broadcast-undoing reductions.
    Check(b.initializer().size() == 6 &&
              b.initializer()[0].data_type() == onnx::TensorProto::FLOAT &&
              b.initializer()[0].dims_size() == 0 &&
              b.initializer()[1].data_type() == onnx::TensorProto::FLOAT &&
              b.initializer()[1].dims_size() == 0,
          "the rule's first two constants are the scalars 1.0 and epsilon");

    // axes as an *attribute*: ReduceMean only moved them to an input at opset
    // 18, and the step graph is opset 17, where the input spelling fails the
    // checker outright ("input size 2 not in range [min=1, max=1]").
    for (const onnx::NodeProto& node : b.nodes()) {
      if (node.op_type() != "ReduceMean") continue;
      Check(node.input_size() == 1 && node.attribute_size() == 2 &&
                node.attribute(0).name() == "axes" &&
                node.attribute(0).ints_size() == 1 &&
                node.attribute(0).ints(0) == 2 &&
                node.attribute(1).name() == "keepdims" &&
                node.attribute(1).i() == 1,
            "the rule's ReduceMeans take axes [2] as an attribute, with "
            "keepdims, at opset 17");
    }
  }
  {
    // axis spans [axis, rank), so a non-default axis widens the reduction
    // rather than moving it.
    const std::vector<onnx::NodeProto> nodes = {Node(
        "LayerNormalization", {"X", "S", "Bn"}, {"Y"}, {IntAttr("axis", 1)})};
    const Shapes shapes = {
        {"X", {2, 3, 4}}, {"S", {3, 4}}, {"Bn", {3, 4}}, {"Y", {2, 3, 4}}};
    GraphBuilder b;
    BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"X", "S", "Bn"});
    Check(b.nodes()[0].op_type() == "ReduceMean" &&
              b.nodes()[0].attribute(0).ints_size() == 2 &&
              b.nodes()[0].attribute(0).ints(0) == 1 &&
              b.nodes()[0].attribute(0).ints(1) == 2,
          "axis = 1 should normalize over axes [1, 2]");
  }
  {
    // The bias is optional, and a real fused LayerNorm often omits it: one
    // gradient fewer, and the two nodes that would have reduced g away are
    // not emitted at all.
    const std::vector<onnx::NodeProto> nodes = {
        Node("LayerNormalization", {"X", "S"}, {"Y"})};
    const Shapes shapes = {{"X", {2, 3, 4}}, {"S", {4}}, {"Y", {2, 3, 4}}};
    GraphBuilder b;
    const std::map<std::string, std::string> grads =
        BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"X", "S"});
    Check(b.nodes().size() == 19 && b.nodes().back().op_type() == "Reshape",
          "a LayerNorm without a bias should emit two nodes fewer");
    Check(grads.size() == 2 && grads.count("Bn") == 0,
          "a LayerNorm without a bias has no third gradient");
  }
}

// If this fails, the C++ BatchNormalization rule has drifted from
// _grad_batch_normalization in graph_grad.py -- pinned the same way and for
// the same reason as the LayerNorm rule above.
// BuildBackwardWithHandWrittenRules rather than plain BuildBackward: Rules()
// now differentiates "BatchNormalization" via the checked-in template
// (GradBatchNormalizationTemplated, wired in for production -- see Rules()'s
// own comment), and this test's whole point is pinning the *hand-written*
// GradBatchNormalization's node sequence as an independent structural
// cross-check, the same reason graph_grad_templates_test.cpp reaches for the
// same override.
void TheBatchNormRuleEmitsTheSameNodesInTheSameOrderAsThePython() {
  GraphBuilder b("bw_");
  const std::map<std::string, std::string> grads =
      BuildBackwardWithHandWrittenRules(b, BatchNormSlice(), BatchNormShapes(),
                                        {{"Y", "dY"}},
                                        {"X", "S", "Bn", "Mn", "Vr"});
  // mean reshaped to [1, C, 1, 1] and subtracted (xc); var reshaped, +eps,
  // sqrt, inv; xhat = xc * inv; scale reshaped, gs = g * scale, dx = gs *
  // inv; g * xhat reduced over every axis but 1 for dscale; g reduced the
  // same way for db; dx reduced and negated for dmean; dx * xhat * inv
  // reduced and scaled by -0.5 for dvar (one more multiply by inv than
  // dscale's numerator needed, since the chain rule through sqrt produces
  // inv^3 rather than inv^2 -- see the Python docstring's note on this).
  const std::vector<std::string> expected = {
      "Reshape", "Sub",       "Reshape",   "Add",       "Sqrt",
      "Div",     "Mul",       "Reshape",   "Mul",       "Mul",
      "Mul",     "ReduceSum", "ReduceSum", "ReduceSum", "Neg",
      "Mul",     "Mul",       "ReduceSum", "Mul"};
  Check(OpTypes(b) == expected,
        "the BatchNormalization rule should emit graph_grad.py's nodes in "
        "its order");
  Check(b.nodes().size() == expected.size() &&
            grads.at("X") == b.nodes()[9].output(0),
        "dX should be the Mul(gs, inv) that closes the dx expression");
  Check(grads.at("S") == b.nodes()[11].output(0),
        "dscale should be the ReduceSum over g * xhat");
  Check(grads.at("Bn") == b.nodes()[12].output(0),
        "db should be the ReduceSum over g alone");
  Check(grads.at("Mn") == b.nodes()[14].output(0),
        "dmean should be the Neg of the reduced dx");
  Check(grads.at("Vr") == b.nodes().back().output(0),
        "dvar should be the last thing emitted");
  // Three broadcast shapes ([1, C, 1, 1] for mean, var and scale), the 1.0
  // numerator, epsilon, four reduce-axes lists (dscale/db/dmean/dvar) and
  // -0.5 -- ten initializers, none shared between uses, the same
  // one-initializer-per-call discipline GraphBuilder.const/int64_const
  // follow in the Python.
  Check(b.initializer().size() == 10, "the rule emits ten initializers");
  const std::set<std::string> emitted = OpTypeSet(b);
  Check(emitted.count("Conv") == 0 && emitted.count("MatMul") == 0 &&
            emitted.count("Expand") == 0,
        "BatchNormalization's gradient needs no weight and no matmul, and "
        "the per-channel broadcast is a Reshape plus an ordinary Mul rather "
        "than an Expand");
}

// The InstanceNormalization analogue of the BatchNormalization pin above.
void TheInstanceNormRuleEmitsTheSameNodesInTheSameOrderAsThePython() {
  GraphBuilder b("bw_");
  const std::map<std::string, std::string> grads =
      BuildBackward(b, InstanceNormSlice(), InstanceNormShapes(), {{"Y", "dY"}},
                    {"X", "S", "Bn"});
  // scale reshaped to [1, C, 1, 1] up front; mu, xc, xc^2, var (both
  // ReduceMeans over the spatial axes only), var+eps, sqrt, inv, xhat --
  // exactly LayerNorm's own sequence with the reduced axes changed; then gs,
  // mean(gs), gs*xhat, mean(gs*xhat), the two mean terms subtracted off and
  // dx; finally g*xhat and g reduced over batch and spatial together for
  // dscale/db.
  const std::vector<std::string> expected = {
      "Reshape",    "ReduceMean", "Sub",        "Mul",       "ReduceMean",
      "Add",        "Sqrt",       "Div",        "Mul",       "Mul",
      "ReduceMean", "Mul",        "ReduceMean", "Sub",       "Mul",
      "Sub",        "Mul",        "Mul",        "ReduceSum", "ReduceSum"};
  Check(OpTypes(b) == expected,
        "the InstanceNormalization rule should emit graph_grad.py's nodes "
        "in its order");
  Check(b.nodes().size() == expected.size() &&
            grads.at("X") == b.nodes()[16].output(0),
        "dX should be the Mul(inv, ...) that closes the dx expression");
  Check(grads.at("S") == b.nodes()[18].output(0),
        "dscale should be the ReduceSum over g * xhat");
  Check(grads.at("Bn") == b.nodes().back().output(0),
        "db should be the last thing emitted");
  Check(b.initializer().size() == 5,
        "the rule emits five initializers: the [1, C, 1, 1] broadcast "
        "shape, the 1.0 numerator, epsilon, and the two reduce-axes lists");
  const std::set<std::string> emitted = OpTypeSet(b);
  Check(emitted.count("Expand") == 0 && emitted.count("Conv") == 0 &&
            emitted.count("MatMul") == 0,
        "InstanceNormalization's gradient needs no weight, no matmul and no "
        "Expand");
}

// If this fails, a BatchNormalization whose rank or per-channel operand
// shapes don't add up would be differentiated against a geometry it
// invented -- the same hazard APoolGeometryThatDoesNotResolveIsRefused
// checks for MaxPool/AveragePool.
void ABatchNormGeometryThatDoesNotAddUpIsRefused() {
  {
    // No batch or channel axis at all.
    const Shapes shapes = {{"X", {3}},  {"S", {3}},  {"Bn", {3}},
                           {"Mn", {3}}, {"Vr", {3}}, {"Y", {3}}};
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(b, BatchNormSlice(), shapes, {{"Y", "dY"}}, {"X"});
        },
        "channel axis",
        "BatchNormalization with no batch or channel axis is refused");
  }
  {
    // scale's shape does not match X's channel count.
    const Shapes shapes = {{"X", {1, 3, 4, 4}}, {"S", {2}},
                           {"Bn", {3}},         {"Mn", {3}},
                           {"Vr", {3}},         {"Y", {1, 3, 4, 4}}};
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(b, BatchNormSlice(), shapes, {{"Y", "dY"}}, {"X"});
        },
        "scale has shape",
        "BatchNormalization with a mismatched scale is refused");
  }
  {
    // training_mode=1 -- refused by BuildBackward's own single-output check
    // before this rule ever runs, since the spec requires three outputs
    // whenever training_mode=1. There is no spec-conformant one-output
    // training_mode=1 graph to construct, which is exactly why this rule
    // carries no training_mode check of its own -- see the rule's comment.
    const Shapes shapes = {
        {"X", {1, 2, 3, 3}}, {"S", {2}},          {"Bn", {2}}, {"Mn", {2}},
        {"Vr", {2}},         {"Y", {1, 2, 3, 3}}, {"RM", {2}}, {"RV", {2}}};
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(
              b,
              {Node("BatchNormalization", {"X", "S", "Bn", "Mn", "Vr"},
                    {"Y", "RM", "RV"}, {IntAttr("training_mode", 1)})},
              shapes, {{"Y", "dY"}}, {"X"});
        },
        "3 outputs", "BatchNormalization with training_mode=1 is refused");
  }
}

// The InstanceNormalization analogue of the refusals above.
void AnInstanceNormGeometryThatDoesNotAddUpIsRefused() {
  {
    // No spatial axis at all (rank 2: batch and channel only).
    const Shapes shapes = {
        {"X", {4, 3}}, {"S", {3}}, {"Bn", {3}}, {"Y", {4, 3}}};
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(b, InstanceNormSlice(), shapes, {{"Y", "dY"}}, {"X"});
        },
        "spatial dimension",
        "InstanceNormalization with no spatial axis is refused");
  }
  {
    const Shapes shapes = {
        {"X", {1, 3, 4, 4}}, {"S", {2}}, {"Bn", {3}}, {"Y", {1, 3, 4, 4}}};
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(b, InstanceNormSlice(), shapes, {{"Y", "dY"}}, {"X"});
        },
        "scale has shape",
        "InstanceNormalization with a mismatched scale is refused");
  }
}

// Little-endian readers for an initializer's raw_data. GraphBuilder writes
// tensors little-endian on purpose (see AppendLittleEndian), so a big-endian
// host has to decode rather than reinterpret -- the same care
// docs/big-endian.md asks of everything else that touches raw_data.
std::vector<int64_t> Int64Data(const onnx::TensorProto& tensor) {
  std::vector<int64_t> values;
  const std::string& raw = tensor.raw_data();
  for (size_t i = 0; i + 8 <= raw.size(); i += 8) {
    uint64_t bits = 0;
    for (int k = 7; k >= 0; --k) {
      bits = (bits << 8) | static_cast<unsigned char>(raw[i + k]);
    }
    values.push_back(static_cast<int64_t>(bits));
  }
  return values;
}

std::vector<float> FloatData(const onnx::TensorProto& tensor) {
  std::vector<float> values;
  const std::string& raw = tensor.raw_data();
  for (size_t i = 0; i + 4 <= raw.size(); i += 4) {
    uint32_t bits = 0;
    for (int k = 3; k >= 0; --k) {
      bits = (bits << 8) | static_cast<unsigned char>(raw[i + k]);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    values.push_back(value);
  }
  return values;
}

// If this fails, the C++ Conv rule has drifted from _grad_conv in
// graph_grad.py. Pinned the same way and for the same reason as the
// LayerNorm rule above: the nodes are numbered by the builder's counter in
// emission order, so a reordering alone renames every tensor a browser-built
// step graph produces from there on.
void TheConvRuleEmitsTheSameNodesInTheSameOrderAsThePython() {
  GraphBuilder b("bw_");
  const std::map<std::string, std::string> grads = BuildBackward(
      b, ConvSlice(), ConvShapes(), {{"Y", "dY"}}, {"X", "Wc", "Bc"});
  // dY reshaped with the group axis split out; then dX -- gather dY per
  // kernel tap, mask the taps that fall outside, matmul against W laid out
  // [group, C/group, (M/group)*taps]; then dW -- im2col of X, matmul against
  // dY, sum over the batch; then dB, dY summed over batch and positions.
  const std::vector<std::string> expected = {
      "Reshape",   "Gather",  "Mul",       "Reshape",   "Reshape",
      "Transpose", "Reshape", "MatMul",    "Reshape",   "Reshape",
      "Gather",    "Mul",     "Reshape",   "Transpose", "MatMul",
      "ReduceSum", "Reshape", "ReduceSum", "Reshape"};
  Check(OpTypes(b) == expected,
        "the Conv rule should emit graph_grad.py's nodes in its order");
  Check(b.nodes().size() == expected.size() &&
            grads.at("X") == b.nodes()[8].output(0) &&
            grads.at("Wc") == b.nodes()[16].output(0) &&
            grads.at("Bc") == b.nodes().back().output(0),
        "dX, dW and dB should be the three Reshapes that close each half");
  // Neither Conv nor ConvTranspose: the whole point of writing the gradient
  // as im2col is that the emitted graph needs no convolution kernel on the
  // backend. See the note beside EpFriendlyOps() in qat_graph_builder.h.
  const std::set<std::string> emitted = OpTypeSet(b);
  Check(emitted.count("Conv") == 0 && emitted.count("ConvTranspose") == 0,
        "a Conv's gradient must not itself contain a convolution");
  for (const onnx::NodeProto& node : b.nodes()) {
    if (node.op_type() != "Gather") continue;
    Check(node.attribute_size() == 1 && node.attribute(0).name() == "axis" &&
              node.attribute(0).i() == 3,
          "both gathers run along axis 3, the flattened spatial axis");
  }
}

// If this fails, the index tables the two gathers read from have been
// computed differently here from the Python -- which no allowlist or op-order
// check would catch, because the graph would have exactly the same shape and
// simply read the wrong elements. The case is small enough to write the
// answer out by hand: a 1-D convolution of a length-3 input by a width-2
// kernel, padded on both sides, whose output is length 4.
void TheConvIndexTablesAreTheOnesTheGeometryImplies() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("Conv", {"X", "Wc"}, {"Y"}, {IntsAttr("pads", {1, 1})})};
  const Shapes shapes = {{"X", {1, 1, 3}}, {"Wc", {1, 1, 2}}, {"Y", {1, 1, 4}}};
  GraphBuilder b("bw_");
  BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"X", "Wc"});

  // dX's gather: for tap t, input position p came from output position
  // p + 1 - t. Tap 0 reads outputs 1, 2, 3; tap 1 reads outputs 0, 1, 2.
  Check(b.initializer().size() == 13, "the rule emits thirteen initializers");
  Check(
      Int64Data(b.initializer()[1]) == std::vector<int64_t>({1, 2, 3, 0, 1, 2}),
      "dX gathers output position p + 1 - t for tap t");
  Check(FloatData(b.initializer()[2]) == std::vector<float>({1, 1, 1, 1, 1, 1}),
        "with a width-2 kernel and one pad each side, every input position "
        "is reached by both taps");

  // dW's im2col: for tap t, output position o reads input o - 1 + t. Tap 0's
  // first read and tap 1's last are the padding, and are masked away rather
  // than gathered from out of range.
  Check(Int64Data(b.initializer()[8]) ==
            std::vector<int64_t>({0, 0, 1, 2, 0, 1, 2, 0}),
        "im2col reads input position o - 1 + t, clamped to 0 where the tap "
        "falls in the padding");
  Check(FloatData(b.initializer()[9]) ==
            std::vector<float>({0, 1, 1, 1, 1, 1, 1, 0}),
        "the mask is 0 exactly where the tap read padding");
}

// If this fails, a convolution the rule cannot invert would be
// differentiated against a geometry it invented, which is the one failure
// mode that produces a plausible-looking wrong gradient rather than an
// error. Each of these is a refusal graph_grad.py makes by the same name.
void AConvWhoseGeometryDoesNotAddUpIsRefused() {
  const Shapes ok = {
      {"X", {1, 2, 5, 5}}, {"Wc", {2, 2, 3, 3}}, {"Y", {1, 2, 3, 3}}};
  {
    // The output shape the node declares has to follow from the attributes;
    // if it does not, one of the two is being misread and neither can be
    // trusted.
    const Shapes wrong = {
        {"X", {1, 2, 5, 5}}, {"Wc", {2, 2, 3, 3}}, {"Y", {1, 2, 4, 4}}};
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(b, {Node("Conv", {"X", "Wc"}, {"Y"})}, wrong,
                        {{"Y", "dY"}}, {"X"});
        },
        "does not follow from", "an inconsistent output shape is refused");
  }
  {
    // "SAME" is not a spelling ONNX has; a typo in the attribute must not
    // silently fall back to no padding.
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(
              b,
              {Node("Conv", {"X", "Wc"}, {"Y"}, {StrAttr("auto_pad", "SAME")})},
              ok, {{"Y", "dY"}}, {"X"});
        },
        "auto_pad", "an unknown auto_pad is refused by name");
  }
  {
    // A group count that does not divide the channels describes no
    // convolution at all.
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(
              b, {Node("Conv", {"X", "Wc"}, {"Y"}, {IntAttr("group", 3)})}, ok,
              {{"Y", "dY"}}, {"X"});
        },
        "group=3", "a group that does not divide the channels is refused");
  }
  {
    // A kernel_shape attribute that disagrees with W's own spatial shape:
    // the two say different things about what the node computes.
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(b,
                        {Node("Conv", {"X", "Wc"}, {"Y"},
                              {IntsAttr("kernel_shape", {2, 2})})},
                        ok, {{"Y", "dY"}}, {"X"});
        },
        "kernel_shape", "a kernel_shape that disagrees with W is refused");
  }
  {
    // No spatial axis at all: a rank-2 "convolution" is a matrix product
    // spelled wrong, and this rule will not guess which.
    const Shapes flat = {{"X", {2, 3}}, {"Wc", {3, 4}}, {"Y", {2, 4}}};
    GraphBuilder b;
    CheckThrows<UnsupportedOpError>(
        [&] {
          BuildBackward(b, {Node("Conv", {"X", "Wc"}, {"Y"})}, flat,
                        {{"Y", "dY"}}, {"X"});
        },
        "spatial dimension", "a Conv with no spatial axis is refused");
  }
}

// If this fails, the rule has grown a special case for the number of spatial
// dimensions -- and since WebNN has conv2d and convTranspose2d and no other
// convolution, a rank-dependent rule is exactly what writing this backward
// as a convolution would have forced. The 1-D and 3-D cases must emit the
// same nodes as the 2-D one, differing only in the tables' contents.
void ConvsOfAnyRankDifferentiateIdentically() {
  const std::vector<std::string> expected = {
      "Reshape", "Gather",    "Mul",     "Reshape",   "Reshape", "Transpose",
      "Reshape", "MatMul",    "Reshape", "Reshape",   "Gather",  "Mul",
      "Reshape", "Transpose", "MatMul",  "ReduceSum", "Reshape"};
  {
    const Shapes shapes = {
        {"X", {1, 2, 7}}, {"Wc", {3, 2, 3}}, {"Y", {1, 3, 3}}};
    GraphBuilder b;
    BuildBackward(
        b, {Node("Conv", {"X", "Wc"}, {"Y"}, {IntsAttr("strides", {2})})},
        shapes, {{"Y", "dY"}}, {"X", "Wc"});
    Check(OpTypes(b) == expected, "a 1-D convolution emits the same nodes");
  }
  {
    const Shapes shapes = {{"X", {1, 1, 3, 3, 3}},
                           {"Wc", {2, 1, 2, 2, 2}},
                           {"Y", {1, 2, 2, 2, 2}}};
    GraphBuilder b;
    BuildBackward(b, {Node("Conv", {"X", "Wc"}, {"Y"})}, shapes, {{"Y", "dY"}},
                  {"X", "Wc"});
    Check(OpTypes(b) == expected, "a 3-D convolution emits the same nodes");
  }
}

// If this fails, the C++ Gather rule has drifted from _grad_gather in
// graph_grad.py -- pinned the same way and for the same reason as the
// LayerNorm and Conv rules above: the nodes are numbered by the builder's
// counter in emission order, so a reordering alone renames every tensor a
// browser-built step graph produces from there on. axis=1 on a rank-3 data
// tensor exercises a non-trivial pre/post split around the gathered axis,
// not just the plain [N, D] embedding-table case.
void TheGatherRuleEmitsTheSameNodesInTheSameOrderAsThePython() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("Gather", {"data", "idx"}, {"Y"}, {IntAttr("axis", 1)})};
  const Shapes shapes = {{"data", {2, 5, 3}}, {"idx", {2}}, {"Y", {2, 2, 3}}};
  GraphBuilder b;
  const std::map<std::string, std::string> grads =
      BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"data"});

  // Flatten and resolve indices (Reshape, Cast, Less, Cast, Mul, Add); build
  // the one-hot mask (Reshape, Reshape, Greater, Cast, Sub, Less, Cast, Sub,
  // Mul); then the batched one-hot matmul that stands in for the scatter-add
  // (Transpose, Reshape, Reshape, MatMul, Reshape).
  const std::vector<std::string> expected = {
      "Reshape",   "Cast",    "Less",    "Cast",    "Mul",
      "Add",       "Reshape", "Reshape", "Greater", "Cast",
      "Sub",       "Less",    "Cast",    "Sub",     "Mul",
      "Transpose", "Reshape", "Reshape", "MatMul",  "Reshape"};
  Check(OpTypes(b) == expected,
        "the Gather rule should emit graph_grad.py's nodes in its order");
  Check(grads.size() == 1 && grads.at("data") == b.nodes().back().output(0),
        "dData should be the last Reshape emitted");

  Check(b.initializer().size() == 11,
        "the rule should emit 11 initializers: the index-resolution "
        "constants, the arange, and four reshape-shape operands");
  const std::vector<onnx::TensorProto>& init = b.initializer();
  Check(Int64Data(init[0]) == std::vector<int64_t>{2},
        "the flat-index reshape target should be [length] = [2]");
  Check(FloatData(init[1]) == std::vector<float>{0.0f} &&
            FloatData(init[2]) == std::vector<float>{5.0f},
        "the resolved-index constants should be 0.0 (the negativity bound) "
        "and count = data.shape[axis] = 5");
  Check(Int64Data(init[3]) == std::vector<int64_t>({2, 1}),
        "the one-hot column reshape should be [length, 1]");
  Check(FloatData(init[4]) == std::vector<float>({0, 1, 2, 3, 4}),
        "the arange constant should be 0..count-1");
  Check(Int64Data(init[5]) == std::vector<int64_t>({1, 5}),
        "the one-hot row reshape should be [1, count]");
  Check(FloatData(init[6]) == std::vector<float>{1.0f} &&
            FloatData(init[7]) == std::vector<float>{1.0f},
        "the two-sided one-hot mask should subtract from 1.0 twice");
  Check(Int64Data(init[8]) == std::vector<int64_t>({1, 5, 2}),
        "the batched one-hot reshape should be [1, count, length]");
  Check(Int64Data(init[9]) == std::vector<int64_t>({2, 2, 3}),
        "dY's flattening reshape should be [pre, length, post] = [2, 2, 3]");
  Check(Int64Data(init.back()) == std::vector<int64_t>({2, 5, 3}),
        "the final reshape should restore data's own shape");

  // indices takes no gradient at all -- neither a node nor an entry.
  Check(grads.count("idx") == 0,
        "Gather's indices input should have no gradient");
}

// If this fails, a batched index tensor (e.g. [batch, seq]) would be
// silently mixed across batch elements by a reshape that only makes sense
// for a scalar or a rank-1 sequence of lookups -- see _grad_gather in
// graph_grad.py for why this is refused rather than risked.
void AGatherWithRankTwoIndicesIsRefused() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("Gather", {"data", "idx"}, {"Y"})};
  const Shapes shapes = {{"data", {5, 3}}, {"idx", {2, 2}}, {"Y", {2, 2, 3}}};
  GraphBuilder b;
  CheckThrows<UnsupportedOpError>(
      [&] { BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"data"}); },
      "rank-2", "a rank-2 index tensor should be refused");
}

// If this fails, the C++ AveragePool rule has drifted from
// _grad_averagepool in graph_grad.py. Pinned the same way and for the same
// reason as the Conv rule above.
void TheAveragePoolRuleEmitsTheSameNodesInTheSameOrderAsThePython() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("AveragePool", {"X"}, {"Y"},
           {IntsAttr("kernel_shape", {2, 2}), IntsAttr("strides", {2, 2})})};
  const Shapes shapes = {{"X", {1, 1, 4, 4}}, {"Y", {1, 1, 2, 2}}};
  GraphBuilder b("bw_");
  const std::map<std::string, std::string> grads =
      BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"X"});
  // dY reshaped to [planes, out_count], scaled by the per-window divisor,
  // then col2im: one gather of the scaled gradient per kernel tap, masked,
  // and summed over taps -- Conv's dX half with the MatMul against W deleted
  // outright, since pooling has no weight to sum over.
  const std::vector<std::string> expected = {
      "Reshape", "Mul", "Gather", "Mul", "Reshape", "ReduceSum", "Reshape"};
  Check(OpTypes(b) == expected,
        "the AveragePool rule should emit graph_grad.py's nodes in its "
        "order");
  Check(b.initializer().size() == 7,
        "the rule emits seven initializers: the reshape-to-[planes,"
        "out_count] shape, the divisor, the gather index, the mask, the "
        "col shape, the reduce axes and the final output shape");
  Check(grads.at("X") == b.nodes().back().output(0),
        "dX should be the last Reshape's output");
  const std::set<std::string> emitted = OpTypeSet(b);
  Check(emitted.count("Conv") == 0 && emitted.count("MatMul") == 0,
        "AveragePool's gradient needs no weight, so unlike Conv's dX it "
        "emits no MatMul");
}

// The same shape of pin as the AveragePool one above, for _grad_maxpool. The
// sequence is Conv's dW-style windowing gather of X, the eq mask built from
// Greater/Less/Cast with no Equal and no fresh ReduceMax, the tie count and
// its Div, then the per-tap-offset scatter this rule needs that
// AveragePool's uniform-per-tap gradient does not.
void TheMaxPoolRuleEmitsTheSameNodesInTheSameOrderAsThePython() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("MaxPool", {"X"}, {"Y"},
           {IntsAttr("kernel_shape", {2, 2}), IntsAttr("strides", {2, 2})})};
  const Shapes shapes = {{"X", {1, 1, 4, 4}}, {"Y", {1, 1, 2, 2}}};
  GraphBuilder b("bw_");
  const std::map<std::string, std::string> grads =
      BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"X"});
  const std::vector<std::string> expected = {
      "Reshape",   "Gather",  "Reshape",   "Reshape", "Greater", "Cast",
      "Sub",       "Less",    "Cast",      "Sub",     "Mul",     "Mul",
      "ReduceSum", "Div",     "Reshape",   "Mul",     "Reshape", "Gather",
      "Mul",       "Reshape", "ReduceSum", "Reshape"};
  Check(OpTypes(b) == expected,
        "the MaxPool rule should emit graph_grad.py's nodes in its order");
  Check(b.initializer().size() == 15, "the rule emits fifteen initializers");
  Check(grads.at("X") == b.nodes().back().output(0),
        "dX should be the last Reshape's output");
  const std::set<std::string> emitted = OpTypeSet(b);
  Check(emitted.count("Equal") == 0,
        "the eq mask is built from Greater/Less/Cast, never Equal");
  Check(emitted.count("Conv") == 0 && emitted.count("MatMul") == 0 &&
            emitted.count("ReduceMax") == 0,
        "MaxPool's gradient needs no weight and reuses the forward's own "
        "output instead of a fresh ReduceMax");
}

// If this fails, a pooling geometry this rule cannot invert would be
// differentiated against one it invented instead -- the same hazard
// AConvWhoseGeometryDoesNotAddUpIsRefused checks for Conv, and the same
// refusals PoolGeometryOf makes by the same name as _pool_geometry.
void APoolGeometryThatDoesNotResolveIsRefused() {
  for (const std::string& op :
       {std::string("MaxPool"), std::string("AveragePool")}) {
    {
      // kernel_shape is required for both -- unlike Conv, there is no
      // weight tensor to read it off.
      const Shapes shapes = {{"X", {1, 1, 4, 4}}, {"Y", {1, 1, 2, 2}}};
      GraphBuilder b;
      CheckThrows<UnsupportedOpError>(
          [&] {
            BuildBackward(b, {Node(op, {"X"}, {"Y"})}, shapes, {{"Y", "dY"}},
                          {"X"});
          },
          "kernel_shape", op + " without kernel_shape is refused");
    }
    {
      // ceil_mode=1 is refused outright rather than reproduced -- see
      // PoolGeometryOf's comment for why.
      const Shapes shapes = {{"X", {1, 1, 5, 5}}, {"Y", {1, 1, 3, 3}}};
      GraphBuilder b;
      CheckThrows<UnsupportedOpError>(
          [&] {
            BuildBackward(
                b,
                {Node(op, {"X"}, {"Y"},
                      {IntsAttr("kernel_shape", {2, 2}),
                       IntsAttr("strides", {2, 2}), IntAttr("ceil_mode", 1)})},
                shapes, {{"Y", "dY"}}, {"X"});
          },
          "ceil_mode", op + " with ceil_mode=1 is refused");
    }
    {
      // A typo'd auto_pad must not silently fall back to no padding.
      const Shapes shapes = {{"X", {1, 1, 5, 5}}, {"Y", {1, 1, 3, 3}}};
      GraphBuilder b;
      CheckThrows<UnsupportedOpError>(
          [&] {
            BuildBackward(b,
                          {Node(op, {"X"}, {"Y"},
                                {IntsAttr("kernel_shape", {3, 3}),
                                 IntsAttr("strides", {2, 2}),
                                 StrAttr("auto_pad", "SAME")})},
                          shapes, {{"Y", "dY"}}, {"X"});
          },
          "auto_pad", op + " with an unknown auto_pad is refused");
    }
    {
      // The declared output shape has to follow from the attributes.
      const Shapes shapes = {{"X", {1, 1, 5, 5}}, {"Y", {1, 1, 4, 4}}};
      GraphBuilder b;
      CheckThrows<UnsupportedOpError>(
          [&] {
            BuildBackward(b,
                          {Node(op, {"X"}, {"Y"},
                                {IntsAttr("kernel_shape", {3, 3}),
                                 IntsAttr("strides", {2, 2})})},
                          shapes, {{"Y", "dY"}}, {"X"});
          },
          "does not follow from",
          op + " with an inconsistent output shape is refused");
    }
    {
      // A window that is entirely padding: kernel_shape=[1] against a
      // length-1 input padded by 1 on each side puts the first and last of
      // three output windows fully in the padding, where neither this
      // rule's divisor (AveragePool) nor its tie count (MaxPool) is
      // defined.
      const Shapes shapes = {{"X", {1, 1, 1}}, {"Y", {1, 1, 3}}};
      GraphBuilder b;
      CheckThrows<UnsupportedOpError>(
          [&] {
            BuildBackward(b,
                          {Node(op, {"X"}, {"Y"},
                                {IntsAttr("kernel_shape", {1}),
                                 IntsAttr("pads", {1, 1})})},
                          shapes, {{"Y", "dY"}}, {"X"});
          },
          "no non-padding elements",
          op + " with a window that is entirely padding is refused");
    }
  }
}

// If this fails, a tensor read by two consumers -- a residual connection's
// own input, which is why this matters -- would keep only one contribution,
// and the parameter upstream of it would train on a fraction of its
// gradient.
void ATensorReadTwiceAccumulatesItsContributions() {
  const std::vector<onnx::NodeProto> nodes = {Node("Mul", {"A", "A"}, {"Y"})};
  const Shapes shapes = {{"A", {2, 3}}, {"Y", {2, 3}}};
  GraphBuilder b;
  const std::map<std::string, std::string> grads =
      BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A"});
  const std::vector<std::string> expected = {"Mul", "Mul", "Add"};
  Check(OpTypes(b) == expected,
        "Mul(A, A) should contribute twice to A and sum the two");
  Check(grads.at("A") == b.nodes().back().output(0),
        "the returned gradient should be the accumulated sum, not one of its "
        "two halves");
}

// If this fails, an alias would cost a copy in every emitted graph -- and,
// worse, would advance the builder's name counter, so the C++ and Python
// emitters would number every subsequent tensor differently.
void AnIdentityAliasesTheSeedInsteadOfEmittingANode() {
  const std::vector<onnx::NodeProto> nodes = {Node("Identity", {"A"}, {"Y"})};
  const Shapes shapes = {{"A", {2, 3}}, {"Y", {2, 3}}};
  GraphBuilder b;
  const std::map<std::string, std::string> grads =
      BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A"});
  Check(b.nodes().empty(), "Identity's rule should emit nothing");
  Check(grads.at("A") == "dY",
        "Identity's gradient should be the seed tensor itself");
}

// If this fails, the caller could not tell which of its targets it actually
// got: an extra key would be a gradient nobody asked for, a missing one a
// parameter that silently never trains.
void TheReturnedMapCoversExactlyTheRequestedTargets() {
  const std::vector<std::string> targets = {"A", "C"};
  GraphBuilder b;
  const std::map<std::string, std::string> grads = BuildBackward(
      b, ElementwiseSlice(), ElementwiseShapes(), {{"Y", "dY"}}, targets);
  Check(grads.size() == targets.size(),
        "the result should hold one entry per requested target");
  for (const std::string& target : targets) {
    Check(grads.count(target) == 1,
          "the result should hold a gradient for '" + target + "'");
    Check(!grads.at(target).empty(),
          "the gradient named for '" + target + "' should not be empty");
  }
  Check(grads.count("B") == 0,
        "the result should not hold gradients that were not asked for");
}

// If this fails, a target the slice does not actually reach would come back
// as a zero (or as nothing at all), and a mis-chosen slice or target list --
// which is what an unreachable target almost always means -- would look like
// a parameter that simply does not want to move.
void ATargetNoGradientReachesIsReportedRatherThanSilentlyMissing() {
  GraphBuilder b;
  CheckThrows<std::invalid_argument>(
      [&] {
        BuildBackward(b, ElementwiseSlice(), ElementwiseShapes(), {{"Y", "dY"}},
                      {"A", "not_in_the_slice"});
      },
      "no gradient reaches", "a target no gradient reaches should be reported");
}

// If this fails, a rule would be silently differentiating against a shape it
// invented, which is exactly the class of error the shapes argument exists to
// make impossible.
void AMissingShapeIsReportedRatherThanAssumed() {
  const std::vector<onnx::NodeProto> nodes = {Node("Add", {"A", "B"}, {"Y"})};
  const Shapes shapes = {{"A", {4, 3}}, {"Y", {4, 3}}};  // B's shape missing
  GraphBuilder b;
  CheckThrows<std::invalid_argument>(
      [&] { BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A"}); },
      "no static shape",
      "a tensor with no shape given should be reported by name");
}

// If this fails, a multi-output node would have its rule applied to the
// gradient of its first output alone, quietly dropping the rest.
void AMultiOutputNodeIsRefused() {
  onnx::NodeProto node = Node("Clip", {"A", "lo", "hi"}, {"Y", "extra"});
  const Shapes shapes = {{"A", {2, 3}}, {"Y", {2, 3}}};
  GraphBuilder b;
  CheckThrows<UnsupportedOpError>(
      [&] { BuildBackward(b, {node}, shapes, {{"Y", "dY"}}, {"A"}); },
      "single-output", "a multi-output node should be refused");
}

}  // namespace

int main() {
  TheSupportedOpsAreExactlyThePythonRuleTable();
  TheBackwardOpsAreThePythonAllowlistAndSitInsideEpFriendlyOps();
  TheEmittedBackwardStaysInsideTheOperatorAllowlist();
  AnOpWithNoRuleIsRefusedByName();
  AnUnsupportedOpIsRefusedEvenWhenNoGradientReachesIt();
  AMatMulWithA1DOperandIsRefused();
  AmbiguousReducedAxesAreRefusedRatherThanGuessed();
  ReducedAxesComeFromTheAttributeWhenThereIsOne();
  ABroadcastGradientIsSummedBackToTheOperandShape();
  TheLayerNormRuleEmitsTheSameNodesInTheSameOrderAsThePython();
  TheBatchNormRuleEmitsTheSameNodesInTheSameOrderAsThePython();
  TheInstanceNormRuleEmitsTheSameNodesInTheSameOrderAsThePython();
  ABatchNormGeometryThatDoesNotAddUpIsRefused();
  AnInstanceNormGeometryThatDoesNotAddUpIsRefused();
  TheConvRuleEmitsTheSameNodesInTheSameOrderAsThePython();
  TheConvIndexTablesAreTheOnesTheGeometryImplies();
  AConvWhoseGeometryDoesNotAddUpIsRefused();
  ConvsOfAnyRankDifferentiateIdentically();
  TheGatherRuleEmitsTheSameNodesInTheSameOrderAsThePython();
  AGatherWithRankTwoIndicesIsRefused();
  TheAveragePoolRuleEmitsTheSameNodesInTheSameOrderAsThePython();
  TheMaxPoolRuleEmitsTheSameNodesInTheSameOrderAsThePython();
  APoolGeometryThatDoesNotResolveIsRefused();
  ATensorReadTwiceAccumulatesItsContributions();
  AnIdentityAliasesTheSeedInsteadOfEmittingANode();
  TheReturnedMapCoversExactlyTheRequestedTargets();
  ATargetNoGradientReachesIsReportedRatherThanSilentlyMissing();
  AMissingShapeIsReportedRatherThanAssumed();
  AMultiOutputNodeIsRefused();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all graph_grad checks passed\n");
  return 0;
}
