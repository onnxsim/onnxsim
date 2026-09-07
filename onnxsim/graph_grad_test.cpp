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
 * stay inside, and the structural shape of the two things the Python
 * docstrings single out as easiest to get subtly wrong -- undoing a broadcast
 * and accumulating a tensor read more than once.
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

std::string Join(const std::set<std::string>& values) {
  std::string out;
  for (const std::string& value : values) {
    if (!out.empty()) out += ", ";
    out += value;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The four forward slices the allowlist test differentiates. Between them
// they reach every rule in the table; individually they are small enough that
// a failure names a rule rather than a graph.
// ---------------------------------------------------------------------------

// Broadcasting elementwise arithmetic, plus a tensor read twice.
std::vector<onnx::NodeProto> ElementwiseSlice() {
  return {
      Node("Sub", {"A", "B"}, {"T0"}),
      Node("Div", {"T0", "C"}, {"T1"}),
      Node("Mul", {"T1", "T1"}, {"Y"}),
  };
}

Shapes ElementwiseShapes() {
  return {{"A", {4, 3}},  {"B", {3}},     {"C", {4, 3}},
          {"T0", {4, 3}}, {"T1", {4, 3}}, {"Y", {4, 3}}};
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// If this fails, the two differentiators disagree about which ops they will
// touch at all: a caller's block discovery (which is told to test against
// SupportedOps rather than to catch refusals) would admit a block in the
// browser that the Python refuses, or the reverse.
void TheSupportedOpsAreExactlyThePythonRuleTable() {
  const std::set<std::string> expected = {
      "Add",        "Clip",      "Div",    "Erf",     "Exp",
      "Gemm",       "Identity",  "MatMul", "Mul",     "Neg",
      "ReduceMean", "ReduceSum", "Relu",   "Reshape", "Sigmoid",
      "Softmax",    "Sqrt",      "Sub",    "Tanh",    "Transpose"};
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
      "Add", "Cast", "Div",       "Exp",     "Greater", "Less",     "MatMul",
      "Mul", "Neg",  "ReduceSum", "Reshape", "Sub",     "Transpose"};
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
  };

  std::set<std::string> emitted;
  for (const Slice& slice : slices) {
    GraphBuilder b("bw_");
    BuildBackward(b, slice.nodes, slice.shapes, {{"Y", "dY"}}, slice.targets);
    const std::set<std::string> types = OpTypeSet(b);
    for (const std::string& op : types) {
      Check(BackwardOps().count(op) != 0,
            "backward graph reached outside the allowlist: " + op);
    }
    emitted.insert(types.begin(), types.end());
  }
  Check(emitted == BackwardOps(),
        "the four slices together should emit every allowlisted op and no "
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
void ABroadcastGradientIsSummedBackToTheOperandShape() {
  const std::vector<onnx::NodeProto> nodes = {Node("Add", {"A", "B"}, {"Y"})};
  {
    // B is rank 1: the leading axis was replicated away entirely, so it is
    // summed with keepdims and then reshaped back down to [3].
    const Shapes shapes = {{"A", {4, 3}}, {"B", {3}}, {"Y", {4, 3}}};
    GraphBuilder b;
    BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A", "B"});
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
    BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A", "B"});
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
        BuildBackward(b, nodes, shapes, {{"Y", "dY"}}, {"A", "B"});
    Check(b.nodes().empty(), "an un-broadcast Add should emit no nodes");
    Check(grads.at("A") == "dY" && grads.at("B") == "dY",
          "an un-broadcast Add's gradients are the seed itself");
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
