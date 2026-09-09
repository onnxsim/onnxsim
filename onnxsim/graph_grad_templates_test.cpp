/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises graph_grad.cpp's "Templated rules (proof of concept)" section:
 * GradAddTemplated/GradBatchNormalizationTemplated, which call into the
 * checked-in onnxscript-compiled FunctionProto templates
 * (graph_grad_templates_gen.h) via GraphBuilder::Call and
 * onnx::inliner::InlineLocalFunctions (run by MakeStepGraph once a builder
 * has accumulated functions) instead of hand-emitting their nodes directly.
 * See graph_grad.py's matching section,
 * scripts/codegen/generate_grad_templates.py, and
 * tests/test_graph_grad_templates.py for the Python side.
 *
 * Numeric validation is not duplicated here, for the reason graph_grad_test.cpp
 * gives for the hand-written rules: nothing in this build evaluates an ONNX
 * graph. tests/test_graph_grad_templates.py already checks
 * GradBatchNormalizationTemplated's actual numbers against torch.autograd on
 * the identical formula, plus the hand-written GradBatchNormalization on the
 * same inputs. What this file checks instead, the same division of labor
 * graph_grad_test.cpp draws for every other rule: that the checked-in
 * template text actually parses and inlines to a real, checker-valid graph
 * with no residual "onnxsim.grad"-domain node, and that the result stays
 * inside the same operator allowlist a hand-written rule does -- which is
 * not automatic (see the second test's comment: the first version of the
 * BatchNormalization template failed exactly this check).
 *
 * Plain asserts and a failure counter, like graph_grad_test.cpp -- this
 * repository vendors no gtest.
 */
#include <onnx/onnx_pb.h>

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "graph_grad.h"
#include "onnx/checker.h"
#include "qat_graph_builder.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

using Shape = std::vector<int64_t>;
using Shapes = std::map<std::string, Shape>;

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

// A step graph whose sole purpose is to expose `outputs` (a target's
// gradient, keyed by the name it should carry) and run them through
// MakeStepGraph -- the real entry point that attaches a builder's
// accumulated functions and inlines them, matching
// qat_graph_builder_test.cpp's own use of `state` purely to declare an
// output (MakeStepGraphDeclaresInputsAndOutputsInOrderAndPassesTheChecker).
StepGraph WrapForInspection(GraphBuilder& b, const Shapes& constant_shapes,
                            const std::map<std::string, Shape>& outputs) {
  StepGraphSpec spec;
  for (const auto& [name, dims] : constant_shapes) {
    spec.constants.push_back({name, dims});
  }
  for (const auto& [name, dims] : outputs) {
    // `input` just needs to be a name nothing else in the graph defines --
    // it becomes an unused, otherwise-harmless declared graph input. Using
    // `name` itself here (as opposed to a distinct placeholder) would
    // declare a graph *input* with the same name as the node output that
    // already produces it, an SSA violation the checker rightly rejects.
    spec.state.push_back(
        {/*input=*/name + "_unused", dims, /*next_output=*/name});
  }
  return MakeStepGraph(b, spec);
}

// Every op in `graph`'s nodes, minus `forward_ops` (the forward node types,
// never a backward rule's own concern) -- what a templated rule's inlined
// result must fit inside BackwardOps() | {"Identity"} for, since a
// checker-valid, fully-inlined graph can still reach for an op the
// execution-provider allowlist does not cover.
std::set<std::string> BackwardEmittedOps(
    const onnx::GraphProto& graph, const std::set<std::string>& forward_ops) {
  std::set<std::string> emitted;
  for (const onnx::NodeProto& node : graph.node()) {
    if (forward_ops.count(node.op_type()) == 0) emitted.insert(node.op_type());
  }
  return emitted;
}

void CheckWithinAllowlist(const std::set<std::string>& emitted,
                          const std::string& what) {
  std::set<std::string> allowed = BackwardOps();
  allowed.insert("Identity");  // this file's own copy-out scaffolding only
  std::string extra;
  for (const std::string& op : emitted) {
    if (allowed.count(op) == 0) {
      if (!extra.empty()) extra += ", ";
      extra += op;
    }
  }
  Check(extra.empty(), what + " reached outside the allowlist: " + extra);
}

// GradAdd's checked-in template is trivial by design (da = db = g, see
// generate_grad_templates.py) -- this proves the plumbing (call node ->
// registered function -> MakeStepGraph's attach-and-inline) rather than
// anything about Add's own math, isolating mechanism bugs from rule bugs.
void TheAddTemplateInlinesToAnAllowlistedGraph() {
  const std::vector<onnx::NodeProto> nodes = {Node("Add", {"A", "B"}, {"Y"})};
  const Shapes shapes = {{"A", {3, 4}}, {"B", {4}}, {"Y", {3, 4}}};

  GraphBuilder b;
  const std::map<std::string, std::string> grads =
      BuildBackwardWithTemplatedRules(b, nodes, shapes, {{"Y", "dY"}},
                                      {"A", "B"});
  Check(grads.size() == 2 && grads.count("A") == 1 && grads.count("B") == 1,
        "both operands get a gradient");
  Check(!b.functions().empty(),
        "the builder accumulated the GradAdd function before inlining");

  const Shapes constants = {{"A", {3, 4}}, {"B", {4}}, {"dY", {3, 4}}};
  const StepGraph step = WrapForInspection(
      b, constants, {{grads.at("A"), {3, 4}}, {grads.at("B"), {4}}});

  Check(step.model.functions_size() == 0,
        "MakeStepGraph inlines every call site -- no residual "
        "onnxsim.grad-domain function");
  bool has_grad_domain_node = false;
  for (const onnx::NodeProto& node : step.model.graph().node()) {
    if (node.domain() == "onnxsim.grad") has_grad_domain_node = true;
  }
  Check(!has_grad_domain_node,
        "no node in the final graph references the private grad domain");

  try {
    onnx::checker::check_model(step.model);
  } catch (const std::exception& e) {
    Check(false, std::string("onnx::checker rejected the templated Add "
                             "backward: ") +
                     e.what());
  }

  CheckWithinAllowlist(BackwardEmittedOps(step.model.graph(), {"Add"}),
                       "the templated Add backward");
}

// The rule that matters: BatchNormalization's own hand-written C++/Python
// rules are where the real dvar-derivation bug this whole design is a
// response to lived. Checks the templated path structurally against the
// hand-written one on the same forward node -- both must resolve all five
// targets without throwing -- and, separately, that the inlined result
// stays inside the allowlist. It did not on the first attempt: the
// checked-in template originally built the literals 1.0/-0.5 in-body via
// Constant/CastLike, neither of which BackwardOps() admits (see
// generate_grad_templates.py's GradBatchNormalization docstring); fixed by
// threading them in as ordinary float32 inputs instead, the same way `eps`
// already was.
void TheBatchNormTemplateInlinesToAnAllowlistedGraph() {
  const std::vector<onnx::NodeProto> nodes = {
      Node("BatchNormalization", {"X", "S", "Bn", "Mn", "Vr"}, {"Y"})};
  const Shapes shapes = {{"X", {2, 3, 4, 4}}, {"S", {3}},  {"Bn", {3}},
                         {"Mn", {3}},         {"Vr", {3}}, {"Y", {2, 3, 4, 4}}};
  const std::vector<std::string> targets = {"X", "S", "Bn", "Mn", "Vr"};

  GraphBuilder hand;
  const std::map<std::string, std::string> hand_grads =
      BuildBackward(hand, nodes, shapes, {{"Y", "dY"}}, targets);

  GraphBuilder templated;
  const std::map<std::string, std::string> templated_grads =
      BuildBackwardWithTemplatedRules(templated, nodes, shapes, {{"Y", "dY"}},
                                      targets);

  Check(hand_grads.size() == 5 && templated_grads.size() == 5,
        "both the hand-written and the templated rule resolve all five "
        "targets");
  Check(!templated.functions().empty(),
        "the templated builder accumulated the GradBatchNormalization "
        "function before inlining");

  Shapes const_shapes = shapes;
  const_shapes["dY"] = {2, 3, 4, 4};
  std::map<std::string, Shape> outputs;
  for (const std::string& target : targets) {
    outputs[templated_grads.at(target)] = shapes.at(target);
  }
  const StepGraph step = WrapForInspection(templated, const_shapes, outputs);

  Check(step.model.functions_size() == 0,
        "MakeStepGraph inlines every call site -- no residual "
        "onnxsim.grad-domain function");
  try {
    onnx::checker::check_model(step.model);
  } catch (const std::exception& e) {
    Check(false, std::string("onnx::checker rejected the templated "
                             "BatchNormalization backward: ") +
                     e.what());
  }

  CheckWithinAllowlist(
      BackwardEmittedOps(step.model.graph(), {"BatchNormalization"}),
      "the templated BatchNormalization backward");
}

}  // namespace

int main() {
  TheAddTemplateInlinesToAnAllowlistedGraph();
  TheBatchNormTemplateInlinesToAnAllowlistedGraph();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
