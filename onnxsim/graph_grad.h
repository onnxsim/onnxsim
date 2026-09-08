#pragma once

// Reverse-mode automatic differentiation over a slice of an ONNX graph, with
// the gradient itself emitted as ordinary ONNX nodes.
//
// onnxsim/graph_grad.py is the original, and its module docstring carries the
// whole argument: why a hand-derived backward pass is plain dataflow and so
// expressible as an inference graph, why the rule table is deliberately
// incomplete, and why every rule has to undo broadcasting. None of that is
// repeated here -- there should be one place to update when a derivation
// changes, and it is the Python.
//
// This is the same differentiator, reachable from the WASM build. The browser
// converter page reaches onnxsim through C++ only, so without this port the
// browser can build a step graph (qat_graph_builder.h) but has nothing to put
// a gradient in it: every QAT rule in the package is Python. The emitted
// nodes go into the same GraphBuilder the forward and the Adam update go
// into, so a browser-built training step composes exactly as the Python one
// does.
//
// **Parity with graph_grad.py is the contract, not an aspiration.** Two
// differentiators that disagree would mean a browser that trains differently
// from the Python for the same model, silently -- and a gradient is the one
// thing whose bugs do not announce themselves, since a slightly wrong one
// still converges, just to a slightly worse answer. So the rules below emit
// the *same nodes in the same order* as their Python counterparts, which
// keeps the two builders' name counters in lockstep and makes the emitted
// graphs comparable node for node. Reordering a rule's emissions is a
// behaviour change even when the arithmetic is untouched.

#include <onnx/onnx_pb.h>

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "qat_graph_builder.h"

// The complete set of operators the rules can emit.
//
// This is graph_grad.py's BACKWARD_OPS, and it must stay identical to it. The
// membership reasoning (no Where, no boolean logic, no Expand, no control
// flow -- a mask is a float 0/1 from Cast(Greater), a broadcast is a multiply
// by a constant) lives next to the Python set, on the same principle
// qat_graph_builder.h states for EpFriendlyOps.
//
// It is a subset of EpFriendlyOps(): a backward is appended to the same
// builder as the forward, so anything it can produce is something the
// execution provider has to be able to run.
const std::set<std::string>& BackwardOps();

// The op types BuildBackward can differentiate -- the keys of the rule table.
//
// Callers that pick the slice themselves (block discovery for QAT, say)
// should test against this rather than rediscovering the list by catching
// UnsupportedOpError.
const std::set<std::string>& SupportedOps();

// Raised for a node whose op type has no VJP rule, and for a node whose op
// type is covered but whose particular configuration is not (a 1-D MatMul
// operand, a Reduce* whose reduced axes cannot be recovered from the shapes
// alone). Either way this module refuses to differentiate that node rather
// than guessing, which is the point -- see the Python class' docstring.
//
// Derived from std::invalid_argument for the reason the Python one derives
// from ValueError: BuildBackward's other refusals (a missing shape, an
// unreachable target) are plain std::invalid_argument, and a caller that
// wants to treat "cannot differentiate this slice" as one condition catches
// the base.
class UnsupportedOpError : public std::invalid_argument {
 public:
  explicit UnsupportedOpError(const std::string& what)
      : std::invalid_argument(what) {}
};

// Appends the reverse-mode gradient of `nodes` to `b` and returns where each
// target's gradient landed, as {target name: tensor holding its gradient}.
//
// Nothing is added to the forward graph and no forward node is modified: the
// rules read forward tensors by name, node *outputs* included where reusing
// them is cheaper than recomputing (Sigmoid, Tanh, Exp, Sqrt, Softmax). So
// the caller must place these nodes after the forward ones in the same graph
// and keep the forward intermediates available.
//
// `nodes` are the forward nodes to differentiate, topologically ordered;
// nodes outside the slice must not be included. `shapes` is the static shape
// of every tensor the slice touches, inputs and outputs included -- needed at
// build time because undoing a broadcast and undoing a reduction are both
// shape arithmetic, and requiring them is why this emits no Shape/Gather
// plumbing. `grad_outputs` maps a forward tensor to the tensor holding
// dL/d(that tensor) and seeds the walk. `targets` are the tensors to return
// gradients for.
//
// A returned name may be one of `grad_outputs`' own values when the path is a
// pure alias (a lone Identity), so it is not guaranteed to name a node in
// `b`.
//
// Throws UnsupportedOpError for a node this module will not differentiate --
// for *every* node in the slice with an unknown op type, including one no
// gradient reaches, so a caller learns the slice is out of scope from the
// shape of the graph rather than from whether a seed happened to reach it.
// Throws std::invalid_argument if a target is not reachable from
// `grad_outputs` through `nodes` (a disconnected target almost always means
// the slice or the target list is wrong, and a zero gradient would hide it),
// or if a shape is missing from `shapes`.
std::map<std::string, std::string> BuildBackward(
    GraphBuilder& b, const std::vector<onnx::NodeProto>& nodes,
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const std::map<std::string, std::string>& grad_outputs,
    const std::vector<std::string>& targets);
