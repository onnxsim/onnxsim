#include "graph_grad.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Every rule below is a transcription of the same-named rule in
// onnxsim/graph_grad.py, node for node and *in the same order*. The
// derivations live there; what is worth saying here is only where C++ forces
// a choice the Python did not have to make.
//
// Two transcription conventions, both load-bearing for parity:
//
//   * Sub-expressions that emit are hoisted into named locals in emission
//     order. C++ leaves the evaluation order of function arguments
//     unspecified, so `b.Mul(g, b.Const(1.0f))` could number its two names
//     either way round; Python's left-to-right argument evaluation does not
//     have that freedom, and the counters have to agree.
//   * Node name hints are spelled out where the Python passed none.
//     GraphBuilder::Op with an empty hint names the node after its op type
//     (`hint or op_type.lower()` in the Python), so "greater", "cast", "neg"
//     and friends below say explicitly what the Python said by omission.

namespace {

using Shape = std::vector<int64_t>;
// One gradient per node input, empty where an input takes none -- a
// Reshape's shape operand, a Clip's bounds.
using OptStr = std::optional<std::string>;

// ---------------------------------------------------------------------------
// Small helpers over the protobuf types
// ---------------------------------------------------------------------------

const onnx::AttributeProto* FindAttr(const onnx::NodeProto& node,
                                     const std::string& name) {
  for (const onnx::AttributeProto& attribute : node.attribute()) {
    if (attribute.name() == name) return &attribute;
  }
  return nullptr;
}

float AttrFloat(const onnx::NodeProto& node, const std::string& name,
                float fallback) {
  const onnx::AttributeProto* a = FindAttr(node, name);
  return a == nullptr ? fallback : a->f();
}

int64_t AttrInt(const onnx::NodeProto& node, const std::string& name,
                int64_t fallback) {
  const onnx::AttributeProto* a = FindAttr(node, name);
  return a == nullptr ? fallback : a->i();
}

onnx::AttributeProto IntAttr(const std::string& name, int64_t value) {
  onnx::AttributeProto attribute;
  attribute.set_name(name);
  attribute.set_type(onnx::AttributeProto::INT);
  attribute.set_i(value);
  return attribute;
}

// Shapes in error messages, spelled the way the Python's f-strings spell
// them, so a refusal reads the same from either implementation.
std::string ShapeStr(const Shape& shape) {
  std::ostringstream out;
  out << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ", ";
    out << shape[i];
  }
  if (shape.size() == 1) out << ",";
  out << ")";
  return out.str();
}

std::string Quoted(const std::string& value) { return "'" + value + "'"; }

// numpy's broadcast_shapes, for the batch axes of a batched MatMul.
Shape BroadcastShapes(const Shape& a, const Shape& b) {
  const size_t rank = std::max(a.size(), b.size());
  Shape out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      throw std::invalid_argument("cannot broadcast MatMul batch shapes " +
                                  ShapeStr(a) + " and " + ShapeStr(b));
    }
    out[i] = std::max(da, db);
  }
  return out;
}

// ---------------------------------------------------------------------------
// _Backward: build-time state shared by the rules
// ---------------------------------------------------------------------------

class Backward {
 public:
  Backward(GraphBuilder& b, const std::map<std::string, Shape>& shapes)
      : b_(b), shapes_(shapes) {}

  GraphBuilder& b() { return b_; }

  const Shape& ShapeOf(const std::string& name) const {
    const auto it = shapes_.find(name);
    if (it == shapes_.end()) {
      throw std::invalid_argument(
          "no static shape given for tensor " + Quoted(name) +
          "; BuildBackward needs the shape of every value the slice touches");
    }
    return it->second;
  }

  // Sums `grad` back down to `target_shape`, undoing a numpy-style
  // broadcast. See _Backward.reduce_to in graph_grad.py.
  std::string ReduceTo(const std::string& grad, const Shape& grad_shape,
                       const Shape& target_shape) {
    if (grad_shape == target_shape) return grad;

    const int64_t offset =
        static_cast<int64_t>(grad_shape.size()) -
        static_cast<int64_t>(target_shape.size());
    if (offset < 0) {
      throw std::invalid_argument(
          "cannot reduce a gradient of shape " + ShapeStr(grad_shape) + " to " +
          ShapeStr(target_shape) +
          ": the gradient has fewer dimensions than the tensor it belongs to");
    }
    std::vector<int64_t> axes;
    for (int64_t i = 0; i < offset; ++i) axes.push_back(i);
    for (size_t i = 0; i < target_shape.size(); ++i) {
      const int64_t dim = target_shape[i];
      const int64_t actual = grad_shape[static_cast<size_t>(offset) + i];
      if (dim == actual) continue;
      if (dim == 1) {
        axes.push_back(offset + static_cast<int64_t>(i));
      } else {
        throw std::invalid_argument("gradient shape " + ShapeStr(grad_shape) +
                                    " is not a broadcast of " +
                                    ShapeStr(target_shape));
      }
    }

    std::string out = grad;
    if (!axes.empty()) {
      const std::string axes_const = b_.ConstInt64(axes, "axes");
      out = b_.Op("ReduceSum", {out, axes_const}, {IntAttr("keepdims", 1)},
                  "unbcast");
    }
    // ONNX's ReduceSum cannot drop *some* axes and keep others as size 1 in
    // one node, so the keepdims=1 result is reshaped when they differ.
    const std::set<int64_t> reduced(axes.begin(), axes.end());
    Shape summed = grad_shape;
    for (size_t i = 0; i < summed.size(); ++i) {
      if (reduced.count(static_cast<int64_t>(i)) != 0) summed[i] = 1;
    }
    if (summed != target_shape) {
      const std::string shape_const = b_.ConstInt64(target_shape, "shape");
      out = b_.Op("Reshape", {out, shape_const}, "unbcast");
    }
    return out;
  }

  // `name` with its last two axes swapped -- what a MatMul's VJP needs, and
  // what a bare Transpose (which reverses *all* axes) would get wrong for a
  // batched operand.
  std::string TransposeLastTwo(const std::string& name, const Shape& shape) {
    const int64_t rank = static_cast<int64_t>(shape.size());
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < rank - 2; ++i) perm.push_back(i);
    perm.push_back(rank - 1);
    perm.push_back(rank - 2);
    return b_.Transpose(name, perm);
  }

  // `(x > bound)` / `(x < bound)` as a float32 0/1 tensor, with `bound` a
  // tensor name rather than GraphBuilder::GreaterMask's float.
  std::string MaskGreater(const std::string& x, const std::string& bound) {
    const std::string gt = b_.Op("Greater", {x, bound}, "greater");
    return b_.Op("Cast", {gt}, {IntAttr("to", onnx::TensorProto::FLOAT)},
                 "cast");
  }

  std::string MaskLess(const std::string& x, const std::string& bound) {
    const std::string lt = b_.Op("Less", {x, bound}, "less");
    return b_.Op("Cast", {lt}, {IntAttr("to", onnx::TensorProto::FLOAT)},
                 "cast");
  }

 private:
  GraphBuilder& b_;
  const std::map<std::string, Shape>& shapes_;
};

// A rule takes the build context, the forward node, and the name of the
// gradient flowing into that node's single output; it appends nodes and
// returns one gradient per node input.
using Rule = std::vector<OptStr> (*)(Backward&, const onnx::NodeProto&,
                                     const std::string&);

// ---------------------------------------------------------------------------
// The rules
// ---------------------------------------------------------------------------

std::vector<OptStr> GradMatMul(Backward& ctx, const onnx::NodeProto& node,
                               const std::string& g) {
  const std::string& a = node.input(0);
  const std::string& b = node.input(1);
  const Shape sa = ctx.ShapeOf(a);
  const Shape sb = ctx.ShapeOf(b);
  if (sa.size() < 2 || sb.size() < 2) {
    throw UnsupportedOpError(
        "MatMul with a 1-D operand is not differentiated here (node " +
        Quoted(node.output(0)) + ", operand shapes " + ShapeStr(sa) + " and " +
        ShapeStr(sb) + ")");
  }
  // dA = G @ B^T, dB = A^T @ G, both then summed back over whatever batch
  // axes broadcasting replicated.
  const Shape batch =
      BroadcastShapes(Shape(sa.begin(), sa.end() - 2),
                      Shape(sb.begin(), sb.end() - 2));
  const std::string bt = ctx.TransposeLastTwo(b, sb);
  const std::string ga = ctx.b().MatMul(g, bt);
  const std::string at = ctx.TransposeLastTwo(a, sa);
  const std::string gb = ctx.b().MatMul(at, g);

  Shape ga_shape = batch;
  ga_shape.push_back(sa[sa.size() - 2]);
  ga_shape.push_back(sa[sa.size() - 1]);
  Shape gb_shape = batch;
  gb_shape.push_back(sb[sb.size() - 2]);
  gb_shape.push_back(sb[sb.size() - 1]);

  const std::string ra = ctx.ReduceTo(ga, ga_shape, sa);
  const std::string rb = ctx.ReduceTo(gb, gb_shape, sb);
  return {ra, rb};
}

std::vector<OptStr> GradGemm(Backward& ctx, const onnx::NodeProto& node,
                             const std::string& g) {
  const float alpha = AttrFloat(node, "alpha", 1.0f);
  const float beta = AttrFloat(node, "beta", 1.0f);
  const bool trans_a = AttrInt(node, "transA", 0) != 0;
  const bool trans_b = AttrInt(node, "transB", 0) != 0;
  const std::string& a = node.input(0);
  const std::string& b = node.input(1);
  const Shape sa = ctx.ShapeOf(a);
  const Shape sb = ctx.ShapeOf(b);
  if (sa.size() != 2 || sb.size() != 2) {
    throw UnsupportedOpError("Gemm expects 2-D A and B, got " + ShapeStr(sa) +
                             " and " + ShapeStr(sb) + " (node " +
                             Quoted(node.output(0)) + ")");
  }

  // Y = alpha * A' B' + beta * C, with A' = A^T when transA. Differentiate
  // against A' and B' first, then transpose back into A's and B's own
  // layouts; alpha scales the incoming gradient once instead of scaling both
  // results.
  std::string gs = g;
  if (alpha != 1.0f) {
    const std::string alpha_const = ctx.b().Const(alpha);
    gs = ctx.b().Mul(g, alpha_const);
  }
  const std::string b_operand = trans_b ? b : ctx.b().Transpose(b, {1, 0});
  std::string ga = ctx.b().MatMul(gs, b_operand);
  if (trans_a) ga = ctx.b().Transpose(ga, {1, 0});
  const std::string a_operand = trans_a ? a : ctx.b().Transpose(a, {1, 0});
  std::string gb = ctx.b().MatMul(a_operand, gs);
  if (trans_b) gb = ctx.b().Transpose(gb, {1, 0});

  std::vector<OptStr> grads{ga, gb};
  if (node.input_size() > 2) {
    if (!node.input(2).empty()) {
      // C broadcasts against [M, N], so its gradient needs the same
      // broadcast-undoing every elementwise rule needs.
      const std::string gc = ctx.ReduceTo(g, ctx.ShapeOf(node.output(0)),
                                          ctx.ShapeOf(node.input(2)));
      if (beta != 1.0f) {
        const std::string beta_const = ctx.b().Const(beta);
        grads.push_back(ctx.b().Mul(gc, beta_const));
      } else {
        grads.push_back(gc);
      }
    } else {
      // C spelled as an omitted optional input ("") rather than left off the
      // node entirely -- there is no tensor to give a gradient to.
      grads.push_back(std::nullopt);
    }
  }
  return grads;
}

std::vector<OptStr> GradAdd(Backward& ctx, const onnx::NodeProto& node,
                            const std::string& g) {
  const Shape out = ctx.ShapeOf(node.output(0));
  const std::string ga = ctx.ReduceTo(g, out, ctx.ShapeOf(node.input(0)));
  const std::string gb = ctx.ReduceTo(g, out, ctx.ShapeOf(node.input(1)));
  return {ga, gb};
}

std::vector<OptStr> GradSub(Backward& ctx, const onnx::NodeProto& node,
                            const std::string& g) {
  const Shape out = ctx.ShapeOf(node.output(0));
  // Negate after reducing, not before: the reduced tensor is the smaller of
  // the two. The second operand's reduction is therefore emitted *first*,
  // which is what the Python does and what the name counters record.
  const std::string gb = ctx.ReduceTo(g, out, ctx.ShapeOf(node.input(1)));
  const std::string ga = ctx.ReduceTo(g, out, ctx.ShapeOf(node.input(0)));
  return {ga, ctx.b().Op("Neg", {gb}, "neg")};
}

std::vector<OptStr> GradMul(Backward& ctx, const onnx::NodeProto& node,
                            const std::string& g) {
  const std::string& a = node.input(0);
  const std::string& b = node.input(1);
  const Shape out = ctx.ShapeOf(node.output(0));
  const std::string gb_times = ctx.b().Mul(g, b);
  const std::string ga = ctx.ReduceTo(gb_times, out, ctx.ShapeOf(a));
  const std::string ga_times = ctx.b().Mul(g, a);
  const std::string gb = ctx.ReduceTo(ga_times, out, ctx.ShapeOf(b));
  return {ga, gb};
}

std::vector<OptStr> GradDiv(Backward& ctx, const onnx::NodeProto& node,
                            const std::string& g) {
  const std::string& a = node.input(0);
  const std::string& b = node.input(1);
  const std::string& y = node.output(0);
  const Shape out = ctx.ShapeOf(y);
  // d/db (a/b) = -a/b^2 = -y/b, reusing the forward quotient rather than
  // recomputing a square.
  const std::string gy = ctx.b().Mul(g, y);
  const std::string quotient = ctx.b().Div(gy, b);
  const std::string gb_raw = ctx.b().Op("Neg", {quotient}, "neg");
  const std::string ga_raw = ctx.b().Div(g, b);
  const std::string ga = ctx.ReduceTo(ga_raw, out, ctx.ShapeOf(a));
  const std::string gb = ctx.ReduceTo(gb_raw, out, ctx.ShapeOf(b));
  return {ga, gb};
}

std::vector<OptStr> GradNeg(Backward& ctx, const onnx::NodeProto& node,
                            const std::string& g) {
  (void)node;
  return {ctx.b().Op("Neg", {g}, "neg")};
}

std::vector<OptStr> GradIdentity(Backward& ctx, const onnx::NodeProto& node,
                                 const std::string& g) {
  // An alias, not a node: the gradient of the output *is* the gradient of the
  // input, and emitting an Identity to say so would only add a copy.
  (void)ctx;
  (void)node;
  return {g};
}

std::vector<OptStr> GradRelu(Backward& ctx, const onnx::NodeProto& node,
                             const std::string& g) {
  // The subgradient at exactly 0 is taken as 0 (strict Greater), matching the
  // straight-through masks adaround.py already builds.
  const std::string mask = ctx.b().GreaterMask(node.input(0), 0.0f);
  return {ctx.b().Mul(g, mask)};
}

std::vector<OptStr> GradSigmoid(Backward& ctx, const onnx::NodeProto& node,
                                const std::string& g) {
  // y (1 - y), from the forward output: the backward never re-runs the
  // sigmoid.
  const std::string& y = node.output(0);
  const std::string one = ctx.b().Const(1.0f);
  const std::string one_minus_y = ctx.b().Sub(one, y);
  const std::string dy = ctx.b().Mul(y, one_minus_y);
  return {ctx.b().Mul(g, dy)};
}

std::vector<OptStr> GradTanh(Backward& ctx, const onnx::NodeProto& node,
                             const std::string& g) {
  const std::string& y = node.output(0);
  const std::string one = ctx.b().Const(1.0f);
  const std::string y_squared = ctx.b().Mul(y, y);
  const std::string dy = ctx.b().Sub(one, y_squared);
  return {ctx.b().Mul(g, dy)};
}

std::vector<OptStr> GradErf(Backward& ctx, const onnx::NodeProto& node,
                            const std::string& g) {
  // 2/sqrt(pi) * exp(-x^2). Here entirely for GELU, which block-wise
  // fine-tuning meets in every transformer FFN.
  const std::string& x = node.input(0);
  const std::string scale = ctx.b().Const(
      static_cast<float>(2.0 / std::sqrt(3.14159265358979323846)));
  const std::string x_squared = ctx.b().Mul(x, x);
  const std::string negated = ctx.b().Op("Neg", {x_squared}, "neg");
  const std::string exponential = ctx.b().Op("Exp", {negated}, "exp");
  const std::string dy = ctx.b().Mul(scale, exponential);
  return {ctx.b().Mul(g, dy)};
}

std::vector<OptStr> GradExp(Backward& ctx, const onnx::NodeProto& node,
                            const std::string& g) {
  return {ctx.b().Mul(g, node.output(0))};
}

std::vector<OptStr> GradSqrt(Backward& ctx, const onnx::NodeProto& node,
                             const std::string& g) {
  // 0.5 / sqrt(x), again reusing the forward result. Singular at x = 0, as
  // the derivative genuinely is.
  const std::string half = ctx.b().Const(0.5f);
  const std::string scaled = ctx.b().Mul(g, half);
  return {ctx.b().Div(scaled, node.output(0))};
}

std::vector<OptStr> GradTranspose(Backward& ctx, const onnx::NodeProto& node,
                                  const std::string& g) {
  const int64_t rank = static_cast<int64_t>(ctx.ShapeOf(node.input(0)).size());
  const onnx::AttributeProto* perm = FindAttr(node, "perm");
  std::vector<int64_t> axes;
  if (perm == nullptr) {
    for (int64_t i = rank - 1; i >= 0; --i) axes.push_back(i);
  } else {
    for (int64_t p : perm->ints()) axes.push_back(p);
  }
  std::vector<int64_t> inverse(static_cast<size_t>(rank), 0);
  for (size_t position = 0; position < axes.size(); ++position) {
    inverse[static_cast<size_t>(axes[position])] =
        static_cast<int64_t>(position);
  }
  return {ctx.b().Transpose(g, inverse)};
}

std::vector<OptStr> GradReshape(Backward& ctx, const onnx::NodeProto& node,
                                const std::string& g) {
  const Shape shape = ctx.ShapeOf(node.input(0));
  const std::string shape_const = ctx.b().ConstInt64(shape, "shape");
  return {ctx.b().Op("Reshape", {g, shape_const}, "reshape"), std::nullopt};
}

// Which axes a Reduce* node reduced over. See _reduced_axes in
// graph_grad.py: from opset 13 the axes are a tensor *input*, which
// BuildBackward is not given, so they have to come back out of the shapes --
// exactly with keepdims=1, and by elimination (refusing a genuine ambiguity)
// without.
std::vector<int64_t> ReducedAxes(const onnx::NodeProto& node,
                                 const Shape& in_shape, const Shape& out_shape) {
  const int64_t rank = static_cast<int64_t>(in_shape.size());
  const onnx::AttributeProto* attribute = FindAttr(node, "axes");
  if (attribute != nullptr) {
    if (rank == 0) {
      // Python would raise ZeroDivisionError on the `% rank` below; a
      // rank-0 input to a Reduce* with an axes attribute is malformed either
      // way, and this says so rather than dividing by zero.
      throw UnsupportedOpError(
          "cannot interpret the axes attribute of " + Quoted(node.output(0)) +
          " against a rank-0 input");
    }
    std::set<int64_t> axes;
    for (int64_t a : attribute->ints()) axes.insert(((a % rank) + rank) % rank);
    return std::vector<int64_t>(axes.begin(), axes.end());
  }

  const bool keepdims = AttrInt(node, "keepdims", 1) != 0;
  if (keepdims) {
    if (static_cast<int64_t>(out_shape.size()) != rank) {
      throw UnsupportedOpError(node.op_type() + " with keepdims=1 changed rank " +
                               std::to_string(rank) + " to " +
                               std::to_string(out_shape.size()) + " (node " +
                               Quoted(node.output(0)) + ")");
    }
    // An axis that is 1 on both sides may or may not have been reduced, and
    // it makes no difference: summing over a length-1 axis and broadcasting
    // back over it are both the identity.
    std::vector<int64_t> axes;
    for (int64_t i = 0; i < rank; ++i) {
      if (out_shape[static_cast<size_t>(i)] == 1 &&
          in_shape[static_cast<size_t>(i)] != 1) {
        axes.push_back(i);
      }
    }
    return axes;
  }

  const int64_t dropped = rank - static_cast<int64_t>(out_shape.size());
  if (dropped < 0 || rank > 16) {
    throw UnsupportedOpError("cannot recover the reduced axes of " +
                             Quoted(node.output(0)) + " from shapes " +
                             ShapeStr(in_shape) + " -> " + ShapeStr(out_shape));
  }
  // itertools.combinations(range(rank), dropped), by an ascending-index
  // cursor -- the same enumeration order, so candidates[0] below is the same
  // candidate the Python picks.
  std::vector<std::vector<int64_t>> candidates;
  std::vector<int64_t> combo(static_cast<size_t>(dropped), 0);
  for (int64_t i = 0; i < dropped; ++i) combo[static_cast<size_t>(i)] = i;
  while (true) {
    Shape kept;
    for (int64_t i = 0; i < rank; ++i) {
      if (std::find(combo.begin(), combo.end(), i) == combo.end()) {
        kept.push_back(in_shape[static_cast<size_t>(i)]);
      }
    }
    if (kept == out_shape) candidates.push_back(combo);
    // Advance the rightmost index that is not already at its limit, then
    // repack everything to its right -- combinations()' own odometer.
    int64_t position = dropped - 1;
    while (position >= 0 &&
           combo[static_cast<size_t>(position)] == rank - dropped + position) {
      --position;
    }
    if (position < 0) break;
    ++combo[static_cast<size_t>(position)];
    for (int64_t i = position + 1; i < dropped; ++i) {
      combo[static_cast<size_t>(i)] = combo[static_cast<size_t>(i - 1)] + 1;
    }
  }
  if (candidates.empty()) {
    throw UnsupportedOpError("no set of reduced axes takes " +
                             ShapeStr(in_shape) + " to " + ShapeStr(out_shape) +
                             " (node " + Quoted(node.output(0)) + ")");
  }
  // Two candidates that disagree only about length-1 axes imply the same
  // backward graph, so compare what actually gets built rather than the axis
  // sets themselves.
  std::set<Shape> expanded;
  for (const std::vector<int64_t>& candidate : candidates) {
    Shape shape = in_shape;
    for (int64_t axis : candidate) shape[static_cast<size_t>(axis)] = 1;
    expanded.insert(shape);
  }
  if (expanded.size() != 1) {
    throw UnsupportedOpError("the reduced axes of " + Quoted(node.output(0)) +
                             " are ambiguous from shapes " +
                             ShapeStr(in_shape) + " -> " + ShapeStr(out_shape) +
                             "; use keepdims=1 so they can be recovered");
  }
  return candidates[0];
}

std::vector<OptStr> GradReduce(Backward& ctx, const onnx::NodeProto& node,
                               const std::string& g) {
  // ReduceSum and ReduceMean: broadcast the gradient back over the axes that
  // were reduced away, scaled by 1/count for the mean. The broadcast is a
  // multiply by a constant rather than an Expand, which keeps the emitted
  // graph inside BackwardOps(); the constant spans only the reduced axes.
  const std::string& x = node.input(0);
  const Shape in_shape = ctx.ShapeOf(x);
  const Shape out_shape = ctx.ShapeOf(node.output(0));
  std::vector<OptStr> rest(static_cast<size_t>(node.input_size() - 1),
                           std::nullopt);
  const std::vector<int64_t> axes_list = ReducedAxes(node, in_shape, out_shape);
  const std::set<int64_t> axes(axes_list.begin(), axes_list.end());
  if (axes.empty()) {
    std::vector<OptStr> grads{g};
    grads.insert(grads.end(), rest.begin(), rest.end());
    return grads;
  }

  Shape keepdims_shape = in_shape;
  for (int64_t axis : axes) keepdims_shape[static_cast<size_t>(axis)] = 1;
  std::string grad = g;
  if (out_shape != keepdims_shape) {
    const std::string shape_const = ctx.b().ConstInt64(keepdims_shape, "shape");
    grad = ctx.b().Op("Reshape", {grad, shape_const}, "reshape");
  }

  double fill = 1.0;
  if (node.op_type() == "ReduceMean") {
    double count = 1.0;
    for (int64_t axis : axes) {
      count *= static_cast<double>(in_shape[static_cast<size_t>(axis)]);
    }
    fill = 1.0 / count;
  }
  Shape ones_shape = in_shape;
  int64_t elements = 1;
  for (size_t i = 0; i < ones_shape.size(); ++i) {
    if (axes.count(static_cast<int64_t>(i)) == 0) ones_shape[i] = 1;
    elements *= ones_shape[i];
  }
  const std::string ones = ctx.b().Const(
      std::vector<float>(static_cast<size_t>(elements),
                         static_cast<float>(fill)),
      ones_shape, "bcast");
  std::vector<OptStr> grads{ctx.b().Mul(grad, ones)};
  grads.insert(grads.end(), rest.begin(), rest.end());
  return grads;
}

std::vector<OptStr> GradSoftmax(Backward& ctx, const onnx::NodeProto& node,
                                const std::string& g) {
  // dx = y * (g - sum(g * y)) along the softmax axis, written from the
  // forward output y.
  const std::string& y = node.output(0);
  const int64_t rank = static_cast<int64_t>(ctx.ShapeOf(node.input(0)).size());
  if (rank == 0) {
    throw UnsupportedOpError("Softmax over a rank-0 input (node " +
                             Quoted(node.output(0)) + ") has no axis to sum");
  }
  const int64_t raw_axis = AttrInt(node, "axis", -1);
  const int64_t axis = ((raw_axis % rank) + rank) % rank;
  const std::string gy = ctx.b().Mul(g, y);
  const std::string axes_const = ctx.b().ConstInt64({axis}, "axes");
  const std::string total = ctx.b().Op(
      "ReduceSum", {gy, axes_const}, {IntAttr("keepdims", 1)}, "reducesum");
  const std::string centred = ctx.b().Sub(g, total);
  return {ctx.b().Mul(y, centred)};
}

std::vector<OptStr> GradClip(Backward& ctx, const onnx::NodeProto& node,
                             const std::string& g) {
  // Pass the gradient through where the input was *strictly* inside the
  // bounds, zero it elsewhere -- the bounds are used as the tensors they are,
  // so their values need not be known at build time.
  const std::string& x = node.input(0);
  std::vector<OptStr> rest(static_cast<size_t>(node.input_size() - 1),
                           std::nullopt);
  OptStr mask;
  if (node.input_size() > 1 && !node.input(1).empty()) {
    mask = ctx.MaskGreater(x, node.input(1));
  }
  if (node.input_size() > 2 && !node.input(2).empty()) {
    const std::string upper = ctx.MaskLess(x, node.input(2));
    mask = mask.has_value() ? ctx.b().Mul(*mask, upper) : upper;
  }
  std::vector<OptStr> grads;
  grads.push_back(mask.has_value() ? ctx.b().Mul(g, *mask) : g);
  grads.insert(grads.end(), rest.begin(), rest.end());
  return grads;
}

const std::map<std::string, Rule>& Rules() {
  static const std::map<std::string, Rule>* rules =
      new std::map<std::string, Rule>{
          {"Add", &GradAdd},
          {"Clip", &GradClip},
          {"Div", &GradDiv},
          {"Erf", &GradErf},
          {"Exp", &GradExp},
          {"Gemm", &GradGemm},
          {"Identity", &GradIdentity},
          {"MatMul", &GradMatMul},
          {"Mul", &GradMul},
          {"Neg", &GradNeg},
          {"ReduceMean", &GradReduce},
          {"ReduceSum", &GradReduce},
          {"Relu", &GradRelu},
          {"Reshape", &GradReshape},
          {"Sigmoid", &GradSigmoid},
          {"Softmax", &GradSoftmax},
          {"Sqrt", &GradSqrt},
          {"Sub", &GradSub},
          {"Tanh", &GradTanh},
          {"Transpose", &GradTranspose},
      };
  return *rules;
}

// "['Add', 'Clip', ...]", the way the Python's sorted(SUPPORTED_OPS) prints
// inside its refusal message.
std::string SupportedOpsList() {
  std::string out = "[";
  bool first = true;
  for (const std::string& op : SupportedOps()) {
    if (!first) out += ", ";
    first = false;
    out += Quoted(op);
  }
  return out + "]";
}

}  // namespace

const std::set<std::string>& BackwardOps() {
  static const std::set<std::string>* ops = new std::set<std::string>{
      "Add", "Cast",  "Div",       "Exp",     "Greater",   "Less",     "MatMul",
      "Mul", "Neg",   "ReduceSum", "Reshape", "Sub",       "Transpose"};
  return *ops;
}

const std::set<std::string>& SupportedOps() {
  static const std::set<std::string>* ops = [] {
    auto* names = new std::set<std::string>();
    for (const auto& entry : Rules()) names->insert(entry.first);
    return names;
  }();
  return *ops;
}

std::map<std::string, std::string> BuildBackward(
    GraphBuilder& b, const std::vector<onnx::NodeProto>& nodes,
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const std::map<std::string, std::string>& grad_outputs,
    const std::vector<std::string>& targets) {
  Backward ctx(b, shapes);
  std::map<std::string, std::string> grads(grad_outputs);

  // Reverse topological order, which is what makes the accumulation below
  // correct: every reader of a tensor is visited before its producer, so by
  // the time a producer asks for its output gradient the sum is complete.
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    const onnx::NodeProto& node = *it;
    const auto rule = Rules().find(node.op_type());
    if (rule == Rules().end()) {
      // The name a refusal points at: the node's own name where it has one,
      // its output otherwise. The Python indexes output[0] unconditionally;
      // a node with no outputs at all would make that an IndexError, so this
      // falls back to the empty string rather than reading past the end.
      const std::string where =
          !node.name().empty()
              ? node.name()
              : (node.output_size() > 0 ? node.output(0) : std::string());
      throw UnsupportedOpError("no gradient rule for op type " +
                               Quoted(node.op_type()) + " (node " +
                               Quoted(where) + "); graph_grad differentiates " +
                               SupportedOpsList());
    }
    if (node.output_size() != 1) {
      throw UnsupportedOpError(
          node.op_type() + " has " + std::to_string(node.output_size()) +
          " outputs; only single-output nodes are differentiated here");
    }
    const auto seed = grads.find(node.output(0));
    if (seed == grads.end()) {
      // Nothing downstream depends on this node, so every gradient it would
      // produce is zero. Emitting those zeros would be correct and pure
      // waste.
      continue;
    }
    const std::string g = seed->second;
    const std::vector<OptStr> contributions = rule->second(ctx, node, g);
    if (contributions.size() != static_cast<size_t>(node.input_size())) {
      throw std::logic_error("the " + node.op_type() + " rule returned " +
                             std::to_string(contributions.size()) +
                             " gradients for " +
                             std::to_string(node.input_size()) + " inputs");
    }
    for (int i = 0; i < node.input_size(); ++i) {
      const std::string& name = node.input(i);
      if (name.empty() || !contributions[static_cast<size_t>(i)].has_value()) {
        continue;
      }
      const std::string& contribution = *contributions[static_cast<size_t>(i)];
      // A tensor read by several nodes -- or twice by one node, as in
      // Mul(x, x) -- accumulates.
      const auto existing = grads.find(name);
      if (existing == grads.end()) {
        grads.emplace(name, contribution);
      } else {
        existing->second = b.Add(existing->second, contribution);
      }
    }
  }

  std::map<std::string, std::string> result;
  for (const std::string& target : targets) {
    const auto found = grads.find(target);
    if (found == grads.end()) {
      std::string seeds;
      for (const auto& entry : grad_outputs) {
        if (!seeds.empty()) seeds += ", ";
        seeds += Quoted(entry.first);
      }
      throw std::invalid_argument("no gradient reaches " + Quoted(target) +
                                  ": it is not downstream of any of [" + seeds +
                                  "] within the given nodes");
    }
    result[target] = found->second;
  }
  return result;
}
