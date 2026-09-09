#include "graph_grad.h"

#include <onnx/defs/parser.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "graph_grad_templates_gen.h"

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

std::vector<int64_t> AttrInts(const onnx::NodeProto& node,
                              const std::string& name,
                              const std::vector<int64_t>& fallback) {
  const onnx::AttributeProto* a = FindAttr(node, name);
  if (a == nullptr) return fallback;
  return std::vector<int64_t>(a->ints().begin(), a->ints().end());
}

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

// An int list the way Python prints one -- "[3, 3]" -- for the Conv
// refusals, whose messages name strides, dilations and pads.
std::string IntsStr(const std::vector<int64_t>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(values[i]);
  }
  return out + "]";
}

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

    const int64_t offset = static_cast<int64_t>(grad_shape.size()) -
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
  const Shape batch = BroadcastShapes(Shape(sa.begin(), sa.end() - 2),
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

// The Conv rule's helpers. graph_grad.py's _prod/_unflatten/_im2col_indices/
// _col2im_indices, transcribed; the derivation and the reason a convolution's
// gradient is written as im2col rather than as a ConvTranspose are in
// _grad_conv's docstring there.
int64_t Prod(const std::vector<int64_t>& dims) {
  int64_t total = 1;
  for (int64_t d : dims) total *= d;
  return total;
}

std::vector<int64_t> Unflatten(int64_t index,
                               const std::vector<int64_t>& dims) {
  std::vector<int64_t> coords(dims.size(), 0);
  for (size_t i = dims.size(); i-- > 0;) {
    coords[i] = index % dims[i];
    index /= dims[i];
  }
  return coords;
}

// Where each (kernel tap, output position) pair reads its input, and a 0/1
// mask of the pairs that read real input rather than padding.
std::pair<std::vector<int64_t>, std::vector<float>> Im2ColIndices(
    const std::vector<int64_t>& in_dims, const std::vector<int64_t>& out_dims,
    const std::vector<int64_t>& kernel, const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dilations,
    const std::vector<int64_t>& pads_begin) {
  const size_t spatial = in_dims.size();
  const int64_t out_count = Prod(out_dims);
  const int64_t taps = Prod(kernel);
  std::vector<int64_t> index(static_cast<size_t>(taps * out_count), 0);
  std::vector<float> mask(index.size(), 0.0f);
  for (int64_t tap = 0; tap < taps; ++tap) {
    const std::vector<int64_t> taps_at = Unflatten(tap, kernel);
    for (int64_t out = 0; out < out_count; ++out) {
      const std::vector<int64_t> position = Unflatten(out, out_dims);
      int64_t flat = 0;
      for (size_t i = 0; i < spatial; ++i) {
        const int64_t p = position[i] * strides[i] - pads_begin[i] +
                          taps_at[i] * dilations[i];
        if (p < 0 || p >= in_dims[i]) {
          flat = -1;
          break;
        }
        flat = flat * in_dims[i] + p;
      }
      if (flat >= 0) {
        index[static_cast<size_t>(tap * out_count + out)] = flat;
        mask[static_cast<size_t>(tap * out_count + out)] = 1.0f;
      }
    }
  }
  return {index, mask};
}

// The same correspondence read the other way: which output position a given
// (kernel tap, input position) pair came from -- what turns dX's scatter-add
// into a gather.
std::pair<std::vector<int64_t>, std::vector<float>> Col2ImIndices(
    const std::vector<int64_t>& in_dims, const std::vector<int64_t>& out_dims,
    const std::vector<int64_t>& kernel, const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dilations,
    const std::vector<int64_t>& pads_begin) {
  const size_t spatial = in_dims.size();
  const int64_t in_count = Prod(in_dims);
  const int64_t taps = Prod(kernel);
  std::vector<int64_t> index(static_cast<size_t>(taps * in_count), 0);
  std::vector<float> mask(index.size(), 0.0f);
  for (int64_t tap = 0; tap < taps; ++tap) {
    const std::vector<int64_t> taps_at = Unflatten(tap, kernel);
    for (int64_t entry = 0; entry < in_count; ++entry) {
      const std::vector<int64_t> position = Unflatten(entry, in_dims);
      int64_t flat = 0;
      for (size_t i = 0; i < spatial; ++i) {
        const int64_t shifted =
            position[i] + pads_begin[i] - taps_at[i] * dilations[i];
        // Divisibility first: C++ truncates towards zero where Python floors,
        // so the quotient is only read once it is known to be exact and the
        // two languages cannot disagree about it.
        if (shifted % strides[i] != 0) {
          flat = -1;
          break;
        }
        const int64_t o = shifted / strides[i];
        if (o < 0 || o >= out_dims[i]) {
          flat = -1;
          break;
        }
        flat = flat * out_dims[i] + o;
      }
      if (flat >= 0) {
        index[static_cast<size_t>(tap * in_count + entry)] = flat;
        mask[static_cast<size_t>(tap * in_count + entry)] = 1.0f;
      }
    }
  }
  return {index, mask};
}

struct ConvGeometry {
  int64_t group;
  std::vector<int64_t> kernel;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  std::vector<int64_t> pads_begin;
};

// Conv's attributes resolved against its actual shapes, with auto_pad turned
// into explicit padding -- _conv_geometry in graph_grad.py, refusal for
// refusal. The last check is the one that matters: the resolved geometry has
// to reproduce the node's own output shape, so a misreading of the attributes
// cannot survive into a wrong gradient.
ConvGeometry ConvGeometryOf(const onnx::NodeProto& node, const Shape& x_shape,
                            const Shape& w_shape, const Shape& y_shape) {
  const std::string name = Quoted(node.output(0));
  const size_t rank = x_shape.size();
  if (rank < 3) {
    throw UnsupportedOpError(
        "Conv needs at least one spatial dimension, got input shape " +
        ShapeStr(x_shape) + " (node " + name + ")");
  }
  const size_t spatial = rank - 2;
  if (w_shape.size() != rank || y_shape.size() != rank) {
    throw UnsupportedOpError("Conv's X, W and Y must have the same rank, got " +
                             ShapeStr(x_shape) + ", " + ShapeStr(w_shape) +
                             " and " + ShapeStr(y_shape) + " (node " + name +
                             ")");
  }
  ConvGeometry geo;
  geo.group = AttrInt(node, "group", 1);
  const int64_t channels = x_shape[1];
  const int64_t features = w_shape[0];
  if (geo.group < 1 || channels % geo.group != 0 || features % geo.group != 0) {
    throw UnsupportedOpError(
        "Conv with group=" + std::to_string(geo.group) +
        " does not divide its " + std::to_string(channels) + " input and " +
        std::to_string(features) + " output channels (node " + name + ")");
  }
  if (w_shape[1] != channels / geo.group) {
    throw UnsupportedOpError(
        "Conv's W has " + std::to_string(w_shape[1]) +
        " channels per group, but group=" + std::to_string(geo.group) +
        " over " + std::to_string(channels) + " input channels needs " +
        std::to_string(channels / geo.group) + " (node " + name + ")");
  }

  geo.kernel.assign(w_shape.begin() + 2, w_shape.end());
  const onnx::AttributeProto* declared = FindAttr(node, "kernel_shape");
  if (declared != nullptr) {
    const std::vector<int64_t> spelled(declared->ints().begin(),
                                       declared->ints().end());
    if (spelled != geo.kernel) {
      throw UnsupportedOpError("Conv's kernel_shape attribute " +
                               IntsStr(spelled) +
                               " disagrees with W's own spatial shape " +
                               IntsStr(geo.kernel) + " (node " + name + ")");
    }
  }
  geo.strides = AttrInts(node, "strides", std::vector<int64_t>(spatial, 1));
  geo.dilations = AttrInts(node, "dilations", std::vector<int64_t>(spatial, 1));
  if (geo.strides.size() != spatial || geo.dilations.size() != spatial) {
    throw UnsupportedOpError("Conv's strides " + IntsStr(geo.strides) +
                             " and dilations " + IntsStr(geo.dilations) +
                             " must have one entry per spatial axis (" +
                             std::to_string(spatial) + ") (node " + name + ")");
  }
  for (size_t i = 0; i < spatial; ++i) {
    if (geo.strides[i] < 1 || geo.dilations[i] < 1) {
      throw UnsupportedOpError(
          "Conv with strides " + IntsStr(geo.strides) + " and dilations " +
          IntsStr(geo.dilations) +
          " is not a convolution this rule can invert (node " + name + ")");
    }
  }

  const onnx::AttributeProto* pad_mode = FindAttr(node, "auto_pad");
  const std::string auto_pad =
      pad_mode == nullptr ? std::string("NOTSET") : pad_mode->s();
  std::vector<int64_t> pads;
  if (auto_pad == "NOTSET") {
    pads = AttrInts(node, "pads", std::vector<int64_t>(2 * spatial, 0));
    if (pads.size() != 2 * spatial) {
      throw UnsupportedOpError("Conv's pads " + IntsStr(pads) +
                               " must have two entries per spatial axis (" +
                               std::to_string(spatial) + ") (node " + name +
                               ")");
    }
  } else if (auto_pad == "VALID") {
    pads.assign(2 * spatial, 0);
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    // The spec's own formula, resolved here rather than left to the runtime:
    // the shapes are static, so "same" is a number at build time.
    pads.assign(2 * spatial, 0);
    for (size_t i = 0; i < spatial; ++i) {
      const int64_t size = x_shape[2 + i];
      const int64_t out = (size + geo.strides[i] - 1) / geo.strides[i];
      const int64_t span = (geo.kernel[i] - 1) * geo.dilations[i] + 1;
      const int64_t needed =
          std::max<int64_t>(0, (out - 1) * geo.strides[i] + span - size);
      pads[i] = auto_pad == "SAME_UPPER" ? needed / 2 : needed - needed / 2;
      pads[spatial + i] = needed - pads[i];
    }
  } else {
    throw UnsupportedOpError("Conv with auto_pad '" + auto_pad +
                             "' is not differentiated here (node " + name +
                             ")");
  }

  if (y_shape[0] != x_shape[0] || y_shape[1] != features) {
    throw UnsupportedOpError("Conv's output shape " + ShapeStr(y_shape) +
                             " does not match its input " + ShapeStr(x_shape) +
                             " and weight " + ShapeStr(w_shape) + " (node " +
                             name + ")");
  }
  for (size_t i = 0; i < spatial; ++i) {
    const int64_t span = (geo.kernel[i] - 1) * geo.dilations[i] + 1;
    const int64_t reach = x_shape[2 + i] + pads[i] + pads[spatial + i] - span;
    // A negative reach is spelled out rather than divided: Python's floor
    // division and C++'s truncation disagree there, and this refusal is the
    // one place the two could have produced different numbers.
    const int64_t expected = reach >= 0 ? reach / geo.strides[i] + 1 : 0;
    if (expected != y_shape[2 + i]) {
      throw UnsupportedOpError(
          "Conv's declared output shape " + ShapeStr(y_shape) +
          " does not follow from input " + ShapeStr(x_shape) + ", kernel " +
          IntsStr(geo.kernel) + ", strides " + IntsStr(geo.strides) +
          ", dilations " + IntsStr(geo.dilations) + " and pads " +
          IntsStr(pads) + ": axis " + std::to_string(i) + " should be " +
          std::to_string(expected) + ", not " + std::to_string(y_shape[2 + i]) +
          " (node " + name + ")");
    }
  }
  geo.pads_begin.assign(pads.begin(), pads.begin() + spatial);
  return geo;
}

std::vector<OptStr> GradConv(Backward& ctx, const onnx::NodeProto& node,
                             const std::string& g) {
  // Conv's three gradients, written as im2col plus a MatMul so that nothing
  // outside BackwardOps() is emitted -- neither Conv nor ConvTranspose is in
  // EpFriendlyOps(), and the note beside that set records why they were left
  // out rather than added for this rule. _grad_conv in graph_grad.py carries
  // the derivation; this is a transcription of it, node for node.
  const std::string& x = node.input(0);
  const std::string& w = node.input(1);
  const Shape x_shape = ctx.ShapeOf(x);
  const Shape w_shape = ctx.ShapeOf(w);
  const Shape y_shape = ctx.ShapeOf(node.output(0));
  const ConvGeometry geo = ConvGeometryOf(node, x_shape, w_shape, y_shape);
  bool has_bias = false;
  if (node.input_size() > 2 && !node.input(2).empty()) {
    has_bias = true;
    const Shape bias_shape = ctx.ShapeOf(node.input(2));
    if (bias_shape.size() != 1 || bias_shape[0] != w_shape[0]) {
      throw UnsupportedOpError("Conv's B has shape " + ShapeStr(bias_shape) +
                               ", not (" + std::to_string(w_shape[0]) +
                               ",) (node " + Quoted(node.output(0)) + ")");
    }
  }

  const int64_t batch = x_shape[0];
  const int64_t features = w_shape[0] / geo.group;
  const int64_t channels = x_shape[1] / geo.group;
  const Shape in_dims(x_shape.begin() + 2, x_shape.end());
  const Shape out_dims(y_shape.begin() + 2, y_shape.end());
  const int64_t taps = Prod(geo.kernel);
  const int64_t in_count = Prod(in_dims);
  const int64_t out_count = Prod(out_dims);

  // The incoming gradient with the group axis split out, which is the layout
  // both halves below want: [N, group, M/group, output positions].
  const std::string g_shape =
      ctx.b().ConstInt64({batch, geo.group, features, out_count}, "shape");
  const std::string g4 = ctx.b().Op("Reshape", {g, g_shape}, "reshape");

  // dX = sum over (m, t) of W[m, c, t] * dY[m, position], one gather of dY
  // per tap.
  const std::pair<std::vector<int64_t>, std::vector<float>> col2im =
      Col2ImIndices(in_dims, out_dims, geo.kernel, geo.strides, geo.dilations,
                    geo.pads_begin);
  const std::string g_index = ctx.b().ConstInt64(col2im.first, "idx");
  const std::string gathered_g =
      ctx.b().Op("Gather", {g4, g_index}, {IntAttr("axis", 3)}, "gather");
  const std::string g_mask =
      ctx.b().Const(col2im.second, {1, 1, 1, taps * in_count}, "mask");
  const std::string masked_g = ctx.b().Mul(gathered_g, g_mask);
  const std::string dcol_shape = ctx.b().ConstInt64(
      {batch, geo.group, features * taps, in_count}, "shape");
  const std::string dcol =
      ctx.b().Op("Reshape", {masked_g, dcol_shape}, "reshape");
  const std::string w4_shape =
      ctx.b().ConstInt64({geo.group, features, channels, taps}, "shape");
  const std::string w4 = ctx.b().Op("Reshape", {w, w4_shape}, "reshape");
  const std::string w4t = ctx.b().Transpose(w4, {0, 2, 1, 3});
  // The leading 1 keeps both MatMul operands rank 4.
  const std::string wt_shape =
      ctx.b().ConstInt64({1, geo.group, channels, features * taps}, "shape");
  const std::string wt = ctx.b().Op("Reshape", {w4t, wt_shape}, "reshape");
  const std::string dx4 = ctx.b().MatMul(wt, dcol);
  const std::string dx_shape = ctx.b().ConstInt64(x_shape, "shape");
  const std::string dx = ctx.b().Op("Reshape", {dx4, dx_shape}, "reshape");

  // dW = sum over (n, o) of dY[n, m, o] * col[n, c, t, o], with col the
  // forward's own im2col of X.
  const std::string x4_shape =
      ctx.b().ConstInt64({batch, geo.group, channels, in_count}, "shape");
  const std::string x4 = ctx.b().Op("Reshape", {x, x4_shape}, "reshape");
  const std::pair<std::vector<int64_t>, std::vector<float>> im2col =
      Im2ColIndices(in_dims, out_dims, geo.kernel, geo.strides, geo.dilations,
                    geo.pads_begin);
  const std::string x_index = ctx.b().ConstInt64(im2col.first, "idx");
  const std::string gathered_x =
      ctx.b().Op("Gather", {x4, x_index}, {IntAttr("axis", 3)}, "gather");
  const std::string x_mask =
      ctx.b().Const(im2col.second, {1, 1, 1, taps * out_count}, "mask");
  const std::string masked_x = ctx.b().Mul(gathered_x, x_mask);
  const std::string col_shape = ctx.b().ConstInt64(
      {batch, geo.group, channels * taps, out_count}, "shape");
  const std::string col =
      ctx.b().Op("Reshape", {masked_x, col_shape}, "reshape");
  const std::string colt = ctx.b().Transpose(col, {0, 1, 3, 2});
  const std::string dw4 = ctx.b().MatMul(g4, colt);
  const std::string dw_axes = ctx.b().ConstInt64({0}, "axes");
  const std::string dw3 = ctx.b().Op("ReduceSum", {dw4, dw_axes},
                                     {IntAttr("keepdims", 0)}, "reducesum");
  const std::string dw_shape = ctx.b().ConstInt64(w_shape, "shape");
  const std::string dw = ctx.b().Op("Reshape", {dw3, dw_shape}, "reshape");

  std::vector<OptStr> grads{dx, dw};
  if (node.input_size() > 2) {
    if (!has_bias) {
      grads.push_back(std::nullopt);
    } else {
      const std::string db_axes = ctx.b().ConstInt64({0, 3}, "axes");
      const std::string db = ctx.b().Op("ReduceSum", {g4, db_axes},
                                        {IntAttr("keepdims", 0)}, "reducesum");
      const std::string db_shape = ctx.b().ConstInt64({w_shape[0]}, "shape");
      grads.push_back(ctx.b().Op("Reshape", {db, db_shape}, "reshape"));
    }
  }
  return grads;
}

struct PoolGeometry {
  std::vector<int64_t> kernel;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  std::vector<int64_t> pads_begin;
};

// MaxPool's/AveragePool's attributes resolved against their actual shapes --
// _pool_geometry in graph_grad.py, refusal for refusal. Unlike Conv there is
// no weight tensor to read kernel_shape off, so it is read directly (it is a
// required attribute for both ops); ceil_mode=1 is refused outright rather
// than reproduced, for the reason the Python docstring gives (it can put a
// window's far edge fully inside the padding, a case runtimes do not agree
// on). The final check is the same one ConvGeometryOf ends on: the resolved
// geometry has to reproduce the node's own declared output shape.
PoolGeometry PoolGeometryOf(const onnx::NodeProto& node, const Shape& x_shape,
                            const Shape& y_shape, bool has_dilations) {
  const std::string& op_type = node.op_type();
  const std::string name = Quoted(node.output(0));
  const size_t rank = x_shape.size();
  if (rank < 3) {
    throw UnsupportedOpError(op_type +
                             " needs at least one spatial dimension, got "
                             "input shape " +
                             ShapeStr(x_shape) + " (node " + name + ")");
  }
  const size_t spatial = rank - 2;
  if (y_shape.size() != rank) {
    throw UnsupportedOpError(op_type +
                             "'s X and Y must have the same rank, got " +
                             ShapeStr(x_shape) + " and " + ShapeStr(y_shape) +
                             " (node " + name + ")");
  }
  if (y_shape[0] != x_shape[0] || y_shape[1] != x_shape[1]) {
    throw UnsupportedOpError(op_type + "'s output shape " + ShapeStr(y_shape) +
                             " does not match its input " + ShapeStr(x_shape) +
                             " in batch or channel size (node " + name + ")");
  }

  PoolGeometry geo;
  const onnx::AttributeProto* declared = FindAttr(node, "kernel_shape");
  if (declared == nullptr) {
    throw UnsupportedOpError(op_type +
                             " without a kernel_shape attribute is not "
                             "differentiated here (node " +
                             name + ")");
  }
  geo.kernel.assign(declared->ints().begin(), declared->ints().end());
  if (geo.kernel.size() != spatial) {
    throw UnsupportedOpError(op_type + "'s kernel_shape " +
                             IntsStr(geo.kernel) +
                             " must have one entry per spatial axis (" +
                             std::to_string(spatial) + ") (node " + name + ")");
  }

  geo.strides = AttrInts(node, "strides", std::vector<int64_t>(spatial, 1));
  geo.dilations = has_dilations ? AttrInts(node, "dilations",
                                           std::vector<int64_t>(spatial, 1))
                                : std::vector<int64_t>(spatial, 1);
  if (geo.strides.size() != spatial || geo.dilations.size() != spatial) {
    throw UnsupportedOpError(op_type + "'s strides " + IntsStr(geo.strides) +
                             " and dilations " + IntsStr(geo.dilations) +
                             " must have one entry per spatial axis (" +
                             std::to_string(spatial) + ") (node " + name + ")");
  }
  for (size_t i = 0; i < spatial; ++i) {
    if (geo.strides[i] < 1 || geo.dilations[i] < 1) {
      throw UnsupportedOpError(
          op_type + " with strides " + IntsStr(geo.strides) +
          " and dilations " + IntsStr(geo.dilations) +
          " is not a pooling this rule can invert (node " + name + ")");
    }
  }

  if (AttrInt(node, "ceil_mode", 0) != 0) {
    throw UnsupportedOpError(op_type +
                             " with ceil_mode=1 is not differentiated here "
                             "(node " +
                             name + ")");
  }

  const onnx::AttributeProto* pad_mode = FindAttr(node, "auto_pad");
  const std::string auto_pad =
      pad_mode == nullptr ? std::string("NOTSET") : pad_mode->s();
  std::vector<int64_t> pads;
  if (auto_pad == "NOTSET") {
    pads = AttrInts(node, "pads", std::vector<int64_t>(2 * spatial, 0));
    if (pads.size() != 2 * spatial) {
      throw UnsupportedOpError(op_type + "'s pads " + IntsStr(pads) +
                               " must have two entries per spatial axis (" +
                               std::to_string(spatial) + ") (node " + name +
                               ")");
    }
  } else if (auto_pad == "VALID") {
    pads.assign(2 * spatial, 0);
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    // The spec's own formula, same as ConvGeometryOf's: output_shape[i] =
    // ceil(input_shape[i] / strides[i]), the odd remainder going to the end
    // for SAME_UPPER and the beginning for SAME_LOWER.
    pads.assign(2 * spatial, 0);
    for (size_t i = 0; i < spatial; ++i) {
      const int64_t size = x_shape[2 + i];
      const int64_t out = (size + geo.strides[i] - 1) / geo.strides[i];
      const int64_t span = (geo.kernel[i] - 1) * geo.dilations[i] + 1;
      const int64_t needed =
          std::max<int64_t>(0, (out - 1) * geo.strides[i] + span - size);
      pads[i] = auto_pad == "SAME_UPPER" ? needed / 2 : needed - needed / 2;
      pads[spatial + i] = needed - pads[i];
    }
  } else {
    throw UnsupportedOpError(op_type + " with auto_pad '" + auto_pad +
                             "' is not differentiated here (node " + name +
                             ")");
  }

  for (size_t i = 0; i < spatial; ++i) {
    const int64_t span = (geo.kernel[i] - 1) * geo.dilations[i] + 1;
    const int64_t reach = x_shape[2 + i] + pads[i] + pads[spatial + i] - span;
    // A negative reach is spelled out rather than divided, for the same
    // reason ConvGeometryOf's does: Python's floor division and C++'s
    // truncation disagree there.
    const int64_t expected = reach >= 0 ? reach / geo.strides[i] + 1 : 0;
    if (expected != y_shape[2 + i]) {
      throw UnsupportedOpError(
          op_type + "'s declared output shape " + ShapeStr(y_shape) +
          " does not follow from input " + ShapeStr(x_shape) + ", kernel " +
          IntsStr(geo.kernel) + ", strides " + IntsStr(geo.strides) +
          ", dilations " + IntsStr(geo.dilations) + " and pads " +
          IntsStr(pads) + ": axis " + std::to_string(i) + " should be " +
          std::to_string(expected) + ", not " + std::to_string(y_shape[2 + i]) +
          " (node " + name + ")");
    }
  }
  geo.pads_begin.assign(pads.begin(), pads.begin() + spatial);
  return geo;
}

std::vector<OptStr> GradAveragePool(Backward& ctx, const onnx::NodeProto& node,
                                    const std::string& g) {
  // AveragePool's gradient: col2im of dY, scaled by each window's own
  // divisor -- GradConv's dX half with the MatMul against W deleted
  // outright, since pooling is depthwise and there is no weight to sum
  // over. _grad_averagepool in graph_grad.py carries the derivation; this is
  // a transcription of it, node for node.
  const std::string& x = node.input(0);
  const Shape x_shape = ctx.ShapeOf(x);
  const Shape y_shape = ctx.ShapeOf(node.output(0));
  const PoolGeometry geo =
      PoolGeometryOf(node, x_shape, y_shape, /*has_dilations=*/false);
  const bool count_include_pad = AttrInt(node, "count_include_pad", 0) != 0;

  const int64_t planes = x_shape[0] * x_shape[1];
  const Shape in_dims(x_shape.begin() + 2, x_shape.end());
  const Shape out_dims(y_shape.begin() + 2, y_shape.end());
  const int64_t taps = Prod(geo.kernel);
  const int64_t in_count = Prod(in_dims);
  const int64_t out_count = Prod(out_dims);

  // Every window's divisor, in double precision to match the Python's
  // np.float64. Always the mask-summing general path when
  // count_include_pad=0, never a "no padding" shortcut keyed on pads_begin
  // alone -- pads_begin can be all zero while pads_end is not (asymmetric,
  // end-only padding), and a shortcut keyed on it once got that case wrong
  // on the Python side. The mask sum below already comes out to
  // prod(kernel) everywhere when there truly is no padding, so nothing is
  // lost by always taking it.
  std::vector<double> divisor(static_cast<size_t>(out_count));
  if (count_include_pad) {
    std::fill(divisor.begin(), divisor.end(), static_cast<double>(taps));
  } else {
    const std::pair<std::vector<int64_t>, std::vector<float>> im2col =
        Im2ColIndices(in_dims, out_dims, geo.kernel, geo.strides, geo.dilations,
                      geo.pads_begin);
    for (int64_t tap = 0; tap < taps; ++tap) {
      for (int64_t out = 0; out < out_count; ++out) {
        divisor[static_cast<size_t>(out)] +=
            im2col.second[static_cast<size_t>(tap * out_count + out)];
      }
    }
    for (double d : divisor) {
      if (d == 0.0) {
        throw UnsupportedOpError(
            "AveragePool has a window with no non-padding elements at all "
            "(node " +
            Quoted(node.output(0)) +
            "); its average, and this rule's divisor, are undefined for "
            "that window");
      }
    }
  }
  std::vector<float> recip(divisor.size());
  for (size_t i = 0; i < divisor.size(); ++i) {
    recip[i] = static_cast<float>(1.0 / divisor[i]);
  }

  const std::string g2_shape = ctx.b().ConstInt64({planes, out_count}, "shape");
  const std::string g2 = ctx.b().Op("Reshape", {g, g2_shape}, "reshape");
  const std::string recip_const = ctx.b().Const(recip, {1, out_count}, "recip");
  const std::string scaled = ctx.b().Mul(g2, recip_const);

  const std::pair<std::vector<int64_t>, std::vector<float>> col2im =
      Col2ImIndices(in_dims, out_dims, geo.kernel, geo.strides, geo.dilations,
                    geo.pads_begin);
  const std::string idx_const = ctx.b().ConstInt64(col2im.first, "idx");
  const std::string gathered =
      ctx.b().Op("Gather", {scaled, idx_const}, {IntAttr("axis", 1)}, "gather");
  const std::string mask_const =
      ctx.b().Const(col2im.second, {1, taps * in_count}, "mask");
  const std::string masked = ctx.b().Mul(gathered, mask_const);
  const std::string col_shape =
      ctx.b().ConstInt64({planes, taps, in_count}, "shape");
  const std::string col = ctx.b().Op("Reshape", {masked, col_shape}, "reshape");
  const std::string axes_const = ctx.b().ConstInt64({1}, "axes");
  const std::string dx_flat = ctx.b().Op("ReduceSum", {col, axes_const},
                                         {IntAttr("keepdims", 0)}, "reducesum");
  const std::string dx_shape = ctx.b().ConstInt64(x_shape, "shape");
  const std::string dx = ctx.b().Op("Reshape", {dx_flat, dx_shape}, "reshape");
  return {dx};
}

std::vector<OptStr> GradMaxPool(Backward& ctx, const onnx::NodeProto& node,
                                const std::string& g) {
  // MaxPool's gradient: route dY to whichever input element(s) achieved
  // node.output(0) in each window, without a fresh ReduceMax and without a
  // MatMul. _grad_maxpool in graph_grad.py carries the full derivation (why
  // no ReduceMax is needed, the tie-splitting convention, the per-tap-offset
  // scatter trick this rule needs that GradConv's dX half does not); this is
  // a transcription of it, node for node. MaxPool's optional Indices output
  // is never reached here at all -- BuildBackward refuses any node with more
  // than one declared output before a rule ever runs.
  const std::string& x = node.input(0);
  const std::string& y = node.output(0);
  const Shape x_shape = ctx.ShapeOf(x);
  const Shape y_shape = ctx.ShapeOf(y);
  const PoolGeometry geo =
      PoolGeometryOf(node, x_shape, y_shape, /*has_dilations=*/true);

  const int64_t planes = x_shape[0] * x_shape[1];
  const Shape in_dims(x_shape.begin() + 2, x_shape.end());
  const Shape out_dims(y_shape.begin() + 2, y_shape.end());
  const int64_t taps = Prod(geo.kernel);
  const int64_t in_count = Prod(in_dims);
  const int64_t out_count = Prod(out_dims);

  const std::string x2_shape = ctx.b().ConstInt64({planes, in_count}, "shape");
  const std::string x2 = ctx.b().Op("Reshape", {x, x2_shape}, "reshape");
  const std::pair<std::vector<int64_t>, std::vector<float>> im2col =
      Im2ColIndices(in_dims, out_dims, geo.kernel, geo.strides, geo.dilations,
                    geo.pads_begin);
  for (int64_t out = 0; out < out_count; ++out) {
    float sum = 0.0f;
    for (int64_t tap = 0; tap < taps; ++tap) {
      sum += im2col.second[static_cast<size_t>(tap * out_count + out)];
    }
    if (sum == 0.0f) {
      // A window every tap of which the padding invented has no real
      // element to route dY to at all: the forward's own max over such a
      // window is -inf by construction, which no real input value equals,
      // so the tie count below would be a division by zero rather than a
      // quietly wrong credit. Caught here, at build time, rather than
      // producing a NaN gradient at run time.
      throw UnsupportedOpError(
          "MaxPool has a window with no non-padding elements at all (node " +
          Quoted(node.output(0)) +
          "); its max, and this rule's gradient for that window, are "
          "undefined");
    }
  }
  const std::string x_index = ctx.b().ConstInt64(im2col.first, "idx");
  const std::string gathered =
      ctx.b().Op("Gather", {x2, x_index}, {IntAttr("axis", 1)}, "gather");
  const std::string windows_shape =
      ctx.b().ConstInt64({planes, taps, out_count}, "shape");
  const std::string windows =
      ctx.b().Op("Reshape", {gathered, windows_shape}, "reshape");
  const std::string y3_shape =
      ctx.b().ConstInt64({planes, 1, out_count}, "shape");
  const std::string y3 = ctx.b().Op("Reshape", {y, y3_shape}, "reshape");

  // eq[plane, tap, out] = 1 iff the tap's input element equals the window's
  // max: neither strictly greater (impossible, Y is the max) nor strictly
  // less -- Cast(Greater)/Cast(Less), no Equal needed. A tap the padding
  // invented is excluded outright by multiplying in the same validity mask
  // GradConv uses, since a masked tap's gathered value is an arbitrary
  // element that could spuriously equal Y.
  const std::string one_a = ctx.b().Const(1.0f);
  const std::string greater_mask = ctx.MaskGreater(windows, y3);
  const std::string not_greater = ctx.b().Sub(one_a, greater_mask);
  const std::string one_b = ctx.b().Const(1.0f);
  const std::string less_mask = ctx.MaskLess(windows, y3);
  const std::string not_less = ctx.b().Sub(one_b, less_mask);
  const std::string eq_raw = ctx.b().Mul(not_greater, not_less);
  const std::string valid_const =
      ctx.b().Const(im2col.second, {1, taps, out_count}, "valid");
  const std::string eq = ctx.b().Mul(eq_raw, valid_const);

  // Ties split the credited gradient so a window's total outgoing gradient
  // still sums to exactly the dY that came in -- see _grad_maxpool's
  // docstring for why, and for the near-zero real-world likelihood of a tie
  // this exists to handle without crashing or misbehaving silently.
  const std::string tie_axes = ctx.b().ConstInt64({1}, "axes");
  const std::string tie_count = ctx.b().Op(
      "ReduceSum", {eq, tie_axes}, {IntAttr("keepdims", 1)}, "reducesum");
  const std::string credit = ctx.b().Div(eq, tie_count);

  const std::string g3_shape =
      ctx.b().ConstInt64({planes, 1, out_count}, "shape");
  const std::string g3 = ctx.b().Op("Reshape", {g, g3_shape}, "reshape");
  const std::string contrib = ctx.b().Mul(credit, g3);
  const std::string contrib2_shape =
      ctx.b().ConstInt64({planes, taps * out_count}, "shape");
  const std::string contrib2 =
      ctx.b().Op("Reshape", {contrib, contrib2_shape}, "reshape");

  std::pair<std::vector<int64_t>, std::vector<float>> scatter =
      Col2ImIndices(in_dims, out_dims, geo.kernel, geo.strides, geo.dilations,
                    geo.pads_begin);
  // Offset each tap's block into its own slice of the flattened
  // [taps, out_count] axis contrib2 holds: unlike GradConv's dX (whose
  // gathered value is the same for every tap, since the weight does the
  // differentiating), here each tap's credited gradient genuinely differs
  // per tap, so the plain Col2ImIndices trick of gathering the same
  // [planes, out_count] tensor for every tap does not apply as-is. Tap t's
  // block starts at t * out_count, so one Gather still reaches the right
  // tap's own contribution for every input element.
  for (int64_t tap = 0; tap < taps; ++tap) {
    const int64_t base = tap * in_count;
    const int64_t offset = tap * out_count;
    for (int64_t entry = 0; entry < in_count; ++entry) {
      scatter.first[static_cast<size_t>(base + entry)] += offset;
    }
  }

  const std::string scatter_idx_const =
      ctx.b().ConstInt64(scatter.first, "idx");
  const std::string gathered_back = ctx.b().Op(
      "Gather", {contrib2, scatter_idx_const}, {IntAttr("axis", 1)}, "gather");
  const std::string scatter_mask_const =
      ctx.b().Const(scatter.second, {1, taps * in_count}, "mask");
  const std::string masked_back =
      ctx.b().Mul(gathered_back, scatter_mask_const);
  const std::string col_shape =
      ctx.b().ConstInt64({planes, taps, in_count}, "shape");
  const std::string col =
      ctx.b().Op("Reshape", {masked_back, col_shape}, "reshape");
  const std::string dx_axes = ctx.b().ConstInt64({1}, "axes");
  const std::string dx_flat = ctx.b().Op("ReduceSum", {col, dx_axes},
                                         {IntAttr("keepdims", 0)}, "reducesum");
  const std::string dx_shape = ctx.b().ConstInt64(x_shape, "shape");
  const std::string dx = ctx.b().Op("Reshape", {dx_flat, dx_shape}, "reshape");
  return {dx};
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
                                 const Shape& in_shape,
                                 const Shape& out_shape) {
  const int64_t rank = static_cast<int64_t>(in_shape.size());
  const onnx::AttributeProto* attribute = FindAttr(node, "axes");
  if (attribute != nullptr) {
    if (rank == 0) {
      // Python would raise ZeroDivisionError on the `% rank` below; a
      // rank-0 input to a Reduce* with an axes attribute is malformed either
      // way, and this says so rather than dividing by zero.
      throw UnsupportedOpError("cannot interpret the axes attribute of " +
                               Quoted(node.output(0)) +
                               " against a rank-0 input");
    }
    std::set<int64_t> axes;
    for (int64_t a : attribute->ints()) axes.insert(((a % rank) + rank) % rank);
    return std::vector<int64_t>(axes.begin(), axes.end());
  }

  const bool keepdims = AttrInt(node, "keepdims", 1) != 0;
  if (keepdims) {
    if (static_cast<int64_t>(out_shape.size()) != rank) {
      throw UnsupportedOpError(
          node.op_type() + " with keepdims=1 changed rank " +
          std::to_string(rank) + " to " + std::to_string(out_shape.size()) +
          " (node " + Quoted(node.output(0)) + ")");
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
  const std::string ones =
      ctx.b().Const(std::vector<float>(static_cast<size_t>(elements),
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
  const std::string total = ctx.b().Op("ReduceSum", {gy, axes_const},
                                       {IntAttr("keepdims", 1)}, "reducesum");
  const std::string centred = ctx.b().Sub(g, total);
  return {ctx.b().Mul(y, centred)};
}

std::vector<OptStr> GradLayerNormalization(Backward& ctx,
                                           const onnx::NodeProto& node,
                                           const std::string& g) {
  // With gs = g * scale: dx = inv * (gs - mean(gs) - xhat * mean(gs * xhat)),
  // dscale = sum(g * xhat), db = sum(g). The derivation, and why mu and inv
  // are recomputed here rather than read from the node's optional
  // Mean/InvStdDev outputs, are in _grad_layer_normalization in
  // graph_grad.py.
  const std::string& x = node.input(0);
  const std::string& scale = node.input(1);
  const Shape shape = ctx.ShapeOf(x);
  const int64_t rank = static_cast<int64_t>(shape.size());
  if (rank == 0) {
    // Python would raise ZeroDivisionError on its `% rank`; in C++ the same
    // modulo is undefined behaviour, so a rank-0 input is refused by name.
    throw UnsupportedOpError("LayerNormalization over a rank-0 input (node " +
                             Quoted(node.output(0)) +
                             ") has no axes to normalize");
  }
  const int64_t raw_axis = AttrInt(node, "axis", -1);
  const int64_t axis = ((raw_axis % rank) + rank) % rank;
  const float eps = AttrFloat(node, "epsilon", 1e-5f);
  // The axes are an *attribute* here, not an input: ReduceSum moved its axes
  // to an input at opset 13 and ReduceMean only at opset 18, so at the step
  // graph's opset 17 the two spell the same idea differently.
  std::vector<int64_t> axes;
  for (int64_t i = axis; i < rank; ++i) axes.push_back(i);
  const std::vector<onnx::AttributeProto> reduce_attrs = {
      IntsAttr("axes", axes), IntAttr("keepdims", 1)};

  const std::string mu =
      ctx.b().Op("ReduceMean", {x}, reduce_attrs, "reducemean");
  const std::string xc = ctx.b().Sub(x, mu);
  const std::string xc_squared = ctx.b().Mul(xc, xc);
  const std::string var =
      ctx.b().Op("ReduceMean", {xc_squared}, reduce_attrs, "reducemean");
  const std::string one = ctx.b().Const(1.0f);
  const std::string eps_const = ctx.b().Const(eps);
  const std::string shifted = ctx.b().Add(var, eps_const);
  const std::string deviation = ctx.b().Sqrt(shifted);
  const std::string inv = ctx.b().Div(one, deviation);
  const std::string xhat = ctx.b().Mul(xc, inv);

  const std::string gs = ctx.b().Mul(g, scale);
  const std::string mean_gs =
      ctx.b().Op("ReduceMean", {gs}, reduce_attrs, "reducemean");
  const std::string gs_xhat = ctx.b().Mul(gs, xhat);
  const std::string mean_gs_xhat =
      ctx.b().Op("ReduceMean", {gs_xhat}, reduce_attrs, "reducemean");
  const std::string centred = ctx.b().Sub(gs, mean_gs);
  const std::string correction = ctx.b().Mul(xhat, mean_gs_xhat);
  const std::string inner = ctx.b().Sub(centred, correction);
  const std::string dx = ctx.b().Mul(inv, inner);

  std::vector<OptStr> grads{dx};
  const std::string g_xhat = ctx.b().Mul(g, xhat);
  grads.push_back(ctx.ReduceTo(g_xhat, shape, ctx.ShapeOf(scale)));
  if (node.input_size() > 2 && !node.input(2).empty()) {
    grads.push_back(ctx.ReduceTo(g, shape, ctx.ShapeOf(node.input(2))));
  }
  return grads;
}

std::vector<OptStr> GradBatchNormalization(Backward& ctx,
                                           const onnx::NodeProto& node,
                                           const std::string& g) {
  // Five gradients, in inference mode (training_mode omitted or 0, the
  // opset-17 default). mean/var are the node's own *inputs* here, fixed
  // per-channel numbers rather than reductions of x, so unlike
  // GradLayerNormalization dx does not depend on every other element in its
  // group -- see _grad_batch_normalization in graph_grad.py for the
  // derivation and for why training_mode=1 needs no check in this rule at
  // all (the spec requires it to carry three outputs, so BuildBackward's own
  // single-output check refuses it before this rule ever runs). This is a
  // transcription of the Python, node for node.
  const std::string& x = node.input(0);
  const std::string& scale = node.input(1);
  const std::string& bias = node.input(2);
  const std::string& mean = node.input(3);
  const std::string& var = node.input(4);
  const std::string name = Quoted(node.output(0));
  const Shape x_shape = ctx.ShapeOf(x);
  const int64_t rank = static_cast<int64_t>(x_shape.size());
  if (rank < 2) {
    throw UnsupportedOpError(
        "BatchNormalization needs a batch and a channel axis, got input "
        "shape " +
        ShapeStr(x_shape) + " (node " + name + ")");
  }
  const int64_t channels = x_shape[1];
  const std::vector<std::pair<std::string, std::string>> operands = {
      {"scale", scale}, {"B", bias}, {"mean", mean}, {"var", var}};
  for (const auto& [label, tensor] : operands) {
    const Shape shape = ctx.ShapeOf(tensor);
    if (shape.size() != 1 || shape[0] != channels) {
      throw UnsupportedOpError(
          "BatchNormalization's " + label + " has shape " + ShapeStr(shape) +
          ", not (" + std::to_string(channels) + ",) (node " + name + ")");
    }
  }
  const float eps = AttrFloat(node, "epsilon", 1e-5f);

  Shape bshape{1, channels};
  for (int64_t i = 2; i < rank; ++i) bshape.push_back(1);

  const std::string mean_shape_const = ctx.b().ConstInt64(bshape, "shape");
  const std::string mean_b =
      ctx.b().Op("Reshape", {mean, mean_shape_const}, "reshape");
  const std::string xc = ctx.b().Sub(x, mean_b);

  const std::string one = ctx.b().Const(1.0f);
  const std::string var_shape_const = ctx.b().ConstInt64(bshape, "shape");
  const std::string var_b =
      ctx.b().Op("Reshape", {var, var_shape_const}, "reshape");
  const std::string eps_const = ctx.b().Const(eps);
  const std::string shifted = ctx.b().Add(var_b, eps_const);
  const std::string deviation = ctx.b().Sqrt(shifted);
  const std::string inv = ctx.b().Div(one, deviation);
  const std::string xhat = ctx.b().Mul(xc, inv);

  const std::string scale_shape_const = ctx.b().ConstInt64(bshape, "shape");
  const std::string scale_b =
      ctx.b().Op("Reshape", {scale, scale_shape_const}, "reshape");
  const std::string gs = ctx.b().Mul(g, scale_b);
  const std::string dx = ctx.b().Mul(gs, inv);

  Shape channel_axes{0};
  for (int64_t i = 2; i < rank; ++i) channel_axes.push_back(i);

  const std::string g_xhat = ctx.b().Mul(g, xhat);
  const std::string dscale_axes = ctx.b().ConstInt64(channel_axes, "axes");
  const std::string dscale = ctx.b().Op("ReduceSum", {g_xhat, dscale_axes},
                                        {IntAttr("keepdims", 0)}, "reducesum");

  const std::string dbias_axes = ctx.b().ConstInt64(channel_axes, "axes");
  const std::string dbias = ctx.b().Op("ReduceSum", {g, dbias_axes},
                                       {IntAttr("keepdims", 0)}, "reducesum");

  const std::string dmean_axes = ctx.b().ConstInt64(channel_axes, "axes");
  const std::string dx_summed = ctx.b().Op(
      "ReduceSum", {dx, dmean_axes}, {IntAttr("keepdims", 0)}, "reducesum");
  const std::string dmean = ctx.b().Op("Neg", {dx_summed}, "neg");

  // dvar = -0.5 * sum(dx * xhat * inv): dx * xhat alone is only inv^2 (one
  // factor from each), so this needs one more multiply by inv to reach the
  // inv^3 the chain rule through sqrt actually produces -- see the Python
  // docstring's note on exactly this point.
  const std::string dx_xhat = ctx.b().Mul(dx, xhat);
  const std::string dx_xhat_inv = ctx.b().Mul(dx_xhat, inv);
  const std::string dvar_axes = ctx.b().ConstInt64(channel_axes, "axes");
  const std::string dvar_summed =
      ctx.b().Op("ReduceSum", {dx_xhat_inv, dvar_axes},
                 {IntAttr("keepdims", 0)}, "reducesum");
  const std::string neg_half = ctx.b().Const(-0.5f);
  const std::string dvar = ctx.b().Mul(dvar_summed, neg_half);

  return {dx, dscale, dbias, dmean, dvar};
}

std::vector<OptStr> GradInstanceNormalization(Backward& ctx,
                                              const onnx::NodeProto& node,
                                              const std::string& g) {
  // Three gradients. Same coupling as GradLayerNormalization -- mean and
  // variance are computed from x itself, so every element's gradient depends
  // on every other element in its group through them -- but the group is one
  // (batch, channel) pair, reduced over the spatial axes [2, rank) only,
  // rather than a suffix of axes named by an axis attribute
  // (InstanceNormalization has none; the channel axis is always 1). scale/B
  // are one entry per channel, GradBatchNormalization's broadcast rather
  // than layer-norm's, so scale is reshaped to [1, C, 1, ..., 1] the same
  // way before use. See _grad_instance_normalization in graph_grad.py for
  // the full derivation; this is a transcription of it, node for node.
  const std::string& x = node.input(0);
  const std::string& scale = node.input(1);
  const std::string& bias = node.input(2);
  const std::string name = Quoted(node.output(0));
  const Shape x_shape = ctx.ShapeOf(x);
  const int64_t rank = static_cast<int64_t>(x_shape.size());
  if (rank < 3) {
    throw UnsupportedOpError(
        "InstanceNormalization needs at least one spatial dimension, got "
        "input shape " +
        ShapeStr(x_shape) + " (node " + name + ")");
  }
  const int64_t channels = x_shape[1];
  const std::vector<std::pair<std::string, std::string>> operands = {
      {"scale", scale}, {"B", bias}};
  for (const auto& [label, tensor] : operands) {
    const Shape shape = ctx.ShapeOf(tensor);
    if (shape.size() != 1 || shape[0] != channels) {
      throw UnsupportedOpError(
          "InstanceNormalization's " + label + " has shape " + ShapeStr(shape) +
          ", not (" + std::to_string(channels) + ",) (node " + name + ")");
    }
  }
  const float eps = AttrFloat(node, "epsilon", 1e-5f);

  Shape spatial_axes;
  for (int64_t i = 2; i < rank; ++i) spatial_axes.push_back(i);
  const std::vector<onnx::AttributeProto> reduce_attrs = {
      IntsAttr("axes", spatial_axes), IntAttr("keepdims", 1)};

  Shape bshape{1, channels};
  for (int64_t i = 2; i < rank; ++i) bshape.push_back(1);
  const std::string scale_shape_const = ctx.b().ConstInt64(bshape, "shape");
  const std::string scale_b =
      ctx.b().Op("Reshape", {scale, scale_shape_const}, "reshape");

  const std::string mu =
      ctx.b().Op("ReduceMean", {x}, reduce_attrs, "reducemean");
  const std::string xc = ctx.b().Sub(x, mu);
  const std::string xc_squared = ctx.b().Mul(xc, xc);
  const std::string var =
      ctx.b().Op("ReduceMean", {xc_squared}, reduce_attrs, "reducemean");
  const std::string one = ctx.b().Const(1.0f);
  const std::string eps_const = ctx.b().Const(eps);
  const std::string shifted = ctx.b().Add(var, eps_const);
  const std::string deviation = ctx.b().Sqrt(shifted);
  const std::string inv = ctx.b().Div(one, deviation);
  const std::string xhat = ctx.b().Mul(xc, inv);

  const std::string gs = ctx.b().Mul(g, scale_b);
  const std::string mean_gs =
      ctx.b().Op("ReduceMean", {gs}, reduce_attrs, "reducemean");
  const std::string gs_xhat = ctx.b().Mul(gs, xhat);
  const std::string mean_gs_xhat =
      ctx.b().Op("ReduceMean", {gs_xhat}, reduce_attrs, "reducemean");
  const std::string centred = ctx.b().Sub(gs, mean_gs);
  const std::string correction = ctx.b().Mul(xhat, mean_gs_xhat);
  const std::string inner = ctx.b().Sub(centred, correction);
  const std::string dx = ctx.b().Mul(inv, inner);

  const std::string g_xhat = ctx.b().Mul(g, xhat);
  Shape channel_axes{0};
  channel_axes.insert(channel_axes.end(), spatial_axes.begin(),
                      spatial_axes.end());
  const std::string dscale_axes = ctx.b().ConstInt64(channel_axes, "axes");
  const std::string dscale = ctx.b().Op("ReduceSum", {g_xhat, dscale_axes},
                                        {IntAttr("keepdims", 0)}, "reducesum");
  const std::string dbias_axes = ctx.b().ConstInt64(channel_axes, "axes");
  const std::string dbias = ctx.b().Op("ReduceSum", {g, dbias_axes},
                                       {IntAttr("keepdims", 0)}, "reducesum");

  return {dx, dscale, dbias};
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

std::vector<OptStr> GradGather(Backward& ctx, const onnx::NodeProto& node,
                               const std::string& g) {
  // Gather's gradient: a scatter-add into `data`, `indices` itself untouched
  // -- the embedding-lookup case, not the constant-index-table use GradConv
  // makes of the *forward* Gather. There is no scatter op in BackwardOps(),
  // so the accumulation is written as a one-hot matmul instead; the full
  // derivation, the shape this rule handles (data: [..., N, ...], indices
  // rank 0 or 1), and what it costs are in _grad_gather in graph_grad.py.
  // This is a transcription of it, node for node.
  const std::string& data = node.input(0);
  const std::string& indices = node.input(1);
  const Shape data_shape = ctx.ShapeOf(data);
  const Shape idx_shape = ctx.ShapeOf(indices);
  if (idx_shape.size() > 1) {
    throw UnsupportedOpError(
        "Gather with rank-" + std::to_string(idx_shape.size()) +
        " indices is not differentiated here (node " + Quoted(node.output(0)) +
        "); only a scalar or a rank-1 index vector is supported");
  }
  const int64_t rank = static_cast<int64_t>(data_shape.size());
  if (rank == 0) {
    // Python would raise ZeroDivisionError on its `% rank` below; in C++ the
    // same modulo is undefined behaviour, so a rank-0 data tensor -- which
    // Gather's own spec never produces anyway -- is refused by name instead.
    throw UnsupportedOpError("Gather over a rank-0 data tensor (node " +
                             Quoted(node.output(0)) + ") has no axis to index");
  }
  const int64_t raw_axis = AttrInt(node, "axis", 0);
  const int64_t axis = ((raw_axis % rank) + rank) % rank;
  const int64_t count = data_shape[static_cast<size_t>(axis)];
  const Shape pre_shape(data_shape.begin(), data_shape.begin() + axis);
  const Shape post_shape(data_shape.begin() + axis + 1, data_shape.end());
  const int64_t pre = Prod(pre_shape);
  const int64_t post = Prod(post_shape);
  const int64_t length = Prod(idx_shape);  // 1 for a scalar, L for a vector

  // indices as a flat float32 [length] vector, with negative values resolved
  // to indices[i] + count -- a run-time value, so this happens in the graph
  // rather than at build time.
  const std::string flat_shape = ctx.b().ConstInt64({length}, "shape");
  const std::string flat_idx =
      ctx.b().Op("Reshape", {indices, flat_shape}, "reshape");
  const std::string idx_f = ctx.b().Op(
      "Cast", {flat_idx}, {IntAttr("to", onnx::TensorProto::FLOAT)}, "cast");
  const std::string zero = ctx.b().Const(0.0f);
  const std::string is_negative = ctx.MaskLess(idx_f, zero);
  const std::string count_const = ctx.b().Const(static_cast<float>(count));
  const std::string negative_amount = ctx.b().Mul(is_negative, count_const);
  const std::string idx_resolved = ctx.b().Add(idx_f, negative_amount);

  // onehot[l, n] = (indices[l] == n), as a float32 [length, count] matrix --
  // the same two-sided (1 - greater) * (1 - less) trick every other rule
  // here uses to build a boolean without an Equal node.
  const std::string col_shape = ctx.b().ConstInt64({length, 1}, "shape");
  const std::string idx_col =
      ctx.b().Op("Reshape", {idx_resolved, col_shape}, "reshape");
  std::vector<float> arange_values(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    arange_values[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  const std::string arange = ctx.b().Const(arange_values, {count}, "arange");
  const std::string row_shape = ctx.b().ConstInt64({1, count}, "shape");
  const std::string arange_row =
      ctx.b().Op("Reshape", {arange, row_shape}, "reshape");
  const std::string one_a = ctx.b().Const(1.0f);
  const std::string greater_mask = ctx.MaskGreater(idx_col, arange_row);
  const std::string not_greater = ctx.b().Sub(one_a, greater_mask);
  const std::string one_b = ctx.b().Const(1.0f);
  const std::string less_mask = ctx.MaskLess(idx_col, arange_row);
  const std::string not_less = ctx.b().Sub(one_b, less_mask);
  const std::string onehot = ctx.b().Mul(not_greater, not_less);

  // dData[n, ...] = sum_l onehot[l, n] * dY[l, ...], batched over `pre` with
  // a broadcasting leading axis so one onehot matrix serves every batch
  // element -- see GradConv's `wt` for the same trick.
  const std::string onehot_t = ctx.b().Transpose(onehot, {1, 0});
  const std::string batched_shape =
      ctx.b().ConstInt64({1, count, length}, "shape");
  const std::string onehot_batched =
      ctx.b().Op("Reshape", {onehot_t, batched_shape}, "reshape");
  const std::string dy_shape = ctx.b().ConstInt64({pre, length, post}, "shape");
  const std::string dy_flat = ctx.b().Op("Reshape", {g, dy_shape}, "reshape");
  const std::string ddata_flat = ctx.b().MatMul(onehot_batched, dy_flat);
  const std::string ddata_shape = ctx.b().ConstInt64(data_shape, "shape");
  const std::string ddata =
      ctx.b().Op("Reshape", {ddata_flat, ddata_shape}, "reshape");

  return {ddata, std::nullopt};
}

// ---------------------------------------------------------------------------
// Templated rules (proof of concept)
// ---------------------------------------------------------------------------
//
// An alternative to hand-transcribing a rule's graph construction directly
// here (and, separately, in graph_grad.py): author the rule once in
// onnxscript, compile it offline to a checked-in ONNX FunctionProto
// (scripts/codegen/generate_grad_templates.py ->
// graph_grad_templates_gen.h/.py), and instantiate the same text from either
// language via ONNX's own function-inlining machinery. See
// graph_grad.py's matching section for the Python original and the full
// rationale (in particular why these functions take no ONNX-level
// attributes).
//
// Not wired into Rules(): these two functions exist to prove the mechanism
// (see graph_grad_templates_test.cpp, which cross-checks
// GradBatchNormalizationTemplated against the hand-written
// GradBatchNormalization above) before any production rule is migrated to
// it. Every real caller of BuildBackward still gets Rules() unchanged.

const onnx::FunctionProto& GradAddTemplate() {
  static const onnx::FunctionProto* fn = [] {
    auto* proto = new onnx::FunctionProto();
    const auto status = onnx::OnnxParser::Parse(*proto, kGradAddTemplate);
    if (!status.IsOK()) {
      throw std::logic_error("failed to parse the GradAdd template: " +
                             status.ErrorMessage());
    }
    return proto;
  }();
  return *fn;
}

const onnx::FunctionProto& GradBatchNormalizationTemplate() {
  static const onnx::FunctionProto* fn = [] {
    auto* proto = new onnx::FunctionProto();
    const auto status =
        onnx::OnnxParser::Parse(*proto, kGradBatchNormalizationTemplate);
    if (!status.IsOK()) {
      throw std::logic_error(
          "failed to parse the GradBatchNormalization template: " +
          status.ErrorMessage());
    }
    return proto;
  }();
  return *fn;
}

// Same shape as GradAdd, but the identity-gradient core (da = db = g) comes
// from a call to the checked-in GradAdd function instead of being written
// here directly. Broadcast-undoing stays outside the template, exactly as
// it does for the hand-written rule.
std::vector<OptStr> GradAddTemplated(Backward& ctx, const onnx::NodeProto& node,
                                     const std::string& g) {
  const std::vector<std::string> outs = ctx.b().Call(GradAddTemplate(), {g});
  const Shape out = ctx.ShapeOf(node.output(0));
  const std::string ga = ctx.ReduceTo(outs[0], out, ctx.ShapeOf(node.input(0)));
  const std::string gb = ctx.ReduceTo(outs[1], out, ctx.ShapeOf(node.input(1)));
  return {ga, gb};
}

// Same validation and shape-resolution as GradBatchNormalization (rank/shape
// checks, the per-channel broadcast reshape, the reduction axes), but the
// actual gradient arithmetic comes from a call to the checked-in
// GradBatchNormalization function -- the rule this repo's own
// dvar-derivation bug was in, so the one most worth proving out this way
// first.
std::vector<OptStr> GradBatchNormalizationTemplated(Backward& ctx,
                                                    const onnx::NodeProto& node,
                                                    const std::string& g) {
  const std::string& x = node.input(0);
  const std::string& scale = node.input(1);
  const std::string& bias = node.input(2);
  const std::string& mean = node.input(3);
  const std::string& var = node.input(4);
  const std::string name = Quoted(node.output(0));
  const Shape x_shape = ctx.ShapeOf(x);
  const int64_t rank = static_cast<int64_t>(x_shape.size());
  if (rank < 2) {
    throw UnsupportedOpError(
        "BatchNormalization needs a batch and a channel axis, got input "
        "shape " +
        ShapeStr(x_shape) + " (node " + name + ")");
  }
  const int64_t channels = x_shape[1];
  const std::vector<std::pair<std::string, std::string>> operands = {
      {"scale", scale}, {"B", bias}, {"mean", mean}, {"var", var}};
  for (const auto& [label, tensor] : operands) {
    const Shape shape = ctx.ShapeOf(tensor);
    if (shape.size() != 1 || shape[0] != channels) {
      throw UnsupportedOpError(
          "BatchNormalization's " + label + " has shape " + ShapeStr(shape) +
          ", not (" + std::to_string(channels) + ",) (node " + name + ")");
    }
  }
  const float eps = AttrFloat(node, "epsilon", 1e-5f);

  Shape bshape{1, channels};
  for (int64_t i = 2; i < rank; ++i) bshape.push_back(1);
  const std::string mean_b = ctx.b().Op(
      "Reshape", {mean, ctx.b().ConstInt64(bshape, "shape")}, "reshape");
  const std::string var_b = ctx.b().Op(
      "Reshape", {var, ctx.b().ConstInt64(bshape, "shape")}, "reshape");
  const std::string scale_b = ctx.b().Op(
      "Reshape", {scale, ctx.b().ConstInt64(bshape, "shape")}, "reshape");

  Shape channel_axes{0};
  for (int64_t i = 2; i < rank; ++i) channel_axes.push_back(i);
  const std::string channel_axes_const =
      ctx.b().ConstInt64(channel_axes, "axes");
  const std::string eps_const = ctx.b().Const(eps);
  const std::string one_const = ctx.b().Const(1.0f);
  const std::string neg_half_const = ctx.b().Const(-0.5f);

  const std::vector<std::string> outs =
      ctx.b().Call(GradBatchNormalizationTemplate(),
                   {g, x, mean_b, var_b, scale_b, eps_const, channel_axes_const,
                    one_const, neg_half_const});
  return {outs[0], outs[1], outs[2], outs[3], outs[4]};
}

const std::map<std::string, Rule>& Rules() {
  static const std::map<std::string, Rule>* rules =
      new std::map<std::string, Rule>{
          {"Add", &GradAdd},
          {"AveragePool", &GradAveragePool},
          {"BatchNormalization", &GradBatchNormalization},
          {"Clip", &GradClip},
          {"Conv", &GradConv},
          {"Div", &GradDiv},
          {"Erf", &GradErf},
          {"Exp", &GradExp},
          {"Gather", &GradGather},
          {"Gemm", &GradGemm},
          {"Identity", &GradIdentity},
          {"InstanceNormalization", &GradInstanceNormalization},
          {"LayerNormalization", &GradLayerNormalization},
          {"MatMul", &GradMatMul},
          {"MaxPool", &GradMaxPool},
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

// "['Add', 'Clip', ...]", the way the Python's sorted(rules) prints inside
// its refusal message. `rules` rather than the global Rules()/SupportedOps()
// so a caller with a different rule table (BuildBackwardWithTemplatedRules)
// gets an accurate list too; std::map already iterates in key order.
std::string SupportedOpsList(const std::map<std::string, Rule>& rules) {
  std::string out = "[";
  bool first = true;
  for (const auto& entry : rules) {
    if (!first) out += ", ";
    first = false;
    out += Quoted(entry.first);
  }
  return out + "]";
}

// The core of BuildBackward, parameterized over the rule table -- see
// BuildBackward and BuildBackwardWithTemplatedRules, its two callers.
std::map<std::string, std::string> BuildBackwardImpl(
    GraphBuilder& b, const std::vector<onnx::NodeProto>& nodes,
    const std::map<std::string, Shape>& shapes,
    const std::map<std::string, std::string>& grad_outputs,
    const std::vector<std::string>& targets,
    const std::map<std::string, Rule>& rules) {
  Backward ctx(b, shapes);
  std::map<std::string, std::string> grads(grad_outputs);

  // Reverse topological order, which is what makes the accumulation below
  // correct: every reader of a tensor is visited before its producer, so by
  // the time a producer asks for its output gradient the sum is complete.
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    const onnx::NodeProto& node = *it;
    const auto rule = rules.find(node.op_type());
    if (rule == rules.end()) {
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
                               SupportedOpsList(rules));
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

}  // namespace

const std::set<std::string>& BackwardOps() {
  // ReduceMean and Sqrt were admitted for GradLayerNormalization, which
  // needs a mean over the normalized axes and the reciprocal square root of
  // the variance; the Python set says at length why that is not a loosening
  // of the criterion.
  // Gather was admitted for GradConv, which writes a convolution's gradient
  // as im2col rather than as the ConvTranspose it naturally is; the Python
  // set says why that is not a loosening either, and the EP_FRIENDLY_OPS note
  // in qat_graph.py records what a Conv/ConvTranspose membership would have
  // cost instead.
  // GradBatchNormalization and GradInstanceNormalization needed nothing from
  // this set at all -- every op their gradients use was already here for
  // GradLayerNormalization or the elementwise rules.
  static const std::set<std::string>* ops = new std::set<std::string>{
      "Add",     "Cast",   "Div", "Exp",      "Gather",     "Greater",
      "Less",    "MatMul", "Mul", "Neg",      "ReduceMean", "ReduceSum",
      "Reshape", "Sqrt",   "Sub", "Transpose"};
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
  return BuildBackwardImpl(b, nodes, shapes, grad_outputs, targets, Rules());
}

std::map<std::string, std::string> BuildBackwardWithTemplatedRules(
    GraphBuilder& b, const std::vector<onnx::NodeProto>& nodes,
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const std::map<std::string, std::string>& grad_outputs,
    const std::vector<std::string>& targets) {
  std::map<std::string, Rule> rules(Rules());
  rules["Add"] = &GradAddTemplated;
  rules["BatchNormalization"] = &GradBatchNormalizationTemplated;
  return BuildBackwardImpl(b, nodes, shapes, grad_outputs, targets, rules);
}
