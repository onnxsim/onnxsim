#include "sym_shape_infer.h"

#include <algorithm>
#include <set>
#include <string>

namespace onnxsim {
namespace {

std::optional<int64_t> ConcreteInt(const SymExpr& e) {
  if (e.is_symbolic()) return std::nullopt;
  return e.to_int();
}

// Ops whose output(0) shape equals input(0) shape. Elementwise ops with extra
// data inputs (Clip's min/max, PRelu's slope, LayerNormalization's scale/bias)
// still take their shape from input(0); so do the in-place scatters
// (ScatterND/ScatterElements), whose output has the shape of `data` (input 0)
// -- this is what lets M2 shape the causal-mask subgraph in encoder/decoder
// models like BART instead of stalling at Shape(ScatterND(...)).
bool IsUnaryShapePreserving(const std::string& op) {
  static const std::set<std::string> kUnary = {
      "Relu",
      "LeakyRelu",
      "PRelu",
      "Sigmoid",
      "HardSigmoid",
      "HardSwish",
      "Tanh",
      "Erf",
      "Gelu",
      "Exp",
      "Log",
      "Sqrt",
      "Reciprocal",
      "Neg",
      "Abs",
      "Sign",
      "Softmax",
      "LogSoftmax",
      "Hardmax",
      "Softplus",
      "Softsign",
      "Elu",
      "Selu",
      "Celu",
      "Mish",
      "Cast",
      "CastLike",
      "Identity",
      "Clip",
      "Round",
      "Floor",
      "Ceil",
      "Not",
      "Shrink",
      "ThresholdedRelu",
      "LayerNormalization",
      "BatchNormalization",
      "InstanceNormalization",
      "GroupNormalization",
      "RMSNormalization",
      "SimplifiedLayerNormalization",
      "Dropout",
      "LpNormalization",
      "QuantizeLinear",
      "DequantizeLinear",
      "ScatterND",
      "ScatterElements",
  };
  return kUnary.count(op) != 0;
}

// Ops that broadcast their (n) inputs to a common shape.
bool IsBroadcastNary(const std::string& op) {
  static const std::set<std::string> kBroadcast = {
      "Add",         "Sub",        "Mul",       "Div",        "Pow",
      "Mod",         "Max",        "Min",       "Mean",       "Sum",
      "Greater",     "Less",       "Equal",     "And",        "Or",
      "Xor",         "BitwiseAnd", "BitwiseOr", "BitwiseXor", "GreaterOrEqual",
      "LessOrEqual",
  };
  return kBroadcast.count(op) != 0;
}

int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0) axis += rank;
  return axis;
}

class ShapeInferer {
 public:
  explicit ShapeInferer(const ShapeGraph& graph) : graph_(graph) {
    for (const auto& [name, shape] : graph.value_info) shapes_[name] = shape;
  }

  std::map<std::string, SymShape> Run() {
    for (const SymNode& node : graph_.node) {
      auto shape = Infer(node);
      if (shape && !node.output.empty() && !node.output[0].empty()) {
        shapes_[node.output[0]] = std::move(*shape);
      }
    }
    return shapes_;
  }

 private:
  const SymShape* Shape(const std::string& name) const {
    if (name.empty()) return nullptr;
    auto it = shapes_.find(name);
    return it == shapes_.end() ? nullptr : &it->second;
  }

  // A distinct symbol for a dim no rule can compute exactly. Deterministic
  // (a monotonic counter, no RNG) and never merged with another, so a shape is
  // only ever left less resolved -- never wrongly equated with a real dim.
  SymExpr Fresh() { return SymExpr::Symbol("unk_" + std::to_string(unk_++)); }

  std::optional<int64_t> IntAttr(const SymNode& node, const std::string& name,
                                 int64_t fallback) const {
    if (const SymAttr* a = node.attr(name)) {
      if (a->i) return *a->i;
    }
    return fallback;
  }

  // The int list from attribute `attr_name`, else from initializer input
  // `input_index` (opset >= 13 moved axes/starts/... to data inputs). Requires
  // every element concrete.
  std::optional<std::vector<int64_t>> IntListOperand(
      const SymNode& node, const std::string& attr_name,
      std::size_t input_index) const {
    if (const SymAttr* a = node.attr(attr_name)) return a->ints;
    if (input_index < node.input.size()) {
      auto it = graph_.initializer.find(node.input[input_index]);
      if (it != graph_.initializer.end()) {
        std::vector<int64_t> out;
        for (const SymExpr& e : it->second.data) {
          auto c = ConcreteInt(e);
          if (!c) return std::nullopt;
          out.push_back(*c);
        }
        return out;
      }
    }
    return std::nullopt;
  }

  // --- broadcasting -------------------------------------------------------

  SymExpr BroadcastDim(const SymExpr& a, const SymExpr& b) {
    auto ca = ConcreteInt(a);
    auto cb = ConcreteInt(b);
    if (ca && *ca == 1) return b;
    if (cb && *cb == 1) return a;
    auto eq = TryEqual(a, b);
    if (eq && *eq) return a;            // provably the same dim
    if (ca && cb) return SymExpr(*ca);  // both concrete (validated equal above)
    return Fresh();  // two differing symbolic dims: unknown, keep it distinct
  }

  SymShape BroadcastShapes(const std::vector<const SymShape*>& shapes) {
    std::size_t rank = 0;
    for (const SymShape* s : shapes) rank = std::max(rank, s->size());
    SymShape out(rank, SymExpr(1));
    for (std::size_t i = 0; i < rank; ++i) {
      bool have = false;
      for (const SymShape* s : shapes) {
        // Right-align: position i (from the left of the rank-`rank` result)
        // maps to this shape's dim only if it is long enough.
        const std::int64_t idx = static_cast<std::int64_t>(s->size()) -
                                 static_cast<std::int64_t>(rank) +
                                 static_cast<std::int64_t>(i);
        if (idx < 0) continue;
        const SymExpr& d = (*s)[static_cast<std::size_t>(idx)];
        out[i] = have ? BroadcastDim(out[i], d) : d;
        have = true;
      }
    }
    return out;
  }

  // --- dispatch -----------------------------------------------------------

  std::optional<SymShape> Infer(const SymNode& node) {
    const std::string& op = node.op_type;
    if (op == "MatMul") return InferMatMul(node);
    if (op == "Gemm") return InferGemm(node);
    if (op == "Conv") return InferConvPool(node, /*is_pool=*/false);
    if (op == "MaxPool" || op == "AveragePool" || op == "LpPool")
      return InferConvPool(node, /*is_pool=*/true);
    if (op == "GlobalAveragePool" || op == "GlobalMaxPool" ||
        op == "GlobalLpPool")
      return InferGlobalPool(node);
    if (op == "Transpose") return InferTranspose(node);
    if (op == "Concat") return InferConcat(node);
    if (op == "Flatten") return InferFlatten(node);
    if (op == "Reshape") return InferReshape(node);
    if (op == "Squeeze") return InferSqueeze(node);
    if (op == "Unsqueeze") return InferUnsqueeze(node);
    if (op == "Expand") return InferExpand(node);
    if (op == "Gather") return InferGather(node);
    if (op == "Slice") return InferSlice(node);
    if (op.rfind("Reduce", 0) == 0) return InferReduce(node);
    if (op == "Where") return InferWhere(node);
    if (IsUnaryShapePreserving(op)) {
      if (const SymShape* s = Shape(node.input.empty() ? "" : node.input[0]))
        return *s;
      return std::nullopt;
    }
    if (IsBroadcastNary(op)) return InferBroadcast(node);
    return std::nullopt;
  }

  std::optional<SymShape> InferBroadcast(const SymNode& node) {
    std::vector<const SymShape*> ins;
    for (const std::string& name : node.input) {
      if (name.empty()) continue;
      const SymShape* s = Shape(name);
      if (!s) return std::nullopt;
      ins.push_back(s);
    }
    if (ins.empty()) return std::nullopt;
    return BroadcastShapes(ins);
  }

  std::optional<SymShape> InferWhere(const SymNode& node) {
    if (node.input.size() < 3) return std::nullopt;
    std::vector<const SymShape*> ins;
    for (int i = 0; i < 3; ++i) {
      const SymShape* s = Shape(node.input[i]);
      if (!s) return std::nullopt;
      ins.push_back(s);
    }
    return BroadcastShapes(ins);
  }

  std::optional<SymShape> InferMatMul(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymShape* a = Shape(node.input[0]);
    const SymShape* b = Shape(node.input[1]);
    if (!a || !b || a->empty() || b->empty()) return std::nullopt;
    // numpy matmul: a rank-1 lhs gets a leading 1, a rank-1 rhs a trailing 1;
    // both are stripped from the result.
    SymShape la = *a, lb = *b;
    const bool a_vec = la.size() == 1;
    const bool b_vec = lb.size() == 1;
    if (a_vec) la.insert(la.begin(), SymExpr(1));
    if (b_vec) lb.push_back(SymExpr(1));
    // Broadcast the batch dims (everything but the trailing 2x2 matrix).
    SymShape a_batch(la.begin(), la.end() - 2);
    SymShape b_batch(lb.begin(), lb.end() - 2);
    SymShape out = BroadcastShapes({&a_batch, &b_batch});
    // M's position is fixed once the batch dims are laid down; erase it there
    // (rather than at out.end() - 2) so a_vec's erase is still correct after
    // b_vec's pop_back has already shortened `out` -- e.g. both inputs rank-1
    // (a plain dot product), where out.end() - 2 would underflow before
    // begin() once N has already been popped.
    const std::size_t m_pos = out.size();
    out.push_back(la[la.size() - 2]);  // M
    out.push_back(lb[lb.size() - 1]);  // N
    if (b_vec) out.pop_back();
    if (a_vec) out.erase(out.begin() + static_cast<std::ptrdiff_t>(m_pos));
    return out;
  }

  std::optional<SymShape> InferGemm(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymShape* a = Shape(node.input[0]);
    const SymShape* b = Shape(node.input[1]);
    if (!a || !b || a->size() != 2 || b->size() != 2) return std::nullopt;
    const bool trans_a = IntAttr(node, "transA", 0).value() != 0;
    const bool trans_b = IntAttr(node, "transB", 0).value() != 0;
    const SymExpr m = trans_a ? (*a)[1] : (*a)[0];
    const SymExpr n = trans_b ? (*b)[0] : (*b)[1];
    return SymShape{m, n};
  }

  // Conv (M from the weight's first dim) and pooling (channels preserved) share
  // the spatial output formula.
  std::optional<SymShape> InferConvPool(const SymNode& node, bool is_pool) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    if (!x || x->size() < 3) return std::nullopt;  // [N, C, spatial...]
    const int64_t spatial = static_cast<int64_t>(x->size()) - 2;

    SymExpr channels_out = (*x)[1];
    std::vector<int64_t> kernel;
    if (!is_pool) {
      if (node.input.size() < 2) return std::nullopt;
      const SymShape* w = Shape(node.input[1]);
      if (!w || static_cast<int64_t>(w->size()) != spatial + 2)
        return std::nullopt;
      channels_out = (*w)[0];  // M
      for (int64_t i = 0; i < spatial; ++i) {
        auto k = ConcreteInt((*w)[static_cast<std::size_t>(i + 2)]);
        if (!k) return std::nullopt;
        kernel.push_back(*k);
      }
    } else {
      if (const SymAttr* a = node.attr("kernel_shape"))
        kernel = a->ints;
      else
        return std::nullopt;
      if (static_cast<int64_t>(kernel.size()) != spatial) return std::nullopt;
    }

    const auto strides = node.attr("strides");
    const auto pads = node.attr("pads");
    const auto dilations = node.attr("dilations");
    const bool ceil_mode = IntAttr(node, "ceil_mode", 0).value() != 0;
    std::string auto_pad;
    if (const SymAttr* ap = node.attr("auto_pad")) {
      (void)ap;  // auto_pad is a string attribute, not modelled by SymAttr;
                 // its presence alone can't be acted on, so treat as NOTSET.
    }

    SymShape out;
    out.push_back((*x)[0]);       // N
    out.push_back(channels_out);  // C or M
    for (int64_t i = 0; i < spatial; ++i) {
      const SymExpr& in_dim = (*x)[static_cast<std::size_t>(i + 2)];
      auto in = ConcreteInt(in_dim);
      const int64_t s =
          strides && i < (int64_t)strides->ints.size() ? strides->ints[i] : 1;
      const int64_t d = dilations && i < (int64_t)dilations->ints.size()
                            ? dilations->ints[i]
                            : 1;
      const int64_t pb =
          pads && 2 * i < (int64_t)pads->ints.size() ? pads->ints[2 * i] : 0;
      const int64_t pe = pads && 2 * i + 1 < (int64_t)pads->ints.size()
                             ? pads->ints[2 * i + 1]
                             : 0;
      // Exact only when the input spatial dim is a concrete int; a strided
      // window over a symbolic length is not a polynomial, so it gets a fresh
      // symbol.
      if (!in) {
        out.push_back(Fresh());
        continue;
      }
      const int64_t k = kernel[static_cast<std::size_t>(i)];
      const int64_t eff = d * (k - 1) + 1;  // dilated kernel extent
      const int64_t numer = *in + pb + pe - eff;
      if (numer < 0) {
        out.push_back(Fresh());
        continue;
      }
      int64_t o = ceil_mode ? (numer + s - 1) / s + 1 : numer / s + 1;
      out.push_back(SymExpr(o));
    }
    return out;
  }

  std::optional<SymShape> InferGlobalPool(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    if (!x || x->size() < 2) return std::nullopt;
    SymShape out{(*x)[0], (*x)[1]};
    for (std::size_t i = 2; i < x->size(); ++i) out.push_back(SymExpr(1));
    return out;
  }

  std::optional<SymShape> InferTranspose(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    if (!x) return std::nullopt;
    const int64_t rank = static_cast<int64_t>(x->size());
    std::vector<int64_t> perm;
    if (const SymAttr* a = node.attr("perm"))
      perm = a->ints;
    else
      for (int64_t i = rank - 1; i >= 0; --i)
        perm.push_back(i);  // default: reverse
    if (static_cast<int64_t>(perm.size()) != rank) return std::nullopt;
    SymShape out;
    for (int64_t p : perm) {
      p = NormalizeAxis(p, rank);
      if (p < 0 || p >= rank) return std::nullopt;
      out.push_back((*x)[static_cast<std::size_t>(p)]);
    }
    return out;
  }

  std::optional<SymShape> InferConcat(const SymNode& node) {
    std::vector<const SymShape*> ins;
    for (const std::string& name : node.input) {
      const SymShape* s = Shape(name);
      if (!s) return std::nullopt;
      ins.push_back(s);
    }
    if (ins.empty()) return std::nullopt;
    const int64_t rank = static_cast<int64_t>(ins[0]->size());
    int64_t axis = NormalizeAxis(IntAttr(node, "axis", 0).value(), rank);
    if (axis < 0 || axis >= rank) return std::nullopt;
    SymShape out = *ins[0];
    SymExpr sum = out[static_cast<std::size_t>(axis)];
    for (std::size_t k = 1; k < ins.size(); ++k) {
      if (static_cast<int64_t>(ins[k]->size()) != rank) return std::nullopt;
      sum = sum + (*ins[k])[static_cast<std::size_t>(axis)];
    }
    out[static_cast<std::size_t>(axis)] = sum;
    return out;
  }

  std::optional<SymShape> InferFlatten(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    if (!x) return std::nullopt;
    const int64_t rank = static_cast<int64_t>(x->size());
    int64_t axis = NormalizeAxis(IntAttr(node, "axis", 1).value(), rank);
    if (axis < 0 || axis > rank) return std::nullopt;
    SymExpr outer(1), inner(1);
    for (int64_t i = 0; i < axis; ++i)
      outer = outer * (*x)[static_cast<std::size_t>(i)];
    for (int64_t i = axis; i < rank; ++i)
      inner = inner * (*x)[static_cast<std::size_t>(i)];
    return SymShape{outer, inner};
  }

  std::optional<SymShape> InferReshape(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    auto target = IntListOperand(node, "shape", 1);
    if (!target) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);  // may be null
    SymShape out(target->size());
    int neg_one = -1;
    for (std::size_t i = 0; i < target->size(); ++i) {
      const int64_t t = (*target)[i];
      if (t == -1) {
        neg_one = static_cast<int>(i);
        continue;
      }
      if (t ==
          0) {  // copy the input dim at this position (allowzero=0 default)
        if (x && i < x->size())
          out[i] = (*x)[i];
        else
          out[i] = Fresh();
      } else {
        out[i] = SymExpr(t);
      }
    }
    if (neg_one >= 0) {
      // Resolve the -1 as total // product(other dims) when the input is fully
      // known; otherwise leave that single slot a fresh symbol.
      SymExpr resolved;
      bool ok = false;
      if (x) {
        SymExpr total(1);
        for (const SymExpr& d : *x) total = total * d;
        SymExpr denom(1);
        for (std::size_t i = 0; i < out.size(); ++i)
          if (static_cast<int>(i) != neg_one) denom = denom * out[i];
        if (auto q = TryExactDivide(total, denom)) {
          resolved = *q;
          ok = true;
        }
      }
      out[static_cast<std::size_t>(neg_one)] = ok ? resolved : Fresh();
    }
    return out;
  }

  std::optional<SymShape> InferSqueeze(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    if (!x) return std::nullopt;
    const int64_t rank = static_cast<int64_t>(x->size());
    auto axes = IntListOperand(node, "axes", 1);
    std::set<int64_t> drop;
    if (axes) {
      for (int64_t a : *axes) drop.insert(NormalizeAxis(a, rank));
    } else {
      // No axes: drop every dim provably equal to 1.
      for (int64_t i = 0; i < rank; ++i) {
        auto c = ConcreteInt((*x)[static_cast<std::size_t>(i)]);
        if (c && *c == 1) drop.insert(i);
      }
    }
    SymShape out;
    for (int64_t i = 0; i < rank; ++i)
      if (!drop.count(i)) out.push_back((*x)[static_cast<std::size_t>(i)]);
    return out;
  }

  std::optional<SymShape> InferUnsqueeze(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    auto axes = IntListOperand(node, "axes", 1);
    if (!x || !axes) return std::nullopt;
    const int64_t out_rank = static_cast<int64_t>(x->size() + axes->size());
    std::set<int64_t> ins;
    for (int64_t a : *axes) ins.insert(NormalizeAxis(a, out_rank));
    SymShape out;
    std::size_t src = 0;
    for (int64_t i = 0; i < out_rank; ++i) {
      if (ins.count(i))
        out.push_back(SymExpr(1));
      else if (src < x->size())
        out.push_back((*x)[src++]);
      else
        return std::nullopt;
    }
    return out;
  }

  std::optional<SymShape> InferExpand(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    auto target = IntListOperand(node, "shape", 1);
    if (!x || !target) return std::nullopt;
    SymShape tshape;
    for (int64_t t : *target) tshape.push_back(SymExpr(t));
    return BroadcastShapes({x, &tshape});
  }

  std::optional<SymShape> InferGather(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymShape* data = Shape(node.input[0]);
    const SymShape* idx = Shape(node.input[1]);
    if (!data || !idx || data->empty()) return std::nullopt;
    const int64_t rank = static_cast<int64_t>(data->size());
    int64_t axis = NormalizeAxis(IntAttr(node, "axis", 0).value(), rank);
    if (axis < 0 || axis >= rank) return std::nullopt;
    // out = data[:axis] + indices.shape + data[axis+1:]
    SymShape out;
    for (int64_t i = 0; i < axis; ++i)
      out.push_back((*data)[static_cast<std::size_t>(i)]);
    for (const SymExpr& d : *idx) out.push_back(d);
    for (int64_t i = axis + 1; i < rank; ++i)
      out.push_back((*data)[static_cast<std::size_t>(i)]);
    return out;
  }

  std::optional<SymShape> InferSlice(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    if (!x) return std::nullopt;
    const int64_t rank = static_cast<int64_t>(x->size());
    auto starts = IntListOperand(node, "starts", 1);
    auto ends = IntListOperand(node, "ends", 2);
    auto axes = IntListOperand(node, "axes", 3);
    auto steps = IntListOperand(node, "steps", 4);
    if (!starts || !ends || starts->size() != ends->size()) return std::nullopt;
    SymShape out = *x;
    for (std::size_t j = 0; j < starts->size(); ++j) {
      int64_t axis =
          axes ? NormalizeAxis((*axes)[j], rank) : static_cast<int64_t>(j);
      if (axis < 0 || axis >= rank) return std::nullopt;
      const int64_t step = (steps && j < steps->size()) ? (*steps)[j] : 1;
      auto dim = ConcreteInt((*x)[static_cast<std::size_t>(axis)]);
      if (!dim || step == 0) {
        out[static_cast<std::size_t>(axis)] = Fresh();
        continue;
      }
      const int64_t len = *dim;
      int64_t s = (*starts)[j], e = (*ends)[j];
      if (s < 0) s += len;
      if (e < 0) e += len;
      int64_t count;
      if (step > 0) {
        s = std::clamp<int64_t>(s, 0, len);
        e = std::clamp<int64_t>(e, 0, len);
        count = e > s ? (e - s + step - 1) / step : 0;
      } else {
        s = std::clamp<int64_t>(s, 0, len - 1);
        e = std::clamp<int64_t>(e, -1, len - 1);
        count = s > e ? (s - e - step - 1) / (-step) : 0;
      }
      out[static_cast<std::size_t>(axis)] = SymExpr(count);
    }
    return out;
  }

  std::optional<SymShape> InferReduce(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymShape* x = Shape(node.input[0]);
    if (!x) return std::nullopt;
    const int64_t rank = static_cast<int64_t>(x->size());
    const bool keepdims = IntAttr(node, "keepdims", 1).value() != 0;
    auto axes = IntListOperand(node, "axes", 1);
    const bool noop_empty =
        IntAttr(node, "noop_with_empty_axes", 0).value() != 0;

    std::set<int64_t> reduce;
    if (axes && !axes->empty()) {
      for (int64_t a : *axes) reduce.insert(NormalizeAxis(a, rank));
    } else if (axes && axes->empty() && noop_empty) {
      return *x;  // explicit empty axes + noop flag: identity
    } else {
      for (int64_t i = 0; i < rank; ++i) reduce.insert(i);  // reduce all
    }
    SymShape out;
    for (int64_t i = 0; i < rank; ++i) {
      if (reduce.count(i)) {
        if (keepdims) out.push_back(SymExpr(1));
      } else {
        out.push_back((*x)[static_cast<std::size_t>(i)]);
      }
    }
    return out;
  }

  const ShapeGraph& graph_;
  std::map<std::string, SymShape> shapes_;
  int64_t unk_ = 0;
};

}  // namespace

std::map<std::string, SymShape> InferSymbolicShapes(const ShapeGraph& graph) {
  return ShapeInferer(graph).Run();
}

}  // namespace onnxsim
