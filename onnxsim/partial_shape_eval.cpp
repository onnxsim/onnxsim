#include "partial_shape_eval.h"

#include <google/protobuf/repeated_ptr_field.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "constant_folding.h"
#include "onnx/common/graph_shape_inference.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"
#include "sym_shape_infer.h"
#include "sym_value_eval.h"

// Mutates the model in place; ``onnx::shape_inference::InferShapes`` already
// works in place, so no extra ModelProto copy is made (the previous ``const&``
// signature forced a defensive ``CopyFrom`` because the input could not be
// mutated).
void _InferShapes(onnx::ModelProto& model) {
  onnx::shape_inference::InferShapes(model);
}

// Build a lookup from tensor name to its type, gathering shapes from every
// place a shape can be declared: value_info (populated by shape inference),
// graph inputs and graph outputs. Pointers reference `model`, so the map must
// not outlive it and `model` must not be mutated while the map is in use.
std::unordered_map<std::string, const onnx::TypeProto*> BuildTypeMap(
    const onnx::ModelProto& model) {
  std::unordered_map<std::string, const onnx::TypeProto*> type_map;
  auto add = [&type_map](const onnx::ValueInfoProto& vi) {
    if (vi.has_type()) {
      type_map[vi.name()] = &vi.type();
    }
  };
  for (const auto& vi : model.graph().value_info()) add(vi);
  for (const auto& vi : model.graph().input()) add(vi);
  for (const auto& vi : model.graph().output()) add(vi);
  return type_map;
}

// Fetch the element type and a fully static shape of `name` from `type_map`.
// Returns false unless the tensor has a known integer (INT64/INT32) element
// type and a shape whose every dimension is a fixed value. A rank-0 (scalar)
// tensor yields an empty `dims` (element count 1).
bool GetStaticIntTensorInfo(
    const std::unordered_map<std::string, const onnx::TypeProto*>& type_map,
    const std::string& name, onnx::TensorProto::DataType& elem_type,
    std::vector<int64_t>& dims) {
  auto iter = type_map.find(name);
  if (iter == type_map.end() || !iter->second->has_tensor_type()) {
    return false;
  }
  const auto& tensor_type = iter->second->tensor_type();
  elem_type = static_cast<onnx::TensorProto::DataType>(tensor_type.elem_type());
  if (elem_type != onnx::TensorProto::INT64 &&
      elem_type != onnx::TensorProto::INT32) {
    return false;
  }
  if (!tensor_type.has_shape()) {
    // Rank is unknown.
    return false;
  }
  dims.clear();
  for (const auto& dim : tensor_type.shape().dim()) {
    if (!dim.has_dim_value()) {
      return false;
    }
    dims.push_back(dim.dim_value());
  }
  return true;
}

// --- Native symbolic shape evaluation (issue #532, milestones M1/M2/M3) ------
//
// The ONNX data-propagation path below stalls at any arithmetic over a dynamic
// dim symbol: it carries a value as a TensorShapeProto whose entries are a
// concrete int or an *opaque* dim_param string, so a Reshape target like
// `[batch, 1024, 128]`, or a `Div`/`Where`/`Equal` over the symbol, cannot be
// evaluated. The dependency-free evaluator in sym_value_eval / sym_shape_infer
// keeps each dynamic dim as a `SymExpr` and computes the shape algebra. These
// helpers adapt an `onnx::ModelProto` into the evaluator's plain structs and
// run M2 (symbolic activation shapes) then M1 (symbolic value evaluation) over
// it.

// Read one little-endian `T` out of `p`. ONNX defines TensorProto::raw_data as
// little-endian on every host, so this is a plain byte-wise decode rather than
// a memcpy into a host integer (which would be correct only on a little-endian
// machine -- see docs/big-endian.md).
template <typename T>
T ReadLittleEndian(const char* p) {
  std::make_unsigned_t<T> v = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    v |= static_cast<std::make_unsigned_t<T>>(static_cast<unsigned char>(p[i]))
         << (8 * i);
  }
  return static_cast<T>(v);
}

// Convert an integer TensorProto (rank 0 or 1, INT64/INT32, inline data) to a
// SymTensor of concrete values. Returns nullopt for other dtypes/ranks or data
// kept in an external file.
std::optional<onnxsim::SymTensor> IntTensorToSymTensor(
    const onnx::TensorProto& tp) {
  if (tp.data_location() == onnx::TensorProto::EXTERNAL) return std::nullopt;
  const auto dt = tp.data_type();
  if (dt != onnx::TensorProto::INT64 && dt != onnx::TensorProto::INT32)
    return std::nullopt;
  if (tp.dims_size() > 1) return std::nullopt;  // rank 0 (scalar) or 1 only
  const bool scalar = tp.dims_size() == 0;
  std::vector<int64_t> vals;
  if (tp.has_raw_data()) {
    const std::string& raw = tp.raw_data();
    if (dt == onnx::TensorProto::INT64) {
      const size_t n = raw.size() / sizeof(int64_t);
      vals.resize(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = ReadLittleEndian<int64_t>(raw.data() + i * sizeof(int64_t));
    } else {
      const size_t n = raw.size() / sizeof(int32_t);
      vals.resize(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = ReadLittleEndian<int32_t>(raw.data() + i * sizeof(int32_t));
    }
  } else if (dt == onnx::TensorProto::INT64) {
    vals.assign(tp.int64_data().begin(), tp.int64_data().end());
  } else {
    vals.assign(tp.int32_data().begin(), tp.int32_data().end());
  }
  const int64_t expect = scalar ? 1 : tp.dims(0);
  if (static_cast<int64_t>(vals.size()) != expect) return std::nullopt;
  onnxsim::SymTensor t;
  t.scalar = scalar;
  for (int64_t v : vals) t.data.emplace_back(v);
  return t;
}

// A TypeProto's shape as a SymShape: dim_value -> SymExpr(v), a non-empty
// dim_param -> its Symbol, an otherwise-unknown dim -> a fresh distinct symbol
// (so the rank is preserved). Returns nullopt when the rank itself is unknown.
std::optional<onnxsim::SymShape> TypeProtoToSymShape(
    const onnx::TypeProto& type, int64_t& fresh) {
  if (!type.has_tensor_type() || !type.tensor_type().has_shape())
    return std::nullopt;
  onnxsim::SymShape shape;
  for (const auto& dim : type.tensor_type().shape().dim()) {
    if (dim.has_dim_value())
      shape.push_back(onnxsim::SymExpr(dim.dim_value()));
    else if (!dim.dim_param().empty())
      shape.push_back(onnxsim::SymExpr::Symbol(dim.dim_param()));
    else
      shape.push_back(
          onnxsim::SymExpr::Symbol("seedunk_" + std::to_string(fresh++)));
  }
  return shape;
}

// One node in the evaluator's plain form. A node from a non-default domain gets
// an empty op_type so no handler matches it (its outputs stay unevaluated).
onnxsim::SymNode ToSymNode(const onnx::NodeProto& node) {
  onnxsim::SymNode n;
  const std::string& domain = node.domain();
  n.op_type = (domain.empty() || domain == "ai.onnx") ? node.op_type() : "";
  n.input.assign(node.input().begin(), node.input().end());
  n.output.assign(node.output().begin(), node.output().end());
  for (const auto& attr : node.attribute()) {
    onnxsim::SymAttr a;
    a.name = attr.name();
    switch (attr.type()) {
      case onnx::AttributeProto::INT:
        a.i = attr.i();
        break;
      case onnx::AttributeProto::INTS:
        a.ints.assign(attr.ints().begin(), attr.ints().end());
        break;
      case onnx::AttributeProto::TENSOR:
        if (auto t = IntTensorToSymTensor(attr.t())) a.t = std::move(*t);
        break;
      default:
        break;
    }
    n.attribute.push_back(std::move(a));
  }
  return n;
}

// A `Constant` node's own embedded value, read directly from its `value` /
// `value_ints` / `value_int` attribute -- mirrors sym_value_eval.cpp's own
// EvalConstant. Needed because a genuine (non-transient) Constant node is
// never folded into a graph initializer any more (constant_folding.cpp
// deliberately leaves it as a producer node -- see kTransientConstantAttr's
// own comment), yet M2's shape rules (Squeeze axes, Reshape/Expand shape,
// Slice starts/ends/axes/steps, Split sizes, ...) only ever consult
// ShapeGraph::initializer for a data-input's concrete values. Without this, an
// int-list operand sourced from an unfolded Constant node is invisible to M2:
// the rule either fails outright, or -- worse, for Squeeze/Unsqueeze's
// "no axes" fallback -- silently substitutes the wrong heuristic (dropping
// every dim that happens to be 1, including ones the real, unseen axes list
// would not have touched). M1 (EvaluateSymbolicValues) does not need this: it
// evaluates every node in topological order, `Constant` included.
std::optional<onnxsim::SymTensor> ConstantNodeValue(
    const onnxsim::SymNode& node) {
  if (node.op_type != "Constant") return std::nullopt;
  if (const onnxsim::SymAttr* a = node.attr("value")) {
    if (a->t) return *a->t;
  }
  if (const onnxsim::SymAttr* a = node.attr("value_ints")) {
    std::vector<onnxsim::SymExpr> v;
    v.reserve(a->ints.size());
    for (int64_t x : a->ints) v.emplace_back(x);
    return onnxsim::SymTensor::Vector(std::move(v));
  }
  if (const onnxsim::SymAttr* a = node.attr("value_int")) {
    if (a->i) return onnxsim::SymTensor::Scalar(onnxsim::SymExpr(*a->i));
  }
  return std::nullopt;
}

// Adds every `Constant` node's own value into `initializers`, so M2's
// ShapeGraph::initializer-only lookups (see ConstantNodeValue's own comment)
// see it exactly as if it were a real graph initializer. `nodes` is the
// already-built SymNode list, so this needs no re-parsing of attributes.
void AddConstantNodeValues(
    const std::vector<onnxsim::SymNode>& nodes,
    std::map<std::string, onnxsim::SymTensor>& initializers) {
  for (const auto& node : nodes) {
    if (node.output.size() != 1 || node.output[0].empty()) continue;
    if (initializers.count(node.output[0])) continue;
    if (auto t = ConstantNodeValue(node)) {
      initializers[node.output[0]] = std::move(*t);
    }
  }
}

// Run M2 (symbolic activation-shape inference) then M1 (symbolic value
// evaluation) over `model`, returning every shape-data tensor the evaluator
// could resolve as a SymTensor (its entries possibly still symbolic).
std::map<std::string, onnxsim::SymTensor> EvaluateModelSymbolicValues(
    const onnx::ModelProto& model) {
  int64_t fresh = 0;
  std::vector<onnxsim::SymNode> nodes;
  nodes.reserve(model.graph().node_size());
  for (const auto& node : model.graph().node())
    nodes.push_back(ToSymNode(node));

  std::map<std::string, onnxsim::SymTensor> initializers;
  std::map<std::string, onnxsim::SymShape> shapes_seed;
  for (const auto& init : model.graph().initializer()) {
    if (auto t = IntTensorToSymTensor(init)) initializers[init.name()] = *t;
    onnxsim::SymShape s;  // an initializer's own shape is fully static
    for (int64_t d : init.dims()) s.emplace_back(d);
    shapes_seed[init.name()] = std::move(s);
  }
  AddConstantNodeValues(nodes, initializers);
  auto seed = [&](const onnx::ValueInfoProto& vi) {
    if (shapes_seed.count(vi.name()))
      return;  // keep the concrete initializer shape
    if (auto s = TypeProtoToSymShape(vi.type(), fresh))
      shapes_seed[vi.name()] = std::move(*s);
  };
  for (const auto& vi : model.graph().input()) seed(vi);
  for (const auto& vi : model.graph().value_info()) seed(vi);
  for (const auto& vi : model.graph().output()) seed(vi);

  onnxsim::ShapeGraph sg;
  sg.node = nodes;
  sg.value_info = shapes_seed;
  sg.initializer = initializers;

  onnxsim::SymGraph vg;
  vg.node = std::move(nodes);
  vg.initializer = std::move(initializers);
  vg.shape = onnxsim::InferSymbolicShapes(sg);
  return onnxsim::EvaluateSymbolicValues(vg);
}

// --- Graph-native counterparts of the two passes above ----------------------
//
// Both InferSymbolicShapes/EvaluateSymbolicValues already operate on the
// plain, representation-agnostic SymNode/SymTensor/ShapeGraph/SymGraph
// structs, so only the *builders* need a Graph-native counterpart -- the
// evaluator logic itself is untouched.

// Graph-native counterpart of IntTensorToSymTensor: identical rank-0/1,
// INT32/INT64-only rules, reading directly from ir.h's Tensor instead of
// TensorProto.
std::optional<onnxsim::SymTensor> IntTensorToSymTensor(const onnx::Tensor& t) {
  if (t.data_location() == onnx::TensorProto::EXTERNAL) return std::nullopt;
  const auto dt = t.elem_type();
  if (dt != onnx::TensorProto::INT64 && dt != onnx::TensorProto::INT32)
    return std::nullopt;
  if (t.sizes().size() > 1) return std::nullopt;
  const bool scalar = t.sizes().empty();
  std::vector<int64_t> vals;
  if (t.is_raw_data()) {
    const std::string& raw = t.raw();
    if (dt == onnx::TensorProto::INT64) {
      const size_t n = raw.size() / sizeof(int64_t);
      vals.resize(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = ReadLittleEndian<int64_t>(raw.data() + i * sizeof(int64_t));
    } else {
      const size_t n = raw.size() / sizeof(int32_t);
      vals.resize(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = ReadLittleEndian<int32_t>(raw.data() + i * sizeof(int32_t));
    }
  } else if (dt == onnx::TensorProto::INT64) {
    vals.assign(t.int64s().begin(), t.int64s().end());
  } else {
    vals.assign(t.int32s().begin(), t.int32s().end());
  }
  const int64_t expect = scalar ? 1 : t.sizes()[0];
  if (static_cast<int64_t>(vals.size()) != expect) return std::nullopt;
  onnxsim::SymTensor r;
  r.scalar = scalar;
  for (int64_t v : vals) r.data.emplace_back(v);
  return r;
}

// Graph-native counterpart of TypeProtoToSymShape, reading a Value's own
// elemType()/sizes() -- no ModelProto value_info map needed, since a Value
// already carries its own current shape/type in the IR.
std::optional<onnxsim::SymShape> ValueToSymShape(onnx::Value* v,
                                                 int64_t& fresh) {
  if (!v->has_sizes()) return std::nullopt;
  onnxsim::SymShape shape;
  for (const auto& d : v->sizes()) {
    if (d.is_int) {
      shape.push_back(onnxsim::SymExpr(d.dim));
    } else if (!d.is_unknown) {
      shape.push_back(onnxsim::SymExpr::Symbol(d.param));
    } else {
      shape.push_back(
          onnxsim::SymExpr::Symbol("seedunk_" + std::to_string(fresh++)));
    }
  }
  return shape;
}

// Graph-native counterpart of ToSymNode.
onnxsim::SymNode ToSymNode(onnx::Node* node) {
  onnxsim::SymNode n;
  const std::string domain =
      node->has_domain() ? node->domain() : std::string();
  n.op_type =
      (domain.empty() || domain == "ai.onnx") ? node->kind().toString() : "";
  for (onnx::Value* v : node->inputs()) {
    n.input.push_back(v->node()->kind() == onnx::kUndefined ? ""
                                                            : v->uniqueName());
  }
  for (onnx::Value* v : node->outputs()) {
    n.output.push_back(v->uniqueName());
  }
  for (onnx::Symbol attr_name : node->attributeNames()) {
    onnxsim::SymAttr a;
    a.name = attr_name.toString();
    switch (node->kindOf(attr_name)) {
      case onnx::AttributeKind::i:
        a.i = node->i(attr_name);
        break;
      case onnx::AttributeKind::is:
        a.ints = node->is(attr_name);
        break;
      case onnx::AttributeKind::t:
        if (auto t = IntTensorToSymTensor(node->t(attr_name)))
          a.t = std::move(*t);
        break;
      default:
        break;
    }
    n.attribute.push_back(std::move(a));
  }
  return n;
}

// Graph-native counterpart of EvaluateModelSymbolicValues.
std::map<std::string, onnxsim::SymTensor> EvaluateGraphSymbolicValues(
    onnx::Graph& g, const std::vector<onnx::Node*>& node_ptrs) {
  int64_t fresh = 0;
  std::vector<onnxsim::SymNode> nodes;
  nodes.reserve(node_ptrs.size());
  for (onnx::Node* node : node_ptrs) {
    if (node->kind() == onnx::kUndefined || node->kind() == onnx::kCaptured)
      continue;
    nodes.push_back(ToSymNode(node));
  }

  std::map<std::string, onnxsim::SymTensor> initializers;
  std::map<std::string, onnxsim::SymShape> shapes_seed;
  const auto& inits = g.initializers();
  const auto& init_names = g.initializer_names();
  for (size_t i = 0; i < inits.size(); ++i) {
    if (auto t = IntTensorToSymTensor(*inits[i]))
      initializers[init_names[i]] = *t;
    onnxsim::SymShape s;
    for (int64_t d : inits[i]->sizes()) s.emplace_back(d);
    shapes_seed[init_names[i]] = std::move(s);
  }
  AddConstantNodeValues(nodes, initializers);
  auto seed = [&](onnx::Value* v) {
    if (shapes_seed.count(v->uniqueName()))
      return;  // keep the concrete initializer shape
    if (auto s = ValueToSymShape(v, fresh))
      shapes_seed[v->uniqueName()] = std::move(*s);
  };
  for (onnx::Value* v : g.inputs()) seed(v);
  for (onnx::Node* node : node_ptrs) {
    for (onnx::Value* v : node->outputs()) seed(v);
  }
  for (onnx::Value* v : g.outputs()) seed(v);

  onnxsim::ShapeGraph sg;
  sg.node = nodes;
  sg.value_info = shapes_seed;
  sg.initializer = initializers;

  onnxsim::SymGraph vg;
  vg.node = std::move(nodes);
  vg.initializer = std::move(initializers);
  vg.shape = onnxsim::InferSymbolicShapes(sg);
  return onnxsim::EvaluateSymbolicValues(vg);
}

// ModelProto-flavoured CollectRankUnsafeDataValues (declared further down,
// next to the Graph-native folder) -- see that function's comment for what
// "rank unsafe" means and why the mark has to be carried forward. Rank comes
// from the model's type map rather than a Value's own sizes(), and a name
// missing from that map counts as "rank not proven", i.e. unsafe.
std::unordered_set<std::string> CollectRankUnsafeDataValuesOnModel(
    const onnx::GraphProto& graph,
    const std::unordered_map<std::string, const onnx::TypeProto*>& type_map,
    const onnx::shape_inference::DataValueMap& data_map) {
  std::unordered_set<std::string> unsafe;
  auto rank_is_modelled = [&type_map](const std::string& name) {
    auto iter = type_map.find(name);
    if (iter == type_map.end() || !iter->second->has_tensor_type()) {
      return false;
    }
    const auto& tensor_type = iter->second->tensor_type();
    return tensor_type.has_shape() && tensor_type.shape().dim_size() <= 1;
  };
  // A GraphProto's nodes are required to be in topological order, so one pass
  // over them propagates the mark completely.
  for (const auto& node : graph.node()) {
    bool tainted = false;
    for (const std::string& in : node.input()) {
      if (data_map.count(in) == 0) {
        continue;
      }
      if (unsafe.count(in) != 0 || !rank_is_modelled(in)) {
        tainted = true;
        break;
      }
    }
    for (const std::string& out : node.output()) {
      if (data_map.count(out) == 0) {
        continue;
      }
      if (tainted || !rank_is_modelled(out)) {
        unsafe.insert(out);
      }
    }
  }
  return unsafe;
}

// Partial shape evaluation (issue #139) via ONNX data propagation.
//
// The plain constant folder only folds a node when *all* of its inputs are
// constant, so shape-computing ops like `Shape` are never folded: their input
// is an activation. Yet those ops depend solely on shapes, which shape
// inference knows -- fully or partially -- even when some dimensions stay
// dynamic.
//
// ONNX shape inference can *propagate* those partially known values: with data
// propagation enabled it fills a DataValueMap mapping each tensor to a
// TensorShapeProto whose entries are either a concrete dim_value or a symbolic
// dim_param. Ops across the shape family (Shape, Gather, Slice, Concat,
// Squeeze/Unsqueeze, Cast, Add/Sub/Mul, ...) participate, so a chain like
//   Shape([batch, C, H, W]) -> Gather([1, 2, 3])  ==>  [C, H, W]
// is propagated end to end and comes out fully concrete even though the batch
// dimension stays dynamic (the mask-rcnn pattern from issue #139).
//
// This pass rewrites every node whose lone output has a fully concrete
// propagated value into a `Constant` node. Downstream ops then fold through the
// ordinary constant folder, and now-dead nodes are removed by the optimizer.
// A graph's ``value_info``/``output`` annotations, snapshotted recursively:
// one entry for the graph itself, followed by one entry per nested subgraph
// (the branches of an ``If``, the body of a ``Loop``/``Scan``, ...) in the
// same pre-order that ``RestoreGraphAnnotations`` below walks them back in.
struct GraphAnnotationSnapshot {
  google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> value_info;
  google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> output;
};

void SnapshotGraphAnnotations(const onnx::GraphProto& graph,
                              std::vector<GraphAnnotationSnapshot>& out) {
  out.push_back({graph.value_info(), graph.output()});
  for (const auto& node : graph.node()) {
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        SnapshotGraphAnnotations(attr.g(), out);
      }
      for (const auto& subgraph : attr.graphs()) {
        SnapshotGraphAnnotations(subgraph, out);
      }
    }
  }
}

// ``restore_self`` gates whether `graph` itself (as opposed to the nested
// subgraphs found in its nodes) gets its snapshot re-applied: callers that
// intentionally keep a graph's own data-propagated annotations (because
// something in that graph did get folded using them) still want every nested
// ``If``/``Loop`` subgraph restored unconditionally -- this pass never folds
// a subgraph-local node, so a subgraph's annotations are never allowed to
// stick regardless of what happened at the level above it.
void RestoreGraphAnnotations(onnx::GraphProto* graph,
                             const std::vector<GraphAnnotationSnapshot>& snaps,
                             size_t& idx, bool restore_self = true) {
  if (restore_self) {
    *graph->mutable_value_info() = snaps[idx].value_info;
    *graph->mutable_output() = snaps[idx].output;
  }
  idx++;
  for (auto& node : *graph->mutable_node()) {
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        RestoreGraphAnnotations(attr.mutable_g(), snaps, idx);
      }
      for (auto& subgraph : *attr.mutable_graphs()) {
        RestoreGraphAnnotations(&subgraph, snaps, idx);
      }
    }
  }
}

void _EvalPartialShape(onnx::ModelProto& model) {
  // This pass runs shape inference with *data propagation* (lenient options)
  // purely to discover foldable shape values; it must not otherwise change the
  // model. InferShapes mutates value_info and output types in place --
  // including inside nested subgraphs (the branches of an ``If``, a ``Loop``
  // body, ...), since data-propagating shape inference recurses into them
  // too -- so snapshot those annotations *recursively* and restore them on
  // the paths that fold nothing, leaving the model byte-for-byte unchanged
  // (the old code returned the untouched input there). Restoring only the
  // top-level graph's annotations left a subgraph's lenient, data-propagated
  // value_info in place even when nothing here was foldable; two sibling
  // branches of the same ``If`` can independently reuse the same local
  // output name (e.g. both a decoder's no-past and with-past branches
  // producing their own "ConstantOfShape_2"), so leaking one branch's
  // data-propagation shape onto the model let onnxruntime's later merge of
  // the ``If`` node's branch output types find the two incompatible and
  // reject the model outright. The snapshot is metadata only -- no tensor
  // weights -- so it is cheap, unlike the full-model ``CopyFrom`` it
  // replaces. Restoring also keeps this pass's data-propagation value_info
  // out of the model, which matters: it differs from the regular
  // shape-inference pass's value_info, and leaving it behind could make the
  // outer fixed point oscillate.
  std::vector<GraphAnnotationSnapshot> saved_annotations;
  SnapshotGraphAnnotations(model.graph(), saved_annotations);
  auto restore = [&]() {
    size_t idx = 0;
    RestoreGraphAnnotations(model.mutable_graph(), saved_annotations, idx);
  };
  // Unlike `restore`, this leaves the top-level graph's own (possibly folded)
  // annotations alone and only reverts every nested subgraph -- for the path
  // below where something *was* foldable at the top level, so the top-level
  // graph's data-propagated value_info is allowed to stick (nothing here
  // ever folds a subgraph-local node, so a subgraph's annotations must
  // revert either way).
  auto restore_subgraphs_only = [&]() {
    size_t idx = 0;
    RestoreGraphAnnotations(model.mutable_graph(), saved_annotations, idx,
                            /*restore_self=*/false);
  };

  onnx::shape_inference::DataValueMap data_map;
  try {
    const onnx::ShapeInferenceOptions options(/*check_type=*/false,
                                              /*error_mode=*/0,
                                              /*enable_data_propagation=*/true);
    onnx::shape_inference::InferShapes(
        model, onnx::OpSchemaRegistry::Instance(), options, &data_map);
  } catch (const std::exception&) {
    // If shape inference fails we simply have no propagated values to exploit.
    restore();
    return;
  }

  // An empty data_map is not a dead end anymore: the native symbolic evaluator
  // (issue #532) runs further below and can resolve chains ONNX data
  // propagation could not, so fall through instead of returning. The loops over
  // data_map simply add nothing, and the final `folded_values && reshape_fixes`
  // empty check restores the model if the symbolic pass also finds nothing.

  const auto type_map = BuildTypeMap(model);

  // Values whose propagated data must not be trusted -- see
  // CollectRankUnsafeDataValues' own comment.
  const std::unordered_set<std::string> rank_unsafe =
      CollectRankUnsafeDataValuesOnModel(model.graph(), type_map, data_map);

  // Maps the output of a foldable node to the constant tensor it produces. Each
  // such node is rewritten into a `Constant` node holding this value.
  std::unordered_map<std::string, onnx::TensorProto> folded_values;

  for (const auto& node : model.graph().node()) {
    // Shape-family ops are single-output; only replace a node when its lone
    // output is fully known, so dropping it can never orphan a second output.
    if (node.output_size() != 1) {
      continue;
    }
    const std::string& output = node.output(0);
    auto data_iter = data_map.find(output);
    if (data_iter == data_map.end()) {
      continue;
    }
    if (rank_unsafe.count(output) != 0) {
      continue;
    }

    // Every element must be statically known. Data propagation represents an
    // unknown element as a dimension with neither dim_value nor dim_param, so
    // requiring dim_value on every entry both proves the value is concrete and
    // filters out activations whose rank alone is known.
    const onnx::TensorShapeProto& value = data_iter->second;
    bool fully_known = true;
    std::vector<int64_t> values;
    for (const auto& dim : value.dim()) {
      if (!dim.has_dim_value()) {
        fully_known = false;
        break;
      }
      values.push_back(dim.dim_value());
    }
    if (!fully_known) {
      continue;
    }

    // Build the constant tensor with the output's real dtype and shape. The
    // propagated data is a flat sequence, so require a fully static shape whose
    // element count matches what was propagated.
    onnx::TensorProto::DataType elem_type;
    std::vector<int64_t> dims;
    if (!GetStaticIntTensorInfo(type_map, output, elem_type, dims)) {
      continue;
    }
    int64_t element_count = 1;
    for (int64_t d : dims) {
      element_count *= d;
    }
    if (element_count != static_cast<int64_t>(values.size())) {
      continue;
    }

    onnx::TensorProto tp;
    tp.set_data_type(elem_type);
    for (int64_t d : dims) {
      tp.add_dims(d);
    }
    if (elem_type == onnx::TensorProto::INT64) {
      for (int64_t v : values) {
        tp.add_int64_data(v);
      }
    } else {
      for (int64_t v : values) {
        tp.add_int32_data(static_cast<int32_t>(v));
      }
    }
    folded_values.emplace(output, std::move(tp));
  }

  // Data propagation for `Reshape` (single dynamic dim). The fully-known folder
  // above only rewrites a node whose propagated value is entirely concrete, so
  // a shape tensor that keeps one symbolic entry -- e.g. `[batch, 1024, 128]`
  // on a graph with a dynamic batch, or `[?, 768]` with a dynamic sequence
  // length -- is left alone, and with it the whole `Shape -> Gather -> Concat`
  // subgraph that computes it at runtime. Those single-dynamic-dim reshapes
  // dominate transformer and speech graphs.
  //
  // When a Reshape's shape input propagates to a value with exactly one unknown
  // entry and all other entries positive constants, materialize the shape as a
  // constant with the unknown slot set to -1. ONNX Reshape infers the -1 dim
  // from the total element count, so for every input the result is identical to
  // the runtime-computed shape, while the shape-producing subgraph becomes dead
  // and is removed by the optimizer. (Correctness is still gated by onnxsim's
  // own equivalence check.)
  struct ReshapeShapeFix {
    std::string shape_name;
    onnx::TensorProto shape_tensor;
  };
  std::unordered_map<std::string, ReshapeShapeFix> reshape_fixes;
  for (const auto& node : model.graph().node()) {
    if (node.op_type() != "Reshape" || node.input_size() < 2 ||
        node.output_size() != 1) {
      continue;
    }
    auto data_iter = data_map.find(node.input(1));
    if (data_iter == data_map.end()) {
      continue;
    }
    if (rank_unsafe.count(node.input(1)) != 0) {
      continue;
    }
    const onnx::TensorShapeProto& shape_value = data_iter->second;
    if (shape_value.dim_size() == 0) {
      continue;
    }
    int unknown = 0;
    bool usable = true;
    std::vector<int64_t> shape;
    shape.reserve(shape_value.dim_size());
    for (const auto& dim : shape_value.dim()) {
      if (dim.has_dim_value()) {
        // A non-positive entry is a literal 0 (copy-dim) or an already
        // materialized -1; leave those for the ordinary folder.
        if (dim.dim_value() <= 0) {
          usable = false;
          break;
        }
        shape.push_back(dim.dim_value());
      } else {
        ++unknown;
        shape.push_back(-1);
      }
    }
    // Exactly one unknown dim maps to Reshape's single -1 sentinel. Zero
    // unknowns is handled by the fully-known folder above; two or more cannot
    // be expressed with a single -1.
    if (!usable || unknown != 1) {
      continue;
    }
    onnx::TensorProto tp;
    tp.set_data_type(onnx::TensorProto::INT64);
    tp.add_dims(static_cast<int64_t>(shape.size()));
    for (int64_t v : shape) {
      tp.add_int64_data(v);
    }
    reshape_fixes.emplace(
        node.output(0),
        ReshapeShapeFix{node.output(0) + "_dp_shape", std::move(tp)});
  }

  // Native symbolic evaluation (issue #532). ONNX data propagation above stops
  // wherever the shape algebra crosses a dynamic-dim symbol; the SymExpr
  // evaluator resolves those chains. Merge whatever it finds that the ONNX path
  // did not already cover into the same two rewrite maps, so the shared rewrite
  // loop below handles both uniformly. Correctness stays gated by check_n.
  {
    const auto sym_values = EvaluateModelSymbolicValues(model);
    // Fully concrete symbolic values fold to a `Constant`, exactly like the
    // ONNX fully-known folder above (same dtype/shape/element-count checks).
    for (const auto& node : model.graph().node()) {
      if (node.output_size() != 1) continue;
      const std::string& output = node.output(0);
      if (folded_values.count(output) || reshape_fixes.count(output)) continue;
      auto sym_iter = sym_values.find(output);
      if (sym_iter == sym_values.end()) continue;
      std::vector<int64_t> flat;
      bool all_concrete = true;
      for (const auto& e : sym_iter->second.data) {
        if (e.is_symbolic()) {
          all_concrete = false;
          break;
        }
        flat.push_back(e.to_int());
      }
      if (!all_concrete) continue;
      onnx::TensorProto::DataType elem_type;
      std::vector<int64_t> dims;
      if (!GetStaticIntTensorInfo(type_map, output, elem_type, dims)) continue;
      int64_t element_count = 1;
      for (int64_t d : dims) element_count *= d;
      if (element_count != static_cast<int64_t>(flat.size())) continue;
      // SymTensor models a rank-0 scalar or a rank-1 vector and nothing else
      // (sym_value_eval.h), and its evaluators say so -- EvalSlice is "rank-1
      // data, axis 0 only", Transpose is evaluated as the identity. Same
      // reasoning as CollectRankUnsafeDataValues: a flat element sequence must
      // not be reinterpreted under a rank >= 2 output's dims (issue #1284).
      if (dims.size() > 1) continue;
      onnx::TensorProto tp;
      tp.set_data_type(elem_type);
      for (int64_t d : dims) tp.add_dims(d);
      if (elem_type == onnx::TensorProto::INT64) {
        for (int64_t v : flat) tp.add_int64_data(v);
      } else {
        for (int64_t v : flat) tp.add_int32_data(static_cast<int32_t>(v));
      }
      folded_values.emplace(output, std::move(tp));
    }
    // A Reshape whose target has exactly one symbolic entry (plus positive
    // constants) becomes `[-1, ...]` -- the same rewrite as the ONNX data-prop
    // Reshape path above, but reached through SymExpr arithmetic.
    for (const auto& node : model.graph().node()) {
      if (node.op_type() != "Reshape" || node.input_size() < 2 ||
          node.output_size() != 1) {
        continue;
      }
      if (reshape_fixes.count(node.output(0))) continue;
      auto sym_iter = sym_values.find(node.input(1));
      if (sym_iter == sym_values.end()) continue;
      const onnxsim::SymTensor& shape_value = sym_iter->second;
      if (shape_value.scalar || shape_value.data.empty()) continue;
      int unknown = 0;
      bool usable = true;
      std::vector<int64_t> shape;
      shape.reserve(shape_value.data.size());
      for (const auto& e : shape_value.data) {
        if (e.is_symbolic()) {
          ++unknown;
          shape.push_back(-1);
        } else {
          const int64_t v = e.to_int();
          if (v <=
              0) {  // a literal 0 (copy) or -1 is left for the ordinary folder
            usable = false;
            break;
          }
          shape.push_back(v);
        }
      }
      if (!usable || unknown != 1) continue;
      onnx::TensorProto tp;
      tp.set_data_type(onnx::TensorProto::INT64);
      tp.add_dims(static_cast<int64_t>(shape.size()));
      for (int64_t v : shape) tp.add_int64_data(v);
      reshape_fixes.emplace(
          node.output(0),
          ReshapeShapeFix{node.output(0) + "_sym_shape", std::move(tp)});
    }
  }

  if (folded_values.empty() && reshape_fixes.empty()) {
    restore();
    return;
  }
  restore_subgraphs_only();

  // Rewrite each foldable node into a `Constant` node in the same position,
  // keeping the graph topologically sorted. Emitting a `Constant` node (rather
  // than injecting an initializer) leaves the value in producer form, so the
  // ordinary constant folder and optimizer decide how to materialize it.
  google::protobuf::RepeatedPtrField<onnx::NodeProto> original_nodes;
  original_nodes.Swap(model.mutable_graph()->mutable_node());
  for (auto& node : original_nodes) {
    auto iter = node.output_size() == 1 ? folded_values.find(node.output(0))
                                        : folded_values.end();
    if (iter != folded_values.end()) {
      onnx::NodeProto* constant = model.mutable_graph()->add_node();
      constant->set_name(node.name());
      constant->set_op_type("Constant");
      constant->add_output(iter->first);
      onnx::AttributeProto* attr = constant->add_attribute();
      attr->set_name("value");
      attr->set_type(onnx::AttributeProto::TENSOR);
      *attr->mutable_t() = std::move(iter->second);
      // Marked transient (see kTransientConstantAttr's own comment): this
      // Constant node is this pass's own intermediate representation for a
      // value it already proved fully known, not a source of "not from
      // initializers" provenance -- the ordinary constant folder should
      // normalize it away like any other node, not treat it as an opaque
      // Constant to leave alone.
      onnx::AttributeProto* transient_attr = constant->add_attribute();
      transient_attr->set_name(kTransientConstantAttr);
      transient_attr->set_type(onnx::AttributeProto::INT);
      transient_attr->set_i(1);
      continue;
    }
    auto fix_iter = node.output_size() == 1 ? reshape_fixes.find(node.output(0))
                                            : reshape_fixes.end();
    if (fix_iter != reshape_fixes.end()) {
      // Emit the materialized shape as a Constant just before the Reshape
      // (preserving topological order), then repoint the Reshape's shape input
      // at it. The original shape-producing subgraph is now unused and is
      // cleaned up by the optimizer's dead-node elimination.
      onnx::NodeProto* shape_const = model.mutable_graph()->add_node();
      shape_const->set_op_type("Constant");
      shape_const->add_output(fix_iter->second.shape_name);
      onnx::AttributeProto* attr = shape_const->add_attribute();
      attr->set_name("value");
      attr->set_type(onnx::AttributeProto::TENSOR);
      *attr->mutable_t() = std::move(fix_iter->second.shape_tensor);
      // Transient -- see the other creation site's own comment above.
      onnx::AttributeProto* transient_attr = shape_const->add_attribute();
      transient_attr->set_name(kTransientConstantAttr);
      transient_attr->set_type(onnx::AttributeProto::INT);
      transient_attr->set_i(1);

      onnx::NodeProto* reshape = model.mutable_graph()->add_node();
      *reshape = std::move(node);
      reshape->set_input(1, fix_iter->second.shape_name);
      continue;
    }
    *model.mutable_graph()->add_node() = std::move(node);
  }
}

// Values whose ONNX-propagated data is meaningless because the propagation ran
// on a tensor of a rank it does not model (onnxsim issue #1284).
//
// Data propagation carries a value as a `TensorShapeProto` -- a flat sequence
// with no rank of its own -- because it exists to resolve *shape* scaffolding,
// which is rank <= 1 by nature. Every `PartialDataPropagationFunction` is
// written for that form: `Slice`'s says outright "Only supports axis = 0 since
// the data comes from Shape", and `DataPropagationContextImpl::getInputData`
// refuses to seed data from an initializer of rank >= 2 for the same reason.
//
// Nothing enforces that on a value produced *inside* the propagation, though.
// Let a rank >= 2 tensor in -- e.g. the `[3, 2]` pads matrix in the
// `Concat -> Reshape([-1, 2]) -> Slice(steps=-1) -> Transpose -> Reshape([-1])`
// construction PyTorch emits for `F.pad` -- and the propagators silently read
// its flat entries as a rank-1 element order: that `Slice` must reverse the 3
// rows, but the propagator reverses all 6 flat entries. The value is still
// "fully known", just wrong, and it keeps propagating to everything downstream,
// so an element-count check on the folded node alone does not catch it.
//
// So mark a value's propagated data unusable when the value's own rank is not
// provably <= 1, and carry that mark forward through any node that consumed a
// marked value's data. `Shape(x)` and friends are unaffected: `x` carries no
// propagated data of its own, so a rank-4 activation feeding a `Shape` marks
// nothing -- which is the whole point of this pass (issue #139).
std::unordered_set<std::string> CollectRankUnsafeDataValues(
    const std::vector<onnx::Node*>& node_ptrs,
    const onnx::shape_inference::DataValueMap& data_map) {
  std::unordered_set<std::string> unsafe;
  // A rank onnx's own data propagation models: a rank-0 scalar or a rank-1
  // vector, and only when shape inference actually proved which.
  auto rank_is_modelled = [](const onnx::Value* v) {
    return v->has_sizes() && v->sizes().size() <= 1;
  };
  // node_ptrs is in topological order, so a producer is always visited before
  // its consumers and one pass suffices.
  for (const onnx::Node* node : node_ptrs) {
    bool tainted = false;
    for (const onnx::Value* in : node->inputs()) {
      const std::string& name = in->uniqueName();
      // Only an input that actually carries propagated data can taint this
      // node's output; an ordinary activation input cannot.
      if (data_map.count(name) == 0) {
        continue;
      }
      if (unsafe.count(name) != 0 || !rank_is_modelled(in)) {
        tainted = true;
        break;
      }
    }
    for (const onnx::Value* out : node->outputs()) {
      const std::string& name = out->uniqueName();
      if (data_map.count(name) == 0) {
        continue;
      }
      if (tainted || !rank_is_modelled(out)) {
        unsafe.insert(name);
      }
    }
  }
  return unsafe;
}

// Graph-native counterpart of _EvalPartialShape: same two rewrites (fold a
// fully-known shape-family output to a Constant; materialize a Reshape's
// shape input as a Constant with a single -1 slot when everything else is
// concrete), reached without any ModelProto <-> Graph round trip. Built on
// the extended InferShapesOnGraph (generated_shape_data out-param, see
// onnx/common/graph_shape_inference.h) instead of onnx's protobuf-based
// InferShapes. Returns whether anything changed, so the fully Graph-native
// outer pipeline can use it as a GraphFnChanged step.
bool _EvalPartialShapeOnGraph(
    onnx::Graph& g, const onnx::shape_inference::ModelLocalFunctionsMap&
                        model_local_functions) {
  // Mirrors _EvalPartialShape's own snapshot/restore of value_info/output:
  // this pass's own data-propagation inference call must not leave its
  // (lenient, check_type=false) shape/type conclusions on the graph --
  // only the two rewrites below should persist. Restore before the rewrite
  // loop, from a snapshot taken before the inference call.
  struct ValueTypeSnapshot {
    int32_t elem_type;
    bool has_sizes;
    std::vector<onnx::Dimension> sizes;
  };
  std::vector<onnx::Node*> node_ptrs(g.nodes().begin(), g.nodes().end());
  std::unordered_map<onnx::Value*, ValueTypeSnapshot> snapshot;
  auto save = [&](onnx::Value* v) {
    snapshot[v] = {
        v->elemType(), v->has_sizes(),
        v->has_sizes() ? v->sizes() : std::vector<onnx::Dimension>{}};
  };
  for (onnx::Value* v : g.inputs()) save(v);
  for (onnx::Node* node : node_ptrs) {
    for (onnx::Value* v : node->outputs()) save(v);
  }
  auto restore = [&]() {
    for (const auto& [v, s] : snapshot) {
      v->setElemType(s.elem_type);
      if (s.has_sizes) {
        v->setSizes(s.sizes);
      } else {
        v->wipeSizes();
      }
    }
  };

  onnx::shape_inference::DataValueMap data_map;
  {
    const onnx::ShapeInferenceOptions options(/*check_type=*/false,
                                              /*error_mode=*/0,
                                              /*enable_data_propagation=*/true);
    try {
      onnx::InferShapesOnGraph(g, options, &data_map, model_local_functions);
    } catch (const std::exception&) {
      restore();
      return false;
    }
  }

  // Values whose propagated data must not be trusted -- see
  // CollectRankUnsafeDataValues' own comment.
  const std::unordered_set<std::string> rank_unsafe =
      CollectRankUnsafeDataValues(node_ptrs, data_map);

  // Maps the output of a foldable node to the constant tensor it produces.
  std::unordered_map<std::string, onnx::Tensor> folded_values;
  for (onnx::Node* node : node_ptrs) {
    if (node->outputs().size() != 1) continue;
    onnx::Value* out = node->outputs()[0];
    auto data_iter = data_map.find(out->uniqueName());
    if (data_iter == data_map.end()) continue;
    if (rank_unsafe.count(out->uniqueName())) continue;

    const onnx::TensorShapeProto& value = data_iter->second;
    bool fully_known = true;
    std::vector<int64_t> values;
    for (const auto& dim : value.dim()) {
      if (!dim.has_dim_value()) {
        fully_known = false;
        break;
      }
      values.push_back(dim.dim_value());
    }
    if (!fully_known) continue;

    const auto elem_type =
        static_cast<onnx::TensorProto::DataType>(out->elemType());
    if (elem_type != onnx::TensorProto::INT64 &&
        elem_type != onnx::TensorProto::INT32) {
      continue;
    }
    if (!out->has_sizes()) continue;
    std::vector<int64_t> dims;
    bool dims_known = true;
    for (const auto& d : out->sizes()) {
      if (!d.is_int) {
        dims_known = false;
        break;
      }
      dims.push_back(d.dim);
    }
    if (!dims_known) continue;
    int64_t element_count = 1;
    for (int64_t d : dims) element_count *= d;
    if (element_count != static_cast<int64_t>(values.size())) continue;

    onnx::Tensor t;
    t.elem_type() = elem_type;
    t.sizes() = dims;
    if (elem_type == onnx::TensorProto::INT64) {
      for (int64_t v : values) t.int64s().push_back(v);
    } else {
      for (int64_t v : values) t.int32s().push_back(static_cast<int32_t>(v));
    }
    folded_values.emplace(out->uniqueName(), std::move(t));
  }

  // Data propagation for `Reshape` (single dynamic dim) -- see
  // _EvalPartialShape's own comment for why this matters.
  struct ReshapeShapeFix {
    onnx::Tensor shape_tensor;
  };
  std::unordered_map<std::string, ReshapeShapeFix> reshape_fixes;
  for (onnx::Node* node : node_ptrs) {
    if (node->kind() != onnx::kReshape || node->inputs().size() < 2 ||
        node->outputs().size() != 1) {
      continue;
    }
    if (folded_values.count(node->outputs()[0]->uniqueName())) continue;
    auto data_iter = data_map.find(node->inputs()[1]->uniqueName());
    if (data_iter == data_map.end()) continue;
    if (rank_unsafe.count(node->inputs()[1]->uniqueName())) continue;
    const onnx::TensorShapeProto& shape_value = data_iter->second;
    if (shape_value.dim_size() == 0) continue;
    int unknown = 0;
    bool usable = true;
    std::vector<int64_t> shape;
    shape.reserve(shape_value.dim_size());
    for (const auto& dim : shape_value.dim()) {
      if (dim.has_dim_value()) {
        if (dim.dim_value() <= 0) {
          usable = false;
          break;
        }
        shape.push_back(dim.dim_value());
      } else {
        ++unknown;
        shape.push_back(-1);
      }
    }
    if (!usable || unknown != 1) continue;
    onnx::Tensor t;
    t.elem_type() = onnx::TensorProto::INT64;
    t.sizes() = {static_cast<int64_t>(shape.size())};
    for (int64_t v : shape) t.int64s().push_back(v);
    reshape_fixes.emplace(node->outputs()[0]->uniqueName(),
                          ReshapeShapeFix{std::move(t)});
  }

  // Native symbolic evaluation (issue #532), merged into the same two
  // rewrite maps exactly as _EvalPartialShape does.
  {
    const auto sym_values = EvaluateGraphSymbolicValues(g, node_ptrs);
    for (onnx::Node* node : node_ptrs) {
      if (node->outputs().size() != 1) continue;
      onnx::Value* out = node->outputs()[0];
      const std::string& output = out->uniqueName();
      if (folded_values.count(output) || reshape_fixes.count(output)) continue;
      auto sym_iter = sym_values.find(output);
      if (sym_iter == sym_values.end()) continue;
      std::vector<int64_t> flat;
      bool all_concrete = true;
      for (const auto& e : sym_iter->second.data) {
        if (e.is_symbolic()) {
          all_concrete = false;
          break;
        }
        flat.push_back(e.to_int());
      }
      if (!all_concrete) continue;
      const auto elem_type =
          static_cast<onnx::TensorProto::DataType>(out->elemType());
      if (elem_type != onnx::TensorProto::INT64 &&
          elem_type != onnx::TensorProto::INT32) {
        continue;
      }
      if (!out->has_sizes()) continue;
      std::vector<int64_t> dims;
      bool dims_known = true;
      for (const auto& d : out->sizes()) {
        if (!d.is_int) {
          dims_known = false;
          break;
        }
        dims.push_back(d.dim);
      }
      if (!dims_known) continue;
      int64_t element_count = 1;
      for (int64_t d : dims) element_count *= d;
      if (element_count != static_cast<int64_t>(flat.size())) continue;
      // SymTensor models a rank-0 scalar or a rank-1 vector and nothing else
      // (sym_value_eval.h), and its evaluators say so -- EvalSlice is "rank-1
      // data, axis 0 only", Transpose is evaluated as the identity. Same
      // reasoning as CollectRankUnsafeDataValues: a flat element sequence must
      // not be reinterpreted under a rank >= 2 output's dims (issue #1284).
      if (dims.size() > 1) continue;
      onnx::Tensor t;
      t.elem_type() = elem_type;
      t.sizes() = dims;
      if (elem_type == onnx::TensorProto::INT64) {
        for (int64_t v : flat) t.int64s().push_back(v);
      } else {
        for (int64_t v : flat) t.int32s().push_back(static_cast<int32_t>(v));
      }
      folded_values.emplace(output, std::move(t));
    }
    for (onnx::Node* node : node_ptrs) {
      if (node->kind() != onnx::kReshape || node->inputs().size() < 2 ||
          node->outputs().size() != 1) {
        continue;
      }
      const std::string& reshape_out = node->outputs()[0]->uniqueName();
      if (reshape_fixes.count(reshape_out)) continue;
      auto sym_iter = sym_values.find(node->inputs()[1]->uniqueName());
      if (sym_iter == sym_values.end()) continue;
      const onnxsim::SymTensor& shape_value = sym_iter->second;
      if (shape_value.scalar || shape_value.data.empty()) continue;
      int unknown = 0;
      bool usable = true;
      std::vector<int64_t> shape;
      shape.reserve(shape_value.data.size());
      for (const auto& e : shape_value.data) {
        if (e.is_symbolic()) {
          ++unknown;
          shape.push_back(-1);
        } else {
          const int64_t v = e.to_int();
          if (v <= 0) {
            usable = false;
            break;
          }
          shape.push_back(v);
        }
      }
      if (!usable || unknown != 1) continue;
      onnx::Tensor t;
      t.elem_type() = onnx::TensorProto::INT64;
      t.sizes() = {static_cast<int64_t>(shape.size())};
      for (int64_t v : shape) t.int64s().push_back(v);
      reshape_fixes.emplace(reshape_out, ReshapeShapeFix{std::move(t)});
    }
  }

  // Restore the graph's shape/type state to what it was before this pass's
  // own data-propagation inference call -- see the snapshot comment above.
  restore();

  if (folded_values.empty() && reshape_fixes.empty()) {
    return false;
  }

  // Apply the two rewrites directly on the Graph: a folded node is replaced
  // in place by a Constant node holding its value (insertBefore + replaceAll
  // UsesWith + destroy preserves topological position without a full
  // node-list rebuild); a Reshape fix inserts a Constant just before the
  // Reshape and repoints its shape input at it, leaving the original
  // shape-producing subgraph dead for the optimizer's own dead-node
  // elimination to remove.
  static const onnx::Symbol kValueAttr("value");
  // Marked transient (see kTransientConstantAttr's own comment): this pass's
  // Constant nodes are its own intermediate representation for a value
  // already proved fully known, not a source of "not from initializers"
  // provenance -- the ordinary constant folder should normalize them away
  // like any other node.
  static const onnx::Symbol kTransientAttr(kTransientConstantAttr);
  for (onnx::Node* node : node_ptrs) {
    if (node->outputs().size() != 1) continue;
    onnx::Value* out = node->outputs()[0];
    auto iter = folded_values.find(out->uniqueName());
    if (iter != folded_values.end()) {
      onnx::Node* constant = g.create(onnx::kConstant, 1);
      constant->t_(kValueAttr, iter->second);
      constant->i_(kTransientAttr, 1);
      constant->outputs()[0]->setElemType(iter->second.elem_type());
      constant->outputs()[0]->setSizes(std::vector<onnx::Dimension>(
          iter->second.sizes().begin(), iter->second.sizes().end()));
      constant->insertBefore(node);
      node->replaceAllUsesWith(constant);
      node->destroy();
      continue;
    }
    auto fix_iter = reshape_fixes.find(out->uniqueName());
    if (fix_iter != reshape_fixes.end() && node->kind() == onnx::kReshape) {
      onnx::Node* shape_const = g.create(onnx::kConstant, 1);
      shape_const->t_(kValueAttr, fix_iter->second.shape_tensor);
      shape_const->i_(kTransientAttr, 1);
      shape_const->outputs()[0]->setElemType(
          fix_iter->second.shape_tensor.elem_type());
      shape_const->outputs()[0]->setSizes(std::vector<onnx::Dimension>(
          fix_iter->second.shape_tensor.sizes().begin(),
          fix_iter->second.shape_tensor.sizes().end()));
      shape_const->insertBefore(node);
      node->replaceInput(1, shape_const->outputs()[0]);
    }
  }
  return true;
}

// Whether every element of `tensor` is zero. Only the storage forms that can be
// inspected locally are accepted: a tensor whose data lives in an external file
// is reported as "not provably zero" rather than loaded.
bool IsAllZeroTensor(const onnx::Tensor& tensor) {
  if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
    return false;
  }
  if (tensor.elem_type() == onnx::TensorProto::STRING ||
      tensor.elem_type() == onnx::TensorProto::UNDEFINED) {
    return false;
  }
  if (tensor.is_raw_data()) {
    const std::string& raw = tensor.raw();
    return std::all_of(raw.begin(), raw.end(), [](char c) { return c == 0; });
  }
  auto all_zero = [](const auto& field) {
    return std::all_of(field.begin(), field.end(),
                       [](auto value) { return value == 0; });
  };
  const size_t element_count =
      tensor.floats().size() + tensor.int32s().size() + tensor.int64s().size() +
      tensor.doubles().size() + tensor.uint64s().size();
  if (element_count == 0) {
    return false;
  }
  return all_zero(tensor.floats()) && all_zero(tensor.int32s()) &&
         all_zero(tensor.int64s()) && all_zero(tensor.doubles()) &&
         all_zero(tensor.uint64s());
}

// Whether `value` is provably an all-zero tensor, following the chain of ops
// that produced it, walking onnx::Value/Node. Only shape-manipulating ops are
// traversed: they move elements around without changing their values, so an
// all-zero input implies an all-zero output. Ops whose output could be empty
// (Slice, Split, Gather) stay sound too, since an empty tensor is vacuously
// all zeros. ``value``'s producer being the graph's single kUndefined
// placeholder node (see ir_pb_converter.cc) means "this optional input was
// not provided".
bool IsAllZeroGraphValue(
    onnx::Value* value,
    const std::unordered_map<std::string, const onnx::Tensor*>&
        initializer_by_name,
    std::unordered_map<const onnx::Value*, bool>& memo) {
  if (value->node()->kind() == onnx::kUndefined) {
    return false;
  }
  auto memo_iter = memo.find(value);
  if (memo_iter != memo.end()) {
    return memo_iter->second;
  }
  // Insert a pessimistic answer up front: it both memoizes the miss and
  // breaks any cycle a malformed graph might contain.
  memo.emplace(value, false);

  auto init_iter = initializer_by_name.find(value->uniqueName());
  if (init_iter != initializer_by_name.end()) {
    const bool result = IsAllZeroTensor(*init_iter->second);
    memo[value] = result;
    return result;
  }

  onnx::Node* producer = value->node();
  if (producer->has_domain() && !producer->domain().empty() &&
      producer->domain() != "ai.onnx") {
    return false;
  }

  static const onnx::Symbol kConstant("Constant");
  static const onnx::Symbol kConstantOfShape("ConstantOfShape");
  static const onnx::Symbol kCast("Cast");
  static const onnx::Symbol kCastLike("CastLike");
  static const onnx::Symbol kIdentity("Identity");
  static const onnx::Symbol kReshape("Reshape");
  static const onnx::Symbol kTranspose("Transpose");
  static const onnx::Symbol kSqueeze("Squeeze");
  static const onnx::Symbol kUnsqueeze("Unsqueeze");
  static const onnx::Symbol kFlatten("Flatten");
  static const onnx::Symbol kTile("Tile");
  static const onnx::Symbol kExpand("Expand");
  static const onnx::Symbol kSlice("Slice");
  static const onnx::Symbol kSplit("Split");
  static const onnx::Symbol kGather("Gather");
  static const onnx::Symbol kGatherElements("GatherElements");
  static const onnx::Symbol kGatherND("GatherND");
  static const onnx::Symbol kConcat("Concat");
  static const onnx::Symbol kValue("value");
  static const onnx::Symbol kValueFloat("value_float");
  static const onnx::Symbol kValueInt("value_int");
  static const onnx::Symbol kValueFloats("value_floats");
  static const onnx::Symbol kValueInts("value_ints");
  static const onnx::Symbol kTo("to");

  bool result = false;
  const onnx::Symbol kind = producer->kind();
  if (kind == kConstant) {
    if (producer->hasAttribute(kValue)) {
      result = IsAllZeroTensor(producer->t(kValue));
    } else if (producer->hasAttribute(kValueFloat)) {
      result = producer->f(kValueFloat) == 0;
    } else if (producer->hasAttribute(kValueInt)) {
      result = producer->i(kValueInt) == 0;
    } else if (producer->hasAttribute(kValueFloats)) {
      const auto& floats = producer->fs(kValueFloats);
      result = !floats.empty() && std::all_of(floats.begin(), floats.end(),
                                              [](double f) { return f == 0; });
    } else if (producer->hasAttribute(kValueInts)) {
      const auto& ints = producer->is(kValueInts);
      result = !ints.empty() && std::all_of(ints.begin(), ints.end(),
                                            [](int64_t i) { return i == 0; });
    }
  } else if (kind == kConstantOfShape) {
    result =
        !producer->hasAttribute(kValue) || IsAllZeroTensor(producer->t(kValue));
  } else if (kind == kCast || kind == kCastLike) {
    const bool to_string = producer->hasAttribute(kTo) &&
                           producer->i(kTo) == onnx::TensorProto::STRING;
    result = !to_string && IsAllZeroGraphValue(producer->inputs()[0],
                                               initializer_by_name, memo);
  } else if (kind == kIdentity || kind == kReshape || kind == kTranspose ||
             kind == kSqueeze || kind == kUnsqueeze || kind == kFlatten ||
             kind == kTile || kind == kExpand || kind == kSlice ||
             kind == kSplit || kind == kGather || kind == kGatherElements ||
             kind == kGatherND) {
    result =
        IsAllZeroGraphValue(producer->inputs()[0], initializer_by_name, memo);
  } else if (kind == kConcat) {
    const auto& inputs = producer->inputs();
    result = !inputs.empty() &&
             std::all_of(inputs.begin(), inputs.end(), [&](onnx::Value* input) {
               return IsAllZeroGraphValue(input, initializer_by_name, memo);
             });
  }

  memo[value] = result;
  return result;
}

// The graph's single placeholder Value standing in for "this optional input
// was not provided" (see ir_pb_converter.cc) -- every kUndefined-producer
// input aliases this same Value. Only scanned when actually needed (an
// all-zero RNN initial state was found), so the common case of a graph with
// no LSTM/RNN/GRU nodes pays nothing for it.
onnx::Value* FindUndefinedGraphValue(onnx::Graph& graph) {
  for (onnx::Node* node : graph.nodes()) {
    if (node->kind() == onnx::kUndefined) {
      return node->outputs()[0];
    }
  }
  return nullptr;
}

// Unset the recurrent initial states of RNN/GRU/LSTM nodes that are provably
// all zeros (issue #314).
//
// paddle2onnx (like several other converters) materializes the zero initial
// hidden/cell state of an LSTM as a *batch-dependent* subgraph, because the
// state's shape is [num_directions, batch_size, hidden_size]:
//   Shape(x) -> Slice -> Concat([batch,1,1]) -> Tile(zeros) -> Transpose
//            -> Slice -> LSTM(initial_h, initial_c)
// When the model has a dynamic batch dimension none of that can be constant
// folded, so the simplified model keeps a Shape/Slice/Concat/Tile chain that
// downstream converters (onnx2ncnn in the issue) reject outright.
//
// The ONNX spec says initial_h/initial_c default to zero when omitted, so an
// input that is provably all zeros can simply be unset. The subgraph feeding it
// then becomes dead and is removed by the ordinary dead-end elimination pass.
// Only the initial states are unset; the equally zero-defaulting B and P inputs
// are left alone because they are plain initializers, so dropping them removes
// no operator while risking a needless behaviour change in consumers that read
// them.
void EliminateZeroRnnInitialState(onnx::Graph& graph) {
  std::unordered_map<std::string, const onnx::Tensor*> initializer_by_name;
  const auto& initializers = graph.initializers();
  const auto& initializer_names = graph.initializer_names();
  initializer_by_name.reserve(initializers.size());
  for (size_t i = 0; i < initializers.size(); ++i) {
    initializer_by_name[initializer_names[i]] = initializers[i].get();
  }
  std::unordered_map<const onnx::Value*, bool> memo;
  onnx::Value* undefined = nullptr;

  static const onnx::Symbol kLSTM("LSTM");
  static const onnx::Symbol kRNN("RNN");
  static const onnx::Symbol kGRU("GRU");

  for (onnx::Node* node : graph.nodes()) {
    // Recurse first, so recurrent ops inside If/Loop/Scan bodies are handled
    // too.
    for (onnx::Symbol attr : node->attributeNames()) {
      if (node->kindOf(attr) == onnx::AttributeKind::g) {
        EliminateZeroRnnInitialState(*node->g(attr));
      } else if (node->kindOf(attr) == onnx::AttributeKind::gs) {
        for (const auto& subgraph : node->gs(attr)) {
          EliminateZeroRnnInitialState(*subgraph);
        }
      }
    }

    if (node->has_domain() && !node->domain().empty() &&
        node->domain() != "ai.onnx") {
      continue;
    }
    const onnx::Symbol kind = node->kind();
    int last_state_index;
    if (kind == kLSTM) {
      last_state_index = 6;
    } else if (kind == kRNN || kind == kGRU) {
      last_state_index = 5;
    } else {
      continue;
    }

    const auto& inputs = node->inputs();
    for (int i = 5;
         i <= last_state_index && i < static_cast<int>(inputs.size()); i++) {
      if (IsAllZeroGraphValue(inputs[i], initializer_by_name, memo)) {
        if (undefined == nullptr) {
          undefined = FindUndefinedGraphValue(graph);
        }
        node->replaceInput(i, undefined);
      }
    }
    // Trailing empty inputs carry no information; drop them so the node ends
    // up in the same shape a converter would have emitted without the state.
    while (!node->inputs().empty() &&
           node->inputs().back()->node()->kind() == onnx::kUndefined) {
      node->removeInput(node->inputs().size() - 1);
    }
  }
}
