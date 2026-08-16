#include "incremental_shape_infer.h"

#include <cstdint>
#include <cstring>
#include <list>
#include <unordered_set>
#include <vector>

#include "onnx/common/constants.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/shape_inference/implementation.h"

namespace onnxsim {
namespace {

using onnx::AttributeProto;
using onnx::GraphProto;
using onnx::ModelProto;
using onnx::NodeProto;
using onnx::OpSchema;
using onnx::OpSchemaRegistry;
using onnx::ShapeInferenceOptions;
using onnx::SparseTensorProto;
using onnx::TensorProto;
using onnx::TypeProto;
using onnx::shape_inference::checkShapesAndTypes;
using onnx::shape_inference::DataValueMap;
using onnx::shape_inference::InferenceContextImpl;
using onnx::shape_inference::MaterializeSymbolicShape;
using onnx::shape_inference::mergeShapesAndTypes;
using onnx::shape_inference::SymbolTableImpl;
using onnx::shape_inference::TraverseGraphsToAddExistingSymbols;

// Appends `s` to `out` preceded by its byte length (8 raw bytes, native
// order). Concatenating several length-prefixed strings this way is
// unambiguous for equality: two different sequences of strings never
// serialize to the same bytes, which is all the cache key/value below ever
// need (neither is ever parsed back apart field-by-field, only compared or,
// for the value, split back into the same fixed list of outputs it was built
// from).
void AppendLP(std::string& out, const std::string& s) {
  uint64_t n = s.size();
  out.append(reinterpret_cast<const char*>(&n), sizeof(n));
  out.append(s);
}

// Reads one length-prefixed string starting at `*pos` (as written by
// AppendLP), advancing `*pos` past it.
std::string ReadLP(const std::string& bytes, size_t* pos) {
  uint64_t n = 0;
  std::memcpy(&n, bytes.data() + *pos, sizeof(n));
  *pos += sizeof(n);
  std::string s = bytes.substr(*pos, n);
  *pos += n;
  return s;
}

bool HasGraphAttribute(const NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.type() == AttributeProto::GRAPH ||
        attr.type() == AttributeProto::GRAPHS)
      return true;
  }
  return false;
}

bool IsConstantWithSparseValue(const NodeProto& node) {
  if (node.op_type() != "Constant" ||
      !(node.domain().empty() || node.domain() == "ai.onnx"))
    return false;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "value" && attr.type() == AttributeProto::SPARSE_TENSOR)
      return true;
  }
  return false;
}

// Mirrors ``ShapeInferenceImplBase::ProcessConstant``/``AddTemporaryConstant``
// in onnx's shape_inference/implementation.cc: a node's *value*, not just its
// type, can matter to a downstream op's shape inference function (Reshape
// reads its target shape from a Constant's tensor, Squeeze reads its axes
// from one, ...). Tracks the same two sources onnx's own driver does -- a
// Constant node's `value` attribute, and any INT/INTS/FLOAT/FLOATS-typed
// attribute of a Constant node (`value_int`/`value_ints`/`value_float`/
// `value_floats`) -- into the shared input-data map InferenceContextImpl
// reads from. (Sparse `value` is excluded: callers bail to the full onnx
// driver before this is used whenever a Constant carries one, via
// IsConstantWithSparseValue.)
template <typename T>
void AddTemporaryConstant(
    const std::string& name, const std::vector<T>& v, bool scalar,
    std::unordered_map<std::string, TensorProto>& owned,
    std::unordered_map<std::string, const TensorProto*>& input_data_by_name) {
  TensorProto t = onnx::ToTensor(v);
  if (!scalar) t.add_dims(static_cast<int64_t>(v.size()));
  owned[name] = std::move(t);
  input_data_by_name[name] = &owned[name];
}

void ProcessConstantNode(
    const NodeProto& n, std::unordered_map<std::string, TensorProto>& owned,
    std::unordered_map<std::string, const TensorProto*>& input_data_by_name) {
  if (n.op_type() != "Constant" ||
      !(n.domain().empty() || n.domain() == "ai.onnx") || n.output_size() != 1)
    return;
  const std::string& output_name = n.output(0);
  for (const auto& attr : n.attribute()) {
    if (attr.name() == "value") {
      if (attr.type() == AttributeProto::TENSOR && attr.has_t()) {
        owned[output_name] = attr.t();
        input_data_by_name[output_name] = &owned[output_name];
      }
      // SPARSE_TENSOR handled by the caller's upfront bail-out.
    } else {
      switch (attr.type()) {
        case AttributeProto::INTS:
          AddTemporaryConstant(
              output_name,
              std::vector<int64_t>(attr.ints().begin(), attr.ints().end()),
              false, owned, input_data_by_name);
          break;
        case AttributeProto::INT:
          AddTemporaryConstant(output_name, std::vector<int64_t>{attr.i()},
                               true, owned, input_data_by_name);
          break;
        case AttributeProto::FLOATS:
          AddTemporaryConstant(
              output_name,
              std::vector<float>(attr.floats().begin(), attr.floats().end()),
              false, owned, input_data_by_name);
          break;
        case AttributeProto::FLOAT:
          AddTemporaryConstant(output_name, std::vector<float>{attr.f()}, true,
                               owned, input_data_by_name);
          break;
        default:
          break;
      }
    }
  }
}

// True if any node needs handling this fast path does not implement, in
// which case the caller falls back to the exact full onnx driver for the
// whole model this round: a graph-valued attribute (If/Loop/Scan/...), a
// Constant with a sparse tensor value, or a schema whose "inference" is
// expanding a function body (rarer still, since RegisterContribOpSchemas/
// RegisterCustomDefaultDomainOpSchemas -- run before Simplify's fixed point
// -- already give onnxsim's own ops and ORT's contrib ops direct C++
// inference functions).
bool HasUnmodeledNode(
    const GraphProto& graph,
    const std::unordered_map<std::string, int>& opset_imports) {
  if (graph.sparse_initializer_size() > 0) return true;
  for (const auto& n : graph.node()) {
    if (HasGraphAttribute(n) || IsConstantWithSparseValue(n)) return true;
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end() && n.domain() == onnx::ONNX_DOMAIN) {
      dit = opset_imports.find(onnx::AI_ONNX_DOMAIN);
    }
    if (dit == opset_imports.end())
      continue;  // left to InferenceContextImpl below: no import, no inference.
    const OpSchema* schema = OpSchemaRegistry::Instance()->GetSchema(
        n.op_type(), dit->second, n.domain());
    if (schema && !schema->has_type_and_shape_inference_function() &&
        schema->HasFunction())
      return true;
  }
  return false;
}

}  // namespace

void IncrementalShapeInferer::Run(ModelProto& model) {
  GraphProto& graph = *model.mutable_graph();

  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset : model.opset_import())
    opset_imports[opset.domain()] = static_cast<int>(opset.version());

  if (model.functions_size() > 0 || HasUnmodeledNode(graph, opset_imports)) {
    onnx::shape_inference::InferShapes(model);
    return;
  }

  const ShapeInferenceOptions options(/*check_type=*/false, /*error_mode=*/0,
                                      /*enable_data_propagation=*/false);
  SymbolTableImpl symbol_table;
  TraverseGraphsToAddExistingSymbols(graph, symbol_table);
  // Data propagation is off (`options.enable_data_propagation == false`), so
  // this is never populated; InferenceContextImpl still wants a valid
  // pointer, matching onnx's own driver (InferShapesImpl defaults it to an
  // empty map rather than null).
  DataValueMap generated_shape_data;

  std::unordered_map<std::string, TypeProto*> value_types_by_name;
  // Names declared (as a graph input/output/value_info) without a type,
  // mapped to that declaration's TypeProto so a type resolved for them below
  // is also written directly into it -- e.g. a graph output declared with no
  // type gets its real inferred type written into graph.output() itself, not
  // only recorded in a same-named value_info entry. Mirrors onnx's own
  // ShapeInferenceImplBase::undefined_value_types_by_name exactly, including
  // its one surprising side effect: calling `vi.mutable_type()` here flips
  // `vi.has_type()` to true from this point on even if nothing is ever
  // written into it (an unresolvable name stays serialized as an explicit
  // empty `type {}` rather than an absent field) -- replicated so a model
  // this pass leaves partly uninferred serializes identically to what the
  // exact full onnx::shape_inference::InferShapes would have produced.
  std::unordered_map<std::string, TypeProto*> undefined_value_types_by_name;
  // Stable addresses for TypeProtos synthesized below (from initializers):
  // std::list never invalidates existing elements on further push_back.
  std::list<TypeProto> owned_types;
  auto register_vi = [&](onnx::ValueInfoProto& vi) {
    if (vi.has_type()) {
      value_types_by_name[vi.name()] = vi.mutable_type();
    } else {
      undefined_value_types_by_name[vi.name()] = vi.mutable_type();
    }
  };
  for (auto& vi : *graph.mutable_value_info()) register_vi(vi);
  for (auto& vi : *graph.mutable_input()) register_vi(vi);
  for (auto& vi : *graph.mutable_output()) register_vi(vi);

  // Seeded with every initializer, then grown as Constant nodes are visited
  // below (in topological order, so a node's inputs are already present by
  // the time it is processed if they were produced by an earlier Constant).
  std::unordered_map<std::string, const TensorProto*> input_data_by_name;
  std::unordered_map<std::string, TensorProto> owned_constants;
  for (const auto& tp : graph.initializer()) {
    TypeProto t;
    auto* tt = t.mutable_tensor_type();
    tt->set_elem_type(tp.data_type());
    auto* shape = tt->mutable_shape();
    for (int64_t d : tp.dims()) shape->add_dim()->set_dim_value(d);
    input_data_by_name[tp.name()] = &tp;
    auto it = value_types_by_name.find(tp.name());
    if (it != value_types_by_name.end()) {
      // Same dual-role precedence as onnx's ProcessInitializer: an
      // initializer that is also a graph input keeps the input's declared
      // type, after checking the two agree.
      checkShapesAndTypes(t, *it->second);
    } else if (model.ir_version() >= 4) {
      owned_types.push_back(std::move(t));
      value_types_by_name[tp.name()] = &owned_types.back();
    }
  }

  auto update_type = [&](const std::string& name, TypeProto* inferred) {
    if (inferred->value_case() == TypeProto::VALUE_NOT_SET) return;
    // Materializing here (rather than before caching, below) means a fresh
    // symbol is assigned per point of use: two node instances that happen to
    // share a cache key (identical op/attrs/input types/input values) can
    // still sit at genuinely different, independently-unknown positions in
    // the graph, so their unresolved output dims must not be merged into the
    // same symbol just because the cache said so.
    MaterializeSymbolicShape(inferred, symbol_table);
    auto it = value_types_by_name.find(name);
    if (it != value_types_by_name.end()) {
      mergeShapesAndTypes(*inferred, it->second);
    } else {
      auto* vi = graph.add_value_info();
      vi->set_name(name);
      *vi->mutable_type() = *inferred;
      value_types_by_name[name] = vi->mutable_type();
      auto uit = undefined_value_types_by_name.find(name);
      if (uit != undefined_value_types_by_name.end()) *uit->second = *inferred;
    }
  };

  // Above this many bytes, a value is never included in a cache key (see the
  // loop below): a shape-computing op's value-carrying inputs (Reshape's
  // target, Slice's starts/ends/axes, Squeeze/Unsqueeze's axes, ...) are
  // always tiny -- bounded by the output's rank, realistically a handful of
  // int64s -- so this comfortably covers every legitimate case while
  // excluding large data tensors (Conv/MatMul/Gemm weights, folded constant
  // blobs, ...) that happen to reach InferenceContextImpl as `getInputData()`
  // candidates but whose *value* no such op's inference function actually
  // reads (only its type/shape, which the key already carries separately).
  constexpr size_t kMaxCachedValueBytes = 4096;

  const std::unordered_set<std::string> empty_unbound;
  auto invoke_inference = [&](NodeProto& node, const OpSchema& schema) {
    std::vector<TypeProto> out_types(static_cast<size_t>(node.output_size()));
    InferenceContextImpl ctx(node, value_types_by_name, input_data_by_name, {},
                             options, &generated_shape_data,
                             /*graphInferenceContext=*/nullptr, &empty_unbound);
    try {
      schema.GetTypeAndShapeInferenceFunction()(ctx);
      for (int i = 0; i < node.output_size(); ++i) {
        if (!node.output(i).empty())
          out_types[static_cast<size_t>(i)] =
              *ctx.getOutputType(static_cast<size_t>(i));
      }
    } catch (const std::exception&) {
      // error_mode == 0: onnx's own driver swallows node-level inference
      // errors the same way (collected, never thrown); match that by leaving
      // this node's outputs unset, same as an unsupported op.
    }
    return out_types;
  };

  for (NodeProto& n : *graph.mutable_node()) {
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end() && n.domain() == onnx::ONNX_DOMAIN) {
      dit = opset_imports.find(onnx::AI_ONNX_DOMAIN);
    }
    const OpSchema* schema = dit != opset_imports.end()
                                 ? OpSchemaRegistry::Instance()->GetSchema(
                                       n.op_type(), dit->second, n.domain())
                                 : nullptr;
    if (!schema || !schema->has_type_and_shape_inference_function()) {
      // No opset import for the node's domain, or a genuinely unsupported
      // (e.g. custom) op: leave its output type unset, same as onnx's own
      // driver does for these (`has_unsupported_op = true; return;`).
      ProcessConstantNode(n, owned_constants, input_data_by_name);
      continue;
    }

    // Cache key: every input InferenceContextImpl can observe for this node
    // -- the operator version, its attributes, and each input's current type
    // and (if known and small, see kMaxCachedValueBytes) constant value.
    // Identical bytes guarantee InferenceContextImpl would see identical
    // inputs and so recompute an identical result; this needs no notion of
    // "the same node as a previous round" at all, which sidesteps the fact
    // that onnx-optimizer passes rebuild the whole NodeProto list every round
    // with no stable node identity across rounds.
    std::string key;
    bool cacheable = true;
    AppendLP(key, n.domain());
    AppendLP(key, n.op_type());
    AppendLP(key, std::to_string(dit->second));
    for (const auto& attr : n.attribute())
      AppendLP(key, attr.SerializeAsString());
    for (const auto& input : n.input()) {
      if (input.empty()) {
        AppendLP(key,
                 "\x01"
                 "absent");
        continue;
      }
      auto tit = value_types_by_name.find(input);
      AppendLP(key, tit != value_types_by_name.end()
                        ? tit->second->SerializeAsString()
                        : std::string("\x01"
                                      "notype"));
      auto dit2 = input_data_by_name.find(input);
      if (dit2 == input_data_by_name.end()) {
        AppendLP(key,
                 "\x01"
                 "novalue");
      } else if (static_cast<size_t>(dit2->second->ByteSizeLong()) >
                 kMaxCachedValueBytes) {
        cacheable = false;
        break;
      } else {
        AppendLP(key, dit2->second->SerializeAsString());
      }
    }

    std::vector<TypeProto> out_types;
    if (!cacheable) {
      out_types = invoke_inference(n, *schema);
    } else if (auto cached = cache_.find(key); cached != cache_.end()) {
      out_types.resize(static_cast<size_t>(n.output_size()));
      size_t pos = 0;
      for (auto& t : out_types) t.ParseFromString(ReadLP(cached->second, &pos));
    } else {
      out_types = invoke_inference(n, *schema);
      std::string serialized;
      for (const auto& t : out_types)
        AppendLP(serialized, t.SerializeAsString());
      cache_.emplace(std::move(key), std::move(serialized));
    }

    for (int i = 0; i < n.output_size(); ++i) {
      if (!n.output(i).empty())
        update_type(n.output(i), &out_types[static_cast<size_t>(i)]);
    }
    ProcessConstantNode(n, owned_constants, input_data_by_name);
  }
}

}  // namespace onnxsim
