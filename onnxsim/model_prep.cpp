#include "model_prep.h"

#include <google/protobuf/io/zero_copy_stream.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <utility>

#include "bev_custom_op_schemas.h"
#include "constant_folding.h"
#include "contrib_schemas.h"
#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/version_converter/convert.h"

void MixBytes(const char* data, size_t n, uint64_t& h1, uint64_t& h2) {
  const size_t n_words = n / sizeof(uint64_t);
  for (size_t i = 0; i < n_words; ++i) {
    // memcpy, not a reinterpret_cast dereference: ``data`` is not guaranteed
    // 8-byte aligned (it may be a chunk of a streaming buffer, or a
    // std::string's internal buffer), and an unaligned uint64_t load is
    // undefined behavior in C++. Any decent compiler lowers this fixed-size
    // memcpy to a single unaligned load.
    uint64_t word;
    std::memcpy(&word, data + i * sizeof(uint64_t), sizeof(uint64_t));
    h1 = (h1 ^ word) * 1099511628211ULL;
    h2 = (h2 + word) * 0x9E3779B97F4A7C15ULL;
    h2 ^= h2 >> 29;
  }
  // Fewer than 8 trailing bytes: fall back to a byte-at-a-time mix so every
  // byte still participates in the hash.
  for (size_t i = n_words * sizeof(uint64_t); i < n; ++i) {
    const unsigned char c = static_cast<unsigned char>(data[i]);
    h1 = (h1 ^ c) * 1099511628211ULL;
    h2 = (h2 + c) * 0x9E3779B97F4A7C15ULL;
    h2 ^= h2 >> 29;
  }
}

// A protobuf output stream that hashes bytes as they are handed out by the
// serializer instead of collecting them into a buffer. Fingerprint() used to
// call ``model.SerializeAsString()`` -- which allocates a buffer the size of
// the whole serialized model (hundreds of MB of initializer weight data on a
// large model) and serializes into it -- and then made a second, separate
// pass over that buffer to hash it. Handing the serializer a small, reused
// scratch buffer instead avoids that large allocation (and the deallocation
// moments later): each time the serializer fills it, this stream
// mixes it into the running hash and hands the same buffer back out, so the
// whole model is hashed in bounded extra memory regardless of its size.
class HashingOutputStream final
    : public google::protobuf::io::ZeroCopyOutputStream {
 public:
  bool Next(void** data, int* size) override {
    FlushPending();
    *data = buf_.data();
    *size = static_cast<int>(buf_.size());
    pending_ = static_cast<int>(buf_.size());
    return true;
  }
  void BackUp(int count) override { pending_ -= count; }
  int64_t ByteCount() const override { return byte_count_; }

  // Call after serialization completes to mix in the final partial buffer.
  void Finish() { FlushPending(); }

  uint64_t h1() const { return h1_; }
  uint64_t h2() const { return h2_; }
  size_t total_bytes() const { return static_cast<size_t>(byte_count_); }

 private:
  void FlushPending() {
    if (pending_ <= 0) {
      return;
    }
    MixBytes(buf_.data(), static_cast<size_t>(pending_), h1_, h2_);
    byte_count_ += pending_;
    pending_ = 0;
  }

  // 64 KiB: large enough that the per-Next() call overhead is negligible next
  // to the memory-bandwidth-bound hashing work, small enough to stay resident
  // in L1/L2 cache across the mix. Heap-allocated, not a fixed-size array
  // member: this object lives on the call stack of Fingerprint(), which is
  // itself invoked from deep inside the (possibly nested) fixed-point
  // machinery, and the WASM build's stack is only tens of KiB total -- a
  // 64 KiB stack-resident array here reliably overflowed it (issue caught by
  // the WASM build's own smoke test, "Aborted(stack overflow ...)" on the
  // very first Simplify() call). std::vector's control block is a few
  // pointer-sized words on the stack; the buffer itself is heap-backed.
  std::vector<char> buf_ = std::vector<char>(1 << 16);
  int pending_ = 0;
  int64_t byte_count_ = 0;
  uint64_t h1_ = 1469598103934665603ULL;  // FNV-1a offset basis
  uint64_t h2_ = 0;
};

ModelFingerprint Fingerprint(const onnx::ModelProto& model) {
  // ModelProto contains no protobuf ``map<>`` fields, so serialization order is
  // stable and equal models serialize to identical bytes.
  //
  // Fingerprint() runs at every step of every (possibly nested) fixed point,
  // so on a model with hundreds of MB of initializer weight data -- which
  // dominates the serialized size but is untouched by most rounds (shape
  // inference, the onnx-optimizer passes) -- it still has to read all of that
  // data, every round, just to prove nothing changed. Profiling a 265 MB
  // model showed this was a multi-second fraction (over half of total wall
  // time on that model) of total simplification time, so it is worth
  // streaming through a small buffer (see HashingOutputStream) rather than
  // materializing the whole serialized model first.
  HashingOutputStream stream;
  // SerializeToZeroCopyStream only returns false if a Next() call on the
  // stream fails (out of disk space, a broken pipe, ...); HashingOutputStream
  // writes to memory and its Next() always succeeds, so this cannot fail.
  model.SerializeToZeroCopyStream(&stream);
  stream.Finish();
  uint64_t h1 = stream.h1();
  uint64_t h2 = stream.h2();
  h2 ^= stream.total_bytes();
  return {h1, h2};
}

// A no-op in-place transform (mutates nothing), used when shape inference or
// constant folding is disabled.
void Identity(onnx::ModelProto&) {}

// Recursively collect the op types of operators that live in ONNX's *default*
// domain but have no registered schema. These are custom operators -- most
// commonly TensorRT plugins such as ``BatchedNMS_TRT`` or ``EfficientNMS_TRT``
// -- that were exported into the default domain instead of a vendor-specific
// one.
//
// Custom ops that already live in a non-default domain (e.g. ``com.microsoft``
// or ``TRT``) are intentionally ignored: onnx::checker::check_model already
// tolerates unknown ops in non-standard domains, which is exactly the manual
// workaround reported in GitHub issue #220.
void CollectCustomDefaultDomainOps(const onnx::GraphProto& graph,
                                   int default_opset_version,
                                   std::set<std::string>& custom_ops) {
  for (const auto& node : graph.node()) {
    const std::string& domain = node.domain();
    const bool is_default_domain = domain.empty() || domain == "ai.onnx";
    if (is_default_domain &&
        onnx::OpSchemaRegistry::Schema(node.op_type(), default_opset_version,
                                       /*domain=*/"") == nullptr) {
      custom_ops.insert(node.op_type());
    }
    // Recurse into subgraphs held in node attributes (If/Loop/Scan bodies).
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectCustomDefaultDomainOps(attr.g(), default_opset_version,
                                      custom_ops);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectCustomDefaultDomainOps(subgraph, default_opset_version,
                                      custom_ops);
      }
    }
  }
}

// Register a permissive placeholder schema for every default-domain custom op
// found in ``model``. Without a schema, onnx::checker::check_model rejects the
// model with "No Op registered for <op> with domain_version of <n>" and
// simplification never even starts (GitHub issues #107 and #220). The
// placeholder accepts any number of inputs/outputs of any tensor type and any
// attributes, so the checker passes and the op is preserved untouched through
// simplification. It carries no shape/type inference function, so shape
// inference simply flows past the op as before.
void RegisterCustomDefaultDomainOpSchemas(const onnx::ModelProto& model) {
  int default_opset_version = 1;
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      default_opset_version =
          std::max(default_opset_version, static_cast<int>(opset.version()));
    }
  }

  std::set<std::string> custom_ops;
  CollectCustomDefaultDomainOps(model.graph(), default_opset_version,
                                custom_ops);

  for (const auto& op_type : custom_ops) {
    onnx::OpSchema schema;
    schema.SetName(op_type)
        .SetDomain("")
        .SinceVersion(1)
        .SetDoc(
            "Placeholder schema registered by onnxsim for a custom operator "
            "(e.g. a TensorRT plugin) exported into the default ONNX domain, "
            "so "
            "that the model passes validation and is simplified with the "
            "operator preserved unchanged.")
        .Input(0, "inputs", "Variadic inputs of the custom operator.", "T",
               onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
               /*min_arity=*/0)
        .Output(0, "outputs", "Variadic outputs of the custom operator.", "T",
                onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
                /*min_arity=*/0)
        .TypeConstraint("T", onnx::OpSchema::all_tensor_types(),
                        "Allow inputs and outputs of any tensor type.")
        // Custom ops carry arbitrary, plugin-specific attributes; accept them
        // all rather than trying to enumerate them.
        .AllowUncheckedAttributes();
    // Never fail or throw: a duplicate registration (e.g. simplifying two
    // models that use the same custom op in one process) is a harmless no-op.
    onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                         /*fail_duplicate_schema=*/false,
                         /*fail_with_exception=*/false);
  }
}

// Collect the names of every node in `graph`, descending into subgraphs held in
// node attributes (If/Loop/Scan bodies). Used to de-duplicate the names later
// assigned to nameless nodes.
void CollectNodeNames(const onnx::GraphProto& graph,
                      std::set<std::string>& names) {
  for (const auto& node : graph.node()) {
    if (!node.name().empty()) {
      names.insert(node.name());
    }
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectNodeNames(attr.g(), names);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectNodeNames(subgraph, names);
      }
    }
  }
}

// Give a unique, deterministic name to every node that has none. Nodes without
// a name survive simplification unnamed -- either because they were nameless in
// the input model or because an onnx-optimizer pass created a replacement node
// without setting a name -- which trips up downstream tooling that keys on node
// names (issue #269). Each generated name is derived from the op type plus a
// running counter and de-duplicated against every name already present in the
// graph (including names generated earlier in this pass). Subgraphs are handled
// recursively so nodes inside If/Loop/Scan bodies are named too.
void AssignMissingNodeNames(onnx::GraphProto& graph,
                            std::set<std::string>& used_names, size_t& counter,
                            std::vector<std::string>& assigned) {
  for (auto& node : *graph.mutable_node()) {
    if (node.name().empty()) {
      std::string name;
      do {
        name = node.op_type() + "_" + std::to_string(counter++);
      } while (used_names.count(name) > 0);
      used_names.insert(name);
      node.set_name(name);
      assigned.push_back(name);
    }
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        AssignMissingNodeNames(*attr.mutable_g(), used_names, counter,
                               assigned);
      }
      for (auto& subgraph : *attr.mutable_graphs()) {
        AssignMissingNodeNames(subgraph, used_names, counter, assigned);
      }
    }
  }
}

// Assign names to any nodes left nameless after simplification (issue #269).
std::vector<std::string> AssignMissingNodeNames(onnx::ModelProto& model) {
  std::set<std::string> used_names;
  CollectNodeNames(model.graph(), used_names);
  size_t counter = 0;
  std::vector<std::string> assigned;
  AssignMissingNodeNames(*model.mutable_graph(), used_names, counter, assigned);
  return assigned;
}

// Shape inference can leave behind a value_info entry that records a shape
// but never resolved an element type (e.g. observed on a real-world model's
// Reshape outputs inside attention blocks: graph_shape_inference.h's
// InferShapesOnGraph propagated the target shape but left elem_type at its
// default UNDEFINED). value_info is purely optional annotation -- nothing
// reads it as ground truth -- but an entry with elem_type == UNDEFINED is a
// malformed TypeProto, and onnx::checker::check_model does not reject it,
// while onnxruntime's own model loader is stricter and refuses to load the
// model at all with "Invalid tensor data type 0". Drop such entries instead
// of leaving broken metadata in the output model; the same op's actual
// output tensor is unaffected either way.
// Recurses into subgraphs (If/Loop/Scan bodies) for the same reason
// AssignMissingNodeNames does.
void DropIncompleteValueInfo(onnx::GraphProto& graph) {
  auto& value_info = *graph.mutable_value_info();
  value_info.erase(
      std::remove_if(value_info.begin(), value_info.end(),
                     [](const onnx::ValueInfoProto& vi) {
                       return vi.type().has_tensor_type() &&
                              vi.type().tensor_type().elem_type() ==
                                  onnx::TensorProto::UNDEFINED;
                     }),
      value_info.end());
  for (auto& node : *graph.mutable_node()) {
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        DropIncompleteValueInfo(*attr.mutable_g());
      }
      for (auto& subgraph : *attr.mutable_graphs()) {
        DropIncompleteValueInfo(subgraph);
      }
    }
  }
}

void DropIncompleteValueInfo(onnx::ModelProto& model) {
  DropIncompleteValueInfo(*model.mutable_graph());
}

void Check(const onnx::ModelProto& model) { onnx::checker::check_model(model); }

// Return the opset version the model imports for the default ONNX domain
// (represented as either the empty string or "ai.onnx"), or std::nullopt when
// the model does not import the default domain at all (a model made purely of
// custom-domain operators).
std::optional<int> DefaultOpsetVersion(const onnx::ModelProto& model) {
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      return static_cast<int>(opset.version());
    }
  }
  return std::nullopt;
}

// Mirrors onnx_simplifier.py's remove_initializer_from_input: an initializer
// that also appears in graph.input is treated as a runtime input by
// onnxoptimizer's is_constant_initializer, which blocks value-baking fusions
// (fuse_bn_into_conv, ...) that only work on non-input initializers -- e.g.
// the plain Conv+BN chains of the opset-8 resnet101-v1-7 were left completely
// unsimplified without this. Removing it from the input list lets it fold
// like any other constant. Kept in sync with the Python function; see that
// one's own comment for the opset-6 floor (Cast's `to` attribute's INT
// encoding wasn't valid before opset 6, so bumping an old-IR/old-opset model
// to IR 4 here would let a later fusion emit a node the opset rejects).
void RemoveInitializerFromInput(onnx::ModelProto& model) {
  constexpr int kMinOpsetForInitializerFold = 6;
  if (model.ir_version() < 4) {
    const auto opset = DefaultOpsetVersion(model);
    if (!opset || *opset < kMinOpsetForInitializerFold) {
      return;
    }
  }
  std::set<std::string> initializer_names;
  for (const auto& init : model.graph().initializer()) {
    initializer_names.insert(init.name());
  }
  auto* inputs = model.mutable_graph()->mutable_input();
  bool removed_any = false;
  for (int i = inputs->size() - 1; i >= 0; --i) {
    if (initializer_names.count((*inputs)[i].name()) > 0) {
      inputs->erase(inputs->begin() + i);
      removed_any = true;
    }
  }
  if (removed_any && model.ir_version() < 4) {
    model.set_ir_version(4);
  }
}

// ONNX TensorProto element types onnxoptimizer's tensor-value hashing
// (cse_util.h) knows how to hash. Any other type makes
// eliminate_common_subexpression/eliminate_duplicate_initializer raise
// "no supported data type: <N>". Enumerates the *supported* types (rather
// than the unsupported ones) so an element type added to ONNX in the future
// is treated as unhashable by default instead of silently crashing. Mirrors
// onnx_simplifier.py's _CSE_HASHABLE_ELEM_TYPES.
bool IsCSEHashableElemType(int32_t elem_type) {
  switch (elem_type) {
    case onnx::TensorProto::UNDEFINED:
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT64:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
    case onnx::TensorProto::FLOAT:
    case onnx::TensorProto::DOUBLE:
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::BFLOAT16:
    case onnx::TensorProto::COMPLEX64:
    case onnx::TensorProto::COMPLEX128:
    case onnx::TensorProto::STRING:
      return true;
    default:
      return false;
  }
}

// Mirrors onnx_simplifier.py's _has_cse_unhashable_tensor /
// _iter_tensor_data_types: walks every tensor CSE might hash -- initializers,
// t/ts node attributes, recursing into subgraphs -- looking for an element
// type IsCSEHashableElemType rejects.
bool GraphHasCSEUnhashableTensor(const onnx::GraphProto& graph) {
  for (const auto& init : graph.initializer()) {
    if (!IsCSEHashableElemType(init.data_type())) {
      return true;
    }
  }
  for (const auto& node : graph.node()) {
    for (const auto& attr : node.attribute()) {
      if (attr.has_t() && !IsCSEHashableElemType(attr.t().data_type())) {
        return true;
      }
      for (const auto& t : attr.tensors()) {
        if (!IsCSEHashableElemType(t.data_type())) {
          return true;
        }
      }
      if (attr.has_g() && GraphHasCSEUnhashableTensor(attr.g())) {
        return true;
      }
      for (const auto& g : attr.graphs()) {
        if (GraphHasCSEUnhashableTensor(g)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Mirrors onnx_simplifier.py's overwrite_input_shapes loop: for each named
// graph input, overwrite its shape's dims with the given values, skipping
// any non-positive entry (which means "keep the original, possibly dynamic,
// dimension" rather than hardcoding an invalid size such as 0 -- GitHub
// issue #237). Throws std::runtime_error if a name isn't a graph input
// (matching onnx_simplifier.py's RuntimeError for the same case).
void ApplyInputShapeOverwrite(
    onnx::ModelProto& model,
    const std::unordered_map<std::string, std::vector<int64_t>>&
        overwrite_input_shapes) {
  for (const auto& [name, shape] : overwrite_input_shapes) {
    bool found = false;
    for (auto& input : *model.mutable_graph()->mutable_input()) {
      if (input.name() != name) {
        continue;
      }
      found = true;
      auto* dims = input.mutable_type()
                       ->mutable_tensor_type()
                       ->mutable_shape()
                       ->mutable_dim();
      for (int i = 0; i < dims->size() && i < static_cast<int>(shape.size());
           ++i) {
        if (shape[i] > 0) {
          dims->Mutable(i)->set_dim_value(shape[i]);
        }
      }
    }
    if (!found) {
      throw std::runtime_error("The model doesn't have input named \"" + name +
                               "\"");
    }
  }
}

// Mirrors onnx_simplifier.py's remove_unused_output: drops the named graph
// outputs. Downstream dead-end elimination cleans up nodes that only fed
// them. Throws std::runtime_error if a name isn't a graph output (matching
// onnx_simplifier.py's RuntimeError for the same case).
void RemoveUnusedOutputs(onnx::ModelProto& model,
                         const std::vector<std::string>& unused_output) {
  for (const auto& name : unused_output) {
    bool found = false;
    for (const auto& output : model.graph().output()) {
      if (output.name() == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("The model doesn't have output named \"" + name +
                               "\"");
    }
  }
  auto* outputs = model.mutable_graph()->mutable_output();
  google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> kept;
  for (auto& output : *outputs) {
    if (std::find(unused_output.begin(), unused_output.end(), output.name()) ==
        unused_output.end()) {
      *kept.Add() = std::move(output);
    }
  }
  outputs->Swap(&kept);
}

// Convert the default ONNX domain of the model to target_version using onnx's
// own version converter. Only the default ONNX domain opset is changed;
// custom/other-domain opset imports are left untouched. Returns the model
// unchanged when it is already at the requested version or does not import the
// default domain.
//
// Takes ``model`` by value, and passes it as a mutable lvalue to
// ConvertVersion below, so the call chain resolves to ConvertVersion's
// consuming overload -- which moves initializer bytes through the Graph IR
// round trip instead of copying them -- rather than the copying, const-ref
// overload. Callers should ``std::move`` in a model they no longer need, so
// this by-value parameter is itself move-constructed rather than copied.
onnx::ModelProto ConvertOpsetVersion(onnx::ModelProto model,
                                     int target_version) {
  const auto current = DefaultOpsetVersion(model);
  if (!current || *current == target_version) {
    return model;
  }
  return onnx::version_conversion::ConvertVersion(model, target_version);
}

// Shared schema setup for the single-pass debug helpers below, mirroring the
// head of ``Simplify``: teach shape inference about ONNX Runtime's quantized
// contrib ops and register permissive placeholders for custom ops exported into
// the default ONNX domain, so neither shape inference nor a later checker
// rejects the model. Unlike ``Simplify`` these helpers do not ``Check`` the
// model up front -- the point of running a step in isolation is to debug a
// model that may not yet be fully valid.
void PrepareSchemasForDebug(const onnx::ModelProto& model) {
  onnxsim::RegisterContribOpSchemas();
  onnxsim::RegisterBevCustomOpSchemas();
  FixupSchemaDeterminism();
  RegisterCustomDefaultDomainOpSchemas(model);
}
