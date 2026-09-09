/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The C++ port of onnxsim/lora.py -- see lora_entry.h for what is ported,
 * what is deliberately not (the driving loop; onnxsim.nf4's own
 * quantization step), and why. None of the technique is repeated here --
 * there should be one place to update when a derivation changes, and it is
 * the Python.
 *
 * Three independent pieces live in this file, in the order lora_entry.h
 * documents them:
 *
 *   1. InjectLora -- graph surgery, no step graph involved. Builds raw
 *      NodeProtos directly with onnx::helper-equivalent code, exactly as
 *      lora.py's inject_lora uses onnx.helper rather than
 *      qat_graph.GraphBuilder for the same reason (this is a one-shot edit
 *      of a model meant to be run as-is, not a step-graph build).
 *
 *   2. FoldFrozenPrefixes plus a small constant evaluator -- lora.py's own
 *      constant folder runs the candidate subgraph through a real ONNX
 *      backend (onnxsim.backend.run_model), which this build has no
 *      equivalent of (see qat_entry_test.cpp's own comment on why: the
 *      wheel does not compile ONNX Runtime, and the WASM build hands
 *      evaluation to onnxruntime-web at run time -- neither is a C++
 *      evaluator this library can call at *build* time). So this port
 *      carries its own small, explicitly-scoped evaluator instead of a
 *      general one; see EvalNode's own comment for exactly what it covers
 *      and why that is enough. BuildLoraStepGraph's own comment records a
 *      finding, checked directly against lora.py: given how SliceBlock
 *      decides what belongs to a block, this function never actually has
 *      anything to fold, in either language -- see that comment before
 *      assuming it is exercised by BuildLoraStepGraph's own tests.
 *      LoraFoldForTesting (lora_entry.h) exists so lora_entry_test.cpp can
 *      still exercise EvalNode directly.
 *
 *   3. BuildLoraStepGraph / WriteBackLoraState -- the step-graph builder and
 *      its write-back tail, structured like qat_entry.cpp's
 *      BuildQatStepGraph / WriteBackQatState but without any of the
 *      fake-quant/scale machinery that exists there only because QAT has a
 *      quantizer to train. LoRA does not, so this is the plainer of the
 *      two: a block's own nodes verbatim, one graph_grad::BuildBackward
 *      call restricted to the adapter's own tensors, one
 *      qat_graph_builder::AdamUpdate per tensor.
 *
 * Several small helpers below (SliceBlock, the shape-inference plumbing,
 * the protobuf decode/encode utilities) are near-duplicates of ones in
 * qat_entry.cpp. They are not shared through a header because qat_entry.cpp
 * keeps them in its own anonymous namespace -- exactly as lora.py needing
 * qat.py's _slice_block/_block_shapes costs nothing in Python (a plain
 * import) but does cost a translation-unit boundary in C++. Duplicating
 * ~150 lines of general graph-slicing/shape-inference plumbing here was
 * judged cheaper and less risky than exporting it from qat_entry.h/.cpp
 * and re-plumbing that file's own tests around a new public surface it
 * does not otherwise need.
 */
#include "lora_entry.h"

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph_grad.h"
#include "onnx/checker.h"
#include "onnx/shape_inference/implementation.h"

namespace {

using Shape = std::vector<int64_t>;
using ShapeMap = std::map<std::string, Shape>;
using ElemTypeMap = std::map<std::string, int32_t>;

const char kPrefix[] = "lora__";

// ---------------------------------------------------------------------------
// Small helpers over the protobuf types -- see this file's top comment on
// why these duplicate (rather than import) qat_entry.cpp's own copies.
// ---------------------------------------------------------------------------

const onnx::AttributeProto* FindAttr(const onnx::NodeProto& node,
                                     const std::string& name) {
  for (const onnx::AttributeProto& attribute : node.attribute()) {
    if (attribute.name() == name) return &attribute;
  }
  return nullptr;
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

std::string Quoted(const std::string& value) { return "'" + value + "'"; }

std::string QuotedList(const std::set<std::string>& values) {
  std::string out = "[";
  bool first = true;
  for (const std::string& value : values) {
    if (!first) out += ", ";
    first = false;
    out += Quoted(value);
  }
  return out + "]";
}

int64_t ElementCount(const Shape& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// raw_data is little-endian on every host -- see onnxsim/passes/endian_read.h
// -- so these decode/encode by shifting bytes, not by casting the host's own
// layout over them.
uint64_t DecodeLE(const char* bytes, int nbytes) {
  uint64_t bits = 0;
  for (int i = 0; i < nbytes; ++i) {
    bits |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[i]))
            << (8 * i);
  }
  return bits;
}

void AppendLE(std::string& out, uint64_t bits, int nbytes) {
  for (int i = 0; i < nbytes; ++i) {
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
}

float DecodeFloat(const char* bytes) {
  const uint32_t bits = static_cast<uint32_t>(DecodeLE(bytes, 4));
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void AppendFloat(std::string& out, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  AppendLE(out, bits, 4);
}

// A numeric tensor's values as doubles, whichever field they are stored in.
// Wider than qat_entry.cpp's own TensorValues (which reads only what
// BuildQatStepGraph needs -- FLOAT/UINT8/INT8) because FoldFrozenPrefixes'
// evaluator has to read whatever an initializer feeding a dequant chain
// happens to be stored as: a codebook or scale (FLOAT), a code array
// (UINT8/INT8), or a Reshape/Gather shape or index tensor (INT64, and INT32
// on some producers).
std::vector<double> TensorValues(const onnx::TensorProto& t) {
  std::vector<double> out;
  const int32_t type = t.data_type();
  if (t.has_raw_data()) {
    const std::string& raw = t.raw_data();
    auto decode = [&](int nbytes, bool is_signed) {
      for (size_t i = 0; i + static_cast<size_t>(nbytes) <= raw.size();
           i += static_cast<size_t>(nbytes)) {
        const uint64_t bits = DecodeLE(raw.data() + i, nbytes);
        if (is_signed) {
          const uint64_t sign_bit = uint64_t{1} << (nbytes * 8 - 1);
          const int64_t value =
              (bits ^ sign_bit) - sign_bit;  // sign-extend nbytes -> 64
          out.push_back(static_cast<double>(value));
        } else {
          out.push_back(static_cast<double>(bits));
        }
      }
    };
    switch (type) {
      case onnx::TensorProto::FLOAT:
        for (size_t i = 0; i + 4 <= raw.size(); i += 4) {
          out.push_back(DecodeFloat(raw.data() + i));
        }
        return out;
      case onnx::TensorProto::DOUBLE:
        for (size_t i = 0; i + 8 <= raw.size(); i += 8) {
          const uint64_t bits = DecodeLE(raw.data() + i, 8);
          double value = 0.0;
          std::memcpy(&value, &bits, sizeof(value));
          out.push_back(value);
        }
        return out;
      case onnx::TensorProto::UINT8:
      case onnx::TensorProto::BOOL:
        decode(1, /*is_signed=*/false);
        return out;
      case onnx::TensorProto::INT8:
        decode(1, /*is_signed=*/true);
        return out;
      case onnx::TensorProto::UINT16:
        decode(2, /*is_signed=*/false);
        return out;
      case onnx::TensorProto::INT16:
        decode(2, /*is_signed=*/true);
        return out;
      case onnx::TensorProto::INT32:
        decode(4, /*is_signed=*/true);
        return out;
      case onnx::TensorProto::INT64:
        decode(8, /*is_signed=*/true);
        return out;
      default:
        break;
    }
  } else {
    switch (type) {
      case onnx::TensorProto::FLOAT:
        for (float v : t.float_data()) out.push_back(v);
        return out;
      case onnx::TensorProto::DOUBLE:
        for (double v : t.double_data()) out.push_back(v);
        return out;
      case onnx::TensorProto::UINT8:
      case onnx::TensorProto::INT8:
      case onnx::TensorProto::UINT16:
      case onnx::TensorProto::INT16:
      case onnx::TensorProto::INT32:
      case onnx::TensorProto::BOOL:
        for (int32_t v : t.int32_data()) out.push_back(v);
        return out;
      case onnx::TensorProto::INT64:
        for (int64_t v : t.int64_data()) out.push_back(static_cast<double>(v));
        return out;
      default:
        break;
    }
  }
  throw std::invalid_argument("cannot read tensor " + Quoted(t.name()) +
                              ": unsupported data type " +
                              std::to_string(type));
}

Shape DimsOf(const onnx::TensorProto& t) {
  return Shape(t.dims().begin(), t.dims().end());
}

onnx::TensorProto MakeFloatTensor(const std::string& name, const Shape& dims,
                                  const std::vector<float>& values) {
  onnx::TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) tensor.add_dims(d);
  std::string raw;
  raw.reserve(values.size() * sizeof(float));
  for (float v : values) AppendFloat(raw, v);
  tensor.set_raw_data(std::move(raw));
  return tensor;
}

onnx::TensorProto MakeZeroTensor(const std::string& name, const Shape& dims) {
  return MakeFloatTensor(
      name, dims, std::vector<float>(static_cast<size_t>(ElementCount(dims))));
}

std::map<std::string, const onnx::TensorProto*> InitializerIndex(
    const onnx::GraphProto& graph) {
  std::map<std::string, const onnx::TensorProto*> index;
  for (const onnx::TensorProto& t : graph.initializer()) index[t.name()] = &t;
  return index;
}

// ---------------------------------------------------------------------------
// Name bookkeeping -- mirrors bias_correction.py's _all_names/_unique_name,
// which lora.py itself imports and uses for exactly this.
// ---------------------------------------------------------------------------

std::set<std::string> AllNames(const onnx::GraphProto& graph) {
  std::set<std::string> names;
  for (const onnx::TensorProto& t : graph.initializer()) names.insert(t.name());
  for (const onnx::ValueInfoProto& vi : graph.input()) names.insert(vi.name());
  for (const onnx::ValueInfoProto& vi : graph.output()) names.insert(vi.name());
  for (const onnx::ValueInfoProto& vi : graph.value_info()) {
    names.insert(vi.name());
  }
  for (const onnx::NodeProto& n : graph.node()) {
    if (!n.name().empty()) names.insert(n.name());
    for (const std::string& i : n.input()) {
      if (!i.empty()) names.insert(i);
    }
    for (const std::string& o : n.output()) {
      if (!o.empty()) names.insert(o);
    }
  }
  return names;
}

std::string UniqueName(const std::string& base, std::set<std::string>& taken) {
  std::string name = base;
  int i = 0;
  while (taken.count(name) != 0) {
    ++i;
    name = base + "_" + std::to_string(i);
  }
  taken.insert(name);
  return name;
}

// ---------------------------------------------------------------------------
// Raw NodeProto construction -- InjectLora edits an already-deployed model
// in place, the same style onnxsim.nf4.quantize_weight_only_nf4 and
// lora.py's own inject_lora use (onnx.helper.make_node), not
// qat_graph_builder::GraphBuilder, which exists for one-shot *step graph*
// construction rather than for editing a model meant to be run as-is. See
// lora.py's module docstring.
// ---------------------------------------------------------------------------

onnx::NodeProto MakeNode(const std::string& op_type,
                         const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs) {
  onnx::NodeProto node;
  node.set_op_type(op_type);
  for (const std::string& in : inputs) node.add_input(in);
  for (const std::string& out : outputs) node.add_output(out);
  return node;
}

void AddIntAttr(onnx::NodeProto& node, const std::string& name, int64_t value) {
  onnx::AttributeProto* a = node.add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::INT);
  a->set_i(value);
}

void AddIntsAttr(onnx::NodeProto& node, const std::string& name,
                 const std::vector<int64_t>& values) {
  onnx::AttributeProto* a = node.add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::INTS);
  for (int64_t v : values) a->add_ints(v);
}

// ---------------------------------------------------------------------------
// InjectLora -- C++ port of lora.py's inject_lora/_inject_matmul/
// _inject_gemm/_inject_conv1x1/_insert_after/_attr_ints/_attr_int/
// _scale_initializer.
// ---------------------------------------------------------------------------

// A Kaiming-normal draw stream for A's initialization. See lora_entry.h's
// top comment for why this does not, and does not need to, reproduce
// numpy's default_rng bit-for-bit.
class KaimingRng {
 public:
  explicit KaimingRng(uint64_t seed) : engine_(seed) {}
  float Draw() { return static_cast<float>(dist_(engine_)); }

 private:
  std::mt19937_64 engine_;
  std::normal_distribution<double> dist_{0.0, 1.0};
};

// lora.py's _scale_initializer: a scalar float32 initializer holding
// alpha / rank, appended to `graph` when `has_alpha`. Returns the empty
// string when there is nothing to scale by (mirrors the Python's
// Optional[Tuple[...]] via an empty-name sentinel, cheaper than an
// std::optional round trip for one caller each in three near-identical
// functions below).
std::string ScaleInitializer(onnx::GraphProto* graph, int64_t rank,
                             bool has_alpha, float alpha,
                             std::set<std::string>& taken_names) {
  if (!has_alpha) return "";
  const std::string name = UniqueName("lora_alpha_over_rank", taken_names);
  *graph->add_initializer() =
      MakeFloatTensor(name, {}, {alpha / static_cast<float>(rank)});
  return name;
}

// Splices `new_nodes` into `output` right after `node` was emitted --
// InjectLora's own caller does this by construction (see its own comment)
// rather than by an in-place onnx.GraphProto splice the way lora.py's
// _insert_after does: protobuf's RepeatedPtrField has no cheap "insert in
// the middle" the way a Python list does, and rebuilding the whole node
// list once at the end (as InjectLora does) is both simpler and gives the
// identical final order.
LoraTarget InjectMatMul(onnx::GraphProto* graph, onnx::NodeProto* node,
                        const std::string& w_name, const Shape& w_dims,
                        int64_t rank, bool has_alpha, float alpha,
                        KaimingRng& rng, std::set<std::string>& taken_names,
                        std::vector<onnx::NodeProto>* new_nodes) {
  const int64_t k = w_dims[0];
  const int64_t n = w_dims[1];
  const std::string x_name = node->input(0);
  const std::string orig_output = node->output(0);

  const std::string a_name = UniqueName(w_name + ".lora_A", taken_names);
  std::vector<float> a_value(static_cast<size_t>(k * rank));
  const float inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(k));
  for (float& v : a_value) v = rng.Draw() * inv_sqrt_k;
  *graph->add_initializer() = MakeFloatTensor(a_name, {k, rank}, a_value);

  const std::string b_name = UniqueName(w_name + ".lora_B", taken_names);
  *graph->add_initializer() = MakeZeroTensor(b_name, {rank, n});

  const std::string base_out =
      UniqueName(w_name + ".lora_base_out", taken_names);
  node->set_output(0, base_out);

  const std::string a_out = UniqueName(w_name + ".lora_a_out", taken_names);
  new_nodes->push_back(MakeNode("MatMul", {x_name, a_name}, {a_out}));
  const std::string ab_out = UniqueName(w_name + ".lora_ab_out", taken_names);
  new_nodes->push_back(MakeNode("MatMul", {a_out, b_name}, {ab_out}));
  std::string delta = ab_out;

  const std::string scale_name =
      ScaleInitializer(graph, rank, has_alpha, alpha, taken_names);
  if (!scale_name.empty()) {
    const std::string scaled_out =
        UniqueName(w_name + ".lora_scaled", taken_names);
    new_nodes->push_back(MakeNode("Mul", {ab_out, scale_name}, {scaled_out}));
    delta = scaled_out;
  }

  new_nodes->push_back(MakeNode("Add", {base_out, delta}, {orig_output}));

  LoraTarget target;
  target.weight_name = w_name;
  target.node_output = orig_output;
  target.op_type = "MatMul";
  target.lora_a_name = a_name;
  target.lora_b_name = b_name;
  target.rank = rank;
  target.has_alpha = has_alpha;
  target.alpha = alpha;
  return target;
}

LoraTarget InjectGemm(onnx::GraphProto* graph, onnx::NodeProto* node,
                      const std::string& w_name, const Shape& w_dims,
                      int64_t rank, bool has_alpha, float alpha,
                      KaimingRng& rng, std::set<std::string>& taken_names,
                      std::vector<onnx::NodeProto>* new_nodes) {
  const int64_t trans_a = AttrInt(*node, "transA", 0);
  const int64_t trans_b = AttrInt(*node, "transB", 0);
  int64_t k, n;
  if (trans_b != 0) {
    n = w_dims[0];
    k = w_dims[1];
  } else {
    k = w_dims[0];
    n = w_dims[1];
  }

  const std::string x_name = node->input(0);
  const std::string orig_output = node->output(0);
  std::string branch_input = x_name;
  if (trans_a != 0) {
    branch_input = UniqueName(w_name + ".lora_xT", taken_names);
    onnx::NodeProto transpose = MakeNode("Transpose", {x_name}, {branch_input});
    AddIntsAttr(transpose, "perm", {1, 0});
    new_nodes->push_back(std::move(transpose));
  }

  const std::string a_name = UniqueName(w_name + ".lora_A", taken_names);
  std::vector<float> a_value(static_cast<size_t>(k * rank));
  const float inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(k));
  for (float& v : a_value) v = rng.Draw() * inv_sqrt_k;
  *graph->add_initializer() = MakeFloatTensor(a_name, {k, rank}, a_value);

  const std::string b_name = UniqueName(w_name + ".lora_B", taken_names);
  *graph->add_initializer() = MakeZeroTensor(b_name, {rank, n});

  const std::string base_out =
      UniqueName(w_name + ".lora_base_out", taken_names);
  node->set_output(0, base_out);

  const std::string a_out = UniqueName(w_name + ".lora_a_out", taken_names);
  new_nodes->push_back(MakeNode("MatMul", {branch_input, a_name}, {a_out}));
  const std::string ab_out = UniqueName(w_name + ".lora_ab_out", taken_names);
  new_nodes->push_back(MakeNode("MatMul", {a_out, b_name}, {ab_out}));
  std::string delta = ab_out;

  const std::string scale_name =
      ScaleInitializer(graph, rank, has_alpha, alpha, taken_names);
  if (!scale_name.empty()) {
    const std::string scaled_out =
        UniqueName(w_name + ".lora_scaled", taken_names);
    new_nodes->push_back(MakeNode("Mul", {ab_out, scale_name}, {scaled_out}));
    delta = scaled_out;
  }

  new_nodes->push_back(MakeNode("Add", {base_out, delta}, {orig_output}));

  LoraTarget target;
  target.weight_name = w_name;
  target.node_output = orig_output;
  target.op_type = "Gemm";
  target.lora_a_name = a_name;
  target.lora_b_name = b_name;
  target.rank = rank;
  target.has_alpha = has_alpha;
  target.alpha = alpha;
  return target;
}

LoraTarget InjectConv1x1(onnx::GraphProto* graph, onnx::NodeProto* node,
                         const std::string& w_name, const Shape& w_dims,
                         int64_t rank, bool has_alpha, float alpha,
                         KaimingRng& rng, std::set<std::string>& taken_names,
                         std::vector<onnx::NodeProto>* new_nodes) {
  const int64_t out_ch = w_dims[0];
  const int64_t in_ch = w_dims[1];
  const std::string x_name = node->input(0);
  const std::string orig_output = node->output(0);

  const Shape strides = AttrInts(*node, "strides", {1, 1});
  const Shape pads = AttrInts(*node, "pads", {0, 0, 0, 0});
  const Shape dilations = AttrInts(*node, "dilations", {1, 1});

  const std::string a_name = UniqueName(w_name + ".lora_A", taken_names);
  std::vector<float> a_value(static_cast<size_t>(rank * in_ch));
  const float inv_sqrt_in = 1.0f / std::sqrt(static_cast<float>(in_ch));
  for (float& v : a_value) v = rng.Draw() * inv_sqrt_in;
  *graph->add_initializer() =
      MakeFloatTensor(a_name, {rank, in_ch, 1, 1}, a_value);

  const std::string b_name = UniqueName(w_name + ".lora_B", taken_names);
  *graph->add_initializer() = MakeZeroTensor(b_name, {out_ch, rank, 1, 1});

  const std::string base_out =
      UniqueName(w_name + ".lora_base_out", taken_names);
  node->set_output(0, base_out);

  const std::string a_out = UniqueName(w_name + ".lora_a_out", taken_names);
  onnx::NodeProto a_node = MakeNode("Conv", {x_name, a_name}, {a_out});
  AddIntsAttr(a_node, "kernel_shape", {1, 1});
  AddIntsAttr(a_node, "strides", strides);
  AddIntsAttr(a_node, "pads", pads);
  AddIntsAttr(a_node, "dilations", dilations);
  AddIntAttr(a_node, "group", 1);
  new_nodes->push_back(std::move(a_node));

  const std::string ab_out = UniqueName(w_name + ".lora_ab_out", taken_names);
  onnx::NodeProto ab_node = MakeNode("Conv", {a_out, b_name}, {ab_out});
  // No strides/pads/dilations here, matching lora.py's own second branch
  // conv exactly: all spatial downsampling already happened in the first,
  // so this one is left at ONNX's plain 1x1/stride-1/no-pad defaults rather
  // than spelling them out.
  AddIntsAttr(ab_node, "kernel_shape", {1, 1});
  AddIntAttr(ab_node, "group", 1);
  new_nodes->push_back(std::move(ab_node));
  std::string delta = ab_out;

  const std::string scale_name =
      ScaleInitializer(graph, rank, has_alpha, alpha, taken_names);
  if (!scale_name.empty()) {
    const std::string scaled_out =
        UniqueName(w_name + ".lora_scaled", taken_names);
    new_nodes->push_back(MakeNode("Mul", {ab_out, scale_name}, {scaled_out}));
    delta = scaled_out;
  }

  new_nodes->push_back(MakeNode("Add", {base_out, delta}, {orig_output}));

  LoraTarget target;
  target.weight_name = w_name;
  target.node_output = orig_output;
  target.op_type = "Conv";
  target.lora_a_name = a_name;
  target.lora_b_name = b_name;
  target.rank = rank;
  target.has_alpha = has_alpha;
  target.alpha = alpha;
  return target;
}

// ---------------------------------------------------------------------------
// FoldFrozenPrefixes -- C++ port of lora.py's _fold_frozen_prefixes, backed
// by a small constant evaluator instead of a real ONNX backend. See this
// file's top comment for why, and EvalNode below for exactly what it
// covers.
// ---------------------------------------------------------------------------

// A folded value: dims plus a flat, row-major, double-valued buffer -- the
// same "decode everything to double" convention TensorValues above uses,
// just carried through intermediate node outputs too instead of only
// initializers.
struct EvalTensor {
  Shape dims;
  int32_t dtype = onnx::TensorProto::FLOAT;
  std::vector<double> data;
};

int64_t Numel(const Shape& dims) { return ElementCount(dims); }

EvalTensor TensorFromInitializer(const onnx::TensorProto& t) {
  EvalTensor et;
  et.dims = DimsOf(t);
  et.dtype = t.data_type();
  et.data = TensorValues(t);
  return et;
}

onnx::TensorProto ToInitializer(const std::string& name, const EvalTensor& t) {
  onnx::TensorProto out;
  out.set_name(name);
  out.set_data_type(t.dtype);
  for (int64_t d : t.dims) out.add_dims(d);
  std::string raw;
  switch (t.dtype) {
    case onnx::TensorProto::FLOAT:
      raw.reserve(t.data.size() * 4);
      for (double v : t.data) AppendFloat(raw, static_cast<float>(v));
      break;
    case onnx::TensorProto::DOUBLE:
      raw.reserve(t.data.size() * 8);
      for (double v : t.data) {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        AppendLE(raw, bits, 8);
      }
      break;
    case onnx::TensorProto::INT64:
      raw.reserve(t.data.size() * 8);
      for (double v : t.data) {
        AppendLE(raw,
                 static_cast<uint64_t>(static_cast<int64_t>(std::llround(v))),
                 8);
      }
      break;
    case onnx::TensorProto::INT32:
      raw.reserve(t.data.size() * 4);
      for (double v : t.data) {
        AppendLE(raw,
                 static_cast<uint32_t>(static_cast<int32_t>(std::llround(v))),
                 4);
      }
      break;
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::BOOL:
      raw.reserve(t.data.size());
      for (double v : t.data) {
        raw.push_back(
            static_cast<char>(static_cast<int64_t>(std::llround(v)) & 0xff));
      }
      break;
    default:
      throw std::invalid_argument(
          "lora fold: cannot materialize a folded tensor of ONNX dtype " +
          std::to_string(t.dtype) +
          " -- this port's constant folder writes "
          "back FLOAT, DOUBLE, INT64, INT32, INT8, UINT8 and BOOL only");
  }
  out.set_raw_data(std::move(raw));
  return out;
}

std::vector<int64_t> IntsFromTensor(const EvalTensor& t) {
  std::vector<int64_t> out;
  out.reserve(t.data.size());
  for (double v : t.data) out.push_back(static_cast<int64_t>(std::llround(v)));
  return out;
}

Shape BroadcastDims(const Shape& a, const Shape& b) {
  const size_t rank = std::max(a.size(), b.size());
  Shape out(rank);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      throw std::invalid_argument(
          "lora fold: cannot broadcast a constant node's operand shapes");
    }
    out[i] = std::max(da, db);
  }
  return out;
}

// The flat index into `operand` (shape `operand_dims`, row-major) that
// numpy-style broadcasting selects for output position `out_multi` (a
// multi-index over `out_dims`, which `operand_dims` broadcasts against).
int64_t BroadcastOperandIndex(const Shape& out_dims,
                              const std::vector<int64_t>& out_multi,
                              const Shape& operand_dims) {
  const size_t rank = out_dims.size();
  const size_t pad = rank - operand_dims.size();
  std::vector<int64_t> op_stride(operand_dims.size());
  int64_t stride = 1;
  for (size_t i = operand_dims.size(); i-- > 0;) {
    op_stride[i] = stride;
    stride *= operand_dims[i];
  }
  int64_t flat = 0;
  for (size_t i = pad; i < rank; ++i) {
    const size_t oi = i - pad;
    const int64_t idx = (operand_dims[oi] == 1) ? 0 : out_multi[i];
    flat += idx * op_stride[oi];
  }
  return flat;
}

std::vector<int64_t> UnflattenRowMajor(int64_t flat, const Shape& dims) {
  std::vector<int64_t> multi(dims.size());
  for (size_t i = dims.size(); i-- > 0;) {
    const int64_t d = dims[i] == 0 ? 1 : dims[i];
    multi[i] = flat % d;
    flat /= d;
  }
  return multi;
}

EvalTensor EvalElementwiseBinary(const std::string& op, const EvalTensor& a,
                                 const EvalTensor& b) {
  if (a.dtype != b.dtype) {
    throw std::invalid_argument("lora fold: " + op +
                                " between tensors of different ONNX dtypes "
                                "is not supported by this port's folder");
  }
  const Shape out_dims = BroadcastDims(a.dims, b.dims);
  const int64_t n = Numel(out_dims);
  EvalTensor out;
  out.dims = out_dims;
  out.dtype = a.dtype;
  out.data.resize(static_cast<size_t>(n));
  for (int64_t flat = 0; flat < n; ++flat) {
    const std::vector<int64_t> multi = UnflattenRowMajor(flat, out_dims);
    const double x = a.data[static_cast<size_t>(
        BroadcastOperandIndex(out_dims, multi, a.dims))];
    const double y = b.data[static_cast<size_t>(
        BroadcastOperandIndex(out_dims, multi, b.dims))];
    double v;
    if (op == "Add") {
      v = x + y;
    } else if (op == "Sub") {
      v = x - y;
    } else if (op == "Mul") {
      v = x * y;
    } else {
      v = x / y;  // Div
    }
    out.data[static_cast<size_t>(flat)] = v;
  }
  return out;
}

EvalTensor EvalCast(const EvalTensor& in, int64_t to) {
  // Data-preserving numeric reinterpretation, not a full ONNX Cast: the only
  // casts a frozen weight's dequant chain ever needs (an integer code
  // widened to the index type Gather takes) are exact, so the double-valued
  // data is carried through unchanged rather than emulating every dtype
  // pair's rounding/truncation rule. See EvalNode's own comment for the
  // scope this is part of.
  EvalTensor out = in;
  out.dtype = static_cast<int32_t>(to);
  return out;
}

EvalTensor EvalReshape(const EvalTensor& data,
                       const std::vector<int64_t>& shape) {
  Shape resolved(shape.size());
  int64_t infer_axis = -1;
  int64_t known_product = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t d = shape[i];
    if (d == -1) {
      infer_axis = static_cast<int64_t>(i);
      resolved[i] = -1;
    } else if (d == 0) {
      if (i >= data.dims.size()) {
        throw std::invalid_argument(
            "lora fold: Reshape's 0 (copy input dim) has no corresponding "
            "input dimension");
      }
      resolved[i] = data.dims[i];
      known_product *= resolved[i];
    } else {
      resolved[i] = d;
      known_product *= d;
    }
  }
  if (infer_axis >= 0) {
    const int64_t total = Numel(data.dims);
    if (known_product == 0 || total % known_product != 0) {
      throw std::invalid_argument(
          "lora fold: cannot infer Reshape's -1 dimension");
    }
    resolved[static_cast<size_t>(infer_axis)] = total / known_product;
  }
  if (Numel(resolved) != Numel(data.dims)) {
    throw std::invalid_argument(
        "lora fold: Reshape target element count does not match the input");
  }
  EvalTensor out = data;
  out.dims = resolved;
  return out;
}

EvalTensor EvalTranspose(const EvalTensor& data, std::vector<int64_t> perm) {
  const size_t rank = data.dims.size();
  if (perm.empty()) {
    perm.resize(rank);
    for (size_t i = 0; i < rank; ++i)
      perm[i] = static_cast<int64_t>(rank - 1 - i);
  }
  Shape out_dims(rank);
  for (size_t i = 0; i < rank; ++i)
    out_dims[i] = data.dims[static_cast<size_t>(perm[i])];
  std::vector<int64_t> in_stride(rank);
  int64_t stride = 1;
  for (size_t i = rank; i-- > 0;) {
    in_stride[i] = stride;
    stride *= data.dims[i];
  }
  const int64_t n = Numel(data.dims);
  EvalTensor out;
  out.dims = out_dims;
  out.dtype = data.dtype;
  out.data.resize(static_cast<size_t>(n));
  for (int64_t flat = 0; flat < n; ++flat) {
    const std::vector<int64_t> multi = UnflattenRowMajor(flat, out_dims);
    int64_t src = 0;
    for (size_t i = 0; i < rank; ++i) {
      src += multi[i] * in_stride[static_cast<size_t>(perm[i])];
    }
    out.data[static_cast<size_t>(flat)] = data.data[static_cast<size_t>(src)];
  }
  return out;
}

EvalTensor EvalGather(const EvalTensor& data, const EvalTensor& indices,
                      int64_t axis) {
  const int64_t rank = static_cast<int64_t>(data.dims.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    throw std::invalid_argument("lora fold: Gather axis out of range");
  }
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= data.dims[static_cast<size_t>(i)];
  const int64_t axis_size = data.dims[static_cast<size_t>(axis)];
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < rank; ++i)
    inner *= data.dims[static_cast<size_t>(i)];

  Shape out_dims;
  for (int64_t i = 0; i < axis; ++i)
    out_dims.push_back(data.dims[static_cast<size_t>(i)]);
  for (int64_t d : indices.dims) out_dims.push_back(d);
  for (int64_t i = axis + 1; i < rank; ++i) {
    out_dims.push_back(data.dims[static_cast<size_t>(i)]);
  }

  EvalTensor out;
  out.dims = out_dims;
  out.dtype = data.dtype;
  out.data.reserve(static_cast<size_t>(outer) * indices.data.size() *
                   static_cast<size_t>(inner));
  for (int64_t o = 0; o < outer; ++o) {
    for (double idx_d : indices.data) {
      int64_t idx = static_cast<int64_t>(std::llround(idx_d));
      if (idx < 0) idx += axis_size;
      if (idx < 0 || idx >= axis_size) {
        throw std::invalid_argument("lora fold: Gather index out of range");
      }
      for (int64_t in = 0; in < inner; ++in) {
        out.data.push_back(
            data.data[static_cast<size_t>((o * axis_size + idx) * inner + in)]);
      }
    }
  }
  return out;
}

EvalTensor EvalSqueeze(const EvalTensor& data, std::vector<int64_t> axes) {
  const int64_t rank = static_cast<int64_t>(data.dims.size());
  if (axes.empty()) {
    for (int64_t i = 0; i < rank; ++i) {
      if (data.dims[static_cast<size_t>(i)] == 1) axes.push_back(i);
    }
  }
  std::set<int64_t> norm;
  for (int64_t a : axes) norm.insert(a < 0 ? a + rank : a);
  Shape out_dims;
  for (int64_t i = 0; i < rank; ++i) {
    if (norm.count(i) == 0)
      out_dims.push_back(data.dims[static_cast<size_t>(i)]);
  }
  EvalTensor out = data;
  out.dims = out_dims;
  return out;
}

EvalTensor EvalUnsqueeze(const EvalTensor& data,
                         const std::vector<int64_t>& axes_in) {
  const int64_t out_rank = static_cast<int64_t>(data.dims.size()) +
                           static_cast<int64_t>(axes_in.size());
  std::set<int64_t> norm;
  for (int64_t a : axes_in) norm.insert(a < 0 ? a + out_rank : a);
  Shape out_dims(static_cast<size_t>(out_rank));
  size_t src = 0;
  for (int64_t i = 0; i < out_rank; ++i) {
    if (norm.count(i) != 0) {
      out_dims[static_cast<size_t>(i)] = 1;
    } else {
      out_dims[static_cast<size_t>(i)] = data.dims[src++];
    }
  }
  EvalTensor out = data;
  out.dims = out_dims;
  return out;
}

EvalTensor EvalConcat(const std::vector<const EvalTensor*>& inputs,
                      int64_t axis) {
  if (inputs.empty()) {
    throw std::invalid_argument("lora fold: Concat needs at least one input");
  }
  const int64_t rank = static_cast<int64_t>(inputs[0]->dims.size());
  if (axis < 0) axis += rank;
  Shape out_dims = inputs[0]->dims;
  int64_t total_axis = 0;
  for (const EvalTensor* t : inputs)
    total_axis += t->dims[static_cast<size_t>(axis)];
  out_dims[static_cast<size_t>(axis)] = total_axis;
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= out_dims[static_cast<size_t>(i)];
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < rank; ++i)
    inner *= out_dims[static_cast<size_t>(i)];

  EvalTensor out;
  out.dims = out_dims;
  out.dtype = inputs[0]->dtype;
  out.data.resize(static_cast<size_t>(Numel(out_dims)));
  for (int64_t o = 0; o < outer; ++o) {
    int64_t axis_off = 0;
    for (const EvalTensor* t : inputs) {
      const int64_t asz = t->dims[static_cast<size_t>(axis)];
      for (int64_t a = 0; a < asz; ++a) {
        for (int64_t in = 0; in < inner; ++in) {
          out.data[static_cast<size_t>(
              (o * out_dims[static_cast<size_t>(axis)] + axis_off + a) * inner +
              in)] = t->data[static_cast<size_t>((o * asz + a) * inner + in)];
        }
      }
      axis_off += asz;
    }
  }
  return out;
}

// Evaluates one node of a constant-only subgraph.
//
// This is deliberately narrower than lora.py's own fold, which hands the
// candidate subgraph to a real ONNX backend and so has no such list at all
// -- see this file's top comment for why this port cannot do the same. The
// op set below is exactly onnxsim.nf4.quantize_weight_only_nf4's dequant
// chain (Cast, Gather, Reshape, Mul) plus Identity/Transpose/Squeeze/
// Unsqueeze/Concat/Add/Sub/Div, cheap generalizations that cost little and
// widen what a future quantization scheme's frozen-prefix chain can look
// like without needing this file touched again. A node whose op type is not
// in this list -- even though FoldFrozenPrefixes has already established
// every one of its inputs is constant -- is refused here with a message
// that says so explicitly, distinct from RefuseUnsupported's "no gradient
// rule" refusal below: the two are different failure modes (this one is
// "this port's folder cannot evaluate it", not "graph_grad cannot
// differentiate it") and a caller should be able to tell them apart.
EvalTensor EvalNode(const onnx::NodeProto& node,
                    const std::map<std::string, EvalTensor>& values) {
  auto get = [&](int i) -> const EvalTensor& {
    const auto it = values.find(node.input(i));
    if (it == values.end()) {
      throw std::invalid_argument("lora fold: no value for " +
                                  Quoted(node.input(i)) + " while evaluating " +
                                  Quoted(node.op_type()));
    }
    return it->second;
  };
  const std::string& op = node.op_type();
  if (op == "Identity") return get(0);
  if (op == "Cast") return EvalCast(get(0), AttrInt(node, "to", 1));
  if (op == "Reshape") return EvalReshape(get(0), IntsFromTensor(get(1)));
  if (op == "Transpose")
    return EvalTranspose(get(0), AttrInts(node, "perm", {}));
  if (op == "Gather")
    return EvalGather(get(0), get(1), AttrInt(node, "axis", 0));
  if (op == "Squeeze") {
    const std::vector<int64_t> axes = node.input_size() > 1
                                          ? IntsFromTensor(get(1))
                                          : AttrInts(node, "axes", {});
    return EvalSqueeze(get(0), axes);
  }
  if (op == "Unsqueeze") {
    const std::vector<int64_t> axes = node.input_size() > 1
                                          ? IntsFromTensor(get(1))
                                          : AttrInts(node, "axes", {});
    return EvalUnsqueeze(get(0), axes);
  }
  if (op == "Concat") {
    std::vector<const EvalTensor*> inputs;
    for (int i = 0; i < node.input_size(); ++i) inputs.push_back(&get(i));
    return EvalConcat(inputs, AttrInt(node, "axis", 0));
  }
  if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div") {
    return EvalElementwiseBinary(op, get(0), get(1));
  }
  throw std::invalid_argument(
      "lora fold: this port's constant folder does not evaluate " + Quoted(op) +
      "; it covers Identity, Cast, Reshape, Transpose, Gather, "
      "Squeeze, Unsqueeze, Concat, Add, Sub, Mul and Div -- see EvalNode's "
      "own comment. If this node's whole input closure really is constant, "
      "either widen EvalNode or move the block boundary so graph_grad never "
      "has to differentiate through it.");
}

struct FoldResult {
  std::vector<onnx::NodeProto> kept_nodes;
  std::vector<onnx::TensorProto> extra_initializers;
};

// C++ port of lora.py's _fold_frozen_prefixes. See that function's own
// docstring for the technique; this transcribes it exactly except for the
// evaluator (EvalNode above, not a real ONNX backend).
FoldResult FoldFrozenPrefixes(const std::vector<onnx::NodeProto>& nodes,
                              const onnx::ModelProto& model,
                              const std::set<std::string>& non_foldable_names) {
  const auto initializer_map = InitializerIndex(model.graph());

  std::set<std::string> foldable_outputs;
  for (const auto& entry : initializer_map) {
    if (non_foldable_names.count(entry.first) == 0) {
      foldable_outputs.insert(entry.first);
    }
  }

  std::vector<onnx::NodeProto> folded_nodes;
  std::vector<onnx::NodeProto> kept_nodes;
  for (const onnx::NodeProto& node : nodes) {
    // A node with no inputs at all (e.g. Constant) is never folded here,
    // matching lora.py's own `if node.input and all(...)` -- an empty
    // `node.input` is falsy in Python, so such a node always falls to the
    // `else` branch there too. Nothing in the quantization dequant chains
    // this exists for produces one.
    bool all_const = node.input_size() > 0;
    for (const std::string& name : node.input()) {
      if (!name.empty() && foldable_outputs.count(name) == 0) {
        all_const = false;
        break;
      }
    }
    if (all_const) {
      folded_nodes.push_back(node);
      for (const std::string& out : node.output()) {
        if (!out.empty()) foldable_outputs.insert(out);
      }
    } else {
      kept_nodes.push_back(node);
    }
  }

  FoldResult result;
  if (folded_nodes.empty()) {
    result.kept_nodes = nodes;
    return result;
  }

  std::set<std::string> folded_output_names;
  for (const onnx::NodeProto& n : folded_nodes) {
    for (const std::string& o : n.output()) {
      if (!o.empty()) folded_output_names.insert(o);
    }
  }
  std::set<std::string> boundary;
  for (const onnx::NodeProto& n : kept_nodes) {
    for (const std::string& in : n.input()) {
      if (folded_output_names.count(in) != 0) boundary.insert(in);
    }
  }
  if (boundary.empty()) {
    // Every consumer of the fold was itself folded away too -- nothing a
    // kept node still needs, so there is nothing to materialize.
    result.kept_nodes = kept_nodes;
    return result;
  }

  std::map<std::string, EvalTensor> values;
  for (const onnx::NodeProto& n : folded_nodes) {
    for (const std::string& in : n.input()) {
      if (in.empty() || values.count(in) != 0) continue;
      const auto it = initializer_map.find(in);
      if (it != initializer_map.end())
        values[in] = TensorFromInitializer(*it->second);
    }
  }
  for (const onnx::NodeProto& n : folded_nodes) {
    EvalTensor value = EvalNode(n, values);
    if (n.output_size() > 0 && !n.output(0).empty()) {
      values[n.output(0)] = std::move(value);
    }
  }

  for (const std::string& name : boundary) {
    const auto it = values.find(name);
    if (it == values.end()) {
      throw std::invalid_argument(
          "lora fold: constant folding did not produce a value for " +
          Quoted(name) +
          " -- an earlier node in its producing chain must "
          "have no output, which should not happen for the op set EvalNode "
          "supports");
    }
    result.extra_initializers.push_back(ToInitializer(name, it->second));
  }
  result.kept_nodes = kept_nodes;
  return result;
}

// ---------------------------------------------------------------------------
// The block slice -- mirrors qat_entry.cpp's SliceBlock/RefuseUnsupported
// exactly (see this file's top comment on why it is a copy rather than a
// shared function).
// ---------------------------------------------------------------------------

struct BlockSlice {
  std::vector<onnx::NodeProto> nodes;
  std::vector<std::string> externals;  // sorted
};

BlockSlice SliceBlock(const onnx::GraphProto& graph,
                      const std::string& block_input_name,
                      const std::string& block_output_name) {
  std::set<std::string> initializers;
  for (const onnx::TensorProto& t : graph.initializer()) {
    initializers.insert(t.name());
  }
  std::map<std::string, int> producer;
  for (int index = 0; index < graph.node_size(); ++index) {
    for (const std::string& output : graph.node(index).output()) {
      if (!output.empty()) producer[output] = index;
    }
  }

  if (producer.find(block_output_name) == producer.end()) {
    throw std::invalid_argument(
        "block output " + Quoted(block_output_name) +
        " is not produced by any node in the model; a block must end at a "
        "computed tensor");
  }

  std::set<std::string> downstream{block_input_name};
  std::set<int> forward;
  for (int index = 0; index < graph.node_size(); ++index) {
    const onnx::NodeProto& node = graph.node(index);
    bool depends = false;
    for (const std::string& name : node.input()) {
      if (downstream.count(name) != 0) depends = true;
    }
    if (!depends) continue;
    forward.insert(index);
    for (const std::string& name : node.output()) {
      if (!name.empty()) downstream.insert(name);
    }
  }

  std::set<int> used;
  std::set<std::string> external;
  std::set<std::string> seen;
  std::vector<std::string> stack{block_output_name};
  while (!stack.empty()) {
    const std::string name = stack.back();
    stack.pop_back();
    if (name.empty() || seen.count(name) != 0) continue;
    seen.insert(name);
    if (initializers.count(name) != 0) continue;
    const auto owner = producer.find(name);
    if (owner == producer.end() || forward.count(owner->second) == 0) {
      external.insert(name);
      continue;
    }
    if (used.count(owner->second) != 0) continue;
    used.insert(owner->second);
    for (const std::string& input : graph.node(owner->second).input()) {
      stack.push_back(input);
    }
  }

  BlockSlice slice;
  for (int index = 0; index < graph.node_size(); ++index) {
    if (used.count(index) != 0) slice.nodes.push_back(graph.node(index));
  }
  slice.externals.assign(external.begin(), external.end());
  return slice;
}

void RefuseUnsupported(const std::vector<onnx::NodeProto>& nodes) {
  std::set<std::string> unsupported;
  for (const onnx::NodeProto& node : nodes) {
    if (SupportedOps().count(node.op_type()) == 0) {
      unsupported.insert(node.op_type());
    }
  }
  if (unsupported.empty()) return;
  throw UnsupportedOpError(
      "the block contains " + QuotedList(unsupported) +
      ", which onnxsim.graph_grad cannot differentiate; it differentiates " +
      QuotedList(SupportedOps()) +
      ". Choose block boundaries that exclude those nodes -- or, if every "
      "one of that node's inputs is really constant, check why "
      "FoldFrozenPrefixes did not remove it (EvalNode's own comment names "
      "the op set it can fold).");
}

// ---------------------------------------------------------------------------
// Block discovery -- faithful ports of qat.py's _liveness_cuts and
// _primary_graph_input, the two helpers lora.py's discover_lora_blocks
// calls. See lora_entry.h's DiscoverLoraBlocks for the argument summary and
// qat.py's own docstrings (read in full before touching either function
// below) for the actual liveness argument -- it is not re-derived here.
// ---------------------------------------------------------------------------

// Every index at which `graph` narrows to a single live activation, paired
// with the tensor that survives it -- index -1 meaning "before the first
// node". A faithful port of qat.py::_liveness_cuts; see that function's
// docstring for why this is the whole of boundary discovery. `primary_input`
// is the one graph input kept in the live set (ordinarily
// PrimaryGraphInput(graph)'s result, or "" for a graph with none); every
// other graph input is excluded from liveness entirely, exactly as
// _liveness_cuts documents.
std::vector<std::pair<int, std::string>> LivenessCuts(
    const onnx::GraphProto& graph, const std::string& primary_input) {
  std::set<std::string> initializers;
  for (const onnx::TensorProto& t : graph.initializer()) {
    initializers.insert(t.name());
  }
  std::set<std::string> graph_inputs;
  for (const onnx::ValueInfoProto& input : graph.input()) {
    if (initializers.count(input.name()) == 0) {
      graph_inputs.insert(input.name());
    }
  }
  std::set<std::string> ignored;
  for (const std::string& name : graph_inputs) {
    if (name != primary_input) ignored.insert(name);
  }

  // A tensor is live until its last consumer; a graph output is live past
  // the end of the graph, so it is never dropped before the final gap.
  const int node_count = graph.node_size();
  std::map<std::string, int> last_use;
  for (int index = 0; index < node_count; ++index) {
    for (const std::string& name : graph.node(index).input()) {
      if (!name.empty() && initializers.count(name) == 0 &&
          ignored.count(name) == 0) {
        last_use[name] = index;
      }
    }
  }
  for (const onnx::ValueInfoProto& out : graph.output()) {
    if (!out.name().empty() && initializers.count(out.name()) == 0 &&
        ignored.count(out.name()) == 0) {
      last_use[out.name()] = node_count;
    }
  }
  auto last_use_of = [&last_use](const std::string& name) -> int {
    const auto it = last_use.find(name);
    return it == last_use.end() ? -1 : it->second;
  };

  std::vector<std::pair<int, std::string>> cuts;
  std::set<std::string> live;
  for (const std::string& name : graph_inputs) {
    if (ignored.count(name) == 0 && last_use_of(name) > -1) live.insert(name);
  }
  if (live.size() == 1) cuts.emplace_back(-1, *live.begin());

  for (int index = 0; index < node_count; ++index) {
    for (const std::string& name : graph.node(index).output()) {
      if (!name.empty() && ignored.count(name) == 0 &&
          last_use_of(name) > index) {
        live.insert(name);
      }
    }
    std::set<std::string> still_live;
    for (const std::string& name : live) {
      if (last_use_of(name) > index) still_live.insert(name);
    }
    live = std::move(still_live);
    if (live.size() == 1) cuts.emplace_back(index, *live.begin());
  }
  return cuts;
}

// The graph input the most nodes depend on -- the main activation path,
// empty when `graph` has no non-initializer input. A faithful port of
// qat.py::_primary_graph_input; see that function's docstring for why "reach
// the most nodes" is the right heuristic and why getting it wrong costs
// block granularity, not correctness. Ties go to the earlier graph input.
std::string PrimaryGraphInput(const onnx::GraphProto& graph) {
  std::set<std::string> initializers;
  for (const onnx::TensorProto& t : graph.initializer()) {
    initializers.insert(t.name());
  }
  std::vector<std::string> candidates;
  for (const onnx::ValueInfoProto& input : graph.input()) {
    if (initializers.count(input.name()) == 0) {
      candidates.push_back(input.name());
    }
  }
  if (candidates.empty()) return "";

  std::string best_name = candidates[0];
  int64_t best_reach = -1;
  for (const std::string& name : candidates) {
    std::set<std::string> reached{name};
    int64_t count = 0;
    for (const onnx::NodeProto& node : graph.node()) {
      bool depends = false;
      for (const std::string& in : node.input()) {
        if (!in.empty() && reached.count(in) != 0) {
          depends = true;
          break;
        }
      }
      if (!depends) continue;
      ++count;
      for (const std::string& out : node.output()) {
        if (!out.empty()) reached.insert(out);
      }
    }
    if (count > best_reach) {
      best_name = name;
      best_reach = count;
    }
  }
  return best_name;
}

// ---------------------------------------------------------------------------
// Shapes -- mirrors qat_entry.cpp's StaticDims/FloatShapeInfo/
// InferFloatShapes/ElemTypeOr/SetValueInfo/BlockShapes.
// ---------------------------------------------------------------------------

Shape StaticDims(const onnx::TypeProto& type, bool* ok) {
  *ok = false;
  if (!type.has_tensor_type() || !type.tensor_type().has_shape()) return {};
  Shape dims;
  for (const onnx::TensorShapeProto::Dimension& d :
       type.tensor_type().shape().dim()) {
    if (!d.has_dim_value() || d.dim_value() <= 0) return {};
    dims.push_back(d.dim_value());
  }
  *ok = true;
  return dims;
}

struct ModelShapeInfo {
  ShapeMap shapes;
  ElemTypeMap elem_types;
};

// Every tensor of `injected_model`, shaped (and typed) as if the caller had
// captured `rows` calibration rows -- the same trick qat_entry.cpp's
// InferFloatShapes plays, and for the same reason: nothing here runs the
// model, so a concrete shape for a block-external tensor can only come from
// pinning the graph's own inputs' leading dimension and re-inferring.
ModelShapeInfo InferInjectedModelShapes(const onnx::ModelProto& injected_model,
                                        int64_t rows) {
  onnx::ModelProto model = injected_model;
  std::set<std::string> initializers;
  for (const onnx::TensorProto& t : model.graph().initializer()) {
    initializers.insert(t.name());
  }
  for (onnx::ValueInfoProto& input : *model.mutable_graph()->mutable_input()) {
    if (initializers.count(input.name()) != 0) continue;
    onnx::TypeProto::Tensor* tensor =
        input.mutable_type()->mutable_tensor_type();
    if (!tensor->has_shape() || tensor->shape().dim_size() == 0) continue;
    tensor->mutable_shape()->mutable_dim(0)->set_dim_value(rows);
  }
  model.mutable_graph()->clear_value_info();
  for (onnx::ValueInfoProto& output :
       *model.mutable_graph()->mutable_output()) {
    output.mutable_type()->mutable_tensor_type()->clear_shape();
  }
  try {
    onnx::shape_inference::InferShapes(model);
  } catch (const std::exception&) {
    // Best-effort, matching qat_entry.cpp: a node this build cannot infer
    // just leaves its output out of the map, and SliceShapes' own strict
    // inference below reports it against the block the caller named.
  }

  ModelShapeInfo info;
  auto collect = [&info](const onnx::ValueInfoProto& value) {
    bool ok = false;
    const Shape dims = StaticDims(value.type(), &ok);
    if (ok) info.shapes[value.name()] = dims;
    const int32_t elem_type = value.type().tensor_type().elem_type();
    if (elem_type != onnx::TensorProto::UNDEFINED) {
      info.elem_types[value.name()] = elem_type;
    }
  };
  for (const onnx::ValueInfoProto& v : model.graph().input()) collect(v);
  for (const onnx::ValueInfoProto& v : model.graph().value_info()) collect(v);
  for (const onnx::ValueInfoProto& v : model.graph().output()) collect(v);
  for (const onnx::TensorProto& t : model.graph().initializer()) {
    info.shapes[t.name()] = DimsOf(t);
    info.elem_types[t.name()] = t.data_type();
  }
  return info;
}

int32_t ElemTypeOr(const ElemTypeMap& types, const std::string& name) {
  const auto it = types.find(name);
  return it == types.end() ? onnx::TensorProto::FLOAT : it->second;
}

void SetValueInfo(onnx::ValueInfoProto* vi, const std::string& name,
                  int32_t elem_type, const Shape& dims) {
  vi->set_name(name);
  onnx::TypeProto::Tensor* tensor = vi->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(elem_type);
  onnx::TensorShapeProto* shape = tensor->mutable_shape();
  for (int64_t d : dims) shape->add_dim()->set_dim_value(d);
}

// Static shapes for every tensor the (post-fold) block slice touches --
// what graph_grad::BuildBackward requires of its caller. `shape_source`
// supplies the initializers a sub-model built from `nodes` needs, which is
// the injected model plus FoldFrozenPrefixes' extra initializers when there
// were any -- and, because it is unfiltered by trainable-vs-not, this is
// also where the adapter's own A/B shapes come from: they are ordinary
// initializers of `shape_source` like any other, just ones BuildLoraStepGraph
// separately declares as step-graph *state* rather than a constant. Mirrors
// qat_entry.cpp's BlockShapes.
ShapeMap SliceShapes(const onnx::ModelProto& shape_source,
                     const std::vector<onnx::NodeProto>& nodes,
                     const std::vector<std::pair<std::string, Shape>>& inputs,
                     const ElemTypeMap& elem_types,
                     const std::string& block_output_name,
                     const Shape& block_output_shape) {
  std::set<std::string> used;
  for (const onnx::NodeProto& node : nodes) {
    for (const std::string& name : node.input()) {
      if (!name.empty()) used.insert(name);
    }
  }

  onnx::ModelProto model;
  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("lora_block");
  for (const onnx::NodeProto& node : nodes) *graph->add_node() = node;
  for (const auto& input : inputs) {
    SetValueInfo(graph->add_input(), input.first,
                 ElemTypeOr(elem_types, input.first), input.second);
  }
  SetValueInfo(graph->add_output(), block_output_name, onnx::TensorProto::FLOAT,
               block_output_shape);
  for (const onnx::TensorProto& t : shape_source.graph().initializer()) {
    if (used.count(t.name()) != 0) *graph->add_initializer() = t;
  }
  onnx::OperatorSetIdProto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(kStepGraphOpset);
  model.set_ir_version(kStepGraphIrVersion);

  try {
    onnx::shape_inference::InferShapes(
        model, onnx::OpSchemaRegistry::Instance(),
        onnx::ShapeInferenceOptions(/*check_type=*/false, /*strict_mode=*/1));
  } catch (const std::exception& error) {
    throw std::invalid_argument(
        std::string(
            "cannot statically infer the block's shapes at opset 17: ") +
        error.what());
  }

  ShapeMap shapes;
  auto collect = [&shapes](const onnx::ValueInfoProto& value) {
    bool ok = false;
    const Shape dims = StaticDims(value.type(), &ok);
    if (!ok) {
      throw std::invalid_argument(
          "tensor " + Quoted(value.name()) +
          " in the block has a non-static shape; LoRA training needs every "
          "shape known at build time");
    }
    shapes[value.name()] = dims;
  };
  for (const onnx::ValueInfoProto& v : model.graph().input()) collect(v);
  for (const onnx::ValueInfoProto& v : model.graph().output()) collect(v);
  for (const onnx::ValueInfoProto& v : model.graph().value_info()) collect(v);
  for (const onnx::TensorProto& t : model.graph().initializer()) {
    shapes[t.name()] = DimsOf(t);
  }

  std::set<std::string> missing;
  for (const onnx::NodeProto& node : nodes) {
    for (const std::string& name : node.input()) {
      if (!name.empty() && shapes.count(name) == 0) missing.insert(name);
    }
    for (const std::string& name : node.output()) {
      if (!name.empty() && shapes.count(name) == 0) missing.insert(name);
    }
  }
  if (!missing.empty()) {
    throw std::invalid_argument("shape inference did not produce a shape for " +
                                QuotedList(missing) + " in the block");
  }
  return shapes;
}

const onnx::TensorProto* FindInitializer(const onnx::ModelProto& model,
                                         const std::string& name) {
  for (const onnx::TensorProto& t : model.graph().initializer()) {
    if (t.name() == name) return &t;
  }
  return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<std::string> LoraAdapter::ParameterNames() const {
  std::vector<std::string> names;
  names.reserve(targets.size() * 2);
  for (const LoraTarget& t : targets) {
    names.push_back(t.lora_a_name);
    names.push_back(t.lora_b_name);
  }
  return names;
}

LoraInjectionResult InjectLora(const onnx::ModelProto& model,
                               const InjectLoraOptions& options) {
  LoraInjectionResult result;
  result.model = model;
  onnx::GraphProto* graph = result.model.mutable_graph();

  std::map<std::string, int> initializer_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    initializer_index[graph->initializer(i).name()] = i;
  }
  std::set<std::string> taken_names = AllNames(*graph);
  KaimingRng rng(options.seed);

  const std::set<std::string> target_op_set(options.target_op_types.begin(),
                                            options.target_op_types.end());
  const std::set<std::string> target_name_set(options.target_names.begin(),
                                              options.target_names.end());

  // Python iterates `list(graph.node)` -- a snapshot -- while `_insert_after`
  // splices new nodes into the *live* graph.node right after the one being
  // processed. protobuf's RepeatedPtrField has no cheap "insert in the
  // middle" the way a Python list does, so this builds the equivalent
  // result differently: walk the original nodes once, and for each one
  // eligible for injection, emit it (with its output possibly renamed)
  // immediately followed by the nodes injection produced, into a fresh
  // list -- which is exactly what "insert right after this node" means for
  // an in-order walk that never revisits an original node. The final order
  // is identical either way.
  const std::vector<onnx::NodeProto> original_nodes(graph->node().begin(),
                                                    graph->node().end());
  std::vector<onnx::NodeProto> rebuilt;
  rebuilt.reserve(original_nodes.size() * 2);

  for (onnx::NodeProto node : original_nodes) {
    bool eligible =
        target_op_set.count(node.op_type()) != 0 && node.input_size() >= 2;
    const onnx::TensorProto* w_init = nullptr;
    std::string w_name;
    if (eligible) {
      w_name = node.input(1);
      if (options.restrict_target_names && target_name_set.count(w_name) == 0) {
        eligible = false;
      } else {
        const auto it = initializer_index.find(w_name);
        if (it == initializer_index.end()) {
          eligible = false;
        } else {
          w_init = &graph->initializer(it->second);
          if (w_init->data_type() != onnx::TensorProto::FLOAT) eligible = false;
        }
      }
    }

    if (eligible) {
      const Shape w_dims = DimsOf(*w_init);
      std::vector<onnx::NodeProto> injected;
      LoraTarget target;
      bool matched = true;
      if (node.op_type() == "MatMul") {
        if (w_dims.size() != 2) {
          matched = false;
        } else {
          target = InjectMatMul(graph, &node, w_name, w_dims, options.rank,
                                options.has_alpha, options.alpha, rng,
                                taken_names, &injected);
        }
      } else if (node.op_type() == "Gemm") {
        if (w_dims.size() != 2) {
          matched = false;
        } else {
          target = InjectGemm(graph, &node, w_name, w_dims, options.rank,
                              options.has_alpha, options.alpha, rng,
                              taken_names, &injected);
        }
      } else {  // "Conv"
        if (w_dims.size() != 4) {
          matched = false;
        } else {
          const Shape kernel_shape =
              AttrInts(node, "kernel_shape", {w_dims[2], w_dims[3]});
          const int64_t group = AttrInt(node, "group", 1);
          if (!(kernel_shape.size() == 2 && kernel_shape[0] == 1 &&
                kernel_shape[1] == 1) ||
              group != 1) {
            matched = false;
          } else {
            target = InjectConv1x1(graph, &node, w_name, w_dims, options.rank,
                                   options.has_alpha, options.alpha, rng,
                                   taken_names, &injected);
          }
        }
      }
      if (matched) {
        rebuilt.push_back(node);  // possibly output-renamed by Inject*
        for (onnx::NodeProto& n : injected) rebuilt.push_back(std::move(n));
        result.adapter.targets.push_back(std::move(target));
        continue;
      }
    }
    rebuilt.push_back(std::move(node));
  }

  graph->clear_node();
  for (onnx::NodeProto& n : rebuilt) *graph->add_node() = std::move(n);

  onnx::checker::check_model(result.model);
  return result;
}

std::vector<LoraBlock> DiscoverLoraBlocks(
    const onnx::ModelProto& injected_model, const LoraAdapter& adapter,
    int64_t max_targets_per_block) {
  if (max_targets_per_block < 1) {
    throw std::invalid_argument("max_targets_per_block must be at least 1");
  }

  const onnx::GraphProto& graph = injected_model.graph();
  const std::vector<std::pair<int, std::string>> cuts =
      LivenessCuts(graph, PrimaryGraphInput(graph));
  std::set<std::string> target_outputs;
  for (const LoraTarget& t : adapter.targets)
    target_outputs.insert(t.node_output);

  // Walk consecutive cut pairs exactly as lora.py's discover_lora_blocks
  // does: `start` is the cut a pending block began at (unset only when
  // `cuts` itself is empty, which skips this loop entirely), `count` how
  // many of `adapter`'s own target outputs have accumulated into it since.
  std::vector<std::pair<std::string, std::string>> pairs;
  bool have_start = !cuts.empty();
  std::pair<int, std::string> start;
  if (have_start) start = cuts.front();
  int64_t count = 0;

  for (size_t i = 0; i + 1 < cuts.size(); ++i) {
    const std::pair<int, std::string>& previous = cuts[i];
    const std::pair<int, std::string>& current = cuts[i + 1];
    bool gap = false;
    for (int index = previous.first + 1; index <= current.first; ++index) {
      if (SupportedOps().count(graph.node(index).op_type()) == 0) {
        gap = true;
        break;
      }
    }
    if (gap) {
      // A gap. Close whatever was pending before it and reopen after.
      if (have_start && count > 0 && start.first < previous.first) {
        pairs.emplace_back(start.second, previous.second);
      }
      start = current;
      have_start = true;
      count = 0;
      continue;
    }
    if (!have_start) {
      start = previous;
      have_start = true;
    }
    for (int index = previous.first + 1; index <= current.first; ++index) {
      for (const std::string& out : graph.node(index).output()) {
        if (target_outputs.count(out) != 0) ++count;
      }
    }
    if (count >= max_targets_per_block) {
      pairs.emplace_back(start.second, current.second);
      start = current;
      count = 0;
    }
  }
  if (have_start && count > 0 && !cuts.empty() &&
      start.first < cuts.back().first) {
    pairs.emplace_back(start.second, cuts.back().second);
  }

  std::vector<LoraBlock> blocks;
  for (const auto& pair : pairs) {
    BlockSlice slice;
    try {
      slice = SliceBlock(graph, pair.first, pair.second);
    } catch (const std::invalid_argument&) {
      // Defensive, matching discover_lora_blocks: the span construction
      // above already guarantees a non-empty, block_output_name-producing
      // slice.
      continue;
    }
    std::set<std::string> block_outputs;
    for (const onnx::NodeProto& node : slice.nodes) {
      for (const std::string& out : node.output()) {
        if (!out.empty()) block_outputs.insert(out);
      }
    }
    std::vector<std::string> block_targets;
    for (const LoraTarget& t : adapter.targets) {
      if (block_outputs.count(t.node_output) != 0) {
        block_targets.push_back(t.node_output);
      }
    }
    // Structurally unreachable given how `pairs` was built above -- every
    // pair spans at least one node whose output is a target -- but kept for
    // symmetry with discover_lora_blocks's own defensive check.
    if (block_targets.empty()) continue;

    std::set<std::string> op_type_set;
    for (const onnx::NodeProto& node : slice.nodes) {
      op_type_set.insert(node.op_type());
    }

    LoraBlock block;
    block.input_name = pair.first;
    block.output_name = pair.second;
    block.target_outputs = std::move(block_targets);
    block.external_inputs = slice.externals;  // already sorted
    block.op_types.assign(op_type_set.begin(), op_type_set.end());
    block.num_nodes = static_cast<int64_t>(slice.nodes.size());
    blocks.push_back(std::move(block));
  }
  return blocks;
}

// A finding worth recording here rather than only in a commit message:
// FoldFrozenPrefixes, called a few lines below, never actually has
// anything to fold, given `nodes` fresh out of SliceBlock -- not just in
// the models this file's own tests build, but *structurally*, for any
// model. SliceBlock puts a node in its `nodes` result only when the node
// is reachable from `block_input_name` through a chain of node inputs
// (its "forward" set); a node whose *entire* input closure is constants
// can, by construction, never have any input in that reachable set, so it
// can never be forward, so it can never be in `nodes`. It is instead
// discovered from the *backward* walk as an "external" -- exactly the
// role `block_input_name` itself plays -- and is captured and bound as an
// ordinary step-graph constant. This was verified against lora.py
// directly (`onnxsim.qat._slice_block` on an `apply_qlora`-composed
// model): the NF4 dequant chain's output ends up in `externals`, not
// `nodes`, and `onnxsim.lora._fold_frozen_prefixes` returns its input
// unchanged. So it is `_slice_block`'s own external-capture path -- not
// `_fold_frozen_prefixes` -- that is what actually lets a QLoRA-quantized
// base weight's non-differentiable `Cast` never reach `build_backward` in
// the first place; the module docstring's framing of the fold as what
// "makes QLoRA trainable at all" over-credits it, at least for every
// composition `apply_qlora` itself can produce. FoldFrozenPrefixes is kept,
// faithfully ported, on the same call site's own reasoning: it costs
// nothing to call when it is a no-op, and stays correct should
// SliceBlock's reachability rule, or a future caller's own hand-assembled
// `nodes`, ever make it live. Because that means it has no coverage
// through this header's own public entry points, LoraFoldForTesting below
// exposes it directly for lora_entry_test.cpp -- exactly the reason
// graph_grad.h exposes BuildBackwardWithTemplatedRules for
// graph_grad_templates_test.cpp.
LoraStepPlan BuildLoraStepGraph(const onnx::ModelProto& injected_model,
                                const LoraAdapter& adapter,
                                const std::string& block_input_name,
                                const std::string& block_output_name,
                                int64_t num_rows, const LoraOptions& options) {
  if (adapter.targets.empty()) {
    throw std::invalid_argument("adapter has no injected targets to train");
  }
  if (num_rows < 1) {
    throw std::invalid_argument("num_rows must be at least 1, got " +
                                std::to_string(num_rows));
  }
  if (options.batch_size < 0) {
    throw std::invalid_argument("batch_size must be at least 1, got " +
                                std::to_string(options.batch_size));
  }

  const std::vector<std::string> param_names = adapter.ParameterNames();
  const std::set<std::string> param_set(param_names.begin(), param_names.end());

  // --- slice + fold --------------------------------------------------------
  const BlockSlice slice =
      SliceBlock(injected_model.graph(), block_input_name, block_output_name);
  if (slice.nodes.empty()) {
    throw std::invalid_argument("no nodes lie between " +
                                Quoted(block_input_name) + " and " +
                                Quoted(block_output_name));
  }
  // A documented finding, not a guess: FoldFrozenPrefixes never actually
  // has anything to fold here, and provably can't -- see this function's
  // own top comment (BuildLoraStepGraph, below the includes) for why, and
  // lora_fold_test.cpp / LoraFoldForTesting for how this port's evaluator
  // is exercised directly instead. `slice.nodes` always plays the role of
  // `folded.kept_nodes` in practice; this call is kept anyway, both because
  // it costs nothing when it is a no-op and because SliceBlock changing
  // its own reachability rule some day (or a future caller assembling
  // `nodes` some other way) would make it live again without this file
  // needing to change.
  const FoldResult folded =
      FoldFrozenPrefixes(slice.nodes, injected_model, param_set);
  RefuseUnsupported(folded.kept_nodes);

  onnx::ModelProto shape_source = injected_model;
  for (const onnx::TensorProto& t : folded.extra_initializers) {
    *shape_source.mutable_graph()->add_initializer() = t;
  }

  // --- shapes for externals + teacher --------------------------------------
  const ModelShapeInfo model_info =
      InferInjectedModelShapes(injected_model, num_rows);
  const ShapeMap& full_shapes = model_info.shapes;
  const ElemTypeMap& elem_types = model_info.elem_types;
  auto shape_of = [&full_shapes](const std::string& name) -> const Shape& {
    const auto it = full_shapes.find(name);
    if (it == full_shapes.end()) {
      throw std::invalid_argument(
          "cannot statically infer the shape of " + Quoted(name) +
          " from the model; LoRA training needs every captured tensor's "
          "shape known at build time");
    }
    return it->second;
  };
  std::vector<std::pair<std::string, Shape>> externals;
  for (const std::string& name : slice.externals) {
    externals.emplace_back(name, shape_of(name));
  }
  const Shape teacher_shape = shape_of(block_output_name);

  // --- minibatch plan (mirrors qat.py's _plan_minibatch / BuildQatStepGraph)
  const bool minibatch =
      options.batch_size > 0 && options.batch_size < num_rows;
  if (minibatch) {
    std::set<int64_t> leading;
    for (const auto& external : externals) {
      leading.insert(external.second.empty() ? 0 : external.second[0]);
    }
    leading.insert(teacher_shape.empty() ? 0 : teacher_shape[0]);
    if (leading.size() != 1 || *leading.begin() != num_rows) {
      throw std::invalid_argument(
          "minibatching slices every captured tensor on axis 0 with one "
          "shared index, so they must all have num_rows rows; leave "
          "batch_size unset to train this block full-batch.");
    }
  }
  const int64_t rows_per_step = minibatch ? options.batch_size : num_rows;

  // Pins a captured tensor's leading dimension to this step's row count --
  // safe, and only ever applied, once minibatching is confirmed: the
  // eligibility check just above already required *every* external (and
  // the teacher) to carry exactly `num_rows` on axis 0, which is what
  // makes "axis 0 is the row axis" true for all of them here. Left as the
  // identity otherwise, unlike qat_entry.cpp's own with_rows (which QAT can
  // apply unconditionally: every QAT external is an activation qat.py's own
  // _capture and _plan_minibatch already assume is row-shaped). LoRA's
  // externals are not all guaranteed to be: onnxsim.nf4's dequant chain
  // output is exactly such an external (see BuildLoraStepGraph's own
  // top-of-file comment on SliceBlock's external-capture path) and its
  // shape has nothing to do with the calibration row count -- pinning it
  // unconditionally the way this once did corrupts a perfectly good shape
  // into a wrong one, which is what TheFrozenNf4DequantChainIsCaptured...
  // test in lora_entry_test.cpp caught (a MatMul shape-inference failure
  // one step further down, in SliceShapes).
  auto with_rows = [rows_per_step, minibatch](Shape dims) {
    if (minibatch && !dims.empty()) dims[0] = rows_per_step;
    return dims;
  };
  std::vector<std::pair<std::string, Shape>> step_inputs;
  for (const auto& external : externals) {
    step_inputs.emplace_back(external.first, with_rows(external.second));
  }
  const Shape step_teacher_shape = with_rows(teacher_shape);
  const ShapeMap shapes =
      SliceShapes(shape_source, folded.kept_nodes, step_inputs, elem_types,
                  block_output_name, step_teacher_shape);

  // --- the block's own untrained constants ---------------------------------
  std::set<std::string> used;
  for (const onnx::NodeProto& node : folded.kept_nodes) {
    for (const std::string& name : node.input()) {
      if (!name.empty()) used.insert(name);
    }
  }
  GraphBuilder b(kPrefix);
  for (const onnx::TensorProto& t : shape_source.graph().initializer()) {
    if (used.count(t.name()) != 0 && param_set.count(t.name()) == 0) {
      b.initializer().push_back(t);
    }
  }

  // 0. The minibatch, if there is one -- identical shape to
  //    BuildQatStepGraph's own section 0 (see that file's comments).
  const std::string teacher = std::string(kPrefix) + "teacher";
  const std::string rows_input = std::string(kPrefix) + "rows";
  std::vector<StepGraphSpec::NamedShape> constants;
  std::vector<LoraCapture> captures;
  if (!minibatch) {
    for (const auto& external : externals) {
      const int32_t elem_type = ElemTypeOr(elem_types, external.first);
      constants.push_back({external.first, external.second, elem_type});
      captures.push_back({external.first, external.first, external.second,
                          elem_type, /*is_teacher=*/false});
    }
    constants.push_back({teacher, teacher_shape, onnx::TensorProto::FLOAT});
    captures.push_back({teacher, block_output_name, teacher_shape,
                        onnx::TensorProto::FLOAT, /*is_teacher=*/true});
  } else {
    for (const auto& external : externals) {
      const int32_t elem_type = ElemTypeOr(elem_types, external.first);
      const std::string table = std::string(kPrefix) + "all_" + external.first;
      constants.push_back({table, external.second, elem_type});
      captures.push_back({table, external.first, external.second, elem_type,
                          /*is_teacher=*/false});
      b.GatherRowsInto(table, rows_input, external.first);
    }
    const std::string teacher_table = std::string(kPrefix) + "teacher_all";
    constants.push_back(
        {teacher_table, teacher_shape, onnx::TensorProto::FLOAT});
    captures.push_back({teacher_table, block_output_name, teacher_shape,
                        onnx::TensorProto::FLOAT, /*is_teacher=*/true});
    b.GatherRowsInto(teacher_table, rows_input, teacher);
  }
  const Shape block_output_shape = step_teacher_shape;

  // 2. The block itself, node for node as the injected model wrote it -- no
  //    substitution, since the base weight is never a training target and
  //    the adapter's own branch is already exactly what should run.
  for (const onnx::NodeProto& node : folded.kept_nodes)
    b.nodes().push_back(node);

  // 3. The objective: MSE of the block's output against the target, and its
  //    gradient, which seeds the backward pass.
  const std::string diff = b.Sub(block_output_name, teacher);
  const int64_t n_elems = ElementCount(block_output_shape);
  const std::string two_over_n =
      b.Const(static_cast<float>(2.0 / static_cast<double>(n_elems)));
  const std::string dl_dy = b.Mul(diff, two_over_n);

  // 4. The backward pass, restricted to the adapter's own A/B tensors --
  //    the base weight (and, when QLoRA folded one in, its dequant chain's
  //    now-constant output) is untargeted and so stays frozen by
  //    construction, exactly as BuildBackward's own contract promises.
  const std::map<std::string, std::string> grads = BuildBackward(
      b, folded.kept_nodes, shapes, {{block_output_name, dl_dy}}, param_names);

  // 5. One Adam step per adapter tensor.
  const std::string lr = std::string(kPrefix) + "lr";
  std::vector<StepGraphSpec::StateEntry> state;
  std::vector<onnx::TensorProto> initial_state;
  for (const std::string& name : param_names) {
    const auto grad_it = grads.find(name);
    if (grad_it == grads.end()) {
      throw std::invalid_argument(
          "the backward pass produced no gradient for " + Quoted(name));
    }
    const auto shape_it = shapes.find(name);
    if (shape_it == shapes.end()) {
      throw std::invalid_argument("no shape found for adapter tensor " +
                                  Quoted(name));
    }
    const Shape& shape = shape_it->second;
    const std::string m_in = std::string(kPrefix) + "m_" + name;
    const std::string v_in = std::string(kPrefix) + "v_" + name;
    const AdamOutputs step = AdamUpdate(b, name, grad_it->second, m_in, v_in,
                                        lr, "m_correction", "v_correction");
    state.push_back({name, shape, step.param_next});
    state.push_back({m_in, shape, step.m_next});
    state.push_back({v_in, shape, step.v_next});

    const onnx::TensorProto* current = FindInitializer(injected_model, name);
    if (current == nullptr) {
      throw std::invalid_argument("adapter tensor " + Quoted(name) +
                                  " is not an initializer of injected_model");
    }
    std::vector<float> values;
    values.reserve(static_cast<size_t>(ElementCount(shape)));
    for (double v : TensorValues(*current))
      values.push_back(static_cast<float>(v));
    initial_state.push_back(MakeFloatTensor(name, shape, values));
    initial_state.push_back(MakeZeroTensor(m_in, shape));
    initial_state.push_back(MakeZeroTensor(v_in, shape));
  }

  const std::vector<std::string> scalars{lr, "m_correction", "v_correction"};

  StepGraphSpec spec;
  spec.constants = constants;
  spec.state = state;
  spec.scalars = scalars;
  if (minibatch) {
    spec.per_step.push_back({rows_input,
                             {options.batch_size},
                             static_cast<int32_t>(onnx::TensorProto::INT64)});
  }
  spec.loss_output = b.MeanSquare(diff);
  spec.graph_name = "onnxsim_lora_step";
  const StepGraph step = MakeStepGraph(b, spec);

  LoraStepPlan plan;
  plan.step_graph = step.model;
  plan.state = step.state;
  plan.scalars = scalars;
  plan.loss_name = step.loss_name;
  plan.captures = std::move(captures);
  plan.initial_state = std::move(initial_state);
  if (minibatch) {
    plan.row_index_input = rows_input;
    plan.row_index_size = options.batch_size;
  }
  plan.num_rows = num_rows;
  plan.parameters = param_names;
  return plan;
}

onnx::ModelProto WriteBackLoraState(
    const onnx::ModelProto& injected_model, const LoraStepPlan& plan,
    const std::map<std::string, onnx::TensorProto>& final_state) {
  std::map<std::string, onnx::TensorProto> updates;
  for (const std::string& name : plan.parameters) {
    const auto it = final_state.find(name);
    if (it == final_state.end()) {
      throw std::invalid_argument("final_state is missing the state tensor " +
                                  Quoted(name));
    }
    const onnx::TensorProto* current = FindInitializer(injected_model, name);
    if (current == nullptr) {
      throw std::invalid_argument("adapter tensor " + Quoted(name) +
                                  " is not an initializer of injected_model");
    }
    const Shape dims = DimsOf(*current);
    const std::vector<double> raw = TensorValues(it->second);
    if (static_cast<int64_t>(raw.size()) != ElementCount(dims)) {
      throw std::invalid_argument("the trained tensor " + Quoted(name) +
                                  " has " + std::to_string(raw.size()) +
                                  " elements, expected " +
                                  std::to_string(ElementCount(dims)));
    }
    std::vector<float> values;
    values.reserve(raw.size());
    for (double v : raw) values.push_back(static_cast<float>(v));
    updates[name] = MakeFloatTensor(name, dims, values);
  }

  onnx::ModelProto tuned = injected_model;
  for (onnx::TensorProto& initializer :
       *tuned.mutable_graph()->mutable_initializer()) {
    const auto it = updates.find(initializer.name());
    if (it != updates.end()) initializer = it->second;
  }
  return tuned;
}

LoraFoldResult LoraFoldForTesting(
    const std::vector<onnx::NodeProto>& nodes, const onnx::ModelProto& model,
    const std::vector<std::string>& non_foldable_names) {
  const std::set<std::string> non_foldable(non_foldable_names.begin(),
                                           non_foldable_names.end());
  const FoldResult folded = FoldFrozenPrefixes(nodes, model, non_foldable);
  LoraFoldResult out;
  out.kept_nodes = folded.kept_nodes;
  out.extra_initializers = folded.extra_initializers;
  return out;
}
