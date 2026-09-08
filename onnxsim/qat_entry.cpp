/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The graph-building half of onnxsim/qat.py, in C++. See qat_entry.h for what
 * that split is and why only this half crossed over; see qat.py for the
 * technique, the two schemes' normalization, the LSQ scale gradient and the
 * straight-through estimator. None of that is repeated here -- there should be
 * one place to update when a derivation changes, and it is the Python.
 *
 * What *is* here, because it exists only on this side:
 *
 *   1. Every emitting sub-expression is hoisted into a named local, in
 *      emission order. Python evaluates a call's arguments left to right;
 *      C++ leaves that order unspecified, so a nested `b.Mul(g, b.Const(...))`
 *      could number its two names either way round. The names are what the
 *      parity fixtures compare, and a graph that numbers differently is a
 *      different graph. qat_graph_builder.cpp and graph_grad.cpp already work
 *      this way; this file follows them.
 *
 *   2. Shapes come from onnx's C++ shape inference rather than from captured
 *      arrays. qat.py's _block_shapes types the slice's inputs with the
 *      *concrete* shapes of the calibration activations it captured; nothing
 *      here runs a model, so the same concreteness is obtained by pinning the
 *      float graph's inputs to `num_rows` rows and inferring. That is the one
 *      place this port reads the float model for something qat.py read out of
 *      data, and the assumption it makes -- axis 0 is the row axis -- is
 *      exactly the assumption qat.py's own _capture and _plan_minibatch make.
 *
 *   3. The driving loop is absent, so `shuffle`/`batch_seed` are carried in
 *      QatOptions for the caller's own minibatch_indices and are not read
 *      here.
 */
#include "qat_entry.h"

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph_grad.h"
#include "onnx/shape_inference/implementation.h"

namespace {

using Shape = std::vector<int64_t>;
using ShapeMap = std::map<std::string, Shape>;

// quantize_weight_only_int4's symmetric INT4 range, quantize_static's
// per-output-channel INT8 one, and its uint8 activation range -- qat.py's
// _N_MIN/_N_MAX, _INT8_N_MIN/_INT8_N_MAX and _ACT_N_MIN/_ACT_N_MAX.
constexpr float kNMin = -7.0f;
constexpr float kNMax = 7.0f;
constexpr float kInt8NMin = -127.0f;
constexpr float kInt8NMax = 127.0f;
constexpr float kActNMin = 0.0f;
constexpr float kActNMax = 255.0f;

// Every name this module introduces into the step graph starts here, so it
// cannot collide with a tensor name carried over from the float model.
const char kPrefix[] = "qat__";

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

std::string Quoted(const std::string& value) { return "'" + value + "'"; }

// A Python-list-shaped rendering, so a refusal reads the same from either
// implementation. graph_grad.cpp's SupportedOpsList does the same.
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

int64_t ElementCount(const Shape& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// raw_data is little-endian on every host, so these decode/encode by shifting
// bytes rather than memcpy'ing the host's own layout.
float DecodeFloat(const char* bytes) {
  uint32_t bits = 0;
  for (int i = 0; i < 4; ++i) {
    bits |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[i]))
            << (8 * i);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void AppendFloat(std::string& out, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
}

// A numeric tensor's values as doubles, whichever field they are stored in.
// Only the dtypes this module actually reads are covered -- a float weight or
// scale, a uint8 zero-point -- and anything else is refused rather than
// guessed at.
std::vector<double> TensorValues(const onnx::TensorProto& t) {
  std::vector<double> out;
  const int32_t type = t.data_type();
  if (t.has_raw_data()) {
    const std::string& raw = t.raw_data();
    if (type == onnx::TensorProto::FLOAT) {
      for (size_t i = 0; i + 4 <= raw.size(); i += 4) {
        out.push_back(DecodeFloat(raw.data() + i));
      }
      return out;
    }
    if (type == onnx::TensorProto::UINT8) {
      for (char c : raw) {
        out.push_back(static_cast<double>(static_cast<unsigned char>(c)));
      }
      return out;
    }
    if (type == onnx::TensorProto::INT8) {
      for (char c : raw) out.push_back(static_cast<double>(c));
      return out;
    }
  } else {
    if (type == onnx::TensorProto::FLOAT) {
      for (float v : t.float_data()) out.push_back(v);
      return out;
    }
    if (type == onnx::TensorProto::UINT8 || type == onnx::TensorProto::INT8) {
      for (int32_t v : t.int32_data()) out.push_back(v);
      return out;
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

onnx::TensorProto MakeFloatScalar(const std::string& name, float value) {
  return MakeFloatTensor(name, {}, {value});
}

// Host-side twin of GraphBuilder::RoundToNearest, i.e. qat.py's
// _round_half_away: the export must round the trained master weights exactly
// the way the trained forward did.
double RoundHalfAway(double x) {
  const double magnitude = std::floor(std::abs(x) + 0.5);
  if (x > 0.0) return magnitude;
  if (x < 0.0) return -magnitude;
  return 0.0;
}

// ---------------------------------------------------------------------------
// Candidate finding: adaround.py's _find_int4_matmul_candidates and
// adaquant.py's _find_static_qdq_candidates, plus qat.py's _from_int4 /
// _from_static / _find_layers, which normalize the two schemes into one shape.
// ---------------------------------------------------------------------------

// adaround.py's _node_outputs: first output -> node, iterated in graph order.
class NodeIndex {
 public:
  explicit NodeIndex(const onnx::GraphProto& graph) {
    for (const onnx::NodeProto& node : graph.node()) {
      if (node.output_size() == 0) continue;
      const std::string& key = node.output(0);
      if (by_output_.find(key) == by_output_.end()) order_.push_back(key);
      by_output_[key] = &node;
    }
  }

  const std::vector<std::string>& order() const { return order_; }

  const onnx::NodeProto* Get(const std::string& output) const {
    const auto it = by_output_.find(output);
    return it == by_output_.end() ? nullptr : it->second;
  }

 private:
  std::vector<std::string> order_;
  std::map<std::string, const onnx::NodeProto*> by_output_;
};

std::map<std::string, const onnx::TensorProto*> InitializerIndex(
    const onnx::GraphProto& graph) {
  std::map<std::string, const onnx::TensorProto*> index;
  for (const onnx::TensorProto& t : graph.initializer()) index[t.name()] = &t;
  return index;
}

const onnx::TensorProto* Lookup(
    const std::map<std::string, const onnx::TensorProto*>& index,
    const std::string& name) {
  const auto it = index.find(name);
  return it == index.end() ? nullptr : it->second;
}

// One activation quantizer this run may train -- qat.py's _ActQuant. It
// describes an *edge*, not a tensor: quantize_static gives two layers reading
// the same activation two independent quantizers, and training one per tensor
// would be a different (lossier) model than the one that ships.
struct ActQuant {
  bool present = false;
  std::string tensor;
  std::string scale_name;
  std::string zp_name;
  double scale_init = 0.0;
  double zp_init = 0.0;
};

// One quantized MatMul/Gemm this run can train, with the two schemes'
// differences already normalized away -- qat.py's _QuantizedLayer. Both
// schemes are "an integer code array plus a scale that tiles the weight", and
// a per-output-channel scale is just a block-wise scale whose block spans the
// whole reduction axis.
struct QuantizedLayer {
  std::string output_name;
  onnx::NodeProto float_node;
  onnx::TensorProto w_float_init;
  std::string wq_name;
  std::string ws_name;
  // The scale in the normalized 2-D blocked view (which is also the shape of
  // the loop's scale state tensor).
  Shape scale_dims;
  std::vector<float> scale_2d;
  int64_t axis = 0;
  int64_t block_size = 0;
  float n_min = 0.0f;
  float n_max = 0.0f;
  bool packed_int4 = false;
  ActQuant act;
  // Whether the step graph fake-quantizes this layer's weight on the way into
  // the block -- qat.py's _QuantizedLayer.fake_quant. False is FromFloat's
  // third scheme: the block reads the master weight directly, the
  // straight-through estimator has nothing to pass through, and the write-back
  // stores fp32 rather than codes. Every field above describing the quantizer
  // is then unread; see FromFloat for the values they carry instead.
  bool fake_quant = true;
};

// adaround.py's _find_int4_matmul_candidates, already turned into the
// normalized layer by qat.py's _from_int4.
//
// A scale that is not stored 2-D is dropped rather than reshaped. qat.py
// would raise an IndexError on it (it indexes scale.shape[1]); this scheme
// never produces one, and a dropped layer surfaces as _plan_block's loud "no
// quantized layer in this block" refusal rather than as a crash.
std::vector<QuantizedLayer> FindInt4Layers(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model) {
  const NodeIndex q_by_output(quantized_model.graph());
  const NodeIndex f_by_output(float_model.graph());
  const auto q_init = InitializerIndex(quantized_model.graph());
  const auto f_init = InitializerIndex(float_model.graph());

  std::vector<QuantizedLayer> layers;
  for (const std::string& out_name : q_by_output.order()) {
    const onnx::NodeProto* qn = q_by_output.Get(out_name);
    if ((qn->op_type() != "MatMul" && qn->op_type() != "Gemm") ||
        qn->input_size() < 2) {
      continue;
    }
    const onnx::NodeProto* fn = f_by_output.Get(out_name);
    if (fn == nullptr || fn->op_type() != qn->op_type() ||
        fn->input_size() < 2) {
      continue;
    }
    const onnx::TensorProto* w_float = Lookup(f_init, fn->input(1));
    if (w_float == nullptr ||
        w_float->data_type() != onnx::TensorProto::FLOAT ||
        w_float->dims_size() != 2) {
      continue;
    }
    const onnx::NodeProto* dq = q_by_output.Get(qn->input(1));
    if (dq == nullptr || dq->op_type() != "DequantizeLinear" ||
        dq->input_size() < 2) {
      continue;
    }
    const onnx::TensorProto* wq = Lookup(q_init, dq->input(0));
    const onnx::TensorProto* ws = Lookup(q_init, dq->input(1));
    if (wq == nullptr || ws == nullptr ||
        wq->data_type() != onnx::TensorProto::INT4 ||
        DimsOf(*wq) != DimsOf(*w_float)) {
      continue;
    }
    const int64_t axis = AttrInt(*dq, "axis", 1);
    const int64_t block_size = AttrInt(*dq, "block_size", 0);
    if (block_size == 0) continue;
    if (ws->dims_size() != 2) continue;

    QuantizedLayer layer;
    layer.output_name = out_name;
    layer.float_node = *fn;
    layer.w_float_init = *w_float;
    layer.wq_name = wq->name();
    layer.ws_name = ws->name();
    layer.scale_dims = DimsOf(*ws);
    for (double v : TensorValues(*ws)) {
      layer.scale_2d.push_back(static_cast<float>(v));
    }
    layer.axis = axis;
    layer.block_size = block_size;
    layer.n_min = kNMin;
    layer.n_max = kNMax;
    layer.packed_int4 = true;
    layers.push_back(std::move(layer));
  }
  return layers;
}

// adaquant.py's _find_static_qdq_candidates, turned into the normalized layer
// by qat.py's _from_static. The shapes _from_static returns None for -- a
// per-channel weight scale that is not 1-D of the output channel's length, an
// activation scale or zero-point that is not a single value -- are dropped
// here for the same reason: a layer this does not recognize is not trained,
// and a block with none left is refused loudly.
std::vector<QuantizedLayer> FindStaticLayers(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model) {
  const NodeIndex q_by_output(quantized_model.graph());
  const NodeIndex f_by_output(float_model.graph());
  const auto q_init = InitializerIndex(quantized_model.graph());
  const auto f_init = InitializerIndex(float_model.graph());

  std::vector<QuantizedLayer> layers;
  for (const std::string& out_name : q_by_output.order()) {
    const onnx::NodeProto* qn = q_by_output.Get(out_name);
    if ((qn->op_type() != "MatMul" && qn->op_type() != "Gemm") ||
        qn->input_size() < 2) {
      continue;
    }
    const onnx::NodeProto* fn = f_by_output.Get(out_name);
    if (fn == nullptr || fn->op_type() != qn->op_type() ||
        fn->input_size() < 2) {
      continue;
    }
    const onnx::TensorProto* w_float = Lookup(f_init, fn->input(1));
    if (w_float == nullptr ||
        w_float->data_type() != onnx::TensorProto::FLOAT ||
        w_float->dims_size() != 2) {
      continue;
    }

    // Weight branch: Wdq = DequantizeLinear(Wq, Ws, axis=...), symmetric.
    const onnx::NodeProto* wdq = q_by_output.Get(qn->input(1));
    if (wdq == nullptr || wdq->op_type() != "DequantizeLinear" ||
        wdq->input_size() != 2) {
      continue;
    }
    const onnx::TensorProto* wq = Lookup(q_init, wdq->input(0));
    const onnx::TensorProto* ws = Lookup(q_init, wdq->input(1));
    if (wq == nullptr || ws == nullptr ||
        wq->data_type() != onnx::TensorProto::INT8 ||
        DimsOf(*wq) != DimsOf(*w_float)) {
      continue;
    }
    const int64_t channel_axis = AttrInt(*wdq, "axis", 1);

    // Activation branch: Xdq = DequantizeLinear(Xq, Xs, Xzp) over
    // Xq = QuantizeLinear(X, Xs, Xzp), both sharing the same initializers.
    const onnx::NodeProto* xdq = q_by_output.Get(qn->input(0));
    if (xdq == nullptr || xdq->op_type() != "DequantizeLinear" ||
        xdq->input_size() != 3) {
      continue;
    }
    const onnx::NodeProto* xq = q_by_output.Get(xdq->input(0));
    if (xq == nullptr || xq->op_type() != "QuantizeLinear" ||
        xq->input_size() != 3) {
      continue;
    }
    const std::string x_scale_name = xdq->input(1);
    const std::string x_zp_name = xdq->input(2);
    if (xq->input(1) != x_scale_name || xq->input(2) != x_zp_name) continue;
    const onnx::TensorProto* x_scale = Lookup(q_init, x_scale_name);
    const onnx::TensorProto* x_zp = Lookup(q_init, x_zp_name);
    if (x_scale == nullptr || x_zp == nullptr ||
        x_zp->data_type() != onnx::TensorProto::UINT8) {
      continue;
    }

    // _from_static's own refusals.
    if (channel_axis != 0 && channel_axis != 1) continue;
    const Shape w_dims = DimsOf(*w_float);
    const std::vector<double> scale = TensorValues(*ws);
    if (static_cast<int64_t>(scale.size()) != w_dims[channel_axis]) continue;
    const std::vector<double> x_scale_values = TensorValues(*x_scale);
    const std::vector<double> x_zp_values = TensorValues(*x_zp);
    if (x_scale_values.size() != 1 || x_zp_values.size() != 1) continue;

    // The blocked axis is the *other* one: one scale covers the whole
    // reduction, which is what "per output channel" means.
    const int64_t blocked = 1 - channel_axis;

    QuantizedLayer layer;
    layer.output_name = out_name;
    layer.float_node = *fn;
    layer.w_float_init = *w_float;
    layer.wq_name = wq->name();
    layer.ws_name = ws->name();
    layer.scale_dims = blocked == 0
                           ? Shape{1, static_cast<int64_t>(scale.size())}
                           : Shape{static_cast<int64_t>(scale.size()), 1};
    for (double v : scale) layer.scale_2d.push_back(static_cast<float>(v));
    layer.axis = blocked;
    layer.block_size = w_dims[blocked];
    layer.n_min = kInt8NMin;
    layer.n_max = kInt8NMax;
    layer.packed_int4 = false;
    layer.act.present = true;
    layer.act.tensor = fn->input(0);
    layer.act.scale_name = x_scale_name;
    layer.act.zp_name = x_zp_name;
    layer.act.scale_init = x_scale_values[0];
    layer.act.zp_init = x_zp_values[0];
    layers.push_back(std::move(layer));
  }
  return layers;
}

// A plain, unquantized MatMul/Gemm -- qat.py's _from_float, the scheme that
// makes this a fine-tuner rather than only a quantizer.
//
// Everything downstream is already written against "a master weight the block
// reads and an optimizer updates"; quantization is what sits *between* those
// two, and QuantizedLayer::fake_quant is the switch that removes it. So this
// only says which weight is trainable and where it is written back -- which,
// with no code array in the picture, is the weight initializer itself.
//
// The quantizer fields are unread here, and are filled in with values chosen
// to *break* rather than to look plausible, exactly as the Python's are: a
// block_size of 0 makes the write-back's block indexing degenerate and
// n_min == n_max == 0 makes a fake-quant forward produce all zeros. A
// neutral-looking block_size of 1 with a unit scale would instead round the
// weights to integers and train on quietly, which is the failure mode worth
// ruling out: it is wrong, and it looks like a model that merely trained
// badly.
QuantizedLayer FromFloat(const onnx::NodeProto& node,
                         const onnx::TensorProto& w_init) {
  QuantizedLayer layer;
  layer.output_name = node.output(0);
  layer.float_node = node;
  layer.w_float_init = w_init;
  // There is no separate code array: the tensor trained and the tensor written
  // back are the same one.
  layer.wq_name = w_init.name();
  layer.ws_name = "";
  layer.scale_dims = Shape{1, 1};
  layer.scale_2d = {0.0f};
  layer.axis = 0;
  layer.block_size = 0;
  layer.n_min = 0.0f;
  layer.n_max = 0.0f;
  layer.packed_int4 = false;
  layer.fake_quant = false;
  return layer;
}

// qat.py's _find_float_layers: every plain MatMul/Gemm in `model` whose weight
// is a 2-D fp32 initializer.
//
// Scanned out of the *student* rather than the teacher, unlike the two
// quantized schemes, and that is the substantive difference between
// fine-tuning and QAT rather than an implementation detail. QAT seeds its
// master weights from the teacher because the student's weights are a lossy
// encoding of them. Fine-tuning has no such relationship: the student's
// weights are the starting point precisely because they are *not* the
// teacher's -- they were pruned, or simplified, or already tuned -- and
// re-seeding from the teacher would throw that away before the first step.
//
// Rank-2 and fp32 are required for the same reason the other two schemes
// require them: the loop carries the weight as a 2-D fp32 state tensor, and a
// layer this does not recognize is simply not trained.
std::vector<QuantizedLayer> FindFloatLayers(const onnx::ModelProto& model) {
  const auto initializers = InitializerIndex(model.graph());
  std::vector<QuantizedLayer> layers;
  for (const onnx::NodeProto& node : model.graph().node()) {
    if ((node.op_type() != "MatMul" && node.op_type() != "Gemm") ||
        node.input_size() < 2) {
      continue;
    }
    if (node.output_size() == 0 || node.output(0).empty()) continue;
    const onnx::TensorProto* w_init = Lookup(initializers, node.input(1));
    if (w_init == nullptr) continue;
    if (w_init->data_type() != onnx::TensorProto::FLOAT ||
        w_init->dims_size() != 2) {
      continue;
    }
    layers.push_back(FromFloat(node, *w_init));
  }
  return layers;
}

// qat.py's _find_layers. The two quantized finders are mutually exclusive by
// construction, so this selects rather than merges; which one is selected is
// the single decision learn_activation_scales makes. fake_quant=false selects
// neither: it is the third scheme, in which there is nothing quantized to look
// for and the trainable layers are just the student's own float ones.
std::vector<QuantizedLayer> FindLayers(const onnx::ModelProto& float_model,
                                       const onnx::ModelProto& quantized_model,
                                       bool activation_quant, bool fake_quant) {
  if (!fake_quant) return FindFloatLayers(quantized_model);
  return activation_quant ? FindStaticLayers(float_model, quantized_model)
                          : FindInt4Layers(float_model, quantized_model);
}

// ---------------------------------------------------------------------------
// The block slice -- qat.py's _slice_block and _refuse_unsupported
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
        " is not produced by any node in the float graph; a block must end at "
        "a computed tensor");
  }

  // Forward: which nodes actually depend on the block input. The graph is
  // topologically ordered, so one pass suffices.
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

  // Backward from the block output, confined to those nodes. Intersecting the
  // two directions is what keeps a residual arriving from before the block
  // from dragging the whole earlier subgraph in with it.
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

// Every op in the slice must have a gradient rule, checked before anything
// expensive happens -- graph_grad's own advice to callers that pick their own
// slice.
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
      ". Choose block boundaries that exclude those nodes.");
}

// ---------------------------------------------------------------------------
// Shapes
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

// Every tensor of the float model, shaped as if the caller had captured
// `rows` calibration rows.
//
// This is what qat.py gets for free by running the model: its captured
// activations are concrete arrays whose leading axis is the concatenated
// calibration batch. Pinning each graph input's leading dimension to `rows`
// and inferring reproduces that, and it is the only reading of the float
// model this port does that qat.py did out of data. Existing value_info and
// output shapes are cleared first so a stale symbolic dimension cannot
// survive the pinning and be refused later as "non-static".
ShapeMap InferFloatShapes(const onnx::ModelProto& float_model, int64_t rows) {
  onnx::ModelProto model = float_model;
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
  // Non-strict: a node this build cannot infer is not fatal here, it just
  // leaves its output out of the map, and the slice's own strict inference
  // below reports it against the block the caller named.
  try {
    onnx::shape_inference::InferShapes(model);
  } catch (const std::exception&) {
    // Fall through with whatever was inferred before the failure.
  }

  ShapeMap shapes;
  auto collect = [&shapes](const onnx::ValueInfoProto& value) {
    bool ok = false;
    const Shape dims = StaticDims(value.type(), &ok);
    if (ok) shapes[value.name()] = dims;
  };
  for (const onnx::ValueInfoProto& v : model.graph().input()) collect(v);
  for (const onnx::ValueInfoProto& v : model.graph().value_info()) collect(v);
  for (const onnx::ValueInfoProto& v : model.graph().output()) collect(v);
  for (const onnx::TensorProto& t : model.graph().initializer()) {
    shapes[t.name()] = DimsOf(t);
  }
  return shapes;
}

void SetValueInfo(onnx::ValueInfoProto* vi, const std::string& name,
                  int32_t elem_type, const Shape& dims) {
  vi->set_name(name);
  onnx::TypeProto::Tensor* tensor = vi->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(elem_type);
  onnx::TensorShapeProto* shape = tensor->mutable_shape();
  for (int64_t d : dims) shape->add_dim()->set_dim_value(d);
}

// qat.py's _block_shapes: static shapes for every tensor the slice touches,
// inferred from a standalone model containing only the slice whose inputs
// carry this step's concrete row count. Inference runs at opset 17, the
// pairing the step graph emits, so a node the step graph could not legally
// carry is refused here rather than at session-creation time.
ShapeMap BlockShapes(const onnx::ModelProto& float_model,
                     const std::vector<onnx::NodeProto>& nodes,
                     const std::vector<std::pair<std::string, Shape>>& inputs,
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
  graph->set_name("qat_block");
  for (const onnx::NodeProto& node : nodes) *graph->add_node() = node;
  for (const auto& input : inputs) {
    SetValueInfo(graph->add_input(), input.first, onnx::TensorProto::FLOAT,
                 input.second);
  }
  SetValueInfo(graph->add_output(), block_output_name, onnx::TensorProto::FLOAT,
               block_output_shape);
  for (const onnx::TensorProto& t : float_model.graph().initializer()) {
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
          " in the block has a non-static shape; block-wise QAT needs every "
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

// ---------------------------------------------------------------------------
// Per-layer training state -- qat.py's _Trained and _plan_trained
// ---------------------------------------------------------------------------

struct Trained {
  const QuantizedLayer* candidate = nullptr;
  std::string w_input, m_input, v_input;
  Shape w_shape;
  std::vector<float> w_init;
  int64_t scale_axis = 0;
  Shape scale_shape;
  std::vector<float> scale_init;
  // Empty unless learn_scales.
  std::string scale_input, ms_input, vs_input;
  // Empty unless learn_activation_scales and the layer has a quantizer.
  bool has_act = false;
  std::string log_scale_input, ma_input, va_input;
  std::string zp_input, mz_input, vz_input;
  // Filled in as the graph is built.
  std::string w_next, m_next, v_next;
  std::string scale_next, ms_next, vs_next;
  std::string log_scale_next, ma_next, va_next;
  std::string zp_next, mz_next, vz_next;
};

// One _Trained per quantized layer, with its master weight seeded from the
// *float* model's own weight -- so step 0 of the loop reproduces
// round-to-nearest exactly, and every later step is a measured improvement on
// it rather than on an arbitrary re-initialization. The activation quantizer,
// when there is one, is seeded the same way, from what calibration chose.
std::vector<Trained> PlanTrained(const std::vector<QuantizedLayer>& candidates,
                                 bool learn_scales,
                                 bool learn_activation_scales) {
  std::vector<Trained> planned;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const QuantizedLayer& candidate = candidates[i];
    const std::string index = std::to_string(i);
    Trained trained;
    trained.candidate = &candidate;
    trained.w_input = std::string(kPrefix) + "w" + index;
    trained.m_input = std::string(kPrefix) + "mw" + index;
    trained.v_input = std::string(kPrefix) + "vw" + index;
    trained.w_shape = DimsOf(candidate.w_float_init);
    for (double v : TensorValues(candidate.w_float_init)) {
      trained.w_init.push_back(static_cast<float>(v));
    }
    trained.scale_axis = candidate.axis;
    trained.scale_shape = candidate.scale_dims;
    trained.scale_init = candidate.scale_2d;
    if (learn_scales) {
      trained.scale_input = std::string(kPrefix) + "s" + index;
      trained.ms_input = std::string(kPrefix) + "ms" + index;
      trained.vs_input = std::string(kPrefix) + "vs" + index;
    }
    if (learn_activation_scales && candidate.act.present) {
      trained.has_act = true;
      trained.log_scale_input = std::string(kPrefix) + "as" + index;
      trained.ma_input = std::string(kPrefix) + "mas" + index;
      trained.va_input = std::string(kPrefix) + "vas" + index;
      trained.zp_input = std::string(kPrefix) + "az" + index;
      trained.mz_input = std::string(kPrefix) + "maz" + index;
      trained.vz_input = std::string(kPrefix) + "vaz" + index;
    }
    planned.push_back(std::move(trained));
  }
  return planned;
}

// ---------------------------------------------------------------------------
// The emitters -- qat.py's _blocked_shapes, _broadcast_scale,
// _sum_over_blocks, _emit_fake_quant and _emit_activation_fake_quant
// ---------------------------------------------------------------------------

// The rank-3 views that turn a blocked scale into a full-size one and a
// full-size gradient back into a blocked one. Returns
// (split weight shape, scale shape with a 1 in the block_size slot).
void BlockedShapes(const Shape& w_shape, const Shape& scale_shape, int64_t axis,
                   int64_t block_size, Shape* split, Shape* with_one) {
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("unsupported blocked axis " +
                                std::to_string(axis) + " for a 2-D weight");
  }
  const int64_t other = 1 - axis;
  if (scale_shape[other] != w_shape[other] ||
      scale_shape[axis] * block_size != w_shape[axis]) {
    // The numpy passes tolerate a ragged final block with a slice; doing the
    // same inside a graph would mean a Slice on a dimension the accelerator
    // backends compile statically, and neither scheme produces one.
    throw std::invalid_argument(
        "weight shape " + ShapeStr(w_shape) +
        " is not an exact block-wise tiling of scale shape " +
        ShapeStr(scale_shape) + " with block_size " +
        std::to_string(block_size) + " on axis " + std::to_string(axis));
  }
  *split = Shape{};
  *with_one = Shape{};
  for (int64_t i = 0; i < 2; ++i) {
    if (i == axis) {
      split->push_back(scale_shape[axis]);
      split->push_back(block_size);
      with_one->push_back(scale_shape[axis]);
      with_one->push_back(1);
    } else {
      split->push_back(w_shape[i]);
      with_one->push_back(scale_shape[i]);
    }
  }
}

// A per-block scale expanded to one value per weight element. Expand would say
// this in one node but is outside EpFriendlyOps; multiplying by a constant of
// ones broadcasts identically and costs the size of one block.
std::string BroadcastScale(GraphBuilder& b, const std::string& scale,
                           const Shape& w_shape, const Shape& scale_shape,
                           int64_t axis, int64_t block_size) {
  Shape split;
  Shape with_one;
  BlockedShapes(w_shape, scale_shape, axis, block_size, &split, &with_one);
  Shape ones_shape(split.size(), 1);
  ones_shape[static_cast<size_t>(axis) + 1] = block_size;
  const std::string with_one_const = b.ConstInt64(with_one, "i64");
  const std::string reshaped = b.Op("Reshape", {scale, with_one_const});
  const std::vector<float> ones(static_cast<size_t>(ElementCount(ones_shape)),
                                1.0f);
  const std::string ones_const = b.Const(ones, ones_shape, "ones");
  const std::string tiled = b.Mul(reshaped, ones_const);
  const std::string w_shape_const = b.ConstInt64(w_shape, "i64");
  return b.Op("Reshape", {tiled, w_shape_const});
}

// The transpose of BroadcastScale: one scale is shared by a whole block of
// weights, so its gradient is the sum of theirs.
std::string SumOverBlocks(GraphBuilder& b, const std::string& grad,
                          const Shape& w_shape, const Shape& scale_shape,
                          int64_t axis, int64_t block_size) {
  Shape split;
  Shape with_one;
  BlockedShapes(w_shape, scale_shape, axis, block_size, &split, &with_one);
  const std::string split_const = b.ConstInt64(split, "i64");
  const std::string reshaped = b.Op("Reshape", {grad, split_const});
  const std::string axes_const = b.ConstInt64({axis + 1}, "i64");
  return b.Op("ReduceSum", {reshaped, axes_const}, {IntAttr("keepdims", 0)},
              "blocksum");
}

// Everything the straight-through backward needs out of the weight
// fake-quant: the integer code, the un-rounded ratio, and a float 0/1 mask of
// the elements strictly inside the clipping range.
struct FakeQuant {
  std::string code;
  std::string ratio;
  std::string active;
};

// w_hat = clip(round(w / s), n_min, n_max) * s, written into `out_name`.
// Clipping happens before rounding: the two commute because the bounds are
// integers, and this order leaves RoundToNearest an argument already bounded
// to the grid, where its float-to-int32 cast is exact.
FakeQuant EmitFakeQuant(GraphBuilder& b, const std::string& w,
                        const std::string& scale_full,
                        const std::string& out_name, float n_min, float n_max) {
  FakeQuant out;
  out.ratio = b.Div(w, scale_full);
  const std::string clipped = b.Clip(out.ratio, n_min, n_max);
  out.code = b.RoundToNearest(clipped);
  b.OpInto("Mul", {out.code, scale_full}, out_name);
  const std::string above = b.GreaterMask(out.ratio, n_min);
  const std::string below = b.LessMask(out.ratio, n_max);
  out.active = b.Mul(above, below);
  return out;
}

// The nodes `b` has accumulated since `start`, removed from it -- qat.py's
// _take_nodes. The activation fake-quant needs its nodes in two lists at once
// (the graph's, and the subset BuildBackward is asked to differentiate) and a
// builder appends to only one.
std::vector<onnx::NodeProto> TakeNodes(GraphBuilder& b, size_t start) {
  std::vector<onnx::NodeProto>& all = b.nodes();
  std::vector<onnx::NodeProto> taken(
      all.begin() + static_cast<std::ptrdiff_t>(start), all.end());
  all.erase(all.begin() + static_cast<std::ptrdiff_t>(start), all.end());
  return taken;
}

struct ActivationFakeQuant {
  std::vector<onnx::NodeProto> all_nodes;
  std::vector<onnx::NodeProto> differentiable;
  ShapeMap shapes;
  std::string xdq;
};

// One layer's uint8 affine quantize-dequantize with a learnable scale and
// zero-point, as nodes. See qat.py's _emit_activation_fake_quant for why the
// rounding is emitted as `r + residual` with the two nodes computing
// `residual` deliberately left out of the differentiated list: a tensor no
// differentiated node produces is a leaf, so the Add's other operand receives
// the whole incoming gradient -- which is exactly what "the derivative of
// round is 1" means.
ActivationFakeQuant EmitActivationFakeQuant(GraphBuilder& b, const Trained& t,
                                            const Shape& x_shape) {
  ActivationFakeQuant out;
  const std::string log_scale = t.log_scale_input;
  const std::string zp = t.zp_input;
  out.shapes[log_scale] = Shape{};
  out.shapes[zp] = Shape{};

  size_t start = b.nodes().size();
  const std::string scale = b.Op("Exp", {log_scale}, "act_s");
  const std::string ratio = b.Div(t.candidate->act.tensor, scale);
  std::vector<onnx::NodeProto> head = TakeNodes(b, start);
  out.shapes[scale] = Shape{};
  out.shapes[ratio] = x_shape;

  // The stop-gradient half: these nodes are in the graph but not in the
  // differentiated list, which is what makes the rounding straight-through.
  start = b.nodes().size();
  const std::string rounded_ratio = b.RoundToNearest(ratio);
  const std::string residual = b.Sub(rounded_ratio, ratio);
  std::vector<onnx::NodeProto> rounding = TakeNodes(b, start);
  out.shapes[residual] = x_shape;

  start = b.nodes().size();
  const std::string rounded = b.Add(ratio, residual);
  const std::string raw = b.Add(rounded, zp);
  const std::string clipped = b.Clip(raw, kActNMin, kActNMax);
  const std::string centred = b.Sub(clipped, zp);
  const std::string xdq = b.Mul(centred, scale);
  std::vector<onnx::NodeProto> tail = TakeNodes(b, start);
  for (const std::string& name : {rounded, raw, clipped, centred, xdq}) {
    out.shapes[name] = x_shape;
  }

  out.all_nodes = head;
  out.all_nodes.insert(out.all_nodes.end(), rounding.begin(), rounding.end());
  out.all_nodes.insert(out.all_nodes.end(), tail.begin(), tail.end());
  out.differentiable = head;
  out.differentiable.insert(out.differentiable.end(), tail.begin(), tail.end());
  out.xdq = xdq;
  return out;
}

// ---------------------------------------------------------------------------
// The refusal message -- qat.py's _no_layers_message
// ---------------------------------------------------------------------------

// Why this block has nothing to train, said in terms of the *scheme* the
// caller asked for. The bare fact is nearly useless when the real cause is
// that the model was quantized by a different quantize_* function than the
// flag selects, so the mismatch is detected and named.
std::string NoLayersMessage(const onnx::ModelProto& float_model,
                            const onnx::ModelProto& quantized_model,
                            const std::string& block_input_name,
                            const std::string& block_output_name,
                            bool learn_activation_scales, bool fake_quant) {
  const std::string where = "the block between " + Quoted(block_input_name) +
                            " and " + Quoted(block_output_name);
  if (!fake_quant) {
    return where +
           " contains no MatMul/Gemm with a 2-D fp32 weight initializer to "
           "fine-tune (fake_quant=False trains the model's own float weights, "
           "so a layer whose weight is computed rather than stored, or stored "
           "at some other rank or dtype, has nothing for the optimizer to "
           "hold)";
  }
  if (learn_activation_scales) {
    if (!FindInt4Layers(float_model, quantized_model).empty()) {
      return "learn_activation_scales targets onnxsim.quantize_static's QDQ "
             "scheme (uint8 activations, per-output-channel INT8 weights), but "
             "this quantized model is an onnxsim.quantize_weight_only_int4 one "
             "-- a weight-only model has no activation quantizer anywhere in "
             "it to train. Re-quantize with onnxsim.quantize_static, or leave "
             "learn_activation_scales off to fine-tune the INT4 weights.";
    }
    return where +
           " contains no quantize_static-quantized MatMul/Gemm layer to train "
           "(learn_activation_scales targets that scheme; see apply_qat's "
           "docstring for why it is the only one with activation quantizers to "
           "train)";
  }
  std::string message =
      where +
      " contains no quantize_weight_only_int4-quantized MatMul/Gemm "
      "layer to train";
  if (!FindStaticLayers(float_model, quantized_model).empty()) {
    message +=
        "; this model's layers match onnxsim.quantize_static's QDQ scheme "
        "instead, which learn_activation_scales=True trains";
  }
  return message;
}

// qat.py's _refuse_quantizer_flags_without_fake_quant. fake_quant=false and
// the two scale flags are a contradiction, not a combination: both flags name
// a parameter of a quantizer, and with the fake-quant gone there is no
// quantizer for them to name. Silently ignoring them would be the worse
// failure of the two available -- a caller who asked to learn scales and got a
// model whose scales are exactly as they were has no way to tell that from a
// run in which learning them did not help.
void RefuseQuantizerFlagsWithoutFakeQuant(bool fake_quant, bool learn_scales,
                                          bool learn_activation_scales) {
  if (fake_quant) return;
  std::string asked;
  if (learn_scales) asked = "learn_scales";
  if (learn_activation_scales) {
    if (!asked.empty()) asked += " and ";
    asked += "learn_activation_scales";
  }
  if (asked.empty()) return;
  throw std::invalid_argument(
      asked +
      " cannot be used with fake_quant=False: both train a quantizer's "
      "parameters, and fake_quant=False is the mode with no quantizer in it. "
      "Fine-tuning trains the weights themselves.");
}

}  // namespace

// ---------------------------------------------------------------------------
// BuildQatStepGraph -- qat.py's _plan_block plus _build_step_graph, and the
// state seeding _train_block does around them
// ---------------------------------------------------------------------------

QatStepPlan BuildQatStepGraph(const onnx::ModelProto& float_model,
                              const onnx::ModelProto& quantized_model,
                              const std::string& block_input_name,
                              const std::string& block_output_name,
                              int64_t num_rows, const QatOptions& options) {
  RefuseQuantizerFlagsWithoutFakeQuant(options.fake_quant, options.learn_scales,
                                       options.learn_activation_scales);
  if (num_rows < 1) {
    throw std::invalid_argument("num_rows must be at least 1, got " +
                                std::to_string(num_rows));
  }
  if (options.batch_size < 0) {
    throw std::invalid_argument("batch_size must be at least 1, got " +
                                std::to_string(options.batch_size));
  }

  // --- _plan_block -------------------------------------------------------
  const BlockSlice slice =
      SliceBlock(float_model.graph(), block_input_name, block_output_name);
  if (slice.nodes.empty()) {
    throw std::invalid_argument("no nodes lie between " +
                                Quoted(block_input_name) + " and " +
                                Quoted(block_output_name));
  }
  RefuseUnsupported(slice.nodes);

  std::set<std::string> slice_outputs;
  for (const onnx::NodeProto& node : slice.nodes) {
    for (const std::string& out : node.output()) {
      if (!out.empty()) slice_outputs.insert(out);
    }
  }
  std::vector<QuantizedLayer> candidates;
  for (QuantizedLayer& layer :
       FindLayers(float_model, quantized_model, options.learn_activation_scales,
                  options.fake_quant)) {
    if (slice_outputs.count(layer.output_name) != 0) {
      candidates.push_back(std::move(layer));
    }
  }
  if (candidates.empty()) {
    throw std::invalid_argument(NoLayersMessage(
        float_model, quantized_model, block_input_name, block_output_name,
        options.learn_activation_scales, options.fake_quant));
  }

  // --- shapes ------------------------------------------------------------
  // The whole captured set's shapes, i.e. what the caller binds once.
  const ShapeMap full_shapes = InferFloatShapes(float_model, num_rows);
  auto shape_of = [&full_shapes](const std::string& name) -> const Shape& {
    const auto it = full_shapes.find(name);
    if (it == full_shapes.end()) {
      throw std::invalid_argument(
          "cannot statically infer the shape of " + Quoted(name) +
          " from the float model; block-wise QAT needs every captured "
          "tensor's shape known at build time");
    }
    return it->second;
  };
  std::vector<std::pair<std::string, Shape>> externals;
  for (const std::string& name : slice.externals) {
    externals.emplace_back(name, shape_of(name));
  }
  const Shape teacher_shape = shape_of(block_output_name);

  // --- _plan_minibatch ---------------------------------------------------
  // batch_size 0 means unset, and a batch covering every row *is* the
  // full-batch objective, so both take the full-batch path -- as qat.py's
  // _plan_minibatch does, and for its reason: wrapping the index stream
  // around would quietly reweight the repeated rows.
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

  // Shape inference sees one step's worth of rows, since that is what the
  // block's nodes -- and therefore the backward pass built from them -- will
  // actually be handed.
  auto with_rows = [rows_per_step](Shape dims) {
    if (!dims.empty()) dims[0] = rows_per_step;
    return dims;
  };
  std::vector<std::pair<std::string, Shape>> step_inputs;
  for (const auto& external : externals) {
    step_inputs.emplace_back(external.first, with_rows(external.second));
  }
  const Shape step_teacher_shape = with_rows(teacher_shape);
  const ShapeMap shapes = BlockShapes(float_model, slice.nodes, step_inputs,
                                      block_output_name, step_teacher_shape);

  // --- _plan_trained and the block's own initializers --------------------
  std::vector<Trained> trained = PlanTrained(candidates, options.learn_scales,
                                             options.learn_activation_scales);
  std::set<std::string> trained_weight_names;
  for (const Trained& t : trained) {
    trained_weight_names.insert(t.candidate->float_node.input(1));
  }
  std::set<std::string> used;
  for (const onnx::NodeProto& node : slice.nodes) {
    for (const std::string& name : node.input()) {
      if (!name.empty()) used.insert(name);
    }
  }

  // --- _build_step_graph -------------------------------------------------
  GraphBuilder b(kPrefix);
  // The block's *untrained* constants -- a LayerNorm's scale and bias, a
  // Gemm's C -- come from whichever model the trained weights came from, and
  // for the same reason. Under QAT that is the teacher, whose weights the
  // student encodes. Under fine-tuning it is the student, because the student
  // is a different model and quietly substituting the teacher's constants into
  // it would train the block to compensate for a substitution the deployed
  // model does not make. qat.py's constant_source.
  const onnx::ModelProto& constant_source =
      options.fake_quant ? float_model : quantized_model;
  for (const onnx::TensorProto& t : constant_source.graph().initializer()) {
    if (used.count(t.name()) != 0 &&
        trained_weight_names.count(t.name()) == 0) {
      b.initializer().push_back(t);
    }
  }

  // 0. The minibatch, if there is one. Each captured tensor is declared at
  //    its full size and a Gather pulls this step's rows out of it under the
  //    name the block's own nodes were written against, so step 2 below can
  //    still splice those nodes in verbatim.
  const std::string teacher = std::string(kPrefix) + "teacher";
  const std::string rows_input = std::string(kPrefix) + "rows";
  std::vector<StepGraphSpec::NamedShape> constants;
  std::vector<QatCapture> captures;
  if (!minibatch) {
    for (const auto& external : externals) {
      constants.push_back({external.first, external.second});
      captures.push_back({external.first, external.first, external.second,
                          /*is_teacher=*/false});
    }
    constants.push_back({teacher, teacher_shape});
    captures.push_back(
        {teacher, block_output_name, teacher_shape, /*is_teacher=*/true});
  } else {
    // A captured tensor's table is `qat__all_<its name>` and the teacher's is
    // `qat__teacher_all`; the two families cannot collide whatever the model
    // calls its tensors.
    for (const auto& external : externals) {
      const std::string table = std::string(kPrefix) + "all_" + external.first;
      constants.push_back({table, external.second});
      captures.push_back(
          {table, external.first, external.second, /*is_teacher=*/false});
      b.GatherRowsInto(table, rows_input, external.first);
    }
    const std::string teacher_table = std::string(kPrefix) + "teacher_all";
    constants.push_back({teacher_table, teacher_shape});
    captures.push_back({teacher_table, block_output_name, teacher_shape,
                        /*is_teacher=*/true});
    b.GatherRowsInto(teacher_table, rows_input, teacher);
  }
  const Shape block_output_shape = step_teacher_shape;

  // 1. Fake-quantize each trained master weight into the tensor name the
  //    block's own node already reads, so the block's nodes need no rewriting
  //    at all -- the weight initializer simply became a computed value.
  //    `fq` is meaningful only when this layer has a fake-quant; step 5
  //    branches on `fake_quant` exactly where qat.py branches on its `active`
  //    slot being None.
  struct PerLayer {
    Trained* t;
    // The tensor this layer's gradient arrives on: the block's own weight
    // name, or -- with no fake-quant between them -- the master weight itself.
    std::string weight_name;
    std::string scale_full;
    FakeQuant fq;
    bool fake_quant = true;
  };
  std::vector<PerLayer> per_layer;
  // Block tensor name -> what the block's own node should read instead. Only
  // fake_quant=false puts anything here: the master weight is substituted for
  // the weight initializer by *renaming one input*, rather than by emitting an
  // Identity, because Identity is not in EpFriendlyOps -- a node whose whole
  // job is to copy a tensor is a node an execution provider would have to
  // implement for no reason.
  std::map<std::string, std::string> weight_rewrites;
  ShapeMap weight_shapes;
  for (Trained& t : trained) {
    if (!t.candidate->fake_quant) {
      const std::string weight_name = t.candidate->float_node.input(1);
      weight_rewrites[weight_name] = t.w_input;
      // The master weight is now a differentiated *leaf* of the block rather
      // than a value computed inside it, so the backward needs its shape the
      // way it needs the block's own tensors'.
      weight_shapes[t.w_input] = t.w_shape;
      // No quantizer, so no straight-through mask: step 5 uses the raw
      // gradient.
      per_layer.push_back({&t, t.w_input, "", FakeQuant(), false});
      continue;
    }
    const int64_t block_size = t.candidate->block_size;
    const std::string scale =
        t.scale_input.empty() ? b.Const(t.scale_init, t.scale_shape, "scale")
                              : t.scale_input;
    const std::string scale_full = BroadcastScale(
        b, scale, t.w_shape, t.scale_shape, t.scale_axis, block_size);
    const std::string weight_name = t.candidate->float_node.input(1);
    const FakeQuant fq = EmitFakeQuant(b, t.w_input, scale_full, weight_name,
                                       t.candidate->n_min, t.candidate->n_max);
    per_layer.push_back({&t, weight_name, scale_full, fq, true});
  }

  // 2. The block itself, node for node as the float graph wrote it -- except
  //    that a layer whose activation quantizer is being trained reads a
  //    fake-quantized copy of its own input instead of the raw tensor. Only
  //    that one node is rewritten (one input name), so the quantizer lands on
  //    the *edge* the deployed QDQ pair occupies rather than on the tensor.
  //    With no fake-quant, one more input name is rewritten per trained
  //    layer: the weight the block reads becomes the master weight.
  ShapeMap act_shapes;
  std::map<std::string, Trained*> quantized_input;
  for (Trained& t : trained) {
    if (t.has_act) quantized_input[t.candidate->output_name] = &t;
  }
  std::vector<onnx::NodeProto> forward;
  std::vector<onnx::NodeProto> differentiated;
  for (const onnx::NodeProto& original : slice.nodes) {
    onnx::NodeProto node = original;
    Trained* layer = nullptr;
    if (node.output_size() > 0) {
      const auto it = quantized_input.find(node.output(0));
      if (it != quantized_input.end()) layer = it->second;
    }
    if (layer != nullptr) {
      const auto shape_it = shapes.find(node.input(0));
      if (shape_it == shapes.end()) {
        throw std::invalid_argument("no shape for the activation " +
                                    Quoted(node.input(0)) +
                                    " whose quantizer is being trained");
      }
      const ActivationFakeQuant emitted =
          EmitActivationFakeQuant(b, *layer, shape_it->second);
      forward.insert(forward.end(), emitted.all_nodes.begin(),
                     emitted.all_nodes.end());
      differentiated.insert(differentiated.end(),
                            emitted.differentiable.begin(),
                            emitted.differentiable.end());
      act_shapes.insert(emitted.shapes.begin(), emitted.shapes.end());
      node.set_input(0, emitted.xdq);
    }
    for (int i = 0; i < node.input_size(); ++i) {
      const auto rewrite = weight_rewrites.find(node.input(i));
      if (rewrite != weight_rewrites.end()) node.set_input(i, rewrite->second);
    }
    forward.push_back(node);
    differentiated.push_back(node);
  }
  for (const onnx::NodeProto& node : forward) b.nodes().push_back(node);

  // 3. The objective: MSE of the student block's output against the
  //    teacher's, and its gradient, which is the seed of the backward pass.
  //    block_output_shape is this step's shape, so the 2/n normalizer is the
  //    batch's element count and one learning rate stays meaningful across
  //    batch sizes.
  const std::string diff = b.Sub(block_output_name, teacher);
  const int64_t n_elems = ElementCount(block_output_shape);
  const std::string two_over_n =
      b.Const(static_cast<float>(2.0 / static_cast<double>(n_elems)));
  const std::string dl_dy = b.Mul(diff, two_over_n);

  // 4. The backward pass over the block, emitted as ONNX nodes. The
  //    activation quantizers' own parameters are targets alongside the
  //    weights: nothing else reaches them, since they are read only by the
  //    fake-quant chain.
  ShapeMap all_shapes = shapes;
  for (const auto& entry : act_shapes) all_shapes[entry.first] = entry.second;
  for (const auto& entry : weight_shapes)
    all_shapes[entry.first] = entry.second;
  std::vector<std::string> targets;
  for (const PerLayer& layer : per_layer) targets.push_back(layer.weight_name);
  for (const Trained& t : trained) {
    if (!t.has_act) continue;
    targets.push_back(t.log_scale_input);
    targets.push_back(t.zp_input);
  }
  const std::map<std::string, std::string> grads = BuildBackward(
      b, differentiated, all_shapes, {{block_output_name, dl_dy}}, targets);
  auto grad_of = [&grads](const std::string& name) -> const std::string& {
    const auto it = grads.find(name);
    if (it == grads.end()) {
      throw std::invalid_argument(
          "the backward pass produced no gradient for " + Quoted(name));
    }
    return it->second;
  };

  // 5. Straight through the fake-quant, into the master weight and (if asked
  //    for) the scale, then one Adam step each.
  const std::string lr = std::string(kPrefix) + "lr";
  const std::string lr_scale = std::string(kPrefix) + "lr_scale";
  const std::string lr_act = std::string(kPrefix) + "lr_act";
  for (PerLayer& layer : per_layer) {
    Trained& t = *layer.t;
    // dL/d(w_hat), in the weight's storage layout.
    const std::string g = grad_of(layer.weight_name);
    // STE: d(w_hat)/d(w) is 1 inside the clipping range and 0 outside. The
    // scale cancels -- w_hat = round(w/s)*s -- which is why a straight-through
    // weight gradient is just the masked output gradient, with no scale factor
    // anywhere. Without a fake-quant there is no clipping range and no
    // estimator: `g` is already dL/dw.
    const std::string masked = layer.fake_quant ? b.Mul(g, layer.fq.active) : g;
    const AdamOutputs w_step =
        AdamUpdate(b, t.w_input, masked, t.m_input, t.v_input, lr,
                   "m_correction", "v_correction");
    t.w_next = w_step.param_next;
    t.m_next = w_step.m_next;
    t.v_next = w_step.v_next;
    if (!t.scale_input.empty()) {
      // LSQ's scale gradient, the same one onnxsim.autoround derives:
      // d(w_hat)/d(s) = code - w/s where the element is inside the clipping
      // range, and just `code` where it saturates.
      const std::string inside = b.Mul(layer.fq.active, layer.fq.ratio);
      const std::string dwhat_ds = b.Sub(layer.fq.code, inside);
      const std::string weighted = b.Mul(g, dwhat_ds);
      const std::string g_scale =
          SumOverBlocks(b, weighted, t.w_shape, t.scale_shape, t.scale_axis,
                        t.candidate->block_size);
      const AdamOutputs scale_step =
          AdamUpdate(b, t.scale_input, g_scale, t.ms_input, t.vs_input,
                     lr_scale, "m_correction", "v_correction");
      t.scale_next = scale_step.param_next;
      t.ms_next = scale_step.m_next;
      t.vs_next = scale_step.v_next;
    }
    if (!t.has_act) continue;
    // The activation quantizer's two parameters. Their gradients were emitted
    // by the backward walk over the fake-quant chain, so all that is left is
    // an Adam step each.
    const AdamOutputs act_step =
        AdamUpdate(b, t.log_scale_input, grad_of(t.log_scale_input), t.ma_input,
                   t.va_input, lr_act, "m_correction", "v_correction");
    t.log_scale_next = act_step.param_next;
    t.ma_next = act_step.m_next;
    t.va_next = act_step.v_next;
    const AdamOutputs zp_step =
        AdamUpdate(b, t.zp_input, grad_of(t.zp_input), t.mz_input, t.vz_input,
                   lr_act, "m_correction", "v_correction");
    t.mz_next = zp_step.m_next;
    t.vz_next = zp_step.v_next;
    // Re-clamped into uint8's range every step rather than only at export:
    // the forward's clip is what the whole activation gradient is derived
    // through, so a zero-point that wandered outside the representable range
    // would saturate every element and silently kill the signal.
    t.zp_next = b.Clip(zp_step.param_next, kActNMin, kActNMax);
  }

  std::vector<StepGraphSpec::StateEntry> state;
  std::vector<onnx::TensorProto> initial_state;
  for (const Trained& t : trained) {
    state.push_back({t.w_input, t.w_shape, t.w_next});
    state.push_back({t.m_input, t.w_shape, t.m_next});
    state.push_back({t.v_input, t.w_shape, t.v_next});
    initial_state.push_back(MakeFloatTensor(t.w_input, t.w_shape, t.w_init));
    initial_state.push_back(MakeZeroTensor(t.m_input, t.w_shape));
    initial_state.push_back(MakeZeroTensor(t.v_input, t.w_shape));
    if (!t.scale_input.empty()) {
      state.push_back({t.scale_input, t.scale_shape, t.scale_next});
      state.push_back({t.ms_input, t.scale_shape, t.ms_next});
      state.push_back({t.vs_input, t.scale_shape, t.vs_next});
      initial_state.push_back(
          MakeFloatTensor(t.scale_input, t.scale_shape, t.scale_init));
      initial_state.push_back(MakeZeroTensor(t.ms_input, t.scale_shape));
      initial_state.push_back(MakeZeroTensor(t.vs_input, t.scale_shape));
    }
    if (t.has_act) {
      state.push_back({t.log_scale_input, {}, t.log_scale_next});
      state.push_back({t.ma_input, {}, t.ma_next});
      state.push_back({t.va_input, {}, t.va_next});
      state.push_back({t.zp_input, {}, t.zp_next});
      state.push_back({t.mz_input, {}, t.mz_next});
      state.push_back({t.vz_input, {}, t.vz_next});
      // Seeded from what calibration chose, so step 0 is the quantized model
      // as shipped. The 1e-8 floor is onnxsim.adaquant's, guarding the
      // degenerate calibrated scale of exactly 0 that log would turn into
      // -inf.
      const double floored = std::max(t.candidate->act.scale_init, 1e-8);
      initial_state.push_back(MakeFloatScalar(
          t.log_scale_input, static_cast<float>(std::log(floored))));
      initial_state.push_back(MakeFloatScalar(t.ma_input, 0.0f));
      initial_state.push_back(MakeFloatScalar(t.va_input, 0.0f));
      const double clamped = std::min(
          std::max(t.candidate->act.zp_init, static_cast<double>(kActNMin)),
          static_cast<double>(kActNMax));
      initial_state.push_back(
          MakeFloatScalar(t.zp_input, static_cast<float>(clamped)));
      initial_state.push_back(MakeFloatScalar(t.mz_input, 0.0f));
      initial_state.push_back(MakeFloatScalar(t.vz_input, 0.0f));
    }
  }

  std::vector<std::string> scalars{lr, "m_correction", "v_correction"};
  if (options.learn_scales) scalars.push_back(lr_scale);
  if (options.learn_activation_scales) scalars.push_back(lr_act);

  StepGraphSpec spec;
  spec.constants = constants;
  spec.state = state;
  spec.scalars = scalars;
  if (minibatch) {
    spec.per_step.push_back({rows_input,
                             {options.batch_size},
                             static_cast<int32_t>(onnx::TensorProto::INT64)});
  }
  // Emitted last, exactly where the Python's `loss=b.mean_square(diff)`
  // argument is evaluated: it consumes two names, and consuming them earlier
  // would renumber everything after it.
  spec.loss_output = b.MeanSquare(diff);
  spec.graph_name = "onnxsim_qat_step";
  const StepGraph step = MakeStepGraph(b, spec);

  QatStepPlan plan;
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
  for (const Trained& t : trained) {
    QatTrainedLayer layer;
    layer.weight_state_input = t.w_input;
    layer.weight_scale_state_input = t.scale_input;
    layer.log_act_scale_state_input = t.log_scale_input;
    layer.act_zero_point_state_input = t.zp_input;
    layer.codes_initializer = t.candidate->wq_name;
    layer.weight_scale_initializer = t.candidate->ws_name;
    if (t.has_act) {
      layer.act_scale_initializer = t.candidate->act.scale_name;
      layer.act_zero_point_initializer = t.candidate->act.zp_name;
    }
    layer.weight_dims = t.w_shape;
    // The normalized 2-D blocked view, which is also the shape of the loop's
    // scale state tensor. The initializer it writes back into keeps whatever
    // shape the model stores it in (1-D for quantize_static's per-channel
    // scale); WriteBackQatState reads that from the model rather than from
    // here, since the element count is the same either way.
    layer.weight_scale_dims = t.scale_shape;
    layer.block_axis = t.scale_axis;
    layer.block_size = t.candidate->block_size;
    layer.code_min = t.candidate->n_min;
    layer.code_max = t.candidate->n_max;
    layer.packed_int4 = t.candidate->packed_int4;
    layer.frozen_weight_scale = MakeFloatTensor(
        t.candidate->ws_name, t.scale_shape, t.candidate->scale_2d);
    layer.fake_quant = t.candidate->fake_quant;
    plan.layers.push_back(std::move(layer));
  }
  return plan;
}

// ---------------------------------------------------------------------------
// WriteBackQatState -- the tail of qat.py's _train_block
// ---------------------------------------------------------------------------

onnx::ModelProto WriteBackQatState(
    const onnx::ModelProto& quantized_model, const QatStepPlan& plan,
    const std::map<std::string, onnx::TensorProto>& final_state) {
  auto state_of =
      [&final_state](const std::string& name) -> const onnx::TensorProto& {
    const auto it = final_state.find(name);
    if (it == final_state.end()) {
      throw std::invalid_argument("final_state is missing the state tensor " +
                                  Quoted(name));
    }
    return it->second;
  };

  // Everything the pass rewrites, keyed by the initializer it lands in. Built
  // first so a layer naming an initializer the model does not have is a
  // no-op rather than a partial rewrite, exactly as the Python's dict lookup
  // over the initializer list is.
  // Keyed by initializer, and already built as the tensor that replaces it:
  // fp32, under the layer's own name, with the dims the plan carries.
  std::map<std::string, onnx::TensorProto> new_weights;
  std::map<std::string, std::vector<int8_t>> new_codes;
  std::map<std::string, std::vector<double>> new_scales;
  std::map<std::string, double> new_act_scales;
  std::map<std::string, int64_t> new_act_zps;

  for (const QatTrainedLayer& layer : plan.layers) {
    const std::vector<double> w =
        TensorValues(state_of(layer.weight_state_input));
    if (!layer.fake_quant) {
      // Nothing to project back onto: the master weight *is* what the model
      // stores, so the write-back is the identity that the two quantized
      // schemes' rounding and clipping stand in for.
      if (static_cast<int64_t>(w.size()) != ElementCount(layer.weight_dims)) {
        throw std::invalid_argument(
            "the trained weight " + Quoted(layer.weight_state_input) + " has " +
            std::to_string(w.size()) + " elements, expected " +
            std::to_string(ElementCount(layer.weight_dims)));
      }
      std::vector<float> values;
      values.reserve(w.size());
      for (double v : w) values.push_back(static_cast<float>(v));
      new_weights[layer.codes_initializer] =
          MakeFloatTensor(layer.codes_initializer, layer.weight_dims, values);
      continue;
    }
    const std::vector<double> scale =
        layer.weight_scale_state_input.empty()
            ? TensorValues(layer.frozen_weight_scale)
            : TensorValues(state_of(layer.weight_scale_state_input));
    if (layer.weight_dims.size() != 2 || layer.weight_scale_dims.size() != 2) {
      throw std::invalid_argument(
          "a trained layer's weight and scale must be "
          "2-D in the normalized blocked view");
    }
    const int64_t rows = layer.weight_dims[0];
    const int64_t cols = layer.weight_dims[1];
    if (static_cast<int64_t>(w.size()) != rows * cols) {
      throw std::invalid_argument(
          "the trained weight " + Quoted(layer.weight_state_input) + " has " +
          std::to_string(w.size()) + " elements, expected " +
          std::to_string(rows * cols));
    }
    const int64_t scale_cols = layer.weight_scale_dims[1];

    // scale_full = np.repeat(scale, block_size, axis=block_axis): the index
    // along the blocked axis is the weight's own divided by the block size.
    std::vector<int8_t> codes(static_cast<size_t>(rows * cols));
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        const int64_t si = layer.block_axis == 0 ? i / layer.block_size : i;
        const int64_t sj = layer.block_axis == 1 ? j / layer.block_size : j;
        const double s = scale[static_cast<size_t>(si * scale_cols + sj)];
        const double code =
            RoundHalfAway(w[static_cast<size_t>(i * cols + j)] / s);
        const double clipped =
            std::min(std::max(code, static_cast<double>(layer.code_min)),
                     static_cast<double>(layer.code_max));
        codes[static_cast<size_t>(i * cols + j)] = static_cast<int8_t>(clipped);
      }
    }
    new_codes[layer.codes_initializer] = std::move(codes);
    if (!layer.weight_scale_state_input.empty()) {
      new_scales[layer.weight_scale_initializer] = scale;
    }
    if (!layer.log_act_scale_state_input.empty()) {
      // Out of log space, and the zero-point back onto uint8's integer grid --
      // the two projections the optimizer's continuous parametrization has to
      // be undone by, because the model can only store what it can store.
      const std::vector<double> log_scale =
          TensorValues(state_of(layer.log_act_scale_state_input));
      const std::vector<double> zp =
          TensorValues(state_of(layer.act_zero_point_state_input));
      if (log_scale.empty() || zp.empty()) {
        throw std::invalid_argument(
            "the activation quantizer's final state tensors must hold one "
            "value each");
      }
      new_act_scales[layer.act_scale_initializer] = std::exp(log_scale[0]);
      // Python's round() is half-to-even, and so is std::nearbyint under the
      // default rounding mode -- unlike the codes above, which round half
      // away from zero to match the forward's own rounding.
      const double rounded = std::nearbyint(zp[0]);
      new_act_zps[layer.act_zero_point_initializer] = static_cast<int64_t>(
          std::min(std::max(rounded, static_cast<double>(kActNMin)),
                   static_cast<double>(kActNMax)));
    }
  }

  onnx::ModelProto tuned = quantized_model;
  for (onnx::TensorProto& initializer :
       *tuned.mutable_graph()->mutable_initializer()) {
    const auto weight = new_weights.find(initializer.name());
    if (weight != new_weights.end()) {
      // The tensor the loop trained, stored as itself. The initializer is
      // replaced wholesale rather than patched -- numpy_helper.from_array's
      // own rewrite in the Python -- so a weight the model happened to store
      // some other way (in float_data, say) comes out canonical raw fp32.
      initializer = weight->second;
      continue;
    }
    const auto codes = new_codes.find(initializer.name());
    if (codes != new_codes.end()) {
      if (initializer.data_type() == onnx::TensorProto::INT4) {
        // Same low-nibble-first packing as
        // weight_only_quantize_int4_matmul.h's own, and as adaround.py's
        // _pack_int4: byte[i] = (code[2i] & 0xF) | ((code[2i+1] & 0xF) << 4).
        std::string raw;
        raw.reserve((codes->second.size() + 1) / 2);
        for (size_t i = 0; i + 1 < codes->second.size(); i += 2) {
          const unsigned lo = static_cast<unsigned>(codes->second[i]) & 0xFu;
          const unsigned hi =
              static_cast<unsigned>(codes->second[i + 1]) & 0xFu;
          raw.push_back(static_cast<char>(lo | (hi << 4)));
        }
        if (codes->second.size() % 2 != 0) {
          const unsigned lo =
              static_cast<unsigned>(codes->second.back()) & 0xFu;
          raw.push_back(static_cast<char>(lo));
        }
        initializer.set_raw_data(std::move(raw));
      } else {
        const Shape dims = DimsOf(initializer);
        onnx::TensorProto replacement;
        replacement.set_name(initializer.name());
        replacement.set_data_type(onnx::TensorProto::INT8);
        for (int64_t d : dims) replacement.add_dims(d);
        std::string raw(codes->second.begin(), codes->second.end());
        replacement.set_raw_data(std::move(raw));
        initializer = replacement;
      }
      continue;
    }
    const auto scale = new_scales.find(initializer.name());
    if (scale != new_scales.end()) {
      const Shape dims = DimsOf(initializer);
      std::vector<float> values;
      values.reserve(scale->second.size());
      for (double v : scale->second) values.push_back(static_cast<float>(v));
      initializer = MakeFloatTensor(initializer.name(), dims, values);
      continue;
    }
    const auto act_scale = new_act_scales.find(initializer.name());
    if (act_scale != new_act_scales.end()) {
      // A 0-d tensor, as numpy_helper.from_array(np.array(x, np.float32))
      // produces -- the shape the Python writes back.
      initializer = MakeFloatScalar(initializer.name(),
                                    static_cast<float>(act_scale->second));
      continue;
    }
    const auto act_zp = new_act_zps.find(initializer.name());
    if (act_zp != new_act_zps.end()) {
      onnx::TensorProto replacement;
      replacement.set_name(initializer.name());
      replacement.set_data_type(onnx::TensorProto::UINT8);
      replacement.set_raw_data(
          std::string(1, static_cast<char>(act_zp->second & 0xff)));
      initializer = replacement;
    }
  }
  return tuned;
}
