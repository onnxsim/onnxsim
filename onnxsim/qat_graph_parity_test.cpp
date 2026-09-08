// The C++ half of the Python<->C++ step-graph emitter parity check.
//
// qat_graph_builder.{h,cpp} re-implements the emitter half of
// onnxsim/qat_graph.py so the browser converter can build a training step graph
// without a Python round trip. The hazard that creates is not a crash -- it is
// two emitters that quietly disagree, both producing valid graphs that run, so
// the browser trains a model differently from the Python and nothing says so.
//
// onnxsim/qat_parity_fixtures.txt is the shared reference.
// tests/test_qat_parity.py asserts fixture == Python; this file asserts fixture
// == C++. Together they give Python == C++, which is the property actually
// wanted and which neither test establishes alone. In particular this file
// CANNOT tell a correct port from one that matches a fixture which stopped
// describing the Python weeks ago -- that is the Python test's job, and it is
// why the pair is not redundant.
//
// The comparison is string equality on a flat text rendering, so a failure
// prints as a readable diff of exactly the lines that moved. The renderer below
// must stay byte-compatible with `_render_case`/`render` in
// scripts/make_qat_parity_fixtures.py; the two are short and deliberately
// parallel. Floats are rendered as IEEE-754 bit patterns because decimal
// formatting differs between the languages and the differences this exists to
// catch are one ulp wide.
//
// Run: qat_graph_parity_test  (ctest -R qat_graph_parity_test)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "graph_grad.h"
#include "onnx/defs/parser.h"
#include "qat_entry.h"
#include "qat_graph_builder.h"
#include "quantize_entry.h"

namespace {

int failures = 0;

// ---------------------------------------------------------------------------
// Rendering, mirroring scripts/make_qat_parity_fixtures.py
// ---------------------------------------------------------------------------

std::string F32Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%08x", bits);
  return buf;
}

std::string Join(const std::vector<std::string>& parts, const char* sep) {
  std::string out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) out += sep;
    out += parts[i];
  }
  return out;
}

// raw_data's byte order is fixed little-endian by ONNX on every host -- see
// onnxsim/passes/endian_read.h, which exists for exactly this -- so these
// decode by shifting bytes in, not by casting the host's own layout over them.
// A memcpy/reinterpret_cast would agree with the fixture on x86 and disagree on
// s390x, and the big-endian CI job runs these tests, so it would be caught --
// but as a confusing parity failure blaming the emitter rather than the reader.
uint32_t LittleEndianU32(const char* p) {
  return (static_cast<uint32_t>(static_cast<unsigned char>(p[0]))) |
         (static_cast<uint32_t>(static_cast<unsigned char>(p[1])) << 8) |
         (static_cast<uint32_t>(static_cast<unsigned char>(p[2])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(p[3])) << 24);
}

uint64_t LittleEndianU64(const char* p) {
  uint64_t bits = 0;
  for (int i = 0; i < 8; ++i) {
    bits |= static_cast<uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
  }
  return bits;
}

std::string F32BitsFromLittleEndian(const char* p) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%08x", LittleEndianU32(p));
  return buf;
}

// A tensor's values, whichever field the producer chose to put them in.
// Encoding is deliberately not part of the comparison: float_data and raw_data
// are indistinguishable to a runtime, so the fixture compares numbers.
std::vector<std::string> TensorValues(const onnx::TensorProto& t) {
  std::vector<std::string> out;
  if (t.data_type() == onnx::TensorProto::INT64) {
    if (t.int64_data_size() > 0) {
      // A typed field needs no swap: protobuf decodes it into host-order
      // scalars itself. Only the raw_data branch below is byte order's problem.
      for (int i = 0; i < t.int64_data_size(); ++i) {
        out.push_back(std::to_string(t.int64_data(i)));
      }
    } else {
      const std::string& raw = t.raw_data();
      for (size_t i = 0; i + 8 <= raw.size(); i += 8) {
        out.push_back(std::to_string(
            static_cast<int64_t>(LittleEndianU64(raw.data() + i))));
      }
    }
    return out;
  }
  if (t.float_data_size() > 0) {
    for (int i = 0; i < t.float_data_size(); ++i) {
      out.push_back(F32Bits(t.float_data(i)));
    }
  } else {
    const std::string& raw = t.raw_data();
    for (size_t i = 0; i + 4 <= raw.size(); i += 4) {
      out.push_back(F32BitsFromLittleEndian(raw.data() + i));
    }
  }
  return out;
}

std::string RenderAttributes(const onnx::NodeProto& node) {
  // Sorted by name, as the Python's `sorted(node["attributes"].items())`.
  std::map<std::string, std::string> attrs;
  for (const auto& a : node.attribute()) {
    if (a.type() == onnx::AttributeProto::INT) {
      attrs[a.name()] = std::to_string(a.i());
    } else if (a.type() == onnx::AttributeProto::INTS) {
      std::vector<std::string> ints;
      for (int64_t v : a.ints()) ints.push_back(std::to_string(v));
      attrs[a.name()] = Join(ints, ",");
    } else {
      // The generator raises on an attribute type it cannot describe rather
      // than dropping it; do the same here, so a new attribute kind cannot
      // slip through the comparison unnoticed.
      std::cerr << "unsupported attribute type on " << node.op_type() << ": "
                << a.name() << "\n";
      ++failures;
    }
  }
  std::vector<std::string> parts;
  for (const auto& kv : attrs) parts.push_back(kv.first + "=" + kv.second);
  return Join(parts, ";");
}

std::vector<std::string> RenderBuilder(const GraphBuilder& b) {
  std::vector<std::string> lines;
  for (const auto& t : b.initializer()) {
    std::vector<std::string> dims;
    for (int64_t d : t.dims()) dims.push_back(std::to_string(d));
    lines.push_back("  init " + t.name() + " " +
                    std::to_string(static_cast<int>(t.data_type())) + " [" +
                    Join(dims, ",") + "] " + Join(TensorValues(t), ","));
  }
  for (const auto& n : b.nodes()) {
    std::vector<std::string> inputs(n.input().begin(), n.input().end());
    std::vector<std::string> outputs(n.output().begin(), n.output().end());
    lines.push_back("  node " + n.op_type() + " [" + Join(inputs, ",") + "] [" +
                    Join(outputs, ",") + "] {" + RenderAttributes(n) + "}");
  }
  return lines;
}

std::string Dims(const onnx::ValueInfoProto& v) {
  std::vector<std::string> dims;
  for (const auto& d : v.type().tensor_type().shape().dim()) {
    dims.push_back(std::to_string(d.dim_value()));
  }
  return Join(dims, ",");
}

// ---------------------------------------------------------------------------
// The cases, mirroring the `_case_*` functions in the generator
// ---------------------------------------------------------------------------

std::vector<std::string> CaseArithmetic() {
  GraphBuilder b;
  std::string s = b.Add("x", "y");
  s = b.Sub(s, "y");
  s = b.Mul(s, "y");
  s = b.Div(s, "y");
  s = b.MatMul(s, "w");
  s = b.Transpose(s);
  s = b.Transpose(s, {1, 0});
  s = b.Sqrt(s);
  s = b.Sigmoid(s);
  const std::string result = b.MeanSquare(s);
  auto lines = RenderBuilder(b);
  lines.push_back("  result " + result);
  return lines;
}

std::vector<std::string> CaseMasksAndClip() {
  GraphBuilder b;
  const std::string clipped = b.Clip("x", -7.0f, 7.0f);
  const std::string gt = b.GreaterMask(clipped, -7.0f);
  const std::string lt = b.LessMask(clipped, 7.0f);
  const std::string result = b.Mul(gt, lt);
  auto lines = RenderBuilder(b);
  lines.push_back("  result " + result);
  return lines;
}

std::vector<std::string> CaseRoundToNearest() {
  GraphBuilder b;
  const std::string result = b.RoundToNearest("x");
  auto lines = RenderBuilder(b);
  lines.push_back("  result " + result);
  return lines;
}

std::vector<std::string> CaseGatherRows() {
  GraphBuilder b;
  const std::string fresh = b.GatherRows("table", "idx");
  b.GatherRowsInto("table", "idx", "block_input");
  auto lines = RenderBuilder(b);
  lines.push_back("  result " + fresh);
  return lines;
}

std::vector<std::string> CaseConsts() {
  GraphBuilder b("pre_");
  b.Const(0.5f);
  b.Const({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  const std::string result = b.Add("x", b.Const(-1.25f));
  auto lines = RenderBuilder(b);
  lines.push_back("  result " + result);
  return lines;
}

std::vector<std::string> CaseAdamUpdate() {
  GraphBuilder b;
  const AdamOutputs out = AdamUpdate(b, "p", "g", "m", "v", "lr", "mc", "vc");
  auto lines = RenderBuilder(b);
  lines.push_back("  result " + out.param_next + "," + out.m_next + "," +
                  out.v_next);
  return lines;
}

std::vector<std::string> CaseStepGraph() {
  GraphBuilder b;
  const std::string diff = b.Sub("student", "teacher");
  const std::string loss = b.MeanSquare(diff);
  const AdamOutputs adam = AdamUpdate(b, "w", diff, "m", "v", "lr", "mc", "vc");

  StepGraphSpec spec;
  spec.constants = {{"teacher", {4, 3}}};
  spec.state = {
      {"w", {4, 3}, adam.param_next},
      {"m", {4, 3}, adam.m_next},
      {"v", {4, 3}, adam.v_next},
  };
  spec.scalars = {"lr", "mc", "vc"};
  spec.per_step = {{"rows", {2}, onnx::TensorProto::INT64}};
  spec.loss_output = loss;
  const StepGraph step = MakeStepGraph(b, spec);

  auto lines = RenderBuilder(b);
  for (const auto& o : step.model.opset_import()) {
    lines.push_back("  opset " + o.domain() + " " +
                    std::to_string(o.version()));
  }
  lines.push_back("  ir_version " + std::to_string(step.model.ir_version()));
  lines.push_back("  graph_name " + step.model.graph().name());
  for (const auto& i : step.model.graph().input()) {
    lines.push_back("  input " + i.name() + " " +
                    std::to_string(i.type().tensor_type().elem_type()) + " [" +
                    Dims(i) + "]");
  }
  for (const auto& o : step.model.graph().output()) {
    lines.push_back("  output " + o.name() + " " +
                    std::to_string(o.type().tensor_type().elem_type()) + " [" +
                    Dims(o) + "]");
  }
  std::map<std::string, std::string> state(step.state.begin(),
                                           step.state.end());
  for (const auto& kv : state) {
    lines.push_back("  state " + kv.first + " " + kv.second);
  }
  lines.push_back("  loss " + step.loss_name);
  return lines;
}

// The planner case's float model, read out of the fixture rather than rebuilt
// here. The fixture is the single definition of that model; two hand-built
// copies that drifted apart would produce two different step graphs, and the
// diff would blame the planner rather than the models. Set by main().
std::string g_planner_model_text;

// W1/W2's values. Must match _planner_weight in the generator bit for bit:
// the arithmetic is done in double and narrowed once, so both languages land
// on the same float.
onnx::TensorProto PlannerWeight(const std::string& name) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(onnx::TensorProto::FLOAT);
  t.add_dims(32);
  t.add_dims(32);
  std::string raw;
  for (int i = 0; i < 32 * 32; ++i) {
    const float v = static_cast<float>(((i % 7) - 3) * 0.1);
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int b = 0; b < 4; ++b) {
      raw.push_back(static_cast<char>((bits >> (8 * b)) & 0xff));
    }
  }
  t.set_raw_data(std::move(raw));
  return t;
}

// A whole step graph, built the way the browser will build it.
//
// Every other case pins one emitter primitive; this one pins the composition
// that onnxsim/qat_entry.cpp performs -- slice the block, find the quantized
// layer, plan the trained state, emit fake-quant + block + loss + backward +
// Adam. It is the case that actually holds qat_entry.cpp to qat.py, and the
// one most likely to catch a reordering, since even this small model's step
// graph is dozens of nodes deep.
std::vector<std::string> CasePlanner() {
  onnx::ModelProto float_model;
  onnx::OnnxParser parser(g_planner_model_text.c_str());
  auto status = parser.Parse(float_model);
  if (!status.IsOK()) {
    std::cerr << "planner: cannot parse the fixture's model_text: "
              << status.ErrorMessage() << "\n";
    ++failures;
    return {};
  }
  *float_model.mutable_graph()->add_initializer() = PlannerWeight("W1");
  *float_model.mutable_graph()->add_initializer() = PlannerWeight("W2");

  const onnx::ModelProto quantized = QuantizeWeightOnlyInt4(float_model);
  const QatStepPlan plan =
      BuildQatStepGraph(float_model, quantized, "X", "Y", 4, QatOptions{});
  const onnx::GraphProto& graph = plan.step_graph.graph();

  std::vector<std::string> lines;
  for (const auto& t : graph.initializer()) {
    std::vector<std::string> dims;
    for (int64_t d : t.dims()) dims.push_back(std::to_string(d));
    lines.push_back("  init " + t.name() + " " +
                    std::to_string(static_cast<int>(t.data_type())) + " [" +
                    Join(dims, ",") + "] " + Join(TensorValues(t), ","));
  }
  for (const auto& n : graph.node()) {
    std::vector<std::string> inputs(n.input().begin(), n.input().end());
    std::vector<std::string> outputs(n.output().begin(), n.output().end());
    lines.push_back("  node " + n.op_type() + " [" + Join(inputs, ",") + "] [" +
                    Join(outputs, ",") + "] {" + RenderAttributes(n) + "}");
  }
  for (const auto& o : plan.step_graph.opset_import()) {
    lines.push_back("  opset " + o.domain() + " " +
                    std::to_string(o.version()));
  }
  lines.push_back("  ir_version " +
                  std::to_string(plan.step_graph.ir_version()));
  lines.push_back("  graph_name " + graph.name());
  for (const auto& i : graph.input()) {
    lines.push_back("  input " + i.name() + " " +
                    std::to_string(i.type().tensor_type().elem_type()) + " [" +
                    Dims(i) + "]");
  }
  for (const auto& o : graph.output()) {
    lines.push_back("  output " + o.name() + " " +
                    std::to_string(o.type().tensor_type().elem_type()) + " [" +
                    Dims(o) + "]");
  }
  std::map<std::string, std::string> state(plan.state.begin(),
                                           plan.state.end());
  for (const auto& kv : state) {
    lines.push_back("  state " + kv.first + " " + kv.second);
  }
  lines.push_back("  loss " + plan.loss_name);
  return lines;
}

// Sorted by name, matching the generator's `for name in sorted(cases)`.
const std::vector<std::pair<std::string, std::vector<std::string> (*)()>>&
Cases() {
  static const std::vector<
      std::pair<std::string, std::vector<std::string> (*)()>>
      cases = {
          {"adam_update", CaseAdamUpdate},
          {"arithmetic", CaseArithmetic},
          {"consts", CaseConsts},
          {"gather_rows", CaseGatherRows},
          {"masks_and_clip", CaseMasksAndClip},
          {"planner", CasePlanner},
          {"round_to_nearest", CaseRoundToNearest},
          {"step_graph", CaseStepGraph},
      };
  return cases;
}

std::string Render() {
  std::vector<std::string> ops(EpFriendlyOps().begin(), EpFriendlyOps().end());
  std::vector<std::string> lines = {
      "# onnxsim QAT step-graph emitter parity fixture, format v1",
      "# Generated by scripts/make_qat_parity_fixtures.py -- do not edit by "
      "hand.",
      "# Asserted against onnxsim/qat_graph.py (tests/test_qat_parity.py) and",
      "# against onnxsim/qat_graph_builder.cpp "
      "(onnxsim/qat_graph_parity_test.cpp).",
      "ops " + Join(ops, ","),
  };
  // The autodiff's two sets, mirroring the generator. A divergence shows up
  // here as a one-line diff; TheAutodiffRuleTableMatchesTheFixture below says
  // which set and which side, since a bare diff of two sorted lists is not
  // obvious to read.
  std::vector<std::string> rules(SupportedOps().begin(), SupportedOps().end());
  std::vector<std::string> back(BackwardOps().begin(), BackwardOps().end());
  lines.push_back("rules " + Join(rules, ","));
  lines.push_back("backward_ops " + Join(back, ","));
  // The planner case's model, verbatim -- the generator writes it and
  // CasePlanner parses it back, so there is one definition of that model.
  std::istringstream model_text(g_planner_model_text);
  for (std::string line; std::getline(model_text, line);) {
    lines.push_back("model_text " + line);
  }
  for (const auto& entry : Cases()) {
    lines.push_back("case " + entry.first);
    const auto case_lines = entry.second();
    lines.insert(lines.end(), case_lines.begin(), case_lines.end());
  }
  return Join(lines, "\n") + "\n";
}

// ---------------------------------------------------------------------------

// What the C++ emitter produces must equal the committed fixture, line for
// line. A mismatch means either the port drifted from the Python emitter or
// somebody changed the Python and regenerated without updating the port; the
// diff below says which lines, and the Python-side test says which of the two
// happened.
void TheCppEmitterReproducesTheCommittedFixture() {
  std::ifstream in(QAT_PARITY_FIXTURE);
  if (!in) {
    std::cerr << "cannot open fixture " << QAT_PARITY_FIXTURE << "\n";
    ++failures;
    return;
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string expected = buffer.str();
  const std::string actual = Render();
  if (expected == actual) return;

  ++failures;
  std::cerr << "C++ emission does not match onnxsim/qat_parity_fixtures.txt\n";
  std::vector<std::string> want, got;
  for (std::stringstream ss(expected); ss.good();) {
    std::string line;
    if (!std::getline(ss, line)) break;
    want.push_back(line);
  }
  for (std::stringstream ss(actual); ss.good();) {
    std::string line;
    if (!std::getline(ss, line)) break;
    got.push_back(line);
  }
  const size_t n = std::max(want.size(), got.size());
  int shown = 0;
  for (size_t i = 0; i < n && shown < 20; ++i) {
    const std::string w = i < want.size() ? want[i] : "<missing>";
    const std::string g = i < got.size() ? got[i] : "<missing>";
    if (w != g) {
      std::cerr << "  line " << (i + 1) << "\n    fixture: " << w
                << "\n    c++    : " << g << "\n";
      ++shown;
    }
  }
}

// The allowlist is part of the contract, not just the graphs: a member present
// on one side only would let one emitter build a graph the other's own tests
// reject. The Python test pins the fixture's `ops` line to EP_FRIENDLY_OPS;
// this pins it to EpFriendlyOps().
void TheAllowlistMatchesTheFixture() {
  std::ifstream in(QAT_PARITY_FIXTURE);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("ops ", 0) != 0) continue;
    std::vector<std::string> ops(EpFriendlyOps().begin(),
                                 EpFriendlyOps().end());
    const std::string expected = "ops " + Join(ops, ",");
    if (line != expected) {
      std::cerr << "allowlist mismatch\n  fixture: " << line
                << "\n  c++    : " << expected << "\n";
      ++failures;
    }
    return;
  }
  std::cerr << "fixture has no `ops` line\n";
  ++failures;
}

// The autodiff's rule table and emittable-op set, pinned by the fixture.
//
// This is the check whose absence let Python and C++ drift apart unnoticed:
// graph_grad_test compares SupportedOps() against a list hardcoded in C++,
// which is a snapshot of the Python rather than the Python, so a rule added
// on the Python side left this side one rule short and nothing failed. It
// happened, with LayerNormalization. Comparing against the shared fixture --
// which the Python test independently pins to graph_grad.py -- makes the
// next divergence a parity failure on whichever side falls behind.
void TheAutodiffRuleTableMatchesTheFixture() {
  std::ifstream in(QAT_PARITY_FIXTURE);
  if (!in) {
    std::cerr << "cannot open fixture " << QAT_PARITY_FIXTURE << "\n";
    ++failures;
    return;
  }
  std::map<std::string, std::string> pinned;
  for (std::string line; std::getline(in, line);) {
    for (const char* key : {"rules", "backward_ops"}) {
      const std::string prefix = std::string(key) + " ";
      if (line.rfind(prefix, 0) == 0) pinned[key] = line.substr(prefix.size());
    }
  }
  const std::pair<const char*, const std::set<std::string>&> checks[] = {
      {"rules", SupportedOps()},
      {"backward_ops", BackwardOps()},
  };
  for (const auto& check : checks) {
    const auto it = pinned.find(check.first);
    if (it == pinned.end()) {
      std::cerr << "fixture has no `" << check.first
                << "` line; regenerate it with "
                   "scripts/make_qat_parity_fixtures.py\n";
      ++failures;
      continue;
    }
    std::vector<std::string> ours(check.second.begin(), check.second.end());
    const std::string expected = Join(ours, ",");
    if (it->second != expected) {
      std::cerr << check.first << " mismatch\n  fixture: " << it->second
                << "\n  c++    : " << expected << "\n";
      ++failures;
    }
  }
}

// Pulls the planner model out of the fixture's `model_text ` lines.
bool LoadPlannerModelText() {
  std::ifstream in(QAT_PARITY_FIXTURE);
  if (!in) {
    std::cerr << "cannot open fixture " << QAT_PARITY_FIXTURE << "\n";
    return false;
  }
  std::vector<std::string> parts;
  for (std::string line; std::getline(in, line);) {
    if (line.rfind("model_text ", 0) == 0) parts.push_back(line.substr(11));
  }
  if (parts.empty()) {
    std::cerr << "fixture has no model_text lines; regenerate it with "
                 "scripts/make_qat_parity_fixtures.py\n";
    return false;
  }
  g_planner_model_text = Join(parts, "\n");
  return true;
}

}  // namespace

int main() {
  if (!LoadPlannerModelText()) return 1;
  TheCppEmitterReproducesTheCommittedFixture();
  TheAllowlistMatchesTheFixture();
  TheAutodiffRuleTableMatchesTheFixture();
  if (failures) {
    std::cerr << failures << " parity check(s) failed\n";
    return 1;
  }
  std::cout << "all qat_graph parity checks passed\n";
  return 0;
}
