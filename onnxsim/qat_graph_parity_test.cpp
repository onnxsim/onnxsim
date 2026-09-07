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

#include "qat_graph_builder.h"

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

// A tensor's values, whichever field the producer chose to put them in.
// Encoding is deliberately not part of the comparison: float_data and raw_data
// are indistinguishable to a runtime, so the fixture compares numbers.
std::vector<std::string> TensorValues(const onnx::TensorProto& t) {
  std::vector<std::string> out;
  if (t.data_type() == onnx::TensorProto::INT64) {
    if (t.int64_data_size() > 0) {
      for (int i = 0; i < t.int64_data_size(); ++i) {
        out.push_back(std::to_string(t.int64_data(i)));
      }
    } else {
      const auto* raw = reinterpret_cast<const int64_t*>(t.raw_data().data());
      const size_t n = t.raw_data().size() / sizeof(int64_t);
      for (size_t i = 0; i < n; ++i) out.push_back(std::to_string(raw[i]));
    }
    return out;
  }
  if (t.float_data_size() > 0) {
    for (int i = 0; i < t.float_data_size(); ++i) {
      out.push_back(F32Bits(t.float_data(i)));
    }
  } else {
    const auto* raw = reinterpret_cast<const float*>(t.raw_data().data());
    const size_t n = t.raw_data().size() / sizeof(float);
    for (size_t i = 0; i < n; ++i) out.push_back(F32Bits(raw[i]));
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

}  // namespace

int main() {
  TheCppEmitterReproducesTheCommittedFixture();
  TheAllowlistMatchesTheFixture();
  if (failures) {
    std::cerr << failures << " parity check(s) failed\n";
    return 1;
  }
  std::cout << "all qat_graph parity checks passed\n";
  return 0;
}
