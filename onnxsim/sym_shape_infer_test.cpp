// Standalone test (all four .cpp on one g++ line):
//   g++ -std=c++17 sym_expr.cpp sym_value_eval.cpp sym_shape_infer.cpp
//       sym_shape_infer_test.cpp -o t && ./t
// (Builds identically under Emscripten.)
//
// Exercises the M2 symbolic activation-shape inference (issue #532): compute
// ops must carry the batch/sequence symbol so that Shape(intermediate) -- which
// M1 folds -- comes back clean. Assembled from the plain ShapeGraph structs the
// real adapter (M3) will fill; no ONNX.
#include "sym_shape_infer.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using onnxsim::ShapeGraph;
using onnxsim::SymAttr;
using onnxsim::SymExpr;
using onnxsim::SymNode;
using onnxsim::SymShape;
using onnxsim::SymTensor;

namespace {

SymAttr AInt(std::string name, int64_t v) {
  SymAttr a;
  a.name = std::move(name);
  a.i = v;
  return a;
}
SymAttr AInts(std::string name, std::vector<int64_t> v) {
  SymAttr a;
  a.name = std::move(name);
  a.ints = std::move(v);
  return a;
}

SymNode N(std::string op, std::vector<std::string> in,
          std::vector<std::string> out, std::vector<SymAttr> attrs = {}) {
  SymNode n;
  n.op_type = std::move(op);
  n.input = std::move(in);
  n.output = std::move(out);
  n.attribute = std::move(attrs);
  return n;
}

SymTensor IVec(std::vector<int64_t> xs) {
  std::vector<SymExpr> d;
  for (int64_t x : xs) d.emplace_back(x);
  return SymTensor::Vector(std::move(d));
}

const std::map<std::string, SymShape>* g_result = nullptr;

// A tensor's inferred shape equals this list of sympy-style dim strings.
bool ShapeIs(const std::string& name, std::vector<std::string> expect) {
  auto it = g_result->find(name);
  if (it == g_result->end()) return false;
  const SymShape& s = it->second;
  if (s.size() != expect.size()) return false;
  for (std::size_t i = 0; i < s.size(); ++i)
    if (s[i].str() != expect[i]) return false;
  return true;
}

// The dim at `pos` of `name` is a fresh "unk_" placeholder symbol.
bool DimIsFresh(const std::string& name, std::size_t pos) {
  auto it = g_result->find(name);
  if (it == g_result->end() || pos >= it->second.size()) return false;
  const std::string s = it->second[pos].str();
  return s.rfind("unk_", 0) == 0;
}

const SymExpr batch = SymExpr::Symbol("batch");
const SymExpr seq = SymExpr::Symbol("seq");

}  // namespace

int main() {
  // === A transformer feed-forward block carries batch & seq through =========
  {
    ShapeGraph g;
    g.value_info["x"] = {batch, seq, SymExpr(768)};
    g.value_info["W"] = {SymExpr(768),
                         SymExpr(3072)};  // weight (initializer shape)
    g.value_info["bias"] = {SymExpr(3072)};
    g.node = {
        N("MatMul", {"x", "W"}, {"mm"}),  // [batch, seq, 3072]
        N("Add", {"mm", "bias"},
          {"biased"}),                   // broadcast -> [batch, seq, 3072]
        N("Gelu", {"biased"}, {"act"}),  // unary -> same
        N("LayerNormalization", {"act"}, {"ln"}),  // unary -> same
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("mm", {"batch", "seq", "3072"}));
    assert(ShapeIs("biased", {"batch", "seq", "3072"}));
    assert(ShapeIs("act", {"batch", "seq", "3072"}));
    assert(ShapeIs("ln", {"batch", "seq", "3072"}));
  }

  // === Scaled-dot-product attention: batched MatMul + Softmax ===============
  {
    ShapeGraph g;
    g.value_info["q"] = {batch, SymExpr(12), seq, SymExpr(64)};
    g.value_info["kT"] = {batch, SymExpr(12), SymExpr(64), seq};
    g.node = {
        N("MatMul", {"q", "kT"}, {"scores"}),  // [batch, 12, seq, seq]
        N("Softmax", {"scores"}, {"attn"}, {AInt("axis", -1)}),
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("scores", {"batch", "12", "seq", "seq"}));
    assert(ShapeIs("attn", {"batch", "12", "seq", "seq"}));
  }

  // === Conv patch-embedding with a dynamic batch (AST/ViT pattern) ==========
  {
    ShapeGraph g;
    g.value_info["img"] = {batch, SymExpr(3), SymExpr(224), SymExpr(224)};
    g.value_info["kern"] = {SymExpr(768), SymExpr(3), SymExpr(16), SymExpr(16)};
    g.node = {N("Conv", {"img", "kern"}, {"emb"},
                {AInts("strides", {16, 16}), AInts("pads", {0, 0, 0, 0})})};
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    // (224 - 16)/16 + 1 = 14 on each spatial axis; batch stays dynamic.
    assert(ShapeIs("emb", {"batch", "768", "14", "14"}));
  }

  // === Conv1d over a symbolic length: batch & channels kept, length fresh ===
  {
    ShapeGraph g;
    g.value_info["feat"] = {batch, SymExpr(512), seq};
    g.value_info["w"] = {SymExpr(512), SymExpr(512), SymExpr(3)};
    g.node = {N("Conv", {"feat", "w"}, {"out"}, {AInts("strides", {2})})};
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    auto it = r.find("out");
    assert(it != r.end() && it->second.size() == 3);
    assert(it->second[0].str() == "batch");
    assert(it->second[1].str() == "512");
    assert(
        DimIsFresh("out", 2));  // strided conv over `seq` is not a polynomial
  }

  // === Pooling + global pool ===============================================
  {
    ShapeGraph g;
    g.value_info["x"] = {batch, SymExpr(64), SymExpr(56), SymExpr(56)};
    g.node = {
        N("MaxPool", {"x"}, {"p"},
          {AInts("kernel_shape", {2, 2}), AInts("strides", {2, 2})}),
        N("GlobalAveragePool", {"p"}, {"g"}),
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("p", {"batch", "64", "28", "28"}));
    assert(ShapeIs("g", {"batch", "64", "1", "1"}));
  }

  // === Gemm with transB (classifier head) ==================================
  {
    ShapeGraph g;
    g.value_info["feat"] = {batch, SymExpr(768)};
    g.value_info["W"] = {SymExpr(1000), SymExpr(768)};
    g.node = {N("Gemm", {"feat", "W", "b"}, {"logits"}, {AInt("transB", 1)})};
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("logits", {"batch", "1000"}));
  }

  // === Gather embedding: ids[batch, seq] into a [V, 768] table ==============
  {
    ShapeGraph g;
    g.value_info["table"] = {SymExpr(30522), SymExpr(768)};
    g.value_info["ids"] = {batch, seq};
    g.node = {N("Gather", {"table", "ids"}, {"emb"}, {AInt("axis", 0)})};
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("emb", {"batch", "seq", "768"}));
  }

  // === Concat along a symbolic axis sums the dim ============================
  {
    ShapeGraph g;
    g.value_info["cls"] = {batch, SymExpr(1), SymExpr(768)};
    g.value_info["tok"] = {batch, seq, SymExpr(768)};
    g.node = {N("Concat", {"cls", "tok"}, {"c"}, {AInt("axis", 1)})};
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("c", {"batch", "seq + 1", "768"}));
  }

  // === Flatten and Transpose ===============================================
  {
    ShapeGraph g;
    g.value_info["x"] = {batch, SymExpr(768), SymExpr(7), SymExpr(7)};
    g.node = {
        N("Flatten", {"x"}, {"f"}, {AInt("axis", 1)}),  // [batch, 768*49]
        N("Transpose", {"x"}, {"t"}, {AInts("perm", {0, 2, 3, 1})}),
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("f", {"batch", "37632"}));  // 768*7*7
    assert(ShapeIs("t", {"batch", "7", "7", "768"}));
  }

  // === Reshape: -1 resolved via exact division; 0 copies the input dim =====
  {
    ShapeGraph g;
    g.value_info["x"] = {batch, SymExpr(197), SymExpr(768)};
    g.initializer["flat"] = IVec({-1, 768});
    g.initializer["copy"] = IVec({0, -1});
    g.node = {
        N("Reshape", {"x", "flat"}, {"r1"}),  // [-1, 768] -> [197*batch, 768]
        N("Reshape", {"x", "copy"}, {"r2"}),  // [0, -1]  -> [batch, 151296]
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("r1", {"197*batch", "768"}));
    assert(ShapeIs("r2", {"batch", "151296"}));  // 197*768
  }

  // === Squeeze / Unsqueeze via axes ========================================
  {
    ShapeGraph g;
    g.value_info["x"] = {batch, SymExpr(1), SymExpr(768)};
    g.initializer["ax"] = IVec({1});
    g.node = {
        N("Squeeze", {"x", "ax"}, {"sq"}),     // -> [batch, 768]
        N("Unsqueeze", {"sq", "ax"}, {"un"}),  // -> [batch, 1, 768]
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("sq", {"batch", "768"}));
    assert(ShapeIs("un", {"batch", "1", "768"}));
  }

  // === ReduceMean over the channel axis (keepdims) =========================
  {
    ShapeGraph g;
    g.value_info["x"] = {batch, seq, SymExpr(768)};
    g.node = {N("ReduceMean", {"x"}, {"m"},
                {AInts("axes", {-1}), AInt("keepdims", 1)})};
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("m", {"batch", "seq", "1"}));
  }

  // === Slice with concrete bounds; symbolic axis stays fresh ===============
  {
    ShapeGraph g;
    g.value_info["x"] = {batch, SymExpr(197), SymExpr(768)};
    g.initializer["st"] = IVec({0, 1});
    g.initializer["en"] = IVec({1, 197});
    g.initializer["ax"] = IVec({0, 1});
    g.node = {N("Slice", {"x", "st", "en", "ax"}, {"s"})};
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    // axis 0 is symbolic (batch) -> fresh; axis 1: [1:197] -> 196.
    assert(DimIsFresh("s", 0));
    auto it = r.find("s");
    assert(it != r.end() && it->second[1].str() == "196" &&
           it->second[2].str() == "768");
  }

  // === ScatterND / ScatterElements keep the data (input 0) shape ===========
  // The causal-mask subgraph in BART reaches a Reshape through
  // Shape(ScatterND(...)); the scatter must carry `data`'s shape so that Shape
  // resolves instead of stalling.
  {
    ShapeGraph g;
    g.value_info["data"] = {seq, seq};
    g.value_info["indices"] = {seq, SymExpr(2)};
    g.value_info["updates"] = {seq};
    g.node = {
        N("ScatterND", {"data", "indices", "updates"}, {"sn"}),
        N("ScatterElements", {"data", "indices", "updates"}, {"se"},
          {AInt("axis", 0)}),
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("sn", {"seq", "seq"}));
    assert(ShapeIs("se", {"seq", "seq"}));
  }

  // === MatMul rank-1 operand handling (onnxsim issue #1275) ================
  // numpy matmul promotes a rank-1 lhs/rhs with a leading/trailing 1, then
  // strips it back out of the result. Getting the "strip" step wrong for the
  // rank-1 x rank-1 case (both promoted dims removed, leaving a 0-D scalar)
  // used to erase past the front of the output vector.
  {
    ShapeGraph g;
    g.value_info["v"] = {SymExpr(768)};
    g.value_info["m"] = {SymExpr(768), SymExpr(1000)};
    g.node = {
        // rank-1 x rank-1 (dot product): self-MatMul is the exact shape NNSmith
        // generated for issue #1275 -- both promoted dims are stripped, so the
        // result is a 0-D scalar.
        N("MatMul", {"v", "v"}, {"dot"}),
        N("MatMul", {"v", "m"}, {"vm"}),  // rank-1 x rank-2 -> [1000]
    };
    const auto r = onnxsim::InferSymbolicShapes(g);
    g_result = &r;
    assert(ShapeIs("dot", {}));
    assert(ShapeIs("vm", {"1000"}));
  }

  std::cout << "sym_shape_infer_test: all assertions passed\n";
  return 0;
}
