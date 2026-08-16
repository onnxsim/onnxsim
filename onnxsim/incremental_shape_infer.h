#pragma once

#include <string>
#include <unordered_map>

#include "onnx/onnx_pb.h"

namespace onnxsim {

// onnxsim's fixed-point loop (see ``Simplify()`` in onnxsim.cpp) alternates
// full-graph ONNX shape inference with an onnx-optimizer pass until the model
// stops changing, which can take dozens of rounds on a large graph. Each round
// re-derives shapes for *every* node from scratch via
// ``onnx::shape_inference::InferShapes``, even though a typical optimizer pass
// only rewrites a small, local part of the graph -- the rest is byte-for-byte
// identical to the previous round.
//
// ``IncrementalShapeInferer`` drives the same per-node ONNX inference
// functions (via ``onnx::shape_inference::InferenceContextImpl``, so the
// actual shape math is untouched, first-party ONNX code) but memoizes each
// node's result keyed by everything that can affect it: its op/domain/
// version, its attributes, and each input's current type *and* known
// constant value. A node whose op-signature and inputs are unchanged since a
// previous round -- the common case once the fixed point is close to
// converged -- is then a hash-map lookup instead of a re-invocation of the
// operator's inference function, turning most rounds after the first into an
// O(changed nodes) pass instead of O(all nodes).
//
// Falls back to the exact, unmodified ``onnx::shape_inference::InferShapes``
// for the handful of graph shapes this fast path does not model: any node
// with a graph-valued attribute (``If``/``Loop``/``Scan``/...), model-local
// functions, sparse initializers, and schema-defined (function-body) op
// inference. Those are rare in onnxsim's typical CNN/Transformer inputs, and
// when present correctness fully wins over speed -- the cache built on other
// calls simply goes unused that round.
class IncrementalShapeInferer {
 public:
  // Mutates `model` in place, same contract as
  // ``onnx::shape_inference::InferShapes(ModelProto&, ...)``.
  void Run(onnx::ModelProto& model);

  // Number of distinct node signatures memoized so far. Exposed for testing
  // and observability only -- callers should not rely on its exact value,
  // only on it staying flat across a Run() that changed nothing.
  size_t CacheSize() const { return cache_.size(); }

 private:
  // Node signature (see Run()) -> its output TypeProtos, length-prefixed and
  // concatenated. Persists across every Run() call on this instance, i.e. for
  // the lifetime of one fixed-point loop.
  std::unordered_map<std::string, std::string> cache_;
};

}  // namespace onnxsim
