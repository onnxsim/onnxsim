#pragma once

// Model-hygiene passes and small utilities shared by Simplify()/SimplifyPath()
// (onnxsim.cpp): fingerprinting + the generic fixed-point combinator that
// drives every fixed-point loop in the simplification pipeline, schema setup
// for custom/default-domain ops, and the assorted single-purpose rewrites
// (node naming, incomplete value_info cleanup, opset conversion, CSE
// hashability, input-shape/output overwrites) that used to live inline in
// onnxsim.cpp.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "profiler.h"

// A 128-bit fingerprint of a model, used by FixedPointFn to detect when an
// iteration stopped changing the model without keeping a second full
// ModelProto around just for the comparison. Two models with the same
// fingerprint are treated as equal; the odds of a false match are ~2^-128 per
// comparison, and a false match would only stop simplification one round
// early (the model stays valid), never produce an incorrect model.
struct ModelFingerprint {
  uint64_t h1;
  uint64_t h2;
  bool operator==(const ModelFingerprint& other) const {
    return h1 == other.h1 && h2 == other.h2;
  }
};

ModelFingerprint Fingerprint(const onnx::ModelProto& model);

// Alternately apply ``f1`` and ``f2`` until the model stops changing (a joint
// fixed point) or ``max_iters`` alternations elapse. Each application produces
// a fresh model, so ``model`` is move-assigned in place and only a single
// ModelProto is held live across the loop; convergence is detected by comparing
// the fingerprints of consecutive states rather than keeping the previous
// ModelProto for a ``MessageDifferencer::Equals`` call. This mirrors the
// original consecutive-pair comparison exactly -- it stops as soon as the last
// applied function left the model unchanged -- while roughly halving the number
// of full model copies held at once (which matters because these fixed points
// nest).
// The transforms mutate the model in place (``std::function<void(T&)>``), so a
// transform that already works in place (e.g. ``_InferShapes``) makes no copy
// at all, and one that must build a fresh model (e.g. ``Optimize``, whose
// underlying ``OptimizeFixed`` returns a new proto) move-assigns it back. The
// returned function likewise mutates in place, so it composes when these fixed
// points nest and a single ModelProto is threaded through the whole thing.
template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](T& model) -> void {
    // Profiled separately from the transforms it gates: on large models this
    // convergence check is not free (see Fingerprint()'s comment), so its cost
    // should be visible in its own right rather than silently inflating
    // whichever of f1/f2's spans happens to run next. A no-op unless
    // ONNXSIM_PROFILE is set.
    auto fingerprint = [](const onnx::ModelProto& m) {
      onnxsim::ProfiledScope scope("Fingerprint");
      return Fingerprint(m);
    };
    size_t _max_iters = max_iters;
    f1(model);
    ModelFingerprint fp_prev = fingerprint(model);
    f2(model);
    ModelFingerprint fp_cur = fingerprint(model);
    while (_max_iters-- > 0) {
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f1(model);
      fp_prev = fp_cur;
      fp_cur = fingerprint(model);
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f2(model);
      fp_prev = fp_cur;
      fp_cur = fingerprint(model);
    }

    if (converged) {
      *converged = false;
    }
  };
}

template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

// Same convergence algorithm as the ``Fingerprint``-based ``FixedPointFn``
// above, specialized for transforms that can cheaply report whether they
// changed the model themselves (``f1``/``f2`` return true iff they did),
// instead of hashing the whole serialized model after every call. This skips
// ``Fingerprint()`` entirely, which matters most here because this is the
// innermost, most-frequently-run fixed point (``OptAndShape`` below runs
// every simplification round). It is only as safe as ``f1``/``f2``'s own
// signal: both onnx-optimizer's per-pass transform counts and onnx's
// InferShapes value-change count are exact for onnxsim's pass list (no
// pass with an ``Empty``/uncounted analysis type is used), so a ``false``
// return means that call provably made no change -- not just "probably".
//
// Convergence requires a whole *round* (one ``f1`` call followed by one
// ``f2`` call) to report no change from either, not just the most recent
// call: ``f1`` reporting false only means it found nothing new, not that
// ``f2``'s previous call is done making downstream cleanup available (e.g.
// a Complete-efficiency onnx-optimizer pass firing late in
// ``OptimizeGraphFixed``'s own pass-list order can leave a now-dead node
// that an earlier-ordered pass in that same call never got a chance to
// revisit -- ``f1`` correctly reports no new shape info either way, but
// ``f2`` still has cleanup work left for its *next* call). Stopping right
// after a lone ``f1`` false, as an earlier version of this function did,
// silently left that cleanup undone whenever nothing above this loop
// happened to re-drive it; matching a whole round instead of a single call
// closes that gap unconditionally, with no dependence on some caller
// incidentally running this again.
template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<bool(T&)>& f1,
                                     const std::function<bool(T&)>& f2,
                                     size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](T& model) -> void {
    size_t _max_iters = max_iters;
    while (_max_iters-- > 0) {
      const bool c1 = f1(model);
      const bool c2 = f2(model);
      if (!c1 && !c2) {
        if (converged) {
          *converged = true;
        }
        return;
      }
    }

    if (converged) {
      *converged = false;
    }
  };
}

template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<bool(T&)>& f1,
                                     const std::function<bool(T&)>& f2,
                                     size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

// A no-op in-place transform (mutates nothing), used when shape inference or
// constant folding is disabled.
void Identity(onnx::ModelProto&);

// Register a permissive placeholder schema for every default-domain custom op
// found in ``model``. Without a schema, onnx::checker::check_model rejects the
// model with "No Op registered for <op> with domain_version of <n>" and
// simplification never even starts (GitHub issues #107 and #220).
void RegisterCustomDefaultDomainOpSchemas(const onnx::ModelProto& model);

// Assign names to any nodes left nameless after simplification (issue #269).
// Returns the names generated, in assignment order, so a caller can record
// which nodes had no author-given name (e.g. as metadata_props).
std::vector<std::string> AssignMissingNodeNames(onnx::ModelProto& model);

// Drop value_info entries left with an UNDEFINED element type (see this
// function's own doc comment in model_prep.cpp for why).
void DropIncompleteValueInfo(onnx::ModelProto& model);

// Raise model.ir_version to the minimum the model's tensor element types need
// (e.g. INT4/UINT4 need IR >= 10). onnx's checker rejects a model whose
// ir_version predates a data type it uses, but such exports exist in the wild
// (an IR 8 file carrying uint4 MatMulNBits weights); the data is unchanged and
// the type is valid in the newer IR, so bump rather than fail.
void RaiseIrVersionForDataTypes(onnx::ModelProto& model);

void Check(const onnx::ModelProto& model);

// Check() for the model handed to Simplify(): tolerates an ir_version older
// than its tensor types need (see RaiseIrVersionForDataTypes) without
// modifying ``model``.
void CheckInput(const onnx::ModelProto& model);

// Return the opset version the model imports for the default ONNX domain
// (represented as either the empty string or "ai.onnx"), or std::nullopt when
// the model does not import the default domain at all (a model made purely of
// custom-domain operators).
std::optional<int> DefaultOpsetVersion(const onnx::ModelProto& model);

// Mirrors onnx_simplifier.py's remove_initializer_from_input.
void RemoveInitializerFromInput(onnx::ModelProto& model);

// Mirrors onnx_simplifier.py's _has_cse_unhashable_tensor: walks every tensor
// CSE might hash looking for an element type onnxoptimizer's tensor-value
// hashing cannot handle.
bool GraphHasCSEUnhashableTensor(const onnx::GraphProto& graph);

// Mirrors onnx_simplifier.py's overwrite_input_shapes loop.
void ApplyInputShapeOverwrite(
    onnx::ModelProto& model,
    const std::unordered_map<std::string, std::vector<int64_t>>&
        overwrite_input_shapes);

// Mirrors onnx_simplifier.py's remove_unused_output.
void RemoveUnusedOutputs(onnx::ModelProto& model,
                         const std::vector<std::string>& unused_output);

// Convert the default ONNX domain of the model to target_version using onnx's
// own version converter. Takes ``model`` by value so a caller who no longer
// needs their copy can ``std::move`` it in and get a move-only, non-copying
// conversion; see the .cpp for why.
onnx::ModelProto ConvertOpsetVersion(onnx::ModelProto model,
                                     int target_version);

// Shared schema setup for the single-pass debug helpers and the quantization
// entry points: teach shape inference about ONNX Runtime's quantized contrib
// ops, correct constant-folding determinism metadata, and register permissive
// placeholders for custom ops exported into the default ONNX domain, so
// neither shape inference nor a later checker call rejects the model.
void PrepareSchemasForDebug(const onnx::ModelProto& model);
