#pragma once

// Calibration-driven importance-matrix (imatrix) weight quantization entry
// point exposed to Python -- C++ port of onnxsim.imatrix_quant's own
// apply_imatrix_quantization/compute_activation_importance (see
// onnxsim/imatrix_quant.py's module docstring for the full technique).
//
// Like structured_pruning_entry.h's own ApplyWandaPruning/
// ApplySparseGptPruning (the closest existing precedent for a *calibration*-
// driven pass in this codebase), this operates directly on onnx::GraphProto
// rather than through onnxoptimizer's Node/Value IR via a
// PredicateBasedPass/OptimizeFixed: threading a live ModelExecutor plus a
// batch of calibration data through OptimizeFixed's single-node-match
// PredicateBasedPass model has no established path in this codebase (every
// PredicateBasedPass in onnxsim/passes/, e.g. passes/quarot.h, is data-free
// by construction), whereas the protobuf-level, "match candidates once, run
// the executor once, rewrite each candidate's own initializer" shape below
// is exactly ApplyWandaPruning's own, already-proven pattern for this exact
// class of problem -- see structured_pruning_entry.h's own top-of-file
// comment for the general rationale this mirrors.
//
// This is a deliberate, reasoned departure from apply_quarot/
// apply_quarot_cpp's PredicateBasedPass shape specifically because
// apply_quarot is data-free (no calibration_data parameter at all); a
// calibration-driven pass in this codebase has never used that shape.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration only -- see structured_pruning_entry.h's own
// identical forward declaration for why (the full ModelExecutor interface
// lives in onnxsim.h, which includes this header back).
struct ModelExecutor;

// Weight-only-quantizes every matched MatMul/vanilla-Gemm node's constant
// 2-D FLOAT32 weight to INT4 (folded as a float32 quantize-dequantize round
// trip, matching ApplyIQ4NL's own "no new tensor type, no new graph nodes"
// convention), using real calibration activations (run through `executor`)
// to bias each weight block's scale search toward minimizing
// importance-weighted squared error -- see onnxsim/imatrix_quant.py's own
// module docstring for the technique and onnxsim/passes/imatrix_quant.h for
// the quantizer itself.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase (structured_pruning_entry.h's own ApplyWandaPruning/
// ApplySparseGptPruning) -- a `calibration_data` batch missing one of
// `model`'s own graph inputs throws `std::invalid_argument`. NOT
// subgraph-aware, matching every one of those passes' own scope decision:
// calibration_data batches are keyed to the top-level graph's own inputs
// only.
//
// A candidate whose activation was never observed as a FLOAT32 tensor
// across `calibration_data`, or whose reduction dimension `K` is not a
// multiple of `block_size`, is left unquantized -- mirrors
// apply_imatrix_quantization's own per-layer skip conditions exactly.
// `skip_names`: weight initializer names to leave unquantized even if
// otherwise eligible.
onnx::ModelProto ApplyImatrixQuantization(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size, int64_t num_scale_candidates, double scale_lo,
    double scale_hi, const std::unordered_set<std::string>& skip_names);
