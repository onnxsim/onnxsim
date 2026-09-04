#pragma once

// Structured (channel) and attention-head pruning entry points exposed to
// Python, mirroring pruning_entry.h's own "not a Quantize* scheme"
// rationale for its separate file. Unlike every pass in onnxsim/passes/
// (which run through onnxoptimizer's Node/Value IR via OptimizeFixed),
// both operate directly on onnx::GraphProto: the algorithm needs
// whole-graph, multi-hop forward/backward tensor-name-based analysis
// (producer/consumer maps, chain walking) that maps far more directly onto
// the same protobuf-level approach onnxsim/pruning.py's own reference
// implementation takes than onto the optimizer's PredicateBasedPass
// single-node-match model -- see structured_pruning_entry.cpp's own
// top-of-file comment for the details and scope of both ports (attention-
// head pruning lives in the same translation unit specifically to reuse
// its producer/consumer/slicing helpers directly, rather than duplicating
// them). See onnxsim.h for the documentation this mirrors.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- the full ``ModelExecutor`` interface (and the
// DLPack machinery its ``Run`` signature needs) lives in onnxsim.h, which
// itself includes *this* header (to reuse EmbeddingVocabPruningResult
// below), so including onnxsim.h back from here would be circular. A
// reference to an incomplete type is all ApplyStructuredWandaPruning's own
// declaration needs; structured_pruning_entry.cpp includes onnxsim.h itself
// for the definition, where ``executor.Run(...)`` is actually called.
struct ModelExecutor;

onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity);

onnx::ModelProto ApplyAttentionHeadPruning(const onnx::ModelProto& model,
                                           double sparsity);

// Removes intermediate (`inter_size`) channels from every expert of a
// matched `com.microsoft::MoE` node at once -- real structural pruning
// (smaller fc1/fc2 weight tensors, smaller per-expert matmuls on any
// runtime), the C++ port of pruning.py's own
// `apply_moe_expert_channel_pruning`. `num_experts` (whole-expert pruning,
// which needs runtime calibration data this build has no ONNX Runtime to
// provide -- see CLAUDE.md) is out of scope; see
// structured_pruning_entry.cpp's own "MoE (com.microsoft::MoE) expert-
// intermediate-channel pruning" section comment for the full scope and
// safety argument.
onnx::ModelProto ApplyMoeExpertChannelPruning(const onnx::ModelProto& model,
                                              double sparsity);

onnx::ModelProto ApplyQMoEExpertChannelPruning(const onnx::ModelProto& model,
                                               double sparsity);

// Removes whole experts (shrinks the `num_experts` leading axis) from a
// matched `com.microsoft::MoE` node and its upstream router projection at
// once -- the complementary technique to ApplyMoeExpertChannelPruning's own
// `inter_size` pruning. The C++ port of pruning.py's own
// `apply_moe_whole_expert_pruning`: experts are ranked by their mean router
// *gate weight* (`softmax(router_probs)` averaged over every calibration
// token, via the shared MoeRouterGateCalibrationStats helper in
// structured_pruning_entry.cpp -- see that function's own comment, and
// pruning.py's own `_moe_router_gate_calibration_stats`, for the full
// safety argument for why Softmax-then-mean, not raw logit magnitude or
// exact top-k frequency), falling back to each expert's own combined
// `fc1`/`fc2`(+`fc1_experts_bias`) L2 weight norm when a chain's
// `router_probs` was never observed during calibration (including
// `calibration_data=[]`, or a chain matched only inside a nested subgraph --
// see structured_pruning_entry.cpp's own "MoE whole-expert pruning" section
// comment). `num_experts_to_keep` is silently floored at the matched node's
// own `k` attribute (`k` itself is NEVER modified) -- pruning below `k`
// experts remaining is a hard onnxruntime execution failure, not merely
// suboptimal.
//
// Subgraph-aware (IterSubgraphs) for matching/slicing, exactly like
// ApplyMoeExpertChannelPruning -- but the calibration-based *ranking* only
// ever runs over chains matched in the TOP-LEVEL graph (mirrors
// pruning.py's own scope decision: `_add_probe_outputs` only ever appends
// to the top-level graph's own `output` list), so a chain matched only
// inside a nested If/Loop/Scan/BeamSearch-family subgraph always falls back
// to the weight-norm-only ranking above -- still correctly pruned, never
// silently skipped.
//
// Same calibration_data contract as ApplyStructuredWandaPruning: a batch
// missing one of `model`'s own graph inputs throws `std::invalid_argument`.
onnx::ModelProto ApplyMoeWholeExpertPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity);

// The quantized-weight (`com.microsoft::QMoE`) counterpart of
// ApplyMoeWholeExpertPruning -- the C++ port of pruning.py's own
// `apply_qmoe_whole_expert_pruning`. Reuses the exact same
// MoeRouterGateCalibrationStats helper unchanged (`router_probs` is QMoE's
// own second input too, upstream of and oblivious to its quantized
// `fc1`/`fc2`), and the exact same `k`-floor safety property. Unlike
// ApplyQMoEExpertChannelPruning's own `inter_size` slicing, whole-expert
// pruning needs no packed-axis unpack/re-pack anywhere: `num_experts` is
// every per-expert QMoE tensor's own LEADING axis regardless of
// `quant_type`/`block_size` (packing always lives on a later axis), so
// every `fc1`/`fc2` weight/scale/bias/zero_point (and, for
// `quant_type='nvfp4'`, `fc1`/`fc2`'s own `global_scale`) is a plain
// raw-element axis-0 index-select -- see structured_pruning_entry.cpp's own
// "QMoE whole-expert pruning" section comment.
onnx::ModelProto ApplyQMoEWholeExpertPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity);

// Return type of ApplyEmbeddingVocabPruning/ApplyEmbeddingVocabMagnitude
// Pruning -- the C++ mirror of pruning.py's own `EmbeddingPruningResult`
// dataclass (see structured_pruning_entry.cpp's own "Embedding vocabulary
// pruning" section comment for the full rationale). Unlike every other
// entry point in this file/onnxsim.h, these two passes change what counts
// as a valid model *input* (a vocabulary-pruned model only accepts
// remapped token ids), so -- exactly like the Python original -- neither
// one returns a bare `onnx::ModelProto`.
//
// Deliberately narrower than pruning.py's own dataclass: `id_map` (the
// old-token-id -> new-token-id mapping) is NOT carried across this
// boundary -- it is exactly `{kept_token_ids[i]: i for i in
// range(len(kept_token_ids))}`, trivial for the Python wrapper
// (onnx_simplifier.py) to reconstruct from `kept_token_ids` alone, so
// there is no reason to also serialize an int64->int64 map through
// nanobind. The Python wrapper builds the real, public
// `onnxsim.pruning.EmbeddingPruningResult` (the exact same dataclass the
// pure-Python entry points already return) from this struct's fields,
// rather than inventing a second, C++-only result type Python callers
// would need to know about -- one canonical return shape for both the
// pure-Python and C++-backed entry points.
struct EmbeddingVocabPruningResult {
  onnx::ModelProto model;
  bool matched = false;
  // Ascending, original-vocabulary token ids that survive. Empty when
  // `matched` is false (mirrors `kept_token_ids: Optional[List[int]] =
  // None` in Python -- an empty vector here is likewise only ever
  // meaningful when `matched` is true, since a non-empty keep-set is
  // required by both entry points below).
  std::vector<int64_t> kept_token_ids;
  bool lm_head_pruned = false;
};

// Shrinks a matched token-embedding table's vocabulary axis (a plain
// `Gather`'s `data` input feeding a graph input's token-id tensor, plus,
// where a tied or confidently-auto-identified untied `lm_head` exists, its
// own vocab-logits projection too) down to a caller-supplied, explicit
// keep-set. The C++ port of pruning.py's own `apply_embedding_vocab_
// pruning` -- give exactly one of `keep_token_ids`/`drop_token_ids`
// (`std::nullopt` means "not given"; an empty-but-present vector is a
// caller value, not "omitted"). See structured_pruning_entry.cpp's own
// "Embedding vocabulary pruning" section comment for the full matched
// topology/scope and structured_pruning_entry.cpp's own
// `MatchEmbeddingChain` for exactly what is/isn't recognized.
//
// `input_name`, when given, names which graph input (by name) the target
// `Gather`'s indices operand must resolve to -- required whenever more
// than one structurally-eligible `Gather` producer exists; a name that
// matches none throws `std::invalid_argument`. When omitted, the whole
// call declines (`matched=false`) rather than guessing if more than one
// eligible producer exists anywhere in the model (including nested
// If/Loop/Scan/BeamSearch-family subgraphs, at any depth -- see
// `IterSubgraphs`).
//
// Unlike pruning.py's own version, this port only ever matches a plain
// `Gather` producer -- not `com.microsoft::EmbedLayerNormalization` or
// `com.microsoft::GatherBlockQuantized` -- and only ever matches a bare
// `MatMul`/vanilla-`Gemm` `lm_head` -- not `com.microsoft::FusedGemm`/
// `GemmFastGelu` -- and only ever admits a plain FLOAT (float32) embedding
// table/lm_head weight/bias, not also FLOAT16/BFLOAT16, matching this
// file's own established narrower-than-pruning.py C++-port scope decision
// elsewhere (e.g. `ApplyMoeExpertChannelPruning`'s own FLOAT32-only
// restriction). See structured_pruning_entry.cpp's own section comment for
// the full list of deliberately out-of-scope shapes and why.
EmbeddingVocabPruningResult ApplyEmbeddingVocabPruning(
    const onnx::ModelProto& model,
    const std::optional<std::vector<int64_t>>& keep_token_ids,
    const std::optional<std::vector<int64_t>>& drop_token_ids,
    const std::optional<std::string>& input_name);

// The importance-ranked variant of ApplyEmbeddingVocabPruning: drops the
// lowest-L2-norm `sparsity` fraction of vocabulary rows (combined,
// root-sum-square, with a matched untied `lm_head`'s own per-row weight
// norm when one is identified), never dropping any id in
// `protect_token_ids`. The C++ port of pruning.py's own
// `apply_embedding_vocab_magnitude_pruning` -- same weaker-safety-bar
// caveat as the Python original applies here too (see that function's own
// docstring): a small row norm means small weights, not that a token is
// safe to drop from a real deployment.
EmbeddingVocabPruningResult ApplyEmbeddingVocabMagnitudePruning(
    const onnx::ModelProto& model, double sparsity,
    const std::optional<std::vector<int64_t>>& protect_token_ids,
    const std::optional<std::string>& input_name);

// The calibration-driven (Wanda-style) upgrade of ApplyStructuredPruning --
// the FIRST calibration-driven pass in this file, and the reusable pattern
// every later one (SparseGPT, MoE whole-expert, transformer-block pruning)
// is meant to build on. Mirrors pruning.py's own
// apply_structured_wanda_pruning: same chain finding as
// ApplyStructuredPruning's own plain (FindChains/FindGatedChains/
// FindConvChains/FindConvResidualChains/FindMatmulResidualChains/
// FindMatmulConcatChains/FindConvConcatChains/FindSplitGatedChains) chain
// families -- deliberately NOT the additional quantized-weight families
// (MatMulNBits, QDQ, Bnb4, Fp8/Fp4 block-quantized, QOperator,
// DynamicQuantizeMatMul) ApplyStructuredPruning also folds in, since
// pruning.py's own apply_structured_wanda_pruning has no quantized-weight
// counterpart either -- but each chain's output channels are ranked by
// ``||W_row||_2 * ||X||_2`` (weight-row/filter L2 norm times the L2 norm of
// the *calibrated activation* actually flowing through that channel) rather
// than weight magnitude alone. See structured_pruning_entry.cpp's own
// "Wanda calibration" section comment for:
//   - the calibration-crossing design (how `calibration_data` -- a plain
//     {graph input name -> TensorProto} map per batch, the same shape as
//     pruning.py's own `Sequence[Tensors]`, crossing the pybind boundary via
//     the ordinary onnx::TensorProto nanobind caster rather than a bespoke
//     bytes encoding -- gets reordered into ModelExecutor::Run's own
//     positional contract), and
//   - how the per-channel activation-norm multiplier threads into
//     ApplyChains/ApplyConcatChains' existing importance computation
//     (a new optional trailing parameter on each, defaulted to nullptr so
//     ApplyStructuredPruning's own call sites are unchanged) rather than
//     duplicating either function.
//
// Unlike ApplyStructuredPruning, this is NOT subgraph-aware (mirrors
// pruning.py's own apply_structured_wanda_pruning, which likewise never
// calls `_iter_subgraphs` and only ever prunes `model.graph` directly) --
// calibration_data batches are keyed to the top-level graph's own inputs,
// and a nested If/Loop/Scan/BeamSearch-family subgraph body has no
// standalone way to be Run at all (it depends on its enclosing loop/branch's
// own runtime state), so there is no meaningful calibrated variant of the
// subgraph recursion ApplyStructuredPruning/ApplyAttentionHeadPruning/etc.
// otherwise share.
//
// A `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument` (ModelExecutor::Run needs every positional
// input filled; there is no meaningful partial-batch fallback). An
// otherwise-empty `calibration_data` (or one where every batch's probed
// activation never resolves -- see the .cpp's own dtype/rank scope notes)
// makes every matched chain fall back to ApplyStructuredPruning's own plain
// ``||W_row||_2`` ranking, exactly mirroring pruning.py's own per-chain "no
// matching activation observed" fallback, just triggered for every chain at
// once instead of one at a time.
onnx::ModelProto ApplyStructuredWandaPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity, double epsilon = 1e-8);

// The calibration-driven (Wanda-style) upgrade of ApplyAttentionHeadPruning
// -- same relationship, and same reused pattern, as
// ApplyStructuredWandaPruning is to ApplyStructuredPruning above. Mirrors
// pruning.py's own apply_attention_head_wanda_pruning: same chain finding
// as ApplyAttentionHeadPruning's own three matched families
// (FindAttentionChains/FindGqaChains/FindOnnxAttentionChains) --
// deliberately NOT the fused `com.microsoft::MatMulNBitsQkv` variant
// ApplyAttentionHeadPruning additionally handles (via
// FindMatMulNBitsQkvChains/ApplyMatMulNBitsQkvChains), since pruning.py's
// own apply_attention_head_wanda_pruning has no quantized-weight
// counterpart either -- but each matched block's head (or, for
// GroupQueryAttention/plain ai.onnx::Attention, whole-KV-group) importance
// is ``||W||_F * ||X||_2`` -- the existing plain Frobenius-norm weight
// score times the combined (root-sum-square) L2 norm of that unit's own
// slice of the *calibrated* output-projection input activation -- rather
// than weight magnitude alone. See structured_pruning_entry.cpp's own
// "Wanda calibration" section comment (WandaCalibrationStats, reused
// as-is here: the attention output-projection's own input, at probe axis
// -1, is exactly the same "probe name -> per-channel activation L2 norm"
// shape that helper already produces for ApplyStructuredWandaPruning) for:
//   - the calibration-crossing design (unchanged from
//     ApplyStructuredWandaPruning -- same `calibration_data` shape, same
//     name -> ModelExecutor::Run-positional reordering), and
//   - how the resulting per-channel activation-norm map threads into
//     ApplyOnePlainAttentionChain/ApplyOneGqaChain's existing per-head/
//     per-group importance computation (new optional trailing parameters
//     on each, and on the ApplyAttentionChains dispatcher between them,
//     defaulted to nullptr so every ApplyAttentionHeadPruning call site is
//     unchanged) rather than duplicating the ranking/top-k/slicing logic.
//
// Unlike ApplyAttentionHeadPruning, this is NOT subgraph-aware (mirrors
// pruning.py's own apply_attention_head_wanda_pruning, which likewise
// never calls `_iter_subgraphs` and only ever prunes `model.graph`
// directly) -- same reasoning as ApplyStructuredWandaPruning's own
// declaration comment above: calibration_data batches are keyed to the
// top-level graph's own inputs, and a nested subgraph body has no
// standalone way to be Run at all.
//
// A `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument` (see ApplyStructuredWandaPruning's own
// declaration comment). An otherwise-empty `calibration_data` (or one
// where every batch's probed activation never resolves) makes every
// matched block fall back to ApplyAttentionHeadPruning's own plain
// ``||W||_F`` ranking, exactly mirroring pruning.py's own per-block "no
// matching activation observed" fallback.
onnx::ModelProto ApplyAttentionHeadWandaPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity, double epsilon = 1e-8);
