#pragma once

// LoRA/QLoRA adapter injection and training, reachable without Python.
//
// onnxsim/lora.py is the original: it injects a trainable low-rank
// ``X @ A @ B`` branch around a frozen weight and trains ``A``/``B`` with
// onnxsim/graph_grad.py's hand-rolled reverse-mode autodiff -- the same
// machinery onnxsim/qat.py already uses for block-wise QAT, and the same
// split qat_entry.h ports for that module applies here: the graph-building
// half is pure graph surgery / dataflow and is what this header ports; the
// driving half (run the step graph in a loop, capture activations) is not
// here and should not be, because the caller already has an inference
// runtime -- onnxruntime-web in the browser -- and running a graph in a loop
// is that runtime's job.
//
// **Scope, and why it is wider here than qat_entry.h's.** qat_entry.h takes
// an already-quantized model as input and never ports the quantization step
// itself. That is not an arbitrary line: quantize_weight_only_int4,
// quantize_static and friends already have their own WASM-reachable C++
// entry points (onnxsim/quantize_entry.h, bound in
// scripts/convertmodel/interface.cpp as onnxsim_quantize_weight_only_int4
// etc.), so qat_entry.h porting them again would be pure duplication -- the
// browser converter page calls one of those first, then hands the result to
// onnxsim_qat_build_step_graph. LoRA's injection step (onnxsim/lora.py's
// inject_lora) has **no such C++ precedent**: it is graph surgery private to
// this module, not a quantizer with its own entry point elsewhere. So unlike
// QAT's step-graph builder, this header also ports the injection itself
// (InjectLora, below) -- otherwise the browser would have a complete QLoRA
// *training* story and no way to have produced the adapter it trains.
// QLoRA's other half, onnxsim.nf4.quantize_weight_only_nf4, has no C++ port
// either and is NOT added here: it is a separate quantization scheme (see
// quantize_entry.h's own precedent) and belongs there if it is ever ported,
// not folded into a LoRA header. What composes an NF4-quantized base with
// LoRA training here is narrower than it might sound, and squarely
// BuildLoraStepGraph's own concern: SliceBlock (see that function's own
// comment) treats the frozen dequant chain as an ordinary block-external
// tensor to capture, exactly like the block's own input, which is what
// keeps the chain's non-differentiable Cast out of graph_grad::BuildBackward
// in the first place. lora_entry.cpp also ports lora.py's own
// _fold_frozen_prefixes (as FoldFrozenPrefixes) for parity, though
// BuildLoraStepGraph's own comment there records a finding: given how
// SliceBlock's reachability works, that function can never actually have
// anything to fold -- it is captured, not folded, and this holds in both
// languages today.
//
// The intended browser flow, for a model without an existing adapter:
//   1. InjectLora(model, options) -> an injected model + a LoraAdapter plan
//   2. BuildLoraStepGraph(injected, adapter, block in/out, rows, opts)
//        -> a step graph, the state ping-pong map, the list of activations
//           to capture, and the initial state (the adapter's *current*
//           A/B values, so a caller can resume a partially trained adapter)
//   3. the caller captures those activations (ort-web) and the block's
//      reconstruction target (from a reference/teacher model, or supplied
//      directly), binds them plus the initial state, and runs the step
//      graph `n` times, feeding the per-step scalars and carrying state
//      outputs back to inputs
//   4. WriteBackLoraState(injected, plan, final state) -> the tuned model
//
// **Parity with the Python is a hard requirement for the graphs each side
// emits**, for the reason qat_entry.h gives: two implementations that
// disagree would train a model differently in the browser than in Python
// and nothing would say so. Unlike qat_graph_builder.cpp, BuildLoraStepGraph
// introduces no emission primitive of its own -- no fake-quant, no
// broadcast-scale, no activation quantizer -- it is a block's own nodes
// verbatim, graph_grad::BuildBackward restricted to the adapter's own
// tensors, and one qat_graph_builder::AdamUpdate per tensor, all of which
// onnxsim/qat_parity_fixtures.txt's "arithmetic"/"adam_update"/
// "gather_rows"/"planner"/"step_graph" cases already pin at the primitive
// level (Sub, Mul, MatMul's VJP, AdamUpdate, GatherRows, MeanSquare). A
// LoRA-specific fixture would be exercising the same already-pinned
// primitives through a different (but not independently risky) composition,
// so none is added here; lora_entry_test.cpp instead checks the composition
// structurally -- the emitted step graph is checker-valid, has the right
// shape/op-type/state/capture/initial-state contract -- exactly as
// qat_entry_test.cpp does for BuildQatStepGraph, and for the same reason
// (nothing in this build evaluates a graph numerically; see that file's own
// comment). Numeric correctness of the injected branch and its gradient is
// tests/test_lora.py's job, already covered there against torch.autograd.
//
// **One deliberate, documented non-parity.** InjectLora's ``A`` matrix is
// Kaiming-normal-initialized from a caller-given seed, exactly as
// inject_lora's is -- but this port draws it from a C++
// std::mt19937_64/std::normal_distribution stream, not from numpy's
// PCG64-backed default_rng, so the *same seed produces different A values*
// in the two languages. This is the same kind of gap qat_entry.h's
// QatOptions::batch_seed already documents and for the same underlying
// reason: reproducing numpy's exact bit stream in C++ is a real undertaking
// unrelated to what either random draw is *for*. It matters less here than
// it might sound: ``B`` always starts at zero (so injection is a numeric
// no-op regardless of what ``A`` drew), and ``A``'s specific initial values
// are training noise, not a claim the trained model depends on -- nothing
// downstream needs this port's A to equal Python's A, only for A to be a
// reasonable Kaiming-scaled random start. A caller that needs the two to
// agree bit-for-bit (a golden-model test, say) should inject in Python and
// hand this header the already-injected model instead of calling
// InjectLora itself.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "qat_graph_builder.h"

// ---------------------------------------------------------------------------
// InjectLora -- the C++ port of onnxsim/lora.py's inject_lora
// ---------------------------------------------------------------------------

// One injected adapter, tied to the base weight it augments. Mirrors
// lora.py's LoraTarget field for field.
struct LoraTarget {
  // The frozen base initializer name. Never modified by injection or
  // training -- the whole point of the low-rank branch.
  std::string weight_name;
  // The original node's output tensor name -- what the closing Add
  // restores, so every existing downstream consumer needs no changes.
  std::string node_output;
  std::string op_type;  // "MatMul" | "Gemm" | "Conv"
  // New initializer name, "<weight_name>.lora_A".
  std::string lora_a_name;
  // New initializer name, "<weight_name>.lora_B".
  std::string lora_b_name;
  int64_t rank = 0;
  // alpha is meaningful only when has_alpha is set -- mirrors lora.py's
  // Optional[float]. With it unset the branch is added unscaled.
  bool has_alpha = false;
  float alpha = 0.0f;
};

// Every adapter one InjectLora call injected. Mirrors lora.py's LoraAdapter.
struct LoraAdapter {
  std::vector<LoraTarget> targets;

  // Every lora_A/lora_B initializer name, in injection order -- the
  // `targets` BuildLoraStepGraph hands graph_grad::BuildBackward, and the
  // skip list a caller composing QLoRA hands its own NF4 quantizer (see
  // this header's own top comment on why that quantizer is not ported
  // here). Mirrors lora.py's LoraAdapter.parameter_names().
  std::vector<std::string> ParameterNames() const;
};

struct InjectLoraOptions {
  int64_t rank = 8;
  // alpha is meaningful only when has_alpha is set -- see LoraTarget::alpha.
  bool has_alpha = false;
  float alpha = 0.0f;
  // Restrict injection to these op types. Defaults to every eligible type,
  // matching lora.py's own default.
  std::vector<std::string> target_op_types{"MatMul", "Gemm", "Conv"};
  // With this false (the default), every eligible node is injected --
  // lora.py's target_names=None. With it true, only nodes whose weight
  // initializer name appears in `target_names` are injected -- including
  // the case where `target_names` is itself empty, which (like the
  // Python's target_names=[]) injects nothing. The two are not the same
  // question a bare "restrict to this list" would collapse: None and []
  // are different requests in lora.py, and this flag is what keeps them
  // distinguishable in a language with no optional-vs-empty distinction on
  // a plain vector.
  bool restrict_target_names = false;
  std::vector<std::string> target_names;
  // Seeds A's Kaiming-normal initialization. See this header's top comment
  // for why this port's random stream does not match numpy's for the same
  // seed. B always starts at zero, so injection is a numeric no-op
  // regardless.
  uint64_t seed = 0;
};

struct LoraInjectionResult {
  onnx::ModelProto model;
  LoraAdapter adapter;
};

// Injects a trainable low-rank adapter branch around every eligible
// MatMul/Gemm/Conv weight, leaving the base weight itself untouched and
// every other byte of `model` unchanged.
//
// Eligible: MatMul/Gemm with a 2-D float32 initializer at input[1]; Conv
// with a 4-D float32 initializer, kernel_shape == [1, 1] and group == 1 --
// the same conditions lora.py's inject_lora checks, which mirror
// tools/onnx-finetune's own lora_surgery.py rather than
// onnxsim.qat's looser _find_float_layers (which admits any-shape Conv, a
// shape a low-rank branch cannot represent).
//
// Throws onnx::checker::ValidationError (via onnx::checker::check_model on
// the result, matching inject_lora's own final check) if the injected
// model is somehow invalid -- which should not happen for a valid input,
// and exists as the same defense-in-depth the Python has.
LoraInjectionResult InjectLora(const onnx::ModelProto& model,
                               const InjectLoraOptions& options = {});

// ---------------------------------------------------------------------------
// DiscoverLoraBlocks -- the C++ port of onnxsim/lora.py's
// discover_lora_blocks, which itself reuses onnxsim/qat.py's
// _liveness_cuts/_primary_graph_input
// ---------------------------------------------------------------------------

// One trainable block DiscoverLoraBlocks found, named the way a caller would
// name one for BuildLoraStepGraph by hand. Mirrors lora.py's LoraBlock field
// for field.
struct LoraBlock {
  std::string input_name;
  std::string output_name;
  std::vector<std::string> target_outputs;   // never empty
  std::vector<std::string> external_inputs;  // sorted, input_name included
  std::vector<std::string> op_types;         // deduplicated, sorted
  int64_t num_nodes = 0;
};

// Partitions `injected_model` into a sequence of blocks BuildLoraStepGraph
// can train, without a caller naming a single block_input_name/
// block_output_name pair by hand -- the C++ port of lora.py's own
// discover_lora_blocks, which itself reuses qat.py's _liveness_cuts /
// _primary_graph_input verbatim (ported here as the same-named
// anonymous-namespace helpers in lora_entry.cpp). Those two functions'
// docstrings carry the actual argument and are not repeated here, on this
// header's own standing policy of one place to update when a derivation
// changes -- but in one sentence: a slice is trainable iff it is
// **differentiable** (every op inside is in graph_grad::SupportedOps()) and
// **self-contained** (the graph narrows to exactly one live tensor at both
// of its boundaries, so cutting it out there severs nothing else still in
// use). _liveness_cuts finds every such boundary by a liveness argument, not
// by recognizing architectures -- initializers and every non-primary graph
// input are excluded from the live set, so a second input (an attention
// mask, say) never suppresses a cut the way it would if it were counted as
// an ordinary activation; _primary_graph_input picks which input keeps that
// power, by reachability, ties going to the earlier input. Neither can see a
// tensor computed purely from initializers for what it is -- such a tensor
// is counted as an ordinary live activation and so suppresses cuts across
// its own live range -- but that is conservative (fewer, larger blocks, or
// none), not incorrect.
//
// This walks consecutive cut pairs in graph order, treats any node in a span
// whose op type is not in SupportedOps() as a hard gap that closes whatever
// block was pending and reopens after it, and otherwise accumulates that
// span's node outputs that are one of `adapter`'s own
// LoraTarget::node_output tensors -- the same tensor InjectLora restored the
// original node's name to -- closing a block once that running count
// reaches `max_targets_per_block`. A final pending block, if any, is closed
// against the last cut once the walk ends.
//
// This does NOT call FoldFrozenPrefixes: like lora.py's own
// discover_lora_blocks, it has no notion of "frozen", so it treats every
// unsupported op as a hard gap even where BuildLoraStepGraph could in
// principle train through it by capturing it as a block-external constant
// (an NF4 dequant chain's Cast, say -- see BuildLoraStepGraph's own comment
// on why SliceBlock captures rather than folds it). That is conservative,
// not incorrect: run this against an apply_qlora-composed model and it may
// propose fewer or smaller blocks than a hand-named BuildLoraStepGraph call
// could actually train; it never proposes one that cannot train.
//
// Throws std::invalid_argument when `max_targets_per_block < 1`, message
// mirroring lora.py's own ValueError ("max_targets_per_block must be at
// least 1"). Boundaries are found in `injected_model`'s own graph -- the one
// BuildLoraStepGraph differentiates -- so pass the model InjectLora/
// apply_qlora produced, not the original float model.
//
// Returns the blocks in graph order, possibly empty. Consecutive blocks need
// not be adjacent: a gap between two of them is a region nothing here can
// train.
std::vector<LoraBlock> DiscoverLoraBlocks(
    const onnx::ModelProto& injected_model, const LoraAdapter& adapter,
    int64_t max_targets_per_block = 2);

// ---------------------------------------------------------------------------
// BuildLoraStepGraph -- the C++ port of the graph-building half of
// onnxsim/lora.py's _build_lora_step_graph / _train_lora_block
// ---------------------------------------------------------------------------

// Which minibatch schedule a run uses. Mirrors QatOptions' own fields and
// their reasoning exactly (see qat_entry.h): batch_size 0 means full batch;
// shuffle/batch_seed are carried for a caller's own minibatch row-selection
// and are not read by BuildLoraStepGraph, since the driving loop that would
// consult them is not here.
struct LoraOptions {
  int64_t batch_size = 0;
  int64_t batch_seed = 0;
  bool shuffle = true;
};

// One tensor the caller must capture from the model being trained (or, for
// the reconstruction target, from a teacher model or its own labels) and
// bind for the life of the loop. Identical in shape and purpose to
// qat_entry.h's QatCapture; see that struct's own comments.
struct LoraCapture {
  std::string step_graph_input;
  std::string source_tensor;
  std::vector<int64_t> dims;
  int32_t elem_type = onnx::TensorProto::FLOAT;
  // True for the block's reconstruction target.
  bool is_teacher = false;
};

// Everything needed to run the loop and then write its result back. Mirrors
// qat_entry.h's QatStepPlan, minus everything that exists there only for
// the two quantized schemes (no quantizer state, no code/scale write-back
// projection) -- LoRA's write-back is the identity on A/B's own values.
struct LoraStepPlan {
  onnx::ModelProto step_graph;
  // Input name -> the output carrying its next value.
  std::vector<std::pair<std::string, std::string>> state;
  // Scalar float inputs supplied fresh each step:
  //   "lora__lr"       the adapter's Adam learning rate
  //   "m_correction"   Adam's first-moment bias correction, 1/(1 - beta1^(t+1))
  //   "v_correction"   ... and its second, 1/(1 - beta2^(t+1))
  // qat_graph_builder.h's AdamBiasCorrections computes the two corrections.
  std::vector<std::string> scalars;
  // The scalar output carrying this step's loss. Empty if none.
  std::string loss_name;
  // Bound once, for the life of the loop.
  std::vector<LoraCapture> captures;
  // The state tensors' starting values: each adapter tensor seeded from its
  // *current* value in the injected model (so a partially trained adapter
  // resumes rather than restarts) and zeroed Adam moments.
  std::vector<onnx::TensorProto> initial_state;
  // The minibatch row index input, when batch_size > 0. Empty otherwise.
  std::string row_index_input;
  int64_t row_index_size = 0;
  // How many rows the captures carry, i.e. what the row index selects from.
  int64_t num_rows = 0;
  // Every adapter parameter this plan trains (lora_A/lora_B initializer
  // names, in the model's own namespace -- these double as the state input
  // names `state` above uses for the weight, as opposed to its Adam
  // moments). WriteBackLoraState's key list.
  std::vector<std::string> parameters;
};

// Builds the step graph for one block of `injected_model` (an InjectLora
// result, or an equivalent model built another way -- e.g. in Python).
//
// `adapter` names the tensors this run trains -- ordinarily every target
// InjectLora produced, but a caller may pass a LoraAdapter with a subset of
// `targets` to train only some of the injected branches.
//
// `num_rows` is how many calibration rows the caller will bind -- the
// leading dimension of every captured activation. It is a shape, not data:
// nothing here runs the model, matching qat_entry.h's BuildQatStepGraph.
//
// Refuses loudly rather than returning something unusable: throws
// std::invalid_argument when `adapter` has no targets, when the block is
// empty or not closed, when any node remaining in it after constant-folding
// (see FoldFrozenPrefixes in lora_entry.cpp) has no gradient rule -- the
// message names the op types and the supported set, as graph_grad does --
// or when minibatching is asked for over captures that disagree on their
// row count.
LoraStepPlan BuildLoraStepGraph(const onnx::ModelProto& injected_model,
                                const LoraAdapter& adapter,
                                const std::string& block_input_name,
                                const std::string& block_output_name,
                                int64_t num_rows,
                                const LoraOptions& options = {});

// Writes a finished loop's state back into the injected model.
//
// `final_state` is the loop's last state values, keyed by the same input
// names LoraStepPlan::state uses. Returns `injected_model` with the
// adapter's own A/B initializers rewritten and every other byte untouched
// -- base weights and any quantization codes (an NF4 dequant chain's
// included) byte-identical, matching lora.py's own write-back tail.
//
// Unlike WriteBackQatState there is no projection to make: an adapter
// tensor's state input *is* its own name in `injected_model`'s namespace,
// so this is the identity on `plan.parameters`' final values, not a
// round-to-a-code-grid.
onnx::ModelProto WriteBackLoraState(
    const onnx::ModelProto& injected_model, const LoraStepPlan& plan,
    const std::map<std::string, onnx::TensorProto>& final_state);

// ---------------------------------------------------------------------------
// Test-only
// ---------------------------------------------------------------------------

// The result of folding a constant-only prefix out of a node list -- see
// LoraFoldForTesting, immediately below.
struct LoraFoldResult {
  std::vector<onnx::NodeProto> kept_nodes;
  std::vector<onnx::TensorProto> extra_initializers;
};

// Test-only: exposes lora_entry.cpp's FoldFrozenPrefixes (lora.py's
// _fold_frozen_prefixes) directly, bypassing SliceBlock.
//
// BuildLoraStepGraph's own top comment records a finding: SliceBlock's
// forward/backward-reachability intersection can *never* hand
// FoldFrozenPrefixes a node whose entire input closure is already
// constant -- such a node can never be reachable from the block input, so
// it can never enter SliceBlock's own `nodes` result, so
// BuildLoraStepGraph's own call to FoldFrozenPrefixes is provably always a
// no-op today. That is true of lora.py's own _fold_frozen_prefixes too (see
// that comment for how this was checked directly against the Python), so
// this is not a divergence to fix -- but it does mean FoldFrozenPrefixes'
// own small constant evaluator (Cast/Gather/Reshape/Mul and friends; see
// lora_entry.cpp's EvalNode) has no coverage through this header's public
// entry points. This function exists purely so lora_entry_test.cpp can
// call it directly and check that evaluator's arithmetic against a
// NF4-shaped dequant chain, the same way graph_grad.h exposes
// BuildBackwardWithTemplatedRules purely for graph_grad_templates_test.cpp.
// Nothing in this header's own intended browser flow calls this.
LoraFoldResult LoraFoldForTesting(
    const std::vector<onnx::NodeProto>& nodes, const onnx::ModelProto& model,
    const std::vector<std::string>& non_foldable_names);
