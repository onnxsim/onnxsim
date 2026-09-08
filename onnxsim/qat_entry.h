#pragma once

// Block-wise QAT, reachable without Python.
//
// onnxsim/qat.py builds a training step graph for one block of a model and
// then drives it: capture the float model's activations, run the step graph in
// a loop, write the trained weights back. The graph-building half is pure
// graph surgery and is what this header ports; the driving half is not here
// and should not be, because the caller already has an inference runtime --
// onnxruntime-web in the browser -- and running a graph in a loop is that
// runtime's job.
//
// That split is what makes the port small enough to be worth trusting.
// qat.py's _build_step_graph reads its captured activations for `value.shape`
// and nothing else: the values become graph *inputs* the caller binds, never
// baked-in constants. So building the step graph needs shapes, not data, and
// the browser can capture the activations itself and hand them straight to the
// loop.
//
// The intended browser flow:
//   1. BuildQatStepGraph(float, quantized, block in/out, rows, opts)
//        -> a step graph, the state ping-pong map, the list of activations to
//           capture, and the initial state tensors
//   2. the caller captures those activations from the float model (ort-web),
//      binds them plus the initial state, and runs the step graph `n` times,
//      feeding the per-step scalars and carrying state outputs back to inputs
//   3. WriteBackQatState(quantized, plan, final state) -> the tuned model
//
// **Parity with the Python is a hard requirement, not an aspiration.** Two
// implementations that disagree would train a model differently in the browser
// than in Python and nothing would say so. onnxsim/qat_parity_fixtures.txt and
// its two tests already enforce that for the emitter primitives; a planner
// case is added there for the same reason. Emit the same nodes in the same
// order as qat.py, because GraphBuilder's name counter makes tensor names out
// of that order and the fixture compares names.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "qat_graph_builder.h"

// Which quantization scheme a run targets, and what it trains.
//
// These mirror apply_qat's own parameters. `learn_activation_scales` is the
// one that selects a *scheme* rather than adding a feature: with it off the
// target is quantize_weight_only_int4's INT4 weight-only layers, with it on it
// is quantize_static's QDQ layers, because a weight-only model has no
// activation quantizer anywhere to train. Asking for the wrong pairing is an
// error, not a silent no-op -- see BuildQatStepGraph's contract below.
struct QatOptions {
  bool learn_scales = false;
  bool learn_activation_scales = false;
  // Rows per optimizer step. 0 means full batch, which is the default and the
  // only mode with no per-step input beyond the scalars.
  int64_t batch_size = 0;
  // Only consulted when batch_size > 0; matches minibatch_indices' own seed.
  int64_t batch_seed = 0;
  bool shuffle = true;
};

// One tensor the caller must capture from the *float* model and bind for the
// life of the loop.
//
// `source_tensor` is the name to read out of the float graph;
// `step_graph_input` is what to bind it to. They differ whenever the step
// graph renames a tensor -- with a minibatch the whole set is bound to a
// resident table (`qat__all_<name>`) and the block reads rows gathered out of
// it, so a caller that assumed the two names were equal would bind the wrong
// input and train on uninitialized memory.
struct QatCapture {
  std::string step_graph_input;
  std::string source_tensor;
  std::vector<int64_t> dims;
  // True for the block's reconstruction target -- the float model's own output
  // for this block, which is the teacher. It is captured the same way as the
  // rest; the flag exists so a caller can label it in a UI.
  bool is_teacher = false;
};

// One trained layer, as the write-back needs to see it.
//
// This is the bridge between the loop's state tensors and the quantized
// model's initializers, and it exists because those two are named differently
// on purpose: state inputs are `qat__`-prefixed so they cannot collide with a
// tensor name carried over from the float model, while the initializers keep
// whatever the quantized model calls them.
//
// The two schemes are already normalized away here. A per-output-channel scale
// *is* a block-wise scale whose block spans the whole reduction axis, so INT4's
// 32-element blocks and INT8's per-channel scales are both just
// (block_axis, block_size, grid), and the write-back needs no special case.
struct QatTrainedLayer {
  // Loop state inputs. The last three are empty when not trained.
  std::string weight_state_input;
  std::string weight_scale_state_input;
  std::string log_act_scale_state_input;
  std::string act_zero_point_state_input;

  // Initializers in the quantized model these write back into.
  std::string codes_initializer;
  std::string weight_scale_initializer;
  std::string act_scale_initializer;
  std::string act_zero_point_initializer;

  std::vector<int64_t> weight_dims;
  std::vector<int64_t> weight_scale_dims;
  int64_t block_axis = 0;
  int64_t block_size = 0;
  // The integer grid the codes are clipped to: [-7, 7] for INT4, [-127, 127]
  // for quantize_static's per-channel INT8.
  float code_min = 0.0f;
  float code_max = 0.0f;
  // INT4 codes are packed two to a byte on export; INT8 codes are not.
  bool packed_int4 = false;
  // The weight scale as the model currently stores it. Needed to re-derive
  // codes when learn_scales is off, since then no state tensor carries it.
  onnx::TensorProto frozen_weight_scale;
};

// Everything needed to run the loop and then write its result back.
struct QatStepPlan {
  onnx::ModelProto step_graph;
  // Input name -> the output carrying its next value. This is what makes the
  // loop a ping-pong of two buffers rather than a rebuild per step.
  std::vector<std::pair<std::string, std::string>> state;
  // Scalar float inputs supplied fresh each step: the learning rates and Adam's
  // two bias-correction factors. AdamBiasCorrections computes the latter.
  std::vector<std::string> scalars;
  // The scalar output carrying this step's loss. Empty if none.
  std::string loss_name;
  // Bound once, for the life of the loop.
  std::vector<QatCapture> captures;
  // The state tensors' starting values: each trained layer's master weight
  // seeded from the *float* model's own weight, and zeroed Adam moments.
  // Seeding from the float weight rather than the quantized one is what makes
  // step 0 reproduce round-to-nearest exactly, so every later step is a
  // measured improvement on it rather than on an arbitrary re-initialization.
  std::vector<onnx::TensorProto> initial_state;
  // The minibatch row index input, when batch_size > 0. Empty otherwise.
  // It is rank-1 int64 and is the only non-float input a step graph ever has.
  std::string row_index_input;
  int64_t row_index_size = 0;
  // How many rows the captures carry, i.e. what the row index selects from.
  int64_t num_rows = 0;
  // The layers this run trains, for WriteBackQatState.
  std::vector<QatTrainedLayer> layers;
};

// Builds the step graph for one block.
//
// `num_rows` is how many calibration rows the caller will bind -- the leading
// dimension of every captured activation. It is a shape, not data: nothing
// here runs the model.
//
// Refuses loudly rather than returning something unusable, matching
// apply_qat's own contract. Throws std::invalid_argument when the block is
// empty or not closed, when any node in it has no gradient rule (the message
// names the op types and the supported set, as graph_grad does), or when the
// slice contains no layer of the scheme `options` selects -- including the
// case where the caller asked for activation training over a weight-only
// model, which is a scheme error rather than a boundary error and says so.
QatStepPlan BuildQatStepGraph(const onnx::ModelProto& float_model,
                              const onnx::ModelProto& quantized_model,
                              const std::string& block_input_name,
                              const std::string& block_output_name,
                              int64_t num_rows, const QatOptions& options);

// Writes a finished loop's state back into the quantized model.
//
// `final_state` is the loop's last state values, keyed by the same input names
// QatStepPlan::state uses. Returns `quantized_model` with this block's
// initializers rewritten and every other byte untouched: the weight codes
// re-derived from the trained master weights, the weight scales if they were
// trained, and the activation scale and zero-point if they were.
//
// The projections back onto what the model can store happen here -- codes
// rounded and clipped to the grid, the activation scale taken out of log
// space, the zero-point rounded onto uint8's integer grid -- for the reason
// qat.py does them here too: the optimizer needs those parameters continuous,
// and the model can only hold what it can hold.
onnx::ModelProto WriteBackQatState(
    const onnx::ModelProto& quantized_model, const QatStepPlan& plan,
    const std::map<std::string, onnx::TensorProto>& final_state);
