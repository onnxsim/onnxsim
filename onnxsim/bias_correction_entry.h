#pragma once

// Graph-surgery half of onnxsim.bias_correction's correct_bias /
// correct_spatial_bias (onnxsim/bias_correction.py), reachable without
// Python -- exposed to the WASM converter page (scripts/convertmodel/
// interface.cpp) the same way qat_entry.h/lora_entry.h split their own
// Python originals.
//
// Measuring a correction means running two models (the original float one
// and a modified one -- quantized, or with a Resize node's mode/
// coordinate_transformation_mode swapped for one a deployment target
// supports) on the same calibration data and averaging their difference.
// That needs a live inference backend, and unlike qat_entry.h/lora_entry.h's
// step-graph approach (which turns "run an optimizer step" into an ordinary
// ONNX graph onnxruntime-web can execute), there is no graph-based way to
// express "run this whole model and reduce its output" -- so, exactly like
// quantize_static's own calibration ranges (see interface.cpp's
// onnxsim_add_graph_outputs / quantize_calibration.mjs), that half stays in
// JS against onnxruntime-web. This header is only the other half: given an
// already-measured correction per output tensor, splice an Add node (and a
// constant initializer holding the correction, broadcast to that tensor's
// own shape) right after its producer -- ports bias_correction.py's own
// _splice_add_correction exactly -- and, separately, list which of a
// model's own Conv/Gemm/MatMul/Resize node outputs are even eligible
// (present, by name, in both the float and modified model), mirroring
// correct_bias's/correct_spatial_bias's own candidate-selection loop so the
// browser doesn't need to reimplement _CORRECTABLE_OPS in JS.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <vector>

// One already-measured correction to splice in: `data` (row-major) is
// added, broadcast to `shape`, right after `output_name`'s producer node.
// `shape` is whatever the caller already decided to broadcast over -- e.g.
// [1, C, 1, 1] for a plain per-channel correction (correct_bias) or
// [1, C, H, W] for a full per-position map (correct_spatial_bias); this
// function does no broadcasting-shape inference of its own, matching
// _splice_add_correction's own contract (the caller reshapes first).
struct BiasCorrectionEntry {
  std::string output_name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

// Applies every entry in `corrections` to a copy of `model`, skipping (not
// erroring on) an entry whose `output_name` is not any node's output in
// `model` -- e.g. a candidate measured against a different model, or one a
// prior graph edit already renamed. Node order in the returned graph keeps
// each correction's Add node immediately after its producer, matching
// bias_correction.py's own node.insert(producer_idx + 1, ...) placement.
onnx::ModelProto ApplyBiasCorrections(
    const onnx::ModelProto& model,
    const std::vector<BiasCorrectionEntry>& corrections);

// One Conv/Gemm/MatMul/Resize node output eligible for bias correction:
// `axis` is the channel axis in _CORRECTABLE_OPS's own convention (1 for
// Conv/Resize's NCHW layout, -1 for Gemm/MatMul's trailing feature dim).
// `spatial` is true exactly when `axis == 1` -- i.e. the case
// correct_spatial_bias's per-position grid can additionally apply to,
// versus correct_bias's per-channel constant, which applies to every
// candidate regardless.
struct CorrectableCandidate {
  std::string output_name;
  int64_t axis;
  bool spatial;
};

// Every node in `float_model` whose op_type is one of _CORRECTABLE_OPS'
// (Conv, Gemm, MatMul, Resize) and whose own first output name is also
// present as some node's output in `modified_model` -- mirrors
// correct_bias's own `candidates` loop over float_model.graph.node exactly
// (including using float_model, not modified_model, as the source of each
// op_type/axis; only presence is checked against modified_model).
std::vector<CorrectableCandidate> ListCorrectableOutputs(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& modified_model);
