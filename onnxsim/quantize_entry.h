#pragma once

// Single-call quantization entry points exposed to Python: each wraps one (or
// a small group of) onnxsim optimizer passes registered in
// custom_optimizer_passes.cpp, running it standalone via OptimizeFixed rather
// than as part of the full Simplify() fixed point. See onnxsim.h for the
// per-function documentation these mirror.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

onnx::ModelProto QuantizeDynamic(const onnx::ModelProto& model);
onnx::ModelProto QuantizeTernary(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnly(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyInt4(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyMatMulNBits(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyInt16(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyInt8Block(const onnx::ModelProto& model);
onnx::ModelProto QuantizeWeightOnlyMXFP4(const onnx::ModelProto& model);

std::vector<std::string> ListQuantizableActivations(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeStatic(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

onnx::ModelProto QuantizeStaticInt16(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorQuantizableOutputs(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperator(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorElementwiseQuantizableTensors(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperatorElementwise(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorActivationQuantizableTensors(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperatorActivation(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

std::vector<std::string> ListQOperatorConcatQuantizableTensors(
    const onnx::ModelProto& model);

onnx::ModelProto QuantizeQOperatorConcat(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges);

onnx::ModelProto QuantizeFp16(const onnx::ModelProto& model,
                              bool keep_io_types);
onnx::ModelProto QuantizeBf16(const onnx::ModelProto& model,
                              bool keep_io_types);
onnx::ModelProto QuantizeFp8(const onnx::ModelProto& model,
                             const std::string& format, bool keep_io_types);

onnx::ModelProto ApplyDoubleQuantization(const onnx::ModelProto& model);
onnx::ModelProto ApplyAnyPrecisionLlm(const onnx::ModelProto& model,
                                      int64_t bits, int64_t max_bits,
                                      int64_t block_size);
onnx::ModelProto ApplyQuarot(const onnx::ModelProto& model, uint64_t seed,
                             int64_t block_size, float epsilon);
onnx::ModelProto ApplyIQ4NL(const onnx::ModelProto& model);
