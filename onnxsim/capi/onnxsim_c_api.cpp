/*
 * SPDX-License-Identifier: Apache-2.0
 */
#define ONNXSIM_C_API_BUILD
#include "onnxsim_c_api.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "custom_optimizer_passes.h"
#include "function_rewriter.h"
#include "model_info.h"
#include "onnx/defs/parser.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/optimize.h"
#include "onnxsim.h"
#include "tensor_pool.h"
#include "tensor_pool_bridge.h"
#include "tensor_pool_gguf_bridge.h"

#ifndef ONNXSIM_HAS_ORT
#error \
    "The onnxsim C API requires the built-in ONNX Runtime (ONNXSIM_BUILTIN_ORT=ON)."
#endif

namespace {

// Duplicate a std::string into a malloc'd, NUL-terminated C string so the
// caller can free it with std::free (via onnxsim_free_string). Returns nullptr
// on allocation failure.
char* DupCString(const std::string& s) {
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (p != nullptr) {
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
  }
  return p;
}

void SetError(char** out_error, const std::string& message) {
  if (out_error != nullptr) {
    *out_error = DupCString(message);
  }
}

// Copy `s` into a freshly malloc'd buffer for an out_data/out_size pair
// (release with onnxsim_free_buffer), same allocation convention
// SimplifyToBuffer below uses: malloc(0) may return nullptr, so a 0-byte
// input still gets a distinct non-null pointer, guaranteeing out_data != NULL
// on success. Returns false (and leaves *out_data/*out_size untouched) on
// allocation failure.
bool CopyToBuffer(const std::string& s, void** out_data, size_t* out_size) {
  void* buffer = std::malloc(s.size() != 0 ? s.size() : 1);
  if (buffer == nullptr) {
    return false;
  }
  std::memcpy(buffer, s.data(), s.size());
  *out_data = buffer;
  *out_size = s.size();
  return true;
}

// Translate the (pointer array, is_null) FFI encoding into the optional vector
// the C++ core expects.
std::optional<std::vector<std::string>> BuildSkipOptimizers(
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null) {
  if (skip_optimizers_is_null != 0) {
    return std::nullopt;
  }
  std::vector<std::string> passes;
  passes.reserve(num_skip_optimizers);
  for (size_t i = 0; i < num_skip_optimizers; ++i) {
    const char* name =
        skip_optimizers != nullptr ? skip_optimizers[i] : nullptr;
    passes.emplace_back(name != nullptr ? name : "");
  }
  return passes;
}

// Translate the (pointer array, count) FFI encoding into the optional vector
// the C++ core expects. Unlike BuildSkipOptimizers there is no is_null flag:
// num_extra_optimizers == 0 and std::nullopt behave identically (no extra
// passes), so a plain count of 0 already means "none".
std::optional<std::vector<std::string>> BuildExtraOptimizers(
    const char* const* extra_optimizers, size_t num_extra_optimizers) {
  if (num_extra_optimizers == 0) {
    return std::nullopt;
  }
  std::vector<std::string> passes;
  passes.reserve(num_extra_optimizers);
  for (size_t i = 0; i < num_extra_optimizers; ++i) {
    const char* name =
        extra_optimizers != nullptr ? extra_optimizers[i] : nullptr;
    passes.emplace_back(name != nullptr ? name : "");
  }
  return passes;
}

// A target opset version of <= 0 means "leave the opset version unchanged".
std::optional<int> BuildTargetOpsetVersion(int target_opset_version) {
  if (target_opset_version <= 0) {
    return std::nullopt;
  }
  return target_opset_version;
}

// Adapts the C rewrite callback to the C++ GraphRewriter interface, exchanging
// models as serialized ModelProto bytes. Mirrors the Python
// `_GraphRewriterAdapter`: a return of 0 means "nothing rewritten" (keep the
// model as-is and skip a copy), > 0 means the callback produced a rewritten
// model, and < 0 is an error.
class CApiGraphRewriter : public GraphRewriter {
 public:
  CApiGraphRewriter(OnnxsimRewriteFn fn, OnnxsimRewriteFreeFn free_fn,
                    void* user_data)
      : fn_(fn), free_fn_(free_fn), user_data_(user_data) {}

  bool _Run(onnx::ModelProto& model) const override {
    std::string in;
    if (!model.SerializeToString(&in)) {
      throw std::runtime_error(
          "failed to serialize the model for the custom rewriter");
    }
    void* out_data = nullptr;
    size_t out_size = 0;
    const int rc = fn_(user_data_, in.data(), in.size(), &out_data, &out_size);
    if (rc < 0) {
      throw std::runtime_error(
          "the custom rewriter callback reported an error");
    }
    if (rc == 0) {
      // Nothing was rewritten this round: keep the model unchanged.
      return false;
    }
    if (out_data == nullptr) {
      throw std::runtime_error(
          "the custom rewriter reported a rewrite but returned no model");
    }
    onnx::ModelProto rewritten;
    const bool parsed = ParseProtoFromBytes(
        &rewritten, static_cast<const char*>(out_data), out_size);
    // Release the callback-owned buffer regardless of whether it parsed.
    if (free_fn_ != nullptr) {
      free_fn_(user_data_, out_data, out_size);
    }
    if (!parsed) {
      throw std::runtime_error(
          "the custom rewriter returned bytes that are not a valid ONNX "
          "ModelProto");
    }
    model.Swap(&rewritten);
    return true;
  }

 private:
  OnnxsimRewriteFn fn_;
  OnnxsimRewriteFreeFn free_fn_;
  void* user_data_;
};

// Adapts the C OnnxsimExecuteFn callback to the C++ ModelExecutor interface,
// exchanging tensors as DLPack DLManagedTensor with no TensorProto round trip.
// Mirrors CApiGraphRewriter: a (fn, free_fn, user_data) triple. The whole C API
// requires the built-in ORT (see the #error at the top), but this adapter needs
// no ORT itself -- it just marshals to the host callback.
class CApiModelExecutor : public ModelExecutor {
 public:
  CApiModelExecutor(OnnxsimExecuteFn fn, OnnxsimExecuteFreeFn free_fn,
                    void* user_data)
      : fn_(fn), free_fn_(free_fn), user_data_(user_data) {}

  std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const override {
    const std::string model_str = model.SerializeAsString();
    DLManagedTensor** out_outputs = nullptr;
    size_t num_outputs = 0;
    const int rc =
        fn_(user_data_, model_str.data(), model_str.size(), inputs.data(),
            inputs.size(), &out_outputs, &num_outputs);
    if (rc != 0) {
      throw std::runtime_error(
          "the custom constant-folding executor callback failed (returned " +
          std::to_string(rc) + ")");
    }
    if (num_outputs != 0 && out_outputs == nullptr) {
      throw std::runtime_error(
          "the custom executor reported outputs but returned a NULL array");
    }
    // Adopt each returned tensor into an owning handle first (so the DLPack
    // deleters run even if a later step throws), then let the host release the
    // array container.
    std::vector<DLManagedTensorPtr> outputs;
    outputs.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      outputs.emplace_back(out_outputs[i]);
    }
    if (free_fn_ != nullptr) {
      free_fn_(user_data_, out_outputs, num_outputs);
    }
    return outputs;
  }

 private:
  OnnxsimExecuteFn fn_;
  OnnxsimExecuteFreeFn free_fn_;
  void* user_data_;
};

// Shared body of onnxsim_simplify / onnxsim_simplify_with_rules: parse the
// model, simplify it (optionally with `rewriter`, and through `executor` when
// non-NULL -- otherwise the built-in ORT executor), and hand back a freshly
// malloc'd buffer of serialized bytes. All out-parameters follow the C API
// contract documented in the header.
OnnxsimStatus SimplifyToBuffer(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const char* const* extra_optimizers, size_t num_extra_optimizers,
    const GraphRewriter* rewriter, const ModelExecutor* executor,
    void** out_data, size_t* out_size, char** out_error) {
  if (out_data != nullptr) {
    *out_data = nullptr;
  }
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (model_data == nullptr || out_data == nullptr || out_size == nullptr) {
    SetError(out_error, "onnxsim_simplify: required argument is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    // Force env initialization to register opsets, matching the Python binding.
    InitEnv();
    onnx::ModelProto model;
    if (!ParseProtoFromBytes(&model, static_cast<const char*>(model_data),
                             model_size)) {
      SetError(out_error, "failed to parse ONNX ModelProto from input bytes");
      return ONNXSIM_ERROR;
    }
    onnx::ModelProto result = Simplify(
        executor != nullptr ? *executor : *GetBuiltinModelExecutor(), model,
        BuildSkipOptimizers(skip_optimizers, num_skip_optimizers,
                            skip_optimizers_is_null),
        constant_folding != 0, shape_inference != 0, tensor_size_threshold,
        BuildTargetOpsetVersion(target_opset_version), rewriter,
        /*initializers_as_constants=*/true,
        /*include_inline_functions=*/false,
        /*mutable_initializer=*/true,
        /*overwrite_input_shapes=*/std::nullopt,
        /*unused_output=*/std::nullopt,
        BuildExtraOptimizers(extra_optimizers, num_extra_optimizers));

    std::string out;
    if (!result.SerializeToString(&out)) {
      SetError(out_error, "failed to serialize the simplified ModelProto");
      return ONNXSIM_ERROR;
    }
    // malloc(0) may return nullptr; hand back a distinct non-null pointer so
    // the caller can always free it and can rely on out_data != NULL on
    // success.
    void* buffer = std::malloc(out.size() != 0 ? out.size() : 1);
    if (buffer == nullptr) {
      SetError(out_error, "out of memory while copying the simplified model");
      return ONNXSIM_ERROR;
    }
    std::memcpy(buffer, out.data(), out.size());
    *out_data = buffer;
    *out_size = out.size();
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while simplifying the model");
    return ONNXSIM_ERROR;
  }
}

// Parse `num_rules` (pattern, replacement) FunctionProto pairs from the given
// byte buffers into rewrite rules. Throws std::runtime_error on a parse
// failure.
std::vector<onnxsim::FunctionRewriteRule> BuildRewriteRules(
    const void* const* pattern_data, const size_t* pattern_sizes,
    const void* const* replacement_data, const size_t* replacement_sizes,
    size_t num_rules) {
  std::vector<onnxsim::FunctionRewriteRule> rules;
  rules.reserve(num_rules);
  for (size_t i = 0; i < num_rules; ++i) {
    onnxsim::FunctionRewriteRule rule;
    if (!ParseProtoFromBytes(&rule.pattern,
                             static_cast<const char*>(pattern_data[i]),
                             pattern_sizes[i])) {
      throw std::runtime_error("failed to parse pattern FunctionProto");
    }
    if (!ParseProtoFromBytes(&rule.replacement,
                             static_cast<const char*>(replacement_data[i]),
                             replacement_sizes[i])) {
      throw std::runtime_error("failed to parse replacement FunctionProto");
    }
    rules.push_back(std::move(rule));
  }
  return rules;
}

}  // namespace

extern "C" {

OnnxsimStatus onnxsim_simplify(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const char* const* extra_optimizers, size_t num_extra_optimizers,
    OnnxsimRewriteFn rewrite_fn, OnnxsimRewriteFreeFn rewrite_free_fn,
    void* rewrite_user_data, void** out_data, size_t* out_size,
    char** out_error) {
  std::optional<CApiGraphRewriter> rewriter;
  if (rewrite_fn != nullptr) {
    rewriter.emplace(rewrite_fn, rewrite_free_fn, rewrite_user_data);
  }
  return SimplifyToBuffer(
      model_data, model_size, skip_optimizers, num_skip_optimizers,
      skip_optimizers_is_null, constant_folding, shape_inference,
      tensor_size_threshold, target_opset_version, extra_optimizers,
      num_extra_optimizers, rewriter.has_value() ? &rewriter.value() : nullptr,
      /*executor=*/nullptr, out_data, out_size, out_error);
}

OnnxsimStatus onnxsim_simplify_with_executor(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const char* const* extra_optimizers, size_t num_extra_optimizers,
    OnnxsimRewriteFn rewrite_fn, OnnxsimRewriteFreeFn rewrite_free_fn,
    void* rewrite_user_data, OnnxsimExecuteFn execute_fn,
    OnnxsimExecuteFreeFn execute_free_fn, void* execute_user_data,
    void** out_data, size_t* out_size, char** out_error) {
  std::optional<CApiGraphRewriter> rewriter;
  if (rewrite_fn != nullptr) {
    rewriter.emplace(rewrite_fn, rewrite_free_fn, rewrite_user_data);
  }
  std::optional<CApiModelExecutor> executor;
  if (execute_fn != nullptr) {
    executor.emplace(execute_fn, execute_free_fn, execute_user_data);
  }
  return SimplifyToBuffer(
      model_data, model_size, skip_optimizers, num_skip_optimizers,
      skip_optimizers_is_null, constant_folding, shape_inference,
      tensor_size_threshold, target_opset_version, extra_optimizers,
      num_extra_optimizers, rewriter.has_value() ? &rewriter.value() : nullptr,
      executor.has_value() ? &executor.value() : nullptr, out_data, out_size,
      out_error);
}

OnnxsimStatus onnxsim_simplify_with_rules(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const void* const* pattern_data, const size_t* pattern_sizes,
    const void* const* replacement_data, const size_t* replacement_sizes,
    size_t num_rules, const char* const* extra_optimizers,
    size_t num_extra_optimizers, void** out_data, size_t* out_size,
    char** out_error) {
  // With no rules this is just onnxsim_simplify (no callback rewriter).
  if (num_rules == 0) {
    return onnxsim_simplify(
        model_data, model_size, skip_optimizers, num_skip_optimizers,
        skip_optimizers_is_null, constant_folding, shape_inference,
        tensor_size_threshold, target_opset_version, extra_optimizers,
        num_extra_optimizers, /*rewrite_fn=*/nullptr,
        /*rewrite_free_fn=*/nullptr, /*rewrite_user_data=*/nullptr, out_data,
        out_size, out_error);
  }
  if (out_data != nullptr) {
    *out_data = nullptr;
  }
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (pattern_data == nullptr || pattern_sizes == nullptr ||
      replacement_data == nullptr || replacement_sizes == nullptr) {
    SetError(out_error, "onnxsim_simplify_with_rules: rule array is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    std::shared_ptr<GraphRewriter> rewriter =
        onnxsim::MakeFunctionProtoRewriter(
            BuildRewriteRules(pattern_data, pattern_sizes, replacement_data,
                              replacement_sizes, num_rules));
    return SimplifyToBuffer(
        model_data, model_size, skip_optimizers, num_skip_optimizers,
        skip_optimizers_is_null, constant_folding, shape_inference,
        tensor_size_threshold, target_opset_version, extra_optimizers,
        num_extra_optimizers, rewriter.get(),
        /*executor=*/nullptr, out_data, out_size, out_error);
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while building rewrite rules");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_simplify_path(
    const char* in_path, const char* out_path,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const char* const* extra_optimizers, size_t num_extra_optimizers,
    OnnxsimRewriteFn rewrite_fn, OnnxsimRewriteFreeFn rewrite_free_fn,
    void* rewrite_user_data, char** out_error) {
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (in_path == nullptr || out_path == nullptr) {
    SetError(out_error, "onnxsim_simplify_path: in_path/out_path is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    InitEnv();
    std::optional<CApiGraphRewriter> rewriter;
    if (rewrite_fn != nullptr) {
      rewriter.emplace(rewrite_fn, rewrite_free_fn, rewrite_user_data);
    }
    const GraphRewriter* rewriter_ptr =
        rewriter.has_value() ? &rewriter.value() : nullptr;

    SimplifyPath(*GetBuiltinModelExecutor(), in_path, out_path,
                 BuildSkipOptimizers(skip_optimizers, num_skip_optimizers,
                                     skip_optimizers_is_null),
                 constant_folding != 0, shape_inference != 0,
                 tensor_size_threshold,
                 BuildTargetOpsetVersion(target_opset_version), rewriter_ptr,
                 /*initializers_as_constants=*/true,
                 /*include_inline_functions=*/false,
                 /*mutable_initializer=*/true,
                 /*overwrite_input_shapes=*/std::nullopt,
                 /*unused_output=*/std::nullopt,
                 BuildExtraOptimizers(extra_optimizers, num_extra_optimizers));
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while simplifying the model");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_parse_model_text(const char* text, void** out_data,
                                       size_t* out_size, char** out_error) {
  if (out_data != nullptr) {
    *out_data = nullptr;
  }
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (text == nullptr) {
    SetError(out_error, "onnxsim_parse_model_text: text is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    onnx::ModelProto model;
    onnx::OnnxParser parser(text);
    const auto status = parser.Parse(model);
    if (!status.IsOK()) {
      SetError(out_error, status.ErrorMessage());
      return ONNXSIM_ERROR;
    }
    std::string bytes;
    if (!model.SerializeToString(&bytes)) {
      SetError(out_error, "failed to serialize the parsed ModelProto");
      return ONNXSIM_ERROR;
    }
    if (out_data != nullptr && out_size != nullptr &&
        !CopyToBuffer(bytes, out_data, out_size)) {
      SetError(out_error, "out of memory while returning the parsed model");
      return ONNXSIM_ERROR;
    }
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while parsing the model text");
    return ONNXSIM_ERROR;
  }
}

char* onnxsim_list_optimizers(void) {
  try {
    std::string joined;
    for (const auto& pass : onnx::optimization::GetFuseAndEliminationPass()) {
      if (!joined.empty()) {
        joined.push_back('\n');
      }
      joined.append(pass);
    }
    return DupCString(joined);
  } catch (...) {
    return nullptr;
  }
}

char* onnxsim_list_other_optimizers(void) {
  try {
    onnxsim::RegisterCustomOptimizerPasses();
    const auto default_passes = onnx::optimization::GetFuseAndEliminationPass();
    const std::unordered_set<std::string> default_set(default_passes.begin(),
                                                      default_passes.end());
    std::string joined;
    for (const auto& pass : onnx::optimization::GetAvailablePasses()) {
      if (default_set.count(pass) != 0) {
        continue;
      }
      if (!joined.empty()) {
        joined.push_back('\n');
      }
      joined.append(pass);
    }
    return DupCString(joined);
  } catch (...) {
    return nullptr;
  }
}

OnnxsimStatus onnxsim_model_info_diff(const void* original_data,
                                      size_t original_size,
                                      const void* simplified_data,
                                      size_t simplified_size, char** out_text,
                                      char** out_error) {
  if (out_text != nullptr) {
    *out_text = nullptr;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (original_data == nullptr || simplified_data == nullptr) {
    SetError(out_error, "onnxsim_model_info_diff: required argument is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    onnx::ModelProto original;
    if (!ParseProtoFromBytes(&original, static_cast<const char*>(original_data),
                             original_size)) {
      SetError(out_error, "failed to parse the original ONNX ModelProto");
      return ONNXSIM_ERROR;
    }
    onnx::ModelProto simplified;
    if (!ParseProtoFromBytes(&simplified,
                             static_cast<const char*>(simplified_data),
                             simplified_size)) {
      SetError(out_error, "failed to parse the simplified ONNX ModelProto");
      return ONNXSIM_ERROR;
    }
    if (out_text != nullptr) {
      char* text = DupCString(FormatSimplifyingInfo(original, simplified));
      if (text == nullptr) {
        SetError(out_error, "out of memory while formatting the model diff");
        return ONNXSIM_ERROR;
      }
      *out_text = text;
    }
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while computing the model diff");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_graph_diff(const void* original_data,
                                 size_t original_size,
                                 const void* simplified_data,
                                 size_t simplified_size, char** out_text,
                                 char** out_error) {
  if (out_text != nullptr) {
    *out_text = nullptr;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (original_data == nullptr || simplified_data == nullptr) {
    SetError(out_error, "onnxsim_graph_diff: required argument is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    onnx::ModelProto original;
    if (!ParseProtoFromBytes(&original, static_cast<const char*>(original_data),
                             original_size)) {
      SetError(out_error, "failed to parse the original ONNX ModelProto");
      return ONNXSIM_ERROR;
    }
    onnx::ModelProto simplified;
    if (!ParseProtoFromBytes(&simplified,
                             static_cast<const char*>(simplified_data),
                             simplified_size)) {
      SetError(out_error, "failed to parse the simplified ONNX ModelProto");
      return ONNXSIM_ERROR;
    }
    if (out_text != nullptr) {
      char* text = DupCString(FormatGraphDiff(original, simplified));
      if (text == nullptr) {
        SetError(out_error, "out of memory while formatting the graph diff");
        return ONNXSIM_ERROR;
      }
      *out_text = text;
    }
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while computing the graph diff");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_export_safetensors(const void* model_data,
                                         size_t model_size,
                                         const char* out_path,
                                         char** out_error) {
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (model_data == nullptr || out_path == nullptr) {
    SetError(out_error,
             "onnxsim_export_safetensors: required argument is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    onnx::ModelProto model;
    if (!ParseProtoFromBytes(&model, static_cast<const char*>(model_data),
                             model_size)) {
      SetError(out_error, "failed to parse ONNX ModelProto from input bytes");
      return ONNXSIM_ERROR;
    }
    onnxsim::tensor_pool::TensorPool pool;
    onnxsim::tensor_pool::SaveModelAsSafetensorsStandalone(model, out_path,
                                                           pool);
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while exporting the safetensors file");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_import_safetensors(const char* in_path, void** out_data,
                                         size_t* out_size, char** out_error) {
  if (out_data != nullptr) {
    *out_data = nullptr;
  }
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (in_path == nullptr) {
    SetError(out_error, "onnxsim_import_safetensors: in_path is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    onnx::ModelProto model;
    onnxsim::tensor_pool::TensorPool pool;
    if (!onnxsim::tensor_pool::LoadModelFromSafetensors(in_path, &model,
                                                        pool)) {
      SetError(out_error,
               "safetensors file has no embedded onnxsim model (a plain "
               "weights-only archive is not importable as a graph)");
      return ONNXSIM_ERROR;
    }
    std::string out;
    if (!model.SerializeToString(&out)) {
      SetError(out_error, "failed to serialize the imported ModelProto");
      return ONNXSIM_ERROR;
    }
    if (out_data != nullptr && out_size != nullptr &&
        !CopyToBuffer(out, out_data, out_size)) {
      SetError(out_error, "out of memory while copying the imported model");
      return ONNXSIM_ERROR;
    }
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while importing the safetensors file");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_export_gguf(const void* model_data, size_t model_size,
                                  const char* out_path, char** out_error) {
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (model_data == nullptr || out_path == nullptr) {
    SetError(out_error, "onnxsim_export_gguf: required argument is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    onnx::ModelProto model;
    if (!ParseProtoFromBytes(&model, static_cast<const char*>(model_data),
                             model_size)) {
      SetError(out_error, "failed to parse ONNX ModelProto from input bytes");
      return ONNXSIM_ERROR;
    }
    onnxsim::tensor_pool::TensorPool pool;
    onnxsim::tensor_pool::SaveModelAsGGUFStandalone(model, out_path, pool);
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while exporting the gguf file");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_import_gguf(const char* in_path, void** out_data,
                                  size_t* out_size, char** out_error) {
  if (out_data != nullptr) {
    *out_data = nullptr;
  }
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (in_path == nullptr) {
    SetError(out_error, "onnxsim_import_gguf: in_path is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    onnx::ModelProto model;
    onnxsim::tensor_pool::TensorPool pool;
    if (!onnxsim::tensor_pool::LoadModelFromGGUF(in_path, &model, pool)) {
      SetError(out_error,
               "gguf file has no embedded onnxsim model (a plain "
               "weights-only archive is not importable as a graph)");
      return ONNXSIM_ERROR;
    }
    std::string out;
    if (!model.SerializeToString(&out)) {
      SetError(out_error, "failed to serialize the imported ModelProto");
      return ONNXSIM_ERROR;
    }
    if (out_data != nullptr && out_size != nullptr &&
        !CopyToBuffer(out, out_data, out_size)) {
      SetError(out_error, "out of memory while copying the imported model");
      return ONNXSIM_ERROR;
    }
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while importing the gguf file");
    return ONNXSIM_ERROR;
  }
}

void onnxsim_free_buffer(void* data) { std::free(data); }

void onnxsim_free_string(char* data) { std::free(data); }

}  // extern "C"
