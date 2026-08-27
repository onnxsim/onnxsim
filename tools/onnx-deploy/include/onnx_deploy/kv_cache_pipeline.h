// onnx_deploy/kv_cache_pipeline.h
//
// Generic C++ glue for the multi-file ONNX export shape that `optimum-onnx`
// produces for autoregressive generation --
//   encoder_model.onnx            (optional -- absent for decoder-only LMs)
//   decoder_model.onnx            (no KV-cache inputs; first decode step)
//   decoder_with_past_model.onnx  (KV-cache in and out; every step after)
// -- with no Python/torch dependency. See ../../README.md for the design
// writeup, the naming-convention assumptions this relies on, and what is
// deliberately left out (tokenization, sampling beyond greedy, batching,
// the merged decoder_model_merged.onnx shape).
//
// Usage contract: this header builds against ONNX Runtime's C++ API with
// ORT_API_MANUAL_INIT (see the block below), so the ORT function table is
// NOT resolved at static-init time and this header does not require linking
// against libonnxruntime at all. The embedder must call `Ort::InitApi(api)`
// with an `OrtApi*` obtained however it likes -- linked directly
// (`OrtGetApiBase()->GetApi(ORT_API_VERSION)`), or resolved at runtime from
// a dlopen'd/LoadLibrary'd libonnxruntime (see onnx_deploy_c_api.cpp, the
// swappable-libort C ABI built on top of this header) -- exactly once,
// before constructing any Ort::* object, including KvCachePipeline.
#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_deploy {

// One decode step's KV-cache tensors, keyed by the *next* step's expected
// input name (e.g. "past_key_values.0.decoder.key"). Values are owned here;
// RunDecoderStep only ever borrows non-owning views of them (see
// detail::BorrowView) so the same buffer can be fed into many Run() calls
// without a copy and without losing ownership.
using KvCache = std::map<std::string, Ort::Value>;

struct GenerationConfig {
  int64_t eos_token_id = -1;  // -1 disables early stop
  int64_t decoder_start_token_id = 0;
  int64_t max_new_tokens = 32;
};

// Which execution provider every session in a KvCachePipeline runs on.
// Defaults to ORT's built-in CPU EP (no explicit provider needed).
//
// "cuda" calls Ort::SessionOptions::AppendExecutionProvider_CUDA -- this
// only works if the libonnxruntime onnx_deploy_load_ort() loaded was
// actually *built* with CUDA EP support (the official CPU-only release
// tarballs are not; a GPU build, e.g. onnxruntime-linux-x64-gpu-*.tgz, is),
// and if a CUDA-capable GPU/driver/toolkit is present at runtime.
//
// "webgpu" calls the generic Ort::SessionOptions::AppendExecutionProvider
// ("WebGPU", {}) -- this is ORT's *native* WebGPU EP (built on Dawn,
// running against a real GPU via Vulkan/Metal/D3D12), a different thing
// from onnxruntime-web's browser WebGPU EP that wasm/ uses. As of ORT
// 1.23.0, the plain prebuilt release tarballs (CPU or GPU) do not include
// it -- it needs a from-source build with --use_webgpu; confirmed
// empirically here via AppendExecutionProvider("WebGPU", {}), which throws
// "WebGPU execution provider is not supported in this build" against the
// plain tarball rather than silently doing nothing. No provider-options
// keys are documented for it in onnxruntime_c_api.h as of this writing, so
// none are exposed here beyond selecting it by name.
//
// Neither EP is baked into this header or its build -- exactly the same
// "swap by pointing at a different libort" story as the rest of this
// library, just choosing a GPU-capable one. If the loaded libort lacks the
// requested EP's support, or no GPU is available, session construction
// throws Ort::Exception with ORT's own message (e.g. "...providers_cuda...
// not found", "not supported in this build", or a GPU driver error), which
// propagates as a normal onnx_deploy_create failure -- not a crash.
struct PipelineOptions {
  enum class ExecutionProvider { kCpu, kCuda, kWebGpu };
  ExecutionProvider execution_provider = ExecutionProvider::kCpu;
  int cuda_device_id = 0;
};

namespace detail {

inline Ort::SessionOptions BuildSessionOptions(const PipelineOptions& options) {
  Ort::SessionOptions session_options;
  if (options.execution_provider == PipelineOptions::ExecutionProvider::kCuda) {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = options.cuda_device_id;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
  } else if (options.execution_provider == PipelineOptions::ExecutionProvider::kWebGpu) {
    session_options.AppendExecutionProvider("WebGPU", {});
  }
  return session_options;
}

#if defined(_WIN32)
inline std::wstring ToOrtPath(const std::string& s) {
  return std::wstring(s.begin(), s.end());  // sketch-level: ASCII paths only
}
#else
inline const std::string& ToOrtPath(const std::string& s) { return s; }
#endif

inline std::vector<std::string> InputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& alloc) {
  std::vector<std::string> names;
  for (size_t i = 0; i < session.GetInputCount(); ++i)
    names.emplace_back(session.GetInputNameAllocated(i, alloc).get());
  return names;
}

inline std::vector<std::string> OutputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& alloc) {
  std::vector<std::string> names;
  for (size_t i = 0; i < session.GetOutputCount(); ++i)
    names.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
  return names;
}

inline std::vector<const char*> AsCStrs(const std::vector<std::string>& v) {
  std::vector<const char*> out;
  out.reserve(v.size());
  for (auto& s : v) out.push_back(s.c_str());
  return out;
}

// Wraps an existing tensor's buffer in a fresh non-owning Ort::Value, so the
// same underlying data (a persistent cache entry, or encoder_hidden_states
// held for the whole Generate() call) can be passed into Run() again without
// moving it out of wherever it's actually owned. Same technique as
// onnxsim/dlpack_bridge.h's BorrowAsOrtValue, just against an existing
// Ort::Value instead of a DLManagedTensor.
//
// Sketch-level limitation: only fp32/int64/int8 tensors are handled -- fp32
// and int64 cover a plain (non-fp16) optimum-onnx export, and int8 covers a
// KV cache quantized by onnxsim.quantize_kv_cache
// (onnxsim/kv_cache_quantization.py), whose past_key_values.*/present.*
// tensors are INT8 by construction. A real deployment of an fp16-weight
// model needs this switch extended further.
inline Ort::Value BorrowView(const Ort::Value& src) {
  auto info = src.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = info.GetShape();
  static Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      auto* data = const_cast<float*>(src.GetTensorData<float>());
      return Ort::Value::CreateTensor<float>(mem_info, data, info.GetElementCount(), shape.data(), shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      auto* data = const_cast<int64_t*>(src.GetTensorData<int64_t>());
      return Ort::Value::CreateTensor<int64_t>(mem_info, data, info.GetElementCount(), shape.data(), shape.size());
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      auto* data = const_cast<int8_t*>(src.GetTensorData<int8_t>());
      return Ort::Value::CreateTensor<int8_t>(mem_info, data, info.GetElementCount(), shape.data(), shape.size());
    }
    default:
      throw std::runtime_error("BorrowView: unsupported dtype (extend for fp16/bf16 models)");
  }
}

}  // namespace detail

// Loads and drives one optimum-onnx export directory's worth of sessions.
class KvCachePipeline {
 public:
  // model_dir must contain decoder_model.onnx + decoder_with_past_model.onnx
  // (the `no_post_process=True` export shape -- see README.md for why the
  // merged decoder_model_merged.onnx shape isn't supported here), plus
  // encoder_model.onnx if this is a seq2seq export. `pipeline_options`
  // selects the execution provider every session runs on (default: CPU) --
  // see PipelineOptions above.
  KvCachePipeline(Ort::Env& env, const std::string& model_dir, const PipelineOptions& pipeline_options = {})
      : env_(env) {
    namespace fs = std::filesystem;
    auto load = [&](const std::string& filename) -> std::unique_ptr<Ort::Session> {
      fs::path p = fs::path(model_dir) / filename;
      if (!fs::exists(p)) return nullptr;
      Ort::SessionOptions options = detail::BuildSessionOptions(pipeline_options);
      return std::make_unique<Ort::Session>(env_, detail::ToOrtPath(p.string()).c_str(), options);
    };

    encoder_session_ = load("encoder_model.onnx");
    decoder_session_ = load("decoder_model.onnx");
    decoder_past_session_ = load("decoder_with_past_model.onnx");

    if (!decoder_session_ || !decoder_past_session_) {
      throw std::runtime_error(
          model_dir +
          ": expected decoder_model.onnx and decoder_with_past_model.onnx (the "
          "optimum-onnx no_post_process=True export shape -- see README.md)");
    }

    if (encoder_session_) {
      encoder_input_names_ = detail::InputNames(*encoder_session_, allocator_);
      encoder_output_names_ = detail::OutputNames(*encoder_session_, allocator_);
    }
    decoder_input_names_ = detail::InputNames(*decoder_session_, allocator_);
    decoder_output_names_ = detail::OutputNames(*decoder_session_, allocator_);
    decoder_past_input_names_ = detail::InputNames(*decoder_past_session_, allocator_);
    decoder_past_output_names_ = detail::OutputNames(*decoder_past_session_, allocator_);
  }

  bool is_seq2seq() const { return encoder_session_ != nullptr; }

  // Greedy decode, batch size 1. `input_ids` is the encoder input for
  // seq2seq models, or the decoder prompt for decoder-only models. Returns
  // the newly generated token ids (not including the prompt).
  std::vector<int64_t> Generate(const std::vector<int64_t>& input_ids, const GenerationConfig& config) {
    std::optional<Ort::Value> encoder_hidden_states;
    int64_t encoder_seq_len = 0;
    if (is_seq2seq()) {
      encoder_hidden_states = RunEncoder(input_ids, encoder_seq_len);
    }

    KvCache cache;
    std::vector<int64_t> decoder_tokens =
        is_seq2seq() ? std::vector<int64_t>{config.decoder_start_token_id} : input_ids;
    std::vector<int64_t> generated;
    bool use_past = false;

    for (int64_t step = 0; step < config.max_new_tokens; ++step) {
      std::vector<int64_t> step_input = use_past ? std::vector<int64_t>{decoder_tokens.back()} : decoder_tokens;
      int64_t total_len = static_cast<int64_t>(decoder_tokens.size());

      Ort::Value logits =
          RunDecoderStep(step_input, encoder_hidden_states ? &*encoder_hidden_states : nullptr, encoder_seq_len,
                          total_len, cache, use_past);
      int64_t next_token = ArgmaxLastToken(logits);
      generated.push_back(next_token);
      decoder_tokens.push_back(next_token);
      use_past = true;

      if (config.eos_token_id >= 0 && next_token == config.eos_token_id) break;
    }
    return generated;
  }

 private:
  Ort::Value RunEncoder(const std::vector<int64_t>& input_ids, int64_t& out_seq_len) {
    out_seq_len = static_cast<int64_t>(input_ids.size());
    std::vector<int64_t> ids_copy = input_ids;
    std::vector<int64_t> mask(input_ids.size(), 1);
    std::vector<int64_t> shape = {1, out_seq_len};

    std::vector<Ort::Value> input_values;
    std::vector<std::string> names_used;
    for (const auto& name : encoder_input_names_) {
      if (name == "input_ids") {
        input_values.push_back(
            Ort::Value::CreateTensor<int64_t>(mem_info_, ids_copy.data(), ids_copy.size(), shape.data(), shape.size()));
      } else if (name == "attention_mask") {
        input_values.push_back(
            Ort::Value::CreateTensor<int64_t>(mem_info_, mask.data(), mask.size(), shape.data(), shape.size()));
      } else {
        throw std::runtime_error("unrecognized encoder input: " + name);
      }
      names_used.push_back(name);
    }

    auto input_c = detail::AsCStrs(names_used);
    auto output_c = detail::AsCStrs(encoder_output_names_);
    auto outputs = encoder_session_->Run(Ort::RunOptions{nullptr}, input_c.data(), input_values.data(),
                                          input_values.size(), output_c.data(), output_c.size());
    // last_hidden_state is conventionally the encoder's only output.
    return std::move(outputs.front());
  }

  // Runs decoder_model.onnx (use_past=false, step 0) or
  // decoder_with_past_model.onnx (use_past=true, every step after), and
  // harvests this call's "present.*" outputs into `cache` for the next call.
  Ort::Value RunDecoderStep(const std::vector<int64_t>& step_input_ids, const Ort::Value* encoder_hidden_states,
                             int64_t encoder_seq_len, int64_t total_len, KvCache& cache, bool use_past) {
    Ort::Session& session = use_past ? *decoder_past_session_ : *decoder_session_;
    const auto& in_names = use_past ? decoder_past_input_names_ : decoder_input_names_;
    const auto& out_names = use_past ? decoder_past_output_names_ : decoder_output_names_;

    std::vector<int64_t> ids_copy = step_input_ids;
    std::vector<int64_t> ids_shape = {1, static_cast<int64_t>(ids_copy.size())};
    std::vector<int64_t> causal_mask(static_cast<size_t>(total_len), 1);
    std::vector<int64_t> causal_mask_shape = {1, total_len};
    std::vector<int64_t> enc_mask(static_cast<size_t>(encoder_seq_len), 1);
    std::vector<int64_t> enc_mask_shape = {1, encoder_seq_len};

    std::vector<Ort::Value> input_values;
    std::vector<std::string> names_used;
    input_values.reserve(in_names.size());
    names_used.reserve(in_names.size());

    for (const auto& name : in_names) {
      if (name == "input_ids") {
        input_values.push_back(Ort::Value::CreateTensor<int64_t>(mem_info_, ids_copy.data(), ids_copy.size(),
                                                                   ids_shape.data(), ids_shape.size()));
      } else if (name == "attention_mask") {
        // Decoder-only (causal LM) graphs: spans the full context so far.
        input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            mem_info_, causal_mask.data(), causal_mask.size(), causal_mask_shape.data(), causal_mask_shape.size()));
      } else if (name == "encoder_attention_mask") {
        input_values.push_back(Ort::Value::CreateTensor<int64_t>(mem_info_, enc_mask.data(), enc_mask.size(),
                                                                   enc_mask_shape.data(), enc_mask_shape.size()));
      } else if (name == "encoder_hidden_states") {
        if (!encoder_hidden_states)
          throw std::runtime_error("decoder graph wants encoder_hidden_states but no encoder ran (bad export?)");
        input_values.push_back(detail::BorrowView(*encoder_hidden_states));
      } else if (name.rfind("past_key_values.", 0) == 0) {
        auto it = cache.find(name);
        if (it == cache.end())
          throw std::runtime_error("missing cache entry for " + name + " (Generate() called out of order?)");
        // Borrow, don't move: some architectures (e.g. T5 cross-attention)
        // only produce a fresh "present.*" for part of the cache each step,
        // so an untouched entry must stay owned in `cache` for reuse next
        // step too -- see HarvestPresentIntoCache below.
        input_values.push_back(detail::BorrowView(it->second));
      } else {
        throw std::runtime_error("unrecognized decoder input: " + name + " (extend RunDecoderStep for this export)");
      }
      names_used.push_back(name);
    }

    auto input_c = detail::AsCStrs(names_used);
    auto output_c = detail::AsCStrs(out_names);
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_c.data(), input_values.data(), input_values.size(),
                                output_c.data(), output_c.size());

    HarvestPresentIntoCache(outputs, out_names, cache);

    for (size_t i = 0; i < out_names.size(); ++i)
      if (out_names[i] == "logits") return std::move(outputs[i]);
    throw std::runtime_error("decoder graph has no 'logits' output");
  }

  // The actual "glue": every output named "present.<rest>" becomes the
  // input named "past_key_values.<rest>" for the *next* RunDecoderStep call
  // -- purely by string substitution, so this needs no knowledge of layer
  // count, head count, or architecture. Ort::Value is move-only, so this is
  // a pointer/ownership handoff, not a tensor copy.
  static void HarvestPresentIntoCache(std::vector<Ort::Value>& outputs, const std::vector<std::string>& output_names,
                                       KvCache& cache) {
    static const std::string kPresentPrefix = "present.";
    static const std::string kPastPrefix = "past_key_values.";
    for (size_t i = 0; i < output_names.size(); ++i) {
      const std::string& name = output_names[i];
      if (name.rfind(kPresentPrefix, 0) != 0) continue;
      // std::map::operator[] would default-construct a Value first (it has
      // no default constructor), so insert_or_assign instead.
      cache.insert_or_assign(kPastPrefix + name.substr(kPresentPrefix.size()), std::move(outputs[i]));
    }
  }

  // Greedy argmax over the last position's logits. Batch size 1 only --
  // batching would need this indexed per batch row instead. Swap this out
  // for temperature/top-k/top-p sampling as needed.
  int64_t ArgmaxLastToken(const Ort::Value& logits) const {
    auto info = logits.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();  // [batch, seq, vocab]
    int64_t vocab = shape.back();
    int64_t seq = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
    const float* data = logits.GetTensorData<float>();
    const float* last_row = data + (seq - 1) * vocab;
    int64_t best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < vocab; ++v) {
      if (last_row[v] > best_val) {
        best_val = last_row[v];
        best = v;
      }
    }
    return best;
  }

  Ort::Env& env_;
  std::unique_ptr<Ort::Session> encoder_session_;
  std::unique_ptr<Ort::Session> decoder_session_;       // decoder_model.onnx (no past)
  std::unique_ptr<Ort::Session> decoder_past_session_;  // decoder_with_past_model.onnx
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo mem_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::string> encoder_input_names_, encoder_output_names_;
  std::vector<std::string> decoder_input_names_, decoder_output_names_;
  std::vector<std::string> decoder_past_input_names_, decoder_past_output_names_;
};

}  // namespace onnx_deploy
