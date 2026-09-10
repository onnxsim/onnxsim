// onnx-finetune-wasm: the same training loop as ../src/main.cpp, exposed to
// JS via Embind instead of a CLI. Runs entirely in-memory -- checkpoint and
// model bytes come in as JS typed arrays via ONNX Runtime training's buffer
// constructors (CheckpointState::LoadCheckpointFromBuffer, the
// std::vector<uint8_t> TrainingSession overload), so no virtual filesystem
// staging is needed for input. The one exception is ExportModelForInferencing,
// which only has a path-based signature upstream; that one write goes through
// Emscripten's in-memory MEMFS and gets read straight back out, never touching
// a real disk.
//
// Knowledge distillation is a separate tool entirely -- see
// ../distill_step_graph/step_graph_runner.mjs, which runs a self-contained
// step graph from ../../scripts/generate_distillation_step_graph.py
// (onnxsim's own graph_grad autodiff) via the official onnxruntime-web
// package's plain, non-training ort.InferenceSession. This binding never
// needs a second "teacher" model.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <fstream>
#include <string>
#include <vector>

#include "onnxruntime_training_cxx_api.h"

using emscripten::val;

namespace {

// One shared Ort::Env for the module's lifetime (matches the native CLI,
// which creates one per process -- here "process" is the wasm module instance).
Ort::Env& GlobalEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-finetune-wasm");
  return env;
}

}  // namespace

class FinetuneSession {
 public:
  // checkpoint_bytes / training_model_bytes / eval_model_bytes /
  // optimizer_model_bytes are JS Uint8Array (or any typed array) holding the
  // four files scripts/generate_artifacts.py produces -- fetch() them on the
  // JS side and pass the resulting ArrayBuffers straight in.
  FinetuneSession(const val& checkpoint_bytes, const val& training_model_bytes,
                   const val& eval_model_bytes, const val& optimizer_model_bytes)
  try : checkpoint_(Ort::CheckpointState::LoadCheckpointFromBuffer(
            emscripten::vecFromJSArray<uint8_t>(checkpoint_bytes))),
        session_(GlobalEnv(), Ort::SessionOptions{}, checkpoint_,
                 emscripten::vecFromJSArray<uint8_t>(training_model_bytes),
                 emscripten::vecFromJSArray<uint8_t>(eval_model_bytes),
                 emscripten::vecFromJSArray<uint8_t>(optimizer_model_bytes)) {
  } catch (const std::exception& e) {
    fprintf(stderr, "FinetuneSession construction threw: %s\n", e.what());
    throw;
  }

  void setLearningRate(float lr) { session_.SetLearningRate(lr); }

  // input/target are JS Float32Array, flattened row-major
  // [batch, inputDim]/[batch, targetDim], matching the raw binary layout the
  // native CLI reads from disk with its --label-dtype default (float32) --
  // there is no --label-dtype int64 support in this binding. Returns the
  // loss for this step.
  float trainStep(const val& input, const val& target, int batch, int input_dim, int target_dim) {
    std::vector<float> input_vec = emscripten::vecFromJSArray<float>(input);
    std::vector<float> target_vec = emscripten::vecFromJSArray<float>(target);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> in_shape = {batch, input_dim};
    std::vector<int64_t> tgt_shape = {batch, target_dim};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, input_vec.data(), input_vec.size(),
                                                       in_shape.data(), in_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, target_vec.data(), target_vec.size(),
                                                       tgt_shape.data(), tgt_shape.size()));

    try {
      auto outputs = session_.TrainStep(inputs);
      return outputs.empty() ? 0.f : *outputs[0].GetTensorData<float>();
    } catch (const std::exception& e) {
      fprintf(stderr, "TrainStep threw: %s\n", e.what());
      throw;
    }
  }

  void optimizerStep() { session_.OptimizerStep(); }
  void lazyResetGrad() { session_.LazyResetGrad(); }

  // output_names: JS array of strings, the *original* model's graph output
  // names (same meaning as --output-names in the native CLI). Returns a
  // Uint8Array of the fine-tuned inference-ready .onnx.
  val exportModel(const val& output_names) {
    std::vector<std::string> names = emscripten::vecFromJSArray<std::string>(output_names);

    // MEMFS is in-memory by default in a browser build (no IDBFS/NODEFS
    // persistence backing it), so this never touches a real disk.
    const char* tmp_path = "/onnx_finetune_export.onnx";
    try {
      session_.ExportModelForInferencing(tmp_path, names);
    } catch (const std::exception& e) {
      fprintf(stderr, "ExportModelForInferencing threw: %s\n", e.what());
      throw;
    }

    std::ifstream f(tmp_path, std::ios::binary);
    if (!f) {
      fprintf(stderr, "failed to reopen %s from MEMFS after export\n", tmp_path);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    fprintf(stderr, "read back %zu bytes from %s\n", bytes.size(), tmp_path);

    val Uint8Array = val::global("Uint8Array");
    val result = Uint8Array.new_(bytes.size());
    result.call<void>("set", val(emscripten::typed_memory_view(bytes.size(), bytes.data())));
    return result;
  }

 private:
  // Declaration order matters: TrainingSession holds a reference to
  // CheckpointState, so checkpoint_ must be constructed (and destroyed) around
  // session_'s lifetime.
  Ort::CheckpointState checkpoint_;
  Ort::TrainingSession session_;
};

EMSCRIPTEN_BINDINGS(onnx_finetune_wasm) {
  emscripten::class_<FinetuneSession>("FinetuneSession")
      .constructor<const val&, const val&, const val&, const val&>()
      .function("setLearningRate", &FinetuneSession::setLearningRate)
      .function("trainStep", &FinetuneSession::trainStep)
      .function("optimizerStep", &FinetuneSession::optimizerStep)
      .function("lazyResetGrad", &FinetuneSession::lazyResetGrad)
      .function("exportModel", &FinetuneSession::exportModel);
}
