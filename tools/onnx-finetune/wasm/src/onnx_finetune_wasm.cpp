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
// Distillation mode (an optional 5th constructor argument, teacher_model_
// bytes) mirrors main.cpp's --teacher-model: a second, plain (non-training)
// session run on each step's input to produce teacher_logits, which
// trainStep then feeds into the 3-input training graph
// generate_artifacts.py --loss distillation produces, instead of the usual
// 2. trainStep's own signature is unchanged either way -- distillation is
// entirely a constructor-time choice, not a per-step one.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <fstream>
#include <memory>
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

// Returns the sole input/output name of a single-input/single-output
// session, or throws -- every model this tool deals with (toy MLPs, the
// teacher in distillation mode) has exactly one of each.
std::string SoleIoName(const Ort::Session& session, bool is_input, const char* kind) {
  size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
  if (count != 1) {
    throw std::runtime_error(
        std::string("expected exactly one ") + (is_input ? "input" : "output") +
        " on the " + kind + " model, found " + std::to_string(count));
  }
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = is_input ? session.GetInputNameAllocated(0, allocator)
                       : session.GetOutputNameAllocated(0, allocator);
  return std::string(name.get());
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

  // Distillation-mode overload: teacher_model_bytes is a plain (non-
  // training) inference-ready .onnx, run internally on each trainStep's
  // input to supply teacher_logits. See generate_artifacts.py --loss
  // distillation for what training_model_bytes must look like for this to
  // make sense (a 3-input training graph, not the usual 2).
  FinetuneSession(const val& checkpoint_bytes, const val& training_model_bytes,
                   const val& eval_model_bytes, const val& optimizer_model_bytes,
                   const val& teacher_model_bytes)
  try : checkpoint_(Ort::CheckpointState::LoadCheckpointFromBuffer(
            emscripten::vecFromJSArray<uint8_t>(checkpoint_bytes))),
        session_(GlobalEnv(), Ort::SessionOptions{}, checkpoint_,
                 emscripten::vecFromJSArray<uint8_t>(training_model_bytes),
                 emscripten::vecFromJSArray<uint8_t>(eval_model_bytes),
                 emscripten::vecFromJSArray<uint8_t>(optimizer_model_bytes)),
        teacher_bytes_(emscripten::vecFromJSArray<uint8_t>(teacher_model_bytes)),
        teacher_session_(std::make_unique<Ort::Session>(
            GlobalEnv(), teacher_bytes_.data(), teacher_bytes_.size(), Ort::SessionOptions{})) {
    teacher_input_name_ = SoleIoName(*teacher_session_, /*is_input=*/true, "teacher");
    teacher_output_name_ = SoleIoName(*teacher_session_, /*is_input=*/false, "teacher");
    auto shape = teacher_session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.empty() || shape.back() <= 0) {
      throw std::runtime_error("could not determine the teacher model's output class count statically");
    }
    teacher_output_dim_ = shape.back();
  } catch (const std::exception& e) {
    fprintf(stderr, "FinetuneSession (distillation) construction threw: %s\n", e.what());
    throw;
  }

  void setLearningRate(float lr) { session_.SetLearningRate(lr); }

  // input/target are JS Float32Array, flattened row-major
  // [batch, inputDim]/[batch, targetDim], matching the raw binary layout the
  // native CLI reads from disk. target is int64-labels-as-Float32Array
  // (each element still a whole class index, just stored as a JS number) if
  // this session was constructed in distillation mode -- see labelsToInt64
  // below, mirroring main.cpp's --label-dtype int64. Returns the combined
  // loss for this step; call lastSubLosses() afterwards for the soft/hard
  // breakdown when distilling.
  float trainStep(const val& input, const val& target, int batch, int input_dim, int target_dim) {
    std::vector<float> input_vec = emscripten::vecFromJSArray<float>(input);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> in_shape = {batch, input_dim};
    std::vector<int64_t> tgt_shape = {batch, target_dim};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, input_vec.data(), input_vec.size(),
                                                       in_shape.data(), in_shape.size()));

    // Kept alive until TrainStep returns: CreateTensor doesn't copy.
    std::vector<float> teacher_logits_vec;
    std::vector<int64_t> target_int64_vec;

    if (teacher_session_) {
      const char* teacher_input_names[] = {teacher_input_name_.c_str()};
      const char* teacher_output_names[] = {teacher_output_name_.c_str()};
      std::vector<int64_t> teacher_in_shape = {batch, input_dim};
      Ort::Value teacher_input = Ort::Value::CreateTensor<float>(
          mem_info, input_vec.data(), input_vec.size(), teacher_in_shape.data(), teacher_in_shape.size());
      auto teacher_outputs = teacher_session_->Run(
          Ort::RunOptions{nullptr}, teacher_input_names, &teacher_input, 1, teacher_output_names, 1);
      const float* logits_data = teacher_outputs[0].GetTensorData<float>();
      teacher_logits_vec.assign(logits_data, logits_data + static_cast<size_t>(batch) * teacher_output_dim_);

      std::vector<int64_t> teacher_logits_shape = {batch, teacher_output_dim_};
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, teacher_logits_vec.data(), teacher_logits_vec.size(),
          teacher_logits_shape.data(), teacher_logits_shape.size()));

      // Distillation artifacts always need int64 labels (SoftmaxCrossEntropyLoss)
      // -- target arrives as a Float32Array from JS regardless (typed arrays
      // of mixed element type are awkward across the Embind boundary), so
      // convert here rather than asking every caller to marshal int64s.
      std::vector<float> target_float_vec = emscripten::vecFromJSArray<float>(target);
      target_int64_vec.resize(target_float_vec.size());
      for (size_t i = 0; i < target_float_vec.size(); ++i) {
        target_int64_vec[i] = static_cast<int64_t>(target_float_vec[i]);
      }
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          mem_info, target_int64_vec.data(), target_int64_vec.size(), tgt_shape.data(), tgt_shape.size()));
    } else {
      std::vector<float> target_vec = emscripten::vecFromJSArray<float>(target);
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, target_vec.data(), target_vec.size(),
                                                         tgt_shape.data(), tgt_shape.size()));
    }

    try {
      auto outputs = session_.TrainStep(inputs);
      last_sub_losses_available_ = teacher_session_ && outputs.size() >= 3;
      if (last_sub_losses_available_) {
        last_soft_loss_ = *outputs[1].GetTensorData<float>();
        last_hard_loss_ = *outputs[2].GetTensorData<float>();
      }
      return outputs.empty() ? 0.f : *outputs[0].GetTensorData<float>();
    } catch (const std::exception& e) {
      fprintf(stderr, "TrainStep threw: %s\n", e.what());
      throw;
    }
  }

  // {soft, hard} sub-losses from the most recent trainStep(), or undefined
  // if this session isn't in distillation mode (or generate_artifacts.py
  // wasn't asked for the additional_output_names that expose them). See
  // distillation_loss.py's DistillationLoss for what these mean.
  val lastSubLosses() const {
    if (!last_sub_losses_available_) return val::undefined();
    val result = val::object();
    result.set("soft", last_soft_loss_);
    result.set("hard", last_hard_loss_);
    return result;
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
  // session_'s lifetime. teacher_bytes_ must outlive teacher_session_ for the
  // same reason (the buffer-based Ort::Session constructor doesn't copy).
  Ort::CheckpointState checkpoint_;
  Ort::TrainingSession session_;
  std::vector<uint8_t> teacher_bytes_;
  std::unique_ptr<Ort::Session> teacher_session_;
  std::string teacher_input_name_;
  std::string teacher_output_name_;
  int64_t teacher_output_dim_ = 0;
  bool last_sub_losses_available_ = false;
  float last_soft_loss_ = 0.f;
  float last_hard_loss_ = 0.f;
};

EMSCRIPTEN_BINDINGS(onnx_finetune_wasm) {
  emscripten::class_<FinetuneSession>("FinetuneSession")
      .constructor<const val&, const val&, const val&, const val&>()
      .constructor<const val&, const val&, const val&, const val&, const val&>()
      .function("setLearningRate", &FinetuneSession::setLearningRate)
      .function("trainStep", &FinetuneSession::trainStep)
      .function("lastSubLosses", &FinetuneSession::lastSubLosses)
      .function("optimizerStep", &FinetuneSession::optimizerStep)
      .function("lazyResetGrad", &FinetuneSession::lazyResetGrad)
      .function("exportModel", &FinetuneSession::exportModel);
}
