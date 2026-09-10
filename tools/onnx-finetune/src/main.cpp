// onnx-finetune: run an ONNX Runtime on-device training loop against
// pre-generated training artifacts (see scripts/generate_artifacts.py) and
// export the result as a normal inference-ready ONNX model.
//
// No Python at runtime: this links only onnxruntime's training C++ API
// (and, in --teacher-model / distillation mode, a second, plain inference
// session for the frozen teacher -- still no Python, ONNX Runtime's normal
// C++ inference API).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "onnxruntime_training_cxx_api.h"

namespace {

struct Args {
  std::string artifacts_dir;
  std::string train_input;
  std::string train_target;
  int64_t input_dim = 0;
  int64_t target_dim = 0;
  int64_t num_samples = 0;
  int64_t batch_size = 8;
  int64_t epochs = 10;
  double lr = 1e-3;
  std::string output_model;
  std::string output_names;   // comma-separated
  std::string save_checkpoint;
  int log_every = 50;
  // "float32" (default, unchanged behavior) or "int64" -- SoftmaxCrossEntropyLoss
  // (--loss cross-entropy / --loss distillation in generate_artifacts.py)
  // needs int64 class-index labels, not float32.
  std::string label_dtype = "float32";
  // Distillation mode: a plain frozen inference model run each step on the
  // same input as the student, to produce teacher_logits. Training against
  // artifacts from `generate_artifacts.py --loss distillation` needs this;
  // omit for every other --loss mode.
  std::string teacher_model;
  std::string teacher_output_name;  // defaults to the teacher's sole output
};

[[noreturn]] void Usage(const char* prog) {
  std::fprintf(stderr,
      "usage: %s --artifacts-dir DIR --train-input FILE --train-target FILE\n"
      "          --input-dim N --target-dim N --num-samples N\n"
      "          --output-model FILE --output-names name1,name2,...\n"
      "          [--batch-size N] [--epochs N] [--lr F] [--save-checkpoint FILE]\n"
      "          [--log-every N] [--label-dtype float32|int64]\n"
      "          [--teacher-model FILE] [--teacher-output-name NAME]\n\n"
      "Trains against artifacts produced by scripts/generate_artifacts.py.\n"
      "--train-input is a raw contiguous float32 binary file\n"
      "  (num_samples * input_dim floats). --train-target is float32 by\n"
      "  default (num_samples * target_dim floats) or, with\n"
      "  --label-dtype int64, num_samples raw int64 class indices\n"
      "  (target_dim must be 1) -- required for --loss cross-entropy/\n"
      "  distillation artifacts, since SoftmaxCrossEntropyLoss expects\n"
      "  int64 labels, not float32.\n\n"
      "--teacher-model switches into distillation mode: artifacts must come\n"
      "  from `generate_artifacts.py --loss distillation`, whose training\n"
      "  graph declares 3 external inputs (the model's own input,\n"
      "  teacher_logits, labels) instead of the usual 2 -- this flag runs a\n"
      "  plain inference session against --teacher-model on each batch's\n"
      "  input to supply teacher_logits, so the training data files\n"
      "  themselves are unchanged (still just input + labels).\n",
      prog);
  std::exit(1);
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) out.push_back(item);
  return out;
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) Usage(argv[0]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--artifacts-dir") a.artifacts_dir = need(i);
    else if (arg == "--train-input") a.train_input = need(i);
    else if (arg == "--train-target") a.train_target = need(i);
    else if (arg == "--input-dim") a.input_dim = std::stoll(need(i));
    else if (arg == "--target-dim") a.target_dim = std::stoll(need(i));
    else if (arg == "--num-samples") a.num_samples = std::stoll(need(i));
    else if (arg == "--batch-size") a.batch_size = std::stoll(need(i));
    else if (arg == "--epochs") a.epochs = std::stoll(need(i));
    else if (arg == "--lr") a.lr = std::stod(need(i));
    else if (arg == "--output-model") a.output_model = need(i);
    else if (arg == "--output-names") a.output_names = need(i);
    else if (arg == "--save-checkpoint") a.save_checkpoint = need(i);
    else if (arg == "--log-every") a.log_every = std::stoi(need(i));
    else if (arg == "--label-dtype") a.label_dtype = need(i);
    else if (arg == "--teacher-model") a.teacher_model = need(i);
    else if (arg == "--teacher-output-name") a.teacher_output_name = need(i);
    else if (arg == "-h" || arg == "--help") Usage(argv[0]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      Usage(argv[0]);
    }
  }
  if (a.artifacts_dir.empty() || a.train_input.empty() || a.train_target.empty() ||
      a.input_dim <= 0 || a.target_dim <= 0 || a.num_samples <= 0 ||
      a.output_model.empty() || a.output_names.empty()) {
    Usage(argv[0]);
  }
  if (a.label_dtype != "float32" && a.label_dtype != "int64") {
    std::fprintf(stderr, "error: --label-dtype must be float32 or int64\n");
    Usage(argv[0]);
  }
  if (a.label_dtype == "int64" && a.target_dim != 1) {
    // SoftmaxCrossEntropyLoss's labels input is rank 1 (batch,), not rank 2
    // (batch, target_dim) -- onnxblock's CrossEntropyLoss/DistillationLoss
    // both build it by dropping the score tensor's trailing class dim
    // entirely, not shrinking it to size 1. --train-target still holds
    // exactly one int64 class index per sample either way, so --target-dim
    // stays the right knob for "how many raw values per sample in the
    // file" -- just constrained to 1 here rather than a separate flag.
    std::fprintf(stderr, "error: --label-dtype int64 requires --target-dim 1\n");
    std::exit(1);
  }
  if (!a.teacher_model.empty() && a.label_dtype == "float32") {
    // Distillation artifacts always use SoftmaxCrossEntropyLoss for the
    // hard-label term (see distillation_loss.py), which needs int64 --
    // rather than silently feeding it the wrong dtype, require the caller
    // to say so explicitly (this also documents the requirement to anyone
    // reading a --teacher-model invocation).
    std::fprintf(stderr, "error: --teacher-model (distillation) needs --label-dtype int64\n");
    std::exit(1);
  }
  return a;
}

std::vector<float> ReadRawFloats(const std::string& path, size_t expected_count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<float> data(expected_count);
  f.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
  if (!f) {
    std::fprintf(stderr, "error: %s is shorter than expected (%zu floats)\n", path.c_str(), expected_count);
    std::exit(1);
  }
  return data;
}

std::vector<int64_t> ReadRawInt64s(const std::string& path, size_t expected_count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<int64_t> data(expected_count);
  f.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(int64_t));
  if (!f) {
    std::fprintf(stderr, "error: %s is shorter than expected (%zu int64s)\n", path.c_str(), expected_count);
    std::exit(1);
  }
  return data;
}

// Returns the sole input/output name of a single-input/single-output
// session, or exits with an error -- every model this tool deals with
// (toy MLPs, the teacher in distillation mode) has exactly one of each.
std::string SoleIoName(const Ort::Session& session, bool is_input, const char* kind) {
  size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
  if (count != 1) {
    std::fprintf(stderr,
        "error: expected exactly one %s on the %s model, found %zu "
        "(use --teacher-output-name to disambiguate outputs)\n",
        is_input ? "input" : "output", kind, count);
    std::exit(1);
  }
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = is_input ? session.GetInputNameAllocated(0, allocator)
                       : session.GetOutputNameAllocated(0, allocator);
  return std::string(name.get());
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  std::vector<std::string> output_names_vec = Split(args.output_names, ',');
  const bool distilling = !args.teacher_model.empty();

  std::vector<float> inputs = ReadRawFloats(args.train_input, static_cast<size_t>(args.num_samples) * args.input_dim);
  std::vector<float> target_floats;
  std::vector<int64_t> target_int64s;
  if (args.label_dtype == "int64") {
    target_int64s = ReadRawInt64s(args.train_target, static_cast<size_t>(args.num_samples) * args.target_dim);
  } else {
    target_floats = ReadRawFloats(args.train_target, static_cast<size_t>(args.num_samples) * args.target_dim);
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-finetune");
  Ort::SessionOptions session_options;

  auto checkpoint_state = Ort::CheckpointState::LoadCheckpoint(args.artifacts_dir + "/checkpoint");
  Ort::TrainingSession train_session(
      env, session_options, checkpoint_state,
      args.artifacts_dir + "/training_model.onnx",
      args.artifacts_dir + "/eval_model.onnx",
      args.artifacts_dir + "/optimizer_model.onnx");

  train_session.SetLearningRate(static_cast<float>(args.lr));

  // Distillation mode: a second, plain (non-training) session for the
  // frozen teacher, run once per batch to produce teacher_logits -- see
  // generate_artifacts.py's DistillationLoss for why the training graph
  // expects that as a plain external input rather than embedding the
  // teacher's own forward pass.
  std::unique_ptr<Ort::Session> teacher_session;
  std::string teacher_input_name, teacher_output_name;
  int64_t teacher_output_dim = 0;
  if (distilling) {
    teacher_session = std::make_unique<Ort::Session>(env, args.teacher_model.c_str(), session_options);
    teacher_input_name = SoleIoName(*teacher_session, /*is_input=*/true, "teacher");
    teacher_output_name = !args.teacher_output_name.empty()
        ? args.teacher_output_name
        : SoleIoName(*teacher_session, /*is_input=*/false, "teacher");

    auto type_info = teacher_session->GetOutputTypeInfo(0);
    auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.empty() || shape.back() <= 0) {
      std::fprintf(stderr, "error: could not determine the teacher model's output class count statically\n");
      std::exit(1);
    }
    teacher_output_dim = shape.back();
  }

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> order(args.num_samples);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);

  std::vector<float> batch_input(static_cast<size_t>(args.batch_size) * args.input_dim);
  std::vector<float> batch_target_floats;
  std::vector<int64_t> batch_target_int64s;
  if (args.label_dtype == "int64") {
    batch_target_int64s.resize(static_cast<size_t>(args.batch_size) * args.target_dim);
  } else {
    batch_target_floats.resize(static_cast<size_t>(args.batch_size) * args.target_dim);
  }
  std::vector<float> batch_teacher_logits;
  if (distilling) {
    batch_teacher_logits.resize(static_cast<size_t>(args.batch_size) * teacher_output_dim);
  }

  int64_t global_step = 0;
  for (int64_t epoch = 0; epoch < args.epochs; ++epoch) {
    std::shuffle(order.begin(), order.end(), rng);

    for (int64_t start = 0; start + args.batch_size <= args.num_samples; start += args.batch_size) {
      for (int64_t b = 0; b < args.batch_size; ++b) {
        int64_t src = order[start + b];
        std::copy_n(inputs.begin() + src * args.input_dim, args.input_dim,
                    batch_input.begin() + b * args.input_dim);
        if (args.label_dtype == "int64") {
          std::copy_n(target_int64s.begin() + src * args.target_dim, args.target_dim,
                      batch_target_int64s.begin() + b * args.target_dim);
        } else {
          std::copy_n(target_floats.begin() + src * args.target_dim, args.target_dim,
                      batch_target_floats.begin() + b * args.target_dim);
        }
      }

      std::vector<int64_t> in_shape = {args.batch_size, args.input_dim};
      // int64 labels are rank 1 (batch,) -- see ParseArgs's --target-dim
      // check above for why -- float32 targets stay rank 2 (batch, target_dim)
      // as before, for the regression --loss modes' arbitrary output_dim.
      std::vector<int64_t> tgt_shape = args.label_dtype == "int64"
          ? std::vector<int64_t>{args.batch_size}
          : std::vector<int64_t>{args.batch_size, args.target_dim};

      std::vector<Ort::Value> step_inputs;
      step_inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, batch_input.data(), batch_input.size(), in_shape.data(), in_shape.size()));

      if (distilling) {
        std::vector<int64_t> teacher_in_shape = {args.batch_size, args.input_dim};
        Ort::Value teacher_input = Ort::Value::CreateTensor<float>(
            mem_info, batch_input.data(), batch_input.size(), teacher_in_shape.data(), teacher_in_shape.size());
        const char* teacher_input_names[] = {teacher_input_name.c_str()};
        const char* teacher_output_names[] = {teacher_output_name.c_str()};
        auto teacher_outputs = teacher_session->Run(
            Ort::RunOptions{nullptr}, teacher_input_names, &teacher_input, 1,
            teacher_output_names, 1);
        const float* logits_data = teacher_outputs[0].GetTensorData<float>();
        std::copy_n(logits_data, batch_teacher_logits.size(), batch_teacher_logits.begin());

        std::vector<int64_t> teacher_logits_shape = {args.batch_size, teacher_output_dim};
        step_inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, batch_teacher_logits.data(), batch_teacher_logits.size(),
            teacher_logits_shape.data(), teacher_logits_shape.size()));
      }

      if (args.label_dtype == "int64") {
        step_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem_info, batch_target_int64s.data(), batch_target_int64s.size(), tgt_shape.data(), tgt_shape.size()));
      } else {
        step_inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, batch_target_floats.data(), batch_target_floats.size(), tgt_shape.data(), tgt_shape.size()));
      }

      auto step_outputs = train_session.TrainStep(step_inputs);
      train_session.OptimizerStep();
      train_session.LazyResetGrad();

      if (args.log_every > 0 && global_step % args.log_every == 0 && !step_outputs.empty()) {
        float loss = *step_outputs[0].GetTensorData<float>();
        if (distilling && step_outputs.size() >= 3) {
          // generate_artifacts.py --loss distillation exposes the soft/hard
          // sub-losses as extra training outputs (additional_output_names)
          // right after the combined loss, for exactly this breakdown.
          float soft_loss = *step_outputs[1].GetTensorData<float>();
          float hard_loss = *step_outputs[2].GetTensorData<float>();
          std::printf("epoch %lld step %lld loss %.6f (soft %.6f, hard %.6f)\n",
                      static_cast<long long>(epoch), static_cast<long long>(global_step),
                      loss, soft_loss, hard_loss);
        } else {
          std::printf("epoch %lld step %lld loss %.6f\n",
                      static_cast<long long>(epoch), static_cast<long long>(global_step), loss);
        }
      }
      ++global_step;
    }
  }

  train_session.ExportModelForInferencing(args.output_model, output_names_vec);
  std::printf("wrote fine-tuned inference model -> %s\n", args.output_model.c_str());

  if (!args.save_checkpoint.empty()) {
    Ort::CheckpointState::SaveCheckpoint(checkpoint_state, args.save_checkpoint, /*include_optimizer_state=*/true);
    std::printf("wrote checkpoint -> %s\n", args.save_checkpoint.c_str());
  }

  return 0;
}
