// onnx-finetune-distill-step-graph: run a knowledge-distillation training
// step graph produced by scripts/generate_distillation_step_graph.py.
//
// That script builds the whole forward pass, the KD loss, the backward pass,
// and an Adam update as ordinary ONNX nodes -- onnxsim's own
// graph_grad/qat_graph autodiff, not onnxruntime.training -- baked into one
// self-contained ONNX graph. This tool's entire job is what
// onnxsim.qat_graph.run_step_graph does in Python: run that graph repeatedly
// on a plain Ort::Session, feeding a batch in and each step's outputs back
// in as the next step's weight inputs.
//
// This is a SEPARATE binary from ../main.cpp (the artifacts-dir/
// TrainingSession tool for every other --loss mode) specifically so it can
// link the plain, non-training onnxruntime C++ API (onnxruntime_cxx_api.h)
// instead of onnxruntime_training_cxx_api.h -- a from-source
// --enable_training_apis build is not needed to compile or run this tool at
// all, see ../README.md's "Knowledge distillation (graph_grad)" section.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

struct Args {
  std::string step_graph;  // manifest/initial-state paths are derived from this
  std::string teacher_model;
  std::string teacher_output_name;
  std::string train_input;
  std::string train_target;  // raw int64 class indices, num_samples of them
  int64_t num_samples = 0;
  int64_t epochs = 10;
  double lr = 1e-3;
  int log_every = 50;
  std::string output_weights;
};

[[noreturn]] void Usage(const char* prog) {
  std::fprintf(stderr,
      "usage: %s --step-graph FILE --teacher-model FILE\n"
      "          --train-input FILE --train-target FILE --num-samples N\n"
      "          --output-weights FILE [--epochs N] [--lr F] [--log-every N]\n"
      "          [--teacher-output-name NAME]\n\n"
      "FILE is a step graph from scripts/generate_distillation_step_graph.py;\n"
      "FILE.manifest.txt and FILE.initial_state.bin (written alongside it) are\n"
      "read too. --train-input is a raw contiguous float32 binary file\n"
      "(num_samples * input_dim floats, input_dim from the manifest);\n"
      "--train-target is num_samples raw int64 class indices (turned into the\n"
      "one-hot matrix the step graph's loss expects, via the manifest's\n"
      "num_classes). The batch size is whatever the step graph was built for\n"
      "(the manifest's input_shape) -- there is no --batch-size here, since\n"
      "the graph's own shapes are fixed at build time.\n\n"
      "Writes the final weights as a raw float32 blob to --output-weights, in\n"
      "the manifest's own order -- reassemble them into an inference-ready\n"
      ".onnx with scripts/apply_trained_weights.py.\n",
      prog);
  std::exit(1);
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) Usage(argv[0]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--step-graph") a.step_graph = need(i);
    else if (arg == "--teacher-model") a.teacher_model = need(i);
    else if (arg == "--teacher-output-name") a.teacher_output_name = need(i);
    else if (arg == "--train-input") a.train_input = need(i);
    else if (arg == "--train-target") a.train_target = need(i);
    else if (arg == "--num-samples") a.num_samples = std::stoll(need(i));
    else if (arg == "--epochs") a.epochs = std::stoll(need(i));
    else if (arg == "--lr") a.lr = std::stod(need(i));
    else if (arg == "--log-every") a.log_every = std::stoi(need(i));
    else if (arg == "--output-weights") a.output_weights = need(i);
    else if (arg == "-h" || arg == "--help") Usage(argv[0]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      Usage(argv[0]);
    }
  }
  if (a.step_graph.empty() || a.teacher_model.empty() || a.train_input.empty() ||
      a.train_target.empty() || a.num_samples <= 0 || a.output_weights.empty()) {
    Usage(argv[0]);
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
// session, or exits with an error -- the only model this tool loads besides
// the step graph itself, the frozen teacher, always has exactly one of each.
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

struct StepGraphManifest {
  std::string input_name;
  std::vector<int64_t> input_shape;
  std::string teacher_logits_name;
  std::vector<int64_t> teacher_logits_shape;
  std::string labels_onehot_name;
  int64_t rows = 0;
  int64_t num_classes = 0;
  std::string loss_name;
  // {state input name, state output name, shape} -- covers a weight and its
  // Adam __m/__v moments alike, all three the same shape.
  std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> state;
  // {weight name, shape}, in the exact order .initial_state.bin/--output-weights
  // concatenate their values in.
  std::vector<std::pair<std::string, std::vector<int64_t>>> weights;
};

StepGraphManifest ReadManifest(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  StepGraphManifest m;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    std::string tag;
    iss >> tag;
    if (tag == "input_name") {
      iss >> m.input_name;
    } else if (tag == "input_shape") {
      int64_t d;
      while (iss >> d) m.input_shape.push_back(d);
    } else if (tag == "teacher_logits_name") {
      iss >> m.teacher_logits_name;
    } else if (tag == "teacher_logits_shape") {
      int64_t d;
      while (iss >> d) m.teacher_logits_shape.push_back(d);
    } else if (tag == "labels_onehot_name") {
      iss >> m.labels_onehot_name;
    } else if (tag == "rows") {
      iss >> m.rows;
    } else if (tag == "num_classes") {
      iss >> m.num_classes;
    } else if (tag == "loss_name") {
      iss >> m.loss_name;
    } else if (tag == "state") {
      std::string state_input, state_output;
      iss >> state_input >> state_output;
      std::vector<int64_t> shape;
      int64_t d;
      while (iss >> d) shape.push_back(d);
      m.state.emplace_back(state_input, state_output, shape);
    } else if (tag == "weight") {
      std::string name;
      iss >> name;
      std::vector<int64_t> shape;
      int64_t d;
      while (iss >> d) shape.push_back(d);
      m.weights.emplace_back(name, shape);
    }
  }
  if (m.input_name.empty() || m.loss_name.empty() || m.state.empty() || m.weights.empty()) {
    std::fprintf(stderr, "error: %s does not look like a step-graph manifest\n", path.c_str());
    std::exit(1);
  }
  return m;
}

int64_t Prod(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (int64_t d : shape) total *= d;
  return total;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  StepGraphManifest manifest = ReadManifest(args.step_graph + ".manifest.txt");
  const int64_t batch_size = manifest.input_shape.empty() ? 0 : manifest.input_shape[0];
  const int64_t input_dim = batch_size > 0 ? Prod(manifest.input_shape) / batch_size : 0;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-finetune-distill-step-graph");
  Ort::SessionOptions session_options;
  Ort::Session step_session(env, args.step_graph.c_str(), session_options);
  Ort::Session teacher_session(env, args.teacher_model.c_str(), session_options);
  std::string teacher_input_name = SoleIoName(teacher_session, /*is_input=*/true, "teacher");
  std::string teacher_output_name = !args.teacher_output_name.empty()
      ? args.teacher_output_name
      : SoleIoName(teacher_session, /*is_input=*/false, "teacher");

  // Every state tensor's current value -- weights start from the student
  // model's own trained-so-far values (.initial_state.bin); Adam's __m/__v
  // moments start at zero, matching generate_distillation_step_graph.py's
  // own `initial_state` exactly.
  std::unordered_map<std::string, std::vector<float>> state;
  {
    std::ifstream state_file(args.step_graph + ".initial_state.bin", std::ios::binary);
    if (!state_file) {
      std::fprintf(stderr, "error: cannot open %s.initial_state.bin\n", args.step_graph.c_str());
      std::exit(1);
    }
    for (const auto& [name, shape] : manifest.weights) {
      std::vector<float> values(static_cast<size_t>(Prod(shape)));
      state_file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
      if (!state_file) {
        std::fprintf(stderr, "error: %s.initial_state.bin is shorter than the manifest expects\n",
                     args.step_graph.c_str());
        std::exit(1);
      }
      state[name] = std::move(values);
    }
  }
  for (const auto& [state_input, state_output, shape] : manifest.state) {
    (void)state_output;
    if (state.count(state_input) == 0) {
      state[state_input] = std::vector<float>(static_cast<size_t>(Prod(shape)), 0.0f);
    }
  }

  std::vector<float> inputs = ReadRawFloats(args.train_input, static_cast<size_t>(args.num_samples) * input_dim);
  std::vector<int64_t> labels = ReadRawInt64s(args.train_target, static_cast<size_t>(args.num_samples));

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> order(args.num_samples);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);

  std::vector<float> batch_input(static_cast<size_t>(batch_size) * input_dim);
  std::vector<int64_t> batch_labels(static_cast<size_t>(batch_size));
  std::vector<float> batch_teacher_logits(static_cast<size_t>(Prod(manifest.teacher_logits_shape)));
  std::vector<float> batch_onehot(static_cast<size_t>(manifest.rows) * static_cast<size_t>(manifest.num_classes));

  std::vector<const char*> output_name_ptrs;
  output_name_ptrs.push_back(manifest.loss_name.c_str());
  for (const auto& [state_input, state_output, shape] : manifest.state) {
    (void)state_input;
    (void)shape;
    output_name_ptrs.push_back(state_output.c_str());
  }

  int64_t global_step = 0;
  for (int64_t epoch = 0; epoch < args.epochs; ++epoch) {
    std::shuffle(order.begin(), order.end(), rng);

    for (int64_t start = 0; start + batch_size <= args.num_samples; start += batch_size) {
      for (int64_t b = 0; b < batch_size; ++b) {
        int64_t src = order[start + b];
        std::copy_n(inputs.begin() + src * input_dim, input_dim, batch_input.begin() + b * input_dim);
        batch_labels[b] = labels[src];
      }

      const char* teacher_input_names[] = {teacher_input_name.c_str()};
      const char* teacher_output_names[] = {teacher_output_name.c_str()};
      Ort::Value teacher_input = Ort::Value::CreateTensor<float>(
          mem_info, batch_input.data(), batch_input.size(),
          manifest.input_shape.data(), manifest.input_shape.size());
      auto teacher_outputs = teacher_session.Run(
          Ort::RunOptions{nullptr}, teacher_input_names, &teacher_input, 1, teacher_output_names, 1);
      std::copy_n(teacher_outputs[0].GetTensorData<float>(), batch_teacher_logits.size(),
                  batch_teacher_logits.begin());

      // The one-hot label matrix, built here on the host rather than in the
      // step graph itself -- see generate_distillation_step_graph.py's
      // module docstring on why (keeps Cast/Greater/Less, which have no
      // graph_grad VJP rule, out of the differentiated slice entirely).
      std::fill(batch_onehot.begin(), batch_onehot.end(), 0.0f);
      for (int64_t r = 0; r < manifest.rows && r < batch_size; ++r) {
        batch_onehot[static_cast<size_t>(r) * manifest.num_classes + batch_labels[r]] = 1.0f;
      }

      std::vector<Ort::Value> feeds;
      std::vector<const char*> feed_names;
      feed_names.push_back(manifest.input_name.c_str());
      feeds.push_back(Ort::Value::CreateTensor<float>(
          mem_info, batch_input.data(), batch_input.size(),
          manifest.input_shape.data(), manifest.input_shape.size()));
      feed_names.push_back(manifest.teacher_logits_name.c_str());
      feeds.push_back(Ort::Value::CreateTensor<float>(
          mem_info, batch_teacher_logits.data(), batch_teacher_logits.size(),
          manifest.teacher_logits_shape.data(), manifest.teacher_logits_shape.size()));
      std::vector<int64_t> onehot_shape = {manifest.rows, manifest.num_classes};
      feed_names.push_back(manifest.labels_onehot_name.c_str());
      feeds.push_back(Ort::Value::CreateTensor<float>(
          mem_info, batch_onehot.data(), batch_onehot.size(), onehot_shape.data(), onehot_shape.size()));

      float lr = static_cast<float>(args.lr);
      float m_correction = static_cast<float>(1.0 / (1.0 - std::pow(0.9, global_step + 1)));
      float v_correction = static_cast<float>(1.0 / (1.0 - std::pow(0.999, global_step + 1)));
      std::vector<int64_t> scalar_shape;  // rank 0
      feed_names.push_back("lr");
      feeds.push_back(Ort::Value::CreateTensor<float>(mem_info, &lr, 1, scalar_shape.data(), 0));
      feed_names.push_back("m_correction");
      feeds.push_back(Ort::Value::CreateTensor<float>(mem_info, &m_correction, 1, scalar_shape.data(), 0));
      feed_names.push_back("v_correction");
      feeds.push_back(Ort::Value::CreateTensor<float>(mem_info, &v_correction, 1, scalar_shape.data(), 0));

      for (const auto& [state_input, state_output, shape] : manifest.state) {
        (void)state_output;
        feed_names.push_back(state_input.c_str());
        feeds.push_back(Ort::Value::CreateTensor<float>(
            mem_info, state[state_input].data(), state[state_input].size(), shape.data(), shape.size()));
      }

      auto outputs = step_session.Run(
          Ort::RunOptions{nullptr}, feed_names.data(), feeds.data(), feeds.size(),
          output_name_ptrs.data(), output_name_ptrs.size());

      for (size_t i = 0; i < manifest.state.size(); ++i) {
        const auto& [state_input, state_output, shape] = manifest.state[i];
        (void)state_output;
        const float* data = outputs[1 + i].GetTensorData<float>();
        std::copy_n(data, static_cast<size_t>(Prod(shape)), state[state_input].begin());
      }

      if (args.log_every > 0 && global_step % args.log_every == 0) {
        float loss = *outputs[0].GetTensorData<float>();
        std::printf("epoch %lld step %lld loss %.6f\n", static_cast<long long>(epoch),
                    static_cast<long long>(global_step), loss);
      }
      ++global_step;
    }
  }

  std::ofstream out(args.output_weights, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "error: cannot open %s for writing\n", args.output_weights.c_str());
    std::exit(1);
  }
  for (const auto& [name, shape] : manifest.weights) {
    (void)shape;
    const auto& values = state[name];
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  }
  std::printf("wrote trained weights -> %s (apply with scripts/apply_trained_weights.py)\n",
              args.output_weights.c_str());
  return 0;
}
