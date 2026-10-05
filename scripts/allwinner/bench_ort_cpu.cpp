// CPU baseline for the NPU benchmark: time an ONNX model on ONNX Runtime's CPU provider on the same device.
//   bench_ort_cpu MODEL.onnx THREADS ITERS      random float32 inputs (dynamic dims -> 1), prints median/min ms of Run()
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: bench_ort_cpu MODEL.onnx THREADS ITERS\n"; return 2; }
  const int threads = std::atoi(argv[2]), iters = std::atoi(argv[3]);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(threads);
  opts.SetInterOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session session(env, argv[1], opts);
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::string> in_names, out_names;
  std::vector<std::vector<float>> data;
  std::vector<Ort::Value> inputs;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    in_names.push_back(session.GetInputNameAllocated(i, alloc).get());
    auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { std::cerr << "only float32 inputs supported\n"; return 1; }
    std::vector<int64_t> shape = info.GetShape();
    size_t n = 1;
    for (auto& d : shape) { if (d < 1) d = 1; n *= static_cast<size_t>(d); }
    data.emplace_back(n);
    for (float& v : data.back()) v = dist(rng);
    inputs.push_back(Ort::Value::CreateTensor<float>(mem, data.back().data(), n, shape.data(), shape.size()));
  }
  for (size_t i = 0; i < session.GetOutputCount(); ++i) out_names.push_back(session.GetOutputNameAllocated(i, alloc).get());
  std::vector<const char*> in_c, out_c;
  for (auto& s : in_names) in_c.push_back(s.c_str());
  for (auto& s : out_names) out_c.push_back(s.c_str());

  session.Run(Ort::RunOptions{nullptr}, in_c.data(), inputs.data(), inputs.size(), out_c.data(), out_c.size());  // warm-up
  std::vector<double> ms;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr}, in_c.data(), inputs.data(), inputs.size(), out_c.data(), out_c.size());
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  std::cout << "ort_cpu threads=" << threads << " median " << ms[ms.size() / 2] << " ms, min " << ms.front() << " ms, max " << ms.back() << " ms over " << iters << " iters\n";
  return 0;
}
