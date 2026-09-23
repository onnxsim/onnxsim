// adb-shell native harness: ONNX Runtime + QNN EP plugin + Qualcomm's own QNN runtime libs,
// all pushed to one directory, no Android app (see ../qnn_shell_findings.md).
//
// usage: qnn_run <model.onnx> <input.bin|-> <mode> <iters> <out_prefix> [ctx.onnx]
//   mode: cpu | htp (strict, no CPU fallback) | htp-fallback
//   input.bin: raw float32 for the model's single input ("-" = zeros)
//   ctx.onnx: if given (htp modes), compile an EP-context model to this path first
//             (or load it directly if it already exists) and run that.
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s model input mode iters out_prefix [ctx]\n", argv[0]);
    return 2;
  }
  const std::string model = argv[1], input_path = argv[2], mode = argv[3];
  const int iters = atoi(argv[4]);
  const std::string out_prefix = argv[5];
  const std::string ctx = argc > 6 ? argv[6] : "";
  const char* qnn_lib = getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB") : "libonnxruntime_providers_qnn.so";
  const int log_level = getenv("ORT_LOG") ? atoi(getenv("ORT_LOG")) : ORT_LOGGING_LEVEL_WARNING;

  try {
    Ort::Env env(static_cast<OrtLoggingLevel>(log_level), "qnn_run");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(getenv("ORT_THREADS") ? atoi(getenv("ORT_THREADS")) : 1);
    so.SetLogSeverityLevel(log_level);

    if (mode != "cpu") {
      env.RegisterExecutionProviderLibrary("QNNExecutionProvider", qnn_lib);
      std::vector<Ort::ConstEpDevice> devs;
      for (const auto& d : env.GetEpDevices()) {
        printf("ep_device %s type=%d vendor=%s\n", d.EpName(), static_cast<int>(d.Device().Type()),
               d.Device().Vendor());
        if (std::string(d.EpName()) == "QNNExecutionProvider" &&
            d.Device().Type() == OrtHardwareDeviceType_NPU)
          devs.push_back(d);
      }
      if (devs.empty()) throw std::runtime_error("no QNN NPU ep device");
      std::unordered_map<std::string, std::string> opts{{"backend_type", "htp"}};
      if (getenv("QNN_PERF")) opts["htp_performance_mode"] = getenv("QNN_PERF");
      if (getenv("QNN_EXTRA")) {  // k=v,k=v
        std::string s = getenv("QNN_EXTRA");
        size_t p = 0;
        while (p < s.size()) {
          size_t c = s.find(',', p);
          std::string kv = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
          size_t e = kv.find('=');
          if (e != std::string::npos) opts[kv.substr(0, e)] = kv.substr(e + 1);
          if (c == std::string::npos) break;
          p = c + 1;
        }
      }
      for (auto& kv : opts) printf("qnn_opt %s=%s\n", kv.first.c_str(), kv.second.c_str());
      if (mode == "htp") so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      so.AppendExecutionProvider_V2(env, devs, opts);
    }

    std::string run_model = model;
    double t_prep = 0;
    if (!ctx.empty() && mode != "cpu") {
      std::ifstream exists(ctx);
      if (!exists) {
        double t0 = now_ms();
        Ort::ModelCompilationOptions co(env, so);
        co.SetInputModelPath(model.c_str());
        co.SetOutputModelPath(ctx.c_str());
        co.SetEpContextEmbedMode(true);
        Ort::Status st = Ort::CompileModel(env, co);
        if (!st.IsOK()) throw std::runtime_error("CompileModel: " + st.GetErrorMessage());
        printf("compile_ms %.1f\n", now_ms() - t0);
      }
      run_model = ctx;
    }

    double t0 = now_ms();
    Ort::Session sess(env, run_model.c_str(), so);
    t_prep = now_ms() - t0;
    printf("session_create_ms %.1f\n", t_prep);

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = sess.GetInputNameAllocated(0, alloc);
    auto in_info = sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto shape = in_info.GetShape();
    size_t n = 1;
    for (auto& d : shape) n *= (d < 0 ? 1 : d);
    std::vector<float> input(n, 0.f);
    if (input_path != "-") {
      std::ifstream f(input_path, std::ios::binary);
      f.read(reinterpret_cast<char*>(input.data()), n * sizeof(float));
      if (!f) throw std::runtime_error("short input file");
    }
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(mem, input.data(), n, shape.data(), shape.size());

    size_t nout = sess.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_names_hold;
    std::vector<const char*> out_names;
    for (size_t i = 0; i < nout; ++i) {
      out_names_hold.push_back(sess.GetOutputNameAllocated(i, alloc));
      out_names.push_back(out_names_hold.back().get());
    }
    const char* in_names[] = {in_name.get()};

    std::vector<Ort::Value> outs;
    std::vector<double> ts;
    for (int it = 0; it < iters; ++it) {
      double a = now_ms();
      outs = sess.Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names.data(), nout);
      ts.push_back(now_ms() - a);
      printf("run %d %.2f ms\n", it, ts.back());
    }
    for (size_t i = 0; i < nout; ++i) {
      auto ti = outs[i].GetTensorTypeAndShapeInfo();
      size_t cnt = ti.GetElementCount();
      std::string p = out_prefix + "_o" + std::to_string(i) + ".bin";
      FILE* f = fopen(p.c_str(), "wb");
      fwrite(outs[i].GetTensorData<float>(), sizeof(float), cnt, f);
      fclose(f);
      printf("out %zu %s elems=%zu\n", i, out_names[i], cnt);
    }
    printf("PASS mode=%s\n", mode.c_str());
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
  return 0;
}
