// Multi-sample evaluation harness: load a model once (ORT + QNN EP, same env/QNN options as qnn_run_multi.cpp), run it on N
// samples one at a time, and report per-sample latency. For accuracy runs (e.g. all 872 SST-2 validation sentences) where one
// process per sample would pay session creation and HTP context load each time.
//
// usage: qnn_eval <model.onnx> <manifest.txt> <mode> <out_prefix> [ctx.onnx]
//   mode: cpu | htp (strict, no CPU fallback) | htp-fallback
//   manifest: one input per line, "<name> <f32|i64|i32|u8|u16> <file.bin> <N>,<d1>,<d2>,..."; the first dimension counts samples,
//             and each run feeds one sample of shape <d1>,<d2>,... (so a batch-1 model with input [1,64] is "N,1,64")
//   outputs: <out_prefix>_o<i>.bin = every sample's output i, concatenated in sample order, native dtype
//   env: WARMUP (default 5) untimed runs on sample 0; REPEAT (default 1) passes over the set for timing (outputs saved from the
//        first pass); QNN_PERF, QNN_EXTRA (k=v,k=v: e.g. enable_htp_fp16_precision=1), ORT_THREADS, ORT_LOG, QNN_EP_LIB as in
//        qnn_run_multi.cpp
//   prints "latency_ms median M p10 A p90 B mean C min D max E (n=K)" over the timed runs, then "PASS mode=<mode>".
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct In {
  std::string name, dtype;
  std::vector<int64_t> shape;  // per-sample shape
  size_t samples = 0, sample_bytes = 0;
  std::vector<char> data;
};

static size_t esize(const std::string& t) { return t == "i64" ? 8 : t == "u8" ? 1 : t == "u16" ? 2 : 4; }

static std::unordered_map<std::string, std::string> parse_kv(const char* env) {
  std::unordered_map<std::string, std::string> out;
  if (!env) return out;
  std::string s = env;
  size_t p = 0;
  while (p < s.size()) {
    size_t c = s.find(',', p);
    std::string kv = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
    size_t e = kv.find('=');
    if (e != std::string::npos) out[kv.substr(0, e)] = kv.substr(e + 1);
    if (c == std::string::npos) break;
    p = c + 1;
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s model manifest mode out_prefix [ctx]\n", argv[0]);
    return 2;
  }
  const std::string model = argv[1], manifest = argv[2], mode = argv[3], out_prefix = argv[4];
  const std::string ctx = argc > 5 ? argv[5] : "";
  const char* qnn_lib = getenv("QNN_EP_LIB") ? getenv("QNN_EP_LIB") : "libonnxruntime_providers_qnn.so";
  const int log_level = getenv("ORT_LOG") ? atoi(getenv("ORT_LOG")) : ORT_LOGGING_LEVEL_WARNING;
  const int warmup = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
  const int repeat = getenv("REPEAT") ? std::max(1, atoi(getenv("REPEAT"))) : 1;

  try {
    std::vector<In> ins;
    std::ifstream mf(manifest);
    std::string line;
    size_t n_samples = 0;
    while (std::getline(mf, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      In in;
      std::string file, dims;
      ss >> in.name >> in.dtype >> file >> dims;
      std::istringstream ds(dims);
      std::string d;
      std::vector<int64_t> all;
      while (std::getline(ds, d, ',')) all.push_back(atoll(d.c_str()));
      if (all.size() < 2) throw std::runtime_error("manifest dims need N,<per-sample dims>: " + line);
      in.samples = static_cast<size_t>(all[0]);
      in.shape.assign(all.begin() + 1, all.end());
      size_t per = 1;
      for (auto v : in.shape) per *= static_cast<size_t>(v);
      in.sample_bytes = per * esize(in.dtype);
      in.data.resize(in.samples * in.sample_bytes);
      std::ifstream f(file, std::ios::binary);
      f.read(in.data.data(), static_cast<std::streamsize>(in.data.size()));
      if (!f) throw std::runtime_error("short input file " + file);
      if (n_samples && n_samples != in.samples) throw std::runtime_error("inputs disagree on the sample count");
      n_samples = in.samples;
      ins.push_back(std::move(in));
    }
    if (!n_samples) throw std::runtime_error("empty manifest");

    Ort::Env env(static_cast<OrtLoggingLevel>(log_level), "qnn_eval");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(getenv("ORT_THREADS") ? atoi(getenv("ORT_THREADS")) : 1);
    so.SetLogSeverityLevel(log_level);

    if (mode != "cpu") {
      env.RegisterExecutionProviderLibrary("QNNExecutionProvider", qnn_lib);
      std::vector<Ort::ConstEpDevice> devs;
      for (const auto& d : env.GetEpDevices())
        if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU) devs.push_back(d);
      if (devs.empty()) throw std::runtime_error("no QNN NPU ep device");
      std::unordered_map<std::string, std::string> opts{{"backend_type", "htp"}};
      if (getenv("QNN_PERF")) opts["htp_performance_mode"] = getenv("QNN_PERF");
      for (auto& kv : parse_kv(getenv("QNN_EXTRA"))) opts[kv.first] = kv.second;
      if (mode == "htp") so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      so.AppendExecutionProvider_V2(env, devs, opts);
    }

    std::string run_model = model;
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
    printf("session_create_ms %.1f\n", now_ms() - t0);

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto elem_type = [](const std::string& t) {
      return t == "i64"   ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
             : t == "i32" ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
             : t == "u8"  ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
             : t == "u16" ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
                          : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };
    std::vector<const char*> in_names;
    for (auto& in : ins) in_names.push_back(in.name.c_str());

    Ort::AllocatorWithDefaultOptions alloc;
    const size_t nout = sess.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> hold;
    std::vector<const char*> out_names;
    for (size_t i = 0; i < nout; ++i) {
      hold.push_back(sess.GetOutputNameAllocated(i, alloc));
      out_names.push_back(hold.back().get());
    }

    auto run_sample = [&](size_t s) {
      std::vector<Ort::Value> xs;
      for (auto& in : ins)
        xs.push_back(Ort::Value::CreateTensor(mem, in.data.data() + s * in.sample_bytes, in.sample_bytes, in.shape.data(), in.shape.size(),
                                              elem_type(in.dtype)));
      return sess.Run(Ort::RunOptions{nullptr}, in_names.data(), xs.data(), xs.size(), out_names.data(), nout);
    };

    for (int w = 0; w < warmup; ++w) run_sample(0);
    std::vector<std::vector<char>> collected(nout);
    std::vector<std::string> out_dtype(nout), out_shape(nout);
    std::vector<double> ts;
    for (int pass = 0; pass < repeat; ++pass) {
      for (size_t s = 0; s < n_samples; ++s) {
        double a = now_ms();
        auto outs = run_sample(s);
        ts.push_back(now_ms() - a);
        if (pass != 0) continue;
        for (size_t i = 0; i < nout; ++i) {
          auto ti = outs[i].GetTensorTypeAndShapeInfo();
          const size_t cnt = ti.GetElementCount();
          const auto et = ti.GetElementType();
          const size_t es = et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? 1 : et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ? 2 : 4;
          const char* raw = static_cast<const char*>(outs[i].GetTensorRawData());
          collected[i].insert(collected[i].end(), raw, raw + cnt * es);
          if (s == 0) {
            out_dtype[i] = es == 8 ? "i64" : es == 1 ? "u8" : es == 2 ? "u16" : "f32";
            for (auto d : ti.GetShape()) out_shape[i] += std::to_string(d) + ",";
          }
        }
      }
    }
    std::vector<double> sorted = ts;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) { return sorted[std::min(sorted.size() - 1, static_cast<size_t>(p * sorted.size()))]; };
    printf("latency_ms median %.3f p10 %.3f p90 %.3f mean %.3f min %.3f max %.3f (n=%zu)\n", pct(0.5), pct(0.1), pct(0.9),
           std::accumulate(ts.begin(), ts.end(), 0.0) / ts.size(), sorted.front(), sorted.back(), ts.size());
    for (size_t i = 0; i < nout; ++i) {
      FILE* f = fopen((out_prefix + "_o" + std::to_string(i) + ".bin").c_str(), "wb");
      fwrite(collected[i].data(), 1, collected[i].size(), f);
      fclose(f);
      printf("out %zu %s %s %zu_samples_of_%s\n", i, out_names[i], out_dtype[i].c_str(), n_samples, out_shape[i].c_str());
    }
    printf("PASS mode=%s\n", mode.c_str());
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    return 1;
  }
  return 0;
}
