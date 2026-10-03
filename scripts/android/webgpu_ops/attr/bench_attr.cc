// Latency benchmark for one ONNX model on the ORT C API: CPU EP or WebGPU EP.
//   bench model.onnx cpu|webgpu WARMUP ITERS [threads=N] [dump=DIR] [dumplast=DIR] [sync=tiny.onnx] [shape=INPUT:d0,d1,...] [key=value ...]
// iobind=1 runs through OrtIoBinding (CPU inputs bound, outputs bound to CPU memory) with run option
// gpu_graph_id=0; needed for enableGraphCapture=1, where plain Run() returns no output tensors on replay.
// sync=tiny.onnx: a second, tiny float[1,4] model (X -> Add) run and read back after every iteration on the
// same WebGPU device. A readback waits for all earlier queued GPU work, so this makes graph-capture
// (async) runs measurable; the tiny model's own latency is measured alone and subtracted.
// Inputs are generated deterministically (seeded) so two providers see identical data: float ->
// uniform [0,1) (or a 0..255 ramp if the name/type suggests an image), int64/int32 -> 1, uint8 -> ramp,
// bool -> false. Dynamic dims become 1. Prints cold-first-run, median, mean, min, p90 in ms and the
// number of nodes that ran on each EP (from the profiler, when PROFILE=1).
#include <unistd.h>
#include <onnxruntime_c_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const OrtApi* g;
static std::string dump_first;
#define CK(x)                                                     \
  do {                                                            \
    OrtStatus* st_ = (x);                                         \
    if (st_) {                                                    \
      fprintf(stderr, "ERR %s: %s\n", #x, g->GetErrorMessage(st_)); \
      return 2;                                                   \
    }                                                             \
  } while (0)

static size_t elem_size(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
    default: return 0;
  }
}

static uint16_t f2h(float f) {  // round-to-nearest-even float -> half (normal range is all we need)
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
  int exp = ((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  uint16_t h = (uint16_t)(sign | (exp << 10) | (mant >> 13));
  if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) h++;
  return h;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: bench model cpu|webgpu WARMUP ITERS [threads=N] [dump=DIR] [k=v ...]\n");
    return 1;
  }
  g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const std::string model = argv[1], prov = argv[2];
  const int warm = atoi(argv[3]), iters = atoi(argv[4]);
  int threads = 4;
  std::string dump, dumplast, sync_model, save_opt;
  bool iobind = false;
  std::map<std::string, std::vector<int64_t>> shape_override;
  std::vector<std::string> keys, vals;
  for (int i = 5; i < argc; i++) {
    std::string a = argv[i];
    size_t eq = a.find('=');
    if (eq == std::string::npos) continue;
    std::string k = a.substr(0, eq), v = a.substr(eq + 1);
    if (k == "threads") threads = atoi(v.c_str());
    else if (k == "dump") dump = v;
    else if (k == "dumplast") dumplast = v;
    else if (k == "sync") sync_model = v;
    else if (k == "save_opt") save_opt = v;
    else if (k == "iobind") iobind = v == "1";
    else if (k == "shape") {
      size_t c = v.find(':');
      std::vector<int64_t> d;
      for (size_t p = c + 1; p < v.size();) {
        size_t e = v.find(',', p);
        d.push_back(atoll(v.substr(p, e == std::string::npos ? e : e - p).c_str()));
        if (e == std::string::npos) break;
        p = e + 1;
      }
      shape_override[v.substr(0, c)] = d;
    }
    else { keys.push_back(k); vals.push_back(v); }
  }
  dump_first = dump;
  OrtEnv* env;
  CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));
  OrtSessionOptions* so;
  CK(g->CreateSessionOptions(&so));
  if (getenv("OPT_LEVEL0")) CK(g->SetSessionGraphOptimizationLevel(so, ORT_DISABLE_ALL));  // attr: keep identical copies from being merged by CSE
  
  CK(g->SetIntraOpNumThreads(so, threads));
  if (getenv("PROFILE")) CK(g->EnableProfiling(so, getenv("PROFILE")));
  if (!save_opt.empty()) CK(g->SetOptimizedModelFilePath(so, save_opt.c_str()));
  if (prov == "webgpu") {
    std::vector<const char*> k, v;
    for (size_t i = 0; i < keys.size(); i++) { k.push_back(keys[i].c_str()); v.push_back(vals[i].c_str()); }
    CK(g->SessionOptionsAppendExecutionProvider(so, "WebGPU", k.data(), v.data(), k.size()));
  }
  OrtSession* s;
  auto t_load0 = std::chrono::steady_clock::now();
  CK(g->CreateSession(env, model.c_str(), so, &s));
  double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();
  OrtSession* sync_s = nullptr;
  if (!sync_model.empty()) {
    OrtSessionOptions* so2;
    CK(g->CreateSessionOptions(&so2));
    CK(g->SetIntraOpNumThreads(so2, threads));
    if (prov == "webgpu") {
      std::vector<const char*> k, v;
      std::vector<std::string> kk, vv;
      for (size_t i = 0; i < keys.size(); i++) {
        if (keys[i] == "enableGraphCapture") continue;  // the sync session runs normally
        kk.push_back(keys[i]); vv.push_back(vals[i]);
      }
      for (size_t i = 0; i < kk.size(); i++) { k.push_back(kk[i].c_str()); v.push_back(vv[i].c_str()); }
      CK(g->SessionOptionsAppendExecutionProvider(so2, "WebGPU", k.data(), v.data(), k.size()));
    }
    CK(g->CreateSession(env, sync_model.c_str(), so2, &sync_s));
  }
  OrtAllocator* al;
  CK(g->GetAllocatorWithDefaultOptions(&al));
  OrtMemoryInfo* mi;
  CK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));

  size_t nin, nout;
  CK(g->SessionGetInputCount(s, &nin));
  CK(g->SessionGetOutputCount(s, &nout));
  std::vector<std::string> in_names, out_names;
  std::vector<std::vector<char>> bufs(nin);
  std::vector<OrtValue*> in_vals(nin);
  for (size_t i = 0; i < nin; i++) {
    char* nm;
    CK(g->SessionGetInputName(s, i, al, &nm));
    in_names.push_back(nm);
    OrtTypeInfo* ti;
    CK(g->SessionGetInputTypeInfo(s, i, &ti));
    const OrtTensorTypeAndShapeInfo* tsi;
    CK(g->CastTypeInfoToTensorInfo(ti, &tsi));
    ONNXTensorElementDataType et;
    CK(g->GetTensorElementType(tsi, &et));
    size_t nd;
    CK(g->GetDimensionsCount(tsi, &nd));
    std::vector<int64_t> dims(nd);
    CK(g->GetDimensions(tsi, dims.data(), nd));
    if (shape_override.count(in_names.back())) { dims = shape_override[in_names.back()]; nd = dims.size(); }
    size_t n = 1;
    for (auto& d : dims) { if (d <= 0) d = 1; n *= d; }
    size_t es = elem_size(et);
    if (!es) { fprintf(stderr, "unsupported input type for %s\n", nm); return 2; }
    bufs[i].assign(n * es, 0);
    unsigned rng = 12345 + i;
    for (size_t j = 0; j < n; j++) {
      rng = rng * 1664525u + 1013904223u;
      float u = (rng >> 8) / 16777216.f;
      char* p = bufs[i].data() + j * es;
      switch (et) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: *(float*)p = u; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: *(uint16_t*)p = f2h(u); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: *(uint8_t*)p = (uint8_t)(u * 255); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: *(int64_t*)p = 1; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: *(int32_t*)p = 1; break;
        default: break;
      }
    }
    if (i == 0 && getenv("INPUT_BIN")) {  // real input (raw file, exactly the input tensor's bytes) instead of the LCG noise
      FILE* f = fopen(getenv("INPUT_BIN"), "rb");
      if (!f) { fprintf(stderr, "cannot open INPUT_BIN\n"); return 2; }
      size_t got = fread(bufs[i].data(), 1, bufs[i].size(), f);
      fclose(f);
      if (got != bufs[i].size()) { fprintf(stderr, "INPUT_BIN size mismatch %zu vs %zu\n", got, bufs[i].size()); return 2; }
    }
    CK(g->CreateTensorWithDataAsOrtValue(mi, bufs[i].data(), bufs[i].size(), dims.data(), nd, et, &in_vals[i]));
    g->ReleaseTypeInfo(ti);
  }
  for (size_t i = 0; i < nout; i++) {
    char* nm;
    CK(g->SessionGetOutputName(s, i, al, &nm));
    out_names.push_back(nm);
  }
  std::vector<const char*> inn, outn;
  for (auto& x : in_names) inn.push_back(x.c_str());
  for (auto& x : out_names) outn.push_back(x.c_str());

  OrtIoBinding* io = nullptr;
  OrtRunOptions* ropts = nullptr;
  if (iobind) {
    CK(g->CreateIoBinding(s, &io));
    for (size_t i = 0; i < nin; i++) CK(g->BindInput(io, inn[i], in_vals[i]));
    for (size_t o = 0; o < nout; o++) CK(g->BindOutputToDevice(io, outn[o], mi));
    CK(g->CreateRunOptions(&ropts));
    CK(g->AddRunConfigEntry(ropts, "gpu_graph_id", "0"));
  }
  auto run_sync = [&]() -> int {
    float x[4] = {1, 2, 3, 4};
    int64_t sh[2] = {1, 4};
    OrtValue* iv;
    if (OrtStatus* e = g->CreateTensorWithDataAsOrtValue(mi, x, sizeof x, sh, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &iv)) { g->ReleaseStatus(e); return 2; }
    const char* in1[] = {"X"};
    char* on;
    if (OrtStatus* e = g->SessionGetOutputName(sync_s, 0, al, &on)) { g->ReleaseStatus(e); return 2; }
    const char* out1[] = {on};
    OrtValue* ov = nullptr;
    if (OrtStatus* e = g->Run(sync_s, nullptr, in1, &iv, 1, out1, 1, &ov)) { fprintf(stderr, "sync run: %s\n", g->GetErrorMessage(e)); return 2; }
    g->ReleaseValue(ov);
    g->ReleaseValue(iv);
    return 0;
  };
  double sync_base = 0;
  if (sync_s) {
    std::vector<double> sm;
    for (int i = 0; i < 40; i++) {
      auto a = std::chrono::steady_clock::now();
      if (run_sync()) return 2;
      if (i >= 10) sm.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(sm.begin(), sm.end());
    sync_base = sm[sm.size() / 2];
  }
  std::vector<double> ms;
  double cold = 0;
  for (int it = 0; it < warm + iters; it++) {
    std::vector<OrtValue*> outs(nout, nullptr);
    auto t0 = std::chrono::steady_clock::now();
    if (iobind) {
      CK(g->RunWithBinding(s, ropts, io));
      OrtValue** bound = nullptr;
      size_t nb = 0;
      CK(g->GetBoundOutputValues(io, al, &bound, &nb));
      for (size_t o = 0; o < nb && o < nout; o++) outs[o] = bound[o];
      if (bound) al->Free(al, bound);
    } else {
      CK(g->Run(s, nullptr, inn.data(), in_vals.data(), nin, outn.data(), nout, outs.data()));
    }
    if (sync_s && run_sync()) return 2;
    double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() - (sync_s ? sync_base : 0.0);
    if (it == 0) cold = d;
    if (it >= warm) ms.push_back(d);
    if (const char* sl = getenv("SLEEP_MS")) usleep(static_cast<useconds_t>(atof(sl) * 1000));  // idle gap between inferences (GPU clock experiment)
    if ((it == 0 && !dump.empty()) || (it == warm + iters - 1 && !dumplast.empty())) {
      const std::string& dump = it == 0 && !::dump_first.empty() ? ::dump_first : dumplast;
      for (size_t o = 0; o < nout; o++) {
        OrtTensorTypeAndShapeInfo* oi;
        CK(g->GetTensorTypeAndShape(outs[o], &oi));
        size_t cnt;
        CK(g->GetTensorShapeElementCount(oi, &cnt));
        ONNXTensorElementDataType et;
        CK(g->GetTensorElementType(oi, &et));
        void* p;
        CK(g->GetTensorMutableData(outs[o], &p));
        std::string fn = dump + "/" + std::to_string(o) + ".bin";
        FILE* f = fopen(fn.c_str(), "wb");
        if (f) { fwrite(p, elem_size(et), cnt, f); fclose(f); }
        g->ReleaseTensorTypeAndShapeInfo(oi);
      }
    }
    for (auto* o : outs) if (o) g->ReleaseValue(o);
  }
  std::sort(ms.begin(), ms.end());
  double sum = 0;
  for (double v : ms) sum += v;
  size_t n = ms.size();
  if (sync_s) printf("(sync run alone: %.2f ms, subtracted)\n", sync_base);
  printf("RESULT %s %s load=%.0fms cold=%.1fms median=%.2fms mean=%.2fms min=%.2fms p90=%.2fms n=%zu\n", model.c_str(),
         prov.c_str(), load_ms, cold, n ? ms[n / 2] : 0.0, n ? sum / n : 0.0, n ? ms[0] : 0.0, n ? ms[(size_t)(n * 0.9)] : 0.0, n);
  if (getenv("PROFILE")) {
    char* pf;
    CK(g->SessionEndProfiling(s, al, &pf));
    printf("profile %s\n", pf);
  }
  return 0;
}
