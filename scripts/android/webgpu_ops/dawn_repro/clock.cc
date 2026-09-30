// Does the Adreno 730 run at a lower effective clock in network-like dispatch patterns than in back-to-back microbenchmarks?
//   clock [M N K]      (default 784 512 128; the GEMM is the register-tile 'sc' kernel, TM=8 NV=2, workgroup 32x4, scalar accumulators)
// Cases (GFLOPS of the GEMM work, best/median of several trials):
//   a  one long batched dispatch (z = 32 GEMMs)
//   b  100 dependent (WAW on C) GEMM dispatches in one pass, one submission
//   c  one submission per GEMM (or per 8 GEMMs) + CPU sleep of 0/0.1/0.5/2 ms between submissions (time = submit -> done)
//   d  GEMM interleaved with a cheap elementwise dispatch, 100 pairs in one pass
//   k  case c again while a second device on another thread keeps the GPU busy with a tiny dispatch stream (keep-alive)
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
using std::string; using std::to_string;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Ctx { wgpu::Instance inst; wgpu::Device dev; };
static Ctx make_ctx(wgpu::Instance* shared = nullptr) {
  Ctx c;
  if (shared) c.inst = *shared;
  else {
    wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
    c.inst = wgpu::CreateInstance(&id);
  }
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  static std::vector<const char*> dev_en = {"skip_validation", "disable_robustness"};
  wgpu::Adapter ad;
  c.inst.WaitAny(c.inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data();
  wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt;
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  c.inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) c.dev = d; }), UINT64_MAX);
  return c;
}
static void wait_done(Ctx& c) { c.inst.WaitAny(c.dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX); }
static wgpu::ComputePipeline pipe(wgpu::Device& dev, const string& src) {
  wgpu::ShaderSourceWGSL w{}; w.code = {src.data(), src.size()}; wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  return dev.CreateComputePipeline(&cpd);
}
static wgpu::Buffer mk(wgpu::Device& dev, size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = n; b.usage = u; return dev.CreateBuffer(&b); }

static string gemm_wgsl(int TM, int NV, int WX, int WY) {
  int TN = 4 * NV;
  string s = "struct U { M: u32, N: u32, K: u32, pad: u32 }\n@group(0) @binding(0) var<storage, read> A : array<vec4<f32>>;\n@group(0) @binding(1) var<storage, read> B : array<vec4<f32>>;\n"
             "@group(0) @binding(2) var<storage, read_write> C : array<vec4<f32>>;\n@group(0) @binding(3) var<uniform> u : U;\n";
  s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
       "  let col4 = g.x * " + to_string(NV) + "u;\n  let row0 = g.y * " + to_string(TM) + "u;\n  let K4 = u.K / 4u; let N4 = u.N / 4u;\n";
  for (int m = 0; m < TM; m++) for (int c = 0; c < TN; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
  s += "  for (var k4 = 0u; k4 < K4; k4++) {\n";
  for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++) s += "    let b" + to_string(n) + "_" + to_string(j) + " = B[(k4 * 4u + " + to_string(j) + "u) * N4 + col4 + " + to_string(n) + "u];\n";
  for (int m = 0; m < TM; m++) {
    s += "    let a" + to_string(m) + " = A[(row0 + " + to_string(m) + "u) * K4 + k4];\n";
    for (int j = 0; j < 4; j++) for (int n = 0; n < NV; n++) for (int q = 0; q < 4; q++) {
      string acc = "c" + to_string(m) + "_" + to_string(n * 4 + q);
      s += "    " + acc + " = fma(a" + to_string(m) + "." + "xyzw"[j] + ", b" + to_string(n) + "_" + to_string(j) + "." + "xyzw"[q] + ", " + acc + ");\n";
    }
  }
  s += "  }\n";
  for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++)
    s += "  if (row0 + " + to_string(m) + "u < u.M && col4 + " + to_string(n) + "u < N4) { C[wid.z * u.M * N4 + (row0 + " + to_string(m) + "u) * N4 + col4 + " + to_string(n) + "u] = vec4<f32>(c" + to_string(m) + "_" + to_string(n * 4) + ", c" + to_string(m) + "_" + to_string(n * 4 + 1) + ", c" + to_string(m) + "_" + to_string(n * 4 + 2) + ", c" + to_string(m) + "_" + to_string(n * 4 + 3) + "); }\n";
  return s + "}\n";
}
static const char* kEw = "@group(0) @binding(0) var<storage, read> a : array<vec4<f32>>;\n@group(0) @binding(1) var<storage, read_write> o : array<vec4<f32>>;\n"
                         "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) { if (g.x < 16384u) { o[g.x] = a[g.x] + vec4<f32>(1.0); } }\n";

static std::atomic<bool> g_stop{false};
static void keepalive_thread(wgpu::Instance inst, int mode) {
  Ctx c = make_ctx(&inst);
  wgpu::ComputePipeline pl = pipe(c.dev, kEw);
  wgpu::Buffer x = mk(c.dev, 65536 * 4, wgpu::BufferUsage::Storage), y = mk(c.dev, 65536 * 4, wgpu::BufferUsage::Storage);
  wgpu::BindGroupEntry e[2]; e[0].binding = 0; e[0].buffer = x; e[0].size = 65536 * 4; e[1].binding = 1; e[1].buffer = y; e[1].size = 65536 * 4;
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 2; bgd.entries = e; wgpu::BindGroup bg = c.dev.CreateBindGroup(&bgd);
  while (!g_stop.load()) {
    wgpu::CommandEncoder enc = c.dev.CreateCommandEncoder();
    { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0, bg); for (int i = 0; i < 4; i++) p.DispatchWorkgroups(256, 1, 1); p.End(); }
    wgpu::CommandBuffer cb = enc.Finish(); c.dev.GetQueue().Submit(1, &cb);
    wait_done(c);
    if (mode == 2) std::this_thread::sleep_for(std::chrono::microseconds(5000));  // light duty cycle (~5 ms period)
    if (mode == 3) std::this_thread::sleep_for(std::chrono::microseconds(1000));
  }
}

int main(int argc, char** argv) {
  dawnProcSetProcs(&dawn::native::GetProcs());
  uint32_t M = argc > 1 ? atoi(argv[1]) : 784, N = argc > 2 ? atoi(argv[2]) : 512, K = argc > 3 ? atoi(argv[3]) : 128;
  const int TM = 8, NV = 2, WX = 32, WY = 4, REPS = 32;
  Ctx c = make_ctx();
  wgpu::ComputePipeline pl = pipe(c.dev, gemm_wgsl(TM, NV, WX, WY)), pe = pipe(c.dev, kEw);
  auto St = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  wgpu::Buffer bA = mk(c.dev, (size_t)M * K * 4, St), bB = mk(c.dev, (size_t)K * N * 4, St), bC = mk(c.dev, (size_t)REPS * M * N * 4, St), bU = mk(c.dev, 16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  std::vector<float> a((size_t)M * K), b((size_t)K * N);
  for (size_t i = 0; i < a.size(); i++) a[i] = ((i * 7919) % 1000) / 1000.f - .5f;
  for (size_t i = 0; i < b.size(); i++) b[i] = ((i * 104729) % 1000) / 1000.f - .5f;
  uint32_t uni[4] = {M, N, K, 0};
  c.dev.GetQueue().WriteBuffer(bA, 0, a.data(), a.size() * 4); c.dev.GetQueue().WriteBuffer(bB, 0, b.data(), b.size() * 4); c.dev.GetQueue().WriteBuffer(bU, 0, uni, 16);
  wgpu::BindGroupEntry e[4]; e[0].binding = 0; e[0].buffer = bA; e[0].size = bA.GetSize(); e[1].binding = 1; e[1].buffer = bB; e[1].size = bB.GetSize();
  e[2].binding = 2; e[2].buffer = bC; e[2].size = bC.GetSize(); e[3].binding = 3; e[3].buffer = bU; e[3].size = 16;
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 4; bgd.entries = e; wgpu::BindGroup bg = c.dev.CreateBindGroup(&bgd);
  wgpu::Buffer ex = mk(c.dev, 65536 * 4, St), ey = mk(c.dev, 65536 * 4, St);
  wgpu::BindGroupEntry ee[2]; ee[0].binding = 0; ee[0].buffer = ex; ee[0].size = 65536 * 4; ee[1].binding = 1; ee[1].buffer = ey; ee[1].size = 65536 * 4;
  wgpu::BindGroupDescriptor bge{}; bge.layout = pe.GetBindGroupLayout(0); bge.entryCount = 2; bge.entries = ee; wgpu::BindGroup bgE = c.dev.CreateBindGroup(&bge);
  uint32_t gx = (N / 4 + NV * WX - 1) / (NV * WX), gy = (M + TM * WY - 1) / (TM * WY);
  const double flops1 = 2.0 * M * N * K;
  auto gflops = [&](double gemms, double ms) { return flops1 * gemms / ms / 1e6; };
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  auto best = [](std::vector<double> v) { return *std::min_element(v.begin(), v.end()); };
  auto submit_gemms = [&](int n, int z) {  // n dispatches of z GEMMs each, one pass, one submission
    wgpu::CommandEncoder enc = c.dev.CreateCommandEncoder();
    { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0, bg); for (int i = 0; i < n; i++) p.DispatchWorkgroups(gx, gy, z); p.End(); }
    wgpu::CommandBuffer cb = enc.Finish(); c.dev.GetQueue().Submit(1, &cb);
  };
  auto warm = [&]() { for (int i = 0; i < 40; i++) { submit_gemms(1, REPS); wait_done(c); } };
  printf("GEMM %ux%ux%u (%.3f GFLOP), sc TM=%d NV=%d wg=%dx%d\n", M, N, K, flops1 / 1e9, TM, NV, WX, WY);
  warm();
  { std::vector<double> t; for (int i = 0; i < 15; i++) { auto t0 = Clock::now(); submit_gemms(1, REPS); wait_done(c); t.push_back(ms_since(t0)); }
    printf("a  one batched dispatch (z=%d):                      %6.0f GFLOPS (best %6.0f)\n", REPS, gflops(REPS, med(t)), gflops(REPS, best(t))); }
  { std::vector<double> t; for (int i = 0; i < 15; i++) { auto t0 = Clock::now(); submit_gemms(100, 1); wait_done(c); t.push_back(ms_since(t0)); }
    printf("b  100 dependent dispatches, one submission:        %6.0f GFLOPS (best %6.0f)\n", gflops(100, med(t)), gflops(100, best(t))); }
  { std::vector<double> t; for (int i = 0; i < 15; i++) {
      auto t0 = Clock::now(); wgpu::CommandEncoder enc = c.dev.CreateCommandEncoder(); wgpu::ComputePassEncoder p = enc.BeginComputePass();
      for (int j = 0; j < 100; j++) { p.SetPipeline(pl); p.SetBindGroup(0, bg); p.DispatchWorkgroups(gx, gy, 1); p.SetPipeline(pe); p.SetBindGroup(0, bgE); p.DispatchWorkgroups(256, 1, 1); }
      p.End(); wgpu::CommandBuffer cb = enc.Finish(); c.dev.GetQueue().Submit(1, &cb); wait_done(c); t.push_back(ms_since(t0)); }
    std::vector<double> te; for (int i = 0; i < 15; i++) {
      auto t0 = Clock::now(); wgpu::CommandEncoder enc = c.dev.CreateCommandEncoder(); wgpu::ComputePassEncoder p = enc.BeginComputePass();
      p.SetPipeline(pe); p.SetBindGroup(0, bgE); for (int j = 0; j < 100; j++) p.DispatchWorkgroups(256, 1, 1);
      p.End(); wgpu::CommandBuffer cb = enc.Finish(); c.dev.GetQueue().Submit(1, &cb); wait_done(c); te.push_back(ms_since(t0)); }
    printf("d  GEMM + elementwise interleaved x100:              %6.0f GFLOPS counting all time (%.2f ms total; elementwise alone %.2f ms; GEMM share %6.0f GFLOPS)\n", gflops(100, med(t)), med(t), med(te), gflops(100, med(t) - med(te))); }
  double gaps[] = {0, 0.1, 0.5, 2.0};
  for (int per : {1, 8}) {
    for (double g : gaps) {
      std::vector<double> t;
      for (int i = 0; i < 60; i++) {
        auto t0 = Clock::now(); submit_gemms(per, 1); wait_done(c); double d = ms_since(t0);
        if (i >= 10) t.push_back(d);
        if (g > 0) std::this_thread::sleep_for(std::chrono::microseconds((int)(g * 1000)));
      }
      printf("c  %d GEMM(s)/submission, gap %.1f ms:                %6.0f GFLOPS (median submit->done %.3f ms, best %.3f)\n", per, g, gflops(per, med(t)), med(t), best(t));
    }
  }
  // burst-length sweep: waiting after every submission (w) vs keeping the queue full (p: submit 400 GEMMs' worth of submissions, wait once)
  for (int per : {1, 2, 4, 8, 16, 32, 100}) {
    std::vector<double> tw, tp;
    int subs = 400 / per;
    for (int r = 0; r < 5; r++) {
      auto t0 = Clock::now(); for (int i = 0; i < subs; i++) { submit_gemms(per, 1); wait_done(c); } tw.push_back(ms_since(t0));
      t0 = Clock::now(); for (int i = 0; i < subs; i++) submit_gemms(per, 1); wait_done(c); tp.push_back(ms_since(t0));
    }
    printf("s  %3d GEMMs/submission x %3d:  wait each %6.0f GFLOPS (%.3f ms/GEMM) | submit ahead %6.0f GFLOPS (%.3f ms/GEMM)\n", per, subs, gflops(400, best(tw)), best(tw) / 400, gflops(400, best(tp)), best(tp) / 400);
  }
  // idle-then-burst: sleep S ms, then time one 100-GEMM submission (does the clock ramp down while idle and how long does the ramp-up take?)
  for (double idle : {0.0, 5.0, 20.0, 100.0, 500.0}) {
    std::vector<double> t;
    for (int r = 0; r < 8; r++) { if (idle > 0) std::this_thread::sleep_for(std::chrono::microseconds((int)(idle * 1000))); auto t0 = Clock::now(); submit_gemms(100, 1); wait_done(c); t.push_back(ms_since(t0)); }
    printf("i  idle %5.0f ms then 100 GEMMs: %6.0f GFLOPS median (%.1f ms), first-run-after-idle best %6.0f\n", idle, gflops(100, med(t)), med(t), gflops(100, best(t)));
  }
  // ramp-up trace: after 1 s idle, run 40 consecutive 20-GEMM submissions (wait each) and print the time of each (ms) -> how long until full clock
  for (int r = 0; r < 2; r++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    printf("r  ramp after 1 s idle (ms per 20 GEMMs, cumulative ms at each):");
    double cum = 0;
    for (int i = 0; i < 40; i++) { auto t0 = Clock::now(); submit_gemms(20, 1); wait_done(c); double d = ms_since(t0); cum += d; if (i < 6 || i % 5 == 4) printf(" [%d] %.1f(@%.0f)", i, d, cum); }
    printf("\n");
  }
  // long ramp trace: 1 s idle, then 8 s of continuous 20-GEMM submissions; print per-GEMM time every ~0.5 s
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    printf("R  long ramp after 1 s idle (t_s: ms/GEMM):");
    auto T0 = Clock::now(); double bucket = 0; int n = 0; double next = 0.5;
    while (ms_since(T0) < 8000) { auto t0 = Clock::now(); submit_gemms(20, 1); wait_done(c); bucket += ms_since(t0); n += 20; if (ms_since(T0) / 1000 >= next) { printf(" %.1f:%.2f", next, bucket / n); bucket = 0; n = 0; next += 0.5; } }
    printf("\n");
  }
  // K: inference-like pattern (20 GEMMs, wait, sleep 20 ms) for 6 s after 1 s idle, without and with a keep-alive stream on a second device
  for (int ka : {0, 1, 2, 3}) {
    g_stop = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    std::thread th;
    if (ka) th = std::thread(keepalive_thread, c.inst, ka);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    printf("K  keep-alive %-22s (t_s: ms/GEMM):", ka == 0 ? "none" : ka == 1 ? "continuous" : ka == 2 ? "tiny dispatch / 5 ms" : "tiny dispatch / 1 ms");
    auto T0 = Clock::now(); double bucket = 0; int n = 0; double next = 1.0;
    while (ms_since(T0) < 6000) { auto t0 = Clock::now(); submit_gemms(20, 1); wait_done(c); bucket += ms_since(t0); n += 20; std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (ms_since(T0) / 1000 >= next) { printf(" %.0f:%.2f", next, bucket / n); bucket = 0; n = 0; next += 1.0; } }
    printf("\n");
    if (ka) { g_stop = true; th.join(); }
  }
  return 0;
}
