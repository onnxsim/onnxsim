// int8 through Dawn/Vulkan on the Adreno 730: feature check, dot4 peak, and an int8 register-tile GEMM with requantize epilogue.
//   int8 info
//   int8 peak  dot|unp|emu|f32  [WORKGROUPS ITERS]     (dot = dot4I8Packed, unp = dot(unpack4xI8,unpack4xI8), emu = extractBits multiply-add)
//   int8 gemm  dot|unp|emu  M N K REPS [TM NV WX WY]   (K % 16 == 0, N % 4 == 0)
// Every mode warms the GPU for WARM seconds first (env WARM, default 5), because the Adreno clock ramps slowly after idle.
// GEMM operands: A = M x K int8 packed 4 per u32 (k contiguous), B = K/4 x N u32 (4 consecutive k of one column per u32), i32 accumulators,
// epilogue: clamp(round(acc * scale[n]), -128, 127) packed 4 columns per u32. Checked against an exact CPU integer reference.
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using std::string; using std::to_string;

static wgpu::Instance inst;
static wgpu::Adapter ad;
static wgpu::Device dev;

static void setup() {
  dawnProcSetProcs(&dawn::native::GetProcs());
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  std::vector<const char*> ad_en, dev_en;
  bool rb_off = getenv("RB") && string(getenv("RB")) == "off", vmm = getenv("VMM") && string(getenv("VMM")) == "1";
  if (vmm) ad_en.push_back("use_vulkan_memory_model");
  if (rb_off) dev_en.push_back("disable_robustness");
  wgpu::DawnTogglesDescriptor adt{}; adt.enabledToggleCount = ad_en.size(); adt.enabledToggles = ad_en.data(); ro.nextInChain = &adt;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data();
  wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt;
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
}

static wgpu::Buffer mk(size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = n; b.usage = u; return dev.CreateBuffer(&b); }
static const auto ST = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;

// compile; returns false and prints the Tint error if the shader is rejected
static bool compile(const string& code, wgpu::ComputePipeline& pl) {
  bool ok = true;
  wgpu::ShaderSourceWGSL w{}; w.code = {code.data(), code.size()};
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  inst.WaitAny(sm.GetCompilationInfo(wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::CompilationInfoRequestStatus, const wgpu::CompilationInfo* ci) {
    for (size_t i = 0; i < ci->messageCount; i++) if (ci->messages[i].type == wgpu::CompilationMessageType::Error) { ok = false; fprintf(stderr, "WGSL error: %.*s\n", (int)ci->messages[i].message.length, ci->messages[i].message.data); }
  }), UINT64_MAX);
  if (!ok) return false;
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  pl = dev.CreateComputePipeline(&cpd);
  return true;
}

// time `run` (one submit+wait); warm for WARM seconds first, then return min of n runs in ms
template <class F> static double timeit(F run, int n = 15) {
  double warm = getenv("WARM") ? atof(getenv("WARM")) : 5.0;
  auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm) run();
  std::vector<double> t; for (int i = 0; i < n; i++) t.push_back(run());
  std::sort(t.begin(), t.end());
  return t[0];
}
static double submit(wgpu::ComputePipeline& pl, wgpu::BindGroup& bg, uint32_t gx, uint32_t gy, uint32_t gz) {
  wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
  { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0, bg); p.DispatchWorkgroups(gx, gy, gz); p.End(); }
  wgpu::CommandBuffer cb = enc.Finish();
  auto t0 = std::chrono::steady_clock::now();
  dev.GetQueue().Submit(1, &cb);
  inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static const char* DOT4E =
    "fn dot4e(a: u32, b: u32) -> i32 {\n"
    "  return extractBits(i32(a), 0u, 8u) * extractBits(i32(b), 0u, 8u) + extractBits(i32(a), 8u, 8u) * extractBits(i32(b), 8u, 8u)\n"
    "       + extractBits(i32(a), 16u, 8u) * extractBits(i32(b), 16u, 8u) + extractBits(i32(a), 24u, 8u) * extractBits(i32(b), 24u, 8u);\n}\n";
static string dotExpr(const string& v, const string& a, const string& b) {
  if (v == "dot") return "dot4I8Packed(" + a + ", " + b + ")";
  if (v == "unp") return "dot(unpack4xI8(" + a + "), unpack4xI8(" + b + "))";
  return "dot4e(" + a + ", " + b + ")";
}
static string header(const string& v) { return string(v == "dot" || v == "unp" || v == "f8" ? "requires packed_4x8_integer_dot_product;\n" : "") + (v == "emu" ? DOT4E : ""); }

static int info() {
  wgpu::AdapterInfo ai{}; ad.GetInfo(&ai);
  printf("adapter: vendor=%.*s arch=%.*s device=%.*s desc=%.*s\n", (int)ai.vendor.length, ai.vendor.data, (int)ai.architecture.length, ai.architecture.data, (int)ai.device.length, ai.device.data, (int)ai.description.length, ai.description.data);
  wgpu::SupportedFeatures sf; ad.GetFeatures(&sf);
  printf("adapter features:"); for (size_t i = 0; i < sf.featureCount; i++) printf(" %d", (int)sf.features[i]); printf("\n");
  printf("Subgroups=%d ShaderF16=%d\n", (int)ad.HasFeature(wgpu::FeatureName::Subgroups), (int)ad.HasFeature(wgpu::FeatureName::ShaderF16));
  printf("wgsl language feature packed_4x8_integer_dot_product: %d\n", (int)inst.HasWGSLLanguageFeature(wgpu::WGSLLanguageFeatureName::Packed4x8IntegerDotProduct));
  printf("wgsl language feature readonly_and_readwrite_storage_textures: %d\n", (int)inst.HasWGSLLanguageFeature(wgpu::WGSLLanguageFeatureName::ReadonlyAndReadwriteStorageTextures));
  wgpu::Limits lim{}; ad.GetLimits(&lim);
  printf("limits: maxComputeInvocationsPerWorkgroup=%u maxComputeWorkgroupStorageSize=%u maxStorageBufferBindingSize=%llu\n", lim.maxComputeInvocationsPerWorkgroup, lim.maxComputeWorkgroupStorageSize, (unsigned long long)lim.maxStorageBufferBindingSize);
  for (string v : {"dot", "unp", "emu"}) {
    string code = header(v) + "@group(0) @binding(0) var<storage, read_write> o : array<i32>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3<u32>) { o[g.x] = " + dotExpr(v, "g.x * 16843009u", "0x01ff7f80u") + "; }\n";
    wgpu::ComputePipeline pl; bool ok = compile(code, pl);
    printf("shader with %s: %s\n", v.c_str(), ok ? "compiles" : "REJECTED");
  }
  return 0;
}

static int peak(const string& v, uint32_t WG, uint32_t ITERS) {
  const bool f32 = v == "f32";
  const int CH = 16;
  string s = header(f32 ? "emu_none" : v);
  s += "@group(0) @binding(0) var<uniform> K : array<vec4<u32>, 4>;\n@group(0) @binding(1) var<storage, read_write> O : array<" + string(f32 ? "f32" : "i32") + ">;\n";
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(num_workgroups) nw : vec3<u32>) {\n";
  if (f32) s += "  var x = f32(g.x) * 0.001 + 0.5;\n"; else s += "  var x = g.x * 2654435761u + 12345u;\n";
  for (int c = 0; c < CH; c++) s += string("  var a") + to_string(c) + (f32 ? " = 0.0;\n" : " = 0;\n");
  s += "  for (var i = 0u; i < " + to_string(ITERS) + "u; i++) {\n";
  for (int c = 0; c < CH; c++) {
    string k = "K[" + to_string(c / 4) + "]." + "xyzw"[c % 4];
    if (f32) s += "    a" + to_string(c) + " = fma(x, bitcast<f32>((" + k + " & 0x3fffffffu) | 0x3f000000u), a" + to_string(c) + ");\n";
    else s += "    a" + to_string(c) + " += " + dotExpr(v, "x", k) + ";\n";
  }
  if (f32) s += "    x = x * 1.0000001 + 0.0001;\n"; else s += "    x = x * 1664525u + 1013904223u;\n";
  s += "  }\n  var t = " + string(f32 ? "0.0" : "0") + ";\n";
  for (int c = 0; c < CH; c++) s += "  t += a" + to_string(c) + ";\n";
  s += "  O[g.x + g.y * nw.x * 64u] = t;\n}\n";
  wgpu::ComputePipeline pl; if (!compile(s, pl)) return 1;
  wgpu::Buffer bk = mk(64, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst), bo = mk((size_t)WG * 64 * 4, ST);
  uint32_t kk[16]; for (int i = 0; i < 16; i++) kk[i] = 0x01020304u * (i + 1) ^ 0x7f80ff01u;
  dev.GetQueue().WriteBuffer(bk, 0, kk, 64);
  wgpu::BindGroupEntry e[2]; e[0].binding = 0; e[0].buffer = bk; e[0].size = 64; e[1].binding = 1; e[1].buffer = bo; e[1].size = bo.GetSize();
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 2; bgd.entries = e; wgpu::BindGroup bg = dev.CreateBindGroup(&bgd);
  double ms = timeit([&] { return submit(pl, bg, WG, 1, 1); });
  double n = (double)WG * 64 * ITERS * CH;  // dot4 (or fma) count
  if (f32) printf("peak f32  fma: %.3f ms  %.0f GFLOPS (2 flops/fma)\n", ms, 2 * n / ms / 1e6);
  else printf("peak %-4s dot4: %.3f ms  %.1f G dot4/s = %.0f GOPS int8 (2 ops/MAC, 4 MAC/dot4; each dot also does one i32 add)\n", v.c_str(), ms, n / ms / 1e6, 8 * n / ms / 1e6);
  return 0;
}

static int gemm(const string& v, uint32_t M, uint32_t N, uint32_t K, uint32_t REPS, int TM, int NV, int WX, int WY) {
  if (K % 16 || N % 4) { fprintf(stderr, "K %% 16 and N %% 4 required\n"); return 1; }
  const int TN = 4 * NV;
  string s = header(v);
  s += "struct U { M: u32, N: u32, K: u32, pad: u32 }\n"
       "@group(0) @binding(0) var<storage, read> A : array<vec4<u32>>;\n"
       "@group(0) @binding(1) var<storage, read> B : array<vec4<u32>>;\n"
       "@group(0) @binding(2) var<storage, read_write> C : array<u32>;\n"
       "@group(0) @binding(3) var<uniform> u : U;\n"
       "@group(0) @binding(4) var<storage, read> S : array<vec4<f32>>;\n"
       "fn q(acc: i32, s: f32) -> u32 { return u32(clamp(i32(round(f32(acc) * s)), -128, 127)) & 0xffu; }\n";
  s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let col4 = g.x * " + to_string(NV) + "u;\n  let row0 = g.y * " + to_string(TM) + "u;\n  let K16 = u.K / 16u; let N4 = u.N / 4u;\n";
  for (int m = 0; m < TM; m++) s += "  let ar" + to_string(m) + " = min(row0 + " + to_string(m) + "u, u.M - 1u);\n";
  for (int n = 0; n < NV; n++) s += "  let bc" + to_string(n) + " = min(col4 + " + to_string(n) + "u, N4 - 1u);\n";
  const bool F8 = v == "f8";
  for (int m = 0; m < TM; m++) for (int c = 0; c < TN; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + (F8 ? " = 0.0;\n" : " = 0;\n");
  s += "  for (var k16 = 0u; k16 < K16; k16++) {\n";
  for (int m = 0; m < TM; m++) s += "    let a" + to_string(m) + " = A[ar" + to_string(m) + " * K16 + k16];\n";
  for (int kk = 0; kk < 4; kk++) {
    for (int n = 0; n < NV; n++) s += "    let b" + to_string(kk) + "_" + to_string(n) + " = B[(k16 * 4u + " + to_string(kk) + "u) * N4 + bc" + to_string(n) + "];\n";
    if (F8) {  // int8 storage, f32 arithmetic: unpack + convert once per operand, dot() in f32
      for (int m = 0; m < TM; m++) s += "    let fa" + to_string(kk) + "_" + to_string(m) + " = vec4<f32>(unpack4xI8(a" + to_string(m) + "." + "xyzw"[kk] + "));\n";
      for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++) s += "    let fb" + to_string(kk) + "_" + to_string(n * 4 + j) + " = vec4<f32>(unpack4xI8(b" + to_string(kk) + "_" + to_string(n) + "." + "xyzw"[j] + "));\n";
      for (int m = 0; m < TM; m++) for (int c = 0; c < TN; c++)
        s += "    c" + to_string(m) + "_" + to_string(c) + " += dot(fa" + to_string(kk) + "_" + to_string(m) + ", fb" + to_string(kk) + "_" + to_string(c) + ");\n";
    } else {
      for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
        s += "    c" + to_string(m) + "_" + to_string(n * 4 + j) + " += " + dotExpr(v, "a" + to_string(m) + "." + "xyzw"[kk], "b" + to_string(kk) + "_" + to_string(n) + "." + "xyzw"[j]) + ";\n";
    }
  }
  s += "  }\n";
  for (int n = 0; n < NV; n++) s += "  let sc" + to_string(n) + " = S[bc" + to_string(n) + "];\n";
  for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) {
    string p = "c" + to_string(m) + "_";
    auto Q = [&](int j, char comp) { return "q(" + string(F8 ? "i32(" : "(") + p + to_string(n * 4 + j) + "), sc" + to_string(n) + "." + comp + ")"; };
    s += "  if (row0 + " + to_string(m) + "u < u.M && col4 + " + to_string(n) + "u < N4) { C[(g.z * u.M + row0 + " + to_string(m) + "u) * N4 + col4 + " + to_string(n) + "u] = " + Q(0, 'x') + " | (" + Q(1, 'y') + " << 8u) | (" + Q(2, 'z') + " << 16u) | (" + Q(3, 'w') + " << 24u); }\n";
  }
  s += "}\n";
  wgpu::ComputePipeline pl; if (!compile(s, pl)) return 1;
  // data
  std::vector<int8_t> a((size_t)M * K), b((size_t)K * N);
  for (size_t i = 0; i < a.size(); i++) a[i] = (int8_t)(((i * 7919 + 13) % 255) - 127);
  for (size_t i = 0; i < b.size(); i++) b[i] = (int8_t)(((i * 104729 + 7) % 255) - 127);
  std::vector<float> sc(N); for (uint32_t n = 0; n < N; n++) sc[n] = 40.0f / (5373.0f * std::sqrt((float)K)) * (1.0f + 0.25f * (n % 5));
  std::vector<uint32_t> ap((size_t)M * K / 4), bp((size_t)K / 4 * N);
  for (uint32_t m = 0; m < M; m++) for (uint32_t k4 = 0; k4 < K / 4; k4++) { uint32_t w = 0; for (int q = 0; q < 4; q++) w |= (uint32_t)(uint8_t)a[(size_t)m * K + 4 * k4 + q] << (8 * q); ap[(size_t)m * (K / 4) + k4] = w; }
  for (uint32_t k4 = 0; k4 < K / 4; k4++) for (uint32_t n = 0; n < N; n++) { uint32_t w = 0; for (int q = 0; q < 4; q++) w |= (uint32_t)(uint8_t)b[(size_t)(4 * k4 + q) * N + n] << (8 * q); bp[(size_t)k4 * N + n] = w; }
  wgpu::Buffer bA = mk(ap.size() * 4, ST), bB = mk(bp.size() * 4, ST), bC = mk((size_t)REPS * M * N, ST), bS = mk(sc.size() * 4, ST), bU = mk(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  uint32_t uni[4] = {M, N, K, 0};
  dev.GetQueue().WriteBuffer(bA, 0, ap.data(), ap.size() * 4); dev.GetQueue().WriteBuffer(bB, 0, bp.data(), bp.size() * 4);
  dev.GetQueue().WriteBuffer(bS, 0, sc.data(), sc.size() * 4); dev.GetQueue().WriteBuffer(bU, 0, uni, 16);
  wgpu::BindGroupEntry e[5];
  e[0].binding = 0; e[0].buffer = bA; e[0].size = bA.GetSize(); e[1].binding = 1; e[1].buffer = bB; e[1].size = bB.GetSize();
  e[2].binding = 2; e[2].buffer = bC; e[2].size = bC.GetSize(); e[3].binding = 3; e[3].buffer = bU; e[3].size = 16; e[4].binding = 4; e[4].buffer = bS; e[4].size = bS.GetSize();
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 5; bgd.entries = e; wgpu::BindGroup bg = dev.CreateBindGroup(&bgd);
  uint32_t gx = ((N / 4 + NV - 1) / NV + WX - 1) / WX, gy = (M + TM * WY - 1) / (TM * WY);
  double ms = timeit([&] { return submit(pl, bg, gx, gy, REPS); });
  // correctness on slice 0 and the last slice
  wgpu::Buffer rb = mk((size_t)REPS * M * N, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  { wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); enc.CopyBufferToBuffer(bC, 0, rb, 0, (size_t)REPS * M * N); wgpu::CommandBuffer cb = enc.Finish(); dev.GetQueue().Submit(1, &cb); }
  bool ok = false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read, 0, (size_t)REPS * M * N, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::MapAsyncStatus st, wgpu::StringView) { ok = st == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
  int bad = 0, off1 = 0, checked = 0;
  if (ok) {
    const uint8_t* c = (const uint8_t*)rb.GetConstMappedRange(0, (size_t)REPS * M * N);
    for (uint32_t z : {0u, REPS - 1}) for (int q = 0; q < 300; q++) {
      size_t r = (q * 2654435761u) % M, cc = (q * 40503u + 7) % N; long long acc = 0;
      for (uint32_t k = 0; k < K; k++) acc += (long long)a[r * K + k] * b[(size_t)k * N + cc];
      float f = (float)(int)acc * sc[cc]; int want = (int)std::nearbyintf(f); want = std::max(-128, std::min(127, want));
      int got = (int8_t)c[((size_t)z * M + r) * N + cc]; checked++;
      if (got != want) { if (std::abs(got - want) == 1) off1++; else bad++; }
    }
  }
  double ops = 2.0 * M * N * K * REPS;
  printf("%-3s M=%u N=%u K=%u reps=%u TM=%d NV=%d wg=%dx%d  best=%.3f ms  %.0f GOPS (%.3f ms per gemm)  check: %d wrong, %d off-by-one of %d\n",
         v.c_str(), M, N, K, REPS, TM, NV, WX, WY, ms, ops / ms / 1e6, ms / REPS, bad, off1, checked);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: int8 info | peak V [WG ITERS] | gemm V M N K REPS [TM NV WX WY]\n"); return 1; }
  setup();
  string mode = argv[1];
  if (mode == "info") return info();
  if (mode == "peak") return peak(argv[2], argc > 3 ? atoi(argv[3]) : 4096, argc > 4 ? atoi(argv[4]) : 2048);
  if (mode == "gemm") return gemm(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argc > 7 ? atoi(argv[7]) : 8, argc > 8 ? atoi(argv[8]) : 2, argc > 9 ? atoi(argv[9]) : 16, argc > 10 ? atoi(argv[10]) : 4);
  return 1;
}
