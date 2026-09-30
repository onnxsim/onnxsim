// 3x3 STRIDE-2 pad-1 NHWC fp32 conv (ResNet-50 v1.5 downsampling), Cin=Cout=C, HxH -> (H/2)x(H/2), batch 1, through Dawn/Vulkan.
//   conv_s2 C H
// direct   : implicit GEMM (M=Ho*Wo, N=C, K=9C), the conv_alt/gemm.cc "sc" register-tile design, swept over (TM,NV,WX,WY).
// poly     : polyphase hybrid Winograd. With xe/xo the even/odd input phases, o[y] = w1 xe[y] + w0 xo[y-1] + w2 xo[y]; the
//            2-tap part uses F(2,2) (3 mults for 2 outputs), the 1-tap part is direct (2 mults), so 5 mults per 2 outputs per
//            dimension, 25 per 2x2 output tile instead of 36 (0.69x). Per tile the 25 transformed inputs are
//            E0=x[4t] E1=x[4t+2] A=x[4t-1]-x[4t+1] B=x[4t+1] C=x[4t+1]-x[4t+3] (per dimension), the weights are
//            (w1, w1, w0, w0+w2, w2) and o0 = E0'+A'+B', o1 = E1'+B'-C'. input transform -> 25 batched GEMMs -> output transform.
// Correctness is checked on a sample against a double-precision CPU reference. Build like gemm.cc (see README.md).
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
using std::string; using std::to_string; using std::vector;

static wgpu::Instance inst; static wgpu::Device dev;

struct Kern { wgpu::ComputePipeline pl; wgpu::BindGroup bg; uint32_t gx, gy, gz; bool ok = true; };

static Kern make(const string& wgsl, const vector<wgpu::Buffer>& bufs, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
  Kern k; k.gx = gx; k.gy = gy; k.gz = gz;
  wgpu::ShaderSourceWGSL w{}; w.code = {wgsl.data(), wgsl.size()};
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  k.pl = dev.CreateComputePipeline(&cpd);
  vector<wgpu::BindGroupEntry> e(bufs.size());
  for (size_t i = 0; i < bufs.size(); i++) { e[i].binding = i; e[i].buffer = bufs[i]; e[i].size = bufs[i].GetSize(); }
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = k.pl.GetBindGroupLayout(0); bgd.entryCount = e.size(); bgd.entries = e.data();
  k.bg = dev.CreateBindGroup(&bgd);
  return k;
}

// `iters` back-to-back repetitions of the whole sequence in one submission; returns best-of-10 ms per repetition
static double timeSeq(const vector<Kern>& ks, int iters) {
  auto run = [&]() {
    wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
    for (int i = 0; i < iters; i++) for (auto& k : ks) {
      wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(k.pl); p.SetBindGroup(0, k.bg); p.DispatchWorkgroups(k.gx, k.gy, k.gz); p.End();
    }
    wgpu::CommandBuffer cb = enc.Finish();
    auto t0 = std::chrono::steady_clock::now();
    dev.GetQueue().Submit(1, &cb);
    inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
  };
  run(); run(); run();
  double best = 1e30; for (int i = 0; i < 10; i++) best = std::min(best, run());
  return best;
}

static wgpu::Buffer mk(size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = (n + 15) & ~size_t(15); b.usage = u; return dev.CreateBuffer(&b); }
static const auto ST = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
static wgpu::Buffer upload(const vector<float>& v) { auto b = mk(v.size() * 4, ST); dev.GetQueue().WriteBuffer(b, 0, v.data(), v.size() * 4); return b; }
static vector<float> download(wgpu::Buffer src, size_t n) {
  wgpu::Buffer rb = mk(n * 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  { wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); enc.CopyBufferToBuffer(src, 0, rb, 0, (n * 4 + 3) & ~size_t(3)); wgpu::CommandBuffer cb = enc.Finish(); dev.GetQueue().Submit(1, &cb); }
  bool ok = false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read, 0, rb.GetSize(), wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::MapAsyncStatus st, wgpu::StringView) { ok = st == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
  vector<float> out(n); if (ok) memcpy(out.data(), rb.GetConstMappedRange(0, rb.GetSize()), n * 4);
  return out;
}

static string hdr(int nin, int nout_rw = 1) {  // bindings 0..nin-1 read-only vec4 storage, then nout_rw read_write
  string s;
  for (int i = 0; i < nin; i++) s += "@group(0) @binding(" + to_string(i) + ") var<storage, read> in" + to_string(i) + " : array<vec4<f32>>;\n";
  s += "@group(0) @binding(" + to_string(nin) + ") var<storage, read_write> outb : array<vec4<f32>>;\n";
  return s;
}
static string U(uint32_t v) { return to_string(v) + "u"; }
static string comp(const string& x, int i) { return x + "." + "xyzw"[i]; }

// register-tile GEMM: C[z][M,N] = A[z][M,K] * B[z][K,N]  (plain), or the 3x3 implicit-GEMM gather (conv) where A is the NHWC image.
static string gemm_wgsl(bool conv, uint32_t M, uint32_t N, uint32_t K, uint32_t H, uint32_t W, uint32_t Cin, int TM, int NV, int WX, int WY) {
  uint32_t N4 = N / 4, K4 = K / 4, CIN4 = Cin / 4;
  string s = hdr(2);  // in0 = A (or image), in1 = B, outb = C
  s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
       "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
       "  let col4 = g.x * " + U(NV) + "; let row0 = g.y * " + U(TM) + "; let z = wid.z;\n";
  for (int n = 0; n < NV; n++) s += "  let cb" + to_string(n) + " = min(col4 + " + U(n) + ", " + U(N4 - 1) + ");\n";
  for (int m = 0; m < TM; m++) for (int c = 0; c < 4 * NV; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
  auto fmas = [&](string ind) {
    string r;
    for (int j = 0; j < 4; j++) for (int n = 0; n < NV; n++) for (int m = 0; m < TM; m++) for (int q = 0; q < 4; q++)
      r += ind + "c" + to_string(m) + "_" + to_string(n * 4 + q) + " = fma(" + comp("a" + to_string(m), j) + ", " + comp("b" + to_string(n) + "_" + to_string(j), q) + ", c" + to_string(m) + "_" + to_string(n * 4 + q) + ");\n";
    return r;
  };
  if (!conv) {
    s += "  for (var k4 = 0u; k4 < " + U(K4) + "; k4++) {\n";
    for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
      s += "    let b" + to_string(n) + "_" + to_string(j) + " = in1[z * " + U(K * N4) + " + (k4 * 4u + " + U(j) + ") * " + U(N4) + " + cb" + to_string(n) + "];\n";
    for (int m = 0; m < TM; m++)
      s += "    let a" + to_string(m) + " = in0[z * " + U(M * K4) + " + min(row0 + " + U(m) + ", " + U(M - 1) + ") * " + U(K4) + " + k4];\n";
    s += fmas("    ") + "  }\n";
  } else {
    for (int m = 0; m < TM; m++)
      s += "  let p" + to_string(m) + " = row0 + " + U(m) + "; let y" + to_string(m) + " = i32(p" + to_string(m) + " / " + U(W) + "); let x" + to_string(m) + " = i32(p" + to_string(m) + " % " + U(W) + ");\n";
    s += "  for (var tap = 0u; tap < 9u; tap++) {\n    let dy = i32(tap / 3u) - 1; let dx = i32(tap % 3u) - 1;\n";
    for (int m = 0; m < TM; m++) {
      string i = to_string(m);
      s += "    let iy" + i + " = y" + i + " + dy; let ix" + i + " = x" + i + " + dx;\n"
           "    let v" + i + " = p" + i + " < " + U(M) + " && iy" + i + " >= 0 && iy" + i + " < " + to_string(H) + " && ix" + i + " >= 0 && ix" + i + " < " + to_string(W) + ";\n"
           "    let base" + i + " = select(0u, u32(iy" + i + " * " + to_string(W) + " + ix" + i + ") * " + U(CIN4) + ", v" + i + ");\n";
    }
    s += "    for (var c4 = 0u; c4 < " + U(CIN4) + "; c4++) {\n";
    for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
      s += "      let b" + to_string(n) + "_" + to_string(j) + " = in1[((tap * " + U(CIN4) + " + c4) * 4u + " + U(j) + ") * " + U(N4) + " + cb" + to_string(n) + "];\n";
    for (int m = 0; m < TM; m++)
      s += "      let a" + to_string(m) + " = select(vec4<f32>(0.0), in0[base" + to_string(m) + " + c4], v" + to_string(m) + ");\n";
    s += fmas("      ") + "    }\n  }\n";
  }
  for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) {
    string c = "c" + to_string(m) + "_";
    s += "  if (row0 + " + U(m) + " < " + U(M) + " && col4 + " + U(n) + " < " + U(N4) + ") { outb[z * " + U(M * N4) + " + (row0 + " + U(m) + ") * " + U(N4) + " + col4 + " + U(n) + "] = vec4<f32>(" +
         c + to_string(n * 4) + ", " + c + to_string(n * 4 + 1) + ", " + c + to_string(n * 4 + 2) + ", " + c + to_string(n * 4 + 3) + "); }\n";
  }
  return s + "}\n";
}

static void initDevice() {
  dawnProcSetProcs(&dawn::native::GetProcs());
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  vector<const char*> ad_en = {"use_vulkan_memory_model"}, dev_en = {"disable_robustness"};  // like ORT (RB=off VMM=1)
  wgpu::DawnTogglesDescriptor adt{}; adt.enabledToggleCount = ad_en.size(); adt.enabledToggles = ad_en.data(); ro.nextInChain = &adt;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data();
  wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt;
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
}

struct Cfg { int TM, NV, WX, WY; };
static const Cfg CFGS[] = {{8, 2, 32, 4}, {8, 2, 16, 4}, {8, 2, 8, 8}, {4, 2, 32, 4}, {4, 2, 16, 4}, {8, 1, 32, 4}, {4, 1, 16, 8}};

// direct stride-2 implicit GEMM: rows are output pixels
static string gemm_s2_wgsl(uint32_t Ho, uint32_t Wo, uint32_t N, uint32_t Hi, uint32_t Wi, uint32_t Cin, int TM, int NV, int WX, int WY) {
  uint32_t M = Ho * Wo, N4 = N / 4, CIN4 = Cin / 4;
  string s = hdr(2);
  s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
       "fn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let col4 = g.x * " + U(NV) + "; let row0 = g.y * " + U(TM) + ";\n";
  for (int n = 0; n < NV; n++) s += "  let cb" + to_string(n) + " = min(col4 + " + U(n) + ", " + U(N4 - 1) + ");\n";
  for (int m = 0; m < TM; m++) for (int c = 0; c < 4 * NV; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
  for (int m = 0; m < TM; m++)
    s += "  let p" + to_string(m) + " = row0 + " + U(m) + "; let y" + to_string(m) + " = i32(p" + to_string(m) + " / " + U(Wo) + ") * 2; let x" + to_string(m) + " = i32(p" + to_string(m) + " % " + U(Wo) + ") * 2;\n";
  s += "  for (var tap = 0u; tap < 9u; tap++) {\n    let dy = i32(tap / 3u) - 1; let dx = i32(tap % 3u) - 1;\n";
  for (int m = 0; m < TM; m++) {
    string i = to_string(m);
    s += "    let iy" + i + " = y" + i + " + dy; let ix" + i + " = x" + i + " + dx;\n"
         "    let v" + i + " = p" + i + " < " + U(M) + " && iy" + i + " >= 0 && iy" + i + " < " + to_string(Hi) + " && ix" + i + " >= 0 && ix" + i + " < " + to_string(Wi) + ";\n"
         "    let base" + i + " = select(0u, u32(iy" + i + " * " + to_string(Wi) + " + ix" + i + ") * " + U(CIN4) + ", v" + i + ");\n";
  }
  s += "    for (var c4 = 0u; c4 < " + U(CIN4) + "; c4++) {\n";
  for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
    s += "      let b" + to_string(n) + "_" + to_string(j) + " = in1[((tap * " + U(CIN4) + " + c4) * 4u + " + U(j) + ") * " + U(N4) + " + cb" + to_string(n) + "];\n";
  for (int m = 0; m < TM; m++)
    s += "      let a" + to_string(m) + " = select(vec4<f32>(0.0), in0[base" + to_string(m) + " + c4], v" + to_string(m) + ");\n";
  for (int j = 0; j < 4; j++) for (int n = 0; n < NV; n++) for (int m = 0; m < TM; m++) for (int q = 0; q < 4; q++)
    s += "      c" + to_string(m) + "_" + to_string(n * 4 + q) + " = fma(" + comp("a" + to_string(m), j) + ", " + comp("b" + to_string(n) + "_" + to_string(j), q) + ", c" + to_string(m) + "_" + to_string(n * 4 + q) + ");\n";
  s += "    }\n  }\n";
  for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) {
    string c = "c" + to_string(m) + "_";
    s += "  if (row0 + " + U(m) + " < " + U(M) + " && col4 + " + U(n) + " < " + U(N4) + ") { outb[(row0 + " + U(m) + ") * " + U(N4) + " + col4 + " + U(n) + "] = vec4<f32>(" +
         c + to_string(n * 4) + ", " + c + to_string(n * 4 + 1) + ", " + c + to_string(n * 4 + 2) + ", " + c + to_string(n * 4 + 3) + "); }\n";
  }
  return s + "}\n";
}

// batched GEMM over z = 0..E-1: C[z][T,N] = A[z][T,K] * B[z][K,N], same register-tile design
static string bgemm_wgsl(uint32_t M, uint32_t N, uint32_t K, int TM, int NV, int WX, int WY) {
  uint32_t N4 = N / 4, K4 = K / 4;
  string s = hdr(2);
  s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
       "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
       "  let col4 = g.x * " + U(NV) + "; let row0 = g.y * " + U(TM) + "; let z = wid.z;\n";
  for (int n = 0; n < NV; n++) s += "  let cb" + to_string(n) + " = min(col4 + " + U(n) + ", " + U(N4 - 1) + ");\n";
  for (int m = 0; m < TM; m++) for (int c = 0; c < 4 * NV; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
  s += "  for (var k4 = 0u; k4 < " + U(K4) + "; k4++) {\n";
  for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
    s += "    let b" + to_string(n) + "_" + to_string(j) + " = in1[z * " + U(K * N4) + " + (k4 * 4u + " + U(j) + ") * " + U(N4) + " + cb" + to_string(n) + "];\n";
  for (int m = 0; m < TM; m++)
    s += "    let a" + to_string(m) + " = in0[z * " + U(M * K4) + " + min(row0 + " + U(m) + ", " + U(M - 1) + ") * " + U(K4) + " + k4];\n";
  for (int j = 0; j < 4; j++) for (int n = 0; n < NV; n++) for (int m = 0; m < TM; m++) for (int q = 0; q < 4; q++)
    s += "    c" + to_string(m) + "_" + to_string(n * 4 + q) + " = fma(" + comp("a" + to_string(m), j) + ", " + comp("b" + to_string(n) + "_" + to_string(j), q) + ", c" + to_string(m) + "_" + to_string(n * 4 + q) + ");\n";
  s += "  }\n";
  for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) {
    string c = "c" + to_string(m) + "_";
    s += "  if (row0 + " + U(m) + " < " + U(M) + " && col4 + " + U(n) + " < " + U(N4) + ") { outb[z * " + U(M * N4) + " + (row0 + " + U(m) + ") * " + U(N4) + " + col4 + " + U(n) + "] = vec4<f32>(" +
         c + to_string(n * 4) + ", " + c + to_string(n * 4 + 1) + ", " + c + to_string(n * 4 + 2) + ", " + c + to_string(n * 4 + 3) + "); }\n";
  }
  return s + "}\n";
}

// per-dimension elements from v[0..4] = x[4t-1 .. 4t+3]: E0=v1 E1=v3 A=v0-v2 B=v2 C=v2-v4
static string poly_in_wgsl(uint32_t Hi, uint32_t Wi, uint32_t C4, uint32_t tW, uint32_t T) {
  string s = hdr(1);
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C4) + ") { return; }\n"
       "  let c4 = idx % " + U(C4) + "; let t = idx / " + U(C4) + "; let ty = i32(t / " + U(tW) + "); let tx = i32(t % " + U(tW) + ");\n";
  for (int r = 0; r < 5; r++) for (int c = 0; c < 5; c++) {
    string i = to_string(r) + to_string(c);
    s += "  var d" + i + " = vec4<f32>(0.0); { let iy = 4 * ty - 1 + " + to_string(r) + "; let ix = 4 * tx - 1 + " + to_string(c) + ";\n"
         "    if (iy >= 0 && iy < " + to_string(Hi) + " && ix >= 0 && ix < " + to_string(Wi) + ") { d" + i + " = in0[(u32(iy) * " + U(Wi) + " + u32(ix)) * " + U(C4) + " + c4]; } }\n";
  }
  // rows: element p of column c
  const char* E[5] = {"%1", "%3", "%0 - %2", "%2", "%2 - %4"};
  auto sub = [&](string tmpl, string pre, int col, bool rowmode) {  // replace %k with pre+index
    string o; for (size_t i = 0; i < tmpl.size(); i++) { if (tmpl[i] == '%') { int k = tmpl[i + 1] - '0'; o += rowmode ? pre + to_string(k) + to_string(col) : pre + to_string(col) + to_string(k); i++; } else o += tmpl[i]; }
    return o; };
  for (int c = 0; c < 5; c++) for (int p = 0; p < 5; p++) s += "  let t" + to_string(p) + to_string(c) + " = " + sub(E[p], "d", c, true) + ";\n";
  for (int p = 0; p < 5; p++) for (int q = 0; q < 5; q++)
    s += "  outb[" + U(p * 5 + q) + " * " + U(T * C4) + " + idx] = " + sub(E[q], "t", p, false) + ";\n";
  return s + "}\n";
}

// o0 = E0'+A'+B', o1 = E1'+B'-C'   (element order E0,E1,A,B,C)
static string poly_out_wgsl(uint32_t Ho, uint32_t Wo, uint32_t C4, uint32_t tW, uint32_t T) {
  string s = hdr(1);
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C4) + ") { return; }\n"
       "  let c4 = idx % " + U(C4) + "; let t = idx / " + U(C4) + "; let ty = t / " + U(tW) + "; let tx = t % " + U(tW) + ";\n";
  for (int p = 0; p < 5; p++) for (int q = 0; q < 5; q++) s += "  let m" + to_string(p) + to_string(q) + " = in0[" + U(p * 5 + q) + " * " + U(T * C4) + " + idx];\n";
  for (int q = 0; q < 5; q++) {  // combine along rows for each column element q
    string Q = to_string(q);
    s += "  let r0" + Q + " = m0" + Q + " + m2" + Q + " + m3" + Q + "; let r1" + Q + " = m1" + Q + " + m3" + Q + " - m4" + Q + ";\n";
  }
  for (int a = 0; a < 2; a++) {
    string A = to_string(a);
    string o[2] = {"r" + A + "0 + r" + A + "2 + r" + A + "3", "r" + A + "1 + r" + A + "3 - r" + A + "4"};
    for (int b = 0; b < 2; b++)
      s += "  { let oy = ty * 2u + " + U(a) + "; let ox = tx * 2u + " + U(b) + "; if (oy < " + U(Ho) + " && ox < " + U(Wo) + ") { outb[(oy * " + U(Wo) + " + ox) * " + U(C4) + " + c4] = " + o[b] + "; } }\n";
  }
  return s + "}\n";
}

static int runConv(uint32_t C, uint32_t Hi) {
  uint32_t Wi = Hi, Ho = (Hi + 1) / 2, Wo = Ho, M = Ho * Wo, C4 = C / 4, tH = (Ho + 1) / 2, tW = tH, T = tH * tW;
  vector<float> X((size_t)Hi * Wi * C), Wt((size_t)C * C * 9);  // Wt[co][ci][ky][kx]
  for (size_t i = 0; i < X.size(); i++) X[i] = ((i * 7919) % 1000) / 1000.f - .5f;
  for (size_t i = 0; i < Wt.size(); i++) Wt[i] = (((i * 104729) % 1000) / 1000.f - .5f) * 0.1f;
  vector<float> Bd((size_t)9 * C * C);  // direct B[(tap*C+ci)][co]
  for (uint32_t co = 0; co < C; co++) for (uint32_t ci = 0; ci < C; ci++) for (int t = 0; t < 9; t++) Bd[((size_t)t * C + ci) * C + co] = Wt[((size_t)co * C + ci) * 9 + t];
  // polyphase weights: U[pq][ci][co] = sum_{ky,kx} a_p[ky] a_q[kx] w[ky][kx]
  const double Acoef[5][3] = {{0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}};
  vector<float> Uw((size_t)25 * C * C);
  for (uint32_t co = 0; co < C; co++) for (uint32_t ci = 0; ci < C; ci++) {
    const float* g = &Wt[((size_t)co * C + ci) * 9];
    for (int p = 0; p < 5; p++) for (int q = 0; q < 5; q++) { double a = 0; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) a += Acoef[p][i] * Acoef[q][j] * g[i * 3 + j]; Uw[((size_t)(p * 5 + q) * C + ci) * C + co] = a; }
  }
  auto bX = upload(X), bBd = upload(Bd), bU = upload(Uw);
  auto bY = mk((size_t)M * C * 4, ST), bV = mk((size_t)25 * T * C * 4, ST), bM = mk((size_t)25 * T * C * 4, ST);
  vector<size_t> pr, pc; vector<double> ref;
  for (int q = 0; q < 300; q++) {
    size_t p = (q * 2654435761u) % M, co = (q * 40503u + 7) % C; int y = p / Wo, x = p % Wo; double a = 0;
    for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) { int iy = 2 * y + ky - 1, ix = 2 * x + kx - 1; if (iy < 0 || iy >= (int)Hi || ix < 0 || ix >= (int)Wi) continue;
      for (uint32_t ci = 0; ci < C; ci++) a += (double)X[((size_t)iy * Wi + ix) * C + ci] * Wt[((size_t)co * C + ci) * 9 + ky * 3 + kx]; }
    pr.push_back(p); pc.push_back(co); ref.push_back(a);
  }
  auto check = [&]() { auto Y = download(bY, (size_t)M * C); double me = 0, mr = 0; for (size_t i = 0; i < ref.size(); i++) { me = std::max(me, fabs(ref[i] - Y[pr[i] * C + pc[i]])); mr = std::max(mr, fabs(ref[i])); } return me / mr; };
  double flops = 2.0 * M * C * C * 9;
  printf("== conv 3x3 s2 C=%u %ux%u -> %ux%u  (%.3f GFLOP, T=%u tiles)\n", C, Hi, Wi, Ho, Wo, flops / 1e9, T);
  double bestD = 1e30; Cfg bd{};
  for (auto c : CFGS) {
    uint32_t gx = ((C4 + c.NV - 1) / c.NV + c.WX - 1) / c.WX, gy = (M + c.TM * c.WY - 1) / (c.TM * c.WY);
    Kern k = make(gemm_s2_wgsl(Ho, Wo, C, Hi, Wi, C, c.TM, c.NV, c.WX, c.WY), {bX, bBd, bY}, gx, gy);
    double ms = timeSeq({k}, 10); if (ms < bestD) { bestD = ms; bd = c; }
  }
  { uint32_t gx = ((C4 + bd.NV - 1) / bd.NV + bd.WX - 1) / bd.WX, gy = (M + bd.TM * bd.WY - 1) / (bd.TM * bd.WY);
    Kern k = make(gemm_s2_wgsl(Ho, Wo, C, Hi, Wi, C, bd.TM, bd.NV, bd.WX, bd.WY), {bX, bBd, bY}, gx, gy); timeSeq({k}, 1);
    printf("direct    best TM=%d NV=%d wg=%dx%d  %.3f ms  %.0f GFLOPS  relerr=%.1e\n", bd.TM, bd.NV, bd.WX, bd.WY, bestD, flops / bestD / 1e6, check()); }
  Kern kin = make(poly_in_wgsl(Hi, Wi, C4, tW, T), {bX, bV}, (T * C4 + 63) / 64);
  Kern kout = make(poly_out_wgsl(Ho, Wo, C4, tW, T), {bM, bY}, (T * C4 + 63) / 64);
  double tin = timeSeq({kin}, 10), tout = timeSeq({kout}, 10);
  double bestG = 1e30; Cfg bg{};
  for (auto c : CFGS) {
    uint32_t gx = ((C4 + c.NV - 1) / c.NV + c.WX - 1) / c.WX, gy = (T + c.TM * c.WY - 1) / (c.TM * c.WY);
    Kern k = make(bgemm_wgsl(T, C, C, c.TM, c.NV, c.WX, c.WY), {bV, bU, bM}, gx, gy, 25);
    double ms = timeSeq({k}, 10); if (ms < bestG) { bestG = ms; bg = c; }
  }
  uint32_t gx = ((C4 + bg.NV - 1) / bg.NV + bg.WX - 1) / bg.WX, gy = (T + bg.TM * bg.WY - 1) / (bg.TM * bg.WY);
  Kern kg = make(bgemm_wgsl(T, C, C, bg.TM, bg.NV, bg.WX, bg.WY), {bV, bU, bM}, gx, gy, 25);
  double tw = timeSeq({kin, kg, kout}, 10);
  printf("polyphase total %.3f ms  %.0f GFLOPS-equiv  relerr=%.1e   | in-xform %.3f + gemm(25x %ux%ux%u, TM=%d NV=%d wg=%dx%d) %.3f + out-xform %.3f ms (%.0f GFLOPS on the reduced work)\n",
         tw, flops / tw / 1e6, check(), tin, T, C, C, bg.TM, bg.NV, bg.WX, bg.WY, bestG, tout, 2.0 * 25 * T * C * C / bestG / 1e6);
  printf("RATIO polyphase/direct time = %.2f\n", tw / bestD);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: conv_s2 C H\n"); return 1; }
  initDevice();
  return runConv(atoi(argv[1]), atoi(argv[2]));
}
