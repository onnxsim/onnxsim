// Whole-conv packed-f16 path (f16 activations + weights in memory, f32 accumulate, f16 output with bias+ReLU epilogue)
// versus the f32 register-tile kernels, on the ResNet-50 conv classes, through Dawn/Vulkan.
//   conv_f16io pw  Cin Cout H   -- 1x1 conv as an NHWC GEMM (M=H*H, N=Cout, K=Cin)
//   conv_f16io c3  C H          -- direct 3x3 stride-1 pad-1 conv, Cin=Cout=C (implicit GEMM, 9 taps)
//   conv_f16io chain C H L      -- L stacked 3x3 layers, C channels, He-init random weights: f16 pipeline error vs f64
// Kernels: f32 = gemm.cc/conv_alt.cc "sc" design (scalar accumulators, TM x NV vec4 outputs/thread, no shared memory);
// f16io = A, B and the output as packed f16 (8 halves per vec4<u32>), f32 accumulators, bias+ReLU on the way out.
// Every kernel is swept over a few (TM, NV, WX, WY); the best per algorithm is printed. Accuracy: 300 sampled outputs vs a
// double-precision CPU reference computed from the ORIGINAL f32 data (so input, weight and output rounding all count).
// Build like conv_alt.cc (see README.md).
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


// ---- packed-f16 helpers: two halves per u32 (low half first, like unpack2x16float) ----
static uint16_t f2hbits(float f) { _Float16 h = (_Float16)f; uint16_t b; memcpy(&b, &h, 2); return b; }
static float h2fbits(uint16_t b) { _Float16 h; memcpy(&h, &b, 2); return (float)h; }
static vector<uint32_t> packHalf(const vector<float>& v) {
  vector<uint32_t> o(v.size() / 2);
  for (size_t i = 0; i < o.size(); i++) o[i] = (uint32_t)f2hbits(v[2 * i]) | ((uint32_t)f2hbits(v[2 * i + 1]) << 16);
  return o;
}
static vector<float> unpackHalf(const vector<uint32_t>& u) {
  vector<float> o(u.size() * 2);
  for (size_t i = 0; i < u.size(); i++) { o[2 * i] = h2fbits(u[i] & 0xffff); o[2 * i + 1] = h2fbits(u[i] >> 16); }
  return o;
}
static wgpu::Buffer uploadRaw(const void* p, size_t n) { auto b = mk(n, ST); dev.GetQueue().WriteBuffer(b, 0, p, (n + 3) & ~size_t(3)); return b; }
static vector<uint32_t> downloadU32(wgpu::Buffer src, size_t n) {
  auto f = download(src, n);  // same 4-byte copy; reinterpret
  vector<uint32_t> o(n); memcpy(o.data(), f.data(), n * 4); return o;
}

// f16io kernel: A [M][K/8] and B [K][N/8] as vec4<u32> (8 halves), bias vec4<f32>, out [M][N/8] packed, f32 accumulators, bias + ReLU.
// conv=true: A is the NHWC image [H*W][C/8], K = 9*C, B rows are (tap*C + ci).
static string p16io_wgsl(bool conv, uint32_t M, uint32_t N, uint32_t K, uint32_t H, uint32_t W, uint32_t C, int TM, int WX, int WY, bool relu) {
  uint32_t N8 = N / 8, K8 = K / 8, C8 = C / 8;
  string s = "@group(0) @binding(0) var<storage, read> A : array<vec4<u32>>;\n@group(0) @binding(1) var<storage, read> B : array<vec4<u32>>;\n"
             "@group(0) @binding(2) var<storage, read> bias : array<vec4<f32>>;\n@group(0) @binding(3) var<storage, read_write> O : array<vec4<u32>>;\n";
  s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let col8 = g.x; let cc = min(col8, " + U(N8 - 1) + "); let row0 = g.y * " + U(TM) + ";\n";
  for (int m = 0; m < TM; m++) for (int j = 0; j < 8; j++) s += "  var c" + to_string(m) + "_" + to_string(j) + " = 0.0;\n";
  auto body = [&](string ind, string bidx /* B row expression for kk (string containing KK) */) {
    string r;
    for (int kk = 0; kk < 8; kk++) {
      string b = bidx; size_t p = b.find("KK"); b.replace(p, 2, U(kk));
      r += ind + "let bw" + to_string(kk) + " = B[(" + b + ") * " + U(N8) + " + cc];\n";
      for (int q = 0; q < 4; q++) r += ind + "let bp" + to_string(kk) + "_" + to_string(q) + " = unpack2x16float(bw" + to_string(kk) + "." + "xyzw"[q] + ");\n";
      for (int m = 0; m < TM; m++) {
        r += ind + "{ let av = unpack2x16float(a" + to_string(m) + "." + "xyzw"[kk / 2] + ")." + (kk % 2 ? "y" : "x") + ";\n";
        for (int j = 0; j < 8; j++)
          r += ind + "  c" + to_string(m) + "_" + to_string(j) + " = fma(av, bp" + to_string(kk) + "_" + to_string(j / 2) + "." + (j % 2 ? "y" : "x") + ", c" + to_string(m) + "_" + to_string(j) + ");\n";
        r += ind + "}\n";
      }
    }
    return r;
  };
  if (!conv) {
    s += "  for (var k8 = 0u; k8 < " + U(K8) + "; k8++) {\n";
    for (int m = 0; m < TM; m++) s += "    let a" + to_string(m) + " = A[min(row0 + " + U(m) + ", " + U(M - 1) + ") * " + U(K8) + " + k8];\n";
    s += body("    ", "k8 * 8u + KK") + "  }\n";
  } else {
    for (int m = 0; m < TM; m++)
      s += "  let p" + to_string(m) + " = row0 + " + U(m) + "; let y" + to_string(m) + " = i32(p" + to_string(m) + " / " + U(W) + "); let x" + to_string(m) + " = i32(p" + to_string(m) + " % " + U(W) + ");\n";
    s += "  for (var tap = 0u; tap < 9u; tap++) {\n    let dy = i32(tap / 3u) - 1; let dx = i32(tap % 3u) - 1;\n";
    for (int m = 0; m < TM; m++) {
      string i = to_string(m);
      s += "    let iy" + i + " = y" + i + " + dy; let ix" + i + " = x" + i + " + dx;\n"
           "    let v" + i + " = p" + i + " < " + U(M) + " && iy" + i + " >= 0 && iy" + i + " < " + to_string(H) + " && ix" + i + " >= 0 && ix" + i + " < " + to_string(W) + ";\n"
           "    let base" + i + " = select(0u, u32(iy" + i + " * " + to_string(W) + " + ix" + i + ") * " + U(C8) + ", v" + i + ");\n";
    }
    s += "    for (var c8 = 0u; c8 < " + U(C8) + "; c8++) {\n";
    for (int m = 0; m < TM; m++) s += "      let a" + to_string(m) + " = select(vec4<u32>(0u), A[base" + to_string(m) + " + c8], v" + to_string(m) + ");\n";
    s += body("      ", "tap * " + U(C) + " + c8 * 8u + KK") + "    }\n  }\n";
  }
  s += "  let bv0 = bias[cc * 2u]; let bv1 = bias[cc * 2u + 1u];\n";
  for (int m = 0; m < TM; m++) {
    string c = "c" + to_string(m) + "_";
    string v[8];
    for (int j = 0; j < 8; j++) { v[j] = c + to_string(j) + " + " + (j < 4 ? "bv0." : "bv1.") + "xyzw"[j % 4]; if (relu) v[j] = "max(" + v[j] + ", 0.0)"; }
    s += "  if (row0 + " + U(m) + " < " + U(M) + " && col8 < " + U(N8) + ") { O[(row0 + " + U(m) + ") * " + U(N8) + " + col8] = vec4<u32>("
         "pack2x16float(vec2<f32>(" + v[0] + ", " + v[1] + ")), pack2x16float(vec2<f32>(" + v[2] + ", " + v[3] + ")), "
         "pack2x16float(vec2<f32>(" + v[4] + ", " + v[5] + ")), pack2x16float(vec2<f32>(" + v[6] + ", " + v[7] + "))); }\n";
  }
  return s + "}\n";
}

struct Cfg { int TM, NV, WX, WY; };
static const Cfg CFGS[] = {{8, 2, 32, 4}, {8, 2, 16, 4}, {8, 2, 8, 8}, {4, 2, 32, 4}, {4, 2, 16, 4}, {8, 1, 32, 4}, {4, 1, 16, 8}};
struct Cfg16 { int TM, WX, WY; };
static const Cfg16 CFG16[] = {{2, 16, 4}, {4, 16, 4}, {8, 16, 4}, {4, 32, 2}, {4, 8, 8}, {8, 8, 8}, {4, 8, 4}, {2, 8, 8}, {4, 4, 8}, {8, 4, 8}, {2, 4, 16}, {4, 4, 16}};

static uint32_t rngState = 12345;
static float frand() { rngState = rngState * 1664525u + 1013904223u; return (rngState >> 8) / 16777216.f; }  // [0,1)

struct Prob { bool conv; uint32_t M, N, K, H, W, C; vector<float> A, B, bias; };  // conv: A=[H*W][C], B=[(tap*C+ci)][co]

static Prob makeProb(bool conv, uint32_t Cin, uint32_t Cout, uint32_t H) {
  Prob p; p.conv = conv; p.H = H; p.W = H; p.C = Cin; p.M = H * H; p.N = Cout; p.K = conv ? 9 * Cin : Cin;
  p.A.resize((size_t)p.M * Cin); for (auto& v : p.A) v = frand();                         // post-ReLU-like: non-negative
  float a = sqrtf(6.f / p.K);                                                             // He-uniform
  p.B.resize((size_t)p.K * Cout); for (auto& v : p.B) v = (2 * frand() - 1) * a;
  p.bias.resize(Cout); for (auto& v : p.bias) v = 0.05f * (frand() - .5f);
  return p;
}

// f64 reference of one output element, from the original f32 data; pre = before bias/ReLU
static double refAt(const Prob& p, size_t row, size_t co, bool full) {
  double a = 0;
  if (!p.conv) { for (uint32_t k = 0; k < p.K; k++) a += (double)p.A[row * p.K + k] * p.B[(size_t)k * p.N + co]; }
  else {
    int y = row / p.W, x = row % p.W;
    for (int t = 0; t < 9; t++) { int iy = y + t / 3 - 1, ix = x + t % 3 - 1; if (iy < 0 || iy >= (int)p.H || ix < 0 || ix >= (int)p.W) continue;
      for (uint32_t ci = 0; ci < p.C; ci++) a += (double)p.A[((size_t)iy * p.W + ix) * p.C + ci] * p.B[((size_t)t * p.C + ci) * p.N + co]; }
  }
  if (full) { a += p.bias[co]; if (a < 0) a = 0; }
  return a;
}

struct Err { double maxrel, rmsrel; };
static Err errOf(const vector<double>& ref, const vector<double>& got) {
  double me = 0, mr = 0, se = 0, sr = 0;
  for (size_t i = 0; i < ref.size(); i++) { double d = fabs(ref[i] - got[i]); me = std::max(me, d); mr = std::max(mr, fabs(ref[i])); se += d * d; sr += ref[i] * ref[i]; }
  return {me / (mr + 1e-30), sqrt(se / (sr + 1e-30))};
}

static int runProb(Prob& p, const char* name) {
  double flops = 2.0 * p.M * p.N * p.K;
  vector<size_t> pr, pc; vector<double> refPre, refFull;
  for (int q = 0; q < 300; q++) { size_t r = (q * 2654435761u) % p.M, c = (q * 40503u + 7) % p.N; pr.push_back(r); pc.push_back(c); refPre.push_back(refAt(p, r, c, false)); refFull.push_back(refAt(p, r, c, true)); }
  printf("== %s  M=%u N=%u K=%u  (%.3f GFLOP)\n", name, p.M, p.N, p.K, flops / 1e9);
  // ---- f32 register tile ----
  { auto bA = upload(p.A), bB = upload(p.B), bO = mk((size_t)p.M * p.N * 4, ST);
    double best = 1e30; Cfg bc{};
    for (auto c : CFGS) {
      uint32_t N4 = p.N / 4, gx = ((N4 + c.NV - 1) / c.NV + c.WX - 1) / c.WX, gy = (p.M + c.TM * c.WY - 1) / (c.TM * c.WY);
      Kern k = make(gemm_wgsl(p.conv, p.M, p.N, p.K, p.H, p.W, p.C, c.TM, c.NV, c.WX, c.WY), {bA, bB, bO}, gx, gy);
      double ms = timeSeq({k}, 10); if (ms < best) { best = ms; bc = c; }
    }
    uint32_t N4 = p.N / 4, gx = ((N4 + bc.NV - 1) / bc.NV + bc.WX - 1) / bc.WX, gy = (p.M + bc.TM * bc.WY - 1) / (bc.TM * bc.WY);
    Kern k = make(gemm_wgsl(p.conv, p.M, p.N, p.K, p.H, p.W, p.C, bc.TM, bc.NV, bc.WX, bc.WY), {bA, bB, bO}, gx, gy); timeSeq({k}, 1);
    auto O = download(bO, (size_t)p.M * p.N); vector<double> got; for (size_t i = 0; i < pr.size(); i++) got.push_back(O[pr[i] * p.N + pc[i]]);
    Err e = errOf(refPre, got);
    printf("f32 reg-tile   best TM=%d NV=%d wg=%dx%d  %.3f ms  %.0f GFLOPS  maxrel=%.1e rmsrel=%.1e   (no epilogue)\n", bc.TM, bc.NV, bc.WX, bc.WY, best, flops / best / 1e6, e.maxrel, e.rmsrel); }
  // ---- packed f16 in/out, f32 accumulate, bias + ReLU ----
  { auto pA = packHalf(p.A), pB = packHalf(p.B);
    auto bA = uploadRaw(pA.data(), pA.size() * 4), bB = uploadRaw(pB.data(), pB.size() * 4), bBias = upload(p.bias), bO = mk((size_t)p.M * p.N / 2 * 4, ST);
    uint32_t N8 = p.N / 8; double best = 1e30; Cfg16 bc{};
    for (auto c : CFG16) {
      uint32_t gx = (N8 + c.WX - 1) / c.WX, gy = (p.M + c.TM * c.WY - 1) / (c.TM * c.WY);
      Kern k = make(p16io_wgsl(p.conv, p.M, p.N, p.K, p.H, p.W, p.C, c.TM, c.WX, c.WY, true), {bA, bB, bBias, bO}, gx, gy);
      double ms = timeSeq({k}, 10); if (ms < best) { best = ms; bc = c; }
    }
    uint32_t gx = (N8 + bc.WX - 1) / bc.WX, gy = (p.M + bc.TM * bc.WY - 1) / (bc.TM * bc.WY);
    Kern k = make(p16io_wgsl(p.conv, p.M, p.N, p.K, p.H, p.W, p.C, bc.TM, bc.WX, bc.WY, true), {bA, bB, bBias, bO}, gx, gy); timeSeq({k}, 1);
    auto O = unpackHalf(downloadU32(bO, (size_t)p.M * p.N / 2)); vector<double> got; for (size_t i = 0; i < pr.size(); i++) got.push_back(O[pr[i] * p.N + pc[i]]);
    Err e = errOf(refFull, got);
    printf("f16io packed   best TM=%d wg=%dx%d  %.3f ms  %.0f GFLOPS  maxrel=%.1e rmsrel=%.1e   (bias+ReLU, vs exact f32 data)\n", bc.TM, bc.WX, bc.WY, best, flops / best / 1e6, e.maxrel, e.rmsrel); }
  return 0;
}

// L stacked 3x3 layers (C channels, HxH) with He-init weights: error growth of the f16-in/out pipeline vs an f64 chain
static int runChain(uint32_t C, uint32_t H, int L) {
  uint32_t M = H * H, N8 = C / 8;
  vector<vector<float>> Wl(L), Bl(L);
  for (int l = 0; l < L; l++) { Prob p = makeProb(true, C, C, H); Wl[l] = p.B; Bl[l] = p.bias; }
  Prob p0 = makeProb(true, C, C, H);
  vector<double> cur(p0.A.begin(), p0.A.end());
  vector<vector<double>> refs;
  for (int l = 0; l < L; l++) {
    vector<double> nxt((size_t)M * C);
    for (uint32_t r = 0; r < M; r++) { int y = r / H, x = r % H;
      for (uint32_t co = 0; co < C; co++) { double a = Bl[l][co];
        for (int t = 0; t < 9; t++) { int iy = y + t / 3 - 1, ix = x + t % 3 - 1; if (iy < 0 || iy >= (int)H || ix < 0 || ix >= (int)H) continue;
          const double* in = &cur[((size_t)iy * H + ix) * C]; const float* w = &Wl[l][(size_t)t * C * C + co];
          for (uint32_t ci = 0; ci < C; ci++) a += in[ci] * w[(size_t)ci * C]; }
        nxt[(size_t)r * C + co] = a > 0 ? a : 0; } }
    refs.push_back(nxt); cur = nxt;
  }
  vector<wgpu::Buffer> act, wb, bb; vector<Kern> ks;
  { auto pk = packHalf(p0.A); act.push_back(uploadRaw(pk.data(), pk.size() * 4)); }
  uint32_t WX = 8, WY = 8, TM = 4, gx = (N8 + WX - 1) / WX, gy = (M + TM * WY - 1) / (TM * WY);
  for (int l = 0; l < L; l++) {
    auto pw = packHalf(Wl[l]); wb.push_back(uploadRaw(pw.data(), pw.size() * 4)); bb.push_back(upload(Bl[l]));
    act.push_back(mk((size_t)M * C / 2 * 4, ST));
    ks.push_back(make(p16io_wgsl(true, M, C, 9 * C, H, H, C, TM, WX, WY, true), {act[l], wb[l], bb[l], act[l + 1]}, gx, gy));
  }
  double ms = timeSeq(ks, 5);
  printf("== chain of %d 3x3 layers, C=%u, %ux%u: f16io pipeline %.3f ms per chain (%.3f ms/layer), error vs f64 chain of the same weights/biases (f32 data)\n", L, C, H, H, ms, ms / L);
  printf("layer   maxrel      rmsrel\n");
  for (int l = 0; l < L; l++) {
    auto got = unpackHalf(downloadU32(act[l + 1], (size_t)M * C / 2)); vector<double> g(got.begin(), got.end());
    Err e = errOf(refs[l], g); printf("%5d   %.2e   %.2e\n", l + 1, e.maxrel, e.rmsrel);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: conv_f16io pw Cin Cout H | c3 C H | chain C H L\n"); return 1; }
  initDevice();
  string m = argv[1];
  if (m == "pw" && argc >= 5) { Prob p = makeProb(false, atoi(argv[2]), atoi(argv[3]), atoi(argv[4])); return runProb(p, (string("1x1 ") + argv[2] + "->" + argv[3] + " @" + argv[4]).c_str()); }
  if (m == "c3" && argc >= 4) { Prob p = makeProb(true, atoi(argv[2]), atoi(argv[2]), atoi(argv[3])); return runProb(p, (string("3x3 ") + argv[2] + " @" + argv[3]).c_str()); }
  if (m == "chain" && argc >= 5) return runChain(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  fprintf(stderr, "bad args\n"); return 1;
}
