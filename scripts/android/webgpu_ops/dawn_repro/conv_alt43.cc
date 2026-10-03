// Winograd F(4,3) vs F(2,3) vs direct register-tile 3x3 convolution through Dawn/Vulkan (NHWC fp32, batch 1, pad 1).
//   conv_alt43 C H     -- 3x3 stride-1 pad-1 conv, Cin=Cout=C, HxH
// Both Winograd variants use one generic generator (transform matrices B^T, G, A^T), so the transforms are timed the same way.
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

struct Wino { int m, a; vector<vector<double>> BT, G, AT; };  // a = m + 2 (input tile), BT a x a, G a x 3, AT m x a
static Wino wino23() {
  return {2, 4, {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}}, {{1, 0, 0}, {.5, .5, .5}, {.5, -.5, .5}, {0, 0, 1}}, {{1, 1, 1, 0}, {0, 1, -1, -1}}};
}
static Wino wino43() {  // Lavin & Gray, points 0, +-1, +-2, inf
  return {4, 6, {{4, 0, -5, 0, 1, 0}, {0, -4, -4, 1, 1, 0}, {0, 4, -4, -1, 1, 0}, {0, -2, -1, 2, 1, 0}, {0, 2, -1, -2, 1, 0}, {0, 4, 0, -5, 0, 1}},
          {{1. / 4, 0, 0}, {-1. / 6, -1. / 6, -1. / 6}, {-1. / 6, 1. / 6, -1. / 6}, {1. / 24, 1. / 12, 1. / 6}, {1. / 24, -1. / 12, 1. / 6}, {0, 0, 1}},
          {{1, 1, 1, 1, 1, 0}, {0, 1, -1, 2, -2, 0}, {0, 1, 1, 4, 4, 0}, {0, 1, -1, 8, -8, 1}}};
}
static string lin(const vector<double>& coef, const vector<string>& v) {  // sum coef[i]*v[i], zeros skipped
  string s; char b[64];
  for (size_t i = 0; i < coef.size(); i++) {
    if (coef[i] == 0) continue;
    double c = coef[i], ac = fabs(c);
    if (ac == 1) s += (c < 0 ? " - " : (s.empty() ? "" : " + ")) + v[i]; else { snprintf(b, sizeof b, "%.9g", ac); s += string(c < 0 ? " - " : (s.empty() ? "" : " + ")) + "(" + b + ") * " + v[i]; }
    if (s.size() && s[0] == ' ' && s.size() > 3 && s.substr(0, 3) == " - ") s = "-" + s.substr(3);
  }
  return s.empty() ? "vec4<f32>(0.0)" : s;
}
static string wino_in_wgsl(const Wino& w, uint32_t H, uint32_t W, uint32_t C4, uint32_t tW, uint32_t T) {
  int a = w.a; string s = hdr(1);
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C4) + ") { return; }\n"
       "  let c4 = idx % " + U(C4) + "; let t = idx / " + U(C4) + "; let ty = i32(t / " + U(tW) + "); let tx = i32(t % " + U(tW) + ");\n";
  for (int r = 0; r < a; r++) for (int c = 0; c < a; c++) {
    string i = to_string(r) + "_" + to_string(c);
    s += "  var d" + i + " = vec4<f32>(0.0); { let iy = " + to_string(w.m) + " * ty - 1 + " + to_string(r) + "; let ix = " + to_string(w.m) + " * tx - 1 + " + to_string(c) + ";\n"
         "    if (iy >= 0 && iy < " + to_string(H) + " && ix >= 0 && ix < " + to_string(W) + ") { d" + i + " = in0[(u32(iy) * " + U(W) + " + u32(ix)) * " + U(C4) + " + c4]; } }\n";
  }
  for (int r = 0; r < a; r++) for (int c = 0; c < a; c++) {  // t = B^T d : t[r][c] = sum_k BT[r][k] d[k][c]
    vector<string> col; for (int k = 0; k < a; k++) col.push_back("d" + to_string(k) + "_" + to_string(c));
    s += "  let t" + to_string(r) + "_" + to_string(c) + " = " + lin(w.BT[r], col) + ";\n";
  }
  for (int r = 0; r < a; r++) for (int c = 0; c < a; c++) {  // v = t B : v[r][c] = sum_k t[r][k] BT[c][k]
    vector<string> row; for (int k = 0; k < a; k++) row.push_back("t" + to_string(r) + "_" + to_string(k));
    s += "  outb[" + U(r * a + c) + " * " + U(T * C4) + " + idx] = " + lin(w.BT[c], row) + ";\n";
  }
  return s + "}\n";
}
static string wino_out_wgsl(const Wino& w, uint32_t H, uint32_t W, uint32_t C4, uint32_t tW, uint32_t T) {
  int a = w.a, m = w.m; string s = hdr(1);
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C4) + ") { return; }\n"
       "  let c4 = idx % " + U(C4) + "; let t = idx / " + U(C4) + "; let ty = t / " + U(tW) + "; let tx = t % " + U(tW) + ";\n";
  for (int r = 0; r < a; r++) for (int c = 0; c < a; c++)
    s += "  let m" + to_string(r) + "_" + to_string(c) + " = in0[" + U(r * a + c) + " * " + U(T * C4) + " + idx];\n";
  for (int r = 0; r < m; r++) for (int c = 0; c < a; c++) {  // s = A^T m
    vector<string> col; for (int k = 0; k < a; k++) col.push_back("m" + to_string(k) + "_" + to_string(c));
    s += "  let s" + to_string(r) + "_" + to_string(c) + " = " + lin(w.AT[r], col) + ";\n";
  }
  for (int r = 0; r < m; r++) for (int c = 0; c < m; c++) {  // y = s A
    vector<string> row; for (int k = 0; k < a; k++) row.push_back("s" + to_string(r) + "_" + to_string(k));
    s += "  { let oy = ty * " + U(m) + " + " + U(r) + "; let ox = tx * " + U(m) + " + " + U(c) + "; if (oy < " + U(H) + " && ox < " + U(W) + ") { outb[(oy * " + U(W) + " + ox) * " + U(C4) + " + c4] = " + lin(w.AT[c], row) + "; } }\n";
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
static const Cfg CFGS[] = {{8, 2, 32, 4}, {8, 2, 16, 4}, {8, 2, 8, 8}, {4, 2, 32, 4}, {4, 2, 16, 4}, {8, 1, 32, 4}, {4, 1, 16, 8}, {2, 2, 16, 8}, {2, 1, 16, 8}, {1, 2, 16, 8}};

struct WinoRes { double total, tin, gemm, tout, relerr; Cfg cfg; };

static int runConv(uint32_t C, uint32_t H) {
  uint32_t W = H, M = H * W, C4 = C / 4;
  vector<float> X((size_t)M * C), Wt((size_t)C * C * 9);  // Wt[co][ci][ky][kx]
  for (size_t i = 0; i < X.size(); i++) X[i] = ((i * 7919) % 1000) / 1000.f - .5f;
  for (size_t i = 0; i < Wt.size(); i++) Wt[i] = (((i * 104729) % 1000) / 1000.f - .5f) * 0.1f;
  vector<float> Bd((size_t)9 * C * C);
  for (uint32_t co = 0; co < C; co++) for (uint32_t ci = 0; ci < C; ci++) for (int t = 0; t < 9; t++) Bd[((size_t)t * C + ci) * C + co] = Wt[((size_t)co * C + ci) * 9 + t];
  auto bX = upload(X), bBd = upload(Bd);
  auto bY = mk((size_t)M * C * 4, ST);
  vector<size_t> pr, pc; vector<double> ref;
  for (int q = 0; q < 400; q++) {
    size_t p = (q * 2654435761u) % M, co = (q * 40503u + 7) % C; int y = p / W, x = p % W; double a = 0;
    for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) { int iy = y + ky - 1, ix = x + kx - 1; if (iy < 0 || iy >= (int)H || ix < 0 || ix >= (int)W) continue;
      for (uint32_t ci = 0; ci < C; ci++) a += (double)X[((size_t)iy * W + ix) * C + ci] * Wt[((size_t)co * C + ci) * 9 + ky * 3 + kx]; }
    pr.push_back(p); pc.push_back(co); ref.push_back(a);
  }
  auto check = [&]() { auto Y = download(bY, (size_t)M * C); double me = 0, mr = 0; for (size_t i = 0; i < ref.size(); i++) { me = std::max(me, fabs(ref[i] - Y[pr[i] * C + pc[i]])); mr = std::max(mr, fabs(ref[i])); } return me / mr; };
  double flops = 2.0 * M * C * C * 9;
  printf("== conv 3x3 C=%u H=W=%u  (%.3f GFLOP)\n", C, H, flops / 1e9);
  double bestD = 1e30; Cfg bd{};
  for (auto c : CFGS) {
    uint32_t gx = ((C4 + c.NV - 1) / c.NV + c.WX - 1) / c.WX, gy = (M + c.TM * c.WY - 1) / (c.TM * c.WY);
    Kern k = make(gemm_wgsl(true, M, C, 9 * C, H, W, C, c.TM, c.NV, c.WX, c.WY), {bX, bBd, bY}, gx, gy);
    double ms = timeSeq({k}, 10); if (ms < bestD) { bestD = ms; bd = c; }
  }
  { uint32_t gx = ((C4 + bd.NV - 1) / bd.NV + bd.WX - 1) / bd.WX, gy = (M + bd.TM * bd.WY - 1) / (bd.TM * bd.WY);
    Kern k = make(gemm_wgsl(true, M, C, 9 * C, H, W, C, bd.TM, bd.NV, bd.WX, bd.WY), {bX, bBd, bY}, gx, gy); timeSeq({k}, 1);
    printf("direct     best TM=%d NV=%d wg=%dx%d  %.3f ms  %.0f GFLOPS  relerr=%.1e\n", bd.TM, bd.NV, bd.WX, bd.WY, bestD, flops / bestD / 1e6, check()); }
  WinoRes res[2];
  const Wino ws[2] = {wino23(), wino43()};
  for (int wi = 0; wi < 2; wi++) {
    const Wino& w = ws[wi]; int a = w.a, E = a * a;
    uint32_t tH = (H + w.m - 1) / w.m, tW = (W + w.m - 1) / w.m, T = tH * tW;
    vector<float> Uw((size_t)E * C * C);
    for (uint32_t co = 0; co < C; co++) for (uint32_t ci = 0; ci < C; ci++) {
      const float* g = &Wt[((size_t)co * C + ci) * 9];
      for (int r = 0; r < a; r++) for (int c = 0; c < a; c++) { double s = 0; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) s += w.G[r][i] * g[i * 3 + j] * w.G[c][j]; Uw[((size_t)(r * a + c) * C + ci) * C + co] = s; }
    }
    auto bU = upload(Uw);
    auto bV = mk((size_t)E * T * C * 4, ST), bM = mk((size_t)E * T * C * 4, ST);
    Kern kin = make(wino_in_wgsl(w, H, W, C4, tW, T), {bX, bV}, (T * C4 + 63) / 64);
    Kern kout = make(wino_out_wgsl(w, H, W, C4, tW, T), {bM, bY}, (T * C4 + 63) / 64);
    double tin = timeSeq({kin}, 10), tout = timeSeq({kout}, 10);
    double bestG = 1e30; Cfg bg{};
    for (auto c : CFGS) {
      uint32_t gx = ((C4 + c.NV - 1) / c.NV + c.WX - 1) / c.WX, gy = (T + c.TM * c.WY - 1) / (c.TM * c.WY);
      Kern k = make(gemm_wgsl(false, T, C, C, 0, 0, 0, c.TM, c.NV, c.WX, c.WY), {bV, bU, bM}, gx, gy, E);
      double ms = timeSeq({k}, 10); if (ms < bestG) { bestG = ms; bg = c; }
    }
    uint32_t gx = ((C4 + bg.NV - 1) / bg.NV + bg.WX - 1) / bg.WX, gy = (T + bg.TM * bg.WY - 1) / (bg.TM * bg.WY);
    Kern kg = make(gemm_wgsl(false, T, C, C, 0, 0, 0, bg.TM, bg.NV, bg.WX, bg.WY), {bV, bU, bM}, gx, gy, E);
    double tw = timeSeq({kin, kg, kout}, 10);
    res[wi] = {tw, tin, bestG, tout, check(), bg};
    printf("winograd F(%d,3) total %.3f ms | in %.3f + gemm(%dx %ux%ux%u, TM=%d NV=%d wg=%dx%d) %.3f + out %.3f | relerr=%.1e | x direct %.2f\n",
           w.m, tw, tin, E, T, C, C, bg.TM, bg.NV, bg.WX, bg.WY, bestG, tout, res[wi].relerr, tw / bestD);
  }
  printf("RATIO F43/F23 = %.2f   F23/direct = %.2f   F43/direct = %.2f\n", res[1].total / res[0].total, res[0].total / bestD, res[1].total / bestD);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: conv_alt43 C H\n"); return 1; }
  initDevice();
  return runConv(atoi(argv[1]), atoi(argv[2]));
}
