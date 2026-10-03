// Fused ReLU linear attention (EfficientViT-SAM encoder block) vs the unfused op chain ORT runs, through Dawn/Vulkan.
//   linattn [HEADS=32 DIM=32 N=256]
// Per block (B=1): X = NHWC conv output [N pixels][HEADS*3*DIM channels] (channel = h*3*DIM + part*DIM + d, part 0/1/2 = q/k/v;
// i.e. the Concat'ed qkv + aggregated outputs after the Reshape [1,HEADS,3*DIM,N]).  out[h][d][n] = (sum_e KV[d][e] relu(q)[e][n]) /
// (sum_e KV[DIM][e] relu(q)[e][n] + 1e-15) with KV[r][e] = sum_n vpad[r][n] relu(k)[e][n], vpad row DIM = 1.  Output written NHWC
// [N][HEADS*DIM] (what the following proj conv reads).
//  chain : the 15 dispatches ORT executes (transpose to NCHW, 3 Slice, 2 Relu, Transpose, Pad, MatMul, MatMul, 2 Slice, Add, Div, transpose back)
//  fused : kernel 1 = split-N partial KV with shared-memory tiles, kernel 2 = sums the partials, applies relu(q) on load, divides, writes NHWC
// Build like gemm.cc / conv_alt.cc (see README.md).  Checked against a double-precision CPU reference.
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
struct Kern { wgpu::ComputePipeline pl; wgpu::BindGroup bg; uint32_t gx, gy, gz; };

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

// `iters` repetitions of the whole sequence in one submission (each dispatch in its own pass); returns ms per repetition
static double runSeq(const vector<Kern>& ks, int iters) {
  wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
  for (int i = 0; i < iters; i++) for (auto& k : ks) {
    wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(k.pl); p.SetBindGroup(0, k.bg); p.DispatchWorkgroups(k.gx, k.gy, k.gz); p.End();
  }
  wgpu::CommandBuffer cb = enc.Finish();
  auto t0 = std::chrono::steady_clock::now();
  dev.GetQueue().Submit(1, &cb);
  inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

static wgpu::Buffer mk(size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = (n + 15) & ~size_t(15); b.usage = u; return dev.CreateBuffer(&b); }
static const auto ST = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
static wgpu::Buffer upload(const vector<float>& v) { auto b = mk(v.size() * 4, ST); dev.GetQueue().WriteBuffer(b, 0, v.data(), v.size() * 4); return b; }
static wgpu::Buffer zeros(size_t nfloats) { vector<float> z(nfloats, 0.f); return upload(z); }
static vector<float> download(wgpu::Buffer src, size_t n) {
  wgpu::Buffer rb = mk(n * 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  { wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); enc.CopyBufferToBuffer(src, 0, rb, 0, (n * 4 + 3) & ~size_t(3)); wgpu::CommandBuffer cb = enc.Finish(); dev.GetQueue().Submit(1, &cb); }
  bool ok = false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read, 0, rb.GetSize(), wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::MapAsyncStatus st, wgpu::StringView) { ok = st == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
  vector<float> out(n); if (ok) memcpy(out.data(), rb.GetConstMappedRange(0, rb.GetSize()), n * 4);
  return out;
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

static string U(uint32_t v) { return to_string(v) + "u"; }
static string bind(int i, bool rw) { return "@group(0) @binding(" + to_string(i) + ") var<storage, " + (rw ? "read_write" : "read") + "> b" + to_string(i) + " : array<f32>;\n"; }

// 1-D element-wise style kernel: `nin` read-only buffers (b0..), one output (b<nin>), body sees `i` and may write `b<nin>[i]`
static string ew(int nin, uint32_t n, const string& body) {
  string s;
  for (int i = 0; i < nin; i++) s += bind(i, false);
  s += bind(nin, true);
  s += "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(num_workgroups) nw : vec3<u32>) {\n"
       "  let i = g.x + g.y * nw.x * 64u;\n  if (i >= " + U(n) + ") { return; }\n" + body + "}\n";
  return s;
}
static void disp(uint64_t n, uint32_t& gx, uint32_t& gy) { uint64_t groups = (n + 63) / 64; gx = (uint32_t)std::min<uint64_t>(groups, 65535); gy = (uint32_t)((groups + gx - 1) / gx); }

int main(int argc, char** argv) {
  uint32_t H = argc > 1 ? atoi(argv[1]) : 32, D = argc > 2 ? atoi(argv[2]) : 32, N = argc > 3 ? atoi(argv[3]) : 256;
  const int NS = 4;                       // N splits in the fused first kernel
  const uint32_t TN = 16;                 // n tile in shared memory
  if (N % (NS * TN) != 0 || D != 32) { fprintf(stderr, "needs D==32 and N %% %u == 0\n", NS * TN); return 1; }
  initDevice();
  const uint32_t C = H * 3 * D, R = D + 1;
  // ---- data: mixed sign, ~half the k/q entries zero after ReLU; some heads/positions have exactly zero denominators
  vector<float> X((size_t)N * C);
  for (size_t i = 0; i < X.size(); i++) X[i] = (((i * 7919u + (i >> 5) * 31u) % 2000) / 1000.f - 1.f) * 0.7f;
  for (uint32_t n = 0; n < N; n += 37) for (uint32_t c = 0; c < C; c++) if ((c % (3 * D)) < D) X[(size_t)n * C + c] = -fabsf(X[(size_t)n * C + c]);  // q<0 -> relu 0 -> num=den=0
  // ---- CPU reference (double), Y[n][h*D+d]
  vector<float> ref((size_t)N * H * D);
  {
    vector<double> KV(R * D);
    for (uint32_t h = 0; h < H; h++) {
      std::fill(KV.begin(), KV.end(), 0.0);
      for (uint32_t n = 0; n < N; n++) for (uint32_t d = 0; d < D; d++) {
        double kk = std::max(0.f, X[(size_t)n * C + h * 3 * D + D + d]);
        for (uint32_t r = 0; r < R; r++) KV[r * D + d] += (r < D ? (double)X[(size_t)n * C + h * 3 * D + 2 * D + r] : 1.0) * kk;
      }
      for (uint32_t n = 0; n < N; n++) {
        double num[64], den = 0;
        for (uint32_t r = 0; r < R; r++) { double a = 0; for (uint32_t e = 0; e < D; e++) a += KV[r * D + e] * std::max(0.f, X[(size_t)n * C + h * 3 * D + e]); if (r < D) num[r] = a; else den = a; }
        for (uint32_t d = 0; d < D; d++) ref[(size_t)n * H * D + h * D + d] = (float)(num[d] / (den + (double)1e-15f));
      }
    }
  }
  auto bX = upload(X);
  // ---- chain (15 dispatches)
  const uint32_t HD3 = H * 3 * D;
  auto bR = zeros((size_t)HD3 * N), bQ = zeros((size_t)H * D * N), bK = zeros((size_t)H * D * N), bV = zeros((size_t)H * D * N);
  auto bQr = zeros((size_t)H * D * N), bKr = zeros((size_t)H * D * N), bKT = zeros((size_t)H * D * N), bVP = zeros((size_t)H * R * N);
  auto bKV = zeros((size_t)H * R * D), bNUM = zeros((size_t)H * R * N), bNs = zeros((size_t)H * D * N), bDs = zeros((size_t)H * N), bD2 = zeros((size_t)H * N);
  auto bO = zeros((size_t)H * D * N), bY = zeros((size_t)N * H * D);
  vector<Kern> chain;
  auto add = [&](const string& w, const vector<wgpu::Buffer>& b, uint64_t n) { uint32_t gx, gy; disp(n, gx, gy); chain.push_back(make(w, b, gx, gy)); };
  const uint32_t HDN = H * D * N;
  // 0 transpose NHWC -> [h*3D + j][n]
  add(ew(1, HD3 * N, "  let nn = i % " + U(N) + "; let hj = i / " + U(N) + ";\n  b1[i] = b0[nn * " + U(C) + " + hj];\n"), {bX, bR}, (uint64_t)HD3 * N);
  for (int part = 0; part < 3; part++)  // Slice q, k, v
    add(ew(1, HDN, "  let h = i / " + U(D * N) + "; let r = (i / " + U(N) + ") % " + U(D) + "; let nn = i % " + U(N) + ";\n  b1[i] = b0[(h * " + U(3 * D) + " + " + U(part * D) + " + r) * " + U(N) + " + nn];\n"), {bR, part == 0 ? bQ : part == 1 ? bK : bV}, HDN);
  add(ew(1, HDN, "  b1[i] = max(b0[i], 0.0);\n"), {bQ, bQr}, HDN);
  add(ew(1, HDN, "  b1[i] = max(b0[i], 0.0);\n"), {bK, bKr}, HDN);
  add(ew(1, HDN, "  let h = i / " + U(D * N) + "; let nn = (i / " + U(D) + ") % " + U(N) + "; let d = i % " + U(D) + ";\n  b1[i] = b0[h * " + U(D * N) + " + d * " + U(N) + " + nn];\n"), {bKr, bKT}, HDN);  // Transpose k
  add(ew(1, (uint32_t)H * R * N, "  let h = i / " + U(R * N) + "; let r = (i / " + U(N) + ") % " + U(R) + "; let nn = i % " + U(N) + ";\n  b1[i] = select(1.0, b0[h * " + U(D * N) + " + min(r, " + U(D - 1) + ") * " + U(N) + " + nn], r < " + U(D) + ");\n"), {bV, bVP}, (uint64_t)H * R * N);  // Pad
  add(ew(2, H * R * D, "  let h = i / " + U(R * D) + "; let m = (i / " + U(D) + ") % " + U(R) + "; let nn = i % " + U(D) + ";\n  var a = 0.0;\n  for (var k = 0u; k < " + U(N) + "; k++) { a += b0[h * " + U(R * N) + " + m * " + U(N) + " + k] * b1[h * " + U(N * D) + " + k * " + U(D) + " + nn]; }\n  b2[i] = a;\n"), {bVP, bKT, bKV}, (uint64_t)H * R * D);  // MatMul 1
  add(ew(2, H * R * N, "  let h = i / " + U(R * N) + "; let m = (i / " + U(N) + ") % " + U(R) + "; let nn = i % " + U(N) + ";\n  var a = 0.0;\n  for (var e = 0u; e < " + U(D) + "; e++) { a += b0[h * " + U(R * D) + " + m * " + U(D) + " + e] * b1[h * " + U(D * N) + " + e * " + U(N) + " + nn]; }\n  b2[i] = a;\n"), {bKV, bQr, bNUM}, (uint64_t)H * R * N);  // MatMul 2
  add(ew(1, HDN, "  let h = i / " + U(D * N) + "; let r = i % " + U(D * N) + ";\n  b1[i] = b0[h * " + U(R * N) + " + r];\n"), {bNUM, bNs}, HDN);  // Slice num
  add(ew(1, H * N, "  let h = i / " + U(N) + "; let nn = i % " + U(N) + ";\n  b1[i] = b0[h * " + U(R * N) + " + " + U(D * N) + " + nn];\n"), {bNUM, bDs}, (uint64_t)H * N);  // Slice den
  add(ew(1, H * N, "  b1[i] = b0[i] + 1.0e-15;\n"), {bDs, bD2}, (uint64_t)H * N);  // Add eps
  add(ew(2, HDN, "  let h = i / " + U(D * N) + "; let nn = i % " + U(N) + ";\n  b2[i] = b0[i] / b1[h * " + U(N) + " + nn];\n"), {bNs, bD2, bO}, HDN);  // Div
  add(ew(1, N * H * D, "  let nn = i / " + U(H * D) + "; let c = i % " + U(H * D) + ";\n  b1[i] = b0[c * " + U(N) + " + nn];\n"), {bO, bY}, (uint64_t)N * H * D);  // back to NHWC
  // ---- fused
  auto bP = zeros((size_t)H * NS * R * D), bY2 = zeros((size_t)N * H * D);
  vector<Kern> fused;
  {
    string s = bind(0, false) + bind(1, true);
    s += "var<workgroup> kt : array<f32, " + to_string(TN * D) + ">;\nvar<workgroup> vt : array<f32, " + to_string(TN * D) + ">;\n";
    s += "@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_index) t : u32, @builtin(workgroup_id) wid : vec3<u32>) {\n"
         "  let h = wid.x; let sp = wid.y; let nbase = sp * " + U(N / NS) + ";\n"
         "  var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0; var acc4 = 0.0;\n";
    s += "  for (var tile = 0u; tile < " + U(N / NS / TN) + "; tile++) {\n"
         "    let n0 = nbase + tile * " + U(TN) + ";\n"
         "    for (var q = t; q < " + U(TN * D) + "; q += 256u) { let nl = q / " + U(D) + "; let d = q % " + U(D) + ";\n"
         "      let base = (n0 + nl) * " + U(C) + " + h * " + U(3 * D) + ";\n"
         "      kt[q] = max(b0[base + " + U(D) + " + d], 0.0); vt[q] = b0[base + " + U(2 * D) + " + d]; }\n"
         "    workgroupBarrier();\n"
         "    for (var nl = 0u; nl < " + U(TN) + "; nl++) {\n";
    for (int i = 0; i < 5; i++) {
      s += "      { let o = t + " + to_string(i * 256) + "u; if (o < " + U(R * D) + ") { let r = o / " + U(D) + "; let d = o % " + U(D) + ";\n"
           "        acc" + to_string(i) + " += select(1.0, vt[nl * " + U(D) + " + min(r, " + U(D - 1) + ")], r < " + U(D) + ") * kt[nl * " + U(D) + " + d]; } }\n";
    }
    s += "    }\n    workgroupBarrier();\n  }\n";
    for (int i = 0; i < 5; i++) s += "  { let o = t + " + to_string(i * 256) + "u; if (o < " + U(R * D) + ") { b1[((h * " + U(NS) + " + sp) * " + U(R * D) + ") + o] = acc" + to_string(i) + "; } }\n";
    s += "}\n";
    fused.push_back(make(s, {bX, bP}, H, NS));
  }
  {
    const uint32_t WG = 64;
    string s = bind(0, false) + bind(1, false) + bind(2, true);  // X, partials, Y
    s += "var<workgroup> kv : array<f32, " + to_string(R * D) + ">;\n";
    s += "@compute @workgroup_size(" + to_string(WG) + ")\nfn main(@builtin(local_invocation_index) t : u32, @builtin(workgroup_id) wid : vec3<u32>) {\n"
         "  let h = wid.y; let n = wid.x * " + U(WG) + " + t;\n"
         "  for (var o = t; o < " + U(R * D) + "; o += " + U(WG) + ") { var a = 0.0;\n"
         "    for (var sp = 0u; sp < " + U(NS) + "; sp++) { a += b1[(h * " + U(NS) + " + sp) * " + U(R * D) + " + o]; }\n"
         "    kv[o] = a; }\n"
         "  workgroupBarrier();\n"
         "  if (n >= " + U(N) + ") { return; }\n"
         "  var q : array<f32, " + to_string(D) + ">;\n"
         "  for (var e = 0u; e < " + U(D) + "; e++) { q[e] = max(b0[n * " + U(C) + " + h * " + U(3 * D) + " + e], 0.0); }\n"
         "  var den = 0.0;\n  for (var e = 0u; e < " + U(D) + "; e++) { den += kv[" + U(D * D) + " + e] * q[e]; }\n"
         "  let inv_den = den + 1.0e-15;\n"
         "  for (var d = 0u; d < " + U(D) + "; d++) { var a = 0.0;\n"
         "    for (var e = 0u; e < " + U(D) + "; e++) { a += kv[d * " + U(D) + " + e] * q[e]; }\n"
         "    b2[n * " + U(H * D) + " + h * " + U(D) + " + d] = a / inv_den; }\n"
         "}\n";
    fused.push_back(make(s, {bX, bP, bY2}, N / WG, H));
  }
  // ---- correctness
  runSeq(chain, 1); runSeq(fused, 1);
  auto err = [&](wgpu::Buffer b) { auto y = download(b, ref.size()); double me = 0, mr = 0; size_t nan = 0; for (size_t i = 0; i < ref.size(); i++) { if (!std::isfinite(y[i])) { nan++; continue; } me = std::max(me, (double)fabsf(y[i] - ref[i])); mr = std::max(mr, (double)fabsf(ref[i])); } return std::make_pair(me / mr, nan); };
  auto ec = err(bY), ef = err(bY2);
  printf("block: heads=%u dim=%u N=%u  (chain %zu dispatches, fused %zu)\n", H, D, N, chain.size(), fused.size());
  printf("relerr vs double CPU reference: chain %.1e (nonfinite %zu), fused %.1e (nonfinite %zu)\n", ec.first, ec.second, ef.first, ef.second);
  // ---- timing: warm GPU >= 5 s, then ABAB rounds
  { auto t0 = std::chrono::steady_clock::now(); while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 5.0) { runSeq(chain, 20); runSeq(fused, 50); } }
  vector<double> tc, tf, tc1, tf1;
  for (int r = 0; r < 12; r++) { tc.push_back(runSeq(chain, 20)); tf.push_back(runSeq(fused, 50)); tc1.push_back(runSeq(chain, 1)); tf1.push_back(runSeq(fused, 1)); }
  auto stat = [](vector<double> v, double& mn, double& md) { std::sort(v.begin(), v.end()); mn = v[0]; md = v[v.size() / 2]; };
  double a, b, c, d2, e, f, g2, h2;
  stat(tc, a, b); stat(tf, c, d2); stat(tc1, e, f); stat(tf1, g2, h2);
  printf("steady state (back-to-back, per block): chain min %.3f median %.3f ms | fused min %.3f median %.3f ms | speedup %.1fx\n", a, b, c, d2, a / c);
  printf("single block latency (1 rep/submit)  : chain min %.3f median %.3f ms | fused min %.3f median %.3f ms\n", e, f, g2, h2);
  printf("x4 blocks (steady state)             : chain %.2f ms, fused %.2f ms, saved %.2f ms\n", 4 * a, 4 * c, 4 * (a - c));
  // per-kernel chain breakdown
  printf("chain per dispatch (min of 5, 20 reps in one submit):");
  for (size_t i = 0; i < chain.size(); i++) { double m = 1e30; for (int r = 0; r < 5; r++) m = std::min(m, runSeq({chain[i]}, 20)); printf(" %.3f", m); }
  printf(" ms\nfused per dispatch:");
  for (size_t i = 0; i < fused.size(); i++) { double m = 1e30; for (int r = 0; r < 5; r++) m = std::min(m, runSeq({fused[i]}, 50)); printf(" %.3f", m); }
  printf(" ms\n");
  return 0;
}
