// Winograd F(2,3) with f16 *intermediates only* (V, U, optionally M packed as f16; f32 accumulation; f32 graph input/output),
// vs the all-f32 Winograd pipeline. NHWC, batch 1, 3x3 stride-1 pad-1, Cin=Cout=C.
//   conv_alt_f16 C H [relu]     -- "relu": post-ReLU-like nonnegative activations instead of zero-mean ones
// Pipelines timed back to back in one submission (input xform -> 16 batched GEMMs -> output xform):
//   f32     : all f32 (conv_alt.cc's Winograd)
//   f16-M32 : V,U packed f16 (8 halves per vec4<u32>), GEMM unpacks + f32 fma, M written f32
//   f16-M16 : same, M also packed f16
// U (weight transform) is computed on the host in double and packed; in ORT it is computed once and cached, so its cost is not timed.
// Accuracy: max |err| / max |ref| over 300 sampled outputs vs a double-precision direct convolution. Build like gemm.cc.
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

static string wino_in_wgsl(uint32_t H, uint32_t W, uint32_t C4, uint32_t tW, uint32_t T) {
  string s = hdr(1);
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C4) + ") { return; }\n"
       "  let c4 = idx % " + U(C4) + "; let t = idx / " + U(C4) + "; let ty = i32(t / " + U(tW) + "); let tx = i32(t % " + U(tW) + ");\n";
  for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
    string i = to_string(r) + to_string(c);
    s += "  var d" + i + " = vec4<f32>(0.0); { let iy = 2 * ty - 1 + " + to_string(r) + "; let ix = 2 * tx - 1 + " + to_string(c) + ";\n"
         "    if (iy >= 0 && iy < " + to_string(H) + " && ix >= 0 && ix < " + to_string(W) + ") { d" + i + " = in0[(u32(iy) * " + U(W) + " + u32(ix)) * " + U(C4) + " + c4]; } }\n";
  }
  for (int c = 0; c < 4; c++) {  // B^T d, column c
    string q = to_string(c);
    s += "  let t0" + q + " = d0" + q + " - d2" + q + "; let t1" + q + " = d1" + q + " + d2" + q + "; let t2" + q + " = d2" + q + " - d1" + q + "; let t3" + q + " = d1" + q + " - d3" + q + ";\n";
  }
  for (int r = 0; r < 4; r++) {
    string q = to_string(r);
    string v[4] = {"t" + q + "0 - t" + q + "2", "t" + q + "1 + t" + q + "2", "t" + q + "2 - t" + q + "1", "t" + q + "1 - t" + q + "3"};
    for (int c = 0; c < 4; c++) s += "  outb[" + U(r * 4 + c) + " * " + U(T * C4) + " + idx] = " + v[c] + ";\n";
  }
  return s + "}\n";
}

static string wino_out_wgsl(uint32_t H, uint32_t W, uint32_t C4, uint32_t tW, uint32_t T) {
  string s = hdr(1);
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C4) + ") { return; }\n"
       "  let c4 = idx % " + U(C4) + "; let t = idx / " + U(C4) + "; let ty = t / " + U(tW) + "; let tx = t % " + U(tW) + ";\n";
  for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
    s += "  let m" + to_string(r) + to_string(c) + " = in0[" + U(r * 4 + c) + " * " + U(T * C4) + " + idx];\n";
  for (int c = 0; c < 4; c++) {  // A^T m
    string q = to_string(c);
    s += "  let s0" + q + " = m0" + q + " + m1" + q + " + m2" + q + "; let s1" + q + " = m1" + q + " - m2" + q + " - m3" + q + ";\n";
  }
  for (int r = 0; r < 2; r++) {
    string q = to_string(r);
    string y[2] = {"s" + q + "0 + s" + q + "1 + s" + q + "2", "s" + q + "1 - s" + q + "2 - s" + q + "3"};
    for (int c = 0; c < 2; c++)
      s += "  { let oy = ty * 2u + " + U(r) + "; let ox = tx * 2u + " + U(c) + "; if (oy < " + U(H) + " && ox < " + U(W) + ") { outb[(oy * " + U(W) + " + ox) * " + U(C4) + " + c4] = " + y[c] + "; } }\n";
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


// ---- f16 helpers (round to nearest even) ----
static uint16_t f2h(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff; int exp = ((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return (uint16_t)sign; if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  uint16_t h = (uint16_t)(sign | (exp << 10) | (mant >> 13)); if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) h++; return h; }
static wgpu::Buffer uploadRaw(const void* p, size_t n) { auto b = mk(n, ST); dev.GetQueue().WriteBuffer(b, 0, p, (n + 3) & ~size_t(3)); return b; }

// binding table helper: types[i] = "f32" or "u32" for each vec4 storage binding, the last one is read_write
static string bindings(const vector<string>& types) {
  string s;
  for (size_t i = 0; i < types.size(); i++) {
    bool last = i + 1 == types.size();
    s += "@group(0) @binding(" + to_string(i) + ") var<storage, " + (last ? "read_write" : "read") + "> " + (last ? "outb" : "in" + to_string(i)) + " : array<vec4<" + types[i] + ">>;\n";
  }
  return s;
}
// input transform, 8 channels per thread (2 vec4<f32> loads per position), V written as packed f16: outb[e*T*C8 + idx] (vec4<u32> = 8 halves)
static string wino_in_f16_wgsl(uint32_t H, uint32_t W, uint32_t C8, uint32_t tW, uint32_t T) {
  string s = bindings({"f32", "u32"});
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C8) + ") { return; }\n"
       "  let c8 = idx % " + U(C8) + "; let t = idx / " + U(C8) + "; let ty = i32(t / " + U(tW) + "); let tx = i32(t % " + U(tW) + ");\n";
  for (int h = 0; h < 2; h++) {  // h = 0: channels 0-3 of the 8, h = 1: channels 4-7
    string P = h ? "B" : "A";
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
      string i = P + to_string(r) + to_string(c);
      s += "  var d" + i + " = vec4<f32>(0.0); { let iy = 2 * ty - 1 + " + to_string(r) + "; let ix = 2 * tx - 1 + " + to_string(c) + ";\n"
           "    if (iy >= 0 && iy < " + to_string(H) + " && ix >= 0 && ix < " + to_string(W) + ") { d" + i + " = in0[(u32(iy) * " + U(W) + " + u32(ix)) * " + U(C8 * 2) + " + c8 * 2u + " + U(h) + "]; } }\n";
    }
    for (int c = 0; c < 4; c++) {
      string q = to_string(c);
      s += "  let t0" + P + q + " = d" + P + "0" + q + " - d" + P + "2" + q + "; let t1" + P + q + " = d" + P + "1" + q + " + d" + P + "2" + q + "; let t2" + P + q + " = d" + P + "2" + q + " - d" + P + "1" + q +
           "; let t3" + P + q + " = d" + P + "1" + q + " - d" + P + "3" + q + ";\n";
    }
    for (int r = 0; r < 4; r++) {
      string q = to_string(r);
      string v[4] = {"t" + q + P + "0 - t" + q + P + "2", "t" + q + P + "1 + t" + q + P + "2", "t" + q + P + "2 - t" + q + P + "1", "t" + q + P + "1 - t" + q + P + "3"};
      for (int c = 0; c < 4; c++) s += "  let v" + P + to_string(r * 4 + c) + " = " + v[c] + ";\n";
    }
  }
  for (int e = 0; e < 16; e++)
    s += "  outb[" + U(e * T * C8) + " + idx] = vec4<u32>(pack2x16float(vA" + to_string(e) + ".xy), pack2x16float(vA" + to_string(e) + ".zw), pack2x16float(vB" + to_string(e) + ".xy), pack2x16float(vB" + to_string(e) + ".zw));\n";
  return s + "}\n";
}

// batched GEMM z=e: M[e][T][N] = V[e][T][K] * U[e][K][N]; V,U packed f16 (vec4<u32> = 8 halves along K resp. N), f32 accumulators.
// Thread: TM rows x NV vec4 columns (NV even, = NV/2 packed loads of B per k). mf16: M written packed f16 (vec4<u32> = 8 columns).
static string gemm_p16_wgsl(uint32_t T, uint32_t N, uint32_t K, int TM, int NV, int WX, int WY, bool mf16) {
  uint32_t N8 = N / 8, K8 = K / 8, N4 = N / 4, NB = NV / 2;
  string s = bindings({"u32", "u32", mf16 ? "u32" : "f32"});
  s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
       "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
       "  let col8 = g.x * " + U(NB) + "; let row0 = g.y * " + U(TM) + "; let z = wid.z;\n";
  for (int n = 0; n < (int)NB; n++) s += "  let cb" + to_string(n) + " = min(col8 + " + U(n) + ", " + U(N8 - 1) + ");\n";
  for (int m = 0; m < TM; m++) for (int c = 0; c < 4 * NV; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
  s += "  for (var k8 = 0u; k8 < " + U(K8) + "; k8++) {\n";
  for (int m = 0; m < TM; m++) {
    s += "    let a" + to_string(m) + " = in0[z * " + U(T * K8) + " + min(row0 + " + U(m) + ", " + U(T - 1) + ") * " + U(K8) + " + k8];\n";
    for (int j = 0; j < 4; j++) s += "    let au" + to_string(m) + "_" + to_string(j) + " = unpack2x16float(a" + to_string(m) + "." + "xyzw"[j] + ");\n";
  }
  for (int kk = 0; kk < 8; kk++) {
    for (int n = 0; n < (int)NB; n++) {
      string b = "b" + to_string(kk) + "_" + to_string(n);
      s += "    let " + b + " = in1[z * " + U(K * N8) + " + (k8 * 8u + " + U(kk) + ") * " + U(N8) + " + cb" + to_string(n) + "];\n";
      for (int j = 0; j < 4; j++) s += "    let " + b + "u" + to_string(j) + " = unpack2x16float(" + b + "." + "xyzw"[j] + ");\n";
    }
    for (int m = 0; m < TM; m++) {
      string av = "au" + to_string(m) + "_" + to_string(kk / 2) + (kk % 2 ? ".y" : ".x");
      for (int n = 0; n < (int)NB; n++) for (int j = 0; j < 4; j++) for (int h = 0; h < 2; h++) {
        string acc = "c" + to_string(m) + "_" + to_string(n * 8 + j * 2 + h);
        s += "    " + acc + " = fma(" + av + ", b" + to_string(kk) + "_" + to_string(n) + "u" + to_string(j) + (h ? ".y" : ".x") + ", " + acc + ");\n";
      }
    }
  }
  s += "  }\n";
  for (int m = 0; m < TM; m++) {
    string cm = "c" + to_string(m) + "_";
    string row = "row0 + " + U(m);
    if (mf16) {
      for (int n = 0; n < (int)NB; n++) {
        string b = cm; int o = n * 8;
        s += "  if (" + row + " < " + U(T) + " && col8 + " + U(n) + " < " + U(N8) + ") { outb[z * " + U(T * N8) + " + (" + row + ") * " + U(N8) + " + col8 + " + U(n) + "] = vec4<u32>(" +
             "pack2x16float(vec2<f32>(" + b + to_string(o) + ", " + b + to_string(o + 1) + ")), pack2x16float(vec2<f32>(" + b + to_string(o + 2) + ", " + b + to_string(o + 3) + ")), " +
             "pack2x16float(vec2<f32>(" + b + to_string(o + 4) + ", " + b + to_string(o + 5) + ")), pack2x16float(vec2<f32>(" + b + to_string(o + 6) + ", " + b + to_string(o + 7) + "))); }\n";
      }
    } else {
      for (int n = 0; n < NV; n++)
        s += "  if (" + row + " < " + U(T) + " && col8 * 2u + " + U(n) + " < " + U(N4) + ") { outb[z * " + U(T * N4) + " + (" + row + ") * " + U(N4) + " + col8 * 2u + " + U(n) + "] = vec4<f32>(" +
             cm + to_string(n * 4) + ", " + cm + to_string(n * 4 + 1) + ", " + cm + to_string(n * 4 + 2) + ", " + cm + to_string(n * 4 + 3) + "); }\n";
    }
  }
  return s + "}\n";
}

// output transform reading M packed f16 (8 channels per thread), writing f32 NHWC
static string wino_out_f16_wgsl(uint32_t H, uint32_t W, uint32_t C8, uint32_t tW, uint32_t T) {
  string s = bindings({"u32", "f32"});
  s += "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) {\n"
       "  let idx = g.x; if (idx >= " + U(T * C8) + ") { return; }\n"
       "  let c8 = idx % " + U(C8) + "; let t = idx / " + U(C8) + "; let ty = t / " + U(tW) + "; let tx = t % " + U(tW) + ";\n";
  for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
    string i = to_string(r) + to_string(c);
    s += "  let w" + i + " = in0[" + U((r * 4 + c) * T * C8) + " + idx];\n"
         "  let mA" + i + " = vec4<f32>(unpack2x16float(w" + i + ".x), unpack2x16float(w" + i + ".y)); let mB" + i + " = vec4<f32>(unpack2x16float(w" + i + ".z), unpack2x16float(w" + i + ".w));\n";
  }
  for (int h = 0; h < 2; h++) {
    string P = h ? "B" : "A";
    for (int c = 0; c < 4; c++) {
      string q = to_string(c);
      s += "  let s0" + P + q + " = m" + P + "0" + q + " + m" + P + "1" + q + " + m" + P + "2" + q + "; let s1" + P + q + " = m" + P + "1" + q + " - m" + P + "2" + q + " - m" + P + "3" + q + ";\n";
    }
    for (int r = 0; r < 2; r++) {
      string q = to_string(r);
      string y[2] = {"s" + q + P + "0 + s" + q + P + "1 + s" + q + P + "2", "s" + q + P + "1 - s" + q + P + "2 - s" + q + P + "3"};
      for (int c = 0; c < 2; c++)
        s += "  { let oy = ty * 2u + " + U(r) + "; let ox = tx * 2u + " + U(c) + "; if (oy < " + U(H) + " && ox < " + U(W) + ") { outb[(oy * " + U(W) + " + ox) * " + U(C8 * 2) + " + c8 * 2u + " + U(h) + "] = " + y[c] + "; } }\n";
    }
  }
  return s + "}\n";
}

struct Cfg { int TM, NV, WX, WY; };
static const Cfg F32CFGS[] = {{8, 2, 16, 4}, {4, 2, 32, 4}, {4, 2, 16, 4}, {4, 1, 16, 8}};
static const Cfg P16CFGS[] = {{4, 2, 16, 4}, {4, 2, 32, 2}, {4, 2, 8, 8}, {2, 2, 16, 8}, {8, 2, 16, 4}, {4, 4, 16, 4}, {2, 4, 16, 8}, {4, 2, 64, 1}};

static int runConv(uint32_t C, uint32_t H, bool relu) {
  uint32_t W = H, M = H * W, C4 = C / 4, C8 = C / 8, tH = (H + 1) / 2, tW = (W + 1) / 2, T = tH * tW;
  vector<float> X((size_t)M * C), Wt((size_t)C * C * 9);
  for (size_t i = 0; i < X.size(); i++) { float v = ((i * 7919) % 1000) / 1000.f - .5f; X[i] = relu ? std::max(0.f, v) * 2.f : v; }
  for (size_t i = 0; i < Wt.size(); i++) Wt[i] = (((i * 104729) % 1000) / 1000.f - .5f) * 0.1f;
  const double G[4][3] = {{1, 0, 0}, {.5, .5, .5}, {.5, -.5, .5}, {0, 0, 1}};
  vector<float> Uw((size_t)16 * C * C);
  for (uint32_t co = 0; co < C; co++) for (uint32_t ci = 0; ci < C; ci++) {
    const float* g = &Wt[((size_t)co * C + ci) * 9];
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) { double a = 0; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) a += G[r][i] * g[i * 3 + j] * G[c][j]; Uw[((size_t)(r * 4 + c) * C + ci) * C + co] = a; }
  }
  vector<uint16_t> Uh(Uw.size()); for (size_t i = 0; i < Uw.size(); i++) Uh[i] = f2h(Uw[i]);
  auto bX = upload(X), bU = upload(Uw), bUh = uploadRaw(Uh.data(), Uh.size() * 2);
  auto bY = mk((size_t)M * C * 4, ST), bV = mk((size_t)16 * T * C * 4, ST), bM = mk((size_t)16 * T * C * 4, ST), bVh = mk((size_t)16 * T * C * 2, ST), bMh = mk((size_t)16 * T * C * 2, ST);
  vector<size_t> pr, pc; vector<double> ref;
  for (int q = 0; q < 300; q++) {
    size_t p = (q * 2654435761u) % M, co = (q * 40503u + 7) % C; int y = p / W, x = p % W; double a = 0;
    for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) { int iy = y + ky - 1, ix = x + kx - 1; if (iy < 0 || iy >= (int)H || ix < 0 || ix >= (int)W) continue;
      for (uint32_t ci = 0; ci < C; ci++) a += (double)X[((size_t)iy * W + ix) * C + ci] * Wt[((size_t)co * C + ci) * 9 + ky * 3 + kx]; }
    pr.push_back(p); pc.push_back(co); ref.push_back(a);
  }
  auto check = [&]() { auto Y = download(bY, (size_t)M * C); double me = 0, mr = 0; for (size_t i = 0; i < ref.size(); i++) { me = std::max(me, fabs(ref[i] - Y[pr[i] * C + pc[i]])); mr = std::max(mr, fabs(ref[i])); } return me / mr; };
  double flops = 2.0 * M * C * C * 9;
  printf("== winograd f16 intermediates, conv 3x3 C=%u H=W=%u %s (%.3f GFLOP, T=%u tiles)\n", C, H, relu ? "relu-like input" : "zero-mean input", flops / 1e9, T);
  auto gdim = [&](const Cfg& c, uint32_t nb8, uint32_t& gx, uint32_t& gy, bool p16) { uint32_t cols = p16 ? (nb8 + c.NV / 2 - 1) / (c.NV / 2) : (C4 + c.NV - 1) / c.NV; gx = (cols + c.WX - 1) / c.WX; gy = (T + c.TM * c.WY - 1) / (c.TM * c.WY); };
  // ---- f32 pipeline
  Kern kin = make(wino_in_wgsl(H, W, C4, tW, T), {bX, bV}, (T * C4 + 63) / 64);
  Kern kout = make(wino_out_wgsl(H, W, C4, tW, T), {bM, bY}, (T * C4 + 63) / 64);
  double tin = timeSeq({kin}, 10), tout = timeSeq({kout}, 10), bestG = 1e30; Cfg bg{};
  for (auto c : F32CFGS) { uint32_t gx, gy; gdim(c, 0, gx, gy, false);
    Kern k = make(gemm_wgsl(false, T, C, C, 0, 0, 0, c.TM, c.NV, c.WX, c.WY), {bV, bU, bM}, gx, gy, 16); double ms = timeSeq({k}, 10); if (ms < bestG) { bestG = ms; bg = c; } }
  double t32; { uint32_t gx, gy; gdim(bg, 0, gx, gy, false); Kern kg = make(gemm_wgsl(false, T, C, C, 0, 0, 0, bg.TM, bg.NV, bg.WX, bg.WY), {bV, bU, bM}, gx, gy, 16);
    t32 = timeSeq({kin, kg, kout}, 10); printf("f32      total %.3f ms  relerr=%.1e  | in %.3f + gemm(TM=%d NV=%d wg=%dx%d) %.3f + out %.3f\n", t32, check(), tin, bg.TM, bg.NV, bg.WX, bg.WY, bestG, tout); }
  // ---- f16 pipelines
  Kern kinh = make(wino_in_f16_wgsl(H, W, C8, tW, T), {bX, bVh}, (T * C8 + 63) / 64);
  double tinh = timeSeq({kinh}, 10);
  for (int mf16 = 0; mf16 < 2; mf16++) {
    wgpu::Buffer bMo = mf16 ? bMh : bM;
    Kern kouth = mf16 ? make(wino_out_f16_wgsl(H, W, C8, tW, T), {bMh, bY}, (T * C8 + 63) / 64) : kout;
    double tout16 = timeSeq({kouth}, 10), best = 1e30; Cfg b{};
    for (auto c : P16CFGS) { uint32_t gx, gy; gdim(c, C8, gx, gy, true);
      Kern k = make(gemm_p16_wgsl(T, C, C, c.TM, c.NV, c.WX, c.WY, mf16), {bVh, bUh, bMo}, gx, gy, 16); double ms = timeSeq({k}, 10); if (ms < best) { best = ms; b = c; } }
    uint32_t gx, gy; gdim(b, C8, gx, gy, true);
    Kern kg = make(gemm_p16_wgsl(T, C, C, b.TM, b.NV, b.WX, b.WY, mf16), {bVh, bUh, bMo}, gx, gy, 16);
    double tt = timeSeq({kinh, kg, kouth}, 10);
    printf("f16-M%s total %.3f ms  relerr=%.1e  speedup vs f32 %.2fx | in %.3f + gemm(TM=%d NV=%d wg=%dx%d) %.3f + out %.3f\n", mf16 ? "16" : "32", tt, check(), t32 / tt, tinh, b.TM, b.NV, b.WX, b.WY, best, tout16);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: conv_alt_f16 C H [relu]\n"); return 1; }
  initDevice();
  return runConv(atoi(argv[1]), atoi(argv[2]), argc > 3 && !strcmp(argv[3], "relu"));
}
