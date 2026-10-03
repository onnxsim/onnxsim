// Memory-layout study for conv / GEMM on the Adreno 730 through Dawn/Vulkan (WGSL), storage buffers and textures.
//   layouts conv KH Cin Cout H     -- KHxKH stride-1 (pad KH/2) conv on HxH NHWC-equivalent data, batch 1, swept over layouts
//   layouts pad                    -- channel padding (multiples of 4/8/16/32) on a 1x1 conv, useful GFLOPS
//   layouts xform                  -- NHWC <-> blocked [C4][H][W] activation transform kernels
// One register-tile implicit-GEMM generator (scalar accumulators, TMxNV vec4 outputs per thread, no shared memory, like
// gemm.cc "sc") parameterised by:
//   activation layout a: nhwc | nc4 ([C/4][HW] vec4) | z4 (4x4-pixel tiles, thread order follows the tiles) | texa (texture x=c4,
//                        y=pixel) | texb (texture x=ix*C4+c4, y=iy) | nhwc16 | nc416 (packed f16, 8 channels per vec4<u32>) |
//                        texa16 | texb16 (RGBA16Float textures)
//   weight layout w:     hwio ([K][N4] vec4 over n) | ok4 ([N4][K], contiguous over k) | nk4 ([N][K4], vec4 over k, dot()) |
//                        texw | texw16 (textures x=n4, y=k) | w16 (packed f16 over n, 8 outputs per vec4<u32>)
//   output layout o:     follows a (nc4 -> nc4, z4 -> z4, everything else nhwc)
//   thread map:          colx (x = output-channel group, y = pixel tile) | rowx (x = pixels spaced apart for coalescing)
// Correctness is checked on 200 sampled outputs against a double-precision CPU reference (computed from the f16-rounded inputs
// for the f16 layouts). Timing: >= 5 s GPU warm-up, then min/median over 12 batched submissions. Build like gemm.cc (build.sh).
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
using std::string; using std::to_string; using std::vector; using std::map;

static wgpu::Instance inst; static wgpu::Device dev;
static string U(uint32_t v) { return to_string(v) + "u"; }
static double nowms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint16_t f2h(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff; int exp = ((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return (uint16_t)sign; if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  uint16_t h = (uint16_t)(sign | (exp << 10) | (mant >> 13)); if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) h++; return h; }
static float h2f(uint16_t h) { uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023, x; if (e == 0) { if (!m) x = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; x = s | ((e + 112) << 23) | (m << 13); } } else x = s | ((e + 112) << 23) | (m << 13); float f; memcpy(&f, &x, 4); return f; }
static float rnd16(float f) { return h2f(f2h(f)); }

static void initDevice() {
  dawnProcSetProcs(&dawn::native::GetProcs());
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  vector<const char*> ad_en = {"use_vulkan_memory_model"}, dev_en = {"disable_robustness"};  // like ORT
  wgpu::DawnTogglesDescriptor adt{}; adt.enabledToggleCount = ad_en.size(); adt.enabledToggles = ad_en.data(); ro.nextInChain = &adt;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data();
  wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt;
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
}

static wgpu::Buffer mk(size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = (n + 15) & ~size_t(15); b.usage = u; return dev.CreateBuffer(&b); }
static const auto ST = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
static wgpu::Buffer upload(const void* p, size_t bytes) { auto b = mk(bytes, ST); dev.GetQueue().WriteBuffer(b, 0, p, (bytes + 3) & ~size_t(3)); return b; }
static vector<float> download(wgpu::Buffer src, size_t n) {
  wgpu::Buffer rb = mk(n * 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  { wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); enc.CopyBufferToBuffer(src, 0, rb, 0, (n * 4 + 3) & ~size_t(3)); wgpu::CommandBuffer cb = enc.Finish(); dev.GetQueue().Submit(1, &cb); }
  bool ok = false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read, 0, rb.GetSize(), wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::MapAsyncStatus st, wgpu::StringView) { ok = st == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
  vector<float> out(n); if (ok) memcpy(out.data(), rb.GetConstMappedRange(0, rb.GetSize()), n * 4);
  return out;
}
// RGBA texture from raw texel data (bytesPerTexel 16 for RGBA32Float, 8 for RGBA16Float)
static wgpu::Texture mkTex(uint32_t w, uint32_t h, bool half, const void* data) {
  wgpu::TextureDescriptor td{}; td.size = {w, h, 1}; td.format = half ? wgpu::TextureFormat::RGBA16Float : wgpu::TextureFormat::RGBA32Float;
  td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
  wgpu::Texture t = dev.CreateTexture(&td);
  uint32_t bpt = half ? 8 : 16;
  wgpu::TexelCopyTextureInfo di{}; di.texture = t; wgpu::TexelCopyBufferLayout lay{}; lay.bytesPerRow = w * bpt; lay.rowsPerImage = h; wgpu::Extent3D ext = {w, h, 1};
  dev.GetQueue().WriteTexture(&di, data, (size_t)w * h * bpt, &lay, &ext);
  return t;
}

struct Res { wgpu::Buffer buf; wgpu::Texture tex; };
struct Kern { wgpu::ComputePipeline pl; wgpu::BindGroup bg; uint32_t gx, gy, gz; };
static Kern make(const string& wgsl, const vector<Res>& rs, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
  Kern k; k.gx = gx; k.gy = gy; k.gz = gz;
  wgpu::ShaderSourceWGSL w{}; w.code = {wgsl.data(), wgsl.size()};
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  k.pl = dev.CreateComputePipeline(&cpd);
  vector<wgpu::BindGroupEntry> e(rs.size());
  for (size_t i = 0; i < rs.size(); i++) { e[i].binding = i; if (rs[i].tex) e[i].textureView = rs[i].tex.CreateView(); else { e[i].buffer = rs[i].buf; e[i].size = rs[i].buf.GetSize(); } }
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = k.pl.GetBindGroupLayout(0); bgd.entryCount = e.size(); bgd.entries = e.data();
  k.bg = dev.CreateBindGroup(&bgd);
  return k;
}
static double submit(const Kern& k, int iters) {
  wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
  for (int i = 0; i < iters; i++) { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(k.pl); p.SetBindGroup(0, k.bg); p.DispatchWorkgroups(k.gx, k.gy, k.gz); p.End(); }
  wgpu::CommandBuffer cb = enc.Finish();
  double t0 = nowms();
  dev.GetQueue().Submit(1, &cb);
  inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
  return (nowms() - t0) / iters;
}
struct Timing { double mn, med; };
static Timing timeKern(const Kern& k) {
  double t1 = 1e9; for (int i = 0; i < 3; i++) t1 = std::min(t1, submit(k, 1));
  int iters = std::max(1, std::min(200, (int)(20.0 / std::max(t1, 0.01))));
  if (t1 > 20) iters = 1;
  vector<double> v; for (int i = 0; i < 3; i++) submit(k, iters);
  for (int i = 0; i < 12; i++) v.push_back(submit(k, iters));
  std::sort(v.begin(), v.end()); return {v[0], v[v.size() / 2]};
}
static void burn(const Kern& k, double seconds) { double t0 = nowms(); while (nowms() - t0 < seconds * 1000) submit(k, 20); }

// ---------------------------------------------------------------------------------------------------------------------
struct Prob { int kh; uint32_t Cin, Cout, H, W; uint32_t M() const { return H * W; } uint32_t K() const { return kh * kh * Cin; } };
struct Lay { string a, w; };
struct Cfg { bool rowx; int TM, NV, WX, WY; };
static bool aIsF16Buf(const string& a) { return a == "nhwc16" || a == "nc416"; }
static bool aIsTex(const string& a) { return a.rfind("tex", 0) == 0; }
static bool aRound(const string& a) { return a == "nhwc16" || a == "nc416" || a == "texa16" || a == "texb16"; }
static bool wRound(const string& w) { return w == "w16" || w == "texw16"; }
static string oLay(const string& a) { return a == "nc4" || a == "nc416" ? "nc4" : a == "z4" ? "z4" : "nhwc"; }
static uint32_t tiled(uint32_t y, uint32_t x, uint32_t W) { return ((y >> 2) * (W >> 2) + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3); }

static bool valid(const Prob& P, const Lay& L, const Cfg& c) {
  if (c.WX * c.WY > 256) return false;
  if (L.a == "z4" && ((P.H | P.W) & 3)) return false;
  if (L.w == "w16" && c.NV != 2) return false;
  if (L.w == "nk4" && (aIsF16Buf(L.a) || P.kh > 1)) return false;
  if (aIsF16Buf(L.a) && P.Cin % 8) return false;
  if (L.w == "w16" && P.Cout % 8) return false;
  return true;
}

static string genKernel(const Prob& P, const Lay& L, const Cfg& c, uint32_t& gx, uint32_t& gy) {
  const uint32_t M = P.M(), N = P.Cout, N4 = N / 4, K = P.K(), Cin = P.Cin, W = P.W, H = P.H, HW = M;
  const bool f16a = aIsF16Buf(L.a), tex = aIsTex(L.a);
  const int KS = f16a ? 8 : 4;
  const uint32_t CB = Cin / 4;  // vec4 channel blocks for f32 storage layouts
  const string ol = oLay(L.a);
  const int TM = c.TM, NV = c.NV;
  string s;
  if (tex) s += "@group(0) @binding(0) var t0 : texture_2d<f32>;\n";
  else s += string("@group(0) @binding(0) var<storage, read> in0 : array<vec4<") + (f16a ? "u32" : "f32") + ">>;\n";
  if (L.w == "texw" || L.w == "texw16") s += "@group(0) @binding(1) var t1 : texture_2d<f32>;\n";
  else s += string("@group(0) @binding(1) var<storage, read> in1 : array<vec4<") + (L.w == "w16" ? "u32" : "f32") + ">>;\n";
  s += "@group(0) @binding(2) var<storage, read_write> outb : array<vec4<f32>>;\n";
  s += "@compute @workgroup_size(" + to_string(c.WX) + "," + to_string(c.WY) + ",1)\n"
       "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {\n";
  if (!c.rowx) {
    s += "  let col4 = g.x * " + U(NV) + "; let row0 = g.y * " + U(TM) + ";\n";
    for (int m = 0; m < TM; m++) s += "  let pr" + to_string(m) + " = row0 + " + U(m) + ";\n";
    gx = (((N4 + NV - 1) / NV) + c.WX - 1) / c.WX; gy = (M + TM * c.WY - 1) / (TM * c.WY);
  } else {
    s += "  let col4 = (wid.y * " + U(c.WY) + " + lid.y) * " + U(NV) + ";\n";
    for (int m = 0; m < TM; m++) s += "  let pr" + to_string(m) + " = wid.x * " + U(c.WX * TM) + " + " + U(m * c.WX) + " + lid.x;\n";
    gx = (M + c.WX * TM - 1) / (c.WX * TM); gy = (((N4 + NV - 1) / NV) + c.WY - 1) / c.WY;
  }
  for (int m = 0; m < TM; m++) s += "  let pv" + to_string(m) + " = pr" + to_string(m) + " < " + U(M) + "; let p" + to_string(m) + " = min(pr" + to_string(m) + ", " + U(M - 1) + ");\n";
  for (int n = 0; n < NV; n++) s += "  let cb" + to_string(n) + " = min(col4 + " + U(n) + ", " + U(N4 - 1) + ");\n";
  for (int m = 0; m < TM; m++) for (int q = 0; q < 4 * NV; q++) s += "  var c" + to_string(m) + "_" + to_string(q) + " = 0.0;\n";
  const bool needXY = P.kh > 1 || L.a.rfind("texb", 0) == 0;
  s += "  for (var tap = 0u; tap < " + U(P.kh * P.kh) + "; tap++) {\n";
  if (P.kh > 1) s += "    let dy = i32(tap / " + U(P.kh) + ") - " + to_string(P.kh / 2) + "; let dx = i32(tap % " + U(P.kh) + ") - " + to_string(P.kh / 2) + ";\n";
  else s += "    let dy = 0; let dx = 0;\n";
  for (int m = 0; m < TM; m++) {
    string i = to_string(m);
    if (needXY) {
      if (L.a == "z4") s += "    let y" + i + " = i32((p" + i + " / 16u / " + U(W / 4) + ") * 4u + ((p" + i + " % 16u) / 4u)); let x" + i + " = i32(((p" + i + " / 16u) % " + U(W / 4) + ") * 4u + (p" + i + " % 4u));\n";
      else s += "    let y" + i + " = i32(p" + i + " / " + U(W) + "); let x" + i + " = i32(p" + i + " % " + U(W) + ");\n";
      s += "    let iy" + i + " = y" + i + " + dy; let ix" + i + " = x" + i + " + dx;\n"
           "    let v" + i + " = iy" + i + " >= 0 && iy" + i + " < " + to_string(H) + " && ix" + i + " >= 0 && ix" + i + " < " + to_string(W) + ";\n";
    } else s += "    let v" + i + " = true;\n";
    // storage pixel index of the source pixel
    if (P.kh > 1) {
      if (L.a == "z4") s += "    let sp" + i + " = select(0u, ((u32(iy" + i + ") >> 2u) * " + U(W / 4) + " + (u32(ix" + i + ") >> 2u)) * 16u + (u32(iy" + i + ") & 3u) * 4u + (u32(ix" + i + ") & 3u), v" + i + ");\n";
      else s += "    let sp" + i + " = select(0u, u32(iy" + i + " * " + to_string(W) + " + ix" + i + "), v" + i + ");\n";
    } else s += "    let sp" + i + " = p" + i + ";\n";
    if (L.a.rfind("texb", 0) == 0) s += "    let tx" + i + " = select(0, ix" + i + ", v" + i + "); let ty" + i + " = select(0, iy" + i + ", v" + i + ");\n";
  }
  const int NB = Cin / KS;
  s += "    for (var c = 0u; c < " + U(NB) + "; c++) {\n";
  for (int m = 0; m < TM; m++) {
    string i = to_string(m), load;
    if (L.a == "nhwc" || L.a == "z4") load = "in0[sp" + i + " * " + U(CB) + " + c]";
    else if (L.a == "nc4") load = "in0[c * " + U(HW) + " + sp" + i + "]";
    else if (L.a == "nhwc16") load = "in0[sp" + i + " * " + U(Cin / 8) + " + c]";
    else if (L.a == "nc416") load = "in0[c * " + U(HW) + " + sp" + i + "]";
    else if (L.a == "texa" || L.a == "texa16") load = "textureLoad(t0, vec2<i32>(i32(c), i32(sp" + i + ")), 0)";
    else load = "textureLoad(t0, vec2<i32>(tx" + i + " * " + to_string(CB) + " + i32(c), ty" + i + "), 0)";
    if (f16a) {
      s += "      let r" + i + " = " + load + ";\n";
      s += "      let a" + i + "_0 = select(vec4<f32>(0.0), vec4<f32>(unpack2x16float(r" + i + ".x), unpack2x16float(r" + i + ".y)), v" + i + ");\n";
      s += "      let a" + i + "_1 = select(vec4<f32>(0.0), vec4<f32>(unpack2x16float(r" + i + ".z), unpack2x16float(r" + i + ".w)), v" + i + ");\n";
    } else s += "      let a" + i + " = select(vec4<f32>(0.0), " + load + ", v" + i + ");\n";
  }
  auto acc = [&](int m, int q) { return "c" + to_string(m) + "_" + to_string(q); };
  auto areg = [&](int m, int j) {  // scalar j (0..KS-1) of row m
    string v = f16a ? "a" + to_string(m) + "_" + to_string(j / 4) : "a" + to_string(m);
    return v + "." + "xyzw"[j % 4];
  };
  if (L.w == "nk4") {
    // dot layout: weights [N][K4] vec4 over k; one vec4 load per output channel and k-block, dot() accumulation
    for (int n = 0; n < NV; n++) for (int q = 0; q < 4; q++)
      s += "      let w" + to_string(n * 4 + q) + " = in1[min(cb" + to_string(n) + " * 4u + " + U(q) + ", " + U(N - 1) + ") * " + U(K / 4) + " + tap * " + U(CB) + " + c];\n";
    for (int m = 0; m < TM; m++) for (int q = 0; q < 4 * NV; q++) s += "      " + acc(m, q) + " += dot(a" + to_string(m) + ", w" + to_string(q) + ");\n";
  } else {
    for (int j = 0; j < KS; j++) {
      string kexpr = "(tap * " + U(Cin) + " + c * " + U(KS) + " + " + U(j) + ")";
      string bj = "b" + to_string(j);
      if (L.w == "w16") {
        for (int h = 0; h < NV / 2; h++) {
          s += "      let " + bj + "_w" + to_string(h) + " = in1[" + kexpr + " * " + U(N / 8) + " + min(col4 / 2u + " + U(h) + ", " + U(N / 8 - 1) + ")];\n";
          for (int q = 0; q < 4; q++) s += "      let " + bj + "_" + to_string(h * 2 + q / 2) + "_" + to_string(q % 2) + " = unpack2x16float(" + bj + "_w" + to_string(h) + "." + "xyzw"[q] + ");\n";
        }
        for (int n = 0; n < NV; n++) s += "      let " + bj + "_" + to_string(n) + " = vec4<f32>(" + bj + "_" + to_string(n) + "_0, " + bj + "_" + to_string(n) + "_1);\n";
      } else {
        for (int n = 0; n < NV; n++) {
          string ld;
          if (L.w == "hwio") ld = "in1[" + kexpr + " * " + U(N4) + " + cb" + to_string(n) + "]";
          else if (L.w == "ok4") ld = "in1[cb" + to_string(n) + " * " + U(K) + " + " + kexpr + "]";
          else ld = "textureLoad(t1, vec2<i32>(i32(cb" + to_string(n) + "), i32(" + kexpr + ")), 0)";
          s += "      let " + bj + "_" + to_string(n) + " = " + ld + ";\n";
        }
      }
      for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) for (int q = 0; q < 4; q++)
        s += "      " + acc(m, n * 4 + q) + " = fma(" + areg(m, j) + ", " + bj + "_" + to_string(n) + "." + "xyzw"[q] + ", " + acc(m, n * 4 + q) + ");\n";
    }
  }
  s += "    }\n  }\n";
  for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) {
    string idx;
    if (ol == "nc4") idx = "(col4 + " + U(n) + ") * " + U(HW) + " + pr" + to_string(m);
    else idx = "pr" + to_string(m) + " * " + U(N4) + " + col4 + " + U(n);
    s += "  if (pv" + to_string(m) + " && col4 + " + U(n) + " < " + U(N4) + ") { outb[" + idx + "] = vec4<f32>(" + acc(m, n * 4) + ", " + acc(m, n * 4 + 1) + ", " + acc(m, n * 4 + 2) + ", " + acc(m, n * 4 + 3) + "); }\n";
  }
  return s + "}\n";
}

// ---------------------------------------------------------------------------------------------------------------------
struct Data {
  Prob P; vector<float> X, Wt;  // X[p][cin], Wt[k][n] with k = tap*Cin + cin
  vector<float> Xr, Wr;         // f16-rounded copies
  vector<int> sp; vector<double> ref[4]; vector<uint32_t> sample_p, sample_n; bool have_ref[4] = {false, false, false, false};
};
static void buildData(Data& d, const Prob& P) {
  d.P = P; d.X.resize((size_t)P.M() * P.Cin); d.Wt.resize((size_t)P.K() * P.Cout);
  for (size_t i = 0; i < d.X.size(); i++) d.X[i] = ((i * 7919) % 1000) / 1000.f - .5f;
  float ws = P.kh == 3 ? 0.1f : 0.2f;
  for (size_t i = 0; i < d.Wt.size(); i++) d.Wt[i] = (((i * 104729) % 1000) / 1000.f - .5f) * ws;
  d.Xr = d.X; d.Wr = d.Wt; for (auto& v : d.Xr) v = rnd16(v); for (auto& v : d.Wr) v = rnd16(v);
  for (int q = 0; q < 200; q++) { d.sample_p.push_back((q * 2654435761u) % P.M()); d.sample_n.push_back((q * 40503u + 7) % P.Cout); }
}
static const vector<double>& getRef(Data& d, bool ar, bool wr) {
  int key = (ar ? 1 : 0) + (wr ? 2 : 0);
  if (d.have_ref[key]) return d.ref[key];
  const Prob& P = d.P; const auto& X = ar ? d.Xr : d.X; const auto& Wt = wr ? d.Wr : d.Wt;
  for (size_t q = 0; q < d.sample_p.size(); q++) {
    uint32_t p = d.sample_p[q], n = d.sample_n[q]; int y = p / P.W, x = p % P.W; double a = 0;
    for (int ky = 0; ky < P.kh; ky++) for (int kx = 0; kx < P.kh; kx++) {
      int iy = y + ky - P.kh / 2, ix = x + kx - P.kh / 2; if (iy < 0 || iy >= (int)P.H || ix < 0 || ix >= (int)P.W) continue;
      for (uint32_t ci = 0; ci < P.Cin; ci++) a += (double)X[((size_t)iy * P.W + ix) * P.Cin + ci] * Wt[((size_t)(ky * P.kh + kx) * P.Cin + ci) * P.Cout + n];
    }
    d.ref[key].push_back(a);
  }
  d.have_ref[key] = true; return d.ref[key];
}

struct Inputs { Res a, w; };
static Inputs buildInputs(const Data& d, const Lay& L) {
  const Prob& P = d.P; const uint32_t HW = P.M(), Cin = P.Cin, N = P.Cout, K = P.K(), C4 = Cin / 4, C8 = Cin / 8, N4 = N / 4, N8 = N / 8;
  const vector<float>& X = aRound(L.a) ? d.Xr : d.X; const vector<float>& Wk = wRound(L.w) ? d.Wr : d.Wt;
  auto sp = [&](uint32_t p) { return L.a == "z4" ? tiled(p / P.W, p % P.W, P.W) : p; };
  Inputs in;
  vector<float> f; vector<uint32_t> u; vector<uint16_t> h;
  if (L.a == "nhwc" || L.a == "z4" || L.a == "texa" || L.a == "texb") {
    f.assign((size_t)HW * Cin, 0); for (uint32_t p = 0; p < HW; p++) for (uint32_t c = 0; c < Cin; c++) f[(size_t)sp(p) * Cin + c] = X[(size_t)p * Cin + c];
    if (L.a == "texa") in.a.tex = mkTex(C4, HW, false, f.data()); else if (L.a == "texb") in.a.tex = mkTex(P.W * C4, P.H, false, f.data()); else in.a.buf = upload(f.data(), f.size() * 4);
  } else if (L.a == "nc4") {
    f.assign((size_t)HW * Cin, 0); for (uint32_t p = 0; p < HW; p++) for (uint32_t c = 0; c < Cin; c++) f[(((size_t)(c / 4) * HW + p) * 4) + c % 4] = X[(size_t)p * Cin + c];
    in.a.buf = upload(f.data(), f.size() * 4);
  } else if (L.a == "nhwc16" || L.a == "nc416") {
    u.assign((size_t)HW * C8 * 4, 0);
    for (uint32_t p = 0; p < HW; p++) for (uint32_t c8 = 0; c8 < C8; c8++) for (int q = 0; q < 4; q++) {
      uint32_t lo = f2h(X[(size_t)p * Cin + c8 * 8 + 2 * q]), hi = f2h(X[(size_t)p * Cin + c8 * 8 + 2 * q + 1]);
      size_t idx = L.a == "nhwc16" ? ((size_t)p * C8 + c8) * 4 + q : ((size_t)c8 * HW + p) * 4 + q; u[idx] = lo | (hi << 16);
    }
    in.a.buf = upload(u.data(), u.size() * 4);
  } else {  // texa16 / texb16
    h.assign((size_t)HW * Cin, 0); for (uint32_t p = 0; p < HW; p++) for (uint32_t c = 0; c < Cin; c++) h[(size_t)p * Cin + c] = f2h(X[(size_t)p * Cin + c]);
    in.a.tex = L.a == "texa16" ? mkTex(C4, HW, true, h.data()) : mkTex(P.W * C4, P.H, true, h.data());
  }
  if (L.w == "hwio") in.w.buf = upload(Wk.data(), Wk.size() * 4);
  else if (L.w == "ok4") { f.assign((size_t)K * N, 0); for (uint32_t k = 0; k < K; k++) for (uint32_t n = 0; n < N; n++) f[(((size_t)(n / 4) * K + k) * 4) + n % 4] = Wk[(size_t)k * N + n]; in.w.buf = upload(f.data(), f.size() * 4); }
  else if (L.w == "nk4") { f.assign((size_t)K * N, 0); for (uint32_t k = 0; k < K; k++) for (uint32_t n = 0; n < N; n++) f[(((size_t)n * (K / 4) + k / 4) * 4) + k % 4] = Wk[(size_t)k * N + n]; in.w.buf = upload(f.data(), f.size() * 4); }
  else if (L.w == "texw") in.w.tex = mkTex(N4, K, false, Wk.data());
  else if (L.w == "texw16") { h.assign((size_t)K * N, 0); for (size_t i = 0; i < h.size(); i++) h[i] = f2h(Wk[i]); in.w.tex = mkTex(N4, K, true, h.data()); }
  else { u.assign((size_t)K * N8 * 4, 0); for (uint32_t k = 0; k < K; k++) for (uint32_t n8 = 0; n8 < N8; n8++) for (int q = 0; q < 4; q++) u[((size_t)k * N8 + n8) * 4 + q] = f2h(Wk[(size_t)k * N + n8 * 8 + 2 * q]) | ((uint32_t)f2h(Wk[(size_t)k * N + n8 * 8 + 2 * q + 1]) << 16); in.w.buf = upload(u.data(), u.size() * 4); }
  return in;
}

struct Result { Lay L; Cfg c; Timing t; double gf, err; };
static double checkOut(Data& d, const Lay& L, wgpu::Buffer out) {
  const Prob& P = d.P; auto Y = download(out, (size_t)P.M() * P.Cout);
  const auto& ref = getRef(d, aRound(L.a), wRound(L.w)); string ol = oLay(L.a); double me = 0, mr = 1e-30;
  for (size_t q = 0; q < ref.size(); q++) {
    uint32_t p = d.sample_p[q], n = d.sample_n[q]; uint32_t s = ol == "z4" ? tiled(p / P.W, p % P.W, P.W) : p; size_t idx;
    if (ol == "nc4") idx = ((size_t)(n / 4) * P.M() + s) * 4 + n % 4; else idx = ((size_t)s * (P.Cout / 4) + n / 4) * 4 + n % 4;
    me = std::max(me, fabs(ref[q] - Y[idx])); mr = std::max(mr, fabs(ref[q]));
  }
  return me / mr;
}

static const vector<Cfg> CFGS = {{false, 8, 2, 16, 4}, {false, 4, 2, 16, 4}, {false, 4, 4, 8, 8}, {true, 8, 2, 32, 4}, {true, 4, 4, 32, 2}};
static string cfgStr(const Cfg& c) { char b[64]; snprintf(b, sizeof b, "%s TM%d NV%d %dx%d", c.rowx ? "rowx" : "colx", c.TM, c.NV, c.WX, c.WY); return b; }

static bool runOne(Data& d, const Inputs& in, wgpu::Buffer out, const Lay& L, const Cfg& c, Result& r, bool quiet = false) {
  if (!valid(d.P, L, c)) return false;
  uint32_t gx, gy; string src = genKernel(d.P, L, c, gx, gy);
  double tc0 = nowms();
  Kern k = make(src, {in.a, in.w, Res{out, {}}}, gx, gy); submit(k, 1);
  double tcomp = nowms() - tc0; (void)tcomp;
  r.L = L; r.c = c; r.t = timeKern(k);
  r.gf = 2.0 * d.P.M() * d.P.Cout * d.P.K() / r.t.mn / 1e6; r.err = checkOut(d, L, out);
  if (!quiet) printf("  a=%-7s w=%-7s o=%-4s %-22s min=%.3f ms med=%.3f ms  %5.0f GFLOPS(min) err=%.1e%s\n", L.a.c_str(), L.w.c_str(), oLay(L.a).c_str(), cfgStr(c).c_str(), r.t.mn, r.t.med, r.gf, r.err, r.err > (aRound(L.a) || wRound(L.w) ? 1e-3 : 1e-4) ? "  WRONG" : "");
  fflush(stdout); return true;
}


// phase 4: interleaved head-to-head of the best configuration of each of the top layouts. The GPU clock drifts during a long
// sweep (and other jobs share the phone), so only kernels timed in alternation, several rounds, are comparable.
static void headToHead(Data& d, wgpu::Buffer out, vector<Result> cand, int rounds = 6) {
  vector<Inputs> ins; vector<Kern> ks; vector<vector<double>> mins(cand.size());
  for (auto& r : cand) { ins.push_back(buildInputs(d, r.L)); uint32_t gx, gy; ks.push_back(make(genKernel(d.P, r.L, r.c, gx, gy), {ins.back().a, ins.back().w, Res{out, {}}}, gx, gy)); }
  burn(ks[0], 3.0);
  for (int round = 0; round < rounds; round++) for (size_t i = 0; i < ks.size(); i++) mins[i].push_back(timeKern(ks[i]).mn);
  vector<size_t> ord(cand.size()); for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
  auto mn = [&](size_t i) { return *std::min_element(mins[i].begin(), mins[i].end()); };
  std::sort(ord.begin(), ord.end(), [&](size_t x, size_t y) { return mn(x) < mn(y); });
  printf("-- phase 4: interleaved head-to-head (%d rounds; min over rounds / median of per-round mins)\n", rounds);
  for (size_t i : ord) { auto v = mins[i]; std::sort(v.begin(), v.end());
    printf("H2H %dx%d C%u>%u@%u  %-7s/%-7s %-22s min %.3f ms  med %.3f ms  %5.0f GFLOPS(min)\n", d.P.kh, d.P.kh, d.P.Cin, d.P.Cout, d.P.H, cand[i].L.a.c_str(), cand[i].L.w.c_str(), cfgStr(cand[i].c).c_str(), v[0], v[v.size() / 2], 2.0 * d.P.M() * d.P.Cout * d.P.K() / v[0] / 1e6); }
  fflush(stdout);
}

static int runConv(int kh, uint32_t Cin, uint32_t Cout, uint32_t H) {
  Prob P{kh, Cin, Cout, H, H}; Data d; buildData(d, P);
  printf("== %dx%d conv Cin=%u Cout=%u %ux%u  (M=%u N=%u K=%u, %.3f GFLOP)\n", kh, kh, Cin, Cout, H, H, P.M(), P.Cout, P.K(), 2.0 * P.M() * P.Cout * P.K() / 1e9);
  auto out = mk((size_t)P.M() * P.Cout * 4, ST);
  // warm-up: >= 5 s of the baseline kernel
  { Lay L0{"nhwc", "hwio"}; Inputs in0 = buildInputs(d, L0); uint32_t gx, gy; Kern k = make(genKernel(P, L0, CFGS[0], gx, gy), {in0.a, in0.w, Res{out, {}}}, gx, gy); burn(k, 5.0); }
  vector<Result> all; map<string, Result> bestA;
  printf("-- phase 1: activation layouts (weights hwio)\n");
  for (string a : {"nhwc", "nc4", "z4", "texa", "texb", "nhwc16", "nc416", "texa16"}) {
    Lay L{a, "hwio"}; Inputs in = buildInputs(d, L);
    for (auto& c : CFGS) { Result r; if (runOne(d, in, out, L, c, r)) { all.push_back(r); if (!bestA.count(a) || r.gf > bestA[a].gf) bestA[a] = r; } }
  }
  printf("-- phase 2: weight layouts (top 2 configs of nhwc and nc4 activations)\n");
  for (string a : {"nhwc", "nc4"}) {
    if (!bestA.count(a)) continue;
    vector<Result> top; for (auto& r : all) if (r.L.a == a && r.L.w == "hwio") top.push_back(r);
    std::sort(top.begin(), top.end(), [](const Result& x, const Result& y) { return x.gf > y.gf; }); top.resize(std::min<size_t>(2, top.size()));
    for (string w : {"ok4", "nk4", "texw", "texw16"}) {
      Lay L{a, w}; Inputs in = buildInputs(d, L);
      for (auto& t : top) { Result r; if (runOne(d, in, out, L, t.c, r)) all.push_back(r); }
    }
  }
  printf("-- phase 3: packed f16 weights x f16 activations (NV=2 configs)\n");
  for (string a : {"nhwc16", "nc416", "texa16"}) for (string w : {"w16", "texw16"}) {
    Lay L{a, w}; Inputs in = buildInputs(d, L);
    for (auto& c : CFGS) { Result r; if (runOne(d, in, out, L, c, r)) all.push_back(r); }
  }
  // summary: best per (a,w)
  printf("-- best per (activation, weight) layout, GFLOPS by min time\n");
  map<string, Result> bp; for (auto& r : all) { string key = r.L.a + "/" + r.L.w; if (r.err < 1e-2 && (!bp.count(key) || r.gf > bp[key].gf)) bp[key] = r; }
  vector<Result> v; for (auto& kv : bp) v.push_back(kv.second);
  std::sort(v.begin(), v.end(), [](const Result& x, const Result& y) { return x.gf > y.gf; });
  { vector<Result> cand; for (auto& r : v) if (cand.size() < 12) cand.push_back(r);
    bool has = false; for (auto& r : cand) if (r.L.a == "nhwc" && r.L.w == "hwio") has = true;
    if (!has && bp.count("nhwc/hwio")) cand.push_back(bp["nhwc/hwio"]);
    headToHead(d, out, cand); }
  for (auto& r : v) printf("BEST %dx%d C%u>%u@%u  %-7s/%-7s %-22s %.3f ms  %5.0f GFLOPS  err=%.1e\n", kh, kh, Cin, Cout, H, r.L.a.c_str(), r.L.w.c_str(), cfgStr(r.c).c_str(), r.t.mn, r.gf, r.err);
  return 0;
}

// channel padding: 1x1 conv on a fixed pixel count, Cin=Cout=padded, useful GFLOPS computed from the logical channel count
static int runPad() {
  for (uint32_t logical : {100u, 60u, 96u}) {
    printf("== channel padding, 1x1 conv 28x28, Cin=Cout=%u logical\n", logical);
    vector<uint32_t> mults = {4u, 8u, 16u, 32u}; vector<Prob> Ps; vector<Data> ds(mults.size()); vector<wgpu::Buffer> outs; vector<Inputs> ins; vector<vector<Kern>> ks(mults.size()); vector<vector<Cfg>> cs(mults.size());
    Lay L{"nhwc", "hwio"};
    for (size_t i = 0; i < mults.size(); i++) {
      uint32_t padc = (logical + mults[i] - 1) / mults[i] * mults[i]; Prob P{1, padc, padc, 28, 28}; Ps.push_back(P); buildData(ds[i], P);
      outs.push_back(mk((size_t)P.M() * P.Cout * 4, ST)); ins.push_back(buildInputs(ds[i], L));
      for (auto& c : CFGS) { if (c.rowx || !valid(P, L, c)) continue; uint32_t gx, gy; ks[i].push_back(make(genKernel(P, L, c, gx, gy), {ins[i].a, ins[i].w, Res{outs[i], {}}}, gx, gy)); cs[i].push_back(c); }
    }
    burn(ks[0][0], 3.0);
    vector<double> best(mults.size(), 1e30); vector<Cfg> bcs(mults.size());
    for (int round = 0; round < 5; round++) for (size_t i = 0; i < mults.size(); i++) for (size_t j = 0; j < ks[i].size(); j++) { double t = timeKern(ks[i][j]).mn; if (t < best[i]) { best[i] = t; bcs[i] = cs[i][j]; } }
    for (size_t i = 0; i < mults.size(); i++) { const Prob& P = Ps[i];
      printf("  pad to x%-2u -> %3u channels: best %s %.3f ms  padded %.0f GFLOPS  useful %.0f GFLOPS\n", mults[i], P.Cin, cfgStr(bcs[i]).c_str(), best[i], 2.0 * P.M() * P.Cin * P.Cout / best[i] / 1e6, 2.0 * P.M() * logical * logical / best[i] / 1e6); }
  }
  return 0;
}

// NHWC <-> [C4][HW] blocked conversion kernels (one vec4 per thread)
static int runXform() {
  struct S { uint32_t C, H; } shapes[] = {{64, 56}, {256, 56}, {128, 28}, {512, 28}, {256, 14}, {1024, 14}};
  for (auto s : shapes) {
    uint32_t HW = s.H * s.H, C4 = s.C / 4, n = HW * C4; vector<float> X((size_t)HW * s.C); for (size_t i = 0; i < X.size(); i++) X[i] = (i % 97) * 0.01f;
    auto in = upload(X.data(), X.size() * 4), out = mk(X.size() * 4, ST);
    for (int dir = 0; dir < 2; dir++) {  // 0: nhwc -> nc4, 1: nc4 -> nhwc
      string w = "@group(0) @binding(0) var<storage, read> in0 : array<vec4<f32>>;\n@group(0) @binding(1) var<storage, read_write> outb : array<vec4<f32>>;\n"
                 "@compute @workgroup_size(64,1,1)\nfn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(num_workgroups) nw : vec3<u32>) {\n"
                 "  let i = g.x + g.y * nw.x * 64u; if (i >= " + U(n) + ") { return; }\n";
      if (dir == 0) w += "  let c4 = i % " + U(C4) + "; let p = i / " + U(C4) + ";\n  outb[c4 * " + U(HW) + " + p] = in0[i];\n}\n";
      else w += "  let p = i % " + U(HW) + "; let c4 = i / " + U(HW) + ";\n  outb[p * " + U(C4) + " + c4] = in0[i];\n}\n";
      uint32_t groups = (n + 63) / 64, gx = std::min(groups, 65535u), gy = (groups + gx - 1) / gx;
      Kern k = make(w, {Res{in, {}}, Res{out, {}}}, gx, gy); burn(k, 0.3); Timing t = timeKern(k);
      double bytes = (double)X.size() * 4 * 2;
      printf("xform %s C=%u %ux%u (%.2f MB moved): min %.3f ms med %.3f ms  %.0f GB/s\n", dir == 0 ? "nhwc->nc4" : "nc4->nhwc", s.C, s.H, s.H, bytes / 1e6, t.mn, t.med, bytes / t.mn / 1e6);
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: layouts conv KH Cin Cout H | pad | xform\n"); return 1; }
  initDevice(); string m = argv[1];
  if (m == "conv" && argc >= 6) return runConv(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  if (m == "pad") return runPad();
  if (m == "xform") return runXform();
  fprintf(stderr, "bad args\n"); return 1;
}
