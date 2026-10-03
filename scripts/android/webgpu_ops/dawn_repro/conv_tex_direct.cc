// Direct convolutions with textures for the convs where ORT's Winograd path is off (YOLO11n/26n: low channel counts, high
// resolution, stride 2), compared with a replica of ORT's Conv2dMM shader, through Dawn/Vulkan.
//   conv_tex_direct list
//   conv_tex_direct run IDX [ROUNDS]        one shape (see `list`), all variants, ABAB-interleaved timing
//   conv_tex_direct conv KS STRIDE CIN COUT H [ROUNDS]   a custom shape (square HxH input, pad (KS-1)/2)
// Variants (all NHWC4 data, f32 math, batch 1; "tex16"/"tex32" = RGBA16F/RGBA32F textures, texel = 4 channels):
//   A    ORT Conv2dMM replica: workgroup 8x8, 4 rows x 1 vec4 per thread, 32x32x32 shared-memory tiles, im2col gather in mm_readA
//   R    direct register-tile conv, buffer activations + buffer weights
//   W16  R with the weights in a tex16 (x = cout/4, y = tap*Cin + c)
//   W32  R with the weights in a tex32
//   B16  activations in a tex16 (x = w*C4 + c4, y = h) + weights in a tex16, buffer output
//   B32  same with tex32
//   B16o B16 with the output also written to a tex16 (cost of producing textures)
//   D    tinygrad-style (scripts/android/tinygrad_aot/r50_kernels): 4 pixels along x x 1 vec4 of output channels per thread,
//        the output-channel index slowest in the workgroup (weight loads uniform across a wave), tex16 in / weights / out
// Every variant except A is swept over a few (pixels, vec4 channels, workgroup shape, thread order) configurations; the best
// per variant then goes into an interleaved (A B C ... A B C ...) timing, min and median over rounds are reported. Also the
// cost of converting a buffer activation to a texture, and correctness against a double-precision CPU reference computed on the
// data each variant really reads (f16-rounded where it uses an RGBA16F texture). Build like gemm.cc (see README.md).
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>
using std::string; using std::to_string; using std::vector;

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
  vector<const char*> ad_en = {"use_vulkan_memory_model"}, dev_en = {"disable_robustness"};  // like ORT (RB=off VMM=1)
  wgpu::DawnTogglesDescriptor adt{}; adt.enabledToggleCount = ad_en.size(); adt.enabledToggles = ad_en.data(); ro.nextInChain = &adt;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data();
  wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt;
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
}

// ---------------------------------------------------------------- resources
static wgpu::Buffer mkbuf(size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = (n + 15) & ~size_t(15); b.usage = u; return dev.CreateBuffer(&b); }
static const auto ST = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
static wgpu::Buffer upload(const void* p, size_t bytes) { auto b = mkbuf(bytes, ST); dev.GetQueue().WriteBuffer(b, 0, p, bytes); return b; }
static vector<float> download(wgpu::Buffer src, size_t n) {
  wgpu::Buffer rb = mkbuf(n * 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  { wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); enc.CopyBufferToBuffer(src, 0, rb, 0, (n * 4 + 3) & ~size_t(3)); wgpu::CommandBuffer cb = enc.Finish(); dev.GetQueue().Submit(1, &cb); }
  bool ok = false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read, 0, rb.GetSize(), wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::MapAsyncStatus st, wgpu::StringView) { ok = st == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
  vector<float> out(n); if (ok) memcpy(out.data(), rb.GetConstMappedRange(0, rb.GetSize()), n * 4);
  return out;
}
struct Tex { wgpu::Texture t; wgpu::TextureView v; int bits = 0; };
// RGBA texture w x h; `data` = w*h*4 floats (converted to half for bits == 16). usage: sampled (+ storage when `storage`)
static Tex mktex(uint32_t w, uint32_t h, int bits, const float* data, bool storage) {
  Tex T; T.bits = bits;
  wgpu::TextureDescriptor td{}; td.size = {w, h, 1};
  td.format = bits == 16 ? wgpu::TextureFormat::RGBA16Float : wgpu::TextureFormat::RGBA32Float;
  td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | (storage ? wgpu::TextureUsage::StorageBinding : wgpu::TextureUsage::None);
  T.t = dev.CreateTexture(&td); T.v = T.t.CreateView();
  if (data) {
    vector<uint16_t> hv; const void* src = data; size_t bpt = 16;
    if (bits == 16) { hv.resize((size_t)w * h * 4); for (size_t i = 0; i < hv.size(); i++) hv[i] = f2h(data[i]); src = hv.data(); bpt = 8; }
    wgpu::TexelCopyTextureInfo di{}; di.texture = T.t; wgpu::TexelCopyBufferLayout lay{}; lay.bytesPerRow = w * bpt; lay.rowsPerImage = h; wgpu::Extent3D ext = {w, h, 1};
    dev.GetQueue().WriteTexture(&di, src, (size_t)w * h * bpt, &lay, &ext);
  }
  return T;
}

enum Kind { RO_BUF, RW_BUF, TEX_RO, TEX_WO16, TEX_WO32 };
struct Res { wgpu::Buffer b; Tex t; };
struct Kern { wgpu::ComputePipeline pl; wgpu::BindGroup bg; uint32_t gx = 1, gy = 1, gz = 1; bool ok = true; };

static Kern make(const string& wgsl, const vector<Kind>& kinds, const vector<Res>& res, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
  Kern k; k.gx = gx; k.gy = gy; k.gz = gz;
  vector<wgpu::BindGroupLayoutEntry> le(kinds.size());
  for (size_t i = 0; i < kinds.size(); i++) {
    le[i].binding = i; le[i].visibility = wgpu::ShaderStage::Compute;
    switch (kinds[i]) {
      case RO_BUF: le[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage; break;
      case RW_BUF: le[i].buffer.type = wgpu::BufferBindingType::Storage; break;
      case TEX_RO: le[i].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat; le[i].texture.viewDimension = wgpu::TextureViewDimension::e2D; break;
      case TEX_WO16: case TEX_WO32:
        le[i].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly; le[i].storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;
        le[i].storageTexture.format = kinds[i] == TEX_WO16 ? wgpu::TextureFormat::RGBA16Float : wgpu::TextureFormat::RGBA32Float; break;
    }
  }
  wgpu::BindGroupLayoutDescriptor bgld{}; bgld.entryCount = le.size(); bgld.entries = le.data();
  wgpu::BindGroupLayout bgl = dev.CreateBindGroupLayout(&bgld);
  wgpu::PipelineLayoutDescriptor pld{}; pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
  wgpu::PipelineLayout plo = dev.CreatePipelineLayout(&pld);
  wgpu::ShaderSourceWGSL w{}; w.code = {wgsl.data(), wgsl.size()};
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.layout = plo; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  k.pl = dev.CreateComputePipeline(&cpd);
  vector<wgpu::BindGroupEntry> e(kinds.size());
  for (size_t i = 0; i < kinds.size(); i++) {
    e[i].binding = i;
    if (kinds[i] == RO_BUF || kinds[i] == RW_BUF) { e[i].buffer = res[i].b; e[i].size = res[i].b.GetSize(); } else e[i].textureView = res[i].t.v;
  }
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = bgl; bgd.entryCount = e.size(); bgd.entries = e.data();
  k.bg = dev.CreateBindGroup(&bgd);
  return k;
}

static void submitWait(wgpu::CommandBuffer& cb) {
  dev.GetQueue().Submit(1, &cb);
  inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
}
// `iters` back-to-back dispatches (one compute pass each) in one submission; ms per dispatch
static double timeBatch(const Kern& k, int iters) {
  wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
  for (int i = 0; i < iters; i++) { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(k.pl); p.SetBindGroup(0, k.bg); p.DispatchWorkgroups(k.gx, k.gy, k.gz); p.End(); }
  wgpu::CommandBuffer cb = enc.Finish();
  double t0 = nowms(); submitWait(cb); return (nowms() - t0) / iters;
}
static int itersFor(const Kern& k, double target_ms = 8.0) {
  timeBatch(k, 2); double t1 = timeBatch(k, 4); return std::max(1, std::min(400, (int)std::ceil(target_ms / std::max(t1, 0.005))));
}
static double quickTime(const Kern& k) {  // used to rank configurations
  int it = itersFor(k, 4.0); double best = 1e30; for (int i = 0; i < 4; i++) best = std::min(best, timeBatch(k, it)); return best;
}

// ---------------------------------------------------------------- problem
struct Prob { int KS, stride, Cin, Cout, H, W; int pad() const { return (KS - 1) / 2; } int OH() const { return (H + 2 * pad() - KS) / stride + 1; } int OW() const { return (W + 2 * pad() - KS) / stride + 1; }
  int C4() const { return Cin / 4; } int N4() const { return Cout / 4; } double flops() const { return 2.0 * OH() * OW() * Cout * Cin * KS * KS; } };

// ---------------------------------------------------------------- A: replica of ORT's Conv2dMM (vec4, NHWC)
static string genA(const Prob& p) {
  uint32_t M = p.OH() * p.OW(), K = p.KS * p.KS * p.Cin, N = p.Cout, N4 = p.N4();
  bool fitA = M % 32 == 0, fitB = N % 32 == 0, fitK = K % 32 == 0;
  string s = "@group(0) @binding(0) var<storage, read> x : array<vec4<f32>>;\n@group(0) @binding(1) var<storage, read> w : array<vec4<f32>>;\n@group(0) @binding(2) var<storage, read_write> y : array<vec4<f32>>;\n";
  s += "fn mm_readA(row : i32, colIn : i32) -> vec4<f32> {\n  let col = colIn * 4;\n";
  if (!fitA || !fitK) s += "  if (!(row < " + to_string(M) + " && col < " + to_string(K) + ")) { return vec4<f32>(0.0); }\n";
  s += "  let outRow = row / " + to_string(p.OW()) + "; let outCol = row % " + to_string(p.OW()) + ";\n"
       "  let WRow = col / " + to_string(p.KS * p.Cin) + "; let WCol = (col / " + to_string(p.Cin) + ") % " + to_string(p.KS) + ";\n"
       "  let xRow = outRow * " + to_string(p.stride) + " + WRow - " + to_string(p.pad()) + "; let xCol = outCol * " + to_string(p.stride) + " + WCol - " + to_string(p.pad()) + ";\n"
       "  let xCh = col % " + to_string(p.Cin) + ";\n  var res = vec4<f32>(0.0);\n"
       "  if (xRow >= 0 && xRow < " + to_string(p.H) + " && xCol >= 0 && xCol < " + to_string(p.W) + ") { res = x[(xRow * " + to_string(p.W) + " + xCol) * " + to_string(p.C4()) + " + xCh / 4]; }\n  return res;\n}\n";
  s += "fn mm_readB(row : i32, colIn : i32) -> vec4<f32> {\n";
  if (!fitB || !fitK) s += "  if (!(row < " + to_string(K) + " && colIn * 4 < " + to_string(N) + ")) { return vec4<f32>(0.0); }\n";
  s += "  return w[row * " + to_string(N4) + " + colIn];\n}\n";
  s += "fn mm_write(row : i32, colIn : i32, v : vec4<f32>) {\n  if (row < " + to_string(M) + " && colIn * 4 < " + to_string(N) + ") { y[row * " + to_string(N4) + " + colIn] = v; }\n}\n";
  s += "var<workgroup> mm_Asub : array<array<vec4<f32>, 8>, 32>;\nvar<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 32>;\n";
  s += "@compute @workgroup_size(8, 8, 1)\nfn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
       "  let localRow = i32(lid.y); let tileRow = localRow * 4; let tileCol = i32(lid.x);\n"
       "  let globalRow = i32(wid.y * 8u + lid.y) * 4; let globalCol = i32(wid.x * 8u + lid.x);\n"
       "  var kStart = 0;\n  var acc : array<vec4<f32>, 4>;\n  let tileRowB = localRow * 4;\n"
       "  for (var t = 0; t < " + to_string((K + 31) / 32) + "; t = t + 1) {\n"
       "    for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) { let inputRow = tileRow + innerRow; let inputCol = tileCol; mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, kStart / 4 + inputCol); }\n"
       "    for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) { let inputRow = tileRowB + innerRow; let inputCol = tileCol; mm_Bsub[inputRow][inputCol] = mm_readB(kStart + inputRow, globalCol); }\n"
       "    kStart = kStart + 32;\n    workgroupBarrier();\n"
       "    for (var k = 0; k < 8; k = k + 1) {\n"
       "      let BCached0 = mm_Bsub[k * 4][tileCol]; let BCached1 = mm_Bsub[k * 4 + 1][tileCol]; let BCached2 = mm_Bsub[k * 4 + 2][tileCol]; let BCached3 = mm_Bsub[k * 4 + 3][tileCol];\n"
       "      for (var i = 0; i < 4; i = i + 1) {\n        let ACached = mm_Asub[tileRow + i][k];\n"
       "        acc[i] = BCached0 * ACached.x + acc[i];\n        acc[i] = BCached1 * ACached.y + acc[i];\n        acc[i] = BCached2 * ACached.z + acc[i];\n        acc[i] = BCached3 * ACached.w + acc[i];\n      }\n    }\n"
       "    workgroupBarrier();\n  }\n"
       "  for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) { mm_write(globalRow + innerRow, globalCol, acc[innerRow]); }\n}\n";
  return s;
}

// ---------------------------------------------------------------- direct register-tile conv
struct Cfg { int TM, NV, R, XC, OC, order; string str() const { return "tm" + to_string(TM) + " nv" + to_string(NV) + " wg" + to_string(R) + "x" + to_string(XC) + "x" + to_string(OC) + (order ? " oc-slow" : " oc-fast"); } };
// order 0: output-channel group fastest in the workgroup, then x chunk, then row; order 1: row fastest, then x chunk, then oc group (slowest)
static string genD(const Prob& p, const Cfg& c, int inBits /*0 buffer*/, int wBits, int outBits) {
  int KS = p.KS, OH = p.OH(), OW = p.OW(), C4 = p.C4(), N4 = p.N4(), Cin = p.Cin;
  string s;
  s += inBits ? "@group(0) @binding(0) var tin : texture_2d<f32>;\n" : "@group(0) @binding(0) var<storage, read> in0 : array<vec4<f32>>;\n";
  s += wBits ? "@group(0) @binding(1) var tw : texture_2d<f32>;\n" : "@group(0) @binding(1) var<storage, read> in1 : array<vec4<f32>>;\n";
  if (outBits) s += string("@group(0) @binding(2) var tout : texture_storage_2d<") + (outBits == 16 ? "rgba16float" : "rgba32float") + ", write>;\n";
  else s += "@group(0) @binding(2) var<storage, read_write> outb : array<vec4<f32>>;\n";
  int S = c.R * c.XC * c.OC;
  s += "@compute @workgroup_size(" + to_string(S) + ", 1, 1)\nfn main(@builtin(local_invocation_index) li : u32, @builtin(workgroup_id) wid : vec3<u32>) {\n";
  if (c.order == 0) s += "  let o = li % " + U(c.OC) + "; let xc = (li / " + U(c.OC) + ") % " + U(c.XC) + "; let r = li / " + U(c.OC * c.XC) + ";\n";
  else s += "  let r = li % " + U(c.R) + "; let xc = (li / " + U(c.R) + ") % " + U(c.XC) + "; let o = li / " + U(c.R * c.XC) + ";\n";
  s += "  let oy = wid.y * " + U(c.R) + " + r; let ox0 = (wid.x * " + U(c.XC) + " + xc) * " + U(c.TM) + "; let n0 = (wid.z * " + U(c.OC) + " + o) * " + U(c.NV) + ";\n"
       "  if (oy >= " + U(OH) + " || ox0 >= " + U(OW) + " || n0 >= " + U(N4) + ") { return; }\n";
  for (int m = 0; m < c.TM; m++) for (int j = 0; j < 4 * c.NV; j++) s += "  var c" + to_string(m) + "_" + to_string(j) + " = 0.0;\n";
  for (int n = 0; n < c.NV; n++) s += "  let nc" + to_string(n) + " = min(n0 + " + U(n) + ", " + U(N4 - 1) + ");\n";
  s += "  for (var ky = 0u; ky < " + U(KS) + "; ky++) {\n"
       "    let iy = i32(oy * " + U(p.stride) + ") - " + to_string(p.pad()) + " + i32(ky);\n"
       "    let rowok = iy >= 0 && iy < " + to_string(p.H) + ";\n    let iyc = u32(clamp(iy, 0, " + to_string(p.H - 1) + "));\n"
       "    for (var kx = 0u; kx < " + U(KS) + "; kx++) {\n      let tap = ky * " + U(KS) + " + kx;\n";
  for (int m = 0; m < c.TM; m++) {
    string i = to_string(m);
    s += "      let ix" + i + " = i32((ox0 + " + U(m) + ") * " + U(p.stride) + ") - " + to_string(p.pad()) + " + i32(kx);\n"
         "      let ok" + i + " = rowok && ix" + i + " >= 0 && ix" + i + " < " + to_string(p.W) + " && (ox0 + " + U(m) + ") < " + U(OW) + ";\n"
         "      let ic" + i + " = u32(clamp(ix" + i + ", 0, " + to_string(p.W - 1) + "));\n";
  }
  s += "      for (var c4 = 0u; c4 < " + U(C4) + "; c4++) {\n";
  for (int m = 0; m < c.TM; m++) {
    string i = to_string(m);
    string ld = inBits ? "textureLoad(tin, vec2<i32>(i32(ic" + i + " * " + U(C4) + " + c4), i32(iyc)), 0)" : "in0[(iyc * " + U(p.W) + " + ic" + i + ") * " + U(C4) + " + c4]";
    s += "        let a" + i + " = select(vec4<f32>(0.0), " + ld + ", ok" + i + ");\n";
  }
  for (int j = 0; j < 4; j++) {
    s += "        let row" + to_string(j) + " = tap * " + U(Cin) + " + c4 * 4u + " + U(j) + ";\n";
    for (int n = 0; n < c.NV; n++) {
      string ld = wBits ? "textureLoad(tw, vec2<i32>(i32(nc" + to_string(n) + "), i32(row" + to_string(j) + ")), 0)" : "in1[row" + to_string(j) + " * " + U(N4) + " + nc" + to_string(n) + "]";
      s += "        let b" + to_string(n) + "_" + to_string(j) + " = " + ld + ";\n";
    }
    for (int n = 0; n < c.NV; n++) for (int m = 0; m < c.TM; m++) for (int q = 0; q < 4; q++)
      s += "        c" + to_string(m) + "_" + to_string(n * 4 + q) + " = fma(a" + to_string(m) + "." + "xyzw"[j] + ", b" + to_string(n) + "_" + to_string(j) + "." + "xyzw"[q] + ", c" + to_string(m) + "_" + to_string(n * 4 + q) + ");\n";
  }
  s += "      }\n    }\n  }\n";
  for (int m = 0; m < c.TM; m++) for (int n = 0; n < c.NV; n++) {
    string v = "vec4<f32>(c" + to_string(m) + "_" + to_string(n * 4) + ", c" + to_string(m) + "_" + to_string(n * 4 + 1) + ", c" + to_string(m) + "_" + to_string(n * 4 + 2) + ", c" + to_string(m) + "_" + to_string(n * 4 + 3) + ")";
    string cond = "(ox0 + " + U(m) + ") < " + U(OW) + " && (n0 + " + U(n) + ") < " + U(N4);
    if (outBits) s += "  if (" + cond + ") { textureStore(tout, vec2<i32>(i32((ox0 + " + U(m) + ") * " + U(N4) + " + n0 + " + U(n) + "), i32(oy)), " + v + "); }\n";
    else s += "  if (" + cond + ") { outb[(oy * " + U(OW) + " + ox0 + " + U(m) + ") * " + U(N4) + " + n0 + " + U(n) + "] = " + v + "; }\n";
  }
  return s + "}\n";
}
static void dispD(const Prob& p, const Cfg& c, uint32_t& gx, uint32_t& gy, uint32_t& gz) {
  gx = (p.OW() + c.XC * c.TM - 1) / (c.XC * c.TM); gy = (p.OH() + c.R - 1) / c.R; gz = (p.N4() + c.OC * c.NV - 1) / (c.OC * c.NV);
}

// ---------------------------------------------------------------- conversions
static string genB2T(uint32_t width, int bits) {  // buffer (vec4 per entry) -> texture, entry i at (i % width, i / width)
  return "@group(0) @binding(0) var<storage, read> src : array<vec4<f32>>;\n@group(0) @binding(1) var dst : texture_storage_2d<" + string(bits == 16 ? "rgba16float" : "rgba32float") + ", write>;\n"
         "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(num_workgroups) nw : vec3<u32>) {\n"
         "  let i = g.x + g.y * nw.x * 64u;\n  if (i >= arrayLength(&src)) { return; }\n  textureStore(dst, vec2<i32>(i32(i % " + U(width) + "), i32(i / " + U(width) + ")), src[i]);\n}\n";
}
static string genT2B(uint32_t width) {  // texture -> buffer (verification)
  return "@group(0) @binding(0) var src : texture_2d<f32>;\n@group(0) @binding(1) var<storage, read_write> dst : array<vec4<f32>>;\n"
         "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(num_workgroups) nw : vec3<u32>) {\n"
         "  let i = g.x + g.y * nw.x * 64u;\n  if (i >= arrayLength(&dst)) { return; }\n  dst[i] = textureLoad(src, vec2<i32>(i32(i % " + U(width) + "), i32(i / " + U(width) + ")), 0);\n}\n";
}
static void dispatch1D(uint64_t n, uint32_t& gx, uint32_t& gy) { uint64_t groups = (n + 63) / 64; gx = (uint32_t)std::min<uint64_t>(groups, 65535); gy = (uint32_t)((groups + gx - 1) / gx); }

// ---------------------------------------------------------------- driver
struct ShapeDef { const char* name; int KS, stride, Cin, Cout, H; int count; };
// YOLO11n convs (optimized graph shapes). Winograd (ORT, min(Cin,Cout) >= 64, stride 1) is off for the rest of the 3x3s.
static const ShapeDef SHAPES[] = {
    {"3x3 s1 32>32 @40 (x4)", 3, 1, 32, 32, 40, 4},     {"3x3 s1 32>64 @40 (x2)", 3, 1, 32, 64, 40, 2},   {"3x3 s1 64>32 @40 (x2)", 3, 1, 64, 32, 40, 2},
    {"3x3 s1 16>32 @80 (x2)", 3, 1, 16, 32, 80, 2},     {"3x3 s1 32>16 @80 (x2)", 3, 1, 32, 16, 80, 2},   {"3x3 s1 8>16 @160", 3, 1, 8, 16, 160, 1},
    {"3x3 s1 16>8 @160", 3, 1, 16, 8, 160, 1},          {"3x3 s2 16>32 @320", 3, 2, 16, 32, 320, 1},      {"3x3 s2 64>64 @160", 3, 2, 64, 64, 160, 1},
    {"3x3 s2 64>64 @80", 3, 2, 64, 64, 80, 1},          {"3x3 s2 128>128 @80", 3, 2, 128, 128, 80, 1},    {"3x3 s2 128>128 @40", 3, 2, 128, 128, 40, 1},
    {"3x3 s2 128>256 @40", 3, 2, 128, 256, 40, 1},      {"3x3 s1 64>64 @80 (Winograd)", 3, 1, 64, 64, 80, 2}, {"3x3 s1 64>64 @20 (Winograd)", 3, 1, 64, 64, 20, 9},
    {"1x1 192>128 @40 (x4)", 1, 1, 192, 128, 40, 4},    {"1x1 384>256 @20 (x3)", 1, 1, 384, 256, 20, 3},  {"1x1 256>64 @80", 1, 1, 256, 64, 80, 1},
    {"1x1 64>64 @80 (x2)", 1, 1, 64, 64, 80, 2},
};
static const int NSHAPES = sizeof(SHAPES) / sizeof(SHAPES[0]);

struct Variant { string name; int inBits, wBits, outBits; bool ort; vector<Cfg> cfgs; };

static vector<Cfg> sweepCfgs() {
  return {{4, 1, 4, 2, 8, 0}, {4, 2, 4, 2, 4, 0}, {4, 1, 4, 2, 16, 1}, {8, 1, 4, 1, 16, 1}, {2, 2, 8, 2, 4, 0}, {4, 2, 2, 4, 4, 1}, {8, 2, 2, 1, 8, 1}, {4, 1, 8, 2, 4, 1}, {4, 1, 16, 2, 4, 0}, {4, 1, 2, 4, 8, 1}};
}
static vector<Cfg> tgCfgs() { return {{4, 1, 4, 2, 16, 1}, {4, 1, 8, 2, 8, 1}, {4, 1, 4, 4, 8, 1}, {4, 1, 4, 2, 8, 1}, {4, 1, 16, 2, 4, 1}}; }

struct Sample { size_t oy, ox, n; };

static int runShape(const Prob& p, const char* name, int rounds) {
  const uint32_t OH = p.OH(), OW = p.OW(), C4 = p.C4(), N4 = p.N4();
  const size_t xs = (size_t)p.H * p.W * p.Cin, ws = (size_t)p.KS * p.KS * p.Cin * p.Cout, os = (size_t)OH * OW * p.Cout;
  vector<float> X(xs), Wb(ws), Wt(ws);  // Wb[(tap*Cin + c)*Cout + n]
  for (size_t i = 0; i < xs; i++) X[i] = (((i * 7919) % 1000) / 1000.f - .5f);
  double wscale = 1.0 / std::sqrt((double)p.KS * p.KS * p.Cin);
  for (int n = 0; n < p.Cout; n++) for (int c = 0; c < p.Cin; c++) for (int t = 0; t < p.KS * p.KS; t++) {
    size_t i = ((size_t)n * p.Cin + c) * p.KS * p.KS + t; float v = (float)((((i * 104729) % 1000) / 1000.f - .5f) * 2 * wscale);
    Wb[((size_t)t * p.Cin + c) * p.Cout + n] = v; Wt[i] = v;
  }
  // reference data (f16-rounded copies for the variants that read RGBA16F textures)
  vector<float> X16(xs), Wb16(ws);
  for (size_t i = 0; i < xs; i++) X16[i] = rnd16(X[i]);
  for (size_t i = 0; i < ws; i++) Wb16[i] = rnd16(Wb[i]);
  vector<Sample> smp; for (int q = 0; q < 200; q++) smp.push_back({(size_t)((q * 2654435761u) % OH), (size_t)((q * 40503u + 11) % OW), (size_t)((q * 7919u + 3) % p.Cout)});
  auto reference = [&](const vector<float>& Xr, const vector<float>& Wr) {
    vector<double> ref;
    for (auto& s : smp) { double a = 0;
      for (int ky = 0; ky < p.KS; ky++) for (int kx = 0; kx < p.KS; kx++) { int iy = (int)s.oy * p.stride - p.pad() + ky, ix = (int)s.ox * p.stride - p.pad() + kx; if (iy < 0 || iy >= p.H || ix < 0 || ix >= p.W) continue;
        for (int c = 0; c < p.Cin; c++) a += (double)Xr[((size_t)iy * p.W + ix) * p.Cin + c] * Wr[((size_t)(ky * p.KS + kx) * p.Cin + c) * p.Cout + s.n]; }
      ref.push_back(a); }
    return ref;
  };
  vector<double> ref32 = reference(X, Wb), refW16 = reference(X, Wb16), ref16 = reference(X16, Wb16);
  auto relerr = [&](const vector<float>& Y, const vector<double>& ref) { double me = 0, mr = 0; for (size_t i = 0; i < smp.size(); i++) { double g = Y[((smp[i].oy * OW) + smp[i].ox) * p.Cout + smp[i].n]; me = std::max(me, fabs(g - ref[i])); mr = std::max(mr, fabs(ref[i])); } return me / mr; };

  // resources
  Res rX{upload(X.data(), xs * 4), {}}, rW{upload(Wb.data(), ws * 4), {}}, rY{mkbuf(os * 4, ST), {}};
  Res tX16{{}, mktex(p.W * C4, p.H, 16, X.data(), false)}, tX32{{}, mktex(p.W * C4, p.H, 32, X.data(), false)};
  Res tW16{{}, mktex(N4, p.KS * p.KS * p.Cin, 16, Wb.data(), false)}, tW32{{}, mktex(N4, p.KS * p.KS * p.Cin, 32, Wb.data(), false)};
  Res tY16{{}, mktex(OW * N4, OH, 16, nullptr, true)};

  printf("== %s  [%dx%d s%d %d>%d @%d]  %.3f GFLOP, out %ux%ux%d\n", name, p.KS, p.KS, p.stride, p.Cin, p.Cout, p.H, p.flops() / 1e9, OH, OW, p.Cout);
  fflush(stdout);

  struct Entry { string name; Kern k; string cfg; double qt = 0; vector<double> t; std::function<double()> err; };
  vector<Entry> ents;
  auto addVariant = [&](const string& vname, int inB, int wB, int oB, const vector<Cfg>& cfgs, const vector<double>* ref, bool tdown) {
    double bestq = 1e30; Entry best;
    for (auto& c : cfgs) {
      if (c.R * c.XC * c.OC > 256 || c.R * c.XC * c.OC < 16) continue;
      vector<Kind> kinds = {inB ? TEX_RO : RO_BUF, wB ? TEX_RO : RO_BUF, oB ? (oB == 16 ? TEX_WO16 : TEX_WO32) : RW_BUF};
      vector<Res> res = {inB ? (inB == 16 ? tX16 : tX32) : rX, wB ? (wB == 16 ? tW16 : tW32) : rW, oB ? tY16 : rY};
      uint32_t gx, gy, gz; dispD(p, c, gx, gy, gz);
      Kern k = make(genD(p, c, inB, wB, oB), kinds, res, gx, gy, gz);
      double q = quickTime(k);
      if (q < bestq) { bestq = q; best.name = vname; best.k = k; best.cfg = c.str(); best.qt = q; }
    }
    if (bestq >= 1e29) { printf("  %s: no valid configuration\n", vname.c_str()); return; }
    // correctness
    if (tdown) {  // read the output texture back
      Res rT{upload(vector<float>(os, 0.f).data(), os * 4), {}};
      Kern kd = make(genT2B(OW * N4), {TEX_RO, RW_BUF}, {tY16, rT}, 1, 1, 1); uint32_t gx, gy; dispatch1D(os / 4, gx, gy); kd.gx = gx; kd.gy = gy;
      wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); wgpu::ComputePassEncoder cp = enc.BeginComputePass(); cp.SetPipeline(kd.pl); cp.SetBindGroup(0, kd.bg); cp.DispatchWorkgroups(kd.gx, kd.gy, 1); cp.End(); wgpu::CommandBuffer cb = enc.Finish(); submitWait(cb);
      vector<float> Y = download(rT.b, os); double e = relerr(Y, *ref); best.err = [e]() { return e; };
    } else { timeBatch(best.k, 1); vector<float> Y = download(rY.b, os); double e = relerr(Y, *ref); best.err = [e]() { return e; }; }
    ents.push_back(best);
  };

  // A: ORT replica
  { uint32_t M = OH * OW; Kern k = make(genA(p), {RO_BUF, RO_BUF, RW_BUF}, {rX, rW, rY}, (N4 + 7) / 8, (M + 31) / 32, 1);
    Entry e; e.name = "A"; e.k = k; e.cfg = "ORT Conv2dMM replica"; e.qt = quickTime(k); timeBatch(k, 1); vector<float> Y = download(rY.b, os); double er = relerr(Y, ref32); e.err = [er]() { return er; }; ents.push_back(e); }
  addVariant("R", 0, 0, 0, sweepCfgs(), &ref32, false);
  addVariant("W16", 0, 16, 0, sweepCfgs(), &refW16, false);
  addVariant("W32", 0, 32, 0, sweepCfgs(), &ref32, false);
  addVariant("B16", 16, 16, 0, sweepCfgs(), &ref16, false);
  addVariant("B32", 32, 32, 0, sweepCfgs(), &ref32, false);
  addVariant("B16o", 16, 16, 16, sweepCfgs(), &ref16, true);
  addVariant("D", 16, 16, 16, tgCfgs(), &ref16, true);

  // interleaved timing
  for (auto& e : ents) e.t.clear();
  vector<int> its; for (auto& e : ents) its.push_back(itersFor(e.k, 8.0));
  for (int r = 0; r < rounds; r++) for (size_t i = 0; i < ents.size(); i++) ents[i].t.push_back(timeBatch(ents[i].k, its[i]));
  auto stat = [](vector<double> v, double& mn, double& md) { std::sort(v.begin(), v.end()); mn = v[0]; md = v[v.size() / 2]; };
  double amin = 0; { double md; stat(ents[0].t, amin, md); }
  printf("  %-5s %-26s %9s %9s %8s %9s  %s\n", "var", "config", "min ms", "median", "GFLOPS", "vs A(min)", "relerr");
  for (auto& e : ents) { double mn, md; stat(e.t, mn, md); printf("  %-5s %-26s %9.3f %9.3f %8.0f %8.2fx  %.1e\n", e.name.c_str(), e.cfg.c_str(), mn, md, p.flops() / mn / 1e6, amin / mn, e.err()); }

  // conversion cost buffer -> texture for the input activation
  { uint32_t gx, gy; dispatch1D((uint64_t)p.H * p.W * C4, gx, gy);
    Kern c16 = make(genB2T(p.W * C4, 16), {RO_BUF, TEX_WO16}, {rX, Res{{}, mktex(p.W * C4, p.H, 16, nullptr, true)}}, gx, gy, 1);
    Kern c32 = make(genB2T(p.W * C4, 32), {RO_BUF, TEX_WO32}, {rX, Res{{}, mktex(p.W * C4, p.H, 32, nullptr, true)}}, gx, gy, 1);
    int i16 = itersFor(c16, 4.0), i32 = itersFor(c32, 4.0); double b16 = 1e30, b32 = 1e30;
    for (int i = 0; i < 6; i++) { b16 = std::min(b16, timeBatch(c16, i16)); b32 = std::min(b32, timeBatch(c32, i32)); }
    printf("  convert buffer->tex16 %.3f ms, ->tex32 %.3f ms (activation %.1f KB f32)\n", b16, b32, xs * 4 / 1024.0); }
  fflush(stdout);
  return 0;
}

static void warmup(double seconds) {  // keep the GPU busy so the clock ramps before anything is timed
  Prob p{3, 1, 64, 64, 80, 80};
  const size_t xs = (size_t)p.H * p.W * p.Cin, ws = (size_t)9 * p.Cin * p.Cout, os = (size_t)p.H * p.W * p.Cout;
  vector<float> X(xs, 0.1f), W(ws, 0.01f);
  Res rX{upload(X.data(), xs * 4), {}}, rW{upload(W.data(), ws * 4), {}}, rY{mkbuf(os * 4, ST), {}};
  Kern k = make(genA(p), {RO_BUF, RO_BUF, RW_BUF}, {rX, rW, rY}, (p.N4() + 7) / 8, (p.OH() * p.OW() + 31) / 32, 1);
  int it = itersFor(k, 20.0); double t0 = nowms(); while (nowms() - t0 < seconds * 1000) timeBatch(k, it);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: conv_tex_direct list | run IDX [ROUNDS] | conv KS STRIDE CIN COUT H [ROUNDS]\n"); return 1; }
  string m = argv[1];
  if (m == "list") { for (int i = 0; i < NSHAPES; i++) printf("%d  %s\n", i, SHAPES[i].name); return 0; }
  initDevice();
  double warm = getenv("WARM") ? atof(getenv("WARM")) : 5.0;
  if (m == "run") {
    int idx = atoi(argv[2]), rounds = argc > 3 ? atoi(argv[3]) : 9;
    if (idx < 0 || idx >= NSHAPES) { fprintf(stderr, "bad index\n"); return 1; }
    warmup(warm); const ShapeDef& d = SHAPES[idx]; return runShape(Prob{d.KS, d.stride, d.Cin, d.Cout, d.H, d.H}, d.name, rounds);
  }
  if (m == "conv" && argc >= 7) {
    warmup(warm); Prob p{atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[6])};
    return runShape(p, "custom", argc > 7 ? atoi(argv[7]) : 9);
  }
  fprintf(stderr, "bad arguments\n"); return 1;
}
