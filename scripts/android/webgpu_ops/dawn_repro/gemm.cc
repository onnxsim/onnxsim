// GEMM kernel-design experiments through Dawn/Vulkan: C[M,N] = A[M,K] * B[K,N], all f32 row-major.
//   gemm VARIANT M N K REPS [TM NV WX WY]
// VARIANT: reg  = no shared memory; each thread owns TM rows x (4*NV) cols, vec4 loads straight from global
//          sh   = shared-memory tiles (the classic design ORT uses), TM x 4 outputs per thread
// REPS output slices are written by z workgroups so one dispatch keeps the whole GPU busy (throughput);
// a single-slice run (REPS=1) shows the latency of one real layer.
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
static uint16_t f2h(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff; int exp = ((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return (uint16_t)sign; if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  uint16_t h = (uint16_t)(sign | (exp << 10) | (mant >> 13)); if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) h++; return h; }
static float h2f(uint16_t h) { uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023, x; if (e == 0) { if (!m) x = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; x = s | ((e + 112) << 23) | (m << 13); } } else x = s | ((e + 112) << 23) | (m << 13); float f; memcpy(&f, &x, 4); return f; }
int main(int argc, char** argv) {
  dawnProcSetProcs(&dawn::native::GetProcs());
  string var = argv[1];
  uint32_t M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), REPS = atoi(argv[5]);
  int TM = argc > 6 ? atoi(argv[6]) : 4, NV = argc > 7 ? atoi(argv[7]) : 1, WX = argc > 8 ? atoi(argv[8]) : 8, WY = argc > 9 ? atoi(argv[9]) : 8;
  if (K % 4 || N % 4) { fprintf(stderr, "K and N must be multiples of 4\n"); return 1; }
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  wgpu::Instance inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  // Dawn toggles like ORT: RB=off disables robust buffer access, VMM=1 enables the Vulkan memory model
  std::vector<const char*> ad_en, dev_en, dev_dis;
  bool rb_off = getenv("RB") && string(getenv("RB")) == "off", vmm = getenv("VMM") && string(getenv("VMM")) == "1";
  if (vmm) ad_en.push_back("use_vulkan_memory_model");
  if (rb_off) dev_en.push_back("disable_robustness");
  if (getenv("SKIPVAL")) dev_en.push_back("skip_validation");
  wgpu::DawnTogglesDescriptor adt{}; adt.enabledToggleCount = ad_en.size(); adt.enabledToggles = ad_en.data(); ro.nextInChain = &adt;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data(); dvt.disabledToggleCount = dev_dis.size(); dvt.disabledToggles = dev_dis.data();
  wgpu::FeatureName sgf = wgpu::FeatureName::Subgroups; wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt; if (ad.HasFeature(sgf)) { dd.requiredFeatureCount = 1; dd.requiredFeatures = &sgf; }
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  { wgpu::SupportedFeatures sf; ad.GetFeatures(&sf); for (size_t i = 0; i < sf.featureCount; i++) if (sf.features[i] == wgpu::FeatureName::Subgroups) fprintf(stderr, "adapter has Subgroups\n"); }
  wgpu::Device dev;
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
  const int TN = 4 * NV;                     // output columns per thread
  const uint32_t tileM = WY * TM, tileN = WX * TN;
  string s = "struct U { M: u32, N: u32, K: u32, pad: u32 }\n"
             "@group(0) @binding(0) var<storage, read> A : array<vec4<f32>>;\n"
             "@group(0) @binding(1) var<storage, read> B : array<vec4<f32>>;\n"
             "@group(0) @binding(2) var<storage, read_write> C : array<vec4<f32>>;\n"
             "@group(0) @binding(3) var<uniform> u : U;\n";
  if (var == "reg") {
    const bool SG = var == "sg";
    if (SG) s = "enable subgroups;\n" + s;
    s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
         "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>, @builtin(subgroup_invocation_id) sid : u32) {\n"
         "  let col4 = g.x * " + to_string(NV) + "u;\n  let row0 = g.y * " + to_string(TM) + "u;\n  let K4 = u.K / 4u; let N4 = u.N / 4u;\n";
    for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) s += "  var c" + to_string(m) + "_" + to_string(n) + " = vec4<f32>(0.0);\n";
    s += "  for (var k4 = 0u; k4 < K4; k4++) {\n";
    for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
      s += "    let b" + to_string(n) + "_" + to_string(j) + " = B[(k4 * 4u + " + to_string(j) + "u) * N4 + col4 + " + to_string(n) + "u];\n";
    for (int m = 0; m < TM; m++) {
      s += "    let a" + to_string(m) + " = A[(row0 + " + to_string(m) + "u) * K4 + k4];\n";
      for (int n = 0; n < NV; n++)
        s += "    c" + to_string(m) + "_" + to_string(n) + " += a" + to_string(m) + ".x * b" + to_string(n) + "_0 + a" + to_string(m) + ".y * b" + to_string(n) + "_1 + a" + to_string(m) + ".z * b" + to_string(n) + "_2 + a" + to_string(m) + ".w * b" + to_string(n) + "_3;\n";
    }
    s += "  }\n";
    for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++)
      s += "  if (row0 + " + to_string(m) + "u < u.M && col4 + " + to_string(n) + "u < N4) { C[wid.z * u.M * N4 + (row0 + " + to_string(m) + "u) * N4 + col4 + " + to_string(n) + "u] = c" + to_string(m) + "_" + to_string(n) + "; }\n";
    s += "}\n";
  } else if (var == "nc4") {  // NC4HW4-like: A4[k4*M + m] (vec4 over 4 consecutive k), C4[n4*M + m]; x = pixels
    // thread handles TM pixels spaced WX apart (coalesced across the wave) x TN=4*NV channels; y picks channel groups
    s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
         "fn main(@builtin(local_invocation_id) l : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
         "  let K4 = u.K / 4u; let N4 = u.N / 4u; let M = u.M;\n"
         "  let pix0 = wid.x * " + to_string(WX * TM) + "u + l.x;\n  let col4 = (wid.y * " + to_string(WY) + "u + l.y) * " + to_string(NV) + "u;\n";
    for (int m = 0; m < TM; m++) for (int c = 0; c < TN; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
    s += "  for (var k4 = 0u; k4 < K4; k4++) {\n";
    for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
      s += "    let b" + to_string(n) + "_" + to_string(j) + " = B[(k4 * 4u + " + to_string(j) + "u) * N4 + col4 + " + to_string(n) + "u];\n";
    for (int m = 0; m < TM; m++) {
      s += "    let a" + to_string(m) + " = A[k4 * M + min(pix0 + " + to_string(m * WX) + "u, M - 1u)];\n";
      for (int j = 0; j < 4; j++) for (int n = 0; n < NV; n++) for (int q = 0; q < 4; q++) {
        string acc = "c" + to_string(m) + "_" + to_string(n * 4 + q);
        s += "    " + acc + " = fma(a" + to_string(m) + "." + "xyzw"[j] + ", b" + to_string(n) + "_" + to_string(j) + "." + "xyzw"[q] + ", " + acc + ");\n";
      }
    }
    s += "  }\n";
    for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++)
      s += "  if (pix0 + " + to_string(m * WX) + "u < M && col4 + " + to_string(n) + "u < N4) { C[wid.z * M * N4 + (col4 + " + to_string(n) + "u) * M + pix0 + " + to_string(m * WX) + "u] = vec4<f32>(c" + to_string(m) + "_" + to_string(n * 4) + ", c" + to_string(m) + "_" + to_string(n * 4 + 1) + ", c" + to_string(m) + "_" + to_string(n * 4 + 2) + ", c" + to_string(m) + "_" + to_string(n * 4 + 3) + "); }\n";
    s += "}\n";
  } else if (var == "tx" || var == "tt") {  // A (and for tt also B) read through textures; scalar accumulators
    bool btex = var == "tt";
    string h = "@group(0) @binding(0) var A : texture_2d<f32>;\n";
    h += btex ? "@group(0) @binding(1) var B : texture_2d<f32>;\n" : "@group(0) @binding(1) var<storage, read> B : array<vec4<f32>>;\n";
    s = "struct U { M: u32, N: u32, K: u32, pad: u32 }\n" + h +
        "@group(0) @binding(2) var<storage, read_write> C : array<vec4<f32>>;\n@group(0) @binding(3) var<uniform> u : U;\n";
    s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
         "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
         "  let col4 = g.x * " + to_string(NV) + "u;\n  let row0 = g.y * " + to_string(TM) + "u;\n  let K4 = u.K / 4u; let N4 = u.N / 4u;\n";
    for (int m = 0; m < TM; m++) for (int c = 0; c < TN; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
    s += "  for (var k4 = 0u; k4 < K4; k4++) {\n";
    for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
      s += btex ? "    let b" + to_string(n) + "_" + to_string(j) + " = textureLoad(B, vec2<u32>(col4 + " + to_string(n) + "u, k4 * 4u + " + to_string(j) + "u), 0);\n"
                : "    let b" + to_string(n) + "_" + to_string(j) + " = B[(k4 * 4u + " + to_string(j) + "u) * N4 + col4 + " + to_string(n) + "u];\n";
    for (int m = 0; m < TM; m++) {
      s += "    let a" + to_string(m) + " = textureLoad(A, vec2<u32>(k4, row0 + " + to_string(m) + "u), 0);\n";
      for (int j = 0; j < 4; j++) for (int n = 0; n < NV; n++) for (int q = 0; q < 4; q++) {
        string acc = "c" + to_string(m) + "_" + to_string(n * 4 + q);
        s += "    " + acc + " = fma(a" + to_string(m) + "." + "xyzw"[j] + ", b" + to_string(n) + "_" + to_string(j) + "." + "xyzw"[q] + ", " + acc + ");\n";
      }
    }
    s += "  }\n";
    for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++)
      s += "  if (row0 + " + to_string(m) + "u < u.M && col4 + " + to_string(n) + "u < N4) { C[wid.z * u.M * N4 + (row0 + " + to_string(m) + "u) * N4 + col4 + " + to_string(n) + "u] = vec4<f32>(c" + to_string(m) + "_" + to_string(n * 4) + ", c" + to_string(m) + "_" + to_string(n * 4 + 1) + ", c" + to_string(m) + "_" + to_string(n * 4 + 2) + ", c" + to_string(m) + "_" + to_string(n * 4 + 3) + "); }\n";
    s += "}\n";
  } else if (var == "p16" || var == "sg") {  // A,B stored as packed f16 (8 values per vec4<u32> load), unpacked to f32, f32 accumulate
    s = "struct U { M: u32, N: u32, K: u32, pad: u32 }\n"
        "@group(0) @binding(0) var<storage, read> A : array<vec4<u32>>;\n"
        "@group(0) @binding(1) var<storage, read> B : array<vec4<u32>>;\n"
        "@group(0) @binding(2) var<storage, read_write> C : array<vec4<f32>>;\n@group(0) @binding(3) var<uniform> u : U;\n";
    const bool SG = var == "sg";
    if (SG) s = "enable subgroups;\n" + s;
    s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
         "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>" + string(SG ? ", @builtin(subgroup_invocation_id) sid : u32" : "") + ") {\n"
         "  let col8 = g.x * " + to_string(NV / 2) + "u;\n  let row0 = g.y * " + to_string(TM) + "u;\n  let K8 = u.K / 8u; let N8 = u.N / 8u; let N4 = u.N / 4u;\n";
    for (int m = 0; m < TM; m++) for (int c = 0; c < TN; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
    if (SG) {  // lane l of a WX-lane row group loads A vec4 (8 k) number kb*WX+l; the group shuffles them through the inner loop
      s += "  let base = sid - lid.x;\n  for (var kb = 0u; kb < K8 / " + to_string(WX) + "u; kb++) {\n";
      for (int m = 0; m < TM; m++) s += "    let mine" + to_string(m) + " = A[min(row0 + " + to_string(m) + "u, u.M - 1u) * K8 + kb * " + to_string(WX) + "u + lid.x];\n";
      s += "    for (var j = 0u; j < " + to_string(WX) + "u; j++) {\n    let k8 = kb * " + to_string(WX) + "u + j;\n";
      for (int m = 0; m < TM; m++) s += "    let a" + to_string(m) + " = subgroupShuffle(mine" + to_string(m) + ", base + j);\n";
    } else {
    s += "  for (var k8 = 0u; k8 < K8; k8++) {\n";
    for (int m = 0; m < TM; m++) s += "    let a" + to_string(m) + " = A[(row0 + " + to_string(m) + "u) * K8 + k8];\n";
    }
    for (int kk = 0; kk < 8; kk++) {
      for (int nb = 0; nb < NV / 2; nb++) {
        string bn = "b" + to_string(kk) + "_" + to_string(nb);
        s += "    let " + bn + " = B[(k8 * 8u + " + to_string(kk) + "u) * N8 + col8 + " + to_string(nb) + "u];\n";
        for (int j = 0; j < 4; j++) s += "    let " + bn + "u" + to_string(j) + " = unpack2x16float(" + bn + "." + "xyzw"[j] + ");\n";
      }
      for (int m = 0; m < TM; m++) {
        string av = "unpack2x16float(a" + to_string(m) + "." + "xyzw"[kk / 2] + ")." + (kk % 2 ? "y" : "x");
        s += "    let av" + to_string(m) + "_" + to_string(kk) + " = " + av + ";\n";
        for (int nb = 0; nb < NV / 2; nb++) for (int j = 0; j < 4; j++) for (int h = 0; h < 2; h++) {
          string acc = "c" + to_string(m) + "_" + to_string(nb * 8 + j * 2 + h);
          s += "    " + acc + " = fma(av" + to_string(m) + "_" + to_string(kk) + ", b" + to_string(kk) + "_" + to_string(nb) + "u" + to_string(j) + "." + (h ? "y" : "x") + ", " + acc + ");\n";
        }
      }
    }
    s += SG ? "  }\n  }\n" : "  }\n";
    for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++)
      s += "  if (row0 + " + to_string(m) + "u < u.M && col8 * 2u + " + to_string(n) + "u < N4) { C[wid.z * u.M * N4 + (row0 + " + to_string(m) + "u) * N4 + col8 * 2u + " + to_string(n) + "u] = vec4<f32>(c" + to_string(m) + "_" + to_string(n * 4) + ", c" + to_string(m) + "_" + to_string(n * 4 + 1) + ", c" + to_string(m) + "_" + to_string(n * 4 + 2) + ", c" + to_string(m) + "_" + to_string(n * 4 + 3) + "); }\n";
    s += "}\n";
  } else if (var == "sc") {  // like reg, but scalar accumulators and scalar fma (no vec4 arithmetic)
    s += "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
         "fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
         "  let col4 = g.x * " + to_string(NV) + "u;\n  let row0 = g.y * " + to_string(TM) + "u;\n  let K4 = u.K / 4u; let N4 = u.N / 4u;\n";
    for (int m = 0; m < TM; m++) for (int c = 0; c < TN; c++) s += "  var c" + to_string(m) + "_" + to_string(c) + " = 0.0;\n";
    s += "  for (var k4 = 0u; k4 < K4; k4++) {\n";
    for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
      s += "    let b" + to_string(n) + "_" + to_string(j) + " = B[(k4 * 4u + " + to_string(j) + "u) * N4 + col4 + " + to_string(n) + "u];\n";
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
    s += "}\n";
  } else {  // shared-memory tiles: A tile [tileM][TK], B tile [TK][tileN/4] vec4
    const int TK = 32;  // inner tile (in scalars)
    s += "const TK4 = " + to_string(TK / 4) + "u;\n"
         "var<workgroup> As : array<vec4<f32>, " + to_string(tileM * TK / 4) + ">;\n"
         "var<workgroup> Bs : array<vec4<f32>, " + to_string(TK * tileN / 4) + ">;\n"
         "@compute @workgroup_size(" + to_string(WX) + "," + to_string(WY) + ",1)\n"
         "fn main(@builtin(local_invocation_id) l : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {\n"
         "  let lid = l.y * " + to_string(WX) + "u + l.x; let nthreads = " + to_string(WX * WY) + "u;\n"
         "  let K4 = u.K / 4u; let N4 = u.N / 4u;\n"
         "  let rowBase = wid.y * " + to_string(tileM) + "u; let colBase4 = wid.x * " + to_string(tileN / 4) + "u;\n";
    for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++) s += "  var c" + to_string(m) + "_" + to_string(n) + " = vec4<f32>(0.0);\n";
    s += "  for (var kt = 0u; kt < K4; kt += TK4) {\n"
         "    for (var i = lid; i < " + to_string(tileM * TK / 4) + "u; i += nthreads) { let r = i / TK4; let c = i % TK4; let gr = rowBase + r;\n"
         "      As[i] = select(vec4<f32>(0.0), A[gr * K4 + kt + c], gr < u.M && kt + c < K4); }\n"
         "    for (var i = lid; i < " + to_string(TK * tileN / 4) + "u; i += nthreads) { let r = i / " + to_string(tileN / 4) + "u; let c = i % " + to_string(tileN / 4) + "u; let gk = kt * 4u + r;\n"
         "      Bs[i] = select(vec4<f32>(0.0), B[gk * N4 + colBase4 + c], gk < u.K && colBase4 + c < N4); }\n"
         "    workgroupBarrier();\n"
         "    for (var k4 = 0u; k4 < TK4; k4++) {\n";
    for (int n = 0; n < NV; n++) for (int j = 0; j < 4; j++)
      s += "      let b" + to_string(n) + "_" + to_string(j) + " = Bs[(k4 * 4u + " + to_string(j) + "u) * " + to_string(tileN / 4) + "u + l.x * " + to_string(NV) + "u + " + to_string(n) + "u];\n";
    for (int m = 0; m < TM; m++) {
      s += "      let a" + to_string(m) + " = As[(l.y * " + to_string(TM) + "u + " + to_string(m) + "u) * TK4 + k4];\n";
      for (int n = 0; n < NV; n++)
        s += "      c" + to_string(m) + "_" + to_string(n) + " += a" + to_string(m) + ".x * b" + to_string(n) + "_0 + a" + to_string(m) + ".y * b" + to_string(n) + "_1 + a" + to_string(m) + ".z * b" + to_string(n) + "_2 + a" + to_string(m) + ".w * b" + to_string(n) + "_3;\n";
    }
    s += "    }\n    workgroupBarrier();\n  }\n";
    for (int m = 0; m < TM; m++) for (int n = 0; n < NV; n++)
      s += "  { let r = rowBase + l.y * " + to_string(TM) + "u + " + to_string(m) + "u; let c4 = colBase4 + l.x * " + to_string(NV) + "u + " + to_string(n) + "u;\n"
           "    if (r < u.M && c4 < N4) { C[wid.z * u.M * N4 + r * N4 + c4] = c" + to_string(m) + "_" + to_string(n) + "; } }\n";
    s += "}\n";
  }
  wgpu::ShaderSourceWGSL w{}; w.code = {s.data(), s.size()};
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  wgpu::ComputePipeline pl = dev.CreateComputePipeline(&cpd);
  auto mk = [&](size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = n; b.usage = u; return dev.CreateBuffer(&b); };
  auto St = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  const bool P16 = var == "p16" || var == "sg"; wgpu::Buffer bA = mk((size_t)M * K * (P16 ? 2 : 4), St), bB = mk((size_t)K * N * (P16 ? 2 : 4), St), bC = mk((size_t)REPS * M * N * 4, St), bU = mk(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  std::vector<float> a((size_t)M * K), b((size_t)K * N);
  for (size_t i = 0; i < a.size(); i++) a[i] = ((i * 7919) % 1000) / 1000.f - .5f;
  for (size_t i = 0; i < b.size(); i++) b[i] = ((i * 104729) % 1000) / 1000.f - .5f;
  uint32_t uni[4] = {M, N, K, 0};
  std::vector<float> a_up = a;
  if (var == "nc4") { for (uint32_t k4 = 0; k4 < K / 4; k4++) for (uint32_t m = 0; m < M; m++) for (int q = 0; q < 4; q++) a_up[((size_t)k4 * M + m) * 4 + q] = a[(size_t)m * K + 4 * k4 + q]; }
  if (P16) { std::vector<uint16_t> ah(a.size()), bh(b.size());
    for (size_t i = 0; i < a.size(); i++) { ah[i] = f2h(a[i]); a[i] = h2f(ah[i]); }
    for (size_t i = 0; i < b.size(); i++) { bh[i] = f2h(b[i]); b[i] = h2f(bh[i]); }
    dev.GetQueue().WriteBuffer(bA, 0, ah.data(), ah.size() * 2); dev.GetQueue().WriteBuffer(bB, 0, bh.data(), bh.size() * 2);
  } else { dev.GetQueue().WriteBuffer(bA, 0, a_up.data(), a_up.size() * 4); dev.GetQueue().WriteBuffer(bB, 0, b.data(), b.size() * 4); }
  {} dev.GetQueue().WriteBuffer(bU, 0, uni, 16);
  wgpu::Texture tA, tB;
  auto mkTex = [&](uint32_t w, uint32_t h, const std::vector<float>& data) {
    wgpu::TextureDescriptor td{}; td.size = {w, h, 1}; td.format = wgpu::TextureFormat::RGBA32Float; td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    wgpu::Texture t = dev.CreateTexture(&td);
    wgpu::TexelCopyTextureInfo di{}; di.texture = t; wgpu::TexelCopyBufferLayout lay{}; lay.bytesPerRow = w * 16; lay.rowsPerImage = h; wgpu::Extent3D ext = {w, h, 1};
    dev.GetQueue().WriteTexture(&di, data.data(), data.size() * 4, &lay, &ext);
    return t;
  };
  if (var == "tx" || var == "tt") tA = mkTex(K / 4, M, a);
  if (var == "tt") tB = mkTex(N / 4, K, b);
  wgpu::BindGroupEntry e[4]; e[0].binding = 0; e[1].binding = 1;
  if (var == "tx" || var == "tt") e[0].textureView = tA.CreateView(); else { e[0].buffer = bA; e[0].size = bA.GetSize(); }
  if (var == "tt") e[1].textureView = tB.CreateView(); else { e[1].buffer = bB; e[1].size = bB.GetSize(); }
  e[2].binding = 2; e[2].buffer = bC; e[2].size = bC.GetSize(); e[3].binding = 3; e[3].buffer = bU; e[3].size = 16;
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 4; bgd.entries = e; wgpu::BindGroup bg = dev.CreateBindGroup(&bgd);
  uint32_t gx = (N + tileN - 1) / tileN, gy = (M + tileM - 1) / tileM;
  if (var == "nc4") { gx = (M + WX * TM - 1) / (WX * TM); gy = ((N / 4) + WY * NV - 1) / (WY * NV); }
  auto run = [&]() {
    wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
    { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0, bg); p.DispatchWorkgroups(gx, gy, REPS); p.End(); }
    wgpu::CommandBuffer cb = enc.Finish();
    auto t0 = std::chrono::steady_clock::now();
    dev.GetQueue().Submit(1, &cb);
    inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  run(); run();
  std::vector<double> t; for (int i = 0; i < 15; i++) t.push_back(run());
  std::sort(t.begin(), t.end());
  // correctness of slice 0 against a CPU reference on a sample of outputs
  wgpu::Buffer rb = mk((size_t)M * N * 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  { wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); enc.CopyBufferToBuffer(bC, 0, rb, 0, (size_t)M * N * 4); wgpu::CommandBuffer cb = enc.Finish(); dev.GetQueue().Submit(1, &cb); }
  bool ok = false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read, 0, (size_t)M * N * 4, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::MapAsyncStatus st, wgpu::StringView) { ok = st == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
  double maxerr = 0;
  if (ok) { const float* c = (const float*)rb.GetConstMappedRange(0, (size_t)M * N * 4);
    for (int q = 0; q < 200; q++) { size_t r = (q * 2654435761u) % M, cc = (q * 40503u + 7) % N; double ref = 0; for (uint32_t k = 0; k < K; k++) ref += (double)a[r * K + k] * b[(size_t)k * N + cc];
      float got = var == "nc4" ? c[((size_t)(cc / 4) * M + r) * 4 + cc % 4] : c[r * N + cc]; maxerr = std::max(maxerr, std::fabs(ref - got)); } }
  double flops = 2.0 * M * N * K * REPS;
  printf("%-4s M=%u N=%u K=%u reps=%u TM=%d NV=%d wg=%dx%d  best=%.3f ms  %.0f GFLOPS  (per gemm %.3f ms)  maxerr=%.1e\n", var.c_str(), M, N, K, REPS, TM, NV, WX, WY, t[0], flops / t[0] / 1e6, t[0] / REPS, maxerr);
  return 0;
}
