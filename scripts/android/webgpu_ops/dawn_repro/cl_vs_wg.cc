// Adreno OpenCL vs WebGPU/Vulkan on the same work: cl_vs_wg info|warm S|peak|bw|gemm|conv
//   info  -- OpenCL platform/device/extension dump
//   warm  -- S seconds of continuous FMA load (put the GPU at its working clock before timing)
//   peak  -- FMA throughput: float / half / half2 / half4 (compare peak.cc: 986 f32, 310 f16 GFLOPS)
//   bw    -- load bandwidth: buffer float4 / half4 / half8, image2d RGBA32F / RGBA16F (compare bw.cc)
//   gemm  -- register-tile GEMM (TM=8 NV=2 scalar accumulators) in float/half, buffer/image (compare gemm.cc sc/p16)
//   conv  -- direct 3x3 pad-1 NHWC conv as an implicit GEMM, buffer and image2d NHWC4 (compare conv_alt.cc direct)
// OpenCL is loaded with dlopen (libOpenCL.so of the vendor partition); only the Khronos headers are needed to build:
//   clang++ -std=c++17 -O2 -I<dir with CL/*.h> cl_vs_wg.cc -o cl_vs_wg -ldl
#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <dlfcn.h>
#include <unistd.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
using std::string;
using std::to_string;
using std::vector;

#define CLFN(ret, name, args) static ret(CL_API_CALL* p_##name) args;
CLFN(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))
CLFN(cl_int, clGetPlatformInfo, (cl_platform_id, cl_platform_info, size_t, void*, size_t*))
CLFN(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*))
CLFN(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void*, size_t*))
CLFN(cl_context, clCreateContext, (const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*))
CLFN(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*))
CLFN(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
CLFN(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void(CL_CALLBACK*)(cl_program, void*), void*))
CLFN(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*))
CLFN(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))
CLFN(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))
CLFN(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*))
CLFN(cl_int, clFinish, (cl_command_queue))
CLFN(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void*, cl_int*))
CLFN(cl_mem, clCreateImage, (cl_context, cl_mem_flags, const cl_image_format*, const cl_image_desc*, void*, cl_int*))
CLFN(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))
CLFN(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))
CLFN(cl_int, clReleaseMemObject, (cl_mem))
CLFN(cl_int, clReleaseKernel, (cl_kernel))
CLFN(cl_int, clReleaseProgram, (cl_program))
CLFN(cl_int, clGetKernelWorkGroupInfo, (cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*))

static void* lib;
#define LOAD(name) \
  p_##name = (decltype(p_##name))dlsym(lib, #name); \
  if (!p_##name) { fprintf(stderr, "missing %s\n", #name); exit(2); }

static cl_context ctx;
static cl_command_queue q;
static cl_device_id dev;
static string g_exts;

static double nowms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static bool has_ext(const char* e) { return g_exts.find(e) != string::npos; }
#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "CL error %d at %s:%d (%s)\n", e_, __FILE__, __LINE__, #x); exit(3); } } while (0)

struct Prog { cl_program p = nullptr; bool ok = false; };
static cl_kernel build(const string& src, const char* name, const string& opts = "", bool quiet = false) {
  const char* s = src.c_str(); size_t len = src.size(); cl_int e;
  cl_program p = p_clCreateProgramWithSource(ctx, 1, &s, &len, &e);
  e = p_clBuildProgram(p, 1, &dev, opts.c_str(), nullptr, nullptr);
  if (e != CL_SUCCESS) {
    if (!quiet) {
      size_t n = 0; p_clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
      string log(n, 0); p_clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
      fprintf(stderr, "build failed (%s): %s\n", name, log.substr(0, 2000).c_str());
    }
    return nullptr;
  }
  cl_kernel k = p_clCreateKernel(p, name, &e);
  if (e != CL_SUCCESS) { fprintf(stderr, "create kernel %s failed %d\n", name, e); return nullptr; }
  return k;
}
template <class T> static void arg(cl_kernel k, int i, const T& v) { CK(p_clSetKernelArg(k, i, sizeof(T), &v)); }

static uint16_t f2h(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff; int exp = ((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return (uint16_t)sign; if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  uint16_t h = (uint16_t)(sign | (exp << 10) | (mant >> 13)); if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) h++; return h; }
static float h2f(uint16_t h) { uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023, x; if (e == 0) { if (!m) x = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; x = s | ((e + 112) << 23) | (m << 13); } } else x = s | ((e + 112) << 23) | (m << 13); float f; memcpy(&f, &x, 4); return f; }

// Time `launch(reps)` (enqueues reps kernel executions or one reps-deep NDRange); warm-up, then n timed batches.
struct Stat { double best, med; };
static Stat timeit(const std::function<void()>& launch, int n = 15, int warm = 3) {
  for (int i = 0; i < warm; i++) { launch(); p_clFinish(q); }
  vector<double> t;
  for (int i = 0; i < n; i++) { double t0 = nowms(); launch(); p_clFinish(q); t.push_back(nowms() - t0); }
  std::sort(t.begin(), t.end());
  return {t[0], t[t.size() / 2]};
}

static void warm_gpu(double seconds) {
  string s = "__kernel void w(__global float* o, int it, float a, float b) { float c0=get_global_id(0), c1=c0+1, c2=c0+2, c3=c0+3, c4=c0+4, c5=c0+5, c6=c0+6, c7=c0+7;\n"
             " for (int i = 0; i < it; i++) { c0=mad(c0,a,b); c1=mad(c1,a,b); c2=mad(c2,a,b); c3=mad(c3,a,b); c4=mad(c4,a,b); c5=mad(c5,a,b); c6=mad(c6,a,b); c7=mad(c7,a,b);} o[get_global_id(0)] = c0+c1+c2+c3+c4+c5+c6+c7; }";
  cl_kernel k = build(s, "w"); cl_int e; cl_mem o = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1 << 22, nullptr, &e);
  int dit = getenv("DBG_IT") ? atoi(getenv("DBG_IT")) : 4096; int dgl = getenv("DBG_G") ? atoi(getenv("DBG_G")) : (1 << 18);
  arg(k, 0, o); arg(k, 1, dit); arg(k, 2, 0.999f); arg(k, 3, 0.001f);
  size_t g = dgl, l = 64;
  double t0 = nowms();
  while (nowms() - t0 < seconds * 1000) { double a = nowms(); for (int i = 0; i < 4; i++) CK(p_clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr)); p_clFinish(q); (void)a; }
  p_clReleaseMemObject(o);
}

// ---------------------------------------------------------------- info
static void do_info() {
  char buf[16384]; size_t n;
  CK(p_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof buf, buf, &n)); printf("device: %s\n", buf);
  CK(p_clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof buf, buf, &n)); printf("version: %s\n", buf);
  CK(p_clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof buf, buf, &n)); printf("driver: %s\n", buf);
  cl_uint cu = 0, mhz = 0; CK(p_clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &cu, nullptr)); CK(p_clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, 4, &mhz, nullptr));
  size_t wg = 0; CK(p_clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wg, &wg, nullptr));
  cl_ulong lm = 0, gm = 0; CK(p_clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, 8, &lm, nullptr)); CK(p_clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &gm, nullptr));
  size_t iw = 0, ih = 0; p_clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof iw, &iw, nullptr); p_clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof ih, &ih, nullptr);
  printf("compute units %u, clock %u MHz, max wg %zu, local mem %llu KB, global mem %llu MB, image2d max %zux%zu\n", cu, mhz, wg, (unsigned long long)lm / 1024, (unsigned long long)gm >> 20, iw, ih);
  cl_device_fp_config fc = 0; p_clGetDeviceInfo(dev, 0x1033 /* CL_DEVICE_HALF_FP_CONFIG */, sizeof fc, &fc, nullptr); printf("half fp config: 0x%llx\n", (unsigned long long)fc);
  printf("extensions:\n");
  string e = g_exts; size_t pos = 0;
  while (pos < e.size()) { size_t sp = e.find(' ', pos); if (sp == string::npos) sp = e.size(); if (sp > pos) printf("  %s\n", e.substr(pos, sp - pos).c_str()); pos = sp + 1; }
}

// ---------------------------------------------------------------- peak FMA
static string fma_kernel(const string& T, int W, int chains, bool fma_builtin) {
  // T scalar type (float|half), W vector width (1,2,4); chains independent accumulators of T{W}
  string V = W == 1 ? T : T + to_string(W);
  string s = string(T == "half" ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" : "") + "__kernel void k(__global " + V + "* o, int it, " + V + " a, " + V + " b) {\n  " + V + " ";
  for (int c = 0; c < chains; c++) s += (c ? ", c" : "c") + to_string(c) + " = (" + V + ")((" + T + ")(get_global_id(0) + " + to_string(c) + "))";
  s += ";\n  for (int i = 0; i < it; i++) {\n";
  for (int c = 0; c < chains; c++) s += fma_builtin ? "    c" + to_string(c) + " = fma(c" + to_string(c) + ", a, b);\n" : "    c" + to_string(c) + " = mad(c" + to_string(c) + ", a, b);\n";
  s += "  }\n  " + V + " r = c0;\n";
  for (int c = 1; c < chains; c++) s += "  r += c" + to_string(c) + ";\n";
  s += "  o[get_global_id(0)] = r;\n}\n";
  return s;
}

static void do_peak() {
  struct Cfg { const char* T; int W, chains; };
  vector<Cfg> cfgs = {{"float", 1, 64}, {"float", 1, 32}, {"float", 2, 32}, {"float", 4, 16}, {"half", 1, 64}, {"half", 2, 32}, {"half", 4, 16}, {"half", 4, 32}};
  size_t l = 256;  // 4096 workgroups of 256, like peak.cc
  // NOTE: OpenCL fma() is emulated (~1000x slower) on this driver, so the fma variants run 1/256 of the work and only mad/a*b+c runs full size.
  for (string opts : {string(""), string("-cl-fast-relaxed-math"), string("-cl-mad-enable")})
    for (auto c : cfgs) {
      if (string(c.T) == "half" && !has_ext("cl_khr_fp16")) { printf("half unsupported\n"); continue; }
      for (bool use_fma : {false, true}) {
        if (use_fma && opts != "") continue;
        size_t g = use_fma ? (1u << 12) : (1u << 20); int iters = use_fma ? 4 : 1024;
        cl_kernel k = build(fma_kernel(c.T, c.W, c.chains, use_fma), "k", opts, true);
        if (!k) { printf("%-5s w=%d chains=%-2d %s: build failed\n", c.T, c.W, c.chains, use_fma ? "fma" : "mad"); continue; }
        cl_int e; size_t esz = (string(c.T) == "half" ? 2 : 4) * c.W;
        cl_mem o = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, g * esz, nullptr, &e);
        arg(k, 0, o); arg(k, 1, iters);
        if (string(c.T) == "half") { cl_half a = f2h(0.999f), b = f2h(0.001f); char av[8], bv[8]; for (int i = 0; i < c.W; i++) { memcpy(av + 2 * i, &a, 2); memcpy(bv + 2 * i, &b, 2); } CK(p_clSetKernelArg(k, 2, 2 * c.W, av)); CK(p_clSetKernelArg(k, 3, 2 * c.W, bv)); }
        else { float a[4] = {0.999f, 0.999f, 0.999f, 0.999f}, b[4] = {0.001f, 0.001f, 0.001f, 0.001f}; CK(p_clSetKernelArg(k, 2, 4 * c.W, a)); CK(p_clSetKernelArg(k, 3, 4 * c.W, b)); }
        Stat st = timeit([&] { CK(p_clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr)); }, 12);
        double flops = 2.0 * g * iters * (double)c.chains * c.W;
        printf("%-5s w=%d chains=%-2d %-3s %-22s best=%.2f ms (%.0f GFLOPS)  median=%.2f ms (%.0f GFLOPS)\n", c.T, c.W, c.chains, use_fma ? "fma" : "mad", opts.empty() ? "(none)" : opts.c_str(), st.best, flops / st.best / 1e6, st.med, flops / st.med / 1e6);
        p_clReleaseMemObject(o); p_clReleaseKernel(k);
      }
    }
}

// ---------------------------------------------------------------- load bandwidth
static void do_bw() {
  // 8 independent vec4-sized loads per iteration from a small working set (kb KB), like bw.cc
  size_t g = 1u << 19, l = 64;  // 8192 workgroups x 64
  int iters = 256;
  for (int kb : {4, 16, 64, 256, 2048}) {
    struct V { const char* name; int bytes_per_load; };
    auto run = [&](const char* name, const string& body_decl, const string& load_expr, int bytes_per_load, bool image, cl_channel_type ct, const string& opts) {
      uint32_t nvec = (uint32_t)((size_t)kb * 1024 / bytes_per_load);  // power of two
      string src = string("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\nconstant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n") +
                   "__kernel void k(" + body_decl + ", __global float* dst, int it) {\n  float4 acc = 0;\n  uint g = get_global_id(0);\n  for (int i = 0; i < it; i++) {\n    uint b = g + i * 8u;\n";
      for (int j = 0; j < 8; j++) {
        string idx = "((b + " + to_string(j) + "u * 3u) & " + to_string(nvec - 1) + "u)";
        src += "    acc += " + string(load_expr) + ";\n";
        size_t p; while ((p = src.find("$I", src.rfind("acc +="))) != string::npos) src.replace(p, 2, idx);
        size_t px; while ((px = src.find("$X", src.rfind("acc +="))) != string::npos) src.replace(px, 2, idx + " & 255u");
        size_t py; while ((py = src.find("$Y", src.rfind("acc +="))) != string::npos) src.replace(py, 2, "(" + idx + " >> 8u) % " + to_string(std::max(1u, nvec / 256)) + "u");
      }
      src += "  }\n  dst[g] = acc.x + acc.y + acc.z + acc.w;\n}\n";
      cl_kernel k = build(src, "k", opts, true);
      if (!k) { printf("%-28s kb=%-5d build failed\n", name, kb); return; }
      cl_int e; cl_mem out = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, g * 4, nullptr, &e), in;
      if (image) {
        cl_image_format fmt = {CL_RGBA, (cl_channel_type)ct}; cl_image_desc d{}; d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = std::min<size_t>(nvec, 256); d.image_height = std::max(1u, nvec / 256);
        vector<char> init(d.image_width * d.image_height * bytes_per_load, 0);
        in = p_clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &d, init.data(), &e);
      } else {
        vector<char> init((size_t)nvec * bytes_per_load, 0);
        in = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, init.size(), init.data(), &e);
      }
      if (e != CL_SUCCESS) { printf("%-28s kb=%-5d alloc failed %d\n", name, kb, e); return; }
      arg(k, 0, in); arg(k, 1, out); arg(k, 2, iters);
      Stat st = timeit([&] { CK(p_clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr)); }, 12);
      double bytes = (double)g * iters * 8 * bytes_per_load;
      printf("%-28s working set %5d KB  best=%.2f ms  %.0f GB/s  (%.0f G loads/s)   median %.0f GB/s\n", name, kb, st.best, bytes / st.best / 1e6, (double)g * iters * 8 / st.best / 1e6, bytes / st.med / 1e6);
      p_clReleaseMemObject(in); p_clReleaseMemObject(out); p_clReleaseKernel(k);
    };
    run("buffer float4", "__global const float4* src", "src[$I]", 16, false, 0, "");
    run("buffer half4 (vload_half4)", "__global const half* src", "vload_half4($I, src)", 8, false, 0, "");
    run("buffer half4 (cvt half4)", "__global const half4* src", "convert_float4(src[$I])", 8, false, 0, "");
    run("buffer half8 (2x cvt half4)", "__global const half8* src", "convert_float4(src[$I].s0123) + convert_float4(src[$I].s4567)", 16, false, 0, "");
    run("buffer uint4 (raw 16 B)", "__global const uint4* src", "as_float4(src[$I])", 16, false, 0, "");
    run("image RGBA32F read_imagef", "__read_only image2d_t src", "read_imagef(src, smp, (int2)($X, $Y))", 16, true, CL_FLOAT, "");
    run("image RGBA16F read_imagef", "__read_only image2d_t src", "read_imagef(src, smp, (int2)($X, $Y))", 8, true, CL_HALF_FLOAT, "");
    run("image RGBA16F read_imageh", "__read_only image2d_t src", "convert_float4(read_imageh(src, smp, (int2)($X, $Y)))", 8, true, CL_HALF_FLOAT, "");
  }
}

// ---------------------------------------------------------------- GEMM
struct Var { const char* name; bool f16; bool bimg; bool hacc; };

static string gemm_src(const Var& v, int TM, int NV) {
  string H = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\nconstant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n";
  string AT = v.f16 ? "half4" : "float4";
  string acc = v.hacc ? "half" : "float", acc4 = v.hacc ? "half4" : "float4";
  string s = H + "__kernel void k(__global const " + AT + "* A, " + (v.bimg ? "__read_only image2d_t B" : "__global const " + AT + "* B") + ", __global float4* C, int M, int N4, int K4) {\n"
             "  int col4 = get_global_id(0) * " + to_string(NV) + ";\n  int row0 = get_global_id(1) * " + to_string(TM) + ";\n  int z = get_global_id(2);\n";
  for (int m = 0; m < TM; m++) for (int c = 0; c < 4 * NV; c++) s += "  " + acc + " c" + to_string(m) + "_" + to_string(c) + " = 0;\n";
  for (int m = 0; m < TM; m++) s += "  int r" + to_string(m) + " = min(row0 + " + to_string(m) + ", M - 1);\n";
  s += "  for (int k4 = 0; k4 < K4; k4++) {\n";
  for (int m = 0; m < TM; m++) s += "    " + acc4 + " a" + to_string(m) + " = " + (v.f16 && !v.hacc ? "convert_float4(A[r" + to_string(m) + " * K4 + k4])" : "A[r" + to_string(m) + " * K4 + k4]") + ";\n";
  for (int kk = 0; kk < 4; kk++) {
    for (int n = 0; n < NV; n++) {
      string bn = "b" + to_string(kk) + "_" + to_string(n);
      string ld;
      if (v.bimg) ld = v.hacc ? "read_imageh(B, smp, (int2)(col4 + " + to_string(n) + ", k4 * 4 + " + to_string(kk) + "))" : "read_imagef(B, smp, (int2)(col4 + " + to_string(n) + ", k4 * 4 + " + to_string(kk) + "))";
      else ld = v.f16 && !v.hacc ? "convert_float4(B[(k4 * 4 + " + to_string(kk) + ") * N4 + col4 + " + to_string(n) + "])" : "B[(k4 * 4 + " + to_string(kk) + ") * N4 + col4 + " + to_string(n) + "]";
      s += "    " + acc4 + " " + bn + " = " + ld + ";\n";
    }
    for (int m = 0; m < TM; m++)
      for (int n = 0; n < NV; n++)
        for (int j = 0; j < 4; j++)
          s += "    c" + to_string(m) + "_" + to_string(n * 4 + j) + " = mad(a" + to_string(m) + ".s" + to_string(kk) + ", b" + to_string(kk) + "_" + to_string(n) + ".s" + to_string(j) + ", c" + to_string(m) + "_" + to_string(n * 4 + j) + ");\n";
  }
  s += "  }\n";
  for (int m = 0; m < TM; m++)
    for (int n = 0; n < NV; n++) {
      string vec = "(float4)((float)c" + to_string(m) + "_" + to_string(n * 4) + ", (float)c" + to_string(m) + "_" + to_string(n * 4 + 1) + ", (float)c" + to_string(m) + "_" + to_string(n * 4 + 2) + ", (float)c" + to_string(m) + "_" + to_string(n * 4 + 3) + ")";
      s += "  if (row0 + " + to_string(m) + " < M) C[(long)z * M * N4 + (long)(row0 + " + to_string(m) + ") * N4 + col4 + " + to_string(n) + "] = " + vec + ";\n";
    }
  return s + "}\n";
}

static void do_gemm(const vector<string>& only) {
  struct Shape { int M, N, K, reps; };
  vector<Shape> shapes = {{784, 512, 128, 32}, {3136, 256, 64, 32}, {196, 256, 2304, 12}};
  vector<Var> vars = {{"f32 buf", false, false, false}, {"f32 img(B)", false, true, false}, {"f16 buf, f32 acc", true, false, false}, {"f16 buf, f16 acc", true, false, true},
                      {"f16 img(B), f32 acc", true, true, false}, {"f16 img(B), f16 acc", true, true, true}};
  struct Lws { int x, y; };
  vector<Lws> lwss = {{16, 4}, {32, 4}, {8, 8}, {32, 2}, {64, 1}, {16, 8}};
  int TMs[] = {8, 4};
  for (auto sh : shapes) {
    int M = sh.M, N = sh.N, K = sh.K, N4 = N / 4, K4 = K / 4;
    vector<float> a((size_t)M * K), b((size_t)K * N);
    for (size_t i = 0; i < a.size(); i++) a[i] = ((i * 7919) % 1000) / 1000.f - .5f;
    for (size_t i = 0; i < b.size(); i++) b[i] = ((i * 104729) % 1000) / 1000.f - .5f;
    vector<uint16_t> ah(a.size()), bh(b.size());
    vector<float> ar = a, br = b;  // values as stored (half-rounded for the f16 variants)
    for (size_t i = 0; i < a.size(); i++) ah[i] = f2h(a[i]);
    for (size_t i = 0; i < b.size(); i++) bh[i] = f2h(b[i]);
    vector<float> ahr(a.size()), bhr(b.size());
    for (size_t i = 0; i < a.size(); i++) ahr[i] = h2f(ah[i]);
    for (size_t i = 0; i < b.size(); i++) bhr[i] = h2f(bh[i]);
    for (auto& v : vars) {
      if (!only.empty() && std::find(only.begin(), only.end(), v.name) == only.end()) continue;
      if (v.f16 && !has_ext("cl_khr_fp16")) continue;
      double bestT = 1e30, medT = 0; int bestTM = 0; Lws bestL{};
      float maxerr = 0; string note;
      for (int TM : TMs) {
        int NV = 2;
        cl_kernel k = build(gemm_src(v, TM, NV), "k", "", true);
        if (!k) { note = "build failed"; continue; }
        cl_int e;
        const vector<float>& ra = v.f16 ? ahr : ar;
        const vector<float>& rb = v.f16 ? bhr : br;
        size_t es = v.f16 ? 2 : 4;
        cl_mem bA = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, a.size() * es, nullptr, &e);
        CK(p_clEnqueueWriteBuffer(q, bA, CL_TRUE, 0, a.size() * es, v.f16 ? (const void*)ah.data() : (const void*)a.data(), 0, nullptr, nullptr));
        cl_mem bB;
        if (v.bimg) {
          cl_image_format fmt = {CL_RGBA, (cl_channel_type)(v.f16 ? CL_HALF_FLOAT : CL_FLOAT)}; cl_image_desc d{}; d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = N4; d.image_height = K;
          bB = p_clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &d, v.f16 ? (void*)bh.data() : (void*)b.data(), &e);
        } else {
          bB = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, b.size() * es, nullptr, &e);
          CK(p_clEnqueueWriteBuffer(q, bB, CL_TRUE, 0, b.size() * es, v.f16 ? (const void*)bh.data() : (const void*)b.data(), 0, nullptr, nullptr));
        }
        if (e != CL_SUCCESS) { note = "alloc failed"; continue; }
        cl_mem bC = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)sh.reps * M * N * 4, nullptr, &e);
        arg(k, 0, bA); arg(k, 1, bB); arg(k, 2, bC); arg(k, 3, M); arg(k, 4, N4); arg(k, 5, K4);
        for (auto l : lwss) {
          if (l.x * l.y > 256) continue;
          size_t gx = ((N4 / NV) + l.x - 1) / l.x * l.x, gy = (((M + TM - 1) / TM) + l.y - 1) / l.y * l.y;
          size_t g[3] = {gx, gy, (size_t)sh.reps}, ll[3] = {(size_t)l.x, (size_t)l.y, 1};
          if (p_clEnqueueNDRangeKernel(q, k, 3, nullptr, g, ll, 0, nullptr, nullptr) != CL_SUCCESS) continue;
          p_clFinish(q);
          Stat st = timeit([&] { p_clEnqueueNDRangeKernel(q, k, 3, nullptr, g, ll, 0, nullptr, nullptr); }, 9, 1);
          if (st.best < bestT) {
            bestT = st.best; medT = st.med; bestTM = TM; bestL = l;
            vector<float> c((size_t)M * N);
            CK(p_clEnqueueReadBuffer(q, bC, CL_TRUE, 0, c.size() * 4, c.data(), 0, nullptr, nullptr));
            maxerr = 0;
            for (int t = 0; t < 200; t++) { size_t r = (t * 2654435761u) % M, cc = (t * 40503u + 7) % N; double ref = 0; for (int kk = 0; kk < K; kk++) ref += (double)ra[r * K + kk] * rb[(size_t)kk * N + cc]; maxerr = std::max(maxerr, (float)std::fabs(ref - c[r * N + cc])); }
          }
        }
        p_clReleaseMemObject(bA); p_clReleaseMemObject(bB); p_clReleaseMemObject(bC); p_clReleaseKernel(k);
      }
      double flops = 2.0 * M * N * K * sh.reps;
      if (bestT > 1e29) printf("OpenCL %-22s M=%d N=%d K=%d: %s\n", v.name, M, N, K, note.c_str());
      else printf("OpenCL %-22s M=%d N=%d K=%d TM=%d NV=2 wg=%dx%d  best=%.3f ms (%.0f GFLOPS)  median %.0f GFLOPS  maxerr=%.1e\n", v.name, M, N, K, bestTM, bestL.x, bestL.y, bestT, flops / bestT / 1e6, flops / medT / 1e6, maxerr);
    }
  }
}

// ---------------------------------------------------------------- direct 3x3 conv (implicit GEMM)
struct CVar { const char* name; bool f16; bool ximg; bool bimg; bool hacc; };

static string conv_src(const CVar& v, int TM, int NV, int H, int W, int C) {
  int C4 = C / 4;
  string AT = v.f16 ? "half4" : "float4";
  string acc = v.hacc ? "half" : "float", acc4 = v.hacc ? "half4" : "float4";
  string s = string("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\nconstant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\nconstant sampler_t smpn = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n") +
             "__kernel void k(" + (v.ximg ? "__read_only image2d_t X" : "__global const " + AT + "* X") + ", " + (v.bimg ? "__read_only image2d_t B" : "__global const " + AT + "* B") + ", __global float4* Y) {\n"
             "  const int H = " + to_string(H) + ", W = " + to_string(W) + ", C4 = " + to_string(C4) + ", M = " + to_string(H * W) + ";\n"
             "  int col4 = get_global_id(0) * " + to_string(NV) + ";\n  int row0 = get_global_id(1) * " + to_string(TM) + ";\n";
  for (int m = 0; m < TM; m++) s += "  int p" + to_string(m) + " = min(row0 + " + to_string(m) + ", M - 1); int y" + to_string(m) + " = p" + to_string(m) + " / W; int x" + to_string(m) + " = p" + to_string(m) + " % W;\n";
  for (int m = 0; m < TM; m++) for (int c = 0; c < 4 * NV; c++) s += "  " + acc + " c" + to_string(m) + "_" + to_string(c) + " = 0;\n";
  s += "  for (int tap = 0; tap < 9; tap++) {\n    int dy = tap / 3 - 1, dx = tap % 3 - 1;\n";
  for (int m = 0; m < TM; m++) s += "    int iy" + to_string(m) + " = y" + to_string(m) + " + dy, ix" + to_string(m) + " = x" + to_string(m) + " + dx;\n";
  if (!v.ximg)
    for (int m = 0; m < TM; m++) s += "    bool ok" + to_string(m) + " = iy" + to_string(m) + " >= 0 && iy" + to_string(m) + " < H && ix" + to_string(m) + " >= 0 && ix" + to_string(m) + " < W;\n    int o" + to_string(m) + " = (iy" + to_string(m) + " * W + ix" + to_string(m) + ") * C4;\n";
  s += "    for (int c4 = 0; c4 < C4; c4++) {\n";
  for (int m = 0; m < TM; m++) {
    string ld;
    if (v.ximg) ld = v.hacc ? "read_imageh(X, smp, (int2)(ix" + to_string(m) + " * C4 + c4, iy" + to_string(m) + "))" : "read_imagef(X, smp, (int2)(ix" + to_string(m) + " * C4 + c4, iy" + to_string(m) + "))";
    else ld = string("ok") + to_string(m) + " ? " + (v.f16 && !v.hacc ? "convert_float4(X[o" + to_string(m) + " + c4])" : "X[o" + to_string(m) + " + c4]") + " : (" + acc4 + ")(0)";
    s += "      " + acc4 + " a" + to_string(m) + " = " + ld + ";\n";
  }
  for (int kk = 0; kk < 4; kk++) {
    for (int n = 0; n < NV; n++) {
      string ld;
      if (v.bimg) ld = string(v.hacc ? "read_imageh" : "read_imagef") + "(B, smpn, (int2)(col4 + " + to_string(n) + ", tap * " + to_string(C) + " + c4 * 4 + " + to_string(kk) + "))";
      else ld = v.f16 && !v.hacc ? "convert_float4(B[(tap * " + to_string(C) + " + c4 * 4 + " + to_string(kk) + ") * C4 + col4 + " + to_string(n) + "])" : "B[(tap * " + to_string(C) + " + c4 * 4 + " + to_string(kk) + ") * C4 + col4 + " + to_string(n) + "]";
      s += "      " + acc4 + " b" + to_string(kk) + "_" + to_string(n) + " = " + ld + ";\n";
    }
    for (int m = 0; m < TM; m++)
      for (int n = 0; n < NV; n++)
        for (int j = 0; j < 4; j++)
          s += "      c" + to_string(m) + "_" + to_string(n * 4 + j) + " = mad(a" + to_string(m) + ".s" + to_string(kk) + ", b" + to_string(kk) + "_" + to_string(n) + ".s" + to_string(j) + ", c" + to_string(m) + "_" + to_string(n * 4 + j) + ");\n";
  }
  s += "    }\n  }\n";
  for (int m = 0; m < TM; m++)
    for (int n = 0; n < NV; n++)
      s += "  if (row0 + " + to_string(m) + " < M) Y[(row0 + " + to_string(m) + ") * C4 + col4 + " + to_string(n) + "] = (float4)((float)c" + to_string(m) + "_" + to_string(n * 4) + ", (float)c" + to_string(m) + "_" + to_string(n * 4 + 1) + ", (float)c" + to_string(m) + "_" + to_string(n * 4 + 2) + ", (float)c" + to_string(m) + "_" + to_string(n * 4 + 3) + ");\n";
  return s + "}\n";
}

static void do_conv() {
  struct P { int C, H; };
  vector<P> probs = {{64, 56}, {256, 14}};
  vector<CVar> vars = {{"f32 buf", false, false, false, false}, {"f32 img(X)", false, true, false, false}, {"f32 img(X,B)", false, true, true, false},
                       {"f16 buf, f32 acc", true, false, false, false}, {"f16 img(X,B), f32 acc", true, true, true, false}, {"f16 img(X,B), f16 acc", true, true, true, true},
                       {"f16 buf, f16 acc", true, false, false, true}};
  struct Lws { int x, y; };
  vector<Lws> lwss = {{16, 4}, {32, 4}, {8, 8}, {32, 2}, {16, 8}, {64, 1}, {8, 4}};
  for (auto pr : probs) {
    int C = pr.C, H = pr.H, W = pr.H, C4 = C / 4, M = H * W;
    vector<float> X((size_t)M * C), Wt((size_t)9 * C * C);  // X[p][c], Wt[(tap*C+ci)][co]
    for (size_t i = 0; i < X.size(); i++) X[i] = ((i * 7919) % 1000) / 1000.f - .5f;
    for (size_t i = 0; i < Wt.size(); i++) Wt[i] = (((i * 104729) % 1000) / 1000.f - .5f) * 0.1f;
    vector<uint16_t> Xh(X.size()), Wh(Wt.size());
    vector<float> Xr(X.size()), Wr(Wt.size());
    for (size_t i = 0; i < X.size(); i++) { Xh[i] = f2h(X[i]); Xr[i] = h2f(Xh[i]); }
    for (size_t i = 0; i < Wt.size(); i++) { Wh[i] = f2h(Wt[i]); Wr[i] = h2f(Wh[i]); }
    for (auto& v : vars) {
      if (v.f16 && !has_ext("cl_khr_fp16")) continue;
      double bestT = 1e30, medT = 0; int bestTM = 0; Lws bestL{}; float maxerr = 0;
      for (int TM : {8, 4}) {
        int NV = 2;
        cl_kernel k = build(conv_src(v, TM, NV, H, W, C), "k", "", true);
        if (!k) continue;
        cl_int e; size_t es = v.f16 ? 2 : 4;
        cl_mem bX, bB;
        if (v.ximg) { cl_image_format fmt = {CL_RGBA, (cl_channel_type)(v.f16 ? CL_HALF_FLOAT : CL_FLOAT)}; cl_image_desc d{}; d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = (size_t)W * C4; d.image_height = H;
          bX = p_clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &d, v.f16 ? (void*)Xh.data() : (void*)X.data(), &e); }
        else { bX = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, X.size() * es, nullptr, &e); CK(p_clEnqueueWriteBuffer(q, bX, CL_TRUE, 0, X.size() * es, v.f16 ? (const void*)Xh.data() : (const void*)X.data(), 0, nullptr, nullptr)); }
        if (e != CL_SUCCESS) continue;
        if (v.bimg) { cl_image_format fmt = {CL_RGBA, (cl_channel_type)(v.f16 ? CL_HALF_FLOAT : CL_FLOAT)}; cl_image_desc d{}; d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = C4; d.image_height = (size_t)9 * C;
          bB = p_clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &d, v.f16 ? (void*)Wh.data() : (void*)Wt.data(), &e); }
        else { bB = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, Wt.size() * es, nullptr, &e); CK(p_clEnqueueWriteBuffer(q, bB, CL_TRUE, 0, Wt.size() * es, v.f16 ? (const void*)Wh.data() : (const void*)Wt.data(), 0, nullptr, nullptr)); }
        if (e != CL_SUCCESS) continue;
        cl_mem bY = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * C * 4, nullptr, &e);
        arg(k, 0, bX); arg(k, 1, bB); arg(k, 2, bY);
        const vector<float>& rx = v.f16 ? Xr : X; const vector<float>& rw = v.f16 ? Wr : Wt;
        for (auto l : lwss) {
          if (l.x * l.y > 256) continue;
          size_t gx = ((C4 / NV) + l.x - 1) / l.x * l.x, gy = (((M + TM - 1) / TM) + l.y - 1) / l.y * l.y;
          size_t g[2] = {gx, gy}, ll[2] = {(size_t)l.x, (size_t)l.y};
          if (p_clEnqueueNDRangeKernel(q, k, 2, nullptr, g, ll, 0, nullptr, nullptr) != CL_SUCCESS) continue;
          p_clFinish(q);
          // dependent batch of 8 back-to-back dispatches (in-order queue)
          Stat st = timeit([&] { for (int i = 0; i < 8; i++) p_clEnqueueNDRangeKernel(q, k, 2, nullptr, g, ll, 0, nullptr, nullptr); }, 9, 1);
          st.best /= 8; st.med /= 8;
          if (st.best < bestT) {
            bestT = st.best; medT = st.med; bestTM = TM; bestL = l;
            vector<float> y((size_t)M * C); CK(p_clEnqueueReadBuffer(q, bY, CL_TRUE, 0, y.size() * 4, y.data(), 0, nullptr, nullptr));
            maxerr = 0; double mr = 0;
            for (int t = 0; t < 200; t++) {
              size_t pp = (t * 2654435761u) % M, co = (t * 40503u + 7) % C; int yy = pp / W, xx = pp % W; double ref = 0;
              for (int tap = 0; tap < 9; tap++) { int iy = yy + tap / 3 - 1, ix = xx + tap % 3 - 1; if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue; for (int ci = 0; ci < C; ci++) ref += (double)rx[((size_t)iy * W + ix) * C + ci] * rw[((size_t)tap * C + ci) * C + co]; }
              maxerr = std::max(maxerr, (float)std::fabs(ref - y[pp * C + co])); mr = std::max(mr, std::fabs(ref));
            }
            maxerr = (float)(maxerr / mr);
          }
        }
        p_clReleaseMemObject(bX); p_clReleaseMemObject(bB); p_clReleaseMemObject(bY); p_clReleaseKernel(k);
      }
      double flops = 2.0 * M * C * C * 9;
      if (bestT > 1e29) printf("OpenCL conv C=%d H=%d %-22s: build/run failed\n", C, H, v.name);
      else printf("OpenCL conv C=%d H=%d %-22s TM=%d NV=2 wg=%dx%d  best=%.3f ms (%.0f GFLOPS)  median %.0f GFLOPS  relerr=%.1e\n", C, H, v.name, bestTM, bestL.x, bestL.y, bestT, flops / bestT / 1e6, flops / medT / 1e6, maxerr);
    }
  }
}


// ---------------------------------------------------------------- cl_qcom_perf_hint: does HIGH avoid the post-idle slowdown?
static void do_hint() {
  typedef cl_int(CL_API_CALL * hint_fn)(cl_context, cl_uint);
  hint_fn setHint = (hint_fn)dlsym(lib, "clSetPerfHintQCOM");
  printf("clSetPerfHintQCOM exported: %s; cl_qcom_perf_hint ext: %s\n", setHint ? "yes" : "no", has_ext("cl_qcom_perf_hint") ? "yes" : "no");
  if (!setHint) return;
  cl_kernel k = build(fma_kernel("float", 1, 32, false), "k", "", true);
  cl_int e; size_t g = 1u << 18, l = 256; int iters = 256;
  cl_mem o = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, g * 4, nullptr, &e);
  float a = 0.999f, b = 0.001f; arg(k, 0, o); arg(k, 1, iters); arg(k, 2, a); arg(k, 3, b);
  double flops = 2.0 * g * iters * 32;
  auto window = [&](double ms) {  // run back-to-back launches for ms, return GFLOPS
    double t0 = nowms(); int n = 0;
    while (nowms() - t0 < ms) { CK(p_clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr)); p_clFinish(q); n++; }
    return flops * n / (nowms() - t0) / 1e6;
  };
  struct H { const char* name; cl_uint v; };
  for (H h : {H{"NORMAL", CL_PERF_HINT_NORMAL_QCOM}, H{"HIGH", CL_PERF_HINT_HIGH_QCOM}, H{"NORMAL", CL_PERF_HINT_NORMAL_QCOM}, H{"HIGH", CL_PERF_HINT_HIGH_QCOM}}) {
    cl_int r = setHint(ctx, h.v);
    printf("hint %-6s (rc=%d):\n", h.name, r);
    for (int idle : {0, 50, 200, 1000, 3000}) {
      warm_gpu(3);
      usleep(idle * 1000);
      printf("  after %4d ms idle: GFLOPS in successive 100 ms windows:", idle);
      for (int w = 0; w < 6; w++) printf(" %.0f", window(100));
      printf("\n");
    }
  }
}

int main(int argc, char** argv) {
  const char* paths[] = {"/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "/system/lib64/libOpenCL.so", "libOpenCL.so"};
  for (auto p : paths) { lib = dlopen(p, RTLD_NOW); if (lib) break; }
  if (!lib) { fprintf(stderr, "no libOpenCL.so: %s\n", dlerror()); return 2; }
  LOAD(clGetPlatformIDs) LOAD(clGetPlatformInfo) LOAD(clGetDeviceIDs) LOAD(clGetDeviceInfo) LOAD(clCreateContext) LOAD(clCreateCommandQueue) LOAD(clCreateProgramWithSource)
  LOAD(clBuildProgram) LOAD(clGetProgramBuildInfo) LOAD(clCreateKernel) LOAD(clSetKernelArg) LOAD(clEnqueueNDRangeKernel) LOAD(clFinish) LOAD(clCreateBuffer) LOAD(clCreateImage)
  LOAD(clEnqueueWriteBuffer) LOAD(clEnqueueReadBuffer) LOAD(clReleaseMemObject) LOAD(clReleaseKernel) LOAD(clReleaseProgram) LOAD(clGetKernelWorkGroupInfo)
  cl_platform_id plat; cl_uint np = 0; CK(p_clGetPlatformIDs(1, &plat, &np));
  cl_uint nd = 0; CK(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nd));
  cl_int e; ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); CK(e);
  q = p_clCreateCommandQueue(ctx, dev, 0, &e); CK(e);
  size_t n = 0; p_clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &n); g_exts.resize(n); p_clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, n, g_exts.data(), nullptr);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  string mode = argc > 1 ? argv[1] : "info";
  if (mode == "info") do_info();
  else if (mode == "warm") { warm_gpu(argc > 2 ? atof(argv[2]) : 5); }
  else if (mode == "peak") { warm_gpu(3); do_peak(); }
  else if (mode == "bw") { warm_gpu(3); do_bw(); }
  else if (mode == "gemm") { warm_gpu(3); vector<string> only; for (int i = 2; i < argc; i++) only.push_back(argv[i]); do_gemm(only); }
  else if (mode == "conv") { warm_gpu(3); do_conv(); }
  else if (mode == "hint") { do_hint(); }
  else if (mode == "hold") {  // hold the QCOM HIGH perf hint for S seconds (idle process) so other processes can be measured under it
    typedef cl_int(CL_API_CALL * hint_fn)(cl_context, cl_uint);
    hint_fn setHint = (hint_fn)dlsym(lib, "clSetPerfHintQCOM");
    cl_int r = setHint ? setHint(ctx, CL_PERF_HINT_HIGH_QCOM) : -1; printf("hold: hint rc=%d\n", r); sleep(argc > 2 ? atoi(argv[2]) : 10);
  }
  else { fprintf(stderr, "usage: cl_vs_wg info|warm S|peak|bw|gemm|conv\n"); return 1; }
  return 0;
}
