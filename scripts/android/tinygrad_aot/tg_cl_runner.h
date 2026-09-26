// Runs a tinygrad ahead-of-time OpenCL bundle (export_cl.py: kernels.cl, plan.txt, consts.bin) on Android's
// vendor OpenCL. No Python and no tinygrad on the phone: the kernels, their launch sizes and the buffer plan were
// fixed on the host; this file only builds the program, allocates the buffers and replays the kernel calls.
//
//   TgClModel m;
//   m.load(dir, cache_dir);                // builds kernels.cl once, caches the device binary in cache_dir
//   m.run(input_bytes, {out0_ptr, ...});   // input as exported (e.g. uint8 NHWC), outputs as float
#pragma once
#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tgcl {

#define TGCL_FNS(X)                                                                                              \
  X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext) X(clCreateCommandQueue)            \
  X(clCreateBuffer) X(clCreateSubBuffer) X(clCreateImage) X(clCreateProgramWithSource)                           \
  X(clCreateProgramWithBinary) X(clBuildProgram) X(clGetProgramBuildInfo) X(clGetProgramInfo) X(clCreateKernel)  \
  X(clSetKernelArg) X(clEnqueueNDRangeKernel) X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer) X(clFinish)          \
  X(clGetEventProfilingInfo) X(clReleaseEvent) X(clWaitForEvents) X(clReleaseMemObject) X(clReleaseKernel)       \
  X(clReleaseProgram) X(clReleaseCommandQueue) X(clReleaseContext)
#define TGCL_DECL(name) inline decltype(&::name) name;
TGCL_FNS(TGCL_DECL)

inline void load_lib() {
  static bool done = false;
  if (done) return;
  void* h = nullptr;
  for (const char* p : {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so"})
    if ((h = dlopen(p, RTLD_NOW | RTLD_LOCAL))) break;
  if (!h) throw std::runtime_error(std::string("dlopen libOpenCL.so: ") + dlerror());
#define TGCL_LOAD(name)                                                  \
  name = reinterpret_cast<decltype(name)>(dlsym(h, #name));              \
  if (!name) throw std::runtime_error("dlsym " #name);
  TGCL_FNS(TGCL_LOAD)
  done = true;
}

#define TGCL_CK(x)                                                                                            \
  do {                                                                                                        \
    cl_int e_ = (x);                                                                                          \
    if (e_ != CL_SUCCESS) throw std::runtime_error(std::string(#x) + " = " + std::to_string(e_));             \
  } while (0)

inline std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("open " + path);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

struct Model {
  cl_context ctx = nullptr;
  cl_device_id dev = nullptr;
  cl_command_queue q = nullptr;
  cl_program prog = nullptr;
  cl_mem arena = nullptr, consts = nullptr;
  std::vector<cl_mem> bufs, own;  // own: everything we release
  std::vector<size_t> nbytes;
  std::vector<int> in_ids, out_ids;
  struct Call {
    cl_kernel k;
    std::string name;
    size_t g[3], l[3];
  };
  std::vector<Call> calls, init_calls;  // init_calls: run once after loading (constants -> persist buffers)
  bool owns_ctx = false;
  double build_ms = 0;

  // ctx/dev/q may come from the caller (share one GPU context with other OpenCL work); null creates our own
  void load(const std::string& dir, const std::string& cache_dir, cl_context c = nullptr, cl_device_id d = nullptr,
            cl_command_queue cq = nullptr, bool profile = false) {
    load_lib();
    if (!c) {
      cl_platform_id plat;
      TGCL_CK(clGetPlatformIDs(1, &plat, nullptr));
      TGCL_CK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
      cl_int e;
      ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
      TGCL_CK(e);
      q = clCreateCommandQueue(ctx, dev, profile ? CL_QUEUE_PROFILING_ENABLE : 0, &e);
      TGCL_CK(e);
      owns_ctx = true;
    } else {
      ctx = c, dev = d, q = cq;
    }
    build_program(dir, cache_dir);
    parse_plan(dir);
    for (auto& c : init_calls) TGCL_CK(clEnqueueNDRangeKernel(q, c.k, 3, nullptr, c.g, c.l, 0, nullptr, nullptr));
    TGCL_CK(clFinish(q));
  }

  void build_program(const std::string& dir, const std::string& cache_dir) {
    auto t0 = std::chrono::steady_clock::now();
    std::string src = read_file(dir + "/kernels.cl");
    std::string cache = cache_dir.empty() ? "" : cache_dir + "/tgcl_" + std::to_string(std::hash<std::string>{}(src)) + ".bin";
    cl_int e = CL_SUCCESS;
    std::string bin;
    if (!cache.empty()) {
      std::ifstream f(cache, std::ios::binary);
      if (f) bin.assign(std::istreambuf_iterator<char>(f), {});
    }
    if (!bin.empty()) {
      const unsigned char* p = reinterpret_cast<const unsigned char*>(bin.data());
      size_t n = bin.size();
      cl_int st;
      prog = clCreateProgramWithBinary(ctx, 1, &dev, &n, &p, &st, &e);
      if (e != CL_SUCCESS || st != CL_SUCCESS || clBuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
        if (prog) clReleaseProgram(prog);
        prog = nullptr;
      }
    }
    if (!prog) {
      const char* s = src.c_str();
      prog = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e);
      TGCL_CK(e);
      if (clBuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        throw std::runtime_error("clBuildProgram failed:\n" + log.substr(0, 4000));
      }
      if (!cache.empty()) {
        size_t n = 0;
        TGCL_CK(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(n), &n, nullptr));
        std::string out(n, '\0');
        unsigned char* p = reinterpret_cast<unsigned char*>(out.data());
        TGCL_CK(clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(p), &p, nullptr));
        std::ofstream(cache, std::ios::binary).write(out.data(), out.size());
      }
    }
    build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }

  cl_mem sub(cl_mem base, size_t off, size_t n) {
    cl_buffer_region r{off, n};
    cl_int e;
    cl_mem m = clCreateSubBuffer(base, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &e);
    TGCL_CK(e);
    own.push_back(m);
    return m;
  }

  cl_mem alloc(size_t n) {
    cl_int e;
    cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n ? n : 1, nullptr, &e);
    TGCL_CK(e);
    own.push_back(m);
    return m;
  }

  void parse_plan(const std::string& dir) {
    std::istringstream plan(read_file(dir + "/plan.txt"));
    std::string cbin = read_file(dir + "/consts.bin");
    cl_int e;
    if (!cbin.empty()) {
      consts = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cbin.size(), cbin.data(), &e);
      TGCL_CK(e);
      own.push_back(consts);
    }
    std::string line;
    while (std::getline(plan, line)) {
      std::istringstream ls(line);
      std::string tag;
      ls >> tag;
      if (tag == "arena") {
        size_t n;
        ls >> n;
        if (n) arena = alloc(n);
      } else if (tag == "buf") {
        int id;
        size_t n;
        std::string kind;
        ls >> id >> n >> kind;
        if ((int)bufs.size() <= id) bufs.resize(id + 1), nbytes.resize(id + 1);
        nbytes[id] = n;
        if (kind == "in" || kind == "out") {
          int k;
          ls >> k;
          auto& v = kind == "in" ? in_ids : out_ids;
          if ((int)v.size() <= k) v.resize(k + 1);
          v[k] = id;
          bufs[id] = alloc(n);
        } else if (kind == "const") {
          size_t off;
          ls >> off;
          bufs[id] = sub(consts, off, n);
        } else if (kind == "persist") {
          bufs[id] = alloc(n);
        } else if (kind == "arena") {
          size_t off;
          ls >> off;
          bufs[id] = sub(arena, off, n);
        }
      } else if (tag == "kern" || tag == "init") {
        Call c;
        int nargs;
        ls >> c.name >> c.g[0] >> c.g[1] >> c.g[2] >> c.l[0] >> c.l[1] >> c.l[2] >> nargs;
        for (int i = 0; i < 3; i++) c.g[i] *= c.l[i];  // tinygrad records work groups; OpenCL wants work items
        c.k = clCreateKernel(prog, c.name.c_str(), &e);
        if (e != CL_SUCCESS) throw std::runtime_error("clCreateKernel " + c.name + " = " + std::to_string(e));
        for (int i = 0; i < nargs; i++) {
          std::string a;
          ls >> a;
          if (a[0] == 'b') {
            cl_mem m = bufs.at(std::stoi(a.substr(1)));
            TGCL_CK(clSetKernelArg(c.k, i, sizeof(m), &m));
          } else if (a[0] == 'i') {
            int id, h, w, isz;
            if (sscanf(a.c_str() + 1, "%d,%d,%d,%d", &id, &h, &w, &isz) != 4) throw std::runtime_error("bad image arg " + a);
            cl_image_format fmt{CL_RGBA, isz == 2 ? (cl_channel_type)CL_HALF_FLOAT : (cl_channel_type)CL_FLOAT};
            cl_image_desc desc{};
            desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width = w, desc.image_height = h, desc.image_row_pitch = (size_t)w * 4 * isz;
            desc.buffer = bufs.at(id);
            cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &e);
            if (e != CL_SUCCESS) throw std::runtime_error("clCreateImage " + a + " = " + std::to_string(e));
            own.push_back(img);
            TGCL_CK(clSetKernelArg(c.k, i, sizeof(img), &img));
          } else if (a[0] == 'v') {
            int v = std::stoi(a.substr(1));
            TGCL_CK(clSetKernelArg(c.k, i, sizeof(v), &v));
          } else
            throw std::runtime_error("bad kernel arg " + a);
        }
        (tag == "init" ? init_calls : calls).push_back(c);
      }
    }
  }

  // one inference: upload the input(s), replay every kernel, read the outputs back (blocking)
  void run(const std::vector<const void*>& ins, const std::vector<void*>& outs) {
    for (size_t i = 0; i < ins.size(); i++)
      TGCL_CK(clEnqueueWriteBuffer(q, bufs[in_ids[i]], CL_FALSE, 0, nbytes[in_ids[i]], ins[i], 0, nullptr, nullptr));
    enqueue();
    for (size_t i = 0; i < outs.size(); i++)
      TGCL_CK(clEnqueueReadBuffer(q, bufs[out_ids[i]], i + 1 == outs.size(), 0, nbytes[out_ids[i]], outs[i], 0, nullptr, nullptr));
  }

  void enqueue(std::vector<cl_event>* evs = nullptr) {
    for (auto& c : calls) {
      cl_event ev = nullptr;
      TGCL_CK(clEnqueueNDRangeKernel(q, c.k, 3, nullptr, c.g, c.l, 0, nullptr, evs ? &ev : nullptr));
      if (evs) evs->push_back(ev);
    }
  }

  // per-kernel GPU time (needs a profiling queue: load(..., profile=true)); returns ms per call
  std::vector<double> profile() {
    std::vector<cl_event> evs;
    enqueue(&evs);
    TGCL_CK(clFinish(q));
    std::vector<double> ms;
    for (auto ev : evs) {
      cl_ulong s = 0, t = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t), &t, nullptr);
      ms.push_back((t - s) * 1e-6);
      clReleaseEvent(ev);
    }
    return ms;
  }

  ~Model() {
    for (auto& c : calls) clReleaseKernel(c.k);
    for (auto& c : init_calls) clReleaseKernel(c.k);
    for (auto it = own.rbegin(); it != own.rend(); ++it) clReleaseMemObject(*it);
    if (prog) clReleaseProgram(prog);
    if (owns_ctx) {
      if (q) clReleaseCommandQueue(q);
      if (ctx) clReleaseContext(ctx);
    }
  }
};

}  // namespace tgcl
