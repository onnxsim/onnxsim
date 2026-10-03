// Runs tinygrad-generated WGSL kernels (gen.py output) through Dawn's Vulkan backend and checks them against the numpy reference.
//   tgrun_dawn PROBLEM_DIR [variant ...]
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <chrono>
#include "tgrun_common.h"
using Clock = std::chrono::steady_clock;
static std::string last_error;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
  dawnProcSetProcs(&dawn::native::GetProcs());
  std::string dir = argv[1];
  std::vector<std::string> sel(argv + 2, argv + argc);
  Manifest m = read_manifest(dir);
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  wgpu::Instance inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  if (!ad) { fprintf(stderr, "no adapter\n"); return 2; }
  // robustness off like onnxruntime's WebGPU EP; validation stays ON so a kernel exceeding WebGPU limits (e.g. >256 invocations
  // per workgroup, which Vulkan allows up to 1024) is reported as a failed variant instead of aborting the process
  std::vector<const char*> dev_en = {"disable_robustness"};
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data();
  wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt;
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView msg) { last_error.assign(msg.data, msg.length); });
  wgpu::Device dev;
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
  wgpu::Queue q = dev.GetQueue();
  auto mk = [&](size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = std::max<size_t>(n, 16); b.usage = u; return dev.CreateBuffer(&b); };
  auto St = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  // buffers (shared by all variants); inputs from in<slot>.bin, output zeroed before every run
  std::vector<wgpu::Buffer> bufs;
  size_t out_bytes = 0; int out_slot = 0;
  for (auto& b : m.bufs) {
    bufs.push_back(mk(b.elems * 4, St));
    if (b.out) { out_bytes = b.elems * 4; out_slot = b.slot; }
    else { auto d = read_file(dir + "/in" + std::to_string(b.slot) + ".bin"); q.WriteBuffer(bufs.back(), 0, d.data(), d.size()); }
  }
  float inf = INFINITY;
  wgpu::Buffer ubuf = mk(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  q.WriteBuffer(ubuf, 0, &inf, 4);
  wgpu::Buffer rb = mk(out_bytes, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
  std::vector<char> zeros(out_bytes, 0), ref = read_file(dir + "/ref.bin");
  // explicit layout: binding 0 = INFINITY uniform, 1.. = storage buffers (tinygrad declares the uniform even when unused)
  std::vector<wgpu::BindGroupLayoutEntry> le(m.bufs.size() + 1);
  le[0].binding = 0; le[0].visibility = wgpu::ShaderStage::Compute; le[0].buffer.type = wgpu::BufferBindingType::Uniform;
  for (size_t i = 0; i < m.bufs.size(); i++) { le[i + 1].binding = i + 1; le[i + 1].visibility = wgpu::ShaderStage::Compute; le[i + 1].buffer.type = wgpu::BufferBindingType::Storage; }
  wgpu::BindGroupLayoutDescriptor bgld{}; bgld.entryCount = le.size(); bgld.entries = le.data();
  wgpu::BindGroupLayout bgl = dev.CreateBindGroupLayout(&bgld);
  wgpu::PipelineLayoutDescriptor pld{}; pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
  wgpu::PipelineLayout pll = dev.CreatePipelineLayout(&pld);
  std::vector<wgpu::BindGroupEntry> be(m.bufs.size() + 1);
  be[0].binding = 0; be[0].buffer = ubuf; be[0].size = 16;
  for (size_t i = 0; i < m.bufs.size(); i++) { be[i + 1].binding = i + 1; be[i + 1].buffer = bufs[i]; be[i + 1].size = bufs[i].GetSize(); }
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = bgl; bgd.entryCount = be.size(); bgd.entries = be.data();
  wgpu::BindGroup bg = dev.CreateBindGroup(&bgd);

  for (auto& v : m.variants) {
    if (!wanted(v, sel)) continue;
    last_error.clear();
    std::string src = read_text(dir + "/" + v.name + ".wgsl");
    auto t0 = Clock::now();
    wgpu::ShaderSourceWGSL w{}; w.code = {src.data(), src.size()};
    wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
    wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
    wgpu::ComputePipelineDescriptor cpd{}; cpd.layout = pll; cpd.compute.module = sm; cpd.compute.entryPoint = v.entry.c_str();
    wgpu::ComputePipeline pl = dev.CreateComputePipeline(&cpd);
    q.WriteBuffer(bufs[out_slot], 0, zeros.data(), out_bytes);
    auto run = [&](int reps) {
      wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
      { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0, bg);
        for (int r = 0; r < reps; r++) p.DispatchWorkgroups(v.g[0], v.g[1], v.g[2]);
        p.End(); }
      wgpu::CommandBuffer cb = enc.Finish();
      auto t = Clock::now();
      q.Submit(1, &cb);
      inst.WaitAny(q.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
      return ms_since(t);
    };
    run(1);  // first use finishes pipeline compilation
    double compile_ms = ms_since(t0);
    if (!last_error.empty()) { printf("RESULT dawn %s FAIL %.120s\n", v.name.c_str(), last_error.c_str()); continue; }
    // correctness on a fresh single dispatch
    q.WriteBuffer(bufs[out_slot], 0, zeros.data(), out_bytes);
    run(1);
    { wgpu::CommandEncoder enc = dev.CreateCommandEncoder(); enc.CopyBufferToBuffer(bufs[out_slot], 0, rb, 0, out_bytes); wgpu::CommandBuffer cb = enc.Finish(); q.Submit(1, &cb); }
    bool ok = false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read, 0, out_bytes, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::MapAsyncStatus s, wgpu::StringView) { ok = s == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
    double err = ok ? rel_err((const float*)rb.GetConstMappedRange(0, out_bytes), ref, out_bytes / 4) : 1e9;
    rb.Unmap();
    // timing: single-dispatch latency, and per-dispatch time inside a back-to-back batch
    std::vector<double> single; for (int i = 0; i < 7; i++) single.push_back(run(1));
    std::sort(single.begin(), single.end());
    // long kernels run one dispatch per submission: batching them trips the Android GPU hang watchdog (device lost)
    int R = single[0] > 20 ? 1 : std::max(4, std::min(200, (int)(40.0 / std::max(single[0], 0.05))));
    run(R);
    std::vector<double> batch; for (int i = 0; i < 5; i++) batch.push_back(run(R) / R);
    std::sort(batch.begin(), batch.end());
    printf("RESULT dawn %s %s err=%.1e compile_ms=%.0f single_ms=%.3f disp_ms=%.3f gflops=%.1f opts=%s\n", v.name.c_str(), err < 1e-3 ? "OK" : "WRONG", err, compile_ms, single[0], batch[0], m.flops / batch[0] / 1e6, v.opts.c_str());
    fflush(stdout);
  }
  return 0;
}
