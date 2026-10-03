// Runs tinygrad-generated GLSL kernels (compiled to SPIR-V by gen.py) directly on Vulkan and checks them against the numpy reference.
//   tgrun_vk PROBLEM_DIR [variant ...]
#include <vulkan/vulkan.h>
#include <chrono>
#include "tgrun_common.h"
using Clock = std::chrono::steady_clock;
#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan error %d at %s:%d: %s\n", r_, __FILE__, __LINE__, #x); exit(2); } } while (0)

static VkPhysicalDeviceMemoryProperties memprops;
static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback) {
  for (int pass = 0; pass < 2; pass++) {
    VkMemoryPropertyFlags f = pass == 0 ? want : fallback;
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & f) == f) return i;
  }
  fprintf(stderr, "no memory type\n"); exit(2);
}

int main(int argc, char** argv) {
  std::string dir = argv[1];
  std::vector<std::string> sel(argv + 2, argv + argc);
  Manifest m = read_manifest(dir);
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
  VkInstance inst; VK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t np = 0; VK(vkEnumeratePhysicalDevices(inst, &np, nullptr));
  std::vector<VkPhysicalDevice> pds(np); VK(vkEnumeratePhysicalDevices(inst, &np, pds.data()));
  VkPhysicalDevice pd = pds[0];
  VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
  vkGetPhysicalDeviceMemoryProperties(pd, &memprops);
  uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
  uint32_t qfi = 0; for (uint32_t i = 0; i < nq; i++) if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits) { qfi = i; break; }
  float prio = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  VkDevice dev; VK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qfi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool pool; VK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VK(vkCreateFence(dev, &fci, nullptr, &fence));
  VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
  VkQueryPool qp; VK(vkCreateQueryPool(dev, &qpci, nullptr, &qp));

  // buffers: host-visible + device-local when available (unified memory on phones), else host-visible
  struct B { VkBuffer b; VkDeviceMemory mem; void* ptr; size_t bytes; };
  std::vector<B> bufs; int out_idx = 0;
  for (size_t i = 0; i < m.bufs.size(); i++) {
    B b; b.bytes = m.bufs[i].elems * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = b.bytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(dev, &bci, nullptr, &b.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.b, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK(vkAllocateMemory(dev, &mai, nullptr, &b.mem)); VK(vkBindBufferMemory(dev, b.b, b.mem, 0)); VK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    if (m.bufs[i].out) out_idx = i;
    else { auto d = read_file(dir + "/in" + std::to_string(m.bufs[i].slot) + ".bin"); memcpy(b.ptr, d.data(), d.size()); }
    bufs.push_back(b);
  }
  std::vector<char> ref = read_file(dir + "/ref.bin");
  // descriptor set: binding i = kernel slot i
  std::vector<VkDescriptorSetLayoutBinding> lb(bufs.size());
  for (size_t i = 0; i < bufs.size(); i++) { lb[i] = {}; lb[i].binding = i; lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].descriptorCount = 1; lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dlci.bindingCount = lb.size(); dlci.pBindings = lb.data();
  VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsl));
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
  VkPipelineLayout pll; VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pll));
  VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)bufs.size()};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
  VkDescriptorPool dpool; VK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds; VK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  std::vector<VkDescriptorBufferInfo> dbi(bufs.size()); std::vector<VkWriteDescriptorSet> wds(bufs.size());
  for (size_t i = 0; i < bufs.size(); i++) { dbi[i] = {bufs[i].b, 0, VK_WHOLE_SIZE}; wds[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1; wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i]; }
  vkUpdateDescriptorSets(dev, wds.size(), wds.data(), 0, nullptr);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &cbai, &cb));

  for (auto& v : m.variants) {
    if (!wanted(v, sel)) continue;
    auto spv = read_file(dir + "/" + v.name + ".spv");
    if (spv.empty()) { printf("RESULT vulkan %s FAIL no spirv\n", v.name.c_str()); continue; }
    auto t0 = Clock::now();
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize = spv.size(); smci.pCode = (const uint32_t*)spv.data();
    VkShaderModule sm; VK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.layout = pll;
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    VkPipeline pl; VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pl);
    double compile_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (pr != VK_SUCCESS) { printf("RESULT vulkan %s FAIL pipeline %d\n", v.name.c_str(), pr); vkDestroyShaderModule(dev, sm, nullptr); continue; }
    // returns {wall ms, gpu ms (timestamps)} for `reps` back-to-back dependent dispatches
    auto run = [&](int reps) {
      VK(vkResetCommandBuffer(cb, 0));
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VK(vkBeginCommandBuffer(cb, &bi));
      vkCmdResetQueryPool(cb, qp, 0, 2);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pll, 0, 1, &ds, 0, nullptr);
      for (int r = 0; r < reps; r++) {
        vkCmdDispatch(cb, v.g[0], v.g[1], v.g[2]);
        if (r + 1 < reps) {
          VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
      }
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
      VK(vkEndCommandBuffer(cb));
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
      auto t = Clock::now();
      VK(vkQueueSubmit(queue, 1, &si, fence)); VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX)); VK(vkResetFences(dev, 1, &fence));
      double wall = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
      uint64_t ts[2]; VK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      return std::make_pair(wall, (ts[1] - ts[0]) * (double)props.limits.timestampPeriod * 1e-6);
    };
    memset(bufs[out_idx].ptr, 0, bufs[out_idx].bytes);
    run(1);
    memset(bufs[out_idx].ptr, 0, bufs[out_idx].bytes);
    run(1);
    double err = rel_err((const float*)bufs[out_idx].ptr, ref, bufs[out_idx].bytes / 4);
    std::vector<double> single; for (int i = 0; i < 7; i++) single.push_back(run(1).first);
    std::sort(single.begin(), single.end());
    // long kernels run one dispatch per submission: batching them trips the Android GPU hang watchdog (device lost)
    int R = single[0] > 20 ? 1 : std::max(4, std::min(200, (int)(40.0 / std::max(single[0], 0.05))));
    run(R);
    std::vector<double> gpu, wall; for (int i = 0; i < 5; i++) { auto w = run(R); gpu.push_back(w.second / R); wall.push_back(w.first / R); }
    std::sort(gpu.begin(), gpu.end()); std::sort(wall.begin(), wall.end());
    printf("RESULT vulkan %s %s err=%.1e compile_ms=%.0f single_ms=%.3f disp_ms=%.3f gflops=%.1f gpu_disp_ms=%.3f opts=%s\n", v.name.c_str(), err < 1e-3 ? "OK" : "WRONG", err, compile_ms, single[0], wall[0], m.flops / wall[0] / 1e6, gpu[0], v.opts.c_str());
    fflush(stdout);
    vkDestroyPipeline(dev, pl, nullptr); vkDestroyShaderModule(dev, sm, nullptr);
  }
  printf("device: %s\n", props.deviceName);
  return 0;
}
