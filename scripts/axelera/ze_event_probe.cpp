// Probe: does the Metis driver return kernel timestamps through Level Zero events?
// Records two timestamp events around a device memory copy and reads them back.
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x) do { ze_result_t r_ = (x); if (r_ != ZE_RESULT_SUCCESS) { \
  std::printf("%s -> 0x%x\n", #x, (unsigned)r_); return 1; } } while (0)

int main() {
  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nd = 1; ze_driver_handle_t drv; CHECK(zeDriverGet(&nd, &drv));
  uint32_t ndev = 1; ze_device_handle_t dev; CHECK(zeDeviceGet(drv, &ndev, &dev));
  ze_device_properties_t dp{}; dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  CHECK(zeDeviceGetProperties(dev, &dp));
  std::printf("device %s, timerResolution=%lu ns, timestampValidBits=%u\n", dp.name,
              (unsigned long)dp.timerResolution, dp.timestampValidBits);

  ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t ctx; CHECK(zeContextCreate(drv, &cd, &ctx));

  ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  pd.count = 2;
  ze_event_pool_handle_t pool; CHECK(zeEventPoolCreate(ctx, &pd, 1, &dev, &pool));
  ze_event_desc_t ed0{ZE_STRUCTURE_TYPE_EVENT_DESC}; ed0.index = 0; ed0.signal = ZE_EVENT_SCOPE_FLAG_HOST; ed0.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_event_desc_t ed1 = ed0; ed1.index = 1;
  ze_event_handle_t e0, e1;
  CHECK(zeEventCreate(pool, &ed0, &e0));
  CHECK(zeEventCreate(pool, &ed1, &e1));

  ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = 0; qd.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ze_command_list_handle_t cl; ze_command_list_desc_t cld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  CHECK(zeCommandListCreate(ctx, dev, &cld, &cl));
  ze_command_queue_handle_t q; CHECK(zeCommandQueueCreate(ctx, dev, &qd, &q));

  ze_device_mem_alloc_desc_t dmd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ze_host_mem_alloc_desc_t hmd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void *src, *dst; CHECK(zeMemAllocHost(ctx, &hmd, 1 << 20, 64, &src));
  CHECK(zeMemAllocDevice(ctx, &dmd, 1 << 20, 64, dev, &dst));
  std::memset(src, 1, 1 << 20);

  CHECK(zeCommandListAppendQueryKernelTimestamps(cl, 0, nullptr, nullptr, nullptr, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(cl, e0, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(cl, dst, src, 1 << 20, e1, 1, &e0));
  CHECK(zeCommandListClose(cl));
  CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr));
  CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));

  uint64_t t0 = 0, t1 = 0;
  ze_result_t r0 = zeEventQueryKernelTimestamp(e0, nullptr);
  std::printf("zeEventQueryKernelTimestamp(e0) probe -> 0x%x\n", (unsigned)r0);
  ze_kernel_timestamp_result_t res0{}, res1{};
  ze_result_t q0 = zeEventQueryKernelTimestamp(e0, &res0);
  ze_result_t q1 = zeEventQueryKernelTimestamp(e1, &res1);
  std::printf("query e0 -> 0x%x, query e1 -> 0x%x\n", (unsigned)q0, (unsigned)q1);
  t0 = res0.global.kernelStart; t1 = res1.global.kernelEnd;
  std::printf("global kernelStart=%lu kernelEnd=%lu (delta=%ld cycles)\n", (unsigned long)t0, (unsigned long)t1,
              (long)(t1 - t0));
  return 0;
}
