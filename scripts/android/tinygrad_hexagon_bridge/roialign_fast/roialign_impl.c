/* DSP side of the RoiAlign FastRPC skel: roialign_kernel.h, optionally split across QuRT threads
 * by RoI (each thread holds the HVX unit via qurt_hvx_lock for its slice). Timed on the DSP with
 * HAP_perf_get_time_us so RPC/copy overhead is reported separately by the client. */
#include <stdlib.h>
#include "roialign_rpc.h"
#include "qurt.h"
#include "HAP_power.h"
#include "roialign_kernel.h"

extern unsigned long long HAP_perf_get_time_us(void);

/* Optional clock vote (same idea as tinygrad's own DSP entry point): leave DCVS and pin the core
 * clock to the TURBO corner, plus power HVX on, so short kernels aren't timed at a DCVS-idle clock. */
AEEResult roialign_rpc_perf_vote(remote_handle64 h, int32 turbo, int32* rc) {
  void* ctx = (void*)(uintptr_t)h;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = 1;
  int r1 = HAP_power_set(ctx, &req);
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2;
  d.dcvs_v2.dcvs_enable = turbo ? FALSE : TRUE;
  d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  d.dcvs_v2.set_latency = TRUE;
  d.dcvs_v2.latency = 40;
  d.dcvs_v2.set_dcvs_params = turbo ? TRUE : FALSE;
  d.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
  d.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
  d.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
  int r2 = HAP_power_set(ctx, &d);
  *rc = r1 ? r1 : r2;
  return 0;
}

int roialign_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return *h ? 0 : -1;
}
int roialign_rpc_close(remote_handle64 h) {
  if (h) free((void*)(uintptr_t)h);
  return 0;
}

typedef struct {
  const float* feat; const float* rois; float* out;
  int H, W, C, R, OH, OW, sr; float scale; int prefetch;
} job_t;

static void run_job(job_t* j) {
  (j->prefetch ? roialign_hwc_pf : roialign_hwc)(j->feat, j->H, j->W, j->C, j->rois, j->R, j->OH, j->OW, j->sr, j->scale, j->out);
}

static void thread_main(void* arg) {
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  run_job((job_t*)arg);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

#define MAX_THREADS 8
#define STACK_SIZE 16384
static char stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));

AEEResult roialign_rpc_run(remote_handle64 h, const float* feat, int featLen, const float* rois,
                           int roisLen, int32 H, int32 W, int32 C, int32 OH, int32 OW, int32 sr,
                           float scale, int32 nthreads, float* out, int resultLen, uint64* dsp_us) {
  int outLen = resultLen;
  int R = roisLen / 4;
  if (C % 32 || C > 32 * ROI_MAX_NV || featLen < H * W * C || outLen < R * OH * OW * C) return -1;
  int prefetch = nthreads >= 100; /* nthreads + 100 selects the l2fetch-prefetching variant */
  if (prefetch) nthreads -= 100;
  if (nthreads < 1) nthreads = 1;
  if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
  if (nthreads > R) nthreads = R > 0 ? R : 1;
  unsigned long long t0 = HAP_perf_get_time_us();
  if (nthreads == 1) {
    job_t j = {feat, rois, out, H, W, C, R, OH, OW, sr, scale, prefetch};
    run_job(&j);
  } else {
    job_t jobs[MAX_THREADS];
    qurt_thread_t tids[MAX_THREADS];
    int per = (R + nthreads - 1) / nthreads, started = 0;
    for (int t = 0; t < nthreads; t++) {
      int r0 = t * per, r1 = r0 + per > R ? R : r0 + per;
      if (r0 >= r1) break;
      jobs[t] = (job_t){feat, rois + 4 * r0, out + (long)r0 * OH * OW * C, H, W, C, r1 - r0, OH, OW, sr, scale, prefetch};
      qurt_thread_attr_t attr;
      qurt_thread_attr_init(&attr);
      qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
      qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
      qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
      if (qurt_thread_create(&tids[t], &attr, thread_main, &jobs[t]) != QURT_EOK) return -2;
      started++;
    }
    for (int t = 0; t < started; t++) { int st; qurt_thread_join(tids[t], &st); }
  }
  *dsp_us = HAP_perf_get_time_us() - t0;
  return 0;
}
