/* FastRPC skel replaying a captured tinygrad kernel sequence (capture.py / emit.py skel): regions are loaded
 * once into the DSP heap, `run` acquires VTCM + HMX (the generated HMX kernels use __hmx_vtcm as their tile
 * cache) and runs the calls on a worker thread holding the HVX and HMX locks. Runtime as ../../hmx_gemm. */
#include <stdlib.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "qurt.h"
#include "replay_rpc.h"

extern unsigned long long HAP_perf_get_time_us(void);
unsigned char* __hmx_vtcm;
unsigned int __hmx_gen;
#include "replay_gen.h" /* NR, RSZ[], NCALL, OUT_R / OUT_O / OUT_N, tg_calls(pc) */

int replay_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return 0;
}
int replay_rpc_close(remote_handle64 h) {
  free((void*)(uintptr_t)h);
  return 0;
}

int replay_rpc_perf_vote(remote_handle64 h, int flags, int* rc) {
  void* ctx = (void*)replay_rpc_perf_vote;
  HAP_power_request_t req = {0};
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = 1;
  int r1 = HAP_power_set(ctx, &req);
  HAP_power_request_t d = {0};
  d.type = HAP_power_set_DCVS_v2;
  d.dcvs_v2.dcvs_enable = 0;
  d.dcvs_v2.set_dcvs_params = 1;
  d.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  if (flags & 1) {
    d.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
    d.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
  }
  int r2 = HAP_power_set(ctx, &d);
  int r3 = 0;
  if (flags & 2) { /* mandatory before any HMX op: without it the first tile op wedges the cDSP */
    HAP_power_request_t x = {0};
    x.type = HAP_power_set_HMX;
    x.hmx.power_up = 1;
    r3 = HAP_power_set(ctx, &x);
  }
  *rc = r3 * 1000000 + r1 * 1000 + r2;
  return 0;
}

int replay_rpc_load_region(remote_handle64 h, int idx, const unsigned char* img, int len) {
  if (idx < 0 || idx >= NR || (unsigned)len != RSZ[idx]) return AEE_EBADPARM;
  /* padded: the generated kernels l2fetch the next K panel / rows ahead, which may run past a buffer's end; tinygrad's own
   * runtime sub-allocates from larger mappings, a region allocated alone can end at an unmapped page (PD fault) */
  if (!R[idx]) R[idx] = memalign(4096, (RSZ[idx] ? RSZ[idx] : 1) + (256 << 10));
  if (!R[idx]) return AEE_ENOMEMORY;
  memcpy(R[idx], img, len);
  return 0;
}

typedef struct {
  unsigned int ctx;
  int iters;
  uint64 *pc, *t;
  int* codes;
} job_t;
#define STACK_SIZE (256 * 1024) /* tinygrad kernels may keep large values on the stack */
static char g_stack[STACK_SIZE] __attribute__((aligned(128)));

static void worker(void* p) {
  job_t* j = (job_t*)p;
  j->codes[1] = qurt_hvx_lock(QURT_HVX_MODE_128B);
  j->codes[2] = j->codes[1] ? -1 : HAP_compute_res_hmx_lock(j->ctx);
  if (j->codes[2] == 0) {
    unsigned long long t0 = HAP_perf_get_time_us();
    for (int it = 0; it < j->iters; it++) tg_calls(j->pc);
    j->t[0] = (HAP_perf_get_time_us() - t0) / j->iters;
    HAP_compute_res_hmx_unlock(j->ctx);
  }
  if (j->codes[1] == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

int replay_rpc_run(remote_handle64 h, int iters, unsigned char* out, int outLen, uint64* pc, int pcLen, uint64* t, int tLen, int* codes,
                   int codesLen) {
  if (iters < 1 || outLen < OUT_N || pcLen < NCALL || tLen < 1 || codesLen < 4) return AEE_EBADPARM;
  for (int i = 0; i < NR; i++)
    if (!R[i]) return AEE_EBADSTATE;
  memset(pc, 0, pcLen * sizeof(uint64));
  memset(t, 0, tLen * sizeof(uint64));
  memset(codes, 0, codesLen * sizeof(int));
  compute_res_attr_t attr;
  HAP_compute_res_attr_init(&attr);
  /* the generated kernels' tile cache spans VTCM_KB (tinygrad's HMX_VTCM_KB) at __hmx_vtcm and is laid out from a 256 KB
   * window boundary: an HMX operand span crossing one page-faults the PD (rc 0x4e, ../../hmx_gemm) -- acquire 256 KB
   * more, align up */
  HAP_compute_res_attr_set_vtcm_param_v2(&attr, (VTCM_KB + 256) * 1024, 0, 0);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);
  unsigned int ctx = HAP_compute_res_acquire(&attr, 100000);
  codes[0] = (int)ctx;
  if (!ctx) return 0;
  void* vp = NULL;
  unsigned int vs = 0;
  HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vp, &vs);
  __hmx_vtcm = (unsigned char*)(((uintptr_t)vp + 0x3FFFF) & ~(uintptr_t)0x3FFFF);
  __hmx_gen++;
  job_t j = {ctx, iters, pc, t, codes};
  qurt_thread_attr_t ta;
  qurt_thread_attr_init(&ta);
  qurt_thread_attr_set_stack_addr(&ta, g_stack);
  qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
  qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
  qurt_thread_t tid;
  int st;
  codes[3] = qurt_thread_create(&tid, &ta, worker, &j);
  if (codes[3] == 0) qurt_thread_join(tid, &st);
  memcpy(out, R[OUT_R] + OUT_O, OUT_N);
  HAP_compute_res_release(ctx);
  return 0;
}
