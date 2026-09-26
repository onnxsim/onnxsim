/* FastRPC skel for mcc_block.h on V69: block weights stay resident in the DSP heap (load_block), `run`
 * acquires VTCM + HMX and runs the blocks on a worker thread that holds the HVX and HMX locks. */
#include <stdlib.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "mcc_hmx_rpc.h"
#include "qurt.h"
#define MB_NOW() qurt_get_core_pcycles()
#include "mcc_decoder.h"

extern unsigned long long HAP_perf_get_time_us(void);

#define MAXB MB_BLOCKS
static uint8_t* g_blob[MAXB];
static uint8_t* g_head;

int mcc_hmx_rpc_open(const char* uri, remote_handle64* h) {
  *h = (remote_handle64)(uintptr_t)malloc(1);
  return 0;
}
int mcc_hmx_rpc_close(remote_handle64 h) {
  free((void*)(uintptr_t)h);
  return 0;
}

int mcc_hmx_rpc_perf_vote(remote_handle64 h, int flags, int* rc) {
  void* ctx = (void*)mcc_hmx_rpc_perf_vote;
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

int mcc_hmx_rpc_load_block(remote_handle64 h, int idx, const unsigned char* blob, int len) {
  if (idx < 0 || idx >= MAXB || (size_t)len != mb_block_bytes()) return AEE_EBADPARM;
  if (!g_blob[idx]) g_blob[idx] = memalign(4096, len);
  if (!g_blob[idx]) return AEE_ENOMEMORY;
  memcpy(g_blob[idx], blob, len);
  return 0;
}

int mcc_hmx_rpc_load_head(remote_handle64 h, const unsigned char* blob, int len) {
  if ((size_t)len != mb_head_bytes()) return AEE_EBADPARM;
  if (!g_head) g_head = memalign(4096, len);
  if (!g_head) return AEE_ENOMEMORY;
  memcpy(g_head, blob, len);
  return 0;
}

int mcc_hmx_rpc_set_kv(remote_handle64 h, const float* k, int kLen, const float* v, int vLen) {
  const int n = MB_BLOCKS * MB_HEADS * MB_SEEN * 32;
  if (kLen != n || vLen != n) return AEE_EBADPARM;
  mb_hf *kt[MB_BLOCKS], *vt[MB_BLOCKS];
  for (int b = 0; b < MB_BLOCKS; b++) {
    if (!g_blob[b]) return AEE_EBADSTATE;
    kt[b] = mb_blob_kt(g_blob[b]), vt[b] = mb_blob_vt(g_blob[b]);
  }
  mb_pack_kv(kt, vt, k, v);
  return 0;
}

int mcc_hmx_rpc_set_kv_tiles(remote_handle64 h, const uint16* tiles, int len) {
  const size_t per = (size_t)2 * MB_HEADS * MB_ST * MB_TH; /* halfwords per block */
  if ((size_t)len != per * MB_BLOCKS) return AEE_EBADPARM;
  for (int b = 0; b < MB_BLOCKS; b++) {
    if (!g_blob[b]) return AEE_EBADSTATE;
    memcpy(mb_blob_kt(g_blob[b]), tiles + per * b, per * 2); /* kt then vt: adjacent in the blob */
  }
  return 0;
}

typedef struct {
  unsigned int ctx;
  uint8_t* vtcm;
  int q, nb, iters, hvx, nthr, mode; /* mode 0: run (blocks on x), 1: decode (xyz -> occ, rgb) */
  const float* xyz;
  float *occ, *rgb;
  mb_head hd;
  const uint16_t* x;
  uint16_t* y;
  uint64* t;
  int* codes;
  mb_ctx c;
  mb_weights w[MAXB];
  qurt_barrier_t bar;
  volatile int ok;
} job_t;

typedef struct {
  job_t* j;
  int tid;
} targ_t;

#define MAXT 4
#define STACK_SIZE (64 * 1024)
static char g_stack[MAXT][STACK_SIZE] __attribute__((aligned(128)));
static float g_pself[1024];

static void bar_wait(void* b) { qurt_barrier_wait((qurt_barrier_t*)b); }

/* SPMD: every thread holds an HVX context; thread 0 also the HMX lock and does the serial parts */
static void worker(void* p) {
  targ_t* a = (targ_t*)p;
  job_t* j = a->j;
  const int tid = a->tid;
  int hv = qurt_hvx_lock(QURT_HVX_MODE_128B);
  if (tid == 0) {
    j->codes[1] = hv;
    j->codes[2] = HAP_compute_res_hmx_lock(j->ctx);
    j->ok = hv == 0 && j->codes[2] == 0;
  } else if (hv)
    j->codes[5]++;
  qurt_barrier_wait(&j->bar);
  if (j->ok && hv == 0) {
    unsigned long long tsum = 0, t0 = 0;
    for (int it = 0; it < j->iters; it++) {
      if (tid == 0) {
        mb_layout(&j->c, j->vtcm, j->q, g_pself);
        j->c.hvx = j->hvx, j->c.nthr = j->nthr, j->c.sync = bar_wait, j->c.sync_arg = &j->bar;
        if (j->mode == 0)
          for (int r = 0; r < j->q; r++)
            for (int d = 0; d < MB_D; d++) *mb_at(j->c.x, MB_KT, r, d) = j->x[(size_t)r * MB_D + d];
        t0 = HAP_perf_get_time_us();
      }
      qurt_barrier_wait(&j->bar);
      if (j->mode == 1)
        mb_decode(&j->c, &j->hd, j->w, j->xyz, j->occ, j->rgb, tid);
      else
        for (int b = 0; b < j->nb; b++) mb_block(&j->c, &j->w[b], tid);
      if (tid == 0) tsum += HAP_perf_get_time_us() - t0;
    }
    if (tid == 0) {
      j->t[0] = tsum / j->iters;
      for (int i = 0; i < MB_NPROF; i++) j->t[1 + i] = j->c.prof[i];
      if (j->mode == 0)
        for (int r = 0; r < j->q; r++)
          for (int d = 0; d < MB_D; d++) j->y[(size_t)r * MB_D + d] = *mb_at(j->c.x, MB_KT, r, d);
    }
  }
  if (tid == 0 && j->codes[2] == 0) HAP_compute_res_hmx_unlock(j->ctx);
  if (hv == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

/* acquire VTCM + HMX, run job `j` on nthr SPMD threads, release */
static void launch(job_t* j, int nthr, int* codes) {
  const size_t need = 7u << 20; /* mcc_block.h layout: up to 6.4 MB */
  compute_res_attr_t attr;
  HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param_v2(&attr, need, 0, 0);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);
  unsigned int ctx = HAP_compute_res_acquire(&attr, 100000);
  codes[0] = (int)ctx;
  if (!ctx) return;
  void* vp = NULL;
  unsigned int vs = 0;
  HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vp, &vs);
  codes[3] = (int)vs;
  if (vp && vs >= need) {
    j->ctx = ctx, j->vtcm = (uint8_t*)vp, j->nthr = nthr, j->codes = codes;
    qurt_barrier_init(&j->bar, nthr);
    targ_t ta_[MAXT];
    qurt_thread_t tids[MAXT];
    int n = 0;
    for (; n < nthr; n++) {
      ta_[n].j = j, ta_[n].tid = n;
      qurt_thread_attr_t ta;
      qurt_thread_attr_init(&ta);
      qurt_thread_attr_set_stack_addr(&ta, g_stack[n]);
      qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
      qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
      if ((codes[4] = qurt_thread_create(&tids[n], &ta, worker, &ta_[n])) != 0) break;
    }
    for (int i = 0; i < n; i++) {
      int st;
      qurt_thread_join(tids[i], &st);
    }
    qurt_barrier_destroy(&j->bar);
  } else
    codes[4] = -1;
  HAP_compute_res_release(ctx);
}

static job_t g_job;

int mcc_hmx_rpc_decode(remote_handle64 h, int q, int hvx, int nthr, const float* xyz, int xyzLen, float* occ, int occLen, float* rgb, int rgbLen,
                       uint64* t, int tLen, int* codes, int codesLen) {
  if (q <= 0 || q > 1024 || q % 32 || xyzLen < q * 3 || occLen < q || rgbLen < q * 3 || tLen < 1 + MB_NPROF || codesLen < 6 || nthr < 1 ||
      nthr > MAXT)
    return AEE_EBADPARM;
  if (!g_head) return AEE_EBADSTATE;
  for (int b = 0; b < MB_BLOCKS; b++)
    if (!g_blob[b]) return AEE_EBADSTATE;
  memset(t, 0, tLen * sizeof(uint64));
  memset(codes, 0, codesLen * sizeof(int));
  job_t* j = &g_job;
  memset(j, 0, sizeof *j);
  j->q = q, j->nb = MB_BLOCKS, j->iters = 1, j->hvx = hvx, j->mode = 1, j->xyz = xyz, j->occ = occ, j->rgb = rgb, j->t = t;
  for (int b = 0; b < MB_BLOCKS; b++) mb_bind(&j->w[b], g_blob[b]);
  mb_bind_head(&j->hd, g_head);
  launch(j, nthr, codes);
  return 0;
}

int mcc_hmx_rpc_run(remote_handle64 h, int q, int nb, int iters, int hvx, int nthr, const uint16* x, int xLen, uint16* y, int yLen, uint64* t,
                    int tLen, int* codes, int codesLen) {
  if (q <= 0 || q > 1024 || q % 32 || nb < 1 || nb > MAXB || xLen < q * MB_D || yLen < q * MB_D || tLen < 1 + MB_NPROF ||
      codesLen < 6 || iters < 1 || nthr < 1 || nthr > MAXT)
    return AEE_EBADPARM;
  for (int b = 0; b < nb; b++)
    if (!g_blob[b]) return AEE_EBADSTATE;
  memset(t, 0, tLen * sizeof(uint64));
  memset(codes, 0, codesLen * sizeof(int));
  job_t* j = &g_job;
  memset(j, 0, sizeof *j);
  j->q = q, j->nb = nb, j->iters = iters, j->hvx = hvx, j->mode = 0, j->x = x, j->y = y, j->t = t;
  for (int b = 0; b < nb; b++) mb_bind(&j->w[b], g_blob[b]);
  launch(j, nthr, codes);
  return 0;
}
