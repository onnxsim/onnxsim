/* FastRPC skel around a tinygrad-generated HMX kernel (tg_kernel.c): the runtime part -- HMX power vote,
 * VTCM + HMX context, HVX + HMX lock on a QuRT worker thread -- comes from hmx_gemm/hmx_runtime.h, exactly as
 * for the hand kernel; the generated kernel only sees __hmx_vtcm / __hmx_gen. */
#include <stdlib.h>
#include <string.h>
#include "tg_hmx_rpc.h"
#include "hmx_runtime.h"
#include "tg_kernel.h"
extern unsigned long long HAP_perf_get_time_us(void);

unsigned char* __hmx_vtcm;
unsigned int __hmx_gen;
#include "tg_kernel.c"

int tg_hmx_rpc_open(const char* uri, remote_handle64* h) { *h = (remote_handle64)(uintptr_t)malloc(1); return 0; }
int tg_hmx_rpc_close(remote_handle64 h) { free((void*)(uintptr_t)h); return 0; }

typedef struct { hmx_rt_t* rt; int iters; tg_a_t* a; tg_b_t* b; tg_c_t* c; uint64* t; int* codes; } job_t;
#define STACK_SIZE (128 * 1024)  /* the generated kernel keeps whole 2 KB tiles as values */
static char g_stack[STACK_SIZE] __attribute__((aligned(128)));

static void worker(void* p) {
  job_t* j = (job_t*)p;
  j->codes[2] = qurt_hvx_lock(QURT_HVX_MODE_128B);
  j->codes[3] = j->codes[2] ? -1 : HAP_compute_res_hmx_lock(j->rt->ctx);
  if (j->codes[3] == 0) {
    unsigned long long t0 = HAP_perf_get_time_us();
    for (int it = 0; it < j->iters; it++) TG_CALL(j->c, j->a, j->b);
    j->t[0] = HAP_perf_get_time_us() - t0;
    HAP_compute_res_hmx_unlock(j->rt->ctx);
  }
  if (j->codes[2] == 0) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

#ifdef TG_SA
#define SA TG_SA  /* a conv's input: the flat padded image grid, not M x K */
#else
#define SA (TG_M * TG_K * (int)sizeof(tg_a_t))
#endif
#define SB (TG_K * TG_N * (int)sizeof(tg_b_t) + TG_B_EXTRA)
#ifdef TG_SC_BYTES
#define SC TG_SC_BYTES
#else
#define SC (TG_M * TG_N * (int)sizeof(tg_c_t))
#endif
int tg_hmx_rpc_run(remote_handle64 h, int iters, const uint8* a, int aLen, const uint8* b, int bLen, uint8* c, int cLen,
                   uint64* t, int tLen, int* codes, int codesLen) {
  if (tLen < 1 || codesLen < 6 || aLen < SA || bLen < SB || cLen < SC) return AEE_EBADPARM;
  memset(t, 0, tLen * sizeof(uint64)); memset(codes, 0, codesLen * sizeof(int));
  codes[0] = hmx_rt_power((void*)tg_hmx_rpc_run, 1);
  if (codes[0]) return 0;
  hmx_rt_t rt;
  if (hmx_rt_acquire(&rt, TG_VTCM_KB * 1024)) { codes[1] = -1; return 0; }  /* the tile pool layout the kernel was built for (HMX_VTCM_KB) */
  codes[1] = (int)rt.ctx; codes[4] = (int)rt.vtcm_bytes;
  /* generated kernels take 128-byte aligned buffers (no memalign in the DSP runtime's libc: align by hand) */
  char *raw = malloc(SA + SB + SC + 3 * 128);
  tg_a_t *ab = raw ? (tg_a_t*)(((uintptr_t)raw + 127) & ~(uintptr_t)127) : 0;
  tg_b_t *bb = ab ? (tg_b_t*)(((uintptr_t)ab + SA + 127) & ~(uintptr_t)127) : 0;
  tg_c_t *cb = bb ? (tg_c_t*)(((uintptr_t)bb + SB + 127) & ~(uintptr_t)127) : 0;
  if (raw) {
    memcpy(ab, a, SA); memcpy(bb, b, SB);
    __hmx_vtcm = rt.vtcm; __hmx_gen++;
    job_t j = {&rt, iters, ab, bb, cb, t, codes};
    qurt_thread_attr_t ta; qurt_thread_attr_init(&ta);
    qurt_thread_attr_set_stack_addr(&ta, g_stack); qurt_thread_attr_set_stack_size(&ta, STACK_SIZE);
    qurt_thread_attr_set_priority(&ta, qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_t tid; int st;
    codes[5] = qurt_thread_create(&tid, &ta, worker, &j);
    if (codes[5] == 0) qurt_thread_join(tid, &st);
    memcpy(c, cb, SC);
  }
  free(raw);
  hmx_rt_release(&rt);
  return 0;
}
