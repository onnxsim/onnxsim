// Single-precision GEMM for Intel Knights Corner (k1om): C = A*B, row-major, M % 64 == 0, K % 256 == 0, N % 128 == 0.
// KNC assembly microkernel (8 rows x 32 columns, 16 accumulators), A and B packed into contiguous panels, a persistent
// pthread pool, and work items of 64 rows x 128 columns over K blocks of 256.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef KC
#define KC 128   // K block (B panel 16 KB: stays in L1 next to the A stream)
#endif
#if KC != 128
#error "the KNC microkernel is written for KC == 128"
#endif
#ifndef PG
#define PG 4     // 32-column panels per work item
#endif
#ifndef RC
#define RC 32    // rows per work item
#endif
#ifndef KS
#define KS 1     // K groups: each group is its own work item and writes its own partial C (reduced at the end)
#endif

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

#ifndef PF_DIST
#define PF_DIST 4
#endif
#if PF_DIST > 0
#define PREFETCH_ASM \
    "vprefetch0 %c[pfb](%%rbx)\n\t" "vprefetch1 %c[pfb2](%%rbx)\n\t" \
    "vprefetch0 %c[pfa](%%rdx)\n\t" "vprefetch0 %c[pfa1](%%rdx)\n\t" "vprefetch0 %c[pfa2](%%rdx)\n\t" "vprefetch0 %c[pfa3](%%rdx)\n\t" \
    "vprefetch0 %c[pfa4](%%rdx)\n\t" "vprefetch0 %c[pfa5](%%rdx)\n\t" "vprefetch0 %c[pfa6](%%rdx)\n\t" "vprefetch0 %c[pfa7](%%rdx)\n\t"
#else
#define PREFETCH_ASM
#endif
// C[8][32] (rows ldc floats apart) = (first ? 0 : C) + Ap[KC][8] * Bp[KC][32], packed panels, in KNC assembly.
// Accumulators: row i in zmm(2i) and zmm(2i+1). B rows are two aligned loads, A values are FMA memory broadcasts.
static inline void kernel_8x32(const float *Ap, const float *Bp, float *C, long ldc, long K, int first) {
  long cnt = K / 4;
  long ldc_b = ldc * 4;
  __asm__ volatile(
    "mov %[C], %%rax\n\t"
    "test %[first], %[first]\n\t"
    "jnz 2f\n\t"
    "vmovaps (%%rax), %%zmm0\n\t"   "vmovaps 64(%%rax), %%zmm1\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps (%%rax), %%zmm2\n\t"   "vmovaps 64(%%rax), %%zmm3\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps (%%rax), %%zmm4\n\t"   "vmovaps 64(%%rax), %%zmm5\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps (%%rax), %%zmm6\n\t"   "vmovaps 64(%%rax), %%zmm7\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps (%%rax), %%zmm8\n\t"   "vmovaps 64(%%rax), %%zmm9\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps (%%rax), %%zmm10\n\t"  "vmovaps 64(%%rax), %%zmm11\n\t"  "add %[ldc], %%rax\n\t"
    "vmovaps (%%rax), %%zmm12\n\t"  "vmovaps 64(%%rax), %%zmm13\n\t"  "add %[ldc], %%rax\n\t"
    "vmovaps (%%rax), %%zmm14\n\t"  "vmovaps 64(%%rax), %%zmm15\n\t"
    "jmp 3f\n\t"
    "2:\n\t"
    "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"   "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
    "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"   "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
    "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"   "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
    "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"   "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
    "vpxord %%zmm8, %%zmm8, %%zmm8\n\t"   "vpxord %%zmm9, %%zmm9, %%zmm9\n\t"
    "vpxord %%zmm10, %%zmm10, %%zmm10\n\t" "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
    "vpxord %%zmm12, %%zmm12, %%zmm12\n\t" "vpxord %%zmm13, %%zmm13, %%zmm13\n\t"
    "vpxord %%zmm14, %%zmm14, %%zmm14\n\t" "vpxord %%zmm15, %%zmm15, %%zmm15\n\t"
    "3:\n\t"
    "mov %[Ap], %%rdx\n\t"
    "mov %[Bp], %%rbx\n\t"
    "mov %[cnt], %%rcx\n\t"
    "1:\n\t"
    PREFETCH_ASM
    "vmovaps 0(%%rbx), %%zmm16\n\t"
    "vmovaps 64(%%rbx), %%zmm17\n\t"
    "vfmadd231ps 0(%%rdx)%{1to16%}, %%zmm16, %%zmm0\n\t"
    "vfmadd231ps 0(%%rdx)%{1to16%}, %%zmm17, %%zmm1\n\t"
    "vfmadd231ps 512(%%rdx)%{1to16%}, %%zmm16, %%zmm2\n\t"
    "vfmadd231ps 512(%%rdx)%{1to16%}, %%zmm17, %%zmm3\n\t"
    "vfmadd231ps 1024(%%rdx)%{1to16%}, %%zmm16, %%zmm4\n\t"
    "vfmadd231ps 1024(%%rdx)%{1to16%}, %%zmm17, %%zmm5\n\t"
    "vfmadd231ps 1536(%%rdx)%{1to16%}, %%zmm16, %%zmm6\n\t"
    "vfmadd231ps 1536(%%rdx)%{1to16%}, %%zmm17, %%zmm7\n\t"
    "vfmadd231ps 2048(%%rdx)%{1to16%}, %%zmm16, %%zmm8\n\t"
    "vfmadd231ps 2048(%%rdx)%{1to16%}, %%zmm17, %%zmm9\n\t"
    "vfmadd231ps 2560(%%rdx)%{1to16%}, %%zmm16, %%zmm10\n\t"
    "vfmadd231ps 2560(%%rdx)%{1to16%}, %%zmm17, %%zmm11\n\t"
    "vfmadd231ps 3072(%%rdx)%{1to16%}, %%zmm16, %%zmm12\n\t"
    "vfmadd231ps 3072(%%rdx)%{1to16%}, %%zmm17, %%zmm13\n\t"
    "vfmadd231ps 3584(%%rdx)%{1to16%}, %%zmm16, %%zmm14\n\t"
    "vfmadd231ps 3584(%%rdx)%{1to16%}, %%zmm17, %%zmm15\n\t"
    "vmovaps 128(%%rbx), %%zmm18\n\t"
    "vmovaps 192(%%rbx), %%zmm19\n\t"
    "vfmadd231ps 4(%%rdx)%{1to16%}, %%zmm18, %%zmm0\n\t"
    "vfmadd231ps 4(%%rdx)%{1to16%}, %%zmm19, %%zmm1\n\t"
    "vfmadd231ps 516(%%rdx)%{1to16%}, %%zmm18, %%zmm2\n\t"
    "vfmadd231ps 516(%%rdx)%{1to16%}, %%zmm19, %%zmm3\n\t"
    "vfmadd231ps 1028(%%rdx)%{1to16%}, %%zmm18, %%zmm4\n\t"
    "vfmadd231ps 1028(%%rdx)%{1to16%}, %%zmm19, %%zmm5\n\t"
    "vfmadd231ps 1540(%%rdx)%{1to16%}, %%zmm18, %%zmm6\n\t"
    "vfmadd231ps 1540(%%rdx)%{1to16%}, %%zmm19, %%zmm7\n\t"
    "vfmadd231ps 2052(%%rdx)%{1to16%}, %%zmm18, %%zmm8\n\t"
    "vfmadd231ps 2052(%%rdx)%{1to16%}, %%zmm19, %%zmm9\n\t"
    "vfmadd231ps 2564(%%rdx)%{1to16%}, %%zmm18, %%zmm10\n\t"
    "vfmadd231ps 2564(%%rdx)%{1to16%}, %%zmm19, %%zmm11\n\t"
    "vfmadd231ps 3076(%%rdx)%{1to16%}, %%zmm18, %%zmm12\n\t"
    "vfmadd231ps 3076(%%rdx)%{1to16%}, %%zmm19, %%zmm13\n\t"
    "vfmadd231ps 3588(%%rdx)%{1to16%}, %%zmm18, %%zmm14\n\t"
    "vfmadd231ps 3588(%%rdx)%{1to16%}, %%zmm19, %%zmm15\n\t"
    "vmovaps 256(%%rbx), %%zmm16\n\t"
    "vmovaps 320(%%rbx), %%zmm17\n\t"
    "vfmadd231ps 8(%%rdx)%{1to16%}, %%zmm16, %%zmm0\n\t"
    "vfmadd231ps 8(%%rdx)%{1to16%}, %%zmm17, %%zmm1\n\t"
    "vfmadd231ps 520(%%rdx)%{1to16%}, %%zmm16, %%zmm2\n\t"
    "vfmadd231ps 520(%%rdx)%{1to16%}, %%zmm17, %%zmm3\n\t"
    "vfmadd231ps 1032(%%rdx)%{1to16%}, %%zmm16, %%zmm4\n\t"
    "vfmadd231ps 1032(%%rdx)%{1to16%}, %%zmm17, %%zmm5\n\t"
    "vfmadd231ps 1544(%%rdx)%{1to16%}, %%zmm16, %%zmm6\n\t"
    "vfmadd231ps 1544(%%rdx)%{1to16%}, %%zmm17, %%zmm7\n\t"
    "vfmadd231ps 2056(%%rdx)%{1to16%}, %%zmm16, %%zmm8\n\t"
    "vfmadd231ps 2056(%%rdx)%{1to16%}, %%zmm17, %%zmm9\n\t"
    "vfmadd231ps 2568(%%rdx)%{1to16%}, %%zmm16, %%zmm10\n\t"
    "vfmadd231ps 2568(%%rdx)%{1to16%}, %%zmm17, %%zmm11\n\t"
    "vfmadd231ps 3080(%%rdx)%{1to16%}, %%zmm16, %%zmm12\n\t"
    "vfmadd231ps 3080(%%rdx)%{1to16%}, %%zmm17, %%zmm13\n\t"
    "vfmadd231ps 3592(%%rdx)%{1to16%}, %%zmm16, %%zmm14\n\t"
    "vfmadd231ps 3592(%%rdx)%{1to16%}, %%zmm17, %%zmm15\n\t"
    "vmovaps 384(%%rbx), %%zmm18\n\t"
    "vmovaps 448(%%rbx), %%zmm19\n\t"
    "vfmadd231ps 12(%%rdx)%{1to16%}, %%zmm18, %%zmm0\n\t"
    "vfmadd231ps 12(%%rdx)%{1to16%}, %%zmm19, %%zmm1\n\t"
    "vfmadd231ps 524(%%rdx)%{1to16%}, %%zmm18, %%zmm2\n\t"
    "vfmadd231ps 524(%%rdx)%{1to16%}, %%zmm19, %%zmm3\n\t"
    "vfmadd231ps 1036(%%rdx)%{1to16%}, %%zmm18, %%zmm4\n\t"
    "vfmadd231ps 1036(%%rdx)%{1to16%}, %%zmm19, %%zmm5\n\t"
    "vfmadd231ps 1548(%%rdx)%{1to16%}, %%zmm18, %%zmm6\n\t"
    "vfmadd231ps 1548(%%rdx)%{1to16%}, %%zmm19, %%zmm7\n\t"
    "vfmadd231ps 2060(%%rdx)%{1to16%}, %%zmm18, %%zmm8\n\t"
    "vfmadd231ps 2060(%%rdx)%{1to16%}, %%zmm19, %%zmm9\n\t"
    "vfmadd231ps 2572(%%rdx)%{1to16%}, %%zmm18, %%zmm10\n\t"
    "vfmadd231ps 2572(%%rdx)%{1to16%}, %%zmm19, %%zmm11\n\t"
    "vfmadd231ps 3084(%%rdx)%{1to16%}, %%zmm18, %%zmm12\n\t"
    "vfmadd231ps 3084(%%rdx)%{1to16%}, %%zmm19, %%zmm13\n\t"
    "vfmadd231ps 3596(%%rdx)%{1to16%}, %%zmm18, %%zmm14\n\t"
    "vfmadd231ps 3596(%%rdx)%{1to16%}, %%zmm19, %%zmm15\n\t"
    "add $512, %%rbx\n\t"
    "add $16, %%rdx\n\t"
    "dec %%rcx\n\t"
    "jnz 1b\n\t"
    "mov %[C], %%rax\n\t"
    "vmovaps %%zmm0, (%%rax)\n\t"   "vmovaps %%zmm1, 64(%%rax)\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps %%zmm2, (%%rax)\n\t"   "vmovaps %%zmm3, 64(%%rax)\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps %%zmm4, (%%rax)\n\t"   "vmovaps %%zmm5, 64(%%rax)\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps %%zmm6, (%%rax)\n\t"   "vmovaps %%zmm7, 64(%%rax)\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps %%zmm8, (%%rax)\n\t"   "vmovaps %%zmm9, 64(%%rax)\n\t"   "add %[ldc], %%rax\n\t"
    "vmovaps %%zmm10, (%%rax)\n\t"  "vmovaps %%zmm11, 64(%%rax)\n\t"  "add %[ldc], %%rax\n\t"
    "vmovaps %%zmm12, (%%rax)\n\t"  "vmovaps %%zmm13, 64(%%rax)\n\t"  "add %[ldc], %%rax\n\t"
    "vmovaps %%zmm14, (%%rax)\n\t"  "vmovaps %%zmm15, 64(%%rax)\n\t"
    :
    : [C] "m"(C), [ldc] "r"(ldc_b), [Ap] "m"(Ap), [Bp] "m"(Bp), [cnt] "m"(cnt), [first] "r"(first),
      [pfb] "i"(PF_DIST * 128), [pfb2] "i"(PF_DIST * 512), [pfa] "i"(PF_DIST * 16),
      [pfa1] "i"(512 + PF_DIST * 16), [pfa2] "i"(1024 + PF_DIST * 16), [pfa3] "i"(1536 + PF_DIST * 16),
      [pfa4] "i"(2048 + PF_DIST * 16), [pfa5] "i"(2560 + PF_DIST * 16), [pfa6] "i"(3072 + PF_DIST * 16), [pfa7] "i"(3584 + PF_DIST * 16)
    : "rax", "rbx", "rcx", "rdx", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
      "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
      "memory", "cc");
}

// ---- persistent thread pool ----
typedef void (*task_fn)(void *, int);
static struct {
  pthread_mutex_t m; pthread_cond_t start, done;
  unsigned gen; int remaining, nthreads; task_fn fn; void *arg; int started;
} pool = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0, NULL, NULL, 0};

// Pin pool thread t to one hardware thread: the card numbers CPUs in groups of four per core, so threads 0..56 go to
// the first hardware thread of each core, and later threads take the other three (measured: 1.4x at 228 threads).
static void pin_thread(int t) {
  int cpu = (4 * (t % 57) + 1 + t / 57) % 228;
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
  sched_setaffinity(0, sizeof set, &set);
}

static void *pool_thread(void *idp) {
  int id = (int)(long)idp;
#ifndef NOPIN
  pin_thread(id);
#endif
  unsigned seen = 0;
  for (;;) {
    pthread_mutex_lock(&pool.m);
    while (pool.gen == seen) pthread_cond_wait(&pool.start, &pool.m);
    seen = pool.gen;
    task_fn fn = pool.fn; void *arg = pool.arg;
    pthread_mutex_unlock(&pool.m);
    fn(arg, id);
    pthread_mutex_lock(&pool.m);
    if (--pool.remaining == 0) pthread_cond_signal(&pool.done);
    pthread_mutex_unlock(&pool.m);
  }
  return NULL;
}

static void pool_init(int n) {
  if (pool.started) return;
  pool.nthreads = n;
  for (int t = 0; t < n; t++) {
    pthread_t th;
    pthread_create(&th, NULL, pool_thread, (void *)(long)t);
    pthread_detach(th);
  }
  pool.started = 1;
}

static void pool_run(task_fn fn, void *arg) {
  pthread_mutex_lock(&pool.m);
  pool.fn = fn; pool.arg = arg; pool.remaining = pool.nthreads; pool.gen++;
  pthread_cond_broadcast(&pool.start);
  while (pool.remaining > 0) pthread_cond_wait(&pool.done, &pool.m);
  pthread_mutex_unlock(&pool.m);
}

// ---- GEMM ----
typedef struct {
  const float *A, *B; float *C; float *Apk, *Bpk;
  int M, N, K, nkc; volatile int next; int nA, nB, items, groups_n, do_pack_B; float *Cpart; int A_src;
} job_t;

static void pack_task(void *argp, int id) {
  job_t *j = (job_t *)argp;
  (void)id;
  const int K = j->K, N = j->N, nkc = j->nkc;
  for (;;) {
    int item = __sync_fetch_and_add(&j->next, 1);
    if (item >= j->nA + j->nB) break;
    if (item < j->nA) {                       // one 8-row block of A: Apk[rb][kc][i][KC] (plain row segments)
      int rb = item;
      for (int kc = 0; kc < nkc; kc++)
        for (int i = 0; i < 8; i++)
          memcpy(j->Apk + (((size_t)rb * nkc + kc) * 8 + i) * KC, j->A + (size_t)(rb * 8 + i) * K + kc * KC, KC * 4);
    } else if (j->do_pack_B) {                // one 32-column panel of B: Bpk[p][kc][kk][32]
      int p = item - j->nA;
      for (int kc = 0; kc < nkc; kc++)
        for (int kk = 0; kk < KC; kk++)
          memcpy(j->Bpk + (((size_t)p * nkc + kc) * KC + kk) * 32, j->B + (size_t)(kc * KC + kk) * N + p * 32, 128);
    }
  }
}

static void compute_task(void *argp, int id) {
  job_t *j = (job_t *)argp;
  (void)id;
  const int N = j->N, K = j->K, nkc = j->nkc;
  float *Ab = aligned_alloc(64, (size_t)RC * KC * 4);   // this item's A rows for one K block (row-major, KC per row)
  for (;;) {
    int item = __sync_fetch_and_add(&j->next, 1);
    if (item >= j->items) break;
    int g = item % KS, rest = item / KS;
    int rchunk = rest / j->groups_n, pg = rest % j->groups_n;
    int rb0 = rchunk * (RC / 8), p0 = pg * PG;
    int kc_lo = g * nkc / KS, kc_hi = (g + 1) * nkc / KS;
    float *Cg = g == 0 ? j->C : j->Cpart + (size_t)(g - 1) * j->M * N;
    for (int kc = kc_lo; kc < kc_hi; kc++) {
      if (j->A_src) {   // pack A in place: rows rb0*8 .. rb0*8+RC-1, K block kc
        for (int r = 0; r < RC; r++)
          memcpy(Ab + (size_t)r * KC, j->A + (size_t)(rb0 * 8 + r) * K + kc * KC, KC * 4);
      }
      for (int q = 0; q < PG; q++) {
        const float *Bp = j->Bpk + (((size_t)(p0 + q)) * nkc + kc) * KC * 32;   // 32 KB, reused over the chunk
        for (int rb = rb0; rb < rb0 + RC / 8; rb++) {
          const float *Ap = j->A_src ? Ab + (size_t)(rb - rb0) * 8 * KC : j->Apk + ((size_t)rb * nkc + kc) * KC * 8;
          kernel_8x32(Ap, Bp, Cg + (size_t)(rb * 8) * N + (p0 + q) * 32, N, KC, kc == kc_lo);
        }
      }
    }
  }
  free(Ab);
}

// C += Cpart[g] for g = 1..KS-1, over row chunks.
static void reduce_task(void *argp, int id) {
  job_t *j = (job_t *)argp;
  (void)id;
  const int N = j->N;
  for (;;) {
    int r = __sync_fetch_and_add(&j->next, 1);
    if (r >= j->M) break;
    float *dst = j->C + (size_t)r * N;
    for (int g = 1; g < KS; g++) {
      const float *src = j->Cpart + ((size_t)(g - 1) * j->M + r) * N;
      for (int c = 0; c < N; c++) dst[c] += src[c];
    }
  }
}

static float *Apk_buf = NULL, *Bpk_buf = NULL, *Cpart_buf = NULL;
static size_t Apk_cap = 0, Bpk_cap = 0, Cpart_cap = 0;

// Pack B[K][N] once (weights): Bpk must hold N*K floats, 64-byte aligned.
void sgemm_knc_pack_B(const float *B, float *Bpk, int K, int N, int nthreads) {
  pool_init(nthreads);
  job_t j = {0};
  j.B = B; j.Bpk = Bpk; j.K = K; j.N = N; j.nkc = K / KC; j.nA = 0; j.nB = N / 32; j.do_pack_B = 1;
  j.next = 0; pool_run(pack_task, &j);
}

// C = A*B with B already packed by sgemm_knc_pack_B. Packs A into Apk (scratch, M*K floats), then computes.
void sgemm_knc_run(const float *A, const float *Bpk, float *C, float *Apk, int M, int N, int K, int nthreads) {
  pool_init(nthreads);
  job_t j = {0};
  j.A = A; j.Bpk = Bpk; j.C = C; j.Apk = Apk;
  j.M = M; j.N = N; j.K = K; j.nkc = K / KC;
  j.nA = M / 8; j.nB = 0; j.do_pack_B = 0;
#ifndef INLINE_A
#define INLINE_A 0
#endif
  j.A_src = INLINE_A;
  j.groups_n = N / (32 * PG);
  j.items = (M / RC) * j.groups_n * KS;
  if (KS > 1) {
    size_t need = (size_t)(KS - 1) * M * N;
    if (need > Cpart_cap) { free(Cpart_buf); Cpart_buf = aligned_alloc(64, need * 4); Cpart_cap = need; }
    j.Cpart = Cpart_buf;
  }
  double t0 = now_s();
  j.next = 0; if (!j.A_src) pool_run(pack_task, &j);
  double t1 = now_s();
  j.next = 0; pool_run(compute_task, &j);
  if (KS > 1) { j.next = 0; pool_run(reduce_task, &j); }
  double t2 = now_s();
  if (getenv("GEMM_TIMING")) printf("  packA %.2f ms  compute %.2f ms\n", (t1 - t0) * 1e3, (t2 - t1) * 1e3);
}

// C = A*B, packing both operands (the general call).
void sgemm_knc(const float *A, const float *B, float *C, int M, int N, int K, int nthreads) {
  if ((size_t)M * K > Apk_cap) { free(Apk_buf); Apk_buf = aligned_alloc(64, (size_t)M * K * 4); Apk_cap = (size_t)M * K; }
  if ((size_t)N * K > Bpk_cap) { free(Bpk_buf); Bpk_buf = aligned_alloc(64, (size_t)N * K * 4); Bpk_cap = (size_t)N * K; }
  sgemm_knc_pack_B(B, Bpk_buf, K, N, nthreads);
  sgemm_knc_run(A, Bpk_buf, C, Apk_buf, M, N, K, nthreads);
}

static double maxerr_sampled(const float *A, const float *B, const float *C, int M, int N, int K) {
  double maxerr = 0;
  for (int t = 0; t < 64; t++) {
    int i = (t * 37) % M, jx = (t * 91) % N; double r = 0;
    for (int k = 0; k < K; k++) r += (double)A[(size_t)i * K + k] * B[(size_t)k * N + jx];
    double e = r - C[(size_t)i * N + jx]; if (e < 0) e = -e; if (e > maxerr) maxerr = e;
  }
  return maxerr;
}

int main(int argc, char **argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048, N = argc > 2 ? atoi(argv[2]) : 2048, K = argc > 3 ? atoi(argv[3]) : 2048;
  int nth = argc > 4 ? atoi(argv[4]) : 171, reps = argc > 5 ? atoi(argv[5]) : 3;
  float *A = aligned_alloc(64, (size_t)M * K * 4), *B = aligned_alloc(64, (size_t)K * N * 4), *C = aligned_alloc(64, (size_t)M * N * 4);
  srand(1);
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (float)rand() / RAND_MAX - 0.5f;
  for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (float)rand() / RAND_MAX - 0.5f;
  float *Bpk = aligned_alloc(64, (size_t)K * N * 4), *Apk = aligned_alloc(64, (size_t)M * K * 4);
  sgemm_knc_pack_B(B, Bpk, K, N, nth);
  sgemm_knc_run(A, Bpk, C, Apk, M, N, K, nth);
  double maxerr = maxerr_sampled(A, B, C, M, N, K);
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    double t0 = now_s(); sgemm_knc_run(A, Bpk, C, Apk, M, N, K, nth); double dt = now_s() - t0; if (dt < best) best = dt;
  }
  double gflops = 2.0 * M * N * K / best * 1e-9;
  printf("M=%d N=%d K=%d threads=%d maxerr(sampled)=%g\n", M, N, K, nth, maxerr);
  printf("best (B prepacked, A packed per call) %.3f ms  %.1f GFLOPS  (%.1f%% of 2006 GFLOPS peak)\n", best * 1e3, gflops, 100.0 * gflops / 2006.4);
  return maxerr < 1e-2 ? 0 : 1;
}
