// Single-precision GEMM for Intel Knights Corner (k1om): C = A*B, row-major. Fast path: M % RC == 0, K % 128 == 0,
// N % (32 * PG) == 0; sgemm_knc_any takes any shape. KNC assembly microkernel (8 rows x 32 columns, 16 accumulators), A and B
// packed into panels, a persistent pinned thread pool, work items of RC rows x PG panels over K blocks of 128.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#ifndef KC
#define KC 128   // K block (B panel 16 KB: stays in L1 next to the A stream)
#endif
#if KC != 128
#error "the KNC microkernel is written for KC == 128"
#endif
#ifndef PG
#define PG 2     // 32-column panels per work item (PG=2, RC=16 measured best overall; PG=1 is faster below 1024^3)
#endif
#ifndef RC
#define RC 16    // rows per work item
#endif
#ifndef KS
#define KS 1     // K groups: each group is its own work item and writes its own partial C (reduced at the end)
#endif


// Alignment of the large buffers.
#ifndef BIGALIGN
#define BIGALIGN 64   // 2 MB alignment and MAP_HUGETLB (HUGETLB=1) were not better in interleaved runs
#endif
// Large-buffer allocation: explicit 2 MB huge pages (MAP_HUGETLB) when HUGETLB=1 is set and pages are reserved
// (/proc/sys/vm/nr_hugepages), else aligned_alloc. Buffers from here are never freed (benchmark-lifetime).
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
static void *big_alloc(size_t n) {
  size_t sz = (n + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
  if (getenv("HUGETLB")) {
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) return p;
    fprintf(stderr, "big_alloc: MAP_HUGETLB failed for %zu bytes, using aligned_alloc\n", sz);
  }
  return aligned_alloc(BIGALIGN, (n + BIGALIGN - 1) / BIGALIGN * BIGALIGN);
}

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// Cores used: 57, or 56 to leave core 56 (which runs the card's OS on CPU 0) alone.
#ifndef NCU
#define NCU 57
#endif
#ifndef PF_DIST
#define PF_DIST 4   // PF=2 and PF=8/16 were not better in interleaved runs (medians within the +-15% noise)
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
  int cpu = (4 * (t % NCU) + 1 + t / NCU) % 228;
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
  sched_setaffinity(0, sizeof set, &set);
}

// Hand-off: a run publishes fn/arg and bumps gen; workers spin on gen (with the KNC `delay` instruction, which frees the
// core's issue slots for the other hardware threads) for up to POOL_SPIN_MS after their last task, then sleep on the
// condition variable. The caller spins on `remaining`. Measured: an empty run cost 4-10 ms with condition variables only.
#ifndef POOL_SPIN_MS
#define POOL_SPIN_MS 100
#endif
static volatile unsigned pool_gen;
static volatile int pool_remaining, pool_sleepers;
static inline void cpu_delay(long cycles) { __asm__ volatile("delay %0" :: "r"(cycles)); }
static void *pool_thread(void *idp) {
  int id = (int)(long)idp;
#ifndef NOPIN
  pin_thread(id);
#endif
  unsigned seen = 0;
  for (;;) {
    double t_end = now_s() + POOL_SPIN_MS * 1e-3;
    long spins = 0;
    while (pool_gen == seen) {
      cpu_delay(200);
      if ((++spins & 1023) == 0 && now_s() > t_end) {
        pthread_mutex_lock(&pool.m);
        pool_sleepers++;
        while (pool_gen == seen) pthread_cond_wait(&pool.start, &pool.m);
        pool_sleepers--;
        pthread_mutex_unlock(&pool.m);
      }
    }
    __sync_synchronize();
    seen = pool_gen;
    task_fn fn = pool.fn; void *arg = pool.arg;
    fn(arg, id);
    __sync_fetch_and_sub(&pool_remaining, 1);
  }
  return NULL;
}

static void pool_init(int n) {
  if (pool.started) return;
  pool.nthreads = n;
  // The calling thread is task 0 (and is pinned like it): with a pinned spinning worker on every hardware thread, an
  // unpinned caller would time-share a CPU with one (measured: 39 ms per empty run at 228 threads).
#ifndef NOPIN
  pin_thread(0);
#endif
  for (int t = 1; t < n; t++) {
    pthread_t th;
    pthread_create(&th, NULL, pool_thread, (void *)(long)t);
    pthread_detach(th);
  }
  pool.started = 1;
}

static void pool_run(task_fn fn, void *arg) {
  pool.fn = fn; pool.arg = arg; pool_remaining = pool.nthreads - 1;
  __sync_synchronize();
  pthread_mutex_lock(&pool.m);                          // a sleeper re-checks pool_gen under the mutex: no lost wake-up
  pool_gen++;
  if (pool_sleepers) pthread_cond_broadcast(&pool.start);
  pthread_mutex_unlock(&pool.m);
  fn(arg, 0);
  while (pool_remaining > 0) cpu_delay(100);
  __sync_synchronize();
}

// ---- GEMM ----
typedef struct {
  const float *A, *B; float *C; float *Apk, *Bpk;
  int M, N, K, nkc; volatile int next; int nA, nB, items, groups_n, do_pack_B; float *Cpart; int A_src; int *pdone; int nPack;
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

// Overlapped run: one pass over a shared counter. Items [0, nPack) pack one 8-row block of A; items after that compute
// one work item, and wait until the blocks of its row chunk are packed (pdone counts packed blocks per chunk).
static void overlap_task(void *argp, int id) {
  job_t *j = (job_t *)argp;
  (void)id;
  const int K = j->K, N = j->N, nkc = j->nkc, BPC = RC / 8;
  for (;;) {
    int item = __sync_fetch_and_add(&j->next, 1);
    if (item >= j->nPack + j->items) break;
    if (item < j->nPack) {
      int rb = item;
      for (int kc = 0; kc < nkc; kc++)
        for (int i = 0; i < 8; i++)
          memcpy(j->Apk + (((size_t)rb * nkc + kc) * 8 + i) * KC, j->A + (size_t)(rb * 8 + i) * K + kc * KC, KC * 4);
      __sync_fetch_and_add(&j->pdone[rb / BPC], 1);
      continue;
    }
    int w = item - j->nPack;
    int g = w % KS, rest = w / KS;
    int rchunk = rest / j->groups_n, pg = rest % j->groups_n;
    while (__sync_fetch_and_add(&j->pdone[rchunk], 0) < BPC) { }   // wait for this chunk's A blocks
    int rb0 = rchunk * BPC, p0 = pg * PG;
    int kc_lo = g * nkc / KS, kc_hi = (g + 1) * nkc / KS;
    float *Cg = g == 0 ? j->C : j->Cpart + (size_t)(g - 1) * j->M * N;
    for (int kc = kc_lo; kc < kc_hi; kc++)
      for (int q = 0; q < PG; q++) {
        const float *Bp = j->Bpk + (((size_t)(p0 + q)) * nkc + kc) * KC * 32;
        for (int rb = rb0; rb < rb0 + BPC; rb++)
          kernel_8x32(j->Apk + ((size_t)rb * nkc + kc) * KC * 8, Bp, Cg + (size_t)(rb * 8) * N + (p0 + q) * 32,
                      N, KC, kc == kc_lo);
      }
  }
}

// C = A*B with A packing overlapped with the compute (see overlap_task). B must be packed.
void sgemm_knc_run_overlap(const float *A, const float *Bpk, float *C, float *Apk, int M, int N, int K, int nthreads) {
  pool_init(nthreads);
  static int *pdone = NULL; static int pdone_cap = 0;
  int nchunks = M / RC;
  if (nchunks > pdone_cap) { free(pdone); pdone = malloc(sizeof(int) * nchunks); pdone_cap = nchunks; }
  memset(pdone, 0, sizeof(int) * nchunks);
  job_t j = {0};
  j.A = A; j.Bpk = Bpk; j.C = C; j.Apk = Apk;
  j.M = M; j.N = N; j.K = K; j.nkc = K / KC;
  j.nA = M / 8; j.nB = 0; j.do_pack_B = 0;
  j.groups_n = N / (32 * PG);
  j.items = nchunks * j.groups_n * KS;
  j.nPack = M / 8;
  j.pdone = pdone;
  if (KS > 1) {
    size_t need = (size_t)(KS - 1) * M * N;
    if (need > Cpart_cap) { free(Cpart_buf); Cpart_buf = aligned_alloc(64, need * 4); Cpart_cap = need; }
    j.Cpart = Cpart_buf;
  }
  j.next = 0; pool_run(overlap_task, &j);
  if (KS > 1) { j.next = 0; pool_run(reduce_task, &j); }
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
  if ((size_t)M * K > Apk_cap) { free(Apk_buf); Apk_buf = aligned_alloc(BIGALIGN, ((size_t)M * K * 4 + BIGALIGN - 1) / BIGALIGN * BIGALIGN); Apk_cap = (size_t)M * K; }
  if ((size_t)N * K > Bpk_cap) { free(Bpk_buf); Bpk_buf = aligned_alloc(BIGALIGN, ((size_t)N * K * 4 + BIGALIGN - 1) / BIGALIGN * BIGALIGN); Bpk_cap = (size_t)N * K; }
  sgemm_knc_pack_B(B, Bpk_buf, K, N, nthreads);
  sgemm_knc_run(A, Bpk_buf, C, Apk_buf, M, N, K, nthreads);
}


// Any shape: pads M to 32, N to 128 and K to 128 with zeros (the fast path needs multiples), runs, and copies the
// M x N result back. Zero padding leaves the valid part of C unchanged.
static size_t pad_to(size_t x, size_t m) { return (x + m - 1) / m * m; }
void sgemm_knc_any(const float *A, const float *B, float *C, int M, int N, int K, int nthreads);   // defined below


// Full-K microkernel: C[8][32] = A[8][nkc*KC] * B[nkc*KC][32] with the accumulators kept in registers across all K
// blocks (C written once). Ap: Apk block of row block rb, kc = 0 (kc blocks are 8*KC floats apart); Bp: panel, kc = 0
// (kc blocks are KC*32 floats apart). The outer loop moves rdx by 3584 bytes so each K block starts at 4 KB.
static inline void kernel_full(const float *Ap, const float *Bp, float *C, long ldc, long nkc) {
  long cnt = KC / 4;
  long ldc_b = ldc * 4;
  __asm__ volatile(
        "vpxord %%zmm0, %%zmm0, %%zmm0\n\t" "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
    "vpxord %%zmm2, %%zmm2, %%zmm2\n\t" "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
    "vpxord %%zmm4, %%zmm4, %%zmm4\n\t" "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
    "vpxord %%zmm6, %%zmm6, %%zmm6\n\t" "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
    "vpxord %%zmm8, %%zmm8, %%zmm8\n\t" "vpxord %%zmm9, %%zmm9, %%zmm9\n\t"
    "vpxord %%zmm10, %%zmm10, %%zmm10\n\t" "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
    "vpxord %%zmm12, %%zmm12, %%zmm12\n\t" "vpxord %%zmm13, %%zmm13, %%zmm13\n\t"
    "vpxord %%zmm14, %%zmm14, %%zmm14\n\t" "vpxord %%zmm15, %%zmm15, %%zmm15\n\t"

    "mov %[Ap], %%rdx\n\t"
    "mov %[Bp], %%rbx\n\t"
    "mov %[nkc], %%r8\n\t"
    "5:\n\t"
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
    "add $3584, %%rdx\n\t"
    "dec %%r8\n\t"
    "jnz 5b\n\t"
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
    : [C] "m"(C), [ldc] "r"(ldc_b), [Ap] "m"(Ap), [Bp] "m"(Bp), [cnt] "m"(cnt), [nkc] "m"(nkc),
      [pfb] "i"(PF_DIST * 128), [pfb2] "i"(PF_DIST * 512), [pfa] "i"(PF_DIST * 16),
      [pfa1] "i"(512 + PF_DIST * 16), [pfa2] "i"(1024 + PF_DIST * 16), [pfa3] "i"(1536 + PF_DIST * 16),
      [pfa4] "i"(2048 + PF_DIST * 16), [pfa5] "i"(2560 + PF_DIST * 16), [pfa6] "i"(3072 + PF_DIST * 16), [pfa7] "i"(3584 + PF_DIST * 16)
    : "rax", "rbx", "rcx", "rdx", "r8", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
      "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
      "memory", "cc");
}


// ---- core-shared schedule ----
// The four hardware threads of a core (threads c, c+57, c+114, c+171 under pin_thread's mapping) work on the same
// 128-column panel group at the same time, so they share one B panel in L1 (measured: 1112 vs 903 GFLOPS kernel-only
// at 228 threads). A unit is one panel group x four row chunks (one per hardware thread of the core); core c takes
// units c, c+57, c+114, ... The packing of A is a separate phase before this (requires M % 128 == 0).
#define NCORES NCU
typedef struct { const float *A, *Bpk, *Apk; float *C; int M, N, K, nkc, nsr, npg; } static_job_t;
static void static_task(void *argp, int id) {
  static_job_t *j = (static_job_t *)argp;
  const int N = j->N, nkc = j->nkc, BPC = RC / 8;
  if (id >= 4 * NCORES) return;
  int c = id % NCORES, jj = id / NCORES;
  int units = j->nsr * j->npg;
  for (int u = c; u < units; u += NCORES) {
    int pg = u / j->nsr, sr = u % j->nsr;
    int rchunk = sr * 4 + jj, rb0 = rchunk * BPC, p0 = pg * PG;
    for (int kc = 0; kc < nkc; kc++)
      for (int q = 0; q < PG; q++) {
        const float *Bp = j->Bpk + (((size_t)(p0 + q)) * nkc + kc) * KC * 32;
        for (int rb = rb0; rb < rb0 + BPC; rb++)
          kernel_8x32(j->Apk + ((size_t)rb * nkc + kc) * KC * 8, Bp, j->C + (size_t)(rb * 8) * N + (p0 + q) * 32, N, KC, kc == 0);
      }
  }
}
void sgemm_knc_run_static(const float *A, const float *Bpk, float *C, float *Apk, int M, int N, int K, int nthreads) {
  pool_init(nthreads);
  job_t pj = {0};
  pj.A = A; pj.Apk = Apk; pj.M = M; pj.N = N; pj.K = K; pj.nkc = K / KC; pj.nA = M / 8; pj.nB = 0; pj.do_pack_B = 0;
  pj.next = 0; pool_run(pack_task, &pj);
  static_job_t j = {A, Bpk, Apk, C, M, N, K, K / KC, M / (RC * 4), N / (32 * PG)};
  pool_run(static_task, &j);
}


// ---- shared-panel schedule with overlapped A packing ----
// A unit is one panel group (PG panels) x four 8-row blocks ("quad"), one block per hardware thread of a core (thread
// c + 57*jj). A quad's blocks are packed into Apk ahead of use: the core that claims a unit also claims the pack
// target quad+4 (leader CAS on sh_state), and its four threads pack one block each while computing. sh_state: 0 not
// packed, 1 being packed, 2 ready. The first four quads are packed before the run.
#define SH_CORES NCU
static volatile int sh_next;
static volatile int sh_arr[SH_CORES];
static volatile int sh_val[SH_CORES];
static volatile int sh_pk[SH_CORES];
static volatile int sh_state[1024];
typedef struct { const float *A, *Bpk; float *Apk; float *C; int M, N, K, nkc, npg, nunits, nquads; } shared_job_t;
static inline void sh_barrier(int c, int m) {
  __sync_fetch_and_add(&sh_arr[c], 1);
  while (__sync_fetch_and_add(&sh_arr[c], 0) < 4 * m) { }
}
static inline void pack_block(const shared_job_t *j, int rb) {
  const int K = j->K, nkc = j->nkc;
  for (int kc = 0; kc < nkc; kc++)
    for (int i = 0; i < 8; i++)
      memcpy(j->Apk + (((size_t)rb * nkc + kc) * 8 + i) * KC, j->A + (size_t)(rb * 8 + i) * K + kc * KC, KC * 4);
}
static void shared_task(void *argp, int id) {
  shared_job_t *j = (shared_job_t *)argp;
  const int c = id % SH_CORES, jj = id / SH_CORES, N = j->N, nkc = j->nkc;
  int m = 0;
  for (;;) {
    if (jj == 0) {
      int u = __sync_fetch_and_add(&sh_next, 1);
      if (u >= j->nunits) { sh_val[c] = -1; sh_pk[c] = -1; }
      else {
        int t = (u / j->npg) + 4;
        sh_val[c] = u;
        sh_pk[c] = (t < j->nquads && __sync_bool_compare_and_swap(&sh_state[t], 0, 1)) ? t : -1;
      }
    }
    sh_barrier(c, ++m);
    int u = sh_val[c];
    if (u < 0) break;
    int t = sh_pk[c];
    if (t >= 0) pack_block(j, t * 4 + jj);
    int quad = u / j->npg, pg = u % j->npg;
    int rb = quad * 4 + jj, p0 = pg * PG;
    while (sh_state[quad] != 2) { __sync_synchronize(); }
    for (int q = 0; q < PG; q++)
      kernel_full(j->Apk + (size_t)rb * nkc * 8 * KC, j->Bpk + ((size_t)(p0 + q) * nkc) * KC * 32,
                  j->C + (size_t)(rb * 8) * N + (p0 + q) * 32, N, nkc);
    sh_barrier(c, ++m);
    if (jj == 0 && t >= 0) { __sync_synchronize(); sh_state[t] = 2; }
  }
}
void sgemm_knc_run_shared(const float *A, const float *Bpk, float *C, float *Apk, int M, int N, int K, int nthreads) {
  pool_init(nthreads);
  const int nquads = M / 32;
  job_t pj = {0};
  pj.A = A; pj.Apk = Apk; pj.M = M; pj.N = N; pj.K = K; pj.nkc = K / KC; pj.nA = nquads < 4 ? M / 8 : 16;
  pj.next = 0; pool_run(pack_task, &pj);                 // quads 0..3 (16 blocks) before the run
  for (int c = 0; c < SH_CORES; c++) { sh_arr[c] = 0; sh_val[c] = 0; sh_pk[c] = -1; }
  for (int q = 0; q < nquads && q < 1024; q++) sh_state[q] = (q < 4) ? 2 : 0;
  sh_next = 0;
  shared_job_t j = {A, Bpk, Apk, C, M, N, K, K / KC, N / (32 * PG), (M / 32) * (N / (32 * PG)), nquads};
  pool_run(shared_task, &j);
}

// ---- any shape ----
// The operands are packed straight from the caller's (unpadded) arrays, in parallel, with the padding written as
// zeros; the compute runs on the padded shape (zero padding leaves the valid part of C unchanged), into C itself when
// the padded C has the same layout (M % RC == 0, N % (32 * PG) == 0), else into a scratch C that is copied back in parallel.
// Scratch buffers are kept between calls. (Measured: 1000^3 from 71 ms with serial padding copies to 42 ms.)
typedef struct { const float *A, *B; float *Apk, *Bpk; const float *Cs; float *C;
                 int M, N, K, Mp, Np, Kp; volatile int next; int nA, nB; } any_job_t;
static void any_pack_task(void *argp, int id) {
  any_job_t *j = (any_job_t *)argp;
  (void)id;
  const int nkc = j->Kp / KC;
  for (;;) {
    int item = __sync_fetch_and_add(&j->next, 1);
    if (item >= j->nA + j->nB) break;
    if (item < j->nA) {                       // 8-row block rb of A: Apk[rb][kc][i][KC]
      int rb = item;
      for (int kc = 0; kc < nkc; kc++) {
        int k0 = kc * KC, nk = j->K - k0; if (nk > KC) nk = KC;
        for (int i = 0; i < 8; i++) {
          float *dst = j->Apk + (((size_t)rb * nkc + kc) * 8 + i) * KC;
          int r = rb * 8 + i;
          if (r >= j->M || nk <= 0) { memset(dst, 0, KC * 4); continue; }
          memcpy(dst, j->A + (size_t)r * j->K + k0, (size_t)nk * 4);
          if (nk < KC) memset(dst + nk, 0, (size_t)(KC - nk) * 4);
        }
      }
    } else {                                  // 32-column panel p of B: Bpk[p][kc][kk][32]
      int p = item - j->nA, c0 = p * 32, nc = j->N - c0; if (nc > 32) nc = 32;
      for (int k = 0; k < j->Kp; k++) {
        float *dst = j->Bpk + ((size_t)p * j->Kp + k) * 32;
        if (k >= j->K || nc <= 0) { memset(dst, 0, 128); continue; }
        memcpy(dst, j->B + (size_t)k * j->N + c0, (size_t)nc * 4);
        if (nc < 32) memset(dst + nc, 0, (size_t)(32 - nc) * 4);
      }
    }
  }
}
static void any_copy_task(void *argp, int id) {   // C[M][N] = Cs[0..M)[0..N) (Cs is Mp x Np)
  any_job_t *j = (any_job_t *)argp;
  (void)id;
  for (;;) {
    int r0 = __sync_fetch_and_add(&j->next, 8);
    if (r0 >= j->M) break;
    for (int r = r0; r < r0 + 8 && r < j->M; r++)
      memcpy(j->C + (size_t)r * j->N, j->Cs + (size_t)r * j->Np, (size_t)j->N * 4);
  }
}
static float *any_scratch(float **buf, size_t *cap, size_t n) {
  if (n > *cap) { free(*buf); *buf = aligned_alloc(BIGALIGN, (n * 4 + BIGALIGN - 1) / BIGALIGN * BIGALIGN); *cap = n; }
  return *buf;
}
static float *Cs_buf = NULL; static size_t Cs_cap = 0;
void sgemm_knc_any(const float *A, const float *B, float *C, int M, int N, int K, int nthreads) {
  pool_init(nthreads);
  int Mp = (int)pad_to(M, RC), Np = (int)pad_to(N, 32 * PG), Kp = (int)pad_to(K, KC);
  any_job_t j = {A, B, any_scratch(&Apk_buf, &Apk_cap, (size_t)Mp * Kp), any_scratch(&Bpk_buf, &Bpk_cap, (size_t)Np * Kp),
                 NULL, C, M, N, K, Mp, Np, Kp, 0, Mp / 8, Np / 32};
  pool_run(any_pack_task, &j);                          // A and B in one pass
  const int direct = (Mp == M && Np == N);
  float *Cd = direct ? C : any_scratch(&Cs_buf, &Cs_cap, (size_t)Mp * Np);
  job_t cj = {0};
  cj.Bpk = j.Bpk; cj.C = Cd; cj.Apk = j.Apk; cj.M = Mp; cj.N = Np; cj.K = Kp; cj.nkc = Kp / KC;
  cj.groups_n = Np / (32 * PG); cj.items = (Mp / RC) * cj.groups_n * KS;
  if (KS > 1) cj.Cpart = any_scratch(&Cpart_buf, &Cpart_cap, (size_t)(KS - 1) * Mp * Np);
  cj.next = 0; pool_run(compute_task, &cj);
  if (KS > 1) { cj.next = 0; pool_run(reduce_task, &cj); }
  if (!direct) { j.Cs = Cd; j.next = 0; pool_run(any_copy_task, &j); }
}

static double maxerr_sampled(const float *A, const float *B, const float *C, int M, int N, int K) {
  double maxerr = 0;
  for (int t = 0; t < 64; t++) {
    int i = (t * 37) % M, jx = (t * 91) % N; double r = 0;
    for (int k = 0; k < K; k++) r += (double)A[(size_t)i * K + k] * B[(size_t)k * N + jx];
    double e = r - C[(size_t)i * N + jx]; if (e != e) return 1e30; if (e < 0) e = -e; if (e > maxerr) maxerr = e;
  }
  return maxerr;
}

int main(int argc, char **argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048, N = argc > 2 ? atoi(argv[2]) : 2048, K = argc > 3 ? atoi(argv[3]) : 2048;
  int nth = argc > 4 ? atoi(argv[4]) : 4 * NCU, reps = argc > 5 ? atoi(argv[5]) : 3;
#define BIGALLOC(n) big_alloc((size_t)(n))
  float *A = BIGALLOC((size_t)M * K * 4), *B = BIGALLOC((size_t)K * N * 4), *C = BIGALLOC((size_t)M * N * 4);
  srand(1);
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (float)rand() / RAND_MAX - 0.5f;
  for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (float)rand() / RAND_MAX - 0.5f;
  float *Bpk = BIGALLOC((size_t)K * N * 4), *Apk = BIGALLOC((size_t)M * K * 4);
  // ANY=1 sends aligned shapes through sgemm_knc_any too (operands packed per call).
  const int fast = !getenv("ANY") && (M % RC == 0 && N % (32 * PG) == 0 && K % KC == 0);
  if (fast) sgemm_knc_pack_B(B, Bpk, K, N, nth);
  double best = 1e30;
  for (int r = 0; r <= reps; r++) {                    // r == 0: warm-up, not timed
    memset(C, 0xff, (size_t)M * N * 4);                // NaN: the error check below sees only this call's output
    double t0 = now_s();
    if (!fast) sgemm_knc_any(A, B, C, M, N, K, nth);
    else if (getenv("PHASED")) sgemm_knc_run(A, Bpk, C, Apk, M, N, K, nth);
    else if (getenv("SHARED") && M % 32 == 0 && nth == 4 * SH_CORES) sgemm_knc_run_shared(A, Bpk, C, Apk, M, N, K, nth);   // opt-in: measured slower
    else if (getenv("STATIC") && M % (RC * 4) == 0 && nth == 4 * NCORES) sgemm_knc_run_static(A, Bpk, C, Apk, M, N, K, nth);   // opt-in: measured slower
    else sgemm_knc_run_overlap(A, Bpk, C, Apk, M, N, K, nth);
    double dt = now_s() - t0; if (r > 0 && dt < best) best = dt;
  }
  double maxerr = maxerr_sampled(A, B, C, M, N, K);   // output of the last timed call (NaN counts as 1e30)
  double gflops = 2.0 * M * N * K / best * 1e-9;
  printf("M=%d N=%d K=%d threads=%d maxerr(sampled)=%g\n", M, N, K, nth, maxerr);
  printf("best (B prepacked, A packed per call) %.3f ms  %.1f GFLOPS  (%.1f%% of 2006 GFLOPS peak)\n", best * 1e3, gflops, 100.0 * gflops / 2006.4);
  return maxerr < 1e-2 ? 0 : 1;
}
