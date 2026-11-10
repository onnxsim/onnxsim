#define _GNU_SOURCE
#define main gemm_main
#include "gemm_knc.c"
#undef main
#include <sched.h>
#include <stdint.h>
static int pin = 0;
// Kernel-only microbenchmark: one 8x32 tile over L1-resident packed panels, repeated; T threads run it independently.
static void bench_task(void *argp, int id) {
  int reps = *(int *)argp;
  if (pin) {
    int cpu = (4 * (id % 57) + 1 + id / 57) % 228;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
  }
  float *Ap = aligned_alloc(64, KC * 8 * 4), *Bp = aligned_alloc(64, KC * 32 * 4), *C = aligned_alloc(64, 8 * 32 * 4 * 2);
  for (int i = 0; i < KC * 8; i++) Ap[i] = 1e-3f;
  for (int i = 0; i < KC * 32; i++) Bp[i] = 1e-3f;
  for (int r = 0; r < reps; r++) kernel_8x32(Ap, Bp, C, 32, KC, r == 0);
  free(Ap); free(Bp); free(C);
}
int main(int argc, char **argv) {
  int T = argc > 1 ? atoi(argv[1]) : 1, reps = argc > 2 ? atoi(argv[2]) : 20000;
  pin = argc > 3 ? atoi(argv[3]) : 1;   // pinned by default (PIN=0 to disable)
  pool_init(T);
  double t0 = now_s();
  pool_run(bench_task, &reps);
  double dt = now_s() - t0;
  double flops = 2.0 * 8 * 32 * KC * (double)reps * T;
  printf("pin=%d threads=%d: %.1f GFLOPS aggregate, %.2f GFLOPS per thread (%.1f%% of peak)\n", pin, T, flops / dt * 1e-9,
         flops / dt * 1e-9 / T, 100.0 * flops / dt * 1e-9 / 2006.4);
  return 0;
}
