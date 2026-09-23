/* Native ARM64 FastRPC client (no TVM): for every real RoiAlign call in calls.txt, runs the DSP
 * kernel at several thread counts, checks the output against ONNX Runtime's, and reports
 * DSP-side kernel time (HAP_perf_get_time_us) and client-side round-trip time. Large buffers come
 * from rpcmem (ION-backed, mapped rather than copied) -- the P2-level feature map is 55.7 MB. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "roialign_rpc.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static float* load(const char* p, long n) {
  float* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n);
  FILE* f = fopen(p, "rb"); if (!f || !b) { perror(p); exit(1); }
  if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p); exit(1); }
  fclose(f); return b;
}
static int cmp_u64(const void* a, const void* b) { unsigned long long x = *(unsigned long long*)a, y = *(unsigned long long*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  int turbo = argc > 3 ? atoi(argv[3]) : 0;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (roialign_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  int vrc = -1;
  if (turbo) { roialign_rpc_perf_vote(h, 1, &vrc); printf("perf_vote(TURBO) rc=%d\n", vrc); }
  FILE* cf = fopen("calls.txt", "r");
  int H, W, C, R, OH, OW, sr, i = 0, bad = 0; float sc;
  int threads[] = {1, 4, 101, 104}; /* +100 = l2fetch-prefetching kernel variant */
  unsigned long long tot[4] = {0};
  while (fscanf(cf, "%d %d %d %d %d %d %d %f", &H, &W, &C, &R, &OH, &OW, &sr, &sc) == 8) {
    char p[64]; long fn = (long)H * W * C * 4, on = (long)R * OH * OW * C * 4;
    snprintf(p, sizeof p, "call%d_feat.bin", i); float* feat = load(p, fn);
    snprintf(p, sizeof p, "call%d_rois.bin", i); float* rois = load(p, (long)R * 16);
    snprintf(p, sizeof p, "call%d_ref.bin", i); float* ref = load(p, on);
    float* out = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, on);
    for (int ti = 0; ti < 4; ti++) {
      unsigned long long d[32]; double rt[32];
      for (int k = 0; k < reps; k++) {
        memset(out, 0, on);
        double t0 = now_us();
        int rc = roialign_rpc_run(h, feat, fn / 4, rois, R * 4, H, W, C, OH, OW, sr, sc, threads[ti], out, on / 4, &d[k]);
        rt[k] = now_us() - t0;
        if (rc) { printf("call%d rc=%d\n", i, rc); return 1; }
      }
      double mx = 0; for (long k = 0; k < on / 4; k++) { double e = fabs(out[k] - ref[k]); if (e > mx) mx = e; }
      if (mx > 1e-3) bad = 1;
      qsort(d, reps, sizeof d[0], cmp_u64);
      double rmin = rt[0]; for (int k = 1; k < reps; k++) if (rt[k] < rmin) rmin = rt[k];
      tot[ti] += d[reps / 2];
      printf("call%d HxW=%dx%d R=%d out=%dx%d %s threads=%d dsp_us(median)=%llu dsp_us(min)=%llu roundtrip_us(min)=%.0f max_abs_err=%.3g\n",
             i, H, W, R, OH, OW, threads[ti] >= 100 ? "prefetch" : "plain", threads[ti] % 100, d[reps / 2], d[0], rmin, mx);
    }
    rpcmem_free(feat); rpcmem_free(rois); rpcmem_free(ref); rpcmem_free(out);
    i++;
  }
  printf("TOTAL dsp_us(median, all 8 calls): plain 1thr=%llu 4thr=%llu | prefetch 1thr=%llu 4thr=%llu\n", tot[0], tot[1], tot[2], tot[3]);
  roialign_rpc_close(h);
  puts(bad ? "FAIL" : "PASS");
  return bad;
}
