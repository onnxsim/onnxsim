/* Native ARM64 FastRPC client (no TVM) for fpn_rpc: prices the NCHW->NHWC conversion the fast
 * RoiAlign kernel needs, on real data, end to end.
 *
 * Inputs (gen_fpn_test_data.py + ../roialign_fast's calls.txt / callN_rois.bin / callN_ref.bin):
 * per FPN level k (0..3 = P5..P2) ORT's real NCHW fp32 map (`lvlK_chw.bin`), and the real FPN
 * output conv's zero-point-padded channels-last uint8 input, packed weight and folded bias.
 *
 *  1. conv_out: the real FPN output convs (fused requant+dequant) writing NHWC vs NCHW, timed;
 *     the NCHW output must equal ORT's real map bitwise.
 *  2. transpose: ORT's NCHW maps -> NHWC on the DSP, several variants, timed; each must equal
 *     the conv's NHWC output bitwise.
 *  3. roialign on the conv-produced NHWC maps, all 8 real calls, vs ORT's real RoiAlign outputs.
 * Large buffers come from rpcmem (mapped into the DSP, not copied). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "remote.h"
#include "rpcmem.h"
#include "fpn_rpc.h"

#define C 256
static void* load(const char* p, long n, long slack) {
  char* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n + slack);
  FILE* f = fopen(p, "rb");
  if (!f || !b) { perror(p); exit(1); }
  if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p); exit(1); }
  fclose(f);
  memset(b + n, 0, slack);
  return b;
}
static void* zalloc(long n) {
  void* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n);
  if (!b) { fprintf(stderr, "rpcmem_alloc %ld failed\n", n); exit(1); }
  memset(b, 0, n);
  return b;
}
static int cmp_u64(const void* a, const void* b) {
  unsigned long long x = *(const unsigned long long*)a, y = *(const unsigned long long*)b;
  return x < y ? -1 : x > y;
}
static unsigned long long median(unsigned long long* d, int n) { qsort(d, n, sizeof d[0], cmp_u64); return d[n / 2]; }
static int bitwise_eq(const float* a, const float* b, long n) { return memcmp(a, b, n * 4) == 0; }

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 5, conv_reps = argc > 3 ? atoi(argv[3]) : 3, turbo = argc > 4 ? atoi(argv[4]) : 0;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (fpn_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  if (turbo) { int vrc = -1; fpn_rpc_perf_vote(h, 1, &vrc); printf("perf_vote(core+bus TURBO) rc=%d\n", vrc); }

  int H[4], W[4], bad = 0;
  FILE* lf = fopen("levels.txt", "r");
  for (int k = 0; k < 4; k++) if (fscanf(lf, "%d %d", &H[k], &W[k]) != 2) { puts("levels.txt"); return 1; }
  float *chw[4], *hwc_t[4], *hwc_c[4], *chw_c[4];
  unsigned char *apad[4], *wp[4];
  int* bias[4];
  char p[64];
  for (int k = 0; k < 4; k++) {
    long hw = (long)H[k] * W[k];
    snprintf(p, sizeof p, "lvl%d_chw.bin", k); chw[k] = load(p, hw * C * 4, 128);
    snprintf(p, sizeof p, "lvl%d_apad.bin", k); apad[k] = load(p, (long)(H[k] + 2) * (W[k] + 2) * C, 0);
    snprintf(p, sizeof p, "lvl%d_wp.bin", k); wp[k] = load(p, 9L * C * C, 0);
    snprintf(p, sizeof p, "lvl%d_bias.bin", k); bias[k] = load(p, C * 4, 0);
    hwc_t[k] = zalloc(hw * C * 4); hwc_c[k] = zalloc(hw * C * 4); chw_c[k] = zalloc(hw * C * 4);
  }

  /* 1. FPN output convs, fused epilogue, NHWC vs NCHW store */
  unsigned long long ctot[2] = {0};
  for (int k = 0; k < 4; k++) {
    long hw = (long)H[k] * W[k];
    for (int lay = 0; lay < 2; lay++) {
      float* o = lay ? chw_c[k] : hwc_c[k];
      unsigned long long d[32];
      for (int r = 0; r < conv_reps; r++) {
        memset(o, 0, hw * C * 4);
        int rc = fpn_rpc_conv_out(h, k, lay, apad[k], (H[k] + 2) * (W[k] + 2) * C, wp[k], 9 * C * C, bias[k], C, o, hw * C, &d[r]);
        if (rc) { printf("conv rc=%d\n", rc); return 1; }
      }
      unsigned long long m = median(d, conv_reps);
      ctot[lay] += m;
      printf("conv_out level %d %s: %llu us\n", k, lay ? "NCHW" : "NHWC", m);
    }
    int e1 = bitwise_eq(chw_c[k], chw[k], hw * C);
    printf("  level %d: NCHW conv output == ORT's real map: %s\n", k, e1 ? "bit-exact" : "MISMATCH");
    if (!e1) bad = 1;
  }

  /* 2. transpose ORT's NCHW maps, each variant checked against the conv's NHWC output */
  const int tv[][2] = {{0, 1}, {1, 1}, {1, 4}, {2, 4}, {3, 1}, {3, 4}};
  const char* tn[] = {"scalar 1thr", "hvx 1thr", "hvx 4thr", "hvx+l2fetch 4thr", "hvx c-outer 1thr", "hvx c-outer 4thr"};
  unsigned long long ttot[6] = {0};
  for (int v = 0; v < 6; v++) {
    for (int k = 0; k < 4; k++) {
      long hw = (long)H[k] * W[k];
      unsigned long long d[32];
      for (int r = 0; r < reps; r++) {
        int rc = fpn_rpc_transpose(h, chw[k], hw * C + 32, C, hw, tv[v][1], tv[v][0], hwc_t[k], hw * C, &d[r]);
        if (rc) { printf("transpose rc=%d\n", rc); return 1; }
      }
      unsigned long long m = median(d, reps);
      ttot[v] += m;
      int eq = bitwise_eq(hwc_t[k], hwc_c[k], hw * C);
      if (!eq) bad = 1;
      memset(hwc_t[k], 0, hw * C * 4);
      printf("transpose %-18s level %d (%dx%d): %llu us, == NHWC conv output: %s\n", tn[v], k, H[k], W[k], m,
             eq ? "bit-exact" : "MISMATCH");
    }
  }

  /* 3. RoiAlign on the conv-produced NHWC maps, all 8 real calls */
  FILE* cf = fopen("calls.txt", "r");
  int h_, w_, c_, R, OH, OW, sr, i = 0;
  float sc;
  unsigned long long rtot = 0;
  double worst = 0;
  while (fscanf(cf, "%d %d %d %d %d %d %d %f", &h_, &w_, &c_, &R, &OH, &OW, &sr, &sc) == 8) {
    int k = i % 4;
    if (h_ != H[k] || w_ != W[k] || c_ != C) { printf("call%d shape mismatch\n", i); return 1; }
    long on = (long)R * OH * OW * C;
    snprintf(p, sizeof p, "call%d_rois.bin", i); float* rois = load(p, (long)R * 16, 0);
    snprintf(p, sizeof p, "call%d_ref.bin", i); float* ref = load(p, on * 4, 0);
    float* out = zalloc(on * 4);
    unsigned long long d[32];
    for (int r = 0; r < reps; r++) {
      int rc = fpn_rpc_roialign(h, hwc_c[k], (long)H[k] * W[k] * C, rois, R * 4, H[k], W[k], C, OH, OW, sr, sc, 4, out, on, &d[r]);
      if (rc) { printf("roialign rc=%d\n", rc); return 1; }
    }
    double mx = 0;
    for (long q = 0; q < on; q++) { double e = fabs(out[q] - ref[q]); if (e > mx) mx = e; }
    if (mx > worst) worst = mx;
    if (mx > 1e-3) bad = 1;
    unsigned long long m = median(d, reps);
    rtot += m;
    printf("roialign call%d (level %d, R=%d, %dx%d) on conv-produced NHWC: %llu us, max_abs_err vs ORT %.3g\n", i, k, R, OH, OW, m, mx);
    rpcmem_free(rois); rpcmem_free(ref); rpcmem_free(out);
    i++;
  }

  printf("\nSUMMARY (DSP-side time, medians, all 4 FPN levels / all 8 real RoiAlign calls)\n");
  printf("roialign (prefetch, 4 threads), 8 calls: %.1f ms, worst max_abs_err vs ORT %.3g\n", rtot / 1e3, worst);
  for (int v = 0; v < 6; v++) printf("transpose NCHW->NHWC %-18s: %.1f ms\n", tn[v], ttot[v] / 1e3);
  printf("FPN output convs, fused epilogue, NHWC store: %.1f ms | NCHW store: %.1f ms\n", ctot[0] / 1e3, ctot[1] / 1e3);
  fpn_rpc_close(h);
  puts(bad ? "FAIL" : "PASS");
  return bad;
}
