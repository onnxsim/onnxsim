/* Host (x86) semantic check of roialign_kernel.h against ONNX Runtime's real outputs. */
/* Build: clang-19 -O2 -o hostcheck roialign_host_check.c -lm; run: ./hostcheck DATA_DIR */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "roialign_kernel.h"
static void* slurp(const char* p, long* n) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = aligned_alloc(128, (*n + 127) / 128 * 128); fread(b, 1, *n, f); fclose(f); return b;
}
int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  char path[512]; snprintf(path, sizeof path, "%s/calls.txt", dir);
  FILE* cf = fopen(path, "r"); int H, W, C, R, OH, OW, sr, i = 0, bad = 0; float sc;
  while (fscanf(cf, "%d %d %d %d %d %d %d %f", &H, &W, &C, &R, &OH, &OW, &sr, &sc) == 8) {
    long n; char p[512];
    snprintf(p, sizeof p, "%s/call%d_feat.bin", dir, i); float* feat = slurp(p, &n);
    snprintf(p, sizeof p, "%s/call%d_rois.bin", dir, i); float* rois = slurp(p, &n);
    snprintf(p, sizeof p, "%s/call%d_ref.bin", dir, i); float* ref = slurp(p, &n);
    long m = (long)R * OH * OW * C; float* o = aligned_alloc(128, m * 4);
    roialign_hwc(feat, H, W, C, rois, R, OH, OW, sr, sc, o);
    double mx = 0, ma = 0; for (long k = 0; k < m; k++) { double e = fabs(o[k] - ref[k]); if (e > mx) mx = e; if (fabs(ref[k]) > ma) ma = fabs(ref[k]); }
    printf("call%d H=%d W=%d R=%d OH=%d max_abs_err=%.3g (max|ref|=%.3g)\n", i, H, W, R, OH, mx, ma);
    if (mx > 1e-3 * (ma > 1 ? ma : 1)) bad = 1;
    i++;
  }
  puts(bad ? "FAIL" : "PASS"); return bad;
}
