/* mcc_hmx_client <uri> <data dir> <Q> <blocks> [iters] [hvx mask, default 30 = all] [threads, default 4]:
 *   load ref.py's packed blocks into the skel, run them on x0 and compare with the float64 reference of
 *   the last block; prints per-phase times.
 * mcc_hmx_client <uri> <data dir> decode <Q> [iters] [hvx] [threads]:
 *   the whole decoder (load_head, set_kv from kv.bin, decode xyz.bin) vs ref_occ / ref_rgb. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mcc_hmx_rpc.h"
#include "remote.h"

static float h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
static void* slurp(const char* dir, const char* name, size_t* n) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* b = malloc(*n);
  if (fread(b, 1, *n, f) != *n) exit(1);
  fclose(f);
  return b;
}
static const char* PH[] = {"layernorm", "weight copy", "qkv/pos", "S=qK", "softmax/color", "PV", "self term", "proj", "fc1", "fc2/pred", "wait/sync", "gelu"};
static int decode_mode(remote_handle64 h, const char* dir, int argc, char** argv) {
  int q = atoi(argv[4]), iters = argc > 5 ? atoi(argv[5]) : 3, hvx = argc > 6 ? atoi(argv[6]) : 30, nthr = argc > 7 ? atoi(argv[7]) : 4;
  size_t n;
  void* head = slurp(dir, "head.bin", &n);
  int rc = mcc_hmx_rpc_load_head(h, head, (int)n);
  float* kv = slurp(dir, "kv.bin", &n);
  const int kvn = (int)(n / 8);
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  rc |= mcc_hmx_rpc_set_kv(h, kv, kvn, kv + kvn, kvn) << 8;
  clock_gettime(CLOCK_MONOTONIC, &b);
  const double kv_ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
  float* xyz = slurp(dir, "xyz.bin", &n);
  float *ro = slurp(dir, "ref_occ.bin", &n), *rr = slurp(dir, "ref_rgb.bin", &n);
  float *occ = malloc(q * 4), *rgb = malloc(q * 12);
  uint64 t[13];
  int codes[6];
  double wall = 0;
  for (int it = 0; it < iters; it++) {
    clock_gettime(CLOCK_MONOTONIC, &a);
    rc |= mcc_hmx_rpc_decode(h, q, hvx, nthr, xyz, q * 3, occ, q, rgb, q * 3, t, 13, codes, 6) << 16;
    clock_gettime(CLOCK_MONOTONIC, &b);
    if (it) wall += (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
  }
  /* colors matter where a point is kept: max over p > 0.1, mean over all */
  double eo = 0, er = 0, em = 0;
  int flip = 0, occd = 0;
  for (int i = 0; i < q; i++) {
    eo = fabs(occ[i] - ro[i]) > eo ? fabs(occ[i] - ro[i]) : eo;
    const int x = 1 / (1 + exp(-occ[i])) > 0.3, y = 1 / (1 + exp(-ro[i])) > 0.3;
    flip += x != y, occd += y;
    for (int c = 0; c < 3; c++) {
      const double d = fabs(rgb[i * 3 + c] - rr[i * 3 + c]);
      em += d / (3.0 * q);
      if (1 / (1 + exp(-ro[i])) > 0.1) er = d > er ? d : er;
    }
  }
  printf("rc %d ctx %d hvx %d hmx %d vtcm %d thread %d | set_kv %.2f ms (wall)\n", rc, codes[0], codes[1], codes[2], codes[3], codes[4], kv_ms);
  printf("decode Q=%d, %d threads: %s occ logit max abs err %.4g, p>0.3 differ %d of %d (%d occupied), rgb err mean %.2f/255, max (p>0.1) %.2f/255;"
         " DSP %.3f ms, wall %.3f ms per call\n",
         q, nthr, flip <= q / 100 && em < 1.0 / 255 ? "PASS" : "FAIL", eo, flip, q, occd, em * 255, er * 255, t[0] / 1e3,
         iters > 1 ? wall / (iters - 1) : 0);
  for (int i = 0; i < 12; i++) printf("  %-13s %8.3f ms\n", PH[i], t[1 + i] / 1.5e6);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 5) return 2;
  const char* dir = argv[2];
  int q = atoi(argv[3]), nb = atoi(argv[4]), iters = argc > 5 ? atoi(argv[5]) : 3, hvx = argc > 6 ? atoi(argv[6]) : 30, nthr = argc > 7 ? atoi(argv[7]) : 4;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  int rc = mcc_hmx_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  int prc = 0;
  mcc_hmx_rpc_perf_vote(h, 3, &prc);
  const int dec = !strcmp(argv[3], "decode");
  if (dec) nb = 8;
  for (int b = 0; b < nb; b++) {
    char n[32];
    size_t len;
    snprintf(n, sizeof n, "blk%d.bin", b);
    void* blob = slurp(dir, n, &len);
    rc = mcc_hmx_rpc_load_block(h, b, blob, (int)len);
    free(blob);
    if (rc) { printf("load_block %d rc %d\n", b, rc); return 1; }
  }
  if (dec) {
    decode_mode(h, dir, argc, argv);
    mcc_hmx_rpc_close(h);
    return 0;
  }
  size_t xl, rl;
  uint16_t* x = slurp(dir, "x0.bin", &xl);
  char n[32];
  snprintf(n, sizeof n, "ref_out%d.bin", nb - 1);
  float* ref = slurp(dir, n, &rl);
  uint16_t* y = malloc(xl);
  uint64 t[13];
  int codes[6];
  rc = mcc_hmx_rpc_run(h, q, nb, iters, hvx, nthr, x, q * 512, y, q * 512, t, 13, codes, 6);
  printf("vote %d rc %d ctx %d hvx %d hmx %d vtcm %d thread %d helper-hvx-fail %d\n", prc, rc, codes[0], codes[1], codes[2], codes[3], codes[4],
         codes[5]);
  double e = 0, mr = 0, dot = 0, na = 0, nr = 0;
  for (int i = 0; i < q * 512; i++) {
    double g = h2f(y[i]), f = ref[i];
    e = fabs(g - f) > e ? fabs(g - f) : e;
    mr = fabs(f) > mr ? fabs(f) : mr;
    dot += g * f, na += g * g, nr += f * f;
  }
  double cs = dot / sqrt(na * nr);
  printf("hvx %d, %d threads, Q=%d, %d blocks: %s cos %.7f max abs err %.4g (max |ref| %.4g); %.3f ms per run (%.3f ms per block)\n", hvx, nthr, q, nb,
         cs > 0.9999 ? "PASS" : "FAIL", cs, e, mr, t[0] / 1e3, t[0] / 1e3 / nb);
  unsigned long long tot = 0;
  for (int i = 0; i < 12; i++) tot += t[1 + i];
  for (int i = 0; i < 12; i++) printf("  %-12s %8.3f ms  %5.1f%%\n", PH[i], t[1 + i] / 1.5e6 / nb * nb, 100.0 * t[1 + i] / (tot ? tot : 1));
  mcc_hmx_rpc_close(h);
  return cs > 0.9999 ? 0 : 3;
}
