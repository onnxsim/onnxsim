/* hmx_gemm_u8_client <uri> <mode> <M> <K> <N> [iters]: random uint8 x int8 GEMM on the DSP's HMX (int8 cm
 * path, hmx_gemm_u8.h) vs an exact host reference (power-of-two column scales, where the HMX conversion is
 * exact); prints mismatches, TMAC/s and the DSP core / HMX clocks. Env HMX_FLAGS (default 3 = turbo + HMX
 * power vote; +4 = HMX_v2 clock vote at turbo), CHECK_STRIDE (check every n-th row, default 1). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmx_gemm_rpc.h"
#include "remote.h"
#include "hmx_gemm_u8.h"
int main(int argc, char** argv) {
  if (argc < 6) return 2;
  int mode = atoi(argv[2]), M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
  int iters = argc > 6 ? atoi(argv[6]) : 10, stride = getenv("CHECK_STRIDE") ? atoi(getenv("CHECK_STRIDE")) : 1;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  int rc = hmx_gemm_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  int clk[3], prc = 0;
  hmx_gemm_rpc_clocks(h, clk, 3);
  printf("clocks before vote: core %d Hz, hmx %d Hz, dcvs %d\n", clk[0], clk[1], clk[2]);
  hmx_gemm_rpc_perf_vote(h, getenv("HMX_FLAGS") ? atoi(getenv("HMX_FLAGS")) : 3, &prc);
  hmx_gemm_rpc_clocks(h, clk, 3);
  printf("vote rc %d; clocks after vote: core %d Hz, hmx %d Hz, dcvs %d\n", prc, clk[0], clk[1], clk[2]);
  uint8_t* A = malloc((size_t)M * K);
  int8_t* W = malloc((size_t)K * N);
  int8_t* Wp = malloc((size_t)K * N);
  uint16_t* S = malloc((size_t)N * 2);
  uint8_t* C = calloc((size_t)M * N, 1);
  srand(7);
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rand() % 256;
  for (size_t i = 0; i < (size_t)K * N; i++) {
    int x = rand() % 256 - 128 + 40;
    W[i] = (int8_t)(x > 127 ? 127 : x);
  }
  /* scale 2^-(e) with e so that a typical accumulator (~K * 128 * 40) lands mid-range after /512 */
  int e = (int)floor(log2((double)K * 128 * 40 / 512 / 128));
  for (int j = 0; j < N; j++) {
    __fp16 x = (__fp16)ldexp(1.0, -(e + rand() % 2));
    memcpy(&S[j], &x, 2);
  }
  hmx_pack_w_u8cm(W, K, N, Wp);
  uint64 t[4];
  int codes[8];
  rc = hmx_gemm_rpc_gemm_u8(h, mode, M, K, N, iters, A, M * K, Wp, K * N, S, N, C, M * N, t, 4, codes, 8);
  hmx_gemm_rpc_clocks(h, clk, 3);
  printf("rc %d codes ctx %d hvx %d hmx %d gemm %d vtcm %d thr %d big %d; clocks after run: core %d hmx %d\n", rc,
         codes[0], codes[1], codes[2], codes[3], codes[4], codes[5], codes[6], clk[0], clk[1]);
  double macs = (double)M * K * N * iters;
  if (mode >= 2) {
    double padded = (double)((M + 63) / 64 * 64) * K * N * iters;
    printf("mode %d %dx%dx%d x%d: MAC loop %llu us, %llu pcycles -> %.2f TMAC/s (%.0f MAC/pcycle, padded rows)\n", mode, M,
           K, N, iters, (unsigned long long)t[0], (unsigned long long)t[1], padded / t[0] / 1e6, padded / t[1]);
    hmx_gemm_rpc_close(h);
    return 0;
  }
  int bad = 0, checked = 0, zero = 0, sat = 0;
  for (int i = 0; i < M; i += stride)
    for (int j = 0; j < N; j++) {
      long acc = 0;
      for (int k = 0; k < K; k++) acc += (long)A[(size_t)i * K + k] * W[(size_t)k * N + j];
      __fp16 s;
      memcpy(&s, &S[j], 2);
      double ex = acc < 0 ? 0 : floor((double)acc * (double)s / 512);
      if (ex >= 255) ex = 255, sat++;
      if (ex == 0) zero++;
      checked++;
      if (C[(size_t)i * N + j] != (uint8_t)ex) {
        if (bad < 4) printf("bad %d,%d got %d want %g acc %ld\n", i, j, C[(size_t)i * N + j], ex, acc);
        bad++;
      }
    }
  printf("mode %d %dx%dx%d x%d: %s %d mismatches of %d checked (%d zero, %d saturated); %.1f us/iter, %.3f TMAC/s\n",
         mode, M, K, N, iters, bad ? "FAIL" : "PASS", bad, checked, zero, sat, (double)t[0] / iters, macs / t[0] / 1e6);
  printf("  pcycles/iter: pack A %.0f, W copy %.0f, MAC+store %.0f, unpack C %.0f\n", (double)t[1] / iters,
         (double)t[2] / iters, (double)(t[3] & 0xffffffffu) / iters, (double)(t[3] >> 32) / iters);
  hmx_gemm_rpc_close(h);
  return bad ? 3 : 0;
}
