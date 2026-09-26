/* hmx_gemm_layers_client <uri> <flags> <M> <C> <L> [iters]: L chained C -> C int8 layers on the DSP (activations
 * in crouton form in VTCM, hmx_layer_u8cm) vs an exact host reference on every CHECK_STRIDE-th row (rows are
 * independent through 1x1 layers; default 16). Per-layer power-of-two scales are picked on the host from the
 * checked rows so outputs stay mid-range. flags: see layers_u8 in hmx_gemm_rpc.idl. Env HMX_FLAGS as usual. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmx_gemm_rpc.h"
#include "remote.h"
#include "hmx_gemm_u8.h"
int main(int argc, char** argv) {
  if (argc < 6) return 2;
  int flags = atoi(argv[2]), M = atoi(argv[3]), C = atoi(argv[4]), L = atoi(argv[5]);
  int iters = argc > 6 ? atoi(argv[6]) : 5, stride = getenv("CHECK_STRIDE") ? atoi(getenv("CHECK_STRIDE")) : 16;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  int rc = hmx_gemm_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  int prc = 0;
  hmx_gemm_rpc_perf_vote(h, getenv("HMX_FLAGS") ? atoi(getenv("HMX_FLAGS")) : 3, &prc);
  size_t cc = (size_t)C * C;
  uint8_t* A = malloc((size_t)M * C);
  int8_t* W = malloc(cc * L);
  int8_t* Wp = malloc(cc * L);
  uint16_t* S = malloc((size_t)L * C * 2);
  uint8_t* O = calloc((size_t)M * C, 1);
  int nr = (M + stride - 1) / stride;
  uint8_t* R = malloc((size_t)nr * C);
  uint8_t* R2 = malloc((size_t)nr * C);
  long* acc = malloc((size_t)nr * C * sizeof(long));
  srand(7);
  for (size_t i = 0; i < (size_t)M * C; i++) A[i] = rand() % 256;
  for (int i = 0; i < nr; i++) memcpy(R + (size_t)i * C, A + (size_t)i * stride * C, C);
  for (int l = 0; l < L; l++) {
    int8_t* w = W + cc * l;
    for (size_t i = 0; i < cc; i++) {
      int x = rand() % 256 - 128 + 30;
      w[i] = (int8_t)(x > 127 ? 127 : x);
    }
    hmx_pack_w_u8cm(w, C, C, Wp + cc * l);
    long mx = 1;
    for (int i = 0; i < nr; i++)
      for (int j = 0; j < C; j++) {
        long s = 0;
        for (int k = 0; k < C; k++) s += (long)R[(size_t)i * C + k] * w[(size_t)k * C + j];
        acc[(size_t)i * C + j] = s;
        if (s > mx) mx = s;
      }
    int e = (int)floor(log2((double)mx / 512 / 255)) + 1;
    __fp16 sc = (__fp16)ldexp(1.0, -e);
    for (int j = 0; j < C; j++) memcpy(&S[(size_t)l * C + j], &sc, 2);
    for (size_t i = 0; i < (size_t)nr * C; i++) {
      double y = acc[i] < 0 ? 0 : floor((double)acc[i] * ldexp(1.0, -e) / 512);
      R2[i] = y > 255 ? 255 : (uint8_t)y;
    }
    memcpy(R, R2, (size_t)nr * C);
  }
  uint64 t[4];
  int codes[8];
  rc = hmx_gemm_rpc_layers_u8(h, flags, M, C, L, iters, A, M * C, Wp, (int)(cc * L), S, L * C, O, M * C, t, 4, codes, 8);
  printf("vote %d rc %d codes ctx %d hvx %d hmx %d vtcm %d thr %d/%d big %d\n", prc, rc, codes[0], codes[1], codes[2],
         codes[4], codes[5], codes[6], codes[7]);
  int bad = 0, zero = 0;
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < C; j++) {
      uint8_t got = O[(size_t)i * stride * C + j], want = R[(size_t)i * C + j];
      if (!want) zero++;
      if (got != want) {
        if (bad < 4) printf("bad row %d col %d got %d want %d\n", i * stride, j, got, want);
        bad++;
      }
    }
  double lmacs = (double)M * C * C, pmacs = (double)((M + 63) / 64 * 64) * C * C;
  printf("layers flags %d M %d C %d L %d: %s %d mismatches of %d checked (%d zero)\n", flags, M, C, L,
         (flags & 2) ? "UNCHECKED(nocopy)" : bad ? "FAIL" : "PASS", (flags & 2) ? 0 : bad, nr * C, zero);
  printf("  first chain %.1f us/layer (%.2f TMAC/s); steady %.1f us/layer (%.2f TMAC/s, %.2f padded); weight wait %.1f us/layer; pack A %llu us\n",
         (double)t[0] / L, lmacs * L / t[0] / 1e6, iters ? (double)t[1] / (iters * L) : 0,
         iters ? lmacs * L * iters / t[1] / 1e6 : 0, iters ? pmacs * L * iters / t[1] / 1e6 : 0,
         (double)t[2] / (L * (1 + iters)), (unsigned long long)t[3]);
  hmx_gemm_rpc_close(h);
  return (!(flags & 2) && bad) ? 3 : 0;
}
