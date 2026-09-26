/* Phone client for tg_hmx_rpc: random fp16 A (MxK), B (KxN), runs the tinygrad-generated HMX kernel and checks it bit
 * for bit against the rounding model of the accumulator-resident codegen: exact accumulation over all of K, rounded once.
 * With I8=1 (the skel built with gen_kernel.py --i8): A uint8, B int8, C int32, checked exactly.
 * With RQ=1 (--rq): the int8 layer with ORT's requantization fused, uint8 out, checked exactly against ORT's formula
 *   y = clamp(rne(fp32(fp32(acc + bias) * m)) + zy, lo, 255) (zy / lo from env ZY, LO: the values the kernel was built with)
 *   [I8=1|RQ=1] tg_hmx_client <uri> M K N [iters] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "remote.h"
#include "tg_hmx_rpc.h"
static float h2f(unsigned short h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
static unsigned short f2h(double f) { __fp16 x = (__fp16)f; unsigned short h; memcpy(&h, &x, 2); return h; }
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s uri M K N [iters]\n", argv[0]); return 2; }
  int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]), iters = argc > 5 ? atoi(argv[5]) : 5;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
  remote_handle64 h; int rc = tg_hmx_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  if (getenv("CASE")) {  /* a case dir from gen_kernel.py --conv: a.bin, b.bin (weights | bias | scale), ref.bin */
    char path[512]; long na, nb, nr; unsigned char *a, *b, *ref;
    FILE* f;
#define RD(nm, buf, n) snprintf(path, sizeof path, "%s/" nm, getenv("CASE")); f = fopen(path, "rb"); if (!f) { printf("no %s\n", path); return 1; } \
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET); buf = malloc(n); fread(buf, 1, n, f); fclose(f);
    RD("a.bin", a, na) RD("b.bin", b, nb) RD("ref.bin", ref, nr)
    unsigned char* c = malloc(nr);
    unsigned long long t[4]; int codes[8];
    rc = tg_hmx_rpc_run(h, 1, a, na, b, nb, c, nr, t, 4, codes, 8);
    int bad = 0; for (long i = 0; i < nr; i++) bad += c[i] != ref[i];
    printf("case %s: rc %d codes power %d ctx %d hvx %d hmx %d vtcm %d thread %d; %d/%ld mismatches vs ORT's formula\n", getenv("CASE"), rc,
           codes[0], codes[1], codes[2], codes[3], codes[4], codes[5], bad, nr);
    rc = tg_hmx_rpc_run(h, iters, a, na, b, nb, c, nr, t, 4, codes, 8);
    printf("%s: %.1f us/call (%d iters) %s\n", getenv("CASE"), (double)t[0] / iters, iters, bad ? "FAIL" : "PASS");
    tg_hmx_rpc_close(h);
    return bad != 0;
  }
  if (getenv("RQ") && atoi(getenv("RQ"))) {
    int zy = getenv("ZY") ? atoi(getenv("ZY")) : 131, lo = getenv("LO") ? atoi(getenv("LO")) : 0;
    int SB = K * N + 8 * N;
    unsigned char* a = malloc(M * K); signed char* b = malloc(SB); unsigned char *c = malloc(M * N), *r = malloc(M * N);
    int* bias = (int*)(b + K * N); float* m = (float*)(b + K * N + 4 * N); long* acc = malloc(sizeof(long) * M * N);
    srand(1);
    for (int i = 0; i < M * K; i++) a[i] = rand() & 255;
    for (int i = 0; i < K * N; i++) b[i] = (signed char)(rand() & 255);
    for (int n = 0; n < N; n++) bias[n] = rand() % 40001 - 20000;
    for (int i = 0; i < M; i++) for (int n = 0; n < N; n++) { long s = bias[n]; for (int k = 0; k < K; k++) s += (long)a[i * K + k] * b[k * N + n]; acc[i * N + n] = s; }
    for (int n = 0; n < N; n++) {  /* the column's spread -> most outputs inside [0, 255]; every 8th a power of two (.5 ties) */
      long mx = 1; for (int i = 0; i < M; i++) mx = labs(acc[i * N + n]) > mx ? labs(acc[i * N + n]) : mx;
      m[n] = n % 8 == 0 ? 1.0f / 256 : (float)((0.5 + 1.5 * (rand() % 1000) / 1000.0) * 50.0 / (double)(mx + 1));
    }
    int ties = 0, sat = 0;
    for (int i = 0; i < M * N; i++) {
      volatile float f = (float)acc[i]; volatile float v = f * m[i % N];
      float l = (float)(lo - zy), hh = (float)(255 - zy); float cl = v < l ? l : v > hh ? hh : v;
      r[i] = (unsigned char)((int)nearbyintf(cl) + zy);
      ties += i % N % 8 == 0 && (acc[i] & 255) == 128; sat += r[i] == 255 || r[i] == lo;
    }
    unsigned long long t[4]; int codes[8];
    rc = tg_hmx_rpc_run(h, 1, a, M * K, (const unsigned char*)b, SB, c, M * N, t, 4, codes, 8);
    int bad = 0; for (int i = 0; i < M * N; i++) bad += c[i] != r[i];
    printf("int8+requant: rc %d codes power %d ctx %d hvx %d hmx %d vtcm %d thread %d; %d/%d mismatches vs ORT's formula "
           "(%d exact .5 ties, %d clamped)\n", rc, codes[0], codes[1], codes[2], codes[3], codes[4], codes[5], bad, M * N, ties, sat);
    rc = tg_hmx_rpc_run(h, iters, a, M * K, (const unsigned char*)b, SB, c, M * N, t, 4, codes, 8);
    double us = (double)t[0] / iters;
    printf("%dx%dx%d int8 + fused requant: %.1f us/call, %.3f TMAC/s (tinygrad-generated HMX :cm + HVX, %d iters) %s\n", M, K, N, us,
           (double)M * K * N / us / 1e6, iters, bad ? "FAIL" : "PASS");
    tg_hmx_rpc_close(h);
    return bad != 0;
  }
  if (getenv("I8") && atoi(getenv("I8"))) {
    unsigned char* a = malloc(M * K); signed char* b = malloc(K * N); int *c = malloc(M * N * 4), *r = malloc(M * N * 4);
    srand(1);
    for (int i = 0; i < M * K; i++) a[i] = rand() & 255;
    for (int i = 0; i < K * N; i++) b[i] = (signed char)(rand() & 255);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { long s = 0; for (int k = 0; k < K; k++) s += (long)a[m * K + k] * b[k * N + n]; r[m * N + n] = (int)s; }
    unsigned long long t[4]; int codes[8];
    rc = tg_hmx_rpc_run(h, 1, a, M * K, (const unsigned char*)b, K * N, (unsigned char*)c, M * N * 4, t, 4, codes, 8);
    int bad = 0; for (int i = 0; i < M * N; i++) bad += c[i] != r[i];
    printf("int8: rc %d codes power %d ctx %d hvx %d hmx %d vtcm %d thread %d; %d/%d mismatches vs exact int32\n", rc, codes[0], codes[1],
           codes[2], codes[3], codes[4], codes[5], bad, M * N);
    rc = tg_hmx_rpc_run(h, iters, a, M * K, (const unsigned char*)b, K * N, (unsigned char*)c, M * N * 4, t, 4, codes, 8);
    double us = (double)t[0] / iters;
    printf("%dx%dx%d int8: %.1f us/call, %.3f TMAC/s (tinygrad-generated HMX :cm, %d iters) %s\n", M, K, N, us, (double)M * K * N / us / 1e6,
           iters, bad ? "FAIL" : "PASS");
    tg_hmx_rpc_close(h);
    return bad != 0;
  }
  unsigned short *a = malloc(M * K * 2), *b = malloc(K * N * 2), *c = malloc(M * N * 2), *r = malloc(M * N * 2);
  srand(1);
  for (int i = 0; i < M * K; i++) a[i] = f2h((rand() % 2001 - 1000) / 2000.0);
  for (int i = 0; i < K * N; i++) b[i] = f2h((rand() % 2001 - 1000) / 2000.0);
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {  /* exact accumulation, one rounding (accumulator kept in HMX) */
    double s = 0; for (int k = 0; k < K; k++) s += (double)h2f(a[m * K + k]) * h2f(b[k * N + n]);
    r[m * N + n] = f2h(s);
  }
  unsigned long long t[4]; int codes[8];
  rc = tg_hmx_rpc_run(h, 1, (const unsigned char*)a, M * K * 2, (const unsigned char*)b, K * N * 2, (unsigned char*)c, M * N * 2, t, 4, codes, 8);  /* warm-up + correctness */
  int bad = 0; for (int i = 0; i < M * N; i++) bad += c[i] != r[i];
  printf("rc %d codes power %d ctx %d hvx %d hmx %d vtcm %d thread %d; %d/%d bit mismatches vs model\n", rc, codes[0], codes[1],
         codes[2], codes[3], codes[4], codes[5], bad, M * N);
  rc = tg_hmx_rpc_run(h, iters, (const unsigned char*)a, M * K * 2, (const unsigned char*)b, K * N * 2, (unsigned char*)c, M * N * 2, t, 4, codes, 8);
  double us = (double)t[0] / iters;
  printf("%dx%dx%d: %.1f us/call, %.3f TMAC/s (tinygrad-generated HMX, %d iters) %s\n", M, K, N, us, (double)M * K * N / us / 1e6,
         iters, bad ? "FAIL" : "PASS");
  tg_hmx_rpc_close(h);
  return bad != 0;
}
