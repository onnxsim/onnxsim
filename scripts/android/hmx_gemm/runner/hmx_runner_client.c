/* hmx_runner_client <uri> <program dir> <input.bin> <ref.bin> [iters]: build a qdq_graph.py program (runner/rn_load.h),
 * load it into the DSP (weights stay in the DSP heap) and run it in one FastRPC call per inference, in both
 * requantization modes; compare with ORT CPU's output (ref.bin, NCHW uint8) and print per-op times.
 * Built with -ffp-contract=off (ORT's exact fp32 Add constants). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hmx_gemm_rpc.h"
#include "remote.h"
#include "rn_load.h"
static uint8_t* rd(const char* p, size_t n) {
  FILE* f = fopen(p, "rb");
  uint8_t* b = malloc(n);
  if (!f || fread(b, 1, n, f) != n) { printf("cannot read %s\n", p); exit(2); }
  fclose(f);
  return b;
}
int main(int argc, char** argv) {
  if (argc < 5) return 2;
  int iters = argc > 5 ? atoi(argv[5]) : 10;
  static rn_build_t B;
  if (rn_build(&B, argv[2])) { printf("build failed: %s\n", B.err); return 2; }
  rn_model_t* m = &B.m;
  const rn_tensor_t *ti = &m->t[m->input], *to = &m->t[m->output];
  size_t xl = (size_t)ti->h * ti->w * ti->c, yl = (size_t)to->c * to->h * to->w, fl;
  uint8_t *x = rd(argv[3], xl), *ref = rd(argv[4], yl), *y = malloc(yl);
  uint8_t* xf = rn_pack_input(m, x, &fl);
  size_t ol = qc_geom_bytes(&to->g, to->cp / 32);
  uint8_t* of = malloc(ol);
  printf("%d tensors %d ops, VTCM plan %u bytes, weights+params %u bytes, input %zu (flat %zu), output %zu (flat %zu)\n", m->nt, m->nops,
         m->vtcm_bytes, m->blob_bytes, xl, fl, yl, ol);
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  int rc = hmx_gemm_rpc_open(argv[1], &h), prc = 0, codes[8];
  if (rc) { printf("open failed %d\n", rc); return 1; }
  hmx_gemm_rpc_perf_vote(h, getenv("HMX_FLAGS") ? atoi(getenv("HMX_FLAGS")) : 3, &prc);
  rc = hmx_gemm_rpc_rn_load(h, (const uint8_t*)m, sizeof *m, B.blob, (int)B.blob_len, codes, 8);
  printf("load rc %d ctx %d vtcm %d plan %d err %d\n", rc, codes[0] != 0, codes[4], codes[5], codes[6] | codes[7]);
  if (rc || !codes[0] || codes[5] || codes[6] || codes[7]) return 1;
  int fails = 0;
  int modes[3] = {0, 1, 2}; /* fast, exact, fast with synchronous weight copies (no prefetch thread) */
  for (int mi = 0; mi < 3; mi++) {
    int mode = modes[mi];
    uint64 t[1 + 6 * RN_MAX_OPS];
    rc = hmx_gemm_rpc_rn_run(h, mode, iters, xf, (int)fl, of, (int)ol, t, 1 + 6 * m->nops, codes, 8);
    rn_unpack_output(m, of, y);
    int bad = 0, hist[5] = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < yl; i++) {
      int d = (int)y[i] - (int)ref[i];
      if (d) bad++;
      hist[d < -1 ? 0 : d == -1 ? 1 : d == 0 ? 2 : d == 1 ? 3 : 4]++;
    }
    printf("%s: rc %d hvx %d hmx %d; %d mismatches of %zu vs ORT (%.2f%%; <-1 %d, -1 %d, +1 %d, >1 %d); scalar fixes %d; %.1f us per inference%s\n",
           mode == 1 ? "exact" : mode == 2 ? "fast, no prefetch" : "fast ", rc, codes[1], codes[2], bad, yl, 100.0 * bad / yl, hist[0], hist[1], hist[3], hist[4], codes[3],
           (double)t[0], mode == 1 ? (bad ? " FAIL" : " PASS") : "");
    if (getenv("RN_OPS")) {
      for (int i = 0; i < m->nops; i++) {
        const rn_op_t* o = &m->op[i];
        const rn_tensor_t* Y = &m->t[o->y];
        const uint64* ph = t + 1 + m->nops + 5 * i;
        printf("  op %2d %-7s k%d s%d -> %dx%dx%d: %llu us", i, o->type == RN_CONV ? "conv" : o->type == RN_ADD ? "add" : "maxpool",
               o->k, o->s, Y->c, Y->h, Y->w, (unsigned long long)t[1 + i]);
        if (o->type == RN_ADD)
          printf(" (loop %llu, rescans %llu us over %llu croutons, %llu scalar fixes, pads %llu)", (unsigned long long)ph[0],
                 (unsigned long long)ph[1], (unsigned long long)ph[2], (unsigned long long)ph[3], (unsigned long long)ph[4]);
        if (o->type == RN_CONV)
          printf(" (weights %llu, sources %llu, stitch %llu, hmx+requant %llu, pads %llu)", (unsigned long long)ph[0], (unsigned long long)ph[1],
                 (unsigned long long)ph[2], (unsigned long long)ph[3], (unsigned long long)ph[4]);
        printf("\n");
      }
    }
    if (mode == 1 && bad) fails++;
  }
  /* wall time of one inference per FastRPC call, as the host sees it (QNN's numbers are ORT Run() wall times) */
  for (int mode = 0; mode < 2; mode++) {
    double best = 1e30, sum = 0;
    int n = 30;
    for (int k = 0; k < n; k++) {
      uint64 t[1];
      struct timespec a, b;
      clock_gettime(CLOCK_MONOTONIC, &a);
      hmx_gemm_rpc_rn_run(h, mode, 0, xf, (int)fl, of, (int)ol, t, 1, codes, 8);
      clock_gettime(CLOCK_MONOTONIC, &b);
      double us = (b.tv_sec - a.tv_sec) * 1e6 + (b.tv_nsec - a.tv_nsec) / 1e3;
      sum += us, best = us < best ? us : best;
    }
    printf("%s: wall per single-inference call %.0f us mean, %.0f us best (incl. FastRPC + input copy)\n", mode ? "exact" : "fast ", sum / n, best);
  }
  hmx_gemm_rpc_rn_unload(h);
  hmx_gemm_rpc_close(h);
  return fails ? 3 : 0;
}
