/* Load a qnn_parity/export_case.py case (host C: phone client or hexagon-sim program) and pack it for
 * hmx_qconv.h. */
#ifndef QC_CASE_H
#define QC_CASE_H
#include <stdio.h>
#include <stdlib.h>
#include "hmx_qconv3.h"

typedef struct {
  int M, K, N, zx, zy, relu, H, W, k, stride, Ho, Wo, Mo; /* M = H*W input rows, Mo = Ho*Wo output rows */
  float sx, sy;
  uint8_t *x, *ref;
  int8_t *w, *wk, *wp; /* ONNX [N, K], k-major [K, N], HMX-packed */
  int32_t* bq;
  float* sw;
  qc_blk_t* blk; /* N/32, 256-byte aligned */
  qc_hdr_t hdr;
} qc_case_t;

static void* qc_read(const char* dir, const char* name, size_t n) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(2); }
  void* b = malloc(n);
  if (fread(b, 1, n, f) != n) { printf("short read %s\n", p); exit(2); }
  fclose(f);
  return b;
}

static int qc_load_case(const char* dir, qc_case_t* c) {
  char p[512];
  snprintf(p, sizeof p, "%s/meta.txt", dir);
  FILE* f = fopen(p, "r");
  if (!f) return -1;
  char sxs[64], sys[64];
  if (fscanf(f, "%d %d %d %d %d %d %63s %63s", &c->M, &c->K, &c->N, &c->zx, &c->zy, &c->relu, sxs, sys) != 8) return -1;
  if (fscanf(f, "%d %d %d %d", &c->H, &c->W, &c->k, &c->stride) != 4) c->H = c->M, c->W = 1, c->k = 1, c->stride = 1;
  if (c->k == 1 && c->H * c->W != c->M) c->H = c->M, c->W = 1; /* a row subset of a 1x1 layer */
  c->Ho = c->k == 1 ? c->H : (c->H - 1) / c->stride + 1, c->Wo = c->k == 1 ? c->W : (c->W - 1) / c->stride + 1;
  c->Mo = c->Ho * c->Wo;
  fclose(f);
  c->sx = strtof(sxs, NULL), c->sy = strtof(sys, NULL);
  c->x = qc_read(dir, "x.bin", (size_t)c->M * c->K);
  c->ref = qc_read(dir, "ref.bin", (size_t)c->Mo * c->N);
  size_t kk = (size_t)c->k * c->k;
  c->w = qc_read(dir, "w.bin", (size_t)c->N * c->K * kk);
  c->bq = qc_read(dir, "bq.bin", (size_t)c->N * 4);
  c->sw = qc_read(dir, "sw.bin", (size_t)c->N * 4);
  c->wk = malloc((size_t)c->K * c->N * kk);
  c->wp = malloc((size_t)c->K * c->N * kk);
  if (c->k == 1) {
    qc_transpose_w(c->w, c->N, c->K, c->wk);
    hmx_pack_w_u8cm(c->wk, c->K, c->N, c->wp);
  } else
    qc_pack_w3(c->w, c->N, c->K, c->wk, c->wp);
  c->blk = aligned_alloc(256, sizeof(qc_blk_t) * (c->N / 32));
  qc_pack_params(c->wk, c->K * (int)kk, c->N, c->bq, c->zx, c->sx, c->sw, c->sy, c->zy, c->relu, c->blk, &c->hdr);
  return 0;
}

/* y: [Mo, N] row-major. Returns mismatches; hist[0..2] = counts of diff -1, +1, other */
static int qc_compare(const qc_case_t* c, const uint8_t* y, int* hist) {
  int bad = 0;
  hist[0] = hist[1] = hist[2] = 0;
  for (size_t i = 0; i < (size_t)c->Mo * c->N; i++) {
    int d = (int)y[i] - (int)c->ref[i];
    if (d) bad++, hist[d == -1 ? 0 : d == 1 ? 1 : 2]++;
  }
  return bad;
}
#endif
