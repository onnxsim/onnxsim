/* hexagon-sim: QDQ 3x3 conv (hmx_qconv3.h, stride 1 or 2) on a qnn_parity/export_case.py case vs ORT CPU, both
 * requantization modes. qconv3_sim <case dir> */
#include <stdio.h>
#include "../qc_case.h"
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static uint8_t* valloc_(uint8_t* v, size_t* off, size_t n) { *off = (*off + 2047) & ~(size_t)2047; uint8_t* p = v + *off; *off += n; return p; }
int main(int argc, char** argv) {
  qc_case_t c;
  if (qc_load_case(argv[1], &c) || c.k != 3) { printf("bad case\n"); return 2; }
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16);
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  int kt = c.K / 32, nt = c.N / 32;
  qc_geom_t gi = qc_geom(c.H, c.W), go = qc_geom(c.Ho, c.Wo);
  size_t off = 0;
  uint8_t* X = valloc_(v, &off, qc_geom_bytes(&gi, kt));
  uint8_t* Y = valloc_(v, &off, qc_geom_bytes(&go, nt));
  uint8_t* W = valloc_(v, &off, (size_t)9 * c.K * c.N);
  qc_blk_t* B = (qc_blk_t*)valloc_(v, &off, sizeof(qc_blk_t) * nt);
  qc_hdr_t* H = (qc_hdr_t*)valloc_(v, &off, sizeof(qc_hdr_t));
  uint8_t* S = valloc_(v, &off, 4 * 2048);
  uint8_t *t0 = valloc_(v, &off, qc_geom_bytes(&go, kt)), *t1 = valloc_(v, &off, qc_geom_bytes(&go, kt));
  uint8_t* ph[4];
  for (int i = 0; i < 4; i++) ph[i] = c.stride == 2 ? valloc_(v, &off, qc_geom_bytes(&go, kt)) : NULL;
  uint8_t* xf = malloc(qc_geom_bytes(&gi, kt));
  qc_flat_pack(c.x, c.H, c.W, c.K, c.zx, &gi, xf);
  memcpy(X, xf, qc_geom_bytes(&gi, kt));
  memcpy(W, c.wp, (size_t)9 * c.K * c.N);
  memcpy(B, c.blk, sizeof(qc_blk_t) * nt);
  *H = c.hdr;
  qc_taps_t tp;
  if (c.stride == 1) {
    qc_shift_copies(X, t0, t1, gi.nblk, kt, c.zx);
    tp = qc_taps_s1(X, t0, t1);
  } else {
    qc_phase_split(X, &gi, ph, &go, kt, c.zx);
    uint8_t* junk = Y; /* p1 output unused: write it into Y, which the conv overwrites */
    qc_shift_copies(ph[1], t0, junk, go.nblk, kt, c.zx);
    qc_shift_copies(ph[3], t1, junk, go.nblk, kt, c.zx);
    tp = qc_taps_s2(ph, t0, t1);
  }
  uint8_t* y = malloc((size_t)c.Mo * c.N);
  uint32_t* atab = malloc(sizeof(uint32_t) * qc_geom_nob(&go) * 9 * kt);
  uint8_t* side = valloc_(v, &off, 64 * 2048);
  qc_stitch_t st[64];
  int ns = qc_conv3x3_plan(&tp, &go, kt, atab, st, side, 64);
  if (ns < 0) { printf("side buffer too small\n"); return 2; }
  qc_conv3x3_stitch(st, ns, kt);
  for (int mode = 0; mode < 2; mode++) {
    int nfix = qc_conv3x3(atab, &go, Y, W, B, H, kt, mode, S);
    qc_flat_unpack(Y, &go, c.N, y);
    int hist[3], bad = qc_compare(&c, y, hist);
    printf("%s 3x3 s%d %dx%dx%d -> %d: %d mismatches of %d vs ORT (-1: %d, +1: %d, other: %d), scalar-fixed groups %d, stitched %d%s\n",
           mode ? "exact" : "fast ", c.stride, c.H, c.W, c.K, c.N, bad, c.Mo * c.N, hist[0], hist[1], hist[2], nfix, ns,
           mode && bad ? " FAIL" : mode ? " PASS" : "");
  }
  return 0;
}
