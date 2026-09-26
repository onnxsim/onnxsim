/* hexagon-sim: QDQ 1x1 conv (hmx_qconv.h) on a qnn_parity/export_case.py case vs ORT CPU's output, both modes.
 * qconv_sim <case dir> */
#include <stdio.h>
#include "../qc_case.h"
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
int main(int argc, char** argv) {
  qc_case_t c;
  if (qc_load_case(argv[1], &c)) { printf("bad case\n"); return 2; }
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16);
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  int mt = (c.M + 63) / 64, kt = c.K / 32, nt = c.N / 32;
  size_t off = 0;
  uint8_t* X = v + off; off += (size_t)mt * kt * 2048;
  uint8_t* Y = v + off; off += (size_t)mt * nt * 2048;
  uint8_t* W = v + off; off += (size_t)c.K * c.N;
  off = (off + 255) & ~(size_t)255;
  qc_blk_t* B = (qc_blk_t*)(v + off); off += sizeof(qc_blk_t) * nt;
  qc_hdr_t* H = (qc_hdr_t*)(v + off); off += sizeof(qc_hdr_t);
  off = (off + 2047) & ~(size_t)2047; /* HMX tile stores need 2 KB alignment */
  uint8_t* S = v + off; off += 4 * 2048;
  memcpy(W, c.wp, (size_t)c.K * c.N);
  memcpy(B, c.blk, sizeof(qc_blk_t) * nt);
  *H = c.hdr;
  for (int mb = 0; mb < mt; mb++) hmx_pack_a_u8cm(c.x, c.M, c.K, mb * 64, X + (size_t)mb * kt * 2048);
  uint8_t* y = malloc((size_t)c.M * c.N);
  for (int mode = 0; mode < 2; mode++) {
    int nfix = qc_conv1x1(X, Y, W, B, H, mt, kt, mode, S);
    hmx_unpack_rows_u8cm(Y, y, c.M, c.N);
    int hist[3], bad = qc_compare(&c, y, hist);
    printf("%s %dx%dx%d relu %d: %d mismatches of %d vs ORT (-1: %d, +1: %d, other: %d), scalar-fixed groups %d%s\n",
           mode ? "exact" : "fast ", c.M, c.K, c.N, c.relu, bad, c.M * c.N, hist[0], hist[1], hist[2], nfix,
           mode && bad ? " FAIL" : mode ? " PASS" : "");
  }
  return 0;
}
