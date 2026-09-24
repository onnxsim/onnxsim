/* hexagon-sim check of mcc_decoder.h (the whole query decoder) against ref.py's float64 QueryDecoder:
 *   decoder_sim <data dir> <Q> [hvx mask, default 30]
 * packs K / V with mb_pack_kv (kv.bin), runs mb_decode on xyz.bin, prints occupancy-logit / p > 0.3 /
 * color agreement. */
#include <stdio.h>
#include <stdlib.h>

#include "../mcc_decoder.h"

static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
static void* slurp(const char* dir, const char* name, size_t n) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(1); }
  void* b = malloc(n);
  size_t got = fread(b, 1, n, f);
  fclose(f);
  if (got != n) { printf("%s: %zu of %zu bytes\n", p, got, n); exit(1); }
  return b;
}
int main(int argc, char** argv) {
  const char* dir = argv[1];
  const int Q = atoi(argv[2]);
  uint8_t* vtcm = (uint8_t*)(cfg(0x38) << 16);
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  mb_weights w[MB_BLOCKS];
  mb_hf *kt[MB_BLOCKS], *vt[MB_BLOCKS];
  for (int b = 0; b < MB_BLOCKS; b++) {
    char n[32];
    snprintf(n, sizeof n, "blk%d.bin", b);
    void* blob = slurp(dir, n, mb_block_bytes());
    mb_bind(&w[b], blob);
    kt[b] = mb_blob_kt(blob), vt[b] = mb_blob_vt(blob);
    memset(kt[b], 0, (size_t)16 * MB_ST * MB_TB); /* prove set_kv fills them */
  }
  float* kv = slurp(dir, "kv.bin", (size_t)2 * MB_BLOCKS * MB_HEADS * MB_SEEN * 32 * 4);
  mb_pack_kv(kt, vt, kv, kv + (size_t)MB_BLOCKS * MB_HEADS * MB_SEEN * 32);
  mb_head hd;
  mb_bind_head(&hd, slurp(dir, "head.bin", mb_head_bytes()));
  float* xyz = slurp(dir, "xyz.bin", (size_t)Q * 12);
  float *ro = slurp(dir, "ref_occ.bin", (size_t)Q * 4), *rr = slurp(dir, "ref_rgb.bin", (size_t)Q * 12);
  float *occ = malloc(Q * 4), *rgb = malloc(Q * 12), *pself = malloc(Q * 4);
  mb_ctx c;
  mb_layout(&c, vtcm, Q, pself);
  c.hvx = argc > 3 ? atoi(argv[3]) : MB_V_ALL;
  mb_decode(&c, &hd, w, xyz, occ, rgb, 0);
  double eo = 0, er = 0;
  int flip = 0, occd = 0;
  for (int i = 0; i < Q; i++) {
    eo = fabs(occ[i] - ro[i]) > eo ? fabs(occ[i] - ro[i]) : eo;
    const int a = 1 / (1 + exp(-occ[i])) > 0.3, b = 1 / (1 + exp(-ro[i])) > 0.3;
    flip += a != b, occd += b;
    for (int ch = 0; ch < 3; ch++) er = fabs(rgb[i * 3 + ch] - rr[i * 3 + ch]) > er ? fabs(rgb[i * 3 + ch] - rr[i * 3 + ch]) : er;
  }
  printf("decoder Q=%d hvx %d: occ logit max abs err %.4g, p>0.3 decisions differ %d of %d (%d occupied), rgb max abs err %.4g (x255 %.2f)\n", Q,
         c.hvx, eo, flip, Q, occd, er, er * 255);
  printf(flip <= Q / 100 && er < 0.02 ? "PASS\n" : "FAIL\n");
  return 0;
}
