/* hexagon-sim check of mcc_block.h against ref.py's float64 references:
 *   block_sim <data dir> <Q> <blocks> [hvx mask (mcc_block.h MB_V_*, default 0 = scalar)]
 * (the residual adds always run on HMX: identity K tile)
 * runs <blocks> decoder blocks on x0 (each on the previous output) and prints, per block, the error
 * vs the float64 reference output (max abs, max |ref|, cosine) and the block's cycle count. */
#include <stdio.h>
#include <stdlib.h>

#include "../mcc_block.h"

static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
static unsigned long long cyc(void) { unsigned long long c; __asm__ volatile("%0 = c15:14" : "=r"(c)); return c; }
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
  const int Q = atoi(argv[2]), nb = atoi(argv[3]);
  uint8_t* vtcm = (uint8_t*)(cfg(0x38) << 16);
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r)); /* HMX enable */
  mb_ctx c;
  float* pself = malloc(Q * sizeof(float));
  mb_layout(&c, vtcm, Q, pself);
  c.hvx = argc > 4 ? atoi(argv[4]) : 0;
  mb_hf* x0 = slurp(dir, "x0.bin", (size_t)Q * MB_D * 2);
  for (int i = 0; i < Q; i++) for (int j = 0; j < MB_D; j++) *mb_at(c.x, MB_KT, i, j) = x0[(size_t)i * MB_D + j];
  int fail = 0;
  for (int b = 0; b < nb; b++) {
    char n[64];
    snprintf(n, sizeof n, "blk%d.bin", b);
    void* blob = slurp(dir, n, mb_block_bytes());
    mb_weights w;
    mb_bind(&w, blob);
    unsigned long long t0 = cyc();
    mb_block(&c, &w, 0);
    unsigned long long t1 = cyc();
    snprintf(n, sizeof n, "ref_out%d.bin", b);
    float* ref = slurp(dir, n, (size_t)Q * MB_D * 4);
    double e = 0, mr = 0, dot = 0, na = 0, nr = 0;
    for (int i = 0; i < Q; i++) for (int j = 0; j < MB_D; j++) {
      double g = mb_h2f(*mb_at(c.x, MB_KT, i, j)), f = ref[(size_t)i * MB_D + j];
      e = fabs(g - f) > e ? fabs(g - f) : e; mr = fabs(f) > mr ? fabs(f) : mr;
      dot += g * f; na += g * g; nr += f * f;
    }
    double cs = dot / sqrt(na * nr);
    printf("block %d: max abs err %.4g (max |ref| %.4g), cos %.7f, %llu cycles\n", b, e, mr, cs, t1 - t0);
    fail |= !(cs > 0.9999);
    free(ref);
    free(blob);
  }
  printf(fail ? "FAIL\n" : "PASS\n");
  return 0;
}
