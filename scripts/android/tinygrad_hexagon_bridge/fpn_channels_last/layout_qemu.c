/* qemu-hexagon-static check of layout_kernels.h: the HVX transpose must equal the scalar one
 * (bit-exact) at every real FPN level shape (C=256; HW = 850, 3400, 13600, 54400 -- including
 * pixel counts that aren't a multiple of 32), with and without prefetch, both loop orders.
 * Build: clang-19 --target=hexagon -mcpu=hexagonv65 -mhvx=v65 -mhvx-length=128b -static -nostdlib
 *        -ffreestanding -fuse-ld=lld -O2 -o layout_qemu layout_qemu.c
 * Run:   qemu-hexagon-static ./layout_qemu   (prints PASS / FAIL) */
#include "layout_kernels.h"

static long sys6(long a, long b, long c, long nr) {
  long rv;
  __asm__ volatile("r0 = %1; r1 = %2; r2 = %3; r6 = %4; trap0(#1); %0 = r0"
                   : "=r"(rv) : "r"(a), "r"(b), "r"(c), "r"(nr) : "r0", "r1", "r2", "r6");
  return rv;
}
static void put(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 64); }

#define MAXHW 54400
#define CH 256
static float in[CH * MAXHW + 64] __attribute__((aligned(128)));
static float a[CH * MAXHW] __attribute__((aligned(128)));
static float b[CH * MAXHW] __attribute__((aligned(128)));

void _start(void) {
  const int hws[] = {850, 3400, 13600, 54400};
  int bad = 0;
  for (int k = 0; k < 4; k++) {
    int HW = hws[k];
    for (long i = 0; i < (long)CH * HW; i++) in[i] = (float)(i % 100003) * 0.25f - 7.0f;
    transpose_chw_hwc_scalar(in, a, CH, HW, 0, HW);
    for (int pf = 0; pf < 3; pf++) {
      for (long i = 0; i < (long)CH * HW; i++) b[i] = -1.0f;
      if (pf < 2) transpose_chw_hwc_hvx(in, b, CH, HW, 0, HW, pf);
      else transpose_chw_hwc_hvx_cout(in, b, CH, HW, 0, HW);
      for (long i = 0; i < (long)CH * HW; i++)
        if (((const unsigned*)a)[i] != ((const unsigned*)b)[i]) { bad = 1; break; }
    }
    put(bad ? "mismatch\n" : "level ok\n");
  }
  put(bad ? "FAIL\n" : "PASS\n");
  sys6(bad, 0, 0, 93);
}
