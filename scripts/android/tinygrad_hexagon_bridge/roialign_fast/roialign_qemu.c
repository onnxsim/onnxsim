/* Build: clang-19 --target=hexagon -mcpu=hexagonv65 -mhvx=v65 -mhvx-length=128b -O2 -static -nostdlib
 *        -ffreestanding -fuse-ld=lld -o roiq roialign_qemu.c
 * Run:   qemu-hexagon-static roiq callN_feat.bin callN_rois.bin callN_ref.bin H W C R OH OW sr 1/scale
 * (-mhvx=v65: qemu 8.2 cannot decode the qfloat HVX ops a v68+/v73 build emits.) */
/* roialign_kernel.h under qemu-hexagon-static (linux-user, freestanding: raw trap0 syscalls, the
 * same bare-metal style tinygrad's MOCKDSP path uses). Reads one call's binaries (paths on argv via
 * a tiny stack parser), runs the kernel, prints max abs error vs ORT's output and QEMU's executed-
 * instruction count for the kernel alone (control register 21, HEX_REG_QEMU_INSN_CNT). */
#include "roialign_kernel.h"
static long sys6(long a, long b, long c, long d, long e, long f, long n) {
  long r; __asm__ volatile("r0=%1;r1=%2;r2=%3;r3=%4;r4=%5;r5=%6;r6=%7;trap0(#1);%0=r0"
    : "=r"(r) : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(n)
    : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "memory"); return r; }
static unsigned inscount(void) { unsigned r; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r"(r) : : "r0"); return r; }
static void* load(const char* p, long bytes) {
  int fd = sys6(-100, (long)p, 0, 0, 0, 0, 56);
  char* b = (char*)sys6(0, (bytes + 4095) & ~4095L, 3, 0x22, -1, 0, 222);
  for (long o = 0; o < bytes;) { long n = sys6(fd, (long)(b + o), bytes - o, 0, 0, 0, 63); if (n <= 0) break; o += n; }
  return b; }
static void puts_(const char* s) { int n = 0; while (s[n]) n++; sys6(1, (long)s, n, 0, 0, 0, 64); }
static void putu(unsigned long v) { char b[24]; int i = 23; b[i] = 0; do { b[--i] = '0' + v % 10; v /= 10; } while (v); puts_(b + i); }
static int atoi_(const char* s) { int v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0'); return v; }
void __attribute__((noreturn)) main_(long* sp) {
  char** argv = (char**)(sp + 1);  /* argv: feat rois ref H W C R OH OW sr scale_inv */
  int H = atoi_(argv[4]), W = atoi_(argv[5]), C = atoi_(argv[6]), R = atoi_(argv[7]);
  int OH = atoi_(argv[8]), OW = atoi_(argv[9]), sr = atoi_(argv[10]);
  float scale = 1.0f / (float)atoi_(argv[11]);
  long m = (long)R * OH * OW * C;
  float* feat = load(argv[1], (long)H * W * C * 4); float* rois = load(argv[2], (long)R * 16);
  float* ref = load(argv[3], m * 4);
  float* out = (float*)sys6(0, (m * 4 + 4095) & ~4095L, 3, 0x22, -1, 0, 222);
  unsigned t0 = inscount();
  roialign_hwc(feat, H, W, C, rois, R, OH, OW, sr, scale, out);
  unsigned t1 = inscount();
  float mx = 0; for (long k = 0; k < m; k++) { float e = out[k] - ref[k]; if (e < 0) e = -e; if (e > mx) mx = e; }
  puts_("max_abs_err_x1e6="); putu((unsigned long)(mx * 1e6f)); puts_(" insns="); putu(t1 - t0); puts_("\n");
  sys6(mx < 1e-3f ? 0 : 1, 0, 0, 0, 0, 0, 93); for (;;) {} }
__asm__(".global _start\n_start:\n r0 = r29\n jump main_\n");
