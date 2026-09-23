/* fp32 NCHW -> NHWC (per image: (C, H*W) -> (H*W, C)) transpose, the conversion the backbone's
 * NCHW FPN outputs need before roialign_fast/roialign_kernel.h can read them.
 *
 * HVX version: 32x32 fp32 blocks (32 channels x 32 pixels). Load 32 channel rows (each 32
 * consecutive pixels, one unaligned 128-byte vector), transpose in registers with five rounds of
 * the perfect-shuffle ("zip") trick -- round k zips rows i and i+16 into rows 2i, 2i+1 via
 * vshuff(.., -4) (word interleave); five rotations of the 10-bit (row, col) index = a transpose --
 * and store each result vector as 32 contiguous channels of one pixel (aligned). No arithmetic:
 * the output is a bit-exact copy. `in` must be readable up to 31 floats past its end (the last
 * pixel block of the last channel over-reads; callers allocate slack). C % 32 == 0.
 * Header-only so the host/qemu checks and the DSP skel compile identical code. */
#ifndef FPN_LAYOUT_KERNELS_H
#define FPN_LAYOUT_KERNELS_H

static void transpose_chw_hwc_scalar(const float* in, float* out, int C, int HW, int p0, int p1) {
  for (int p = p0; p < p1; p += 32) {
    int pe = p + 32 < p1 ? p + 32 : p1;
    for (int c = 0; c < C; c++)
      for (int q = p; q < pe; q++) out[(long)q * C + c] = in[(long)c * HW + q];
  }
}

#if defined(__hexagon__) && defined(__HVX__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline void fpn_l2fetch(const void* p, unsigned bytes) {
  unsigned long long ctl = ((unsigned long long)bytes << 32) | ((unsigned long long)bytes << 16) | 1ull;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(ctl));
}

/* pixels [p0, p1); p0 % 32 == 0. `pf`: l2fetch the next pixel block's 128-byte row chunks. */
static void transpose_chw_hwc_hvx(const float* in, float* out, int C, int HW, int p0, int p1, int pf) {
  for (int p = p0; p < p1; p += 32) {
    const int np = p1 - p < 32 ? p1 - p : 32;
    for (int c0 = 0; c0 < C; c0 += 32) {
      HVX_Vector v[32], t[32];
      if (pf && p + 32 < p1)
        for (int i = 0; i < 32; i++) fpn_l2fetch(in + (long)(c0 + i) * HW + p + 32, 128);
      for (int i = 0; i < 32; i++) v[i] = *(const HVX_UVector*)(in + (long)(c0 + i) * HW + p);
      for (int r = 0; r < 5; r++) {
        for (int i = 0; i < 16; i++) {
          HVX_VectorPair w = Q6_W_vshuff_VVR(v[i + 16], v[i], -4);
          t[2 * i] = Q6_V_lo_W(w);
          t[2 * i + 1] = Q6_V_hi_W(w);
        }
        for (int i = 0; i < 32; i++) v[i] = t[i];
      }
      if (np == 32) {
        for (int j = 0; j < 32; j++) *(HVX_Vector*)(out + (long)(p + j) * C + c0) = v[j];
      } else {
        for (int j = 0; j < np; j++) *(HVX_Vector*)(out + (long)(p + j) * C + c0) = v[j];
      }
    }
  }
}

/* Same blocks, channel-group-outer order: each pass reads only 32 channel rows (32 streams instead
 * of C) and writes one 128-byte column of every pixel's C-float record. */
static void transpose_chw_hwc_hvx_cout(const float* in, float* out, int C, int HW, int p0, int p1) {
  for (int c0 = 0; c0 < C; c0 += 32) {
    for (int p = p0; p < p1; p += 32) {
      const int np = p1 - p < 32 ? p1 - p : 32;
      HVX_Vector v[32], t[32];
      for (int i = 0; i < 32; i++) v[i] = *(const HVX_UVector*)(in + (long)(c0 + i) * HW + p);
      for (int r = 0; r < 5; r++) {
        for (int i = 0; i < 16; i++) {
          HVX_VectorPair w = Q6_W_vshuff_VVR(v[i + 16], v[i], -4);
          t[2 * i] = Q6_V_lo_W(w);
          t[2 * i + 1] = Q6_V_hi_W(w);
        }
        for (int i = 0; i < 32; i++) v[i] = t[i];
      }
      for (int j = 0; j < np; j++) *(HVX_Vector*)(out + (long)(p + j) * C + c0) = v[j];
    }
  }
}
#endif

#endif
