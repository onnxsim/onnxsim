/* int8 HMX GEMM on V69 through the "cm" path QNN's own int8 convs use (see hmx_block.h):
 *   C[M, N] (uint8) = min(255, floor(max(A . W, 0) * s[n] / 512)),  A uint8 [M, K], W int8 [K, N], s fp16 [N]
 * row-major A and C; W prepacked once with hmx_pack_w_u8cm. K and N multiples of 64 (vector pack/unpack
 * paths need K % 128 == 0 and N % 128 == 0, otherwise a scalar path is used); any M.
 *
 * One instruction is 64 rows x 32 K x 64 columns (activation crouton :cm + weight :deep), so per 2 KB of
 * activation read the HMX does 4x the MACs of the fp16 path (32 x 32 x 32 per 2 KB + 2 KB of weights).
 * Callers must hold HVX + HMX exactly as for hmx_gemm.h. */
#ifndef HMX_GEMM_U8_H
#define HMX_GEMM_U8_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hmx_gemm.h" /* hmx_block.h, hmx_l2fetch, hmx_copy_hvx */

#define HMX_U8_ROWS 64

/* W int8 [K, N] row-major -> per 64-column block jb, per 32-deep K block kb: 2 KB = columns jb*64 + 0..31 then
 * + 32..63, each 1 KB block with W(k, c) at byte 128*(k/4) + 4*c + k%4. out holds K * N bytes. */
static inline void hmx_pack_w_u8cm(const int8_t* W, int K, int N, int8_t* out) {
  for (int jb = 0; jb < N / 64; jb++)
    for (int kb = 0; kb < K / 32; kb++) {
      int8_t* t = out + ((size_t)jb * (K / 32) + kb) * 2048;
      for (int h = 0; h < 2; h++)
        for (int k = 0; k < 32; k++)
          for (int c = 0; c < 32; c++)
            t[1024 * h + 128 * (k / 4) + 4 * c + k % 4] = W[(size_t)(kb * 32 + k) * N + jb * 64 + 32 * h + c];
    }
}

/* VTCM layout: A row blocks (mt x kt croutons), two weight buffers of one 128-column group (kt x 8 KB each),
 * 4 output tiles, the 4 column tables of a group. Returns bytes; offsets may be NULL. resident_w: all of W instead. */
static inline size_t hmx_gemm_u8_layout(int M, int K, int N, int resident_w, size_t* a_off, size_t* w_off,
                                        size_t* c_off, size_t* t_off) {
  size_t mt = (M + 63) / 64, kt = K / 32, off = 0;
  size_t wgrp = kt * 8192; /* 128 columns */
  if (a_off) *a_off = off;
  off += mt * kt * 2048;
  if (w_off) w_off[0] = off;
  off += resident_w ? (size_t)K * N : wgrp;
  if (w_off) w_off[1] = off;
  off += resident_w ? 0 : wgrp;
  if (c_off) *c_off = off;
  off += 4 * 2048;
  if (t_off) *t_off = off;
  off += 4 * 256;
  return off;
}

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hexagon_protos.h>
typedef HVX_Vector __attribute__((aligned(1))) hmx_u8_uvec;

/* 4x4 transpose of 32-byte chunks across 4 vectors: out[j] chunk i = in[i] chunk j. */
static inline void hmx_tr4x32(HVX_Vector v0, HVX_Vector v1, HVX_Vector v2, HVX_Vector v3, HVX_Vector* o) {
  HVX_VectorPair p01 = Q6_W_vshuff_VVR(v1, v0, -32), p23 = Q6_W_vshuff_VVR(v3, v2, -32);
  HVX_VectorPair lo = Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
  HVX_VectorPair hi = Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
  o[0] = Q6_V_lo_W(lo), o[1] = Q6_V_hi_W(lo), o[2] = Q6_V_lo_W(hi), o[3] = Q6_V_hi_W(hi);
}

/* rows m0..m0+63 of A (zero past M) -> kt croutons at out (A(s, k) = byte 32*s + k%32 of crouton k/32) */
static inline void hmx_pack_a_u8cm(const uint8_t* A, int M, int K, int m0, uint8_t* out) {
  int kt = K / 32;
  if (K % 128 == 0) {
    HVX_Vector z = Q6_V_vzero(), o[4];
    for (int s = 0; s < 64; s += 4) {
      const uint8_t* r[4];
      for (int i = 0; i < 4; i++) r[i] = m0 + s + i < M ? A + (size_t)(m0 + s + i) * K : NULL;
      for (int k = 0; k < K; k += 128) {
        HVX_Vector v[4];
        for (int i = 0; i < 4; i++) v[i] = r[i] ? *(const hmx_u8_uvec*)(r[i] + k) : z;
        hmx_tr4x32(v[0], v[1], v[2], v[3], o);
        for (int j = 0; j < 4; j++) *(HVX_Vector*)(out + (size_t)(k / 32 + j) * 2048 + 32 * s) = o[j];
      }
    }
    return;
  }
  for (int kb = 0; kb < kt; kb++)
    for (int s = 0; s < 64; s++) {
      uint8_t* d = out + (size_t)kb * 2048 + 32 * s;
      if (m0 + s < M) memcpy(d, A + (size_t)(m0 + s) * K + kb * 32, 32);
      else memset(d, 0, 32);
    }
}

/* 4 output tiles (columns n0..n0+127) of rows m0.. -> C (rows < M only) */
static inline void hmx_unpack4_u8cm(const uint8_t* t, uint8_t* C, int M, int N, int m0, int n0) {
  HVX_Vector o[4];
  for (int s = 0; s < 64 && m0 + s < M; s += 4) {
    hmx_tr4x32(*(const HVX_Vector*)(t + 32 * s), *(const HVX_Vector*)(t + 2048 + 32 * s),
               *(const HVX_Vector*)(t + 4096 + 32 * s), *(const HVX_Vector*)(t + 6144 + 32 * s), o);
    for (int i = 0; i < 4 && m0 + s + i < M; i++) *(hmx_u8_uvec*)(C + (size_t)(m0 + s + i) * N + n0) = o[i];
  }
}
static inline void hmx_unpack_u8cm_scalar(const uint8_t* t, int ntiles, uint8_t* C, int M, int N, int m0, int n0) {
  for (int j = 0; j < ntiles; j++)
    for (int s = 0; s < 64 && m0 + s < M; s++) memcpy(C + (size_t)(m0 + s) * N + n0 + 32 * j, t + 2048 * j + 32 * s, 32);
}

#ifndef HMX_NOW
#define HMX_NOW() 0ull
#endif
#ifndef HMX_COPY
#define HMX_COPY hmx_copy_hvx
#endif

/* prof (may be NULL): += pack A, weight copy, MAC + store, C unpack (HMX_NOW units).
 * resident_w: Wp is already the VTCM weight area (mode "resident"): no weight copies. */
static inline int hmx_gemm_u8_prof(const uint8_t* A, const int8_t* Wp, const uint16_t* scale, uint8_t* C, int M,
                                   int K, int N, uint8_t* vtcm, size_t vtcm_bytes, int resident_w,
                                   unsigned long long* prof) {
  unsigned long long pr[4] = {0, 0, 0, 0}, t0 = HMX_NOW(), t1;
  int mt = (M + 63) / 64, kt = K / 32;
  size_t a_off, w_off[2], c_off, t_off;
  if (K % 64 || N % 64 || hmx_gemm_u8_layout(M, K, N, resident_w, &a_off, w_off, &c_off, &t_off) > vtcm_bytes) return -1;
  uint8_t *va = vtcm + a_off, *ct = vtcm + c_off;
  uint32_t* tbl = (uint32_t*)(vtcm + t_off);
  for (int mb = 0; mb < mt; mb++) {
    if (mb + 1 < mt) hmx_l2fetch(A + (size_t)(mb + 1) * 64 * K, (size_t)((M - (mb + 1) * 64) < 64 ? M - (mb + 1) * 64 : 64) * K);
    hmx_pack_a_u8cm(A, M, K, mb * 64, va + (size_t)mb * kt * 2048);
  }
  t1 = HMX_NOW(), pr[0] += t1 - t0, t0 = t1;
  for (int n0 = 0; n0 < N; n0 += 128) {
    int nc = N - n0 < 128 ? N - n0 : 128, groups = nc / 64; /* 64-column deep blocks in this group */
    const uint8_t* wg;
    if (resident_w)
      wg = (const uint8_t*)Wp + (size_t)(n0 / 64) * kt * 2048;
    else {
      HMX_COPY(vtcm + w_off[0], Wp + (size_t)(n0 / 64) * kt * 2048, (size_t)groups * kt * 2048);
      wg = vtcm + w_off[0];
    }
    for (int j = 0; j < 2 * groups; j++)
      for (int c = 0; c < 32; c++) tbl[64 * j + c] = scale[n0 + 32 * j + c];
    t1 = HMX_NOW(), pr[1] += t1 - t0, t0 = t1;
    for (int mb = 0; mb < mt; mb++) {
      const uint8_t* a = va + (size_t)mb * kt * 2048;
      for (int g = 0; g < groups; g++) {
        hmx_blk_mac_u8cm_deep(a, wg + (size_t)g * kt * 2048, kt);
        for (int h = 0; h < 2; h++) {
          hmx_blk_set_table(tbl + 64 * (2 * g + h));
          hmx_blk_store_u8cm(ct + (size_t)(2 * g + h) * 2048);
        }
      }
      t1 = HMX_NOW(), pr[2] += t1 - t0, t0 = t1;
      if (nc == 128) hmx_unpack4_u8cm(ct, C, M, N, mb * 64, n0);
      else hmx_unpack_u8cm_scalar(ct, 2 * groups, C, M, N, mb * 64, n0);
      t1 = HMX_NOW(), pr[3] += t1 - t0, t0 = t1;
    }
  }
  if (prof)
    for (int i = 0; i < 4; i++) prof[i] += pr[i];
  return 0;
}
/* One layer entirely in VTCM, activations in crouton form on both sides: Y = sat_u8(X . W) where X holds mt row
 * blocks of kt input croutons (row block mb at X + mb*kt*2 KB) and Y receives mt row blocks of N/32 croutons --
 * an output tile (64 rows x 32 columns, byte 32*s + c) is exactly the next layer's activation crouton, so layers
 * chain with no repacking. W = hmx_pack_w_u8cm layout (in VTCM), tbl = N/32 column tables of 256 B (fp16 scale
 * of column c in the low half of word c). */
static inline void hmx_layer_u8cm(const uint8_t* X, uint8_t* Y, const uint8_t* W, const uint32_t* tbl, int mt, int kt,
                                  int N) {
  int nt = N / 32;
  for (int g = 0; g < N / 64; g++)
    for (int mb = 0; mb < mt; mb++) {
      hmx_blk_mac_u8cm_deep(X + (size_t)mb * kt * 2048, W + (size_t)g * kt * 2048, kt);
      for (int h = 0; h < 2; h++) {
        hmx_blk_set_table(tbl + 64 * (2 * g + h));
        hmx_blk_store_u8cm(Y + ((size_t)mb * nt + 2 * g + h) * 2048);
      }
    }
}

/* Crouton-form Y (mt row blocks x N/32 croutons) -> row-major C [M, N] */
static inline void hmx_unpack_rows_u8cm(const uint8_t* Y, uint8_t* C, int M, int N) {
  int nt = N / 32;
  for (int mb = 0; mb < (M + 63) / 64; mb++)
    for (int j = 0; j < nt; j++)
      for (int s = 0; s < 64 && mb * 64 + s < M; s++)
        memcpy(C + (size_t)(mb * 64 + s) * N + 32 * j, Y + ((size_t)mb * nt + j) * 2048 + 32 * s, 32);
}
#endif /* __hexagon__ */
#endif
