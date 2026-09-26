/* QDQ-exact int8 1x1 convolution on V69 HMX (the ":cm" path of hmx_gemm_u8.h), as ORT CPU computes a QDQ
 * Conv (QLinearConv after its QDQ fusion):
 *
 *   acc[m, c] = sum_k (xq[m, k] - zx) * wq[k, c] + bq[c]                      (exact int32)
 *   y[m, c]   = clamp(rne(fp32(fp32(acc) * M[c])) + zy, lo, 255),  M[c] = fp32(fp32(sx * sw[c]) / sy)
 *   lo = zy under a fused Relu, else 0.  xq uint8 (zero point zx), wq int8 per-channel symmetric.
 *
 * The zero point folds into the bias (xq goes to the HMX unchanged): acc = sum xq*wq + (bq - zx * sum_k wq).
 * That int32 is the high word of the HMX 64-bit column table (bias = mxmem2), added exactly (hexagon-sim).
 *
 * Two requantization modes:
 *  QC_FAST   one `:after:cm:sat.ub` store per tile: HMX converts floor(trunc(acc + B) * s / 512 + 0.5) with
 *            s = fp16(512 * M) (table low word; bit 22 = the +0.5) and B = bias + round(zy / M). The HMX keeps
 *            only ~4 fractional output bits (acc truncated to a multiple of 2^(5 - exp(s))) and an 11-bit
 *            scale, so ~5-8% of outputs are 1 off -- the same class as QNN's own HTP output (measured 5.6-7.7%
 *            off vs ORT on the same layers).
 *  QC_EXACT  four non-saturating `:cm.ub` stores from the same accumulator (`:retain`) at scales 1, 2^-8,
 *            2^-16, 2^-24 give the four bytes of acc (exact int32, two's complement, hexagon-sim + phone). HVX
 *            then requantizes in integers: r = round(acc * 2^L * bm / 2^31) ~ v * 2^F (v = acc * M), y = round
 *            half up of r / 2^F. Outputs whose r lies within a per-column window of a .5 boundary (the window
 *            covers our integer error and fp32's own rounding in ORT's formula) are recomputed on the scalar
 *            core with ORT's exact fp32 formula (IEEE sfmpy + convert_sf2w round-to-nearest-even). Bit-exact.
 *
 * Layouts as hmx_gemm_u8.h: activations / outputs in crouton form (64 rows x 32 channels, byte 32*s + c,
 * row block mb of kt croutons at X + mb*kt*2 KB), weights packed by hmx_pack_w_u8cm (64 columns per :deep). */
#ifndef HMX_QCONV_H
#define HMX_QCONV_H
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "hmx_gemm_u8.h"

enum { QC_FAST = 0, QC_EXACT = 1 };

/* per 32-output-column block; lives in VTCM (the HMX reads the tables from there), 256-byte aligned: a 64-bit
 * table load (bias = mxmem2) needs a 256-byte-aligned address (a 128-aligned one loads garbage, hexagon-sim) */
typedef struct {
  uint32_t tbl_fast[64];                                    /* QC_FAST column table (lo words, hi words) */
  uint32_t tbl_plane[4][64];                                /* QC_EXACT byte-plane tables */
  int32_t L[32], bm[32], F[32], half[32], mask[32], win[32]; /* QC_EXACT HVX constants, per column */
  float mf[32];                                             /* M[c] as ORT's fp32, for the scalar fix */
  int32_t pad[32];                                          /* size 2304 = 9 x 256 */
} qc_blk_t;

typedef struct {
  int32_t zy, lo, relu, n;
  int32_t pad[28];
} qc_hdr_t; /* 128 bytes */

static inline uint16_t qc_f2h(float f) {
  __fp16 h = (__fp16)f;
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
static inline int qc_bitlen(uint64_t v) {
  int n = 0;
  while (v) n++, v >>= 1;
  return n;
}

/* Host-side parameter packing for an N-output layer (N a multiple of 32). w: int8 [K, N] (k-major, i.e. W^T of
 * ONNX's [N, K]), bq int32 [N], sw fp32 [N]. Fills blk[N/32] and hdr. */
static inline void qc_pack_params(const int8_t* w, int K, int N, const int32_t* bq, int zx, float sx, const float* sw,
                                  float sy, int zy, int relu, qc_blk_t* blk, qc_hdr_t* hdr) {
  static const uint16_t plane_scale[4] = {0x6000, 0x4000, 0x2000, 0x0800}; /* 512, 2, 2^-7, 2^-15: x1, /2^8, /2^16, /2^24 */
  memset(hdr, 0, sizeof *hdr);
  hdr->zy = zy, hdr->lo = relu ? zy : 0, hdr->relu = relu, hdr->n = N;
  for (int j = 0; j < N / 32; j++) {
    qc_blk_t* b = &blk[j];
    memset(b, 0, sizeof *b);
    for (int c = 0; c < 32; c++) {
      int n = 32 * j + c;
      long sumw = 0, suma = 0;
      for (int k = 0; k < K; k++) sumw += w[(size_t)k * N + n], suma += labs(w[(size_t)k * N + n]);
      long bex = (long)bq[n] - (long)zx * sumw;
      float m = (sx * sw[n]) / sy; /* ORT: fp32(fp32(sx * sw) / sy) */
      volatile float mv = m;
      m = mv;
      b->mf[c] = m;
      /* QC_FAST */
      /* the HMX truncates acc + B to a multiple of 2^(5 - E) (E = exponent of s): add half of that to center the
       * error instead of always rounding down */
      uint16_t sh = qc_f2h(512.0f * m);
      int E = ((sh >> 10) & 31) - 15;
      long bf = bex + lrint((double)zy / m) + (4 - E >= 0 ? 1L << (4 - E) : 0);
      b->tbl_fast[c] = (1u << 22) | sh;
      b->tbl_fast[32 + c] = (uint32_t)(int32_t)bf;
      for (int p = 0; p < 4; p++) b->tbl_plane[p][c] = plane_scale[p], b->tbl_plane[p][32 + c] = (uint32_t)(int32_t)bex;
      /* QC_EXACT: |acc| <= bound; a = acc << L stays below 2^30 */
      uint64_t bound = (uint64_t)255 * suma + (uint64_t)labs(bex);
      int L = 30 - qc_bitlen(bound);
      if (L < 0) L = 0;
      /* r = acc * 2^L * bm / 2^31 = v * 2^F: bm = M * 2^(F + 31 - L) < 2^31 and |v| < 512 unsaturated (F <= 21) */
      int e = (int)floor(log2((double)m));        /* M in [2^e, 2^(e+1)) */
      int F = L - e - 1;                           /* largest F with bm < 2^31 */
      if (F > 21) { /* |v| < 512 must not saturate: lower L instead, keeping bm in [2^30, 2^31) (a tiny-M column) */
        F = 21, L = F + e + 1;
        if (L < 0) L = 0;
      }
      if (F < 1) F = 1;
      double bmd = ldexp((double)m, F + 31 - L);
      long bm = lrint(bmd);
      if (bm >= 2147483647L) bm = 2147483647L;
      b->L[c] = L, b->bm[c] = (int32_t)bm, b->F[c] = F;
      b->half[c] = F > 0 ? 1 << (F - 1) : 0, b->mask[c] = (int32_t)((1u << F) - 1);
      /* error of r (units of 2^-F): 1 (rounding of the high multiply) + the bm quantization over |r| < 2^31,
       * plus ORT's own fp32 error: half an ulp of |v| < 512 (2^-16) and, for |acc| >= 2^24, fp32(acc)'s rounding */
      double err = 2.0 + ldexp(1.0, 31) * fabs(bmd - (double)bm) / (bmd > 0 ? bmd : 1) + ldexp(1.0, F - 16);
      int ab = qc_bitlen(bound);
      /* fp32(acc) rounds only for |acc| >= 2^24, i.e. |v| >= M 2^24: irrelevant when those outputs saturate anyway */
      if (ab > 24 && ldexp((double)m, 24) < 1024) err += ldexp((double)m, F + ab - 25);
      b->win[c] = (int32_t)ceil(err) + 1;
      if ((double)bound * m < 0.25) /* a (near-)dead column: |v| < 0.25 always, so y = zy exactly (r = 0, never flagged) */
        b->L[c] = 0, b->bm[c] = 0, b->F[c] = 1, b->half[c] = 1, b->mask[c] = 1, b->win[c] = 0;
    }
  }
}

/* weights: ONNX [N, K] int8 -> k-major [K, N] */
static inline void qc_transpose_w(const int8_t* wo, int N, int K, int8_t* wk) {
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) wk[(size_t)k * N + n] = wo[(size_t)n * K + k];
}

#ifdef __hexagon__
static inline HVX_Vector qc_ror_or(HVX_Vector v) {
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 64));
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 32));
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 16));
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 8));
  return Q6_V_vor_VV(v, Q6_V_vror_VR(v, 4));
}

static inline int32_t qc_acc_from_planes(const uint8_t* planes, int i) {
  return (int32_t)((uint32_t)planes[i] | (uint32_t)planes[2048 + i] << 8 | (uint32_t)planes[4096 + i] << 16 |
                   (uint32_t)planes[6144 + i] << 24);
}

/* ORT's formula on the scalar core, for the rows of group q (4 rows x 32 columns). Scalar loads from VTCM are
 * slow (~40 cycles each: a flagged tile cost 38k cycles), so HVX first copies the group's four plane slices and
 * the column scales into cached stack memory. */
static inline void qc_fix_group(const uint8_t* planes, uint8_t* out, const qc_blk_t* b, int zy, int lo, int q) {
  HVX_Vector st[6];
  const HVX_Vector* p = (const HVX_Vector*)(planes + 128 * q);
  st[0] = p[0], st[1] = p[16], st[2] = p[32], st[3] = p[48], st[4] = *(const HVX_Vector*)b->mf, st[5] = *(const HVX_Vector*)(out + 128 * q);
  const uint8_t* c = (const uint8_t*)st;
  const float* mf = (const float*)&st[4];
  uint8_t* o = (uint8_t*)&st[5];
  for (int i = 0; i < 128; i++) {
    int32_t acc = (int32_t)((uint32_t)c[i] | (uint32_t)c[128 + i] << 8 | (uint32_t)c[256 + i] << 16 | (uint32_t)c[384 + i] << 24);
    float v = __builtin_HEXAGON_F2_conv_w2sf(acc) * mf[i & 31];
    int y = __builtin_HEXAGON_F2_conv_sf2w(v) + zy;
    o[i] = (uint8_t)(y < lo ? lo : y > 255 ? 255 : y);
  }
  *(HVX_Vector*)(out + 128 * q) = st[5];
}

/* planes: 4 x 2 KB byte planes of one 64 x 32 accumulator tile -> out (2 KB crouton tile). Returns the number of
 * 4-row groups that needed the scalar fix. */
static inline int qc_requant_tile(const uint8_t* planes, uint8_t* out, const qc_blk_t* b, const qc_hdr_t* h) {
  HVX_Vector flags[18];
  const HVX_Vector* bv = (const HVX_Vector*)__builtin_assume_aligned(b->L, 128);
  HVX_Vector L = bv[0], bm = bv[1], F = bv[2], half = bv[3], mask = bv[4], win = bv[5];
  /* zy folds into the rounding constant (y = (r + 2^(F-1) + zy * 2^F) >> F); the saturating packs clamp to 0..255 */
  HVX_Vector hz = Q6_Vw_vadd_VwVw(half, Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(h->zy), F)), one = Q6_V_vsplat_R(1);
  HVX_Vector zero = Q6_V_vzero(), any = zero, lo = Q6_Vb_vsplat_R(h->lo);
  int clamp_lo = h->lo > 0;
  /* one row (32 columns, one word each): r ~ v * 2^F, y = round-half-up(r / 2^F) + zy; FLAG |= near .5 */
#define QC_ROW(ACC, Y, FL)                                                                              \
  do {                                                                                                  \
    HVX_Vector a_ = Q6_Vw_vasl_VwVw(ACC, L);                                                            \
    HVX_Vector r_ = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(Q6_Vw_vmpye_VwVuh(a_, bm), a_, bm);          \
    HVX_Vector t_ = Q6_Vw_vasr_VwVw(Q6_Vw_vadd_VwVw_sat(r_, hz), F);                                    \
    HVX_Vector d_ = Q6_Vw_vabs_Vw(Q6_Vw_vsub_VwVw(Q6_V_vand_VV(r_, mask), half));                        \
    FL = Q6_Q_or_QQ(FL, Q6_Q_vcmp_gt_VwVw(win, d_));                                                    \
    Y = t_;                                                                                             \
  } while (0)
#pragma unroll 2
  for (int q = 0; q < 16; q++) {
    const HVX_Vector* p = (const HVX_Vector*)(planes + 128 * q);
    HVX_VectorPair b01 = Q6_W_vshuff_VVR(p[16], p[0], -1), b23 = Q6_W_vshuff_VVR(p[48], p[32], -1);
    HVX_VectorPair w0 = Q6_W_vshuff_VVR(Q6_V_lo_W(b23), Q6_V_lo_W(b01), -2);
    HVX_VectorPair w1 = Q6_W_vshuff_VVR(Q6_V_hi_W(b23), Q6_V_hi_W(b01), -2);
    HVX_Vector y0, y1, y2, y3;
    HVX_VectorPred fl = Q6_Q_vcmp_gt_VwVw(zero, one); /* all false */
    QC_ROW(Q6_V_lo_W(w0), y0, fl);
    QC_ROW(Q6_V_hi_W(w0), y1, fl);
    QC_ROW(Q6_V_lo_W(w1), y2, fl);
    QC_ROW(Q6_V_hi_W(w1), y3, fl);
    HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(y1, y0), h23 = Q6_Vh_vpack_VwVw_sat(y3, y2);
    HVX_Vector yb = Q6_Vub_vpack_VhVh_sat(h23, h01);
    *(HVX_Vector*)(out + 128 * q) = clamp_lo ? Q6_Vub_vmax_VubVub(yb, lo) : yb;
    HVX_Vector f = Q6_V_vmux_QVV(fl, one, zero);
    flags[q] = f;
    any = Q6_V_vor_VV(any, f);
  }
#undef QC_ROW
  /* one vector -> scalar round trip per tile (a store followed by a scalar load stalls); groups are rescanned
   * only when some output of the tile is near a .5 boundary. flags: cached (stack) memory, not VTCM. */
  flags[16] = qc_ror_or(any);
  if (!*(volatile int32_t*)&flags[16]) {
    return 0;
  }
  int nfix = 0;
  int zyv = h->zy, lov = h->lo;
  for (int q = 0; q < 16; q++) {
    flags[17] = qc_ror_or(flags[q]);
    if (*(volatile int32_t*)&flags[17]) qc_fix_group(planes, out, b, zyv, lov, q), nfix++;
  }
  return nfix;
}

/* One 1x1 QDQ conv layer: X (mt row blocks x kt croutons) -> Y (mt x N/32 tiles), W packed by hmx_pack_w_u8cm,
 * blk[N/32] (256-byte aligned) + hdr in VTCM. scratch (QC_EXACT): 4 x 2 KB byte planes, VTCM, 2 KB
 * aligned (an HMX tile store drops the low 11 address bits). Returns the number of scalar-fixed 4-row groups. */
static inline int qc_conv1x1(const uint8_t* X, uint8_t* Y, const uint8_t* W, const qc_blk_t* blk, const qc_hdr_t* h,
                             int mt, int kt, int mode, uint8_t* scratch) {
  int nt = h->n / 32, nfix = 0;
  for (int g = 0; g < h->n / 64; g++)
    for (int mb = 0; mb < mt; mb++) {
      hmx_blk_mac_u8cm_deep(X + (size_t)mb * kt * 2048, W + (size_t)g * kt * 2048, kt);
      for (int hh = 0; hh < 2; hh++) {
        const qc_blk_t* b = &blk[2 * g + hh];
        uint8_t* yt = Y + ((size_t)mb * nt + 2 * g + hh) * 2048;
        if (mode == QC_FAST) {
          hmx_blk_set_table2(b->tbl_fast);
          hmx_blk_store_u8cm(yt);
        } else {
          for (int p = 0; p < 4; p++) {
            hmx_blk_set_table2(b->tbl_plane[p]);
            if (p < 3)
              __asm__ volatile("mxmem(%0,%1):after:retain:cm.ub = acc" ::"r"(scratch + 2048 * p), "r"(0) : "memory");
            else
              __asm__ volatile("mxmem(%0,%1):after:cm.ub = acc" ::"r"(scratch + 2048 * p), "r"(0) : "memory");
          }
          nfix += qc_requant_tile(scratch, yt, b, h);
        }
      }
    }
  if (mode == QC_FAST && h->relu && h->lo > 0) { /* the sat.ub clamp is at 0; a Relu with zy > 0 clamps at zy */
    HVX_Vector lo = Q6_Vb_vsplat_R(h->lo);
    for (size_t i = 0; i < (size_t)mt * nt * 2048; i += 128) *(HVX_Vector*)(Y + i) = Q6_Vub_vmax_VubVub(*(HVX_Vector*)(Y + i), lo);
  }
  return nfix;
}
#endif /* __hexagon__ */
#endif
