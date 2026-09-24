/* HVX versions of mcc_block.h's element-wise steps, on the HMX tile layout (V69: qfloat arithmetic only).
 *
 * A tile's vector p (128 B, 64 halfwords) holds rows 2p and 2p+1 interleaved: lane 2j = (2p, j), lane
 * 2j+1 = (2p+1, j). The widening hf x hf -> qf32 multiply splits exactly that way (lo = even lanes = row
 * 2p, hi = odd lanes = row 2p+1; checked on hexagon-sim, sim/qf_probe.c), and the narrowing qf32 pair
 * -> hf re-interleaves, so per-row statistics are plain 32-lane reductions and a per-row scalar becomes
 * an interleaved vector with one narrowing. qf32 -> hf rounds to nearest, ties away from zero (sim/
 * cvt_probe.c); qf16 -> hf can be one ulp off, so the residual stream and LayerNorm go through qf32 and
 * only exp / GELU use qf16. */
#ifndef MB_HVX_H
#define MB_HVX_H
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "../hmx_gemm/hmx_gemm.h" /* hmx_copy_hvx (l2fetch-prefetched DDR -> VTCM copy) */

typedef HVX_Vector mbV;
typedef HVX_VectorPair mbW;
#define MBV(p) (*(mbV*)(p))

static inline mbV mbv_hsplat(float f) { return Q6_Vh_vsplat_R(mb_f2h(f)); }
static inline mbV mbv_ssplat(float f) {
  union { float f; int i; } u = {f};
  return Q6_V_vsplat_R(u.i);
}
#define MBV_HMUL(a, b) Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf((a), (b)))
#define MBV_HADD(a, b) Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf((a), (b)))
#define MBV_HSUB(a, b) Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf((a), (b)))
#define MBV_SF(q) Q6_Vsf_equals_Vqf32(q)
#define MBV_QF(s) Q6_Vqf32_vadd_VsfVsf((s), Q6_V_vzero())

static inline mbW mbv_widen(mbV x) { return Q6_Wqf32_vmpy_VhfVhf(x, Q6_Vh_vsplat_R(0x3C00)); }
static inline mbV mbv_narrow(mbV hi, mbV lo) { return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo)); }
/* 32-lane sum of a qf32 vector, in every lane (cyclic butterfly) */
static inline mbV mbv_lanesum(mbV a) {
  for (int s = 64; s >= 4; s >>= 1) a = Q6_Vqf32_vadd_Vqf32Vqf32(a, Q6_V_vror_VR(a, s));
  return a;
}
/* max over the 32 same-parity lanes of an hf vector (= per row of the row pair), in every lane */
static inline mbV mbv_rowmax(mbV a) {
  for (int s = 64; s >= 4; s >>= 1) a = Q6_Vhf_vmax_VhfVhf(a, Q6_V_vror_VR(a, s));
  return a;
}
/* 1 / x, sf, 3 Newton steps from the bit-trick estimate */
static inline mbV mbv_srecip(mbV x) {
  mbV y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C7), x), two = mbv_ssplat(2.f);
  for (int k = 0; k < 3; k++)
    y = MBV_SF(Q6_Vqf32_vmpy_VsfVsf(y, MBV_SF(Q6_Vqf32_vsub_VsfVsf(two, MBV_SF(Q6_Vqf32_vmpy_VsfVsf(x, y))))));
  return y;
}
/* qf16 -> hf. V69 qf16 arithmetic is loose (sim/qf16_ops.c: an exact product 0.0078125 * 128 comes back
 * 1.00098, 2 - d r two ulps low, biased the same way), so chained qf16 compounds: a Newton reciprocal
 * stalls near 1e-3 and a 5-step polynomial reaches ~1%. Here every qf16 op is converted back to hf
 * (one ulp each, measured fine end to end), and whatever needs more -- per-row sums, 1 / sum, the self
 * score -- runs in qf32. */
#define MBV_Q2H(q) Q6_Vhf_equals_Vqf16(q)

/* 2^t for hf t in [-13, 13] (callers clamp), N vectors in lockstep (independent chains interleave):
 * n = round(t) through the 1536.0 magic number (hf bits 0x6600 + n), g = n - t (in [-0.5, 1] even
 * when the qf16 rounding is one ulp off), 2^-g by a degree-4 Horner polynomial (hf after every op,
 * <= 1.3e-3 relative at g = 1), n added to the exponent field. In place. */
#define MB_EXPN 4
static inline void mbv_hexp2_n(mbV* t) {
  const mbV magic = Q6_Vh_vsplat_R(0x6600), c4 = mbv_hsplat(0.00961813f), c3 = mbv_hsplat(-0.0555041f), c2 = mbv_hsplat(0.2402265f),
            c1 = mbv_hsplat(-0.6931472f), c0 = mbv_hsplat(1.f);
  mbV n[MB_EXPN], g[MB_EXPN], p[MB_EXPN];
  for (int i = 0; i < MB_EXPN; i++) {
    mbV r = MBV_HADD(t[i], magic);
    n[i] = Q6_Vh_vsub_VhVh(r, magic);
    g[i] = MBV_HSUB(MBV_HSUB(r, magic), t[i]);
  }
  for (int i = 0; i < MB_EXPN; i++) p[i] = MBV_HADD(MBV_HMUL(g[i], c4), c3);
  for (int i = 0; i < MB_EXPN; i++) p[i] = MBV_HADD(MBV_HMUL(p[i], g[i]), c2);
  for (int i = 0; i < MB_EXPN; i++) p[i] = MBV_HADD(MBV_HMUL(p[i], g[i]), c1);
  for (int i = 0; i < MB_EXPN; i++) p[i] = MBV_HADD(MBV_HMUL(p[i], g[i]), c0);
  for (int i = 0; i < MB_EXPN; i++) t[i] = Q6_Vh_vadd_VhVh(p[i], Q6_Vh_vasl_VhR(n[i], 10));
}
static inline mbV mbv_hexp2(mbV t) {
  mbV v[MB_EXPN] = {t, t, t, t};
  mbv_hexp2_n(v);
  return v[0];
}
/* 1 / d for hf d in [2^-14, 2^14] (result normal fp16): hf bit trick (0x7800 - bits) + `it` Newton steps in qf16 */
static inline mbV mbv_hrecip(mbV d, int it) {
  const mbV two = mbv_hsplat(2.f);
  mbV r = Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(0x7800), d);
  for (int k = 0; k < it; k++) r = MBV_Q2H(Q6_Vqf16_vmpy_VhfVhf(MBV_Q2H(Q6_Vqf16_vsub_VhfVhf(two, MBV_Q2H(Q6_Vqf16_vmpy_VhfVhf(d, r)))), r));
  return r;
}
/* hf sum over the 32 same-parity lanes (= per row of the row pair), in every lane, via qf16 */
static inline mbV mbv_rowsum(mbV a) {
  mbV q = Q6_Vqf16_vadd_VhfVhf(a, Q6_V_vzero());
  for (int s = 64; s >= 4; s >>= 1) q = Q6_Vqf16_vadd_Vqf16Vqf16(q, Q6_V_vror_VR(q, s));
  return MBV_Q2H(q);
}

/* GELU in place (tanh form x * sigmoid(1.5957691 (x + 0.044715 x^3)), |error vs erf| < ~1e-3):
 * sigmoid = 1 / (1 + 2^t), t = -1.5957691 log2(e) x (1 + 0.044715 x^2) clamped to [-12, 12]; the
 * reciprocal of 1 + 2^t in [1, 4097] is the hf bit trick + 2 Newton steps; x < -4 -> 0 (|gelu| < 1.3e-4). */
static inline void mbv_tile_gelu(uint8_t* dst) {
  const mbV c3 = mbv_hsplat(0.044715f), one = mbv_hsplat(1.f), k = mbv_hsplat(-2.3022082f), lo = mbv_hsplat(-12.f), hi = mbv_hsplat(12.f),
            cut = mbv_hsplat(-4.f), z = Q6_V_vzero();
  for (int i = 0; i < 16; i++) {
    mbV x = MBV(dst + 128 * i);
    mbV a = MBV_Q2H(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(x, x), c3), one));
    mbV t = MBV_Q2H(Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(a, x), k));
    t = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(t, lo), hi);
    mbV d = MBV_Q2H(Q6_Vqf16_vadd_VhfVhf(mbv_hexp2(t), one));
    MBV(dst + 128 * i) = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, cut), MBV_Q2H(Q6_Vqf16_vmpy_VhfVhf(x, mbv_hrecip(d, 2))), z);
  }
}

/* H = LayerNorm(X) over 512 columns per row, eps 1e-6: per row pair, qf32 sums of x and x^2 per row
 * (lo / hi halves), one-pass variance, rstd on the scalar core, then (x - mean) * (rstd * gamma) + beta
 * in qf32 per half. */
static inline void mbv_layernorm(uint8_t* x, uint8_t* h, int rt, const mb_hf* ln, int tid, int nthr) {
  float gb[2][MB_KT][32] __attribute__((aligned(128)));
  for (int j = 0; j < MB_D; j++) gb[0][j / 32][j % 32] = mb_h2f(ln[j]), gb[1][j / 32][j % 32] = mb_h2f(ln[MB_D + j]);
  float st[4][32] __attribute__((aligned(128)));
  const mbV one = Q6_Vh_vsplat_R(0x3C00);
  for (int rb = tid; rb < rt; rb += nthr)
    for (int p = 0; p < 16; p++) {
      mbV s0 = Q6_V_vzero(), s1 = s0, q0 = s0, q1 = s0;
      for (int kb = 0; kb < MB_KT; kb++) {
        mbV v = MBV(x + ((size_t)rb * MB_KT + kb) * MB_TB + 128 * p);
        mbW w = Q6_Wqf32_vmpy_VhfVhf(v, one), w2 = Q6_Wqf32_vmpy_VhfVhf(v, v);
        s0 = Q6_Vqf32_vadd_Vqf32Vqf32(s0, Q6_V_lo_W(w)), s1 = Q6_Vqf32_vadd_Vqf32Vqf32(s1, Q6_V_hi_W(w));
        q0 = Q6_Vqf32_vadd_Vqf32Vqf32(q0, Q6_V_lo_W(w2)), q1 = Q6_Vqf32_vadd_Vqf32Vqf32(q1, Q6_V_hi_W(w2));
      }
      MBV(st[0]) = MBV_SF(mbv_lanesum(s0)), MBV(st[1]) = MBV_SF(mbv_lanesum(s1));
      MBV(st[2]) = MBV_SF(mbv_lanesum(q0)), MBV(st[3]) = MBV_SF(mbv_lanesum(q1));
      float mean[2], rs[2];
      for (int r = 0; r < 2; r++) {
        mean[r] = st[r][0] / MB_D;
        float var = st[2 + r][0] / MB_D - mean[r] * mean[r];
        rs[r] = 1.f / sqrtf((var > 0 ? var : 0) + 1e-6f);
      }
      const mbV m0 = mbv_ssplat(mean[0]), m1 = mbv_ssplat(mean[1]), r0 = mbv_ssplat(rs[0]), r1 = mbv_ssplat(rs[1]);
      for (int kb = 0; kb < MB_KT; kb++) {
        uint8_t* xo = x + ((size_t)rb * MB_KT + kb) * MB_TB + 128 * p;
        mbW w = mbv_widen(MBV(xo));
        const mbV g = MBV(gb[0][kb]), b = MBV(gb[1][kb]);
        mbV y0 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vsub_Vqf32Vsf(Q6_V_lo_W(w), m0), Q6_Vqf32_vmpy_VsfVsf(g, r0)), b);
        mbV y1 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vsub_Vqf32Vsf(Q6_V_hi_W(w), m1), Q6_Vqf32_vmpy_VsfVsf(g, r1)), b);
        MBV(h + ((size_t)rb * MB_KT + kb) * MB_TB + 128 * p) = mbv_narrow(y1, y0);
      }
    }
}

/* Base-2 softmax of one head, in place: S (row block stride 8 tiles, 7 used; scores already in log2
 * units; the padded columns 197..223 hold -65504 from the S table) plus the self score s_self = q . k *
 * scale2 per row; P = e / sum written back into S, p_self as an interleaved hf vector per row pair into
 * pv (row block rb, vector p at pv + rb * 2048 + 128 p). e = 2^(t), t = s - (max - 7) clamped at -13
 * (so e <= 128; the clamp floor, 2^-20 of the max, stands in for 0: padded columns and anything that
 * small), exp in hf, 4 vectors in lockstep, e parked in S between the passes; the row sums, their
 * reciprocal and the self score in qf32 (lo / hi halves = the two rows); P = e * (128 / sum) / 128. */
static inline void mbv_softmax(uint8_t* s, const uint8_t* q, const uint8_t* k, uint8_t* pv, int rt, int tid, int nthr) {
  const mbV seven = mbv_hsplat(7.f), lo = mbv_hsplat(-13.f), k2m7 = mbv_hsplat(1.f / 128.f), sc = mbv_ssplat(MB_SCALE2),
            k128 = mbv_ssplat(128.f), one = Q6_Vh_vsplat_R(0x3C00);
  for (int rb = tid; rb < rt; rb += nthr)
    for (int p = 0; p < 16; p++) {
      uint8_t* sp[MB_ST + 1];
      for (int cb = 0; cb < MB_ST; cb++) sp[cb] = s + ((size_t)rb * 8 + cb) * MB_TB + 128 * p;
      sp[MB_ST] = pv + (size_t)rb * MB_TB + 128 * p; /* the self term's e / p lives in pv */
      mbW qk = Q6_Wqf32_vmpy_VhfVhf(MBV(q + (size_t)rb * MB_TB + 128 * p), MBV(k + (size_t)rb * MB_TB + 128 * p));
      mbV ss = mbv_narrow(Q6_Vqf32_vmpy_VsfVsf(MBV_SF(mbv_lanesum(Q6_V_hi_W(qk))), sc), Q6_Vqf32_vmpy_VsfVsf(MBV_SF(mbv_lanesum(Q6_V_lo_W(qk))), sc));
      mbV m = MBV(sp[0]);
      for (int cb = 1; cb < MB_ST; cb++) m = Q6_Vhf_vmax_VhfVhf(m, MBV(sp[cb]));
      const mbV m7 = MBV_HSUB(Q6_Vhf_vmax_VhfVhf(mbv_rowmax(m), ss), seven);
      MBV(sp[MB_ST]) = ss;
      mbV a0 = Q6_V_vzero(), a1 = a0;
      for (int c0 = 0; c0 <= MB_ST; c0 += MB_EXPN) {
        mbV t[MB_EXPN];
        for (int i = 0; i < MB_EXPN; i++) t[i] = Q6_Vhf_vmax_VhfVhf(MBV_HSUB(MBV(sp[c0 + i]), m7), lo);
        mbv_hexp2_n(t);
        for (int i = 0; i < MB_EXPN; i++) {
          MBV(sp[c0 + i]) = t[i];
          if (c0 + i < MB_ST) {
            mbW w = Q6_Wqf32_vmpy_VhfVhf(t[i], one);
            a0 = Q6_Vqf32_vadd_Vqf32Vqf32(a0, Q6_V_lo_W(w)), a1 = Q6_Vqf32_vadd_Vqf32Vqf32(a1, Q6_V_hi_W(w));
          }
        }
      }
      mbW es = Q6_Wqf32_vmpy_VhfVhf(MBV(sp[MB_ST]), one); /* the self term: once per row (it is in all 32 lanes) */
      mbV t0 = Q6_Vqf32_vadd_Vqf32Vqf32(mbv_lanesum(a0), Q6_V_lo_W(es)), t1 = Q6_Vqf32_vadd_Vqf32Vqf32(mbv_lanesum(a1), Q6_V_hi_W(es));
      const mbV inv = mbv_narrow(Q6_Vqf32_vmpy_VsfVsf(mbv_srecip(MBV_SF(t1)), k128), Q6_Vqf32_vmpy_VsfVsf(mbv_srecip(MBV_SF(t0)), k128));
      for (int cb = 0; cb <= MB_ST; cb++) MBV(sp[cb]) = MBV_HMUL(MBV_HMUL(MBV(sp[cb]), inv), k2m7);
    }
}

/* O[:, head h] += p_self * v_h (qf16) */
static inline void mbv_self_term(uint8_t* o, const uint8_t* v, const uint8_t* pv, int rt, int h) {
  for (int rb = 0; rb < rt; rb++)
    for (int p = 0; p < 16; p++) {
      uint8_t* op = o + ((size_t)rb * MB_KT + h) * MB_TB + 128 * p;
      MBV(op) = Q6_Vhf_equals_Vqf16(
          Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(MBV(pv + (size_t)rb * MB_TB + 128 * p), MBV(v + (size_t)rb * MB_TB + 128 * p)), MBV(op)));
    }
}
#endif
