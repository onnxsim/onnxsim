/* V65-ISA HVX (128B) int8 kernels for openpilot's conv nets, plus a hexagon-sim harness.
 *
 * Built with upstream clang (-mcpu=hexagonv65 -mhvx=v65: the Qualcomm 19.x toolchain no longer
 * accepts v65), linked and run on hexagon-sim (which only models v68+; V65 code runs unchanged there,
 * so cycle counts are a V68-pipeline proxy for the 845's V65 cDSP -- see README).
 *
 * Kernels (all activations uint8 with a zero point, weights int8 symmetric per output channel, int32
 * bias, Q31 multiplier + right shift requantization, the QDQ contract onnxsim's quantize_full_qdq emits):
 *   pw   1x1 conv / GEMM: act [P/32][K/4][32][4] u8, out [P/32][N/4][32][4] u8 (same layout, so layers
 *        chain without re-layout). vrmpy(Vu.ub = 32 pixels x 4 input channels, Rt.b = 4 weights of one
 *        output channel): 128 MACs per instruction, 4 output channels per activation load.
 *   pw16 the same with uint16 activations (W8A16 layers): two vrmpy passes over the low and high
 *        activation bytes (u16 = hi*256 + lo), combined at requantization: exactly 2x the MACs.
 *   dw   depthwise kxk conv (k = 3, 7), stride 1 or 2, planar [C][H][W] u8, 32-bit accumulation from
 *        16-bit widened pixels (vmpyacc Vh x Rt.h); out planar u8.
 *   gelu uint8 -> uint8 lookup table (the op is elementwise on quantized values), 8 x vlut32.
 *
 * Usage: kernels <op> <args...> [check]
 *   pw   P K N (8 output channels per pass when N % 8 == 0; "pw4" forces the 4-channel kernel)
 *   pw16 P K N    dw C H W k s (planar)    dwc C H W k s (channels-last)    gelu n
 *   gemv M K N [a16] [prefetch]  (heads: int8 weights streamed from DDR, u8 or u16 activations)
 *   dw3 C H W s           (3x3 only: weights in registers, sliding input window)
 *   dwc4 C H W k s        (channels-last, vrmpy over taps, taps shuffled once per input row)
 *   dwc3 C H W k s [pix]  (channels-last, vrmpy over 4 taps; pix = 1|2)
 *   dwc2 C H W k s [pix]  (channels-last, per-layer shift, narrowing requant; pix = 2|4 per iteration)
 * Prints cycles (upcycle counter around the timed call) and, with "check", compares against a scalar
 * C reference of the same fixed-point spec (bit exact expected).
 */
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VLEN 128
typedef HVX_Vector V;

static void enable_cycle_counter(void) {
  uint32_t ssr;
  __asm__ __volatile__("%0 = ssr" : "=r"(ssr));
  ssr |= 1u << 23; /* SSR.CE: user cycle counter enable (standalone sim runs in supervisor mode) */
  __asm__ __volatile__("ssr = %0; isync" : : "r"(ssr));
}

static inline uint64_t cycles(void) {
  uint64_t c;
  __asm__ __volatile__("%0 = c15:14" : "=r"(c)); /* UPCYCLE */
  return c;
}

static uint32_t rng_state = 12345;
static inline uint32_t rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static void *amalloc(size_t n) {
  void *p = NULL;
  if (posix_memalign(&p, VLEN, (n + VLEN - 1) / VLEN * VLEN)) abort();
  return p;
}

/* ---- requantization: out = clamp(((acc * mult + 2^30) >> 31 + round) >> shift + zp, 0, 255) ---- */
static inline int32_t q31_mul_ref(int32_t a, int32_t m) {
  int64_t p = (int64_t)a * m;
  int64_t r = (p + (1ll << 30)) >> 31;
  if (r > INT32_MAX) r = INT32_MAX;
  if (r < INT32_MIN) r = INT32_MIN;
  return (int32_t)r;
}
static inline int32_t rshift_rnd_ref(int32_t x, int s) { return s ? (int32_t)(((int64_t)x + (1 << (s - 1))) >> s) : x; }
static inline int32_t clampi(int32_t x, int32_t lo, int32_t hi) { return x < lo ? lo : x > hi ? hi : x; }

/* HVX Q31 multiply: vmpye(Vu.w, Vv.uh) + vmpyo(Vu.w, Vv.h):<<1:rnd:sat:shift == (Vu*Vv + 2^30) >> 31, sat */
static inline V q31_mul(V a, V m) {
  V t = Q6_Vw_vmpye_VwVuh(a, m);
  return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(t, a, m);
}
static inline V requant(V acc, int32_t mult, int shift, int zp, int hi) {
  V x = q31_mul(acc, Q6_V_vsplat_R(mult));
  if (shift) x = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(x, Q6_V_vsplat_R(1 << (shift - 1))), shift);
  x = Q6_Vw_vadd_VwVw(x, Q6_V_vsplat_R(zp));
  x = Q6_Vw_vmax_VwVw(x, Q6_V_vzero());
  return Q6_Vw_vmin_VwVw(x, Q6_V_vsplat_R(hi));
}

/* ---- pointwise (1x1) conv, u8 activations ---- */
typedef struct {
  int P, K, N;
  const uint8_t *act;   /* [P/32][K/4][32][4] */
  const int32_t *w;     /* [N][K/4], 4 int8 per word (input channels 4k..4k+3, little endian) */
  const int32_t *bias;  /* [N] */
  const int32_t *mult;  /* [N] */
  const int8_t *shift;  /* [N] */
  int zp_in, zp_out;
  uint8_t *out;         /* [P/32][N/4][32][4] */
} pw_args;

/* bias must already fold -zp_in * sum(w) (done once, offline, like every int8 runtime) */
__attribute__((noinline)) void pw_u8(const pw_args *a) {
  const int K4 = a->K / 4, N4 = a->N / 4;
  for (int pb = 0; pb < a->P / 32; pb++) {
    const V *act = (const V *)(a->act + (size_t)pb * K4 * VLEN);
    V *out = (V *)(a->out + (size_t)pb * N4 * VLEN);
    for (int n4 = 0; n4 < N4; n4++) {
      const int32_t *w0 = a->w + (size_t)(4 * n4) * K4, *w1 = w0 + K4, *w2 = w1 + K4, *w3 = w2 + K4;
      V c0 = Q6_V_vsplat_R(a->bias[4 * n4]), c1 = Q6_V_vsplat_R(a->bias[4 * n4 + 1]);
      V c2 = Q6_V_vsplat_R(a->bias[4 * n4 + 2]), c3 = Q6_V_vsplat_R(a->bias[4 * n4 + 3]);
#pragma unroll 4
      for (int k = 0; k < K4; k++) {
        V x = act[k];
        c0 = Q6_Vw_vrmpyacc_VwVubRb(c0, x, w0[k]);
        c1 = Q6_Vw_vrmpyacc_VwVubRb(c1, x, w1[k]);
        c2 = Q6_Vw_vrmpyacc_VwVubRb(c2, x, w2[k]);
        c3 = Q6_Vw_vrmpyacc_VwVubRb(c3, x, w3[k]);
      }
      int n = 4 * n4;
      V r0 = requant(c0, a->mult[n], a->shift[n], a->zp_out, 255);
      V r1 = requant(c1, a->mult[n + 1], a->shift[n + 1], a->zp_out, 255);
      V r2 = requant(c2, a->mult[n + 2], a->shift[n + 2], a->zp_out, 255);
      V r3 = requant(c3, a->mult[n + 3], a->shift[n + 3], a->zp_out, 255);
      /* pixel i's 4 channels -> bytes 4i..4i+3: the input layout of the next layer */
      V o = Q6_V_vor_VV(r0, Q6_Vw_vasl_VwR(r1, 8));
      o = Q6_V_vor_VV(o, Q6_Vw_vasl_VwR(r2, 16));
      out[n4] = Q6_V_vor_VV(o, Q6_Vw_vasl_VwR(r3, 24));
    }
  }
}

/* 8 output channels per activation load, weights repacked [N/8][K/4][8] words so the 8 weights of one
 * k step are one contiguous 32-byte run (4 memd). 8 accumulators: each register is updated every 4
 * packets, hiding the vrmpy accumulate latency that the 4-channel version stalls on. */
__attribute__((noinline)) void pw_u8_n8(const pw_args *a, const int32_t *w8) {
  const int K4 = a->K / 4, N4 = a->N / 4;
  for (int pb = 0; pb < a->P / 32; pb++) {
    const V *act = (const V *)(a->act + (size_t)pb * K4 * VLEN);
    V *out = (V *)(a->out + (size_t)pb * N4 * VLEN);
    for (int n8 = 0; n8 < a->N / 8; n8++) {
      const int64_t *w = (const int64_t *)(w8 + (size_t)n8 * K4 * 8);
      const int32_t *bias = a->bias + 8 * n8;
      V c0 = Q6_V_vsplat_R(bias[0]), c1 = Q6_V_vsplat_R(bias[1]), c2 = Q6_V_vsplat_R(bias[2]), c3 = Q6_V_vsplat_R(bias[3]);
      V c4 = Q6_V_vsplat_R(bias[4]), c5 = Q6_V_vsplat_R(bias[5]), c6 = Q6_V_vsplat_R(bias[6]), c7 = Q6_V_vsplat_R(bias[7]);
#pragma unroll 2
      for (int k = 0; k < K4; k++) {
        V x = act[k];
        int64_t w01 = w[4 * k], w23 = w[4 * k + 1], w45 = w[4 * k + 2], w67 = w[4 * k + 3];
        c0 = Q6_Vw_vrmpyacc_VwVubRb(c0, x, (int32_t)w01);
        c1 = Q6_Vw_vrmpyacc_VwVubRb(c1, x, (int32_t)(w01 >> 32));
        c2 = Q6_Vw_vrmpyacc_VwVubRb(c2, x, (int32_t)w23);
        c3 = Q6_Vw_vrmpyacc_VwVubRb(c3, x, (int32_t)(w23 >> 32));
        c4 = Q6_Vw_vrmpyacc_VwVubRb(c4, x, (int32_t)w45);
        c5 = Q6_Vw_vrmpyacc_VwVubRb(c5, x, (int32_t)(w45 >> 32));
        c6 = Q6_Vw_vrmpyacc_VwVubRb(c6, x, (int32_t)w67);
        c7 = Q6_Vw_vrmpyacc_VwVubRb(c7, x, (int32_t)(w67 >> 32));
      }
      const int n = 8 * n8;
      V C[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
      for (int h = 0; h < 2; h++) {
        V r0 = requant(C[4 * h], a->mult[n + 4 * h], a->shift[n + 4 * h], a->zp_out, 255);
        V r1 = requant(C[4 * h + 1], a->mult[n + 4 * h + 1], a->shift[n + 4 * h + 1], a->zp_out, 255);
        V r2 = requant(C[4 * h + 2], a->mult[n + 4 * h + 2], a->shift[n + 4 * h + 2], a->zp_out, 255);
        V r3 = requant(C[4 * h + 3], a->mult[n + 4 * h + 3], a->shift[n + 4 * h + 3], a->zp_out, 255);
        V o = Q6_V_vor_VV(Q6_V_vor_VV(r0, Q6_Vw_vasl_VwR(r1, 8)), Q6_V_vor_VV(Q6_Vw_vasl_VwR(r2, 16), Q6_Vw_vasl_VwR(r3, 24)));
        out[2 * n8 + h] = o;
      }
    }
  }
}

/* uint16 activations as two byte planes: lo [P/32][K/4][32][4] and hi (same layout). acc = hi*256 + lo.
 * Requantization: q31(acc_hi, mult) * 256 + q31(acc_lo, mult) (each int32-safe for any K), then the
 * same shift/zp/clamp into uint16 [0, 65535], stored as two byte planes again. */
typedef struct {
  pw_args b;
  const uint8_t *act_hi;
  uint8_t *out_hi;
} pw16_args;

__attribute__((noinline)) void pw_u16(const pw16_args *a16) {
  const pw_args *a = &a16->b;
  const int K4 = a->K / 4, N4 = a->N / 4;
  for (int pb = 0; pb < a->P / 32; pb++) {
    const V *al = (const V *)(a->act + (size_t)pb * K4 * VLEN);
    const V *ah = (const V *)(a16->act_hi + (size_t)pb * K4 * VLEN);
    V *ol = (V *)(a->out + (size_t)pb * N4 * VLEN), *oh = (V *)(a16->out_hi + (size_t)pb * N4 * VLEN);
    for (int n4 = 0; n4 < N4; n4++) {
      const int32_t *w0 = a->w + (size_t)(4 * n4) * K4, *w1 = w0 + K4, *w2 = w1 + K4, *w3 = w2 + K4;
      V l0 = Q6_V_vzero(), l1 = l0, l2 = l0, l3 = l0, h0 = l0, h1 = l0, h2 = l0, h3 = l0;
#pragma unroll 2
      for (int k = 0; k < K4; k++) {
        V x = al[k], y = ah[k];
        l0 = Q6_Vw_vrmpyacc_VwVubRb(l0, x, w0[k]);
        l1 = Q6_Vw_vrmpyacc_VwVubRb(l1, x, w1[k]);
        l2 = Q6_Vw_vrmpyacc_VwVubRb(l2, x, w2[k]);
        l3 = Q6_Vw_vrmpyacc_VwVubRb(l3, x, w3[k]);
        h0 = Q6_Vw_vrmpyacc_VwVubRb(h0, y, w0[k]);
        h1 = Q6_Vw_vrmpyacc_VwVubRb(h1, y, w1[k]);
        h2 = Q6_Vw_vrmpyacc_VwVubRb(h2, y, w2[k]);
        h3 = Q6_Vw_vrmpyacc_VwVubRb(h3, y, w3[k]);
      }
      V L[4] = {l0, l1, l2, l3}, H[4] = {h0, h1, h2, h3}, R[4];
      for (int j = 0; j < 4; j++) {
        int n = 4 * n4 + j;
        V m = Q6_V_vsplat_R(a->mult[n]);
        /* bias (already zp-folded, in lo units) goes with the lo part */
        V lo = Q6_Vw_vadd_VwVw(L[j], Q6_V_vsplat_R(a->bias[n]));
        V x = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(q31_mul(H[j], m), 8), q31_mul(lo, m));
        int s = a->shift[n];
        if (s) x = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(x, Q6_V_vsplat_R(1 << (s - 1))), s);
        x = Q6_Vw_vadd_VwVw(x, Q6_V_vsplat_R(a->zp_out));
        R[j] = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(x, Q6_V_vzero()), Q6_V_vsplat_R(65535));
      }
      V ff = Q6_V_vsplat_R(0xff);
      V o = Q6_V_vand_VV(R[0], ff), p = Q6_Vuw_vlsr_VuwR(R[0], 8);
      for (int j = 1; j < 4; j++) {
        o = Q6_V_vor_VV(o, Q6_Vw_vasl_VwR(Q6_V_vand_VV(R[j], ff), 8 * j));
        p = Q6_V_vor_VV(p, Q6_Vw_vasl_VwR(Q6_Vuw_vlsr_VuwR(R[j], 8), 8 * j));
      }
      ol[n4] = o;
      oh[n4] = p;
    }
  }
}

/* ---- depthwise kxk, stride s, pad k/2, planar u8. W and output width padded to multiples of 64 ---- */
typedef struct {
  int C, H, W, k, s, Ho, Wo, Wp;  /* Wp: padded input row pitch (>= W + k - 1, multiple of 128), input pre-padded */
  const uint8_t *in;              /* [C][H + k - 1][Wp], zero-point-padded borders */
  const int8_t *w;                /* [C][k*k] */
  const int32_t *bias, *mult;     /* [C], bias zp-folded */
  const int8_t *shift;
  int zp_out;
  uint8_t *out;                   /* [C][Ho][Wop], Wop = roundup(Wo, 128) */
  int Wop;
} dw_args;

/* stride 1: out[y][x..x+127] = sum_{dy,dx} w * in[y+dy][x+dx..]; pixels widened to u16 (vzxt), products
 * accumulated in 32 bits with vmpyacc(Ww, Vh, Rt.h) -- 64 lanes per instruction; 2 per 128 outputs per tap.
 * stride 2: the same over the even output columns (inputs deinterleaved by vdeal on the fly). */
__attribute__((noinline)) void dw_u8(const dw_args *a) {
  const int k = a->k, Hp = a->H + k - 1;
  for (int c = 0; c < a->C; c++) {
    const uint8_t *in = a->in + (size_t)c * Hp * a->Wp;
    const int8_t *wc = a->w + c * k * k;
    V bias = Q6_V_vsplat_R(a->bias[c]);
    for (int y = 0; y < a->Ho; y++) {
      for (int x = 0; x < a->Wo; x += 128) {
        HVX_VectorPair acc_lo = Q6_W_vcombine_VV(bias, bias), acc_hi = acc_lo;
        for (int dy = 0; dy < k; dy++) {
          const uint8_t *row = in + (size_t)(y * a->s + dy) * a->Wp + x * a->s;
          for (int dx = 0; dx < k; dx++) {
            V px;
            if (a->s == 1) {
              px = *(const HVX_UVector *)(row + dx);
            } else { /* even bytes of 256 consecutive pixels */
              V p0 = *(const HVX_UVector *)(row + dx), p1 = *(const HVX_UVector *)(row + dx + 128);
              px = Q6_Vb_vpacke_VhVh(p1, p0);
            }
            HVX_VectorPair wide = Q6_Wuh_vzxt_Vub(px); /* lo: even pixels, hi: odd pixels (vzxt interleaves) */
            int wt = (uint16_t)(int16_t)wc[dy * k + dx]; /* Rt.h: even lanes use the low half, odd the high */
            acc_lo = Q6_Ww_vmpyacc_WwVhRh(acc_lo, Q6_V_lo_W(wide), wt | (wt << 16));
            acc_hi = Q6_Ww_vmpyacc_WwVhRh(acc_hi, Q6_V_hi_W(wide), wt | (wt << 16));
          }
        }
        /* vzxt splits pixels even/odd, vmpy(Vh, Rt.h) splits each again even/odd: word i of
         * acc_lo.lo / acc_hi.lo / acc_lo.hi / acc_hi.hi is pixel 4i / 4i+1 / 4i+2 / 4i+3. Requantize the
         * four, then byte-pack pixel 4i+j into byte j of word i: 128 u8 outputs in pixel order. */
        V q0 = requant(Q6_V_lo_W(acc_lo), a->mult[c], a->shift[c], a->zp_out, 255);
        V q1 = requant(Q6_V_lo_W(acc_hi), a->mult[c], a->shift[c], a->zp_out, 255);
        V q2 = requant(Q6_V_hi_W(acc_lo), a->mult[c], a->shift[c], a->zp_out, 255);
        V q3 = requant(Q6_V_hi_W(acc_hi), a->mult[c], a->shift[c], a->zp_out, 255);
        V b = Q6_V_vor_VV(Q6_V_vor_VV(q0, Q6_Vw_vasl_VwR(q1, 8)), Q6_V_vor_VV(Q6_Vw_vasl_VwR(q2, 16), Q6_Vw_vasl_VwR(q3, 24)));
        *(V *)(a->out + ((size_t)c * a->Ho + y) * a->Wop + x) = b;
      }
    }
  }
}

/* ---- depthwise, channels-last: 128 channels per vector, any spatial size (the planar dw_u8 above wastes
 * lanes when W < 128 and serializes on one accumulator; this one is the one the projection uses) ----
 * in [Hp][Wp][Cp] u8 (pre-padded by k/2 with the input zero point, Cp = roundup(C, 128)),
 * w16 [Cp/128][k*k][2] vectors of int16 (even channels, odd channels: vzxt's split),
 * per-channel requant vectors M/S/B [Cp/128][4] (word i of vector j = channel 4i + j),
 * out [Ho][Wo][Cp] u8. Two output pixels per iteration: four independent accumulator pairs. */
typedef struct {
  int C, Hp, Wp, k, s, Ho, Wo;
  const uint8_t *in;
  const V *w16, *M, *S, *B;
  int zp_out;
  uint8_t *out;
} dwc_args;

static inline V requant_v(V acc, V m, V sh, V bias_zp_unused, int zp) {
  (void)bias_zp_unused;
  V x = q31_mul(acc, m);
  /* round: + (1 << (sh - 1)) for sh > 0 (all shifts here are >= 1) */
  V one = Q6_V_vsplat_R(1);
  x = Q6_Vw_vadd_VwVw(x, Q6_Vw_vasl_VwVw(one, Q6_Vw_vsub_VwVw(sh, one)));
  x = Q6_Vw_vasr_VwVw(x, sh);
  x = Q6_Vw_vadd_VwVw(x, Q6_V_vsplat_R(zp));
  return Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(x, Q6_V_vzero()), Q6_V_vsplat_R(255));
}

__attribute__((noinline)) void dwc_u8(const dwc_args *a) {
  const int k = a->k, kk = k * k, Cv = a->C / 128;
  const size_t rowb = (size_t)a->Wp * a->C;
  for (int cv = 0; cv < Cv; cv++) {
    const V *w = a->w16 + (size_t)cv * kk * 2;
    const V *M = a->M + cv * 4, *S = a->S + cv * 4, *B = a->B + cv * 4;
    for (int y = 0; y < a->Ho; y++) {
      for (int x = 0; x < a->Wo; x += 2) {
        const int two = x + 1 < a->Wo;
        HVX_VectorPair e0 = Q6_W_vcombine_VV(B[2], B[0]), o0 = Q6_W_vcombine_VV(B[3], B[1]);
        HVX_VectorPair e1 = e0, o1 = o0;
        const uint8_t *base0 = a->in + (size_t)(y * a->s) * rowb + (size_t)(x * a->s) * a->C + cv * 128;
        const uint8_t *base1 = base0 + (size_t)a->s * a->C;
        for (int dy = 0; dy < k; dy++) {
          for (int dx = 0; dx < k; dx++) {
            size_t off = (size_t)dy * rowb + (size_t)dx * a->C;
            V we = w[2 * (dy * k + dx)], wo = w[2 * (dy * k + dx) + 1];
            HVX_VectorPair x0 = Q6_Wuh_vzxt_Vub(*(const V *)(base0 + off));
            e0 = Q6_Ww_vmpyacc_WwVhVh(e0, Q6_V_lo_W(x0), we);
            o0 = Q6_Ww_vmpyacc_WwVhVh(o0, Q6_V_hi_W(x0), wo);
            HVX_VectorPair x1 = Q6_Wuh_vzxt_Vub(*(const V *)(base1 + off));
            e1 = Q6_Ww_vmpyacc_WwVhVh(e1, Q6_V_lo_W(x1), we);
            o1 = Q6_Ww_vmpyacc_WwVhVh(o1, Q6_V_hi_W(x1), wo);
          }
        }
        /* e.lo: channels 4i, e.hi: 4i+2, o.lo: 4i+1, o.hi: 4i+3 */
        for (int p = 0; p < 1 + two; p++) {
          HVX_VectorPair e = p ? e1 : e0, o = p ? o1 : o0;
          V q0 = requant_v(Q6_V_lo_W(e), M[0], S[0], B[0], a->zp_out);
          V q1 = requant_v(Q6_V_lo_W(o), M[1], S[1], B[1], a->zp_out);
          V q2 = requant_v(Q6_V_hi_W(e), M[2], S[2], B[2], a->zp_out);
          V q3 = requant_v(Q6_V_hi_W(o), M[3], S[3], B[3], a->zp_out);
          V b = Q6_V_vor_VV(Q6_V_vor_VV(q0, Q6_Vw_vasl_VwR(q1, 8)), Q6_V_vor_VV(Q6_Vw_vasl_VwR(q2, 16), Q6_Vw_vasl_VwR(q3, 24)));
          *(V *)(a->out + ((size_t)y * a->Wo + x + p) * a->C + cv * 128) = b;
        }
      }
    }
  }
}

/* ---- depthwise v2: the MAC loop of dwc_u8, with a requantization that costs ~12 instead of ~45 ops
 * per 128 outputs. Spec change (the quantizer can always produce it): per-channel Q31 multipliers but
 * one right shift per layer, so the shift + round + saturate is the narrowing
 * vasr(Vu.w, Vv.w, Rt):rnd:sat (two word vectors -> one halfword vector), then + zp and a saturating
 * narrow to bytes. The even/odd split of vzxt x vmpy (e.lo = ch 4i, o.lo = 4i+1, e.hi = 4i+2,
 * o.hi = 4i+3) is undone by the two narrowing steps' own interleave, so bytes come out in channel
 * order with no shuffle. out = clamp(sat16(rnd(q31(acc, M_c) >> s)) + zp, 0, 255). */
typedef struct {
  int C, Hp, Wp, k, s, Ho, Wo;
  const uint8_t *in;
  const V *w16, *M, *B; /* as dwc_args */
  int shift, zp_out;
  uint8_t *out;
  int pix; /* output pixels per inner iteration: 2 or 4 */
} dwc2_args;

static inline V narrow_requant(HVX_VectorPair e, HVX_VectorPair o, const V *M, int shift, V zp) {
  V q0 = q31_mul(Q6_V_lo_W(e), M[0]), q1 = q31_mul(Q6_V_lo_W(o), M[1]);
  V q2 = q31_mul(Q6_V_hi_W(e), M[2]), q3 = q31_mul(Q6_V_hi_W(o), M[3]);
  V a = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vasr_VwVwR_rnd_sat(q2, q0, shift), zp); /* h: 4i, 4i+2 */
  V b = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vasr_VwVwR_rnd_sat(q3, q1, shift), zp); /* h: 4i+1, 4i+3 */
  return Q6_Vub_vasr_VhVhR_sat(b, a, 0);
}

__attribute__((noinline)) void dwc2_u8(const dwc2_args *a) {
  const int k = a->k, kk = k * k, Cv = a->C / 128;
  const size_t rowb = (size_t)a->Wp * a->C, colb = (size_t)a->s * a->C;
  const V zp = Q6_Vh_vsplat_R(a->zp_out);
  for (int cv = 0; cv < Cv; cv++) {
    const V *w = a->w16 + (size_t)cv * kk * 2;
    const V *M = a->M + cv * 4, *B = a->B + cv * 4;
    const HVX_VectorPair be = Q6_W_vcombine_VV(B[2], B[0]), bo = Q6_W_vcombine_VV(B[3], B[1]);
    for (int y = 0; y < a->Ho; y++) {
      const uint8_t *row = a->in + (size_t)(y * a->s) * rowb + cv * 128;
      uint8_t *orow = a->out + (size_t)y * a->Wo * a->C + cv * 128;
      int x = 0;
      if (a->pix == 4) {
        for (; x + 4 <= a->Wo; x += 4) {
          HVX_VectorPair e0 = be, o0 = bo, e1 = be, o1 = bo, e2 = be, o2 = bo, e3 = be, o3 = bo;
          const uint8_t *p = row + (size_t)x * colb;
          for (int dy = 0; dy < k; dy++) {
            const uint8_t *pr = p + (size_t)dy * rowb;
            const V *wr = w + 2 * dy * k;
            for (int dx = 0; dx < k; dx++) {
              V we = wr[2 * dx], wo = wr[2 * dx + 1];
              const uint8_t *pt = pr + (size_t)dx * a->C;
              HVX_VectorPair x0 = Q6_Wuh_vzxt_Vub(*(const V *)pt);
              HVX_VectorPair x1 = Q6_Wuh_vzxt_Vub(*(const V *)(pt + colb));
              HVX_VectorPair x2 = Q6_Wuh_vzxt_Vub(*(const V *)(pt + 2 * colb));
              HVX_VectorPair x3 = Q6_Wuh_vzxt_Vub(*(const V *)(pt + 3 * colb));
              e0 = Q6_Ww_vmpyacc_WwVhVh(e0, Q6_V_lo_W(x0), we);
              o0 = Q6_Ww_vmpyacc_WwVhVh(o0, Q6_V_hi_W(x0), wo);
              e1 = Q6_Ww_vmpyacc_WwVhVh(e1, Q6_V_lo_W(x1), we);
              o1 = Q6_Ww_vmpyacc_WwVhVh(o1, Q6_V_hi_W(x1), wo);
              e2 = Q6_Ww_vmpyacc_WwVhVh(e2, Q6_V_lo_W(x2), we);
              o2 = Q6_Ww_vmpyacc_WwVhVh(o2, Q6_V_hi_W(x2), wo);
              e3 = Q6_Ww_vmpyacc_WwVhVh(e3, Q6_V_lo_W(x3), we);
              o3 = Q6_Ww_vmpyacc_WwVhVh(o3, Q6_V_hi_W(x3), wo);
            }
          }
          V *o = (V *)(orow + (size_t)x * a->C);
          o[0] = narrow_requant(e0, o0, M, a->shift, zp);
          o[Cv] = narrow_requant(e1, o1, M, a->shift, zp);
          o[2 * Cv] = narrow_requant(e2, o2, M, a->shift, zp);
          o[3 * Cv] = narrow_requant(e3, o3, M, a->shift, zp);
        }
      }
      for (; x < a->Wo; x++) {
        HVX_VectorPair e0 = be, o0 = bo;
        const uint8_t *p = row + (size_t)x * colb;
        for (int dy = 0; dy < k; dy++)
          for (int dx = 0; dx < k; dx++) {
            HVX_VectorPair x0 = Q6_Wuh_vzxt_Vub(*(const V *)(p + (size_t)dy * rowb + (size_t)dx * a->C));
            e0 = Q6_Ww_vmpyacc_WwVhVh(e0, Q6_V_lo_W(x0), w[2 * (dy * k + dx)]);
            o0 = Q6_Ww_vmpyacc_WwVhVh(o0, Q6_V_hi_W(x0), w[2 * (dy * k + dx) + 1]);
          }
        *(V *)(orow + (size_t)x * a->C) = narrow_requant(e0, o0, M, a->shift, zp);
      }
    }
  }
}

/* ---- depthwise v3: vrmpy over the kernel taps (u8 x s8 -> s32, 4 taps per word, no widening) ----
 * dwc/dwc2 widen pixels to 16 bits and use vmpy(Vh, Vh), a double-resource instruction: at most 64 MACs
 * per packet, and the compiler reaches a fraction of that. Here the taps become vrmpy's 4-way reduction:
 * for each group of 4 taps (the k*k taps flattened, the last group zero-padded), the 4 input vectors
 * (128 channels each) are byte- then halfword-shuffled so word i of vector g holds channel 32g+i's four
 * tap pixels, and vrmpy(Vu.ub, Vv.b) against the same-shuffled weights (prepared offline) accumulates
 * 32 channels x 4 taps. vrmpy (vector x vector) is a single-resource instruction, so 2 issue per packet.
 * Requantization: per-channel Q31 multiplier, one shift per layer (as dwc2); the two narrowing steps
 * leave byte 4i+j = channel 32j+i, which two vdeal.b undo. */
typedef struct {
  int C, Hp, Wp, k, s, Ho, Wo;
  const uint8_t *in;      /* [Hp][Wp][C] */
  const V *w4;            /* [C/128][groups][4] shuffled int8 weights */
  const V *M, *B;         /* [C/128][4] natural channel order (vector g = channels 32g..32g+31) */
  const int32_t *tap_off; /* [groups*4] byte offsets of each tap from the output pixel's window origin */
  int groups, shift, zp_out;
  uint8_t *out;
  int pix;  /* 1 or 2 output pixels per iteration */
  int c64;  /* C = 64, stride 1: 2 pixels per vector (lane b = channel b % 64), unaligned tap loads */
} dwc3_args;

static inline void shuf4(V t0, V t1, V t2, V t3, V *x) {
  HVX_VectorPair s01 = Q6_W_vshuff_VVR(t1, t0, -1), s23 = Q6_W_vshuff_VVR(t3, t2, -1);
  HVX_VectorPair lo = Q6_W_vshuff_VVR(Q6_V_lo_W(s23), Q6_V_lo_W(s01), -2);
  HVX_VectorPair hi = Q6_W_vshuff_VVR(Q6_V_hi_W(s23), Q6_V_hi_W(s01), -2);
  x[0] = Q6_V_lo_W(lo), x[1] = Q6_V_hi_W(lo), x[2] = Q6_V_lo_W(hi), x[3] = Q6_V_hi_W(hi);
}

static inline V requant4(V a0, V a1, V a2, V a3, const V *M, int shift, V zp) {
  V q0 = q31_mul(a0, M[0]), q1 = q31_mul(a1, M[1]), q2 = q31_mul(a2, M[2]), q3 = q31_mul(a3, M[3]);
  V a = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vasr_VwVwR_rnd_sat(q2, q0, shift), zp);
  V b = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vasr_VwVwR_rnd_sat(q3, q1, shift), zp);
  V u = Q6_Vub_vasr_VhVhR_sat(b, a, 0); /* byte 4i+j = channel 32j+i */
  return Q6_Vb_vdeal_Vb(Q6_Vb_vdeal_Vb(u));
}

static inline __attribute__((always_inline)) void dwc3_body(const dwc3_args *a, const int c64) {
  const int Cv = c64 ? 1 : a->C / 128, G = a->groups, pixb = c64 ? 64 : a->C;
  const size_t rowb = (size_t)a->Wp * pixb, colb = (size_t)a->s * pixb;
  const int step = c64 ? 2 : 1; /* output pixels per vector */
  const V zp = Q6_Vh_vsplat_R(a->zp_out);
#define LD(p) (c64 ? *(const HVX_UVector *)(p) : *(const V *)(p))
  for (int cv = 0; cv < Cv; cv++) {
    const V *w = a->w4 + (size_t)cv * G * 4;
    const V *M = a->M + cv * 4, *B = a->B + cv * 4;
    for (int y = 0; y < a->Ho; y++) {
      const uint8_t *row = a->in + (size_t)(y * a->s) * rowb + cv * 128;
      uint8_t *orow = a->out + (size_t)y * a->Wo * pixb + cv * 128;
      int x = 0;
      if (a->pix == 2) {
        for (; x + 2 * step <= a->Wo; x += 2 * step) {
          const uint8_t *p0 = row + (size_t)x * colb, *p1 = p0 + step * colb;
          V a0 = B[0], a1 = B[1], a2 = B[2], a3 = B[3], b0 = B[0], b1 = B[1], b2 = B[2], b3 = B[3];
          for (int g = 0; g < G; g++) {
            const int32_t *o = a->tap_off + 4 * g;
            const V *wg = w + 4 * g;
            V xa[4], xb[4];
            shuf4(LD(p0 + o[0]), LD(p0 + o[1]), LD(p0 + o[2]), LD(p0 + o[3]), xa);
            shuf4(LD(p1 + o[0]), LD(p1 + o[1]), LD(p1 + o[2]), LD(p1 + o[3]), xb);
            V w0 = wg[0], w1 = wg[1], w2 = wg[2], w3 = wg[3];
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, xa[0], w0);
            a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, xa[1], w1);
            a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, xa[2], w2);
            a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, xa[3], w3);
            b0 = Q6_Vw_vrmpyacc_VwVubVb(b0, xb[0], w0);
            b1 = Q6_Vw_vrmpyacc_VwVubVb(b1, xb[1], w1);
            b2 = Q6_Vw_vrmpyacc_VwVubVb(b2, xb[2], w2);
            b3 = Q6_Vw_vrmpyacc_VwVubVb(b3, xb[3], w3);
          }
          *(V *)(orow + (size_t)x * pixb) = requant4(a0, a1, a2, a3, M, a->shift, zp);
          *(V *)(orow + (size_t)(x + step) * pixb) = requant4(b0, b1, b2, b3, M, a->shift, zp);
        }
      }
      for (; x < a->Wo; x += step) {
        const uint8_t *p0 = row + (size_t)x * colb;
        V a0 = B[0], a1 = B[1], a2 = B[2], a3 = B[3];
        for (int g = 0; g < G; g++) {
          const int32_t *o = a->tap_off + 4 * g;
          const V *wg = w + 4 * g;
          V xa[4];
          shuf4(LD(p0 + o[0]), LD(p0 + o[1]), LD(p0 + o[2]), LD(p0 + o[3]), xa);
          a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, xa[0], wg[0]);
          a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, xa[1], wg[1]);
          a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, xa[2], wg[2]);
          a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, xa[3], wg[3]);
        }
        *(HVX_UVector *)(orow + (size_t)x * pixb) = requant4(a0, a1, a2, a3, M, a->shift, zp);
      }
    }
  }
#undef LD
}

__attribute__((noinline)) void dwc3_u8(const dwc3_args *a) {
  if (a->c64)
    dwc3_body(a, 1);
  else
    dwc3_body(a, 0);
}

/* ---- depthwise v4: dwc3's vrmpy-over-taps, with the tap shuffle done once per input row ----
 * dwc3 re-shuffles every tap window for every output pixel and reloads 4 input vectors per 4 taps per
 * pixel: with one vector load per packet, loads bound it. Here each input row r is transformed once into
 * Q_r(p) = shuf4(in[r][p], in[r][p+1], in[r][p+2], in[r][p+3]) (4 vectors: word i of vector v = channel
 * 32v+i's pixels p..p+3) in a ring buffer of k+s rows; an output pixel then needs, per kernel row dy and
 * per group of 4 horizontal taps, 4 loads of Q and 4 vrmpy. The shuffle is amortized over the k output
 * rows that read each input row (stride 1). Requantization as dwc3. */
typedef struct {
  int C, Hp, Wp, k, s, Ho, Wo;
  const uint8_t *in; /* [Hp][Wp][C], Wp >= (Wo-1)*s + 4*gx + 4 */
  const V *w4;       /* [C/128][k][gx][4]: row dy, horizontal group g, shuffled like Q */
  const V *M, *B;
  int shift, zp_out;
  uint8_t *out;
  V *ring; /* scratch: (k + s) * Wq * 4 vectors, Wq = Wp - 3 */
} dwc4_args;

__attribute__((noinline)) void dwc4_u8(const dwc4_args *a) {
  const int k = a->k, s = a->s, gx = (k + 3) / 4, Cv = a->C / 128, R = k + s, Wq = a->Wp - 3;
  const V zp = Q6_Vh_vsplat_R(a->zp_out);
  for (int cv = 0; cv < Cv; cv++) {
    const V *w = a->w4 + (size_t)cv * k * gx * 4;
    const V *M = a->M + cv * 4, *B = a->B + cv * 4;
    int built = 0; /* rows [0, built) have been transformed */
    for (int y = 0; y < a->Ho; y++) {
      for (; built < y * s + k; built++) {
        const uint8_t *row = a->in + (size_t)built * a->Wp * a->C + cv * 128;
        V *q = a->ring + (size_t)(built % R) * Wq * 4;
        for (int p = 0; p < Wq; p += (s == 2 ? 2 : 1)) {
          const uint8_t *pp = row + (size_t)p * a->C;
          shuf4(*(const V *)pp, *(const V *)(pp + a->C), *(const V *)(pp + 2 * a->C), *(const V *)(pp + 3 * a->C), q + 4 * p);
        }
      }
      uint8_t *orow = a->out + (size_t)y * a->Wo * a->C + cv * 128;
      int x = 0;
      for (; x + 2 <= a->Wo; x += 2) {
        V a0 = B[0], a1 = B[1], a2 = B[2], a3 = B[3], b0 = B[0], b1 = B[1], b2 = B[2], b3 = B[3];
        for (int dy = 0; dy < k; dy++) {
          const V *q = a->ring + (size_t)((y * s + dy) % R) * Wq * 4;
          for (int g = 0; g < gx; g++) {
            const V *wg = w + ((size_t)dy * gx + g) * 4;
            const V *qa = q + 4 * (x * s + 4 * g), *qb = qa + 4 * s;
            V w0 = wg[0], w1 = wg[1], w2 = wg[2], w3 = wg[3];
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, qa[0], w0);
            a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, qa[1], w1);
            a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, qa[2], w2);
            a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, qa[3], w3);
            b0 = Q6_Vw_vrmpyacc_VwVubVb(b0, qb[0], w0);
            b1 = Q6_Vw_vrmpyacc_VwVubVb(b1, qb[1], w1);
            b2 = Q6_Vw_vrmpyacc_VwVubVb(b2, qb[2], w2);
            b3 = Q6_Vw_vrmpyacc_VwVubVb(b3, qb[3], w3);
          }
        }
        *(V *)(orow + (size_t)x * a->C) = requant4(a0, a1, a2, a3, M, a->shift, zp);
        *(V *)(orow + (size_t)(x + 1) * a->C) = requant4(b0, b1, b2, b3, M, a->shift, zp);
      }
      for (; x < a->Wo; x++) {
        V a0 = B[0], a1 = B[1], a2 = B[2], a3 = B[3];
        for (int dy = 0; dy < k; dy++) {
          const V *q = a->ring + (size_t)((y * s + dy) % R) * Wq * 4;
          for (int g = 0; g < gx; g++) {
            const V *wg = w + ((size_t)dy * gx + g) * 4, *qa = q + 4 * (x * s + 4 * g);
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, qa[0], wg[0]);
            a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, qa[1], wg[1]);
            a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, qa[2], wg[2]);
            a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, qa[3], wg[3]);
          }
        }
        *(V *)(orow + (size_t)x * a->C) = requant4(a0, a1, a2, a3, M, a->shift, zp);
      }
    }
  }
}

/* ---- depthwise 3x3 v5: dwc3's vrmpy-over-taps with everything the sim says is expensive removed ----
 * hexagon-sim sustains ~1 HVX vector load per 2 cycles from L2 (measured: 2.0-2.2 cycles/vector,
 * 16 KB-1 MB), so the kernel is load-bound, not multiply-bound. For k = 3 the 12 weight vectors (3 rows
 * x 4 shuffled vectors) stay in registers for a whole output row, and a 3x3 input window slides along x:
 * stride 1 loads 3 new vectors per output pixel (1 per kernel row), stride 2 loads 6. Each kernel row is
 * one shuf4 (taps dx = 0, 1, 2 and a dummy 4th with zero weight) and 4 vrmpy. */
typedef struct {
  int C, Hp, Wp, s, Ho, Wo;
  const uint8_t *in; /* [Hp][Wp][C], pre-padded */
  const V *w4;       /* [C/128][3][4]: kernel row dy, shuffled like shuf4 (4th tap 0) */
  const V *M, *B;
  int shift, zp_out;
  uint8_t *out;
} dw3_args;

/* C = 64, stride 1: a vector holds 2 pixels x 64 channels (channels-last memory as is), so no lane is
 * padding. Output pixels 2x, 2x+1 need input vectors starting at pixels 2x, 2x+1, 2x+2 per kernel row:
 * the aligned V(2x), V(2x+2) and V(2x+1) = valign(V(2x+2), V(2x), 64). Weights/M/B vectors hold each
 * channel's value in both 64-byte halves (lane b = channel b % 64). */
__attribute__((noinline)) void dw3_c64_u8(const dw3_args *a) {
  const size_t rowb = (size_t)a->Wp * 64;
  const V *w = a->w4;
  const V w00 = w[0], w01 = w[1], w02 = w[2], w03 = w[3], w10 = w[4], w11 = w[5], w12 = w[6], w13 = w[7];
  const V w20 = w[8], w21 = w[9], w22 = w[10], w23 = w[11];
  const V zp = Q6_Vh_vsplat_R(a->zp_out);
  for (int y = 0; y < a->Ho; y++) {
    const uint8_t *r0 = a->in + (size_t)y * rowb, *r1 = r0 + rowb, *r2 = r1 + rowb;
    uint8_t *orow = a->out + (size_t)y * a->Wo * 64;
    V p0 = *(const V *)r0, p1 = *(const V *)r1, p2 = *(const V *)r2;
    for (int x = 0; x < a->Wo; x += 2) {
      const size_t o = (size_t)(x + 2) * 64;
      V n0 = *(const V *)(r0 + o), n1 = *(const V *)(r1 + o), n2 = *(const V *)(r2 + o);
      V a0 = a->B[0], a1 = a->B[1], a2 = a->B[2], a3 = a->B[3], t[4];
      shuf4(p0, Q6_V_valign_VVR(n0, p0, 64), n0, n0, t);
      a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, t[0], w00);
      a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, t[1], w01);
      a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, t[2], w02);
      a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, t[3], w03);
      shuf4(p1, Q6_V_valign_VVR(n1, p1, 64), n1, n1, t);
      a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, t[0], w10);
      a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, t[1], w11);
      a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, t[2], w12);
      a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, t[3], w13);
      shuf4(p2, Q6_V_valign_VVR(n2, p2, 64), n2, n2, t);
      a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, t[0], w20);
      a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, t[1], w21);
      a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, t[2], w22);
      a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, t[3], w23);
      *(V *)(orow + (size_t)x * 64) = requant4(a0, a1, a2, a3, a->M, a->shift, zp);
      p0 = n0, p1 = n1, p2 = n2;
    }
  }
}

__attribute__((noinline)) void dw3_u8(const dw3_args *a) {
  const int Cv = a->C / 128, s = a->s;
  const size_t rowb = (size_t)a->Wp * a->C, C = a->C;
  const V zp = Q6_Vh_vsplat_R(a->zp_out);
  for (int cv = 0; cv < Cv; cv++) {
    const V *w = a->w4 + (size_t)cv * 12;
    const V w00 = w[0], w01 = w[1], w02 = w[2], w03 = w[3], w10 = w[4], w11 = w[5], w12 = w[6], w13 = w[7];
    const V w20 = w[8], w21 = w[9], w22 = w[10], w23 = w[11];
    const V *M = a->M + cv * 4, *B = a->B + cv * 4;
    for (int y = 0; y < a->Ho; y++) {
      const uint8_t *r0 = a->in + (size_t)(y * s) * rowb + cv * 128, *r1 = r0 + rowb, *r2 = r1 + rowb;
      uint8_t *orow = a->out + (size_t)y * a->Wo * C + cv * 128;
      /* window columns 0, 1 of each row (column 2 is loaded per pixel) */
      V p00 = *(const V *)r0, p01 = *(const V *)(r0 + C);
      V p10 = *(const V *)r1, p11 = *(const V *)(r1 + C);
      V p20 = *(const V *)r2, p21 = *(const V *)(r2 + C);
      for (int x = 0; x < a->Wo; x++) {
        const size_t o = (size_t)(x * s + 2) * C;
        V p02 = *(const V *)(r0 + o), p12 = *(const V *)(r1 + o), p22 = *(const V *)(r2 + o);
        V a0 = B[0], a1 = B[1], a2 = B[2], a3 = B[3], t[4];
        shuf4(p00, p01, p02, p02, t);
        a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, t[0], w00);
        a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, t[1], w01);
        a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, t[2], w02);
        a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, t[3], w03);
        shuf4(p10, p11, p12, p12, t);
        a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, t[0], w10);
        a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, t[1], w11);
        a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, t[2], w12);
        a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, t[3], w13);
        shuf4(p20, p21, p22, p22, t);
        a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, t[0], w20);
        a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, t[1], w21);
        a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, t[2], w22);
        a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, t[3], w23);
        *(V *)(orow + (size_t)x * C) = requant4(a0, a1, a2, a3, M, a->shift, zp);
        if (s == 1) {
          p00 = p01, p01 = p02, p10 = p11, p11 = p12, p20 = p21, p21 = p22;
        } else {
          const size_t n = (size_t)(x * s + 3) * C;
          p00 = p02, p01 = *(const V *)(r0 + n), p10 = p12, p11 = *(const V *)(r1 + n), p20 = p22,
          p21 = *(const V *)(r2 + n);
        }
      }
    }
  }
}

/* ---- GEMV / skinny GEMM for the driving heads: y[M][N] = x[M][K] . W[K][N], M <= 9 ----
 * Weights int8, prepacked [N/32][K/4][32][4] (one 128-byte vector = 32 outputs x 4 k), streamed once;
 * activations uint8 (a16: uint16 as lo/hi byte planes, two vrmpy per weight vector), each 4-byte k-group
 * splatted across a vector: vrmpy(Vu.ub = splat(x[m][4k..4k+3]), Vv.b = W) -> 32 int32 partial sums.
 * The weight stream (23.5 MB for the whole driving head) does not fit L2, so the next 32-output block
 * is l2fetch'ed while the current one computes. Output: raw int32 accumulators (the heads' requant /
 * float epilogue is not modeled). */
static inline void l2fetch_box(const void *p, unsigned stride, unsigned width, unsigned height) {
  unsigned long long ctl = ((unsigned long long)stride << 32) | ((unsigned long long)width << 16) | height;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(ctl));
}

/* MM output rows at a time, compile-time, so the accumulators stay in registers */
static inline __attribute__((always_inline)) void gemv_rows(const uint8_t *x, const uint8_t *xhi, int K4, int N, const V *w,
                                                            int32_t *y, const int MM, const int a16, int prefetch) {
  const unsigned blk = (unsigned)K4 * 128;
  for (int n = 0; n < N / 32; n++) {
    const V *wb = w + (size_t)n * K4;
    if (prefetch && n + 1 < N / 32) {
      const uint8_t *nx = (const uint8_t *)(wb + K4);
      if (blk <= 32768)
        l2fetch_box(nx, blk, blk, 1);
      else
        l2fetch_box(nx, 32768, 32768, blk / 32768);
    }
    V acc[9], acch[9];
    for (int m = 0; m < MM; m++) acc[m] = acch[m] = Q6_V_vzero();
    const int32_t *xw = (const int32_t *)x, *xhw = (const int32_t *)xhi;
#pragma unroll 2
    for (int k = 0; k < K4; k++) {
      V wv = wb[k];
      for (int m = 0; m < MM; m++) {
        acc[m] = Q6_Vw_vrmpyacc_VwVubVb(acc[m], Q6_V_vsplat_R(xw[m * K4 + k]), wv);
        if (a16) acch[m] = Q6_Vw_vrmpyacc_VwVubVb(acch[m], Q6_V_vsplat_R(xhw[m * K4 + k]), wv);
      }
    }
    for (int m = 0; m < MM; m++) {
      V r = a16 ? Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(acch[m], 8), acc[m]) : acc[m];
      *(HVX_UVector *)(y + (size_t)m * N + 32 * n) = r;
    }
  }
}

/* 9 rows (the attention block's tokens) x 2 output blocks per step, u8 activations: each splat feeds two
 * vrmpy, halving the splats that bound the 9-row case (18 accumulators) */
static void gemv9x2_u8(const uint8_t *x, int K4, int N, const V *w, int32_t *y, int prefetch) {
  const unsigned blk = (unsigned)K4 * 128;
  const int32_t *xw = (const int32_t *)x;
  for (int n = 0; n + 2 <= N / 32; n += 2) { /* caller guarantees an even number of 32-output blocks */
    const V *w0 = w + (size_t)n * K4, *w1 = w0 + K4;
    if (prefetch && n + 2 < N / 32) {
      const uint8_t *nx = (const uint8_t *)(w1 + K4);
      if (2 * blk <= 32768)
        l2fetch_box(nx, 2 * blk, 2 * blk, 1);
      else
        l2fetch_box(nx, 32768, 32768, 2 * blk / 32768);
    }
    V a[9], b[9];
    for (int m = 0; m < 9; m++) a[m] = b[m] = Q6_V_vzero();
    for (int k = 0; k < K4; k++) {
      V u = w0[k], v = w1[k];
      for (int m = 0; m < 9; m++) {
        V sx = Q6_V_vsplat_R(xw[m * K4 + k]);
        a[m] = Q6_Vw_vrmpyacc_VwVubVb(a[m], sx, u);
        b[m] = Q6_Vw_vrmpyacc_VwVubVb(b[m], sx, v);
      }
    }
    for (int m = 0; m < 9; m++) {
      *(HVX_UVector *)(y + (size_t)m * N + 32 * n) = a[m];
      *(HVX_UVector *)(y + (size_t)m * N + 32 * (n + 1)) = b[m];
    }
  }
}

__attribute__((noinline)) void gemv_u8(const uint8_t *x, int M, int K, int N, const V *w, int32_t *y, int a16,
                                       const uint8_t *xhi, int prefetch) {
  const int K4 = K / 4;
  /* rows in groups of 9 (the attention block's 9 tokens: one pass over the weights), else one at a time */
  for (int m = 0; m < M;) {
    int r = M - m >= 9 ? 9 : 1;
    const uint8_t *xm = x + (size_t)m * K, *xhm = xhi + (size_t)m * K;
    int32_t *ym = y + (size_t)m * N;
    if (r == 9 && !a16 && (N / 32) % 2 == 0) {
      gemv9x2_u8(xm, K4, N, w, ym, prefetch);
    } else if (r == 9) {
      if (a16)
        gemv_rows(xm, xhm, K4, N, w, ym, 9, 1, prefetch);
      else
        gemv_rows(xm, xhm, K4, N, w, ym, 9, 0, prefetch);
    } else {
      if (a16)
        gemv_rows(xm, xhm, K4, N, w, ym, 1, 1, prefetch);
      else
        gemv_rows(xm, xhm, K4, N, w, ym, 1, 0, prefetch);
    }
    m += r;
  }
}

/* ---- GELU (or any u8 -> u8 map) by table: 8 x vlut32 over the 256-entry table ---- */
__attribute__((noinline)) void lut_u8(const uint8_t *in, uint8_t *out, int n, const V *tbl /* 8 vectors */) {
  for (int i = 0; i < n; i += 128) {
    V x = *(const V *)(in + i);
    V r = Q6_Vb_vlut32_VbVbR(x, tbl[0], 0);
    for (int j = 1; j < 8; j++) r = Q6_Vb_vlut32or_VbVbVbR(r, x, tbl[j], j);
    *(V *)(out + i) = r;
  }
}

/* ======================= harness ======================= */
static void rand_requant(int N, int32_t *mult, int8_t *shift, int32_t *bias, int bias_range) {
  for (int n = 0; n < N; n++) {
    mult[n] = (int32_t)(0x40000000u + (rnd() % 0x3fffffff));
    shift[n] = 6 + rnd() % 5;
    bias[n] = (int32_t)(rnd() % (2 * bias_range)) - bias_range;
  }
}

static int getenv_n4 = 0;
static int warm = 0; /* "warm" as the last-but-check argument: run once untimed first (caches hot) */
static int run_pw(int P, int K, int N, int check, int u16) {
  size_t asz = (size_t)P * K, osz = (size_t)P * N;
  uint8_t *act = amalloc(asz), *act_hi = u16 ? amalloc(asz) : NULL;
  uint8_t *out = amalloc(osz), *out_hi = u16 ? amalloc(osz) : NULL;
  int32_t *w = amalloc((size_t)N * K), *bias = amalloc(4 * N), *mult = amalloc(4 * N);
  int8_t *shift = amalloc(N);
  for (size_t i = 0; i < asz; i++) act[i] = rnd();
  if (u16)
    for (size_t i = 0; i < asz; i++) act_hi[i] = rnd();
  int8_t *wb = (int8_t *)w;
  for (size_t i = 0; i < (size_t)N * K; i++) wb[i] = (int8_t)(rnd() % 255 - 127);
  rand_requant(N, mult, shift, bias, 100000);
  if (!u16)
    for (int n = 0; n < N; n++) shift[n] += 3; /* keep u8 outputs off the rails for a meaningful check */
  pw_args a = {P, K, N, act, w, bias, mult, shift, 0, 7, out};
  pw16_args a16 = {a, act_hi, out_hi};
  if (u16) {
    for (int n = 0; n < N; n++) shift[n] += 8;
  }
  int32_t *w8 = NULL;
  int n8 = !u16 && N % 8 == 0 && !getenv_n4;
  if (n8) {
    w8 = amalloc((size_t)N * K);
    for (int n = 0; n < N; n++)
      for (int k4 = 0; k4 < K / 4; k4++) w8[((size_t)(n / 8) * (K / 4) + k4) * 8 + n % 8] = w[(size_t)n * (K / 4) + k4];
  }
  uint64_t t0 = cycles();
  if (u16)
    pw_u16(&a16);
  else if (n8)
    pw_u8_n8(&a, w8);
  else
    pw_u8(&a);
  uint64_t t1 = cycles();
  double macs = (double)P * K * N * (u16 ? 2 : 1);
  printf("%s P=%d K=%d N=%d cycles=%llu macs_per_cycle=%.1f\n", u16 ? "pw16" : "pw", P, K, N,
         (unsigned long long)(t1 - t0), macs / (double)(t1 - t0));
  if (!check) return 0;
  int bad = 0, pc = P < 64 ? P : 64;
  for (int p = 0; p < pc && bad < 5; p++)
    for (int n = 0; n < N && bad < 5; n++) {
      int pb = p / 32, pi = p % 32;
      int32_t lo = 0, hi = 0;
      for (int k = 0; k < K; k++) {
        size_t ai = ((size_t)pb * (K / 4) + k / 4) * VLEN + pi * 4 + k % 4;
        int8_t wv = wb[(size_t)n * K + k];
        lo += act[ai] * wv;
        if (u16) hi += act_hi[ai] * wv;
      }
      size_t oi = ((size_t)pb * (N / 4) + n / 4) * VLEN + pi * 4 + n % 4;
      int ref, got;
      if (u16) {
        int32_t x = q31_mul_ref(hi, mult[n]) * 256 + q31_mul_ref(lo + bias[n], mult[n]);
        ref = clampi(rshift_rnd_ref(x, shift[n]) + 7, 0, 65535);
        got = out[oi] | (out_hi[oi] << 8);
      } else {
        ref = clampi(rshift_rnd_ref(q31_mul_ref(lo + bias[n], mult[n]), shift[n]) + 7, 0, 255);
        got = out[oi];
      }
      if (ref != got) {
        printf("  MISMATCH p=%d n=%d ref=%d got=%d\n", p, n, ref, got);
        bad++;
      }
    }
  printf("  check %s (%d pixels x %d channels)\n", bad ? "FAIL" : "PASS", pc, N);
  return bad;
}

static int run_dw(int C, int H, int W, int k, int s, int check) {
  int Ho = (H + 2 * (k / 2) - k) / s + 1, Wo = (W + 2 * (k / 2) - k) / s + 1;
  int Wop = (Wo + 127) / 128 * 128;
  int Wp = ((Wop * s + k + 128) + 127) / 128 * 128;
  int Hp = H + k - 1;
  uint8_t *in = amalloc((size_t)C * Hp * Wp), *out = amalloc((size_t)C * Ho * Wop);
  int8_t *w = amalloc(C * k * k), *shift = amalloc(C);
  int32_t *bias = amalloc(4 * C), *mult = amalloc(4 * C);
  for (size_t i = 0; i < (size_t)C * Hp * Wp; i++) in[i] = rnd();
  for (int i = 0; i < C * k * k; i++) w[i] = (int8_t)(rnd() % 255 - 127);
  rand_requant(C, mult, shift, bias, 20000);
  dw_args a = {C, H, W, k, s, Ho, Wo, Wp, in, w, bias, mult, shift, 5, out, Wop};
  uint64_t t0 = cycles();
  dw_u8(&a);
  uint64_t t1 = cycles();
  printf("dw C=%d H=%d W=%d k=%d s=%d cycles=%llu macs_per_cycle=%.1f\n", C, H, W, k, s,
         (unsigned long long)(t1 - t0), (double)C * Ho * Wo * k * k / (double)(t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int c = 0; c < (C < 4 ? C : 4) && bad < 5; c++)
    for (int y = 0; y < Ho && bad < 5; y++)
      for (int x = 0; x < Wo && bad < 5; x++) {
        int32_t acc = bias[c];
        for (int dy = 0; dy < k; dy++)
          for (int dx = 0; dx < k; dx++)
            acc += in[((size_t)c * Hp + y * s + dy) * Wp + x * s + dx] * w[c * k * k + dy * k + dx];
        int ref = clampi(rshift_rnd_ref(q31_mul_ref(acc, mult[c]), shift[c]) + 5, 0, 255);
        int got = out[((size_t)c * Ho + y) * Wop + x];
        if (ref != got) {
          printf("  MISMATCH c=%d y=%d x=%d ref=%d got=%d\n", c, y, x, ref, got);
          bad++;
        }
      }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int run_lut(int n, int check) {
  uint8_t *in = amalloc(n), *out = amalloc(n), table[256];
  V *tbl = amalloc(8 * VLEN);
  for (int i = 0; i < n; i++) in[i] = rnd();
  for (int i = 0; i < 256; i++) table[i] = (uint8_t)(i * 7 + 3);
  /* vlut32 table layout: for segment j, entries j*32..j*32+31 at even byte positions of both halves */
  uint8_t *tb = (uint8_t *)tbl;
  memset(tb, 0, 8 * VLEN);
  for (int j = 0; j < 8; j++)
    for (int e = 0; e < 32; e++) {
      /* vlut32 (V65): index low 5 bits select the entry; element e sits at byte 2*e (+1 mirror) */
      tb[j * VLEN + 2 * e] = table[j * 32 + e];
      tb[j * VLEN + 2 * e + 1] = table[j * 32 + e];
      tb[j * VLEN + 64 + 2 * e] = table[j * 32 + e];
      tb[j * VLEN + 64 + 2 * e + 1] = table[j * 32 + e];
    }
  uint64_t t0 = cycles();
  lut_u8(in, out, n, tbl);
  uint64_t t1 = cycles();
  printf("lut n=%d cycles=%llu bytes_per_cycle=%.1f\n", n, (unsigned long long)(t1 - t0), (double)n / (t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int i = 0; i < n && bad < 5; i++)
    if (out[i] != table[in[i]]) {
      printf("  MISMATCH i=%d in=%d ref=%d got=%d\n", i, in[i], table[in[i]], out[i]);
      bad++;
    }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int run_dwc(int C0, int H, int W, int k, int s, int check) {
  int C = (C0 + 127) / 128 * 128, pad = k / 2;
  int Ho = (H + 2 * pad - k) / s + 1, Wo = (W + 2 * pad - k) / s + 1;
  int Hp = H + 2 * pad + s, Wp = W + 2 * pad + s; /* slack so the odd second pixel never reads past the end */
  uint8_t *in = amalloc((size_t)Hp * Wp * C), *out = amalloc((size_t)Ho * Wo * C);
  int8_t *w = amalloc(C * k * k), *shift = amalloc(C);
  int32_t *bias = amalloc(4 * C), *mult = amalloc(4 * C);
  V *w16 = amalloc((size_t)C / 128 * k * k * 2 * VLEN), *M = amalloc(C / 128 * 4 * VLEN), *S = amalloc(C / 128 * 4 * VLEN),
    *B = amalloc(C / 128 * 4 * VLEN);
  for (size_t i = 0; i < (size_t)Hp * Wp * C; i++) in[i] = rnd();
  for (int i = 0; i < C * k * k; i++) w[i] = (int8_t)(rnd() % 255 - 127);
  rand_requant(C, mult, shift, bias, 20000);
  for (int cv = 0; cv < C / 128; cv++) {
    for (int t = 0; t < k * k; t++) {
      int16_t *we = (int16_t *)&w16[((size_t)cv * k * k + t) * 2], *wo = we + 64;
      for (int i = 0; i < 64; i++) {
        we[i] = w[(cv * 128 + 2 * i) * k * k + t];
        wo[i] = w[(cv * 128 + 2 * i + 1) * k * k + t];
      }
    }
    for (int j = 0; j < 4; j++)
      for (int i = 0; i < 32; i++) {
        int c = cv * 128 + 4 * i + j;
        ((int32_t *)&M[cv * 4 + j])[i] = mult[c];
        ((int32_t *)&S[cv * 4 + j])[i] = shift[c];
        ((int32_t *)&B[cv * 4 + j])[i] = bias[c];
      }
  }
  dwc_args a = {C, Hp, Wp, k, s, Ho, Wo, in, w16, M, S, B, 5, out};
  uint64_t t0 = cycles();
  dwc_u8(&a);
  uint64_t t1 = cycles();
  printf("dwc C=%d(%d) H=%d W=%d k=%d s=%d cycles=%llu macs_per_cycle=%.1f (useful channels)\n", C0, C, H, W, k, s,
         (unsigned long long)(t1 - t0), (double)C0 * Ho * Wo * k * k / (double)(t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int y = 0; y < Ho && bad < 5; y++)
    for (int x = 0; x < Wo && bad < 5; x++)
      for (int c = 0; c < C && bad < 5; c += 7) {
        int32_t acc = bias[c];
        for (int dy = 0; dy < k; dy++)
          for (int dx = 0; dx < k; dx++) acc += in[((size_t)(y * s + dy) * Wp + x * s + dx) * C + c] * w[c * k * k + dy * k + dx];
        int ref = clampi(rshift_rnd_ref(q31_mul_ref(acc, mult[c]), shift[c]) + 5, 0, 255);
        int got = out[((size_t)y * Wo + x) * C + c];
        if (ref != got) {
          printf("  MISMATCH y=%d x=%d c=%d ref=%d got=%d\n", y, x, c, ref, got);
          bad++;
        }
      }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int run_dwc2(int C0, int H, int W, int k, int s, int pix, int check) {
  int C = (C0 + 127) / 128 * 128, pad = k / 2;
  int Ho = (H + 2 * pad - k) / s + 1, Wo = (W + 2 * pad - k) / s + 1;
  int Hp = H + 2 * pad + s, Wp = W + 2 * pad + 4 * s;
  uint8_t *in = amalloc((size_t)Hp * Wp * C), *out = amalloc((size_t)Ho * Wo * C);
  int8_t *w = amalloc(C * k * k);
  int32_t *bias = amalloc(4 * C), *mult = amalloc(4 * C);
  int8_t *dummy = amalloc(C);
  V *w16 = amalloc((size_t)C / 128 * k * k * 2 * VLEN), *M = amalloc(C / 128 * 4 * VLEN), *B = amalloc(C / 128 * 4 * VLEN);
  for (size_t i = 0; i < (size_t)Hp * Wp * C; i++) in[i] = rnd();
  for (int i = 0; i < C * k * k; i++) w[i] = (int8_t)(rnd() % 255 - 127);
  rand_requant(C, mult, dummy, bias, 20000);
  int shift = 9 + rnd() % 4;
  for (int cv = 0; cv < C / 128; cv++) {
    for (int t = 0; t < k * k; t++) {
      int16_t *we = (int16_t *)&w16[((size_t)cv * k * k + t) * 2], *wo = we + 64;
      for (int i = 0; i < 64; i++) {
        we[i] = w[(cv * 128 + 2 * i) * k * k + t];
        wo[i] = w[(cv * 128 + 2 * i + 1) * k * k + t];
      }
    }
    for (int j = 0; j < 4; j++)
      for (int i = 0; i < 32; i++) {
        int c = cv * 128 + 4 * i + j;
        ((int32_t *)&M[cv * 4 + j])[i] = mult[c];
        ((int32_t *)&B[cv * 4 + j])[i] = bias[c];
      }
  }
  dwc2_args a = {C, Hp, Wp, k, s, Ho, Wo, in, w16, M, B, shift, 5, out, pix};
  uint64_t t0 = cycles();
  dwc2_u8(&a);
  uint64_t t1 = cycles();
  printf("dwc2 C=%d(%d) H=%d W=%d k=%d s=%d pix=%d cycles=%llu macs_per_cycle=%.1f (useful channels)\n", C0, C, H, W, k, s, pix,
         (unsigned long long)(t1 - t0), (double)C0 * Ho * Wo * k * k / (double)(t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int y = 0; y < Ho && bad < 5; y++)
    for (int x = 0; x < Wo && bad < 5; x++)
      for (int c = 0; c < C && bad < 5; c += 3) {
        int32_t acc = bias[c];
        for (int dy = 0; dy < k; dy++)
          for (int dx = 0; dx < k; dx++) acc += in[((size_t)(y * s + dy) * Wp + x * s + dx) * C + c] * w[c * k * k + dy * k + dx];
        int ref = clampi(clampi(rshift_rnd_ref(q31_mul_ref(acc, mult[c]), shift), -32768, 32767) + 5, 0, 255);
        int got = out[((size_t)y * Wo + x) * C + c];
        if (ref != got) {
          printf("  MISMATCH y=%d x=%d c=%d ref=%d got=%d\n", y, x, c, ref, got);
          bad++;
        }
      }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int run_dwc3(int C0, int H, int W, int k, int s, int pix, int check) {
  int c64 = C0 == 64 && s == 1;
  int C = c64 ? 64 : (C0 + 127) / 128 * 128, pad = k / 2, kk = k * k, G = (kk + 3) / 4;
  int Ho = (H + 2 * pad - k) / s + 1, Wo = (W + 2 * pad - k) / s + 1;
  int Hp = H + 2 * pad + s, Wp = W + 2 * pad + 2 * s + 2;
  int Cb = c64 ? 128 : C; /* channel lanes per vector block */
  uint8_t *in = amalloc((size_t)Hp * Wp * C), *out = amalloc((size_t)(Ho * Wo + 2) * C);
  int8_t *w = amalloc(C * kk), *dummy = amalloc(C);
  int32_t *bias = amalloc(4 * C), *mult = amalloc(4 * C), *off = amalloc(4 * G * 4);
  V *w4 = amalloc((size_t)Cb / 128 * G * 4 * VLEN), *M = amalloc(Cb / 128 * 4 * VLEN), *B = amalloc(Cb / 128 * 4 * VLEN);
  for (size_t i = 0; i < (size_t)Hp * Wp * C; i++) in[i] = rnd();
  for (int i = 0; i < C * kk; i++) w[i] = (int8_t)(rnd() % 255 - 127);
  rand_requant(C, mult, dummy, bias, 20000);
  int shift = 9 + rnd() % 4;
  for (int t = 0; t < 4 * G; t++) off[t] = t < kk ? ((t / k) * Wp + t % k) * C : 0;
  for (int cv = 0; cv < Cb / 128; cv++) {
    for (int g = 0; g < G; g++)
      for (int v = 0; v < 4; v++) {
        int8_t *wb = (int8_t *)&w4[((size_t)cv * G + g) * 4 + v];
        for (int i = 0; i < 32; i++)
          for (int j = 0; j < 4; j++) {
            int t = 4 * g + j, c = (cv * 128 + 32 * v + i) % C;
            wb[4 * i + j] = t < kk ? w[c * kk + t] : 0;
          }
      }
    for (int v = 0; v < 4; v++)
      for (int i = 0; i < 32; i++) {
        ((int32_t *)&M[cv * 4 + v])[i] = mult[(cv * 128 + 32 * v + i) % C];
        ((int32_t *)&B[cv * 4 + v])[i] = bias[(cv * 128 + 32 * v + i) % C];
      }
  }
  dwc3_args a = {C, Hp, Wp, k, s, Ho, Wo, in, w4, M, B, off, G, shift, 5, out, pix, c64};
  if (warm) dwc3_u8(&a);
  uint64_t t0 = cycles();
  dwc3_u8(&a);
  uint64_t t1 = cycles();
  printf("dwc3 C=%d(%d) H=%d W=%d k=%d s=%d pix=%d cycles=%llu macs_per_cycle=%.1f (useful channels)\n", C0, C, H, W, k, s, pix,
         (unsigned long long)(t1 - t0), (double)C0 * Ho * Wo * kk / (double)(t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int y = 0; y < Ho && bad < 5; y++)
    for (int x = 0; x < Wo && bad < 5; x++)
      for (int c = 0; c < C && bad < 5; c += 3) {
        int32_t acc = bias[c];
        for (int dy = 0; dy < k; dy++)
          for (int dx = 0; dx < k; dx++) acc += in[((size_t)(y * s + dy) * Wp + x * s + dx) * C + c] * w[c * kk + dy * k + dx];
        int ref = clampi(clampi(rshift_rnd_ref(q31_mul_ref(acc, mult[c]), shift), -32768, 32767) + 5, 0, 255);
        int got = out[((size_t)y * Wo + x) * C + c];
        if (ref != got) {
          printf("  MISMATCH y=%d x=%d c=%d ref=%d got=%d\n", y, x, c, ref, got);
          bad++;
        }
      }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int run_dwc4(int C0, int H, int W, int k, int s, int check) {
  int C = (C0 + 127) / 128 * 128, pad = k / 2, kk = k * k, gx = (k + 3) / 4;
  int Ho = (H + 2 * pad - k) / s + 1, Wo = (W + 2 * pad - k) / s + 1;
  int Hp = H + 2 * pad + s, Wp = (Wo - 1) * s + 4 * gx + 4 + s;
  uint8_t *in = amalloc((size_t)Hp * Wp * C), *out = amalloc((size_t)Ho * Wo * C);
  int8_t *w = amalloc(C * kk), *dummy = amalloc(C);
  int32_t *bias = amalloc(4 * C), *mult = amalloc(4 * C);
  V *w4 = amalloc((size_t)C / 128 * k * gx * 4 * VLEN), *M = amalloc(C / 128 * 4 * VLEN), *B = amalloc(C / 128 * 4 * VLEN);
  V *ring = amalloc((size_t)(k + s) * (Wp - 3) * 4 * VLEN);
  for (size_t i = 0; i < (size_t)Hp * Wp * C; i++) in[i] = rnd();
  for (int i = 0; i < C * kk; i++) w[i] = (int8_t)(rnd() % 255 - 127);
  rand_requant(C, mult, dummy, bias, 20000);
  int shift = 9 + rnd() % 4;
  for (int cv = 0; cv < C / 128; cv++) {
    for (int dy = 0; dy < k; dy++)
      for (int g = 0; g < gx; g++)
        for (int v = 0; v < 4; v++) {
          int8_t *wb = (int8_t *)&w4[(((size_t)cv * k + dy) * gx + g) * 4 + v];
          for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++) {
              int dx = 4 * g + j, c = cv * 128 + 32 * v + i;
              wb[4 * i + j] = dx < k ? w[c * kk + dy * k + dx] : 0;
            }
        }
    for (int v = 0; v < 4; v++)
      for (int i = 0; i < 32; i++) {
        ((int32_t *)&M[cv * 4 + v])[i] = mult[cv * 128 + 32 * v + i];
        ((int32_t *)&B[cv * 4 + v])[i] = bias[cv * 128 + 32 * v + i];
      }
  }
  dwc4_args a = {C, Hp, Wp, k, s, Ho, Wo, in, w4, M, B, shift, 5, out, ring};
  if (warm) dwc4_u8(&a);
  uint64_t t0 = cycles();
  dwc4_u8(&a);
  uint64_t t1 = cycles();
  printf("dwc4 C=%d(%d) H=%d W=%d k=%d s=%d cycles=%llu macs_per_cycle=%.1f (useful channels)\n", C0, C, H, W, k, s,
         (unsigned long long)(t1 - t0), (double)C0 * Ho * Wo * kk / (double)(t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int y = 0; y < Ho && bad < 5; y++)
    for (int x = 0; x < Wo && bad < 5; x++)
      for (int c = 0; c < C && bad < 5; c += 3) {
        int32_t acc = bias[c];
        for (int dy = 0; dy < k; dy++)
          for (int dx = 0; dx < k; dx++) acc += in[((size_t)(y * s + dy) * Wp + x * s + dx) * C + c] * w[c * kk + dy * k + dx];
        int ref = clampi(clampi(rshift_rnd_ref(q31_mul_ref(acc, mult[c]), shift), -32768, 32767) + 5, 0, 255);
        int got = out[((size_t)y * Wo + x) * C + c];
        if (ref != got) {
          printf("  MISMATCH y=%d x=%d c=%d ref=%d got=%d\n", y, x, c, ref, got);
          bad++;
        }
      }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int run_dw3(int C0, int H, int W, int s, int check) {
  const int k = 3, kk = 9;
  int c64 = C0 == 64 && s == 1;
  int C = c64 ? 64 : (C0 + 127) / 128 * 128, Cb = c64 ? 128 : C;
  int Ho = (H + 2 - k) / s + 1, Wo = (W + 2 - k) / s + 1;
  int Hp = H + 2 + s, Wp = (Wo - 1) * s + 4 + s + 2;
  uint8_t *in = amalloc((size_t)Hp * Wp * C), *out = amalloc((size_t)(Ho * Wo + 2) * C);
  int8_t *w = amalloc(C * kk), *dummy = amalloc(C);
  int32_t *bias = amalloc(4 * C), *mult = amalloc(4 * C);
  V *w4 = amalloc((size_t)Cb / 128 * 12 * VLEN), *M = amalloc(Cb / 128 * 4 * VLEN), *B = amalloc(Cb / 128 * 4 * VLEN);
  for (size_t i = 0; i < (size_t)Hp * Wp * C; i++) in[i] = rnd();
  for (int i = 0; i < C * kk; i++) w[i] = (int8_t)(rnd() % 255 - 127);
  rand_requant(C, mult, dummy, bias, 20000);
  int shift = 9 + rnd() % 4;
  for (int cv = 0; cv < Cb / 128; cv++) {
    for (int dy = 0; dy < 3; dy++)
      for (int v = 0; v < 4; v++) {
        int8_t *wb = (int8_t *)&w4[((size_t)cv * 3 + dy) * 4 + v];
        for (int i = 0; i < 32; i++)
          for (int j = 0; j < 4; j++) wb[4 * i + j] = j < 3 ? w[((cv * 128 + 32 * v + i) % C) * kk + dy * 3 + j] : 0;
      }
    for (int v = 0; v < 4; v++)
      for (int i = 0; i < 32; i++) {
        ((int32_t *)&M[cv * 4 + v])[i] = mult[(cv * 128 + 32 * v + i) % C];
        ((int32_t *)&B[cv * 4 + v])[i] = bias[(cv * 128 + 32 * v + i) % C];
      }
  }
  dw3_args a = {C, Hp, Wp, s, Ho, Wo, in, w4, M, B, shift, 5, out};
  void (*fn)(const dw3_args *) = c64 ? dw3_c64_u8 : dw3_u8;
  if (warm) fn(&a);
  uint64_t t0 = cycles();
  fn(&a);
  uint64_t t1 = cycles();
  printf("dw3 C=%d(%d) H=%d W=%d k=3 s=%d cycles=%llu macs_per_cycle=%.1f (useful channels)\n", C0, C, H, W, s,
         (unsigned long long)(t1 - t0), (double)C0 * Ho * Wo * kk / (double)(t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int y = 0; y < Ho && bad < 5; y++)
    for (int x = 0; x < Wo && bad < 5; x++)
      for (int c = 0; c < C && bad < 5; c += 3) {
        int32_t acc = bias[c];
        for (int dy = 0; dy < 3; dy++)
          for (int dx = 0; dx < 3; dx++) acc += in[((size_t)(y * s + dy) * Wp + x * s + dx) * C + c] * w[c * kk + dy * 3 + dx];
        int ref = clampi(clampi(rshift_rnd_ref(q31_mul_ref(acc, mult[c]), shift), -32768, 32767) + 5, 0, 255);
        int got = out[((size_t)y * Wo + x) * C + c];
        if (ref != got) {
          printf("  MISMATCH y=%d x=%d c=%d ref=%d got=%d\n", y, x, c, ref, got);
          bad++;
        }
      }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

static int run_gemv(int M, int K, int N, int a16, int prefetch, int check) {
  const int K4 = K / 4;
  uint8_t *x = amalloc((size_t)M * K), *xh = amalloc((size_t)M * K);
  int8_t *wt = amalloc((size_t)K * N);
  V *w = amalloc((size_t)K * N);
  int32_t *y = amalloc((size_t)4 * M * N + 128);
  for (size_t i = 0; i < (size_t)M * K; i++) x[i] = rnd(), xh[i] = rnd();
  for (size_t i = 0; i < (size_t)K * N; i++) wt[i] = (int8_t)(rnd() % 255 - 127); /* wt[k][n] */
  int8_t *dst = (int8_t *)w;
  for (int nb = 0; nb < N / 32; nb++)
    for (int k4 = 0; k4 < K4; k4++)
      for (int i = 0; i < 32; i++)
        for (int j = 0; j < 4; j++) *dst++ = wt[(size_t)(4 * k4 + j) * N + 32 * nb + i];
  /* x is packed so that 32-bit word m*K4+k holds x[m][4k..4k+3] (natural row-major bytes already are) */
  /* evict: the weights must stream from DDR as they would every frame */
  size_t junk_n = 4u << 20;
  volatile uint8_t *junk = amalloc(junk_n);
  for (size_t i = 0; i < junk_n; i += 32) junk[i] = (uint8_t)i;
  uint64_t t0 = cycles();
  gemv_u8(x, M, K, N, w, y, a16, xh, prefetch);
  uint64_t t1 = cycles();
  double bytes = (double)K * N;
  printf("gemv M=%d K=%d N=%d a16=%d pf=%d cycles=%llu weight_bytes_per_cycle=%.1f macs_per_cycle=%.1f\n", M, K, N, a16, prefetch,
         (unsigned long long)(t1 - t0), bytes / (t1 - t0), (double)M * K * N / (t1 - t0));
  if (!check) return 0;
  int bad = 0;
  for (int m = 0; m < M && bad < 5; m++)
    for (int n = 0; n < N && bad < 5; n += 7) {
      int64_t r = 0;
      for (int k = 0; k < K; k++) r += (int64_t)(a16 ? (x[(size_t)m * K + k] + 256 * xh[(size_t)m * K + k]) : x[(size_t)m * K + k]) * wt[(size_t)k * N + n];
      if ((int32_t)r != y[(size_t)m * N + n]) {
        printf("  MISMATCH m=%d n=%d ref=%lld got=%ld\n", m, n, (long long)r, (long)y[(size_t)m * N + n]);
        bad++;
      }
    }
  printf("  check %s\n", bad ? "FAIL" : "PASS");
  return bad;
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  enable_cycle_counter();
  int check = !strcmp(argv[argc - 1], "check");
  for (int i = 2; i < argc; i++)
    if (!strcmp(argv[i], "warm")) {
      warm = 1;
      for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
      argc--;
      break;
    }
  if (!strcmp(argv[1], "pw4")) {
    getenv_n4 = 1;
    return run_pw(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), check, 0);
  }
  if (!strcmp(argv[1], "pw") || !strcmp(argv[1], "pw16"))
    return run_pw(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), check, argv[1][2] == '1');
  if (!strcmp(argv[1], "dw")) return run_dw(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), check);
  if (!strcmp(argv[1], "dwc")) return run_dwc(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), check);
  if (!strcmp(argv[1], "dwc2"))
    return run_dwc2(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argc > 7 && argv[7][0] != 'c' ? atoi(argv[7]) : 4, check);
  if (!strcmp(argv[1], "dwc3"))
    return run_dwc3(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argc > 7 && argv[7][0] != 'c' ? atoi(argv[7]) : 2, check);
  if (!strcmp(argv[1], "dwc4")) return run_dwc4(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), check);
  if (!strcmp(argv[1], "dw3")) return run_dw3(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), check);
  if (!strcmp(argv[1], "gemv"))
    return run_gemv(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argc > 5 && argv[5][0] != 'c' ? atoi(argv[5]) : 0,
                    argc > 6 && argv[6][0] != 'c' ? atoi(argv[6]) : 1, check);
  if (!strcmp(argv[1], "gelu")) return run_lut(atoi(argv[2]), check);
  return 2;
}
