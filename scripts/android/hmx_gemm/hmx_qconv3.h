/* QDQ 3x3 convolution (pad 1, stride 1 or 2) on V69 HMX, requantized exactly as hmx_qconv.h (QC_FAST / QC_EXACT).
 *
 * Activations live in a *flat padded* crouton layout: pixel (y, x) of an H x W map is flat pixel
 *   p = M0 + (y + 1) * Wp + x,   Wp = roundup(W + 1, 4),  M0 = (-Wp) mod 64 (64 if that is 0)
 * i.e. rows of Wp pixels (columns W..Wp-1 are padding), one padding row above (y = -1) and below, and M0 margin
 * pixels in front so that output pixel o = y * Wp + x sits at flat pixel (M0 + Wp) + o with M0 + Wp a multiple of
 * 64: an output tile block is an input block of the next layer. Pixels are grouped 64 per crouton (the :cm row
 * dimension); crouton (block b, channel block kb) is at (b * kt + kb) * 2 KB, pixel row r at byte 32 * r.
 * Padding pixels hold the activation zero point (zx), so padding is exact (xq - zx = 0).
 *
 * A 3x3 tap (dy, dx) of output block ob reads the 64 input pixels starting at flat (M0 + Wp) + ob*64 + dy*Wp + dx.
 *  - dy*Wp is a multiple of 4 pixels: `activation.ub = mxmem(Rs, Rt):single:cm` reads a 64-row window starting
 *    4 * Rs[10:7] rows into the crouton at Rs and continuing into the crouton at Rs + Rt[31:11] (hexagon-sim map,
 *    full HMX rate); Rt[31:11] = kt * 2 KB = the next pixel block of the same channel block.
 *  - dx = -1 / +1 read one-pixel-shifted copies of the input (HVX valign by 32 bytes, 2 extra copies).
 * So a 3x3 conv is 9 * kt :single instructions per 64 x 64 output tile pair, with no im2col.
 * Stride 2: HVX splits the input into row/column phases (in the output's flat geometry): tap (dy, dx) reads
 * phase (dy != 0, dx != 0) at row offset -1 for dy = -1 and column offset -1 for dx = -1 (a shifted copy of the
 * odd-column phase), so the same stride-1 machinery runs on quarter-size maps. */
#ifndef HMX_QCONV3_H
#define HMX_QCONV3_H
#include "hmx_qconv.h"

#define QC_MAX_TAPS 49 /* up to 7 x 7 */

typedef struct {
  int H, W, Wp, M0, nblk; /* nblk: pixel blocks of the whole flat buffer (incl. margins) */
  int P, Q;               /* padding rows above and below, minimum padding columns (general k x k: P = Q = k/2) */
} qc_geom_t;

/* H x W map with P padding rows above/below and >= Q padding columns: pixel (y, x) at M0 + (y + P) * Wp + x */
static inline qc_geom_t qc_geom2(int H, int W, int P, int Q) {
  qc_geom_t g;
  g.H = H, g.W = W, g.P = P, g.Q = Q, g.Wp = (W + Q + 3) & ~3;
  g.M0 = (64 - (P * g.Wp) % 64) % 64;
  if (g.M0 < Q + 1) g.M0 += 64; /* room for the leftmost column shift */
  int last = g.M0 + (H + 2 * P) * g.Wp + 64; /* + a margin block for the column shifts and :single windows */
  g.nblk = (last + 63) / 64 + 1;
  return g;
}
static inline qc_geom_t qc_geom(int H, int W) { return qc_geom2(H, W, 1, 1); }
static inline int qc_geom_pix(const qc_geom_t* g, int y, int x) { return g->M0 + (y + g->P) * g->Wp + x; }
static inline size_t qc_geom_bytes(const qc_geom_t* g, int kt) { return (size_t)g->nblk * kt * 2048; }
/* first pixel block of the output region (o = 0) */
static inline int qc_geom_oblk(const qc_geom_t* g) { return (g->M0 + g->P * g->Wp) / 64; }
static inline int qc_geom_nob(const qc_geom_t* g) { return (g->H * g->Wp + 63) / 64; }

/* host: NHWC uint8 [H, W, C] -> flat padded crouton buffer (zx everywhere else) */
static inline void qc_flat_pack(const uint8_t* x, int H, int W, int C, int zx, const qc_geom_t* g, uint8_t* out) {
  int kt = C / 32;
  memset(out, zx, qc_geom_bytes(g, kt));
  for (int y = 0; y < H; y++)
    for (int xx = 0; xx < W; xx++) {
      int p = qc_geom_pix(g, y, xx);
      for (int kb = 0; kb < kt; kb++) memcpy(out + ((size_t)(p / 64) * kt + kb) * 2048 + 32 * (p % 64), x + ((size_t)y * W + xx) * C + 32 * kb, 32);
    }
}
/* host: output region (flat o = y*Wp + x from block qc_geom_oblk) -> NHWC-rows [H*W, N] */
static inline void qc_flat_unpack(const uint8_t* buf, const qc_geom_t* g, int N, uint8_t* y) {
  int nt = N / 32;
  for (int yy = 0; yy < g->H; yy++)
    for (int x = 0; x < g->W; x++) {
      int p = qc_geom_pix(g, yy, x);
      for (int j = 0; j < nt; j++) memcpy(y + ((size_t)yy * g->W + x) * N + 32 * j, buf + ((size_t)(p / 64) * nt + j) * 2048 + 32 * (p % 64), 32);
    }
}

/* host: ONNX weights [N, C, 3, 3] -> tap-major k-major [9*C, N] (for qc_pack_params) and the HMX packing
 * [N/64 groups][9 taps][kt][2 KB :deep block] */
static inline void qc_pack_w3(const int8_t* w, int N, int C, int8_t* wk, int8_t* wp) {
  for (int t = 0; t < 9; t++)
    for (int c = 0; c < C; c++)
      for (int n = 0; n < N; n++) wk[((size_t)t * C + c) * N + n] = w[((size_t)n * C + c) * 9 + t];
  int kt = C / 32;
  for (int g = 0; g < N / 64; g++)
    for (int t = 0; t < 9; t++)
      for (int kb = 0; kb < kt; kb++) {
        int8_t* d = wp + (((size_t)g * 9 + t) * kt + kb) * 2048;
        for (int h = 0; h < 2; h++)
          for (int k = 0; k < 32; k++)
            for (int cc = 0; cc < 32; cc++)
              d[1024 * h + 128 * (k / 4) + 4 * cc + k % 4] = wk[((size_t)t * C + 32 * kb + k) * N + 64 * g + 32 * h + cc];
      }
}

/* host: ONNX weights [N, C, k, k] -> tap-major k-major [k*k*Cp, N] (for qc_pack_params; channels C..Cp-1 zero) and the
 * HMX packing [N/64 groups][k*k taps][Cp/32][2 KB :deep block]. Cp = C rounded up to 32. */
static inline void qc_pack_wk(const int8_t* w, int N, int C, int k, int Cp, int8_t* wk, int8_t* wp) {
  int kk = k * k, kt = Cp / 32;
  memset(wk, 0, (size_t)kk * Cp * N);
  for (int t = 0; t < kk; t++)
    for (int c = 0; c < C; c++)
      for (int n = 0; n < N; n++) wk[((size_t)t * Cp + c) * N + n] = w[((size_t)n * C + c) * kk + t];
  for (int g = 0; g < N / 64; g++)
    for (int t = 0; t < kk; t++)
      for (int kb = 0; kb < kt; kb++) {
        int8_t* d = wp + (((size_t)g * kk + t) * kt + kb) * 2048;
        for (int h = 0; h < 2; h++)
          for (int kq = 0; kq < 32; kq++)
            for (int cc = 0; cc < 32; cc++)
              d[1024 * h + 128 * (kq / 4) + 4 * cc + kq % 4] = wk[((size_t)t * Cp + 32 * kb + kq) * N + 64 * g + 32 * h + cc];
      }
}

#ifdef __hexagon__
static inline void qc_fill(uint8_t* buf, size_t n, int v) {
  HVX_Vector z = Q6_Vb_vsplat_R(v);
  for (size_t i = 0; i < n; i += 128) *(HVX_Vector*)(buf + i) = z;
}

/* general column shift: out[p] = in[p + c], |c| <= 3 pixels (zx beyond the ends). Per channel block the pixel blocks
 * are one stream of vectors (4 pixels each); c > 0 and c < 0 are separate straight-line loops (a select inside the
 * loop ran at 6 cycles per vector in hexagon-sim, these at ~1.5). */
#define QC_SHIFT_BODY(OP)                                                                                   \
  for (int b = 0; b < nblk; b++) {                                                                        \
    const HVX_Vector* __restrict v = (const HVX_Vector*)(in + b * bs + (size_t)kb * 2048);                \
    HVX_Vector* __restrict o = (HVX_Vector*)(out + b * bs + (size_t)kb * 2048);                           \
    HVX_Vector next = b + 1 < nblk ? *(const HVX_Vector*)(in + (b + 1) * bs + (size_t)kb * 2048) : z;     \
    HVX_Vector x0 = v[0], x1 = v[1], x2 = v[2], x3 = v[3], x4 = v[4], x5 = v[5], x6 = v[6], x7 = v[7];      \
    HVX_Vector x8 = v[8], x9 = v[9], x10 = v[10], x11 = v[11], x12 = v[12], x13 = v[13], x14 = v[14], x15 = v[15]; \
    OP;                                                                                                   \
  }
static inline void qc_shift_copy(const uint8_t* __restrict in, uint8_t* __restrict out, int nblk, int kt, int c, int zx) {
  HVX_Vector z = Q6_Vb_vsplat_R(zx);
  size_t bs = (size_t)kt * 2048;
  int sh = 32 * (c < 0 ? -c : c);
  for (int kb = 0; kb < kt; kb++) {
    HVX_Vector prev = z;
    if (c > 0) {
      QC_SHIFT_BODY(
          o[0] = Q6_V_valign_VVR(x1, x0, sh); o[1] = Q6_V_valign_VVR(x2, x1, sh); o[2] = Q6_V_valign_VVR(x3, x2, sh);
          o[3] = Q6_V_valign_VVR(x4, x3, sh); o[4] = Q6_V_valign_VVR(x5, x4, sh); o[5] = Q6_V_valign_VVR(x6, x5, sh);
          o[6] = Q6_V_valign_VVR(x7, x6, sh); o[7] = Q6_V_valign_VVR(x8, x7, sh); o[8] = Q6_V_valign_VVR(x9, x8, sh);
          o[9] = Q6_V_valign_VVR(x10, x9, sh); o[10] = Q6_V_valign_VVR(x11, x10, sh); o[11] = Q6_V_valign_VVR(x12, x11, sh);
          o[12] = Q6_V_valign_VVR(x13, x12, sh); o[13] = Q6_V_valign_VVR(x14, x13, sh); o[14] = Q6_V_valign_VVR(x15, x14, sh);
          o[15] = Q6_V_valign_VVR(next, x15, sh))
    } else {
      QC_SHIFT_BODY(
          o[0] = Q6_V_vlalign_VVR(x0, prev, sh); o[1] = Q6_V_vlalign_VVR(x1, x0, sh); o[2] = Q6_V_vlalign_VVR(x2, x1, sh);
          o[3] = Q6_V_vlalign_VVR(x3, x2, sh); o[4] = Q6_V_vlalign_VVR(x4, x3, sh); o[5] = Q6_V_vlalign_VVR(x5, x4, sh);
          o[6] = Q6_V_vlalign_VVR(x6, x5, sh); o[7] = Q6_V_vlalign_VVR(x7, x6, sh); o[8] = Q6_V_vlalign_VVR(x8, x7, sh);
          o[9] = Q6_V_vlalign_VVR(x9, x8, sh); o[10] = Q6_V_vlalign_VVR(x10, x9, sh); o[11] = Q6_V_vlalign_VVR(x11, x10, sh);
          o[12] = Q6_V_vlalign_VVR(x12, x11, sh); o[13] = Q6_V_vlalign_VVR(x13, x12, sh); o[14] = Q6_V_vlalign_VVR(x14, x13, sh);
          o[15] = Q6_V_vlalign_VVR(x15, x14, sh); prev = x15)
    }
  }
}
#undef QC_SHIFT_BODY

/* one-pixel shifted copies of a flat buffer: m1[p] = in[p - 1], p1[p] = in[p + 1] (zx beyond the ends) */
static inline void qc_shift_copies(const uint8_t* in, uint8_t* m1, uint8_t* p1, int nblk, int kt, int zx) {
  qc_shift_copy(in, m1, nblk, kt, -1, zx);
  qc_shift_copy(in, p1, nblk, kt, 1, zx);
}

/* stride-2 phase split (out[2*py] and out[2*py + 1] both NULL: row parity py not needed): out[ph] (ph = 2*py + px, flat buffers in the output geometry go) holds input pixel
 * (2i + py, 2j + px) at output position (i, j); rows i = -1 and columns j >= Wo are zx (so is any input pixel
 * outside the map). gi: input geometry. 4 pixels (one vector) at a time: vdeal by 32 bytes splits a pair of
 * vectors (8 consecutive input pixels) into even and odd pixels. */
/* one output row (py, i) of the phase split, one channel block; VT = the load type (aligned HVX_Vector, or hmx_uvec
 * when the source is a misaligned DDR buffer: the compiler otherwise emits unaligned loads for both) */
#define QC_PHASE_ROW(VT)                                                                                         \
  _Pragma("unroll 4") for (int j = 0; j < nfull; j++) {                                                        \
    int pp = p + 8 * j, qq = q + 4 * j;                                                                        \
    HVX_Vector a = *(const VT*)(ib + (size_t)(pp >> 6) * bs + ((pp & 63) << 5));                               \
    HVX_Vector b = *(const VT*)(ib + (size_t)((pp + 4) >> 6) * bs + (((pp + 4) & 63) << 5));                   \
    HVX_VectorPair d = Q6_W_vdeal_VVR(b, a, -32); /* lo: pixels 0,2,4,6; hi: 1,3,5,7 */                        \
    size_t o = (size_t)(qq >> 6) * bs + okb + ((qq & 63) << 5);                                                \
    *(HVX_Vector*)(o0 + o) = Q6_V_lo_W(d);                                                                     \
    *(HVX_Vector*)(o1 + o) = Q6_V_hi_W(d);                                                                     \
  }                                                                                                            \
  if (tail) { /* the last, partial vector: padding columns stay zx */                                          \
    int pp = p + 8 * nfull, qq = q + 4 * nfull;                                                                \
    HVX_Vector a = *(const VT*)(ib + (size_t)(pp >> 6) * bs + ((pp & 63) << 5));                               \
    HVX_Vector b = *(const VT*)(ib + (size_t)((pp + 4) >> 6) * bs + (((pp + 4) & 63) << 5));                   \
    HVX_VectorPair d = Q6_W_vdeal_VVR(b, a, -32);                                                              \
    size_t o = (size_t)(qq >> 6) * bs + okb + ((qq & 63) << 5);                                                \
    *(HVX_Vector*)(o0 + o) = Q6_V_vmux_QVV(keep, Q6_V_lo_W(d), z);                                             \
    *(HVX_Vector*)(o1 + o) = Q6_V_vmux_QVV(keep, Q6_V_hi_W(d), z);                                             \
  }
static inline void qc_phase_split(const uint8_t* in, const qc_geom_t* gi, uint8_t* const out[4], const qc_geom_t* go, int kt,
                                  int zx) {
  HVX_Vector z = Q6_Vb_vsplat_R(zx);
  size_t bs = (size_t)kt * 2048;
  for (int ph = 0; ph < 4; ph++)
    if (out[ph]) qc_fill(out[ph], qc_geom_bytes(go, kt), zx);
  int nfull = go->W / 4, tail = go->W % 4; /* full 4-pixel vectors per output row, pixels in the last one */
  HVX_VectorPred keep = Q6_Q_vsetq2_R(tail ? 32 * tail : 128);
  int aligned = !((uintptr_t)in & 127);
  for (int py = 0; py < 2; py++) {
    uint8_t *o0 = out[2 * py], *o1 = out[2 * py + 1];
    if (!o0 || !o1) continue; /* phases come in (even, odd column) pairs per row parity; a missing pair is not built */
    for (int i = 0; i < go->H; i++) {
      int r = 2 * i + py;
      if (r >= gi->H) break; /* stays zx */
      int p0 = qc_geom_pix(gi, r, 0), q0 = qc_geom_pix(go, i, 0);
      if (kt == 1 && r + 2 < gi->H) { /* one contiguous row per channel block: prefetch the row after next (DDR sources) */
        int pn = qc_geom_pix(gi, r + 2, 0);
        hmx_l2fetch(in + (size_t)(pn >> 6) * bs + ((pn & 63) << 5), (size_t)2 * go->W * 32 + 256);
      }
      for (int kb = 0; kb < kt; kb++) {
        const uint8_t* ib = in + (size_t)kb * 2048;
        size_t okb = (size_t)kb * 2048;
        int p = p0, q = q0;
        if (aligned) {
          QC_PHASE_ROW(HVX_Vector)
        } else {
          QC_PHASE_ROW(hmx_uvec)
        }
      }
    }
  }
}
#undef QC_PHASE_ROW

/* tap sources of one k x k conv in the output geometry go: src[t] = buffer, drow[t] = row offset; t = ky * k + kx
 * (the weight packing order of qc_pack_wk) */
typedef struct {
  const uint8_t* src[QC_MAX_TAPS];
  int drow[QC_MAX_TAPS];
  int n;
} qc_taps_t;

/* The HMX faults when one instruction's operands span a 256 KB VTCM boundary (phone only; hexagon-sim does not
 * model it). A :single window whose two croutons (kt * 2 KB apart) straddle a boundary is therefore stitched into a
 * side crouton by HVX (16 vector moves: window offsets are multiples of 4 rows = 128 bytes) and read with a
 * one-crouton instruction. qc_conv3x3_plan (once per layer and buffer set) fills atab[(ob * 9 + t) * kt + kb] with
 * the Rs of every instruction (offset bits included; offset 0 = no second crouton) and stitch[] with the straddling
 * windows (src Rs, side crouton); returns their number, or -1 if side_cap is too small. qc_conv3x3_stitch copies
 * them (every run, after the tap sources are written). */
typedef struct {
  uint32_t rs;         /* the original window (address | offset bits) */
  uint8_t* side;       /* its side crouton */
} qc_stitch_t;

static inline int qc_conv3x3_plan(const qc_taps_t* tp, const qc_geom_t* go, int kt, uint32_t* atab, qc_stitch_t* stitch,
                                  uint8_t* side, int side_cap) {
  int ob0 = qc_geom_oblk(go), nob = qc_geom_nob(go), ns = 0, nt9 = tp->n;
  for (int t = 0; t < nt9; t++) {
    int s = tp->drow[t] * go->Wp; /* pixels, multiple of 4, may be negative */
    int sb = s >= 0 ? s / 64 : -((-s + 63) / 64), off = s - 64 * sb, o4 = off / 4;
    for (int ob = 0; ob < nob; ob++)
      for (int kb = 0; kb < kt; kb++) {
        const uint8_t* a0 = tp->src[t] + ((size_t)(ob0 + sb + ob) * kt + kb) * 2048;
        const uint8_t* a1 = a0 + (size_t)kt * 2048;
        uint32_t rs = (uint32_t)(uintptr_t)a0 | ((uint32_t)o4 << 7);
#ifndef QC_STITCH_ALL /* test hook: stitch every offset window */
        if (o4 && ((uintptr_t)a0 >> 18) != ((uintptr_t)a1 >> 18)) {
#else
        if (o4) {
#endif
          if (ns >= side_cap) return -1;
          stitch[ns].rs = rs, stitch[ns].side = side + (size_t)ns * 2048;
          rs = (uint32_t)(uintptr_t)stitch[ns].side, ns++;
        }
        atab[((size_t)ob * nt9 + t) * kt + kb] = rs;
      }
  }
  return ns;
}
static inline void qc_conv3x3_stitch(const qc_stitch_t* st, int ns, int kt) {
  for (int i = 0; i < ns; i++) {
    int o4 = (st[i].rs >> 7) & 15;
    const HVX_Vector* v0 = (const HVX_Vector*)(uintptr_t)(st[i].rs & ~2047u);
    const HVX_Vector* v1 = (const HVX_Vector*)((uintptr_t)(st[i].rs & ~2047u) + (size_t)kt * 2048);
    HVX_Vector* d = (HVX_Vector*)st[i].side;
    for (int j = 0; j < 16; j++) d[j] = j + o4 < 16 ? v0[j + o4] : v1[j + o4 - 16];
  }
}

/* 3x3 conv: Y = output flat buffer in geometry go (tiles for blocks oblk .. oblk+nob-1; the rest of Y untouched),
 * W3 from qc_pack_w3, blk/h from qc_pack_params(K = 9*C), atab from qc_conv3x3_prep. */
static inline int qc_convk(const uint32_t* atab, int ntaps, const qc_geom_t* go, uint8_t* Y, const uint8_t* W3, const qc_blk_t* blk,
                           const qc_hdr_t* h, int kt, int mode, uint8_t* scratch) {
  int nt = h->n / 32, nfix = 0, ob0 = qc_geom_oblk(go), nob = qc_geom_nob(go);
  unsigned rt2 = ((unsigned)kt * 2048) | 0x7ff;
  for (int g = 0; g < h->n / 64; g++)
    for (int ob = 0; ob < nob; ob++) {
      const uint8_t* w = W3 + (size_t)g * ntaps * kt * 2048;
      const uint32_t* at = atab + (size_t)ob * ntaps * kt;
      for (int i = 0; i < ntaps * kt; i++, w += 2048) {
        uint32_t rs = at[i];
        __asm__ volatile("{ activation.ub = mxmem(%0,%1):single:cm\n weight.b = mxmem(%2,%3):deep }" ::"r"(rs),
                         "r"(rs & 0x780 ? rt2 : 0x7ffu), "r"(w), "r"(0x7ff)
                         : "memory");
      }
      for (int hh = 0; hh < 2; hh++) {
        const qc_blk_t* b = &blk[2 * g + hh];
        uint8_t* yt = Y + ((size_t)(ob0 + ob) * nt + 2 * g + hh) * 2048;
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
  if (mode == QC_FAST && h->relu && h->lo > 0) {
    HVX_Vector lo = Q6_Vb_vsplat_R(h->lo);
    for (size_t i = (size_t)ob0 * nt * 2048; i < (size_t)(ob0 + nob) * nt * 2048; i += 128)
      *(HVX_Vector*)(Y + i) = Q6_Vub_vmax_VubVub(*(HVX_Vector*)(Y + i), lo);
  }
  return nfix;
}

static inline int qc_conv3x3(const uint32_t* atab, const qc_geom_t* go, uint8_t* Y, const uint8_t* W3, const qc_blk_t* blk,
                             const qc_hdr_t* h, int kt, int mode, uint8_t* scratch) {
  return qc_convk(atab, 9, go, Y, W3, blk, h, kt, mode, scratch);
}

/* taps of a stride-1 conv: X = input, xm1/xp1 = qc_shift_copies of it (all in geometry go = the input's) */
static inline qc_taps_t qc_taps_s1(const uint8_t* X, const uint8_t* xm1, const uint8_t* xp1) {
  qc_taps_t tp;
  for (int t = 0; t < 9; t++) tp.src[t] = t % 3 == 0 ? xm1 : t % 3 == 1 ? X : xp1, tp.drow[t] = t / 3 - 1;
  tp.n = 9;
  return tp;
}
/* taps of a stride-2 conv from qc_phase_split's ph[4] and ph1m1 = the m1 shift of ph[1] (odd columns, x - 1) and
 * ph3m1 = the m1 shift of ph[3] */
static inline qc_taps_t qc_taps_s2(uint8_t* const ph[4], const uint8_t* ph1m1, const uint8_t* ph3m1) {
  qc_taps_t tp;
  for (int t = 0; t < 9; t++) {
    int dy = t / 3 - 1, dx = t % 3 - 1, py = dy != 0, px = dx != 0;
    const uint8_t* s = ph[2 * py + px];
    if (dx == -1) s = py ? ph3m1 : ph1m1;
    tp.src[t] = s, tp.drow[t] = dy == -1 ? -1 : 0;
  }
  tp.n = 9;
  return tp;
}
#endif /* __hexagon__ */
#endif
