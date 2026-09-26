/* DSP side of the graph runner: executes an rn_model_t (rn_model.h) over VTCM, one op after another, on the thread
 * that holds the HVX + HMX locks. rn_plan once after the model and VTCM are in place (conv instruction address
 * tables, 256 KB-straddle stitching), then rn_run per input. */
#ifndef RN_EXEC_H
#define RN_EXEC_H
#include <stdlib.h>

#include "rn_model.h"

#ifndef RN_NOW
#define RN_NOW() 0ull /* the skel defines it as HAP_perf_get_time_us() */
#endif

typedef struct {
  const rn_model_t* m;
  const uint8_t* blob; /* packed weights + params (DDR) */
  uint8_t* vtcm;
  uint32_t* atab[RN_MAX_OPS];
  qc_stitch_t* stitch[RN_MAX_OPS];
  int nstitch[RN_MAX_OPS];
  unsigned long long us[RN_MAX_OPS]; /* per-op time of the last run */
  unsigned long long ph[RN_MAX_OPS][5]; /* conv phases: weights copy, sources, stitch, HMX + requant, pads */
  int nfix[RN_MAX_OPS];
  /* optional asynchronous weight prefetch (the skel runs the copies on a second HVX thread): prefetch(c, op) starts
   * copying op's weights + params, wait(c, op) returns when they are in VTCM. NULL: synchronous copies. */
  void (*prefetch)(void* user, int op);
  void (*wait)(void* user, int op);
  void* user;
} rn_ctx_t;

static inline void rn_copy_weights(rn_ctx_t* c, int op) {
  const rn_op_t* o = &c->m->op[op];
  hmx_copy_hvx(c->vtcm + o->vw, c->blob + o->w_off, o->w_len);
  hmx_copy_hvx(c->vtcm + o->vprm, c->blob + o->prm_off, o->prm_len);
}

static inline uint8_t* rn_tbuf(rn_ctx_t* c, int t, const uint8_t* input) {
  const rn_tensor_t* T = &c->m->t[t];
  return T->in_ddr ? (uint8_t*)input : c->vtcm + T->off;
}

/* tap sources of a conv / maxpool: src pointers into VTCM (or the input tensor) */
static inline void rn_taps(rn_ctx_t* c, const rn_op_t* o, const uint8_t* input, qc_taps_t* tp) {
  const uint8_t* x = rn_tbuf(c, o->x, input);
  tp->n = o->ntaps;
  for (int t = 0; t < o->ntaps; t++)
    tp->src[t] = o->tap_src[t] < 0 ? x : c->vtcm + o->src[(int)o->tap_src[t]].off, tp->drow[t] = o->tap_drow[t];
}

static inline int rn_plan(rn_ctx_t* c) {
  for (int i = 0; i < c->m->nops; i++) {
    const rn_op_t* o = &c->m->op[i];
    if (o->type != RN_CONV) continue;
    qc_taps_t tp;
    rn_taps(c, o, NULL, &tp);
    int kt = c->m->t[o->x].cp / 32;
    for (int t = 0; t < tp.n; t++)
      if (!tp.src[t]) return -10 - i; /* a conv reading the DDR graph input directly: not supported */
    c->atab[i] = (uint32_t*)malloc(sizeof(uint32_t) * qc_geom_nob(&o->gsrc) * tp.n * kt);
    c->stitch[i] = (qc_stitch_t*)malloc(sizeof(qc_stitch_t) * (o->side_cap ? o->side_cap : 1));
    c->nstitch[i] = qc_conv3x3_plan(&tp, &o->gsrc, kt, c->atab[i], c->stitch[i], c->vtcm + o->vside, o->side_cap);
    if (c->nstitch[i] < 0) return -100 - i;
  }
  return 0;
}

static inline void rn_unplan(rn_ctx_t* c) {
  for (int i = 0; i < RN_MAX_OPS; i++) free(c->atab[i]), free(c->stitch[i]), c->atab[i] = NULL, c->stitch[i] = NULL;
}

/* zero point everywhere outside the H x W map of a flat buffer (after an op wrote its output region) */
static inline void rn_fix_pads(uint8_t* buf, const qc_geom_t* g, int kt, int zp) {
  HVX_Vector z = Q6_Vb_vsplat_R(zp);
  int ob0 = qc_geom_oblk(g), nob = qc_geom_nob(g);
  size_t bs = (size_t)kt * 2048;
  qc_fill(buf, (size_t)ob0 * bs, zp);
  qc_fill(buf + (size_t)(ob0 + nob) * bs, (size_t)(g->nblk - ob0 - nob) * bs, zp);
  for (int y = 0; y <= g->H; y++) {
    int q = y < g->H ? qc_geom_pix(g, y, g->W) : qc_geom_pix(g, g->H, 0);
    int e = y < g->H ? qc_geom_pix(g, y + 1, 0) : (ob0 + nob) * 64;
    for (int p = q & ~3; p < e; p += 4) {
      HVX_VectorPred keep = Q6_Q_vsetq_R(p < q ? 32 * (q - p) : 0);
      for (int kb = 0; kb < kt; kb++) {
        HVX_Vector* v = (HVX_Vector*)(buf + ((size_t)(p / 64) * kt + kb) * 2048 + 32 * (p % 64));
        *v = p < q ? Q6_V_vmux_QVV(keep, *v, z) : z;
      }
    }
  }
}

/* sources of a conv / maxpool: phases and column shifts, in the order of o->src (a shift reads an earlier source or
 * the input) */
static inline void rn_build_sources(rn_ctx_t* c, const rn_op_t* o, const uint8_t* input) {
  const rn_tensor_t* X = &c->m->t[o->x];
  const uint8_t* x = rn_tbuf(c, o->x, input);
  int kt = X->cp / 32;
  uint8_t* ph[4] = {NULL, NULL, NULL, NULL};
  for (int i = 0; i < o->nsrc; i++)
    if (o->src[i].kind == RN_SRC_PHASE) ph[o->src[i].a] = c->vtcm + o->src[i].off;
  if (ph[0] || ph[1] || ph[2] || ph[3]) qc_phase_split(x, &X->g, ph, &o->gsrc, kt, X->zp);
  for (int i = 0; i < o->nsrc; i++)
    if (o->src[i].kind == RN_SRC_SHIFT) {
      const uint8_t* in = o->src[i].a < 0 ? x : c->vtcm + o->src[o->src[i].a].off;
      qc_shift_copy(in, c->vtcm + o->src[i].off, o->gsrc.nblk, kt, o->src[i].c, X->zp);
    }
}

static inline int rn_conv(rn_ctx_t* c, int i, const uint8_t* input, int mode) {
  const rn_op_t* o = &c->m->op[i];
  const rn_tensor_t *X = &c->m->t[o->x], *Y = &c->m->t[o->y];
  int kt = X->cp / 32;
  unsigned long long t0 = RN_NOW(), t1;
  if (o->prefetched && c->wait)
    c->wait(c->user, i);
  else
    rn_copy_weights(c, i);
  if (o->prefetch_next >= 0 && c->prefetch) c->prefetch(c->user, o->prefetch_next); /* overlaps with this op and the next */
  t1 = RN_NOW(), c->ph[i][0] = t1 - t0, t0 = t1;
  rn_build_sources(c, o, input);
  t1 = RN_NOW(), c->ph[i][1] = t1 - t0, t0 = t1;
  qc_conv3x3_stitch(c->stitch[i], c->nstitch[i], kt);
  t1 = RN_NOW(), c->ph[i][2] = t1 - t0, t0 = t1;
  const qc_blk_t* B = (const qc_blk_t*)(c->vtcm + o->vprm);
  const qc_hdr_t* H = (const qc_hdr_t*)(c->vtcm + o->vprm + sizeof(qc_blk_t) * (o->n / 32));
  uint8_t* y = c->vtcm + Y->off;
  int nfix = qc_convk(c->atab[i], o->ntaps, &o->gsrc, y, c->vtcm + o->vw, B, H, kt, mode, c->vtcm + o->vplanes);
  (void)*(volatile uint8_t*)y;
  t1 = RN_NOW(), c->ph[i][3] = t1 - t0, t0 = t1;
  rn_fix_pads(y, &Y->g, Y->cp / 32, Y->zp);
  c->ph[i][4] = RN_NOW() - t0;
  return nfix;
}

/* ORT's QLinearAdd exactly: HVX fixed point v * 2^F from the exact products a * mantissa(ra) (12-bit halves),
 * round half up; lanes whose fraction is within win of .5 are recomputed with ORT's fp32 sequence
 * (rne(rb*b + (ra*a + fixed))) on the scalar core. Whole flat buffers (same geometry), pads fixed afterwards.
 * Near-tie flags are collected per 2 KB and checked with one vector -> scalar round trip. Straight-line (no
 * branches, shift counts in registers: 0 <= sa, sb <= 12 is checked by the loader). */
#define RN_ADD_HALF(VA, VB, Y, FL)                                                                             \
  do {                                                                                                          \
    HVX_VectorPair pah = Q6_Ww_vmpy_VhRh(VA, ah), pal = Q6_Ww_vmpy_VhRh(VA, al);                                \
    HVX_VectorPair pbh = Q6_Ww_vmpy_VhRh(VB, bh), pbl = Q6_Ww_vmpy_VhRh(VB, bl);                                \
    HVX_Vector v0 = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_V_lo_W(pah), sah), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(pal), o->sa)); \
    HVX_Vector v1 = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_V_hi_W(pah), sah), Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(pal), o->sa)); \
    v0 = Q6_Vw_vadd_VwVw(v0, Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_V_lo_W(pbh), sbh), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(pbl), o->sb))); \
    v1 = Q6_Vw_vadd_VwVw(v1, Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_V_hi_W(pbh), sbh), Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(pbl), o->sb))); \
    v0 = Q6_Vw_vadd_VwVw(v0, fq), v1 = Q6_Vw_vadd_VwVw(v1, fq);                                                  \
    HVX_Vector r0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(v0, half), o->F), r1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(v1, half), o->F); \
    HVX_VectorPred q0 = Q6_Q_vcmp_gt_VwVw(win, Q6_Vw_vabs_Vw(Q6_Vw_vsub_VwVw(Q6_V_vand_VV(v0, mask), half)));   \
    HVX_VectorPred q1 = Q6_Q_vcmp_gt_VwVw(win, Q6_Vw_vabs_Vw(Q6_Vw_vsub_VwVw(Q6_V_vand_VV(v1, mask), half)));   \
    Y = Q6_Vh_vsat_VwVw(r1, r0);                                                                                \
    FL = Q6_Vh_vsat_VwVw(Q6_V_vmux_QVV(q1, one, zero), Q6_V_vmux_QVV(q0, one, zero));                           \
  } while (0)
static inline int rn_add(rn_ctx_t* c, int i, const uint8_t* input) {
  const rn_op_t* o = &c->m->op[i];
  const rn_tensor_t *A = &c->m->t[o->x], *Y = &c->m->t[o->y];
  const uint8_t *a = rn_tbuf(c, o->x, input), *b = rn_tbuf(c, o->x2, input);
  uint8_t* y = c->vtcm + Y->off;
  int kt = A->cp / 32, ob0 = qc_geom_oblk(&A->g), nob = qc_geom_nob(&A->g), nfix = 0;
  size_t beg = (size_t)ob0 * kt * 2048, end = (size_t)(ob0 + nob) * kt * 2048;
  HVX_Vector fq = Q6_V_vsplat_R(o->fq), half = Q6_V_vsplat_R(1 << (o->F - 1)), mask = Q6_V_vsplat_R((1 << o->F) - 1);
  HVX_Vector win = Q6_V_vsplat_R(o->win), one = Q6_V_vsplat_R(1), zero = Q6_V_vzero();
  int ah = (o->a_hi << 16) | o->a_hi, al = (o->a_lo << 16) | o->a_lo, bh = (o->b_hi << 16) | o->b_hi, bl = (o->b_lo << 16) | o->b_lo;
  int sah = 12 - o->sa, sbh = 12 - o->sb;
  HVX_Vector fl[16];
  unsigned long long t0 = RN_NOW(), tres = 0;
  int nres = 0;
  for (size_t c0 = beg; c0 < end; c0 += 2048) {
    HVX_Vector any = zero;
    const HVX_Vector *pa = (const HVX_Vector*)(a + c0), *pb = (const HVX_Vector*)(b + c0);
    HVX_Vector* py = (HVX_Vector*)(y + c0);
    for (int k = 0; k < 16; k++) {
      HVX_VectorPair ua = Q6_Wuh_vunpack_Vub(pa[k]), ub = Q6_Wuh_vunpack_Vub(pb[k]);
      HVX_Vector y0, y1, f0, f1;
      RN_ADD_HALF(Q6_V_lo_W(ua), Q6_V_lo_W(ub), y0, f0);
      RN_ADD_HALF(Q6_V_hi_W(ua), Q6_V_hi_W(ub), y1, f1);
      py[k] = Q6_Vub_vpack_VhVh_sat(y1, y0);
      HVX_Vector f = Q6_Vub_vpack_VhVh_sat(f1, f0);
      fl[k] = f;
      any = Q6_V_vor_VV(any, f);
    }
    HVX_Vector st[2];
    st[0] = qc_ror_or(any);
    if (!*(volatile int32_t*)st) continue;
    unsigned long long r0 = RN_NOW();
    nres++;
    HVX_Vector fs[16] __attribute__((aligned(128)));
    for (int k = 0; k < 16; k++) fs[k] = fl[k];
    const uint64_t* fw = (const uint64_t*)fs; /* scan 64-bit words, skip the (almost always) zero ones */
    for (int w = 0; w < 256; w++) {
      uint64_t bits = fw[w];
      while (bits) {
        int j = __builtin_ctzll(bits) / 8;
        bits &= ~(0xffull << (8 * j));
        size_t off = c0 + 8 * w + j;
        float t1 = __builtin_HEXAGON_F2_sfmpy(o->ra, __builtin_HEXAGON_F2_conv_w2sf(a[off]));
        float t2 = __builtin_HEXAGON_F2_sfadd(t1, o->fixed);
        float t3 = __builtin_HEXAGON_F2_sfmpy(o->rb, __builtin_HEXAGON_F2_conv_w2sf(b[off]));
        int v = __builtin_HEXAGON_F2_conv_sf2w(__builtin_HEXAGON_F2_sfadd(t3, t2));
        y[off] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        nfix++;
      }
    }
    tres += RN_NOW() - r0;
  }
  unsigned long long t1 = RN_NOW();
  rn_fix_pads(y, &Y->g, Y->cp / 32, Y->zp);
  c->ph[i][0] = t1 - t0 - tres, c->ph[i][1] = tres, c->ph[i][2] = nres, c->ph[i][3] = nfix, c->ph[i][4] = RN_NOW() - t1;
  return nfix;
}
#undef RN_ADD_HALF

/* MaxPool 3x3 s2 p1 as the max over the 9 tap windows of the phase sources (input zero point 0 = padding min) */
static inline void rn_maxpool(rn_ctx_t* c, int i, const uint8_t* input) {
  const rn_op_t* o = &c->m->op[i];
  const rn_tensor_t *X = &c->m->t[o->x], *Y = &c->m->t[o->y];
  int kt = X->cp / 32, ob0 = qc_geom_oblk(&o->gsrc), nob = qc_geom_nob(&o->gsrc);
  rn_build_sources(c, o, input);
  qc_taps_t tp;
  rn_taps(c, o, input, &tp);
  uint8_t* y = c->vtcm + Y->off;
  for (int ob = 0; ob < nob; ob++)
    for (int kb = 0; kb < kt; kb++) {
      HVX_Vector m[16];
      for (int j = 0; j < 16; j++) m[j] = Q6_V_vzero();
      for (int t = 0; t < tp.n; t++) {
        int st = (ob0 + ob) * 64 + tp.drow[t] * o->gsrc.Wp, blk = st >> 6, o4 = (st & 63) >> 2;
        for (int j = 0; j < 16; j++) {
          int q = o4 + j;
          HVX_Vector v = *(const HVX_Vector*)(tp.src[t] + ((size_t)(blk + (q >> 4)) * kt + kb) * 2048 + 128 * (q & 15));
          m[j] = Q6_Vub_vmax_VubVub(m[j], v);
        }
      }
      HVX_Vector* d = (HVX_Vector*)(y + ((size_t)(ob0 + ob) * kt + kb) * 2048);
      for (int j = 0; j < 16; j++) d[j] = m[j];
    }
  rn_fix_pads(y, &Y->g, Y->cp / 32, Y->zp);
}

static inline int rn_run(rn_ctx_t* c, const uint8_t* input, int mode) {
  int nfix = 0;
  const rn_tensor_t* I = &c->m->t[c->m->input];
  if (!I->in_ddr) hmx_copy_hvx(c->vtcm + I->off, input, qc_geom_bytes(&I->g, I->cp / 32));
  for (int i = 0; i < c->m->nops; i++) {
    unsigned long long t0 = RN_NOW();
    const rn_op_t* o = &c->m->op[i];
    int f = 0;
    if (o->type == RN_CONV) f = rn_conv(c, i, input, mode);
    else if (o->type == RN_ADD) f = rn_add(c, i, input);
    else rn_maxpool(c, i, input);
    c->nfix[i] = f, nfix += f;
    c->us[i] = RN_NOW() - t0;
  }
  return nfix;
}
#endif
