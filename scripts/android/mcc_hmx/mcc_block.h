/* One MCC decoder block (../vision_models/mcc/model.py QueryDecoder, per block) for a chunk of Q queries,
 * fused on Hexagon V69: the GEMMs on HMX (../hmx_gemm/hmx_block.h), everything else on HVX (mb_hvx.h; a
 * scalar-C fallback per step for checking), with every activation in VTCM in the HMX tile layout.
 *
 *   h = LN1(x); q, k, v = h Wqkv + b                      (HMX, per head: its q / k / v column blocks)
 *   S = q_h K_h^T * scale * log2(e) (K pre-scaled; 197 -> 224 cols) (HMX, one K tile; padded cols -65504)
 *   s_self = q_h . k_h * scale * log2(e); P = 2^[S, s_self] / sum  (HVX, row-wise: a base-2 softmax)
 *   o_h = P V_h + p_self v_h                              (HMX, 7 K tiles; + HVX)
 *   x = x + o Wproj + b                                   (HMX; the residual is one more K tile: x . I)
 *   g = GELU(LN2(x) Wfc1 + b); x = x + g Wfc2 + b         (HMX; GELU on HVX; residual as above)
 *
 * head_dim = 32 = one HMX tile: q / k / v of head h are qkv column blocks h / 16+h / 32+h, and o_h is
 * column block h of o, which is directly the proj GEMM's K tile h (outputs and activations share the tile
 * layout). The residual add costs one extra tile MAC per output tile (the x tile times an identity weight
 * tile, into the same exact fp16 accumulator), so x + y is rounded once and HVX never touches it.
 * Weights come prepacked (ref.py pack_block) and stream one output-column block ("job") at a time into
 * one of two VTCM buffers: with >= 2 threads, thread 1 copies job g + 1 while thread 0's HMX runs job g
 * (sequence counters + spin waits, no barrier per job); the tables carry the biases.
 *
 * Threads (SPMD, mb_block(c, w, tid) on every thread): thread 0 holds the HMX lock and issues every HMX
 * op; the row-wise HVX steps hand out row blocks through an atomic counter per phase, so thread 0 joins
 * as soon as its HMX work for the phase is done. Overlap: attention is pipelined by head (phase k:
 * thread 0 finishes head k-1 -- P V, self term -- and computes q / k / v / S of head k+1 into the other
 * half of a double buffer while everyone runs head k's softmax); the MLP runs in two row halves (fc1 of
 * the second half beside GELU of the first, fc2 of the first beside GELU of the second). With nthr = 1
 * (hexagon-sim: no QuRT threads) thread 0 runs the same phases in order.
 *
 * VTCM (8 MB; Q <= 1024 = RT row tiles of 32):
 *   [0, 1M)   X: residual stream, RT x 16 tiles        [1M, 2M)  H: LayerNorm output (GEMM A operand)
 *   [2M, 6M)  scratch: attention -- O (1M) | 2 x QKV of a head (3 x RT tiles) | 2 x S (RT x 8 tiles, 7
 *             used); MLP -- the fc1 / GELU output, RT x 64 tiles (row block = 128 KB)
 *   [6M, ...) two weight column-block buffers (<= 64 tiles each), tables, identity tile, 2 x p_self
 * Row blocks start at multiples of their span, so no HMX operand span crosses a 256 KB window. */
#ifndef MCC_BLOCK_H
#define MCC_BLOCK_H
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "../hmx_gemm/hmx_block.h"

#define MB_D 512
#define MB_HEADS 16
#define MB_KT 16    /* 512 / 32 */
#define MB_HT 64    /* 2048 / 32 */
#define MB_ST 7     /* 224 / 32: seen tokens (197) padded */
#define MB_SEEN 197
#define MB_TB 2048  /* bytes per tile */
#define MB_TH 1024  /* halfwords per tile */
#define MB_SCALE 0.17677669529663687f
#define MB_SCALE2 (MB_SCALE * 1.4426950408889634f) /* scores are in log2 units (K packed pre-scaled) */

typedef uint16_t mb_hf;

/* per-phase time accounting (the skel defines MB_NOW as the DSP's pcycle counter; 0 = off) */
#ifndef MB_NOW
#define MB_NOW() 0ull
#endif
/* which steps run as HVX (mb_hvx.h) instead of scalar C; MB_V_SM covers softmax + the self term.
 * Scalar steps are serial (thread 0 only). */
enum { MB_V_GELU = 2, MB_V_LN = 4, MB_V_SM = 8, MB_V_COPY = 16, MB_V_ALL = 30 };
enum { MB_P_LN, MB_P_WCOPY, MB_P_QKV, MB_P_S, MB_P_SOFTMAX, MB_P_PV, MB_P_SELF, MB_P_PROJ, MB_P_FC1, MB_P_FC2, MB_P_WAIT, MB_P_GELU, MB_NPROF };

/* one block's weights, as ref.py BLOCK_LAYOUT writes them (fp16 tiles, u32 tables) */
typedef struct {
  const mb_hf *wqkv, *wproj, *wfc1, *wfc2, *kt, *vt, *ln1, *ln2;
  const uint32_t *tqkv, *tproj, *tfc1, *tfc2;
} mb_weights;

static inline size_t mb_block_bytes(void) {
  return (size_t)(48 * 16 + 16 * 16 + 64 * 16 + 16 * 64 + 16 * 7 + 16 * 7) * MB_TB + (48 + 16 + 64 + 16) * 256 + 4 * MB_D * 2;
}

static inline void mb_bind(mb_weights* w, const void* blob) {
  const uint8_t* p = (const uint8_t*)blob;
#define TAKE(f, T, n) (w->f = (const T*)p, p += (n))
  TAKE(wqkv, mb_hf, 48 * 16 * MB_TB);
  TAKE(tqkv, uint32_t, 48 * 256);
  TAKE(wproj, mb_hf, 16 * 16 * MB_TB);
  TAKE(tproj, uint32_t, 16 * 256);
  TAKE(wfc1, mb_hf, 64 * 16 * MB_TB);
  TAKE(tfc1, uint32_t, 64 * 256);
  TAKE(wfc2, mb_hf, 16 * 64 * MB_TB);
  TAKE(tfc2, uint32_t, 16 * 256);
  TAKE(kt, mb_hf, 16 * 7 * MB_TB);
  TAKE(vt, mb_hf, 16 * 7 * MB_TB);
  TAKE(ln1, mb_hf, 2 * MB_D * 2);
  TAKE(ln2, mb_hf, 2 * MB_D * 2);
#undef TAKE
}

typedef struct {
  const uint8_t* a;
  size_t a_stride;
  int rb0, nrb; /* row blocks rb0 .. rb0 + nrb - 1 */
  int kt;
  const mb_hf* w; /* kt weight tiles (DDR) */
  const void* tab; /* 256 B column table, 0 = zeros */
  uint8_t* c;
  size_t c_stride;
  int mode, ph;
} mb_job;

typedef struct {
  uint8_t *x, *h, *o, *qkv[2], *s[2], *hid, *wbuf[2], *tbuf[2], *tzero, *ts6, *ident, *pv[2];
  volatile int ctr[64]; /* per-phase work counters (atomic), zeroed by thread 0 at block start */
  volatile int ready, consumed; /* job sequence counters: copied into wbuf[g % 2] / done by HMX */
  int seq0, seq1;                 /* next job number of thread 0 / of the copier (each counts its own) */
  int rt;  /* row tiles (Q / 32) */
  int hvx; /* MB_V_* */
  int nthr;
  void (*sync)(void*); /* barrier over the nthr threads (unused when nthr == 1) */
  void* sync_arg;
  float* pself; /* Q floats (scalar softmax) */
  unsigned long long prof[MB_NPROF]; /* thread 0's time */
} mb_ctx;

static inline void mb_layout(mb_ctx* c, uint8_t* vtcm, int q, float* pself) {
  const size_t M1 = 1u << 20;
  c->rt = q / 32;
  c->x = vtcm;
  c->h = vtcm + M1;
  c->o = vtcm + 2 * M1;
  c->qkv[0] = vtcm + 3 * M1;               /* <= 192 KB each */
  c->qkv[1] = vtcm + 3 * M1 + (1u << 18);
  c->s[0] = vtcm + 3 * M1 + (1u << 19);     /* <= 512 KB each */
  c->s[1] = vtcm + 4 * M1;
  c->hid = vtcm + 2 * M1;
  c->wbuf[0] = vtcm + 6 * M1;        /* 2 x 128 KB */
  c->wbuf[1] = c->wbuf[0] + (1u << 17);
  c->tbuf[0] = c->wbuf[0] + (1u << 18); /* 2 x 256 B */
  c->tbuf[1] = c->tbuf[0] + 256;
  c->tzero = c->tbuf[0] + 512;
  c->ts6 = c->tbuf[0] + 768;          /* S column block 6: columns 197..223 (j >= 5) biased to -65504 */
  c->ident = c->tbuf[0] + MB_TB;      /* identity weight tile (residual adds), 2 KB aligned */
  c->pv[0] = c->tbuf[0] + 2 * MB_TB;  /* p_self, one tile per row block (HVX softmax), x 2 */
  c->pv[1] = c->pv[0] + 32 * MB_TB;
  c->ready = c->consumed = -1;
  memset(c->tzero, 0, 256);
  for (int j = 0; j < 64; j++) ((uint32_t*)c->ts6)[j] = j >= 5 && j < 32 ? 0xFBFFu << 16 : 0;
  memset(c->ident, 0, MB_TB);
  for (int i = 0; i < 32; i++) ((mb_hf*)c->ident)[HMX_IDX(i, i)] = 0x3C00;
  c->hvx = 0;
  c->nthr = 1;
  c->seq0 = c->seq1 = 0;
  c->sync = 0;
  c->pself = pself;
  memset(c->prof, 0, sizeof c->prof);
}

static inline void mb_sync(mb_ctx* c) {
  if (c->nthr > 1) {
    unsigned long long t = MB_NOW();
    c->sync(c->sync_arg);
    c->prof[MB_P_WAIT] += MB_NOW() - t; /* all threads add, thread 0's view is what prints (roughly) */
  }
}

/* ---- scalar element access in the tile layout ---- */
static inline float mb_h2f(mb_hf h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
static inline mb_hf mb_f2h(float f) { __fp16 x = (__fp16)f; mb_hf h; memcpy(&h, &x, 2); return h; }
static inline int mb_idx(int i, int j) { return 64 * (i / 2) + 2 * j + (i % 2); }
/* element (r, c) of a tiled matrix whose row block spans `stride` tiles */
static inline mb_hf* mb_at(uint8_t* base, int stride, int r, int c) {
  return (mb_hf*)(base + ((size_t)(r / 32) * stride + c / 32) * MB_TB) + mb_idx(r % 32, c % 32);
}

static inline float mb_gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }
static void mb_tile_gelu(uint8_t* dst) {
  mb_hf* d = (mb_hf*)dst;
  for (int i = 0; i < MB_TH; i++) d[i] = mb_f2h(mb_gelu(mb_h2f(d[i])));
}

/* H = LayerNorm(X) (eps 1e-6), gamma / beta = ln[0..511] / ln[512..1023] */
static void mb_layernorm(mb_ctx* c, const mb_hf* ln) {
  for (int r = 0; r < c->rt * 32; r++) {
    float s = 0, s2 = 0, v[MB_D];
    for (int j = 0; j < MB_D; j++) {
      v[j] = mb_h2f(*mb_at(c->x, MB_KT, r, j));
      s += v[j];
    }
    const float mean = s / MB_D;
    for (int j = 0; j < MB_D; j++) s2 += (v[j] - mean) * (v[j] - mean);
    const float rstd = 1.f / sqrtf(s2 / MB_D + 1e-6f);
    for (int j = 0; j < MB_D; j++) *mb_at(c->h, MB_KT, r, j) = mb_f2h((v[j] - mean) * rstd * mb_h2f(ln[j]) + mb_h2f(ln[MB_D + j]));
  }
}

/* row-wise softmax of a head's scores S (+ the self score) in place -> P; pself[r] (row block rb) */
static void mb_softmax_rb(mb_ctx* c, uint8_t* qkv, uint8_t* s, int rb) {
  const int rt = c->rt;
  for (int r = rb * 32; r < rb * 32 + 32; r++) {
    float ss = 0, e[MB_SEEN];
    for (int d = 0; d < 32; d++) ss += mb_h2f(*mb_at(qkv, 1, r, d)) * mb_h2f(*mb_at(qkv + (size_t)rt * MB_TB, 1, r, d));
    ss *= MB_SCALE2;
    float mx = ss;
    for (int j = 0; j < MB_SEEN; j++) {
      e[j] = mb_h2f(*mb_at(s, 8, r, j));
      mx = e[j] > mx ? e[j] : mx;
    }
    float sum = exp2f(ss - mx);
    const float es = sum;
    for (int j = 0; j < MB_SEEN; j++) sum += (e[j] = exp2f(e[j] - mx));
    const float inv = 1.f / sum;
    for (int j = 0; j < MB_ST * 32; j++) *mb_at(s, 8, r, j) = mb_f2h(j < MB_SEEN ? e[j] * inv : 0.f);
    c->pself[r] = es * inv;
  }
}

/* O[:, head h] += pself * v_h */
static void mb_self_term(mb_ctx* c, const uint8_t* qkv, int h) {
  for (int r = 0; r < c->rt * 32; r++)
    for (int d = 0; d < 32; d++) {
      mb_hf* o = mb_at(c->o, MB_KT, r, h * 32 + d);
      *o = mb_f2h(mb_h2f(*o) + c->pself[r] * mb_h2f(*mb_at((uint8_t*)qkv + (size_t)2 * c->rt * MB_TB, 1, r, d)));
    }
}

#ifdef __HVX__
#include "mb_hvx.h"
#define MB_HVX_ON(c, f) ((c)->hvx & (f))
#else
#define MB_HVX_ON(c, f) 0
#endif

/* prefetch the next weight column block (DDR -> L2) while this one runs */
static inline void mb_prefetch(const void* p, size_t n) {
#ifdef __HVX__
  if (p) hmx_l2fetch(p, n);
#else
  (void)p, (void)n;
#endif
}

enum { MB_STORE, MB_RESID };

static inline void mb_pause(void) {
#ifdef __hexagon__
  __asm__ volatile("pause(#16)" ::: "memory");
#endif
}

static void mb_copy_job(mb_ctx* x, const mb_job* j, int buf) {
#ifdef __HVX__
  if (MB_HVX_ON(x, MB_V_COPY))
    hmx_copy_hvx(x->wbuf[buf], j->w, (size_t)j->kt * MB_TB);
  else
#endif
    memcpy(x->wbuf[buf], j->w, (size_t)j->kt * MB_TB);
  memcpy(x->tbuf[buf], j->tab ? j->tab : x->tzero, 256);
}

/* thread 0: job j's HMX work from buffer buf. C[:, cb] = A . W_cb + table (+ the C tile itself for
 * MB_RESID: its tile times the identity is one more K tile) */
static void mb_hmx_job(mb_ctx* x, const mb_job* j, int buf) {
#ifdef __hexagon__
  hmx_blk_set_table(x->tbuf[buf]);
  for (int rb = j->rb0; rb < j->rb0 + j->nrb; rb++) {
    hmx_blk_mac_f16(j->a + rb * j->a_stride, x->wbuf[buf], j->kt);
    if (j->mode == MB_RESID) hmx_blk_mac_f16(j->c + rb * j->c_stride, x->ident, 1);
    hmx_blk_store_f16(j->c + rb * j->c_stride);
  }
#else /* host builds (the app's ARM side) only use the packing helpers */
  (void)x, (void)j, (void)buf;
#endif
}

/* A phase of n jobs. Thread 0 runs the HMX side; with >= 2 threads thread 1 copies ahead (double
 * buffer), otherwise thread 0 copies each job itself. Other threads return at once. */
static void mb_jobs(mb_ctx* x, const mb_job* jobs, int n, int tid) {
  const int piped = x->nthr > 1;
  if (tid == 1 && piped) {
    for (int i = 0; i < n; i++, x->seq1++) {
      const int g = x->seq1;
      while (x->consumed < g - 2) mb_pause(); /* buffer g % 2 is free once job g - 2 is done */
      unsigned long long t = MB_NOW();
      mb_copy_job(x, &jobs[i], g & 1);
      if (i + 1 < n) mb_prefetch(jobs[i + 1].w, (size_t)jobs[i + 1].kt * MB_TB);
      x->prof[MB_P_WCOPY] += MB_NOW() - t;
#ifdef __hexagon__
      __asm__ volatile("syncht" ::: "memory"); /* the copy's stores land before HMX (thread 0) reads them */
#endif
      x->ready = g;
    }
    return;
  }
  if (tid) return;
  for (int i = 0; i < n; i++) {
    const int g = x->seq0++;
    unsigned long long t = MB_NOW();
    if (piped) {
      while (x->ready < g) mb_pause();
      x->prof[MB_P_WAIT] += MB_NOW() - t;
    } else {
      mb_copy_job(x, &jobs[i], g & 1);
      if (i + 1 < n) mb_prefetch(jobs[i + 1].w, (size_t)jobs[i + 1].kt * MB_TB);
      x->prof[MB_P_WCOPY] += MB_NOW() - t;
    }
    t = MB_NOW();
    mb_hmx_job(x, &jobs[i], g & 1);
    x->prof[jobs[i].ph] += MB_NOW() - t;
#ifdef __hexagon__
    __asm__ volatile("syncht" ::: "memory"); /* HMX has read buffer g % 2 before the copier refills it */
#endif
    x->consumed = g;
  }
}

/* the jobs of a whole GEMM, column block by column block: W = ncb blocks of kt tiles (+ tables) */
static int mb_gemm_jobs(mb_job* out, const uint8_t* a, size_t a_stride, int rb0, int nrb, int kt, const mb_hf* w, const uint32_t* tab,
                        int ncb, uint8_t* c, size_t c_stride, int mode, int ph) {
  for (int cb = 0; cb < ncb; cb++)
    out[cb] = (mb_job){a, a_stride, rb0, nrb, kt, w + (size_t)cb * kt * MB_TH, tab + cb * 64, c + (size_t)cb * MB_TB, c_stride, mode, ph};
  return ncb;
}

/* next work item of phase `ph` (atomic) */
static inline int mb_take(mb_ctx* c, int ph) {
#ifdef __hexagon__
  return __atomic_fetch_add((int*)&c->ctr[ph], 1, __ATOMIC_RELAXED);
#else
  return c->ctr[ph]++;
#endif
}
#define MB_TIMED(c, tid, ph, call)                    \
  do {                                                \
    unsigned long long _t = MB_NOW();                 \
    call;                                             \
    if (!(tid)) (c)->prof[ph] += MB_NOW() - _t;       \
  } while (0)

static void mb_ln(mb_ctx* c, const mb_hf* ln, int tid) {
#ifdef __HVX__
  if (MB_HVX_ON(c, MB_V_LN)) {
    mbv_layernorm(c->x, c->h, c->rt, ln, tid, c->nthr);
    return;
  }
#endif
  if (!tid) mb_layernorm(c, ln);
}

/* softmax of one row block (HVX or scalar) */
static void mb_softmax_one(mb_ctx* c, uint8_t* qkv, uint8_t* s, uint8_t* pv, int rb) {
#ifdef __HVX__
  if (MB_HVX_ON(c, MB_V_SM)) {
    mbv_softmax(s, qkv, qkv + (size_t)c->rt * MB_TB, pv, rb + 1, rb, 1 << 30);
    return;
  }
#endif
  mb_softmax_rb(c, qkv, s, rb);
}
static void mb_gelu_rb(mb_ctx* c, int rb) {
  for (int cb = 0; cb < MB_HT; cb++) {
    uint8_t* tile = c->hid + ((size_t)rb * MB_HT + cb) * MB_TB;
#ifdef __HVX__
    if (MB_HVX_ON(c, MB_V_GELU))
      mbv_tile_gelu(tile);
    else
#endif
      mb_tile_gelu(tile);
  }
}

/* jobs: head h's q / k / v column blocks and its S = q K^T (buffer b) */
static int mb_qkvs_jobs(mb_ctx* c, const mb_weights* w, int h, int b, mb_job* jobs) {
  const int rt = c->rt;
  int n = 0;
  for (int part = 0; part < 3; part++) {
    const int cb = part * 16 + h;
    jobs[n++] = (mb_job){c->h, (size_t)MB_KT * MB_TB, 0, rt, MB_KT, w->wqkv + (size_t)cb * MB_KT * MB_TH, w->tqkv + cb * 64,
                         c->qkv[b] + (size_t)part * rt * MB_TB, MB_TB, MB_STORE, MB_P_QKV};
  }
  for (int cb = 0; cb < MB_ST; cb++)
    jobs[n++] = (mb_job){c->qkv[b], MB_TB, 0, rt, 1, w->kt + ((size_t)h * MB_ST + cb) * MB_TH, cb == MB_ST - 1 ? c->ts6 : 0,
                         c->s[b] + (size_t)cb * MB_TB, (size_t)8 * MB_TB, MB_STORE, MB_P_S};
  return n;
}

/* one decoder block on X (in place); every thread calls it with its tid */
static void mb_block(mb_ctx* c, const mb_weights* w, int tid) {
  const int rt = c->rt, half = rt / 2;
  const size_t row16 = (size_t)MB_KT * MB_TB, row64 = (size_t)MB_HT * MB_TB;
  mb_job jobs[MB_HT];
  if (!tid) memset((void*)c->ctr, 0, sizeof c->ctr);
  MB_TIMED(c, tid, MB_P_LN, mb_ln(c, w->ln1, tid));
  mb_sync(c);
  mb_jobs(c, jobs, mb_qkvs_jobs(c, w, 0, 0, jobs), tid);
  mb_sync(c);
  for (int k = 0; k <= MB_HEADS; k++) { /* phase k: finish head k-1, prepare head k+1, softmax of head k */
    if (k >= 1) {
      const int b = (k - 1) & 1;
      jobs[0] = (mb_job){c->s[b], (size_t)8 * MB_TB, 0, rt, MB_ST, w->vt + (size_t)(k - 1) * MB_ST * MB_TH, 0, c->o + (size_t)(k - 1) * MB_TB,
                         row16, MB_STORE, MB_P_PV};
      mb_jobs(c, jobs, 1, tid);
      if (!tid) {
#ifdef __HVX__
        if (MB_HVX_ON(c, MB_V_SM))
          MB_TIMED(c, tid, MB_P_SELF, mbv_self_term(c->o, c->qkv[b] + (size_t)2 * rt * MB_TB, c->pv[b], rt, k - 1));
        else
#endif
          MB_TIMED(c, tid, MB_P_SELF, mb_self_term(c, c->qkv[b], k - 1));
      }
    }
    if (k + 1 < MB_HEADS) mb_jobs(c, jobs, mb_qkvs_jobs(c, w, k + 1, (k + 1) & 1, jobs), tid);
    if (k < MB_HEADS) {
      const int b = k & 1;
      unsigned long long t = MB_NOW();
      if (MB_HVX_ON(c, MB_V_SM) || !tid)
        for (int rb; (rb = mb_take(c, k)) < rt;) mb_softmax_one(c, c->qkv[b], c->s[b], c->pv[b], rb);
      if (!tid) c->prof[MB_P_SOFTMAX] += MB_NOW() - t;
    }
    mb_sync(c);
  }
  mb_jobs(c, jobs, mb_gemm_jobs(jobs, c->o, row16, 0, rt, MB_KT, w->wproj, w->tproj, MB_KT, c->x, row16, MB_RESID, MB_P_PROJ), tid);
  mb_sync(c);
  MB_TIMED(c, tid, MB_P_LN, mb_ln(c, w->ln2, tid));
  mb_sync(c);
  /* MLP in two row halves: A fc1(0) | B fc1(1) + GELU(0) | C fc2(0) + GELU(1) | D fc2(1) */
  mb_jobs(c, jobs, mb_gemm_jobs(jobs, c->h, row16, 0, half, MB_KT, w->wfc1, w->tfc1, MB_HT, c->hid, row64, MB_STORE, MB_P_FC1), tid);
  mb_sync(c);
  for (int ph = 0; ph < 2; ph++) {
    if (ph == 0)
      mb_jobs(c, jobs, mb_gemm_jobs(jobs, c->h, row16, half, rt - half, MB_KT, w->wfc1, w->tfc1, MB_HT, c->hid, row64, MB_STORE, MB_P_FC1), tid);
    else
      mb_jobs(c, jobs, mb_gemm_jobs(jobs, c->hid, row64, 0, half, MB_HT, w->wfc2, w->tfc2, MB_KT, c->x, row16, MB_RESID, MB_P_FC2), tid);
    unsigned long long t = MB_NOW();
    const int r0 = ph ? half : 0, n = ph ? rt - half : half;
    if (MB_HVX_ON(c, MB_V_GELU) || !tid)
      for (int i; (i = mb_take(c, 32 + ph)) < n;) mb_gelu_rb(c, r0 + i);
    if (!tid) c->prof[MB_P_GELU] += MB_NOW() - t;
    mb_sync(c);
  }
  mb_jobs(c, jobs, mb_gemm_jobs(jobs, c->hid, row64, half, rt - half, MB_HT, w->wfc2, w->tfc2, MB_KT, c->x, row16, MB_RESID, MB_P_FC2), tid);
  mb_sync(c);
}
#endif
