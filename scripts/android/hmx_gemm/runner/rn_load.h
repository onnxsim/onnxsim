/* Host side of the graph runner (portable C: the phone client and hexagon-sim programs): read qdq_graph.py's
 * program.txt + weights.bin, pack weights and requantization params, choose each tensor's flat geometry, build the
 * conv / maxpool sources and taps, and plan VTCM with liveness-based reuse. Build with -ffp-contract=off: the Add
 * constants must be ORT's exact fp32 values. */
#ifndef RN_LOAD_H
#define RN_LOAD_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rn_model.h"

#pragma clang fp contract(off) /* ORT's Add constants are separate fp32 multiplies and adds, never fused */

typedef struct {
  rn_model_t m;
  uint8_t* blob;
  size_t blob_len, blob_cap;
  char err[256];
} rn_build_t;

static uint32_t rn_put(rn_build_t* b, const void* p, size_t n) {
  size_t off = (b->blob_len + 2047) & ~(size_t)2047;
  if (off + n > b->blob_cap) {
    b->blob_cap = (off + n) * 2;
    b->blob = (uint8_t*)realloc(b->blob, b->blob_cap);
  }
  memset(b->blob + b->blob_len, 0, off - b->blob_len);
  memcpy(b->blob + off, p, n);
  b->blob_len = off + n;
  return (uint32_t)off;
}

/* union-find over tensors that must share a flat geometry */
static int rn_find(int* par, int i) { return par[i] == i ? i : (par[i] = rn_find(par, par[i])); }

static int rn_floordiv2(int v) { return v >= 0 ? v / 2 : -((-v + 1) / 2); }

/* the phase / shift sources and taps of a stride-s k x k window op reading tensor x into geometry gsrc */
static void rn_sources(rn_op_t* o, int k, int s) {
  int p = k / 2;
  o->nsrc = 0, o->ntaps = k * k;
  int idx[4][8]; /* [phase or 0][shift + 3] -> src index, -1 = input, -2 = not built yet */
  for (int a = 0; a < 4; a++)
    for (int c = 0; c < 8; c++) idx[a][c] = -2;
  if (s == 1) idx[0][3] = -1;
  for (int t = 0; t < k * k; t++) {
    int dy = t / k - p, dx = t % k - p, ph = 0, c, drow;
    if (s == 1)
      c = dx, drow = dy;
    else {
      int py = dy & 1, px = dx & 1;
      ph = 2 * py + px, drow = rn_floordiv2(dy), c = rn_floordiv2(dx);
      for (int q = 0; q < 2; q++) { /* the phase split builds (even, odd column) pairs */
        int pq = 2 * py + q;
        if (idx[pq][3] == -2) {
          o->src[o->nsrc].kind = RN_SRC_PHASE, o->src[o->nsrc].a = pq, o->src[o->nsrc].c = 0;
          idx[pq][3] = o->nsrc++;
        }
      }
    }
    if (idx[ph][c + 3] == -2) {
      o->src[o->nsrc].kind = RN_SRC_SHIFT, o->src[o->nsrc].a = idx[ph][3], o->src[o->nsrc].c = c;
      idx[ph][c + 3] = o->nsrc++;
    }
    o->tap_src[t] = (int8_t)idx[ph][c + 3], o->tap_drow[t] = (int8_t)drow;
  }
}

/* VTCM first-fit allocator over [begin, end) op intervals */
typedef struct {
  uint32_t off, len;
  int end; /* last op using it */
} rn_blk_t;
typedef struct {
  rn_blk_t b[512];
  int n;
  uint32_t peak;
} rn_alloc_t;
static uint32_t rn_alloc(rn_alloc_t* a, uint32_t len, int end, int now) {
  len = (len + 2047) & ~2047u;
  /* live blocks sorted by offset: find the first gap */
  for (int i = 0; i < a->n; i++)
    if (a->b[i].end < now) a->b[i] = a->b[--a->n], i--;
  for (int i = 0; i < a->n; i++)
    for (int j = i + 1; j < a->n; j++)
      if (a->b[j].off < a->b[i].off) {
        rn_blk_t t = a->b[i];
        a->b[i] = a->b[j], a->b[j] = t;
      }
  uint32_t off = 0;
  for (int i = 0; i < a->n; i++) {
    if (a->b[i].off >= off + len) break;
    if (a->b[i].off + a->b[i].len > off) off = a->b[i].off + a->b[i].len;
  }
  a->b[a->n].off = off, a->b[a->n].len = len, a->b[a->n].end = end, a->n++;
  if (off + len > a->peak) a->peak = off + len;
  return off;
}

static int rn_build(rn_build_t* B, const char* dir) {
  char p[512];
  memset(B, 0, sizeof *B);
  rn_model_t* m = &B->m;
  snprintf(p, sizeof p, "%s/weights.bin", dir);
  FILE* f = fopen(p, "rb");
  if (!f) return snprintf(B->err, sizeof B->err, "no %s", p), -1;
  fseek(f, 0, SEEK_END);
  long wl = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* W = (uint8_t*)malloc(wl);
  if (fread(W, 1, wl, f) != (size_t)wl) return snprintf(B->err, sizeof B->err, "short weights"), -1;
  fclose(f);
  snprintf(p, sizeof p, "%s/program.txt", dir);
  f = fopen(p, "r");
  if (!f) return snprintf(B->err, sizeof B->err, "no %s", p), -1;
  if (fscanf(f, "tensors %d ops %d input %d output %d", &m->nt, &m->nops, &m->input, &m->output) != 4 || m->nt > RN_MAX_T ||
      m->nops > RN_MAX_OPS)
    return snprintf(B->err, sizeof B->err, "bad program header"), -1;
  for (int i = 0; i < m->nt; i++) {
    int id, c, h, w, zp;
    char sc[64];
    if (fscanf(f, " T %d %d %d %d %63s %d", &id, &c, &h, &w, sc, &zp) != 6) return snprintf(B->err, sizeof B->err, "bad T line"), -1;
    rn_tensor_t* t = &m->t[id];
    t->c = c, t->cp = (c + 31) & ~31, t->h = h, t->w = w, t->zp = zp, t->scale = strtof(sc, NULL);
  }
  struct { uint32_t w, b, sw; int cin; } cv[RN_MAX_OPS];
  for (int i = 0; i < m->nops; i++) {
    char ty[16];
    rn_op_t* o = &m->op[i];
    if (fscanf(f, " %15s", ty) != 1) return snprintf(B->err, sizeof B->err, "bad op line"), -1;
    if (!strcmp(ty, "conv")) {
      o->type = RN_CONV;
      if (fscanf(f, "%d %d %d %d %d %d %u %u %u", &o->x, &o->y, &o->k, &o->s, &o->n, &cv[i].cin, &cv[i].w, &cv[i].b, &cv[i].sw) != 9)
        return -1;
      if (o->n % 64) return snprintf(B->err, sizeof B->err, "op %d: conv output channels %d not a multiple of 64", i, o->n), -1;
    } else if (!strcmp(ty, "add")) {
      o->type = RN_ADD;
      if (fscanf(f, "%d %d %d", &o->x, &o->x2, &o->y) != 3) return -1;
    } else if (!strcmp(ty, "maxpool")) {
      o->type = RN_MAXPOOL;
      if (fscanf(f, "%d %d %d %d", &o->x, &o->y, &o->k, &o->s) != 4) return -1;
      if (m->t[o->x].zp != 0) return snprintf(B->err, sizeof B->err, "op %d: MaxPool input zero point %d != 0", i, m->t[o->x].zp), -1;
    } else
      return snprintf(B->err, sizeof B->err, "op %d: unknown op %s", i, ty), -1;
  }
  fclose(f);
  /* geometry: P/Q per class of tensors that must share one */
  int par[RN_MAX_T], P[RN_MAX_T], Q[RN_MAX_T];
  for (int i = 0; i < m->nt; i++) par[i] = i, P[i] = Q[i] = 1;
  for (int i = 0; i < m->nops; i++) {
    rn_op_t* o = &m->op[i];
    if (o->type == RN_ADD) par[rn_find(par, o->x2)] = rn_find(par, o->x), par[rn_find(par, o->y)] = rn_find(par, o->x);
    if (o->type == RN_CONV && o->s == 1) par[rn_find(par, o->y)] = rn_find(par, o->x);
  }
  for (int i = 0; i < m->nops; i++) {
    rn_op_t* o = &m->op[i];
    int need = o->type == RN_ADD ? 1 : o->s == 1 ? o->k / 2 : (o->k / 2 + 1) / 2, r = rn_find(par, o->y);
    if (need > P[r]) P[r] = Q[r] = need;
  }
  for (int i = 0; i < m->nt; i++) {
    rn_tensor_t* t = &m->t[i];
    int r = rn_find(par, i);
    t->g = qc_geom2(t->h, t->w, P[r], Q[r]);
  }
  /* the graph input stays in DDR (the RPC buffer) when only stride-2 ops read it (their phase split reads any memory) */
  m->t[m->input].in_ddr = 1;
  for (int i = 0; i < m->nops; i++) {
    rn_op_t* o = &m->op[i];
    if ((o->x == m->input || (o->type == RN_ADD && o->x2 == m->input)) && (o->type == RN_ADD || o->s == 1)) m->t[m->input].in_ddr = 0;
  }
  /* ops: sources, packing, params */
  for (int i = 0; i < m->nops; i++) {
    rn_op_t* o = &m->op[i];
    rn_tensor_t *X = &m->t[o->x], *Y = &m->t[o->y];
    if (o->type == RN_ADD) {
      rn_tensor_t* X2 = &m->t[o->x2];
      volatile float ra = X->scale / Y->scale, rb = X2->scale / Y->scale;
      volatile float p1 = ra * (float)X->zp, p2 = rb * (float)X2->zp, s12 = p1 + p2;
      volatile float fixed = (float)Y->zp - s12;
      o->ra = ra, o->rb = rb, o->fixed = fixed;
      double bound = 255.0 * (ra + rb) + fabs(fixed) + 1;
      int F = (int)floor(log2(ldexp(1.0, 30) / bound));
      if (F > 21) F = 21;
      /* a * ra exactly: ra = ma * 2^ea with a 24-bit mantissa ma; a * ma < 2^32 is split in 12-bit halves */
      int ea, eb;
      double fa = frexp(ra, &ea), fb = frexp(rb, &eb); /* ra = fa * 2^ea, fa in [0.5, 1) */
      long ma = lrint(ldexp(fa, 24)), mb = lrint(ldexp(fb, 24));
      ea -= 24, eb -= 24; /* ra = ma * 2^ea */
      o->F = F, o->sa = -ea - F, o->sb = -eb - F;
      if (o->sa > 12 || o->sb > 12 || o->sa < 0 || o->sb < 0)
        return snprintf(B->err, sizeof B->err, "op %d: Add scale ratios %g / %g out of the supported range", i, ra, rb), -1;
      o->a_hi = (int32_t)(ma >> 12), o->a_lo = (int32_t)(ma & 4095), o->b_hi = (int32_t)(mb >> 12), o->b_lo = (int32_t)(mb & 4095);
      o->fq = (int32_t)lrint(ldexp(fixed, F));
      /* our error: two truncating shifts per input + the rounding of fq (< 3 units); ORT's error: its four fp32
       * roundings (half an ulp each of |ra*a|, |ra*a + fixed|, |rb*b| and |v|) */
      double ulp_ra = ldexp(1.0, (int)floor(log2(255.0 * ra)) - 23), ulp_rb = ldexp(1.0, (int)floor(log2(255.0 * rb)) - 23);
      double ulp_t = ldexp(1.0, (int)floor(log2(255.0 * ra + fabs(fixed) + 1)) - 23), ulp_v = ldexp(1.0, (int)floor(log2(bound)) - 23);
      o->win = (int32_t)ceil(3.0 + ldexp(0.5 * (ulp_ra + ulp_rb + ulp_t + ulp_v), F)) + 2;
      continue;
    }
    o->gsrc = Y->g;
    if (o->type == RN_MAXPOOL) {
      rn_sources(o, o->k, o->s);
      continue;
    }
    /* conv */
    int k = o->k, cin = cv[i].cin, cp = X->cp, n = o->n, kk = k * k;
    if (cin != X->c) return snprintf(B->err, sizeof B->err, "op %d: weight channels %d != input %d", i, cin, X->c), -1;
    if (o->s == 1 && (X->g.P != Y->g.P || X->g.Wp != Y->g.Wp)) return snprintf(B->err, sizeof B->err, "op %d: geometry mismatch", i), -1;
    rn_sources(o, k, o->s);
    int8_t* wk = (int8_t*)malloc((size_t)kk * cp * n);
    int8_t* wp = (int8_t*)malloc((size_t)kk * cp * n);
    qc_pack_wk((const int8_t*)(W + cv[i].w), n, cin, k, cp, wk, wp);
    size_t plen = sizeof(qc_blk_t) * (n / 32) + sizeof(qc_hdr_t);
    uint8_t* prm = (uint8_t*)aligned_alloc(256, (plen + 255) & ~(size_t)255);
    qc_pack_params(wk, kk * cp, n, (const int32_t*)(W + cv[i].b), X->zp, X->scale, (const float*)(W + cv[i].sw), Y->scale, Y->zp, 0,
                   (qc_blk_t*)prm, (qc_hdr_t*)(prm + sizeof(qc_blk_t) * (n / 32)));
    o->w_len = (uint32_t)((size_t)kk * cp * n), o->w_off = rn_put(B, wp, o->w_len);
    o->prm_len = (uint32_t)plen, o->prm_off = rn_put(B, prm, plen);
    free(wk), free(wp), free(prm);
  }
  free(W);
  /* VTCM plan */
  int last[RN_MAX_T];
  for (int i = 0; i < m->nt; i++) last[i] = -1;
  for (int i = 0; i < m->nops; i++) {
    rn_op_t* o = &m->op[i];
    last[o->x] = i;
    if (o->type == RN_ADD) last[o->x2] = i;
  }
  last[m->output] = m->nops;
  rn_alloc_t al;
  memset(&al, 0, sizeof al);
  /* conv j's weight buffers live from the previous conv i (prefetched while ops i .. j-1 run): allocate them at i */
  int prevconv = -1;
  for (int i = 0; i < m->nops; i++) m->op[i].prefetch_next = -1;
  for (int i = 0; i < m->nops; i++)
    if (m->op[i].type == RN_CONV) {
      if (prevconv >= 0) m->op[prevconv].prefetch_next = i, m->op[i].prefetched = 1;
      prevconv = i;
    }
  if (!m->t[m->input].in_ddr)
    m->t[m->input].off = rn_alloc(&al, (uint32_t)qc_geom_bytes(&m->t[m->input].g, m->t[m->input].cp / 32), last[m->input], 0);
  for (int i = 0; i < m->nops; i++) {
    rn_op_t* o = &m->op[i];
    rn_tensor_t* Y = &m->t[o->y];
    Y->off = rn_alloc(&al, (uint32_t)qc_geom_bytes(&Y->g, Y->cp / 32), last[o->y], i);
    int kt = m->t[o->x].cp / 32;
    for (int s = 0; s < o->nsrc; s++) o->src[s].off = rn_alloc(&al, (uint32_t)qc_geom_bytes(&o->gsrc, kt), i, i);
    if (o->type == RN_CONV) {
      /* weights + params are allocated from the previous conv on: the executor may prefetch them during that op */
      if (!o->prefetched) {
        o->vw = rn_alloc(&al, o->w_len, i, i);
        o->vprm = rn_alloc(&al, o->prm_len, i, i);
      }
      o->vplanes = rn_alloc(&al, 4 * 2048, i, i);
      size_t span = qc_geom_bytes(&o->gsrc, kt);
      o->side_cap = o->ntaps * kt * (int)(span / 262144 + 2);
      if (o->side_cap > 1024) o->side_cap = 1024;
      o->vside = rn_alloc(&al, (uint32_t)o->side_cap * 2048, i, i);
      if (o->prefetch_next >= 0) {
        rn_op_t* nx = &m->op[o->prefetch_next];
        nx->vw = rn_alloc(&al, nx->w_len, o->prefetch_next, i);
        nx->vprm = rn_alloc(&al, nx->prm_len, o->prefetch_next, i);
      }
    }
  }
  m->vtcm_bytes = al.peak, m->blob_bytes = (uint32_t)B->blob_len;
  return 0;
}

/* NHWC uint8 [H, W, C] graph input -> its flat buffer (Cp channels, zero point elsewhere) */
static uint8_t* rn_pack_input(const rn_model_t* m, const uint8_t* x, size_t* len) {
  const rn_tensor_t* t = &m->t[m->input];
  int kt = t->cp / 32;
  *len = qc_geom_bytes(&t->g, kt);
  /* page-aligned: FastRPC keeps the offset within a page, so the DSP sees a 128-byte-aligned buffer (aligned loads) */
  uint8_t* out = (uint8_t*)aligned_alloc(4096, (*len + 4095) & ~(size_t)4095);
  memset(out, t->zp, *len);
  for (int y = 0; y < t->h; y++)
    for (int xx = 0; xx < t->w; xx++) {
      int p = qc_geom_pix(&t->g, y, xx);
      for (int c = 0; c < t->c; c++) out[((size_t)(p / 64) * kt + c / 32) * 2048 + 32 * (p % 64) + c % 32] = x[((size_t)y * t->w + xx) * t->c + c];
    }
  return out;
}
/* the output tensor's flat buffer -> NCHW uint8 */
static void rn_unpack_output(const rn_model_t* m, const uint8_t* buf, uint8_t* y) {
  const rn_tensor_t* t = &m->t[m->output];
  int kt = t->cp / 32;
  for (int c = 0; c < t->c; c++)
    for (int yy = 0; yy < t->h; yy++)
      for (int x = 0; x < t->w; x++) {
        int p = qc_geom_pix(&t->g, yy, x);
        y[((size_t)c * t->h + yy) * t->w + x] = buf[((size_t)(p / 64) * kt + c / 32) * 2048 + 32 * (p % 64) + c % 32];
      }
}
#endif
