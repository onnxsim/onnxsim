/* The whole MCC query decoder for a chunk of Q queries on the DSP (model.py QueryDecoder), around
 * mcc_block.h's blocks:
 *
 *   x = xyz Wpos + b                (HMX, one K tile: xyz padded 3 -> 32; upstream's shrink is the
 *                                   identity on the query grid: |xyz| <= 3 sqrt(3) < 10)
 *   x = block_7(... block_0(x))     (mcc_block.h)
 *   y = LN_f(x) Wpred + b           (HMX; columns reordered, ref.py PRED_ORDER: color channel c's 256
 *                                   logits = 8 whole tiles at 256 c, the occupancy logit at 768)
 *   occ = y[:, 768]; rgb_c = sum_l softmax(y_c / 0.1)_l * l / 255   (HVX, per row, base 2)
 *
 * K / V of the seen tokens come per image from the encoder (float [8, 16, 197, 32] each) and are packed
 * once (mb_pack_kv) into every resident block's kt / vt slots (K pre-scaled by scale * log2(e)). */
#ifndef MCC_DECODER_H
#define MCC_DECODER_H
#include "mcc_block.h"

#define MB_BLOCKS 8
#define MB_PT 25 /* 800 / 32 prediction column tiles */
#define MB_OCC_COL 768

typedef struct {
  const mb_hf *wpos, *lnf, *wpred;
  const uint32_t *tpos, *tpred;
} mb_head;

static inline size_t mb_head_bytes(void) { return (size_t)(16 + MB_PT * 16) * MB_TB + (16 + MB_PT) * 256 + 2 * MB_D * 2; }

static inline void mb_bind_head(mb_head* hd, const void* blob) {
  const uint8_t* p = (const uint8_t*)blob;
  hd->wpos = (const mb_hf*)p, p += 16 * MB_TB;
  hd->tpos = (const uint32_t*)p, p += 16 * 256;
  hd->lnf = (const mb_hf*)p, p += 2 * MB_D * 2;
  hd->wpred = (const mb_hf*)p, p += MB_PT * 16 * MB_TB;
  hd->tpred = (const uint32_t*)p;
}

/* K / V (float, [8 blocks][16 heads][197][32] each) -> every block's kt / vt tiles (writable blob slots) */
static void mb_pack_kv(mb_hf* const kt[MB_BLOCKS], mb_hf* const vt[MB_BLOCKS], const float* k, const float* v) {
  for (int b = 0; b < MB_BLOCKS; b++)
    for (int h = 0; h < MB_HEADS; h++) {
      const float *kh = k + ((size_t)b * MB_HEADS + h) * MB_SEEN * 32, *vh = v + ((size_t)b * MB_HEADS + h) * MB_SEEN * 32;
      mb_hf *kp = kt[b] + (size_t)h * MB_ST * MB_TH, *vp = vt[b] + (size_t)h * MB_ST * MB_TH;
      memset(kp, 0, (size_t)MB_ST * MB_TB);
      memset(vp, 0, (size_t)MB_ST * MB_TB);
      for (int t = 0; t < MB_SEEN; t++)
        for (int d = 0; d < 32; d++) {
          kp[(size_t)(t / 32) * MB_TH + mb_idx(d, t % 32)] = mb_f2h(kh[t * 32 + d] * MB_SCALE2); /* W(d, token) */
          vp[(size_t)(t / 32) * MB_TH + mb_idx(t % 32, d)] = mb_f2h(vh[t * 32 + d]);             /* W(token, d) */
        }
    }
}

/* the kt / vt slots inside a block blob (ref.py BLOCK_LAYOUT) */
static inline mb_hf* mb_blob_kt(void* blob) { return (mb_hf*)((uint8_t*)blob + (size_t)(48 * 16 + 16 * 16 + 64 * 16 + 16 * 64) * MB_TB + (48 + 16 + 64 + 16) * 256); }
static inline mb_hf* mb_blob_vt(void* blob) { return mb_blob_kt(blob) + (size_t)16 * MB_ST * MB_TH; }

/* scalar color head for one row block (reference / fallback) */
static void mb_color_rb(const uint8_t* y, int rb, float* occ, float* rgb) {
  for (int r = rb * 32; r < rb * 32 + 32; r++) {
    occ[r] = mb_h2f(*mb_at((uint8_t*)y, 32, r, MB_OCC_COL));
    for (int ch = 0; ch < 3; ch++) {
      float mx = -1e30f, s = 0, sl = 0;
      for (int l = 0; l < 256; l++) mx = fmaxf(mx, mb_h2f(*mb_at((uint8_t*)y, 32, r, 256 * ch + l)));
      for (int l = 0; l < 256; l++) {
        const float e = expf((mb_h2f(*mb_at((uint8_t*)y, 32, r, 256 * ch + l)) - mx) * 10.f);
        s += e, sl += e * l / 255.f;
      }
      rgb[r * 3 + ch] = sl / s;
    }
  }
}

#ifdef __HVX__
/* HVX color head for one row block: per row pair and channel, max over the 8 tiles, e = 2^((y - max) *
 * 10 log2(e) + 7) in hf (clamped at -13), sum e and sum e * level in qf32, rgb = ratio; occ read out. */
static void mbv_color_rb(const uint8_t* y, int rb, float* occ, float* rgb) {
  static const float lvf = 1.f / 255.f;
  mbV lev[8];
  for (int tb = 0; tb < 8; tb++) {
    mb_hf l[64] __attribute__((aligned(128)));
    for (int i = 0; i < 64; i++) l[i] = mb_f2h((tb * 32 + i / 2) * lvf);
    lev[tb] = MBV(l);
  }
  const mbV k = mbv_hsplat(14.426950f), seven = mbv_hsplat(7.f), lo = mbv_hsplat(-13.f), one = Q6_Vh_vsplat_R(0x3C00);
  float st[4][32] __attribute__((aligned(128)));
  for (int p = 0; p < 16; p++) {
    for (int ch = 0; ch < 3; ch++) {
      const uint8_t* base = y + ((size_t)rb * 32 + ch * 8) * MB_TB + 128 * p;
      mbV m = MBV(base);
      for (int tb = 1; tb < 8; tb++) m = Q6_Vhf_vmax_VhfVhf(m, MBV(base + tb * MB_TB));
      m = mbv_rowmax(m);
      mbV s0 = Q6_V_vzero(), s1 = s0, w0 = s0, w1 = s0;
      for (int t0 = 0; t0 < 8; t0 += MB_EXPN) {
        mbV t[MB_EXPN];
        for (int i = 0; i < MB_EXPN; i++)
          t[i] = Q6_Vhf_vmax_VhfVhf(MBV_HADD(MBV_HMUL(MBV_HSUB(MBV(base + (t0 + i) * MB_TB), m), k), seven), lo);
        mbv_hexp2_n(t);
        for (int i = 0; i < MB_EXPN; i++) {
          mbW a = Q6_Wqf32_vmpy_VhfVhf(t[i], one), b = Q6_Wqf32_vmpy_VhfVhf(t[i], lev[t0 + i]);
          s0 = Q6_Vqf32_vadd_Vqf32Vqf32(s0, Q6_V_lo_W(a)), s1 = Q6_Vqf32_vadd_Vqf32Vqf32(s1, Q6_V_hi_W(a));
          w0 = Q6_Vqf32_vadd_Vqf32Vqf32(w0, Q6_V_lo_W(b)), w1 = Q6_Vqf32_vadd_Vqf32Vqf32(w1, Q6_V_hi_W(b));
        }
      }
      MBV(st[0]) = MBV_SF(mbv_lanesum(s0)), MBV(st[1]) = MBV_SF(mbv_lanesum(s1));
      MBV(st[2]) = MBV_SF(mbv_lanesum(w0)), MBV(st[3]) = MBV_SF(mbv_lanesum(w1));
      const int r = rb * 32 + 2 * p;
      rgb[r * 3 + ch] = st[2][0] / st[0][0];
      rgb[(r + 1) * 3 + ch] = st[3][0] / st[1][0];
    }
  }
  for (int r = rb * 32; r < rb * 32 + 32; r++) occ[r] = mb_h2f(*mb_at((uint8_t*)y, 32, r, MB_OCC_COL));
}
#endif

/* the whole decoder on one chunk; every thread calls it with its tid. xyz: Q x 3 floats; occ: Q; rgb: Q x 3 */
static void mb_decode(mb_ctx* c, const mb_head* hd, const mb_weights* w, const float* xyz, float* occ, float* rgb, int tid) {
  const int rt = c->rt;
  const size_t row16 = (size_t)MB_KT * MB_TB, row32 = (size_t)32 * MB_TB;
  mb_job jobs[MB_PT];
  uint8_t* a = c->hid; /* xyz A tiles (one per row block), then the prediction output (row block = 32 tiles) */
  if (!tid) {
    memset(a, 0, (size_t)rt * MB_TB);
    for (int r = 0; r < rt * 32; r++)
      for (int d = 0; d < 3; d++) *mb_at(a, 1, r, d) = mb_f2h(xyz[r * 3 + d]);
  }
  mb_sync(c);
  mb_jobs(c, jobs, mb_gemm_jobs(jobs, a, MB_TB, 0, rt, 1, hd->wpos, hd->tpos, MB_KT, c->x, row16, MB_STORE, MB_P_QKV), tid);
  mb_sync(c);
  for (int b = 0; b < MB_BLOCKS; b++) mb_block(c, &w[b], tid);
  if (!tid) memset((void*)c->ctr, 0, sizeof c->ctr);
  MB_TIMED(c, tid, MB_P_LN, mb_ln(c, hd->lnf, tid));
  mb_sync(c);
  mb_jobs(c, jobs, mb_gemm_jobs(jobs, c->h, row16, 0, rt, MB_KT, hd->wpred, hd->tpred, MB_PT, a, row32, MB_STORE, MB_P_FC2), tid);
  mb_sync(c);
  unsigned long long t = MB_NOW();
#ifdef __HVX__
  if (MB_HVX_ON(c, MB_V_SM))
    for (int rb; (rb = mb_take(c, 0)) < rt;) mbv_color_rb(a, rb, occ, rgb);
  else
#endif
    if (!tid)
      for (int rb = 0; rb < rt; rb++) mb_color_rb(a, rb, occ, rgb);
  if (!tid) c->prof[MB_P_SOFTMAX] += MB_NOW() - t;
  mb_sync(c);
}
#endif
