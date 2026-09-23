/* RoiAlign (ONNX opset <16 semantics, mode=avg) over a channels-last (H, W, C) fp32 feature map.
 *
 * Data dependence is only in *which pixel* each bilinear tap reads; every tap reads C contiguous
 * floats (C/32 whole 128-byte HVX vectors). So instead of a per-lane gather (vgathermh: 16-bit
 * lanes, VTCM-only source), the scalar core computes each tap's row/col + weight and the HVX unit
 * does plain aligned vector loads and a 4-tap FMA across channels. Output is channels-last
 * (R, OH, OW, C). Header-only so the qemu harness and the FastRPC skel compile the same code. */
#ifndef ROIALIGN_KERNEL_H
#define ROIALIGN_KERNEL_H

typedef float roi_f32x32 __attribute__((vector_size(128)));

#define ROI_MAX_NV 8 /* C <= 256 */

static inline float roi_maxf(float a, float b) { return a > b ? a : b; }

static void roialign_hwc(const float* feat, int H, int W, int C, const float* rois, int R, int OH,
                         int OW, int sr, float scale, float* out) {
  const int nv = C / 32;
  const float inv_count = 1.0f / (float)(sr * sr);
  for (int r = 0; r < R; r++) {
    const float x1 = rois[4 * r + 0] * scale, y1 = rois[4 * r + 1] * scale;
    const float x2 = rois[4 * r + 2] * scale, y2 = rois[4 * r + 3] * scale;
    const float bin_w = roi_maxf(x2 - x1, 1.0f) / (float)OW;
    const float bin_h = roi_maxf(y2 - y1, 1.0f) / (float)OH;
    for (int ph = 0; ph < OH; ph++) {
      for (int pw = 0; pw < OW; pw++) {
        roi_f32x32 acc[ROI_MAX_NV];
        for (int v = 0; v < nv; v++) acc[v] = (roi_f32x32){0};
        for (int iy = 0; iy < sr; iy++) {
          float y = y1 + ph * bin_h + (iy + 0.5f) * bin_h / (float)sr;
          for (int ix = 0; ix < sr; ix++) {
            float x = x1 + pw * bin_w + (ix + 0.5f) * bin_w / (float)sr;
            if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) continue;
            float yy = y <= 0.0f ? 0.0f : y, xx = x <= 0.0f ? 0.0f : x;
            int y_lo = (int)yy, x_lo = (int)xx, y_hi, x_hi;
            if (y_lo >= H - 1) { y_hi = y_lo = H - 1; yy = (float)y_lo; } else { y_hi = y_lo + 1; }
            if (x_lo >= W - 1) { x_hi = x_lo = W - 1; xx = (float)x_lo; } else { x_hi = x_lo + 1; }
            const float ly = yy - y_lo, lx = xx - x_lo, hy = 1.0f - ly, hx = 1.0f - lx;
            const float w1 = hy * hx * inv_count, w2 = hy * lx * inv_count;
            const float w3 = ly * hx * inv_count, w4 = ly * lx * inv_count;
            const roi_f32x32* p1 = (const roi_f32x32*)(feat + ((long)y_lo * W + x_lo) * C);
            const roi_f32x32* p2 = (const roi_f32x32*)(feat + ((long)y_lo * W + x_hi) * C);
            const roi_f32x32* p3 = (const roi_f32x32*)(feat + ((long)y_hi * W + x_lo) * C);
            const roi_f32x32* p4 = (const roi_f32x32*)(feat + ((long)y_hi * W + x_hi) * C);
            for (int v = 0; v < nv; v++) acc[v] += p1[v] * w1 + p2[v] * w2 + p3[v] * w3 + p4[v] * w4;
          }
        }
        roi_f32x32* o = (roi_f32x32*)(out + (((long)r * OH + ph) * OW + pw) * C);
        for (int v = 0; v < nv; v++) o[v] = acc[v];
      }
    }
  }
}

#ifdef __hexagon__
/* 1-D l2fetch of `bytes` starting at p (Rtt = dir|stride|width|height, per the SDK's qhl_hvx
 * hvx_internal.h helper; height=1 since a feature-map row stride here exceeds the 16-bit field). */
static inline void roi_l2fetch(const void* p, unsigned bytes) {
  unsigned long long ctl = ((unsigned long long)bytes << 32) | ((unsigned long long)bytes << 16) | 1ull;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(ctl));
}
#else
static inline void roi_l2fetch(const void* p, unsigned bytes) { (void)p; (void)bytes; }
#endif

/* Same result as roialign_hwc; before computing bin (ph, pw) it l2fetch-es the two 2-pixel row
 * segments (2*C floats each) every sample of the *next* bin will read, so DDR latency for bin b+1
 * overlaps bin b's HVX work. */
static void roialign_hwc_pf(const float* feat, int H, int W, int C, const float* rois, int R,
                            int OH, int OW, int sr, float scale, float* out) {
  const int nv = C / 32;
  const float inv_count = 1.0f / (float)(sr * sr);
  const unsigned seg = 2u * C * 4u;
  for (int r = 0; r < R; r++) {
    const float x1 = rois[4 * r + 0] * scale, y1 = rois[4 * r + 1] * scale;
    const float x2 = rois[4 * r + 2] * scale, y2 = rois[4 * r + 3] * scale;
    const float bin_w = roi_maxf(x2 - x1, 1.0f) / (float)OW;
    const float bin_h = roi_maxf(y2 - y1, 1.0f) / (float)OH;
    for (int b = 0; b < OH * OW; b++) {
      const int ph = b / OW, pw = b % OW;
      if (b + 1 < OH * OW) {
        const int nph = (b + 1) / OW, npw = (b + 1) % OW;
        for (int iy = 0; iy < sr; iy++) {
          float y = y1 + nph * bin_h + (iy + 0.5f) * bin_h / (float)sr;
          if (y < -1.0f || y > (float)H) continue;
          int y_lo = y <= 0.0f ? 0 : (int)y; if (y_lo > H - 1) y_lo = H - 1;
          int y_hi = y_lo + 1 > H - 1 ? H - 1 : y_lo + 1;
          for (int ix = 0; ix < sr; ix++) {
            float x = x1 + npw * bin_w + (ix + 0.5f) * bin_w / (float)sr;
            if (x < -1.0f || x > (float)W) continue;
            int x_lo = x <= 0.0f ? 0 : (int)x; if (x_lo > W - 2) x_lo = W - 2 < 0 ? 0 : W - 2;
            roi_l2fetch(feat + ((long)y_lo * W + x_lo) * C, seg);
            roi_l2fetch(feat + ((long)y_hi * W + x_lo) * C, seg);
          }
        }
      }
      roi_f32x32 acc[ROI_MAX_NV];
      for (int v = 0; v < nv; v++) acc[v] = (roi_f32x32){0};
      for (int iy = 0; iy < sr; iy++) {
        float y = y1 + ph * bin_h + (iy + 0.5f) * bin_h / (float)sr;
        for (int ix = 0; ix < sr; ix++) {
          float x = x1 + pw * bin_w + (ix + 0.5f) * bin_w / (float)sr;
          if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) continue;
          float yy = y <= 0.0f ? 0.0f : y, xx = x <= 0.0f ? 0.0f : x;
          int y_lo = (int)yy, x_lo = (int)xx, y_hi, x_hi;
          if (y_lo >= H - 1) { y_hi = y_lo = H - 1; yy = (float)y_lo; } else { y_hi = y_lo + 1; }
          if (x_lo >= W - 1) { x_hi = x_lo = W - 1; xx = (float)x_lo; } else { x_hi = x_lo + 1; }
          const float ly = yy - y_lo, lx = xx - x_lo, hy = 1.0f - ly, hx = 1.0f - lx;
          const float w1 = hy * hx * inv_count, w2 = hy * lx * inv_count;
          const float w3 = ly * hx * inv_count, w4 = ly * lx * inv_count;
          const roi_f32x32* p1 = (const roi_f32x32*)(feat + ((long)y_lo * W + x_lo) * C);
          const roi_f32x32* p2 = (const roi_f32x32*)(feat + ((long)y_lo * W + x_hi) * C);
          const roi_f32x32* p3 = (const roi_f32x32*)(feat + ((long)y_hi * W + x_lo) * C);
          const roi_f32x32* p4 = (const roi_f32x32*)(feat + ((long)y_hi * W + x_hi) * C);
          for (int v = 0; v < nv; v++) acc[v] += p1[v] * w1 + p2[v] * w2 + p3[v] * w3 + p4[v] * w4;
        }
      }
      roi_f32x32* o = (roi_f32x32*)(out + (((long)r * OH + ph) * OW + pw) * C);
      for (int v = 0; v < nv; v++) o[v] = acc[v];
    }
  }
}

#endif
