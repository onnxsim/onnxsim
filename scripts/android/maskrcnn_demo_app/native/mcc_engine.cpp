// MCC mode of the demo app (MccActivity, its own process): one photo -> tap an object -> its full 3D
// shape and color, every network strict on the HTP from EP-context models.
//   1. SAM (sam_l0_enc / sam_l0_dec, the SAM mode's models, see sam_engine.cpp): the tapped object's
//      mask. The working image is W x H = 480 x 640 (portrait) or 640 x 480 (landscape).
//   2. MoGe-2 ViT-S (moge_<H>x<W>.onnx, ../vision_models/mcc/depth.py static): float NCHW image in
//      [0, 1] -> metric point map "points" [1, H, W, 3] in the OpenCV camera frame; y and z are
//      flipped into MCC's frame (x right, y up, z toward the viewer), as depth.py does. It needs only
//      the image, so it starts on its own thread as soon as an image is encoded and runs while the
//      user picks the object; "3D" waits for it (usually done by then).
//   3. model.py prep() + xyz_windows() on the CPU: points outside the mask -> inf, scale by the mean
//      per-axis std, center, crop to the mask box + 40 px, pad square, image bilinear -> 800 -> 224
//      (normalized), points bilinear -> 112 (a non-finite tap makes a point invalid, as in torch),
//      invalid -> -100, shrink, 8x8 windows.
//   4. MCC encoder (mcc_enc.onnx = ../vision_models/mcc enc.onnx): img, xyz_win, valid -> the decoder's
//      seen K/V [8, 16, 197, 32] each, once per reconstruction.
//   5. MCC decoder chunks: 1024 query points each against that K/V -> occupancy logit + color. Default
//      (opts dec=hmx): the hand-written DSP decoder of ../../mcc_hmx (HMX GEMMs + 4 HVX threads, a
//      FastRPC skel, libmcc_hmx_rpc.so; weights mcc_hmx_blk0..7.bin + mcc_hmx_head.bin loaded once, the
//      image's K/V packed here and sent once), ~22 ms a chunk, and the last chunk of a level only as
//      long as it needs (multiples of 32 queries). dec=qnn: mcc_dec_q1024.onnx on the HTP through QNN
//      (the w8a16 dec_opt.py a16c build: 47 ms a chunk). Queries coarse-to-fine exactly as mcc.py recon: every point of the coarsest grid,
//      then at each finer level the 27-neighborhood of every cell with p > lo (default 15^3 -> 30^3 ->
//      60^3, granularity 0.1, lo 0.1: recall >= 0.9975 of the dense grid's occupied points on the three
//      references in ../vision_models/mcc/dec_opt.py, 13-20% fewer queries than 0.05). The result is
//      the target grid's points with p > thr.
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "htp_session.h"
#include "yuv_upright.h"
#include "remote.h"
#include "mcc_hmx_rpc.h"
#include "mcc_decoder.h" /* mb_pack_kv: the K/V tile layout the skel expects */

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MccDemo", __VA_ARGS__)

namespace {
using demo::now_ms;
constexpr int S = 512, PROMPT = 1024, LR = 256, EMB = 256 * 64 * 64;
constexpr uint8_t kPad[3] = {124, 116, 104};  // round(SAM pixel mean), as sam_engine.cpp
constexpr int SEEN = 197, KV = 8 * 16 * SEEN * 32, Q = 1024, XYZ = 112, WIN = 8, NWIN = (XYZ / WIN) * (XYZ / WIN);
constexpr float kInf = std::numeric_limits<float>::infinity();

demo::Htp g_htp;
std::string g_dir, g_perf;
std::unique_ptr<Ort::Session> g_sam_enc, g_sam_dec, g_enc, g_dec;
remote_handle64 g_hmx = 0; /* the DSP decoder (dec=hmx) */
std::map<std::string, std::unique_ptr<Ort::Session>> g_moge;  // per orientation, loaded on first use
int g_w = 0, g_h = 0;                                          // working image
std::vector<uint8_t> g_rgb, g_mask, g_q(S * S * 3);
std::vector<float> g_emb(EMB), g_lr(4 * LR * LR);
std::shared_future<std::vector<float>> g_moge_pts;  // the current image's MoGe-2 points (MCC frame)
float g_moge_ms = 0;                                  // its run time (set by the MoGe thread)
bool g_have_emb = false, g_have_mask = false;
std::vector<float> g_pts;    // result: xyz per point
std::vector<int32_t> g_col;  // result: ARGB per point
std::string g_err;
std::mutex g_mu;
std::unordered_map<std::string, std::string> g_opts;

Ort::MemoryInfo cpu() { return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); }

// torch F.interpolate(mode="bilinear", align_corners=False, no antialias) taps along one axis:
// src = scale * (dst + 0.5) - 0.5 clamped at 0, in float like torch's CPU kernel.
struct Taps {
  std::vector<int> i0, i1;
  std::vector<float> l0, l1;
};
Taps taps(int in, int out, float scale) {
  Taps t;
  for (int d = 0; d < out; ++d) {
    float src = std::max(0.f, scale * (d + 0.5f) - 0.5f);
    int i0 = std::min((int)src, in - 1);
    t.i0.push_back(i0);
    t.i1.push_back(i0 + (i0 < in - 1 ? 1 : 0));
    t.l1.push_back(src - i0);
    t.l0.push_back(1.f - (src - i0));
  }
  return t;
}
// Interleaved C-channel bilinear resize. Every tap is multiplied in, weight 0 included, so a
// non-finite source point poisons its neighbors exactly as in torch (0 * inf = nan).
void resize(const float* src, int sh, int sw, int C, float* dst, int dh, int dw, float sy, float sx) {
  const Taps ty = taps(sh, dh, sy), tx = taps(sw, dw, sx);
  for (int y = 0; y < dh; ++y) {
    const float* r0 = src + (size_t)ty.i0[y] * sw * C;
    const float* r1 = src + (size_t)ty.i1[y] * sw * C;
    for (int x = 0; x < dw; ++x) {
      const int a = tx.i0[x] * C, b = tx.i1[x] * C;
      for (int c = 0; c < C; ++c)
        dst[((size_t)y * dw + x) * C + c] = ty.l0[y] * (tx.l0[x] * r0[a + c] + tx.l1[x] * r0[b + c]) +
                                            ty.l1[y] * (tx.l0[x] * r1[a + c] + tx.l1[x] * r1[b + c]);
    }
  }
}

void dump(const char* name, const void* p, size_t bytes) {
  if (g_opts["dump"] != "1") return;
  std::string path = g_dir + "/../mcc_dump";
  mkdir(path.c_str(), 0755);
  FILE* f = fopen((path + "/" + name).c_str(), "wb");
  if (!f) return;
  fwrite(p, 1, bytes, f);
  fclose(f);
}

// ---- SAM ----------------------------------------------------------------------------------------
// working image -> SAM encoder input (longest side 512, padded bottom/right) -> image embeddings
float sam_encode() {
  const double t = now_ms();
  const float r = (float)S / std::max(g_w, g_h);
  const int fw = std::max(1, (int)(g_w * r + 0.5f)), fh = std::max(1, (int)(g_h * r + 0.5f));
  std::vector<float> src(g_rgb.begin(), g_rgb.end()), dst((size_t)fw * fh * 3);
  resize(src.data(), g_h, g_w, 3, dst.data(), fh, fw, (float)g_h / fh, (float)g_w / fw);
  for (int y = 0; y < S; ++y)
    for (int x = 0; x < S; ++x) {
      uint8_t* o = g_q.data() + ((size_t)y * S + x) * 3;
      if (x < fw && y < fh)
        for (int c = 0; c < 3; ++c) o[c] = (uint8_t)std::lround(std::min(255.f, std::max(0.f, dst[((size_t)y * fw + x) * 3 + c])));
      else
        memcpy(o, kPad, 3);
    }
  int64_t is[4] = {1, S, S, 3}, es[4] = {1, 256, 64, 64};
  Ort::Value in = Ort::Value::CreateTensor<uint8_t>(cpu(), g_q.data(), g_q.size(), is, 4);
  Ort::Value out = Ort::Value::CreateTensor<float>(cpu(), g_emb.data(), g_emb.size(), es, 4);
  const char* in_n[] = {"pixels_u8"};
  const char* out_n[] = {"image_embeddings"};
  g_sam_enc->Run(Ort::RunOptions{nullptr}, in_n, &in, 1, out_n, &out, 1);
  g_have_emb = true;
  g_have_mask = false;
  return (float)(now_ms() - t);
}

// tap (working pixels) -> g_mask (working res; the chosen low-res logits upsampled bilinearly, > 0)
float sam_decode(float x, float y, float* iou) {
  const float k = (float)PROMPT / std::max(g_w, g_h);
  float pc[4] = {x * k, y * k, 0.f, 0.f}, pl[2] = {1.f, -1.f};
  int64_t es[4] = {1, 256, 64, 64}, cs[3] = {1, 2, 2}, ls[2] = {1, 2}, is[2] = {1, 4}, ms[4] = {1, 4, LR, LR};
  Ort::Value in[3] = {Ort::Value::CreateTensor<float>(cpu(), g_emb.data(), g_emb.size(), es, 4),
                      Ort::Value::CreateTensor<float>(cpu(), pc, 4, cs, 3),
                      Ort::Value::CreateTensor<float>(cpu(), pl, 2, ls, 2)};
  Ort::Value out[2] = {Ort::Value::CreateTensor<float>(cpu(), iou, 4, is, 2),
                       Ort::Value::CreateTensor<float>(cpu(), g_lr.data(), g_lr.size(), ms, 4)};
  const char* in_n[] = {"image_embeddings", "point_coords", "point_labels"};
  const char* out_n[] = {"iou_predictions", "low_res_masks"};
  g_sam_dec->Run(Ort::RunOptions{nullptr}, in_n, in, 3, out_n, out, 2);
  const int slot = 1 + (int)(std::max_element(iou + 1, iou + 4) - (iou + 1));  // sam.py's point rule
  // one low-res pixel is PROMPT / LR = 4 prompt pixels = max(W, H) / LR working pixels; resize's scale
  // is source pixels per output pixel
  const float sc = (float)LR / std::max(g_w, g_h);
  std::vector<float> m((size_t)g_w * g_h);
  resize(g_lr.data() + (size_t)slot * LR * LR, LR, LR, 1, m.data(), g_h, g_w, sc, sc);
  g_mask.assign((size_t)g_w * g_h, 0);
  for (size_t i = 0; i < m.size(); ++i) g_mask[i] = m[i] > 0.f;
  g_have_mask = true;
  return (float)slot;
}

// ---- MoGe-2 + MCC -------------------------------------------------------------------------------
// Runs on the MoGe thread only (g_moge is touched nowhere else).
Ort::Session& moge(int w, int h) {
  const std::string stem = "moge_" + std::to_string(h) + "x" + std::to_string(w);
  auto& s = g_moge[stem];
  if (!s) s = g_htp.session(g_dir, stem, g_perf, "MccDemo");
  return *s;
}

// image (h, w, 3) -> MoGe points (h, w, 3), flipped into MCC's frame
std::vector<float> moge_points(const std::vector<uint8_t>& rgb, int w, int h) {
  const double t0 = now_ms();
  const size_t n = (size_t)w * h;
  std::vector<float> img(3 * n), pts(3 * n), mk(n);
  for (size_t i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) img[c * n + i] = rgb[3 * i + c] / 255.f;
  int64_t is[4] = {1, 3, h, w}, ps[4] = {1, h, w, 3}, ms[3] = {1, h, w};
  Ort::Value in = Ort::Value::CreateTensor<float>(cpu(), img.data(), img.size(), is, 4);
  Ort::Value out[2] = {Ort::Value::CreateTensor<float>(cpu(), pts.data(), pts.size(), ps, 4),
                       Ort::Value::CreateTensor<float>(cpu(), mk.data(), mk.size(), ms, 3)};
  const char* in_n[] = {"image"};
  const char* out_n[] = {"points", "mask"};
  moge(w, h).Run(Ort::RunOptions{nullptr}, in_n, &in, 1, out_n, out, 2);
  dump("moge_points.f32", pts.data(), pts.size() * 4);
  for (size_t i = 0; i < n; ++i) {  // OpenCV camera frame -> x right, y up, z toward the viewer
    pts[3 * i + 1] = -pts[3 * i + 1];
    pts[3 * i + 2] = -pts[3 * i + 2];
  }
  g_moge_ms = (float)(now_ms() - t0);
  LOGI("MoGe-2 %dx%d: %.1f ms (background)", w, h, g_moge_ms);
  return pts;
}

// A new working image: start its MoGe-2 run (after the previous one, which may still be running).
void start_moge() {
  if (g_moge_pts.valid()) g_moge_pts.wait();
  g_moge_pts = std::async(std::launch::async, moge_points, g_rgb, g_w, g_h).share();
}

struct EncIn {
  std::vector<float> img = std::vector<float>(3 * 224 * 224), win = std::vector<float>(NWIN * WIN * WIN * 3),
                     valid = std::vector<float>(NWIN * WIN * WIN);
};

// model.py prep() + xyz_windows() for the working image, its mask and seen points
EncIn prep(std::vector<float> xyz) {
  const int W = g_w, H = g_h;
  const size_t n = (size_t)W * H;
  for (size_t i = 0; i < n; ++i)
    if (!g_mask[i]) xyz[3 * i] = xyz[3 * i + 1] = xyz[3 * i + 2] = kInf;
  auto fin = [&](size_t i) { return std::isfinite(xyz[3 * i] + xyz[3 * i + 1] + xyz[3 * i + 2]); };
  double sum[3] = {0, 0, 0}, sq[3] = {0, 0, 0};
  size_t cnt = 0;
  for (size_t i = 0; i < n; ++i)
    if (fin(i)) {
      ++cnt;
      for (int c = 0; c < 3; ++c) sum[c] += xyz[3 * i + c], sq[c] += (double)xyz[3 * i + c] * xyz[3 * i + c];
    }
  if (cnt < 2) throw std::runtime_error("the mask has no valid depth points");
  double sd = 0;  // mean of the per-axis std (torch var: unbiased)
  for (int c = 0; c < 3; ++c) sd += std::sqrt((sq[c] - sum[c] * sum[c] / cnt) / (cnt - 1)) / 3;
  double mean[3] = {0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    for (int c = 0; c < 3; ++c) xyz[3 * i + c] = (float)(xyz[3 * i + c] / sd);
    if (fin(i))
      for (int c = 0; c < 3; ++c) mean[c] += xyz[3 * i + c];
  }
  for (size_t i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) xyz[3 * i + c] -= (float)(mean[c] / cnt);
  // mask box + 40 px (torch slicing clips the far side)
  int top = H, left = W, bottom = -1, right = -1;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (g_mask[(size_t)y * W + x]) top = std::min(top, y), bottom = std::max(bottom, y), left = std::min(left, x), right = std::max(right, x);
  top = std::max(top - 40, 0), left = std::max(left - 40, 0);
  bottom = std::min(bottom + 40, H - 1), right = std::min(right + 40, W - 1);
  const int ch = bottom - top + 1, cw = right - left + 1, sq_n = std::max(ch, cw);
  // pad square (bottom or right): points with inf, the image with 0
  std::vector<float> rgb((size_t)sq_n * sq_n * 3, 0.f), pts((size_t)sq_n * sq_n * 3, kInf);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      const size_t s = (size_t)(top + y) * W + left + x, d = (size_t)y * sq_n + x;
      for (int c = 0; c < 3; ++c) rgb[3 * d + c] = g_rgb[3 * s + c] / 255.f, pts[3 * d + c] = xyz[3 * s + c];
    }
  std::vector<float> i800(800 * 800 * 3), i224(224 * 224 * 3), x112(XYZ * XYZ * 3);
  resize(rgb.data(), sq_n, sq_n, 3, i800.data(), 800, 800, (float)sq_n / 800, (float)sq_n / 800);
  const float s224 = (float)(1.0 / (224.0 / 800.0));  // F.interpolate(scale_factor=224/800)
  resize(i800.data(), 800, 800, 3, i224.data(), 224, 224, s224, s224);
  resize(pts.data(), sq_n, sq_n, 3, x112.data(), XYZ, XYZ, (float)sq_n / XYZ, (float)sq_n / XYZ);
  EncIn e;
  const float mu[3] = {0.485f, 0.456f, 0.406f}, sg[3] = {0.229f, 0.224f, 0.225f};
  for (int i = 0; i < 224 * 224; ++i)
    for (int c = 0; c < 3; ++c) e.img[c * 224 * 224 + i] = (i224[3 * i + c] - mu[c]) / sg[c];
  // xyz_windows: invalid -> -100, shrink(threshold 10), (14, 8, 14, 8) -> (196, 64)
  for (int y = 0; y < XYZ; ++y)
    for (int x = 0; x < XYZ; ++x) {
      float* p = x112.data() + ((size_t)y * XYZ + x) * 3;
      const bool ok = std::isfinite(p[0] + p[1] + p[2]);
      float v[3] = {ok ? p[0] : -100.f, ok ? p[1] : -100.f, ok ? p[2] : -100.f};
      const float dist = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      if (dist > 10.f) {
        const float f = 10.f * (2.f - 10.f / dist) / dist;
        for (float& t : v) t *= f;
      }
      const int w = (y / WIN) * (XYZ / WIN) + x / WIN, k = (y % WIN) * WIN + x % WIN;
      memcpy(e.win.data() + ((size_t)w * WIN * WIN + k) * 3, v, sizeof v);
      e.valid[(size_t)w * WIN * WIN + k] = ok ? 1.f : 0.f;
    }
  return e;
}

// times: 0 MoGe-2 run (background), 1 waited for it, 2 prep, 3 encoder, 4 decoder (all chunks), 5 total;
// counts: queries, chunks, points
void reconstruct(float* times, int* counts) {
  if (!g_have_mask) throw std::runtime_error("tap an object first");
  const double t0 = now_ms();
  const float gran = std::stof(g_opts["gran"]), lo = std::stof(g_opts["lo"]), thr = std::stof(g_opts["thr"]);
  const int levels = std::stoi(g_opts["levels"]), nt = (int)std::lround(6 / gran);
  if (nt % (1 << levels)) throw std::runtime_error("6 / gran must be divisible by 2^levels");
  const int32_t dims[4] = {g_w, g_h, nt, levels};
  dump("dims.i32", dims, sizeof dims);
  dump("rgb.u8", g_rgb.data(), g_rgb.size());
  dump("mask.u8", g_mask.data(), g_mask.size());
  if (!g_moge_pts.valid()) throw std::runtime_error("no image encoded yet");
  std::vector<float> seen = g_moge_pts.get();  // a copy: the same image can be reconstructed again
  const double t1 = now_ms();
  EncIn e = prep(std::move(seen));
  const double t2 = now_ms();
  dump("img.f32", e.img.data(), e.img.size() * 4);
  dump("xyz_win.f32", e.win.data(), e.win.size() * 4);
  dump("valid.f32", e.valid.data(), e.valid.size() * 4);
  std::vector<float> k(KV), v(KV);
  {
    int64_t is[4] = {1, 3, 224, 224}, ws[3] = {NWIN, WIN * WIN, 3}, vs[2] = {NWIN, WIN * WIN}, ks[4] = {8, 16, SEEN, 32};
    Ort::Value in[3] = {Ort::Value::CreateTensor<float>(cpu(), e.img.data(), e.img.size(), is, 4),
                        Ort::Value::CreateTensor<float>(cpu(), e.win.data(), e.win.size(), ws, 3),
                        Ort::Value::CreateTensor<float>(cpu(), e.valid.data(), e.valid.size(), vs, 2)};
    Ort::Value out[2] = {Ort::Value::CreateTensor<float>(cpu(), k.data(), k.size(), ks, 4),
                         Ort::Value::CreateTensor<float>(cpu(), v.data(), v.size(), ks, 4)};
    const char* in_n[] = {"img", "xyz_win", "valid"};
    const char* out_n[] = {"k", "v"};
    g_enc->Run(Ort::RunOptions{nullptr}, in_n, in, 3, out_n, out, 2);
  }
  const double t3 = now_ms();
  dump("k.f32", k.data(), k.size() * 4);
  dump("v.f32", v.data(), v.size() * 4);
  // decoder: one set of tensors, rebound per chunk (QNN), or the DSP decoder with this image's K/V
  std::vector<float> qx(Q * 3), qo(Q), qc(Q * 3);
  int64_t xs[3] = {1, Q, 3}, os[2] = {1, Q}, ks[4] = {8, 16, SEEN, 32};
  Ort::Value din[3] = {Ort::Value::CreateTensor<float>(cpu(), qx.data(), qx.size(), xs, 3),
                       Ort::Value::CreateTensor<float>(cpu(), k.data(), k.size(), ks, 4),
                       Ort::Value::CreateTensor<float>(cpu(), v.data(), v.size(), ks, 4)};
  Ort::Value dout[2] = {Ort::Value::CreateTensor<float>(cpu(), qo.data(), qo.size(), os, 2),
                        Ort::Value::CreateTensor<float>(cpu(), qc.data(), qc.size(), xs, 3)};
  const char* din_n[] = {"xyz", "k", "v"};
  const char* dout_n[] = {"occ", "rgb"};
  const int hmx_thr = std::stoi(g_opts["hmx_threads"]);
  if (g_hmx) {
    const size_t per = (size_t)2 * MB_HEADS * MB_ST * MB_TH; /* halfwords per block: K tiles, V tiles */
    std::vector<mb_hf> kv(per * MB_BLOCKS);
    mb_hf *kt[MB_BLOCKS], *vt[MB_BLOCKS];
    for (int b = 0; b < MB_BLOCKS; b++) kt[b] = kv.data() + per * b, vt[b] = kt[b] + per / 2;
    mb_pack_kv(kt, vt, k.data(), v.data());
    const int rc = mcc_hmx_rpc_set_kv_tiles(g_hmx, kv.data(), (int)kv.size());
    if (rc) throw std::runtime_error("mcc_hmx set_kv_tiles " + std::to_string(rc));
  }
  std::vector<float> p, rgb, p_prev, rgb_prev;
  std::vector<uint8_t> want, q_prev;
  int n_prev = 0, nq = 0, nchunks = 0;
  for (int li = 0; li <= levels; ++li) {
    const int n = nt >> (levels - li);
    const size_t N = (size_t)n * n * n;
    auto at = [n](int i, int j, int l) { return ((size_t)i * n + j) * n + l; };
    std::vector<uint8_t> have(N, 0);
    want.assign(N, li == 0);
    p.assign(N, 0.f);
    rgb.assign(3 * N, 0.f);
    if (li) {  // queried coarse points -> even fine indices; refine around coarse cells with p > lo
      for (int i = 0; i < n_prev; ++i)
        for (int j = 0; j < n_prev; ++j)
          for (int l = 0; l < n_prev; ++l) {
            const size_t c = ((size_t)i * n_prev + j) * n_prev + l, f = at(2 * i, 2 * j, 2 * l);
            p[f] = p_prev[c];
            for (int t = 0; t < 3; ++t) rgb[3 * f + t] = rgb_prev[3 * c + t];
            if (q_prev[c]) have[f] = want[f] = 1;
            if (p_prev[c] > lo)
              for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                  for (int dl = -1; dl <= 1; ++dl) {
                    const int a = 2 * i + di, b = 2 * j + dj, d = 2 * l + dl;
                    if (a >= 0 && b >= 0 && d >= 0 && a < n && b < n && d < n) want[at(a, b, d)] = 1;
                  }
          }
    }
    std::vector<size_t> todo;
    for (size_t f = 0; f < N; ++f)
      if (want[f] && !have[f]) todo.push_back(f);
    for (size_t s0 = 0; s0 < todo.size(); s0 += Q) {
      const size_t m = std::min<size_t>(Q, todo.size() - s0);
      std::fill(qx.begin(), qx.end(), 0.f);
      for (size_t r = 0; r < m; ++r) {
        const size_t f = todo[s0 + r];
        const size_t idx[3] = {f / ((size_t)n * n), (f / n) % n, f % n};
        for (int t = 0; t < 3; ++t) qx[3 * r + t] = (float)((idx[t] - n / 2.0) / ((n / 2.0) / 3.0));
      }
      if (g_hmx) { /* only the rows this chunk uses, rounded up to 32 */
        const int q = (int)((m + 31) / 32 * 32);
        uint64 t[13];
        int codes[6];
        const int rc = mcc_hmx_rpc_decode(g_hmx, q, MB_V_ALL, hmx_thr, qx.data(), q * 3, qo.data(), q, qc.data(), q * 3, t, 13, codes, 6);
        if (rc || codes[2] || !codes[0]) throw std::runtime_error("mcc_hmx decode rc " + std::to_string(rc) + " ctx " + std::to_string(codes[0]) +
                                                                  " hmx lock " + std::to_string(codes[2]));
      } else
        g_dec->Run(Ort::RunOptions{nullptr}, din_n, din, 3, dout_n, dout, 2);
      for (size_t r = 0; r < m; ++r) {
        const size_t f = todo[s0 + r];
        p[f] = 1.f / (1.f + std::exp(-qo[r]));
        for (int t = 0; t < 3; ++t) rgb[3 * f + t] = qc[3 * r + t];
      }
      ++nchunks;
    }
    nq += (int)todo.size();
    LOGI("level n=%d: %zu queries", n, todo.size());
    p_prev.swap(p), rgb_prev.swap(rgb), q_prev.swap(want), n_prev = n;
  }
  const double t4 = now_ms();
  dump("p.f32", p_prev.data(), p_prev.size() * 4);
  dump("rgb.f32", rgb_prev.data(), rgb_prev.size() * 4);
  g_pts.clear(), g_col.clear();
  for (size_t f = 0; f < p_prev.size(); ++f)
    if (p_prev[f] > thr) {
      const size_t idx[3] = {f / ((size_t)nt * nt), (f / nt) % nt, f % nt};
      for (int t = 0; t < 3; ++t) g_pts.push_back((float)((idx[t] - nt / 2.0) / ((nt / 2.0) / 3.0)));
      int32_t c = (int32_t)0xFF000000;
      for (int t = 0; t < 3; ++t)
        c |= (int32_t)std::lround(std::min(1.f, std::max(0.f, rgb_prev[3 * f + t])) * 255) << (16 - 8 * t);
      g_col.push_back(c);
    }
  const float tt[6] = {g_moge_ms, (float)(t1 - t0), (float)(t2 - t1), (float)(t3 - t2), (float)(t4 - t3),
                       (float)(now_ms() - t0)};
  memcpy(times, tt, sizeof tt);
  counts[0] = nq, counts[1] = nchunks, counts[2] = (int)g_col.size();
}

struct Locked {
  JNIEnv* e;
  jobject bmp;
  uint8_t* px = nullptr;
  AndroidBitmapInfo bi{};
  Locked(JNIEnv* e_, jobject b) : e(e_), bmp(b) {
    if (b && AndroidBitmap_getInfo(e, b, &bi) == 0 && bi.format == ANDROID_BITMAP_FORMAT_RGBA_8888 &&
        AndroidBitmap_lockPixels(e, b, (void**)&px) != 0)
      px = nullptr;
  }
  ~Locked() {
    if (px) AndroidBitmap_unlockPixels(e, bmp);
  }
};

std::string jstr(JNIEnv* e, jstring s) {
  const char* c = e->GetStringUTFChars(s, nullptr);
  std::string r(c);
  e->ReleaseStringUTFChars(s, c);
  return r;
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_MccEngine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                       jstring jlib, jstring jopts) {
  std::lock_guard<std::mutex> l(g_mu);
  try {
    g_opts = demo::parse_opts(jstr(e, jopts), {{"htp_performance_mode", "burst"},
                                               {"gran", "0.1"},
                                               {"levels", "2"},
                                               {"lo", "0.1"},
                                               {"thr", "0.3"},
                                               {"dump", "0"},
                                               {"dec", "hmx"},
                                               {"hmx_threads", "4"}});
    g_dir = jstr(e, jdir);
    g_perf = g_opts["htp_performance_mode"];
    g_htp.init(jstr(e, jlib), "mcc");
    g_sam_enc = g_htp.session(g_dir, "sam_l0_enc", g_perf, "MccDemo");
    g_sam_dec = g_htp.session(g_dir, "sam_l0_dec", g_perf, "MccDemo");
    g_enc = g_htp.session(g_dir, "mcc_enc", g_perf, "MccDemo");
    if (g_opts["dec"] == "hmx") {
      if (!g_hmx) {
        struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
        remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
        if (mcc_hmx_rpc_open("file:///libmcc_hmx_rpc.so?mcc_hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &g_hmx))
          throw std::runtime_error("mcc_hmx_rpc_open failed (skel libmcc_hmx_rpc.so)");
        int prc = 0;
        mcc_hmx_rpc_perf_vote(g_hmx, 3, &prc); /* turbo + the HMX power vote (mandatory before HMX ops) */
        const double t = now_ms();
        auto load = [](const std::string& path) {
          std::ifstream f(path, std::ios::binary);
          if (!f) throw std::runtime_error("missing " + path);
          return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        };
        for (int b = 0; b < MB_BLOCKS; b++) {
          auto blob = load(g_dir + "/mcc_hmx_blk" + std::to_string(b) + ".bin");
          if (int rc = mcc_hmx_rpc_load_block(g_hmx, b, blob.data(), (int)blob.size())) throw std::runtime_error("mcc_hmx load_block " + std::to_string(rc));
        }
        auto head = load(g_dir + "/mcc_hmx_head.bin");
        if (int rc = mcc_hmx_rpc_load_head(g_hmx, head.data(), (int)head.size())) throw std::runtime_error("mcc_hmx load_head " + std::to_string(rc));
        LOGI("DSP decoder: weights loaded in %.0f ms", now_ms() - t);
      }
    } else
      g_dec = g_htp.session(g_dir, "mcc_dec_q1024", g_perf, "MccDemo");
    return nullptr;
  } catch (const std::exception& ex) {
    return e->NewStringUTF(ex.what());
  }
}

// Camera frame -> the upright working bitmap disp (480x640 or 640x480, the frame scaled to it; the
// camera frames are 4:3); with encode, it becomes the working image and SAM encodes it. times[0] ms.
extern "C" JNIEXPORT jboolean JNICALL Java_org_onnxsim_maskrcnndemo_MccEngine_nativeYuv(
    JNIEnv* e, jclass, jobject jy, jobject ju, jobject jv, jint ys, jint uvs, jint uvps, jint w, jint h, jint rot,
    jobject disp, jboolean enc, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  try {
    const demo::YuvPlanes P{(const uint8_t*)e->GetDirectBufferAddress(jy), (const uint8_t*)e->GetDirectBufferAddress(ju),
                            (const uint8_t*)e->GetDirectBufferAddress(jv), ys, uvs, uvps, w, h};
    Locked d(e, disp);
    if (!d.px) throw std::runtime_error("display bitmap must be RGBA_8888");
    const int W = d.bi.width, H = d.bi.height;
    if (enc) g_w = W, g_h = H, g_rgb.resize((size_t)W * H * 3);
    demo::yuv_upright(P, rot, W, H, 4, [&](int oy, int ox, uint8_t r, uint8_t g, uint8_t b) {
      uint8_t* p = d.px + (size_t)oy * d.bi.stride + 4 * ox;
      p[0] = r; p[1] = g; p[2] = b; p[3] = 255;
      if (enc) {
        uint8_t* q = g_rgb.data() + ((size_t)oy * W + ox) * 3;
        q[0] = r; q[1] = g; q[2] = b;
      }
    });
    if (enc) {
      float t = sam_encode();
      start_moge();
      e->SetFloatArrayRegion(jt, 0, 1, &t);
    }
    return JNI_TRUE;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return JNI_FALSE;
  }
}

// Images mode: the upright working bitmap (480x640 or 640x480) -> working image + SAM encoder.
extern "C" JNIEXPORT jboolean JNICALL Java_org_onnxsim_maskrcnndemo_MccEngine_nativeEncode(JNIEnv* e, jclass,
                                                                                          jobject bmp, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  try {
    Locked d(e, bmp);
    if (!d.px) throw std::runtime_error("bitmap must be RGBA_8888");
    g_w = d.bi.width, g_h = d.bi.height;
    g_rgb.resize((size_t)g_w * g_h * 3);
    for (int y = 0; y < g_h; ++y)
      for (int x = 0; x < g_w; ++x) memcpy(g_rgb.data() + ((size_t)y * g_w + x) * 3, d.px + (size_t)y * d.bi.stride + 4 * x, 3);
    float t = sam_encode();
    start_moge();
    e->SetFloatArrayRegion(jt, 0, 1, &t);
    return JNI_TRUE;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return JNI_FALSE;
  }
}

// Tap (working pixels) -> the object mask (W x H bytes, 1 = inside) and iou[4]. Returns the slot, -1 on
// error. jt[0] = decoder ms.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_MccEngine_nativeSegment(JNIEnv* e, jclass, jfloat x,
                                                                                       jfloat y, jbyteArray jmask,
                                                                                       jfloatArray jiou, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  try {
    if (!g_have_emb) throw std::runtime_error("no image encoded yet");
    const double t0 = now_ms();
    float iou[4];
    const int slot = (int)sam_decode(x, y, iou);
    e->SetByteArrayRegion(jmask, 0, (jsize)g_mask.size(), (const jbyte*)g_mask.data());
    e->SetFloatArrayRegion(jiou, 0, 4, iou);
    float t = (float)(now_ms() - t0);
    e->SetFloatArrayRegion(jt, 0, 1, &t);
    return slot;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

// MoGe-2 + MCC for the current mask. times[6]: MoGe-2 run, MoGe-2 wait, prep, encoder, decoder, total
// ms; counts[3]: queries, chunks, points. Returns the number of points, -1 on error.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_MccEngine_nativeReconstruct(JNIEnv* e, jclass,
                                                                                           jfloatArray jt, jintArray jc) {
  std::lock_guard<std::mutex> l(g_mu);
  try {
    float t[6];
    int c[3];
    reconstruct(t, c);
    e->SetFloatArrayRegion(jt, 0, 6, t);
    e->SetIntArrayRegion(jc, 0, 3, c);
    return c[2];
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

// The last reconstruction: xyz (3 per point, MCC's frame, in [-3, 3]) and ARGB colors.
extern "C" JNIEXPORT void JNICALL Java_org_onnxsim_maskrcnndemo_MccEngine_nativePoints(JNIEnv* e, jclass, jfloatArray jxyz,
                                                                                      jintArray jcol) {
  std::lock_guard<std::mutex> l(g_mu);
  e->SetFloatArrayRegion(jxyz, 0, (jsize)g_pts.size(), g_pts.data());
  e->SetIntArrayRegion(jcol, 0, (jsize)g_col.size(), (const jint*)g_col.data());
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_MccEngine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}
