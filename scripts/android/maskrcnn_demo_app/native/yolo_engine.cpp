// YOLO mode of the demo app: one HTP session (ORT + QNN EP, strict, no CPU fallback) running a
// deploy-pipeline YOLO model (../../deploy/models/yolo26n.yaml, yolo11n.yaml: uint8 NHWC 640x640
// letterboxed RGB in, the head's decoded (1, 4+nc, N) float out), preprocessing and the head's
// post-processing done here in C++ (no ORT CPU graph):
//   post=end2end  YOLO26's NMS-free one-to-one head, x1,y1,x2,y2 + class scores: the two-stage
//                 top-k of Ultralytics' Detect.get_topk_index (== the deploy pipeline's
//                 yolo_end2end post graph): top max_det anchors by best class score, then the top
//                 max_det (anchor, class) pairs among them.
//   post=nms      YOLO11's head, cx,cy,w,h + class scores: per-class greedy NMS (iou 0.7, conf 0.25,
//                 at most max_det boxes), what the deploy pipeline's yolo_detect graph does with
//                 ONNX NonMaxSuppression.
//   post=detr     RF-DETR (../../vision_models/rfdetr, export.py: uint8 NHWC SxS in, `logits`
//                 (1, 300, 91) + `boxes` (1, 300, 4) out): RF-DETR's PostProcess -- sigmoid, top
//                 max_det over queries x classes, cx,cy,w,h normalized to the square input, labels
//                 are COCO category ids. The input is the frame *stretched* to SxS (RF-DETR's
//                 predict() preprocessing), not letterboxed; boxes scale back by the frame size.
//   -seg models (../../deploy/models/yolo26n-seg.yaml, yolo11n-seg.yaml): the head output carries
//                 nm = 32 mask coefficients after the class scores, (1, 4+nc+nm, N), and a second output
//                 holds the (1, nm, 160, 160) mask prototypes. Boxes are selected as above; each shown
//                 detection's mask is sigmoid(coefficients . prototypes) (Ultralytics' process_mask),
//                 sampled on a side x side grid over its box (the box crop), for the overlay.
// engine=tinygrad runs the same model as a tinygrad ahead-of-time OpenCL bundle on the Adreno GPU instead of the HTP
// (../tinygrad_aot: <models>/<model>.tg/{kernels.cl,plan.txt,consts.bin,meta.txt}, same uint8 NHWC input and float
// outputs, so pre/post-processing are shared); engine=qnn (the default) is the HTP session above.
// Built into its own libyolo_demo.so, loaded only by YoloActivity (its own process).
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "htp_session.h"
#include "tg_cl_runner.h"
#include "yuv_upright.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "YoloDemo", __VA_ARGS__)

namespace {
constexpr int S = 640;           // model input side
constexpr uint8_t kPad = 114;    // Ultralytics letterbox fill
using demo::now_ms;

demo::Htp g_htp;
std::unique_ptr<Ort::Session> g_sess;
// engine=tinygrad: the AOT bundle and its outputs' names and shapes (meta.txt), in bundle output order
std::unique_ptr<tgcl::Model> g_tg;
struct TgOut {
  std::string name;
  std::vector<int64_t> shape;
};
std::vector<TgOut> g_tg_outs;
std::string g_in, g_out, g_post, g_err;
std::vector<uint8_t> g_q(S * S * 3);
std::vector<float> g_head;
int g_ch = 0, g_n = 0, g_nc = 0, g_maxdet = 300;
// -seg models: prototype output name, mask count, prototype grid
bool g_seg = false;
std::string g_out_proto;
std::vector<float> g_proto;
int g_nm = 0, g_ph = 0, g_pw = 0;
std::mutex g_mu;
float g_conf = 0.25f, g_iou = 0.7f;
// post=detr (RF-DETR): input side, second output, query/class counts, the upright display-size
// RGB the stretched input is sampled from, and COCO category id -> Coco.java's contiguous index
bool g_detr = false;
int g_S = S, g_nq = 0, g_ncls = 0;
std::string g_out2;
std::vector<float> g_boxes;
std::vector<uint8_t> g_rgb;
int g_cat2idx[128];
void init_cat_map() {
  // the 80 COCO category ids in order (the 91-id space has gaps)
  static const int ids[80] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                              22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                              46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                              67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
  for (int& v : g_cat2idx) v = 0;
  for (int i = 0; i < 80; ++i) g_cat2idx[ids[i]] = i + 1;  // Coco.name(): 1-based, 0 = background
}

// Nearest-neighbour stretch of an RGB(A) image (w x h, `ch` bytes per pixel, row stride `stride`)
// to the model's g_S x g_S uint8 RGB input (pixel-centre sampling, as yuv_upright).
void stretch_into_input(const uint8_t* src, int w, int h, int ch, size_t stride) {
  std::vector<int> xs(g_S);
  for (int x = 0; x < g_S; ++x) xs[x] = std::min(w - 1, (int)(((long)x * w + w / 2) / g_S));
  for (int y = 0; y < g_S; ++y) {
    const uint8_t* row = src + (size_t)std::min(h - 1, (int)(((long)y * h + h / 2) / g_S)) * stride;
    uint8_t* o = g_q.data() + (size_t)y * g_S * 3;
    for (int x = 0; x < g_S; ++x) {
      const uint8_t* p = row + (size_t)xs[x] * ch;
      o[3 * x] = p[0];
      o[3 * x + 1] = p[1];
      o[3 * x + 2] = p[2];
    }
  }
}

// Letterbox geometry for an upright w x h frame: scale to fit 640x640, centered (as
// deploy/stages/images.py: r = min(640/h, 640/w), round, pad (640 - n) // 2).
struct Fit {
  int fw, fh, left, top;
};
Fit fit(int w, int h) {
  const float r = std::min((float)S / w, (float)S / h);
  Fit f;
  f.fw = std::max(1, (int)std::lround(w * r));
  f.fh = std::max(1, (int)std::lround(h * r));
  f.left = (S - f.fw) / 2;
  f.top = (S - f.fh) / 2;
  return f;
}
void pad_rows(int top, int fh) {
  memset(g_q.data(), kPad, (size_t)top * S * 3);
  memset(g_q.data() + (size_t)(top + fh) * S * 3, kPad, (size_t)(S - top - fh) * S * 3);
}

// times: 0 total, 1 pre, 2 htp, 3 post
void infer(float* times, double t0) {
  const double t1 = now_ms();
  if (g_tg) {  // outputs in bundle order, mapped to the same host buffers the HTP path fills
    std::vector<void*> outs;
    for (auto& o : g_tg_outs)
      outs.push_back(o.name == g_out2 && g_detr ? g_boxes.data() : o.name == g_out ? g_head.data() : g_proto.data());
    g_tg->run({g_q.data()}, outs);
    times[1] = (float)(t1 - t0);
    times[2] = (float)(now_ms() - t1);
    return;
  }
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  if (g_detr) {
    int64_t ishape[4] = {1, g_S, g_S, 3};
    Ort::Value in = Ort::Value::CreateTensor<uint8_t>(mi, g_q.data(), g_q.size(), ishape, 4);
    int64_t lshape[3] = {1, g_nq, g_ncls}, bshape[3] = {1, g_nq, 4};
    Ort::Value outs[2] = {Ort::Value::CreateTensor<float>(mi, g_head.data(), g_head.size(), lshape, 3),
                          Ort::Value::CreateTensor<float>(mi, g_boxes.data(), g_boxes.size(), bshape, 3)};
    const char* in_names[] = {g_in.c_str()};
    const char* out_names[] = {g_out.c_str(), g_out2.c_str()};
    g_sess->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, outs, 2);
    times[1] = (float)(t1 - t0);
    times[2] = (float)(now_ms() - t1);
    return;
  }
  int64_t ishape[4] = {1, S, S, 3};
  Ort::Value in = Ort::Value::CreateTensor<uint8_t>(mi, g_q.data(), g_q.size(), ishape, 4);
  int64_t oshape[3] = {1, g_ch, g_n}, pshape[4] = {1, g_nm, g_ph, g_pw};
  Ort::Value outs[2] = {Ort::Value::CreateTensor<float>(mi, g_head.data(), g_head.size(), oshape, 3), Ort::Value{nullptr}};
  if (g_seg) outs[1] = Ort::Value::CreateTensor<float>(mi, g_proto.data(), g_proto.size(), pshape, 4);
  const char* in_names[] = {g_in.c_str()};
  const char* out_names[] = {g_out.c_str(), g_out_proto.c_str()};
  g_sess->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, outs, g_seg ? 2 : 1);
  times[1] = (float)(t1 - t0);
  times[2] = (float)(now_ms() - t1);
}

struct Det {
  float x1, y1, x2, y2, s;
  int c, a = -1;  // a: the anchor (head column), for the -seg models' mask coefficients
};

std::vector<Det> post_end2end() {
  const int nc = g_nc, N = g_n, k = std::min(g_maxdet, N);
  const float* h = g_head.data();
  std::vector<float> best(N, -1e30f);
  for (int c = 0; c < nc; ++c) {
    const float* row = h + (size_t)(4 + c) * N;
    for (int a = 0; a < N; ++a) best[a] = std::max(best[a], row[a]);
  }
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  // ties: ORT TopK keeps the lower index first
  auto by = [&](const std::vector<float>& v) {
    return [&v](int a, int b) { return v[a] > v[b] || (v[a] == v[b] && a < b); };
  };
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), by(best));
  std::vector<float> cand((size_t)k * nc);
  for (int i = 0; i < k; ++i)
    for (int c = 0; c < nc; ++c) cand[(size_t)i * nc + c] = h[(size_t)(4 + c) * N + idx[i]];
  std::vector<int> ci(cand.size());
  std::iota(ci.begin(), ci.end(), 0);
  std::partial_sort(ci.begin(), ci.begin() + k, ci.end(), by(cand));
  std::vector<Det> d;
  for (int j = 0; j < k; ++j) {
    const int a = idx[ci[j] / nc];
    const float s = cand[ci[j]];
    if (s < g_conf) break;  // sorted: the rest are lower (display threshold, not part of the top-k)
    d.push_back({h[a], h[(size_t)N + a], h[(size_t)2 * N + a], h[(size_t)3 * N + a], s, ci[j] % nc, a});
  }
  return d;
}

float iou(const Det& a, const Det& b) {
  const float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1), hh = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (w <= 0 || hh <= 0) return 0.f;
  const float i = w * hh, u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i;
  return u > 0 ? i / u : 0.f;
}

std::vector<Det> post_nms() {
  const int nc = g_nc, N = g_n;
  const float* h = g_head.data();
  std::vector<Det> cand;
  for (int c = 0; c < nc; ++c) {
    const float* row = h + (size_t)(4 + c) * N;
    for (int a = 0; a < N; ++a)
      if (row[a] > g_conf) {
        const float cx = h[a], cy = h[(size_t)N + a], w = h[(size_t)2 * N + a] * 0.5f, hh = h[(size_t)3 * N + a] * 0.5f;
        cand.push_back({cx - w, cy - hh, cx + w, cy + hh, row[a], c, a});
      }
  }
  std::stable_sort(cand.begin(), cand.end(), [](const Det& a, const Det& b) { return a.s > b.s; });
  std::vector<Det> keep;
  for (const Det& d : cand) {
    bool ok = true;
    for (const Det& k : keep)
      if (k.c == d.c && iou(k, d) > g_iou) { ok = false; break; }
    if (ok) keep.push_back(d);
    if ((int)keep.size() >= g_maxdet) break;
  }
  return keep;
}

// RF-DETR PostProcess: top max_det (query, class) pairs by logit (sigmoid is monotonic), boxes
// from normalized cx,cy,w,h to the w x h display frame (the stretch resize maps back per axis).
std::vector<Det> post_detr(int w, int h) {
  const int n = g_nq * g_ncls, k = std::min(g_maxdet, n);
  const float* lg = g_head.data();
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [lg](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
  std::vector<Det> d;
  for (int j = 0; j < k; ++j) {
    const float s = 1.f / (1.f + std::exp(-lg[idx[j]]));
    if (s < g_conf) break;
    const int q = idx[j] / g_ncls, c = idx[j] % g_ncls;
    const float* b = g_boxes.data() + (size_t)q * 4;
    d.push_back({(b[0] - b[2] / 2) * w, (b[1] - b[3] / 2) * h, (b[0] + b[2] / 2) * w, (b[1] + b[3] / 2) * h, s,
                 (c < 128 ? g_cat2idx[c] : 0) - 1});
  }
  return d;
}

// A -seg detection's mask as side x side probabilities over its box (row-major, box-relative): the
// nm prototypes are combined with its coefficients only over the prototype cells the box covers,
// then sampled bilinearly (cell centres) at the grid points and passed through the sigmoid.
void seg_mask(const Det& d, int side, float* out) {
  const float sx = (float)g_pw / S, sy = (float)g_ph / S;  // input pixels -> prototype cells (1/4)
  const float bx1 = d.x1 * sx, by1 = d.y1 * sy, bx2 = d.x2 * sx, by2 = d.y2 * sy;
  const int x0 = std::max(0, (int)std::floor(bx1 - 0.5f)), x1 = std::min(g_pw - 1, (int)std::ceil(bx2 - 0.5f));
  const int y0 = std::max(0, (int)std::floor(by1 - 0.5f)), y1 = std::min(g_ph - 1, (int)std::ceil(by2 - 0.5f));
  if (x1 < x0 || y1 < y0) {
    std::fill(out, out + (size_t)side * side, 0.f);
    return;
  }
  const int rw = x1 - x0 + 1, rh = y1 - y0 + 1;
  std::vector<float> m((size_t)rw * rh, 0.f);
  for (int k = 0; k < g_nm; ++k) {
    const float c = g_head[(size_t)(4 + g_nc + k) * g_n + d.a];
    const float* P = g_proto.data() + (size_t)k * g_ph * g_pw;
    for (int y = 0; y < rh; ++y) {
      const float* row = P + (size_t)(y0 + y) * g_pw + x0;
      float* o = m.data() + (size_t)y * rw;
      for (int x = 0; x < rw; ++x) o[x] += c * row[x];
    }
  }
  for (int v = 0; v < side; ++v) {
    const float fy = std::min((float)(rh - 1), std::max(0.f, by1 + (v + 0.5f) * (by2 - by1) / side - 0.5f - y0));
    const int iy = std::min(rh - 2, (int)fy) < 0 ? 0 : std::min(rh - 2, (int)fy);
    const float wy = rh > 1 ? fy - iy : 0.f;
    for (int u = 0; u < side; ++u) {
      const float fx = std::min((float)(rw - 1), std::max(0.f, bx1 + (u + 0.5f) * (bx2 - bx1) / side - 0.5f - x0));
      const int ix = std::min(rw - 2, (int)fx) < 0 ? 0 : std::min(rw - 2, (int)fx);
      const float wx = rw > 1 ? fx - ix : 0.f;
      auto at = [&](int y, int x) { return m[(size_t)std::min(y, rh - 1) * rw + std::min(x, rw - 1)]; };
      const float l = (1 - wy) * ((1 - wx) * at(iy, ix) + wx * at(iy, ix + 1)) + wy * ((1 - wx) * at(iy + 1, ix) + wx * at(iy + 1, ix + 1));
      out[(size_t)v * side + u] = 1.f / (1.f + std::exp(-l));
    }
  }
}

// Detections back into the upright frame's pixel coordinates (the display bitmap: fw x fh), and
// for the -seg models their masks (jm: side x side per detection; empty = none).
int emit(const Fit& f, float* times, double t0, JNIEnv* e, jfloatArray jb, jintArray jl, jfloatArray js,
         jfloatArray jm, int side) {
  const double t2 = now_ms();
  std::vector<Det> d = g_detr ? post_detr(f.fw, f.fh) : g_post == "nms" ? post_nms() : post_end2end();
  const int cap = std::min<int>(e->GetArrayLength(js), (int)d.size());
  std::vector<float> b(4 * (size_t)cap), s(cap);
  std::vector<int> l(cap);
  for (int i = 0; i < cap; ++i) {
    b[4 * i] = d[i].x1 - f.left;
    b[4 * i + 1] = d[i].y1 - f.top;
    b[4 * i + 2] = d[i].x2 - f.left;
    b[4 * i + 3] = d[i].y2 - f.top;
    s[i] = d[i].s;
    l[i] = d[i].c + 1;  // Coco.name() is 1-based (0 = background)
  }
  e->SetFloatArrayRegion(jb, 0, 4 * cap, b.data());
  e->SetFloatArrayRegion(js, 0, cap, s.data());
  e->SetIntArrayRegion(jl, 0, cap, l.data());
  const int per = side * side;
  if (g_seg && jm && per > 0 && e->GetArrayLength(jm) >= (jsize)cap * per) {
    std::vector<float> m((size_t)cap * per);
    for (int i = 0; i < cap; ++i)
      if (d[i].s >= g_conf) seg_mask(d[i], side, m.data() + (size_t)i * per);  // shown ones only
    e->SetFloatArrayRegion(jm, 0, cap * per, m.data());
  }
  const double t3 = now_ms();
  times[3] = (float)(t3 - t2);
  times[0] = (float)(t3 - t0);
  return cap;
}

// engine=tinygrad: load <bundle>, read its input/outputs from meta.txt ("input <name> 1x640x640x3 uchar",
// "output <name> 1x84x8400 float", ...) and size the same buffers the HTP path uses
void init_tinygrad(const std::string& bundle, const std::string& cache_dir) {
  const double t = now_ms();
  g_tg = std::make_unique<tgcl::Model>();
  g_tg->load(bundle, cache_dir);
  g_tg_outs.clear();
  std::istringstream meta(tgcl::read_file(bundle + "/meta.txt"));
  std::string kind, name, dims, dt;
  std::vector<int64_t> in_shape;
  while (meta >> kind >> name >> dims >> dt) {
    std::vector<int64_t> shp;
    std::istringstream ds(dims);
    for (std::string d; std::getline(ds, d, 'x');) shp.push_back(std::stoll(d));
    if (kind == "input") g_in = name, in_shape = shp;
    else g_tg_outs.push_back({name, shp});
  }
  if (in_shape.size() != 4 || in_shape[3] != 3) throw std::runtime_error("tinygrad bundle: expected a uint8 NHWC input");
  g_S = (int)in_shape[1];
  g_q.assign((size_t)g_S * g_S * 3, 0);
  g_nm = 0;
  if (g_detr) {
    init_cat_map();
    for (auto& o : g_tg_outs)
      if (o.name == "logits") g_out = o.name, g_nq = (int)o.shape[1], g_ncls = (int)o.shape[2];
      else if (o.name == "boxes") g_out2 = o.name;
    if (!g_nq || g_out2.empty()) throw std::runtime_error("post=detr expects `logits` and `boxes` outputs");
    g_head.assign((size_t)g_nq * g_ncls, 0.f);
    g_boxes.assign((size_t)g_nq * 4, 0.f);
  } else {
    for (auto& o : g_tg_outs)
      if (o.shape.size() == 3) g_out = o.name, g_ch = (int)o.shape[1], g_n = (int)o.shape[2];
      else if (o.shape.size() == 4) g_seg = true, g_out_proto = o.name, g_nm = (int)o.shape[1], g_ph = (int)o.shape[2], g_pw = (int)o.shape[3];
    if (!g_ch) throw std::runtime_error("expected a (1, 4+nc[+nm], N) head output");
    g_nc = g_ch - 4 - g_nm;
    g_head.assign((size_t)g_ch * g_n, 0.f);
    g_proto.assign(g_seg ? (size_t)g_nm * g_ph * g_pw : 0, 0.f);
  }
  LOGI("tinygrad %s: %zu kernel calls, program %.0f ms, load %.0f ms", bundle.c_str(), g_tg->calls.size(), g_tg->build_ms,
       now_ms() - t);
}

void init(const std::string& dir, const std::string& lib_dir, const std::string& model, const std::string& opts) {
  const bool rf = model.rfind("rfdetr", 0) == 0;
  auto o = demo::parse_opts(opts, {{"post", rf ? "detr" : model.rfind("yolo26", 0) == 0 ? "end2end" : "nms"},
                                   {"htp_performance_mode", "burst"},
                                   {"engine", "qnn"},
                                   {"conf", rf ? "0.5" : "0.25"}});  // RF-DETR predict()'s default 0.5
  g_post = o["post"];
  g_detr = g_post == "detr";
  g_seg = false;
  g_conf = std::stof(o["conf"]);
  g_sess.reset();  // one model at a time: the previous session goes before the next loads
  g_tg.reset();
  if (o["engine"] == "tinygrad") {
    init_tinygrad(dir + "/" + model + ".tg", dir);
    return;
  }
  g_htp.init(lib_dir, "yolo");
  g_sess = g_htp.session(dir, model, o["htp_performance_mode"], "YoloDemo");
  Ort::AllocatorWithDefaultOptions a;
  g_in = g_sess->GetInputNameAllocated(0, a).get();
  g_out = g_sess->GetOutputNameAllocated(0, a).get();
  if (g_detr) {
    init_cat_map();
    g_S = (int)g_sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
    for (size_t i = 0; i < g_sess->GetOutputCount(); ++i) {
      std::string nm = g_sess->GetOutputNameAllocated(i, a).get();
      auto shp = g_sess->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      if (nm == "logits") {
        g_out = nm;
        g_nq = (int)shp[1];
        g_ncls = (int)shp[2];
      } else if (nm == "boxes") {
        g_out2 = nm;
      }
    }
    if (!g_nq || g_out2.empty()) throw std::runtime_error("post=detr expects `logits` and `boxes` outputs");
    g_q.assign((size_t)g_S * g_S * 3, 0);
    g_head.assign((size_t)g_nq * g_ncls, 0.f);
    g_boxes.assign((size_t)g_nq * 4, 0.f);
    LOGI("%s: %s (1,%d,%d,3) -> logits (1,%d,%d) + boxes, post detr", model.c_str(), g_in.c_str(), g_S, g_S, g_nq,
         g_ncls);
    return;
  }
  g_S = S;
  g_q.assign((size_t)S * S * 3, 0);
  g_nm = 0;
  for (size_t i = 0; i < g_sess->GetOutputCount(); ++i) {  // the head (rank 3) and, for -seg, the prototypes (rank 4)
    auto shp = g_sess->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::string nm = g_sess->GetOutputNameAllocated(i, a).get();
    if (shp.size() == 3) {
      g_out = nm;
      g_ch = (int)shp[1];
      g_n = (int)shp[2];
    } else if (shp.size() == 4) {
      g_seg = true;
      g_out_proto = nm;
      g_nm = (int)shp[1];
      g_ph = (int)shp[2];
      g_pw = (int)shp[3];
    }
  }
  if (!g_ch) throw std::runtime_error("expected a (1, 4+nc[+nm], N) head output");
  g_nc = g_ch - 4 - g_nm;
  g_head.assign((size_t)g_ch * g_n, 0.f);
  g_proto.assign(g_seg ? (size_t)g_nm * g_ph * g_pw : 0, 0.f);
  LOGI("%s: %s -> %s (1,%d,%d)%s, post %s", model.c_str(), g_in.c_str(), g_out.c_str(), g_ch, g_n,
       g_seg ? (" + " + g_out_proto + " prototypes (" + std::to_string(g_nm) + " masks)").c_str() : "", g_post.c_str());
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeInit(JNIEnv* e, jclass, jstring jdir,
                                                                                        jstring jlib, jstring jmodel,
                                                                                        jstring jopts) {
  auto str = [&](jstring s) {
    const char* c = e->GetStringUTFChars(s, nullptr);
    std::string r(c);
    e->ReleaseStringUTFChars(s, c);
    return r;
  };
  std::lock_guard<std::mutex> l(g_mu);
  try {
    init(str(jdir), str(jlib), str(jmodel), str(jopts));
    return nullptr;
  } catch (const std::exception& ex) {
    return e->NewStringUTF(ex.what());
  }
}

// Upright display size of a w x h sensor frame rotated by rot (the letterboxed content, <= 640).
extern "C" JNIEXPORT void JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeFitDims(JNIEnv* e, jclass, jint w,
                                                                                        jint h, jint rot, jintArray out) {
  Fit f = (rot % 180) ? fit(h, w) : fit(w, h);
  jint d[2] = {f.fw, f.fh};
  e->SetIntArrayRegion(out, 0, 2, d);
}

// Camera: YUV_420_888 planes -> rotated (clockwise rot), letterboxed RGB uint8 NHWC model input in
// one pass (nearest-neighbour; JFIF full-range BT.601, fixed point, as maskrcnn_engine.cpp), plus
// the upright letterboxed content as an RGBA display bitmap. Returns the detection count, -1 on
// error (nativeLastError), boxes in the display bitmap's pixels.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeRunYuv(
    JNIEnv* e, jclass, jobject jy, jobject ju, jobject jv, jint ys, jint uvs, jint uvps, jint w, jint h, jint rot,
    jobject disp, jfloatArray jb, jintArray jl, jfloatArray js, jfloatArray jm, jint side, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float times[4] = {0, 0, 0, 0};
  try {
    const demo::YuvPlanes P{(const uint8_t*)e->GetDirectBufferAddress(jy), (const uint8_t*)e->GetDirectBufferAddress(ju),
                            (const uint8_t*)e->GetDirectBufferAddress(jv), ys, uvs, uvps, w, h};
    int RW, RH;
    demo::upright_dims(P, rot, &RW, &RH);
    const Fit f = fit(RW, RH);
    AndroidBitmapInfo bi;
    uint8_t* dp = nullptr;
    if (disp && AndroidBitmap_getInfo(e, disp, &bi) == 0 && (int)bi.width == f.fw && (int)bi.height == f.fh &&
        AndroidBitmap_lockPixels(e, disp, (void**)&dp) != 0)
      dp = nullptr;
    if (g_detr) {  // the upright frame at display size, then stretched to the model's SxS input
      g_rgb.resize((size_t)f.fw * f.fh * 3);
      demo::yuv_upright(P, rot, f.fw, f.fh, 4, [&](int oy, int ox, uint8_t r, uint8_t g, uint8_t b) {
        uint8_t* o = g_rgb.data() + ((size_t)oy * f.fw + ox) * 3;
        o[0] = r;
        o[1] = g;
        o[2] = b;
        if (dp) {
          uint8_t* d = dp + (size_t)oy * bi.stride + 4 * ox;
          d[0] = r; d[1] = g; d[2] = b; d[3] = 255;
        }
      });
      if (dp) AndroidBitmap_unlockPixels(e, disp);
      stretch_into_input(g_rgb.data(), f.fw, f.fh, 3, (size_t)f.fw * 3);
      infer(times, t0);
      Fit fd = f;
      fd.left = fd.top = 0;  // no letterbox: boxes are already in the display frame's pixels
      int n = emit(fd, times, t0, e, jb, jl, js, jm, side);
      e->SetFloatArrayRegion(jt, 0, 4, times);
      return n;
    }
    pad_rows(f.top, f.fh);
    for (int oy = 0; oy < f.fh; ++oy) {  // the letterbox's left/right bars
      uint8_t* o = g_q.data() + ((size_t)(f.top + oy) * S) * 3;
      memset(o, kPad, (size_t)f.left * 3);
      memset(o + (size_t)(f.left + f.fw) * 3, kPad, (size_t)(S - f.left - f.fw) * 3);
    }
    demo::yuv_upright(P, rot, f.fw, f.fh, 4, [&](int oy, int ox, uint8_t r, uint8_t g, uint8_t b) {
      uint8_t* o = g_q.data() + ((size_t)(f.top + oy) * S + f.left + ox) * 3;
      o[0] = r;
      o[1] = g;
      o[2] = b;
      if (dp) {
        uint8_t* d = dp + (size_t)oy * bi.stride + 4 * ox;
        d[0] = r; d[1] = g; d[2] = b; d[3] = 255;
      }
    });
    if (dp) AndroidBitmap_unlockPixels(e, disp);
    infer(times, t0);
    int n = emit(f, times, t0, e, jb, jl, js, jm, side);
    e->SetFloatArrayRegion(jt, 0, 4, times);
    return n;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

// Images mode: an upright RGBA bitmap already scaled to fit 640x640 (YoloActivity.decodeFit) is
// centered into the letterbox. Boxes come back in the bitmap's pixels.
extern "C" JNIEXPORT jint JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeRun(JNIEnv* e, jclass, jobject bmp,
                                                                                    jfloatArray jb, jintArray jl,
                                                                                    jfloatArray js, jfloatArray jm,
                                                                                    jint side, jfloatArray jt) {
  std::lock_guard<std::mutex> l(g_mu);
  const double t0 = now_ms();
  float times[4] = {0, 0, 0, 0};
  try {
    AndroidBitmapInfo bi;
    uint8_t* px = nullptr;
    if (AndroidBitmap_getInfo(e, bmp, &bi) || bi.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        AndroidBitmap_lockPixels(e, bmp, (void**)&px))
      throw std::runtime_error("bitmap must be RGBA_8888");
    Fit f;
    f.fw = std::min<int>(bi.width, S);
    f.fh = std::min<int>(bi.height, S);
    f.left = (S - f.fw) / 2;
    f.top = (S - f.fh) / 2;
    if (g_detr) {  // stretch the whole bitmap; boxes come back in its pixels
      f.left = f.top = 0;
      stretch_into_input(px, f.fw, f.fh, 4, bi.stride);
      AndroidBitmap_unlockPixels(e, bmp);
      infer(times, t0);
      int n = emit(f, times, t0, e, jb, jl, js, jm, side);
      e->SetFloatArrayRegion(jt, 0, 4, times);
      return n;
    }
    pad_rows(f.top, f.fh);
    for (int y = 0; y < f.fh; ++y) {
      uint8_t* o = g_q.data() + ((size_t)(f.top + y) * S) * 3;
      memset(o, kPad, (size_t)f.left * 3);
      memset(o + (size_t)(f.left + f.fw) * 3, kPad, (size_t)(S - f.left - f.fw) * 3);
      o += (size_t)f.left * 3;
      const uint8_t* row = px + (size_t)y * bi.stride;
      for (int x = 0; x < f.fw; ++x) {
        o[3 * x] = row[4 * x];
        o[3 * x + 1] = row[4 * x + 1];
        o[3 * x + 2] = row[4 * x + 2];
      }
    }
    AndroidBitmap_unlockPixels(e, bmp);
    infer(times, t0);
    int n = emit(f, times, t0, e, jb, jl, js, jm, side);
    e->SetFloatArrayRegion(jt, 0, 4, times);
    return n;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return -1;
  }
}

extern "C" JNIEXPORT jstring JNICALL Java_org_onnxsim_maskrcnndemo_YoloEngine_nativeLastError(JNIEnv* e, jclass) {
  return e->NewStringUTF(g_err.c_str());
}
