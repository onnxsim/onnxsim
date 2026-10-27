// BEVFormer-tiny's whole frame on the phone in one process: backbone (int8 HTP) -> feats dequantize
// (CPU) -> encoder (7 fp16 HTP pieces + 6 HVX MSDA calls, chain.h) -> decoder (fp16 HTP), the previous
// frame's BEV carried in-process (prev_bev = rot_idx gather of the last bev_embed, has_prev per frame).
//
// usage: frame_run <piece dir> <backbone.onnx> <decoder.onnx|split> <mode> <warmup> <reps> <out dir> <frame dir>...
//   decoder "split": dec_split.py's pieces (dpre, dmid0-4, dpost + dec_const/) in the piece dir, around
//   6 more HVX MSDA calls (one 50x50 map, 900 queries); the MSDA skel is shared under a mutex.
//   mode seq : frames one after another, each fully finished before the next; per-frame latency.
//   mode pipe: three stage threads (backbone | encoder | decoder) with two slots between stages, so the
//              HTP / HVX / CPU work of neighbouring frames overlaps; throughput and per-frame latency.
//              The encoder is still strictly sequential (frame t needs frame t-1's BEV).
//   mode conc: does the HTP run while the HVX runs? backbone alone, TSA+SCA calls alone, then both at
//              once from two threads.
//   frame dir: img.u8 (6,480,800,3; the backbone's quantized NHWC input), has_prev.f32 (1), can_bus.f32
//              (18), tsa_ref.f32 (2,2500,1,2), ref_cam.f32 (6,2500,4,2), vis.u8 (6,2500), rot_idx.i32
//              (2500; prev_bev row source, -1 = zero; from torchvision's own rotate, see e2e_msda.py)
//   piece dir also holds feats_q.txt: the backbone output's "scale zero_point". The backbone is batch 6
//   (one execute) or batch 1 (backbone1.q8: one execute per camera).
//   seq and pipe write <out dir>/f<i>_{bev,cls,bbox}.f32 for the last pass over the frames; pipe also
//   prints the max |diff| to seq's outputs when those exist (the two must agree bit for bit).
//   TG_ENC=<enc3.tghx> / TG_DEC=<decoder.tghx>: the encoder / decoder as tinygrad-compiled programs instead (the fork's
//   examples/dsp_graph_onnx.py --artifact on export_msda_op.py's enc3.msda.onnx / decoder.msda.onnx; tg_prog.h). TG_ENC also
//   reads shift.f32 (2) and bev_mask.f32 (6,2500,4) from each frame dir; the program's own inputs are staged in place. The two
//   programs both take the HMX and VTCM, so their runs are serialized (TG_CONC=1: not). A TG_ENC program with three outputs
//   (bev_embed, cls_scores, bbox_preds: the encoder and decoder merged into one graph) is the whole frame after the backbone;
//   the decoder stage then only hands its outputs on. (Two separate programs didn't fit the DSP process together: the second's
//   weight upload failed.)
// env: MSDA_THREADS (default 4), QNN_PERF, QNN_EXTRA, ORT_LOG
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "chain.h"
#include "tg_prog.h"

namespace {

struct Frame {
  std::vector<uint8_t> img, vis;
  std::vector<float> has_prev, can_bus, tsa_ref, ref_cam, shift, bev_mask;
  std::vector<int32_t> rot_idx;
};

template <class T>
std::vector<T> read_vec(const std::string& path, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(path, std::ios::binary);
  f.read((char*)v.data(), (std::streamsize)(n * sizeof(T)));
  if (!f) throw std::runtime_error("short/missing " + path);
  return v;
}

const int NQ = 2500, E = 256;
const size_t IMG = 6ull * 480 * 800 * 3, FEATS = 6ull * 256 * 15 * 25;
float feats_scale = 1;
int feats_zp = 0;

// Encoder-side per-frame inputs into the store (the encoder thread's own tensors).
void stage_inputs(const Frame& fr, const float* prev_bev, const uint8_t* feats_q) {
  float* feats = (float*)get("feats").p;
  for (size_t i = 0; i < FEATS; ++i) feats[i] = ((int)feats_q[i] - feats_zp) * feats_scale;
  float* pb = (float*)get("prev_bev").p;
  for (int q = 0; q < NQ; ++q) {
    int s = fr.rot_idx[q];
    if (s < 0 || !prev_bev) memset(pb + (size_t)q * E, 0, E * 4);
    else memcpy(pb + (size_t)q * E, prev_bev + (size_t)s * E, E * 4);
  }
  memcpy(get("has_prev").p, fr.has_prev.data(), 4);
  memcpy(get("can_bus").p, fr.can_bus.data(), 18 * 4);
  memcpy(get("tsa_ref").p, fr.tsa_ref.data(), fr.tsa_ref.size() * 4);
  memcpy(get("ref_cam").p, fr.ref_cam.data(), fr.ref_cam.size() * 4);
  memcpy(get("vis").p, fr.vis.data(), fr.vis.size());
}

// the same for a tinygrad encoder program (TG_ENC): straight into its packed input buffer, in its ONNX input order
// (feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask). A program whose first input is uint8 is export_msda_op.py's
// frame --resident: (feats_q, bev_state, rot_idx, has_prev, shift, can_bus, ref_cam, bev_mask), the backbone's uint8 feats as
// they are and the previous BEV kept on the DSP (sent once, as zeros), gathered there by rot_idx
void stage_tg(TgProg& p, const Frame& fr, const float* prev_bev, const uint8_t* feats_q) {
  if (p.in.at(0).dtype == 2) {
    memcpy(p.input(0), feats_q, FEATS);
    if (void* st = p.input(1)) memset(st, 0, (size_t)NQ * E * 4);
    float* ri = (float*)p.input(2);
    for (int q = 0; q < NQ; ++q) ri[q] = (float)fr.rot_idx[q];
    const std::vector<float>* rest[] = {&fr.has_prev, &fr.shift, &fr.can_bus, &fr.ref_cam, &fr.bev_mask};
    for (int i = 0; i < 5; ++i)
      if (void* d = p.input(3 + i)) memcpy(d, rest[i]->data(), rest[i]->size() * 4);
    return;
  }
  float* feats = (float*)p.input(0);
  for (size_t i = 0; i < FEATS; ++i) feats[i] = ((int)feats_q[i] - feats_zp) * feats_scale;
  float* pb = (float*)p.input(1);
  for (int q = 0; q < NQ; ++q) {
    int s = fr.rot_idx[q];
    if (s < 0 || !prev_bev) memset(pb + (size_t)q * E, 0, E * 4);
    else memcpy(pb + (size_t)q * E, prev_bev + (size_t)s * E, E * 4);
  }
  const std::vector<float>* rest[] = {&fr.has_prev, &fr.shift, &fr.can_bus, &fr.ref_cam, &fr.bev_mask};
  for (int i = 0; i < 5; ++i)
    if (void* d = p.input(2 + i)) memcpy(d, rest[i]->data(), rest[i]->size() * 4);
}

double med(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// A slot handoff between two stage threads: `full[s]` once the producer wrote slot s.
struct Slots {
  std::mutex m;
  std::condition_variable cv;
  int produced = 0, consumed = 0;  // frames
  const int n;
  explicit Slots(int n) : n(n) {}
  void wait_free(int f) {  // producer, before writing frame f's slot
    std::unique_lock<std::mutex> l(m);
    cv.wait(l, [&] { return f - consumed < n; });
  }
  void publish(int f) {
    { std::lock_guard<std::mutex> l(m); produced = f + 1; }
    cv.notify_all();
  }
  void wait_full(int f) {
    std::unique_lock<std::mutex> l(m);
    cv.wait(l, [&] { return produced > f; });
  }
  void release(int f) {
    { std::lock_guard<std::mutex> l(m); consumed = f + 1; }
    cv.notify_all();
  }
};

void write_out(const std::string& dir, int i, const std::string& what, const Buf& b) {
  FILE* f = fopen((dir + "/f" + std::to_string(i) + "_" + what + ".f32").c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write to " + dir);
  fwrite(b.p, 1, b.bytes, f);
  fclose(f);
}

float max_diff_to_file(const std::string& path, const Buf& b) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return -1;
  std::vector<float> v(b.bytes / 4);
  f.read((char*)v.data(), (std::streamsize)b.bytes);
  if (!f) return -1;
  float m = 0;
  for (size_t i = 0; i < v.size(); ++i) m = std::max(m, std::fabs(v[i] - ((const float*)b.p)[i]));
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 9) {
    fprintf(stderr, "usage: %s piece_dir backbone.onnx decoder.onnx seq|pipe|conc warmup reps out_dir frame_dir...\n", argv[0]);
    return 2;
  }
  const std::string pdir = argv[1], bb_path = argv[2], dec_path = argv[3], mode = argv[4], odir = argv[7];
  const int warmup = atoi(argv[5]), reps = atoi(argv[6]);
  try {
    struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
    if (msda_rpc_open("file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h_msda))
      throw std::runtime_error("msda_rpc_open failed");
    int32 prc = 0;
    msda_rpc_perf_vote(h_msda, 0, &prc);
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "frame_run");
    env->RegisterExecutionProviderLibrary("QNNExecutionProvider", "libonnxruntime_providers_qnn.so");
    for (const auto& d : env->GetEpDevices())
      if (std::string(d.EpName()) == "QNNExecutionProvider" && d.Device().Type() == OrtHardwareDeviceType_NPU)
        npu.push_back(d);
    if (npu.empty()) throw std::runtime_error("no QNN NPU ep device");

    const char* tg_enc = getenv("TG_ENC");
    const char* tg_dec = getenv("TG_DEC");
    std::map<std::string, Piece> pc;
    double create = 0, ms = 0;
    for (auto n : {"pre", "mid0", "post0", "mid1", "post1", "mid2", "post2"}) {
      if (tg_enc) break;
      pc[n] = load(pdir, n, &ms);
      create += ms;
    }
    TgProg tenc, tdec;
    std::mutex tg_mu;  // the two programs' runs (both take the HMX + VTCM) one at a time, unless TG_CONC=1
    const bool tg_conc = getenv("TG_CONC") && atoi(getenv("TG_CONC"));
    auto tg_run = [&](TgProg& p) {
      if (tg_conc) return p.run();
      std::lock_guard<std::mutex> l(tg_mu);
      p.run();
    };
    if (tg_enc) {
      double a = now_ms();
      tenc.load(tg_enc, ".", "tg_enc.so");
      create += now_ms() - a;
      tenc.omit_state_out = getenv("TG_OMIT_STATE_OUT") && atoi(getenv("TG_OMIT_STATE_OUT"));
      printf("tinygrad encoder %s: %d calls, %zu input bytes (%zu with the state on the DSP)%s\n", tg_enc, tenc.ncalls, tenc.ibytes,
             tenc.ibytes_res, tenc.omit_state_out ? ", state output left on the DSP (no bev_embed out)" : "");
    }
    if (tg_dec) {
      double a = now_ms();
      tdec.load(tg_dec, ".", "tg_dec.so");
      create += now_ms() - a;
      printf("tinygrad decoder %s: %d calls\n", tg_dec, tdec.ncalls);
    }
    auto split_path = [](const std::string& p) {
      size_t s = p.rfind('/');
      std::string d = s == std::string::npos ? "." : p.substr(0, s), f = s == std::string::npos ? p : p.substr(s + 1);
      return std::make_pair(d, f.substr(0, f.size() - 5));  // drop ".onnx"
    };
    auto [bd, bn] = split_path(bb_path);
    auto [dd, dn] = split_path(dec_path);
    Piece bb = load(bd, bn, &ms);
    create += ms;
    // decoder: one fp16 HTP graph, or "split" = dec_split.py's pieces around 6 HVX MSDA calls
    const bool dsplit = dec_path == "split";
    Piece dec;
    std::map<std::string, Piece> dp;
    if (dsplit) {
      for (auto n : {"dpre", "dmid0", "dmid1", "dmid2", "dmid3", "dmid4", "dpost"}) {
        dp[n] = load(pdir, n, &ms);
        create += ms;
      }
    } else if (!tg_dec && !(tg_enc && tenc.out.size() == 3)) {
      dec = load(dd, dn, &ms);
      create += ms;
    }
    printf("sessions create_ms %.1f\n", create);
    {
      std::ifstream q(pdir + "/feats_q.txt");
      if (!(q >> feats_scale >> feats_zp)) throw std::runtime_error("missing " + pdir + "/feats_q.txt");
    }
    if (bb.in.size() != 1 || bb.out.size() != 1 || (dec.s && dec.in.size() != 1))
      throw std::runtime_error("backbone: 1 in / 1 out, decoder: 1 in expected");

    std::vector<Frame> frames;
    for (int i = 8; i < argc; ++i) {
      const std::string d = argv[i];
      Frame fr;
      fr.img = read_vec<uint8_t>(d + "/img.u8", IMG);
      fr.vis = read_vec<uint8_t>(d + "/vis.u8", 6 * NQ);
      fr.has_prev = read_vec<float>(d + "/has_prev.f32", 1);
      fr.can_bus = read_vec<float>(d + "/can_bus.f32", 18);
      fr.tsa_ref = read_vec<float>(d + "/tsa_ref.f32", 2 * NQ * 2);
      fr.ref_cam = read_vec<float>(d + "/ref_cam.f32", 6 * NQ * 4 * 2);
      fr.rot_idx = read_vec<int32_t>(d + "/rot_idx.i32", NQ);
      if (tg_enc) {
        fr.shift = read_vec<float>(d + "/shift.f32", 2);
        fr.bev_mask = read_vec<float>(d + "/bev_mask.f32", 6 * NQ * 4);
      }
      frames.push_back(std::move(fr));
    }
    const int nf = (int)frames.size();

    // every tensor up front, so the stage threads only ever look up (never insert into) the store
    const int SL = 2;
    buf("feats", {6, 256, 15, 25});
    buf("prev_bev", {NQ, E});
    buf("has_prev", {1});
    buf("can_bus", {18});
    buf("tsa_ref", {2, NQ, 1, 2});
    buf("ref_cam", {6, NQ, 4, 2});
    buf("vis", {6, NQ}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    memset(buf("tsa_vis", {2, NQ}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8).p, 1, 2 * NQ);
    for (int s = 0; s < SL; ++s) {
      buf("img" + std::to_string(s), {6, 480, 800, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      buf("fq" + std::to_string(s), {6, 256, 15, 25}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      buf("bev" + std::to_string(s), {NQ, E});
    }
    const int DQ = 900;
    const bool tg_frame = tg_enc && tenc.out.size() == 3;
    if (tg_dec || tg_frame) {
      buf("cls_scores", {DQ, 10});
      buf("bbox_preds", {DQ, 10});
    }
    if (tg_frame)
      for (int s = 0; s < SL; ++s) {
        buf("cls" + std::to_string(s), {DQ, 10});
        buf("bbox" + std::to_string(s), {DQ, 10});
      }
    if (dsplit) {  // layer 0's kernel inputs are constants (dec_split.py const)
      std::vector<std::pair<std::string, std::vector<int64_t>>> cs = {
          {"q1", {DQ, E}}, {"doff", {DQ, 64}}, {"dw", {DQ, 32}}, {"refp", {DQ, 3}}, {"dref", {DQ, 2}}};
      for (auto& [n, shp] : cs) {
        Buf& b = buf("c_" + n, shp);
        std::vector<float> v = read_vec<float>(pdir + "/dec_const/" + n + ".f32", b.bytes / 4);
        memcpy(b.p, v.data(), b.bytes);
      }
      memset(buf("dvis", {1, DQ}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8).p, 1, DQ);
    }
    std::vector<float> last_bev(NQ * E);
    auto noop = [](double) {};
    // a batch-1 backbone (quantize.py's backbone1.q8) runs once per camera on slices of the same
    // buffers: six short HTP executes the pipelined runner can interleave with the encoder's pieces
    const int64_t bb_batch = bb.s->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[0];
    if (bb_batch != 1 && bb_batch != 6) throw std::runtime_error("backbone batch must be 1 or 6");
    printf("backbone batch %lld\n", (long long)bb_batch);
    auto do_bb = [&](int f, int s) {
      Buf &img = get("img" + std::to_string(s)), &fq = get("fq" + std::to_string(s));
      memcpy(img.p, frames[f % nf].img.data(), IMG);
      if (bb_batch == 6) {
        run(bb, {{bb.in[0], "img" + std::to_string(s)}}, {{bb.out[0], "fq" + std::to_string(s)}});
        return;
      }
      const char* in_name = bb.in[0].c_str();
      const char* out_name = bb.out[0].c_str();
      std::vector<int64_t> xs{1, 480, 800, 3}, ys{1, 256, 15, 25};
      for (int c = 0; c < 6; ++c) {
        Ort::Value x = Ort::Value::CreateTensor(cpu_mem, (uint8_t*)img.p + c * (IMG / 6), IMG / 6, xs.data(), 4,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
        Ort::Value y = Ort::Value::CreateTensor(cpu_mem, (uint8_t*)fq.p + c * (FEATS / 6), FEATS / 6, ys.data(), 4,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
        bb.s->Run(Ort::RunOptions{nullptr}, &in_name, &x, 1, &out_name, &y, 1);
      }
    };
    auto do_enc = [&](int f, int s_in, int s_out, auto&& mark) {
      const Frame& fr = frames[f % nf];
      const float* prev = fr.has_prev[0] > 0 ? last_bev.data() : nullptr;
      const uint8_t* fq = (const uint8_t*)get("fq" + std::to_string(s_in)).p;
      if (tg_enc) {
        stage_tg(tenc, fr, prev, fq);
        mark(-1);
        tg_run(tenc);
        mark(1);
        if (const void* b = tenc.output(0)) memcpy(get("bev" + std::to_string(s_out)).p, b, NQ * E * 4);
        if (tg_frame) {
          memcpy(get("cls" + std::to_string(s_out)).p, tenc.output(1), DQ * 10 * 4);
          memcpy(get("bbox" + std::to_string(s_out)).p, tenc.output(2), DQ * 10 * 4);
        }
      } else {
        stage_inputs(fr, prev, fq);
        mark(-1);  // host staging (dequantize + prev_bev gather)
        encoder(pc, mark, "bev" + std::to_string(s_out));
      }
      memcpy(last_bev.data(), get("bev" + std::to_string(s_out)).p, NQ * E * 4);
    };
    auto do_dec = [&](int s) {
      if (tg_frame) {
        memcpy(get("cls_scores").p, get("cls" + std::to_string(s)).p, DQ * 10 * 4);
        memcpy(get("bbox_preds").p, get("bbox" + std::to_string(s)).p, DQ * 10 * 4);
        return;
      }
      if (tg_dec) {
        memcpy(tdec.input(0), get("bev" + std::to_string(s)).p, NQ * E * 4);
        tg_run(tdec);
        memcpy(get("cls_scores").p, tdec.output(0), DQ * 10 * 4);
        memcpy(get("bbox_preds").p, tdec.output(1), DQ * 10 * 4);
        return;
      }
      if (!dsplit) {
        run(dec, {{dec.in[0], "bev" + std::to_string(s)}});
        return;
      }
      run(dp["dpre"], {{"bev_embed", "bev" + std::to_string(s)}});
      std::string q1 = "c_q1", rp = "c_refp", off = "c_doff", w = "c_dw", ref = "c_dref";
      for (int l = 0; l < 6; ++l) {
        msda("dv", (size_t)l * NQ * E, ref, off, w, "dvis", 1, 50, 50, 1, 1, 4, "dout", 0, DQ);
        if (l == 5) {
          run(dp["dpost"], {{"q1", q1}, {"refp", rp}});
          break;
        }
        const std::string nq = "dq1_" + std::to_string(l % 2), nr = "drefp_" + std::to_string(l % 2);
        run(dp["dmid" + std::to_string(l)], {{"q1", q1}, {"refp", rp}}, {{"q1o", nq}, {"refpo", nr}});
        q1 = nq, rp = nr, off = "doff", w = "dw", ref = "dref";
      }
    };
    auto save = [&](int f, int s, const std::string& dir) {
      write_out(dir, f % nf, "bev", get("bev" + std::to_string(s)));
      write_out(dir, f % nf, "cls", get("cls_scores"));
      write_out(dir, f % nf, "bbox", get("bbox_preds"));
    };
    // one warm frame creates every piece's output buffers before any threads start
    do_bb(0, 0);
    do_enc(0, 0, 0, noop);
    do_dec(0);

    if (mode == "seq") {
      std::vector<double> t_bb, t_st, t_enc, t_msda, t_dec, t_tot, t_edsp, t_ddsp;
      for (int it = 0; it < warmup + reps; ++it)
        for (int f = 0; f < nf; ++f) {
          double t0 = now_ms();
          do_bb(f, 0);
          double t1 = now_ms(), a = t1, st = 0, dsp = 0;
          do_enc(f, 0, 0, [&](double d) {
            double b = now_ms();
            if (d < 0) st += b - a;
            else if (d > 0) dsp += b - a;
            a = b;
          });
          double t2 = now_ms();
          do_dec(0);
          double t3 = now_ms();
          if (it >= warmup) {
            t_bb.push_back(t1 - t0); t_st.push_back(st); t_enc.push_back(t2 - t1 - st); t_msda.push_back(dsp);
            t_dec.push_back(t3 - t2); t_tot.push_back(t3 - t0);
            if (tg_enc) t_edsp.push_back(tenc.last_dsp_ms);
            if (tg_dec) t_ddsp.push_back(tdec.last_dsp_ms);
          }
          if (it == warmup + reps - 1) save(f, 0, odir);
        }
      printf("seq frame_ms median %.2f (n=%zu): backbone %.2f, host staging %.2f, encoder %.2f (msda calls %.2f), decoder %.2f\n",
             med(t_tot), t_tot.size(), med(t_bb), med(t_st), med(t_enc), med(t_msda), med(t_dec));
      if (tg_enc || tg_dec) printf("tinygrad programs' own DSP time: encoder %.2f, decoder %.2f\n", med(t_edsp), med(t_ddsp));
    } else if (mode == "pipe") {
      const int N = (warmup + reps) * nf;
      Slots bb2enc(SL), enc2dec(SL);
      std::vector<double> start(N), done(N);
      std::mutex em;
      auto guard = [&](auto&& fn) {
        return [&, fn] {
          try {
            fn();
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> l(em);
            printf("FAIL %s\n", e.what());
            fflush(stdout);
            _exit(3);
          }
        };
      };
      double T0 = now_ms();
      std::thread tb(guard([&] {
        for (int f = 0; f < N; ++f) {
          bb2enc.wait_free(f);
          start[f] = now_ms();
          do_bb(f, f % SL);
          bb2enc.publish(f);
        }
      }));
      std::thread te(guard([&] {
        for (int f = 0; f < N; ++f) {
          bb2enc.wait_full(f);
          enc2dec.wait_free(f);
          do_enc(f, f % SL, f % SL, noop);
          bb2enc.release(f);
          enc2dec.publish(f);
        }
      }));
      std::thread td(guard([&] {
        for (int f = 0; f < N; ++f) {
          enc2dec.wait_full(f);
          do_dec(f % SL);
          done[f] = now_ms();
          if (f >= N - nf) save(f, f % SL, odir);
          enc2dec.release(f);
        }
      }));
      tb.join(); te.join(); td.join();
      const int W0 = warmup * nf;
      std::vector<double> lat;
      for (int f = W0; f < N; ++f) lat.push_back(done[f] - start[f]);
      double span = done[N - 1] - done[W0 > 0 ? W0 - 1 : 0];
      int n = N - (W0 > 0 ? W0 : 1);
      printf("pipe frames %d: throughput %.2f ms/frame (%.1f FPS), latency median %.2f ms (wall %.0f ms total)\n", n,
             span / n, 1000.0 * n / span, med(lat), now_ms() - T0);
    } else if (mode == "conc") {
      // backbone alone, the 6 sampling calls of one encoder pass alone, then both from two threads
      do_enc(0, 0, 0, noop);  // encoder tensors hold a real frame now
      auto sampling = [&] {
        for (int i = 0; i < 3; ++i) {
          msda("tsa_v", 0, "tsa_ref", "tsa_off", "tsa_w", "tsa_vis", 2, 50, 50, 1, 2, 4, "tsa_out", i);
          msda("sca_v", (size_t)i * 6 * 375 * 256, "ref_cam", "sca_off", "sca_w", "vis", 6, 15, 25, 4, 1, 8, "sca_out", i);
        }
      };
      auto timeit = [&](auto&& fn) {
        std::vector<double> t;
        for (int it = 0; it < warmup + reps; ++it) {
          double a = now_ms();
          fn();
          if (it >= warmup) t.push_back(now_ms() - a);
        }
        return med(t);
      };
      double tb_alone = timeit([&] { do_bb(0, 0); });
      double ts_alone = timeit(sampling);
      double tdec_alone = timeit([&] { do_dec(0); });
      // both at once: each thread loops its work `reps` times; per-iteration medians while overlapping
      std::vector<double> tb_c, ts_c;
      std::thread t1([&] {
        for (int it = 0; it < reps; ++it) { double a = now_ms(); do_bb(0, 0); tb_c.push_back(now_ms() - a); }
      });
      std::thread t2([&] {
        for (int it = 0; it < reps; ++it) { double a = now_ms(); sampling(); ts_c.push_back(now_ms() - a); }
      });
      t1.join(); t2.join();
      std::vector<double> td_c, ts_c2;
      std::thread t3([&] {
        for (int it = 0; it < reps; ++it) { double a = now_ms(); do_dec(0); td_c.push_back(now_ms() - a); }
      });
      std::thread t4([&] {
        for (int it = 0; it < reps; ++it) { double a = now_ms(); sampling(); ts_c2.push_back(now_ms() - a); }
      });
      t3.join(); t4.join();
      printf("conc alone: backbone %.2f ms, 6 sampling calls %.2f ms, decoder %.2f ms\n", tb_alone, ts_alone, tdec_alone);
      printf("conc together: backbone %.2f ms || sampling %.2f ms;  decoder %.2f ms || sampling %.2f ms\n", med(tb_c),
             med(ts_c), med(td_c), med(ts_c2));
    } else {
      throw std::runtime_error("mode must be seq, pipe or conc");
    }
    if (mode == "pipe") {
      float m = 0;
      bool have = true;
      for (int f = 0; f < nf; ++f) {
        for (auto w : {"bev", "cls", "bbox"}) {
          std::string p = odir + "/../seq/f" + std::to_string(f) + "_" + w + ".f32";
          Buf b;
          std::vector<float> mine = read_vec<float>(odir + "/f" + std::to_string(f) + "_" + w + ".f32",
                                                    std::string(w) == "bev" ? NQ * E : 9000);
          b.p = mine.data();
          b.bytes = mine.size() * 4;
          float d = max_diff_to_file(p, b);
          if (d < 0) have = false;
          else m = std::max(m, d);
        }
      }
      if (have) printf("pipe vs seq outputs: max |diff| %g\n", m);
    }
    tenc.close();
    tdec.close();
    msda_rpc_close(h_msda);
    printf("PASS\n");
    fflush(stdout);
    _exit(0);
  } catch (const std::exception& e) {
    printf("FAIL %s\n", e.what());
    fflush(stdout);
    return 1;
  }
}
