/* Phone-CPU baseline: time each real single-node RoiAlign model (make_roialign_single_node_models.py)
 * in ONNX Runtime's CPU EP on the phone itself, via the C API from the stock onnxruntime-android
 * AAR's arm64 libonnxruntime.so -- the op's actual current execution path in this pipeline. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "onnxruntime_c_api.h"
static const OrtApi* g;
#define CK(x) do { OrtStatus* s_ = (x); if (s_) { fprintf(stderr, "%s\n", g->GetErrorMessage(s_)); exit(1); } } while (0)
static void* slurp(const char* p, size_t* n) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); } fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }
int main(void) {
  g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env; CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "b", &env));
  OrtMemoryInfo* mi; CK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
  FILE* cf = fopen("calls.txt", "r"); int H, W, C, R, OH, OW, sr; float sc; int i = 0;
  double tot[2] = {0, 0};
  while (fscanf(cf, "%d %d %d %d %d %d %d %f", &H, &W, &C, &R, &OH, &OW, &sr, &sc) == 8) {
    char p[64]; size_t n;
    snprintf(p, sizeof p, "call%d_X.bin", i); void* x = slurp(p, &n); size_t xn = n;
    snprintf(p, sizeof p, "call%d_rois_nchw.bin", i); void* r = slurp(p, &n); size_t rn = n;
    snprintf(p, sizeof p, "call%d_bidx.bin", i); void* b = slurp(p, &n); size_t bn = n;
    int64_t xs[4] = {1, C, H, W}, rs[2] = {R, 4}, bs[1] = {R};
    OrtValue* in[3];
    CK(g->CreateTensorWithDataAsOrtValue(mi, x, xn, xs, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
    CK(g->CreateTensorWithDataAsOrtValue(mi, r, rn, rs, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]));
    CK(g->CreateTensorWithDataAsOrtValue(mi, b, bn, bs, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[2]));
    const char* inames[3] = {"X", "rois", "batch_indices"}; const char* onames[1] = {"Y"};
    for (int ti = 0; ti < 2; ti++) {
      OrtSessionOptions* so; CK(g->CreateSessionOptions(&so));
      CK(g->SetIntraOpNumThreads(so, ti == 0 ? 1 : 0));
      snprintf(p, sizeof p, "call%d.onnx", i);
      OrtSession* s; CK(g->CreateSession(env, p, so, &s));
      double ts[9];
      for (int k = 0; k < 10; k++) {
        OrtValue* y = NULL; double t0 = now_ms();
        CK(g->Run(s, NULL, inames, (const OrtValue* const*)in, 3, onames, 1, &y));
        if (k) ts[k - 1] = now_ms() - t0;
        g->ReleaseValue(y);
      }
      qsort(ts, 9, sizeof ts[0], cmpd); tot[ti] += ts[4];
      printf("call%d HxW=%dx%d R=%d out=%d ort_%s_ms(median)=%.3f\n", i, H, W, R, OH, ti ? "default" : "1thr", ts[4]);
      g->ReleaseSession(s); g->ReleaseSessionOptions(so);
    }
    i++;
  }
  printf("TOTAL ort_1thr_ms=%.2f ort_default_ms=%.2f\n", tot[0], tot[1]);
  return 0;
}
