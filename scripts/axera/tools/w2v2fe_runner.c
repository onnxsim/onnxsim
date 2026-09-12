/* Resident runner for the wav2vec2 feature-extractor training-step
 * .axmodel (`../build_w2v2_feature_extractor_step.py`). I/O order confirmed
 * via probe_io on a real compiled model: inputs [x, y,
 * fe.conv_layers.0.conv.weight, lr, grad_seed], outputs [updated weight,
 * loss] -- one trainable state tensor, unlike resnet18's four or Whisper's
 * fourteen, so this is its own small runner rather than a generalization of
 * resident_runner.c's N_STATE=4 layout.
 *
 * Usage: w2v2fe_runner model.axmodel steps [warmup]
 *
 * Seeds the trainable weight from <model.axmodel>.state0 (raw float32
 * bytes, matching resident_runner.c's convention) and prints w[0] alongside
 * loss every step so a dying gradient (the weight freezing bit-identical
 * after a real step-0 update) is visible directly, the same diagnostic this
 * project has needed twice before (PRs #1343/#1346's loss=0 investigations,
 * PR #1359's Whisper step-1 death) -- reading the raw state buffer, not
 * trusting the runner's own loss output alone.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "axcl.h"

#define CK(e) do { axclError _r = (e); if (_r != 0) { \
    fprintf(stderr, "%s failed: 0x%x\n", #e, _r); return 1; } } while (0)

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.axmodel steps [warmup]\n", argv[0]);
        return 2;
    }
    int steps = atoi(argv[2]);
    int warmup = argc > 3 ? atoi(argv[3]) : 3;

    CK(axclInit(NULL));
    axclrtDeviceList devs; CK(axclrtGetDeviceList(&devs));
    if (!devs.num) { fprintf(stderr, "no device\n"); return 1; }
    CK(axclrtSetDevice(devs.devices[0]));
    CK(axclrtEngineInit(AXCL_VNPU_DISABLE));

    uint64_t modelId = 0, ctx = 0;
    CK(axclrtEngineLoadFromFile(argv[1], &modelId));
    CK(axclrtEngineCreateContext(modelId, &ctx));

    axclrtEngineIOInfo info; CK(axclrtEngineGetIOInfo(modelId, &info));
    uint32_t ni = axclrtEngineGetNumInputs(info), no = axclrtEngineGetNumOutputs(info);
    fprintf(stderr, "inputs=%u outputs=%u\n", ni, no);

    axclrtEngineIO io; CK(axclrtEngineCreateIO(info, &io));

    const int x_in = 0, y_in = 1, w_in = 2, lr_in = 3, seed_in = 4;
    const int w_out = 0, loss_out = 1;

    void *in_bufs[5] = {0}, *out_bufs[2] = {0};
    uint64_t in_sz[5], out_sz[2];

    for (uint32_t i = 0; i < ni; i++) {
        in_sz[i] = axclrtEngineGetInputSizeByIndex(info, 0, i);
        CK(axclrtMalloc(&in_bufs[i], in_sz[i], AXCL_MEM_MALLOC_NORMAL_ONLY));
        CK(axclrtEngineSetInputBufferByIndex(io, i, in_bufs[i], in_sz[i]));
    }
    for (uint32_t i = 0; i < no; i++) {
        out_sz[i] = axclrtEngineGetOutputSizeByIndex(info, 0, i);
        CK(axclrtMalloc(&out_bufs[i], out_sz[i], AXCL_MEM_MALLOC_NORMAL_ONLY));
        CK(axclrtEngineSetOutputBufferByIndex(io, i, out_bufs[i], out_sz[i]));
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s.state0", argv[1]);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path); return 1; }
    void *host = malloc(in_sz[w_in]);
    size_t got = fread(host, 1, in_sz[w_in], f);
    fclose(f);
    if (got != in_sz[w_in]) {
        fprintf(stderr, "short read %s (%zu != %llu)\n", path, got,
                (unsigned long long)in_sz[w_in]);
        return 1;
    }
    CK(axclrtMemcpy(in_bufs[w_in], host, in_sz[w_in], AXCL_MEMCPY_HOST_TO_DEVICE));
    free(host);

    { float lr = 1e-4f; CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE)); }
    { float seed = 1.0f; CK(axclrtMemcpy(in_bufs[seed_in], &seed, sizeof(seed), AXCL_MEMCPY_HOST_TO_DEVICE)); }

    void *hx = malloc(in_sz[x_in]);
    void *hy = malloc(in_sz[y_in]);
    memset(hx, 0x11, in_sz[x_in]);
    memset(hy, 0x22, in_sz[y_in]);

    float loss_host, w0;
    CK(axclrtMemcpy(&w0, in_bufs[w_in], sizeof(float), AXCL_MEMCPY_DEVICE_TO_HOST));
    fprintf(stderr, "initial w[0] = %g\n", w0);

    for (int i = 0; i < warmup; i++) {
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        CK(axclrtMemcpy(in_bufs[w_in], out_bufs[w_out], out_sz[w_out], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host), AXCL_MEMCPY_DEVICE_TO_HOST));
    }

    double best = 1e30, sum = 0, t_start = now_ms();
    for (int i = 0; i < steps; i++) {
        double t0 = now_ms();
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        CK(axclrtMemcpy(in_bufs[w_in], out_bufs[w_out], out_sz[w_out], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host), AXCL_MEMCPY_DEVICE_TO_HOST));
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        sum += dt;
        CK(axclrtMemcpy(&w0, in_bufs[w_in], sizeof(float), AXCL_MEMCPY_DEVICE_TO_HOST));
        fprintf(stderr, "step %d: %.3f ms  loss=%g  w[0]=%.10g\n", i, dt, loss_host, w0);
    }
    double total = now_ms() - t_start;
    printf("steps=%d min=%.3fms avg=%.3fms total=%.3fms throughput=%.1f steps/s\n",
           steps, best, sum / steps, total, 1000.0 * steps / total);

    for (uint32_t i = 0; i < ni; i++) axclrtFree(in_bufs[i]);
    for (uint32_t i = 0; i < no; i++) axclrtFree(out_bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
