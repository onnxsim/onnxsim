/* w2v2fe_runner_realdata.c variant: drops lr to a second value partway
 * through a continuous resident run, to test whether wav2vec2's PR #1376
 * loss plateau is ordinary SGD convergence at an oversized lr (loss should
 * resume decreasing once lr drops) or a genuinely stalled gradient (loss
 * stays flat regardless of lr). Weight state stays device-resident the
 * whole run -- lr changes are a host->device scalar write, not a restart.
 *
 * Usage: w2v2fe_runner_lrdrop model.axmodel steps switch_step lr1 lr2 [warmup]
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
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.axmodel steps switch_step lr1 lr2 [warmup]\n", argv[0]);
        return 2;
    }
    int steps = atoi(argv[2]);
    int switch_step = atoi(argv[3]);
    float lr1 = (float)atof(argv[4]);
    float lr2 = (float)atof(argv[5]);
    int warmup = argc > 6 ? atoi(argv[6]) : 3;

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
    if (got != in_sz[w_in]) { fprintf(stderr, "short read state0\n"); return 1; }
    CK(axclrtMemcpy(in_bufs[w_in], host, in_sz[w_in], AXCL_MEMCPY_HOST_TO_DEVICE));
    free(host);

    { float lr = lr1; fprintf(stderr, "lr=%g (phase 1)\n", lr); CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE)); }
    { float seed = 1.0f; CK(axclrtMemcpy(in_bufs[seed_in], &seed, sizeof(seed), AXCL_MEMCPY_HOST_TO_DEVICE)); }

    void *hx = malloc(in_sz[x_in]);
    void *hy = malloc(in_sz[y_in]);
    {
        char xpath[1024], ypath[1024];
        snprintf(xpath, sizeof(xpath), "%s.x0", argv[1]);
        snprintf(ypath, sizeof(ypath), "%s.y0", argv[1]);
        FILE *fx = fopen(xpath, "rb");
        FILE *fy = fopen(ypath, "rb");
        if (!fx || !fy) { fprintf(stderr, "missing x0/y0\n"); return 1; }
        if (fread(hx, 1, in_sz[x_in], fx) != in_sz[x_in]) { fprintf(stderr, "short read x0\n"); return 1; }
        if (fread(hy, 1, in_sz[y_in], fy) != in_sz[y_in]) { fprintf(stderr, "short read y0\n"); return 1; }
        fclose(fx); fclose(fy);
    }

    float loss_host, w0;
    for (int i = 0; i < warmup; i++) {
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        CK(axclrtMemcpy(in_bufs[w_in], out_bufs[w_out], out_sz[w_out], AXCL_MEMCPY_DEVICE_TO_DEVICE));
    }

    for (int i = 0; i < steps; i++) {
        if (i == switch_step) {
            float lr = lr2;
            fprintf(stderr, "=== switching lr -> %g at step %d ===\n", lr2, i);
            CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE));
        }
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        CK(axclrtMemcpy(in_bufs[w_in], out_bufs[w_out], out_sz[w_out], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host), AXCL_MEMCPY_DEVICE_TO_HOST));
        if (i % 25 == 0 || i == steps - 1 || (i > switch_step - 3 && i < switch_step + 20)) {
            CK(axclrtMemcpy(&w0, in_bufs[w_in], sizeof(float), AXCL_MEMCPY_DEVICE_TO_HOST));
            fprintf(stderr, "step %d: loss=%.8g  w[0]=%.10g\n", i, loss_host, w0);
        }
    }

    for (uint32_t i = 0; i < ni; i++) axclrtFree(in_bufs[i]);
    for (uint32_t i = 0; i < no; i++) axclrtFree(out_bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
