/* Resident runner for the wav2vec2 encoder-attention training-step
 * .axmodel (`../build_w2v2_encoder_attn_step.py`) -- the first wav2vec2
 * build in this project whose trainable tail spans attention output,
 * exercising `onnxsim.graph_grad._grad_where`/`_grad_is_nan` on real
 * hardware. I/O order confirmed via probe_io on a real compiled model:
 * inputs [x, y, onnx::MatMul_371 (layer 0's q_proj weight), lr,
 * grad_seed], outputs [updated weight, loss] -- structurally identical to
 * `w2v2fe_runner_realdata.c`'s own I/O shape (one trainable state
 * tensor), so this is that runner with only the header comment and
 * variable naming updated, not new logic.
 *
 * Usage: w2v2_encoder_attn_runner model.axmodel steps [warmup] [lr] [-v]
 *
 * Seeds the trainable weight from <model.axmodel>.state0 (raw float32
 * bytes) and prints w[0] alongside loss every step -- this model's own
 * gradient is orders of magnitude smaller than the feature extractor's
 * own conv-weight gradient (confirmed on host,~1e-6 per element), so a
 * frozen w[0] here is a much weaker signal of a dead gradient than it was
 * for that script; compare against a real host-side reference run instead
 * of trusting this alone.
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
    float lr_arg = argc > 4 ? (float)atof(argv[4]) : 1e-4f;
    int vnpu_enable = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) vnpu_enable = 1;
    }

    CK(axclInit(NULL));
    axclrtDeviceList devs; CK(axclrtGetDeviceList(&devs));
    if (!devs.num) { fprintf(stderr, "no device\n"); return 1; }
    CK(axclrtSetDevice(devs.devices[0]));
    CK(axclrtEngineInit(vnpu_enable ? AXCL_VNPU_ENABLE : AXCL_VNPU_DISABLE));
    fprintf(stderr, "vnpu: %s\n", vnpu_enable ? "AXCL_VNPU_ENABLE" : "AXCL_VNPU_DISABLE");

    uint64_t modelId = 0, ctx = 0;
    CK(axclrtEngineLoadFromFile(argv[1], &modelId));
    CK(axclrtEngineCreateContext(modelId, &ctx));

    axclrtEngineIOInfo info; CK(axclrtEngineGetIOInfo(modelId, &info));
    uint32_t ni = axclrtEngineGetNumInputs(info), no = axclrtEngineGetNumOutputs(info);
    fprintf(stderr, "inputs=%u outputs=%u\n", ni, no);

    int64_t sys_bytes = 0, cmm_bytes = 0;
    CK(axclrtEngineGetUsageFromModelId(modelId, &sys_bytes, &cmm_bytes));
    double cmm_mib = cmm_bytes / (1024.0 * 1024.0);
    fprintf(stderr, "engine usage: sys=%lld B cmm=%.3f MiB\n",
            (long long)sys_bytes, cmm_mib);

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

    { float lr = lr_arg; fprintf(stderr, "lr=%g\n", lr); CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE)); }
    { float seed = 1.0f; CK(axclrtMemcpy(in_bufs[seed_in], &seed, sizeof(seed), AXCL_MEMCPY_HOST_TO_DEVICE)); }

    void *hx = malloc(in_sz[x_in]);
    void *hy = malloc(in_sz[y_in]);
    {
        char xpath[1024], ypath[1024];
        snprintf(xpath, sizeof(xpath), "%s.x0", argv[1]);
        snprintf(ypath, sizeof(ypath), "%s.y0", argv[1]);
        FILE *fx = fopen(xpath, "rb");
        FILE *fy = fopen(ypath, "rb");
        if (!fx || !fy) { fprintf(stderr, "missing %s or %s\n", xpath, ypath); return 1; }
        if (fread(hx, 1, in_sz[x_in], fx) != in_sz[x_in]) { fprintf(stderr, "short read x0\n"); return 1; }
        if (fread(hy, 1, in_sz[y_in], fy) != in_sz[y_in]) { fprintf(stderr, "short read y0\n"); return 1; }
        fclose(fx); fclose(fy);
    }

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
    printf("steps=%d min=%.3fms avg=%.3fms total=%.3fms throughput=%.1f steps/s cmm=%.3fMiB\n",
           steps, best, sum / steps, total, 1000.0 * steps / total, cmm_mib);

    for (uint32_t i = 0; i < ni; i++) axclrtFree(in_bufs[i]);
    for (uint32_t i = 0; i < no; i++) axclrtFree(out_bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
