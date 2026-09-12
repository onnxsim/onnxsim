/* w2v2fe_runner_realdata.c variant that also captures the full trainable
 * weight tensor (not just its w[0] readback) at a window of late-training
 * steps, plus the final state -- the missing ingredient for a multi-phase
 * calibration-swap phase 2 (docs/axera-audio-speech-op-coverage.md's "The
 * 2,000-step plateau" section): no earlier run persisted anything beyond a
 * per-step scalar printed to stderr, so there was no real trajectory data
 * on disk to recalibrate against, only w[0].
 *
 * Usage: w2v2fe_runner_capture model.axmodel steps warmup lr
 *        capture_start capture_stride capture_count out_dir
 *
 * Writes `out_dir/w_capture_<i>.bin` (raw float32, full weight tensor) for
 * `capture_count` steps starting at `capture_start` every `capture_stride`
 * steps, `out_dir/loss_capture.txt` (one loss value per captured step, same
 * order), and `out_dir/final.state0` (the last step's weight state, same
 * raw-float32 convention as resident_runner.c's own `.state0` seed files --
 * hand this directly to a phase-2 compiled model as its own `.state0`).
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
    if (argc < 9) {
        fprintf(stderr,
            "usage: %s model.axmodel steps warmup lr capture_start "
            "capture_stride capture_count out_dir\n", argv[0]);
        return 2;
    }
    int steps = atoi(argv[2]);
    int warmup = atoi(argv[3]);
    float lr_arg = (float)atof(argv[4]);
    int capture_start = atoi(argv[5]);
    int capture_stride = atoi(argv[6]);
    int capture_count = atoi(argv[7]);
    const char *out_dir = argv[8];

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

    void *w_host = malloc(out_sz[w_out]);
    char cap_path[1200], loss_path[1200];
    snprintf(loss_path, sizeof(loss_path), "%s/loss_capture.txt", out_dir);
    FILE *loss_f = fopen(loss_path, "w");
    if (!loss_f) { fprintf(stderr, "cannot write %s\n", loss_path); return 1; }
    int captured = 0;

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

        if (captured < capture_count && i >= capture_start &&
            (i - capture_start) % capture_stride == 0) {
            CK(axclrtMemcpy(w_host, in_bufs[w_in], out_sz[w_out], AXCL_MEMCPY_DEVICE_TO_HOST));
            snprintf(cap_path, sizeof(cap_path), "%s/w_capture_%d.bin", out_dir, captured);
            FILE *cf = fopen(cap_path, "wb");
            if (!cf) { fprintf(stderr, "cannot write %s\n", cap_path); return 1; }
            fwrite(w_host, 1, out_sz[w_out], cf);
            fclose(cf);
            fprintf(loss_f, "%d %.9g\n", i, loss_host);
            fflush(loss_f);
            captured++;
        }

        if ((i % 100) == 0 || i == steps - 1) {
            CK(axclrtMemcpy(&w0, in_bufs[w_in], sizeof(float), AXCL_MEMCPY_DEVICE_TO_HOST));
            fprintf(stderr, "step %d: %.3f ms  loss=%g  w[0]=%.10g captured=%d\n",
                    i, dt, loss_host, w0, captured);
        }
    }
    fclose(loss_f);

    {
        CK(axclrtMemcpy(w_host, in_bufs[w_in], out_sz[w_out], AXCL_MEMCPY_DEVICE_TO_HOST));
        char final_path[1200];
        snprintf(final_path, sizeof(final_path), "%s/final.state0", out_dir);
        FILE *ff = fopen(final_path, "wb");
        if (!ff) { fprintf(stderr, "cannot write %s\n", final_path); return 1; }
        fwrite(w_host, 1, out_sz[w_out], ff);
        fclose(ff);
    }
    free(w_host);

    double total = now_ms() - t_start;
    printf("steps=%d min=%.3fms avg=%.3fms total=%.3fms throughput=%.1f steps/s captured=%d\n",
           steps, best, sum / steps, total, 1000.0 * steps / total, captured);

    for (uint32_t i = 0; i < ni; i++) axclrtFree(in_bufs[i]);
    for (uint32_t i = 0; i < no; i++) axclrtFree(out_bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
