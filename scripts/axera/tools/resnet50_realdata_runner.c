/* Resident runner for the resnet50 layer4.2+fc.weight training-step
 * .axmodel, generalizing resident_runner.c (N_STATE=4, same I/O layout:
 * inputs x, y, layer4.2.conv1.weight, layer4.2.conv2.weight,
 * layer4.2.conv3.weight, fc.weight, lr[, grad_seed]; outputs the four
 * updated states then loss) but reading REAL x0/y0/state0-3 data from
 * files instead of resident_runner.c's fixed memset test pattern, and
 * printing loss every step (not just the first 5) -- resident_runner.c's
 * memset pattern makes loss=0 ambiguous by construction (near-zero at both
 * ends, see docs/axera-on-device-training-handoff.md's resnet50 section);
 * this runner exists to give an unambiguous, monotonically-checkable real
 * loss curve across a real batch sweep.
 *
 * Usage: resnet50_realdata_runner model.axmodel steps [warmup] [lr] [-v]
 * Reads model.axmodel.state0/.state1/.state2/.state3/.x0/.y0.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "axcl.h"

#define CK(e) do { axclError _r = (e); if (_r != 0) { \
    fprintf(stderr, "%s failed: 0x%x\n", #e, _r); return 1; } } while (0)

#define N_STATE 4

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.axmodel steps [warmup] [lr] [-v]\n", argv[0]);
        return 2;
    }
    int steps = atoi(argv[2]);
    int warmup = argc > 3 ? atoi(argv[3]) : 3;
    float lr_arg = argc > 4 ? (float)atof(argv[4]) : 1e-4f;
    int vnpu_enable = 0;
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "-v") == 0) vnpu_enable = 1;

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

    /* input order: x, y, conv1, conv2, conv3, fc.weight, lr[, grad_seed]
     * output order: conv1', conv2', conv3', fc.weight', loss */
    const int state_in[N_STATE]  = {2, 3, 4, 5};
    const int state_out[N_STATE] = {0, 1, 2, 3};
    const int x_in = 0, y_in = 1, lr_in = 6, loss_out = 4;
    const int seed_in = 7;
    if (ni != 7 && ni != 8) {
        fprintf(stderr, "unexpected input count %u (expected 7 or 8)\n", ni);
        return 1;
    }

    void *in_bufs[8] = {0}, *out_bufs[5] = {0};
    uint64_t in_sz[8], out_sz[5];

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
    for (int k = 0; k < N_STATE; k++) {
        int i = state_in[k];
        snprintf(path, sizeof(path), "%s.state%d", argv[1], k);
        FILE *f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", path); return 1; }
        void *host = malloc(in_sz[i]);
        size_t got = fread(host, 1, in_sz[i], f);
        fclose(f);
        if (got != in_sz[i]) { fprintf(stderr, "short read %s (%zu != %llu)\n", path, got, (unsigned long long)in_sz[i]); return 1; }
        CK(axclrtMemcpy(in_bufs[i], host, in_sz[i], AXCL_MEMCPY_HOST_TO_DEVICE));
        free(host);
    }
    { float lr = lr_arg; fprintf(stderr, "lr=%g\n", lr); CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE)); }
    if (ni == 8) {
        float seed = 1.0f;
        CK(axclrtMemcpy(in_bufs[seed_in], &seed, sizeof(seed), AXCL_MEMCPY_HOST_TO_DEVICE));
    }

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

    float loss_host;
    for (int i = 0; i < warmup; i++) {
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        for (int k = 0; k < N_STATE; k++)
            CK(axclrtMemcpy(in_bufs[state_in[k]], out_bufs[state_out[k]],
                             out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host), AXCL_MEMCPY_DEVICE_TO_HOST));
    }
    fprintf(stderr, "post-warmup loss=%g\n", loss_host);

    double best = 1e30, sum = 0, t_start = now_ms();
    for (int i = 0; i < steps; i++) {
        double t0 = now_ms();
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        for (int k = 0; k < N_STATE; k++)
            CK(axclrtMemcpy(in_bufs[state_in[k]], out_bufs[state_out[k]],
                             out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host), AXCL_MEMCPY_DEVICE_TO_HOST));
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        sum += dt;
        fprintf(stderr, "step %d: %.3f ms  loss=%g\n", i, dt, loss_host);
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
