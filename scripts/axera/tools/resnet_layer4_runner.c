/* Generalizes resnet50_realdata_runner.c's N_STATE=4 (`layer4.2` + `fc.weight`
 * only) resident runner to an arbitrary trainable-tensor count, for
 * `build_resnet50_layer4_step.py`'s `layer4_1_2` (7 states) and `layer4_all`
 * (10 states) scopes -- the concrete "remaining two bottleneck blocks of
 * layer4" next step docs/axera-on-device-training-handoff.md's "resnet50,
 * first compile" section names. Same I/O layout convention as every other
 * resident runner in this project (`qat_graph.make_step_graph`'s own
 * ordering): inputs x, y, state_0..N-1, lr[, grad_seed]; outputs
 * state_0'..N-1', loss.
 *
 * Usage: resnet_layer4_runner model.axmodel n_state steps [warmup] [lr] [-v]
 * Reads model.axmodel.state0..state<n_state-1>, .x0, .y0.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "axcl.h"

#define CK(e) do { axclError _r = (e); if (_r != 0) { \
    fprintf(stderr, "%s failed: 0x%x\n", #e, _r); return 1; } } while (0)

#define MAX_STATE 16

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.axmodel n_state steps [warmup] [lr] [-v]\n", argv[0]);
        return 2;
    }
    int n_state = atoi(argv[2]);
    if (n_state < 1 || n_state > MAX_STATE) {
        fprintf(stderr, "n_state out of range: %d\n", n_state);
        return 2;
    }
    int steps = atoi(argv[3]);
    int warmup = argc > 4 ? atoi(argv[4]) : 3;
    float lr_arg = argc > 5 ? (float)atof(argv[5]) : 1e-4f;
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
    fprintf(stderr, "inputs=%u outputs=%u n_state=%d\n", ni, no, n_state);

    int64_t sys_bytes = 0, cmm_bytes = 0;
    CK(axclrtEngineGetUsageFromModelId(modelId, &sys_bytes, &cmm_bytes));
    double cmm_mib = cmm_bytes / (1024.0 * 1024.0);
    fprintf(stderr, "engine usage: sys=%lld B cmm=%.3f MiB\n",
            (long long)sys_bytes, cmm_mib);

    axclrtEngineIO io; CK(axclrtEngineCreateIO(info, &io));

    /* input order: x, y, state_0..n_state-1, lr[, grad_seed]
     * output order: state_0'..n_state-1', loss */
    int state_in[MAX_STATE], state_out[MAX_STATE];
    for (int k = 0; k < n_state; k++) { state_in[k] = 2 + k; state_out[k] = k; }
    const int x_in = 0, y_in = 1;
    const int lr_in = 2 + n_state;
    const int seed_in = 3 + n_state;
    const int loss_out = n_state;
    int expect_ni_a = 3 + n_state, expect_ni_b = 4 + n_state;
    if ((int)ni != expect_ni_a && (int)ni != expect_ni_b) {
        fprintf(stderr, "unexpected input count %u (expected %d or %d)\n",
                ni, expect_ni_a, expect_ni_b);
        return 1;
    }

    void *in_bufs[MAX_STATE + 4] = {0}, *out_bufs[MAX_STATE + 1] = {0};
    uint64_t in_sz[MAX_STATE + 4], out_sz[MAX_STATE + 1];

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
    for (int k = 0; k < n_state; k++) {
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
    if ((int)ni == expect_ni_b) {
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
        for (int k = 0; k < n_state; k++)
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
        for (int k = 0; k < n_state; k++)
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
