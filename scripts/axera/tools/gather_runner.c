/* Minimal resident runner for the resident-dataset-Gather A/B comparison.
 * Explicitly feeds grad_seed (neither resident_runner.c nor
 * whisper_resident_runner.c do -- a real latent gap found while building
 * this), and supports both I/O layouts build_resident_train_step.py
 * produces:
 *   baseline: inputs = x, y, state[4], lr, grad_seed   (8 inputs)
 *   gather:   inputs = batch_index, state[4], lr, grad_seed (7 inputs)
 * outputs (both): state_out[4], loss (5 outputs)
 *
 * Usage: gather_runner model.axmodel steps [warmup] [-g] [-rN]
 *   -g    gather variant (batch_index input instead of x/y)
 *   -rN   dataset row count to index into, e.g. -r128 (default 4096,
 *         matching the original N=4096 investigation) -- indices are drawn
 *         mod N, so this must match (or undershoot) the row count the
 *         model was actually compiled/calibrated against, or the NPU reads
 *         an out-of-range row and faults.
 *
 * `batch_index` is declared `int64` in the ONNX graph
 * (`build_resident_train_step.add_resident_dataset`), but Pulsar2's
 * compiled `.axmodel` silently downcasts it to **int32** -- confirmed via
 * both the frontend's own calibration-time error message (`'indices':
 * Tensor(S32, name=batch_index, ...)`) and `probe_model_io`'s reported
 * input size (4 bytes for a declared-int64, shape-`[1]` tensor; int64 would
 * be 8). Writing `int64_t` values into that 4-byte buffer (an earlier
 * version of this file did) only half-initializes it, so the NPU reads a
 * garbage index and faults `axclrtEngineExecute` with `0x8030070c` --
 * found chasing `docs/axera-on-device-training-handoff.md`'s pre-flattened-
 * Gather workaround once it got past the original compile-time blocker.
 * This file writes `int32_t` indices to match what Pulsar2 actually
 * expects on-device, regardless of the ONNX-declared dtype.
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
    if (argc < 3) { fprintf(stderr, "usage: %s model.axmodel steps [warmup] [-g]\n", argv[0]); return 2; }
    int steps = atoi(argv[2]);
    int warmup = 5;
    int gather_mode = 0;
    int n_rows = 4096;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0) gather_mode = 1;
        else if (strncmp(argv[i], "-r", 2) == 0) n_rows = atoi(argv[i] + 2);
        else warmup = atoi(argv[i]);
    }
    fprintf(stderr, "mode: %s\n", gather_mode ? "gather (resident dataset)" : "baseline (x/y re-upload)");

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

    int64_t sys_bytes = 0, cmm_bytes = 0;
    CK(axclrtEngineGetUsageFromModelId(modelId, &sys_bytes, &cmm_bytes));
    double cmm_mib = cmm_bytes / (1024.0 * 1024.0);
    fprintf(stderr, "engine usage: cmm=%.3f MiB\n", cmm_mib);

    axclrtEngineIO io; CK(axclrtEngineCreateIO(info, &io));

    /* index layout, depending on mode */
    int state_in[N_STATE], state_out[N_STATE] = {0, 1, 2, 3};
    int x_in = -1, y_in = -1, batch_index_in = -1, lr_in, seed_in, loss_out = 4;
    if (gather_mode) {
        batch_index_in = 0;
        state_in[0]=1; state_in[1]=2; state_in[2]=3; state_in[3]=4;
        lr_in = 5; seed_in = 6;
    } else {
        x_in = 0; y_in = 1;
        state_in[0]=2; state_in[1]=3; state_in[2]=4; state_in[3]=5;
        lr_in = 6; seed_in = 7;
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
        if (got != in_sz[i]) { fprintf(stderr, "short read %s\n", path); return 1; }
        CK(axclrtMemcpy(in_bufs[i], host, in_sz[i], AXCL_MEMCPY_HOST_TO_DEVICE));
        free(host);
    }
    { float lr = 1e-4f; CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE)); }
    { float seed = 1.0f; CK(axclrtMemcpy(in_bufs[seed_in], &seed, sizeof(seed), AXCL_MEMCPY_HOST_TO_DEVICE)); }

    void *hx = NULL, *hy = NULL, *hidx = NULL;
    if (gather_mode) {
        hidx = malloc(in_sz[batch_index_in]);
        int32_t *idx = (int32_t *)hidx;
        uint64_t n_idx = in_sz[batch_index_in] / sizeof(int32_t);
        for (uint64_t i = 0; i < n_idx; i++) idx[i] = (int32_t)(i % (uint64_t)n_rows);
    } else {
        hx = malloc(in_sz[x_in]);
        hy = malloc(in_sz[y_in]);
        memset(hx, 0x11, in_sz[x_in]);
        memset(hy, 0x22, in_sz[y_in]);
    }

    float loss_host;
    void *state_host[N_STATE];
    for (int k = 0; k < N_STATE; k++) state_host[k] = malloc(out_sz[state_out[k]]);

    for (int i = 0; i < warmup; i++) {
        if (gather_mode) CK(axclrtMemcpy(in_bufs[batch_index_in], hidx, in_sz[batch_index_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        else {
            CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
            CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        }
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        for (int k = 0; k < N_STATE; k++)
            CK(axclrtMemcpy(in_bufs[state_in[k]], out_bufs[state_out[k]], out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host), AXCL_MEMCPY_DEVICE_TO_HOST));
    }

    double best = 1e30, sum = 0, t_start = now_ms();
    for (int i = 0; i < steps; i++) {
        double t0 = now_ms();
        if (gather_mode) CK(axclrtMemcpy(in_bufs[batch_index_in], hidx, in_sz[batch_index_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        else {
            CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
            CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        }
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        for (int k = 0; k < N_STATE; k++)
            CK(axclrtMemcpy(in_bufs[state_in[k]], out_bufs[state_out[k]], out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host), AXCL_MEMCPY_DEVICE_TO_HOST));
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        sum += dt;
        if (i < 5 || i == steps - 1) fprintf(stderr, "step %d: %.3f ms  loss=%g\n", i, dt, loss_host);
    }
    double total = now_ms() - t_start;
    printf("mode=%s steps=%d min=%.3fms avg=%.3fms total=%.3fms throughput=%.1f steps/s cmm=%.3fMiB\n",
           gather_mode ? "gather" : "baseline", steps, best, sum / steps, total, 1000.0 * steps / total, cmm_mib);

    for (uint32_t i = 0; i < ni; i++) axclrtFree(in_bufs[i]);
    for (uint32_t i = 0; i < no; i++) axclrtFree(out_bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
