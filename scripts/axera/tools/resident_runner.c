/* Resident runner for a compiled onnxsim training-step .axmodel: loads the
 * model once, keeps every trainable-weight state buffer device-resident
 * between steps (bound in/out buffers are separate; after each Execute the
 * output is copied device-to-device back into the input buffer), and only
 * streams the batch (x, y) in and the loss out across the host boundary.
 *
 * Usage: resident_runner model.axmodel steps [warmup] [-n] [-v]
 *
 * -n: non-resident comparison mode. Instead of copying each step's updated
 * weight straight back device-to-device, round-trip it through a host
 * buffer (device -> host -> device) -- what a loop that treats the compiled
 * step graph as a stateless function, the same way the pre-residency design
 * did, would have to do. Isolates the residency win from the in-graph-update
 * graph-structure change itself.
 *
 * -v: run with AXCL_VNPU_ENABLE instead of AXCL_VNPU_DISABLE. Confirmed
 * non-corrupting (bit-identical output against -disable on this model) and
 * the lever for real concurrent throughput -- run several copies of this
 * binary at once, each against its own model-file copy, and the NPU
 * schedules them concurrently rather than serializing: aggregate throughput
 * scales (measured 1.7x at 2 concurrent contexts, 2.6x at 4, saturating by
 * 8) at the cost of per-context latency and ~7% off solo throughput even
 * alone. See docs/axera-on-device-training-handoff.md's "Execution overlap"
 * section for the numbers and for why axclrtEngineExecuteAsync (the other
 * overlap primitive AXCL exposes) is not an option here -- it returns
 * AXCL_ERR_UNSUPPORT on this device/SDK build.
 *
 * The model's own I/O order is fixed here rather than discovered generically
 * (see probe_io's dump): inputs x, y, _v_231, _v_233, _v_235, fc.weight, lr;
 * outputs sub_950 (_v_231'), sub_952 (_v_233'), sub_954 (_v_235'),
 * sub_956 (fc.weight'), loss. State pairing is positional: input i (for
 * i in 2..5) pairs with output i-2.
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
        fprintf(stderr, "usage: %s model.axmodel steps [warmup]\n", argv[0]);
        return 2;
    }
    int steps = atoi(argv[2]);
    int warmup = 5;
    int non_resident = 0, vnpu_enable = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0) non_resident = 1;
        else if (strcmp(argv[i], "-v") == 0) vnpu_enable = 1;
        else warmup = atoi(argv[i]);
    }
    fprintf(stderr, "mode: %s, %s\n",
            non_resident ? "non-resident (host round trip)" : "resident (device-to-device)",
            vnpu_enable ? "AXCL_VNPU_ENABLE" : "AXCL_VNPU_DISABLE");

    CK(axclInit(NULL));
    axclrtDeviceList devs; CK(axclrtGetDeviceList(&devs));
    if (!devs.num) { fprintf(stderr, "no device\n"); return 1; }
    CK(axclrtSetDevice(devs.devices[0]));
    CK(axclrtEngineInit(vnpu_enable ? AXCL_VNPU_ENABLE : AXCL_VNPU_DISABLE));

    uint64_t modelId = 0, ctx = 0;
    CK(axclrtEngineLoadFromFile(argv[1], &modelId));
    CK(axclrtEngineCreateContext(modelId, &ctx));

    axclrtEngineIOInfo info; CK(axclrtEngineGetIOInfo(modelId, &info));
    uint32_t ni = axclrtEngineGetNumInputs(info), no = axclrtEngineGetNumOutputs(info);
    fprintf(stderr, "inputs=%u outputs=%u\n", ni, no);

    axclrtEngineIO io; CK(axclrtEngineCreateIO(info, &io));

    /* input indices: 0=x 1=y 2.._v_231 3.._v_233 4.._v_235 5=fc.weight 6=lr
     * output indices: 0.._v_231' 1.._v_233' 2.._v_235' 3=fc.weight' 4=loss */
    const int state_in[N_STATE]  = {2, 3, 4, 5};
    const int state_out[N_STATE] = {0, 1, 2, 3};
    const int x_in = 0, y_in = 1, lr_in = 6, loss_out = 4;

    void *in_bufs[7] = {0}, *out_bufs[5] = {0};
    uint64_t in_sz[7], out_sz[5];

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

    /* seed the 4 trainable state inputs and lr from host files written
     * alongside the model (see push_and_run.sh): <argv[1]>.state<k> and
     * <argv[1]>.lr */
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
    {
        float lr = 1e-4f;
        CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE));
    }

    /* x/y: reused synthetic batch, re-uploaded every step exactly as a real
     * loop would upload a fresh batch (the point being measured is that nothing
     * ELSE crosses the bus, not that x/y are literally static). */
    void *hx = malloc(in_sz[x_in]);
    void *hy = malloc(in_sz[y_in]);
    memset(hx, 0x11, in_sz[x_in]);
    memset(hy, 0x22, in_sz[y_in]);

    float loss_host;
    void *state_host[N_STATE];
    for (int k = 0; k < N_STATE; k++) state_host[k] = malloc(out_sz[state_out[k]]);

    /* warmup */
    for (int i = 0; i < warmup; i++) {
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        for (int k = 0; k < N_STATE; k++) {
            if (non_resident) {
                CK(axclrtMemcpy(state_host[k], out_bufs[state_out[k]],
                                 out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_HOST));
                CK(axclrtMemcpy(in_bufs[state_in[k]], state_host[k],
                                 out_sz[state_out[k]], AXCL_MEMCPY_HOST_TO_DEVICE));
            } else {
                CK(axclrtMemcpy(in_bufs[state_in[k]], out_bufs[state_out[k]],
                                 out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_DEVICE));
            }
        }
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host),
                         AXCL_MEMCPY_DEVICE_TO_HOST));
    }

    double best = 1e30, sum = 0, t_start = now_ms();
    for (int i = 0; i < steps; i++) {
        double t0 = now_ms();
        CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        for (int k = 0; k < N_STATE; k++) {
            if (non_resident) {
                CK(axclrtMemcpy(state_host[k], out_bufs[state_out[k]],
                                 out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_HOST));
                CK(axclrtMemcpy(in_bufs[state_in[k]], state_host[k],
                                 out_sz[state_out[k]], AXCL_MEMCPY_HOST_TO_DEVICE));
            } else {
                CK(axclrtMemcpy(in_bufs[state_in[k]], out_bufs[state_out[k]],
                                 out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_DEVICE));
            }
        }
        CK(axclrtMemcpy(&loss_host, out_bufs[loss_out], sizeof(loss_host),
                         AXCL_MEMCPY_DEVICE_TO_HOST));
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        sum += dt;
        if (i < 5 || i == steps - 1)
            fprintf(stderr, "step %d: %.3f ms  loss=%g\n", i, dt, loss_host);
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
