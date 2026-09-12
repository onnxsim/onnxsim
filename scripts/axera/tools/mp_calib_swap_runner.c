/* Minimal resident runner for the multi-phase calibration-swap demo
 * (docs/axera-on-device-training-handoff.md's "Multi-phase calibration
 * swap" section), against the small Conv+Gemm training step
 * build_multiphase_calib_swap_probe.py builds. Inputs: x, y, cw, gw, lr
 * (fixed order, matching that model's own I/O -- confirm with probe_io.c
 * before reusing against a different compile). Outputs:
 * resident_step__sub_74 (cw_next), resident_step__sub_76 (gw_next), loss.
 *
 * Usage: mp_calib_swap_runner model.axmodel steps cw.bin gw.bin y.bin lr
 *
 * Seeds cw/gw state from host .bin files (raw float32, no header) so the
 * same weight values can be handed from one compiled phase to the next
 * without any graph-level coupling between them. x is a small fixed
 * pattern baked in below (not a calibration/demo concern); y and lr are
 * host files/an argv float, respectively, so the exact "does the update
 * survive" experiment (a controlled lr change, or a near-converged y) can
 * be run without rebuilding this binary.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "axcl.h"

#define CK(e) do { axclError _r = (e); if (_r != 0) { \
    fprintf(stderr, "%s failed: 0x%x\n", #e, _r); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s model.axmodel steps cw.bin gw.bin y.bin lr\n", argv[0]);
        return 2;
    }
    int steps = atoi(argv[2]);

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

    /* indices per probe_io: 0=x 1=y 2=cw 3=gw 4=lr ; out 0=cw_next 1=gw_next 2=loss */
    void *in_bufs[5] = {0}, *out_bufs[3] = {0};
    uint64_t in_sz[5], out_sz[3];
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

    /* seed cw, gw from host files */
    {
        FILE *f = fopen(argv[3], "rb");
        void *h = malloc(in_sz[2]); size_t n = fread(h, 1, in_sz[2], f); (void)n; fclose(f);
        CK(axclrtMemcpy(in_bufs[2], h, in_sz[2], AXCL_MEMCPY_HOST_TO_DEVICE));
        free(h);
    }
    {
        FILE *f = fopen(argv[4], "rb");
        void *h = malloc(in_sz[3]); size_t n = fread(h, 1, in_sz[3], f); (void)n; fclose(f);
        CK(axclrtMemcpy(in_bufs[3], h, in_sz[3], AXCL_MEMCPY_HOST_TO_DEVICE));
        free(h);
    }
    { float lr = atof(argv[6]); CK(axclrtMemcpy(in_bufs[4], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE)); }

    /* x, y: fixed small pattern (not all-zero, matches calibration x_scale~0.3) */
    float hx[16]; for (int i = 0; i < 16; i++) hx[i] = 0.1f * ((i % 5) - 2);
    float hy[10];
    { FILE *f = fopen(argv[5], "rb"); size_t n = fread(hy, 1, sizeof(hy), f); (void)n; fclose(f); }
    CK(axclrtMemcpy(in_bufs[0], hx, in_sz[0], AXCL_MEMCPY_HOST_TO_DEVICE));
    CK(axclrtMemcpy(in_bufs[1], hy, in_sz[1], AXCL_MEMCPY_HOST_TO_DEVICE));

    float cw_before[18], cw_after[18], loss_v;
    CK(axclrtMemcpy(cw_before, in_bufs[2], sizeof(cw_before), AXCL_MEMCPY_DEVICE_TO_HOST));

    for (int i = 0; i < steps; i++) {
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        /* resident: copy state output back into state input, device-to-device */
        CK(axclrtMemcpy(in_bufs[2], out_bufs[0], out_sz[0], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[3], out_bufs[1], out_sz[1], AXCL_MEMCPY_DEVICE_TO_DEVICE));
    }
    CK(axclrtMemcpy(cw_after, out_bufs[0], sizeof(cw_after), AXCL_MEMCPY_DEVICE_TO_HOST));
    CK(axclrtMemcpy(&loss_v, out_bufs[2], sizeof(loss_v), AXCL_MEMCPY_DEVICE_TO_HOST));

    double max_abs_delta = 0;
    for (int i = 0; i < 18; i++) {
        double d = cw_after[i] - cw_before[i];
        if (d < 0) d = -d;
        if (d > max_abs_delta) max_abs_delta = d;
    }
    printf("cw[0] before=%.9g after=%.9g  max|delta|=%.9g  loss=%.9g  steps=%d\n",
           cw_before[0], cw_after[0], max_abs_delta, loss_v, steps);

    for (uint32_t i = 0; i < ni; i++) axclrtFree(in_bufs[i]);
    for (uint32_t i = 0; i < no; i++) axclrtFree(out_bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
