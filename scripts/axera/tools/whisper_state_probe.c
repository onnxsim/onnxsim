/* Minimal probe: run N steps of a Whisper last_half resident step, dumping
 * one state tensor's raw device buffer directly before/after each step
 * (not the loss, which this project's history has shown can be a
 * misleading proxy at this quantization level -- see docs/axera-on-device-
 * training-handoff.md's "last_half actually trains" and "Recalibrating
 * Whisper for its real gradient scale" sections, both found this way).
 *
 * Reads x/y from fixed host files (/root/whisper_x.bin, /root/whisper_y.bin,
 * raw float32) rather than baking a pattern in, so the exact same batch can
 * be reused across differently-calibrated compiles of the same graph shape.
 *
 * Usage: whisper_state_probe model.axmodel steps
 * (seeds state from <model.axmodel>.state0..13, same convention as
 * whisper_resident_runner.c)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "axcl.h"

#define CK(e) do { axclError _r = (e); if (_r != 0) { \
    fprintf(stderr, "%s failed: 0x%x\n", #e, _r); return 1; } } while (0)

#define N_STATE 14

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.axmodel steps\n", argv[0]); return 2; }
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

    /* input indices: 0=x 1=y 2..15=14 state tensors 16=lr
     * output indices: 0..13=14 updated state tensors 14=loss */
    const int state_in[N_STATE]  = {2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    const int state_out[N_STATE] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13};
    const int x_in = 0, y_in = 1, lr_in = 16, loss_out = 14;

    void *in_bufs[17] = {0}, *out_bufs[15] = {0};
    uint64_t in_sz[17], out_sz[15];
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
        void *h = malloc(in_sz[i]); size_t got = fread(h, 1, in_sz[i], f); (void)got; fclose(f);
        CK(axclrtMemcpy(in_bufs[i], h, in_sz[i], AXCL_MEMCPY_HOST_TO_DEVICE));
        free(h);
    }
    { float lr = 1e-2f; CK(axclrtMemcpy(in_bufs[lr_in], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE)); }

    void *hx = malloc(in_sz[x_in]); void *hy = malloc(in_sz[y_in]);
    { FILE *f = fopen("/root/whisper_x.bin", "rb"); if (!f) { fprintf(stderr, "no x file\n"); return 1; }
      size_t got=fread(hx,1,in_sz[x_in],f); (void)got; fclose(f); }
    { FILE *f = fopen("/root/whisper_y.bin", "rb"); if (!f) { fprintf(stderr, "no y file\n"); return 1; }
      size_t got=fread(hy,1,in_sz[y_in],f); (void)got; fclose(f); }
    CK(axclrtMemcpy(in_bufs[x_in], hx, in_sz[x_in], AXCL_MEMCPY_HOST_TO_DEVICE));
    CK(axclrtMemcpy(in_bufs[y_in], hy, in_sz[y_in], AXCL_MEMCPY_HOST_TO_DEVICE));

    float before[8], after[8], loss_v;
    for (int step = 0; step < steps; step++) {
        CK(axclrtMemcpy(before, in_bufs[state_in[0]], sizeof(before), AXCL_MEMCPY_DEVICE_TO_HOST));
        CK(axclrtEngineExecute(modelId, ctx, 0, io));
        CK(axclrtMemcpy(after, out_bufs[state_out[0]], sizeof(after), AXCL_MEMCPY_DEVICE_TO_HOST));
        CK(axclrtMemcpy(&loss_v, out_bufs[loss_out], sizeof(loss_v), AXCL_MEMCPY_DEVICE_TO_HOST));
        double maxd = 0;
        for (int i = 0; i < 8; i++) { double d = after[i]-before[i]; if (d<0) d=-d; if (d>maxd) maxd=d; }
        printf("step %d: loss=%.9g before=[%.6g,%.6g,...] after=[%.6g,%.6g,...] max|delta|=%.6g\n",
               step, loss_v, before[0], before[1], after[0], after[1], maxd);
        for (int k = 0; k < N_STATE; k++) {
            CK(axclrtMemcpy(in_bufs[state_in[k]], out_bufs[state_out[k]], out_sz[state_out[k]], AXCL_MEMCPY_DEVICE_TO_DEVICE));
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
