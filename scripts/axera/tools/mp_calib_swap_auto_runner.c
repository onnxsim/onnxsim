/* Automatic-swap-trigger variant of mp_calib_swap_runner.c: runs a live,
 * self-monitoring resident loop against the same Conv+Gemm multi-phase
 * calibration-swap demo (docs/axera-on-device-training-handoff.md's
 * "Multi-phase calibration swap" section), and decides *on its own*,
 * step by step, whether the update has died -- instead of a human running
 * a fixed step count and eyeballing the printed numbers afterward (what
 * mp_calib_swap_runner.c and PR #1356's own demonstration did).
 *
 * Death detector: same zero-fraction principle as finetune.py's
 * LossScaler.zero_fraction, but checks the state OUTPUT itself for
 * collapsing to zero (`fabsf(after[i]) < eps`), not "did it change from
 * before" -- this project's own prior investigations found *two* distinct
 * death signatures, and a detector keyed on "unchanged" would miss the
 * second one entirely: PR #1357's real Whisper run found grad=0 exactly,
 * so `w_next == w` (unchanged); PR #1356's own manual demo, feeding a
 * weight far outside its calibrated range, found `w_next` reads as exactly
 * 0 regardless of a clearly nonzero `w` (the input itself gets crushed at
 * quantization, before any gradient is even computed) -- `before != after`
 * there, so an "unchanged" check would have called that step healthy. A
 * near-total fraction of the *output* reading as zero (>=0.99) covers both:
 * whether `w` stayed put or was written to hard zero, the update carried
 * no real information forward either way.
 *
 * `inject_step` (-1 = never) optionally overwrites the live state with a
 * deliberately different-scale value at a chosen step, from `inject_cw`/
 * `inject_gw` host files -- simulating "training has reached a later point
 * with much smaller weights/gradients" without needing an actual multi-
 * thousand-step real convergence run to get there (the same shortcut
 * PR #1356's own hardware demonstration used: feed a late-training-scale
 * weight directly, rather than train to it). The detector runs completely
 * blind to whether/when this injection happens -- it only ever sees the
 * live buffer contents, which is what makes its trigger genuinely
 * automatic rather than a human-timed swap.
 *
 * On death: writes the last-known-good state to out_cw/out_gw (raw
 * float32, no header, same convention as mp_calib_swap_runner.c's input
 * files) so a host orchestrator can hand it to the next compiled phase's
 * run of this same binary, and exits 3. On completing every step without
 * death, writes the final state the same way and exits 0.
 *
 * Usage: mp_calib_swap_auto_runner model.axmodel max_steps cw.bin gw.bin
 *        y.bin lr inject_step inject_cw.bin inject_gw.bin out_cw.bin out_gw.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "axcl.h"

#define CK(e) do { axclError _r = (e); if (_r != 0) { \
    fprintf(stderr, "%s failed: 0x%x\n", #e, _r); return 1; } } while (0)

#define CW_N 18
#define GW_N 320
#define DEATH_ZERO_FRACTION 0.99

static void read_bin(const char *path, void *buf, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path); exit(1); }
    size_t got = fread(buf, 1, n, f);
    (void)got;
    fclose(f);
}

static void write_bin(const char *path, const void *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite(buf, 1, n, f);
    fclose(f);
}

/* Fraction of `after` that carries no signal: either it stayed exactly at
 * `before` (grad computed as exactly zero) or it collapsed to hard zero
 * while `before` was not already zero (the input got crushed at
 * quantization). Either way, `after` is not usable as a real update. */
static double dead_fraction(const float *before, const float *after, int n) {
    int dead = 0;
    for (int i = 0; i < n; i++) {
        int unchanged = (after[i] == before[i]);
        int crushed_to_zero = (after[i] == 0.0f && before[i] != 0.0f);
        if (unchanged || crushed_to_zero) dead++;
    }
    return (double)dead / n;
}

int main(int argc, char **argv) {
    if (argc < 12) {
        fprintf(stderr,
            "usage: %s model.axmodel max_steps cw.bin gw.bin y.bin lr "
            "inject_step inject_cw.bin inject_gw.bin out_cw.bin out_gw.bin\n",
            argv[0]);
        return 2;
    }
    int max_steps = atoi(argv[2]);
    float lr = atof(argv[6]);
    int inject_step = atoi(argv[7]);
    const char *inject_cw_path = argv[8], *inject_gw_path = argv[9];
    const char *out_cw_path = argv[10], *out_gw_path = argv[11];

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

    { float h[CW_N]; read_bin(argv[3], h, sizeof(h));
      CK(axclrtMemcpy(in_bufs[2], h, in_sz[2], AXCL_MEMCPY_HOST_TO_DEVICE)); }
    { float h[GW_N]; read_bin(argv[4], h, sizeof(h));
      CK(axclrtMemcpy(in_bufs[3], h, in_sz[3], AXCL_MEMCPY_HOST_TO_DEVICE)); }
    CK(axclrtMemcpy(in_bufs[4], &lr, sizeof(lr), AXCL_MEMCPY_HOST_TO_DEVICE));

    float hx[16]; for (int i = 0; i < 16; i++) hx[i] = 0.1f * ((i % 5) - 2);
    float hy[10]; read_bin(argv[5], hy, sizeof(hy));
    CK(axclrtMemcpy(in_bufs[0], hx, in_sz[0], AXCL_MEMCPY_HOST_TO_DEVICE));
    CK(axclrtMemcpy(in_bufs[1], hy, in_sz[1], AXCL_MEMCPY_HOST_TO_DEVICE));

    float cw_before[CW_N], gw_before[GW_N], cw_after[CW_N], gw_after[GW_N], loss_v;
    int died = 0, death_step = -1;

    for (int step = 0; step < max_steps; step++) {
        if (step == inject_step) {
            float h_cw[CW_N]; read_bin(inject_cw_path, h_cw, sizeof(h_cw));
            float h_gw[GW_N]; read_bin(inject_gw_path, h_gw, sizeof(h_gw));
            CK(axclrtMemcpy(in_bufs[2], h_cw, in_sz[2], AXCL_MEMCPY_HOST_TO_DEVICE));
            CK(axclrtMemcpy(in_bufs[3], h_gw, in_sz[3], AXCL_MEMCPY_HOST_TO_DEVICE));
            fprintf(stderr, "step %d: injected alternate-scale state\n", step);
        }
        CK(axclrtMemcpy(cw_before, in_bufs[2], sizeof(cw_before), AXCL_MEMCPY_DEVICE_TO_HOST));
        CK(axclrtMemcpy(gw_before, in_bufs[3], sizeof(gw_before), AXCL_MEMCPY_DEVICE_TO_HOST));

        CK(axclrtEngineExecute(modelId, ctx, 0, io));

        CK(axclrtMemcpy(cw_after, out_bufs[0], sizeof(cw_after), AXCL_MEMCPY_DEVICE_TO_HOST));
        CK(axclrtMemcpy(gw_after, out_bufs[1], sizeof(gw_after), AXCL_MEMCPY_DEVICE_TO_HOST));
        CK(axclrtMemcpy(&loss_v, out_bufs[2], sizeof(loss_v), AXCL_MEMCPY_DEVICE_TO_HOST));

        float combined_before[CW_N + GW_N], combined_after[CW_N + GW_N];
        memcpy(combined_before, cw_before, sizeof(cw_before));
        memcpy(combined_before + CW_N, gw_before, sizeof(gw_before));
        memcpy(combined_after, cw_after, sizeof(cw_after));
        memcpy(combined_after + CW_N, gw_after, sizeof(gw_after));
        double zf = dead_fraction(combined_before, combined_after, CW_N + GW_N);

        fprintf(stderr, "step %d: loss=%.9g dead_fraction=%.4f cw[0]=%.9g->%.9g\n",
                step, loss_v, zf, cw_before[0], cw_after[0]);

        if (zf >= DEATH_ZERO_FRACTION) {
            died = 1;
            death_step = step;
            /* last-known-good state is the pre-step input, since the
             * update that just happened contributed (numerically)
             * nothing. */
            write_bin(out_cw_path, cw_before, sizeof(cw_before));
            write_bin(out_gw_path, gw_before, sizeof(gw_before));
            break;
        }

        CK(axclrtMemcpy(in_bufs[2], out_bufs[0], out_sz[0], AXCL_MEMCPY_DEVICE_TO_DEVICE));
        CK(axclrtMemcpy(in_bufs[3], out_bufs[1], out_sz[1], AXCL_MEMCPY_DEVICE_TO_DEVICE));

        if (step == max_steps - 1) {
            write_bin(out_cw_path, cw_after, sizeof(cw_after));
            write_bin(out_gw_path, gw_after, sizeof(gw_after));
        }
    }

    if (died) {
        printf("DEATH_DETECTED step=%d\n", death_step);
    } else {
        printf("COMPLETED steps=%d\n", max_steps);
    }

    for (uint32_t i = 0; i < ni; i++) axclrtFree(in_bufs[i]);
    for (uint32_t i = 0; i < no; i++) axclrtFree(out_bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return died ? 3 : 0;
}
