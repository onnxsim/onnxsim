/* Time one shape group of a compiled model -- group 1 is an llm_build
 * layer's prefill subgraph, which axcl_run_model will not select. */
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
    if (argc < 3) { fprintf(stderr, "usage: %s model.axmodel group [repeat]\n", argv[0]); return 2; }
    uint32_t group = (uint32_t)atoi(argv[2]);
    int repeat = argc > 3 ? atoi(argv[3]) : 10;

    CK(axclInit(NULL));
    axclrtDeviceList devs; CK(axclrtGetDeviceList(&devs));
    if (!devs.num) { fprintf(stderr, "no device\n"); return 1; }
    CK(axclrtSetDevice(devs.devices[0]));
    CK(axclrtEngineInit(AXCL_VNPU_DISABLE));

    uint64_t modelId = 0, ctx = 0;
    CK(axclrtEngineLoadFromFile(argv[1], &modelId));
    CK(axclrtEngineCreateContext(modelId, &ctx));

    axclrtEngineIOInfo info; CK(axclrtEngineGetIOInfo(modelId, &info));
    int32_t groups = 0; axclrtEngineGetShapeGroupsCount(info, &groups);
    if ((int32_t)group >= groups) { fprintf(stderr, "group %u of %d\n", group, groups); return 1; }
    uint32_t ni = axclrtEngineGetNumInputs(info), no = axclrtEngineGetNumOutputs(info);

    axclrtEngineIO io; CK(axclrtEngineCreateIO(info, &io));
    void **bufs = calloc(ni + no, sizeof(void *));
    uint64_t total_in = 0;
    for (uint32_t i = 0; i < ni; i++) {
        uint64_t sz = axclrtEngineGetInputSizeByIndex(info, group, i);
        total_in += sz;
        CK(axclrtMalloc(&bufs[i], sz, AXCL_MEM_MALLOC_NORMAL_ONLY));
        CK(axclrtMemset(bufs[i], 0, sz));
        CK(axclrtEngineSetInputBufferByIndex(io, i, bufs[i], sz));
    }
    for (uint32_t i = 0; i < no; i++) {
        uint64_t sz = axclrtEngineGetOutputSizeByIndex(info, group, i);
        CK(axclrtMalloc(&bufs[ni + i], sz, AXCL_MEM_MALLOC_NORMAL_ONLY));
        CK(axclrtEngineSetOutputBufferByIndex(io, i, bufs[ni + i], sz));
    }

    for (int i = 0; i < 3; i++) CK(axclrtEngineExecute(modelId, ctx, group, io));
    double best = 1e30, sum = 0;
    for (int i = 0; i < repeat; i++) {
        double t0 = now_ms();
        CK(axclrtEngineExecute(modelId, ctx, group, io));
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        sum += dt;
    }
    printf("group %u: min %.3f ms  avg %.3f ms  (inputs %llu B)\n",
           group, best, sum / repeat, (unsigned long long)total_in);

    for (uint32_t i = 0; i < ni + no; i++) if (bufs[i]) axclrtFree(bufs[i]);
    axclrtEngineDestroyIO(io);
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
