/* Ask the AXCL engine what a compiled llm_build layer actually exposes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "axcl.h"

#define CK(e) do { axclError _r = (e); if (_r != 0) { \
    fprintf(stderr, "%s failed: 0x%x\n", #e, _r); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.axmodel\n", argv[0]); return 2; }
    CK(axclInit(NULL));
    axclrtDeviceList devs;
    CK(axclrtGetDeviceList(&devs));
    if (devs.num == 0) { fprintf(stderr, "no device\n"); return 1; }
    int32_t dev = devs.devices[0];
    CK(axclrtSetDevice(dev));
    CK(axclrtEngineInit(AXCL_VNPU_DISABLE));

    uint64_t modelId = 0;
    CK(axclrtEngineLoadFromFile(argv[1], &modelId));
    printf("compiler version: %s\n", axclrtEngineGetModelCompilerVersion(modelId));

    axclrtEngineIOInfo info;
    CK(axclrtEngineGetIOInfo(modelId, &info));
    int32_t groups = -1;
    axclrtEngineGetShapeGroupsCount(info, &groups);
    uint32_t ni = axclrtEngineGetNumInputs(info), no = axclrtEngineGetNumOutputs(info);
    printf("shape groups: %d\ninputs: %u\noutputs: %u\n", groups, ni, no);
    for (uint32_t g = 0; g < (groups > 0 ? (uint32_t)groups : 1u); g++) {
        printf("-- group %u\n", g);
        for (uint32_t i = 0; i < ni; i++)
            printf("   in  %2u %-16s %llu B\n", i, axclrtEngineGetInputNameByIndex(info, i),
                   (unsigned long long)axclrtEngineGetInputSizeByIndex(info, g, i));
        for (uint32_t i = 0; i < no; i++)
            printf("   out %2u %-16s %llu B\n", i, axclrtEngineGetOutputNameByIndex(info, i),
                   (unsigned long long)axclrtEngineGetOutputSizeByIndex(info, g, i));
    }
    axclrtEngineDestroyIOInfo(info);
    axclrtEngineUnload(modelId);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
