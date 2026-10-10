// Declarations-only stand-in for the AXCL SDK's <axcl.h>, for syntax-checking
// remote_axcl_worker.cpp on a host without the SDK:
//
//   cd tools/onnx-remote
//   g++ -std=c++17 -fsyntax-only -Wall -Wextra -I . -I test/axcl_stub remote_axcl_worker.cpp
//
// It declares only what that file uses, with signatures written from the uses
// in scripts/axera/vm/axcl_batch_runner.c (which is built against the real
// SDK; AXCL_MEMCPY_DEVICE_TO_DEVICE is used there) and, for axclrtMemset, in
// scripts/axera/tools/bench_shape_group.c. There are no definitions: nothing
// can link against it, and the real
// build (-DONNX_REMOTE_AXCL=ON) never looks here -- it finds axcl.h through
// AXCL_INCLUDE_DIR.
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t axclError;

typedef struct {
  uint32_t num;
  int32_t devices[256];
} axclrtDeviceList;

typedef enum { AXCL_VNPU_DISABLE = 0 } axclrtEngineVNpuKind;
typedef enum { AXCL_MEM_MALLOC_NORMAL_ONLY = 1 } axclrtMemMallocPolicy;
typedef enum {
  AXCL_MEMCPY_HOST_TO_DEVICE = 1,
  AXCL_MEMCPY_DEVICE_TO_HOST = 2,
  AXCL_MEMCPY_DEVICE_TO_DEVICE = 3
} axclrtMemcpyKind;
typedef enum { AXCL_DATA_TYPE_NONE = 0 } axclrtEngineDataType;

typedef void* axclrtEngineIOInfo;
typedef void* axclrtEngineIO;

typedef struct {
  int32_t dimCount;
  int32_t dims[32];
} axclrtEngineIODims;

axclError axclInit(const char* config);
axclError axclFinalize(void);
axclError axclrtGetDeviceList(axclrtDeviceList* devices);
axclError axclrtSetDevice(int32_t device);

axclError axclrtMalloc(void** device_ptr, size_t size, axclrtMemMallocPolicy policy);
axclError axclrtFree(void* device_ptr);
axclError axclrtMemcpy(void* dst, const void* src, size_t count, axclrtMemcpyKind kind);
axclError axclrtMemset(void* device_ptr, uint8_t value, size_t count);

axclError axclrtEngineInit(axclrtEngineVNpuKind kind);
axclError axclrtEngineFinalize(void);
axclError axclrtEngineLoadFromFile(const char* path, uint64_t* model_id);
axclError axclrtEngineUnload(uint64_t model_id);
axclError axclrtEngineCreateContext(uint64_t model_id, uint64_t* context_id);
axclError axclrtEngineGetIOInfo(uint64_t model_id, axclrtEngineIOInfo* info);
axclError axclrtEngineDestroyIOInfo(axclrtEngineIOInfo info);
axclError axclrtEngineCreateIO(axclrtEngineIOInfo info, axclrtEngineIO* io);
axclError axclrtEngineDestroyIO(axclrtEngineIO io);
uint32_t axclrtEngineGetNumInputs(axclrtEngineIOInfo info);
uint32_t axclrtEngineGetNumOutputs(axclrtEngineIOInfo info);
axclError axclrtEngineGetInputDataType(axclrtEngineIOInfo info, uint32_t index,
                                       axclrtEngineDataType* type);
axclError axclrtEngineGetOutputDataType(axclrtEngineIOInfo info, uint32_t index,
                                        axclrtEngineDataType* type);
uint64_t axclrtEngineGetInputSizeByIndex(axclrtEngineIOInfo info, uint32_t group,
                                         uint32_t index);
uint64_t axclrtEngineGetOutputSizeByIndex(axclrtEngineIOInfo info, uint32_t group,
                                          uint32_t index);
axclError axclrtEngineGetInputDims(axclrtEngineIOInfo info, uint32_t group,
                                   uint32_t index, axclrtEngineIODims* dims);
axclError axclrtEngineGetOutputDims(axclrtEngineIOInfo info, uint32_t group,
                                    uint32_t index, axclrtEngineIODims* dims);
const char* axclrtEngineGetInputNameByIndex(axclrtEngineIOInfo info, uint32_t index);
const char* axclrtEngineGetOutputNameByIndex(axclrtEngineIOInfo info, uint32_t index);
axclError axclrtEngineSetInputBufferByIndex(axclrtEngineIO io, uint32_t index,
                                            const void* buffer, uint64_t size);
axclError axclrtEngineSetOutputBufferByIndex(axclrtEngineIO io, uint32_t index,
                                             const void* buffer, uint64_t size);
axclError axclrtEngineExecute(uint64_t model_id, uint64_t context_id, uint32_t group,
                              axclrtEngineIO io);

#ifdef __cplusplus
}
#endif
