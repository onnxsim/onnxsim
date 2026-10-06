// Peak-ish DRAM read bandwidth of a Jetson GPU: the roofline for batch-1 LLM decode.
//   nvcc -O3 -arch=sm_87 read_bandwidth.cu -o read_bandwidth && ./read_bandwidth
// Run it with clocks locked (see edgellm_bench.sh) or the number reflects DVFS.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void read_kernel(const float4* __restrict__ p, size_t n, float* out) {
  float sum = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    float4 v = p[i];
    sum += v.x + v.y + v.z + v.w;
  }
  if (sum == 123.f) *out = sum;  // keeps the loads alive
}

int main() {
  const size_t bytes = 1ull << 30;
  float4* p;
  float* out;
  cudaMalloc(&p, bytes);
  cudaMalloc(&out, 4);
  cudaMemset(p, 1, bytes);
  const size_t n = bytes / 16;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 3; ++i) read_kernel<<<1024, 256>>>(p, n, out);
  cudaEventRecord(a);
  for (int i = 0; i < 20; ++i) read_kernel<<<1024, 256>>>(p, n, out);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  printf("read BW: %.1f GB/s\n", 20.0 * bytes / (ms * 1e-3) / 1e9);
}
