/* hexagon-sim: rounding of the V69 qfloat narrowing ops (qf32 pair -> hf, qf16 -> hf) */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdio.h>
#include <string.h>
static unsigned short f2h(float f) { __fp16 x = (__fp16)f; unsigned short h; memcpy(&h, &x, 2); return h; }
#define A128 __attribute__((aligned(128)))
int main(void) {
  float lo[32] A128, hi[32] A128;
  unsigned short back[64] A128, a[64] A128, b[64] A128;
  const char* nm[] = {"just above half-ulp", "exact half-ulp (tie)", "just below half-ulp", "random"};
  for (int kind = 0; kind < 4; kind++) {
    int bad = 0, n = 0;
    for (int rep = 0; rep < 64; rep++) {
      for (int i = 0; i < 32; i++) {
        float base = (1 + (rep * 32 + i) % 1024 / 1024.f) * (1 << (rep % 5));
        float ulp = (1 << (rep % 5)) / 1024.f;
        lo[i] = kind == 0 ? base + ulp * 0.5f + ulp * 0.01f : kind == 1 ? base + ulp * 0.5f : kind == 2 ? base + ulp * 0.49f
                                                                                             : base + ulp * ((rep * 37 + i * 11) % 100) / 100.f;
        hi[i] = -lo[i];
      }
      HVX_Vector ql = Q6_Vqf32_vadd_VsfVsf(*(HVX_Vector*)lo, Q6_V_vzero()), qh = Q6_Vqf32_vadd_VsfVsf(*(HVX_Vector*)hi, Q6_V_vzero());
      *(HVX_Vector*)back = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qh, ql));
      for (int i = 0; i < 64; i++) { n++; bad += back[i] != f2h(i % 2 ? hi[i / 2] : lo[i / 2]); }
    }
    printf("qf32 -> hf, %s: %d of %d differ from RNE\n", nm[kind], bad, n);
  }
  /* hf + hf through qf32 (widen by x1 = exact) vs RNE(a + b) */
  int bad = 0;
  for (int rep = 0; rep < 64; rep++) {
    for (int i = 0; i < 64; i++) { a[i] = f2h((rep * 64 + i) * 0.0137f - 20.f); b[i] = f2h(((rep * 7 + i * 3) % 97) * 0.0311f - 1.5f); }
    HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
    HVX_VectorPair wa = Q6_Wqf32_vmpy_VhfVhf(*(HVX_Vector*)a, one), wb = Q6_Wqf32_vmpy_VhfVhf(*(HVX_Vector*)b, one);
    HVX_VectorPair s = Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(wa), Q6_V_hi_W(wb)), Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(wa), Q6_V_lo_W(wb)));
    *(HVX_Vector*)back = Q6_Vhf_equals_Wqf32(s);
    for (int i = 0; i < 64; i++) {
      __fp16 x, y; memcpy(&x, &a[i], 2); memcpy(&y, &b[i], 2);
      bad += back[i] != f2h((float)x + (float)y);
    }
  }
  printf("hf + hf via qf32: %d of 4096 differ from RNE\n", bad);
  return 0;
}
