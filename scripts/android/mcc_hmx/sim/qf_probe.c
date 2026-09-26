/* hexagon-sim: lane mapping of the V69 qfloat widening/narrowing ops the HVX epilogues rely on */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdio.h>
#include <string.h>
static float h2f(unsigned short h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
static unsigned short f2h(float f) { __fp16 x = (__fp16)f; unsigned short h; memcpy(&h, &x, 2); return h; }
int main(void) {
  unsigned short in[64] __attribute__((aligned(128))), one[64] __attribute__((aligned(128))), back[64] __attribute__((aligned(128)));
  float lo[32] __attribute__((aligned(128))), hi[32] __attribute__((aligned(128)));
  for (int i = 0; i < 64; i++) in[i] = f2h(i + 0.5f), one[i] = f2h(1.f);
  HVX_Vector x = *(HVX_Vector*)in, o = *(HVX_Vector*)one;
  HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(x, o);
  *(HVX_Vector*)lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w));
  *(HVX_Vector*)hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w));
  printf("widen lo:"); for (int i = 0; i < 6; i++) printf(" %g", lo[i]); printf(" ... %g\n", lo[31]);
  printf("widen hi:"); for (int i = 0; i < 6; i++) printf(" %g", hi[i]); printf(" ... %g\n", hi[31]);
  *(HVX_Vector*)back = Q6_Vhf_equals_Wqf32(w);
  int ok = 1; for (int i = 0; i < 64; i++) ok &= back[i] == in[i];
  printf("narrow(widen(x)) == x: %d  (back[0..3] %g %g %g %g)\n", ok, h2f(back[0]), h2f(back[1]), h2f(back[2]), h2f(back[3]));
  /* sf -> qf32 -> narrowing from two sf vectors */
  HVX_Vector ql = Q6_Vqf32_vadd_VsfVsf(*(HVX_Vector*)lo, Q6_V_vzero()), qh = Q6_Vqf32_vadd_VsfVsf(*(HVX_Vector*)hi, Q6_V_vzero());
  *(HVX_Vector*)back = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qh, ql));
  ok = 1; for (int i = 0; i < 64; i++) ok &= back[i] == in[i];
  printf("narrow(combine(hi, lo) from sf) == x: %d\n", ok);
  /* qf16 add/mul round trip */
  *(HVX_Vector*)back = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x, o));
  ok = 1; for (int i = 0; i < 64; i++) ok &= back[i] == in[i];
  printf("qf16 mul by 1 == x: %d:", ok);
  for (int i = 0; i < 64; i += 9) printf(" %g->%g", h2f(in[i]), h2f(back[i]));
  printf("\n");
  for (int i = 0; i < 64; i++) in[i] = f2h(1.f + i / 1024.f);
  x = *(HVX_Vector*)in;
  *(HVX_Vector*)back = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x, o));
  int bad = 0; for (int i = 0; i < 64; i++) bad += back[i] != in[i];
  printf("qf16 mul by 1 on 1+i/1024 (every fp16 ulp): %d of 64 differ\n", bad);
  *(HVX_Vector*)back = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(x, Q6_V_vzero()));
  bad = 0; for (int i = 0; i < 64; i++) bad += back[i] != in[i];
  printf("qf16 add 0: %d of 64 differ\n", bad);
  return 0;
}
