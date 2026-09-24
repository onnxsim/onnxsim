/* hexagon-sim: one Newton reciprocal step, op by op, in qf16 */
#include <stdio.h>
#include <stdlib.h>
#include "../mcc_block.h"
#define A128 __attribute__((aligned(128)))
int main(void) {
  mb_hf d[64] A128, r[64] A128, a[64] A128, b[64] A128, c[64] A128;
  for (int i = 0; i < 64; i++) d[i] = mb_f2h(1.f / 128 + i * 3.f);
  MBV(r) = Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(0x7800), MBV(d));
  MBV(a) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(MBV(d), MBV(r)));
  MBV(b) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(mbv_hsplat(2.f), MBV(a)));
  MBV(c) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(MBV(b), MBV(r)));
  for (int i = 0; i < 64; i += 7)
    printf("d %9.4f r0 %.6g (1/d %.6g) d*r %.6g [%.6g] 2-dr %.6g [%.6g] r1 %.6g [%.6g]\n", mb_h2f(d[i]), mb_h2f(r[i]), 1 / mb_h2f(d[i]),
           mb_h2f(a[i]), mb_h2f(d[i]) * mb_h2f(r[i]), mb_h2f(b[i]), 2 - mb_h2f(a[i]), mb_h2f(c[i]), mb_h2f(b[i]) * mb_h2f(r[i]));
  return 0;
}
