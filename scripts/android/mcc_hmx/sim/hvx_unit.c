/* hexagon-sim: unit checks of mb_hvx.h helpers (rowsum, rowmax, hrecip, hexp2) */
#include <stdio.h>
#include <stdlib.h>

#include "../mcc_block.h"
#define A128 __attribute__((aligned(128)))
int main(void) {
  mb_hf in[64] A128, out[64] A128;
  /* rowsum: even lanes 1..32 step, odd lanes constant */
  double se = 0, so = 0;
  for (int i = 0; i < 64; i++) { in[i] = mb_f2h(i % 2 ? 3.0f : (float)(i / 2) * 7.5f); if (i % 2) so += 3; else se += mb_h2f(in[i]); }
  MBV(out) = mbv_rowsum(MBV(in));
  printf("rowsum even %g (want %g) lane62 %g, odd %g (want %g)\n", mb_h2f(out[0]), se, mb_h2f(out[62]), mb_h2f(out[1]), so);
  for (int i = 0; i < 64; i++) in[i] = mb_f2h(i % 2 ? -(float)i : (float)(i * 3 % 50));
  MBV(out) = mbv_rowmax(MBV(in));
  printf("rowmax even %g odd %g\n", mb_h2f(out[0]), mb_h2f(out[1]));
  double worst = 0;
  for (int i = 0; i < 64; i++) in[i] = mb_f2h(1.f / 128 + i * 3.f);
  MBV(out) = mbv_hrecip(MBV(in), 3);
  for (int i = 0; i < 64; i++) { double w = 1.0 / mb_h2f(in[i]), e = fabs(mb_h2f(out[i]) - w) / w; worst = e > worst ? e : worst; }
  printf("hrecip(1/128..189) 3 NR: max rel err %.3g (d=%g -> %g)\n", worst, mb_h2f(in[5]), mb_h2f(out[5]));
  worst = 0;
  double wi = 0;
  for (int k = 0; k < 64; k++) {
  for (int i = 0; i < 64; i++) in[i] = mb_f2h(-13.f + (k * 64 + i) * 20.f / 4095);
  MBV(out) = mbv_hexp2(MBV(in));
  for (int i = 0; i < 64; i++) { double w = exp2(mb_h2f(in[i])), e = fabs(mb_h2f(out[i]) - w) / w; if (e > worst) worst = e, wi = mb_h2f(in[i]); }
  }
  printf("hexp2(-13..7), 4096 points: max rel err %.3g at t = %g\n", worst, wi);
  return 0;
}
