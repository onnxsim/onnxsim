/* hexagon-sim: mb_hvx.h mbv_softmax vs a double reference on one row block (32 rows), scores ~ N(0, sd) */
#include <stdio.h>
#include <stdlib.h>

#include "../mcc_block.h"

static double gauss(void) {
  double u = (rand() + 1.0) / (RAND_MAX + 2.0), v = (rand() + 1.0) / (RAND_MAX + 2.0);
  return sqrt(-2 * log(u)) * cos(6.283185307 * v);
}
static unsigned cfg(int off) { unsigned b; __asm__ volatile("%0 = cfgbase" : "=r"(b)); b <<= 16; return *(volatile unsigned*)(b + off); }
int main(int argc, char** argv) {
  const double sd = argc > 1 ? atof(argv[1]) : 3;
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16);
  uint8_t *s = v, *q = v + 8 * MB_TB, *k = q + MB_TB, *pv = k + MB_TB;
  srand(3);
  double sref[32][MB_SEEN + 1];
  for (int r = 0; r < 32; r++) {
    for (int j = 0; j < MB_ST * 32; j++) {
      mb_hf h = mb_f2h(j < MB_SEEN ? (float)(gauss() * sd) : -65504.f);
      *mb_at(s, 8, r, j) = h;
      if (j < MB_SEEN) sref[r][j] = mb_h2f(h);
    }
    double ss = 0;
    for (int d = 0; d < 32; d++) {
      mb_hf a = mb_f2h((float)gauss() * 1.5f), b = mb_f2h((float)gauss() * 1.5f);
      *mb_at(q, 1, r, d) = a, *mb_at(k, 1, r, d) = b;
      ss += (double)mb_h2f(a) * mb_h2f(b);
    }
    sref[r][MB_SEEN] = ss * MB_SCALE2;
  }
  mbv_softmax(s, q, k, pv, 1, 0, 1);
  double worst = 0, worst_self = 0, worst_sum = 0;
  for (int r = 0; r < 32; r++) {
    double mx = -1e30, sum = 0, got = 0;
    for (int j = 0; j <= MB_SEEN; j++) mx = sref[r][j] > mx ? sref[r][j] : mx;
    for (int j = 0; j <= MB_SEEN; j++) sum += exp2(sref[r][j] - mx);
    for (int j = 0; j < MB_SEEN; j++) {
      double p = exp2(sref[r][j] - mx) / sum, g = mb_h2f(*mb_at(s, 8, r, j));
      got += g;
      double e = fabs(g - p) / (p + 1e-4);
      worst = e > worst ? e : worst;
    }
    double ps = exp2(sref[r][MB_SEEN] - mx) / sum, gs = mb_h2f(*mb_at(pv, 1, r, 0)), gs2 = mb_h2f(*mb_at(pv, 1, r, 31));
    got += gs;
    worst_self = fabs(gs - ps) / (ps + 1e-4) > worst_self ? fabs(gs - ps) / (ps + 1e-4) : worst_self;
    worst_self = fabs(gs2 - ps) / (ps + 1e-4) > worst_self ? fabs(gs2 - ps) / (ps + 1e-4) : worst_self;
    worst_sum = fabs(got - 1) > worst_sum ? fabs(got - 1) : worst_sum;
  }
  printf("sd %.1f: max rel err P %.4g, p_self %.4g, |sum P - 1| %.4g\n", sd, worst, worst_self, worst_sum);
  return 0;
}
