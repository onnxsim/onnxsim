/* hexagon-sim: the whole graph runner (runner/rn_load.h + rn_exec.h) on a qdq_graph.py program, vs ORT CPU's output,
 * both requantization modes. runner_sim <program dir> <input.bin> <ref.bin> */
#include <stdio.h>
#include "../runner/rn_load.h"
#include "../runner/rn_exec.h"
static uint8_t* rd(const char* p, size_t n) {
  FILE* f = fopen(p, "rb");
  uint8_t* b = malloc(n);
  if (!f || fread(b, 1, n, f) != n) { printf("cannot read %s\n", p); exit(2); }
  fclose(f);
  return b;
}
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
int main(int argc, char** argv) {
  static rn_build_t B;
  if (rn_build(&B, argv[1])) { printf("build failed: %s\n", B.err); return 2; }
  rn_model_t* m = &B.m;
  uint8_t* v = (uint8_t*)(cfg(0x38) << 16); unsigned vs = cfg(0x3c) * 1024;
  unsigned r; __asm__ volatile("%0 = ssr" : "=r"(r)); r |= 1u << 26; __asm__ volatile("ssr = %0; isync" ::"r"(r));
  printf("%d tensors %d ops, VTCM plan %u of %u bytes, blob %u bytes\n", m->nt, m->nops, m->vtcm_bytes, vs, m->blob_bytes);
  if (m->vtcm_bytes > vs) return 2;
  const rn_tensor_t* ti = &m->t[m->input], *to = &m->t[m->output];
  size_t xl = (size_t)ti->h * ti->w * ti->c, yl = (size_t)to->c * to->h * to->w, fl;
  uint8_t *x = rd(argv[2], xl), *ref = rd(argv[3], yl), *y = malloc(yl);
  uint8_t* xf = rn_pack_input(m, x, &fl);
  static rn_ctx_t c;
  c.m = m, c.blob = B.blob, c.vtcm = v;
  int pr = rn_plan(&c);
  if (pr) { printf("plan failed %d\n", pr); return 2; }
  if (argc > 4) {
    const rn_tensor_t* I = &m->t[m->input];
    if (!I->in_ddr) memcpy(v + I->off, xf, qc_geom_bytes(&I->g, I->cp / 32));
  }
  if (argc > 4) { /* debug: run op by op (exact mode), dump each output right after its op: <dir>/t<i>.bin (NCHW) */
    for (int i = 0; i < m->nops; i++) {
      const rn_op_t* o = &m->op[i];
      if (o->type == RN_CONV) rn_conv(&c, i, xf, 1);
      else if (o->type == RN_ADD) rn_add(&c, i, xf);
      else rn_maxpool(&c, i, xf);
      const rn_tensor_t* t = &m->t[o->y];
      rn_model_t mm = *m;
      mm.output = o->y;
      uint8_t* d = malloc((size_t)t->c * t->h * t->w);
      rn_unpack_output(&mm, v + t->off, d);
      char p[256];
      snprintf(p, sizeof p, "%s/t%d.bin", argv[4], o->y);
      FILE* f = fopen(p, "wb");
      fwrite(d, 1, (size_t)t->c * t->h * t->w, f);
      fclose(f);
      free(d);
    }
  }
  for (int mode = 0; mode < 2; mode++) {
    int nfix = rn_run(&c, xf, mode);
    rn_unpack_output(m, v + to->off, y);
    int bad = 0;
    for (size_t i = 0; i < yl; i++) bad += y[i] != ref[i];
    printf("%s: %d mismatches of %zu vs ORT (scalar fixes %d)%s\n", mode ? "exact" : "fast ", bad, yl, nfix, mode ? (bad ? " FAIL" : " PASS") : "");
  }
  return 0;
}
