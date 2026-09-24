/* replay_client <uri> <bundle dir> [iters] [ref float32 file]: load a captured tinygrad kernel sequence's regions,
 * run it on the DSP, print the time per iteration and per kernel (summed over its calls), compare the output
 * (fp16) with a float32 reference if given. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "remote.h"
#include "replay_rpc.h"

static void* slurp(const char* dir, const char* name, size_t* n) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* b = malloc(*n + 1);
  if (fread(b, 1, *n, f) != *n) exit(1);
  ((char*)b)[*n] = 0;
  fclose(f);
  return b;
}
static float h2f(unsigned short h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const char* dir = argv[2];
  int iters = argc > 3 ? atoi(argv[3]) : 5;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  if (replay_rpc_open(argv[1], &h)) { printf("open failed\n"); return 1; }
  int prc = 0;
  replay_rpc_perf_vote(h, 3, &prc);
  size_t n;
  char* regions = slurp(dir, "regions.txt", &n);
  int nr = 0;
  for (char* l = strtok(regions, "\n"); l; l = strtok(0, "\n")) {
    int idx, sz;
    sscanf(l, "%d %d", &idx, &sz);
    char name[32];
    snprintf(name, sizeof name, "r%d.bin", idx);
    size_t m;
    void* img = slurp(dir, name, &m);
    int rc = replay_rpc_load_region(h, idx, img, (int)m);
    free(img);
    if (rc) { printf("load_region %d rc %d\n", idx, rc); return 1; }
    nr++;
  }
  char* calls = slurp(dir, "calls.txt", &n);
  int kcall[4096], nc = 0;
  for (char* l = strtok(calls, "\n"); l; l = strtok(0, "\n")) kcall[nc++] = atoi(l);
  char* outs = slurp(dir, "out_bytes.txt", &n);
  int outn = atoi(outs);
  unsigned char* out = malloc(outn);
  unsigned long long* pc = calloc(nc, 8);
  unsigned long long t[1];
  int codes[4];
  int rc = replay_rpc_run(h, iters, out, outn, pc, nc, t, 1, codes, 4);
  printf("vote %d rc %d ctx %d hvx %d hmx %d thread %d | %d regions, %d calls: %.3f ms per iteration\n", prc, rc, codes[0], codes[1], codes[2],
         codes[3], nr, nc, t[0] / 1e3);
  /* per kernel: sum over its calls, kernel name from k<i>.c's first line */
  unsigned long long per[256] = {0};
  int cnt[256] = {0}, nk = 0;
  for (int i = 0; i < nc; i++) per[kcall[i]] += pc[i], cnt[kcall[i]]++, nk = kcall[i] + 1 > nk ? kcall[i] + 1 : nk;
  unsigned long long tot = 0;
  for (int k = 0; k < nk; k++) tot += per[k];
  for (int k = 0; k < nk; k++) {
    char name[32];
    snprintf(name, sizeof name, "k%d.c", k);
    size_t m;
    char* src = slurp(dir, name, &m);
    char* e = strchr(src, '\n');
    if (e) *e = 0;
    printf("  k%-3d x%-3d %8.3f ms  %5.1f%%  %s\n", k, cnt[k], per[k] / 1.5e6 / iters, 100.0 * per[k] / (tot ? tot : 1), src);
    free(src);
  }
  if (argc > 4) {
    float* ref = slurp(".", argv[4], &n);
    const int q = outn / 2;
    double e = 0, mr = 0, dot = 0, na = 0, nb = 0;
    for (int i = 0; i < q; i++) {
      double g = h2f(((unsigned short*)out)[i]), f = ref[i];
      e = fabs(g - f) > e ? fabs(g - f) : e;
      mr = fabs(f) > mr ? fabs(f) : mr;
      dot += g * f, na += g * g, nb += f * f;
    }
    printf("vs %s: max abs err %.4g (max |ref| %.4g), cos %.7f\n", argv[4], e, mr, dot / sqrt(na * nb));
  }
  replay_rpc_close(h);
  return 0;
}
