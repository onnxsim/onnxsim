// Phone-side check of a tinygrad AOT OpenCL bundle: runs it on in.bin, writes out<k>.bin, times it.
//   tg_cl_bench <bundle dir> <in.bin[,in2.bin,...]> <out prefix> [iters] [profile]
// with several inputs, each one's outputs go to <out prefix><k>_<i>.bin (timing uses the first)
#include "tg_cl_runner.h"

#include <algorithm>

int main(int argc, char** argv) {
  if (argc < 4) return fprintf(stderr, "usage: %s dir in.bin outprefix [iters] [profile]\n", argv[0]), 2;
  std::string dir = argv[1];
  int iters = argc > 4 ? atoi(argv[4]) : 20;
  bool prof = argc > 5 && atoi(argv[5]);
  try {
    tgcl::Model m;
    m.load(dir, dir, nullptr, nullptr, nullptr, prof);
    printf("build/load %.1f ms, %zu kernel calls\n", m.build_ms, m.calls.size());
    std::vector<std::string> ins;
    {
      std::string a = argv[2];
      for (size_t p = 0, q; p <= a.size(); p = q + 1) {
        q = a.find(',', p);
        if (q == std::string::npos) q = a.size();
        ins.push_back(tgcl::read_file(a.substr(p, q - p)));
      }
    }
    std::string& in = ins[0];
    if (in.size() != m.nbytes[m.in_ids[0]]) return fprintf(stderr, "input is %zu bytes, model wants %zu\n", in.size(), m.nbytes[m.in_ids[0]]), 1;
    std::vector<std::vector<char>> outs;
    std::vector<void*> optr;
    for (int id : m.out_ids) outs.emplace_back(m.nbytes[id]), optr.push_back(outs.back().data());
    std::vector<double> t;
    for (int i = 0; i < iters + 3; i++) {
      auto t0 = std::chrono::steady_clock::now();
      m.run({in.data()}, optr);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (i >= 3) t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    printf("run: median %.2f ms, min %.2f ms (%d iters, incl. input upload + output readback)\n", t[t.size() / 2], t[0], iters);
    for (size_t k = 0; k < outs.size(); k++) {
      std::string p = std::string(argv[3]) + std::to_string(k) + ".bin";
      std::ofstream(p, std::ios::binary).write(outs[k].data(), outs[k].size());
    }
    for (size_t i = 1; ins.size() > 1 && i <= ins.size(); i++) {
      m.run({ins[i - 1].data()}, optr);
      for (size_t k = 0; k < outs.size(); k++)
        std::ofstream(std::string(argv[3]) + std::to_string(k) + "_" + std::to_string(i - 1) + ".bin", std::ios::binary)
            .write(outs[k].data(), outs[k].size());
    }
    if (prof) {
      std::vector<double> sum(m.calls.size(), 0);
      for (int r = 0; r < 5; r++) {
        auto ms = m.profile();
        for (size_t i = 0; i < ms.size(); i++) sum[i] += ms[i] / 5;
      }
      double tot = 0;
      for (double s : sum) tot += s;
      std::vector<size_t> idx(sum.size());
      for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return sum[a] > sum[b]; });
      printf("GPU kernel time %.2f ms total; top kernels:\n", tot);
      for (size_t j = 0; j < std::min<size_t>(12, idx.size()); j++)
        printf("  %6.3f ms  %-40s g=%zux%zux%zu l=%zux%zux%zu\n", sum[idx[j]], m.calls[idx[j]].name.c_str(), m.calls[idx[j]].g[0],
               m.calls[idx[j]].g[1], m.calls[idx[j]].g[2], m.calls[idx[j]].l[0], m.calls[idx[j]].l[1], m.calls[idx[j]].l[2]);
    }
  } catch (std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
