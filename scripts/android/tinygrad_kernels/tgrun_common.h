// Shared by the Dawn (WGSL) and raw-Vulkan (SPIR-V) runners: manifest parsing, file IO, output check.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Buf { int slot; size_t elems; bool out; };
struct Variant { std::string name, entry; uint32_t g[3], l[3]; std::string opts; };
struct Manifest { std::string problem; double flops = 0; std::vector<Buf> bufs; std::vector<Variant> variants; };

inline Manifest read_manifest(const std::string& dir) {
  Manifest m;
  std::ifstream f(dir + "/manifest.txt");
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string kw;
    ss >> kw;
    if (kw == "problem") { std::string fl; ss >> m.problem >> fl >> m.flops; }
    else if (kw == "buf") { Buf b; std::string role; ss >> b.slot >> b.elems >> role; b.out = role == "out"; m.bufs.push_back(b); }
    else if (kw == "variant") {
      Variant v; std::string bar;
      ss >> v.name >> v.entry >> v.g[0] >> v.g[1] >> v.g[2] >> v.l[0] >> v.l[1] >> v.l[2] >> bar;
      std::getline(ss, v.opts);
      m.variants.push_back(v);
    }
  }
  return m;
}
inline std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
inline std::string read_text(const std::string& p) { auto v = read_file(p); return std::string(v.begin(), v.end()); }
// max |got-ref| / max|ref|
inline double rel_err(const float* got, const std::vector<char>& ref, size_t n) {
  const float* r = reinterpret_cast<const float*>(ref.data());
  double mx = 1e-12, e = 0;
  for (size_t i = 0; i < n; i++) { mx = std::max(mx, (double)std::fabs(r[i])); e = std::max(e, (double)std::fabs(got[i] - r[i])); if (std::isnan(got[i])) return 1e9; }
  return e / mx;
}
inline bool wanted(const Variant& v, const std::vector<std::string>& sel) {
  if (sel.empty()) return true;
  return std::find(sel.begin(), sel.end(), v.name) != sel.end();
}
