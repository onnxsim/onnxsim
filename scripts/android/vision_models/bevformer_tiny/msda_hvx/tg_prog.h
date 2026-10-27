// One tinygrad-compiled program (a .tghx artifact from the fork's examples/dsp_graph_onnx.py --artifact, the same container
// onnx-remote's hexagon worker loads: tools/onnx-remote/remote_hexagon_worker.cpp) in this process: its skel written next to
// the binary and opened over FastRPC, its weights uploaded, and its packed input / output buffers in rpcmem, so a run passes
// them without a copy. frame_run.cpp's TG_ENC / TG_DEC.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "remote.h"
#include "rpcmem.h"
#include "tg_graph.h"
}

struct TgProg {
  struct In { int onnx = 0; uint32_t dtype = 0; uint64_t bytes = 0; int slot = -1; size_t off = 0; };
  struct Out { uint32_t dtype = 1; uint64_t elements = 0; size_t off = 0; };
  int ncalls = 0, threads = 1;
  uint64_t output_bytes = 0, output_align = 1;
  std::vector<In> in;  // ONNX graph order
  std::vector<Out> out;
  remote_handle64 h = 0;
  uint8_t *ibuf = nullptr, *obuf = nullptr;
  size_t ibytes = 0;
  double last_dsp_ms = 0;

  // the packed input slot of ONNX input i (nullptr: the program doesn't read it)
  void* input(int i) {
    for (auto& s : in) if (s.onnx == i) return s.slot < 0 ? nullptr : ibuf + s.off;
    throw std::runtime_error("no program input " + std::to_string(i));
  }
  const void* output(int k) const { return obuf + out.at(k).off; }

  void load(const std::string& tghx, const std::string& so_dir, const std::string& so_name) {
    std::ifstream f(tghx, std::ios::binary);
    std::vector<uint8_t> a((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (a.size() < 12 || memcmp(a.data(), "TGHXV", 5)) throw std::runtime_error("not a tghx artifact: " + tghx);
    std::map<std::string, std::pair<size_t, size_t>> files;
    size_t at = 8;
    auto take = [&](auto& v) {
      if (at + sizeof v > a.size()) throw std::runtime_error("truncated " + tghx);
      memcpy(&v, a.data() + at, sizeof v);
      at += sizeof v;
    };
    uint32_t count = 0;
    take(count);
    for (uint32_t k = 0; k < count; ++k) {
      uint32_t nl = 0;
      uint64_t n = 0;
      take(nl);
      std::string name((const char*)a.data() + at, nl);
      at += nl;
      take(n);
      files[name] = {at, (size_t)n};
      at += n;
    }
    for (auto need : {"tg_graph.so", "blob.bin", "program.txt"})
      if (!files.count(need)) throw std::runtime_error(tghx + " lacks " + need);
    parse(std::string((const char*)a.data() + files["program.txt"].first, files["program.txt"].second));
    {
      std::ofstream so(so_dir + "/" + so_name, std::ios::binary | std::ios::trunc);
      so.write((const char*)a.data() + files["tg_graph.so"].first, (std::streamsize)files["tg_graph.so"].second);
      if (!so) throw std::runtime_error("cannot write " + so_name);
    }
    const std::string uri = "file:///" + so_name + "?tg_graph_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    if (int rc = tg_graph_open(uri.c_str(), &h)) throw std::runtime_error("tg_graph_open " + so_name + ": " + std::to_string(rc));
    const size_t boff = files["blob.bin"].first, bn = files["blob.bin"].second, chunk = 8u << 20;
    for (size_t o = 0; o < bn || o == 0; o += chunk) {
      if (int rc = tg_graph_load(h, (int)o, (int)bn, a.data() + boff + o, (int)std::min(chunk, bn - o)))
        throw std::runtime_error("tg_graph_load " + so_name + ": " + std::to_string(rc));
      if (bn == 0) break;
    }
    // the input buffer: the used inputs in slot order, each 128-byte aligned (the worker's packing)
    std::vector<In*> by_slot;
    for (auto& s : in) if (s.slot >= 0) by_slot.push_back(&s);
    std::sort(by_slot.begin(), by_slot.end(), [](const In* x, const In* y) { return x->slot < y->slot; });
    for (In* s : by_slot) { s->off = ibytes; ibytes += (s->bytes + 127) / 128 * 128; }
    ibuf = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, std::max<size_t>(ibytes, 128));
    obuf = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, std::max<size_t>(output_bytes, 128));
    if (!ibuf || !obuf) throw std::runtime_error("rpcmem_alloc for " + so_name);
    memset(ibuf, 0, ibytes);
  }

  void run() {
    uint64 t = 0;
    if (int rc = tg_graph_run(h, 0, ncalls, threads, 0, ibuf, (int)ibytes, obuf, (int)output_bytes, &t, 1))
      throw std::runtime_error("tg_graph_run: " + std::to_string(rc));
    last_dsp_ms = t / 1000.0;
  }

  void close() { if (h) tg_graph_close(h); h = 0; }

 private:
  void parse(const std::string& text) {
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
      std::istringstream f(line);
      std::string key;
      f >> key;
      if (key == "ncalls") f >> ncalls;
      else if (key == "threads") f >> threads;
      else if (key == "output_bytes") f >> output_bytes;
      else if (key == "output_align") f >> output_align;
      else if (key == "input") { In s; f >> s.onnx >> s.dtype >> s.bytes >> s.slot; in.push_back(s); }
      else if (key == "output") { Out o; f >> o.dtype >> o.elements; out.push_back(o); }
      else if (key == "state") throw std::runtime_error("programs with recurrent state aren't supported here");
    }
    size_t o = 0;
    for (auto& x : out) {
      x.off = o;
      o = (o + x.elements * (x.dtype == 1 ? 4 : 1) + output_align - 1) / output_align * output_align;
    }
    if (ncalls <= 0 || out.empty() || o != output_bytes) throw std::runtime_error("inconsistent program.txt");
  }
};
