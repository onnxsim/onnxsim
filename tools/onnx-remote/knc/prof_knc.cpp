// Profiling check: detailed remote profile events from the Knights Corner worker, for elementwise ops on 1M floats.
#include "remote_transport.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace onnx_remote;
int main(int argc, char** argv) {
  const char* host = argc > 1 ? argv[1] : "172.31.1.1";
  uint16_t port = argc > 2 ? (uint16_t)std::atoi(argv[2]) : 39501;
  const int64_t n = 1 << 20;
  Tensor x; x.shape = {n}; x.data.assign(n, 0.25f);
  const char* ops[] = {"relu", "add", "tanh", "exp"};
  for (const char* op : ops) {
    for (int rep = 0; rep < 3; rep++) {
      int fd = connect_tcp(host, port);
      if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
      Request req; req.request_id = rep + 1; req.op = op; req.inputs = {x};
      if (std::string(op) == "add") req.inputs.push_back(x);
      req.profiling = ProfilingLevel::Detailed;
      std::string err; Response resp;
      auto t0 = std::chrono::steady_clock::now();
      bool ok = send_request(fd, req, err) && receive_response(fd, resp, err);
      auto t1 = std::chrono::steady_clock::now();
      close_socket(fd);
      if (!ok || !resp.ok) { std::cerr << op << ": " << (err.empty() ? resp.error : err) << "\n"; return 1; }
      double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      for (const auto& ev : resp.profile)
        std::printf("%-5s rep%d  worker: start %8llu us  dur %8llu us  [%s, %s]  | host round trip %.3f ms\n",
                    op, rep, (unsigned long long)ev.start_us, (unsigned long long)ev.duration_us,
                    ev.category.c_str(), ev.detail.c_str(), wall_ms);
    }
  }
  return 0;
}
