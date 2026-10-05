#include "remote_transport.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

using namespace onnx_remote;

static bool parse_floats(const std::string& text, std::vector<float>& values,
                         std::string& error) {
  values.clear();
  std::string item;
  std::istringstream stream(text);
  while (std::getline(stream, item, ',')) {
    char* end = nullptr;
    const float value = std::strtof(item.c_str(), &end);
    if (end == item.c_str() || *end != '\0' || !std::isfinite(value)) {
      error = "invalid float value: " + item;
      return false;
    }
    values.push_back(value);
  }
  if (values.empty()) error = "input values must not be empty";
  return !values.empty();
}

static int subgraph_request(int argc, char** argv) {
  // onnx-remote-client --subgraph HOST PORT MODEL.onnx V0,V1,... [--shape D0,D1]
  // Repeat [--input VALUES [--shape DIMS]] for additional model inputs.
  std::string shape_text;
  int positional = 0;
  std::string host, port_text, model_path, values_text;
  std::vector<std::pair<std::string, std::string>> extra_inputs;
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--shape" && i + 1 < argc) {
      shape_text = argv[++i];
    } else if (argument == "--input" && i + 1 < argc) {
      std::string spec = argv[++i];
      std::string input_shape;
      const size_t at = spec.find('@');
      if (at != std::string::npos) {
        input_shape = spec.substr(at + 1);
        spec.resize(at);
      }
      extra_inputs.emplace_back(spec, input_shape);
    } else if (positional == 0) {
      host = argument;
      ++positional;
    } else if (positional == 1) {
      port_text = argument;
      ++positional;
    } else if (positional == 2) {
      model_path = argument;
      ++positional;
    } else if (positional == 3) {
      values_text = argument;
      ++positional;
    } else {
      std::cerr << "unexpected argument: " << argument << '\n';
      return 2;
    }
  }
  if (positional != 4) {
    std::cerr << "usage: onnx-remote-client --subgraph HOST PORT MODEL.onnx "
                 "V0,V1,... [--shape D0,D1] [--input VALUES[@D0,D1] ...]\n";
    return 2;
  }
  std::ifstream model_file(model_path, std::ios::binary);
  if (!model_file) {
    std::cerr << "cannot open model\n";
    return 1;
  }
  Request request;
  request.op = kSubgraphOperation;
  request.profiling = ProfilingLevel::Detailed;
  request.model.assign(std::istreambuf_iterator<char>(model_file), {});
  auto append_input = [&](const std::string& values_text,
                          const std::string& shape_text) {
    std::string error;
    std::vector<float> values;
    if (!parse_floats(values_text, values, error)) {
      std::cerr << error << '\n';
      return false;
    }
    Tensor input;
    if (shape_text.empty()) {
      input.shape = {static_cast<int64_t>(values.size())};
    } else {
      std::string item;
      std::istringstream stream(shape_text);
      while (std::getline(stream, item, ',')) {
        char* end = nullptr;
        const long dimension = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || dimension <= 0) {
          std::cerr << "invalid shape dimension: " << item << '\n';
          return false;
        }
        input.shape.push_back(dimension);
      }
      size_t elements = 1;
      for (int64_t dimension : input.shape) elements *= static_cast<size_t>(dimension);
      if (elements != values.size()) {
        std::cerr << "values count does not match shape\n";
        return false;
      }
    }
    input.data = values;
    request.inputs.push_back(std::move(input));
    return true;
  };
  if (!append_input(values_text, shape_text)) return 2;
  for (const auto& extra : extra_inputs) {
    if (!append_input(extra.first, extra.second)) return 2;
  }
  const int fd = connect_tcp(host,
                             static_cast<uint16_t>(std::strtoul(port_text.c_str(), nullptr, 10)));
  if (fd < 0) {
    std::cerr << "connect failed\n";
    return 1;
  }
  std::string error;
  Response response;
  const bool ok =
      send_request(fd, request, error) && receive_response(fd, response, error);
  close_socket(fd);
  if (!ok || !response.ok) {
    std::cerr << (error.empty() ? response.error : error) << '\n';
    return 1;
  }
  for (size_t i = 0; i < response.outputs.size(); ++i) {
    const Tensor& output = response.outputs[i];
    std::cout << "output[" << i << "] dtype=" << static_cast<int>(output.dtype)
              << " shape=[";
    for (size_t j = 0; j < output.shape.size(); ++j) {
      if (j != 0) std::cout << ',';
      std::cout << output.shape[j];
    }
    std::cout << "]\n";
    for (float value : output.data) std::cout << value << '\n';
  }
  for (const ProfileEvent& event : response.profile) {
    std::cout << "profile: " << event.name << " " << event.duration_us
              << "us " << event.detail << '\n';
  }
  return 0;
}

static int self_test() {
  Request payload_request;
  payload_request.request_id = 7;
  payload_request.op = "relu";
  payload_request.profiling = ProfilingLevel::Detailed;
  payload_request.inputs.push_back(Tensor{{5}, {-2, -1, 0, 1, 2}});
  std::vector<uint8_t> payload;
  std::string error;
  if (!encode_request_payload(payload_request, payload, error)) {
    std::cerr << "request payload encode failed: " << error << '\n';
    return 1;
  }
  Request decoded_request;
  if (!decode_request_payload(payload.data(), payload.size(), decoded_request, error) ||
      decoded_request.op != payload_request.op ||
      decoded_request.profiling != payload_request.profiling) {
    std::cerr << "request payload round-trip failed\n";
    return 1;
  }
  struct Operation {
    const char* name;
    std::vector<float> lhs;
    std::vector<float> rhs;
    std::vector<float> expected;
  };
  const std::vector<Operation> operations = {
      {"relu", {-2, -1, 0, 1, 2}, {}, {0, 0, 0, 1, 2}},
      {"add", {1, 2, 3}, {4, 5, 6}, {5, 7, 9}},
      {"mul", {1, 2, 3}, {4, 5, 6}, {4, 10, 18}},
      {"sub", {1, 2, 3}, {4, 5, 6}, {-3, -3, -3}},
      {"div", {4, 9, 12}, {2, 3, 4}, {2, 3, 3}},
      {"max", {1, 5, 3}, {4, 2, 6}, {4, 5, 6}},
      {"min", {1, 5, 3}, {4, 2, 6}, {1, 2, 3}},
      {"abs", {-2, 0, 3}, {}, {2, 0, 3}},
      {"neg", {-2, 0, 3}, {}, {2, 0, -3}},
      {"sqrt", {0, 1, 4}, {}, {0, 1, 2}},
      {"exp", {0, 1, 2}, {}, {1, std::exp(1.0f), std::exp(2.0f)}},
      {"log", {1, std::exp(1.0f), std::exp(2.0f)}, {}, {0, 1, 2}},
      {"tanh", {-1, 0, 1}, {}, {-0.76159414f, 0, 0.76159414f}},
  };
  size_t profile_events = 0;
  for (const Operation& operation : operations) {
    int fd = connect_tcp("127.0.0.1", 39501);
    if (fd < 0) {
      std::cerr << "connect failed (start onnx-remote-worker --port 39501)\n";
      return 1;
    }
    Request request;
    request.request_id = 7 + profile_events;
    request.op = operation.name;
    request.profiling = ProfilingLevel::Detailed;
    request.inputs.push_back(Tensor{{static_cast<int64_t>(operation.lhs.size())},
                                    operation.lhs});
    if (!operation.rhs.empty())
      request.inputs.push_back(Tensor{{static_cast<int64_t>(operation.rhs.size())},
                                      operation.rhs});
    if (!send_request(fd, request, error)) {
      std::cerr << error << '\n'; close_socket(fd); return 1;
    }
    Response response;
    if (!receive_response(fd, response, error) || !response.ok ||
        response.outputs.size() != 1) {
      std::cerr << (error.empty() ? response.error : error) << '\n';
      close_socket(fd); return 1;
    }
    if (response.request_id != request.request_id ||
        response.outputs[0].data.size() != operation.expected.size()) {
      std::cerr << "response mismatch for " << operation.name << '\n';
      close_socket(fd); return 1;
    }
    for (size_t i = 0; i < operation.expected.size(); ++i) {
      if (std::fabs(response.outputs[0].data[i] - operation.expected[i]) >
          1e-6f) {
        std::cerr << "value mismatch for " << operation.name << '\n';
        close_socket(fd); return 1;
      }
    }
    profile_events += response.profile.size();
    close_socket(fd);
  }
  {
    int fd = connect_tcp("127.0.0.1", 39501);
    if (fd < 0) return 1;
    Request request;
    request.request_id = 99;
    request.op = "add";
    request.profiling = ProfilingLevel::Detailed;
    request.inputs.push_back(Tensor{{2, 3}, {1, 2, 3, 1, 2, 3}});
    request.inputs.push_back(Tensor{{3}, {4, 5, 6}});
    if (!send_request(fd, request, error)) {
      close_socket(fd);
      return 1;
    }
    Response response;
    if (!receive_response(fd, response, error) || !response.ok ||
        response.outputs.size() != 1 || response.outputs[0].shape !=
            std::vector<int64_t>({2, 3}) ||
        response.outputs[0].data !=
            std::vector<float>({5, 7, 9, 5, 7, 9})) {
      std::cerr << "broadcast add self-test failed\n";
      close_socket(fd);
      return 1;
    }
    profile_events += response.profile.size();
    close_socket(fd);
  }
  std::cout << "remote transport self-test passed (profile events: "
            << profile_events << ")\n";
  return 0;
}

static bool exchange(const std::string& host, uint16_t port, const Request& request, Response& response) {
  int fd = connect_tcp(host, port);
  if (fd < 0) { std::cerr << "connect to " << host << ':' << port << " failed\n"; return false; }
  std::string error;
  const bool ok = send_request(fd, request, error) && receive_response(fd, response, error);
  close_socket(fd);
  if (!ok || !response.ok) { std::cerr << request.op << ": " << (error.empty() ? response.error : error) << '\n'; return false; }
  return true;
}

static int compile_run(int argc, char** argv) {
  // onnx-remote-client --compile-run COMPILER_HOST PORT RUNNER_HOST PORT MODEL.onnx
  //     [--input-raw DTYPE:D0,D1,...:FILE]... [--expect FILE] [--dump FILE] [--iters N] [--profile] [--resident IN:OUT,...]
  // The compiler/runner split end to end: COMPILE the model, load_compiled the artifact on the runner once, then
  // run_compiled without artifact bytes. Inputs are raw little-endian files in ONNX graph order; --expect compares the
  // outputs' bytes, concatenated, byte for byte (of the first run). --resident names recurrent-state inputs and the outputs that
  // are their next values (e.g. 4:1 for state_img_q <- next_state_img_q): after the first run they are sent empty so the runner
  // uses its resident copy, and a check compares that against sending the state back explicitly.
  if (argc < 7) { std::cerr << "usage: onnx-remote-client --compile-run CHOST CPORT RHOST RPORT MODEL.onnx [...]\n"; return 2; }
  const std::string chost = argv[2], rhost = argv[4], model_path = argv[6];
  const auto cport = static_cast<uint16_t>(std::strtoul(argv[3], nullptr, 10));
  const auto rport = static_cast<uint16_t>(std::strtoul(argv[5], nullptr, 10));
  std::vector<Tensor> inputs;
  std::string expect_path, dump_path;
  int iters = 3;
  bool profiling = false;
  bool native_out = false;  // --native-out: ask the runner for run_compiled_native (outputs in the network's own dtype, quantization in the manifest)
  std::vector<std::pair<size_t, size_t>> resident;  // (input index, output index)
  for (int i = 7; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--input-raw" && i + 1 < argc) {
      std::istringstream spec(argv[++i]);
      std::string dtype, dims, file;
      std::getline(spec, dtype, ':'); std::getline(spec, dims, ':'); std::getline(spec, file);
      Tensor t;
      t.dtype = static_cast<uint8_t>(std::strtoul(dtype.c_str(), nullptr, 10));
      std::istringstream ds(dims);
      for (std::string d; std::getline(ds, d, ',');) t.shape.push_back(std::strtoll(d.c_str(), nullptr, 10));
      std::ifstream f(file, std::ios::binary);
      std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
      if (!f.good() && !f.eof()) { std::cerr << "cannot read " << file << '\n'; return 1; }
      if (t.dtype == 1) { t.data.resize(bytes.size() / 4); std::memcpy(t.data.data(), bytes.data(), t.data.size() * 4); }
      else t.raw_data = std::move(bytes);
      inputs.push_back(std::move(t));
    } else if (a == "--expect" && i + 1 < argc) expect_path = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--profile") profiling = true;
    else if (a == "--native-out") native_out = true;
    else if (a == "--resident" && i + 1 < argc) {
      std::istringstream rs(argv[++i]);
      for (std::string pair; std::getline(rs, pair, ',');) {
        const auto colon = pair.find(':');
        if (colon == std::string::npos) { std::cerr << "--resident takes IN:OUT pairs\n"; return 2; }
        resident.emplace_back(std::strtoul(pair.substr(0, colon).c_str(), nullptr, 10), std::strtoul(pair.substr(colon + 1).c_str(), nullptr, 10));
      }
    }
    else { std::cerr << "unknown argument: " << a << '\n'; return 2; }
  }
  Request compile;
  compile.op = "compile";
  { std::ifstream f(model_path, std::ios::binary); compile.model.assign(std::istreambuf_iterator<char>(f), {}); }
  // the model's own static shapes define the program; no inputs, so this hits the same cache entry as --compile
  auto t0 = std::chrono::steady_clock::now();
  auto ms = [&](auto a) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); };
  Response compiled;
  if (!exchange(chost, cport, compile, compiled)) return 1;
  std::cout << "compiled: artifact " << compiled.artifact_id << ", " << compiled.artifact.size() << " bytes, "
            << ms(t0) << " ms\n" << compiled.manifest << '\n';
  Request load;
  load.op = "load_compiled";
  load.artifact_id = compiled.artifact_id;
  load.artifact = std::move(compiled.artifact);
  t0 = std::chrono::steady_clock::now();
  Response loaded;
  if (!exchange(rhost, rport, load, loaded)) return 1;
  std::cout << "loaded on the runner in " << ms(t0) << " ms\n";
  Request run;
  run.op = native_out ? "run_compiled_native" : "run_compiled";
  run.artifact_id = load.artifact_id;
  run.inputs = std::move(inputs);
  run.profiling = profiling ? ProfilingLevel::Detailed : ProfilingLevel::Summary;
  for (const auto& [in, out] : resident)
    if (in >= run.inputs.size()) { std::cerr << "--resident input " << in << " out of range\n"; return 2; }
  const std::vector<Tensor> full_inputs = run.inputs;
  auto state_empty = [&](Request& r) { for (const auto& [in, out] : resident) r.inputs[in] = Tensor{{0}, {}, r.inputs[in].dtype, {}}; };
  Response result, first;
  double best = 1e30, first_ms = 0;
  for (int it = 0; it < std::max(iters, 1); ++it) {
    t0 = std::chrono::steady_clock::now();
    if (!exchange(rhost, rport, run, result)) return 1;
    // with --resident the first run sends the state (its time is reported separately), the rest leave it on the runner
    if (it == 0) { first = result; first_ms = ms(t0); if (!resident.empty()) { state_empty(run); if (iters > 1) continue; } }
    best = std::min(best, ms(t0));
  }
  std::cout << "run_compiled: " << result.outputs.size() << " outputs, best " << best << " ms of " << std::max(iters, 1)
            << " (RPC-inclusive)";
  if (!resident.empty()) std::cout << ", state resident on the runner after the first run (" << first_ms << " ms)";
  std::cout << '\n';
  if (native_out) {
    size_t payload = 0;
    for (const Tensor& t : result.outputs) payload += t.dtype == 1 ? t.data.size() * 4 : t.raw_data.size();
    std::cout << "native outputs: " << payload << " bytes; manifest " << result.manifest << '\n';
  }
  auto bytes_of = [](const Tensor& t) {
    if (t.dtype != 1) return t.raw_data;
    const auto* b = reinterpret_cast<const uint8_t*>(t.data.data());
    return std::vector<uint8_t>(b, b + t.data.size() * 4);
  };
  if (!resident.empty()) {
    // the second step two ways: the first run's state outputs sent back explicitly, vs (after re-sending the original state,
    // which resets the runner's copy to the same values) sent empty
    Request explicit_run = run, reset = run, resident_run = run;
    explicit_run.profiling = reset.profiling = resident_run.profiling = ProfilingLevel::Off;
    explicit_run.inputs = full_inputs;
    for (const auto& [in, out] : resident) {
      if (out >= first.outputs.size()) { std::cerr << "--resident output " << out << " out of range\n"; return 2; }
      explicit_run.inputs[in] = first.outputs[out];
      explicit_run.inputs[in].shape = full_inputs[in].shape;
    }
    reset.inputs = full_inputs;
    Response a, b, c;
    if (!exchange(rhost, rport, explicit_run, a) || !exchange(rhost, rport, reset, b) || !exchange(rhost, rport, resident_run, c)) return 1;
    bool same = a.outputs.size() == c.outputs.size();
    for (size_t k = 0; same && k < a.outputs.size(); ++k) {
      const bool is_state = std::any_of(resident.begin(), resident.end(), [&](const auto& r) { return r.second == k; });
      if (!is_state) same = bytes_of(a.outputs[k]) == bytes_of(c.outputs[k]);
      else same = c.outputs[k].data.empty() && c.outputs[k].raw_data.empty();
    }
    std::cout << "resident state: " << (same ? "bit-exact to sending it back" : "DIFFERS from sending it back") << '\n';
    if (!same) return 1;
  }
  std::vector<const ProfileEvent*> calls;
  for (const auto& e : result.profile) {
    if (e.category == "hexagon_call") calls.push_back(&e);
    else std::cout << "  " << e.name << ' ' << e.duration_us << " us " << e.detail << '\n';
  }
  if (profiling && !calls.empty()) {
    // per-kernel device time of the last run, slowest first (PROF_TOP, default 15)
    uint64_t total = 0;
    for (const auto* e : calls) total += e->duration_us;
    std::sort(calls.begin(), calls.end(), [](const ProfileEvent* a, const ProfileEvent* b) { return a->duration_us > b->duration_us; });
    const char* top_env = std::getenv("PROF_TOP");
    const size_t top = std::min(calls.size(), static_cast<size_t>(top_env ? std::atoi(top_env) : 15));
    std::cout << "  " << calls.size() << " profiled calls, " << total << " us:\n";
    for (size_t i = 0; i < top; ++i)
      std::cout << "    " << calls[i]->name << ' ' << calls[i]->duration_us << " us " << (100.0 * calls[i]->duration_us / total)
                << "% " << calls[i]->detail << '\n';
  }
  if (!dump_path.empty()) {
    std::ofstream f(dump_path, std::ios::binary | std::ios::trunc);
    for (const auto& t : first.outputs) {
      const auto b = bytes_of(t);
      f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    }
  }
  if (!expect_path.empty()) {
    std::ifstream f(expect_path, std::ios::binary);
    std::vector<uint8_t> expect((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint8_t> got;
    for (const auto& t : first.outputs) {
      const auto b = bytes_of(t);
      got.insert(got.end(), b.begin(), b.end());
    }
    size_t bad = got.size() == expect.size() ? 0 : std::max(got.size(), expect.size());
    for (size_t i = 0; i < std::min(got.size(), expect.size()) && bad == 0; ++i) bad += got[i] != expect[i];
    std::cout << (bad ? "outputs DIFFER from " : "outputs bit-exact to ") << expect_path << '\n';
    if (bad) return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
  if (argc >= 2 && std::string(argv[1]) == "--subgraph")
    return subgraph_request(argc, argv);
  if (argc >= 2 && std::string(argv[1]) == "--compile-run") return compile_run(argc, argv);
  if (argc == 4 && std::string(argv[1]) == "--capabilities") {
    int fd = connect_tcp(argv[2], static_cast<uint16_t>(std::strtoul(argv[3], nullptr, 10)));
    if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
    Request request;
    request.op = "capabilities";
    std::string error;
    Response response;
    const bool ok = send_request(fd, request, error) &&
                    receive_response(fd, response, error);
    close_socket(fd);
    if (!ok || !response.ok) {
      std::cerr << (error.empty() ? response.error : error) << '\n';
      return 1;
    }
    std::cout << response.manifest << '\n';
    return 0;
  }
  if (argc == 5 && std::string(argv[1]) == "--compile") {
    std::ifstream input(argv[4], std::ios::binary);
    if (!input) { std::cerr << "cannot open model\n"; return 1; }
    Request request;
    request.op = "compile";
    request.model.assign(std::istreambuf_iterator<char>(input), {});
    int fd = connect_tcp(argv[2], static_cast<uint16_t>(std::strtoul(argv[3], nullptr, 10)));
    if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
    std::string error;
    Response response;
    bool ok = send_request(fd, request, error) && receive_response(fd, response, error);
    close_socket(fd);
    if (!ok || !response.ok) {
      std::cerr << (error.empty() ? response.error : error) << '\n';
      return 1;
    }
    std::cout << "artifact_id=" << response.artifact_id
              << " bytes=" << response.artifact.size() << '\n'
              << response.manifest << '\n';
    return 0;
  }
  if (argc != 4) {
    std::cerr << "usage: onnx-remote-client HOST PORT OP\n"
                 "   or: onnx-remote-client --subgraph HOST PORT MODEL.onnx "
                 "V0,V1,... [--shape D0,D1]\n";
    return 2;
  }
  int fd = connect_tcp(argv[1], static_cast<uint16_t>(std::strtoul(argv[2], nullptr, 10)));
  if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
  Request r; r.op = argv[3]; r.inputs.push_back(Tensor{{5}, {-2, -1, 0, 1, 2}});
  std::string error; Response response;
  bool ok = send_request(fd, r, error) && receive_response(fd, response, error);
  close_socket(fd);
  if (!ok || !response.ok) { std::cerr << (error.empty() ? response.error : error) << '\n'; return 1; }
  for (float x : response.outputs[0].data) std::cout << x << '\n';
  return 0;
}
