// Axelera Metis runner for Voyager SDK artifacts (compiled by scripts/axelera/onnx_remote_compile.py).
//
// An artifact is an ONXAXL1 bundle: the 8-byte magic "ONXAXL1\0", a little-endian u32 entry count, then per entry a
// u32 name length, the relative name, a u64 byte size and the bytes. The compiler writes the Voyager output directory
// (model.json, kernels, weights) into it, and this worker unpacks it under --cache-dir and loads model.json with the SDK's
// axr_load_model().
//
// Operations:
//   capabilities                  runner manifest: runner id, devices, supported dtypes
//   load_compiled(id, artifact)   unpack and load the model onto a connected Metis device once; keep it resident
//   run_compiled(id, tensors)     float32 inputs in graph order -> float32 outputs in graph order
//
// Quantization follows the runtime's documented contract (docs/reference/apis/axruntime-py.md): an input is quantized as
// q = round(x / scale) + zero_point, clipped to the integer type, and padded with zero_point; an output is depadded and
// dequantized as (q - zero_point) * scale. Models that need host-side preamble or postamble graphs are rejected. When the
// model's input_tensor_layout is NHWC, rank-4 NCHW tensors are transposed on the way in and out.
//
// Builds against the Voyager SDK's axruntime C API (axruntime/axruntime.h and libaxruntime.so; see the CMake options).
#include "remote_transport.h"

#include "axruntime/axruntime.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace onnx_remote;

namespace {

constexpr char kBundleMagic[8] = {'O', 'N', 'X', 'A', 'X', 'L', '1', '\0'};
constexpr size_t kMaxBundleNameLen = 512;

std::string g_cache_dir;
std::string g_manifest;          // capabilities document, built once at startup
axrContext* g_ctx = nullptr;

// Host trace written by the SDK runtime (AXE_LZ_PERF_LOG_PATH, set in main): one line per frame, appended before
// axr_run_model_instance returns. Runs are serialized under g_run_mu so each new line belongs to exactly one request.
std::string g_trace_path;
uint64_t g_trace_offset = 0;
std::mutex g_run_mu;
constexpr uintmax_t kTraceMaxBytes = 64ull << 20;  // truncate the trace past this size; the runtime appends from 0 again

uint64_t now_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

template <typename T>
const axrObject* as_object(const T* p) {
  return reinterpret_cast<const axrObject*>(p);
}

std::string last_error(const axrObject* o) {
  const char* s = o ? axr_last_error_string(o) : nullptr;
  return s ? std::string(s) : std::string("unknown error");
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (static_cast<unsigned char>(c) < 0x20) { out += ' '; }
    else out += c;
  }
  return out;
}

bool safe_artifact_id(const std::string& id) {
  if (id.empty() || id.size() > 128 || id[0] == '.') return false;
  return std::all_of(id.begin(), id.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '.' || c == '_' || c == '-';
  });
}

bool safe_entry_name(const std::string& name) {
  if (name.empty() || name[0] == '/' || name.find('\\') != std::string::npos) return false;
  // Reject any ".." path component: entries must stay inside the artifact directory.
  std::istringstream parts(name);
  for (std::string part; std::getline(parts, part, '/');)
    if (part == ".." || part.empty()) return false;
  return true;
}

// Parses an ONXAXL1 bundle into name -> bytes. Every length is checked before it is used.
bool parse_bundle(const std::vector<uint8_t>& bytes, std::map<std::string, std::vector<uint8_t>>& files,
                  std::string& error) {
  if (bytes.size() < 12 || std::memcmp(bytes.data(), kBundleMagic, sizeof(kBundleMagic)) != 0) {
    error = "artifact is not an ONXAXL1 bundle";
    return false;
  }
  size_t pos = sizeof(kBundleMagic);
  auto read_le = [&](size_t n, uint64_t& v) {
    if (bytes.size() - pos < n) return false;
    v = 0;
    for (size_t i = 0; i < n; ++i) v |= static_cast<uint64_t>(bytes[pos + i]) << (8 * i);
    pos += n;
    return true;
  };
  uint64_t count = 0;
  if (!read_le(4, count)) { error = "truncated bundle header"; return false; }
  for (uint64_t k = 0; k < count; ++k) {
    uint64_t name_len = 0, size = 0;
    if (!read_le(4, name_len) || name_len == 0 || name_len > kMaxBundleNameLen || bytes.size() - pos < name_len) {
      error = "bad bundle entry name";
      return false;
    }
    std::string name(bytes.begin() + pos, bytes.begin() + pos + name_len);
    pos += name_len;
    if (!read_le(8, size) || size > bytes.size() - pos) { error = "truncated bundle entry " + name; return false; }
    if (!safe_entry_name(name)) { error = "unsafe bundle entry name: " + name; return false; }
    files[name] = std::vector<uint8_t>(bytes.begin() + pos, bytes.begin() + pos + size);
    pos += size;
  }
  return true;
}

// Layout helpers. Shapes are row-major; a compiled tensor's logical shape is its full shape minus its padding.
std::vector<size_t> logical_shape(const axrTensorInfo& t) {
  std::vector<size_t> s(t.ndims);
  for (size_t d = 0; d < t.ndims; ++d) s[d] = t.dims[d] - t.padding[d][0] - t.padding[d][1];
  return s;
}

// Permutes a row-major tensor: output dim i is input dim perm[i].
template <typename T>
std::vector<T> permute(const std::vector<T>& src, const std::vector<size_t>& shape, const std::vector<size_t>& perm) {
  const size_t rank = shape.size();
  std::vector<size_t> istride(rank), oshape(rank);
  size_t acc = 1;
  for (size_t i = rank; i-- > 0;) { istride[i] = acc; acc *= shape[i]; }
  for (size_t i = 0; i < rank; ++i) oshape[i] = shape[perm[i]];
  std::vector<T> out(src.size());
  for (size_t o = 0; o < out.size(); ++o) {
    size_t rem = o, off = 0;
    for (size_t i = rank; i-- > 0;) {
      const size_t c = rem % oshape[i];
      rem /= oshape[i];
      off += c * istride[perm[i]];
    }
    out[o] = src[off];
  }
  return out;
}

// One frame's host-side timing, as the runtime reports it: all fields are nanoseconds, beg on the monotonic clock.
// Stages run back to back in this order, so prep + inp + krn + out is tot (within a few microseconds).
struct HostTrace {
  uint64_t beg_ns = 0, prep_ns = 0, inp_ns = 0, krn_ns = 0, out_ns = 0, tot_ns = 0;
};

// Parses "[T] idx:0:00000001 beg:<hex> prep:<hex> inp:<hex> krn:<hex> out:<hex> tot:<hex>".
bool parse_host_trace(const std::string& line, HostTrace& t) {
  std::istringstream in(line);
  std::string tok;
  int found = 0;
  while (in >> tok) {
    const size_t colon = tok.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = tok.substr(0, colon);
    uint64_t* dst = nullptr;
    if (key == "beg") dst = &t.beg_ns;
    else if (key == "prep") dst = &t.prep_ns;
    else if (key == "inp") dst = &t.inp_ns;
    else if (key == "krn") dst = &t.krn_ns;
    else if (key == "out") dst = &t.out_ns;
    else if (key == "tot") dst = &t.tot_ns;
    if (!dst) continue;
    try {
      *dst = std::stoull(tok.substr(colon + 1), nullptr, 16);
    } catch (const std::exception&) {
      return false;
    }
    ++found;
  }
  return found == 6;
}

// Returns the complete trace lines appended since the last call and advances the offset past them. Caller holds g_run_mu.
std::vector<HostTrace> read_new_host_trace() {
  std::vector<HostTrace> rows;
  std::error_code ec;
  const auto size = fs::file_size(g_trace_path, ec);
  if (ec) return rows;  // no trace yet (runtime writes it on the first run)
  if (size < g_trace_offset) g_trace_offset = 0;  // the file was truncated outside this process
  if (size <= g_trace_offset) return rows;
  std::ifstream f(g_trace_path, std::ios::binary);
  f.seekg(static_cast<std::streamoff>(g_trace_offset));
  std::string chunk(static_cast<size_t>(size - g_trace_offset), '\0');
  f.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
  chunk.resize(static_cast<size_t>(f.gcount()));
  const size_t last_nl = chunk.rfind('\n');
  if (last_nl == std::string::npos) return rows;  // a partial line; read it once it is complete
  std::istringstream lines(chunk.substr(0, last_nl));
  for (std::string line; std::getline(lines, line);) {
    HostTrace t;
    if (parse_host_trace(line, t)) rows.push_back(t);
  }
  g_trace_offset += last_nl + 1;
  if (size > static_cast<uintmax_t>(kTraceMaxBytes)) {  // bound the file; the runtime keeps appending from offset 0
    fs::resize_file(g_trace_path, 0, ec);
    g_trace_offset = 0;
  }
  return rows;
}

// Emits the host stages of one frame as profile events, positioned on the request's timeline (t0 is the request start).
void add_host_stages(Response& r, const Request& req, uint64_t t0, const HostTrace& t) {
  if (req.profiling != ProfilingLevel::Detailed) return;
  auto rel_us = [t0](uint64_t ns) { const uint64_t us = ns / 1000; return us > t0 ? us - t0 : 0; };
  const uint64_t b = t.beg_ns;
  const struct {
    const char* name;
    uint64_t start_ns, dur_ns;
  } stages[] = {
      {"axelera_host_prep", b, t.prep_ns},
      {"axelera_host_input", b + t.prep_ns, t.inp_ns},
      {"axelera_host_kernel", b + t.prep_ns + t.inp_ns, t.krn_ns},
      {"axelera_host_output", b + t.prep_ns + t.inp_ns + t.krn_ns, t.out_ns},
  };
  for (const auto& s : stages)
    r.profile.push_back(ProfileEvent{s.name, "axelera", rel_us(s.start_ns), s.dur_ns / 1000, "host trace"});
}

struct Net {
  axrModel* model = nullptr;
  axrConnection* connection = nullptr;
  axrModelInstance* instance = nullptr;
  std::vector<axrTensorInfo> inputs, outputs;
  bool nhwc = false;

  ~Net() {
    if (instance) axr_destroy(as_object(instance));
    if (connection) axr_destroy(as_object(connection));
    if (model) axr_destroy(as_object(model));
  }
};

std::mutex g_nets_mu;
std::map<std::string, std::shared_ptr<Net>> g_nets;

std::shared_ptr<Net> get_net(const std::string& id) {
  std::lock_guard<std::mutex> lk(g_nets_mu);
  auto it = g_nets.find(id);
  return it == g_nets.end() ? nullptr : it->second;
}

void add_profile(Response& r, const Request& req, const char* name, uint64_t begin, uint64_t duration,
                 const std::string& detail) {
  if (req.profiling != ProfilingLevel::Off) r.profile.push_back(ProfileEvent{name, "axelera", begin, duration, detail});
}

// Loads the bundle as a resident model. Replaces any earlier net with the same id.
bool load(const std::string& id, const std::vector<uint8_t>& artifact, Response& response) {
  if (!safe_artifact_id(id)) { response.error = "artifact_id must be 1-128 characters of [A-Za-z0-9._-]"; return false; }
  std::map<std::string, std::vector<uint8_t>> files;
  if (!parse_bundle(artifact, files, response.error)) return false;
  if (!files.count("model.json")) { response.error = "bundle has no model.json"; return false; }

  const fs::path dir = fs::path(g_cache_dir) / id;
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  for (const auto& [name, data] : files) {
    const fs::path p = dir / name;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!f) { response.error = "cannot write " + p.string(); return false; }
  }

  auto net = std::make_shared<Net>();
  net->model = axr_load_model(g_ctx, (dir / "model.json").string().c_str());
  if (!net->model) { response.error = "axr_load_model: " + last_error(as_object(g_ctx)); return false; }

  // Host-side preamble/postamble graphs would have to run in an ONNX runtime on the host; this worker does not run them.
  if (const axrProperties* mp = axr_get_model_properties(net->model)) {
    const char* pre = axr_get_property(const_cast<axrProperties*>(mp), "preamble_graph");
    const char* post = axr_get_property(const_cast<axrProperties*>(mp), "postamble_graph");
    if ((pre && *pre) || (post && *post)) {
      response.error = "model needs host-side preamble/postamble graphs, which this worker does not run";
      return false;
    }
    const char* layout = axr_get_property(const_cast<axrProperties*>(mp), "input_tensor_layout");
    net->nhwc = layout && std::string(layout) == "NHWC";
  }

  net->connection = axr_device_connect(g_ctx, nullptr, 1, nullptr);
  if (!net->connection) { response.error = "axr_device_connect: " + last_error(as_object(g_ctx)); return false; }
  // host_profiling makes the runtime append one trace line per frame (see read_new_host_trace).
  axrProperties* ip = axr_create_properties(g_ctx, "aipu_cores=1\nnum_sub_devices=1\nhost_profiling=1\n");
  net->instance = axr_load_model_instance(net->connection, net->model, ip);
  axr_destroy(as_object(ip));
  if (!net->instance) { response.error = "axr_load_model_instance: " + last_error(as_object(net->connection)); return false; }

  for (size_t i = 0; i < axr_num_model_inputs(net->model); ++i) net->inputs.push_back(axr_get_model_input(net->model, i));
  for (size_t i = 0; i < axr_num_model_outputs(net->model); ++i) net->outputs.push_back(axr_get_model_output(net->model, i));
  for (const auto& t : net->inputs)
    if (t.ndims == 0 || std::isnan(t.scale) || t.scale == 0.0) {
      response.error = "input tensor needs a non-zero per-tensor scale (per-channel quantization is not supported)";
      return false;
    }
  for (const auto& t : net->outputs)
    if (std::isnan(t.scale)) { response.error = "output tensor uses per-channel quantization, which is not supported"; return false; }

  std::lock_guard<std::mutex> lk(g_nets_mu);
  g_nets[id] = std::move(net);
  return true;
}

// Quantizes one float32 input (ONNX order) into the padded device buffer for `info`.
bool quantize_input(const axrTensorInfo& info, bool nhwc, const Tensor& t, std::vector<uint8_t>& buf,
                    std::string& error) {
  if (t.dtype != 1) { error = "only FLOAT inputs are supported"; return false; }
  std::vector<size_t> onnx_shape(t.shape.begin(), t.shape.end());
  std::vector<size_t> lshape = logical_shape(info);
  std::vector<float> nat;
  if (onnx_shape == lshape) {
    nat = t.data;
  } else if (nhwc && onnx_shape.size() == 4 && lshape.size() == 4 && lshape[0] == onnx_shape[0] &&
             lshape[1] == onnx_shape[2] && lshape[2] == onnx_shape[3] && lshape[3] == onnx_shape[1]) {
    nat = permute(t.data, onnx_shape, {0, 2, 3, 1});  // NCHW -> NHWC
  } else {
    error = "input shape does not match the compiled model";
    return false;
  }
  size_t count = 1;
  for (size_t d : lshape) count *= d;
  if (nat.size() != count) { error = "input element count does not match the compiled model"; return false; }

  const bool is_signed = info.type == AXR_SIGNED;
  const int qmin = is_signed ? -128 : 0, qmax = is_signed ? 127 : 255;
  const int zp = info.zero_point;
  const float inv_scale = static_cast<float>(1.0 / info.scale);

  // Padded row-major strides of the device tensor.
  const size_t rank = info.ndims;
  std::vector<size_t> pstride(rank);
  size_t acc = 1;
  for (size_t i = rank; i-- > 0;) { pstride[i] = acc; acc *= info.dims[i]; }
  buf.assign(axr_tensor_size(&info), static_cast<uint8_t>(static_cast<int8_t>(zp)));

  std::vector<size_t> coord(rank, 0);
  for (size_t li = 0; li < count; ++li) {
    long q = std::lrint(static_cast<double>(nat[li]) * static_cast<double>(inv_scale)) + zp;
    q = std::clamp<long>(q, qmin, qmax);
    size_t off = 0;
    for (size_t d = 0; d < rank; ++d) off += (coord[d] + info.padding[d][0]) * pstride[d];
    buf[off] = static_cast<uint8_t>(static_cast<int8_t>(q));
    for (size_t d = rank; d-- > 0;) {  // advance the logical coordinate
      if (++coord[d] < lshape[d]) break;
      coord[d] = 0;
    }
  }
  return true;
}

// Dequantizes one padded device output into float32 in ONNX order.
void dequantize_output(const axrTensorInfo& info, bool nhwc, const std::vector<uint8_t>& buf, Tensor& out) {
  const std::vector<size_t> lshape = logical_shape(info);
  size_t count = 1;
  for (size_t d : lshape) count *= d;
  const bool is_signed = info.type == AXR_SIGNED;
  const float scale = static_cast<float>(info.scale);
  const int zp = info.zero_point;

  const size_t rank = info.ndims;
  std::vector<size_t> pstride(rank);
  size_t acc = 1;
  for (size_t i = rank; i-- > 0;) { pstride[i] = acc; acc *= info.dims[i]; }

  std::vector<float> nat(count);
  std::vector<size_t> coord(rank, 0);
  for (size_t li = 0; li < count; ++li) {
    size_t off = 0;
    for (size_t d = 0; d < rank; ++d) off += (coord[d] + info.padding[d][0]) * pstride[d];
    const int q = is_signed ? static_cast<int>(static_cast<int8_t>(buf[off])) : static_cast<int>(buf[off]);
    nat[li] = static_cast<float>(q - zp) * scale;
    for (size_t d = rank; d-- > 0;) {
      if (++coord[d] < lshape[d]) break;
      coord[d] = 0;
    }
  }

  std::vector<size_t> onnx_shape = lshape;
  if (nhwc && lshape.size() == 4) {
    nat = permute(nat, lshape, {0, 3, 1, 2});  // NHWC -> NCHW
    onnx_shape = {lshape[0], lshape[3], lshape[1], lshape[2]};
  }
  out.dtype = 1;
  out.shape.assign(onnx_shape.begin(), onnx_shape.end());
  out.data = std::move(nat);
}

Response run(const Request& request, const std::shared_ptr<Net>& net_ptr, Response& response, uint64_t t0) {
  Net& net = *net_ptr;
  if (request.inputs.size() != net.inputs.size()) {
    response.error = "input count " + std::to_string(request.inputs.size()) + " differs from the model's " +
                     std::to_string(net.inputs.size());
    return response;
  }
  const uint64_t q_begin = now_us() - t0;
  std::vector<std::vector<uint8_t>> in_bufs(net.inputs.size());
  std::vector<axrArgument> in_args(net.inputs.size());
  for (size_t i = 0; i < net.inputs.size(); ++i) {
    if (!quantize_input(net.inputs[i], net.nhwc, request.inputs[i], in_bufs[i], response.error)) return response;
    in_args[i] = axrArgument{};
    in_args[i].ptr = in_bufs[i].data();
    in_args[i].size = in_bufs[i].size();
  }
  add_profile(response, request, "axelera_quantize", q_begin, now_us() - t0 - q_begin, "");

  std::vector<std::vector<uint8_t>> out_bufs(net.outputs.size());
  std::vector<axrArgument> out_args(net.outputs.size());
  for (size_t i = 0; i < net.outputs.size(); ++i) {
    out_bufs[i].assign(axr_tensor_size(&net.outputs[i]), 0);
    out_args[i] = axrArgument{};
    out_args[i].ptr = out_bufs[i].data();
    out_args[i].size = out_bufs[i].size();
  }

  std::vector<HostTrace> trace;
  {
    std::lock_guard<std::mutex> run_lk(g_run_mu);
    const uint64_t run_begin = now_us() - t0;
    const axrResult r = axr_run_model_instance(net.instance, in_args.data(), in_args.size(), out_args.data(), out_args.size());
    if (r != AXR_SUCCESS) {
      response.error = std::string("axr_run_model_instance: ") + axr_error_string(r) + ": " +
                       last_error(as_object(net.instance));
      return response;
    }
    add_profile(response, request, "axelera_device_run", run_begin, now_us() - t0 - run_begin, "");
    trace = read_new_host_trace();
  }
  // One line per frame; this request ran one frame, so take the newest line if the trace has any.
  if (!trace.empty()) add_host_stages(response, request, t0, trace.back());

  const uint64_t d_begin = now_us() - t0;
  response.outputs.resize(net.outputs.size());
  for (size_t i = 0; i < net.outputs.size(); ++i) dequantize_output(net.outputs[i], net.nhwc, out_bufs[i], response.outputs[i]);
  add_profile(response, request, "axelera_dequantize", d_begin, now_us() - t0 - d_begin, "");
  return response;
}

std::string manifest_document() { return g_manifest; }

Response execute(const Request& request) {
  Response response;
  response.request_id = request.request_id;
  const uint64_t t0 = now_us();
  if (request.op == "capabilities") {
    response.ok = true;
    response.artifact_id = "axelera-metis";
    response.manifest = manifest_document();
    return response;
  }
  if (request.op != "load_compiled" && request.op != "run_compiled") {
    response.error = "axelera runner accepts capabilities/load_compiled/run_compiled";
    return response;
  }

  std::shared_ptr<Net> net;
  if (request.op == "load_compiled" && request.artifact.empty()) {
    net = get_net(request.artifact_id);
    if (!net) { response.error = "load_compiled needs artifact bytes for an unknown artifact_id"; return response; }
  } else if (!request.artifact.empty() || request.op == "load_compiled") {
    if (!load(request.artifact_id, request.artifact, response)) return response;
    add_profile(response, request, "axelera_load", 0, now_us() - t0, request.artifact_id);
    net = get_net(request.artifact_id);
  } else {
    net = get_net(request.artifact_id);
    if (!net) { response.error = "artifact not loaded: send the artifact bytes or load_compiled first"; return response; }
  }
  response.artifact_id = request.artifact_id;
  if (request.op == "load_compiled") { response.ok = true; return response; }

  Response out = run(request, net, response, t0);
  if (out.error.empty()) {
    out.ok = true;
    out.artifact_id = request.artifact_id;
  }
  return out;
}

int listen_on(const std::string& host, uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1 || ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) ||
      ::listen(fd, 16)) {
    ::close(fd);
    return -1;
  }
  return fd;
}

// Lists the Metis devices and builds the capabilities document. Returns false when no device is visible.
bool build_manifest(std::string& error) {
  axrDeviceInfo* devices = nullptr;
  const size_t n = axr_list_devices(g_ctx, &devices);
  if (n == 0 || !devices) { error = "no Axelera device visible: " + last_error(as_object(g_ctx)); return false; }
  std::ostringstream m;
  m << "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\",\"runner_id\":\"axelera-metis\",\"ready\":true,"
       "\"graph_execution\":false,\"supported_ops\":[\"load_compiled\",\"run_compiled\"],"
       "\"supported_dtypes\":[\"FLOAT\"],\"profiling\":true,\"artifact\":{\"format\":\"axelera-bundle\"},\"hardware\":{\"devices\":[";
  for (size_t i = 0; i < n; ++i) {
    if (i) m << ',';
    m << "{\"name\":\"" << json_escape(devices[i].name) << "\",\"board\":\""
      << json_escape(axr_board_type_string(devices[i].board_type)) << "\",\"firmware\":\""
      << json_escape(devices[i].firmware_version) << "\",\"subdevices\":" << devices[i].subdevice_count << '}';
  }
  m << "]}}";
  g_manifest = m.str();
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  uint16_t port = 39505;
  int io_timeout_ms = 60000;
  g_cache_dir = std::getenv("HOME") ? std::string(std::getenv("HOME")) + "/.cache/onnx-remote-axelera" : "/tmp/onnx-remote-axelera";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) host = argv[++i];
    else if (a == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::stoul(argv[++i]));
    else if (a == "--cache-dir" && i + 1 < argc) g_cache_dir = argv[++i];
    else if (a == "--io-timeout-ms" && i + 1 < argc) io_timeout_ms = std::atoi(argv[++i]);
    else if (a == "--help") {
      std::cout << "usage: onnx-remote-axelera-worker [--host ADDR (default 127.0.0.1)] [--port PORT (default 39505)]\n"
                   "                                  [--cache-dir DIR] [--io-timeout-ms MS (default 60000)]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << a << '\n';
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);

  // Host trace: the runtime reads these when it starts. The file is per worker process, so start it empty.
  std::error_code ec;
  fs::create_directories(g_cache_dir, ec);
  g_trace_path = g_cache_dir + "/host_trace.log";
  fs::remove(g_trace_path, ec);
  g_trace_offset = 0;
  setenv("AXE_LZ_PERF_LOG_PATH", g_trace_path.c_str(), 1);
  setenv("AXE_PROFILING_CONFIG", "HOST_TEMPLATE", 1);

  g_ctx = axr_create_context();
  if (!g_ctx) { std::cerr << "axr_create_context failed\n"; return 1; }
  std::string error;
  if (!build_manifest(error)) { std::cerr << error << '\n'; return 1; }

  const int listener = listen_on(host, port);
  if (listener < 0) { std::cerr << "listen failed: " << std::strerror(errno) << '\n'; return 1; }
  std::cerr << "onnx-remote-axelera-worker listening on " << host << ':' << port << " (cache " << g_cache_dir << ")\n";

  // Thread per connection (the transport is one request per socket); the cap bounds memory held by in-flight tensors.
  std::atomic<int> in_flight{0};
  constexpr int kMaxInFlight = 8;
  for (;;) {
    const int fd = accept_tcp(listener);
    if (fd < 0) continue;
    if (in_flight >= kMaxInFlight) {
      Response busy;
      busy.error = "axelera runner busy (too many concurrent requests)";
      std::string e;
      send_response(fd, busy, e);
      close_socket(fd);
      continue;
    }
    ++in_flight;
    std::thread([fd, &in_flight, io_timeout_ms] {
      Request request;
      Response response;
      std::string e;
      set_socket_io_timeout(fd, io_timeout_ms);
      if (!receive_request(fd, request, e)) response.error = e;
      else response = execute(request);
      if (!send_response(fd, response, e)) std::cerr << "response failed: " << e << '\n';
      close_socket(fd);
      --in_flight;
    }).detach();
  }
}
