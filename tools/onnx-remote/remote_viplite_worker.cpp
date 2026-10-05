// Allwinner VIPLite (Vivante VIP9000 NPU) runner for compiled NBG artifacts: the Allwinner A733 / T527 / T736 / V853 family.
//
// An artifact is a plain Network Binary Graph (`.nb`) as written by Acuity's `pegasus export ovxlib --pack-nbg-unify`
// (scripts/allwinner/compile_nbg.py is the onnx-remote-compiler command that makes one). Operations:
//   capabilities                     the runner manifest, with the NPU's hardware ID and driver version
//   load_compiled(id, artifact)      create + prepare the network once and keep it resident (several can be resident)
//   run_compiled(id, tensors)        ONNX inputs in graph order -> ONNX outputs in graph order
//
// The NBG carries each input's and output's data format and quantization (affine scale/zero-point or dynamic fixed point),
// so the worker converts on the way in and out: a FLOAT tensor is quantized to the input's format, and every output comes back
// as FLOAT. A tensor already in the input's native integer format (UINT8/INT8/INT16) is copied through untouched, which is what
// an application feeding camera frames to a uint8 network wants. Shapes come back in ONNX (NCHW) order: VIPLite lists
// dimensions innermost first, so the worker reverses them.
//
// Builds against the VIPLite SDK's headers and libNBGlinker.so (see the CMake options); the headers are not in this repo.
#include "remote_transport.h"

#include <vip_lite.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace onnx_remote;
namespace fs = std::filesystem;

namespace {

// ONNX TensorProto.DataType values the transport carries.
constexpr uint8_t kFloat = 1, kUint8 = 2, kInt8 = 3, kInt16 = 5, kFloat16 = 10;

struct Port {
  std::string name;
  vip_buffer_create_params_t params{};
  uint64_t elements = 0;
  size_t bytes = 0;
  vip_buffer buffer = nullptr;
  // The buffer wraps this cached, aligned host allocation (vip_create_buffer_from_handle): converting in place is then ordinary
  // memory traffic, where reading or writing the driver's own mapping of a vip_create_buffer buffer is uncached and very slow.
  uint8_t* host = nullptr;
  size_t host_bytes = 0;
};

// One independent set of input/output buffers. A network has two, so one request can convert its input (or read its output) while
// another request's NPU run uses the other set: conversion on the CPU overlaps with execution on the NPU.
struct Slot {
  std::vector<Port> inputs, outputs;
  bool busy = false;
  ~Slot() {
    for (auto* ports : {&inputs, &outputs})
      for (auto& p : *ports) {
        if (p.buffer) vip_destroy_buffer(p.buffer);
        std::free(p.host);
      }
  }
};
constexpr size_t kSlots = 2;

struct Net {
  vip_network network = nullptr;
  std::vector<std::unique_ptr<Slot>> slots;
  std::mutex mu;  // guards Slot::busy and the spare pool
  std::condition_variable cv;
  // Output float vectors handed back after a response was sent (see recycle), one set (a vector per output) each. A fresh multi-MB
  // std::vector costs more to zero-fill and page-fault (about 5 ms for YOLOv5s' 1.6M-element output) than the dequantization itself.
  std::vector<std::vector<std::vector<float>>> spare_pool;
  Slot* bound = nullptr;  // the slot whose buffers are set on the network; only touched under g_npu
  ~Net() {
    slots.clear();
    if (network) { vip_finish_network(network); vip_destroy_network(network); }
  }
};

fs::path g_cache_dir = "/data/local/tmp/onnx-remote-viplite";
std::mutex g_mu;   // guards g_nets and loading
std::mutex g_npu;  // one vip_run_network (and the buffer rebinding before it) at a time
std::map<std::string, std::shared_ptr<Net>> g_nets;

std::shared_ptr<Net> get_net(const std::string& id) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_nets.find(id);
  return it == g_nets.end() ? nullptr : it->second;
}

uint64_t now_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool valid_artifact_id(const std::string& id) {
  if (id.empty() || id.size() > kMaxArtifactIdBytes) return false;
  for (char c : id)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
      return false;
  return true;
}

size_t format_bytes(vip_enum format) {
  switch (format) {
    case VIP_BUFFER_FORMAT_FP32: case VIP_BUFFER_FORMAT_INT32: case VIP_BUFFER_FORMAT_UINT32: return 4;
    case VIP_BUFFER_FORMAT_FP16: case VIP_BUFFER_FORMAT_BFP16: case VIP_BUFFER_FORMAT_INT16: case VIP_BUFFER_FORMAT_UINT16: return 2;
    case VIP_BUFFER_FORMAT_UINT8: case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_CHAR: case VIP_BUFFER_FORMAT_BOOL8: return 1;
    case VIP_BUFFER_FORMAT_INT64: case VIP_BUFFER_FORMAT_UINT64: case VIP_BUFFER_FORMAT_FP64: return 8;
    default: return 0;
  }
}

float half_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff, bits;
  if (exp == 0) {
    if (mant == 0) bits = sign;
    else {  // subnormal
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; --exp; }
      bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float f; std::memcpy(&f, &bits, 4); return f;
}

uint16_t float_to_half(float f) {
  uint32_t x; std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  if (mant & 0x1000u) ++half;  // round to nearest; a carry into the exponent is the correct result
  return static_cast<uint16_t>(half);
}

// Range of the integer formats, for saturating quantization.
bool int_range(vip_enum format, double& lo, double& hi) {
  switch (format) {
    case VIP_BUFFER_FORMAT_UINT8: lo = 0; hi = 255; return true;
    case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_CHAR: lo = -128; hi = 127; return true;
    case VIP_BUFFER_FORMAT_UINT16: lo = 0; hi = 65535; return true;
    case VIP_BUFFER_FORMAT_INT16: lo = -32768; hi = 32767; return true;
    case VIP_BUFFER_FORMAT_INT32: lo = -2147483648.0; hi = 2147483647.0; return true;
    default: return false;
  }
}

double read_integer(const uint8_t* p, vip_enum format) {
  switch (format) {
    case VIP_BUFFER_FORMAT_UINT8: case VIP_BUFFER_FORMAT_BOOL8: return *p;
    case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_CHAR: return static_cast<int8_t>(*p);
    case VIP_BUFFER_FORMAT_UINT16: { uint16_t v; std::memcpy(&v, p, 2); return v; }
    case VIP_BUFFER_FORMAT_INT16: { int16_t v; std::memcpy(&v, p, 2); return v; }
    case VIP_BUFFER_FORMAT_INT32: { int32_t v; std::memcpy(&v, p, 4); return v; }
    case VIP_BUFFER_FORMAT_UINT32: { uint32_t v; std::memcpy(&v, p, 4); return v; }
    default: return 0;
  }
}

void write_integer(uint8_t* p, vip_enum format, double q) {
  switch (format) {
    case VIP_BUFFER_FORMAT_UINT8: *p = static_cast<uint8_t>(q); break;
    case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_CHAR: *reinterpret_cast<int8_t*>(p) = static_cast<int8_t>(q); break;
    case VIP_BUFFER_FORMAT_UINT16: { uint16_t v = static_cast<uint16_t>(q); std::memcpy(p, &v, 2); break; }
    case VIP_BUFFER_FORMAT_INT16: { int16_t v = static_cast<int16_t>(q); std::memcpy(p, &v, 2); break; }
    case VIP_BUFFER_FORMAT_INT32: { int32_t v = static_cast<int32_t>(q); std::memcpy(p, &v, 4); break; }
    default: break;
  }
}

// real = (q - zero_point) * scale (affine) or q * 2^-fixed_point_pos (dynamic fixed point).
double dequantize(double q, const vip_buffer_create_params_t& p) {
  if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) return (q - p.quant_data.affine.zeroPoint) * p.quant_data.affine.scale;
  if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) return std::ldexp(q, -p.quant_data.dfp.fixed_point_pos);
  return q;
}

double quantize(double x, const vip_buffer_create_params_t& p) {
  double q = x;
  if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) q = std::nearbyint(x / p.quant_data.affine.scale) + p.quant_data.affine.zeroPoint;
  else if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) q = std::nearbyint(std::ldexp(x, p.quant_data.dfp.fixed_point_pos));
  else if (p.data_format != VIP_BUFFER_FORMAT_FP32 && p.data_format != VIP_BUFFER_FORMAT_FP16) q = std::nearbyint(x);
  double lo, hi;
  if (int_range(p.data_format, lo, hi)) q = std::min(hi, std::max(lo, q));
  return q;
}

// Bulk converters. The per-element helpers above are exact but branch on the format per value; these hoist the format, scale and range
// out of the loop so the compiler can vectorize the 1-2 byte formats that real networks use.
struct QuantParams { float scale, zero; float lo, hi; };  // real -> q: clamp(rint(x / scale + zero))

template <typename T>
void quantize_bulk(const float* src, T* dst, uint64_t n, const QuantParams& q) {
  const float inv = 1.0f / q.scale;
  for (uint64_t i = 0; i < n; ++i) {
    float v = std::nearbyintf(src[i] * inv + q.zero);
    dst[i] = static_cast<T>(std::min(q.hi, std::max(q.lo, v)));
  }
}

template <typename T>
void dequantize_bulk(const T* src, float* dst, uint64_t n, const QuantParams& q) {
  for (uint64_t i = 0; i < n; ++i) dst[i] = (static_cast<float>(src[i]) - q.zero) * q.scale;
}

// The affine/DFP/none triple expressed as one (scale, zero) pair: DFP is scale 2^-pos with zero 0, "none" is the identity.
QuantParams quant_params(const vip_buffer_create_params_t& p) {
  QuantParams q{1.0f, 0.0f, -INFINITY, INFINITY};
  if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) { q.scale = p.quant_data.affine.scale; q.zero = static_cast<float>(p.quant_data.affine.zeroPoint); }
  else if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) q.scale = std::ldexp(1.0f, -p.quant_data.dfp.fixed_point_pos);
  double lo, hi;
  if (int_range(p.data_format, lo, hi)) { q.lo = static_cast<float>(lo); q.hi = static_cast<float>(hi); }
  return q;
}

bool query_port(vip_network net, bool is_input, uint32_t index, Port& port, std::string& error) {
  auto query = [&](vip_enum prop, void* out) {
    return (is_input ? vip_query_input(net, index, prop, out) : vip_query_output(net, index, prop, out)) == VIP_SUCCESS;
  };
  vip_buffer_create_params_t& p = port.params;
  std::memset(&p, 0, sizeof(p));
  p.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
  char name[256] = {0};
  if (!query(VIP_BUFFER_PROP_DATA_FORMAT, &p.data_format) || !query(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &p.num_of_dims) ||
      !query(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p.sizes) || !query(VIP_BUFFER_PROP_QUANT_FORMAT, &p.quant_format)) {
    error = "cannot query NBG tensor properties";
    return false;
  }
  query(VIP_BUFFER_PROP_NAME, name);
  port.name = name;
  if (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) query(VIP_BUFFER_PROP_FIXED_POINT_POS, &p.quant_data.dfp.fixed_point_pos);
  else if (p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
    query(VIP_BUFFER_PROP_TF_SCALE, &p.quant_data.affine.scale);
    query(VIP_BUFFER_PROP_TF_ZERO_POINT, &p.quant_data.affine.zeroPoint);
  }
  if (p.num_of_dims == 0 || p.num_of_dims > 6 || format_bytes(p.data_format) == 0) {
    error = "unsupported NBG tensor (dims/format) for " + port.name;
    return false;
  }
  port.elements = 1;
  for (uint32_t k = 0; k < p.num_of_dims; ++k) port.elements *= p.sizes[k];
  port.bytes = static_cast<size_t>(port.elements) * format_bytes(p.data_format);
  return true;
}

// Wrap an aligned host allocation as the NPU buffer. Alignment is 64 bytes up to driver 2.0.3 and 256 after (as in Allwinner's runtime).
bool create_buffer(Port& port, std::string& error) {
  const size_t align = vip_get_version() <= 0x00020003 ? 64 : 256;
  port.host_bytes = (port.bytes + align - 1) / align * align;
  void* mem = nullptr;
  if (posix_memalign(&mem, align, port.host_bytes)) { error = "out of memory for " + port.name; return false; }
  std::memset(mem, 0, port.host_bytes);
  port.host = static_cast<uint8_t*>(mem);
  port.params.memory_type = VIP_BUFFER_MEMORY_TYPE_HOST;
  const vip_status_e st = vip_create_buffer_from_handle(&port.params, port.host, static_cast<vip_uint32_t>(port.host_bytes), &port.buffer);
  if (st != VIP_SUCCESS) { error = "vip_create_buffer_from_handle failed (" + std::to_string(st) + ") for " + port.name; return false; }
  return true;
}

// Point the network's inputs and outputs at a slot's buffers (before a run). Callers hold g_npu, or own the network exclusively.
bool bind(Net& net, Slot& slot, std::string& error) {
  if (net.bound == &slot) return true;
  for (uint32_t i = 0; i < slot.inputs.size(); ++i)
    if (vip_set_input(net.network, i, slot.inputs[i].buffer) != VIP_SUCCESS) { error = "vip_set_input failed"; return false; }
  for (uint32_t i = 0; i < slot.outputs.size(); ++i)
    if (vip_set_output(net.network, i, slot.outputs[i].buffer) != VIP_SUCCESS) { error = "vip_set_output failed"; return false; }
  net.bound = &slot;
  return true;
}

bool load(const std::string& id, const std::vector<uint8_t>& nb, Response& response) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!valid_artifact_id(id)) { response.error = "invalid artifact id"; return false; }
  std::error_code ec;
  fs::create_directories(g_cache_dir, ec);
  const fs::path path = g_cache_dir / (id + ".nb");
  if (!nb.empty()) {
    if (nb.size() > kMaxArtifactBytes) { response.error = "artifact exceeds transport limit"; return false; }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(nb.data()), static_cast<std::streamsize>(nb.size()));
    if (!out) { response.error = "cannot write artifact cache"; return false; }
    g_nets.erase(id);  // a re-upload replaces the resident network (requests still running keep the old one alive)
  }
  if (g_nets.count(id)) return true;
  std::ifstream in(path, std::ios::binary);
  std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (bytes.empty()) { response.error = "artifact is not cached: " + id; return false; }

  auto net = std::make_unique<Net>();
  vip_status_e st = vip_create_network(bytes.data(), static_cast<vip_uint32_t>(bytes.size()), VIP_CREATE_NETWORK_FROM_MEMORY, &net->network);
  if (st != VIP_SUCCESS) {
    response.error = "vip_create_network failed (" + std::to_string(st) + "): not a valid NBG for this NPU generation";
    return false;
  }
  vip_uint32_t n_in = 0, n_out = 0;
  vip_query_network(net->network, VIP_NETWORK_PROP_INPUT_COUNT, &n_in);
  vip_query_network(net->network, VIP_NETWORK_PROP_OUTPUT_COUNT, &n_out);
  for (size_t k = 0; k < kSlots; ++k) net->slots.push_back(std::make_unique<Slot>());
  Slot& first = *net->slots[0];
  first.inputs.resize(n_in);
  first.outputs.resize(n_out);
  for (int side = 0; side < 2; ++side) {
    auto& ports = side == 0 ? first.inputs : first.outputs;
    for (uint32_t i = 0; i < ports.size(); ++i)
      if (!query_port(net->network, side == 0, i, ports[i], response.error)) return false;
  }
  for (size_t k = 0; k < kSlots; ++k) {  // every slot has the same tensors and its own buffers
    Slot& slot = *net->slots[k];
    if (k) { slot.inputs = first.inputs; slot.outputs = first.outputs; }
    for (auto* ports : {&slot.inputs, &slot.outputs})
      for (Port& port : *ports) {
        port.buffer = nullptr;
        port.host = nullptr;
        if (!create_buffer(port, response.error)) return false;
      }
  }
  if ((st = vip_prepare_network(net->network)) != VIP_SUCCESS) { response.error = "vip_prepare_network failed (" + std::to_string(st) + ")"; return false; }
  if (!bind(*net, first, response.error)) return false;
  g_nets[id] = std::move(net);
  return true;
}

// Fill one input buffer from a request tensor, converting straight into the buffer's host memory.
bool fill_input(const Port& port, const Tensor& t, std::string& error) {
  uint64_t elements = 1;
  for (int64_t d : t.shape) elements *= static_cast<uint64_t>(d);
  const vip_enum fmt = port.params.data_format;
  const size_t native = format_bytes(fmt);
  if (elements != port.elements) {
    error = "input " + port.name + " has " + std::to_string(elements) + " elements, the NBG expects " + std::to_string(port.elements);
    return false;
  }
  uint8_t* dst = port.host;
  if (t.dtype == kFloat) {
    const float* src = t.data.data();
    if (t.data.size() != elements) { error = "float input payload size mismatch"; return false; }
    const QuantParams q = quant_params(port.params);
    switch (fmt) {
      case VIP_BUFFER_FORMAT_FP32: std::memcpy(dst, src, port.bytes); break;
      case VIP_BUFFER_FORMAT_FP16: for (uint64_t i = 0; i < elements; ++i) { uint16_t h = float_to_half(src[i]); std::memcpy(dst + 2 * i, &h, 2); } break;
      case VIP_BUFFER_FORMAT_UINT8: quantize_bulk(src, dst, elements, q); break;
      case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_CHAR: quantize_bulk(src, reinterpret_cast<int8_t*>(dst), elements, q); break;
      case VIP_BUFFER_FORMAT_INT16: quantize_bulk(src, reinterpret_cast<int16_t*>(dst), elements, q); break;
      case VIP_BUFFER_FORMAT_UINT16: quantize_bulk(src, reinterpret_cast<uint16_t*>(dst), elements, q); break;
      default: for (uint64_t i = 0; i < elements; ++i) write_integer(dst + i * native, fmt, quantize(src[i], port.params)); break;
    }
  } else if ((t.dtype == kUint8 && fmt == VIP_BUFFER_FORMAT_UINT8) || (t.dtype == kInt8 && (fmt == VIP_BUFFER_FORMAT_INT8 || fmt == VIP_BUFFER_FORMAT_CHAR)) ||
             (t.dtype == kInt16 && fmt == VIP_BUFFER_FORMAT_INT16) || (t.dtype == kFloat16 && fmt == VIP_BUFFER_FORMAT_FP16)) {
    if (t.raw_data.size() != port.bytes) { error = "raw input payload size mismatch for " + port.name; return false; }
    std::memcpy(dst, t.raw_data.data(), port.bytes);  // already in the network's native format: pass through
  } else {
    error = "input " + port.name + ": dtype " + std::to_string(t.dtype) + " cannot feed this NBG input (send FLOAT, or its native integer format)";
    return false;
  }
  if (vip_flush_buffer(port.buffer, VIP_BUFFER_OPER_TYPE_FLUSH) != VIP_SUCCESS) { error = "input cache flush failed"; return false; }
  return true;
}

void read_output(const Port& port, Tensor& out, std::vector<float>& spare) {
  vip_flush_buffer(port.buffer, VIP_BUFFER_OPER_TYPE_INVALIDATE);
  const uint8_t* src = port.host;
  const vip_enum fmt = port.params.data_format;
  const size_t native = format_bytes(fmt);
  out.dtype = kFloat;
  for (int k = static_cast<int>(port.params.num_of_dims) - 1; k >= 0; --k) out.shape.push_back(port.params.sizes[k]);
  out.data = std::move(spare);
  out.data.resize(static_cast<size_t>(port.elements));  // no-op (and no page faults) when the recycled vector is already this size
  const QuantParams q = quant_params(port.params);
  float* dst = out.data.data();
  const uint64_t n = port.elements;
  switch (fmt) {
    case VIP_BUFFER_FORMAT_FP32: std::memcpy(dst, src, n * 4); break;
    case VIP_BUFFER_FORMAT_FP16: for (uint64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); dst[i] = half_to_float(h); } break;
    case VIP_BUFFER_FORMAT_BFP16: for (uint64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); uint32_t b = static_cast<uint32_t>(h) << 16; std::memcpy(&dst[i], &b, 4); } break;
    case VIP_BUFFER_FORMAT_UINT8: case VIP_BUFFER_FORMAT_BOOL8: dequantize_bulk(src, dst, n, q); break;
    case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_CHAR: dequantize_bulk(reinterpret_cast<const int8_t*>(src), dst, n, q); break;
    case VIP_BUFFER_FORMAT_INT16: dequantize_bulk(reinterpret_cast<const int16_t*>(src), dst, n, q); break;
    case VIP_BUFFER_FORMAT_UINT16: dequantize_bulk(reinterpret_cast<const uint16_t*>(src), dst, n, q); break;
    default: for (uint64_t i = 0; i < n; ++i) dst[i] = static_cast<float>(dequantize(read_integer(src + i * native, fmt), port.params)); break;
  }
}

void add_profile(Response& r, const Request& req, const char* name, uint64_t begin, uint64_t duration, const std::string& detail) {
  if (req.profiling != ProfilingLevel::Off) r.profile.push_back(ProfileEvent{name, "viplite", begin, duration, detail});
}

std::string manifest() {
  vip_uint32_t cid = 0, devices = 0;
  vip_query_hardware(VIP_QUERY_HW_PROP_CID, sizeof(cid), &cid);
  vip_query_hardware(VIP_QUERY_HW_PROP_DEVICE_COUNT, sizeof(devices), &devices);
  std::ostringstream m;
  m << "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\",\"runner_id\":\"viplite-runner\",\"ready\":true,"
       "\"graph_execution\":false,\"supported_ops\":[\"load_compiled\",\"run_compiled\"],"
       "\"supported_dtypes\":[\"FLOAT\",\"UINT8\",\"INT8\",\"INT16\",\"FLOAT16\"],\"profiling\":true,"
       "\"artifact\":{\"format\":\"nbg\"},\"hardware\":{\"cid\":\"0x"
    << std::hex << cid << std::dec << "\",\"devices\":" << devices << ",\"driver_version\":\"0x" << std::hex << vip_get_version() << "\"}}";
  return m.str();
}

Response execute(const Request& request) {
  Response response;
  response.request_id = request.request_id;
  const uint64_t t0 = now_us();
  if (request.op == "capabilities") {
    response.ok = true;
    response.artifact_id = "viplite-runner";
    response.manifest = manifest();
    return response;
  }
  if (request.op != "load_compiled" && request.op != "run_compiled") {
    response.error = "viplite runner accepts capabilities/load_compiled/run_compiled";
    return response;
  }
  std::shared_ptr<Net> net_ptr = get_net(request.artifact_id);
  if (!request.artifact.empty() || request.op == "load_compiled" || !net_ptr) {
    if (!load(request.artifact_id, request.artifact, response)) return response;
    add_profile(response, request, "viplite_load", 0, now_us() - t0, request.artifact_id);
    net_ptr = get_net(request.artifact_id);
  }
  if (request.op == "load_compiled") { response.ok = true; response.artifact_id = request.artifact_id; return response; }

  Net& net = *net_ptr;
  const size_t n_in = net.slots[0]->inputs.size(), n_out = net.slots[0]->outputs.size();
  if (request.inputs.size() != n_in) {
    response.error = "input count " + std::to_string(request.inputs.size()) + " differs from the NBG's " + std::to_string(n_in);
    return response;
  }
  // Take a free slot (waits while both are in use); the guard hands it back, and wakes a waiter, however this function exits.
  Slot* slot = nullptr;
  {
    std::unique_lock<std::mutex> lk(net.mu);
    net.cv.wait(lk, [&] { return !net.slots[0]->busy || !net.slots[1]->busy; });
    slot = net.slots[0]->busy ? net.slots[1].get() : net.slots[0].get();
    slot->busy = true;
  }
  struct Release {
    Net& n; Slot* s;
    ~Release() { { std::lock_guard<std::mutex> lk(n.mu); s->busy = false; } n.cv.notify_one(); }
  } release{net, slot};

  const uint64_t fill_begin = now_us() - t0;
  for (size_t i = 0; i < n_in; ++i)
    if (!fill_input(slot->inputs[i], request.inputs[i], response.error)) return response;
  uint64_t run_begin, run_end, bind_us = 0;
  vip_inference_profile_t hw{};
  bool have_hw = false;
  vip_uint32_t layers = 0;
  {
    std::lock_guard<std::mutex> npu(g_npu);  // only the NPU run itself is serialized; conversions on other slots proceed meanwhile
    const uint64_t bind_begin = now_us() - t0;
    if (!bind(net, *slot, response.error)) return response;
    bind_us = now_us() - t0 - bind_begin;
    run_begin = now_us() - t0;
    const vip_status_e st = vip_run_network(net.network);
    run_end = now_us() - t0;
    if (st != VIP_SUCCESS) { response.error = "vip_run_network failed (" + std::to_string(st) + ")"; return response; }
    if (request.profiling != ProfilingLevel::Off) {
      have_hw = vip_query_network(net.network, VIP_NETWORK_PROP_PROFILING, &hw) == VIP_SUCCESS;
      vip_query_network(net.network, VIP_NETWORK_PROP_LAYER_COUNT, &layers);
    }
  }
  std::vector<std::vector<float>> spare;
  {
    std::lock_guard<std::mutex> lk(net.mu);
    if (!net.spare_pool.empty()) { spare = std::move(net.spare_pool.back()); net.spare_pool.pop_back(); }
  }
  spare.resize(n_out);
  response.outputs.resize(n_out);
  for (size_t i = 0; i < n_out; ++i) read_output(slot->outputs[i], response.outputs[i], spare[i]);
  add_profile(response, request, "viplite_run", run_begin, run_end - run_begin, "NPU execution (wall, around vip_run_network)");
  if (have_hw) {
    // The driver's own counters for this run: hardware inference time and NPU cycles (the ratio is the effective NPU clock).
    // They are whole-network figures; VIPLite exposes no per-layer timing.
    std::ostringstream d;
    d << "cycles=" << hw.total_cycle << " layers=" << layers;
    if (hw.inference_time) d << " clock_mhz=" << static_cast<double>(hw.total_cycle) / hw.inference_time;
    add_profile(response, request, "viplite_hw", run_begin, hw.inference_time, d.str());
  }
  if (request.profiling == ProfilingLevel::Detailed) {
    add_profile(response, request, "viplite_input", fill_begin, run_begin - fill_begin, "quantize + upload inputs, and waiting for the NPU lock");
    add_profile(response, request, "viplite_bind", run_begin - bind_us, bind_us, "vip_set_input/output to switch slots");
    add_profile(response, request, "viplite_output", run_end, now_us() - t0 - run_end, "dequantize outputs");
  }
  response.ok = true;
  return response;
}

// Take the output vectors back from a response that has been sent (or discarded) so the next run on that network reuses them.
void recycle(const std::string& artifact_id, Response& response) {
  std::shared_ptr<Net> net = get_net(artifact_id);
  if (!net || response.outputs.size() != net->slots[0]->outputs.size()) return;
  std::vector<std::vector<float>> set(response.outputs.size());
  for (size_t i = 0; i < set.size(); ++i) set[i] = std::move(response.outputs[i].data);
  std::lock_guard<std::mutex> lk(net->mu);
  if (net->spare_pool.size() < 4) net->spare_pool.push_back(std::move(set));
}

// A client hands this worker NBG machine code for the NPU, so unlike the shared listen_tcp (which binds every interface) it listens on
// loopback by default; forward it with `adb forward` / an SSH tunnel, or pass --host 0.0.0.0 to expose it on a trusted network.
int listen_on(const std::string& host, uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1 || ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) || ::listen(fd, 16)) {
    ::close(fd);
    return -1;
  }
  return fd;
}

// --bench FILE.nb [ITERS] [--threads N]: load a network, feed it random inputs, run it and print timing and an output summary. This is
// the smoke test for a fresh device: it needs no client, compiler or network. With N > 1, N callers run concurrently (as N pipelined
// clients would), which exercises the conversion/NPU overlap and reports throughput; the outputs of every call must match the first.
int bench(const char* path, int iters, int threads) {
  std::ifstream in(path, std::ios::binary);
  std::vector<uint8_t> nb{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (nb.empty()) { std::cerr << "cannot read " << path << '\n'; return 1; }
  Response loaded;
  const uint64_t l0 = now_us();
  if (!load("bench", nb, loaded)) { std::cerr << "load: " << loaded.error << '\n'; return 1; }
  std::cout << "load+prepare: " << (now_us() - l0) / 1000.0 << " ms\n";
  Request req;
  req.op = "run_compiled";
  req.artifact_id = "bench";
  req.profiling = ProfilingLevel::Detailed;
  std::mt19937 rng(1);
  std::shared_ptr<Net> net = get_net("bench");
  for (const Port& p : net->slots[0]->inputs) {
    Tensor t;
    t.dtype = kFloat;
    for (int k = static_cast<int>(p.params.num_of_dims) - 1; k >= 0; --k) t.shape.push_back(p.params.sizes[k]);
    t.data.resize(static_cast<size_t>(p.elements));
    std::uniform_real_distribution<float> d(0.f, 1.f);
    for (float& v : t.data) v = d(rng);
    std::cout << "input  " << p.name << " fmt=" << p.params.data_format << " quant=" << p.params.quant_format << " dims=";
    for (int64_t s : t.shape) std::cout << s << ' ';
    std::cout << '\n';
    req.inputs.push_back(std::move(t));
  }
  // Two warm-up calls (one per slot), then `iters` timed ones split over the callers; per-phase medians come from the responses' own
  // profile events.
  Response reference = execute(req);
  if (!reference.ok) { std::cerr << "run: " << reference.error << '\n'; return 1; }
  { Response w = execute(req); if (!w.ok) { std::cerr << "run: " << w.error << '\n'; return 1; } }
  std::map<std::string, std::vector<double>> phases;
  std::vector<double> ms;
  std::mutex result_mu;
  std::atomic<int> mismatches{0}, checked{0};
  Response last;
  const int per_thread = std::max(1, iters / threads);
  const uint64_t wall0 = now_us();
  auto worker = [&](int tid) {
    Response mine;
    for (int i = 0; i < per_thread; ++i) {
      const uint64_t t0 = now_us();
      mine = execute(req);
      const double dt = (now_us() - t0) / 1000.0;
      if (!mine.ok) { std::cerr << "run: " << mine.error << '\n'; std::exit(1); }
      {
        std::lock_guard<std::mutex> lk(result_mu);
        ms.push_back(dt);
        for (const ProfileEvent& e : mine.profile) phases[e.name].push_back(e.duration_us / 1000.0);
      }
      if (checked++ < 40) {  // outside the timed region
        bool same = mine.outputs.size() == reference.outputs.size();
        for (size_t o = 0; same && o < mine.outputs.size(); ++o) same = mine.outputs[o].data == reference.outputs[o].data;
        if (!same) ++mismatches;
      }
      if (i + 1 < per_thread) recycle("bench", mine);  // as the server does after sending a response
    }
    if (tid == 0) { std::lock_guard<std::mutex> lk(result_mu); last = std::move(mine); }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
  for (auto& t : pool) t.join();
  const double wall_s = (now_us() - wall0) / 1e6;
  auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  std::sort(ms.begin(), ms.end());
  std::cout << "run (incl. quantize/IO) median " << median(ms) << " ms, min " << ms.front() << " ms, max " << ms.back() << " ms over " << ms.size() << " iters\n";
  if (threads > 1)
    std::cout << "throughput with " << threads << " concurrent callers: " << ms.size() / wall_s << " calls/s (" << wall_s * 1000.0 / ms.size() << " ms per call)\n";
  std::cout << "outputs identical to the first call across slots: " << (mismatches ? "NO (" + std::to_string(mismatches.load()) + " mismatches)" : "yes (" + std::to_string(std::min(checked.load(), 40)) + " calls checked)") << '\n';
  for (const ProfileEvent& e : last.profile) if (e.name == "viplite_hw") std::cout << "  hw counters: " << e.detail << '\n';
  for (const auto& [name, v] : phases)
    std::cout << "  " << name << ": median " << median(v) << " ms, min " << *std::min_element(v.begin(), v.end()) << " ms\n";
  for (size_t i = 0; i < last.outputs.size(); ++i) {
    const Tensor& o = last.outputs[i];
    double sum = 0; float lo = INFINITY, hi = -INFINITY;
    for (float v : o.data) { sum += v; lo = std::min(lo, v); hi = std::max(hi, v); }
    std::cout << "output " << net->slots[0]->outputs[i].name << " dims=";
    for (int64_t s : o.shape) std::cout << s << ' ';
    std::cout << " min=" << lo << " max=" << hi << " mean=" << sum / std::max<size_t>(1, o.data.size()) << '\n';
  }
  return mismatches ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  uint16_t port = 39503;
  std::string host = "127.0.0.1";
  int fscale = 0;
  const char* bench_path = nullptr;
  int bench_iters = 10, bench_threads = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) host = argv[++i];
    else if (a == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::stoul(argv[++i]));
    else if (a == "--fscale" && i + 1 < argc) fscale = std::atoi(argv[++i]);
    else if (a == "--threads" && i + 1 < argc) bench_threads = std::max(1, std::atoi(argv[++i]));
    else if (a == "--cache-dir" && i + 1 < argc) g_cache_dir = argv[++i];
    else if (a == "--bench" && i + 1 < argc) { bench_path = argv[++i]; if (i + 1 < argc && argv[i + 1][0] != '-') bench_iters = std::atoi(argv[++i]); }
    else if (a == "--help") {
      std::cout << "usage: onnx-remote-viplite-worker [--host ADDR (default 127.0.0.1)] [--port PORT] [--cache-dir DIR] [--fscale 1-100] | --bench FILE.nb [ITERS] [--threads N]\n";
      return 0;
    } else { std::cerr << "unknown argument: " << a << '\n'; return 2; }
  }
  std::signal(SIGPIPE, SIG_IGN);
  if (vip_status_e st = vip_init(); st != VIP_SUCCESS) { std::cerr << "vip_init failed: " << st << " (is /dev/vipcore accessible?)\n"; return 1; }
  if (fscale) {  // scale the NPU clock (percent of full); useful to tell compute-bound from memory-bound networks
    vip_power_frequency_t f{static_cast<vip_uint8_t>(fscale)};
    const vip_status_e st = vip_power_management(0, VIP_POWER_PROPERTY_SET_FREQUENCY, &f);
    std::cerr << "NPU clock scale " << fscale << "%: " << (st == VIP_SUCCESS ? "ok" : "failed " + std::to_string(st)) << '\n';
  }
  int rc = 0;
  if (bench_path) rc = bench(bench_path, std::max(1, bench_iters), bench_threads);
  else {
    int listener = listen_on(host, port);
    if (listener < 0) { std::cerr << "listen failed: " << std::strerror(errno) << '\n'; vip_destroy(); return 1; }
    std::cerr << "onnx-remote-viplite-worker listening on " << host << ":" << port << '\n';
    // One thread per connection (the transport is one request per socket), so several pipelined clients overlap their conversions with
    // each other's NPU runs; the g_npu lock keeps the NPU itself to one run at a time. The cap bounds memory held by in-flight tensors.
    std::atomic<int> in_flight{0};
    constexpr int kMaxInFlight = 8;
    for (;;) {
      int fd = accept_tcp(listener);
      if (fd < 0) continue;
      if (in_flight >= kMaxInFlight) {  // shed load instead of queueing unbounded multi-MB requests
        Response busy; busy.error = "viplite runner busy (too many concurrent requests)"; std::string e;
        send_response(fd, busy, e); close_socket(fd);
        continue;
      }
      ++in_flight;
      std::thread([fd, &in_flight] {
        Request request; Response response; std::string error;
        if (!receive_request(fd, request, error)) response.error = error;
        else response = execute(request);
        if (!send_response(fd, response, error)) std::cerr << "response failed: " << error << '\n';
        close_socket(fd);
        if (response.ok && request.op == "run_compiled") recycle(request.artifact_id, response);
        --in_flight;
      }).detach();
    }
  }
  g_nets.clear();
  vip_destroy();
  return rc;
}
