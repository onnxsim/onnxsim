#include "remote_transport.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <limits>

namespace onnx_remote {
namespace {

constexpr uint32_t kMagic = 0x4f525452u;  // ORTR
constexpr uint16_t kVersion = 2;
constexpr uint16_t kRun = 1;
constexpr uint16_t kOk = 2;
constexpr uint16_t kError = 3;

uint64_t ntoh64(uint64_t x) {
  return (static_cast<uint64_t>(ntohl(static_cast<uint32_t>(x))) << 32) |
         ntohl(static_cast<uint32_t>(x >> 32));
}
uint64_t hton64(uint64_t x) { return ntoh64(x); }

bool read_all(int fd, void* p, size_t n) {
  auto* out = static_cast<char*>(p);
  while (n) {
    ssize_t got = ::recv(fd, out, n, 0);
    if (got <= 0) return false;
    out += got;
    n -= static_cast<size_t>(got);
  }
  return true;
}
bool write_all(int fd, const void* p, size_t n) {
  auto* in = static_cast<const char*>(p);
  while (n) {
    ssize_t put = ::send(fd, in, n, MSG_NOSIGNAL);
    if (put <= 0) return false;
    in += put;
    n -= static_cast<size_t>(put);
  }
  return true;
}
bool take_u32(const std::vector<char>& b, size_t& at, uint32_t& out) {
  if (at + 4 > b.size()) return false;
  uint32_t v;
  std::memcpy(&v, b.data() + at, 4);
  at += 4; out = ntohl(v); return true;
}
bool take_u64(const std::vector<char>& b, size_t& at, uint64_t& out) {
  if (at + 8 > b.size()) return false;
  uint64_t v;
  std::memcpy(&v, b.data() + at, 8);
  at += 8; out = ntoh64(v); return true;
}
void put_u32(std::vector<char>& b, uint32_t v) {
  v = htonl(v); auto* p = reinterpret_cast<char*>(&v); b.insert(b.end(), p, p + 4);
}
void put_u64(std::vector<char>& b, uint64_t v) {
  v = hton64(v); auto* p = reinterpret_cast<char*>(&v); b.insert(b.end(), p, p + 8);
}
void put_string(std::vector<char>& b, const std::string& value) {
  put_u32(b, static_cast<uint32_t>(value.size()));
  b.insert(b.end(), value.begin(), value.end());
}
bool take_string(const std::vector<char>& b, size_t& at, uint32_t max_bytes,
                 std::string& out) {
  uint32_t n;
  if (!take_u32(b, at, n) || n > max_bytes || at + n > b.size()) return false;
  out.assign(b.data() + at, n);
  at += n;
  return true;
}
bool checked_tensor(const Tensor& t, std::string& error) {
  if (t.shape.size() > kMaxRank) { error = "tensor rank exceeds limit"; return false; }
  uint64_t elements = 1;
  for (int64_t d : t.shape) {
    if (d <= 0 || static_cast<uint64_t>(d) > kMaxTensorBytes ||
        elements > kMaxTensorBytes / static_cast<uint64_t>(d)) {
      error = "invalid tensor shape"; return false;
    }
    elements *= static_cast<uint64_t>(d);
  }
  if (elements * sizeof(float) != t.data.size() * sizeof(float)) {
    error = "tensor shape/data size mismatch"; return false;
  }
  if (elements * sizeof(float) > kMaxTensorBytes) { error = "tensor too large"; return false; }
  return true;
}
bool encode_tensors(const std::vector<Tensor>& ts, std::vector<char>& b, std::string& error) {
  if (ts.size() > kMaxTensors) { error = "too many tensors"; return false; }
  put_u32(b, static_cast<uint32_t>(ts.size()));
  for (const Tensor& t : ts) {
    if (!checked_tensor(t, error)) return false;
    put_u32(b, static_cast<uint32_t>(t.shape.size()));
    for (int64_t d : t.shape) put_u64(b, static_cast<uint64_t>(d));
    put_u64(b, static_cast<uint64_t>(t.data.size()));
    const auto* p = reinterpret_cast<const char*>(t.data.data());
    b.insert(b.end(), p, p + t.data.size() * sizeof(float));
  }
  return b.size() <= kMaxMessageBytes;
}
bool decode_tensors(const std::vector<char>& b, size_t& at, std::vector<Tensor>& ts, std::string& error) {
  uint32_t count;
  if (!take_u32(b, at, count) || count > kMaxTensors) { error = "invalid tensor count"; return false; }
  ts.resize(count);
  for (Tensor& t : ts) {
    uint32_t rank; uint64_t n;
    if (!take_u32(b, at, rank) || rank > kMaxRank) { error = "invalid tensor rank"; return false; }
    t.shape.resize(rank); uint64_t elements = 1;
    for (auto& d : t.shape) {
      uint64_t ud;
      if (!take_u64(b, at, ud) || ud == 0 || ud > kMaxTensorBytes || elements > kMaxTensorBytes / ud) {
        error = "invalid tensor dimension"; return false;
      }
      d = static_cast<int64_t>(ud); elements *= ud;
    }
    if (!take_u64(b, at, n) || n != elements || n * sizeof(float) > kMaxTensorBytes ||
        at + n * sizeof(float) > b.size()) { error = "invalid tensor payload"; return false; }
    t.data.resize(static_cast<size_t>(n));
    std::memcpy(t.data.data(), b.data() + at, static_cast<size_t>(n) * sizeof(float));
    at += static_cast<size_t>(n) * sizeof(float);
  }
  return true;
}
bool encode_profile(const std::vector<ProfileEvent>& events,
                    std::vector<char>& b, std::string& error) {
  if (events.size() > kMaxProfileEvents) {
    error = "too many profile events";
    return false;
  }
  put_u32(b, static_cast<uint32_t>(events.size()));
  for (const ProfileEvent& event : events) {
    if (event.name.empty() || event.name.size() > kMaxProfileNameBytes ||
        event.category.size() > kMaxProfileNameBytes ||
        event.detail.size() > kMaxProfileDetailBytes) {
      error = "invalid profile event string";
      return false;
    }
    put_string(b, event.name);
    put_string(b, event.category);
    put_u64(b, event.start_us);
    put_u64(b, event.duration_us);
    put_string(b, event.detail);
  }
  return b.size() <= kMaxMessageBytes;
}
bool decode_profile(const std::vector<char>& b, size_t& at,
                    std::vector<ProfileEvent>& events, std::string& error) {
  uint32_t count;
  if (!take_u32(b, at, count) || count > kMaxProfileEvents) {
    error = "invalid profile event count";
    return false;
  }
  events.resize(count);
  for (ProfileEvent& event : events) {
    if (!take_string(b, at, kMaxProfileNameBytes, event.name) ||
        event.name.empty() ||
        !take_string(b, at, kMaxProfileNameBytes, event.category) ||
        !take_u64(b, at, event.start_us) ||
        !take_u64(b, at, event.duration_us) ||
        !take_string(b, at, kMaxProfileDetailBytes, event.detail)) {
      error = "invalid profile event";
      return false;
    }
  }
  return true;
}
bool encode_request_bytes(const Request& request, std::vector<char>& b,
                          std::string& error) {
  if (request.op.empty() || request.op.size() > kMaxOpBytes) {
    error = "invalid operation";
    return false;
  }
  put_u32(b, static_cast<uint32_t>(request.op.size()));
  b.insert(b.end(), request.op.begin(), request.op.end());
  put_u64(b, request.model.size());
  b.insert(b.end(), request.model.begin(), request.model.end());
  put_u32(b, static_cast<uint32_t>(request.profiling));
  if (!encode_tensors(request.inputs, b, error)) return false;
  return b.size() <= kMaxMessageBytes;
}
bool decode_request_bytes(const std::vector<char>& b, Request& request,
                          std::string& error) {
  size_t at = 0;
  uint32_t op_len;
  if (!take_u32(b, at, op_len) || op_len == 0 || op_len > kMaxOpBytes ||
      at + op_len > b.size()) {
    error = "invalid operation";
    return false;
  }
  request.op.assign(b.data() + at, op_len);
  at += op_len;
  uint64_t model_len;
  if (!take_u64(b, at, model_len) || model_len > kMaxMessageBytes ||
      at + model_len > b.size()) {
    error = "invalid model payload";
    return false;
  }
  request.model.resize(static_cast<size_t>(model_len));
  if (model_len) {
    std::memcpy(request.model.data(), b.data() + at,
                static_cast<size_t>(model_len));
  }
  at += static_cast<size_t>(model_len);
  uint32_t profiling;
  if (!take_u32(b, at, profiling) ||
      profiling > static_cast<uint32_t>(ProfilingLevel::Detailed)) {
    error = "invalid profiling level";
    return false;
  }
  request.profiling = static_cast<ProfilingLevel>(profiling);
  if (!decode_tensors(b, at, request.inputs, error) || at != b.size()) {
    if (error.empty()) error = "trailing request bytes";
    return false;
  }
  return true;
}
bool encode_response_bytes(const Response& response, std::vector<char>& b,
                           std::string& error) {
  if (!response.ok) {
    if (response.error.size() > kMaxProfileDetailBytes * 4) {
      error = "error response too large";
      return false;
    }
    put_u32(b, static_cast<uint32_t>(response.error.size()));
    b.insert(b.end(), response.error.begin(), response.error.end());
    return true;
  }
  return encode_tensors(response.outputs, b, error) &&
         encode_profile(response.profile, b, error);
}
bool decode_response_bytes(const std::vector<char>& b, bool ok,
                           Response& response, std::string& error) {
  size_t at = 0;
  if (!ok) {
    uint32_t len;
    if (!take_u32(b, at, len) || len > kMaxProfileDetailBytes * 4 ||
        at + len != b.size()) {
      error = "invalid error response";
      return false;
    }
    response.ok = false;
    response.error.assign(b.data() + at, len);
    return true;
  }
  response.ok = true;
  response.error.clear();
  return decode_tensors(b, at, response.outputs, error) &&
         decode_profile(b, at, response.profile, error) && at == b.size();
}
bool send_message(int fd, uint16_t kind, const std::vector<char>& payload, std::string& error) {
  if (payload.size() > kMaxMessageBytes) { error = "message too large"; return false; }
  uint32_t magic = htonl(kMagic); uint16_t version = htons(kVersion), k = htons(kind);
  uint64_t n = hton64(payload.size());
  if (!write_all(fd, &magic, 4) || !write_all(fd, &version, 2) || !write_all(fd, &k, 2) ||
      !write_all(fd, &n, 8) || (!payload.empty() && !write_all(fd, payload.data(), payload.size()))) {
    error = std::strerror(errno); return false;
  }
  return true;
}

}  // namespace

int listen_tcp(uint16_t port, int backlog) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0); if (fd < 0) return -1;
  int one = 1; ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons(port);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) || ::listen(fd, backlog)) { ::close(fd); return -1; }
  return fd;
}
int accept_tcp(int listener) { return ::accept(listener, nullptr, nullptr); }
int connect_tcp(const std::string& host, uint16_t port) {
  addrinfo hints{}; hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC;
  addrinfo* result = nullptr; std::string service = std::to_string(port);
  if (::getaddrinfo(host.c_str(), service.c_str(), &hints, &result)) return -1;
  int fd = -1;
  for (addrinfo* p = result; p; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd >= 0 && !::connect(fd, p->ai_addr, p->ai_addrlen)) break;
    if (fd >= 0) ::close(fd); fd = -1;
  }
  ::freeaddrinfo(result); return fd;
}
void close_socket(int fd) { if (fd >= 0) ::close(fd); }

bool receive_request(int fd, Request& request, std::string& error) {
  uint32_t magic; uint16_t version, kind; uint64_t n;
  if (!read_all(fd, &magic, 4) || !read_all(fd, &version, 2) || !read_all(fd, &kind, 2) || !read_all(fd, &n, 8)) {
    error = "truncated message header"; return false;
  }
  if (ntohl(magic) != kMagic || ntohs(version) != kVersion || ntohs(kind) != kRun || ntoh64(n) > kMaxMessageBytes) {
    error = "invalid message header"; return false;
  }
  std::vector<char> b(static_cast<size_t>(ntoh64(n))); if (!b.empty() && !read_all(fd, b.data(), b.size())) { error = "truncated message"; return false; }
  return decode_request_bytes(b, request, error);
}
bool send_response(int fd, const Response& response, std::string& error) {
  std::vector<char> b;
  if (!encode_response_bytes(response, b, error)) return false;
  if (!response.ok) return send_message(fd, kError, b, error);
  return send_message(fd, kOk, b, error);
}

bool send_request(int fd, const Request& request, std::string& error) {
  std::vector<char> b;
  if (!encode_request_bytes(request, b, error)) return false;
  return send_message(fd, kRun, b, error);
}

bool receive_response(int fd, Response& response, std::string& error) {
  uint32_t magic; uint16_t version, kind; uint64_t n;
  if (!read_all(fd, &magic, 4) || !read_all(fd, &version, 2) || !read_all(fd, &kind, 2) || !read_all(fd, &n, 8)) {
    error = "truncated response header"; return false;
  }
  n = ntoh64(n);
  if (ntohl(magic) != kMagic || ntohs(version) != kVersion || (ntohs(kind) != kOk && ntohs(kind) != kError) || n > kMaxMessageBytes) {
    error = "invalid response header"; return false;
  }
  std::vector<char> b(static_cast<size_t>(n));
  if (!b.empty() && !read_all(fd, b.data(), b.size())) { error = "truncated response"; return false; }
  return decode_response_bytes(b, ntohs(kind) == kOk, response, error);
}

bool encode_request_payload(const Request& request, std::vector<uint8_t>& payload,
                            std::string& error) {
  std::vector<char> bytes;
  if (!encode_request_bytes(request, bytes, error)) return false;
  payload.clear();
  payload.assign(bytes.begin(), bytes.end());
  return true;
}

bool decode_request_payload(const uint8_t* data, size_t size, Request& request,
                            std::string& error) {
  if (size > kMaxMessageBytes || (size != 0 && data == nullptr)) {
    error = "invalid request payload";
    return false;
  }
  std::vector<char> bytes;
  if (size != 0) {
    bytes.assign(reinterpret_cast<const char*>(data),
                 reinterpret_cast<const char*>(data) + size);
  }
  return decode_request_bytes(bytes, request, error);
}

bool encode_response_payload(const Response& response,
                             std::vector<uint8_t>& payload, std::string& error) {
  std::vector<char> bytes;
  if (!encode_response_bytes(response, bytes, error)) return false;
  payload.clear();
  payload.reserve(bytes.size() + 1);
  payload.push_back(response.ok ? 1 : 0);
  payload.insert(payload.end(), bytes.begin(), bytes.end());
  return true;
}

bool decode_response_payload(const uint8_t* data, size_t size,
                             Response& response, std::string& error) {
  if (size > kMaxMessageBytes || (size != 0 && data == nullptr)) {
    error = "invalid response payload";
    return false;
  }
  std::vector<char> bytes;
  if (size != 0) {
    bytes.assign(reinterpret_cast<const char*>(data),
                 reinterpret_cast<const char*>(data) + size);
  }
  // Message-oriented adapters carry the response status as the first byte:
  // 1 = OK, 0 = error, followed by the normal response payload.
  if (bytes.empty()) {
    error = "empty response payload";
    return false;
  }
  const bool ok = bytes[0] != 0;
  bytes.erase(bytes.begin());
  return decode_response_bytes(bytes, ok, response, error);
}

}  // namespace onnx_remote
