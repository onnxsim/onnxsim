#include "remote_transport.h"

#include <cerrno>
#include <atomic>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace onnx_remote;

namespace {

struct Options {
  uint16_t port = 39502;
  fs::path cache_dir;
  std::string command;
  std::string target = "qnn-htp";
  std::string compiler_id = "unidentified";
};

std::atomic<uint64_t> cache_write_counter{0};

std::string hex_u64(uint64_t value) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result(16, '0');
  for (int i = 15; i >= 0; --i) {
    result[static_cast<size_t>(i)] = kHex[value & 0xf];
    value >>= 4;
  }
  return result;
}

// This is an artifact cache key, not a security digest. Two independent FNV-1a
// streams make accidental collisions sufficiently unlikely for a local cache,
// while keeping this worker dependency-free on small compile hosts.
std::string cache_key(const Request& request, const Options& options) {
  uint64_t a = 1469598103934665603ull;
  uint64_t b = 1099511628211ull;
  auto add = [&](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
      a ^= bytes[i];
      a *= 1099511628211ull;
      b ^= static_cast<uint64_t>(bytes[i]) + 0x9d;
      b *= 14029467366897019727ull;
    }
  };
  add(options.target.data(), options.target.size());
  add(options.compiler_id.data(), options.compiler_id.size());
  add(options.command.data(), options.command.size());
  add(request.model.data(), request.model.size());
  return hex_u64(a) + hex_u64(b);
}

bool read_file(const fs::path& path, std::vector<uint8_t>& bytes, std::string& error) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    error = "cannot open " + path.string();
    return false;
  }
  const auto end = input.tellg();
  if (end < 0 || static_cast<uint64_t>(end) > kMaxArtifactBytes) {
    error = "file is missing or exceeds artifact limit: " + path.string();
    return false;
  }
  bytes.resize(static_cast<size_t>(end));
  input.seekg(0);
  if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), end);
  if (!input && !bytes.empty()) {
    error = "cannot read " + path.string();
    return false;
  }
  return true;
}

bool read_text(const fs::path& path, std::string& text) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::ostringstream stream;
  stream << input.rdbuf();
  text = stream.str();
  return text.size() <= kMaxManifestBytes;
}

bool write_file(const fs::path& path, const std::vector<uint8_t>& bytes,
                std::string& error) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    error = "cannot write " + path.string();
    return false;
  }
  if (!bytes.empty()) output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!output) {
    error = "cannot write " + path.string();
    return false;
  }
  return true;
}

bool publish_cache(const fs::path& artifact_path, const fs::path& manifest_path,
                  const fs::path& complete_path,
                  const std::vector<uint8_t>& artifact,
                  const std::string& manifest, std::string& error) {
  // Never expose a partially written artifact to a second compiler process.
  // The service is currently single-request-at-a-time, but the cache may be
  // shared by multiple service instances on one compile host.
  const auto suffix = ".tmp." + std::to_string(static_cast<long long>(getpid())) +
                      "." + std::to_string(cache_write_counter.fetch_add(1));
  const fs::path artifact_tmp = artifact_path.string() + suffix;
  const fs::path manifest_tmp = manifest_path.string() + suffix;
  const fs::path complete_tmp = complete_path.string() + suffix;
  fs::remove(complete_path);
  if (!write_file(artifact_tmp, artifact, error)) return false;
  {
    std::ofstream output(manifest_tmp, std::ios::binary | std::ios::trunc);
    if (!output) {
      error = "cannot write " + manifest_tmp.string();
      fs::remove(artifact_tmp);
      return false;
    }
    output << manifest;
    if (!output) {
      error = "cannot write " + manifest_tmp.string();
      fs::remove(artifact_tmp);
      fs::remove(manifest_tmp);
      return false;
    }
  }
  std::error_code ec;
  fs::rename(artifact_tmp, artifact_path, ec);
  if (ec) {
    error = "cannot publish " + artifact_path.string() + ": " + ec.message();
    fs::remove(artifact_tmp);
    fs::remove(manifest_tmp);
    fs::remove(complete_tmp);
    return false;
  }
  fs::rename(manifest_tmp, manifest_path, ec);
  if (ec) {
    error = "cannot publish " + manifest_path.string() + ": " + ec.message();
    fs::remove(manifest_tmp);
    fs::remove(complete_tmp);
    return false;
  }
  std::ofstream complete(complete_tmp, std::ios::binary | std::ios::trunc);
  if (!complete) {
    error = "cannot write " + complete_tmp.string();
    fs::remove(complete_tmp);
    return false;
  }
  complete.close();
  fs::rename(complete_tmp, complete_path, ec);
  if (ec) {
    error = "cannot publish " + complete_path.string() + ": " + ec.message();
    fs::remove(complete_tmp);
    return false;
  }
  return true;
}

std::string shell_quote(const std::string& value) {
  std::string result = "'";
  for (char c : value) {
    if (c == '\'') result += "'\\''";
    else result += c;
  }
  return result + "'";
}

std::string replace_all(std::string command, const std::string& token,
                        const std::string& value) {
  size_t position = 0;
  while ((position = command.find(token, position)) != std::string::npos) {
    command.replace(position, token.size(), value);
    position += value.size();
  }
  return command;
}

Response compile(const Request& request, const Options& options) {
  Response response;
  if (request.op != "compile") {
    response.error = "compiler service only accepts op=compile";
    return response;
  }
  if (request.model.empty()) {
    response.error = "compile requires a serialized model";
    return response;
  }

  const std::string key = cache_key(request, options);
  const fs::path artifact_path = options.cache_dir / (key + ".artifact");
  const fs::path manifest_path = options.cache_dir / (key + ".manifest");
  const fs::path complete_path = options.cache_dir / (key + ".complete");
  if (!options.cache_dir.empty() && fs::exists(artifact_path) &&
      fs::exists(manifest_path) && fs::exists(complete_path)) {
    std::string error;
    if (read_file(artifact_path, response.artifact, error) &&
        read_text(manifest_path, response.manifest)) {
      response.ok = true;
      response.artifact_id = key;
      return response;
    }
    response.artifact.clear();
  }

  fs::path work_dir;
  if (!options.command.empty()) {
    std::string pattern = "/tmp/onnxsim-compiler-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    if (mkdtemp(name.data()) == nullptr) {
      response.error = "cannot create compiler work directory";
      return response;
    }
    work_dir = name.data();
  }

  const fs::path input_path = work_dir / "model.onnx";
  const fs::path output_path = work_dir / "compiled.bin";
  const fs::path manifest_output_path = work_dir / "manifest.json";
  std::string error;
  if (options.command.empty()) {
    // Useful for validating transport, caching, and runner integration before
    // the proprietary QNN/QAIRT compiler is installed.
    response.artifact = request.model;
    response.manifest = "{\"backend\":\"passthrough\",\"target\":\"" +
                        options.target + "\",\"compiler_id\":\"" +
                        options.compiler_id + "\"}";
  } else if (!write_file(input_path, request.model, error)) {
    response.error = error;
  } else {
    std::string command = options.command;
    command = replace_all(command, "{input}", shell_quote(input_path.string()));
    command = replace_all(command, "{output}", shell_quote(output_path.string()));
    command = replace_all(command, "{manifest}", shell_quote(manifest_output_path.string()));
    command = replace_all(command, "{target}", shell_quote(options.target));
    const int status = std::system(command.c_str());
    if (status != 0) {
      response.error = "compiler command failed with status " + std::to_string(status);
    } else if (!read_file(output_path, response.artifact, error)) {
      response.error = "compiler did not produce {output}: " + error;
    } else if (!read_text(manifest_output_path, response.manifest)) {
      response.error = "compiler did not produce {manifest} or it exceeds the manifest limit";
    }
  }

  if (!work_dir.empty()) fs::remove_all(work_dir);
  if (!response.error.empty()) return response;
  response.artifact_id = key;
  response.ok = true;
  if (!options.cache_dir.empty()) {
    std::error_code ec;
    fs::create_directories(options.cache_dir, ec);
    if (!ec) publish_cache(artifact_path, manifest_path, complete_path,
                           response.artifact, response.manifest, error);
  }
  return response;
}

bool parse_options(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--port" && i + 1 < argc) {
      options.port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (argument == "--cache-dir" && i + 1 < argc) {
      options.cache_dir = argv[++i];
    } else if (argument == "--command" && i + 1 < argc) {
      options.command = argv[++i];
    } else if (argument == "--target" && i + 1 < argc) {
      options.target = argv[++i];
    } else if (argument == "--compiler-id" && i + 1 < argc) {
      options.compiler_id = argv[++i];
    } else if (argument == "--help") {
      std::cout << "usage: onnx-remote-compiler [--port PORT] [--cache-dir DIR]"
                   " [--target TARGET] [--compiler-id ID] [--command COMMAND]\n"
                   "COMMAND placeholders: {input} {output} {manifest} {target}\n"
                   "Without COMMAND, copies the model as a transport smoke-test artifact.\n";
      return false;
    } else {
      std::cerr << "unknown or incomplete argument: " << argument << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options)) return argc > 1 && std::string(argv[1]) == "--help" ? 0 : 2;
  std::signal(SIGPIPE, SIG_IGN);
  int listener = listen_tcp(options.port);
  if (listener < 0) {
    std::cerr << "listen failed: " << std::strerror(errno) << '\n';
    return 1;
  }
  std::cerr << "onnx-remote-compiler listening on " << options.port
            << " for target " << options.target << '\n';
  for (;;) {
    int fd = accept_tcp(listener);
    if (fd < 0) continue;
    Request request;
    Response response;
    std::string error;
    if (!receive_request(fd, request, error)) response.error = error;
    else response = compile(request, options);
    if (!response.ok && response.error.empty()) response.error = "compile failed";
    send_response(fd, response, error);
    close_socket(fd);
  }
}
