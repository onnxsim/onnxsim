#!/usr/bin/env bash
# Cross-build the onnxsim CPython extension for s390x (big endian) Linux from an
# x86_64 host, so the test suite can be run big-endian under qemu-user.
#
# Same shape as build_windows_wheel.sh: a host protoc runs ONNX's code
# generation, abseil + protobuf are cross-built for the target, and the
# extension links against them. The target sysroot is an Ubuntu s390x rootfs
# (see scripts/cross/README.md) which also supplies CPython, numpy and onnx for
# the actual test run.
#
# Everything is compiled by the *host* toolchain (s390x-linux-gnu-g++), so the
# build runs at native speed; only the resulting binaries execute under qemu.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-${REPO_ROOT}/.cross-build-s390x}"
SYSROOT="${SYSROOT:-/rootfs-s390x}"
JOBS="${JOBS:-$(nproc)}"
PYVER="${PYVER:-3.12}"

ONNX_DIR="${REPO_ROOT}/third_party/onnx"
SBOM="${ONNX_DIR}/sbom.cdx.json"

sbom_ver() {  # sbom_ver <component-name>
  python3 - "$SBOM" "$1" <<'PY'
import json, sys
sbom, name = sys.argv[1], sys.argv[2]
data = json.load(open(sbom))
for c in data.get("components", []):
    if c.get("name") == name:
        print(c["version"]); break
else:
    sys.exit(f"component {name} not found in SBOM")
PY
}

PROTOBUF_VER="${PROTOBUF_VER:-$(sbom_ver protobuf)}"
ABSL_VER="${ABSL_VER:-$(sbom_ver abseil-cpp)}"

DEPS_HOST="${WORK}/deps-host"      # host protobuf install -> protoc
DEPS_TARGET="${WORK}/deps-target"  # target (s390x) absl + protobuf install
DL="${WORK}/dl"
mkdir -p "${DL}"

fetch() {  # fetch <url> <dest>
  [[ -s "$2" ]] && return 0
  echo "-- fetching $1"
  curl -fsSL --retry 3 -o "$2" "$1"
}

TOOLCHAIN_ARGS=(
  -DCMAKE_TOOLCHAIN_FILE="${REPO_ROOT}/scripts/cross/linux-s390x.toolchain.cmake"
  -DONNXSIM_S390X_SYSROOT="${SYSROOT}"
)

# When sccache is on PATH it launches every compile (host protoc, target
# abseil/protobuf, onnx/onnxsim), so object files survive across runs -- the
# same arrangement build_windows_wheel.sh uses. The workflow provides it via
# mozilla-actions/sccache-action with the GitHub Actions cache backend.
SCCACHE_ARGS=()
if command -v sccache >/dev/null; then
  SCCACHE_ARGS=( -DCMAKE_C_COMPILER_LAUNCHER=sccache
                 -DCMAKE_CXX_COMPILER_LAUNCHER=sccache )
  sccache --start-server >/dev/null 2>&1 || true
  echo "sccache enabled: $(command -v sccache)"
fi

# ---------------------------------------------------------------------------
# 1. Host protoc -- ONNX runs this during code generation. A cross-built
#    s390x protoc could not execute on this host.
# ---------------------------------------------------------------------------
HOST_PROTOC="${DEPS_HOST}/bin/protoc"
if [[ ! -x "${HOST_PROTOC}" ]]; then
  echo "== [1/3] building host abseil ${ABSL_VER} + protobuf ${PROTOBUF_VER} (for protoc) =="
  fetch "https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz" "${DL}/absl.tar.gz"
  fetch "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz" "${DL}/protobuf.tar.gz"

  rm -rf "${WORK}/absl-host-src" "${WORK}/protobuf-host-src"
  mkdir -p "${WORK}/absl-host-src" "${WORK}/protobuf-host-src"
  tar -C "${WORK}/absl-host-src"     --strip-components=1 -xf "${DL}/absl.tar.gz"
  tar -C "${WORK}/protobuf-host-src" --strip-components=1 -xf "${DL}/protobuf.tar.gz"

  cmake -S "${WORK}/absl-host-src" -B "${WORK}/absl-host-build" -G Ninja \
    "${SCCACHE_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DABSL_PROPAGATE_CXX_STD=ON -DABSL_ENABLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX="${DEPS_HOST}"
  cmake --build "${WORK}/absl-host-build" --target install -j "${JOBS}"

  cmake -S "${WORK}/protobuf-host-src" -B "${WORK}/protobuf-host-build" -G Ninja \
    "${SCCACHE_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
    -DCMAKE_PREFIX_PATH="${DEPS_HOST}" -DCMAKE_INSTALL_PREFIX="${DEPS_HOST}"
  cmake --build "${WORK}/protobuf-host-build" --target install -j "${JOBS}"
fi
"${HOST_PROTOC}" --version

# ---------------------------------------------------------------------------
# 2. Target abseil + protobuf -- ONNX links these
# ---------------------------------------------------------------------------
if [[ ! -f "${DEPS_TARGET}/.done" ]]; then
  echo "== [2/3] cross-building abseil ${ABSL_VER} + protobuf ${PROTOBUF_VER} for s390x =="
  fetch "https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz" "${DL}/absl.tar.gz"
  fetch "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz" "${DL}/protobuf.tar.gz"

  rm -rf "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  mkdir -p "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  tar -C "${WORK}/absl-tgt-src"     --strip-components=1 -xf "${DL}/absl.tar.gz"
  tar -C "${WORK}/protobuf-tgt-src" --strip-components=1 -xf "${DL}/protobuf.tar.gz"

  common_target_args=(
    -G Ninja
    "${TOOLCHAIN_ARGS[@]}"
    "${SCCACHE_ARGS[@]}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
  )

  cmake -S "${WORK}/absl-tgt-src" -B "${WORK}/absl-tgt-build" "${common_target_args[@]}" \
    -DABSL_PROPAGATE_CXX_STD=ON -DABSL_ENABLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX="${DEPS_TARGET}"
  cmake --build "${WORK}/absl-tgt-build" --target install -j "${JOBS}"

  cmake -S "${WORK}/protobuf-tgt-src" -B "${WORK}/protobuf-tgt-build" "${common_target_args[@]}" \
    -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
    -DCMAKE_PREFIX_PATH="${DEPS_TARGET}" -DCMAKE_INSTALL_PREFIX="${DEPS_TARGET}"
  cmake --build "${WORK}/protobuf-tgt-build" --target install -j "${JOBS}"
  touch "${DEPS_TARGET}/.done"
fi

# ---------------------------------------------------------------------------
# 3. Cross-build the extension (and the dependency-free C++ unit tests)
# ---------------------------------------------------------------------------
echo "== [3/3] cross-building onnxsim for s390x =="

NANOBIND_CMAKE_DIR="$(python3 -m nanobind --cmake_dir)"
BUILD_DIR="${WORK}/onnxsim-build"
# CMake's FindPython validates the interpreter against the target headers and
# refuses a version mismatch, so the host interpreter must be the same X.Y as
# the target CPython in the sysroot (it only runs ONNX's codegen scripts).
HOSTPY="${HOSTPY:-$(command -v "python${PYVER}")}"
[[ -x "${HOSTPY}" ]] || { echo "need a host python${PYVER} to match the target"; exit 1; }

TGT_PY_INC="${SYSROOT}/usr/include/python${PYVER}"
TGT_PY_LIB="${SYSROOT}/usr/lib/s390x-linux-gnu/libpython${PYVER}.so"
TGT_PY_SABI="${SYSROOT}/usr/lib/s390x-linux-gnu/libpython3.so"

# onnx splits the Python roles when cross-compiling (target dev libs land in the
# Python3 namespace, the host interpreter in Python); onnxsim and onnx-optimizer
# use the Python namespace for target dev libs. Provide hints for both, exactly
# as the Windows cross-build does.
python_args=(
  -DPython_EXECUTABLE="${HOSTPY}"
  -DPython_INCLUDE_DIR="${TGT_PY_INC}"
  -DPython_LIBRARY="${TGT_PY_LIB}"
  -DPython_SABI_LIBRARY="${TGT_PY_SABI}"
  -DPython3_EXECUTABLE="${HOSTPY}"
  -DPython3_INCLUDE_DIR="${TGT_PY_INC}"
  -DPython3_LIBRARY="${TGT_PY_LIB}"
  -DPython3_SABI_LIBRARY="${TGT_PY_SABI}"
)

# Let CTest run the cross-built unit tests under qemu. Set here rather than in
# the toolchain file so it applies only to onnxsim: with an emulator configured,
# CMake also starts *running* try_run checks, which the abseil/protobuf
# configures above have no need to do.
emulator_args=()
if command -v qemu-s390x-static >/dev/null; then
  emulator_args=(
    -DCMAKE_CROSSCOMPILING_EMULATOR="$(command -v qemu-s390x-static);-L;${SYSROOT}"
  )
fi

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja -Wno-dev -Wdeprecated \
  "${TOOLCHAIN_ARGS[@]}" \
  "${SCCACHE_ARGS[@]}" \
  "${emulator_args[@]}" \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="${REPO_ROOT}/scripts/cross/find_python_early.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_BUILD_PYTHON=ON \
  -DONNX_INSTALL=OFF \
  -DONNXSIM_PYTHON=ON \
  -DONNXSIM_BUILTIN_ORT=OFF \
  -DONNXSIM_TESTS=ON \
  -DONNX_USE_LITE_PROTO=OFF \
  -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF \
  -DONNX_CUSTOM_PROTOC_EXECUTABLE="${HOST_PROTOC}" \
  -DCMAKE_PREFIX_PATH="${DEPS_TARGET};${NANOBIND_CMAKE_DIR}" \
  -Dnanobind_DIR="${NANOBIND_CMAKE_DIR}" \
  "${python_args[@]}"

cmake --build "${BUILD_DIR}" --target onnxsim_cpp2py_export -j "${JOBS}"
# onnx's own extension. Not a dependency of onnxsim's -- onnxsim links the onnx
# C++ library, not its Python bindings -- so it has to be asked for by name.
# run_s390x_tests.sh installs it into the rootfs as part of the vendored onnx,
# which the distro's much older python3-onnx is shadowed by.
cmake --build "${BUILD_DIR}" --target onnx_cpp2py_export -j "${JOBS}"
# The dependency-free unit tests (ONNXSIM_TESTS=ON above). dlpack_dtype_test is
# the one that covers the byte-order conversion directly; the rest come along
# because they are cheap and exercise the same cross-built toolchain.
# tensor_pool_dtype_test/tensor_pool_test (safetensors) and
# gguf_dtype_test/tensor_pool_gguf_test (GGUF) are likewise dependency-free
# (each format's own byte-order handling; see tensor_pool.h's "Byte order"
# note and tensor_pool_gguf.cpp's mirror of it) and belong in this list for
# the same reason. ggml_kquant_test (the GGML K-quant dequantization
# ggml_kquant.h implements -- decoded values are real numeric output, not
# just copied bytes, so this is exactly the kind of logic a big-endian run
# needs to check) is dependency-free too, for the same reason.
# tensor_pool_hash_test (TensorPool::ContentHash's BLAKE3 / SHA-256
# backends) is dependency-free too and belongs here for the same reason --
# CMake still registers each of these as a ctest target even if left off
# this list, so omitting one here doesn't skip its test, it makes ctest try
# to exec a binary that was never built: an instant, silent "Failed 0.00
# sec" with no output, indistinguishable at a glance from a real crash.
cmake --build "${BUILD_DIR}" --target sym_expr_test model_metrics_test \
  sym_value_eval_test sym_shape_infer_test dlpack_dtype_test \
  tensor_pool_dtype_test tensor_pool_test tensor_pool_hash_test \
  gguf_dtype_test ggml_kquant_test tensor_pool_gguf_test -j "${JOBS}"
# tensor_pool_bridge_test, tensor_pool_gguf_bridge_test, and
# tensor_pool_archive_test are NOT dependency-free (they exercise the
# onnx::TensorProto <-> TensorPool bridges), but the onnx/onnx-optimizer
# static libraries they need are already built as a side effect of the
# onnxsim_cpp2py_export target above, so they cost only their own link step
# here rather than a second onnx build.
cmake --build "${BUILD_DIR}" --target tensor_pool_bridge_test \
  tensor_pool_gguf_bridge_test tensor_pool_archive_test -j "${JOBS}"

SO="$(find "${BUILD_DIR}" -name 'onnxsim_cpp2py_export*.so' -print -quit)"
[[ -n "${SO}" ]] || { echo "no extension module produced"; exit 1; }
echo "built: ${SO}"
file "${SO}"
