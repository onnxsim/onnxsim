cmake_minimum_required(VERSION 3.22)

# For std::filesystem in onnx optimizer, and std::to_chars (used by onnx's
# text printer) which libc++ only marks available from macOS 13.3 onwards.
# ONNX itself raised its wheel deployment target to 13.3 for the same reason,
# so onnxsim follows to keep building the bundled onnx on macOS.
# Must be a cache variable and be set before project()
# Reference: https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_DEPLOYMENT_TARGET.html
# It can be a normal variable if policy CMP0126 is set to NEW.
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.3 CACHE STRING "Minimum OS X deployment version")

project(onnxsim CXX)

# The BLAKE3 sources backing TensorPool's content hashing (onnxsim/
# tensor_pool_hash.cpp, third_party/blake3/c/*.c) are plain C, unlike
# the rest of this project, and are compiled into the `onnxsim` target in
# EVERY configuration (see ONNXSIM_BLAKE3_SOURCES below) -- unlike the
# ONNXSIM_BUILTIN_ORT path's own enable_language(C) call below, which only
# runs when that option is ON and is skipped under Emscripten (that path
# only ever configures ONNX Runtime, whose own from-source CMake build
# enables the C/ASM it needs on its own). BLAKE3 has no such transitive
# helper, and Emscripten does NOT enable C on its own just because CXX was
# requested in project() above -- confirmed by the WASM build failing with
# "CMAKE_C_COMPILE_OBJECT ... not set" without this. So enable C
# unconditionally, including under Emscripten (emcc compiles C same as
# C++), before any target needs it. A second enable_language(C) call below
# is a harmless no-op.
enable_language(C)

set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

option(ONNXSIM_PYTHON "" OFF)
option(ONNXSIM_BUILTIN_ORT "" ON)
option(ONNXSIM_PREBUILT_ORT "Link a prebuilt ONNX Runtime release instead of building it from source" OFF)
option(ONNXSIM_WASM_NODE "For node (enable NODERAWFS etc.)" OFF)
option(ONNXSIM_C_API "Build the C ABI shared library used by the Rust wrapper and other FFI consumers" OFF)
option(ONNXSIM_COVERAGE "Instrument onnxsim's own C++ targets for gcov/llvm-cov coverage" OFF)
option(ONNXSIM_TESTS "Build the dependency-free C++ unit tests (e.g. sym_expr)" OFF)
option(ONNXSIM_WASM_ORT_WEB "For the Emscripten build: delegate constant folding to the page's onnxruntime-web instead of compiling and linking ONNX Runtime into the module" OFF)
# Opt-in additional ModelExecutor backend (see onnxsim/xnnpack_executor.h and
# docs/dlpack-executor.md): runs constant-folding fold groups through Google's
# XNNPACK instead of ONNX Runtime, for a small, explicitly-supported subset of
# ops. Independent of ONNXSIM_BUILTIN_ORT -- both may be built at once; the
# caller picks which GetXxxModelExecutor() to hand to Simplify.
option(ONNXSIM_BUILTIN_XNNPACK "Build and link Google's XNNPACK as an additional (opt-in) ModelExecutor backend for constant folding" OFF)

if (ONNXSIM_WASM_ORT_WEB)
  # The whole point of this mode is to NOT bundle ONNX Runtime -- constant
  # folding is delegated to the page's onnxruntime-web at runtime -- so force the
  # built-in ORT off. onnxsim then compiles without ONNXSIM_HAS_ORT (all Ort::
  # code #ifdef'd out) and no ORT source is fetched or compiled.
  set(ONNXSIM_BUILTIN_ORT OFF)
endif()

# onnxsim's own onnx/onnx-optimizer fork (third_party/onnx, added below) is
# always built independently, in every configuration -- never provided
# transitively through ONNX Runtime's own build the way a from-source
# in-tree ORT used to (for Emscripten in particular: build_ort.cmake used to
# hand onnx a wasm protobuf ORT itself built via onnxruntime_USE_FULL_PROTOBUF
# + ONNX_TARGET_NAME=onnxruntime_webassembly). Without that transitive
# protobuf-for-wasm, onnx's generated .pb.* code fails to compile for the wasm
# target unless it builds its own bundled protobuf instead -- see
# ONNX_BUILD_CUSTOM_PROTOBUF below, set unconditionally for exactly this
# reason (the host protoc is still used for codegen via
# ONNX_CUSTOM_PROTOC_EXECUTABLE, passed by build_wasm.sh, when EMSCRIPTEN).

if (ONNXSIM_PYTHON AND EMSCRIPTEN)
  message(STATUS "python and emscripten cannot be built at the same time")
endif()

if (NOT ONNXSIM_BUILTIN_ORT AND EMSCRIPTEN AND NOT ONNXSIM_WASM_ORT_WEB)
  message(STATUS "emscripten needs builtin ort")
endif()

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>>
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>>
  $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:Clang>:-fcolor-diagnostics>>)
if (WIN32)
  add_compile_definitions(NOMINMAX)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ONNXSIM_PREBUILT_ORT AND NOT ONNXSIM_BUILTIN_ORT)
  message(FATAL_ERROR "ONNXSIM_PREBUILT_ORT requires ONNXSIM_BUILTIN_ORT=ON")
endif()
if (ONNXSIM_PREBUILT_ORT AND EMSCRIPTEN)
  message(FATAL_ERROR "ONNXSIM_PREBUILT_ORT is not supported for Emscripten; the "
                      "WebAssembly build must compile ONNX Runtime from source")
endif()
if (ONNXSIM_WASM_ORT_WEB AND NOT EMSCRIPTEN)
  message(FATAL_ERROR "ONNXSIM_WASM_ORT_WEB only applies to the Emscripten/"
                      "WebAssembly build (it routes folding through onnxruntime-web)")
endif()

if (ONNXSIM_BUILTIN_ORT)
  if (ONNXSIM_PREBUILT_ORT)
    # Link an official ONNX Runtime release (headers + libonnxruntime) and skip
    # the from-source ORT build entirely. This defines an imported `onnxruntime`
    # target and sets ONNXRUNTIME_INCLUDE_DIR, mirroring build_ort.cmake's
    # outputs so the rest of this file is unchanged.
    include(cmake/prebuilt_ort.cmake)
  else()
    # The top-level project only enables CXX, but ONNX Runtime (built below, out
    # of tree) compiles C and assembly sources. Enable the languages it needs
    # before configuring it. Emscripten's toolchain handles this on its own.
    if (NOT EMSCRIPTEN)
      enable_language(C)
      enable_language(ASM)
    endif()
    include(cmake/build_ort.cmake)
  endif()
  set(ORT_NAME onnxruntime)
endif()

if (ONNXSIM_BUILTIN_XNNPACK)
  if (EMSCRIPTEN)
    message(FATAL_ERROR "ONNXSIM_BUILTIN_XNNPACK is not supported for the "
                        "Emscripten/WebAssembly build")
  endif()
  include(cmake/build_xnnpack.cmake)
endif()

# onnxsim's own onnx/onnx-optimizer fork provides the `onnx` target -- used for
# every build (onnxsim's own Graph-native optimizer/shape-inference fixed
# point needs onnxsim-fork-only extensions, e.g. onnx/common/
# graph_shape_inference.h, that plain upstream onnx does not have), not just
# when ONNX Runtime is unavailable. When ONNXSIM_BUILTIN_ORT is ON, ONNX
# Runtime is now built/linked fully out-of-tree (see cmake/build_ort.cmake /
# cmake/prebuilt_ort.cmake) specifically so its own, separately-versioned
# vendored onnx never enters this project's target graph and never collides
# with the `onnx`/`onnx_proto` targets defined here (onnx's own CMakeLists.txt
# hardcodes those names, so two copies cannot coexist in one CMake project).
option(ONNX_OPT_USE_SYSTEM_PROTOBUF "" OFF)
if (NOT ONNX_OPT_USE_SYSTEM_PROTOBUF)
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_MSVC_STATIC_RUNTIME "" ${ONNX_USE_MSVC_STATIC_RUNTIME})
  set(protobuf_FORCE_FETCH_DEPENDENCIES ON)
  set(ONNX_BUILD_CUSTOM_PROTOBUF ON)
endif()
add_subdirectory(third_party/onnx EXCLUDE_FROM_ALL)

add_subdirectory(third_party/onnx-optimizer EXCLUDE_FROM_ALL)

# BLAKE3 (third_party/blake3, a git submodule pinned to a specific upstream
# commit -- see .gitmodules / `git submodule status`, and LICENSE_A2 /
# LICENSE_CC0 in that directory) backing TensorPool::ContentHash's default
# HashAlgorithm -- see tensor_pool_hash.h's file comment. Its C sources live
# under the submodule's own c/ subdirectory (upstream's layout, alongside
# Rust/other-language bindings this repo doesn't use). Only the portable
# files are compiled (no per-architecture SIMD), with every SIMD backend
# disabled: that makes blake3_dispatch.c's runtime CPU-feature dispatch fall
# through to blake3_portable.c unconditionally, which is what makes omitting
# the SIMD source files (and their assembly) correct rather than merely
# omitted. Applied via target_sources / target_compile_definitions below to
# each target that (transitively, via tensor_pool.cpp) needs TensorPool's
# hashing -- mirroring how tensor_pool.cpp itself is listed per-target
# rather than built once into a shared object library, matching this file's
# existing tensor_pool_test / tensor_pool_gguf_test pattern below.
set(ONNXSIM_BLAKE3_SOURCES
    third_party/blake3/c/blake3.c
    third_party/blake3/c/blake3_dispatch.c
    third_party/blake3/c/blake3_portable.c)
set(ONNXSIM_BLAKE3_COMPILE_DEFS
    BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512
    BLAKE3_USE_NEON=0)

add_library(onnxsim onnxsim/onnxsim.cpp onnxsim/constant_folding.cpp onnxsim/partial_shape_eval.cpp onnxsim/model_prep.cpp onnxsim/quantize_entry.cpp onnxsim/cross_layer_equalization_entry.cpp onnxsim/pruning_entry.cpp onnxsim/structured_pruning_entry.cpp onnxsim/contrib_schemas.cpp onnxsim/bev_custom_op_schemas.cpp onnxsim/custom_optimizer_passes.cpp onnxsim/function_rewriter.cpp onnxsim/profiler.cpp onnxsim/model_info.cpp onnxsim/sym_expr.cpp onnxsim/sym_value_eval.cpp onnxsim/sym_shape_infer.cpp onnxsim/model_metrics.cpp onnxsim/memory_planning.cpp onnxsim/xnnpack_codegen.cpp onnxsim/precision_estimator.cpp onnxsim/tensor_pool.cpp onnxsim/tensor_pool_gguf.cpp onnxsim/tensor_pool_hash.cpp onnxsim/gemm_fusion_backend.cpp ${ONNXSIM_BLAKE3_SOURCES})
target_compile_definitions(onnxsim PRIVATE ${ONNXSIM_BLAKE3_COMPILE_DEFS})
if (ONNXSIM_BUILTIN_ORT)
  # Both ways of obtaining ONNX Runtime ship/install headers flat under this
  # directory (an official release tarball, or cmake/build_ort.cmake's own
  # `cmake --install` of the ExternalProject build -- see that file), so
  # onnxsim.cpp/dlpack_bridge.h include onnxruntime_cxx_api.h directly.
  target_include_directories(onnxsim PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()
if (ONNXSIM_BUILTIN_XNNPACK)
  # onnx_to_xnnpack_subgraph.cpp / xnnpack_executor.cpp include xnnpack.h
  # directly; the XNNPACK CMake target itself (linked below) already exports
  # this include directory to its own sources but not to onnxsim's.
  target_sources(onnxsim PRIVATE onnxsim/onnx_to_xnnpack_subgraph.cpp
                                 onnxsim/xnnpack_executor.cpp)
  target_include_directories(onnxsim PRIVATE ${XNNPACK_INCLUDE_DIR})
  # The actual `target_link_libraries(onnxsim ... XNNPACK)` call is further
  # below, alongside onnx/onnx_optimizer/ORT_NAME: CMake requires every
  # target_link_libraries call for a given target to use the same
  # keyword/plain signature, and those existing calls are plain (no
  # PUBLIC/PRIVATE/INTERFACE), so this one must be too rather than introduced
  # here as its own PRIVATE call.
  # Guards the xnnpack.h-facing glue (onnx_to_xnnpack_subgraph.h,
  # GetXnnpackModelExecutor) the same way ONNXSIM_HAS_ORT guards the Ort::
  # glue above.
  target_compile_definitions(onnxsim PUBLIC ONNXSIM_HAS_XNNPACK)
endif()
target_include_directories(onnxsim PUBLIC onnxsim)
# dlpack.h (at third_party/dlpack/dlpack.h) is the tensor-exchange type at the
# ModelExecutor / C-ABI executor boundary (see docs/dlpack-executor.md). The
# include root is third_party/ so `#include "dlpack/dlpack.h"` resolves. PUBLIC
# so the C API and any target that includes onnxsim.h / dlpack_bridge.h finds it.
target_include_directories(onnxsim PUBLIC third_party)
if (ONNXSIM_BUILTIN_ORT)
  # Guards the Ort:: C++ API glue (dlpack_bridge.h, GetBuiltinModelExecutor) --
  # the only code that still needs to know whether ONNX Runtime is available,
  # since onnxsim's optimizer/shape-inference pipeline itself no longer forks
  # on this (see onnxsim.cpp).
  target_compile_definitions(onnxsim PUBLIC ONNXSIM_HAS_ORT)
endif()
if (EMSCRIPTEN)
  target_link_libraries(onnxsim onnx_optimizer onnx)
  if (ONNXSIM_BUILTIN_ORT)
    target_link_libraries(onnxsim ${ORT_NAME})
  endif()
else()
  # The profiler samples RSS on a background std::thread, so link the platform
  # threading library explicitly rather than relying on it arriving transitively
  # through ONNX Runtime. Emscripten provides threading through its own toolchain
  # flags, so skip it there.
  find_package(Threads REQUIRED)
  target_link_libraries(onnxsim onnx_optimizer onnx Threads::Threads)
  if (ONNXSIM_BUILTIN_ORT)
    target_link_libraries(onnxsim ${ORT_NAME})
  endif()
  if (ONNXSIM_BUILTIN_XNNPACK)
    target_link_libraries(onnxsim XNNPACK)
  endif()
endif()

# ORT is always a separate shared library now, whether built from source
# out-of-tree (cmake/build_ort.cmake) or an official prebuilt release
# (cmake/prebuilt_ort.cmake), so every executable/shared-object that links
# onnxsim needs to find libonnxruntime.* at runtime. rpath is a property of
# the final linked binary (not the static onnxsim library in between), so bake
# it into each such consumer individually via this helper rather than
# requiring LD_LIBRARY_PATH/DYLD_LIBRARY_PATH to be set by hand.
function(onnxsim_add_ort_rpath target)
  if (ONNXSIM_BUILTIN_ORT AND NOT WIN32 AND NOT EMSCRIPTEN)
    set_target_properties(${target} PROPERTIES
      INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
      BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
  endif()
endfunction()

if(EMSCRIPTEN)
  list(APPEND ONNXSIM_WASM_INTERFACE scripts/convertmodel/interface.cpp)
  if (ONNXSIM_WASM_ORT_WEB)
    # The onnxruntime-web-backed executor (compiled only in this mode).
    list(APPEND ONNXSIM_WASM_INTERFACE scripts/convertmodel/js_model_executor.cpp)
  endif()
endif()

# onnxsim_bin needs a constant-folding executor: GetBuiltinModelExecutor()
# (ONNXSIM_HAS_ORT) natively, or GetJsModelExecutor() (ONNXSIM_WASM_ORT_WEB)
# under Emscripten. With neither -- ONNXSIM_BUILTIN_ORT=OFF on a native,
# non-Emscripten build, e.g. the Python wheel's own configure -- there is no
# executor available at all, so exclude it from the default `all` target
# rather than failing that build (setup.py already only ever targets
# onnxsim_cpp2py_export explicitly; this just keeps a plain `cmake --build .`
# from tripping over a target that cannot link in that configuration).
if (ONNXSIM_HAS_ORT OR ONNXSIM_WASM_ORT_WEB)
  add_executable(
    onnxsim_bin
    onnxsim/bin/onnxsim_bin.cpp
    onnxsim/bin/onnxsim_option.cpp
    ${ONNXSIM_WASM_INTERFACE})
else()
  add_executable(
    onnxsim_bin EXCLUDE_FROM_ALL
    onnxsim/bin/onnxsim_bin.cpp
    onnxsim/bin/onnxsim_option.cpp
    ${ONNXSIM_WASM_INTERFACE})
endif()
set_target_properties(onnxsim_bin PROPERTIES OUTPUT_NAME onnxsim)

# Bake the onnxsim and onnx-optimizer version strings into the module so the
# WASM converter's interface.cpp can report them (the "versions" panel / issue
# report). Read from the VERSION files that ship in the tree; the onnx and
# protobuf versions are read at runtime from the linked libraries instead.
if (EMSCRIPTEN)
  set(_onnxsim_ver "unknown")
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _onnxsim_ver LIMIT_COUNT 1)
    string(STRIP "${_onnxsim_ver}" _onnxsim_ver)
  endif()
  set(_onnxopt_ver "unknown")
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-optimizer/VERSION_NUMBER")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-optimizer/VERSION_NUMBER" _onnxopt_ver LIMIT_COUNT 1)
    string(STRIP "${_onnxopt_ver}" _onnxopt_ver)
  endif()
  target_compile_definitions(onnxsim_bin PRIVATE
    "ONNXSIM_VERSION_STRING=\"${_onnxsim_ver}\""
    "ONNX_OPTIMIZER_VERSION_STRING=\"${_onnxopt_ver}\"")
endif()

if (EMSCRIPTEN)
  # ALLOW_MEMORY_GROWTH=1 lets the heap grow on demand, but without
  # MAXIMUM_MEMORY Emscripten caps that growth at 2 GiB and aborts with
  # "Cannot enlarge memory, requested N bytes, but the limit is 2147483648
  # bytes!" the moment a model needs more. Raise the cap to 4 GiB, the hard
  # addressing limit of a wasm32 module (2^32 bytes), so larger models can be
  # converted in the browser. Going beyond 4 GiB would require MEMORY64/wasm64.
  set_target_properties(onnxsim_bin PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4294967296 -s EXIT_RUNTIME=0 -s FORCE_FILESYSTEM=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_onnxsim\"' -s
  'EXPORTED_RUNTIME_METHODS=[ENV]' -Wl,--threads=8 -Wl,--lto-partitions=8 -Wl,--lto-O0 -Wl,--lto-CGO0 -s ERROR_ON_UNDEFINED_SYMBOLS=0 -sASSERTIONS=2 -sINVOKE_RUN=0")
  target_link_libraries(onnxsim_bin embind "-Wl,--whole-archive" onnxsim "-Wl,--no-whole-archive")
else()
  target_link_libraries(onnxsim_bin onnxsim)
  onnxsim_add_ort_rpath(onnxsim_bin)
endif()

if (ONNXSIM_WASM_ORT_WEB)
  # interface.cpp / onnxsim_bin.cpp select GetJsModelExecutor() under this define
  # and js_model_executor.cpp implements it. onnxsim_bin.cpp lives in onnxsim/bin,
  # so add scripts/convertmodel to its include path to find js_model_executor.h.
  target_compile_definitions(onnxsim_bin PRIVATE ONNXSIM_WASM_ORT_WEB)
  target_include_directories(onnxsim_bin PRIVATE scripts/convertmodel)
  # JsModelExecutor::_Run blocks on an onnxruntime-web Promise via
  # emscripten::val::await(), which requires Asyncify. Asyncify also makes the
  # exported onnxsimplify_export return a Promise (the worker awaits it). The
  # larger Asyncify stack covers the deep Simplify -> RunOps -> _Run call chain
  # that is unwound and rewound across each fold group's await.
  target_link_options(onnxsim_bin PRIVATE "-sASYNCIFY" "-sASYNCIFY_STACK_SIZE=131072")
endif()

if (ONNXSIM_C_API)
  if (NOT ONNXSIM_BUILTIN_ORT)
    message(FATAL_ERROR "ONNXSIM_C_API requires ONNXSIM_BUILTIN_ORT=ON")
  endif()
  # onnx_optimizer, onnx and onnx_proto (onnxsim's own fork, always built --
  # see the add_subdirectory(third_party/onnx) above) are absorbed statically
  # into onnxsim_c. They inherit the project-wide "hidden" visibility preset
  # (CMakeLists.txt's CMAKE_CXX_VISIBILITY_PRESET), which onnxsim_c needs
  # undone: it calls onnx_optimizer entry points (OptimizeFixed,
  # GetFuseAndEliminationPass, ...) and, historically when these were separate
  # shared libraries, needed them exported, so restore default visibility
  # (harmless now that they link in statically).
  foreach(_onnxsim_shared_dep onnx_optimizer onnx onnx_proto)
    if (TARGET ${_onnxsim_shared_dep})
      set_target_properties(${_onnxsim_shared_dep} PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN FALSE)
    endif()
  endforeach()
  add_library(onnxsim_c SHARED onnxsim/capi/onnxsim_c_api.cpp)
  # With the onnx stack built static, the linker prunes archive members
  # nothing references directly: onnx's operator schemas and shape-inference fns
  # (e.g. onnx::RNNShapeInference, self-registered via static initializers) and
  # onnx.pb.cc (onnx::ModelProto's vtable/typeinfo, referenced only through a
  # weak COMDAT reference). Whole-archive onnxsim + onnx + onnx_proto so every
  # such object is retained.
  #
  # Whole-archive by ARCHIVE FILE ($<TARGET_FILE:...>) rather than by target
  # name. onnx and onnx_proto also reach onnxsim_c transitively (as ordinary
  # link items through onnxsim/onnx_optimizer), and CMake refuses to link the
  # same *target* both with the WHOLE_ARCHIVE feature and plain -- the
  # "$<LINK_LIBRARY:WHOLE_ARCHIVE,...>" genex fails to configure with "link item
  # 'onnx' ... has already occurred with the feature 'WHOLE_ARCHIVE', which is
  # not allowed". Passing the archive paths as raw linker inputs sidesteps both
  # that per-target feature check and CMake's de-duplication (which otherwise
  # drops onnx_proto's onnx.pb.cc -- ModelProto's vtable/typeinfo -- back out of
  # a raw --whole-archive group). The transitive plain occurrences still link
  # normally for their referenced members; the whole-archive pass additionally
  # retains the members nothing references directly: onnx's self-registered
  # operator schemas / shape-inference fns (e.g. onnx::RNNShapeInference) and
  # onnx.pb.cc's ModelProto RTTI (reached only through a weak COMDAT reference).
  # The onnx-operators *.pb.cc object is archived into both libonnx and
  # libonnx_proto, so --allow-multiple-definition tolerates the duplicate:
  # protobuf's per-file descriptor_table merges to one copy whose once-guarded
  # registration runs exactly once. onnx_optimizer and ONNX Runtime link normally
  # (referenced directly / via the Ort:: C++ API through onnxsim's interface).
  target_link_libraries(onnxsim_c PRIVATE onnxsim onnx_optimizer)
  target_link_options(onnxsim_c PRIVATE
    "LINKER:--allow-multiple-definition"
    "LINKER:--whole-archive"
    "$<TARGET_FILE:onnxsim>"
    "$<TARGET_FILE:onnx>"
    "$<TARGET_FILE:onnx_proto>"
    "LINKER:--no-whole-archive")
  # $<TARGET_FILE:...> in link options does not itself create a build-order
  # edge, so require the archives explicitly (onnxsim already is a link dep).
  add_dependencies(onnxsim_c onnx onnx_proto)
  target_include_directories(onnxsim_c PUBLIC onnxsim/capi)
  # The exported symbols opt back into default visibility via ONNXSIM_C_API in
  # the header, so the project-wide "hidden" preset keeps everything else local.
  set_target_properties(onnxsim_c PROPERTIES C_VISIBILITY_PRESET hidden
                                             CXX_VISIBILITY_PRESET hidden)
  onnxsim_add_ort_rpath(onnxsim_c)
endif()

if (ONNXSIM_PYTHON)
  find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module REQUIRED
    OPTIONAL_COMPONENTS Development.SABIModule)

  # nanobind ships its CMake package config inside its own pip-installed
  # wheel (not on any default CMake search path), so it must be located via
  # its Python API before find_package(nanobind) can locate it -- this is
  # nanobind's documented CMake integration pattern. Set nanobind_DIR
  # directly to the directory --cmake_dir prints (which already *is* the
  # directory containing nanobind-config.cmake).
  #
  # nanobind-config.cmake self-computes its OWN "NB_DIR" (nanobind's package
  # root, used e.g. as "${NB_DIR}/cmake/darwin-ld-cpython.sym" for the macOS
  # linker response file) via get_filename_component(... CMAKE_CURRENT_LIST_FILE
  # ...) and caches it as NB_DIR CACHE INTERNAL. That collides by name with a
  # local (non-cache) variable this block used to set for the very same
  # --cmake_dir output -- on at least one macOS cibuildwheel build this left
  # nanobind's own NB_DIR one directory level too deep (pointing at the
  # cmake/ subdirectory instead of the package root), producing a doubled
  # ".../nanobind/cmake/cmake/darwin-ld-cpython.sym" link failure. Use a
  # differently-named local variable to avoid any chance of that collision,
  # and force nanobind's own NB_DIR to the independently-known-correct value
  # (one directory above --cmake_dir's output) as a defensive follow-up, in
  # case a stale NB_DIR cache entry from an earlier configure attempt in the
  # same build directory would otherwise have survived (plain `CACHE
  # INTERNAL` sets, as nanobind's config script uses, only take effect the
  # first time -- they do not overwrite an existing cache entry).
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE ONNXSIM_NANOBIND_CMAKE_DIR)
  set(nanobind_DIR "${ONNXSIM_NANOBIND_CMAKE_DIR}" CACHE PATH "" FORCE)
  find_package(nanobind CONFIG REQUIRED)
  get_filename_component(ONNXSIM_NANOBIND_PACKAGE_ROOT
                          "${ONNXSIM_NANOBIND_CMAKE_DIR}" DIRECTORY)
  set(NB_DIR "${ONNXSIM_NANOBIND_PACKAGE_ROOT}" CACHE INTERNAL "" FORCE)

  # FREE_THREADED marks the module Py_MOD_GIL_NOT_USED so free-threaded
  # interpreters (e.g. the cp314t wheel) keep the GIL disabled instead of
  # re-enabling it at import. nanobind ignores STABLE_ABI on free-threaded
  # builds (the limited API is unavailable there), so the two keywords coexist:
  # STABLE_ABI applies on regular interpreters, FREE_THREADED on cp314t.
  nanobind_add_module(onnxsim_cpp2py_export onnxsim/cpp2py_export.cc STABLE_ABI FREE_THREADED)
  target_link_libraries(onnxsim_cpp2py_export PRIVATE onnxsim)
  onnxsim_add_ort_rpath(onnxsim_cpp2py_export)

  # onnxsim links the ONNX and protobuf C++ libraries statically into this
  # extension, but the running Python process ALSO loads the separate pip
  # `onnx` package (onnx_cpp2py_export), which carries its own copy of the same
  # onnx::/google::protobuf:: symbols built against a possibly different
  # protobuf major (onnx <= 1.22 -> protobuf 5.x; our bundled onnx 1.23 ->
  # protobuf 6.x). ONNX and protobuf annotate their public classes with
  # default-visibility export macros (ONNX_API / PROTOBUF_EXPORT), so those
  # symbols leak out of this module even under the project-wide hidden preset.
  # On macOS, dyld coalesces such weak definitions (vtables, typeinfo, inline
  # methods) across every loaded image regardless of the two-level namespace,
  # so onnxsim's protobuf-6 message objects can end up dispatching through pip
  # onnx's protobuf-5 ClassData -- an ABI mismatch that corrupts the message
  # vtable/merge function pointer and crashes with SIGBUS inside TypeProto
  # merges during shape inference. Linux escapes it because CPython dlopens
  # extension modules with RTLD_LOCAL. Export ONLY the module init symbol so
  # none of the bundled onnx/protobuf symbols are visible to coalesce.
  if (APPLE)
    target_link_options(onnxsim_cpp2py_export PRIVATE
      "-Wl,-exported_symbol,_PyInit_onnxsim_cpp2py_export")
  endif()
endif()

if (ONNXSIM_COVERAGE)
  # Instrument onnxsim's own C++ sources (onnxsim.cpp, contrib_schemas.cpp,
  # cpp2py_export.cc) with gcov/llvm-cov counters. When the Python extension is
  # imported and called from pytest, the exercised C++ writes .gcda profiles
  # next to the .gcno files in the build directory, which gcovr then turns into
  # a C++ coverage report to sit alongside coverage.py's Python report.
  #
  # STABLE_ABI is irrelevant here: coverage builds are Debug/-O0 dev builds, not
  # distributable wheels. Only GCC and Clang support --coverage; MSVC does not.
  if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
    message(FATAL_ERROR
      "ONNXSIM_COVERAGE requires a GCC- or Clang-compatible compiler that "
      "understands --coverage (got '${CMAKE_CXX_COMPILER_ID}').")
  endif()

  # --coverage expands to -fprofile-arcs -ftest-coverage at compile time and
  # links the coverage runtime (libgcov / clang's profile runtime) at link time.
  # Instrument the static library and every consumer that links it so the
  # runtime is pulled into the final Python module / executable.
  set(_onnxsim_cov_targets onnxsim)
  if (TARGET onnxsim_cpp2py_export)
    list(APPEND _onnxsim_cov_targets onnxsim_cpp2py_export)
  endif()
  if (TARGET onnxsim_bin)
    list(APPEND _onnxsim_cov_targets onnxsim_bin)
  endif()
  if (TARGET onnxsim_c)
    list(APPEND _onnxsim_cov_targets onnxsim_c)
  endif()
  foreach(_t IN LISTS _onnxsim_cov_targets)
    target_compile_options(${_t} PRIVATE --coverage -O0 -g)
    target_link_options(${_t} PRIVATE --coverage)
  endforeach()
endif()

if (ONNXSIM_TESTS)
  # sym_expr is a self-contained, dependency-free unit (no ONNX / ORT), so its
  # test compiles and runs on its own -- including under Emscripten -- without
  # configuring the rest of the project.
  enable_testing()
  add_executable(sym_expr_test onnxsim/sym_expr_test.cpp onnxsim/sym_expr.cpp)
  target_include_directories(sym_expr_test PRIVATE onnxsim)
  add_test(NAME sym_expr_test COMMAND sym_expr_test)

  add_executable(model_metrics_test onnxsim/model_metrics_test.cpp
                                    onnxsim/model_metrics.cpp onnxsim/sym_expr.cpp)
  target_include_directories(model_metrics_test PRIVATE onnxsim)
  add_test(NAME model_metrics_test COMMAND model_metrics_test)

  # memory_planning builds on model_metrics' GraphView/liveness core, so it is
  # likewise ONNX-free and tests standalone the same way.
  add_executable(memory_planning_test onnxsim/memory_planning_test.cpp
                                      onnxsim/memory_planning.cpp
                                      onnxsim/model_metrics.cpp onnxsim/sym_expr.cpp)
  target_include_directories(memory_planning_test PRIVATE onnxsim)
  add_test(NAME memory_planning_test COMMAND memory_planning_test)

  # The M1 symbolic value evaluator (issue #532) is likewise dependency-free
  # (only SymExpr), so it builds and runs standalone -- including under
  # Emscripten -- without configuring ONNX / onnxruntime.
  add_executable(sym_value_eval_test onnxsim/sym_value_eval_test.cpp
                                     onnxsim/sym_value_eval.cpp onnxsim/sym_expr.cpp)
  target_include_directories(sym_value_eval_test PRIVATE onnxsim)
  add_test(NAME sym_value_eval_test COMMAND sym_value_eval_test)

  # M2 symbolic activation-shape inference (issue #532): dependency-free (only
  # SymExpr + the SymNode/SymTensor structs), so it builds and runs standalone.
  add_executable(sym_shape_infer_test onnxsim/sym_shape_infer_test.cpp
                                      onnxsim/sym_shape_infer.cpp
                                      onnxsim/sym_value_eval.cpp onnxsim/sym_expr.cpp)
  target_include_directories(sym_shape_infer_test PRIVATE onnxsim)
  add_test(NAME sym_shape_infer_test COMMAND sym_shape_infer_test)

  # The ONNX<->DLPack dtype mapping (dlpack_dtype.h) is pure: it operates on the
  # integer ONNX dtype codes and depends only on dlpack.h, so its test builds
  # and runs standalone -- including under Emscripten -- without ONNX / ORT.
  add_executable(dlpack_dtype_test onnxsim/dlpack_dtype_test.cpp)
  target_include_directories(dlpack_dtype_test PRIVATE onnxsim third_party)
  add_test(NAME dlpack_dtype_test COMMAND dlpack_dtype_test)

  # The ONNX<->safetensors dtype mapping (tensor_pool_dtype.h) is likewise
  # pure and dependency-free, mirroring dlpack_dtype_test.cpp above.
  add_executable(tensor_pool_dtype_test onnxsim/tensor_pool_dtype_test.cpp)
  target_include_directories(tensor_pool_dtype_test PRIVATE onnxsim)
  add_test(NAME tensor_pool_dtype_test COMMAND tensor_pool_dtype_test)

  # TensorPool's storage, safetensors read/write, and content hashing
  # (tensor_pool.{h,cpp}, tensor_pool_hash.{h,cpp}) depend only on
  # tensor_pool_dtype.h and the vendored BLAKE3, so this too builds and
  # runs standalone without ONNX / ONNX Runtime configured.
  add_executable(tensor_pool_test onnxsim/tensor_pool_test.cpp
                                  onnxsim/tensor_pool.cpp
                                  onnxsim/tensor_pool_hash.cpp
                                  ${ONNXSIM_BLAKE3_SOURCES})
  target_include_directories(tensor_pool_test PRIVATE onnxsim third_party)
  target_compile_definitions(tensor_pool_test PRIVATE ${ONNXSIM_BLAKE3_COMPILE_DEFS})
  add_test(NAME tensor_pool_test COMMAND tensor_pool_test)

  # tensor_pool_hash.{h,cpp}'s two backends (BLAKE3, this repo's own
  # SHA-256) checked against each algorithm's published test vectors, plus
  # ContentDigest's dtype/shape domain separation -- pure and
  # dependency-free like tensor_pool_dtype_test.cpp above.
  add_executable(tensor_pool_hash_test onnxsim/tensor_pool_hash_test.cpp
                                       onnxsim/tensor_pool_hash.cpp
                                       ${ONNXSIM_BLAKE3_SOURCES})
  target_include_directories(tensor_pool_hash_test PRIVATE onnxsim third_party)
  target_compile_definitions(tensor_pool_hash_test PRIVATE ${ONNXSIM_BLAKE3_COMPILE_DEFS})
  add_test(NAME tensor_pool_hash_test COMMAND tensor_pool_hash_test)

  # tensor_pool_bridge.h is the one piece that DOES need onnx (it converts
  # onnx::TensorProto <-> TensorPool), so its test links the fully-configured
  # `onnxsim` target rather than building standalone like the tests above.
  add_executable(tensor_pool_bridge_test onnxsim/tensor_pool_bridge_test.cpp)
  target_link_libraries(tensor_pool_bridge_test onnxsim)
  onnxsim_add_ort_rpath(tensor_pool_bridge_test)
  add_test(NAME tensor_pool_bridge_test COMMAND tensor_pool_bridge_test)

  # precision_estimator.{h,cpp} needs onnx (ModelProto/shape inference), like
  # tensor_pool_bridge_test above -- links the fully-configured `onnxsim`
  # target rather than building standalone.
  add_executable(precision_estimator_test onnxsim/precision_estimator_test.cpp)
  target_link_libraries(precision_estimator_test onnxsim)
  onnxsim_add_ort_rpath(precision_estimator_test)
  add_test(NAME precision_estimator_test COMMAND precision_estimator_test)

  # xnnpack_codegen.{h,cpp} needs onnx (ModelProto/shape inference), like
  # precision_estimator_test above -- links the fully-configured `onnxsim`
  # target rather than building standalone. Needs no XNNPACK build itself
  # (ONNXSIM_BUILTIN_XNNPACK): it only emits text naming XNNPACK's C API.
  add_executable(xnnpack_codegen_test onnxsim/xnnpack_codegen_test.cpp)
  target_link_libraries(xnnpack_codegen_test onnxsim)
  onnxsim_add_ort_rpath(xnnpack_codegen_test)
  add_test(NAME xnnpack_codegen_test COMMAND xnnpack_codegen_test)

  # contrib_schemas.cpp's MoE/QMoE registration needs onnx's schema
  # registry and function-building API, like precision_estimator_test
  # above -- links the fully-configured `onnxsim` target rather than
  # building standalone.
  add_executable(contrib_schemas_moe_test onnxsim/contrib_schemas_moe_test.cpp)
  target_link_libraries(contrib_schemas_moe_test onnxsim)
  onnxsim_add_ort_rpath(contrib_schemas_moe_test)
  add_test(NAME contrib_schemas_moe_test COMMAND contrib_schemas_moe_test)

  # Dev convenience only, never part of any default build target or CI job:
  # re-runs scripts/codegen/generate_moe_function_templates.py to refresh
  # onnxsim/contrib_schemas_moe_templates.gen.h in place. That header is
  # checked into the repository specifically so ordinary builds -- including
  # every cross-compile matrix (Windows/macOS/ARM, WASM/Pyodide, s390x) --
  # never need onnxscript, which this script (unlike anything else the C++
  # build touches) depends on; see that script's own module docstring. Run
  # explicitly with `cmake --build . --target regenerate_moe_templates` when
  # iterating on the MoE decomposition itself.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_custom_target(regenerate_moe_templates
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/scripts/codegen/generate_moe_function_templates.py
              ${CMAKE_CURRENT_SOURCE_DIR}/onnxsim/contrib_schemas_moe_templates.gen.h
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Regenerating contrib_schemas_moe_templates.gen.h from onnxscript"
      VERBATIM)
  endif()

  # The ONNX<->GGML dtype mapping (gguf_dtype.h), mirroring
  # tensor_pool_dtype_test.cpp above for the same reason.
  add_executable(gguf_dtype_test onnxsim/gguf_dtype_test.cpp)
  target_include_directories(gguf_dtype_test PRIVATE onnxsim)
  add_test(NAME gguf_dtype_test COMMAND gguf_dtype_test)

  # GGML K-quant dequantization (ggml_kquant.h) depends only on gguf_dtype.h,
  # so it too builds and runs standalone -- mirroring gguf_dtype_test.cpp.
  add_executable(ggml_kquant_test onnxsim/ggml_kquant_test.cpp)
  target_include_directories(ggml_kquant_test PRIVATE onnxsim)
  add_test(NAME ggml_kquant_test COMMAND ggml_kquant_test)

  # GGML MXFP4 dequantization (ggml_mxfp4.h) -- same standalone-buildable
  # dependency footprint as ggml_kquant_test above.
  add_executable(ggml_mxfp4_test onnxsim/ggml_mxfp4_test.cpp)
  target_include_directories(ggml_mxfp4_test PRIVATE onnxsim)
  add_test(NAME ggml_mxfp4_test COMMAND ggml_mxfp4_test)

  # GGML legacy quant (Q4_0/Q4_1/Q5_0/Q5_1) dequantization
  # (ggml_legacy_quant.h) -- same standalone-buildable dependency footprint
  # as ggml_kquant_test above (it reuses that header's fp16 conversion, but
  # still needs only gguf_dtype.h/ggml_kquant.h, no protobuf/onnx).
  add_executable(ggml_legacy_quant_test onnxsim/ggml_legacy_quant_test.cpp)
  target_include_directories(ggml_legacy_quant_test PRIVATE onnxsim)
  add_test(NAME ggml_legacy_quant_test COMMAND ggml_legacy_quant_test)

  # TensorPool's GGUF read/write (tensor_pool_gguf.cpp) depends only on
  # tensor_pool.{h,cpp} and gguf_dtype.h, so it too builds and runs
  # standalone without ONNX / ONNX Runtime configured.
  add_executable(tensor_pool_gguf_test onnxsim/tensor_pool_gguf_test.cpp
                                       onnxsim/tensor_pool.cpp
                                       onnxsim/tensor_pool_hash.cpp
                                       onnxsim/tensor_pool_gguf.cpp
                                       ${ONNXSIM_BLAKE3_SOURCES})
  target_include_directories(tensor_pool_gguf_test PRIVATE onnxsim third_party)
  target_compile_definitions(tensor_pool_gguf_test PRIVATE ${ONNXSIM_BLAKE3_COMPILE_DEFS})
  add_test(NAME tensor_pool_gguf_test COMMAND tensor_pool_gguf_test)

  # ReadGGUFMetadata (tensor_pool_gguf.cpp) -- same standalone-buildable
  # dependency footprint as tensor_pool_gguf_test above.
  add_executable(read_gguf_metadata_test onnxsim/read_gguf_metadata_test.cpp
                                         onnxsim/tensor_pool.cpp
                                         onnxsim/tensor_pool_hash.cpp
                                         onnxsim/tensor_pool_gguf.cpp
                                         ${ONNXSIM_BLAKE3_SOURCES})
  target_include_directories(read_gguf_metadata_test PRIVATE onnxsim third_party)
  target_compile_definitions(read_gguf_metadata_test PRIVATE ${ONNXSIM_BLAKE3_COMPILE_DEFS})
  add_test(NAME read_gguf_metadata_test COMMAND read_gguf_metadata_test)

  # tensor_pool_gguf_bridge.h needs onnx, same reason as
  # tensor_pool_bridge_test above.
  add_executable(tensor_pool_gguf_bridge_test
                onnxsim/tensor_pool_gguf_bridge_test.cpp)
  target_link_libraries(tensor_pool_gguf_bridge_test onnxsim)
  onnxsim_add_ort_rpath(tensor_pool_gguf_bridge_test)
  add_test(NAME tensor_pool_gguf_bridge_test
          COMMAND tensor_pool_gguf_bridge_test)

  # The self-describing single-file archive functions (EmbedModel/
  # ExtractModel and their Save/LoadModelAs* wrappers) span both bridge
  # headers, so this one test exercises both the safetensors and the GGUF
  # archive path; needs onnx for the same reason as the two tests above.
  add_executable(tensor_pool_archive_test
                onnxsim/tensor_pool_archive_test.cpp)
  target_link_libraries(tensor_pool_archive_test onnxsim)
  onnxsim_add_ort_rpath(tensor_pool_archive_test)
  add_test(NAME tensor_pool_archive_test COMMAND tensor_pool_archive_test)

  if (ONNXSIM_BUILTIN_XNNPACK)
    # Exercises GetXnnpackModelExecutor() end to end (ONNX ModelProto in,
    # DLManagedTensor out), so it links the fully-configured `onnxsim` target
    # like the tests above, and additionally needs xnnpack.h for building its
    # own input/expected-output tensors.
    add_executable(xnnpack_executor_test onnxsim/xnnpack_executor_test.cpp)
    target_link_libraries(xnnpack_executor_test onnxsim)
    target_include_directories(xnnpack_executor_test PRIVATE ${XNNPACK_INCLUDE_DIR})
    onnxsim_add_ort_rpath(xnnpack_executor_test)
    add_test(NAME xnnpack_executor_test COMMAND xnnpack_executor_test)
  endif()
endif()
