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

if (ONNXSIM_WASM_ORT_WEB)
  # The whole point of this mode is to NOT bundle ONNX Runtime -- constant
  # folding is delegated to the page's onnxruntime-web at runtime -- so force the
  # built-in ORT off. onnxsim then compiles with NO_BUILTIN_ORT (all Ort:: code
  # #ifdef'd out) and no ORT source is fetched or compiled.
  set(ONNXSIM_BUILTIN_ORT OFF)
  # In the default WASM build, ONNX Runtime builds protobuf for the wasm target
  # and hands it to onnx (build_ort.cmake sets onnxruntime_USE_FULL_PROTOBUF and
  # ONNX_TARGET_NAME). With ORT gone, onnx has no wasm protobuf and its generated
  # .pb.* code fails to compile. Have onnx build its own bundled protobuf for the
  # target instead; the host protoc is still used for codegen via
  # ONNX_CUSTOM_PROTOC_EXECUTABLE (passed by build_wasm.sh).
  set(ONNX_BUILD_CUSTOM_PROTOBUF ON CACHE BOOL "" FORCE)
endif()

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
    set(ORT_NAME onnxruntime)
  else()
    # The top-level project only enables CXX, but ONNX Runtime (added below as a
    # subdirectory) compiles C and assembly sources. Enable the languages it needs
    # before configuring it. Emscripten's toolchain handles this on its own.
    if (NOT EMSCRIPTEN)
      enable_language(C)
      enable_language(ASM)
    endif()
    include(cmake/build_ort.cmake)
    if (EMSCRIPTEN)
      set(ORT_NAME onnxruntime_webassembly)
    else()
      set(ORT_NAME onnxruntime)
    endif()
  endif()
endif()

# Provide the `onnx` target from onnxsim's own top-level submodule instead of
# letting onnx-optimizer's nested third_party/onnx submodule provide it (its
# cmake/utils.cmake add_subdirectory_if_no_target() skips its own copy once a
# target named `onnx` already exists -- must run before
# add_subdirectory(third_party/onnx-optimizer) below). This keeps onnx's
# pinned commit its own top-level submodule entry in onnxsim's history,
# rather than folded invisibly into an onnx-optimizer submodule bump.
# Mirrors the protobuf-related CMake variables onnx-optimizer's own
# CMakeLists.txt otherwise sets before its (now-skipped) call, so onnx's
# build configuration is unchanged.
#
# Left to ONNXSIM_BUILTIN_ORT's own path when that's ON: onnxruntime (added
# above) already provides its own `onnx` target first there, and onnx-
# optimizer's add_subdirectory_if_no_target() links against that instead, "because
# they both depend on onnx and onnxruntime has its own flags for onnx" --
# unrelated to and unaffected by this change.
if (NOT ONNXSIM_BUILTIN_ORT)
  option(ONNX_OPT_USE_SYSTEM_PROTOBUF "" OFF)
  if (NOT ONNX_OPT_USE_SYSTEM_PROTOBUF)
    option(protobuf_BUILD_TESTS "" OFF)
    option(protobuf_MSVC_STATIC_RUNTIME "" ${ONNX_USE_MSVC_STATIC_RUNTIME})
    set(protobuf_FORCE_FETCH_DEPENDENCIES ON)
    set(ONNX_BUILD_CUSTOM_PROTOBUF ON)
  endif()
  add_subdirectory(third_party/onnx EXCLUDE_FROM_ALL)
endif()

# configure onnx-optimizer after onnxruntime, because they both depend on onnx and onnxruntime has its own flags for onnx
add_subdirectory(third_party/onnx-optimizer EXCLUDE_FROM_ALL)

add_library(onnxsim onnxsim/onnxsim.cpp onnxsim/contrib_schemas.cpp onnxsim/custom_optimizer_passes.cpp onnxsim/function_rewriter.cpp onnxsim/profiler.cpp onnxsim/model_info.cpp onnxsim/sym_expr.cpp onnxsim/sym_value_eval.cpp onnxsim/sym_shape_infer.cpp onnxsim/incremental_shape_infer.cpp onnxsim/model_metrics.cpp)
if (ONNXSIM_BUILTIN_ORT)
  target_include_directories(onnxsim PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  if (ONNXSIM_PREBUILT_ORT)
    # Prebuilt releases ship the public headers flat under include/, so onnxsim.cpp
    # includes onnxruntime_cxx_api.h directly instead of the nested source path.
    target_compile_definitions(onnxsim PRIVATE ONNXSIM_ORT_FLAT_HEADERS)
  endif()
endif()
target_include_directories(onnxsim PUBLIC onnxsim)
# dlpack.h (at third_party/dlpack/dlpack.h) is the tensor-exchange type at the
# ModelExecutor / C-ABI executor boundary (see docs/dlpack-executor.md). The
# include root is third_party/ so `#include "dlpack/dlpack.h"` resolves. PUBLIC
# so the C API and any target that includes onnxsim.h / dlpack_bridge.h finds it.
target_include_directories(onnxsim PUBLIC third_party)
if (NOT ONNXSIM_BUILTIN_ORT)
  target_compile_definitions(onnxsim PUBLIC NO_BUILTIN_ORT)
endif()
if (EMSCRIPTEN)
  if (ONNXSIM_BUILTIN_ORT)
    target_link_libraries(onnxsim ${ORT_NAME} onnx_optimizer)
  else()
    # ORT-web build: no ONNX Runtime is linked in, so onnx is no longer provided
    # transitively by it. Link onnx (and onnx_optimizer) directly instead.
    target_link_libraries(onnxsim onnx_optimizer onnx)
  endif()
else()
  # The profiler samples RSS on a background std::thread, so link the platform
  # threading library explicitly rather than relying on it arriving transitively
  # through ONNX Runtime. Emscripten provides threading through its own toolchain
  # flags, so skip it there.
  find_package(Threads REQUIRED)
  target_link_libraries(onnxsim ${ORT_NAME} onnx_optimizer onnx Threads::Threads)
endif()

if(EMSCRIPTEN)
  list(APPEND ONNXSIM_WASM_INTERFACE scripts/convertmodel/interface.cpp)
  if (ONNXSIM_WASM_ORT_WEB)
    # The onnxruntime-web-backed executor (compiled only in this mode).
    list(APPEND ONNXSIM_WASM_INTERFACE scripts/convertmodel/js_model_executor.cpp)
  endif()
endif()

add_executable(
  onnxsim_bin
  onnxsim/bin/onnxsim_bin.cpp
  onnxsim/bin/onnxsim_option.cpp
  ${ONNXSIM_WASM_INTERFACE})
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
  # onnx_optimizer, onnx and onnx_proto are built as part of the builtin-ORT
  # subtree and absorbed statically into onnxsim_c (option 1; see
  # build_ort.cmake). Two settings they inherit from that subtree must be undone
  # for onnxsim_c to link:
  #
  #  * Visibility -- they inherit the project-wide "hidden" preset. onnxsim_c
  #    calls onnx_optimizer entry points (OptimizeFixed, GetFuseAndElimination
  #    Pass, ...) and, historically when these were separate shared libraries,
  #    needed them exported, so restore default visibility (harmless now that
  #    they link in statically).
  #  * RTTI -- ONNX Runtime configures its subtree with onnxruntime_DISABLE_RTTI
  #    =ON (-fno-rtti), so onnx, the generated *.pb.cc and the bundled protobuf
  #    omit their typeinfo (onnx-ml.pb.cc emits onnx::ModelProto's vtable but not
  #    its typeinfo; libprotobuf omits google::protobuf::Message/MessageLite
  #    typeinfo). onnxsim is compiled RTTI-on -- it has to be, because
  #    onnx-optimizer's cse_util.h uses typeid(T) -- and references
  #    onnx::ModelProto's typeinfo, so the link fails with "undefined symbol:
  #    typeinfo for onnx::ModelProto" (and, once that resolves, its
  #    google::protobuf::Message base). Compiling onnxsim -fno-rtti to match the
  #    stack is therefore not possible; instead force -frtti back on for the onnx
  #    and protobuf targets so the whole typeinfo chain is emitted (-frtti wins
  #    as the last -f(no-)rtti on the command line).
  foreach(_onnxsim_shared_dep onnx_optimizer onnx onnx_proto)
    if (TARGET ${_onnxsim_shared_dep})
      set_target_properties(${_onnxsim_shared_dep} PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN FALSE)
    endif()
  endforeach()
  foreach(_onnxsim_rtti_dep onnx onnx_proto libprotobuf libprotobuf-lite)
    if (TARGET ${_onnxsim_rtti_dep})
      target_compile_options(${_onnxsim_rtti_dep} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-frtti>)
    endif()
  endforeach()
  add_library(onnxsim_c SHARED onnxsim/capi/onnxsim_c_api.cpp)
  # With the onnx stack built static (option 1) the linker prunes archive members
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
endif()

if (ONNXSIM_PYTHON)
  find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module REQUIRED
    OPTIONAL_COMPONENTS Development.SABIModule)
  # FREE_THREADED marks the module Py_MOD_GIL_NOT_USED so free-threaded
  # interpreters (e.g. the cp314t wheel) keep the GIL disabled instead of
  # re-enabling it at import. nanobind ignores STABLE_ABI on free-threaded
  # builds (the limited API is unavailable there), so the two keywords coexist:
  # STABLE_ABI applies on regular interpreters, FREE_THREADED on cp314t.
  nanobind_add_module(onnxsim_cpp2py_export onnxsim/cpp2py_export.cc STABLE_ABI FREE_THREADED)
  target_link_libraries(onnxsim_cpp2py_export PRIVATE onnxsim)

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

  # The incremental (per-node, memoized) shape inference driver exercises real
  # ONNX schemas and inference functions, so -- unlike the dependency-free
  # tests above -- it links against onnx.
  add_executable(incremental_shape_infer_test onnxsim/incremental_shape_infer_test.cpp
                                              onnxsim/incremental_shape_infer.cpp)
  target_include_directories(incremental_shape_infer_test PRIVATE onnxsim third_party)
  target_link_libraries(incremental_shape_infer_test PRIVATE onnx)
  add_test(NAME incremental_shape_infer_test COMMAND incremental_shape_infer_test)
endif()
