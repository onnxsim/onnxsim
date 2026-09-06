# Build ONNX Runtime from source as a fully isolated CMake build (via
# ExternalProject_Add) and expose it the same way cmake/prebuilt_ort.cmake
# exposes an official release: an imported `onnxruntime` target carrying
# ONNXRUNTIME_INCLUDE_DIR + the compiled shared library.
#
# This used to `add_subdirectory(third_party/onnxruntime-1.29.0/cmake)` ORT's
# own CMake project directly into onnxsim's own CMake project graph. That
# meant ORT's own vendored onnx copy (fetched by ORT's build via
# FetchContent, a different onnx version than onnxsim's own third_party/onnx
# fork) became part of onnxsim's target graph too -- which is fine as long as
# onnxsim's own third_party/onnx is never *also* configured in the same
# build (that's exactly what ONNXSIM_BUILTIN_ORT gated: only one of the two
# onnx copies is ever present). Now that onnxsim's own onnx/onnx-optimizer
# fork is configured unconditionally (see CMakeLists.txt), both onnx copies
# would try to define CMake targets literally named `onnx`/`onnx_proto` --
# onnx's own CMakeLists.txt hardcodes those names, they are not
# ONNX_NAMESPACE-parameterized -- which is a hard CMake configure error.
# Building ORT out-of-tree, so its internal onnx never enters this project's
# target graph at all, sidesteps the collision entirely (and matches how the
# official prebuilt releases prebuilt_ort.cmake consumes already behave: they
# are precompiled, so their internal onnx never appears here either).
include(ExternalProject)

set(_ort_ext_source_dir "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-1.29.0")
set(_ort_ext_prefix "${CMAKE_BINARY_DIR}/onnxruntime-ext")
set(_ort_ext_install_dir "${_ort_ext_prefix}/install")

set(_ort_cmake_args
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_INSTALL_PREFIX=${_ort_ext_install_dir}"
  "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  # For MessageDifferencer::Equals, used by onnxsim's own model-diffing code.
  "-Donnxruntime_USE_FULL_PROTOBUF=ON"
  "-Donnxruntime_BUILD_SHARED_LIB=ON"
  "-Donnxruntime_BUILD_UNIT_TESTS=OFF")

# Forward the toolchain explicitly: unlike add_subdirectory, ExternalProject_Add
# configures ORT as a fully separate CMake invocation and inherits nothing from
# the parent configure automatically.
if (CMAKE_TOOLCHAIN_FILE)
  list(APPEND _ort_cmake_args "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
endif()
if (CMAKE_C_COMPILER)
  list(APPEND _ort_cmake_args "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
endif()
if (CMAKE_CXX_COMPILER)
  list(APPEND _ort_cmake_args "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()
if (APPLE)
  if (CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND _ort_cmake_args "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  if (CMAKE_OSX_ARCHITECTURES)
    list(APPEND _ort_cmake_args "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
  endif()
endif()

if (EMSCRIPTEN)
  if (NOT DEFINED ONNX_CUSTOM_PROTOC_EXECUTABLE)
    message(FATAL_ERROR "ONNX_CUSTOM_PROTOC_EXECUTABLE must be set for emscripten")
  endif()
  list(APPEND _ort_cmake_args
    "-DONNX_CUSTOM_PROTOC_EXECUTABLE=${ONNX_CUSTOM_PROTOC_EXECUTABLE}"
    "-Donnxruntime_BUILD_WEBASSEMBLY=ON"
    "-Donnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB=ON"
    "-Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=ON"
    "-Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING=ON"
    "-Donnxruntime_ENABLE_WEBASSEMBLY_THREADS=ON"
    "-Donnxruntime_EMSCRIPTEN_SETTINGS=MALLOC=dlmalloc")
  set(_ort_lib_name "libonnxruntime_webassembly.a")
elseif (WIN32)
  set(_ort_lib_name "onnxruntime.dll")
elseif (APPLE)
  set(_ort_lib_name "libonnxruntime.dylib")
else()
  set(_ort_lib_name "libonnxruntime.so")
endif()

ExternalProject_Add(onnxruntime_external
  SOURCE_DIR "${_ort_ext_source_dir}/cmake"
  PREFIX "${_ort_ext_prefix}"
  CMAKE_ARGS ${_ort_cmake_args}
  BUILD_BYPRODUCTS "${_ort_ext_install_dir}/lib/${_ort_lib_name}"
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
  EXCLUDE_FROM_ALL TRUE)

# ORT's own `install(TARGETS onnxruntime PUBLIC_HEADER DESTINATION
# ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime ...)` rule installs its public
# headers flat under include/onnxruntime/ (not nested under
# include/onnxruntime/core/session/ the way the in-tree source layout is), the
# same flat shape the official release tarballs ship -- so this path sets
# ONNXSIM_ORT_FLAT_HEADERS too (see CMakeLists.txt).
set(ONNXRUNTIME_INCLUDE_DIR "${_ort_ext_install_dir}/include/onnxruntime")
# CMake validates that an IMPORTED target's INTERFACE_INCLUDE_DIRECTORIES
# exist at generate time, even though this directory is only populated later
# by onnxruntime_external's own build+install step. Pre-create it (empty) so
# that check passes; the actual headers land here before anything that
# #includes onnxruntime_cxx_api.h is compiled, since onnxsim depends on
# onnxruntime_external (see add_dependencies below).
file(MAKE_DIRECTORY "${ONNXRUNTIME_INCLUDE_DIR}")

add_library(onnxruntime SHARED IMPORTED GLOBAL)
add_dependencies(onnxruntime onnxruntime_external)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${_ort_ext_install_dir}/lib/${_ort_lib_name}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
if (WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${_ort_ext_install_dir}/lib/onnxruntime.lib")
endif()

set(ONNXRUNTIME_LIB_DIR "${_ort_ext_install_dir}/lib" CACHE INTERNAL
    "Directory holding the from-source-built onnxruntime shared library")
