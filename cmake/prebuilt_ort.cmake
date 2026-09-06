# Use a prebuilt ONNX Runtime release instead of compiling ONNX Runtime from
# source. The full source build (cmake/build_ort.cmake) recompiles ONNX Runtime,
# protobuf and friends, which is the slowest part of a native onnxsim build; the
# official releases at https://github.com/microsoft/onnxruntime/releases ship a
# ready-to-link libonnxruntime shared library and the public C/C++ API headers,
# so pointing at one lets a build skip that step entirely.
#
# Inputs (cache variables):
#   ONNXSIM_ORT_HOME    - path to an already-extracted release directory (the one
#                         that contains include/ and lib/). When set it is used
#                         as-is and nothing is downloaded.
#   ONNXSIM_ORT_VERSION - release version to download when ONNXSIM_ORT_HOME is
#                         unset (default 1.29.0).
#   ONNXSIM_ORT_URL     - full download URL, overriding the auto-detected release
#                         asset (useful for GPU builds or mirrors).
#
# Outputs:
#   ONNXRUNTIME_INCLUDE_DIR - the release's public-header directory
#   imported target `onnxruntime` - links libonnxruntime and carries the include
#                                   directory, matching ORT_NAME in CMakeLists.txt
#
# The shipped lib/cmake/onnxruntime config package is deliberately *not* used:
# its generated paths assume a CMake `install()` layout (lib64/, include/onnxruntime/)
# that does not match the release tarball, so find_package(onnxruntime) fails on
# the archive. Locating the artifacts directly is more robust.

set(ONNXSIM_ORT_VERSION "1.29.0" CACHE STRING
    "Prebuilt ONNX Runtime release version to use when ONNXSIM_PREBUILT_ORT is ON")
set(ONNXSIM_ORT_HOME "" CACHE PATH
    "Path to an extracted prebuilt ONNX Runtime release (skips the download)")
set(ONNXSIM_ORT_URL "" CACHE STRING
    "Override URL for the prebuilt ONNX Runtime release archive")

# Map the host platform to the release asset name microsoft/onnxruntime publishes.
# Only the pieces onnxsim actually needs (Linux/macOS/Windows CPU) are covered;
# anything else should set ONNXSIM_ORT_HOME or ONNXSIM_ORT_URL explicitly.
function(_onnxsim_ort_asset out_stem out_ext)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (_arch MATCHES "aarch64|arm64")
      set(_a "onnxruntime-linux-aarch64-${ONNXSIM_ORT_VERSION}")
    else()
      set(_a "onnxruntime-linux-x64-${ONNXSIM_ORT_VERSION}")
    endif()
    set(_e "tgz")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_a "onnxruntime-osx-universal2-${ONNXSIM_ORT_VERSION}")
    set(_e "tgz")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_a "onnxruntime-win-x64-${ONNXSIM_ORT_VERSION}")
    set(_e "zip")
  else()
    message(FATAL_ERROR
      "ONNXSIM_PREBUILT_ORT: no known release asset for platform "
      "'${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}'. Set ONNXSIM_ORT_HOME to "
      "an extracted release or ONNXSIM_ORT_URL to a download URL.")
  endif()
  set(${out_stem} "${_a}" PARENT_SCOPE)
  set(${out_ext} "${_e}" PARENT_SCOPE)
endfunction()

set(_ort_home "${ONNXSIM_ORT_HOME}")

if (NOT _ort_home)
  _onnxsim_ort_asset(_ort_stem _ort_ext)
  if (ONNXSIM_ORT_URL)
    set(_ort_url "${ONNXSIM_ORT_URL}")
  else()
    set(_ort_url
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXSIM_ORT_VERSION}/${_ort_stem}.${_ort_ext}")
  endif()

  set(_ort_root "${CMAKE_BINARY_DIR}/prebuilt-ort")
  set(_ort_home "${_ort_root}/${_ort_stem}")
  set(_ort_archive "${_ort_root}/${_ort_stem}.${_ort_ext}")

  # Extraction marker: the extracted tree contains include/, so treat its
  # presence as "already downloaded and unpacked" and skip the network on
  # reconfigure.
  if (NOT EXISTS "${_ort_home}/include")
    file(MAKE_DIRECTORY "${_ort_root}")
    message(STATUS "Downloading prebuilt ONNX Runtime ${ONNXSIM_ORT_VERSION} from ${_ort_url}")
    file(DOWNLOAD "${_ort_url}" "${_ort_archive}" SHOW_PROGRESS STATUS _ort_dl_status)
    list(GET _ort_dl_status 0 _ort_dl_code)
    if (NOT _ort_dl_code EQUAL 0)
      list(GET _ort_dl_status 1 _ort_dl_msg)
      message(FATAL_ERROR "Failed to download ONNX Runtime from ${_ort_url}: ${_ort_dl_msg}")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_ort_archive}" DESTINATION "${_ort_root}")
    if (NOT EXISTS "${_ort_home}/include")
      message(FATAL_ERROR
        "Extracted ${_ort_archive} but ${_ort_home}/include is missing; the "
        "archive layout may have changed. Set ONNXSIM_ORT_HOME manually.")
    endif()
  endif()
endif()

# Resolve the include directory (public headers live flat under include/).
set(ONNXRUNTIME_INCLUDE_DIR "${_ort_home}/include")
if (NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNXSIM_PREBUILT_ORT: ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h not "
    "found. Is '${_ort_home}' an extracted ONNX Runtime release?")
endif()

# Resolve the shared library. Releases keep it in lib/; prefer the linker-name
# symlink (libonnxruntime.so / .dylib / onnxruntime.lib) and fall back to a glob.
if (WIN32)
  set(_ort_libname "onnxruntime.lib")
elseif (APPLE)
  set(_ort_libname "libonnxruntime.dylib")
else()
  set(_ort_libname "libonnxruntime.so")
endif()

set(_ort_lib "")
foreach(_libdir "${_ort_home}/lib" "${_ort_home}/lib64")
  if (EXISTS "${_libdir}/${_ort_libname}")
    set(_ort_lib "${_libdir}/${_ort_libname}")
    break()
  endif()
  file(GLOB _ort_lib_candidates
       "${_libdir}/libonnxruntime.so*"
       "${_libdir}/libonnxruntime*.dylib"
       "${_libdir}/onnxruntime.lib")
  if (_ort_lib_candidates)
    list(GET _ort_lib_candidates 0 _ort_lib)
    break()
  endif()
endforeach()

if (NOT _ort_lib)
  message(FATAL_ERROR
    "ONNXSIM_PREBUILT_ORT: could not find the onnxruntime library under "
    "${_ort_home}/lib. Set ONNXSIM_ORT_HOME to a complete release.")
endif()

message(STATUS "Using prebuilt ONNX Runtime")
message(STATUS "  headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  library: ${_ort_lib}")

# Expose an imported target named `onnxruntime` so the rest of CMakeLists.txt can
# link ${ORT_NAME} uniformly, whether ORT was built from source or prefetched.
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${_ort_lib}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

# On Windows the import library and the runtime DLL are separate files; point the
# loader at the DLL so downstream consumers can copy it next to their binaries.
if (WIN32)
  file(GLOB _ort_dll "${_ort_home}/lib/onnxruntime.dll")
  if (_ort_dll)
    list(GET _ort_dll 0 _ort_dll0)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB "${_ort_lib}"
                                                 IMPORTED_LOCATION "${_ort_dll0}")
  endif()
endif()

# Surface the library directory so callers can extend rpath / copy the runtime.
get_filename_component(ONNXRUNTIME_LIB_DIR "${_ort_lib}" DIRECTORY)
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIB_DIR}" CACHE INTERNAL
    "Directory holding the prebuilt onnxruntime shared library")
