cmake_minimum_required(VERSION 3.16)
project(onnx_remote_transport LANGUAGES CXX)

include(CTest)

option(ONNX_REMOTE_AXCL "Build the AXCL-backed remote worker" OFF)
option(ONNX_REMOTE_VIPLITE
       "Build the Allwinner VIPLite (Vivante NPU) remote worker" OFF)
set(VIPLITE_INCLUDE_DIR "" CACHE PATH
    "VIPLite SDK include directory (contains vip_lite.h)")
set(VIPLITE_LIBRARY "" CACHE FILEPATH
    "VIPLite runtime library (libNBGlinker.so, or libVIPlite.so on older SoCs)")
option(ONNXSIM_REMOTE_DORA "Build the DORA node adapter" OFF)
option(ONNXSIM_REMOTE_DORA_STUB
       "Use the dependency-free DORA C API stub for adapter compile tests" OFF)
option(ONNXSIM_REMOTE_ROS2 "Build the optional ROS2 binary bridge" OFF)
option(ONNXSIM_REMOTE_GRPC "Generate the optional host-side gRPC bindings" OFF)
option(ONNXSIM_REMOTE_ORT_PUBLIC_PLUGIN
       "Build the public OrtEp remote execution-provider plugin"
       OFF)
option(ONNXSIM_REMOTE_ORT_WORKER
       "Build the optional ORT-backed serialized subgraph worker" OFF)
set(ONNXSIM_ORT_WORKER_INCLUDE_DIR "" CACHE PATH
    "ONNX Runtime public include directory for the ORT worker")
set(ONNXSIM_ORT_WORKER_LIBRARY "" CACHE FILEPATH
    "ONNX Runtime library for the ORT worker")
set(ONNXSIM_ORT_PUBLIC_INCLUDE_DIR "" CACHE PATH
    "ONNX Runtime public include directory for the OrtEp plugin")
set(ONNXSIM_ORT_PUBLIC_LIBRARY "" CACHE FILEPATH
    "Optional ONNX Runtime library to link into the OrtEp plugin")
set(DORA_NODE_API_INCLUDE_DIR "" CACHE PATH
    "DORA C node API include directory (contains node_api.h)")
set(DORA_NODE_API_LIBRARY "" CACHE FILEPATH
    "DORA C node API static library")

if (ONNXSIM_REMOTE_DORA_STUB)
  if (NOT ONNXSIM_REMOTE_DORA)
    message(FATAL_ERROR "ONNXSIM_REMOTE_DORA_STUB requires ONNXSIM_REMOTE_DORA=ON")
  endif()
  add_library(onnx_remote_dora_api_stub STATIC
              test/dora_node_api_stub/node_api.cpp)
  target_include_directories(onnx_remote_dora_api_stub PRIVATE
                             "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(onnx_remote_dora_api_stub PUBLIC onnx_remote_transport)
  set(DORA_NODE_API_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/test/dora_node_api_stub")
  set(DORA_NODE_API_LIBRARY onnx_remote_dora_api_stub)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(onnx_remote_transport STATIC remote_transport.cpp remote_profile.cpp
            remote_capabilities.cpp)
target_include_directories(onnx_remote_transport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (ONNXSIM_REMOTE_ORT_PUBLIC_PLUGIN)
  set_target_properties(onnx_remote_transport PROPERTIES
                        POSITION_INDEPENDENT_CODE ON)
endif()
if (WIN32)
  target_link_libraries(onnx_remote_transport PUBLIC ws2_32)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(onnx_remote_transport PUBLIC Threads::Threads)
endif()
install(TARGETS onnx_remote_transport ARCHIVE DESTINATION lib)

if (ONNXSIM_REMOTE_ORT_PUBLIC_PLUGIN)
  if (NOT ONNXSIM_ORT_PUBLIC_INCLUDE_DIR OR
      NOT EXISTS "${ONNXSIM_ORT_PUBLIC_INCLUDE_DIR}/onnxruntime_cxx_api.h" OR
      NOT EXISTS "${ONNXSIM_ORT_PUBLIC_INCLUDE_DIR}/onnxruntime_ep_c_api.h")
    message(FATAL_ERROR
      "ONNXSIM_REMOTE_ORT_PUBLIC_PLUGIN requires a public ORT include "
      "directory containing onnxruntime_cxx_api.h and onnxruntime_ep_c_api.h")
  endif()
  add_library(onnxsim_remote_ep SHARED ort_plugin/public_remote_ep.cpp)
  target_compile_definitions(onnxsim_remote_ep PRIVATE ORT_API_MANUAL_INIT)
  target_include_directories(onnxsim_remote_ep PRIVATE
                             ${ONNXSIM_ORT_PUBLIC_INCLUDE_DIR})
  target_link_libraries(onnxsim_remote_ep PRIVATE onnx_remote_transport)
  if (ONNXSIM_ORT_PUBLIC_LIBRARY)
    target_link_libraries(onnxsim_remote_ep PRIVATE
                          ${ONNXSIM_ORT_PUBLIC_LIBRARY})
    add_executable(onnxsim_remote_ep_load_test
                   ort_plugin/public_ep_load_test.cpp)
    add_executable(onnxsim_remote_ep_run_test
                   ort_plugin/public_ep_run_test.cpp)
    target_include_directories(onnxsim_remote_ep_load_test PRIVATE
                               ${ONNXSIM_ORT_PUBLIC_INCLUDE_DIR})
    target_include_directories(onnxsim_remote_ep_run_test PRIVATE
                               ${ONNXSIM_ORT_PUBLIC_INCLUDE_DIR})
    target_link_libraries(onnxsim_remote_ep_load_test PRIVATE
                          ${ONNXSIM_ORT_PUBLIC_LIBRARY})
    target_link_libraries(onnxsim_remote_ep_run_test PRIVATE
                          ${ONNXSIM_ORT_PUBLIC_LIBRARY})
    add_test(NAME onnxsim-remote-public-ep-load
             COMMAND onnxsim_remote_ep_load_test
                     $<TARGET_FILE:onnxsim_remote_ep>)
  endif()
  set_target_properties(onnxsim_remote_ep PROPERTIES OUTPUT_NAME onnxsim_remote_ep)
  install(TARGETS onnxsim_remote_ep LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
endif()

add_executable(onnx-remote-worker remote_worker.cpp)
target_link_libraries(onnx-remote-worker PRIVATE onnx_remote_transport)

if (ONNXSIM_REMOTE_ORT_WORKER)
  if (NOT ONNXSIM_ORT_WORKER_INCLUDE_DIR OR
      NOT EXISTS "${ONNXSIM_ORT_WORKER_INCLUDE_DIR}/onnxruntime_cxx_api.h" OR
      NOT ONNXSIM_ORT_WORKER_LIBRARY)
    message(FATAL_ERROR
      "ONNXSIM_REMOTE_ORT_WORKER requires the ORT public headers and library")
  endif()
  add_executable(onnx-remote-ort-worker remote_ort_worker.cpp)
  add_executable(onnx-remote-ort-worker-test remote_ort_worker_test.cpp)
  target_include_directories(onnx-remote-ort-worker PRIVATE
                             ${ONNXSIM_ORT_WORKER_INCLUDE_DIR})
  target_link_libraries(onnx-remote-ort-worker PRIVATE
                        onnx_remote_transport ${ONNXSIM_ORT_WORKER_LIBRARY})
  target_include_directories(onnx-remote-ort-worker-test PRIVATE
                             ${ONNXSIM_ORT_WORKER_INCLUDE_DIR})
  target_link_libraries(onnx-remote-ort-worker-test PRIVATE
                        onnx_remote_transport ${ONNXSIM_ORT_WORKER_LIBRARY})
  install(TARGETS onnx-remote-ort-worker RUNTIME DESTINATION bin)
  install(TARGETS onnx-remote-ort-worker-test RUNTIME DESTINATION bin)
endif()

add_executable(onnx-remote-client remote_client.cpp)
target_link_libraries(onnx-remote-client PRIVATE onnx_remote_transport)

add_executable(onnx-remote-compiler remote_compiler.cpp)
target_link_libraries(onnx-remote-compiler PRIVATE onnx_remote_transport)

add_executable(onnx-remote-mock-runner remote_mock_runner.cpp)
target_link_libraries(onnx-remote-mock-runner PRIVATE onnx_remote_transport)

add_executable(onnx-remote-attach-test remote_attach_test.cpp)
target_link_libraries(onnx-remote-attach-test PRIVATE onnx_remote_transport)

# These dependency-free binaries form the host-side compiler/runner bundle:
# the compiler service produces artifacts, the mock runner supports CI and
# constrained-host bring-up, and the client/attach test provide diagnostics
# without requiring Python or gRPC.
install(TARGETS onnx-remote-worker onnx-remote-client onnx-remote-compiler
                onnx-remote-mock-runner onnx-remote-attach-test
        RUNTIME DESTINATION bin)

add_executable(onnx-remote-profile-dump remote_profile_dump.cpp)
target_link_libraries(onnx-remote-profile-dump PRIVATE onnx_remote_transport)
install(TARGETS onnx-remote-profile-dump RUNTIME DESTINATION bin)

add_executable(onnx-remote-profile-fixture remote_profile_fixture.cpp)
target_link_libraries(onnx-remote-profile-fixture PRIVATE onnx_remote_transport)

# The schema is source-only and does not pull protobuf or gRPC into the
# default build.  Install it so an optional host-side gateway can generate
# bindings with the toolchain available on that host.
install(FILES proto/onnxsim_remote.proto proto/README.md
        DESTINATION share/onnx-remote/proto)
install(FILES remote_transport.h remote_profile.h remote_capabilities.h
        DESTINATION include/onnx-remote)
install(FILES ort_plugin/onnxsim_remote_ep.manifest.json ort_plugin/README.md
        DESTINATION share/onnx-remote/ort-plugin)

if (ONNXSIM_REMOTE_GRPC)
  # Prefer a package config when the host distribution provides one, but also
  # support CMake's FindProtobuf module (common on Debian/Ubuntu and ROS
  # workspaces).
  find_package(absl CONFIG REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(Protobuf CONFIG QUIET)
  if (NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
  endif()
  find_package(gRPC CONFIG REQUIRED)

  if (TARGET protobuf::protoc)
    set(ONNXSIM_PROTOC "$<TARGET_FILE:protobuf::protoc>")
  else()
    find_program(ONNXSIM_PROTOC protoc REQUIRED)
  endif()
  if (TARGET gRPC::grpc_cpp_plugin)
    set(ONNXSIM_GRPC_CPP_PLUGIN "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")
  else()
    find_program(ONNXSIM_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
  endif()

  set(ONNXSIM_GRPC_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc")
  set(ONNXSIM_GRPC_PROTO_CPP "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.pb.cc")
  set(ONNXSIM_GRPC_PROTO_H "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.pb.h")
  set(ONNXSIM_GRPC_SERVICE_CPP "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.grpc.pb.cc")
  set(ONNXSIM_GRPC_SERVICE_H "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.grpc.pb.h")
  add_custom_command(
    OUTPUT ${ONNXSIM_GRPC_PROTO_CPP} ${ONNXSIM_GRPC_PROTO_H}
           ${ONNXSIM_GRPC_SERVICE_CPP} ${ONNXSIM_GRPC_SERVICE_H}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ONNXSIM_GRPC_GEN_DIR}
    COMMAND ${ONNXSIM_PROTOC}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
            --cpp_out=${ONNXSIM_GRPC_GEN_DIR}
            --grpc_out=${ONNXSIM_GRPC_GEN_DIR}
            --plugin=protoc-gen-grpc=${ONNXSIM_GRPC_CPP_PLUGIN}
            onnxsim_remote.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/onnxsim_remote.proto
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/proto
    VERBATIM)
  add_library(onnx_remote_grpc STATIC
              ${ONNXSIM_GRPC_PROTO_CPP} ${ONNXSIM_GRPC_SERVICE_CPP})
  target_include_directories(onnx_remote_grpc PUBLIC ${ONNXSIM_GRPC_GEN_DIR})
  target_link_libraries(onnx_remote_grpc PUBLIC protobuf::libprotobuf gRPC::grpc++)
  add_executable(onnx-remote-grpc-gateway grpc_gateway.cpp grpc_gateway_main.cpp)
  target_include_directories(onnx-remote-grpc-gateway PRIVATE ${ONNXSIM_GRPC_GEN_DIR})
  target_link_libraries(onnx-remote-grpc-gateway PRIVATE
                        onnx_remote_transport onnx_remote_grpc gRPC::grpc++)
  if (BUILD_TESTING)
    add_executable(onnx-remote-grpc-gateway-test grpc_gateway.cpp
                   grpc_gateway_test.cpp)
    target_include_directories(onnx-remote-grpc-gateway-test PRIVATE
                               ${ONNXSIM_GRPC_GEN_DIR})
    target_link_libraries(onnx-remote-grpc-gateway-test PRIVATE
                          onnx_remote_transport onnx_remote_grpc gRPC::grpc++)
    add_test(NAME onnx-remote-grpc-gateway-smoke
             COMMAND onnx-remote-grpc-gateway-test)
  endif()
  install(TARGETS onnx-remote-grpc-gateway RUNTIME DESTINATION bin)
  install(TARGETS onnx_remote_grpc ARCHIVE DESTINATION lib)
  install(FILES ${ONNXSIM_GRPC_PROTO_H} ${ONNXSIM_GRPC_SERVICE_H}
          DESTINATION include/onnx-remote/grpc)
endif()

if (BUILD_TESTING)
  add_executable(onnx-remote-op-mapping-test remote_op_mapping_test.cpp)
  target_link_libraries(onnx-remote-op-mapping-test PRIVATE onnx_remote_transport)
  add_test(NAME onnx-remote-op-mapping COMMAND onnx-remote-op-mapping-test)

  add_executable(onnx-remote-transport-test remote_transport_test.cpp)
  target_link_libraries(onnx-remote-transport-test PRIVATE onnx_remote_transport)
  add_test(NAME onnx-remote-transport-roundtrip
           COMMAND onnx-remote-transport-test)

  # The VIPLite worker's half-float and quantize/dequantize routines; header-only, so no SDK or device is needed.
  add_executable(onnx-remote-viplite-convert-test viplite_convert_test.cpp)
  add_test(NAME onnx-remote-viplite-convert
           COMMAND onnx-remote-viplite-convert-test)
endif()

if (ONNX_REMOTE_AXCL)
  find_path(AXCL_INCLUDE_DIR axcl.h REQUIRED)
  find_library(AXCL_RT_LIBRARY axcl_rt REQUIRED)
  find_library(AXCL_SYS_LIBRARY axcl_sys REQUIRED)
  add_executable(onnx-remote-axcl-worker remote_axcl_worker.cpp)
  target_include_directories(onnx-remote-axcl-worker PRIVATE ${AXCL_INCLUDE_DIR})
  target_link_libraries(onnx-remote-axcl-worker PRIVATE
                        onnx_remote_transport ${AXCL_RT_LIBRARY} ${AXCL_SYS_LIBRARY})
endif()

if (ONNX_REMOTE_VIPLITE)
  if (NOT VIPLITE_INCLUDE_DIR OR NOT EXISTS "${VIPLITE_INCLUDE_DIR}/vip_lite.h" OR
      NOT VIPLITE_LIBRARY)
    message(FATAL_ERROR
      "ONNX_REMOTE_VIPLITE requires VIPLITE_INCLUDE_DIR (with vip_lite.h) "
      "and VIPLITE_LIBRARY")
  endif()
  add_executable(onnx-remote-viplite-worker remote_viplite_worker.cpp)
  target_include_directories(onnx-remote-viplite-worker PRIVATE
                             ${VIPLITE_INCLUDE_DIR})
  target_link_libraries(onnx-remote-viplite-worker PRIVATE
                        onnx_remote_transport ${VIPLITE_LIBRARY})
  install(TARGETS onnx-remote-viplite-worker RUNTIME DESTINATION bin)
endif()

if (ONNXSIM_REMOTE_DORA)
  if (NOT DORA_NODE_API_INCLUDE_DIR OR NOT DORA_NODE_API_LIBRARY)
    message(FATAL_ERROR
      "ONNXSIM_REMOTE_DORA requires DORA_NODE_API_INCLUDE_DIR and "
      "DORA_NODE_API_LIBRARY")
  endif()
  add_executable(onnx-remote-dora-node dora_node.cpp)
  target_include_directories(onnx-remote-dora-node PRIVATE
                             ${DORA_NODE_API_INCLUDE_DIR})
  target_link_libraries(onnx-remote-dora-node PRIVATE
                        onnx_remote_transport ${DORA_NODE_API_LIBRARY})
  if (UNIX AND NOT APPLE)
    target_link_libraries(onnx-remote-dora-node PRIVATE dl m rt pthread)
  endif()
  if (BUILD_TESTING AND ONNXSIM_REMOTE_DORA_STUB)
    add_executable(onnx-remote-dora-status-test test/dora_status_test.cpp)
    target_include_directories(onnx-remote-dora-status-test PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME onnx-remote-dora-status
             COMMAND onnx-remote-dora-status-test)
  endif()
endif()

if (ONNXSIM_REMOTE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  add_executable(onnx-remote-ros2-bridge ros2_bridge.cpp)
  add_executable(onnx-remote-ros2-service-smoke ros2_service_smoke.cpp)
  add_executable(onnx-remote-ros2-discovery-smoke ros2_discovery_smoke.cpp)
  add_executable(onnx-remote-ros2-failover-smoke ros2_failover_smoke.cpp)
  # ament_target_dependencies uses CMake's plain target_link_libraries
  # signature, so keep this target on the same signature for compatibility
  # with ament-cmake releases used by the Nix ROS2 overlay.
  target_link_libraries(onnx-remote-ros2-bridge onnx_remote_transport)
  ament_target_dependencies(onnx-remote-ros2-bridge rclcpp std_msgs std_srvs)
  ament_target_dependencies(onnx-remote-ros2-service-smoke rclcpp std_srvs)
  ament_target_dependencies(onnx-remote-ros2-discovery-smoke rclcpp std_msgs)
  ament_target_dependencies(onnx-remote-ros2-failover-smoke rclcpp std_msgs)
  install(TARGETS onnx-remote-ros2-bridge
          DESTINATION lib/${PROJECT_NAME})
  install(TARGETS onnx-remote-ros2-service-smoke
          DESTINATION lib/${PROJECT_NAME})
  install(TARGETS onnx-remote-ros2-discovery-smoke
          DESTINATION lib/${PROJECT_NAME})
  install(TARGETS onnx-remote-ros2-failover-smoke
          DESTINATION lib/${PROJECT_NAME})
  ament_package()
endif()
