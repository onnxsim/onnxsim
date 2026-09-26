cmake_minimum_required(VERSION 3.16)
project(onnx_remote_transport LANGUAGES CXX)

include(CTest)

option(ONNX_REMOTE_AXCL "Build the AXCL-backed remote worker" OFF)
option(ONNXSIM_REMOTE_DORA "Build the DORA node adapter" OFF)
option(ONNXSIM_REMOTE_ROS2 "Build the optional ROS2 binary bridge" OFF)
set(DORA_NODE_API_INCLUDE_DIR "" CACHE PATH
    "DORA C node API include directory (contains node_api.h)")
set(DORA_NODE_API_LIBRARY "" CACHE FILEPATH
    "DORA C node API static library")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(onnx_remote_transport STATIC remote_transport.cpp)
target_include_directories(onnx_remote_transport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (WIN32)
  target_link_libraries(onnx_remote_transport PUBLIC ws2_32)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(onnx_remote_transport PUBLIC Threads::Threads)
endif()

add_executable(onnx-remote-worker remote_worker.cpp)
target_link_libraries(onnx-remote-worker PRIVATE onnx_remote_transport)

add_executable(onnx-remote-client remote_client.cpp)
target_link_libraries(onnx-remote-client PRIVATE onnx_remote_transport)

add_executable(onnx-remote-compiler remote_compiler.cpp)
target_link_libraries(onnx-remote-compiler PRIVATE onnx_remote_transport)

add_executable(onnx-remote-mock-runner remote_mock_runner.cpp)
target_link_libraries(onnx-remote-mock-runner PRIVATE onnx_remote_transport)

add_executable(onnx-remote-attach-test remote_attach_test.cpp)
target_link_libraries(onnx-remote-attach-test PRIVATE onnx_remote_transport)

if (BUILD_TESTING)
  add_executable(onnx-remote-transport-test remote_transport_test.cpp)
  target_link_libraries(onnx-remote-transport-test PRIVATE onnx_remote_transport)
  add_test(NAME onnx-remote-transport-roundtrip
           COMMAND onnx-remote-transport-test)
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
endif()

if (ONNXSIM_REMOTE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  add_executable(onnx-remote-ros2-bridge ros2_bridge.cpp)
  target_link_libraries(onnx-remote-ros2-bridge PRIVATE onnx_remote_transport)
  ament_target_dependencies(onnx-remote-ros2-bridge rclcpp std_msgs std_srvs)
endif()
