cmake_minimum_required(VERSION 3.16)
project(onnx_remote_transport LANGUAGES CXX)

option(ONNX_REMOTE_AXCL "Build the AXCL-backed remote worker" OFF)

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

if (ONNX_REMOTE_AXCL)
  find_path(AXCL_INCLUDE_DIR axcl.h REQUIRED)
  find_library(AXCL_RT_LIBRARY axcl_rt REQUIRED)
  find_library(AXCL_SYS_LIBRARY axcl_sys REQUIRED)
  add_executable(onnx-remote-axcl-worker remote_axcl_worker.cpp)
  target_include_directories(onnx-remote-axcl-worker PRIVATE ${AXCL_INCLUDE_DIR})
  target_link_libraries(onnx-remote-axcl-worker PRIVATE
                        onnx_remote_transport ${AXCL_RT_LIBRARY} ${AXCL_SYS_LIBRARY})
endif()
