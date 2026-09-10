cmake_minimum_required(VERSION 3.18)
project(onnx-finetune CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# `onnx-finetune` (src/main.cpp) links onnxruntime's training C++ API, which
# needs a from-source --enable_training_apis build -- there is no published
# package that includes it (not the pip wheels, not the prebuilt release
# tarballs). See README.md for the exact command, and note that
# `cmake --install` is not part of that recipe (the training headers under
# orttraining/ are not guaranteed to land in an install tree), so this points
# directly at the source checkout and the build output instead of an
# installed prefix:
#   cmake -B build \
#     -DORT_SOURCE_DIR=/path/to/onnxruntime \
#     -DORT_BUILD_DIR=/path/to/onnxruntime/build/Release
if(ORT_SOURCE_DIR OR ORT_BUILD_DIR)
  if(NOT ORT_SOURCE_DIR)
    message(FATAL_ERROR "ORT_BUILD_DIR is set but ORT_SOURCE_DIR is not -- both are needed for onnx-finetune. See README.md.")
  endif()
  if(NOT ORT_BUILD_DIR)
    message(FATAL_ERROR "ORT_SOURCE_DIR is set but ORT_BUILD_DIR is not -- both are needed for onnx-finetune. See README.md.")
  endif()

  add_executable(onnx-finetune src/main.cpp)

  target_include_directories(onnx-finetune PRIVATE
    ${ORT_SOURCE_DIR}/include/onnxruntime/core/session
    ${ORT_SOURCE_DIR}/orttraining/orttraining/training_api/include
  )

  find_library(ORT_LIBRARY onnxruntime PATHS ${ORT_BUILD_DIR} REQUIRED NO_DEFAULT_PATH)
  target_link_libraries(onnx-finetune PRIVATE ${ORT_LIBRARY})
  set_target_properties(onnx-finetune PROPERTIES
    BUILD_RPATH ${ORT_BUILD_DIR}
    INSTALL_RPATH ${ORT_BUILD_DIR}
  )
endif()

# `onnx-finetune-distill-step-graph` (src/distill_step_graph_main.cpp) runs a
# step graph from scripts/generate_distillation_step_graph.py -- forward,
# loss, backward and an Adam update already baked into one ordinary ONNX
# graph by onnxsim's own graph_grad/qat_graph autodiff -- on a *plain*
# Ort::Session. It needs nothing but onnxruntime_cxx_api.h and
# libonnxruntime, so ORT_HOME can point at any onnxruntime distribution that
# has them: an official prebuilt release tarball
# (https://github.com/microsoft/onnxruntime/releases, no build at all), a
# `pip install onnxruntime`'s bundled shared library, or the same from-source
# checkout as above -- unlike onnx-finetune, this target never needs
# --enable_training_apis.
#   cmake -B build -DORT_HOME=/path/to/onnxruntime-linux-x64-1.19.2
if(ORT_HOME)
  add_executable(onnx-finetune-distill-step-graph src/distill_step_graph_main.cpp)
  target_include_directories(onnx-finetune-distill-step-graph PRIVATE ${ORT_HOME}/include)
  find_library(ORT_PLAIN_LIBRARY onnxruntime PATHS ${ORT_HOME}/lib REQUIRED NO_DEFAULT_PATH)
  target_link_libraries(onnx-finetune-distill-step-graph PRIVATE ${ORT_PLAIN_LIBRARY})
  set_target_properties(onnx-finetune-distill-step-graph PROPERTIES
    BUILD_RPATH ${ORT_HOME}/lib
    INSTALL_RPATH ${ORT_HOME}/lib
  )
endif()

if(NOT ORT_SOURCE_DIR AND NOT ORT_BUILD_DIR AND NOT ORT_HOME)
  message(FATAL_ERROR "Set either ORT_HOME (plain onnxruntime, for onnx-finetune-distill-step-graph) or both ORT_SOURCE_DIR and ORT_BUILD_DIR (a training-enabled build, for onnx-finetune). See README.md.")
endif()
