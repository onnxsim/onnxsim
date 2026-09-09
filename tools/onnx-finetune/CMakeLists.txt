cmake_minimum_required(VERSION 3.18)
project(onnx-finetune CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# This tool links against a training-enabled ONNX Runtime build. There is no
# published package that includes the training C++ API (not the pip wheels,
# not the prebuilt release tarballs) -- it must be built from source with
# --enable_training_apis --build_shared_lib. See README.md for the exact
# command, and note that `cmake --install` is not part of that recipe (the
# training headers under orttraining/ are not guaranteed to land in an
# install tree), so this points directly at the source checkout and the
# build output instead of an installed prefix:
#   cmake -B build \
#     -DORT_SOURCE_DIR=/path/to/onnxruntime \
#     -DORT_BUILD_DIR=/path/to/onnxruntime/build/Release
if(NOT ORT_SOURCE_DIR)
  message(FATAL_ERROR "ORT_SOURCE_DIR is not set -- point it at the onnxruntime source checkout used for the training build. See README.md.")
endif()
if(NOT ORT_BUILD_DIR)
  message(FATAL_ERROR "ORT_BUILD_DIR is not set -- point it at that build's output directory (containing libonnxruntime.so). See README.md.")
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
