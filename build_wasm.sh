#!/usr/bin/env bash
set -ex

# Check if emcmake is available
command -v emcmake

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WITH_NODE_RAW_FS=${1:-OFF}
# ORT_WEB=ON builds the experimental variant that delegates constant folding to
# the page's onnxruntime-web instead of compiling ONNX Runtime into the module
# (see docs/wasm_ort_web.md). It needs no ONNX Runtime source and no ORT compile.
ORT_WEB=${ORT_WEB:-OFF}

cd $SCRIPT_DIR
if which protoc ; then
    PROTOC=$(which protoc)
else
    . ./third_party/onnx/workflow_scripts/protobuf/build_protobuf_unix.sh $(nproc) $PWD/protobuf
    PROTOC=$(which protoc)
fi

set -u -o pipefail

ORT_VER=1.29.0
# The ORT-web variant links no ONNX Runtime, so the source tarball is not needed.
if [ "$ORT_WEB" != "ON" ] && [ ! -d "third_party/onnxruntime-${ORT_VER}" ] ; then
    pushd third_party
    wget -q "https://github.com/microsoft/onnxruntime/archive/refs/tags/v${ORT_VER}.zip"
    unzip -q "v${ORT_VER}.zip"
    popd
fi

BUILD_DIR=build-wasm-node-$WITH_NODE_RAW_FS
ORT_WEB_CMAKE=()
if [ "$ORT_WEB" = "ON" ] ; then
    # Separate build dir so the two variants can coexist without reconfiguring.
    BUILD_DIR=${BUILD_DIR}-ortweb
    ORT_WEB_CMAKE=(-DONNXSIM_WASM_ORT_WEB=ON)
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
emcmake cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-sNO_DISABLE_EXCEPTION_CATCHING" "${ORT_WEB_CMAKE[@]}" ..
cmake --build . -t onnxsim_bin
