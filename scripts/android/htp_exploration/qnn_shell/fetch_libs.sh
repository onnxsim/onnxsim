#!/bin/bash
# Download the exact ORT + QNN EP + Qualcomm QNN runtime libraries qnn_run needs into ./libs/
# (not committed). Versions must line up:
#   - onnxruntime-android 1.26.0      host ORT (plugin-EP C API: RegisterExecutionProviderLibrary)
#   - onnxruntime-android-qnn 2.6.0   the QNN EP plugin; built against QNN 2.50.x
#                                     (onnxruntime-qnn 2.6.0 wheel: qnn_version = '2.50.40')
#   - com.qualcomm.qti:qnn-runtime 2.50.0   libQnnHtp/libQnnSystem/libQnnHtpPrepare + the
#                                     V69 stub (ARM) and V69 skel (DSP side, loaded unsigned)
set -euo pipefail
cd "$(dirname "$0")"
M=https://repo1.maven.org/maven2
mkdir -p dl libs
fetch() { [ -f "dl/$2" ] || curl -sfL -o "dl/$2" "$1"; }
fetch "$M/com/microsoft/onnxruntime/onnxruntime-android/1.26.0/onnxruntime-android-1.26.0.aar" ort.aar
# the QNN EP plugin is published by Qualcomm (group com.qualcomm.qti), not com.microsoft
fetch "$M/com/qualcomm/qti/onnxruntime-android-qnn/2.6.0/onnxruntime-android-qnn-2.6.0.aar" ort-qnn.aar
fetch "$M/com/qualcomm/qti/qnn-runtime/2.50.0/qnn-runtime-2.50.0.aar" qnn-runtime.aar
rm -rf dl/x && mkdir dl/x
unzip -q -o dl/ort.aar 'jni/arm64-v8a/libonnxruntime.so' 'headers/*' -d dl/x/ort
unzip -q -o dl/ort-qnn.aar 'jni/arm64-v8a/*' -d dl/x/ortqnn
unzip -q -o dl/qnn-runtime.aar 'jni/arm64-v8a/*' -d dl/x/qnn
cp dl/x/ort/jni/arm64-v8a/libonnxruntime.so dl/x/ortqnn/jni/arm64-v8a/libonnxruntime_providers_qnn.so libs/
for l in libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so libQnnHtpV69Stub.so libQnnHtpV69Skel.so; do
  cp "dl/x/qnn/jni/arm64-v8a/$l" libs/
done
rm -rf headers && cp -r dl/x/ort/headers headers
ls -la libs
