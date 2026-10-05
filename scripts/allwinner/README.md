# Allwinner NPU (Vivante VIP9000): RPC runner and ONNX -> NBG converter

Targets the NPU in Allwinner A733 / T736 (VIP9000NANODI_PLUS, hardware ID `0x1000003b`) and the same VIPLite runtime on T527,
V853 and friends. Two pieces, both plugged into the existing `onnx-remote` compiler/runner split (`tools/onnx-remote`,
`docs/rpc.md`):

| piece | where | what |
|---|---|---|
| **runner** | `tools/onnx-remote/remote_viplite_worker.cpp` | on-device server: `load_compiled` / `run_compiled` for NBG artifacts through VIPLite (`libNBGlinker.so`) |
| **converter** | `scripts/allwinner/compile_nbg.py` | `onnx-remote-compiler --command`: ONNX -> `.nb` by driving Acuity's `pegasus` |

## Status: what is verified and what is not

**Verified on hardware** (iPlay 70 S tablet, A733, Android 15, adb shell, no root): the runner builds with the NDK, opens
`/dev/vipcore` as the unprivileged shell user, loads and runs Allwinner's shipped `.nb` files, and serves
`capabilities` / `load_compiled` / `run_compiled` over TCP to the stock `onnx-remote-client` and `onnx-remote-compiler`
(passthrough mode). Bad artifacts and wrong input sizes come back as errors without killing the worker.

**Not verified: the converter against real Acuity.** Acuity is not publicly downloadable (Allwinner distributes it as a Docker image
through its customer portal), so `compile_nbg.py` has only been exercised against a fake `pegasus`
(`tests/test_allwinner_compile_nbg.py`: step order, flags, per-SoC `--optimize` target, calibration hand-off, manifest, errors). The
command lines mirror Allwinner's `awnpu_model_zoo` scripts, but real-toolkit behaviour -- in particular the `acuitylib` calls in
`acuity_inputmeta.py` and `.npy` calibration files in a `TEXT` dataset -- is untested. Expect to adjust on first contact.

There is no from-scratch ONNX -> NBG compiler here: an NBG is machine code for the NPU and only Acuity emits it.

Measured on the A733 (NPU = `vip_run_network`; "IO" = float32 quantize/dequantize in the worker):

| model | NPU | IO in+out |
|---|---|---|
| deepHeadPose uint8 (112x336) | 0.35 ms | 0.09 ms |
| RetinaFace pcq (640x640) | 8.2 ms | 2.9 ms |
| YOLOv5n uint8 (640x640) | 10.2 ms | 8.3 ms |
| YOLOv5s uint8, from the zoo (640x640) | 23 ms | 30 ms |

Feeding a uint8 network its native `UINT8` tensor skips input quantization. Over `adb forward` the float32 tensors dominate
wall time for large models (YOLOv5s: ~440 ms RPC-inclusive); run the client on the device or on the same LAN for real numbers.

## Build and deploy the runner

The VIPLite headers and `libNBGlinker.so` come from Allwinner's public model zoo
(`https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz`, `common/npuruntime/`); they are not vendored here.

```bash
ZOO=awnpu_model_zoo-*/common/npuruntime
cmake -S tools/onnx-remote -B build-aw -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
  -DBUILD_TESTING=OFF -DONNX_REMOTE_VIPLITE=ON \
  -DVIPLITE_INCLUDE_DIR=$ZOO/include -DVIPLITE_LIBRARY=$ZOO/lib_android/A733/arm64-v8a/libNBGlinker.so
cmake --build build-aw --target onnx-remote-viplite-worker      # Release matters: -O0 is ~10x slower in the conversion loops

adb push build-aw/onnx-remote-viplite-worker model.nb /data/local/tmp/viplite/
adb shell /data/local/tmp/viplite/onnx-remote-viplite-worker --bench /data/local/tmp/viplite/model.nb 20   # smoke test, no network
adb shell "cd /data/local/tmp/viplite && nohup ./onnx-remote-viplite-worker --port 39503 --cache-dir ./cache >worker.log 2>&1 &"
adb forward tcp:39503 tcp:39503
```

At link time `libNBGlinker.so` is the zoo's copy; at run time the device's own `/system/lib64/libNBGlinker.so` is used. For older
SoCs (T527, ...) link their `libVIPlite.so` instead.

The worker listens on **127.0.0.1** by default (`--host 0.0.0.0` to change): it submits whatever NBG a client sends to the NPU.

## Run a model

```bash
onnx-remote-client --capabilities 127.0.0.1 39503          # hardware cid, driver version
# compile (Acuity host) -> load_compiled -> run_compiled, against separate endpoints; ONNX -> .nb needs Acuity:
onnx-remote-compiler --port 39510 --cache-dir ~/.cache/onnxsim-aw --target allwinner-a733 --compiler-id "$ACUITY_VERSION" \
  --command 'python3 scripts/allwinner/compile_nbg.py {input} {output} {manifest} --platform a733 --quant pcq --calib-dir calib/'
onnx-remote-client --compile-run 127.0.0.1 39510 127.0.0.1 39503 model.onnx --input-raw 1:1,3,224,224:input.f32 --iters 10 --profile
# Already have an .nb? Compiler passthrough mode (no --command) accepts it as the "model":
onnx-remote-compiler --port 39510 --cache-dir cc --target allwinner-a733 &
onnx-remote-client --compile-run 127.0.0.1 39510 127.0.0.1 39503 model.nb --input-raw 1:1,3,224,224:input.f32
```

I/O contract: `run_compiled` takes the model's inputs in graph order and returns every output as FLOAT in ONNX (NCHW) order. The
worker quantizes/dequantizes using the scale/zero-point (or fixed-point position) stored in the NBG. An input already in the
network's native integer format is copied through. `--profile` reports `viplite_run` (NPU time), plus input/output conversion
with `Detailed`.

## Converter

```
compile_nbg.py INPUT.onnx OUTPUT.nb MANIFEST.json [--platform a733] [--quant pcq|uint8|int16|bf16|float]
               [--calib-dir DIR] [--calib-count N] [--input-shape NAME:1,3,224,224] [--no-simplify]
               [--docker-image IMAGE] [--inputmeta-args ' --preproc IMAGE_RGB --mean 0,0,0 --scale 0.0039216']
```

Steps (the zoo's `pegasus_import.sh`, `pegasus_quantize.sh`, `pegasus_export_ovx_nbg.sh`): onnxsim + static-shape check ->
`pegasus import onnx` -> `generate inputmeta` / `postprocess-file` -> `acuity_inputmeta.py` -> `quantize` ->
`export ovxlib --pack-nbg-unify --optimize VIP9000NANODI_PLUS_PID0X1000003B` -> `network_binary.nb`. Environment: either a native
Acuity install (`ACUITY_PATH`, `VIV_SDK`) or `--docker-image` (`AW_NPU_DOCKER_IMAGE`; the image presets those itself).

Limits: single-input models only when quantizing (calibration is one dataset; use `--quant float` for multi-input); Acuity needs
static shapes; the default keeps ONNX semantics (no baked-in image preprocessing, `TENSOR` input) -- pass `--preproc IMAGE_RGB`
to get a uint8 camera-frame input instead. The SoC -> `--optimize` table is copied from the zoo's script; `a733` and `t736` share a
target, and the A733's reported hardware ID (`0x1000003b`) matches it.
