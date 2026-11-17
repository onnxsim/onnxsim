# Real-device testing on a Radxa Dragon Q8B with an Axelera Metis AIPU

This runbook covers bringing up an aarch64 Ubuntu host with an Axelera Metis
AIPU, so the Voyager SDK compiler tests in `tests/test_axelera_voyager_*.py`
and `tests/test_axelera_legalize.py` can run against onnxsim's own build.

`scripts/axelera/q8b_device_setup.sh` automates the steps below. This document
explains them, records the problems found, and lists the workarounds that are
**not** in the script because they patch Voyager SDK source.

## Tested configuration

| Item | Value |
| --- | --- |
| Board | Radxa Dragon Q8B (Qualcomm SC8280XP, aarch64) |
| OS / kernel | Ubuntu 26.04 (resolute), kernel 7.0.11-qcom |
| Accelerator | Axelera AI Metis AIPU, PCIe `0002:01:00.0` |
| Metis driver | `metis-dkms` 1.7.16 (Axelera apt, `ubuntu24` suite) |
| Voyager SDK | 1.9.0 |
| Python for SDK | 3.12.16 (SDK supports 3.10-3.13; the system has 3.14 only) |
| Toolchain | GCC 15, system OpenCV 4, cmake 4.2.3, ninja, ccache 4.12.3 |

Axelera's docs list Ubuntu 22.04/24.04 and kernels around 5.15/6.x. This
kernel and OS combination worked, but it is outside what they document.

## 1. Metis driver

Axelera publishes the driver as `metis-dkms`. The apt suite published at the
time of writing is `ubuntu24`; there is no `ubuntu26` suite.

```bash
sudo sh -c "curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public | gpg --dearmor -o /etc/apt/keyrings/axelera.gpg"
sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu24 main' > /etc/apt/sources.list.d/axelera.list"
sudo apt-get update
sudo apt-get install -y metis-dkms linux-headers-$(uname -r)
sudo modprobe metis
cat /sys/class/metis/version        # 1.7.16
```

DKMS built the module for both installed kernels without changes.

**Device node permissions.** The apt package installs
`/etc/udev/rules.d/72-axelera.rules`, which sets group `axelera` on
`/dev/metis-*`. On this board the rule did not apply to a node created before
the package was installed, and `udevadm trigger --action=add` did not fix
that. Options:

- reboot, which should apply the rule at device creation, or
- `sudo chgrp axelera /dev/metis-2-1-0 && sudo chmod 660 /dev/metis-2-1-0`
  (not persistent), then `sudo usermod -aG axelera $USER` and log in again.

## 2. Voyager SDK 1.9.0

```bash
git clone --depth 1 https://github.com/axelera-ai-hub/voyager-sdk.git ~/voyager-sdk
cd ~/voyager-sdk
mkdir -p ~/pyshim && ln -sf "$(uv python find 3.12)" ~/pyshim/python3   # Python 3.12 via uv
PATH=$HOME/pyshim:$PATH ./quick-start.sh
axdevice                          # Device 0: metis-2:1:0 ... flver=1.8.0
```

`quick-start.sh` calls plain `python3`, so the 3.12 shim has to come first on
`PATH`. On this board the SDK's operator build failed; see
[Known SDK issues](#known-sdk-issues-voyager-sdk-190).

Two packages needed separately on this image: `python3.14-venv` (for
`uv`-created venvs) and `uv` itself (installed from PyPI into a venv).

## 3. onnxsim wheel (ccache + Ninja)

`setup.py` has no generator flag. CMake picks up `CMAKE_GENERATOR`, and
`CMAKE_ARGS` is appended to the configure command.

```bash
cd <onnxsim checkout>
git submodule update --init --recursive   # onnx-optimizer carries its own onnx and blake3
export PATH=/usr/bin:/bin:/usr/sbin:/sbin # keep the SDK venv's cmake script off PATH
export CMAKE_GENERATOR=Ninja
export CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
export CCACHE_DIR=$HOME/.cache/ccache
~/voyager-sdk/venv_1.9.0/bin/python3 -m pip wheel . --no-deps -w ~/wheels
~/voyager-sdk/venv_1.9.0/bin/python3 -m pip install --force-reinstall --no-deps ~/wheels/onnxsim-*.whl
```

Cold build: 7m08s on 8 cores, ccache hit rate 4% (26 / 622). A later build from
a fresh clone took 57s, but its ccache log showed 11 cacheable calls and 0 hits,
so the speedup is not explained by ccache. Not investigated further.

Two build failures on this host and their causes:

- `ModuleNotFoundError: No module named 'cmake'` and a 2-second failure: `setup.py`
  resolves `cmake` with `shutil.which`, which found the SDK venv's cmake
  script. That script needs its own Python module, which pip's build isolation
  does not provide. Fix: keep `/usr/bin` ahead of the venv on `PATH`.
- `Could not find ... third_party/onnx/cmake/Utils.cmake`: `onnx-optimizer`
  includes `${PROJECT_SOURCE_DIR}/third_party/onnx/cmake/Utils.cmake`, so it
  needs its own nested `onnx` and `blake3` submodules. Use
  `git submodule update --init --recursive`.

## 4. Device tests

```bash
source ~/voyager-sdk/venv_1.9.0/bin/activate
cd <onnxsim checkout>
# Use `pytest`, not `python -m pytest`: -m puts the repo root on sys.path,
# so `import onnxsim` loads the source tree (no compiled extension) and fails.
pytest tests/test_axelera_voyager_real_compiler.py -v
pytest tests/test_axelera_legalize.py tests/test_axelera_voyager_op_support.py tests/test_voyager_sdk_patterns.py -q
```

Results on this board, with the wheel from step 3:

- `test_axelera_voyager_real_compiler.py`: 3 passed. These run through
  `axelera.compiler.quantize()`. One of them checks that a Conv with
  `auto_pad=SAME_UPPER` fails quantization, matching the rule in
  `voyager_op_support_data.py`.
- `test_axelera_legalize.py`, `test_axelera_voyager_op_support.py`,
  `test_voyager_sdk_patterns.py`: 45 passed, no skips. These do not need the
  device.

Smoke inference through the SDK (`inference.py yolov8n-coco <image> -o
result.jpg`) detected real COCO objects with plausible confidences, so the
device path works end to end. Device-side inference time was about 1.85 ms per
frame.

## Known SDK issues (Voyager SDK 1.9.0)

These are problems in `axelera-ai-hub/voyager-sdk`, not in onnxsim. They are
listed here so the next person does not have to rediscover them. The fixes
below were applied to a local clone only and are not upstream.

1. **Operator build fails with `-Werror`.** `operators/CMakeLists.txt` uses
   `-Wall -Werror`. System OpenCV 4 headers trigger
   `-Werror=overloaded-virtual` in `stitching/detail/exposure_compensate.hpp`.
   Workaround: add `-Wno-error=overloaded-virtual` to `add_compile_options`.
   Setting `CXXFLAGS` does not work here because the CMake configure step
   already runs with its own flags.
2. **Missing `#include <algorithm>`** in `operators/axstreamer/include/AxDataInterface.h`
   (`std::transform`). GCC 15 no longer pulls it in transitively.
3. **`Gst.init(None)` crashes with PyGObject 3.50.0.** `axelera/app/pipe/gst.py:490`
   and `axelera/app/pipe/gst_helper.py:491` pass `None`, which PyGObject 3.50
   rejects with `TypeError: Argument 1 does not allow None as a value`. The error
   surfaces only after the model has compiled. Workaround: `Gst.init([])`.
4. **GStreamer plugin path.** The operators build produces
   `operators/lib/libgstaxstreamer.so`, which the SDK finds only through
   `GST_PLUGIN_PATH` or `LD_LIBRARY_PATH`. Set both to `<voyager>/operators/lib`
   when running `inference.py`.
5. **Statistics read 0.** `--show-stats` reports `End-to-end: 0.0fps` and zero
   time for post-processing elements, even though detections are correct. The
   cause is not known. The inference element itself reports a time.

## Open items

- The Axelera apt repository has no `ubuntu26` suite. The `ubuntu24` package
  was used on Ubuntu 26.04.
- The SDK's own `quick-start.sh` does not complete on this image without the
  three source fixes above.
