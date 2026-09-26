# tinygrad ahead-of-time OpenCL bundles for the Android demo app

The demo app (`../maskrcnn_demo_app`) runs its models through ORT + the QNN EP on the HTP. This directory
replaces that, per mode, with **tinygrad-generated OpenCL kernels on the Adreno GPU**, run without Python or
tinygrad on the phone at app run time:

1. `export_cl.py` lowers an ONNX model with tinygrad (`OnnxRunner` + `TinyJit`, our fork `onnxsim/tinygrad`) on an
   OpenCL device, captures the JIT (the ordered kernel calls, their launch sizes and the buffers they touch) and writes a
   bundle: `kernels.cl` (every kernel's OpenCL C), `plan.txt` (buffers + calls, format in the script's docstring),
   `consts.bin` (weights), `meta.txt` (input/output names, shapes, dtypes).
2. `tg_cl_runner.h` (header-only C++, dlopens the vendor `libOpenCL.so`) builds `kernels.cl` once (the device binary
   is cached next to the models), allocates the buffers (scratch buffers share one arena, first-fit by lifetime),
   runs the `init` calls once (weights -> fp16 images) and replays the per-frame calls.
3. The app's engines pick it with `engine=tinygrad` in their opts (`--es opts engine=tinygrad`); `engine=qnn` (the
   default) keeps the HTP path, so both stay comparable.

## Why the export runs on the phone

tinygrad's fast Adreno path is openpilot's: `IMAGE=1 FLOAT16=1` (weights and activations as fp16 `image2d_t`,
read through the texture cache). The kernels depend on the device (image pitch alignment, extensions), the capture
runs every kernel, and a desktop OpenCL can't stand in: NVIDIA's rejects half-float images from buffers. So the
export runs **on the phone**, with tinygrad's `DEV=CL` backend on the vendor OpenCL, under the official CPython
Android build (python.org `python-3.14.x-aarch64-linux-android.tar.gz`, plus a 3-line `Py_BytesMain` launcher).
The app itself still has no Python: it only reads the bundle. (`DEV=QCOM`, tinygrad's kgsl backend that openpilot
uses on its Adreno 630, refuses the Adreno 730 in this phone: `gpu_id >= (7, 3)` is unsupported.)

`--adreno` adds what the vendor compiler needs:
- `AdrenoCLRenderer`: `QCOMCLRenderer`'s dtype workarounds (half only with IMAGE+FLOAT16, bool buffers as uchar)
  with plain OpenCL C output.
- Every kernel is built once in a **child process** (`clc`) before tinygrad uses it. Qualcomm's compiler crashes on
  some kernels ("Custom lowering code for this instruction is not implemented yet: 150", seen on an unrolled 3x3
  conv loop with a masked image-row index; building it with `-cl-opt-disable` or rewriting one `(x&7)*360` as
  `mul24` avoids it) and after a crash **every later build in the same process fails** ("Program not built!"). A
  rejected kernel is retried with BEAM (the rejected candidates are skipped) and last with NOOPT.

## Setup (phone)

```bash
R=/data/local/tmp/codex-demo-app-tinygrad
curl -LO https://www.python.org/ftp/python/3.14.4/python-3.14.4-aarch64-linux-android.tar.gz   # -> prefix/lib
$NDK/aarch64-linux-android29-clang -I prefix/include/python3.14 -o python3 pymain.c -L prefix/lib -lpython3.14
# push prefix/lib (symlinks resolved, test/ dropped), python3, tinygrad/ (the fork), export_cl.py, clc to $R
PYTHONHOME=$R/py LD_LIBRARY_PATH=$R/py/lib PYTHONPATH=$R/tgpkg OPENCL_PATH=/vendor/lib64/libOpenCL.so \
  PARALLEL=0 TG_CLC=$R/clc DEV=CL IMAGE=1 FLOAT16=1 JIT_BATCH_SIZE=0 \
  $R/py/python3 export_cl.py yolo11n.onnx yolo11n.tg --u8-nhwc --adreno --check coco139_u8.bin
```

(`PARALLEL=0`: the Android build has no `_multiprocessing`.) `tg_cl_bench.cpp` runs a bundle standalone on the
phone (timing, per-kernel GPU profile, outputs to compare with the export's `ref_*.bin`).

## Results

Xiaomi 12S (Snapdragon 8+ Gen 1, Adreno 730), QNN baseline = the app's existing engine (ORT 1.26 + QNN EP 2.6.0, QNN
2.50, strict HTP, int8 QDQ model from `../deploy`, EP-context, burst). Agreement is `compare_yolo.py` on the 20
`eval` COCO images of `../deploy/models/yolo11n.yaml` (letterboxed 640, uint8): detections after the app's own
post-processing (conf 0.25, per-class NMS 0.7) matched to fp32 ONNX Runtime (same class, IoU >= 0.5).

| mode | engine | inference (app, images mode) | end-to-end FPS | matched / fp32 ORT | extra boxes |
|---|---|---:|---:|---:|---:|
| YOLO11n | QNN HTP, int8 | **2.58 ms** | **94** | 79 / 91 | 21 |
| YOLO11n | tinygrad, Adreno OpenCL, `IMAGE=1 FLOAT16=1` | 44.3 ms | 16.2 | **90 / 91** | **0** |
| YOLO26n | QNN HTP, int8 | **2.5 ms** (README above) | **100-111** | 68 / 78 | 7 |
| YOLO26n | tinygrad | 44.8 ms | 16.0 | **78 / 78** | **0** |
| YOLO26n-seg | QNN HTP, int8 | **3.4 ms** (README above) | **85** | see below | |
| YOLO26n-seg | tinygrad | 86-88 ms | 9.4 | **82 / 83** | **0** |
| YOLO11n-seg | QNN HTP, int8 | **3.4 ms** (README above) | **70** | see below | |
| YOLO11n-seg | tinygrad | 82-84 ms | 9.8 | **89 / 90** | **0** |

| RF-DETR-Nano @320 | QNN HTP, fp16 | **26.5 ms** (README above) | **25-26** | | |
| RF-DETR-Nano @320 | tinygrad | 267 ms | 3.3 | same 11 detections (> 0.5) as fp32 on COCO #139 | |

(YOLO26n uses the app's post=end2end: x1,y1,x2,y2, best class per anchor, top 300, no NMS. The -seg rows compare
boxes; the mask prototypes' worst max |diff| relative to fp32 over the 20 images is 1.9% for YOLO11n-seg and 10% for
YOLO26n-seg, fp16 on the GPU. The HTP -seg numbers in `../maskrcnn_demo_app/README.md` use `seg_check.py`'s own
matching: 69/84 and 78/91 fp32 detections.)

- tinygrad's fp16 GPU graph is much closer to fp32 than the int8 HTP graph (max score difference on matched boxes
  0.020 vs 0.352), but **17x slower** than the HTP: 122 kernel calls per frame, 38.8 ms of GPU time (the largest
  kernel 4.6 ms, the next ones 1.4-1.7 ms each). tinygrad's default heuristics are used; only the kernel the
  vendor compiler rejects is BEAM-searched (the full-model BEAM on the phone is the next step).
- The runner's output is bit-exact against tinygrad's own run on the phone (`ref_output0.bin`), and fp32 host
  tinygrad matches fp32 ORT to 7.7e-3 on 637-pixel box coordinates.
- Load: 2.5 s the first time (vendor compile of `kernels.cl`), 7 ms after (cached device binary). The export itself
  takes ~3 minutes on the phone (17 minutes the first time, with the one BEAM-searched kernel uncached).
- The -seg models need two more things: tinygrad's IMAGE rewrite crashed on their prototype upsampling conv
  (`_drop_valid_stmts` KeyError, fixed in onnxsim/tinygrad#10; `export_cl.py` also renders such a kernel on plain
  buffers as a fallback), and `AdrenoCLRenderer` keeps fp16 buffers native (QCOMCLRenderer's rule would emulate
  them as ushort: 2 s for that one kernel). Their largest kernel (18-19 ms) is one fused 20x20/40x40/80x80 head kernel.
- RF-DETR needed two exporter additions: a `GridSample` (bilinear, zero padding) for its deformable attention, which
  tinygrad's OnnxRunner lacks, and reading scalar constants (`CAST(CONST)` initializers used as Gather/Reshape
  arguments) straight off the graph, since OnnxRunner otherwise realizes them with a kernel on tinygrad's CPU device,
  which needs a C compiler the phone's Python doesn't have. Its time is spread over many kernels (the largest six are
  identical 9.4 ms attention kernels); max sigmoid-score difference vs fp32 over all 300x91 scores is 0.24, but on
  low-score queries: every score above 0.5 agrees to 0.004.
- Fixed per-frame overhead in the app: the 1.2 MB input upload and 2.8 MB output readback are inside the 44.3 ms.
