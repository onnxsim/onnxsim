# Mask R-CNN live demo app (Android, Hexagon HTP + HVX)

Model buttons (top): Mask R-CNN, YOLO26n, YOLO11n, RT-DETR, RF-DETR, SAM, MCC 3D, Super-res and Game
upscaling, each engine in its own process (one model loaded at a time); see the YOLO, RT-DETR, SAM, MCC,
super-resolution and game-upscaling sections below for the newer modes.

An Android app that runs the full Mask R-CNN (ONNX model zoo `MaskRCNN-12-qdq`) on the phone
(Xiaomi 12S, Snapdragon 8+ Gen 1), frame by frame, from the camera or a set of test images, with
boxes, labels, instance masks and an FPS / latency / per-stage counter. It runs the pipeline
`../e2e_pipeline/` built and measured (PR #1841, plus the changes below): `native/maskrcnn_engine.cpp`
`#include`s `../e2e_pipeline/e2e_run.cpp` and only swaps the image source and the output path.

| region | engine |
|---|---|
| backbone (optimized per PR #1833), box head, mask head | HTP via ORT + QNN EP (`com.qualcomm.qti:qnn-runtime` 2.50.0, bundled), EP-context models with uint8 graph boundaries |
| RPN post-processing (TopK, decode, NMS, merge) | HVX DSP, FastRPC skel `../tinygrad_hexagon_bridge/rpn_fused` |
| RoiAlign + level merge, straight to the heads' uint8 input | HVX DSP, FastRPC skel `../tinygrad_hexagon_bridge/roialign_fast/roialign_u8_*` (PR #1848) |
| image preprocessing (camera YUV or RGBA -> rotated, letterboxed uint8 NHWC) | native, one pass |
| everything else (per-class NMS, box decode, ...) | ORT CPU, 4 threads |

## YOLO mode (YOLO26n / YOLO11n)

The model buttons (top right) switch between Mask R-CNN and the deploy pipeline's YOLO models
(`../deploy/models/yolo26n.yaml`, `yolo11n.yaml`; PR #1865). `YoloActivity` runs in its own
process (`:yolo`), since each native engine holds process-wide HTP/DSP state and `onDestroy` ends its
process; the two YOLO buttons switch models in place (the engine re-inits its HTP session).

- `native/yolo_engine.cpp` (`libyolo_demo.so`): one HTP session (ORT + QNN EP, strict, EP-context
  model `<model>.ctx0.onnx` compiled on the first launch), fed the camera YUV frame (or a test image)
  rotated upright and letterboxed to 640x640 (centered, pad 114, as `deploy/stages/images.py`) as
  uint8 NHWC in one native pass -- the model's input *is* the RGB bytes (scale 1/255, zero point 0).
  The head's `(1, 84, 8400)` output is post-processed in C++: YOLO26's NMS-free two-stage top-k
  (the deploy pipeline's `yolo_end2end`, no NMS) or YOLO11's per-class NMS (iou 0.7, conf 0.25).
- Camera: the smallest 4:3 size covering 640 (640x480), and the fastest fixed AE frame-rate range
  (auto-exposure otherwise drops to ~14 FPS indoors, which was the cap).
- Models: `YOLO="<deploy work>/yolo26n/pipe/yolo26n.onnx <deploy work>/yolo11n/pipe/yolo11n.onnx" ./deploy.sh`,
  then `adb shell am start -n org.onnxsim.maskrcnndemo/.YoloActivity [--es mode images] [--es model yolo11n]`.

Measured on the phone (medians of the app's running averages, under the shared phone lock):

| model, mode | end-to-end FPS | inference | pre | HTP | post |
|---|---:|---:|---:|---:|---:|
| YOLO26n, test images | 100-111 | 2.9 ms | 0.1 | 2.5 | 0.3 |
| YOLO11n, test images | 91 | 3.3 ms | 0.1 | 2.6 | 0.6 |
| YOLO26n, camera, AE default | 14.2 (camera-capped) | 9.1 ms | 5.4 | 3.0 | 0.8 |
| YOLO11n, camera, AE default | 14.2 (camera-capped) | 10.0 ms | 5.5 | 3.0 | 1.4 |
| **YOLO26n, camera, fixed 30 FPS AE** | **30.0-30.6 (camera-capped)** | 8.2-9.0 ms | 4.5-5.6 | 2.9 | 0.5-0.8 |

<img src="docs/yolo26n_images.jpg" width="240" alt="YOLO26n on COCO val2017 #139 in the app">

Inference alone would allow ~110 FPS from the camera and ~330 FPS from decoded images; end to end is
bounded by the camera (30 FPS) and, in images mode, by the Java JPEG decode + UI draw per frame.
The camera preprocessing (4.5-5.6 ms at 640x480) is the column-wise plane reads of the 90-degree
rotation, as in the Mask R-CNN path. Box placement was checked visually on COCO val2017 #139.
YOLO26's end-to-end top-k is cheaper than YOLO11's NMS here too (0.3 vs 0.6 ms on images).

## RT-DETR mode (RT-DETR-r18, NMS-free, HTP + HVX)

The "RT-DETR" button (`RtDetrActivity`, process `:rtdetr`, the YOLO mode's loop with another engine)
runs RT-DETR-r18 from `../vision_models/rtdetr` (PR #1867) the way its phone runner does: 4 strict
HTP pieces (backbone uint8 + hybrid encoder W8A16, uint8 value maps) around 3 multi-scale deformable
attention calls on the HVX (`../msda_hvx` skel, `MSDA_FLAGS=260`). `native/rtdetr_engine.cpp`
`#include`s `../vision_models/rtdetr/msda_hvx/dec_run.cpp` for its rpcmem tensor store, piece
runner and msda call; the input is the frame stretched to 640x640 (HF's processor: no letterbox),
detections are sigmoid + top-k over queries x classes (no NMS), shown at score >= 0.4.
Models: `RTDETR=$HOME/.cache/onnxsim-rtdetr/work/split ./deploy.sh` (`split.py export` + `quant
--policy bb8enc16 --u8-values`).

| mode | end-to-end FPS | inference | pre | HTP pieces | MSDA calls (in-DSP) |
|---|---:|---:|---:|---:|---:|
| test images | 34.7-35.0 | 19.4-19.7 ms | 1.1-1.2 | 15.0-15.2 | 3.3 (2.6) |
| camera (fixed 30 FPS AE) | 29.8-30.1 (camera-capped) | 23.7-24.4 ms | 4.1-4.8 | 15.5 | 4.0 (3.3) |

Same as the runner's 18.3 ms plus the app's preprocessing. Against YOLO26n (2.9 ms), RT-DETR is ~7x
the inference per frame but needs no NMS and keeps small, crowded detections (19 on this image).

<img src="docs/rtdetr_images.jpg" width="240" alt="RT-DETR-r18 on a crowded COCO image in the app">

## RF-DETR mode (RF-DETR-Nano, NMS-free, all-HTP fp16)

The "RF-DETR" button runs RF-DETR-Nano from `../vision_models/rfdetr` in the YOLO activity (process
`:yolo`, model `rfdetr_nano`), as one strict-HTP fp16 graph: `native/yolo_engine.cpp`'s `post=detr`.
- **Input:** RF-DETR is trained on a *stretched* square input (its `predict()` resizes without
  keeping the aspect ratio), so the camera frame is converted upright at display size in the usual
  YUV pass, then nearest-sampled to the model's S x S uint8 NHWC input; test images are stretched
  straight from the bitmap. The normalization is folded into the model's patch-embed conv
  (`../vision_models/rfdetr/graph.py`), so the input is the RGB bytes.
- **Output:** `logits` (1, 300, 91) + `boxes` (1, 300, 4), decoded as RF-DETR's PostProcess
  (sigmoid, top 300 over queries x classes, threshold 0.5 as `predict()`); boxes are normalized to
  the stretched input, so they scale back by the frame's width and height; labels are COCO category
  ids, mapped to `Coco.java`'s contiguous table.
- **Model:** `RFDETR=$HOME/.cache/onnxsim-rfdetr/work/nano@320.u8.onnx ./deploy.sh` (stored as
  `rfdetr_nano.onnx`), then `adb shell am start -n org.onnxsim.maskrcnndemo/.YoloActivity --es model
  rfdetr_nano [--es mode images]`. Nano at 320 is the default: its 26 ms keep up with the 30 FPS
  camera; `nano.u8.onnx` (384, 47 ms, a bit more accurate) drops in the same way.

Measured on the phone under the shared phone lock (the app's running averages):

| model, mode | end-to-end FPS | inference | pre | HTP | post |
|---|---:|---:|---:|---:|---:|
| RF-DETR-Nano @320, test images | 25-26 | 27 ms | 0.2 | 26.5 | 0.3 |
| RF-DETR-Nano @320, camera | 30 (camera-capped) | 30 ms | 4 | 26 | 0.3 |

<img src="docs/rfdetr_images.jpg" width="240" alt="RF-DETR-Nano on a COCO image in the app">

## SAM mode (tap to segment, EfficientViT-SAM-L0)

The "SAM" button runs Segment Anything (`SamActivity`, its own process `:sam`,
`native/sam_engine.cpp` -> `libsam_demo.so`): EfficientViT-SAM-L0 from `../vision_models/sam`
(PR #1876, with its exact bicubic-as-depthwise-conv neck rewrite), encoder and decoder both strict on
the HTP from EP-context models.

- **images:** each test image is encoded once (longest side -> 512, padded bottom/right with the SAM
  mean pixel, uint8 NHWC; normalization is in the graph); every tap runs only the decoder with that
  point (+ a padding point), and the mask of slot 1 + argmax(iou[1:]) is overlaid (a low-res pixel =
  2x2 encoder pixels). "Next image" moves on.
- **camera:** a live preview (no inference); a tap freezes that frame and runs the encoder, then the
  decoder; further taps only decode; "Live" unfreezes.
- Models: `SAM=$HOME/.cache/onnxsim-sam/efficientvit_sam_l0 ./deploy.sh` (the `sam.py` work dir:
  `enc.fp16.onnx`, `dec.sim.onnx`); `--es tap 0.5,0.55` taps automatically after each encode.

| | ms |
|---|---:|
| encoder, per image / frozen frame (pre 0.3-0.4, camera 5.6) | 41.3-42.5 |
| decoder, per tap | 11.2-13.6 |
| first tap on a new image / frozen camera frame | ~53-60 |
| init, first launch (compiles both HTP graphs) / later launches | 11.0 s / 0.44 s |

<img src="docs/sam_l0_images.jpg" width="240" alt="SAM mode: a tap on the snow segments the slope around the skier">

(phone, under the shared phone lock; the encoder matches #1876's 41.8 ms.) Each extra tap costs
~11 ms, so segmenting feels immediate after the one-off encode.

## MCC 3D mode (one photo -> tap -> the object's full 3D shape)

The "MCC 3D" button (`MccActivity`, its own process `:mcc`, `native/mcc_engine.cpp` -> `libmcc_demo.so`)
is the follow-up that `../vision_models/mcc` left open: tap an object, press "3D", and get its
reconstructed colored point cloud, including the sides the photo doesn't show. Every network runs strict
on the HTP from EP-context models:

| step | model | where |
|---|---|---|
| segment the tapped object | EfficientViT-SAM-L0 encoder + decoder (the SAM mode's `sam_l0_enc/dec`) | HTP |
| monocular metric point map | MoGe-2 ViT-S, static 640x480 / 480x640 (`depth.py static`) | HTP, fp16 |
| upstream `prep` + XYZ window partition (`model.py`) | ported to C++ | CPU |
| encoder -> the decoder's seen K/V | MCC `enc.onnx` (`model.py` K/V-cache split) | HTP, fp16 |
| occupancy + color queries, coarse-to-fine as `mcc.py recon` (15^3 -> 30^3 -> 60^3, refine where p > 0.1) | the hand-written DSP decoder `../mcc_hmx` (HMX GEMMs + 4 HVX threads, FastRPC skel `libmcc_hmx_rpc.so`), up to 1024 queries a call; `--es opts dec=qnn`: MCC `dec_q1024.a16c.onnx` (`dec_opt.py`) | DSP (HMX + HVX, fp16); QNN: HTP w8a16 |

- The working image is 480x640 (portrait) or 640x480 (landscape): camera frames (4:3) scaled, test
  images center-cropped to 3:4 / 4:3. MoGe-2's points are flipped into MCC's frame (y up, z toward the
  viewer), as `depth.py` does. MoGe-2 needs only the image, so it starts on its own thread when an image
  is encoded and is done by the time "3D" is pressed. There is no gravity alignment yet (see `../vision_models/mcc` on the
  17.5 deg rotation this leaves against the iPhone demo cloud).
- **images:** the test images; tap an object (SAM mask in blue), "3D" reconstructs, "Photo" goes back,
  "Next image" moves on. **camera:** live preview; a tap freezes the frame and segments, "Live" unfreezes.
- 3D view: z-buffered colored splats of the points with p > 0.3; drag to turn, pinch to zoom; the photo
  with its mask sits in the corner.
- Models: `SAM=... MCC=$HOME/.cache/onnxsim-mcc/work MOGE=$HOME/.cache/onnxsim-mcc/moge MCC_HMX=<../mcc_hmx/ref.py weights dir> ./deploy.sh`
  (`mcc.py export --chunks 1024`, `dec_opt.py prep` + `quant --policy a16c`, `MCC_DEC=dec_q1024.onnx` for
  the fp16 decoder; `depth.py static` at `--h 640 --w 480` and `--h 480 --w 640`).
  Scripted: `--es image quest2.jpg --es tap 0.506,0.491 --ez recon true`; `--es opts` takes `gran`,
  `levels`, `lo`, `thr` (defaults 0.1 / 2 / 0.1 / 0.3) and `dump=1`.
- **License:** MCC's code and weights are CC BY-NC 4.0 (non-commercial); nothing of it is in the APK,
  `deploy.sh` pushes the exported models. The screenshot's input is upstream MCC's `demo/quest2.jpg`.

Phone (Xiaomi 12S), upstream's quest2 photo, the tap on the headset ("3D" pressed again on the same
mask for the steady state):

| | first version | optimized (QNN w8a16 decoder) |
|---|---:|---:|
| SAM encoder (once per image) / decoder (per tap) | 43-53 / 12-15 | same |
| MoGe-2 ViT-S 640x480 | 268-276 | 270-590, in the background: 5-12 waited |
| prep (C++) | 6-21 | 12-24 |
| MCC encoder | 205-207 | 203-206 |
| MCC decoder chunks x ms (queries: coarse-to-fine refine threshold 0.05 -> 0.1) | 59 x 64 (59,212) | 48 x 46.6 (47,317) |
| **total, "3D" to points** (19,652-19,666 points) | **3.9-4.3 s** | **2.48-2.52 s** |
| vs the dense host fp32 grid: recall / precision / chamfer / color L1 | 0.993 / 0.992 / 0.0008 / 0.46 | 0.990 / 0.988 / 0.0011 / 0.76 |
| init, first launch (compiles MCC's two graphs) / later launches | 48.7 s / 1.6 s | ~49 s / 1.6 s |
| MoGe-2 compile, first image per orientation (in the background now) | 17-24 s | same |

**DSP decoder** (`../mcc_hmx`, now the default): the decoder chunks on the Hexagon DSP by hand -- every
GEMM on HMX, softmax / LayerNorm / GELU on 4 HVX threads, activations in VTCM between steps -- 22 ms a
1024-query chunk vs QNN's 47 (w8a16): decoder **1.06 s** (48 calls), **"3D" 1.30 s**, and closer to the
float64 model than the w8a16 graph (recall 0.993, precision 0.992, chamfer 0.0008, color L1 0.43/255 vs
the dense host fp32 grid). Details and the step-by-step timings in `../mcc_hmx/README.md`.

The optimizations, one at a time (`../vision_models/mcc/dec_opt.py`, `README.md` "Decoder
optimization" there for everything tried):
- **w8a16 decoder** (`onnxsim.full_qdq`, uint16 activations, int8 weights, the color head's
  temperature-0.1 softmax kept fp16): 63 -> 47 ms a chunk. The per-op profile had softmax at 44% of the
  fp16 chunk; quantizing only softmax (or softmax + Gelu) is *slower* (72 / 83 ms: the fp16 <-> int16
  conversions around it), the whole graph has to go integer.
- **refine threshold `lo` 0.05 -> 0.1**: 13-20% fewer queries, recall >= 0.9975 of the dense grid's
  occupied points on all three references (exact: host dense grids).
- **MoGe-2 off the critical path**: it only needs the image, so it runs while the user taps.

**Checked against the Python pipeline** (`../vision_models/mcc/app_check.py` on the app's `dump=1`
tensors -- its own mask and MoGe-2 points): the C++ prep's image input is identical (max abs 0), the
valid-point pattern identical, xyz within 1.9e-6; the phone encoder's K / V cos 0.999985 / 0.99988 vs
host fp32; the reconstruction vs the dense host fp32 grid in the table above.

<img src="docs/mcc_quest2.jpg" width="480" alt="MCC 3D mode: the headset's reconstruction from the photo's viewpoint and turned to show its far side">

(left: from the photo's viewpoint; right: turned by a drag, showing the visor's back and inside.)

## Super-resolution mode (x4 neural upscaling, XLSR int8 on the HTP)

The "Super-res" button (`SrActivity`, its own process `:sr`, `native/sr_engine.cpp` ->
`libsr_demo.so`) is the open, camera-usable counterpart of DLSS-style upscaling (DLSS / XeSS / FSR4 /
MetalFX are closed; see `../vision_models/superres/README.md`). Every frame:

1. the camera's centered 16:9 crop at 1920x1080 (1080x1920 upright in portrait; 1:1 sensor pixels of
   the 1920x1440 stream) is the **original**;
2. its 4x4 box average, 480x270 (1/16 the pixels -- what a game would render), is the **low-res**
   input; the camera path averages Y/U/V per 4x4 block and converts once per low-res pixel;
3. an x4 model from `../vision_models/superres` (PR #1906) upscales it back to 1920x1080 on the HTP,
   uint8 NHWC in and out;
4. the screen shows the **SR** output right of a draggable divider and a reference left of it:
   **bicubic** of the same low-res input (CPU, PIL's a = -0.5, run alongside the HTP inference), the
   **original**, or the **low-res** pixels (nearest).

Buttons: the model (**XLSR int8**, the default: the best PSNR kept through int8 in #1906;
QuickSRNet-M int8; XLSR fp16; Real-ESRGAN animevideov3 int8, GAN-trained for perceived sharpness)
and the left-side reference. One HTP session per orientation (`<model>_270x480.onnx`,
`<model>_480x270.onnx`), EP-context compiled on first use. Models:
`SR=$HOME/.cache/superres/models ./deploy.sh` (from `superres.py build <model> 270 480` and
`... 480 270`); extras `--es model sr_xlsr_int8 --es ref bicubic|original|lowres --es mode images`.

Phone, under the shared phone lock (images mode: 3 nuScenes CAM_FRONT frames, center-cropped;
camera mode: 30 FPS stream):

| mode, model | reference | HTP (SR) | frame -> low-res | SR -> bitmap | end to end |
|---|---|---:|---:|---:|---:|
| images, XLSR int8 | original | 6.1 ms | 5.1 ms | 2.6 ms | **54 FPS** |
| images, QuickSRNet-M int8 | original | 5.1-5.3 ms | 5.2 ms | 2.5 ms | 56-57 FPS |
| images, XLSR fp16 | original | 15.1 ms | 6.1 ms | 2.6 ms | 35 FPS |
| images, Real-ESRGAN int8 | original | 13.3 ms | 5.4 ms | 2.5 ms | 39 FPS |
| images, XLSR int8 | bicubic (CPU 24-27 ms, in parallel) | 6.3 ms | 5.1 ms | 7 ms | 28 FPS |
| camera, XLSR int8 | bicubic (CPU 31-34 ms, in parallel) | 6.2 ms | 6.9 ms | 5-8 ms | 24-26 FPS |
| camera, XLSR int8 | original (full-res YUV -> RGB for display) | 5.8 ms | 23-27 ms | 2.6-3.7 ms | 26-30 FPS (camera-capped) |

<img src="docs/sr_images.jpg" width="270" alt="Super-res mode: bicubic left of the divider, XLSR int8 right">
<img src="docs/sr_detail.jpg" width="540" alt="1:1 detail: bicubic, XLSR int8, original">

(the detail strip is a 300x200 1:1 crop, shown 2x, of the same HTP graph run on the host: bicubic
40.90 dB vs XLSR int8 41.92 dB PSNR against the original on that frame.) The SR itself is ~6 ms of a
33 ms camera frame; what bounds the loop is the CPU work around it (the bicubic reference, or
converting the full-res frame when the original is shown). In the camera path the low-res input
costs 6.9 ms since only the 130K low-res pixels are converted -- converting all 2M pixels first
took 23 ms.

## Game-upscaling mode (NSS super sampling + NFRU frame generation, replayed Bistro)

The "Game upscaling" button (`GameActivity`, its own process `:game`, `native/game_engine.cpp` ->
`libgame_demo.so`) replays a short stretch of Arm's rendered test sequence -- what a game hands its
upscaler every frame: a 960x540 colour render, depth, motion vectors, camera matrix, jitter -- through

1. **NSS** (`../vision_models/nss`): temporal super sampling to 1920x1080, the int8 CNN on the HTP and
   the pre/post-processing as OpenCL kernels on the Adreno 730;
2. with **NFRU on** (`../vision_models/nfru`): one generated frame between each two NSS outputs, the
   int8 network on the HTP and the block-matching flow, motion splats and blend on the Adreno.

The same kernels (compiled into the library from `nss_kernels.cl` / `nfru_kernels.cl`) and networks as
the standalone runners; everything stays in GPU buffers/images, the networks read and write host-mapped
buffers shared with ORT, and only the displayed RGBA8 frames are copied into bitmaps (1.1-3 ms). NFRU
consumes NSS's linear output through its own colour pipeline (the same exposure + Reinhard NSS displays
with) and NSS's motion negated: NFRU's vectors point back, NSS's forward (checked against the camera:
the camera-induced motion matches `-motion_lr` to 0.05% median). The sequence has no forward vectors,
so NFRU's block-matching hint for m1 -> p1 is NSS's motion of p1 (an approximation; the hint only
competes with the 7x7 search).

The screen: NSS (or NSS + NFRU) right of a draggable divider, the native 540p render (nearest) left of
it, the NFRU on/off button, per-frame ms. Data: `python game_seq.py` (the first 48 frames of
`Arm/neural-graphics-dataset` `nss/test`, 379 MiB, + Arm's license text next to it), then
`GAME=~/.cache/arm-nss/game ./deploy.sh` (also pushes NSS's and NFRU's int8 ONNX; EP-context compiled
on the first launch); `adb shell am start -n org.onnxsim.maskrcnndemo/.GameActivity [--ez nfru true]`.
The sequence and weights are under the Arm AI Model Community License, which permits redistribution
with its text and notices (see `../vision_models/nss/README.md`); at 379 MiB it is still deployed, not
bundled into the APK.

Phone (Xiaomi 12S), `cl_qcom_perf_hint` high, HTP burst, in the app:

| NFRU | NSS per frame (CNN on the HTP) | NFRU per generated frame (net on the HTP) | upload + copy-out | displayed FPS (1080p) |
|---|---:|---:|---:|---:|
| off | 26.4 ms (3.7) | - | 1.7 + 1.7 ms | **31-33** |
| on | 26.4 ms (3.7) | 29.3 ms (1.8) | 2 + 3 ms | **30-33** (15-16 rendered) |

<img src="docs/game_nss.jpg" width="540" alt="Game upscaling, NFRU off: native 960x540 left, NSS 1920x1080 right">
<img src="docs/game_nfru.jpg" width="540" alt="Game upscaling, NFRU on: native 960x540 left, NSS + NFRU right">

With the replay's rendering free (the frames come from a file), NFRU doesn't raise the displayed rate
here: one generated frame (29 ms) costs about as much as upscaling a rendered one (26 ms). It pays off
once the game's own 540p rendering costs more than ~5 ms a frame (`../vision_models/nfru/README.md`).
The generated frame is shown right away and frame t half a step later (posted, so the next step's
work overlaps it).

## More modes: follow-ups (not built)

- **BEV replay** (Fast-BEV++, PR #1873 / #1895 pipelined): bundle a few nuScenes-mini scene-0103
  frames (6 cameras each; nuScenes is CC BY-NC-SA 4.0, so keep it to a handful and credit it),
  run the image piece + the in-graph view transform + BEV head on the HTP per frame, decode on the
  CPU, and draw the 6 camera thumbnails with projected 3D boxes plus a top-down BEV view. Needs the
  per-rig lookup table and the camera calibration shipped with the frames, and a 3D-box projection
  in Java. A toggle for StreamPETR (#1884) / BEVFormer (#1879) on the same frames would reuse the
  data and the drawing.
- **LLM chat** (SmolLM2-135M fp16 on the HTP, #1886: 24.5 ms prefill for 128 tokens, 110 tok/s):
  a text box streaming greedy tokens. Needs the prefill and single-token decode graphs with the KV
  cache as uint8/fp16 I/O driven from a native loop, plus the tokenizer (a small BPE in C++ or
  Java) -- the decode step graph is the piece to port from `../llm_tinygrad`.

## Result

Defaults since this round: `pipe_e_u8ra_ctx.txt` + `quant=lut;merge=seg2,seg4;pipeline=box_head`.
Steady state in the app, medians over 20-35 logged frames, with the UI drawing (and, in camera mode,
the camera running). FPS is how often a processed frame reaches the screen; latency is capture to
result. Startup is from process start to the first / tenth result on screen. ("Now" FPS rows were
measured with warm-up on, which only affects startup; startup "now" is without it, the new default.)

| | FPS | latency | first result | 10th result |
|---|---:|---:|---:|---:|
| test images, #1843 as merged (JIT, no options) | 6.3 | 140 ms | 7.3 s | 8.8 s |
| test images, #1843's best options (lut+merge+pipeline) | 10.1 | 175 ms | 6.5 s | 7.4 s |
| **test images, now** | **13.6** | **118 ms** | **0.78 s** | **1.33 s** |
| camera 1280x960, #1843's best options | 9.1 | 220 ms | 6.9 s | 7.9 s |
| **camera, now** | **14.0-14.2** | **117-142 ms** | **0.97-1.0 s** | **1.60-1.64 s** |

The same pipeline in the one-process `adb shell` driver (`../e2e_pipeline`, `ORT_SPIN=0`, no UI):
78-88 ms per image, against 116-142 ms for #1841's `e_opt`, with 59/61 detections matched vs
all-ONNX-Runtime (e_opt: 58/61) and mask IoU 0.887 (0.861).

## Optimizations, one lever at a time

Each is an option, so the #1841 path stays available for A/B: `--es pipe pipe_e_opt.txt --es opts ""`.
`./bench.sh "label|mode|pipe|opts|overlap" ...` runs configurations back to back and prints the
table below. Images mode unless noted. All numbers are from the phone, measured when no other agent
was using the DSP/HTP (it is shared; contended runs were discarded).

| config | FPS | latency | pre | backbone | RoiAlign | heads | CPU | stage B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. #1843 as merged | 6.33 | 140 | 6.8 | 17.2 | 25.7 | 43.6 | 45.8 | - |
| 2. + `quant=lut` + `merge=seg2,seg4` + `pipeline=box_head` | 10.10 | 175 | 2.2 | 30.9 | 27.0 | 48.8 | 35.0 | 68.2 |
| 3. + ORT pool spinning off | 10.04 | 171 | 2.9 | 23.0 | 26.8 | 56.5 | 38.8 | 81.6 |
| 4. + uint8 heads from EP-context models (`pipe_e_u8_ctx.txt`) | 9.96 | 170 | 2.4 | 19.9 | 26.1 | 37.8 | 48.1 | 69.4 |
| 5. + uint8 RoiAlign skel (`pipe_e_u8ra_ctx.txt`), no pipelining | 9.20 | 87 | 2.4 | 17.1 | 12.0 | 34.7 | 17.8 | - |
| **6. = 5 + `pipeline=box_head` (default)** | **13.56** | 118 | 2.8 | 20.3 | 13.3 | 36.6 | 24.2 | 57.9 |
| camera: 2's options | 9.09 | 220 | 12.6 | 38.2 | 24.6 | 42.7 | 55.7 | 71.4 |
| **camera: default** | **14.03** | 142 | 15.8 | 31.7 | 12.9 | 32.8 | 28.2 | 56.4 |

(ms; "RoiAlign" includes the map staging, "heads" both HTP heads, "CPU" every ORT CPU segment and
native pass. Pipelined rows run two frames at once, so stage times overlap and share the cores.)

- **Frame pipelining (`pipeline=<step>`)** runs the steps before `<step>` (stage A) and the rest
  (stage B) on two threads, so frame N+1's stage A overlaps frame N's stage B. Hand-off is one frame
  deep; stage-A buffers alternate by frame parity so frame N+2 never overwrites what stage B reads.
  Split points measured with the current pipe: `box_head` 13.6 FPS, `seg3` 9.0, `mask_ra` 7.8, none
  9.2. Overlapping only the image capture/decode with inference (`--ez overlap true`) did not help:
  the extra decode thread slowed preprocessing (12 -> 21 ms) through CPU contention.
- **LUT quantize (`quant=lut`)**: a per-channel 256-entry table for the image quantize, exact.
- **Native merge (`merge=seg2,seg4`)**: a 4-thread row scatter instead of the ScatterND chains. The
  uint8 RoiAlign skel now writes merged rows itself, so this only matters for the older pipes.
- **Camera YUV fast path:** `quant_yuv` does YUV_420_888 -> RGB (BT.601 full range, fixed point),
  the rotation, the letterbox scale and the quantize in one native pass straight from the camera
  planes, and also writes the displayed RGBA frame. No Java bitmap conversion. Still ~15 ms: the
  camera planes are slow to read column-wise for the 90-degree rotation, and copying them to cached
  memory first did not help.
- **ORT pool spinning off** (`env.ORT_SPIN=1` restores it): ORT's global intra-op threads spin after
  every op and hold the big cores, which starved every threaded native pass between ORT segments
  (several times slower and noisy). In the adb driver it is worth 15 ms per frame; in the app the
  effect shows in the steadier pre / quant times rather than in FPS.
- **ORT threads:** 3 or 4 is best (6: 9.6 FPS vs 10.5 with the #1843 options).
- **uint8 heads from EP-context models:** see "Startup" below; per frame it takes the heads from
  48.8 to 37.8 ms.
- **uint8 RoiAlign (PR #1848, by the RoiAlign work):** one DSP call per head reads the backbone's
  uint8 maps and writes the head's uint8 input rows. It replaces the maps' dequantize, 4 RoiAlign
  calls, the merge and the quantize, 65.6 ms of the old span on image 139, with 6.5 (box) + 2.6 (mask)
  ms plus 1.9 ms staging the maps into rpcmem once per frame.
- **Not done here:** CPU segment threading beyond ORT's pool. The largest CPU piece left is `seg3`
  (per-class NMS, box decode: ~13 ms), which PR #1825 found slower on the DSP.

### DSP calls are serialized

Every skel call (RPN, RoiAlign, uint8 RoiAlign) takes one process-wide mutex (`exec_step`). With
pipelining, stage A (RPN, box RoiAlign) and stage B (mask RoiAlign) would otherwise call the DSP at
the same time. The same skel called from two threads at once failed with `rc=78`: the skels keep
per-handle scratch state and aren't reentrant. The uint8 RoiAlign kernel also assumes it never runs
next to the RPN kernel, since both size their HVX threads for a DSP of their own. The calls are short
(2-7 ms), so the lock costs little.

## Startup

Before: the HTP graphs were compiled at every launch (6.2-7.4 s). EP-context models (precompiled
QNN context binaries) loaded in 0.75 s but made the box head ~20 ms slower per frame, so they weren't
the default. Root cause (details in `../e2e_pipeline/README.md`): not the compile options, but the
heads' **fp32 graph boundaries**. The box head's 50 MB fp32 input is reshaped and quantized inside
the HTP graph, and the mask head ends in dequantize + sigmoid to fp32; a deserialized graph is much
slower at both. QNN's profiler puts the extra time inside the accelerator. With uint8 inputs (the
CPU quantizes, or now the RoiAlign skel writes uint8) and uint8 mask logits (a `mask_sel` step
dequantizes and sigmoids only the detection's class channel), the box head runs 25.3 ms from
EP-context vs 24.6 JIT (was 48.3 vs 27.9).

| images mode, pipelined | init | first result | 10th result | per-frame heads |
|---|---:|---:|---:|---:|
| JIT (`pipe_e_opt.txt`) | 6.2-7.4 s | 6.5-7.6 s | 7.4-8.8 s | 48.8 ms |
| old EP-context (`pipe_e_opt_ctx.txt`, embedded) | 0.69 s | 0.96 s | 2.01 s | 88.4 ms |
| **new EP-context, uint8 heads, embed mode 0 (default)** | **0.60-0.61 s** | **0.77-0.78 s** | **1.32-1.33 s** | 37.8-41.3 ms |
| + warm-up (`warmup=1`) | 0.69-0.71 s | 0.84-0.86 s | 1.40-1.41 s | |
| + parallel HTP session creation (`par_load=1`, with warm-up) | 0.67-0.68 s | 0.82-0.83 s | 1.38-1.40 s | |

(two launches each for the last three rows; the first row's heads are with #1843's options.) In the
adb driver, embed mode 0 (`ctx=2`: the context binary in its own file) creates the sessions in
0.66 s vs 1.0 s embedded.

- **DSP skels open on their own thread** while the ORT sessions are created. Not before the first
  HTP session exists, though: opening our FastRPC sessions while QNN sets up its HTP device made that
  setup fail ("Failed to create device ... INVALID_CONFIG"), and the session silently fell back to
  the (disabled) CPU EP.
- **The camera opens while the models load** (camera mode), instead of after.
- **Warm-up doesn't pay** any more: one gray frame plus every `ortpad` bucket costs ~120 ms of init,
  more than the first cold frame now costs, so results appeared ~80 ms later with it. Kept as
  `warmup=1`, off by default.
- **Parallel session creation (`par_load=1`)** saves ~30 ms of session setup (QNN seems to
  serialize most of the context loading), inside launch-to-launch noise. Off by default.
- **First launch on a new phone:** the EP-context files come from compiling on the device. `deploy.sh`
  copies them if `../e2e_pipeline` already made them; otherwise the app compiles them at first
  launch (the JIT cost, once; `NO_CTX=1 ./deploy.sh` to try) and reuses them after.
- Models stay in internal storage (see below).

## Orientation

The camera frame is rotated by `(sensorOrientation - displayRotation + 360) % 360` in the native
YUV pass, so the network always sees an upright image; boxes and masks come back in that upright
frame and are drawn over it. The live thumbnail gets the matching `setTransform`. The activity uses
`fullUser`, which follows the rotation lock (`fullSensor` ignores it). Images mode applies the JPEG's
EXIF orientation. Checked on the phone at all four `user_rotation` values with auto-rotate off
(settings restored afterwards).

## Does this work from an app, not just an adb shell? Yes.

Everything before this ran as an `adb shell` native process. PR #1810 had found that an app can't
`dlopen` the *vendor's* `/vendor/lib64/libQnnHtp.so` (it isn't in `/vendor/etc/public.libraries.txt`).
Bundling Qualcomm's own runtime from Maven (as PR #1829 did for the shell) avoids that. Measured in
this app, an `untrusted_app_27` process on device `239dbd8f` (Xiaomi 12S, Snapdragon 8+ Gen 1,
Hexagon V69), no root, no system changes:

- **Bundled QNN runtime loads from the APK's `nativeLibraryDir`** (`useLegacyPackaging true`, so the
  libraries are real files): `QNN EP registered, 1 NPU device(s)`, and every HTP session is created
  strict (`session.disable_cpu_ep_fallback=1`).
- **DSP-side skels load from the app's lib dir.** `ADSP_LIBRARY_PATH` is set in-process (`setenv`,
  before the first FastRPC call) to the app's `nativeLibraryDir` first. `libQnnHtpV69Skel.so`,
  `librpn_rpc.so` and `libroialign_rpc.so` (our skels, renamed `lib*.so` so the package manager
  extracts them) all load: `rpn skel open`, `roialign skel open`.
- **Unsigned PD is allowed** for the app: `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`
  returns 0, and the unsigned skels run.
- **`libcdsprpc.so` loads** (it is in `public.libraries.txt`; declared with `<uses-native-library>`).
- **SELinux denials seen, all harmless:** `untrusted_app_27` is denied `search`/`getattr` on
  `/vendor/dsp` (`adsprpcd_file`), which is only FastRPC's default skel search path, and `open` on
  `/dev/adsprpc-smd` / `adsprpc-smd-secure`. `libcdsprpc` then falls back to the CDSP device node the
  app *is* allowed, and every skel above loads and runs.

One app-context gotcha: files `adb` creates under `/sdcard/Android/data/<pkg>` are owned by the
shell user, so the app can't enter them (`chdir ... models` failed). `deploy.sh` instead copies the
models into the app's **internal** files dir with `run-as` (the APK is debuggable).


## Build and run

```bash
# 1. models: ../e2e_pipeline/README.md "Reproduce" (build_models.py, then u8_heads.py). If
#    ../e2e_pipeline/build.sh already ran on this phone, deploy.sh copies them on-device from
#    /data/local/tmp/e2e (with any EP-context files compiled there).
# 2. build (heavy step; cap it on a shared machine)
systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0 \
  -E HEXAGON_SDK_ROOT=... -E HEXAGON_TOOLCHAIN=... -E ANDROID_HOME=~/android-sdk ./build_app.sh
# 3. install + models + test images
IMGS="cats.jpg img_000000000139.jpg ..." ./deploy.sh        # or MODELS=<build_models.py --out dir>
# 4. run: camera mode (default) or a loop over the test images; --es pipe / --es opts for A/B
adb shell am start -n org.onnxsim.maskrcnndemo/.MainActivity --es mode images
adb shell am start -n org.onnxsim.maskrcnndemo/.MainActivity --es pipe pipe_e_opt.txt --es opts ""
```

Options (`--es opts "k=v;..."`, see `native/maskrcnn_engine.cpp`): `quant=lut`, `merge=a,b`,
`pipeline=<step>`, `par_load=1`, `warmup=1`, `env.NAME=value` (`ORT_THREADS`, `ORT_SPIN`, ...).
Images mode logs a per-image output checksum (`check img ...`) to A/B options for equality.

Toolchain used: Android SDK platform 34 + build-tools 34, NDK 27.2, AGP 8.5.2, Gradle 8.7 (offline),
JDK 21. No CameraX: the app uses the framework Camera2 API, so it needs no extra Maven dependencies.

## Bugs found building it

- **Wrong labels:** the first version used torchvision's 91-id COCO list, and the cats came out as
  "bird". This model (maskrcnn-benchmark lineage) uses **81 contiguous classes** (cat = 16,
  remote = 66); `Coco.java` now has that list.
- **Every box had the last detection's mask:** the overlay reused one mutable 28x28 bitmap for all
  detections. A hardware-accelerated canvas records draw calls and uploads bitmap contents only at
  render time, so all masks came out as the last one drawn. It now creates one bitmap per detection.
- **Images shown rotated 90 degrees:** the frame rotation ignored the display rotation; see
  "Orientation".
- **Camera-mode crash (SIGSEGV in libQnnHtp's memcpy) when no RoI survived:** `ortpad` sized its pad
  buffer from `count / n`, which is 0 rows' worth for n = 0, while the HTP still copies a full bucket
  from it. Only an empty camera scene hits it (the test images always have detections). Fixed in
  `e2e_run.cpp`.

## Known limits

- **The displayed image is the processed frame**, not the live preview. That keeps boxes aligned
  with what was inferred; a small live camera thumbnail sits in the corner.
- **The APK doesn't bundle the models** (about 100 MB); `deploy.sh` copies them in with `run-as`, so
  this needs a debuggable build. A release build would download them into internal storage.
- **Fixed 800x1088 input**, like the rest of this pipeline: frames are scaled to fit, top-left
  aligned.
- **Camera preprocessing is ~15 ms** (vs ~2.5 ms from a bitmap), see above.
