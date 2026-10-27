# BEVFormer-tiny on the Hexagon HTP

The real BEVFormer-tiny (official `bevformer_tiny_epoch_24.pth`), rebuilt in plain PyTorch
(no mmcv / mmdet / mmdet3d / mmdeploy), validated against upstream semantics on real nuScenes-mini
frames, then exported and run piece by piece on the phone's HTP through QNN.

## Files

| file | what |
|---|---|
| `fetch_data.sh` | sha256-pinned checkpoint + the camera-only prefix of nuScenes-mini (no account needed) |
| `model.py` | backbone+FPN, 3-layer encoder (TSA + SCA), 6-layer decoder + head, NMS-free decode, host-side geometry; `load_official()` name map; rank-5 MSDA + a verbatim copy of mmcv's `multi_scale_deformable_attn_pytorch` |
| `nuscenes.py` | lidar2img / image preprocessing / can_bus exactly as BEVFormer's converter + test pipeline, GT for a sanity match |
| `validate.py` | rank-5 path vs the upstream-literal path (6-D MSDA, SCA `nonzero()` rebatch), temporal, detections vs GT |
| `export.py` | one piece -> ONNX (TorchScript exporter, opset 17) -> ORT CPU check -> onnxsim -> check; phone inputs + fp32 reference outputs |
| `run_phone.sh`, `compare_out.py` | partition report + strict all-HTP run of a piece (`../../vision_models_probe/partition_report.sh`), outputs vs fp32 |
| `e2e_phone.py` | whole model on the HTP frame after frame (phone outputs chained, HTP prev_bev carried) vs fp32 torch and GT |
| `profile_ops.py` | per-op-type share of an HTP execute from a QNN detailed-profiling CSV |
| `sensitivity.py` | ORT CPU sweep: which op types int8 hurts (all-but-T / only-T cosines) |
| `quantize.py` | int8 QDQ pieces with onnxsim's whole-graph quantizer (`onnxsim.full_qdq`): calibration set, backbone, mixed-precision encoder/decoder policies |
| `bisect_precision.py`, `bisect_run.sh` | expose chosen intermediates as outputs, run on the HTP, per-tensor cosine vs ORT CPU |
| `msda_hvx/` | the encoder split around the generic HVX MSDA kernel (`../../msda_hvx/`) and its phone runner: see "Encoder with the sampling on the HVX" below |
| `tinygrad_port/` | the same encoder in tinygrad (`bevformer_tg.py`), its DSP codegen (`capture_msda.py`) and the phone build that runs the tinygrad MSDA beside the hand kernel: see `tinygrad_port/README.md` |

Reproduce (each heavy step under `systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0`):

```sh
./fetch_data.sh                                   # ~/.cache/onnxsim-bevformer/{*.pth,nuscenes-mini}
C=~/.cache/onnxsim-bevformer
python3 validate.py --ckpt $C/bevformer_tiny_epoch_24.pth --data $C/nuscenes-mini --work $C/work
for p in backbone1 backbone6 enc1 enc3 decoder; do
  python3 export.py $p --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work
  ./run_phone.sh $C/work $p
done
```

## Validation (CPU fp32, scene-0103 frames 0-2, peak 1.3 GB)

* `msda_rank5` vs mmcv's reference: max abs diff 0.
* checkpoint: all 643 tensors mapped; unused = `code_weights` (loss weights) + `cls_branches.0-4`
  (aux classifiers of the 5 intermediate decoder layers; inference decodes only the last layer).
* rank-5 encoder/decoder vs the upstream-literal path: max abs diff 0 on every frame (the SCA
  all-queries + visibility-mask formulation is exact).
* detections (score >= 0.3, same class within 2 m of a GT center): frame 0 8/23 GT, frame 1 11/29,
  frame 2 16/30 -- temporal context (prev_bev) helps as expected. The 9 CAN-bus signals are 0
  (CAN-bus expansion not in v1.0-mini), so this is a sanity check, not an mAP number.

## Export (each piece its own systemd-run, MemoryMax=16G)

| piece | inputs | ONNX nodes (sim) | ORT CPU vs torch | peak RSS |
|---|---|---|---|---|
| backbone1 | img 1x3x480x800 | 121 | 1.4e-5 max abs | 1.5 GB |
| backbone6 | img 6x3x480x800 | 121 | 2.2e-5 | 2.6 GB |
| enc1 | feats 6x256x15x25, prev_bev, has_prev, shift, can_bus, ref_cam 6x2500x4x2, bev_mask | 107 | 7.7e-6 | 1.8 GB |
| enc3 | same | 245 | 1.1e-5 | 2.0 GB |
| decoder | bev_embed 2500x256 | 482 | 2.1e-4 | 1.3 GB |

Nothing came close to the 16 GB cap; no external data was needed (largest file: backbone, 98 MB).

## Phone: Snapdragon HTP (V69) via ORT 1.26 + QNN EP 2.6.0 / QNN 2.50, fp32 graph run as fp16

Frame 1 of scene-0103 (has_prev = 1), median of 10, burst perf mode:

| piece | QNN refused ops | strict all-HTP | median ms | cos vs fp32 torch |
|---|---|---|---|---|
| backbone1 | 0 | PASS | 20.4 | 1.00000 |
| backbone6 | 0 | PASS | 129-184 (shared phone, varies) | 1.00000 |
| enc1 | 0 | PASS | 105 | 0.99999 |
| enc3 | 0 | PASS | 241 | 0.99999 |
| decoder (+head) | 0 | PASS | 25 | 1.00000 (cls and bbox) |

Every piece of the real model runs entirely on the HTP (rank-5 MSDA; nothing refused).

### End to end on real frames (`e2e_phone.py`, scene-0103 frames 0-5)

The three pieces chained on the phone, each fed the previous piece's HTP output, with the HTP's
own BEV carried to the next frame as prev_bev (rotated/shifted on the host):

| | fp32 torch (CPU) | HTP fp16 |
|---|---|---|
| per-frame cos vs fp32 (feats / bev / cls / bbox) | - | 1.00000 / 0.99999 / >=0.99999 / 0.99999, no drift over 6 frames |
| GT matched (same class, <2 m, score >= 0.3), 6 frames | 106 / 190 (273 dets) | 105 / 190 (274 dets) |
| latency per frame | - | backbone6 126 + enc3 241 + decoder 25 = ~392 ms (2.5 FPS) |

fp16 on the HTP is accuracy-neutral for BEVFormer-tiny. The encoder (60% of the time, 6
GridSamples on 2500 queries) is the next target; int8 PTQ of the backbone the other.

### fp16 precision bisect (the "cos 0.986" of the plan's synthetic probe; 0.916 on the real model)

Two independent causes, both fixed in `model.py` with exact (fp32-identical) rewrites:

1. **ref_cam overflow.** Upstream divides projected points by `max(depth, 1e-5)`, so pillar
   points behind a camera land at |xy| up to 6.7e6, beyond fp16's 65504. Clamping to [-1, 2] (originally [-5, 6]) on
   the host is exact (validate.py: clamped vs unclamped max abs diff 0) because such points still
   sample zero padding.
2. **QNN miscompiles stack -> reshape -> Gemm.** TSA projects its 2-frame value queue as
   `value_proj(stack([prev, cur]))` = reshape (5000, 256) -> Gemm. On the HTP that Gemm output
   is scrambled (cos 0.12 vs ORT CPU), but only when the intermediates are *not* graph outputs:
   exposing all of them (the first bisect attempt) hides the bug (cos 0.99999). Found by
   exposing a few tensors at a time (`bisect_run.sh`): the Concat is correct, the Gemm right
   after the reshape is not. Projecting each frame separately and stacking afterwards fixes it
   (enc1 cos 0.916 -> 0.99999) and is 36% faster (165 -> 105 ms).

## int8 on the HTP (`quantize.py`, `onnxsim.full_qdq`)

`onnxsim.quantize_static` only wraps MatMul/Gemm/Conv *inputs* in Q/DQ, which leaves every Conv
output (and each Relu/Add/MaxPool) as float, so the HTP runs it in fp16. `onnxsim.full_qdq`
(new) quantizes the whole graph into QDQ node units: calibrated uint8 activations
(`onnxsim.calibration.calibrate`), int8 per-channel weights, int32 biases, data-movement ops
sharing their input's qparams, Relu folded into the producer's Q (zp 0), plus `quantized_io` for
uint8 (and NHWC) graph I/O. That covers the Mask R-CNN backbone's hand rewrites
(`../../htp_exploration/ceiling/`: int8 residual Adds, uint8 I/O, NHWC image input) by
construction.

Calibration: fp32 torch over the first 3 keyframes of scene-0061, -0553, -0757 and -1077 (night),
chained with prev_bev: 12 samples, all disjoint from the scene-0103 evaluation frames.

```sh
C=~/.cache/onnxsim-bevformer; S="systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0"
$S python3 quantize.py calib --ckpt $C/bevformer_tiny_epoch_24.pth --data $C/nuscenes-mini --work $C/work
$S python3 quantize.py backbone --work $C/work          # -> backbone6.q8.onnx (+ .json io qparams)
adb push $C/work/backbone6.q8.onnx /data/local/tmp/bevformer_tiny/
$S python3 e2e_phone.py --ckpt ... --data ... --work $C/work --backbone backbone6.q8
```

The backbone is calibrated on `backbone1` (one camera per batch, 72 batches; peak 4.2 GB) and
the ranges are applied to `backbone6` (identical tensor names). Result, strict all-HTP, burst:

| backbone (6 cameras) | ms | feats cos vs fp32 | e2e GT matched (6 frames, scene-0103) |
|---|---|---|---|
| fp16 (fp32 graph) | 126 | 1.00000 | 105 / 190 |
| int8 QDQ, uint8 NHWC image in, uint8 feats out | **20.9** | 0.994 | **108** / 190 (fp32 torch: 106) |

6.0x faster at the same detection quality (the feats cos of 0.994 becomes bev 0.997-0.998 and
cls/bbox >= 0.9997 after the encoder). The host quantizes the normalized image
(`round(x / 0.01866) + 114`, NHWC) where it normalizes it anyway.

### Encoder: exact rewrite first, then mixed precision

QNN's per-op profile of one fp16 encoder layer (`QNN_EXTRA='profiling_level=detailed,
profiling_file_path=...'` on `qnn_run_multi`, then `profile_ops.py`) shows the Linear layers
are not the cost: **Gemm 0.6%**, GridSample 40%, and most of the rest is elementwise work on the
SCA's (cams x queries x heads x points x 2) sampling-coordinate tensors (Gather 8.7%, Expand
5.7%, Sub 6.6%, Mul 10.9%, ...). So int8 Linears alone cannot help. First, an exact fp16 rewrite
of the SCA (`model.py`): build the grid in grid_sample's layout from its small factors (the
[0, 1] -> [-1, 1] map, the anchor repeat and the head permute on ref_cam and the offsets
*before* they broadcast), with every tensor rank <= 4 (a rank-5 broadcast version fails to
execute on the HTP, `QNN_COMMON_ERROR_SYSTEM`). validate.py vs upstream: max abs 7e-6.

| encoder | enc1 ms | enc3 ms | bev cos |
|---|---|---|---|
| fp16, as in #1849 | 104 | 241 | 0.99999 |
| fp16, SCA rewrite | **55.4** | **151** | 0.99999 |

After the rewrite GridSample is 56% of the layer (SCA 48%, TSA 9%), then Mul + ReduceSum over
the sampled values (17%). The next lever is an HVX deformable-sampling kernel (the RoiAlign
channels-last kernel is the template), not quantization: done below, "Encoder with the sampling
on the HVX" (151 -> 52 ms).

Mixed-precision policies (`quantize.py enc1 --policy ...`), strict all-HTP, enc1 vs fp32:

| policy | what | ms | cos |
|---|---|---|---|
| fp16 | - | 55.4 | 1.00000 |
| lin8 | Gemm/MatMul int8 only | 74.9 | 0.985 |
| all8 | int8 except LayerNorm, Softmax, GridSample | 68.8 | 0.971 |
| all16 | same, uint16 activations | 89.5 | 0.99998 |
| all8gs | int8 everything but LayerNorm/Softmax (GridSample int8) | 40.9 | 0.971 -> **0.991** with the [-1, 2] ref_cam clamp |
| all16gs | same, uint16 | 65.3 | 0.99998 |
| mix8 | all8gs + uint16 sampling coordinates (`sampling_coordinate_tensors`) | 78.4 | 0.994 |

Only an int8 GridSample beats fp16. Its uint8 grid is what costs accuracy (`sensitivity.py`: the
Sub/Add/Reshape feeding the grid are the worst single op types), and a uint16 grid makes
GridSample slower than fp16. Clamping ref_cam to [-1, 2] instead of [-5, 6] (still exact, max
abs 0; [-0.5, 1.5] is not) shrinks the grid range 3.7x: all8gs cos 0.971 -> 0.991. Over three
layers and six chained frames that is still a visible loss:

| whole model, scene-0103 frames 0-5 | backbone | enc3 | decoder | total | FPS | GT matched |
|---|---|---|---|---|---|---|
| fp32 torch (CPU) | | | | | | 106 / 190 |
| #1849: fp16 everywhere | 126 | 241 | 25 | 392 ms | 2.6 | 105 |
| **int8 backbone + fp16 encoder (SCA rewrite) + fp16 decoder** | 21 | 151 | 25 | **197 ms** | **5.1** | **107** |
| same, encoder all8gs | 21 | 121 | 25 | 167 ms | 6.0 | 103 (bev cos 0.98) |

The first is the default (no accuracy cost); all8gs trades 3 GT matches for 30 ms. The decoder
stays fp16: every policy was slower there (lin8 34 ms, mix8 42, all8gs 54 vs fp16 25). The
QDQ decoder has many more nodes, and its tensors are small.

`onnxsim.full_qdq` mixed-precision details that the HTP needs:
- An `exclude`d node sandwiched between quantized nodes would still form a QDQ unit (DQ in, Q
  out) and run quantized. `quantize_full_qdq` keeps its output float instead.
- A quantized node computes in its output's dtype. An input of the other dtype gets a
  DQ -> Q' -> DQ' convert; GridSample's grid is exempt. Without this, QNN rejects a Gemm with a
  uint8 input and a uint16 output, and its weight DQ is stranded on the CPU.
- Weights are quantized only for nodes that end up real QDQ units.

## Encoder with the sampling on the HVX (`msda_hvx/`)

The HTP spends 73% of an fp16 encoder layer building sampling grids, in GridSample, and in the
Mul + ReduceSum after it. That whole span, per TSA and per SCA, becomes one FastRPC call to the
generic multi-scale deformable attention kernel `../../msda_hvx/` on the CDSP's HVX; its README
has the kernel, its C API and how it got fast. The HTP keeps everything else: the Linears,
softmax, LayerNorm and the FFN. In the kernel's terms:
- **TSA** is `NV = 2` value maps (the queue) with per-map offsets (`NO = 2`), averaged.
- **SCA** is `NV = 6` cameras with shared offsets, pillar anchor `p % 4` (`R = 4`), and a
  per-(camera, query) visibility mask. 81% of the pairs are invisible on a real frame (2863 of
  15000 visible), and the kernel skips them.
- Both are one level with reference points + pixel offsets (`MSDA_REF_PIX`).

| file | what |
|---|---|
| `split.py` | BEVFormer's calls in the kernel's terms (`msda_fused`); the encoder split into 7 pieces around them (`pre`, `mid0-2`, `post0-2`). `check`: split vs `Encoder` (max abs 7e-6 on 3 frames). `dump`: real kernel calls as case directories. `export`: the pieces to ONNX + onnxsim |
| `enc_run.cpp` | the encoder on the phone: 7 HTP pieces + 6 kernel calls in one process, all tensors in rpcmem buffers ORT writes into |
| `e2e_msda.py` | backbone (int8) -> `enc_run` -> decoder on scene-0103's frames, like `e2e_phone.py`, in a phone directory of its own, every adb call under the host's phone lock (`PHONE_RUN`) |
| `build.sh` | the core's skel + stub, then `enc_run` and `qnn_run_multi` |

Checks:
- `split.py check` / `dump`, then `../../msda_hvx/msda_host_check` on all 6 calls of a real frame:
  rel <= 3.4e-6.
- `tests/test_msda_hvx.py` checks this glue against the model's own `msda_rank5` math (TSA and
  SCA), plus the kernel itself on BEVFormer-shaped synthetic calls.

Kernel alone on the phone, real frame, 4 threads, median of 10, under the phone lock:
- TSA 4.4 ms (4.8 wall incl. FastRPC);
- SCA 4.0 ms (4.4 wall).

Encoder on the phone (`enc_run`, frame 1, median of 10, burst, under the phone lock):

| step | ms |
|---|---|
| `pre` (3 layers' SCA value projections + layer 0's TSA inputs) | 6.8 |
| `mid0-2` (TSA output proj + LN + SCA offsets/weights), each | 2.3 |
| `post0-1` (SCA output proj + LN + FFN + LN + next TSA inputs), each | 5.8-6.0 |
| `post2` | 2.1 |
| TSA call, each (in-DSP / wall) | 4.5 / 4.8 |
| SCA call, each | 4.3-4.4 / 4.6 |
| **encoder** | **55.2** (HTP pieces 27.1, kernel calls 27.9 incl. 1.4 FastRPC) |

End to end (`e2e_msda.py`, scene-0103 frames 0-5, the phone's own BEV carried as prev_bev):
- **The encoder alone** (fp32 torch on the phone's own encoder inputs vs the phone's output):
  cos 0.999999 on every frame.
- **Whole model:** bev cos 0.9966-0.9980 vs fp32, cls >= 0.99979. The gap is the int8
  backbone's (feats cos 0.994), the same as with the fp16 HTP encoder.

| whole model, scene-0103 frames 0-5 | backbone | encoder | decoder | total | FPS | GT matched |
|---|---|---|---|---|---|---|
| fp32 torch (CPU) | | | | | | 106 / 190 |
| int8 backbone + fp16 HTP encoder + fp16 decoder (above) | 21 | 151 | 25 | 197 ms | 5.1 | 107 |
| **int8 backbone + split encoder (HTP + HVX MSDA) + fp16 decoder** | 21 | **55** | 24 | **100 ms** | **10.0** | **107** |

Reproduce (heavy host steps under `systemd-run ... MemoryMax=16G`):

```sh
cd msda_hvx
python3 split.py check  --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work
python3 split.py dump   --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work   # -> $C/work/msda_io/l{0,1,2}_{tsa,sca}
python3 split.py export --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work   # -> $C/work/msda_split/*.sim.onnx
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... OUT=build ./build.sh
PHONE_RUN=~/.cache/android-phone/phone-run PHONE_LOCK_OWNER=<branch> \
  python3 e2e_msda.py --ckpt $C/bevformer_tiny_epoch_24.pth --data $C/nuscenes-mini --work $C/work --build build
```

**uint8 TSA value maps: tried, no gain in the chain.** The kernel reads uint8 value maps (see
`../../msda_hvx/README.md`), and in isolation that makes a real TSA call 9% faster (4.52 ->
4.10 ms). So:
- `split.py calib-tsa` measures per-layer ranges on quantize.py's 12 calibration frames (other
  scenes). Quantizing costs nothing measurable there: encoder cos 1.0000000 vs fp32.
- `split.py export --tsa-u8` appends a QuantizeLinear to the `tsa_v` outputs of `pre` and
  `post0/1`, so the HTP emits them as uint8 (`msda_split_u8/`, `e2e_msda.py --pieces
  msda_split_u8`).

On the phone (scene-0103, same 6 frames) it doesn't pay off:
- the encoder median over the 6 frames is 54.9 ms, vs 55.2 with fp32 values: within run-to-run
  noise (55-58 ms);
- `pre`/`post0-1` are unchanged (6.7 / 5.9 ms): the smaller outputs save nothing on the HTP side;
- the uint8 TSA calls take 4.6-5.1 ms in the chain, vs ~4.5 ms with fp32.

GT matched: 107 / 190, the same as fp32 values. fp32 values stay the default. The SCA was left on fp32: uint8 is
slower for it even in isolation.

Where the remaining 55 ms go, and the next levers:
- **The HTP pieces' graph I/O is fp32:** ~12 MB in and out per frame. Making the TSA value
  outputs uint8 didn't shorten the pieces (above). The per-piece time is dominated by the
  Linears and the fixed per-execute cost, not the output conversion.
- **`pre` computes all 3 layers' SCA value projections up front** (6.9 MB of fp32 output). They
  depend only on the image features, so they could also run in the backbone's call.
- **The kernel's multiply-accumulate phase is its larger half.** See `../../msda_hvx/README.md`.

### The whole frame in one process (`msda_hvx/frame_run.cpp`, `frame_e2e.py`)

`e2e_msda.py` runs each piece as its own process and carries tensors through the host. `frame_run`
chains backbone -> feats dequantize (CPU) -> encoder -> decoder in one process, every tensor in rpcmem,
and carries the BEV itself: prev_bev is a row gather of its own last BEV (`rot_idx.i32`, torchvision's
nearest rotate of an index image, checked exact against `rotate_prev_bev`). The host only prepares
per-frame inputs that don't depend on any output. Modes:
- `seq`: one frame at a time (latency);
- `pipe`: backbone | encoder | decoder threads, two slots between stages (throughput); its outputs are
  compared bit for bit with seq's;
- `conc`: the backbone / decoder alone, the 6 sampling calls alone, then each pair at once.

Phone, scene-0103 frames 0-5 x 5 passes, under the phone lock (`FRAME_RUN_TABLE`):

| configuration | backbone | encoder (msda) | decoder | seq ms/frame | pipe ms/frame (FPS) | pipe latency | GT matched |
|---|---|---|---|---|---|---|---|
| #1859 chain, one process per piece (`e2e_msda.py`) | 21 | 55 | 24 | ~100 | - | - | 107 / 190 |
| `frame_run`, backbone6 + fp16 decoder | 22.1 | 55.6 (27.7) | 25.3 | 103.2 | 97.4 (10.3) | 222 ms | 107 / 190 |
| **`frame_run`, backbone1 x 6 + fp16 decoder** | 21.3 | 55.1 (27.8) | 24.9 | 101.8 | **90.5 (11.1)** | 208 ms | **107 / 190** |
| `frame_run`, backbone1 x 6 + split decoder (HVX sampling) | 21.1 | 55.1 (27.8) | 29.9 | 106.6 | 89.7 (11.2) | 275 ms | 107 / 190 |

bev cos vs fp32 0.9965-0.9980 and cls >= 0.99979 on every frame in every row (the int8 backbone's
gap, as before); `pipe` outputs are bit-identical to `seq`'s in every configuration.

Findings:
- **The HTP and the HVX run concurrently.** The backbone takes 21.9 ms with the 6 sampling calls running
  alongside (22.0 alone); the calls take 31.2 ms (29.4 alone).
- **The HTP is the bottleneck of a pipelined frame:** ~75 ms of HTP pieces per frame (backbone 22,
  encoder pieces 28, decoder 25) against ~28 ms of HVX. So pipelining alone gives little (97 vs 103
  ms/frame); what helps is moving HTP work to the HVX.
- **A per-camera backbone pipelines better:** `quantize.py backbone` also writes `backbone1.q8` (same
  calibrated ranges; `backbone6.q8` regenerates byte-identical). Six batch-1 executes cost the same as
  one batch-6 (21.3 vs 22.1 ms) but let the encoder's HTP pieces interleave: 97.4 -> 90.5 ms/frame.
- **Decoder per-op profile (fp16):** the self-attention's softmax x V MatMul is 31% (900 x 900
  attention in layers 1-5), the deformable sampling (GridSample, grid math, ReduceSum) ~35-40%.
- **Decoder sampling on the HVX: tried, slower.** `dec_split.py` splits the decoder like the encoder
  (dpre, dmid0-4, dpost around 6 kernel calls; layer 0's inputs are constants; torch split vs
  `Decoder` max abs 0). On the phone it takes 29.9 ms vs 24.9 for the fp16 graph, with the same
  accuracy: seven fp32-I/O pieces and six RPCs cost more than the GridSample they replace at 900
  queries x 4 points. The fp16 graph stays the default. The self-attention is the bigger decoder cost.

Next levers (not done here):
- the HTP pieces' ~75 ms/frame is the pipelined bottleneck: int8 on the encoder pieces' Linears now
  that GridSample is off the HTP (the earlier int8-encoder loss came from the GridSample grid), and
  int8 on the decoder's self-attention matmuls;
- the encoder's `pre` (6.8 ms, 3 layers' SCA value projections) could fold into the backbone graph.

### Follow-up levers on the frame runner (`msda_hvx/quantize_split.py`, `frame_e2e.py --pieces`)

Phone, scene-0103 frames 0-5 x 5 passes, `frame_run` with backbone1 x 6 + fp16 decoder, under the phone
lock (same session as the baseline row, which reproduces #1879's 90.5 ms):

| encoder HTP pieces | host bev cos vs fp32 (scene-0103 / calib) | encoder (msda) | seq ms/frame | pipe ms/frame (FPS) | phone bev cos | GT matched |
|---|---|---|---|---|---|---|
| **fp16 (`msda_split`, default)** | - | **54.4 (27.4)** | **101.3** | **89.6 (11.2)** | 0.9965-0.9980 | **107 / 190** |
| int8 except LayerNorm/Softmax (`all8`) | 0.99754 / 0.99727 | slower* | - | - | 0.9941-0.9956 | 108 / 190 |
| `all8`, sampling offsets uint16 (`all8o16`) | 0.99810 / 0.99785 | 73.2 (27.5) | 120.2 | 108.3 (9.2) | - | 110 / 190 |
| int8 Gemm/MatMul only (`lin8`) | 0.99821 / 0.99768 | 102.6 (28.2) | 150.1 | 137.0 (7.3) | - | 107 / 190 |

\* `all8` was measured while the phone was contended (the baseline read 216 ms/frame pipelined in the
same run), so only its accuracy counts; it is also below the 0.996 bev-cos bar.

**Lever 1, int8 encoder pieces: accurate now, but slower on the HTP.** With the sampling on the HVX,
int8 no longer costs accuracy: `all8o16` even matches 110 GT, and host bev cos is 0.998. But every int8
variant makes the encoder slower (54 -> 73 -> 103 ms): the pieces are small (2500 x 256 Linears), their
graph I/O is fp32, and each QDQ unit adds per-op conversion work that the int8 matmuls don't win back.
fp16 stays the default. (Uint8 graph I/O would need frame_run to carry uint8 buffers between pieces; the
earlier uint8 TSA-output test saved nothing on the HTP side, so it is not pursued here.)

**Lever 2, decoder self-attention int8 / 16-bit** (`quantize.py decoder --policy mm8|mm16`: only the
decoder's 10 activation x activation MatMuls, i.e. Q x K^T and softmax x V of layers 1-5):

| decoder | decoder ms | seq ms/frame | pipe ms/frame (FPS) | cls cos (min) | GT matched |
|---|---|---|---|---|---|
| **fp16 (`decoder.sim`, default)** | **25.0** | **101.9** | **90.2 (11.1)** | 0.99979 | **107 / 190** |
| `mm8` | 24.0 | 100.9 | 90.0 (11.1) | 0.99965 | 107 / 190 |
| `mm16` | 27.0 | 103.5 | 92.1 (10.9) | 0.99979 | 107 / 190 |

`mm8` saves 1 ms of decoder time at the same GT, but the pipelined frame doesn't move (the HTP total per
frame is ~75 ms, and 1 ms is within run-to-run noise), so fp16 stays the default.

**Lever 3, fold `pre` into the backbone: bounded, not done.** Timed alone on the HTP (`qnn_run_multi`,
median of 20): `pre` 6.46 ms, `pre` without the SCA value projections 4.97 ms, the SCA value projections
alone 1.58 ms. Folding moves those ~1.5 ms of HTP work into the backbone graph rather than removing it,
and makes the backbone's output 3x larger (3 layers' `sca_v` instead of `feats`). The ceiling is
therefore < 1.5 ms of a 90 ms pipelined frame, for a frame_run change; not worth it.

**Lever 4, backbone recalibration** (`quantize.py backbone --method mse|percentile`, same graph, only the
scales change, so the speed is unchanged):

| backbone calibration | phone bev cos (min / max) | GT matched |
|---|---|---|
| **minmax (default)** | 0.9965 / 0.9980 | **107 / 190** |
| mse | 0.9970 / 0.9979 | 104 / 190 |
| percentile (99.999) | 0.9971 / 0.9980 | 105 / 190 |

mse and percentile raise bev cos slightly but match fewer GT boxes: minmax stays.

**Where this leaves BEVFormer-tiny:** 89.6-90.2 ms/frame pipelined (11.1-11.2 FPS), 107/190 GT. The
pipelined frame is bounded by ~75 ms of HTP work that neither int8 (slower on these small pieces) nor
re-partitioning shrinks. The remaining lever is moving more HTP work to the HVX, where the kernel has
headroom (~28 ms per frame).

Phone-contention note: the Mask R-CNN demo app (`org.onnxsim.maskrcnndemo`) was running on the phone
during parts of this session and slowed the HTP ~2x. Every timing in the tables above was taken with the
app idle (checked with `top` right after each run); runs taken while it was busy were discarded.

Reproduce:

```sh
cd msda_hvx
python3 quantize_split.py quant --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work --policy all8o16
python3 quantize_split.py eval  --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work --pieces msda_split_all8o16
PHONE_RUN=~/.cache/android-phone/phone-run PHONE_LOCK_OWNER=<branch> R=/data/local/tmp/<dir> \
  python3 frame_e2e.py --ckpt ... --data ... --work $C/work --build <build> --backbone backbone1.q8 \
  --pieces msda_split_all8o16 --modes seq,pipe
```

### Encoder and decoder from the tinygrad fork through onnx-remote (`msda_hvx/e2e_remote.py`)

The int8 backbone stays on the HTP. The 3-layer encoder (`enc3.msda.onnx`) and the decoder (`decoder.msda.onnx`), from
`export_msda_op.py`, are compiled by the tinygrad fork. That is fork commit `9edd28d9e`, where every kernel is generated
(see `tools/onnx-remote/README.md`). They run on the cDSP through onnx-remote. Each piece is fed the previous piece's phone
output, and the phone's own BEV is carried to the next frame, as in `e2e_msda.py`. Results for scene-0103 frames 0-5:

| | backbone (HTP) | encoder | decoder | frame | GT matched |
|---|---:|---:|---:|---:|---|
| fp32 torch (CPU) | | | | | 106 / 190 |
| previous best: `frame_run`, HTP pieces + hand HVX MSDA (above), seq | 21.3 | 55.1 | 24.9 | 101.8 | 107 / 190 |
| **tinygrad encoder + decoder (cDSP DSP time)** | 21.4 | **29.6** | **22.7** | **73.8** | 106 / 190 |
| same BEV, fp16 HTP decoder instead | | | 24.8 | | 107 / 190 |

- **Accuracy:** the encoder alone is exact against fp32 torch on the phone's own inputs: cos 1.000000 on every frame. The
  whole model has bev cos 0.9966-0.9980 and cls >= 0.99979, the int8 backbone's gap as before. Detections match the HTP
  decoder's within one GT.
- **Encoder and decoder times** are the program's DSP time (median of the 6 frames, 10 runs each).
- **The frame time** is the sum of the backbone's HTP time and the two programs' DSP times. The pieces don't run in one
  process yet.
- **Through onnx-remote** (`run_compiled` RPC-inclusive, adb-forwarded TCP from the host), the encoder takes 268 ms and the
  decoder 104 ms. That is the transport (about 12 MB of fp32 encoder inputs and BEV over USB), not the phone.
  `frame_run` with the tinygrad artifacts loaded in-process is the next step for an app-level number.

```sh
CLIENT=<onnx-remote-client> WORKER=<onnx-remote-hexagon-worker> COMPILER_PORT=<compile_v69.sh service> \
  PHONE_RUN=~/.cache/android-phone/phone-run PHONE_LOCK_OWNER=<branch> \
  python3 msda_hvx/e2e_remote.py --ckpt $C/bevformer_tiny_epoch_24.pth --data $C/nuscenes-mini --work $C/work \
  --htp-decoder decoder.sim
```

### The tinygrad programs in the frame runner (`frame_run` TG_ENC, `frame_e2e.py --tg-enc`)

`frame_run` can load tinygrad-compiled programs in-process (`tg_prog.h`). A program is a `.tghx` artifact, the container
onnx-remote's worker loads. `frame_run` writes its skel next to the binary and opens it over FastRPC. The program's packed
input and output buffers live in rpcmem, and the frame's inputs are staged straight into them: dequantized feats, the
prev_bev gather, and the host geometry. `frame_e2e.py` now also writes each frame's `shift.f32` and `bev_mask.f32`.
- **Two separate programs** (TG_ENC and TG_DEC) don't fit the DSP process together: the second program's weight upload
  fails (encoder 9.3 MB weights + 36.6 MB scratch, decoder 11.8 + 12.9 MB).
- **One program instead:** `export_msda_op.py frame` merges `enc3.msda.onnx` and `decoder.msda.onnx` into one graph, with
  `bev_embed` kept as an output. The emitter reuses scratch across both parts: 21.1 MB weights, 36.7 MB scratch. A TG_ENC
  program with three outputs is the whole frame after the backbone.
- **Output packing:** that program first ran at 64.2 ms of DSP time, against 52.3 ms for the two programs separately. One
  generated kernel concatenated the three outputs into the output buffer, a select per element over all 658 K floats,
  16.4 ms. The fork (`b5eb951ad`) now stores each output into its own slice: 48.9 ms.

Phone, scene-0103 frames 0-5 x 5 passes, backbone1.q8 x 6 on the HTP, under the phone lock:

| configuration | backbone | encoder + decoder | seq ms/frame | pipe ms/frame (FPS) | pipe latency | GT matched |
|---|---|---|---|---|---|---|
| `frame_run`, HTP pieces + hand HVX MSDA + fp16 HTP decoder (above) | 21.3 | 55.1 + 24.9 | 101.8 | 90.5 (11.1) | 208 ms | 107 / 190 |
| **`frame_run`, one tinygrad program (TG_ENC=frame)** | 21.4 | **54.5** (DSP 48.9) | **76.3** | **71.9 (13.9)** | 128 ms | 106 / 190 |

- **Accuracy:** bev cos 0.9965-0.9980 and cls >= 0.99979 on every frame, the int8 backbone's gap as before. Pipe outputs
  are bit-identical to seq's. fp32 torch also matches 106 GT.
- **FastRPC call cost:** the call costs 5.6 ms over the program's own DSP time, from about 5.6 MB of fp32 inputs and 2.6 MB
  of outputs going through FastRPC.
- **Pipelining:** it gains less than with the HTP pieces (76 -> 72 ms). The backbone (QNN HTP) and the tinygrad program
  share the cDSP.

```sh
python3 export_msda_op.py frame --work $C/work           # after enc3 / decoder
python3 <tinygrad>/examples/dsp_graph_onnx.py $C/work/frame.msda.onnx <out> --arch v69 --artifact frame.tghx  # GEMM_FP16=1 HMX=1 ...
TINYGRAD_ROOT=<tinygrad> OUT=<build> ./build.sh
PHONE_RUN=~/.cache/android-phone/phone-run PHONE_LOCK_OWNER=<branch> R=/data/local/tmp/<dir> \
  python3 frame_e2e.py --ckpt ... --data ... --work $C/work --build <build> --backbone backbone1.q8 --tg-enc frame.tghx \
  --modes seq,pipe
```

### Why not a `scripts/android/deploy` spec (yet)

The deploy pipeline (#1853) takes one graph through fetch -> simplify -> quantize -> rewrite ->
bench -> accuracy, with images as calibration/eval data and a detection-match accuracy. Four
things are missing for BEVFormer, none small:
- **a chain of pieces**: backbone6 -> enc3 -> decoder, each with its own precision policy, where
  the backbone's uint8 feats feed the encoder;
- **host inputs per frame**: ref_cam/bev_mask from lidar2img, can_bus, shift;
- **temporal state**: the previous BEV, rotated and shifted on the host, is an encoder input;
- **nuScenes data and a 3D accuracy kind**: calibration/eval frames, and GT matching in the ego
  frame.

`quantize.py` + `e2e_phone.py` cover all four for this model. The piece that generalizes is the
quantizer: the deploy spec's `quantize` section could call `onnxsim.full_qdq.quantize_full_qdq`
(`op_types` / `exclude_nodes` / `tensor_dtypes` map directly onto spec keys).
