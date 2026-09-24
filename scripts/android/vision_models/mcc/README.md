# MCC (single-image 3D reconstruction) on the Xiaomi 12S

[Multiview Compressive Coding](https://mcc3d.github.io/) (Wu, Johnson, Malik, Feichtenhofer,
Gkioxari, CVPR 2023) reconstructs an object's full 3D shape and color from one RGB-D view: an
encoder reads the image and the seen points, and a decoder is queried at 3D points for
occupancy and color.

**License: the upstream code and weights are CC BY-NC 4.0 (non-commercial).** Nothing from
upstream is vendored here: `model.py` imports the upstream `mcc_model.py` from a clone
(`MCC_REPO`, default `~/.cache/onnxsim-mcc/MCC`) and loads the released checkpoint
`co3dv2_all_categories.pth` (CO3D v2, all categories) from `dl.fbaipublicfiles.com/MCC/`.

## Model facts (from the code)

- RGB encoder: ViT-B (`get_mcc_model`: width 768, 12 blocks, 12 heads) on 224x224, 197 tokens.
- XYZ encoder: seen points resampled to 112x112, a 1-block transformer per 8x8 window
  (196 windows x 65 tokens), then another ViT-B stack (12 blocks) -> 197 tokens.
- Decoder: 8 blocks, width 512, 16 heads, on `[197 seen tokens ; Q queries]`; output 1 occupancy
  logit + 3x256 color logits (softmax with temperature 0.1 -> expected value) per query.
- Queries: a grid over [-3, 3]^3. The demo's default granularity 0.05 is 120^3 = 1.73M queries;
  the training/eval default 0.1 is 60^3 = 216k.

## Deployment split (`model.py`, exact)

Upstream masks the decoder so seen tokens attend only to seen tokens and each query only to the
seen tokens and itself. So the seen stream is query-independent:

- `Encoder`: both encoders + the decoder's seen-token stream, run once per image, outputs each
  decoder block's seen K/V (8 x 16 heads x 197 x 32, twice).
- `QueryDecoder`: a fixed-size chunk of queries, each attending to `[K_seen ; k_self]`. Cost is
  linear in the chunk (about 27 MMAC per query) instead of upstream's (197+Q)^2 masked attention.
- The XYZ encoder's 8x8 window partition (a rank-6 view upstream) moves to the host; everything
  in the graphs is rank <= 4.

## Results

Checkpoint `co3dv2_all_categories.pth`, sha256
`ca861bee4c2cb27acc6855da34227ce7026cf9eb275171da3c5a33976b3d86bd` (2.4 GB with optimizer state).
Demo input: upstream `demo/quest2` (image + iPhone point cloud + mask).

**Split vs upstream** (`validate.py`, 2000 queries, host fp32): occupancy logit max abs 1.5e-5,
color max abs 1.7e-5 -- the K/V-cache split is exact.

**Host fp32 reference** (split, 8 threads; the reference every phone run is scored against):

| granularity | queries | host fp32 | occupied (p > 0.1 / 0.3 / 0.5) |
|---|---|---|---|
| 0.1 | 216,000 | 29.6 s | 20,684 / 13,604 / 9,725 |
| 0.05 (demo default) | 1,728,000 | 243 s | 165,139 / 108,881 / 78,007 |

**Phone (Xiaomi 12S, strict all-HTP, fp16, ORT + QNN EP, medians under the phone lock):**

| piece | ms | vs host fp32 |
|---|---|---|
| encoder (RGB + XYZ ViT-B + decoder seen stream -> K/V) | 202 | K cos 0.99997, V cos 0.9997 |
| decoder, 512-query chunk | 38 (74 us/query) | occ cos 0.999999 |
| **decoder, 1024-query chunk** | **65 (64 us/query)** | occ cos 0.999999, color cos 0.9998 |
| decoder, 2048 / 4096 / 8192 | 214 / 502 / 769 | (per-query cost grows past 1024) |

Rank <= 4 everywhere: upstream/timm build `(B,N,3,heads,d)` (rank 5) for q/k/v; `split_qkv` uses
three linears instead (exact). Softmax is 44% of a decoder chunk (per-op QNN profile); an exact
"split" softmax (max/exp/sum without the 198-wide concat) was slower (608 vs 502 ms at 4096) and
inexact on the HTP (fp16 Exp), so the concat softmax stays. A variant that pre-scaled q before the
concat softmax failed QNN graph finalize (`QNN_COMMON_ERROR_MEM_ALLOC`); the exported form scales
the scores.

**Query reduction** (`queries.py`, exact: each query's prediction depends only on its xyz and the
0.4/0.2/0.1/0.05 grids nest, so a strategy only loses occupied points it never queries):

| target | strategy | queries | recall of dense occupied (p > 0.3) | projected phone |
|---|---|---|---|---|
| 0.1 | dense | 216,000 | 1.000 | 14.0 s |
| 0.1 | 0.2 -> 0.1, refine where p > 0.05 | 55,703 | 1.000 | 3.8 s |
| 0.1 | **0.4 -> 0.2 -> 0.1, refine where p > 0.05** | **36,455** | **1.000** | **2.5 s** |
| 0.1 | 0.4 -> ..., p > 0.3 | 24,132 | 0.979 | 1.8 s |
| 0.05 | dense | 1,728,000 | 1.000 | 110.8 s |
| 0.05 | **0.4 -> ... -> 0.05, p > 0.05** | **238,470** | **1.000** | **15.5 s** |
| 0.05 | 0.4 -> ... -> 0.05, p > 0.2 | 164,480 | 0.997 | 10.7 s |

**End to end on the phone** (`mcc.py recon`: phone encoder K/V -> adaptive coarse-to-fine phone
decoder chunks, scored against the dense host fp32 grid):

| target | queries | phone (steady state) | recall | precision | chamfer | color L1 |
|---|---|---|---|---|---|---|
| 0.1 (0.4 -> 0.2 -> 0.1, p > 0.05) | 36,666 | **2.47 s** (enc 204 ms + 36 x 63 ms) | 0.991 | 0.984 | 0.0013 | 0.50 / 255 |

"Steady state" = encoder + chunks x per-chunk median; `phone.sh` starts one process per chunk
set (session load dominates its wall time), an app keeps one session.

## Decoder optimization (`dec_opt.py`)

The demo app's MCC mode spends ~85% of a reconstruction in decoder chunks. `dec_opt.py` builds
variants and scores each end to end: the phone runs the whole coarse-to-fine reconstruction for a K/V
cache, scored against that cache's dense host fp32 grid. Calibration on quest2 (iPhone points), held
out: **spyro** (upstream demo, a different object) and **quest2m** (the demo app's own run on quest2.jpg:
SAM mask + MoGe-2 points). `prep` also records each set's exact coarse-to-fine query set; calibration
samples chunks from it (the surface-heavy distribution the phone queries).

| decoder (1024-query chunk) | ms / chunk | spyro recall / precision | quest2m recall / precision | color L1 /255 |
|---|---:|---|---|---|
| fp16 (baseline) | 63 | 0.998 / 0.999 | 0.999 / 0.999 | 0.16-0.21 |
| uint16 QDQ around Softmax only | 72 | 0.998 / 0.994 | 0.999 / 0.997 | 0.12-0.21 |
| uint16 around Softmax + Gelu | 83 | 0.998 / 0.995 | 0.999 / 0.997 | 0.10-0.22 |
| w8a16 whole graph | 47 | 0.994 / 0.989 | 0.993 / 0.991 | **88-92 (broken)** |
| **w8a16, color tail fp16 (`a16c`)** | **47** | 0.994 / 0.989 | 0.993 / 0.991 | 0.65-0.90 |
| `a16c`, K/V ranges pinned (+-16 / +-8) | 47 | 0.989 / 0.984 | 0.991 / 0.991 | 0.80-1.35 |
| `a16c`, pinned + uint16 K/V graph inputs | 47 | same | same | same |
| `a16c`, MLP hidden uint8 | 54 | 0.993 / 0.987 | 0.991 / 0.989 | 0.84-1.34 |
| `a16c` at 512 / 2048 queries a chunk | 30 / 175 (59 / 85 us a query) | | 0.992 / 0.992, 0.994 / 0.991 | |
| w8a8 (color tail fp16) | 29 | 0.977 / 0.854 | 0.956 / 0.912 | 3.7-7.0 |
| w8a8 + uint16 islands (xyz embed / attention / residual / output head / MLP, alone and combined) | 29-54 | <= 0.982 / <= 0.869 | <= 0.962 / <= 0.916 | 3.3-6.9 |
| w8a8, MSE calibration | 29 | | 0.944 / 0.915 | 3.9 |

- **Softmax (44% of an fp16 chunk) can't be fixed locally**: QDQ islands add fp16 <-> int conversions
  that cost more than they save; only a whole-graph integer decoder is faster.
- **The color head needs float**: logits / 0.1 into a 256-way softmax is nearly one-hot, and uint16
  there gives L1 ~90/255 while occupancy is fine.
- **uint8 loses 5-15% precision (extra points) wherever the uint16 islands go**: the error is spread over
  the 8 blocks, not in one tensor group; `a16c` is the pick.
- Quantized K/V inputs save nothing measurable (the per-chunk K/V conversion is not the cost).
- 1024 stays the chunk size: 2048 falls off a cliff (the 16 x Q x 198 attention intermediate outgrows
  on-chip memory).
- **Encoder**: w8a16 (`quant-enc`) is only 172 vs 201 ms and breaks it (K cos 0.85, recall 0.08), so it
  stays fp16.
- **Refine threshold** (exact, from the dense host grids): `lo` 0.1 queries 13-20% fewer points than 0.05
  with recall >= 0.9975 on all three sets (0.15: 20-31% fewer, recall 0.992 on spyro); the app uses 0.1.

## NU-MCC: evaluated, MCC stays the phone target

[NU-MCC](https://arxiv.org/abs/2307.09112) (Lionar et al., NeurIPS 2023; `sail-sg/numcc`, code
Apache-2.0 -- though several utility files carry Meta headers from MCC, and the released weights
`udf-ep99.pth` (CO3D-v2, sha256 `8e92765aefe6d81b084c5e0f495d7a11dcf9b13aeede5815986a6ddd3ab21b49`)
have no separate license statement: treat them as research use) replaces MCC's global decoder
with 200 predicted anchors + a neighborhood decoder, and occupancy with a repulsive UDF. The paper
reports > 5x faster and +9.7% F1 on CO3D-v2 than MCC (upstream MCC's quadratic masked decoder).

`numcc_ref.py` runs upstream's demo inference on the host, on the same quest2 input (pytorch3d,
imported at module level but only used for training/data, is stubbed):

| | MCC (this split) | NU-MCC (upstream inference) |
|---|---|---|
| per-query decoder, host CPU | 142 us | 227 us (kNN sorts all 12,544 seen points per query) |
| queries at granularity 0.1 | 36.7k coarse-to-fine (lossless) / 216k dense | 106k (anchor box) + 23k candidates x 10 gradient steps + color ~= 827k forward-equivalents |
| host time (8 threads) | 5 s coarse-to-fine / 30 s dense | 91 s |
| phone-exportable as is | yes (static chunks) | no: the point updates use autograd (-grad UDF) through the decoder; per-query top-k over 12.5k points |

NU-MCC's "5x faster" is against upstream MCC's (197+Q)^2 masked decoder; the exact K/V split here
removes that, and coarse-to-fine removes most queries, so NU-MCC is not cheaper on this pipeline --
and its gradient-driven point refinement would need finite differences (6 extra forwards per
step) or a hand-written backward on the phone. Its quality gain can't be checked without CO3D
(its points are 0.07 chamfer from MCC's occupied set on quest2 -- no ground truth there).
**MCC stays the phone target**; NU-MCC's neighborhood gather would be an HVX-kernel job if it's
revisited.

## Depth for phone photos

The 12S has no depth sensor, so the demo needs monocular geometry. Candidates:

| model | license | size / shapes | output |
|---|---|---|---|
| **MoGe-2 ViT-S** (`Ruicheng/moge-2-vits-normal`) | MIT | DINOv2-small, official ONNX | metric point map + normals (no scale/shift fit) |
| Depth Anything V2 Metric Small | not stated on the model card | ViT-S | metric depth (needs intrinsics to unproject) |
| Depth Pro | Apple AMLR (research only) | ViT-L at 1536 px | metric depth + focal length; too heavy for the phone |

MoGe-2 ViT-S is the pick: permissive, small, and it outputs the point map MCC consumes directly.

`depth.py static` makes the official dynamic ONNX static (image 640x480, `num_tokens` 1200 as a
constant; onnxsim folds the shape math) and fuses its 12 erf-GELU chains into `Gelu` at opset 20
(QNN has no `Erf`); outputs match the dynamic model to 1.7e-6 (model sha256
`24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0`).

| MoGe-2 ViT-S, 640x480, 1200 tokens | phone (strict all-HTP, fp16) | vs host fp32 |
|---|---|---|
| points + normals + mask + scale | **257 ms** | points cos 0.999995, mask cos 1.0 |

MoGe-2 vs the demo's iPhone LiDAR cloud (quest2, same pixels, masked object):
- frames differ: MoGe is the OpenCV camera frame (x right, y down, z forward); the demo cloud is
  x right, y up, z toward the viewer -> `depth.py` flips y and z. A further **17.5 deg** rotation
  remains (the demo cloud looks gravity-aligned, ARKit world frame, not camera frame).
- depth: after the best scale, per-pixel depth error median **6.9%**, p90 17.8% (MoGe's metric
  scale says ~1.1 m where the iPhone says 0.35 m; MCC normalizes scale away).
- MCC's reconstruction is sensitive to that: MCC fed MoGe points vs fed iPhone points at
  granularity 0.1 (p > 0.3): IoU 0.13 / chamfer 0.24 (camera frame), IoU 0.17 / chamfer 0.22
  after rotating MoGe into the iPhone frame (oracle rotation, diagnostic only). No ground truth
  here, so this measures disagreement, not which is better; an app has the gravity vector (not
  the full ARKit pose) to reproduce a gravity-aligned frame.

Phone budget for one photo -> 3D: MoGe 257 ms + MCC encoder 202 ms + ~36 x 64 ms decoder chunks
~= **2.8 s** at granularity 0.1 (coarse-to-fine).

## Related work (why not these on the phone)

- **MCC-HO** (hand-held objects, MCC + hand geometry): same decoder cost as MCC plus a hand model;
  a candidate once MCC runs, not before.
- **TripoSR / SF3D** (Stability, feed-forward image -> mesh via triplane transformers): ~0.3-1B
  parameters, triplane decoding + marching cubes; heavier than MCC and meshes rather than point
  queries -- desktop-GPU class today.
- **DUSt3R / MASt3R / VGGT** (pairwise/multi-view point maps, ViT-L/ViT-g): reconstruct the visible
  scene from several views rather than completing an object from one; ViT-L at 512 px is several
  x MCC's encoder, possible on the HTP but a different task.

## Files

| file | what |
|---|---|
| `model.py` | deployment split on top of the upstream module; demo preprocessing without pytorch3d |
| `validate.py` | split vs upstream forward; fp32 full-grid reference `ref_<demo>_<g>.npz` |
| `mcc.py` | export (onnxsim, ORT check), phone runs (strict all-HTP via `phone.sh`), scoring |
| `numcc_ref.py` | NU-MCC upstream inference on the host, same input, cost/accuracy comparison |
| `depth.py` | MoGe-2 static-shape ONNX (+ erf-GELU -> Gelu), MCC fed MoGe vs iPhone points |
| `queries.py` | query-reduction strategies scored exactly against the dense references |
| `dec_opt.py` | decoder/encoder variants (quantization policies, chunk sizes) scored end to end on the phone vs dense host references |
| `app_check.py` | the demo app's MCC mode (its `dump=1` tensors) vs `model.prep` + the host fp32 model |
| `phone.sh` | one ONNX piece on the phone (ORT + QNN EP), under the shared phone lock; md5-skips unchanged inputs |

## Follow-ups

- **Demo-app mode: built** -- `../../maskrcnn_demo_app` "MCC 3D" (photo -> tap (SAM mask) -> MoGe-2 -> MCC
  -> rotatable colored point cloud, 2.5 s for the quest2 headset after "Decoder optimization", 1.3 s with
  the hand-written DSP decoder `../../mcc_hmx`; see its README). `app_check.py`
  checks the app's C++ prep and its reconstruction against this directory's pipeline. Still open: a
  gravity-aligned frame from the phone's accelerometer.
- uint8 decoder: 29 ms a chunk (vs 47 w8a16) but 5-15% of the points wrong (see "Decoder optimization");
  would need QAT or a per-block sensitivity search. Encoder quantization needs a working calibration.

- The decoder on HMX + HVX by hand: `../../mcc_hmx` (22 ms a 1024-query chunk vs QNN's 47 w8a16, at
  fp16-level accuracy).
