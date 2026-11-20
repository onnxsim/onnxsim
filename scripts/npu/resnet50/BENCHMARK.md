# ResNet50 on the Intel NPU: baseline measurements

Baseline for the lowering executor in this directory. All numbers come from one test machine (Intel NPU 5020,
5 CPU cores, Ubuntu 26.04) and are not a controlled benchmark: the NPU was shared with other users' jobs, which caused
transient device faults and some run-to-run variance. Figures are medians or best-of-N, as noted.

## Latency and throughput

ResNet50 is about 4.1 GMAC per image (224x224, batch 1).

| Path | Per image | Effective rate | Source of the number |
|---|---|---|---|
| OpenVINO precompiled blob on the NPU (`resnet50_npu.blob`) | 3.7 to 4.2 ms | about 1.0 TMAC/s | median of 20 warm inferences, two sessions |
| onnxruntime on the CPU (float model) | 73 to 88 ms | about 47 GMAC/s | median of 5 runs, two sessions |
| This executor, fp16 vector convs (`conv_vec_run`, in this PR's dependencies) | about 32 s | about 0.13 GMAC/s | one warm walk |
| This executor, int8 convs on SAU `dp4` (`conv_int8_v2`, not in the PR yet) | 7.85 s | about 0.5 GMAC/s | median of 3 warm walks |

Single conv layer (64 to 64 channels, 3x3, 56x56, one launch, 115.6M MAC):

| Kernel | Time | Rate |
|---|---|---|
| scalar fp16 (`conv_npu`) | 0.79 s | 0.15 GMAC/s |
| fp16 vector, 8 output channels per step (`conv_vec`) | 0.18 s | 0.64 GMAC/s |
| int8 `dp4`, v1 (one channel) | 0.169 s | 0.68 GMAC/s |
| int8 `dp4`, v2 (four channels, shared input word) | 0.156 s | 0.74 GMAC/s |
| int8 `dp4`, v3 (two channels on `sacc0`/`sacc1`) | 0.141 s (best of 5; average 0.225 s) | 0.82 GMAC/s |
| int8 `dp4`, v4 (two pixels x four channels) | no better than v2 | |

Where the int8 time goes (fork microbenchmark, v2, same layer): `dp4` calls account for about 55% of the time; loads
and loop overhead for the rest. Launch size alone does not change the time. Each extra launch costs about 6 ms.

## Accuracy (held-out, int8 path)

- Calibration: onnxsim `quantize_full_qdq` (per-channel int8 weights, uint8 activations), 128 augmented views of 300
  Imagenette training images. Validation images were not used for calibration.
- Evaluation: 100 Imagenette validation images (10 per class), ImageNet labels, standard resize and center crop.

| Model | Top-1 |
|---|---|
| float (onnxruntime) | 86.0% |
| onnxsim QDQ (onnxruntime) | 87.0% |
| this executor, all 53 convs int8 on the NPU | 87.0% |

The NPU and float models agree on the top-1 class for 99 of 100 images. With 100 images the 95% interval is about
+/-7 points, so the 86 vs 87 gap is not significant; the result shows that int8 costs no measurable top-1 accuracy here.

An earlier calibration on random Gaussian inputs gave much worse agreement with the float model (34/56) than
calibration on real photos (50/56) on the same evaluation views.

## Accuracy of the fp16 path

The fp16 vector-conv walk matches onnxruntime to a maximum absolute logit difference of 0.018 (reference maximum 7.5)
and gives the same top-1 class on a random input.

## Interpretation

- The gap to the OpenVINO blob is about 2000x. Our kernels run one scalar SHAVE core with `dp4` (4 MACs per
  instruction) plus loads and loop overhead; the vector and multi-core paths are not reachable with the current
  toolchain (the `vector` dialect is not loaded in `vpux-opt`, and the 5000xx target does not expose the vector `dp4`
  builtins).
- Launch count and host packing are a secondary cost: about 300 to 340 launches per image.

## Reproduce

Scripts are on the test machine: `perf_ov.py` (OpenVINO blob and onnxruntime timings), `perf_ours.py` (warm walk),
`eval_inet.py` (held-out evaluation with per-image progress). Set `KERNEL=v2` for the int8 path.
