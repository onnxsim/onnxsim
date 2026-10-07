# Which layers should stay in float? (`onnxsim.quant_sensitivity`)

Mixed-precision quantization comes down to one question: *which layers does quantization hurt the
most?* The brute-force answer (quantize one layer at a time and measure) costs one evaluation pass
per layer. `onnxsim.quant_sensitivity` puts several cheaper **estimators** behind one API next to
that brute-force measurement, so they can be compared honestly, and this page reports how they did
on real pretrained models and real data. The short version is below; the evidence follows it.

## Summary

Measured on 7 runs with real pretrained models and real data (ResNet18 and MobileNetV2 on ImageNet validation
images, DistilBERT on SST-2 validation; weights and 4-bit activations), ground truth = quantize one layer at a time and
measure the KL to the float model on held-out samples:

* **`taylor` and `fisher` track real sensitivity.** Both are gradient × quantization-perturbation scores from one backward
  pass per calibration sample. Spearman against measured KL: +0.89 … +0.98 (`taylor`), +0.86 … +0.98 (`fisher`), against
  ground-truth ceilings of 0.87 … 1.00. `fisher` finds the single most sensitive layer in 6 of 7 runs, `taylor` in 5 of 7.
  Used to pick which layers to keep in float they beat random choices in 26 of 28 (run, k) cells and come within 5 % of
  the single-site oracle in 16 (`fisher`) and 18 (`taylor`) of 28.
* **But they are not better than just measuring.** Quantizing one layer at a time and measuring KL on the *same* 64
  calibration images ranks the CNN layers at Spearman +0.95 … +0.99 in 6–10 s per pass over all layers, and its
  budgeted selections are equal to or better than `fisher`'s in all 5 CNN runs. The gradient estimators win on
  DistilBERT, where each forward pass is batch-1: for weights +0.96 vs +0.86 at comparable wall-clock, and better
  budgeted selections in both of its runs (for its activations the rankings tie at +0.89 for `taylor`). What they really buy is that one
  backward pass covers *every* layer: about 3× cheaper than the like-for-like brute force here (21 and 53 layers), with the
  gap growing with the layer count.
* **Data-free proxies are unreliable.** `‖W−Q(W)‖`, its relative form and the activation quantization step reach Spearman
  −0.48 … +0.57, and pick a *worse-than-random* set in 13 of 44 (run, k) cells. Do not choose layers by weight norm.
* **HAWQ-V2 (`hessian_trace`) was worse than `fisher` in all 4 weight runs** (Spearman +0.21 … +0.94 vs +0.86 … +0.98)
  and was the slowest estimator (41–241 s).
* **The certified bound is for safety, not for choosing layers.** It cannot run on real-size models (dense zonotopes), and on
  10 small trained digits nets it ranks layers poorly: mean Spearman +0.32 (3-bit weights), +0.17 (8-bit activations),
  and **−0.78** when the activation quantizers come from interval ranges.
* **Where the gradient estimators fail:** when joint quantization collapses the model (ResNet18 at int3: 0.3 % accuracy with
  everything quantized), a correct one-at-a-time ranking does not give a good budgeted selection; and for activations at
  256 calibration samples one unlucky subset dropped `fisher` to +0.54 (stability section).

## The module

A *site* is a `MatMul`/`Gemm`/`Conv` with a constant weight. Quantizing a site means its **weights**
(per-output-channel symmetric, min-max) or its **input activation** (per-tensor affine, calibrated).
Sites can be grouped (all linear layers of a transformer block); every estimator scores *groups*, and a
singleton group is a site.

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import quant_sensitivity as qs

rng = np.random.default_rng(0)
dims, scales = [12, 12, 12, 5], (0.5, 1.0, 6.0)  # the last layer has 12x the weight scale of the first
init, lines, prev = {}, [], "x"
for i, s in enumerate(scales):
    init[f"W{i}"] = (rng.standard_normal((dims[i], dims[i + 1])) * s / np.sqrt(dims[i])).astype(np.float32)
    init[f"B{i}"] = (rng.standard_normal(dims[i + 1]) * 0.1).astype(np.float32)
    out = "logits" if i == 2 else f"r{i}"
    lines += [f"h{i} = MatMul({prev}, W{i})", f"a{i} = Add(h{i}, B{i})"]
    lines.append("logits = Identity(a2)" if i == 2 else f"r{i} = Relu(a{i})")
    prev = out
model = parser.parse_model(
    '<ir_version: 9, opset_import: ["" : 17]> '
    "m (float[N,12] x) => (float[N,5] logits) { " + " ".join(lines) + " }"
)
model.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in init.items())

calib = {"x": rng.standard_normal((128, 12)).astype(np.float32)}
report = qs.rank(model, calib=calib, kind="weights", bits=2, methods=("weight_err", "fisher"))
print("most sensitive first (fisher):    ", report.ranking("fisher"))
print("most sensitive first (||W-Q(W)||):", report.ranking("weight_err"))
print("keep in float (k=1, fisher):      ", qs.select_float_sites(report, 1, "fisher"))

groups = qs.as_groups(qs.find_sites(model))
truth = qs.measure(model, groups, "weights", 2, {"x": rng.standard_normal((2000, 12)).astype(np.float32)})
print("measured KL per layer:", {g: round(v["kl"], 4) for g, v in truth.items()})
```

Output (the `‖W−Q(W)‖` ranking is deterministic and is checked exactly by the test that keeps this page honest;
the `fisher` ranking depends on seeded random labels and the measured numbers on the platform's float rounding, so
the test only checks that those lines have the right form):

```
most sensitive first (fisher):     ['MatMul_0', 'MatMul_3', 'MatMul_6']
most sensitive first (||W-Q(W)||): ['MatMul_6', 'MatMul_3', 'MatMul_0']
keep in float (k=1, fisher):       ['MatMul_0']
measured KL per layer: {'MatMul_0': 0.0918, 'MatMul_3': 0.1064, 'MatMul_6': 0.0626}
```

This 3-layer toy is *not* evidence about the estimators: the measured values are nearly tied and
`fisher` ranks the layers differently from the measurement. The tables below are the evidence.

| estimator (`methods=`) | what it computes | needs | cost |
|---|---|---|---|
| `weight_err`, `weight_err_rel` | ‖W − Q(W)‖<sub>F</sub> and the same divided by ‖W‖<sub>F</sub> | nothing | free |
| `act_scale` | the calibrated quantization step of the site's input | calibration data | free |
| `fisher` | `mean_i (g_i · δ)²`, with `g_i` the gradient of log p(y<sub>i</sub>\|x<sub>i</sub>) w.r.t. the site's weight (or input activation), `δ` the quantization perturbation (Q(W) − W, or the activation's own rounding error on sample *i*), and y<sub>i</sub> drawn from the model's own distribution. Twice the diagonal-Fisher estimate of the KL a perturbation δ causes | calibration samples | one backward pass per sample covers *every* site |
| `taylor` | `mean_i abs(g_i · δ)`: the first-order change of log p itself (same gradients) | calibration samples | same as `fisher` |
| `hessian_trace` | HAWQ-V2: `tr(H)/n · ‖δ‖²`, Hutchinson trace of the cross-entropy Hessian (8 Rademacher probes, double backward) | calibration samples | much slower (below) |
| `certified` | a *sound* bound on max abs(float − quantized) over an input box, from `onnxsim.zonotope` (weights) or `onnxsim.quant_verify` (8-bit activations) | an input box; a small graph | dense zonotopes: small graphs only |

The gradient estimators run the ONNX graph through a small differentiable executor
(`quant_sensitivity.TorchGraph`: the ops CNNs and transformer encoders use; an unsupported op raises
an error naming it). `onnxsim.to_torch` is not used: it targets LLM graphs and has no `Conv` rule.
`quant_sensitivity.measure` is the brute-force oracle, `select_float_sites(report, k, method)` the
decision helper, and `spearman`/`kendall` the rank correlations used below.

## Evaluation protocol

Everything here uses **real pretrained weights and real labelled data**; nothing is synthetic except the
unit tests. The models were downloaded, not trained, with one stated exception (the *digits* nets).

| task | model (weights) | data | sites → groups |
|---|---|---|---|
| image classification | torchvision ResNet18 and MobileNetV2, `IMAGENET1K_V1`, exported to ONNX (BatchNorm folded into `Conv`) | ImageNet-1k **validation** images: every 10th image of the set, class stratified (5 per class, 5000 images, short side 256, center crop 224), from the ungated mirror `Tsomaros/Imagenet-1k_validation` | 21 and 53 sites, one per group |
| sentence classification | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` exported with batch 1 and 64 tokens | the real SST-2 **validation** set, 872 labelled sentences (none truncated) | 38 linear layers → 7 groups (6 transformer blocks + the head) |
| small trained nets (certified method only) | 10 deep MLPs (8 layers × 32) **trained here** on real data for 400 Adam steps | scikit-learn `load_digits`, 1797 real 8×8 handwritten digits | 8 sites per net |

The float models' accuracy on the evaluation splits checks the whole pipeline: ResNet18 68.6 %
(published 69.8 %), MobileNetV2 71.7 % (published 71.9 %), DistilBERT 91.4 % on the 616 evaluation
sentences (91.1 % on all 872). The ImageNet subset was regenerated from scratch with
`scripts/quant_sensitivity_prep.py` and is byte-identical to the one used for the runs; the regenerated SST-2 tokens are
identical and both regenerated ONNX models give identical logits.

* **Splits.** The estimators and the activation quantizers see only a **calibration** split; ground truth is
  measured on a **disjoint evaluation** split. ImageNet: per class the first 2 images are calibration (2000)
  and the other 3 evaluation (3000). SST-2: 256 random sentences calibrate, the other 616 evaluate.
  The gradient estimators use the first 256 calibration images (128 sentences for DistilBERT) and the activation
  quantizers the first 512. The calibration split is stored class by class, so those are the first 128 and 256 classes
  rather than a random draw, which is a limitation of these runs; the stability experiment draws random subsets from the
  whole pool of 2000 and lands on the same numbers (ResNet18 int4 weights, 256 samples: +0.97 ± 0.02 vs +0.970 here).
* **Quantizers.** Weights: per-output-channel symmetric min-max, 4 bits (and 3 bits for ResNet18).
  Activations: per-tensor affine, 4 bits, range = the 99.99th percentile of 512 calibration activations
  (4-bit activations are emulated with `Div`/`Round`/`Clip`/`Mul`; 8-bit ones are real
  `QuantizeLinear`/`DequantizeLinear`). The bit widths were chosen from a 500-image pilot so that
  one-layer effects are visible without the whole model collapsing; 3-bit weights *do* collapse it (below).
  Fake quantization only: no integer kernels, no QAT, no calibration search.
* **Ground truth.** Quantize **one group at a time**; measure the mean KL(float ‖ quantized) of the output
  distribution on the evaluation split. KL is the target the estimators model; accuracy drop is also reported but
  is a noisy, discrete signal (the binomial standard error of an accuracy is about 0.85 points on 3000 images and 1.1
  points on 616 sentences, comparable to the per-layer effects). A split-half check (ground truth on two
  random halves of the evaluation images) gives the *ceiling* any estimator can reach.
* **Intervals.** Bootstrap over evaluation samples (300 resamples, 95 % percentile). **That interval covers noise in
  the ground truth only**, not the choice of calibration samples; the stability section measures that separately.
* **Decision test.** For k = 1, 2, 4, 8 (ResNet18), 2, 4, 8, 16 (MobileNetV2) or 1, 2, 3, 4 (DistilBERT), keep the k top-ranked groups in float, quantize *all the
  rest at once*, and measure. Compared against random choices (mean of 5 draws) and a *single-site oracle*:
  the top k by measured one-at-a-time KL. The oracle is **not** the optimal set, because layers interact; where an
  estimator beats it, that is why.
* **Environment.** Python 3.12.13, onnxruntime 1.30.0, onnx 1.23.1, numpy 2.5.3, torch 2.14.1 (CPU), torchvision 0.29.1,
  transformers 5.19.0, datasets 5.1.0; AMD Ryzen AI MAX+ 395, 32 threads. Seeds: 0 everywhere
  (random-label draws, random selections, resamples).

## Results

### Rank agreement with measured sensitivity

Spearman correlation between each estimator's scores and the measured one-at-a-time KL, one column per
run (+1 = same ranking, 0 = unrelated, −1 = reversed). The last two rows give the ceiling and the number of
groups: with 7 groups (DistilBERT) a single swapped adjacent pair moves ρ by 0.04.

| estimator | ResNet18 · int4 weights | ResNet18 · int3 weights | ResNet18 · 4-bit acts | MobileNetV2 · int4 weights | MobileNetV2 · 4-bit acts | DistilBERT/SST-2 · int4 weights | DistilBERT/SST-2 · 4-bit acts |
|---|---|---|---|---|---|---|---|
| `weight_err` ‖W−Q(W)‖ | -0.48 | -0.26 | – | +0.17 | – | +0.46 | – |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | +0.05 | +0.15 | – | +0.33 | – | +0.57 | – |
| `act_scale` (quantization step) | – | – | +0.18 | – | +0.56 | – | +0.07 |
| `fisher` | +0.97 | +0.86 | +0.87 | +0.98 | +0.97 | +0.96 | +0.86 |
| `taylor` | +0.97 | +0.89 | +0.92 | +0.98 | +0.98 | +0.96 | +0.89 |
| `hessian_trace` (HAWQ-V2) | +0.94 | +0.64 | – | +0.85 | – | +0.21 | – |
| *ceiling: ground truth vs itself (split-half)* | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.87 | 0.98 |
| *number of groups ranked* | 21 | 21 | 21 | 53 | 53 | 7 | 7 |

Per-run detail, with bootstrap intervals:

#### ResNet18 · int4 weights
n_calibration=2000, n_evaluation=3000, 21 groups, float accuracy 0.6860, ground-truth split-half Spearman 0.998

| estimator | Spearman vs KL (95% CI) | Kendall vs KL | Spearman vs accuracy drop | picks the true #1 | true top-3 found |
|---|---|---|---|---|---|
| `weight_err` ‖W−Q(W)‖ | -0.479 [-0.49, -0.47] | -0.400 | -0.364 | no | 0/3 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | +0.055 [+0.02, +0.06] | +0.048 | -0.163 | no | 1/3 |
| `fisher` | +0.970 [+0.96, +0.97] | +0.886 | +0.669 | yes | 3/3 |
| `taylor` | +0.974 [+0.97, +0.98] | +0.886 | +0.635 | no | 3/3 |
| `hessian_trace` (HAWQ-V2) | +0.939 [+0.93, +0.94] | +0.819 | +0.687 | yes | 3/3 |

#### ResNet18 · int3 weights
n_calibration=2000, n_evaluation=3000, 21 groups, float accuracy 0.6860, ground-truth split-half Spearman 0.996

| estimator | Spearman vs KL (95% CI) | Kendall vs KL | Spearman vs accuracy drop | picks the true #1 | true top-3 found |
|---|---|---|---|---|---|
| `weight_err` ‖W−Q(W)‖ | -0.261 [-0.30, -0.24] | -0.181 | -0.225 | no | 0/3 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | +0.145 [+0.12, +0.17] | +0.095 | +0.170 | no | 0/3 |
| `fisher` | +0.862 [+0.86, +0.89] | +0.733 | +0.842 | yes | 3/3 |
| `taylor` | +0.886 [+0.88, +0.91] | +0.743 | +0.869 | yes | 3/3 |
| `hessian_trace` (HAWQ-V2) | +0.636 [+0.63, +0.66] | +0.467 | +0.618 | no | 2/3 |

#### ResNet18 · 4-bit activations
n_calibration=2000, n_evaluation=3000, 21 groups, float accuracy 0.6860, ground-truth split-half Spearman 0.989

| estimator | Spearman vs KL (95% CI) | Kendall vs KL | Spearman vs accuracy drop | picks the true #1 | true top-3 found |
|---|---|---|---|---|---|
| `act_scale` (quantization step) | +0.180 [+0.15, +0.21] | +0.139 | +0.197 | no | 2/3 |
| `fisher` | +0.870 [+0.84, +0.89] | +0.714 | +0.269 | yes | 3/3 |
| `taylor` | +0.923 [+0.90, +0.93] | +0.800 | +0.344 | yes | 3/3 |

#### MobileNetV2 · int4 weights
n_calibration=2000, n_evaluation=3000, 53 groups, float accuracy 0.7170, ground-truth split-half Spearman 0.998

| estimator | Spearman vs KL (95% CI) | Kendall vs KL | Spearman vs accuracy drop | picks the true #1 | true top-3 found |
|---|---|---|---|---|---|
| `weight_err` ‖W−Q(W)‖ | +0.169 [+0.16, +0.18] | +0.120 | +0.297 | no | 0/3 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | +0.332 [+0.32, +0.34] | +0.234 | +0.179 | no | 0/3 |
| `fisher` | +0.981 [+0.98, +0.98] | +0.900 | +0.783 | yes | 2/3 |
| `taylor` | +0.984 [+0.98, +0.99] | +0.917 | +0.781 | yes | 2/3 |
| `hessian_trace` (HAWQ-V2) | +0.853 [+0.85, +0.86] | +0.698 | +0.715 | yes | 2/3 |

#### MobileNetV2 · 4-bit activations
n_calibration=2000, n_evaluation=3000, 53 groups, float accuracy 0.7170, ground-truth split-half Spearman 0.999

| estimator | Spearman vs KL (95% CI) | Kendall vs KL | Spearman vs accuracy drop | picks the true #1 | true top-3 found |
|---|---|---|---|---|---|
| `act_scale` (quantization step) | +0.564 [+0.56, +0.57] | +0.414 | +0.374 | yes | 1/3 |
| `fisher` | +0.974 [+0.97, +0.98] | +0.869 | +0.770 | yes | 2/3 |
| `taylor` | +0.978 [+0.98, +0.98] | +0.880 | +0.761 | yes | 2/3 |

#### DistilBERT / SST-2 · int4 weights (7 block-level groups)
n_calibration=256, n_evaluation=616, 7 groups, float accuracy 0.9140, ground-truth split-half Spearman 0.866

| estimator | Spearman vs KL (95% CI) | Kendall vs KL | Spearman vs accuracy drop | picks the true #1 | true top-3 found |
|---|---|---|---|---|---|
| `weight_err` ‖W−Q(W)‖ | +0.464 [+0.32, +0.54] | +0.333 | +0.162 | no | 1/3 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | +0.571 [+0.39, +0.75] | +0.429 | +0.468 | no | 2/3 |
| `fisher` | +0.964 [+0.75, +1.00] | +0.905 | +0.847 | yes | 2/3 |
| `taylor` | +0.964 [+0.75, +1.00] | +0.905 | +0.847 | yes | 2/3 |
| `hessian_trace` (HAWQ-V2) | +0.214 [+0.00, +0.36] | +0.143 | +0.306 | yes | 2/3 |

#### DistilBERT / SST-2 · 4-bit activations (7 block-level groups)
n_calibration=256, n_evaluation=616, 7 groups, float accuracy 0.9140, ground-truth split-half Spearman 0.977

| estimator | Spearman vs KL (95% CI) | Kendall vs KL | Spearman vs accuracy drop | picks the true #1 | true top-3 found |
|---|---|---|---|---|---|
| `act_scale` (quantization step) | +0.071 [+0.00, +0.21] | +0.048 | +0.071 | no | 1/3 |
| `fisher` | +0.857 [+0.86, +0.89] | +0.714 | +0.857 | no | 3/3 |
| `taylor` | +0.893 [+0.86, +0.89] | +0.810 | +0.893 | no | 3/3 |

### The decision test: which layers to keep in float

How close each estimator's budgeted selection comes to random and to the single-site oracle, as the mean over k of
`KL(selection) / KL(random)` and `KL(selection) / KL(oracle)` (**lower is better**; 1.00× random means no better than
chance; 1.00× oracle means as good as measuring every layer):

| estimator | ResNet18 · int4 weights | ResNet18 · int3 weights | ResNet18 · 4-bit acts | MobileNetV2 · int4 weights | MobileNetV2 · 4-bit acts | DistilBERT/SST-2 · int4 weights | DistilBERT/SST-2 · 4-bit acts |
|---|---|---|---|---|---|---|---|
| `weight_err` ‖W−Q(W)‖ | 1.32× random · 3.49× oracle | 0.90× random · 1.06× oracle | – | 0.94× random · 3.74× oracle | – | 0.81× random · 2.05× oracle | – |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | 1.01× random · 2.58× oracle | 0.95× random · 1.12× oracle | – | 0.86× random · 3.31× oracle | – | 0.95× random · 2.88× oracle | – |
| `act_scale` (quantization step) | – | – | 0.78× random · 3.42× oracle | – | 0.77× random · 3.63× oracle | – | 1.18× random · 3.50× oracle |
| `fisher` | 0.45× random · 1.01× oracle | 1.03× random · 1.21× oracle | 0.29× random · 1.03× oracle | 0.34× random · 1.06× oracle | 0.49× random · 1.19× oracle | 0.45× random · 1.01× oracle | 0.48× random · 1.20× oracle |
| `taylor` | 0.50× random · 1.08× oracle | 1.03× random · 1.21× oracle | 0.29× random · 1.01× oracle | 0.32× random · 1.00× oracle | 0.49× random · 1.19× oracle | 0.45× random · 1.01× oracle | 0.47× random · 1.16× oracle |
| `hessian_trace` (HAWQ-V2) | 0.46× random · 1.04× oracle | 1.12× random · 1.32× oracle | – | 0.34× random · 1.07× oracle | – | 0.81× random · 2.80× oracle | – |
| *brute force on 64 calibration samples* | 0.45× random · 1.01× oracle | 0.92× random · 1.06× oracle | 0.29× random · 1.00× oracle | 0.32× random · 0.97× oracle | 0.42× random · 0.97× oracle | 0.54× random · 1.17× oracle | 0.54× random · 1.43× oracle |

Full tables (KL of the *whole model* with everything outside the selection quantized):

#### ResNet18 · int4 weights
| keep k groups in float | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| *none (everything quantized)* | KL 1.566 · acc 0.429 | KL 1.566 · acc 0.429 | KL 1.566 · acc 0.429 | KL 1.566 · acc 0.429 |
| `weight_err` ‖W−Q(W)‖ | KL 1.522 · acc 0.435 | KL 1.478 · acc 0.438 | KL 1.374 · acc 0.453 | KL 1.239 · acc 0.470 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | KL 1.473 · acc 0.443 | KL 1.119 · acc 0.498 | KL 0.987 · acc 0.518 | KL 0.846 · acc 0.530 |
| `fisher` | KL 0.861 · acc 0.542 | KL 0.634 · acc 0.573 | KL 0.394 · acc 0.616 | KL 0.201 · acc 0.659 |
| `taylor` | KL 1.211 · acc 0.484 | KL 0.634 · acc 0.573 | KL 0.398 · acc 0.618 | KL 0.179 · acc 0.664 |
| `hessian_trace` (HAWQ-V2) | KL 0.861 · acc 0.542 | KL 0.634 · acc 0.573 | KL 0.394 · acc 0.616 | KL 0.223 · acc 0.651 |
| single-site oracle (top-k by measured one-at-a-time KL) | KL 0.861 · acc 0.542 | KL 0.634 · acc 0.573 | KL 0.394 · acc 0.616 | KL 0.195 · acc 0.658 |
| random (mean of 5 draws) | KL 1.499 · acc 0.437 | KL 1.230 · acc 0.480 | KL 0.817 · acc 0.547 | KL 0.901 · acc 0.532 |

Float model: accuracy 0.686, KL 0.

#### ResNet18 · int3 weights
| keep k groups in float | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| *none (everything quantized)* | KL 15.659 · acc 0.003 | KL 15.659 · acc 0.003 | KL 15.659 · acc 0.003 | KL 15.659 · acc 0.003 |
| `weight_err` ‖W−Q(W)‖ | KL 14.494 · acc 0.002 | KL 11.640 · acc 0.005 | KL 10.582 · acc 0.006 | KL 8.100 · acc 0.016 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | KL 15.200 · acc 0.004 | KL 12.101 · acc 0.008 | KL 12.279 · acc 0.005 | KL 8.186 · acc 0.046 |
| `fisher` | KL 14.922 · acc 0.003 | KL 15.291 · acc 0.009 | KL 13.487 · acc 0.014 | KL 8.249 · acc 0.072 |
| `taylor` | KL 14.922 · acc 0.003 | KL 15.291 · acc 0.009 | KL 13.487 · acc 0.014 | KL 8.249 · acc 0.072 |
| `hessian_trace` (HAWQ-V2) | KL 15.671 · acc 0.005 | KL 15.291 · acc 0.009 | KL 13.036 · acc 0.017 | KL 10.775 · acc 0.034 |
| single-site oracle (top-k by measured one-at-a-time KL) | KL 14.922 · acc 0.003 | KL 14.245 · acc 0.008 | KL 10.765 · acc 0.023 | KL 5.494 · acc 0.093 |
| random (mean of 5 draws) | KL 15.667 · acc 0.004 | KL 14.066 · acc 0.007 | KL 13.614 · acc 0.008 | KL 7.593 · acc 0.048 |

Float model: accuracy 0.686, KL 0.

#### ResNet18 · 4-bit activations
| keep k groups in float | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| *none (everything quantized)* | KL 0.324 · acc 0.634 | KL 0.324 · acc 0.634 | KL 0.324 · acc 0.634 | KL 0.324 · acc 0.634 |
| `act_scale` (quantization step) | KL 0.312 · acc 0.632 | KL 0.163 · acc 0.659 | KL 0.155 · acc 0.660 | KL 0.133 · acc 0.664 |
| `fisher` | KL 0.175 · acc 0.660 | KL 0.063 · acc 0.676 | KL 0.042 · acc 0.676 | KL 0.026 · acc 0.684 |
| `taylor` | KL 0.175 · acc 0.660 | KL 0.063 · acc 0.676 | KL 0.042 · acc 0.676 | KL 0.024 · acc 0.688 |
| single-site oracle (top-k by measured one-at-a-time KL) | KL 0.175 · acc 0.660 | KL 0.063 · acc 0.676 | KL 0.042 · acc 0.676 | KL 0.024 · acc 0.690 |
| random (mean of 5 draws) | KL 0.320 · acc 0.632 | KL 0.260 · acc 0.639 | KL 0.168 · acc 0.659 | KL 0.226 · acc 0.647 |

Float model: accuracy 0.686, KL 0.

#### MobileNetV2 · int4 weights
| keep k groups in float | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| *none (everything quantized)* | KL 5.095 · acc 0.119 | KL 5.095 · acc 0.119 | KL 5.095 · acc 0.119 | KL 5.095 · acc 0.119 |
| `weight_err` ‖W−Q(W)‖ | KL 4.622 · acc 0.149 | KL 4.448 · acc 0.162 | KL 4.317 · acc 0.162 | KL 2.891 · acc 0.286 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | KL 4.601 · acc 0.149 | KL 4.817 · acc 0.126 | KL 2.949 · acc 0.257 | KL 2.640 · acc 0.291 |
| `fisher` | KL 2.643 · acc 0.329 | KL 2.068 · acc 0.393 | KL 1.083 · acc 0.525 | KL 0.464 · acc 0.626 |
| `taylor` | KL 2.643 · acc 0.329 | KL 1.807 · acc 0.426 | KL 0.953 · acc 0.542 | KL 0.464 · acc 0.626 |
| `hessian_trace` (HAWQ-V2) | KL 2.643 · acc 0.329 | KL 1.836 · acc 0.422 | KL 1.197 · acc 0.509 | KL 0.477 · acc 0.617 |
| single-site oracle (top-k by measured one-at-a-time KL) | KL 2.643 · acc 0.329 | KL 1.807 · acc 0.426 | KL 1.012 · acc 0.541 | KL 0.447 · acc 0.629 |
| random (mean of 5 draws) | KL 4.825 · acc 0.133 | KL 4.840 · acc 0.130 | KL 3.946 · acc 0.195 | KL 3.625 · acc 0.230 |

Float model: accuracy 0.717, KL 0.

#### MobileNetV2 · 4-bit activations
| keep k groups in float | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| *none (everything quantized)* | KL 5.901 · acc 0.015 | KL 5.901 · acc 0.015 | KL 5.901 · acc 0.015 | KL 5.901 · acc 0.015 |
| `act_scale` (quantization step) | KL 5.442 · acc 0.135 | KL 4.892 · acc 0.178 | KL 4.248 · acc 0.207 | KL 2.886 · acc 0.318 |
| `fisher` | KL 5.547 · acc 0.122 | KL 3.475 · acc 0.251 | KL 1.926 · acc 0.418 | KL 0.436 · acc 0.631 |
| `taylor` | KL 5.547 · acc 0.122 | KL 3.475 · acc 0.251 | KL 1.926 · acc 0.418 | KL 0.436 · acc 0.631 |
| single-site oracle (top-k by measured one-at-a-time KL) | KL 5.080 · acc 0.138 | KL 3.267 · acc 0.254 | KL 1.632 · acc 0.461 | KL 0.308 · acc 0.653 |
| random (mean of 5 draws) | KL 5.847 · acc 0.016 | KL 5.734 · acc 0.019 | KL 5.858 · acc 0.034 | KL 5.227 · acc 0.069 |

Float model: accuracy 0.717, KL 0.

#### DistilBERT / SST-2 · int4 weights
| keep k groups in float | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| *none (everything quantized)* | KL 0.048 · acc 0.903 | KL 0.048 · acc 0.903 | KL 0.048 · acc 0.903 | KL 0.048 · acc 0.903 |
| `weight_err` ‖W−Q(W)‖ | KL 0.048 · acc 0.906 | KL 0.029 · acc 0.906 | KL 0.022 · acc 0.904 | KL 0.009 · acc 0.904 |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | KL 0.047 · acc 0.903 | KL 0.027 · acc 0.903 | KL 0.028 · acc 0.906 | KL 0.019 · acc 0.903 |
| `fisher` | KL 0.030 · acc 0.903 | KL 0.016 · acc 0.904 | KL 0.012 · acc 0.906 | KL 0.003 · acc 0.909 |
| `taylor` | KL 0.030 · acc 0.903 | KL 0.016 · acc 0.904 | KL 0.012 · acc 0.906 | KL 0.003 · acc 0.909 |
| `hessian_trace` (HAWQ-V2) | KL 0.030 · acc 0.903 | KL 0.023 · acc 0.907 | KL 0.023 · acc 0.907 | KL 0.023 · acc 0.909 |
| single-site oracle (top-k by measured one-at-a-time KL) | KL 0.030 · acc 0.903 | KL 0.016 · acc 0.904 | KL 0.012 · acc 0.904 | KL 0.003 · acc 0.909 |
| random (mean of 5 draws) | KL 0.038 · acc 0.906 | KL 0.036 · acc 0.903 | KL 0.030 · acc 0.906 | KL 0.022 · acc 0.905 |

Float model: accuracy 0.914, KL 0.

#### DistilBERT / SST-2 · 4-bit activations
| keep k groups in float | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| *none (everything quantized)* | KL 0.429 · acc 0.748 | KL 0.429 · acc 0.748 | KL 0.429 · acc 0.748 | KL 0.429 · acc 0.748 |
| `act_scale` (quantization step) | KL 0.543 · acc 0.761 | KL 0.543 · acc 0.799 | KL 0.308 · acc 0.869 | KL 0.195 · acc 0.888 |
| `fisher` | KL 0.371 · acc 0.795 | KL 0.202 · acc 0.864 | KL 0.086 · acc 0.890 | KL 0.043 · acc 0.901 |
| `taylor` | KL 0.371 · acc 0.795 | KL 0.179 · acc 0.881 | KL 0.086 · acc 0.890 | KL 0.043 · acc 0.901 |
| single-site oracle (top-k by measured one-at-a-time KL) | KL 0.266 · acc 0.847 | KL 0.142 · acc 0.886 | KL 0.086 · acc 0.890 | KL 0.043 · acc 0.901 |
| random (mean of 5 draws) | KL 0.419 · acc 0.792 | KL 0.417 · acc 0.791 | KL 0.215 · acc 0.862 | KL 0.284 · acc 0.854 |

Float model: accuracy 0.914, KL 0.

### The like-for-like baseline: brute force on a few calibration samples

The ground truth above uses the *whole* evaluation split (3000 images / 616 sentences), far more data than the
gradient estimators see, so comparing their cost to it would flatter them. The fair competitor is the brute-force
oracle run on **the same few calibration samples**: quantize one group at a time, measure KL on n samples, rank.
Below, n = 16, 64 and 256 random calibration samples (5 draws each, mean ± std of Spearman against the same
full-split ground truth, and the wall-clock of one pass over all groups with ONNX Runtime), next to `fisher`/`taylor`
at their sample budget (256 images, 128 sentences; wall-clock of the torch executor):

| run | groups | brute force, 16 samples | brute force, 64 samples | brute force, 256 samples | `fisher` (est. samples) | `taylor` (est. samples) |
|---|---|---|---|---|---|---|
| ResNet18 · int4 weights | 21 | ρ +0.949 ± 0.021 · 3.3 s | ρ +0.975 ± 0.009 · 5.9 s | ρ +0.989 ± 0.002 · 15.7 s | ρ +0.970 · 5.4 s (256) | ρ +0.974 · 5.4 s (256) |
| ResNet18 · int3 weights | 21 | ρ +0.958 ± 0.011 · 3.2 s | ρ +0.977 ± 0.006 · 6.0 s | ρ +0.989 ± 0.005 · 15.0 s | ρ +0.862 · 5.4 s (256) | ρ +0.886 · 5.4 s (256) |
| ResNet18 · 4-bit acts | 21 | ρ +0.902 ± 0.048 · 3.1 s | ρ +0.954 ± 0.011 · 6.1 s | ρ +0.937 ± 0.087 · 16.3 s | ρ +0.870 · 4.5 s (256) | ρ +0.923 · 4.5 s (256) |
| MobileNetV2 · int4 weights | 53 | ρ +0.966 ± 0.010 · 3.7 s | ρ +0.988 ± 0.003 · 8.8 s | ρ +0.995 ± 0.001 · 22.7 s | ρ +0.981 · 6.6 s (256) | ρ +0.984 · 6.6 s (256) |
| MobileNetV2 · 4-bit acts | 53 | ρ +0.968 ± 0.006 · 3.8 s | ρ +0.992 ± 0.003 · 9.7 s | ρ +0.995 ± 0.005 · 24.1 s | ρ +0.974 · 8.1 s (256) | ρ +0.978 · 8.1 s (256) |
| DistilBERT/SST-2 · int4 weights | 7 | ρ +0.829 ± 0.073 · 4.5 s | ρ +0.857 ± 0.093 · 9.1 s | ρ +0.929 ± 0.000 · 25.9 s | ρ +0.964 · 6.5 s (128) | ρ +0.964 · 6.5 s (128) |
| DistilBERT/SST-2 · 4-bit acts | 7 | ρ +0.807 ± 0.062 · 4.6 s | ρ +0.893 ± 0.060 · 9.2 s | ρ +1.000 ± 0.000 · 28.5 s | ρ +0.857 · 4.7 s (128) | ρ +0.893 · 4.7 s (128) |

The last row of the decision-test scorecard above uses the n = 64 brute-force ranking for the same budgeted selections.

### Cost

| run | brute force (all groups, one at a time) | `fisher`+`taylor` (256 calibration samples) | `hessian_trace` | groups |
|---|---|---|---|---|
| ResNet18 · int4 weights | 144 s | 5.4 s | 83 s | 21 |
| ResNet18 · int3 weights | 125 s | 5.4 s | 41 s | 21 |
| ResNet18 · 4-bit acts | 137 s | 4.5 s | – | 21 |
| MobileNetV2 · int4 weights | 180 s | 6.6 s | 110 s | 53 |
| MobileNetV2 · 4-bit acts | 285 s | 8.1 s | – | 53 |
| DistilBERT/SST-2 · int4 weights | 49 s | 6.5 s | 241 s | 7 |
| DistilBERT/SST-2 · 4-bit acts | 46 s | 4.7 s | – | 7 |

The brute-force column is the full-split ground truth (every group, 3000 images / 616 sentences) and is shown only for scale:
it is *not* a like-for-like comparison with the estimators, which see 256 images (128 sentences). The like-for-like
comparison is the baseline section above.

### The certified bound on small trained nets

The `certified` estimator needs dense zonotopes, so it cannot run on the real-size models above; its skip reasons
are recorded, not hidden:



To test it anyway, 10 independently trained deep MLPs on real digits (8 layers each, 64 input pixels, input box
[0, 1]⁶⁴; held-out accuracy 74–94 %) were scored with every estimator and compared to the measured KL on held-out
digits. Mean Spearman over the 10 nets, ± standard error of the mean:

**3-bit weights**

| estimator | mean Spearman ± SEM | median | picks the true #1 |
|---|---|---|---|
| `weight_err` ‖W−Q(W)‖ | +0.543 ± 0.058 | +0.560 | 40% |
| `weight_err_rel` ‖W−Q(W)‖/‖W‖ | +0.674 ± 0.078 | +0.762 | 30% |
| `fisher` | +0.931 ± 0.014 | +0.929 | 60% |
| `taylor` | +0.945 ± 0.015 | +0.964 | 70% |
| `hessian_trace` (HAWQ-V2) | +0.717 ± 0.052 | +0.690 | 40% |
| `certified` | +0.324 ± 0.069 | +0.321 | 40% |

**8-bit activations, quantizers calibrated on data (99.99th percentile)**

| estimator | mean Spearman ± SEM | median | picks the true #1 |
|---|---|---|---|
| `act_scale` (quantization step) | -0.005 ± 0.083 | -0.060 | 30% |
| `fisher` | +0.852 ± 0.041 | +0.881 | 70% |
| `taylor` | +0.883 ± 0.036 | +0.917 | 80% |
| `certified` | +0.171 ± 0.065 | +0.155 | 0% |

**8-bit activations, quantizers calibrated by interval analysis** (the range is the worst-case range over the box, so
nothing can clip and the certified bound is not inflated by saturation):

| estimator | mean Spearman ± SEM | median | picks the true #1 |
|---|---|---|---|
| `act_scale` (quantization step) | +0.993 ± 0.003 | +1.000 | 100% |
| `fisher` | +0.974 ± 0.010 | +0.988 | 100% |
| `taylor` | +0.986 ± 0.007 | +1.000 | 100% |
| `certified` | -0.783 ± 0.038 | -0.810 | 0% |

### Stability of the gradient estimators

The bootstrap above ignores the estimators' own variability. Here the estimators are recomputed on random subsets of
the calibration pool (5 subsets per size), against the fixed measured ground truth of the matching ResNet18 run, with
labels drawn from the model's distribution (`y~model`, the default) or fixed to its prediction (`y=argmax`):

**ResNet18 · int4 weights**

| calibration samples | fisher, y~model (mean ± std, min) | fisher, y=argmax | taylor, y~model | taylor, y=argmax |
|---|---|---|---|---|
| 8 | +0.684 ± 0.061, +0.61 | +0.768 ± 0.078, +0.67 | +0.719 ± 0.093, +0.56 | +0.789 ± 0.069, +0.71 |
| 16 | +0.828 ± 0.102, +0.64 | +0.841 ± 0.049, +0.79 | +0.848 ± 0.060, +0.74 | +0.855 ± 0.060, +0.78 |
| 32 | +0.895 ± 0.015, +0.88 | +0.889 ± 0.040, +0.82 | +0.909 ± 0.018, +0.88 | +0.919 ± 0.051, +0.82 |
| 64 | +0.935 ± 0.025, +0.90 | +0.919 ± 0.041, +0.84 | +0.951 ± 0.027, +0.91 | +0.933 ± 0.022, +0.90 |
| 128 | +0.950 ± 0.023, +0.91 | +0.964 ± 0.018, +0.94 | +0.956 ± 0.015, +0.94 | +0.970 ± 0.012, +0.95 |
| 256 | +0.972 ± 0.023, +0.93 | +0.978 ± 0.007, +0.96 | +0.983 ± 0.007, +0.97 | +0.978 ± 0.009, +0.96 |

**ResNet18 · 4-bit activations**

| calibration samples | fisher, y~model (mean ± std, min) | fisher, y=argmax | taylor, y~model | taylor, y=argmax |
|---|---|---|---|---|
| 8 | +0.589 ± 0.069, +0.51 | +0.647 ± 0.161, +0.37 | +0.598 ± 0.085, +0.47 | +0.657 ± 0.162, +0.38 |
| 16 | +0.757 ± 0.132, +0.57 | +0.742 ± 0.061, +0.66 | +0.789 ± 0.130, +0.61 | +0.779 ± 0.040, +0.71 |
| 32 | +0.853 ± 0.020, +0.83 | +0.816 ± 0.072, +0.76 | +0.867 ± 0.036, +0.80 | +0.859 ± 0.045, +0.81 |
| 64 | +0.910 ± 0.027, +0.86 | +0.882 ± 0.042, +0.84 | +0.924 ± 0.031, +0.86 | +0.917 ± 0.035, +0.86 |
| 128 | +0.906 ± 0.024, +0.88 | +0.913 ± 0.053, +0.81 | +0.937 ± 0.015, +0.92 | +0.938 ± 0.023, +0.91 |
| 256 | +0.855 ± 0.157, +0.54 | +0.909 ± 0.039, +0.86 | +0.914 ± 0.094, +0.73 | +0.937 ± 0.023, +0.90 |

## Findings

**1. The gradient estimators are good rankers, and `taylor` is at least as good as `fisher`.** Over the seven real runs the
two agree with measured sensitivity at Spearman +0.86 … +0.98 (tables above). Against *accuracy drop* instead of KL the
agreement is lower (`fisher` +0.27 … +0.86, `taylor` +0.34 … +0.89): accuracy moves by fractions of a point per layer,
which is mostly noise on 3000 images (616 sentences for DistilBERT), whereas KL is the quantity these scores model.

**2. They are an optimisation of brute force, not a replacement for it, and the saving is modest.** The like-for-like
baseline (quantize each layer, measure on the same calibration samples) is as good or better on all five CNN runs and
costs about 3× more at 256 samples (ResNet18 15.7 s vs 5.4 s; MobileNetV2 22.7 s vs 6.6 s). Two caveats that cut in
opposite directions: the gradient estimators' wall-clock is a plain per-sample torch executor, not an optimised one
(a batched per-sample-gradient implementation would be faster), while brute force on ONNX Runtime gets batching for free
on a dynamic-batch CNN but not on DistilBERT's batch-1 graph. That is why DistilBERT is the one place the
estimators clearly win at equal time (weights: +0.96 in 6.5 s vs +0.86 in 9.1 s for brute force on 64 sentences).
The advantage grows with the number of layers (brute force is linear in it, the backward pass is not); with 21–53 layers
it is a factor of about 3.

**3. Calibration-set size and noise.** Against the fixed ground truth of the matching ResNet18 run, with 5 random
calibration subsets per size (`y~model`, the default):

| calibration samples | int4 weights, `fisher` | int4 weights, `taylor` | 4-bit activations, `fisher` | 4-bit activations, `taylor` |
|---|---|---|---|---|
| 8 | +0.68 | +0.72 | +0.59 | +0.60 |
| 32 | +0.90 | +0.91 | +0.85 | +0.87 |
| 128 | +0.95 | +0.96 | +0.91 | +0.94 |
| 256 | +0.97 | +0.98 | +0.86 (std 0.16, worst +0.54) | +0.91 (std 0.09, worst +0.73) |

Weights are stable from about 32 samples on. **Activations at 256 samples are not**: with labels drawn from the model's
own distribution (the default, `label_mode="sample"`), one subset dropped `fisher` to +0.54, because the score squares a
per-sample inner product and a few large ones (here from low-confidence samples with unlucky sampled labels) dominate the
mean. The label choice is what matters: with `label_mode="argmax"` on the *same* subsets, 256 samples give `fisher`
+0.91 ± 0.04 (worst +0.86) and `taylor` +0.94 ± 0.02 (worst +0.90) for activations. For weights `argmax` is equal or better
too (n = 8: +0.77 vs +0.68 for `fisher`; n = 256: +0.98 vs +0.97). The benchmark runs above used the default `"sample"`; the
data say `label_mode="argmax"` with `taylor`, at least 64 samples, is the more robust setting (full table in the stability
section). Average over several subsets before trusting a single ranking either way.

**4. Data-free proxies mislead.** `‖W−Q(W)‖` has Spearman −0.48 (ResNet18 int4), −0.26 (ResNet18 int3), +0.17
(MobileNetV2), +0.46 (DistilBERT): not even a consistent sign. Choosing by it was worse than choosing at random in 7 of 16
cells; the relative norm in 3 of 16; the activation quantization step in 3 of 12. (We did not test *why*; layer size
and downstream amplification are not in these scores.)

**5. HAWQ-V2 did not earn its cost here.** Spearman +0.94 / +0.64 / +0.85 / +0.21 against `fisher`'s
+0.97 / +0.86 / +0.98 / +0.96 on the four weight runs, and 41–241 s against 5–7 s. On DistilBERT it took longer
(241 s) than the *whole brute-force ground truth* (49 s). Possible reasons (untested): 8 probes is few; the loss Hessian
includes terms the Fisher form ignores. Not a statement about HAWQ in its original setting (trained models with true
labels, QAT-style sensitivity).

**6. The certified bound.** A *sound* bound answers "what can go wrong for any input in the box", which is a different
question from "which layer hurts most on real data". It could not be computed for ResNet18, MobileNetV2 or DistilBERT (skip reasons
above), and on the digits nets, where it can: mean Spearman +0.32 ± 0.07 (3-bit weights), +0.17 ± 0.07 (8-bit
activations, data-calibrated) and −0.78 ± 0.04 (8-bit activations, interval-calibrated; it picked the true most-sensitive
layer in 0 of 10 nets, while the quantization step alone ranked at +0.99 there). Worst-case bounds are dominated by
amplification through the downstream layers, which grows toward the input, while the measured effect grows toward the
output where the interval-calibrated quantizer is coarsest. Use `quant_verify`/`quant_int_verify` to certify a *chosen*
configuration; do not use the bound to choose it.

**7. Where the gradient estimators fail.**

* *Collapse.* With int3 weights everywhere ResNet18 drops to 0.3 % accuracy (KL 15.7). The one-at-a-time ranking is still
  good (+0.86 … +0.89), but selections built from it are no better than random (`fisher` 1.03× random, worse than the
  random mean at k = 2 and k = 8), while brute force on 64 samples reached 0.92× random and 1.06× the oracle. When the
  model is far outside the linear regime, a gradient at the float point says little about joint quantization.
* *Few groups, noisy truth.* DistilBERT has 7 groups and the ground truth's own split-half ceiling is 0.87 (weights):
  `fisher` +0.96 has a bootstrap interval of [+0.75, +1.00] and one swapped adjacent pair moves ρ by 0.04. Per-layer
  accuracy differences there (about 1.1 points for *all* layers quantized) are within noise.
* *Top-1.* `fisher` found the single most sensitive group in 6 of 7 runs and `taylor` in 5 of 7; neither is a guarantee.
* *Interactions.* The "oracle" is the top-k by one-at-a-time KL, which is not the best set when layers interact: `taylor`
  beat it in 2 of 28 cells. It is also scored on the same evaluation split that defined it, so it is slightly optimistic;
  the estimators never saw that split.

## What does not generalise

* Three real models (two CNNs, one encoder-only text model) and 10 small MLPs; no LLM, no detection or segmentation model,
  no model with attention scores in the quantized set, nothing larger than ResNet18 / DistilBERT.
* One quantizer family: per-channel symmetric min-max weights, per-tensor affine percentile activations, fake quantization
  only; no QAT, no GPTQ/AWQ-style weight rounding, no activation-outlier handling, no integer kernels.
* Sensitivity is "quantize one group (or everything but k groups) and measure KL/accuracy". Joint effects beyond that are
  only probed through the budgeted-selection test at 4–5 values of k, with 5 random draws (no confidence intervals on
  those cells).
* The ImageNet subset is a 5000-image, class-stratified slice of the validation set, the SST-2 split 872 sentences.
  Bootstrap intervals cover evaluation-sample noise only; calibration-subset variability was measured separately and only
  for ResNet18.
* Timings are one CPU (32 threads), ONNX Runtime 1.30 for brute force and an unoptimised torch executor for the gradient
  estimators; do not read them as general ratios.
* The digits nets were trained here (400 full-batch Adam steps on 900 images, held-out accuracy 74–94 %), not downloaded;
  they exist only because the certified method needs a graph small enough for dense zonotopes.

## Reproduce

```
python scripts/quant_sensitivity_prep.py imagenet && python scripts/quant_sensitivity_prep.py cnn resnet18
python scripts/quant_sensitivity_prep.py cnn mobilenet_v2 && python scripts/quant_sensitivity_prep.py bert
python scripts/quant_sensitivity_bench.py cnn --model resnet18 --kind weights --bits 4 --est-samples 256 --ks 1,2,4,8
python scripts/quant_sensitivity_bench.py cnn --model resnet18 --kind activations --bits 4 --est-samples 256 --ks 1,2,4,8
python scripts/quant_sensitivity_bench.py cnn --model mobilenet_v2 --kind weights --bits 4 --est-samples 256 --ks 2,4,8,16
python scripts/quant_sensitivity_bench.py bert --kind weights --bits 4 --est-samples 128 --ks 1,2,3,4
python scripts/quant_sensitivity_bench.py digits --kind weights --bits 3 --nets 10 --est-samples 300
python scripts/quant_sensitivity_bench.py digits --kind activations --bits 8 --nets 10 --est-samples 300 --act-calib interval
python scripts/quant_sensitivity_bench.py stability --model resnet18 --kind weights --bits 4
```

Each benchmark writes a JSON file under `--workdir` (default `/mnt/data/cache/claude-work/qs-work`) with every
per-group score, the ground truth and the selection results; the tables above were generated from those files.
The runs are single-threaded-deterministic given the seeds, but ONNX Runtime threading and the platform's float
rounding can move the last digits.
