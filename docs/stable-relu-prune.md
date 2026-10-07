# Exact pruning from provably stable ReLUs (`onnxsim.stable_relu_prune`)

**Short version.** Over a *declared input box*, some ReLU units are provably constant-signed. A unit
whose pre-activation upper bound is `<= 0` is **dead**: it outputs exactly 0 for every input in the
box, so it can be removed (together with the consumer's matching weights) with *zero* output change
inside the box. A unit whose lower bound is `>= 0` is **always on**: the ReLU is the identity there.
`analyze` counts them; `apply` removes the dead ones (and, opt-in, fuses layers that have no
nonlinearity left).

The idea is the standard one from the neural-network-verification literature (background knowledge,
not re-checked here). What this module adds is the ONNX plumbing, three bound providers, and an
honest measurement of how often it finds anything on real models. **On the real models measured
here, it mostly does not** -- see [Results](#results).

## The result is exact only inside the box

`apply` returns a model that equals the original for inputs **inside the declared ranges**, and may
differ arbitrarily outside them (a removed unit might fire; a fused layer might have needed its ReLU).
Therefore:

* `apply` is **opt-in** and never part of `simplify()`;
* it refuses to run without a finite range for every float input (an unbounded input proves
  nothing stable anyway): pass `input_ranges={name: (lo, hi)}` or annotate the model with
  `onnxsim.ranges.set_range`;
* the box is stored in the returned model's `metadata_props` as `onnxsim.precondition.range.<input>`
  (the same JSON as `onnxsim.range.*`) plus `onnxsim.precondition.note`;
* a box *derived from data* (for example per-pixel min/max of a training set) is a statement about
  that data, **not a certificate for all inputs**. Using one is your decision, and the code does not
  pretend otherwise.

Float32 caveat: the bounds are real-arithmetic enclosures, and float32 execution can differ from them by
rounding (~1e-6 relative). A unit is therefore called stable only with a margin (`margin`, default `1e-5`
in pre-activation units). `apply` also runs a **self-check**: it samples `verify_samples` (default 32)
random inputs inside the box, runs the original and the pruned model in ONNX Runtime and raises if they
differ beyond float rounding.

## Example

<!-- doctest -->
```python
import numpy as np
import onnxruntime as ort
from onnx import numpy_helper, parser

from onnxsim import stable_relu_prune as S

rng = np.random.default_rng(0)
w1 = (rng.standard_normal((4, 6)) * 0.3).astype(np.float32)
b1 = np.array([-9.0, 0.1, -9.0, 9.0, 0.2, -9.0], np.float32)  # three units are dead on [0, 1]^4
w2 = (rng.standard_normal((6, 2)) * 0.3).astype(np.float32)
b2 = np.array([0.1, -0.2], np.float32)
model = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 13]> m (float[N,4] x) => (float[N,2] y) '
    "{ a = MatMul(x, W1)  c = Add(a, B1)  r = Relu(c)  d = MatMul(r, W2)  y = Add(d, B2) }"
)
model.graph.initializer.extend(
    numpy_helper.from_array(v, k) for k, v in dict(W1=w1, B1=b1, W2=w2, B2=b2).items()
)

box = {"x": (0.0, 1.0)}
print(S.analyze(model, box))

pruned, rep = S.apply(model, box)
print("removed units:", rep.removed_units, "| parameters:", rep.params_removed, "| MACs:", rep.macs_removed)
print("self-check on", rep.verified_samples, "inputs inside the box: max |diff| < 1e-5:", rep.verified_max_abs_diff < 1e-5)
print(sorted(p.key for p in pruned.metadata_props))


def run(m, x):
    return ort.InferenceSession(m.SerializeToString()).run(None, {"x": x})[0]


inside = rng.uniform(0, 1, (1000, 4)).astype(np.float32)
outside = rng.uniform(-50, 50, (1000, 4)).astype(np.float32)
print("inside the box : max |diff| < 1e-5:", bool(np.abs(run(model, inside) - run(pruned, inside)).max() < 1e-5))
print("outside the box: max |diff| > 0.1 :", bool(np.abs(run(model, outside) - run(pruned, outside)).max() > 0.1))
```
```text
relu                                          units   dead     on  unst. prune    params        MACs
r                                                 6      3      1      2   yes        21          18
total                                             6      3      1      2              21          18
  all prunable dead units together: 21 parameters, 18 MACs per sample (exact; per-layer figures are standalone)
  interval: 3 dead, 1 always-on
  crown: 3 dead, 1 always-on
removed units: 3 | parameters: 21 | MACs: 18
self-check on 32 inputs inside the box: max |diff| < 1e-5: True
['onnxsim.precondition.note', 'onnxsim.precondition.range.x']
inside the box : max |diff| < 1e-5: True
outside the box: max |diff| > 0.1 : True
```

The per-layer figures are *standalone* (that layer alone). Where two adjacent layers both prune the weight
between them, the figures overlap, so the report also carries exact combined `params_removed` /
`macs_removed`, computed from the pruned weight shapes.

## API

* `analyze(model, input_ranges=None, methods=("interval", "crown"), margin=1e-5, max_elements=None)`
  returns a `PruneReport` with per-layer counts (`dead`, `always_on`, `unstable`), what each provider
  proved on its own (`by_method`), whether the layer's structure is prunable (and why not), and the
  parameters / per-sample MACs that removing the dead units would free. A unit is stable if *any* provider
  proves it (all are sound). Providers over their size budget are skipped with a reason in
  `skipped_methods`; defaults: `interval` has no cap, `crown` 60000 pre-activation elements, `zonotope`
  (opt-in) 4000.
* `apply(model, input_ranges=None, remove_dead=True, merge_active=False, ...)` returns
  `(pruned_model, report)`. `merge_active=True` also fuses `Gemm/MatMul -> Relu -> Gemm/MatMul` when every
  remaining unit of the layer is always on (the ReLU is then the identity), as `W = W1 @ W2`,
  `b = b1 @ W2 (+ b2)`. Conv layers are never merged (the composition would enlarge the kernel) and
  partially active layers are left alone.
* `prune_units(model, {relu_name: unit_indices})` is the same surgery on **units you choose**, with **no
  proof**. It exists for experiments (for example removing channels that were never positive on a data set)
  and marks the output model as approximate.

Supported structure (anything else is left alone, with the reason in the report): the producer is
`MatMul [+ Add]`, `Gemm` or `Conv` (group 1), optionally followed by `BatchNormalization` / `Add` / `Mul`
by a per-unit constant; the ReLU output has exactly one consumer path through `MaxPool`, `AveragePool`,
`GlobalAveragePool`, `Identity`, `Dropout` (and `Flatten` between a conv and a MatMul/Gemm) ending at a
`MatMul` / `Gemm` / `Conv` (group 1) with an unshared constant weight. Residual joins (the ReLU output also
feeds an `Add`), depthwise or grouped convs, ReLU6 / `Clip`, graph outputs and shared weights are reported
as unprunable; their stable units are still counted by `analyze`. If every unit of a layer is dead, one is
kept (a zero-width layer is not representable) and the report says so.

## Results

Environment: Python 3.12.13, onnx 1.23.1, onnxruntime 1.30.0, numpy 2.5.3, torch 2.14.1 (CPU), one CPU.
Reproduce with `scripts/stable_relu_prune_eval.py` (`digits` and `resnet18` tasks); raw JSON is written by
the script.

### Trained digits MLPs (10 nets trained here: 8 layers x 32 units = 224 ReLU units each)

These are small MLPs trained on the real sklearn digits (900 + 300 images for fitting and calibration, 597
held-out; held-out accuracy 74.2 - 94.1 %, mean 87.2 %). They are trained here (seeds 0-9, with the training recipe of
the quantization-sensitivity benchmark's digits task), not downloaded.

| input box | interval | CROWN | zonotope | removed by `apply` |
|---|---|---|---|---|
| `[0, 1]^64` (every valid image; **certified**) | 0 / 2240 | 0 / 2240 | 0 / 2240 | 0 units |
| per-pixel min/max of the fitting data (3 of 64 pixels constant; **not certified**) | 0 / 2240 | 0 / 2240 | 0 / 2240 | 0 units |

(units proven dead or always-on, summed over the 10 nets). **No unit is provably stable over either box,
with any provider.** `apply` therefore changes nothing, and the pruned logits are bit-identical to the
original (trivially).

That is not because nothing is stable on the data. *Empirically*, on the 1200 fitting images, a mean of
**50.4 units per net (22.5 %) are never positive and 38.9 (17.4 %) are never negative**. The boxes are just
too coarse to prove it. How fast provable stability disappears as the box grows (the data box scaled toward
the mean image; mean over the 10 nets, of 224 units; "union" = proven by at least one provider):

| scale of the data box | 0 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| interval | 224.0 | 157.4 | 126.7 | 80.4 | 47.1 | 22.7 | 2.9 | 0.0 |
| CROWN | 224.0 | 220.5 | 217.3 | 205.8 | 171.6 | 77.9 | 4.1 | 0.0 |
| zonotope | 224.0 | 220.5 | 217.1 | 204.7 | 161.5 | 67.9 | 3.3 | 0.0 |
| union | 224.0 | 220.5 | 217.3 | 206.1 | 172.2 | 78.7 | 4.1 | 0.0 |

Tighter bounds find several times more stable units on a narrow box (at 0.05: CROWN 206 against intervals'
80; at 0.2: 78 against 23), which is why `analyze` runs more than one provider. At the full data box all of
them find nothing.

**Unchecked data-driven pruning** (`prune_units` with the units that were never positive on the fitting data;
*not* certified, *not* exact): it removes 33.3 % of the parameters (42 - 69 units per net) and on the 597
held-out digits gives top-1 agreement with the float model of 100 % for every net (accuracy 87.15 % before
and after; max logit difference 0.029, mean KL below 1e-5). So on this data the *statistical* version works and the
*provable* version does not, which is the central finding.

### ResNet18 (torchvision, ImageNet weights, BatchNorm folded) on real ImageNet images

5000 real validation images (every 10th, class-stratified), a seeded random split into 2000 calibration and
3000 evaluation images; float top-1 on the evaluation images is 68.4 %. 17 ReLU layers, 3904 channels.

| | channels dead | channels always-on |
|---|---|---|
| provable, valid-image box `((0-mean)/std, (1-mean)/std)` per channel (**certified**), interval | 1 | 0 |
| provable, per-pixel min/max of the 5000 images (**not certified**; identical to the valid box because 5000 images cover `[0, 1]` at every pixel), interval | 1 | 0 |
| empirical, never positive / never negative over the 2000 calibration images (**a statement about those images**) | 8 | 0 |

CROWN was skipped: 2,308,096 pre-activation elements exceed its 60,000-element budget. Interval analysis
takes about 3 s.

**Nothing is prunable.** The single provably dead channel and all 8 empirically dead channels are in the
stem ReLU, whose output (through the MaxPool) fans out into both a convolution and the first residual
`Add`, so it is reported unprunable. Of the 17 ReLU layers, 8 have the supported structure
(`Conv -> Relu -> Conv`, the first ReLU of each residual block) and none of those has a stable channel;
the other 9 (the stem and the 8 block-output ReLUs, which feed the residual `Add`) are reported with
reasons. `apply` removes 0 units; `prune_units` on the 8 empirically dead channels removes 0 as well
(skipped: fan-out). Accuracy and outputs are unchanged (68.4 % either way, because nothing was removed).

For a BatchNorm-folded ResNet18 on ImageNet, pre-activations are essentially never constant-signed, even
empirically, so neither the exact nor the data-driven version has anything to remove.

## Limits and what is not verified

* **Boxes are the whole story.** Provable stability needs a narrow input box. Over a full valid-input box
  (all of `[0,1]^64`, all valid images) nothing was provable on any model tested here. The method is
  useful where the input domain really is narrow (a fixed sensor, a known operating region), which these
  experiments do not test.
* Only two real models (the trained digits MLPs and ResNet18) were evaluated; no transformer, no LLM, no
  other CNN. MobileNetV2-style ReLU6 / depthwise layers are not supported.
* The empirical numbers describe the specific images used (2000 calibration images for ResNet18, 1200 for
  digits). They are an upper bound on what any box could prove *for that data*, not for other data.
* Bounds are real-arithmetic enclosures widened slightly; the `margin` is a float32 safety allowance, not
  a proof about float32 execution.
* CROWN and zonotope store dense per-row coefficients: they suit small and medium nets and were skipped by
  budget on ResNet18.
* `merge_active` was verified on constructed all-on MLPs (exact to float rounding), but never fired on a real
  trained net here, because no real layer was entirely always-on.
* Timings are one run on one CPU.
