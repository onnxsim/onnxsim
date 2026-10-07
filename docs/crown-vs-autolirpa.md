# onnxsim.crown vs auto_LiRPA's CROWN: why ours is tighter on some rows and looser on others

PR #2110 (VNN-LIB conformance) found no soundness violations but reported that `onnxsim.crown` is
**tighter** than auto_LiRPA 0.7.2's CROWN on many rows (41 of 90 oval21, 36 of 135 MNIST-FC at box
scale 1.0) and **looser** on others (63 of 108 ERAN rows), and left both unexplained. "Tighter than a
sound reference" is either a legitimate algorithmic difference or a soundness bug. This page decides
which, with measurements. Everything here is reproducible with `scripts/vnncomp_tightness_*.py`
(not run in CI); the raw data is under `/mnt/data/cache/claude-work/crown-audit/` and is not committed.

## Verdicts

| Question | Verdict | Evidence |
|---|---|---|
| Is "tighter than auto_LiRPA" a soundness bug? | **No: a legitimate, sound difference.** | Intersecting two valid enclosures is valid (argument below); removing that one step puts our CROWN exactly on auto_LiRPA's dense CROWN; 0 violations in the exact-reference checks (strength stated below). |
| What causes it? | **Refinement intersects each refined box with the interval box; auto_LiRPA's default does not.** | `no_intersect` ablation (tables 3, 5). |
| Is float32 noise in the old reference the cause? | Only for a few ERAN rows. | Table 1: float32 vs float64 reference differ on 22 + 14 of 108 ERAN rows, 1 + 0 of 52 ACAS Xu rows. |
| Is it the ReLU lower-slope rule? | No, the rule is the same adaptive one. | Forcing slope 1 or 0 breaks agreement everywhere (table 3). |
| The final intersection with the interval bound? | No. | `no_intersect_no_final` is identical to the default (table 3). |
| The 4096-element refinement cap, the #2106 leaf fix, float64 widening? | Not factors on these networks. | `refine_all` identical to the default; chains have no repeated leaf (#2110 measured the pre-#2106 code identical); widening is at the 1e-17 level (table 7). |
| Why is ours *looser* on ERAN? | **The Sigmoid network, not ReLU.** The ReLU network matches to 4e-11. | Table 4. Partly the validity-grid slack of our Sigmoid lines; the rest is not isolated. |
| Why *looser* on MNIST 256x4 and ACAS Xu 5_4? | auto_LiRPA's **sparse** default is tighter than dense CROWN (256x4); on 5_4 a tighter intermediate box flips the adaptive slope and loosens the result. | Tables 3, 5. |
| Can ours be made as tight as auto_LiRPA's own intersection option? | **Yes, cheaply: re-propagate the interval bounds from each refined box ("cascade").** Prototype matches `compare_crown_with_ibp=True` on 135/135 MNIST-FC and 90/90 oval21 rows. | Table 5, 6. Not applied to `crown.py` here. |

## Method

Rows are spec rows `a . Y` over the property box (`scripts/vnncomp_bench.py export-specs`), scale 1.0
unless stated. "tighter/same/looser" compares **our lower bound to the reference's lower bound** with a
relative threshold `tol * (1 + |ref|)`. The reference is auto_LiRPA 0.7.2 (BSD-3-Clause; run as an
external oracle, no code copied) in **float64** (`vnncomp_tightness_ref64.py`); the float32 numbers of the
first conformance run are shown for comparison. Our `crown` is the default `crown.bounds(method="crown")`.
Widths are `ub - lb` of the spec row.

## 1. Float32 was not the main cause

| bench | rows | tighter / same / looser vs float64 ref (1e-6) | same at 1e-9 | same at 1e-12 | float32 ref vs float64 ref (1e-6) |
|---|---|---|---|---|---|
| acasxu | 52 | 20 / 28 / 4 | 28 | 28 | 1 / 51 / 0 |
| eran | 108 | 0 / 54 / 54 | 54 | 27 | 22 / 72 / 14 |
| mnistfc | 135 | 36 / 81 / 18 | 81 | 81 | 0 / 135 / 0 |
| oval21 | 90 | 41 / 45 / 4 | 45 | 45 | 3 / 82 / 5 |

The first run's ERAN count (0 / 45 / 63) became 0 / 54 / 54 in float64: 9 rows were float32 artifacts. The rest is real.

## 2. Per-case numbers (largest differences; `rel = (ours_lb - ref_lb) / (1 + |ref_lb|)`)

| kind | network | row | ours lb | ours ub | ref64 lb | ref64 ub | rel diff | width ratio |
|---|---|---|---|---|---|---|---|---|
| tighter | ACAS Xu 3_1, g0 | 3 | -1.3113 | 1.7665 | -9.3281 | 8.8517 | +7.76e-1 | 0.1693 |
| tighter | ACAS Xu 3_1, g0 | 1 | -1.0651 | 1.2414 | -7.8813 | 6.4799 | +7.67e-1 | 0.1606 |
| tighter | MNIST 256x2, g0 | 8 | -3.3724 | 3.8964 | -4.5749 | 4.8133 | +2.16e-1 | 0.7742 |
| tighter | MNIST 256x2, g0 | 4 | -0.014391 | 1.634 | -0.25008 | 1.9462 | +1.89e-1 | 0.7505 |
| tighter | oval21 cifar_deep, g0 | 8 | -3.9451 | 13.951 | -3.9465 | 13.945 | +2.82e-4 | 1.0003 |
| tighter | oval21 cifar_base, g0 | 1 | 3.841 | 8.1456 | 3.8397 | 8.146 | +2.64e-4 | 0.9996 |
| looser | ACAS Xu 5_4, g0 | 0 | -912.42 | 503.28 | -800.07 | 486.34 | -1.40e-1 | 1.1005 |
| looser | ERAN ffnnSIGMOID 6x200 | 3 | -3.6681 | 36.567 | -2.8306 | 35.498 | -2.19e-1 | 1.0497 |
| looser | MNIST 256x4, g0 | 6 | -4.8664 | 2.8109 | -3.8164 | 2.2834 | -2.18e-1 | 1.2586 |
| looser | oval21 cifar_deep, g0 | 0 | -5.088 | 9.5097 | -5.0838 | 9.5101 | -6.94e-4 | 1.0003 |

The oval21 differences are at the 1e-4 level (width ratio within 0.04% of 1): a numerical-level difference,
not a structural one. The ACAS Xu and MNIST ones are large.

## 3. Ablation (one switch at a time) against auto_LiRPA CROWN float64, tol 1e-9, tighter / same / looser

| variant | acasxu | eran | mnistfc | oval21 |
|---|---|---|---|---|
| default | 20 / 28 / 4 | 0 / 54 / 54 | 36 / 81 / 18 | 41 / 45 / 4 |
| `no_refine` (interval boxes only) | 0 / 0 / 52 | 0 / 0 / 108 | 8 / 0 / 127 | 0 / 0 / 90 |
| **`no_intersect`** (refined boxes = CROWN only) | **4 / 47 / 1** | 0 / 54 / 54 | **0 / 81 / 54** | **0 / 63 / 27** |
| `no_intersect_no_final` (only the final intersection removed) | 20 / 28 / 4 | 0 / 54 / 54 | 36 / 81 / 18 | not run |
| `refine_all` (no 4096-element cap) | 20 / 28 / 4 | 0 / 54 / 54 | 36 / 81 / 18 | not run |
| `alpha0_one` (lower slope always 1) | 0 / 0 / 52 | 0 / 0 / 108 | 0 / 0 / 135 | 0 / 0 / 90 |
| `alpha0_zero` (lower slope always 0) | 26 / 0 / 26 | 0 / 0 / 108 | 38 / 0 / 97 | 4 / 0 / 86 |
| `ibp` | 0 / 0 / 52 | 0 / 0 / 108 | 0 / 0 / 135 | not run |

Per network (default vs `no_intersect`): ACAS Xu 3_1 and 1_4 go 4 / 0 / 0 to 0 / 4 / 0 (the intersection fully
explains them); **ACAS Xu 5_4 goes 0 / 0 / 4 to 0 / 4 / 0** (the intersection made it *looser*);
MNIST 256x2 goes 36 / 9 / 0 to 0 / 9 / 36; 256x4 stays 0 / 27 / 18; 256x6 stays 0 / 45 / 0.

## 4. Our plain CROWN reproduces auto_LiRPA's dense CROWN exactly

auto_LiRPA defaults: `compare_crown_with_ibp=False` and `sparse_intermediate_bounds=True` (the interval bound is
kept for neurons the interval pass finds stable). Our `no_intersect` against auto_LiRPA with
`sparse_intermediate_bounds=False` (read from its source; its options exist as `bound_opts`):

| | acasxu | eran | mnistfc | oval21 |
|---|---|---|---|---|
| ours `no_intersect` vs `CROWN+dense` | **0 / 52 / 0** | 0 / 54 / 54 | **0 / 135 / 0** | **0 / 90 / 0** |

ACAS Xu, MNIST-FC and oval21 agree on every row at 1e-9. The 54 ERAN rows that differ are all the Sigmoid
network (`ffnnSIGMOID__Point_6x200` 0 / 0 / 54); the ReLU network `mnist_relu_9_200` matches to 4e-11.

## 5. The mechanism, and the "cascade" prototype

Our `refine()` takes each Relu input box from CROWN and intersects it with the interval box computed **once
from the raw input**. On ACAS Xu 3_1 (group 0) the interval box beats the CROWN-only box on only a handful of
neurons (1 upper bound at layer 2, 2 lower bounds at layer 3), but a stable or tighter neuron early tightens
every later relaxation:

| layer | unstable (default) | unstable (CROWN-only) | median width (default) | median width (CROWN-only) |
|---|---|---|---|---|
| 1 | 4 | 4 | 0.1028 | 0.1028 |
| 2 | 7 | 7 | 0.268 | 0.268 |
| 3 | 8 | 11 | 0.7097 | 1.088 |
| 4 | 19 | 23 | 1.52 | 2.418 |
| 5 | 31 | 45 | 3.568 | 7.676 |
| 6 | 49 | 50 | 8.378 | 38 |

auto_LiRPA's own `compare_crown_with_ibp=True` computes the interval bound of a layer **from the already refined
bounds of the layers before it**, which ours does not. The `cascade` ablation does exactly that (after each
refined box, re-run the interval pass from it over the graph suffix and intersect):

| ours vs | acasxu | eran | mnistfc | oval21 |
|---|---|---|---|---|
| default vs `CROWN+ibpcmp` | 0 / 7 / 45 | 0 / 34 / 74 | 0 / 81 / 54 | 0 / 81 / 9 |
| **cascade vs `CROWN+ibpcmp`** | 10 / 38 / 4 | 0 / 34 / 74 | **0 / 135 / 0** | 9 / 81 / 0 |
| **cascade vs `CROWN+dense+ibpcmp`** | 0 / 48 / 4 | 0 / 34 / 74 | **0 / 135 / 0** | **0 / 90 / 0** |

(`ibpcmp` = auto_LiRPA with `compare_crown_with_ibp=True`.) The cascade matches it on every MNIST-FC and oval21
row and on 48 of 52 ACAS Xu rows. On 3_1 group 0 it narrows the spec bounds from lb `[-1.088, -1.065, -1.499,
-1.311]`, ub `[0.932, 1.241, 1.329, 1.767]` to lb `[-0.901, -0.895, -1.269, -1.136]`, ub `[0.805, 1.074, 1.173,
1.544]`. Against auto_LiRPA's *default* it is tighter on 37 of 52 (ACAS Xu) and 86 of 135 (MNIST-FC) rows, looser
on 4 and 0.

| cost (all spec groups of the benchmark, seconds) | acasxu | eran | mnistfc | oval21 |
|---|---|---|---|---|
| default | 0.2 | 1.2 | 1.1 | 9.1 |
| cascade | 0.3 | 1.4 | 1.3 | 9.1 |

Not applied to `crown.py` in this PR: the measured win (tighter bounds, about +20% time on small networks) is
real, but a default change needs its own PR and review.

## 6. The Sigmoid network (ERAN) is looser, partly from the validity-grid slack

Our Sigmoid/Tanh lines are picked from several candidates (secant, tangents, tangent-through-endpoint), each shifted to be
valid on a 33-point grid plus a margin `sup|f''| h^2 / 8`, keeping the smallest mean gap.

| variant (ERAN Sigmoid network, 54 rows) | tighter / same / looser vs auto_LiRPA CROWN | seconds (all of ERAN) |
|---|---|---|
| default (grid 33) | 0 / 0 / 54 | 1.3 |
| grid 129 | 0 / 0 / 54 | 1.4 |
| grid 1025 | 20 / 0 / 34 | 1.5 |

A finer grid flips 20 rows to tighter at negligible cost. The 34 rows that stay looser are **not explained**:
the candidate-selection rule (mean gap rather than the downstream bound) is the remaining suspect and was not
isolated. The Sigmoid network was not checked against an exact reference (no sigmoid support in the MILP/PGD).

## 7. Soundness against an exact reference

The exact side (`vnncomp_tightness_exact.py`) extracts chain MLPs (ACAS Xu, MNIST-FC) and computes the true
range of a spec row with a MILP (big-M with independent interval bounds, HiGHS) and a batched PGD search. The
rigorous test uses only values **attained by a real input through the network** (PGD points; the MILP objective
can overshoot by the solver's 1e-6 feasibility tolerance and is not used for this): a bound is *violated* if an
attained value lies outside it.

| run | variant | rows | violations (attained values) | smallest margin lb / ub | MILP closed the gap (exact range known) |
|---|---|---|---|---|---|
| ACAS Xu, scale 0.01 | default | 10 | **0** | 2.2e-17 / 2.0e-17 | 10 of 10 |
| ACAS Xu, scale 0.1 | default | 10 | **0** | 4.7e-17 / 3.5e-17 | 6 of 10 |
| ACAS Xu, scale 0.1 | cascade | 10 | **0** | 4.7e-17 / 3.5e-17 | 6 of 10 |
| ACAS Xu, scale 1.0, largest "tighter" rows | default | 6 | **0** | 1.07 / 0.93 | 0 of 6 |
| ACAS Xu, scale 1.0, largest "tighter" rows | cascade | 6 | **0** | 0.89 / 0.80 | 0 of 6 |
| MNIST-FC 256x2, scale 1.0, PGD only | cascade | 6 | **0** | 1.01 / 0.55 | n/a |

How strong this is, honestly:

* **Strong** at scales 0.01 and 0.1: the bounds nearly touch the exact range (margins down to 1e-17, i.e.
  machine precision), the MILP proves the exact range on 16 of 20 rows, and the MILP dual bound certifies our
  lower bound on 16 of 16 rows where it exists. Ours is on the safe side everywhere; where auto_LiRPA's
  float64 bound sits at -1.4e-17, ours is positive thanks to its small outward widening.
* **Weak** at scale 1.0, which is where the intersection and cascade differences appear: the MILP did not close
  within 90 s there, and the bounds are loose (the true range of an ACAS Xu 3_1 row is far inside), so margins of
  about 1 cannot separate a sound bound from a mildly unsound one. For those rows the soundness case rests on
  the argument below plus the sampling checks of #2110 (66 replayed counterexamples, 0 `unsat` against them,
  0 bound violations), not on an exact reference.
* oval21 (conv) and the Sigmoid network have no exact reference (the MILP/PGD extraction handles ReLU chains).

**Why the tighter rows are sound by construction.** `refine()` intersects the CROWN box of a Relu input with the
interval box; both enclose the true pre-activation range, so their intersection does too. The cascade re-runs the interval
pass starting from a refined (valid) box, which encloses every downstream tensor, and intersects it with the
existing (valid) box. No unsound step is introduced; the difference to auto_LiRPA is that it takes more of the
valid information.

**A mistake in my own checking, found and fixed.** My first exact-reference script looked up each spec by a
shortened key (`prop_1.vnnlib`) that is shared by several networks, so for most rows it evaluated the wrong
network and box. It was caught because the (sound) reference violated its own bound by identical amounts.
The lookup now uses the full key and asserts the network name; every exact number above is from the fixed script,
and an earlier "exact range of ACAS Xu 3_1" I had seen was discarded.

## 8. What stays unexplained

* The 4 ACAS Xu 5_4 rows where the cascade is still looser than `CROWN+dense+ibpcmp` (and where the plain intersection
  makes ours looser than the reference). The adaptive-slope non-monotonicity (a tighter intermediate box flips a
  neuron's lower slope) is consistent with the data (`no_intersect` is identical to the reference there) but I did not isolate it per neuron.
* The 34 Sigmoid rows that stay looser at grid 1025.
* The remaining 9 oval21 rows where the cascade is tighter than the sparse default (`CROWN+ibpcmp`) but
  identical to the dense one: consistent with the sparse option, not separately verified.
* Soundness of the full-box "tighter" rows against an exact reference (see section 7).

## Recommendations (not applied here)

1. Adopt the cascade in `crown.refine()` as its own PR with its own tests: tighter bounds, about +20% time,
   identical to auto_LiRPA's own intersection option on two of the four benchmarks, sound by construction.
2. Refine the Sigmoid/Tanh validity grid (33 to 1025 costs about 15% on the Sigmoid network and removes 20 of 54 "looser" rows).
3. Treat "tighter/looser than auto_LiRPA" counts only against a float64 reference; the float32 reference alone moved 9 ERAN rows.

## Reproduce

```
python scripts/vnncomp_tightness_ref64.py specs.json --out ref64.json          # in the auto_LiRPA venv
python scripts/vnncomp_tightness_audit.py compare   [--scale 1.0] [--benches acasxu,...]
python scripts/vnncomp_tightness_audit.py ablate    [--variants default,no_intersect,cascade,...]
python scripts/vnncomp_tightness_pernet.py acasxu:ACASXU_run2a_3_1 mnistfc:mnist-net_256x4 ...
python scripts/vnncomp_tightness_refvariants.py     # needs ref64 runs with --methods CROWN,CROWN+ibpcmp,CROWN+dense,...
python scripts/vnncomp_tightness_mech.py --net ACASXU_run2a_3_1
python scripts/vnncomp_tightness_exact.py --benches acasxu --milp [--scale 0.01 --kind any] [--variant cascade]
python scripts/vnncomp_tightness_summary.py "exact_*.json"
```

Limits: one machine, one CPU, single runs; the benchmark instances are the four VNN-COMP 2021 sets of #2110.
