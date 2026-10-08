# VNN-LIB / VNN-COMP conformance of onnxsim's bound engines

Question: can `onnxsim.crown` (crown / alpha / prima / bab), `onnxsim.interval` (ibp) and
`onnxsim.zonotope` be **trusted**, measured on the standard neural-network-verification format
against independent references instead of only our own sampling tests?

Engines under test: onnxsim at master `044f3dbf` (includes the CROWN leaf-cancellation fix #2106,
box noise, `backward_diff`). CPU only, single run, no tuning.

Headline: **0 soundness violations** (see the first table). Coverage is modest: these are dense
bound-propagation engines with small BaB budgets, far from state-of-the-art branch-and-bound
verifiers. The soundness result is the main finding; the coverage numbers are not a claim of
competitiveness.

## What is implemented

`onnxsim/vnnlib.py` (pure Python; torch is imported lazily, only by the counterexample search):

* `parse` / `parse_file`: a VNN-LIB subset. Input boxes (`X_i` against constants, single-variable
  atoms only), output atoms that are linear in `Y_j` (`+`, `-`, `*` / `/` by constants), `and` /
  `or` nesting (expanded to a union of clauses), `<=`, `>=`, `<`, `>`, `=`.
  Anything else (mixed X/Y atoms, nonlinear terms, `assert` forms we do not know, unbounded boxes,
  ...) raises `VnnlibError` -> verdict `unsupported`. It never guesses.
* `verify(model, prop, engine, timeout, budget, attack, ...) -> Verdict`.
  The convention is VNN-COMP's: the file describes the **unsafe** region.
  `unsat` = proved safe, `sat` = a counterexample that **replays on onnxruntime**, `unknown`,
  `unsupported`.
* Proof rule: a clause (input box + atoms `a.Y <= b`) is empty if for one atom the proven lower
  bound of `a.Y` exceeds `b` (small relative margin). The property is safe iff every clause is empty.
  The engines bound a spec model (`MatMul(Flatten(Y), A^T)`), `A` must be exactly float32.
* BaB (`bab`, `bab-beta`) reuses `crown._Bab`; a region is decided when ANY atom of its clause is
  infeasible.
* `verify` never returns `unsat` without first attacking every clause (PGD + corners + samples);
  if that finds a replayable counterexample the verdict is `sat` with `inconsistent=True` and a
  "SOUNDNESS ALARM" detail (an alarm would be a bug in an engine).

Engines: `ibp`, `zonotope`, `crown` (backward CROWN with intermediate refinement), `alpha`
(torch alpha-CROWN), `prima`, `bab` (CROWN leaves), `bab-beta`.

## Benchmarks and where the numbers come from

* Instances: `stanleybak/vnncomp2021`, `benchmarks/{acasxu,mnistfc,eran,oval21}` (VNN-COMP 2021
  files, fetched at the repo's final commits; `cifar2020` is supported by the script but not run).
  `oval21` was run on 30 instances, `eran` 72 (the mnist_relu nets), `mnistfc` 90, `acasxu` 186
  (the instances listed in each benchmark's `*_instances.csv`).
* Published verdicts: the per-tool `results_csv/*.csv` of `stanleybak/vnncomp2021_results`
  (VNN-COMP **2021**; tools DNNF, Debona, ERAN, Marabou, NV.jl, RPM, VeriNet, a-b-CROWN, nnenum, nnv,
  oval, randgen, venus2). Two evidence tiers:
  *hard* = a counterexample that replays on onnxruntime (ours), *soft* = published claims.
  Tool claims conflict for some instances (32 conflicts on mnistfc), so a consensus is not trusted
  as ground truth; `gt=conflict/open` instances are excluded from the "proved" columns.
* Independent reference bounds: **auto_LiRPA 0.7.2** (BSD-3-Clause, read from the installed
  package's metadata; installed into a separate venv from source because the PyPI pins an old
  torch). It is only *run* (`scripts/vnncomp_ref_bounds.py` calls its public API: `BoundedModule`,
  `PerturbationLpNorm`, CROWN / alpha-CROWN with `C=A`); **no auto_LiRPA code is copied** into the
  repository. The script is not run in CI.
* Our own counterexample search is the independent oracle for "unsat despite a real counterexample".

Reproduce (not part of CI, needs downloads): `scripts/vnncomp_bench.py run|summary|export-specs|tightness`,
`scripts/vnncomp_ref_bounds.py`.

## Running as a VNN-COMP tool (2025 harness)

`scripts/vnncomp/{install_tool,prepare_instance,run_instance}.sh` implement the tool side of the
VNN-COMP 2025 harness (`run_single_instance.sh v1 <tool> <category> <onnx> <vnnlib> <timeout> ...`
calls `prepare_instance.sh v1 <category> <onnx> <vnnlib>`, then
`run_instance.sh v1 <category> <onnx> <vnnlib> out.txt <timeout>`, and reads `out.txt`).
`run_instance.sh` writes the first line of `<results>`:

* `unsat` (proved safe); `unknown` (also for properties outside the supported subset, which the
  protocol has no separate word for); `error` (an exception, e.g. a missing file; the traceback
  goes to stderr);
* `sat`, a counterexample that replays on onnxruntime, followed by a block of lines
  `(X_i v)` for the inputs and `(Y_j v)` for the outputs, wrapped in `(` / `)` on their own lines;
* `timeout`: written by a watchdog `min(5 s, 10 %)` before the limit, after which the process
  exits with status 0. The file is replaced atomically, so the harness never reads a partial word.

The same logic is `python -m onnxsim.vnnlib run ONNX VNNLIB RESULTS TIMEOUT [--engine bab]
[--budget 200]`. Tests: `tests/test_vnnlib.py` (words, counterexample block, unsupported / error
mapping, the watchdog in a subprocess, the CLI).

The counterexample block follows the harness (`sed 's/^sat (/sat\n(/'` and `tail --lines=+2`). The
2024 and 2025 rules, read through a summarising fetch rather than verbatim, show the witness as
`((X_0 ...)` ... `(Y_j ...))` after `sat`: the same s-expression with different whitespace. Not
checked against the official checker itself.

Not implemented:

* The benchmark argument is ignored: every benchmark gets the same engine and budget.

## 1. Soundness (first, because it is the point)

Checks:

1. **Verdict level.** Any engine `unsat` on an instance where the independent attack pass
   (10 s per instance) found a replayed counterexample = hard violation. Also counted: internal
   alarms, `unsat` on a published-unsafe instance, replayed `sat` on a published-safe instance.
2. **Bound level.** For each engine and each spec row the proven lower bound must not exceed the
   smallest value observed by sampling (tolerance 1e-6 relative) on boxes of several sizes
   (ACAS Xu 0.01 / 0.1 / 1.0 of the property box, MNIST-FC and ERAN and oval21 0.3 / 1.0).

| benchmark | engines run | instances | `unsat` despite replayed counterexample | internal alarms | `unsat` but published unsafe | replayed `sat` but published safe |
|---|---|---|---|---|---|---|
| acasxu | ibp, zonotope, crown, alpha, prima, bab | 186 | 0 | 0 | 0 | 0 |
| mnistfc | ibp, zonotope, crown, alpha, bab | 90 | 0 | 0 | 0 | 0 |
| eran | ibp, zonotope, crown, alpha, bab | 72 | 0 | 0 | 0 | 0 |
| oval21 | ibp, zonotope, crown, alpha, bab | 30 | 0 | 0 | 0 | 0 |

The attack pass found replayed counterexamples on 66 instances in total, so check 1 had material
to bite on. **Total hard soundness violations: 0.**

Bound level (check 2): 0 bounds above an observed value at 1e-6 relative, for ibp / zonotope /
crown / alpha, on all 9 (benchmark, box scale) configurations. The margin `observed_min - lb`
shows the check has teeth where it matters: on ACAS Xu at scale 0.01 the median margin of crown is
1.6e-6 (minimum -5e-8, i.e. float32 rounding of onnxruntime's outputs, not an unsound bound), so the
bounds are nearly attained by real samples.

## 2. Verdicts versus the published verdicts

Columns: `pub:safe` / `pub:unsafe` = instances with a published verdict (soft tier, conflicts
excluded); `proved` = our `unsat` on a published-safe instance; `found` = our replayed `sat` on a
published-unsafe instance. Engine rows use `--no-attack` (bounds only) except `attack` which is the
counterexample search alone (10 s). Timeouts: 60 s (oval21 120 s); BaB budget 600 / 200 / 200 / 100
regions (acasxu / mnistfc / eran / oval21).

| benchmark | engine | n | unsat | sat | unknown | pub:safe | proved | pub:unsafe | found | avg s | max s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| acasxu | ibp | 186 | 0 | 0 | 186 | 133 | 0 | 47 | 0 | 0.00 | 0.1 |
| acasxu | zonotope | 186 | 2 | 0 | 184 | 133 | 2 | 47 | 0 | 0.00 | 0.0 |
| acasxu | crown | 186 | 14 | 0 | 172 | 133 | 14 | 47 | 0 | 0.01 | 0.1 |
| acasxu | alpha | 186 | 16 | 0 | 170 | 133 | 16 | 47 | 0 | 0.20 | 0.9 |
| acasxu | prima | 186 | 16 | 0 | 170 | 133 | 16 | 47 | 0 | 0.38 | 1.7 |
| acasxu | bab | 186 | 15 | 0 | 171 | 133 | 15 | 47 | 0 | 2.28 | 4.8 |
| acasxu | attack | 186 | 0 | 45 | 141 | 133 | 0 | 47 | 45 | 7.61 | 10.0 |
| mnistfc | ibp | 90 | 0 | 0 | 90 | 9 | 0 | 41 | 0 | 0.15 | 0.3 |
| mnistfc | zonotope | 90 | 17 | 0 | 73 | 9 | 3 | 41 | 0 | 0.45 | 0.7 |
| mnistfc | crown | 90 | 26 | 0 | 64 | 9 | 7 | 41 | 0 | 0.87 | 1.5 |
| mnistfc | alpha | 90 | 26 | 0 | 64 | 9 | 7 | 41 | 0 | 1.61 | 2.8 |
| mnistfc | bab | 90 | 26 | 0 | 64 | 9 | 7 | 41 | 0 | 5.32 | 9.3 |
| mnistfc | attack | 90 | 0 | 19 | 71 | 9 | 0 | 41 | 17 | 8.18 | 11.1 |
| eran | ibp | 72 | 3 | 0 | 69 | 51 | 3 | 5 | 0 | 0.16 | 0.3 |
| eran | zonotope | 72 | 14 | 0 | 58 | 51 | 14 | 5 | 0 | 0.50 | 0.8 |
| eran | crown | 72 | 36 | 0 | 36 | 51 | 34 | 5 | 0 | 1.19 | 1.5 |
| eran | alpha | 72 | 36 | 0 | 36 | 51 | 34 | 5 | 0 | 2.24 | 4.7 |
| eran | bab | 72 | 36 | 0 | 36 | 51 | 34 | 5 | 0 | 4.72 | 22.0 |
| eran | attack | 72 | 0 | 1 | 71 | 51 | 0 | 5 | 1 | 10.16 | 11.6 |
| oval21 | ibp | 30 | 0 | 0 | 30 | 17 | 0 | 0 | 0 | 5.50 | 8.3 |
| oval21 | zonotope | 30 | 0 | 0 | 30 | 17 | 0 | 0 | 0 | 34.68 | 65.5 |
| oval21 | crown | 30 | 1 | 0 | 29 | 17 | 1 | 0 | 0 | 18.54 | 29.8 |
| oval21 | alpha | 30 | 1 | 0 | 29 | 17 | 1 | 0 | 0 | 32.44 | 70.0 |
| oval21 | bab | 30 | 2 | 0 | 28 | 17 | 2 | 0 | 0 | 66.85 | 124.7 |
| oval21 | attack | 30 | 0 | 1 | 29 | 17 | 0 | 0 | 0 | 19.59 | 25.9 |

(Numbers are from the final summary over all result files.)

Honest reading:

* Our engines prove few ACAS Xu properties. One-shot CROWN proves 14 of 133 published-safe
  instances and BaB with a 600-region budget does not add any (15). State-of-the-art verifiers
  (a-b-CROWN, nnenum, Marabou) solve nearly all of them with far larger, GPU- / LP-based
  branch-and-bound. This is a dense, small-budget BaB, not a competitor.
* ERAN mnist_relu nets are the strongest case: crown proves 34 of 51 in about 1 s each; alpha and
  BaB add nothing on this set.
* The counterexample search finds 45 of 47 published-unsafe ACAS Xu instances, only 17 of 41 on
  mnistfc and 1 of 5 on eran. Misses on mnistfc are mostly instances that a single tool (RPM)
  claimed; see the discrepancy below.
* `alpha` (here with a short iteration budget) and `prima` improve ACAS Xu by two instances over
  crown; they do not help on mnistfc / eran / oval21.

## 3. Tightness versus auto_LiRPA (identical network and box)

Metric: share of the IBP-to-reference gap recovered, `(lb_ours - lb_ibp) / (lb_ref - lb_ibp)`
(median over spec rows; 1.0 = as tight as the reference), plus per-row counts at 1e-6 relative:
tighter / same / looser than the reference. Rows are spec rows `a.Y` over the property box,
shrunk by the given factor. Reference: auto_LiRPA 0.7.2 CROWN and alpha-CROWN. Our `crown` uses
intermediate refinement; our reference alpha-CROWN run used **weak settings** (early stop), so
the alpha comparison is not a fair comparison against a fully tuned alpha-CROWN.

vs auto_LiRPA CROWN, ours = `crown` (tighter / same / looser, median share):

| benchmark | box scale | rows | median share | tighter | same | looser |
|---|---|---|---|---|---|---|
| acasxu | 0.01 | 52 | 1.000 | 0 | 51 | 1 |
| acasxu | 0.1 | 52 | 1.000 | 1 | 47 | 4 |
| acasxu | 1.0 | 52 | 1.000 | 20 | 27 | 5 |
| eran | 0.3 | 108 | 1.000 | 9 | 54 | 45 |
| eran | 1.0 | 108 | 1.000 | 0 | 45 | 63 |
| mnistfc | 0.3 | 135 | 1.000 | 13 | 122 | 0 |
| mnistfc | 1.0 | 135 | 1.000 | 36 | 81 | 18 |
| oval21 | 0.3 | 90 | 1.000 | 16 | 72 | 2 |
| oval21 | 1.0 | 90 | 1.000 | 41 | 45 | 4 |

Same, ours = `zonotope` (looser on most rows; median share 0.95 acasxu 1.0, 0.84 eran 1.0,
0.81 mnistfc 1.0, 0.99 oval21 1.0), and ours = `alpha` vs auto_LiRPA CROWN: tighter on 50 / 52 (acasxu 1.0),
54 / 108 (eran 1.0), 119 / 135 (mnistfc 1.0), 89 / 90 (oval21 1.0).

A check that this is not the #2106 change: on ACAS Xu at scale 1.0, the pre-#2106 `crown.py`
gives the identical 20 / 27 / 5 split.

Findings:

* **Ours is tighter than a sound reference on some rows** (e.g. 41 of 90 oval21 rows, 36 of 135
  mnistfc rows). This is not by itself a soundness problem: every such bound still satisfied
  `lb <= min observed` in check 2, and the auto_LiRPA numbers there are plain CROWN; our crown
  additionally refines intermediate bounds and uses a different ReLU lower-slope choice. I did
  **not** isolate which of these causes each individual tighter row, so that explanation is
  a hypothesis. The soundness evidence for those rows is check 2 (observed values) only, not an
  independent proof of the bound.
* **Ours is looser than auto_LiRPA CROWN on many ERAN rows** (63 of 108 at scale 1.0, 45 of 108 at
  0.3) while the median share still reads 1.000. The median hides that the loose rows are loose;
  I did not quantify the size of the gap per row.
* The per-case table of width ratios requested is not reproduced here; the data per row is in the
  `tight_*.json` files produced by `scripts/vnncomp_bench.py tightness` (not committed, large).

## 4. Unresolved discrepancy: `mnistfc mnist-net_256x2 prop_0_0.05`

Published as `violated` by 8 of the 2021 tools. On this repository's files the network predicts
class 8 over the whole box with a margin of about +1.0 at the centre and both corners, and a 120 s
PGD finds no counterexample (closest margin +0.96). Our `ibp` / `zonotope` / `crown` return
`unknown` (no unsound `unsat`), auto_LiRPA CROWN's lower bounds are also negative, so neither
engine contradicts the published claim by proving the property. The property file was regenerated
upstream ("updated to competition seed", 2021-07-01); I compared both versions (they differ) and
neither has a counterexample on the same network. I could not determine why the published tools
report `violated`. Treated as open, not as an engine bug and not as evidence for either side.

Also open: ACAS Xu `1_9 prop_7` is borderline (closest `f = +5.5e-5` after the strong attack).

## 5. Limits

* One run, CPU only, no repeats; timings are on a loaded 32-core machine (load average about 21).
* Only VNN-COMP 2021 benchmarks, four of them, with the instance counts above; `cifar2020`
  and the larger convolutional benchmarks were not run.
* Published verdicts are soft evidence (conflicting for some instances); the hard evidence is
  only replayed counterexamples.
* The independent attack is a PGD / sampling search, so absence of a counterexample proves nothing;
  "0 violations" means none of the engines contradicted a counterexample we could find, plus
  the bound-level check against sampled values.
* The parser supports a subset of VNN-LIB (see above); unsupported input is reported, not guessed.
* auto_LiRPA alpha-CROWN reference settings were weak; comparison against a tuned alpha-CROWN
  was not done.
* BaB budgets are small (see table 2); `bab-beta` and a cifar run were not benchmarked.
