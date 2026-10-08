# Verified computation and onnxsim: a survey

"Verified computation" covers three different guarantees. Each one maps to a different
part of onnxsim, so the survey is split the same way.

| Question | Field | Where it lands in onnxsim |
|---|---|---|
| Is this *rewrite* semantics-preserving? | Verified compilers, translation validation, SMT/proof-assistant rule verification | `Simplify()` and the optimizer passes |
| Did this *run* produce the right output? | Interactive/succinct proofs, Freivalds checks, optimistic fraud proofs, LSH commitments | Constant folding and `--check` |
| Can someone else *check* the result cheaply, possibly without seeing the model? | zkML (SNARK/GKR circuits) | onnxsim as the pre-processor for zkML toolchains |

Items marked (bg) come from background knowledge and were not re-checked during this
survey. The rest come from the searches linked under Sources.

## 1. What onnxsim already has

- **Hand-written Z3 specs.** There are 118 `tests/test_formal_verify_*.py` files. Each
  proves that a pass's documented algebra is sound for all indices and values.
  `scripts/formal_verify_coverage.py` reports how many registered passes have such a proof.
  The `verify` extra provides z3-solver, and CI has a dedicated job for it.
- **Differential check of the compiled pass.** `simplify_isolated` runs one pass alone
  and compares against the same spec on concrete numbers.
- **Random-sampling equivalence.** `onnxsim/model_checking.py` (`--check`) compares
  original and simplified outputs on random inputs.
- **Backend fuzzing.** `scripts/onnx_op_fuzzer.py` does differential fuzzing between
  onnxruntime and the ONNX reference evaluator. `scripts/_nnsmith_gen.py` also exists.

The stated gap, in `tests/_formal_verify_common.py`, is that the proof covers the
*spec*, not the C++ pass code. Nothing connects the two except sampled concrete
inputs. Most of the work below is about closing that gap or making it checkable per run.

## 2. Verifying rewrites (the compiler side)

- **Translation validation** (Pnueli et al.; Alive2 for LLVM (bg); CompCert's validators
  (bg)). Instead of proving the optimizer correct once, you check each (input, output)
  pair it produces. The checker is much smaller than the optimizer, so the trusted base
  shrinks. Alive2 is bounded and SMT-based, and it accepts being incomplete.
- **TensorRight** (POPL 2025). It verifies tensor graph rewrites for *arbitrary rank and
  size*. It reduces each rewrite to a finite set of bounded proof obligations and
  discharges them with an SMT solver using symbolic execution. This is the closest match
  to onnxsim's layout and shape passes (Transpose, Reshape, Slice, Concat, Pad, Gather).
  Onnxsim's current Z3 tests use fixed example shapes.
- **SuperTensor-lean.** Equality saturation over tensor graphs in Lean 4, where every
  rewrite rule carries a machine-checked proof. This is correct by construction for
  the rule set, but it means redefining the rules in Lean. It does not verify existing
  C++.
- **Tensat / TASO / PET.** These synthesise or search over rewrites. TASO and PET verify
  their substitutions automatically. Tensat adds e-graph exploration, which removes
  the phase-ordering problem. Onnxsim runs passes to a fixed point, so it has that
  problem in a mild form.
- **PolyJuice.** It fuzzed the rewrite rules of production tensor compilers and found
  84 bugs, 49 confirmed. This is evidence that unverified rewrite sets do contain
  silent miscompiles, which is the case for verifying onnxsim's own.
- **Floating point.** Z3 proofs over reals or integers do not cover fp32. Herbie,
  FPTaylor and Gappa (bg) bound the error. This matters for `fuse_bn_into_conv`, the
  constant folds, and the quantization passes. It is also where the existing
  specs are weakest: they are exact-arithmetic statements.

## 3. Verifying executions

- **Freivalds' algorithm.** To check `C = A·B`, sample a random `r` and test
  `(rᵀA)B = rᵀC`. This costs O(n²) instead of O(n³), with one-sided error of at most
  1/|F| per trial over a field. Over fp32 it needs a tolerance, so it is a check
  against bit-flips and wrong kernels, not an exact proof.
- **opML.** Optimistic execution with an interactive fraud-proof game, as in optimistic
  rollups. It is cheap on the happy path but needs a deterministic VM for the
  bisection game.
- **TOPLOC.** A locality-sensitive hash of intermediate activations. It detects changed
  models, prompts or precision, and tolerates GPU and reordering nondeterminism. It
  is a commitment, not a proof.
- **SafetyNets, Slalom, zkCNN (bg).** These are earlier interactive-proof and
  TEE-offload designs.

## 4. zkML

Frameworks:

- **EZKL.** It compiles ONNX to Halo2/Plonkish circuits.
- **DeepProve.** A GKR/sum-check approach with layers for Transformers. It claims
  50–150x faster proving than EZKL. This is the vendor's figure, not independently measured.
- **zkLLM.** It reports a 13B-parameter inference proof in under 15 minutes, with a
  proof under 200 kB.

The relevant facts for onnxsim:

- Proving cost scales with node and constraint count, so graph simplification
  directly lowers prover time. Constant folding, BN fusion and dead-code removal
  are exactly what these toolchains want before circuit compilation.
- Circuits work over fixed-point or field elements, so ops need integer-friendly
  forms. Onnxsim's quantization and Gelu/Softmax approximation passes are adjacent.
- The catch is trust. If the user proves a statement about the *simplified* model but
  the claim is about the *original*, the simplifier is in the trusted base. That makes
  a per-run equivalence certificate (section 5, item 1) the valuable piece.

## 5. Proposed work for onnxsim, in priority order

1. **Rewrite log plus translation validation.** Have each pass application record
   (pass, matched node names, replaced node names). A separate checker takes each
   local before/after subgraph and proves equivalence with Z3 at small concrete shapes
   (Alive2 style), or via TensorRight-style bounded obligations. Run it as an
   opt-in `--certify` flag. This attacks the spec-vs-code gap directly and gives
   zkML users a certificate. Risk: the nanobind boundary exposes only whole-model
   `optimize()`, so the log needs C++ support.
2. **Arbitrary-shape proofs for layout passes.** Port the existing Transpose/Reshape/
   Slice/Concat/Pad specs to TensorRight-style symbolic rank. Small and contained,
   and the passes are already identified by `formal_verify_coverage.py`.
3. **Floating-point error bounds** for BN fusion and constant folding, using
   FPTaylor or Gappa. Output a documented ulp bound per pass, replacing "within
   `rtol` on random inputs".
4. **Exact constant-fold check.** Re-evaluate folded constants in float64 or rationals
   and compare. Freivalds can spot-check large folded MatMuls cheaply. This catches
   the backend gaps the fuzzer already looks for.
5. **zkML front-end recipe.** A documented preset (fixed-point-friendly passes, no
   dynamic shapes) plus a test that feeds the simplified model to EZKL. Low effort,
   and it shows the proving-cost benefit with a measurement.
6. **Equality saturation** as an experimental pass driver. Highest effort, and it
   changes the architecture. I would only do it after 1, since extraction needs a
   checker anyway.

Not recommended: verifiable *inference* protocols (opML, TOPLOC) belong to a runtime,
not a simplifier.

## 6. Interval arithmetic

Interval arithmetic (IA) propagates `[lo, hi]` per tensor so the result is a sound
enclosure of every value the real computation can take. Variants (bg): affine arithmetic
(tracks correlations, so it avoids IA's dependency blow-up), zonotopes, and the
DeepPoly/CROWN family of linear relaxations used in neural-network verification.

What onnxsim has: no general range-propagation pass. The closest things are the
hand-derived worst-case bound in `tests/test_formal_verify_quantized_mac_bound.py`
(`eps_x*sum|w| + eps_w*sum|x| + K*eps_w*eps_x`) and its round-trip lemma. Both are
closed-form IA over a single dot product, proved with Z3 per op rather than computed
per model.

Where IA fits:

1. **Sound floating-point bounds** (replaces survey item 3's tooling). Run the original
   and rewritten subgraph in outward-rounded intervals, or in a reference format with
   a known error, and report a certified `|orig - simplified|` bound per output.
   Unlike `--check`, this holds for every input in the declared range, not a random
   sample. Needs input ranges from the user, or defaults such as `[0,1]` for images.
2. **New simplifications that need ranges.** A sound range lets onnxsim delete a
   `Clip`/`Relu` that can never fire, fold `Where` with a decided condition, drop a
   saturating cast, or narrow a Softmax's stabilising max-subtraction. Each is
   justified by the interval, so the interval is also the certificate. This is
   the part that adds capability and is not only assurance.
3. **Quantization range derivation.** Calibrating scales from data (the repo's PTQ
   work) is statistical. IA gives a worst-case range, which is far looser but
   overflow-safe: it bounds int32 accumulators of integer MatMul/Conv for a given
   weight set exactly (`sum|w| * max|x|`), and can prove no saturation.
4. **zkML.** Circuits over fields or fixed-point need a proof of no overflow or
   wraparound. IA gives that bound statically and picks the bit width.
5. **Shape and index bounds.** `sym_shape_infer` already reasons symbolically about
   dims. Adding integer intervals for dynamic dims would justify slice/gather
   index-in-range facts.

Limits:

- **Dependency problem.** Plain IA treats `x - x` as `[lo-hi, hi-lo]`. Through deep
  networks, bounds grow exponentially with depth, so they are only useful for
  shallow subgraphs, local rewrites, or with affine arithmetic and linear relaxations.
- **Not a proof of equivalence.** IA bounds error or range. It cannot show two graphs
  compute the same function. Use it next to Z3, not instead of it.
- **Rounding mode.** Sound IA needs directed rounding. ORT and numpy do not expose
  it, so it needs a small outward-rounding reference evaluator (nextafter widening is
  a cheap, sound approximation), and it will not match fused or reordered kernels
  bit for bit.

Suggested first step: a `onnxsim.ranges` module with interval transfer functions
for the common ops (Add, Mul, MatMul/Gemm/Conv via `sum|w|`, Relu/Clip, Sigmoid and
Tanh by monotonicity, Softmax in [0,1]), plus value-range annotations in `value_info`.
Use it first for item 2 (dead `Clip`/`Relu`) and item 3 (accumulator overflow), and
validate each transfer function against Z3 and random sampling, like the existing
`test_formal_verify_*` files.

### Existing libraries

Search results only; none was built or run here, and the licence, ONNX op coverage and
build requirements of each still need checking before adoption.

| Tool | What it is | Fit for onnxsim |
|---|---|---|
| **Luna** (arXiv 2603.23878) | C++ bound propagator: IBP, CROWN and alpha-CROWN over a general computational graph. Described as the first C++ alpha-CROWN with a stable interface for foreign-function integration, and competitive with the Python reference on VNN-COMP 2025 benchmarks. | Best match. Onnxsim is C++ with nanobind, and a Python-only dependency would be awkward in the C++/WASM builds. Check licence and whether it reads ONNX directly. |
| **auto_LiRPA** (PyPI `auto-LiRPA`) | Python/PyTorch library for automatic linear relaxation bounds (IBP, CROWN, alpha-CROWN), the engine under alpha,beta-CROWN. | Easiest for a Python prototype, but needs PyTorch and an ONNX-to-PyTorch route. Optional extra only. |
| **alpha,beta-CROWN** | Verifier built on auto_LiRPA plus branch and bound. It won VNN-COMP 2021 through 2025. It takes ONNX plus VNNLIB specs and has a Python API. | A full verifier. Heavy for bounds-only needs, but it fits "prove the simplified model satisfies the same property". |
| **Marabou** | SMT-based verifier with ONNX as its main format. | Complete (it can return counterexamples), but scales poorly. Suits small subgraphs. |
| **PyRAT, nnenum, NNV, ERAN, MN-BaB** | Abstract-interpretation or star-set verifiers, mostly ONNX-capable. | Alternatives for cross-checking. PyRAT is Python, so it is easier to wrap. |
| **MPFI / Rival** | Arbitrary-precision interval arithmetic with outward rounding. MPFI is C on MPFR/GMP. Rival is a Rust library. | Sound scalar kernels for a reference evaluator, in place of hand-rolled `nextafter`. |

Notes:

- The CROWN family gives *linear relaxation* bounds, which are much tighter than
  plain intervals on deep networks. That removes the main limit listed above,
  at the cost of heavier machinery.
- These are neural-network verifiers. They answer "does this input region map to
  an output region", which is a different question from "do two graphs agree".
  For equivalence you can build a product graph, `orig(x) - simp(x)`, and bound its
  output over the input region. That is the certified error bound from use 1,
  obtained with an existing tool instead of a new module.
- Their ONNX op coverage targets standard layers. Onnxsim's custom or contrib ops
  and quantized ops may not be supported, so check coverage early.
- Soundness of these tools is usually w.r.t. real arithmetic or with floating-point
  caveats. Check each tool's treatment of fp32 rounding before calling a result
  certified.

### Luna spike (2026-10-06)

Script: `spike.py` in the scratch dir (not in the repo). Luna at `github.com/ai-ar-research/luna`,
CPU build, torch 2.14 CPU. Each case is a product graph `orig(x) - simplified(x)` bounded over an input box.

| Case | Input box | CROWN bound on the difference |
|---|---|---|
| A: Conv+BN vs BN folded into Conv (folded in numpy, **not** real onnxsim output) | [-1,1] and [0,1] | 1.19e-7, zero width |
| B: dead `Relu` after `Sigmoid` | [-10,10] | exactly 0 |
| C (control): `Relu(x)` vs `x` | [-1,1] | width 2.0 (non-zero, as it must be) |
| C (control): same, Relu dead | [0,1] | exactly 0 |

Findings:

- The idea works. Luna proves a rewrite equivalent or bounds the difference, and the negative control
  shows it does not return zero unconditionally.
- Case A's 1.19e-7 is real-arithmetic reasoning over the float32 constants, so it reflects the
  rounding of the folded weights, not a sound fp32 evaluation bound. It is not yet a "certified" figure.
- The Python API exposes only `CROWN` and `alpha-CROWN` (no IBP, despite the README).
- `Clip` is unsupported, and the op list is small (Linear, Conv, BN, Relu, Sigmoid, reshape-family,
  Concat, Add/Sub). Most onnxsim passes involve ops outside it.
- alpha-CROWN crashed on Sigmoid (a torch `scatter` shape error inside `BoundedSigmoidNode`); CROWN was fine.
- The repo as cloned did not work out of the box: `lunapy/CMakeLists.txt` never links `LUNAOnnxParser`
  (undefined `TorchModel(String)` symbol), and `PYBIND11_MODULE(LirpaPyCore, ...)` has a stale name
  (`PyInit_LunaPyCore` missing). I patched both in a scratch clone only.
- Licence: `COPYING` reads "All rights reserved" with no open-source grant. Do not vendor or link Luna into
  onnxsim until the authors clarify. Using it as an external, optional tool is the safe reading.

Not yet done: run on real onnxsim output (needs a built extension), a fused-BN case with a non-trivial
BN scale range, and ops beyond Luna's list.

#### Follow-up: real onnxsim output (`onnxsim 0.7.3.dev4719` from TestPyPI, no source build needed)

`onnxsim.simplify(orig)` merged with `orig` into a product graph, bounded by Luna CROWN over `[-1,1]`:

| Model | What onnxsim did | CROWN bound on orig - simplified |
|---|---|---|
| Add c1 -> Add c2 -> MatMul | nothing (already minimal) | 0 |
| Conv -> BN (no Relu) | BN folded into Conv | 2.0e-6 (certified) |
| Conv -> BN -> Relu | BN folded | 23.6 (useless) |
| MatMul -> Add -> Relu -> MatMul -> Add | MatMul+Add -> Gemm | 51.4 (useless) |

The large bounds are the dependency problem from the section above, not a bug: with the Relu present,
CROWN relaxes the Relu in each branch independently and the two relaxations do not cancel. Shrinking the
box confirms it: Conv+BN+Relu gives 23.6 at +-1, 1.69 at +-0.1 and 6e-8 at +-0.01 (all Relus stable).
Every rewrite is exact, and onnxsim's own check passed in each case.

Consequences for a `--certify` design:

- Plain bound propagation on the whole product graph only certifies Relu-free (or stable-Relu, narrow-box)
  graphs. It is useful for linear rewrites (BN fusion, bias fusion), not for general graphs.
- The fix is structural: certify each rewrite on the *matched subgraph* (before/after the pass touched),
  where the surrounding nonlinearity is outside the window, or share the common prefix and compare only the
  diverging linear part. A rewrite log (survey item 1) makes this possible.
- alpha-CROWN failed in these bindings whenever an unstable Relu was present ("autograd engine was called
  while holding the GIL"), so only CROWN is usable from Python here. That is another upstream bug.
- The sampled difference through onnxruntime was exactly 0, because ORT fuses Conv+BN itself in both
  branches. Random sampling via ORT can therefore hide a bug in a BN fold. This is a real argument for a
  bound or SMT check next to `--check`.
- Product graph note: unnamed nodes must get unique names before merging, or onnxruntime rejects it.

### Prototype: `onnxsim/certify.py` (Z3, 2026-10-06)

`certify(orig, simplified, input_ranges=None, atol=1e-5, rtol=1e-4)` tries to prove the two models agree.
No rewrite log is needed, and the log onnxsim already writes into `metadata_props` (`removed_nodes`,
`changed_nodes`, ...) could not serve as one anyway: it is name-level, and names are reused for different
values (after a BN fold, `c` is the BN output, not the Conv output). It works in layers:

1. structural hashing of both graphs into one id space (identical computation => equal, free);
2. congruence peeling (same op + equal inputs => equal), which keeps an untouched `Relu` out of the solver;
3. a Z3 window over exact rationals for what a pass actually rewrote, cut at structurally shared tensors.

Verdicts per output: `proved-structural`, `proved-congruence`, `proved-smt`, `refuted` (with a counterexample),
or `skipped` (unsupported op, symbolic shape, over budget) -- never silently "ok".

Measured on real `onnxsim.simplify` output: MatMul/Add -> Gemm proved in under 0.1 s; Conv+BN+Relu proved in
1.3 s with `input_ranges={"x": (-1, 1)}`. Without ranges it correctly refutes Conv+BN (the folded weights differ
by ~1e-8, a huge enough input exposes it) and says so in the message, so ranges are effectively required --
which is what the range annotation proposal is for. Wrong rewrites (bad bias, RGB/BGR weight swap) are refuted.

Design points worth keeping:

- Proofs under cut points are sound, but a counterexample under cuts may not be reachable from any real input,
  so a cut-level witness is re-solved uncut before being reported as `refuted`.
- Op encodings (Conv incl. stride/pad/dilation/group, Gemm, MatMul, BatchNormalization, Clip, Relu, elementwise,
  Transpose/Flatten/Reshape) are part of the trusted base; `tests/test_certify.py` checks each against the ONNX
  reference evaluator on concrete inputs.
- That check turned up a discrepancy: at opset 13 the reference evaluator's `BatchNormalization` is off by
  about 0.09 from both onnxruntime and this encoding (epsilon-independent); from opset 15 it agrees. The cause
  was not investigated. It matters because onnxsim falls back to the reference evaluator when onnxruntime is
  absent.

Limits of the prototype: windows are encoded at their real shape (budget-limited, 300k multiply-adds), so large
Convs are `skipped`; shrinking spatial dims for shape-generic windows is the obvious next step. 2-D Conv only.
Reals, not fp32 evaluation. Not yet wired into `simplify()` or the CLI.

## 7. Training graphs: gradient vanishing and precision

A training step is an ONNX graph whose outputs include gradients, so the interval and roundoff
analyses apply to it directly. `onnxsim/grad_health.py` checks two things per gradient over an
input box: whether its magnitude bound is below the precision's underflow range (`dead`,
`flushed`, `subnormal`; a proof, because the bound holds for every point of the box), and whether
its roundoff bound from `fp_error.roundoff_bound` reaches the magnitude (`imprecise`).

What the literature covers:

- Vanishing and exploding gradients were characterised for recurrent nets by Bengio, Simard and
  Frasconi (1994), and analysed further with gradient clipping by Pascanu, Mikolov and Bengio
  (2013). A 2024 paper revisits the question for recurrent networks.
- Low-precision underflow is the reason mixed-precision training keeps FP32 master weights and
  scales the loss before the backward pass (Micikevicius et al., 2017). Adaptive loss scaling
  (2019) chooses the scale automatically.
- Training-run monitors look for silent numerical faults at runtime (TrainCheck; mechanism-driven
  monitors for LLM training instability). They observe a run; they do not bound a gradient over
  a box before the run.
- Static floating-point analysis is mature for straight-line code (Higham's rounding-error
  analysis; Satire's rigorous mixed-precision bounds; a reduced product of absolute and relative
  error bounds). Recent work applies it to neural-network libraries and operators: automatic
  precision estimation, backward error analysis of networks in floating point, and numerical
  stability analysis of deep-learning operators.
- Interval bounds can be loose on deep networks; a 2024 paper revisits interval bound propagation
  for verification.

Gap: the searches here found no work that combines a sound per-gradient magnitude bound with a
roundoff bound on an ONNX training graph and reports underflow proofs. That combination is what
`grad_health` does. Its limits: interval magnitudes grow with depth, so `imprecise` can come from
loose bounds; ops without an error model make their gradients imprecise; `flushed` assumes
round-to-nearest in the stated precision.

## Sources

- zkML overview: <https://kudelskisecurity.com/modern-ciso-blog/zkml-verifiable-machine-learning-using-zero-knowledge-proof>
- zkML survey (June 2017 – Dec 2024): <https://sotaverified.org/papers/a-survey-of-zero-knowledge-proof-based>
- zkML frameworks, 2025 analysis: <https://dev.to/extropy/the-zkml-singularity-a-comprehensive-analysis-of-the-2025-cryptographic-convergence-iln>
- TensorRight (POPL 2025): <https://popl25.sigplan.org/details/POPL-2025-popl-research-papers/29/TensorRight-Automated-Verification-of-Tensor-Graph-Rewrites>
- SuperTensor-lean: <https://reservoir.lean-lang.org/@lambdaclass/SuperTensor>
- Tensat: <https://arxiv.org/pdf/2101.01332>
- TOPLOC: <https://arxiv.org/pdf/2501.16007>
- opML: <https://www.themoonlight.io/review/opml-optimistic-machine-learning-on-blockchain>
- Freivalds-based delegation (Maverick): <https://arxiv.org/pdf/2609.10264>
- alpha,beta-CROWN: <https://github.com/Verified-Intelligence/alpha-beta-CROWN>
- auto_LiRPA: <https://pypi.org/project/auto-LiRPA>
- Luna: <https://arxiv.org/abs/2603.23878>
- VNN-COMP 2023 summary: <https://arxiv.org/pdf/2312.16760>
- PyRAT: <https://arxiv.org/html/2410.23903v1>
- Marabou: <https://github.com/neuralnetworkverification/Marabou/>
- MPFI: <https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/inria-00100985>

### Training graphs and precision

- Bengio, Simard, Frasconi, "Learning long-term dependencies with gradient descent is difficult", IEEE TNN 5(2), 1994: <https://doi.org/10.1109/72.279181>
- Pascanu, Mikolov, Bengio, "On the difficulty of training Recurrent Neural Networks" (2013): <https://arxiv.org/abs/1211.5063>
- Recurrent neural networks: vanishing and exploding gradients are not the end of the story (2024): <https://arxiv.org/abs/2405.21064>
- Micikevicius et al., "Mixed Precision Training" (2017): <https://arxiv.org/abs/1710.03740>
- Adaptive Loss Scaling for Mixed Precision Training (2019): <https://arxiv.org/abs/1910.12385>
- A Convergence Analysis of Adaptive Optimizers under Floating-point Quantization (2025): <https://arxiv.org/abs/2510.21314>
- TrainCheck: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks (2025): <https://arxiv.org/abs/2506.14813>
- Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability (2026): <https://arxiv.org/abs/2606.28116>
- Satire: Computing Rigorous Bounds for Floating-Point Rounding Error in Mixed-Precision Loop-Free Programs (2025): <https://arxiv.org/abs/2503.05924>
- A Reduced Product of Absolute and Relative Error Bounds for Floating-Point Analysis: <https://researchportal.ip-paris.fr/en/publications/a-reduced-product-of-absolute-and-relative-error-bounds-for-float/>
- Algorithms and data structures for automatic precision estimation of neural networks (2025): <https://arxiv.org/abs/2509.24607>
- Deterministic and probabilistic backward error analysis of neural networks in floating-point arithmetic: <https://hal.sorbonne-universite.fr/NUMPEX/hal-04663142v1>
- Automated Numerical Stability Analysis of Deep Learning Operators (2026): <https://arxiv.org/abs/2607.25494>
- When AllClose Fails: Round Off Error Estimation for Deep Learning Programs (ASE 2025): <https://conf.researchr.org/details/ase-2025/ase-2025-papers/118/When-AllClose-Fails-Round-Off-Error-Estimation-for-Deep-Learning-Programs>
- Make Interval Bound Propagation great again (2024): <https://arxiv.org/abs/2410.03373>
