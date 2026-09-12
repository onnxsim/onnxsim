# Moving vendor legalize rules into onnxsim's core -- what moved, what stayed, and why

Answers: "could we move legalize passes to onnxsim library code and remove
them from scripts?" There's already a real precedent for this
(`git log`: `4073e4b7` "Port the Voyager legalizer's three rewrites to
onnxsim's C++ core", `7757d73a` "Make the three legalization passes
target-agnostic in onnxsim's core") -- this records finishing that
precedent's open half, and doing the equivalent survey and one full
promotion for `scripts/axera/legalize.py`, which is much larger.

## A real constraint the precedent already discovered, worth stating plainly

`scripts/axelera/legalize.py`'s own docstring is explicit about why its
three already-ported rules (`explicit_auto_pad`, `gemm_transA_to_transpose`,
`maxpool_rowmajor_when_indices_unused`) still carry a full standalone Python
implementation *alongside* the new C++ core pass, rather than delegating to
it: *"This file stays useful on its own: it needs nothing beyond the `onnx`
package (no onnxsim build)."* Making the Python function call
`onnxsim.simplify(model, extra_optimizers=[...])` internally would introduce
a hard dependency on onnxsim's *compiled* extension being built and
importable -- breaking that stated property for anyone using
`legalize.py in.onnx out.onnx` as a bare script against just `pip install
onnx`.

`scripts/axera/legalize.py` doesn't document this as explicitly, but the
same property holds today: it imports only `onnx`/`numpy`, nothing from
`onnxsim`. **"Remove it from scripts" is not the right read of the
precedent's own design** -- what the precedent actually does, and what this
continues, is: promote the *rewrite* to a shared, target-agnostic C++ pass
usable from every onnxsim binding, cross-reference it from the vendor
script's docstring, and keep the vendor script's own Python version working
standalone. Not deleted, not silently duplicated-and-forgotten -- documented
as intentionally dual.

## The rebuild gap, found and closed

The three already-ported passes had never actually been rebuilt+tested in
this checkout: `import onnxsim` resolved fine, but the compiled
`onnxsim_cpp2py_export` extension predated the port, so all three
(`explicit_auto_pad`, `gemm_transA_to_transpose`,
`maxpool_rowmajor_when_indices_unused`) failed with `pass %s is
unknown.<name>` from `onnxoptimizer`'s own `pass_registry.h`. An incremental
`pip install --no-build-isolation -e .` (52s, not a from-scratch rebuild --
protobuf/ONNX/onnx-optimizer were already built) picked them up; all 17
existing tests across the three then pass. Worth knowing for anyone who
lands a new `onnxsim/passes/*.h`/`custom_optimizer_passes.cpp` change in
this environment: build artifacts aren't tracked by git, so a fresh
checkout (or a worktree that never ran the C++ build) needs this rebuild
step before its tests can pass, independent of anything about the
change itself.

## Classification: every top-level rule in `scripts/axera/legalize.py`

Thirteen entries in `RULES`, eight of them also in `TRAINING_RULES` (the
subset that makes a live-weight *training* graph, not just an inference
graph, compile on Pulsar2).

### Promoted, or clearly promotable on the same reasoning (generic ONNX
rewrite, target-agnostic, the vendor-specific "why" stays in the script)

- **`neg_to_mul`** -- **promoted this session**. `onnxsim/passes/neg_to_mul.h`,
  registered, tested (`tests/test_neg_to_mul.py`, 3 cases including a real
  dtype-scoping edge case), cross-referenced from the vendor's own
  docstring. `Neg(x) -> Mul(x, -1)` is exact and the exact shape of rewrite
  the precedent already promoted three of ("backend lacks this op" class).
  Scoped to `float32` in the core pass -- the emitted `-1` constant is
  written via `Tensor::floats()` (ONNX's `float_data` field), the wrong wire
  representation for `float64`/`float16`/integer `Neg`; the vendor script's
  own Python version has the identical latent limitation (it always emits a
  `float32` constant too), just without a guard that declines other types.
- **`pow2_to_mul`** -- not ported this session (time-boxed to one complete
  promotion per the task's own instruction), but the same class exactly:
  `Pow(x, 2) -> Mul(x, x)`, exact, no vendor-specific formula in the
  rewrite itself (the vendor-specific part is *why* -- avoiding a fused
  `AxQuantizedSnake` activation that fails Pulsar2's tiler -- which would
  stay in `scripts/axera/legalize.py`/its README exactly like the
  precedent's own three rules keep their vendor rationale).
- **`float16_to_float32`** -- promotable. Retyping an fp16 graph to fp32
  throughout (initializers, `Constant` values, `Cast` targets, value_info)
  is a generic normalization need for any tool/backend that only wants
  fp32, not an AX650N-specific formula.
- **`explicit_conv_padding`** -- promotable, and **not a duplicate of
  axelera's `explicit_auto_pad`** despite the similar name: `explicit_auto_pad`
  converts a symbolic `auto_pad` mode (`SAME_UPPER`/`SAME_LOWER`/`VALID`)
  into explicit `pads`; `explicit_conv_padding` converts an *already*-explicit
  but *asymmetric* `pads` attribute into a separate `Pad` node plus symmetric
  (zero) `pads` on the convolution. Different backend limitation (auto_pad
  support vs. asymmetric-padding support), genuinely worth two separate
  passes, not one.
- **`dilated_conv_to_taps`** -- promotable. `y[t] = sum_j w[:,:,j] .
  xp[t+j*d]` decomposed into one 1x1 conv per tap, summed, is exact and
  carries no Pulsar2-specific math -- any backend without dilated-conv
  support could use it. (The rule's own docstring notes it *also* happens to
  match how the AX650N's weight table stores a dilated conv internally --
  that's a bonus property of this target, not a dependency the rewrite has
  on it.)
- **`rank0_to_rank1`** -- promotable, on the same "generic rewrite,
  vendor-specific why" split the precedent already established. A scalar
  (rank-0) graph output getting a trailing axis is not itself
  training-specific or AX650N-specific; the *reason* this project needs it
  (Pulsar2's calibration step can't concatenate a rank-0 tensor across
  samples, and a training graph's loss is always scalar) is specific and
  stays put.

### Not a graph-rewrite pass -- doesn't fit `PredicateBasedPass`'s shape

- **`filename_safe_io_names`** -- a whole-graph I/O *renaming* utility (fixing
  names like `/Add_10_output_0` that break as filenames), not a per-node
  pattern-triggered semantic rewrite. It also exists for a specific
  *consumer's* behavior (`axcl_run_model` writing `<input name>.bin` files),
  not an ONNX-level or compiler-level limitation any other backend would
  recognize. Onnxsim's `PredicateBasedPass` architecture -- `patternMatchPredicate(Node*)`
  triggering per node -- isn't the right shape for "rename every I/O value
  in the graph," and the motivating problem is genuinely this one tool's,
  not a target's compiler constraint. Stays in `scripts/axera`.

### Stay in `scripts/axera` -- specific to a *live-weight training* graph and
Pulsar2's own compiler, not general ONNX inference legalization

- **`inline_local_functions`** -- already delegates its actual work to
  `onnx.inliner.inline_local_functions` (ONNX's own stdlib), so there's
  nothing to "promote" -- the generic part is already shared. What's left
  in this function is opset-import bookkeeping specific to
  `onnxsim.graph_grad`'s own generated `FunctionProto`s (fixing an opset
  mismatch between a hand-assembled model and the functions `graph_grad`
  ships) -- training-pipeline-specific, correctly stays.
- **`avgpool_ceil_to_floor`, `flatten_to_reshape`, `global_pool_to_reduce`**
  -- exist because `onnxsim.graph_grad.build_backward` has no gradient rule
  for `ceil_mode=1` pooling, `Flatten`, or `GlobalAveragePool`, so a
  *forward* graph destined for differentiation needs these cleared before
  `build_backward` ever runs -- an autodiff-coverage question, not an ONNX
  inference-legalization one. A generic inference-graph legalizer (onnxsim's
  pass architecture, used from CLI/C/Rust/npm bindings that never touch
  `graph_grad`) has no use for "make this pool differentiable."
- **`gemm_to_matmul`, `act_weight_conv_to_matmul`** -- exist specifically
  because a **live** (runtime-input, not constant) weight breaks Pulsar2's
  own `Gemm`/`Conv` lowering in ways a normal inference graph's constant
  weight never does (`NotImplementedError('Should fuse Gemm ... to
  MatMul.')`, `AxQuantizedActWeightConv, shapefn failed`). "The weight
  might be a graph input, not a constant" is a training-graph-specific
  precondition an inference-only legalizer has no reason to check for.
- **`TRAINING_RULES`** itself -- an ordered tuple naming which of the above
  apply to a training step specifically, not a rule.

**Bottom line on the training-specific group**: onnxsim's pass architecture
is built for legalizing *inference* graphs across arbitrary targets. "Make a
live-weight backward-pass graph compile on one specific vendor's compiler"
is a narrower, different problem than any of the already-promoted rules
solve, and forcing it into the same architecture would mean either (a)
teaching the generic pass system about `graph_grad`'s own conventions (a
real coupling this project's onnxsim core has otherwise avoided), or (b)
promoting a pass that's only ever meaningfully invoked from one training
pipeline -- neither is a good trade for "runs from every onnxsim binding."
These rules are correctly scoped to `scripts/axera`.

## Regression verification

`tests/test_axera_legalize.py`, `tests/test_axera_training_legalize.py`,
`tests/test_axelera_legalize.py`, `tests/test_explicit_auto_pad.py`,
`tests/test_gemm_transa_to_transpose.py`,
`tests/test_maxpool_rowmajor_when_indices_unused.py`,
`tests/test_neg_to_mul.py`, `tests/test_build_resident_train_step.py`: **77
passed**, after the rebuild above. `ruff format`/`ruff check` clean on all
touched Python; `clang-format` clean on the new C++.

## What's left, if this is picked up further

`pow2_to_mul`, `float16_to_float32`, `explicit_conv_padding`,
`dilated_conv_to_taps` are classified as promotable above but not yet
ported (time-boxed to one complete round-trip this session, per the task's
own instruction to prioritize getting the survey right over rushing every
promotion partially) -- each would follow `neg_to_mul.h`'s exact structure:
new `onnxsim/passes/<name>.h`, register in `custom_optimizer_passes.cpp`,
`onnx.parser`-based test in `tests/test_<name>.py`, a cross-reference
paragraph added to the vendor function's docstring (Python implementation
kept, not deleted, matching the standalone-usability constraint discussed
above).
