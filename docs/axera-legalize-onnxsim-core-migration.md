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
- **`pow2_to_mul`**, **`explicit_conv_padding`** -- promoted on a separate
  branch (`axera-legalize-more-rules`, PR #1378, open but not yet merged as
  of this survey): `onnxsim/passes/pow2_to_mul.h`,
  `onnxsim/passes/explicit_conv_padding.h`, both `PredicateBasedPass`. Not
  re-verified here (this survey's own branch forked from `master`, which
  does not yet contain that PR) -- noted for completeness, not part of this
  round's work. `explicit_conv_padding` is **not a duplicate of axelera's
  `explicit_auto_pad`** despite the similar name: `explicit_auto_pad`
  converts a symbolic `auto_pad` mode (`SAME_UPPER`/`SAME_LOWER`/`VALID`)
  into explicit `pads`; `explicit_conv_padding` converts an *already*-explicit
  but *asymmetric* `pads` attribute into a separate `Pad` node plus symmetric
  (zero) `pads` on the convolution. Different backend limitation (auto_pad
  support vs. asymmetric-padding support), genuinely worth two separate
  passes, not one.
- **`float16_to_float32`** -- **promoted this session**.
  `onnxsim/passes/float16_to_float32.h`, registered, tested
  (`tests/test_float16_to_float32.py`, 4 cases), cross-referenced from the
  vendor's own docstring. Retyping an fp16 graph to fp32 throughout
  (initializers, `Constant` values, `Cast` targets, every `Value`'s own
  elemType) is a generic normalization need for any tool/backend that only
  wants fp32, not an AX650N-specific formula -- and, unlike `neg_to_mul`/
  `pow2_to_mul`/`explicit_conv_padding`, it is graph-*output*-driven rather
  than node-pattern-driven (see the architecture note below), so it is a
  `FullGraphBasedPass`, not a `PredicateBasedPass`.
- **`dilated_conv_to_taps`** -- **promoted this session**.
  `onnxsim/passes/dilated_conv_to_taps.h`, registered, tested
  (`tests/test_dilated_conv_to_taps.py`, 7 cases -- three dilation/padding
  combinations, a bias-added-exactly-once check, low-dilation and grouped-
  conv skip cases, and the disabled-by-default case), cross-referenced from
  the vendor's own docstring. `y[t] = sum_j w[:,:,j] . xp[t+j*d]` decomposed
  into one 1x1 conv per tap, summed, is exact and carries no Pulsar2-specific
  math -- any backend without dilated-conv support could use it. (The rule's
  own docstring notes it *also* happens to match how the AX650N's weight
  table stores a dilated conv internally -- that's a bonus property of this
  target, not a dependency the rewrite has on it.) A `PredicateBasedPass`,
  following `rewrite_deform_conv_to_gather.h`'s own precedent for "one node
  becomes many" via `graph.create()`/`insertBefore()` (see the architecture
  note below); `min_dilation` is fixed at 2 in the core pass rather than
  exposed as a parameter, since every call site in this project uses the
  vendor rule's own default.
- **`rank0_to_rank1`** -- **promoted this session**.
  `onnxsim/passes/rank0_to_rank1.h`, registered, tested
  (`tests/test_rank0_to_rank1.py`, 6 cases, including both the attribute-
  form and opset>=18 input-form of a Reduce*'s `axes`), cross-referenced
  from the vendor's own docstring. A scalar (rank-0) graph output getting a
  trailing axis is not itself training-specific or AX650N-specific; the
  *reason* this project needs it (Pulsar2's calibration step can't
  concatenate a rank-0 tensor across samples, and a training graph's loss is
  always scalar) is specific and stays in the vendor script. Driven by the
  graph's own output list, not a node pattern, so a `FullGraphBasedPass`,
  the same split `float16_to_float32.h` uses.

### An architecture note this survey exists to settle

Before this session, three rules above (`float16_to_float32`,
`dilated_conv_to_taps`, `rank0_to_rank1`) sat in "promotable" limbo without
being ported, and it was worth checking first whether that was because they
genuinely didn't fit anything onnxsim's C++ core already had, or simply
hadn't been gotten to. They fit, on two different existing mechanisms, both
real precedent already compiled into onnxsim before this session touched it:

- **`PredicateBasedPass`** (`third_party/onnx-optimizer/onnxoptimizer/pass.h`)
  -- the one every already-ported rule (`neg_to_mul`, `pow2_to_mul`,
  `explicit_conv_padding`) uses: `patternMatchPredicate(Node*)` triggers a
  per-node match, `runTransform(Node*, Graph&, NodeDestroyType&)` rewrites
  it. Its `runTransform` receives the whole `Graph&`, not just the matched
  node, so "one node becomes many" is not actually out of scope for it --
  `onnxsim/passes/rewrite_deform_conv_to_gather.h` was already real
  precedent for exactly that ("one node becomes dozens to low hundreds")
  before this session, via `graph.create()` + `insertBefore()`. This is what
  `dilated_conv_to_taps.h` follows.
- **`FullGraphBasedPass`** (same header) -- "the most general pass which
  allows the user to run a pass given only a graph," its own doc comment
  says: `runPass(Graph&)` with no per-node predicate at all. Already real,
  compiled-in precedent for this before this session too:
  `onnxsim/passes/quantize_fp16.h` (and `quantize_fp8.h`/`quantize_bf16.h`/
  `magnitude_pruning.h`) -- `quantize_fp16.h` in particular is close to a
  mirror image of `float16_to_float32.h`'s own need (a whole-graph dtype
  retype touching initializers, `Constant` values, and every `Value`'s own
  elemType, including graph inputs/outputs). Both `float16_to_float32.h` and
  `rank0_to_rank1.h` use this base class, registered into the same registry
  via the same `RegisterOrReplace<T>` template as any `PredicateBasedPass`
  (`custom_optimizer_passes.cpp` neither knows nor cares which base class a
  pass uses).

Net: nothing here needed new C++ pass infrastructure. The earlier
"promotable, not yet ported" state was a scheduling gap (each of the
three already-ported rules was one complete round-trip's worth of work,
time-boxed one at a time), not an architectural one.

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
`tests/test_neg_to_mul.py`, `tests/test_quantize_fp16.py`,
`tests/test_float16_to_float32.py`, `tests/test_rank0_to_rank1.py`,
`tests/test_dilated_conv_to_taps.py`, `tests/test_build_resident_train_step.py`:
**100 passed**, after an incremental rebuild (`pip install
--no-build-isolation -e .`, ~25s) in a fresh worktree off `origin/master`.
`ruff format`/`ruff check` clean on all touched Python; `clang-format` clean
on all three new C++ headers and `custom_optimizer_passes.cpp`.

## What's left, if this is picked up further

Every rule classified "promotable" above is now ported (`pow2_to_mul` and
`explicit_conv_padding` on PR #1378, not yet merged; `float16_to_float32`,
`dilated_conv_to_taps`, `rank0_to_rank1` this session) -- nothing left in
that category. What remains unported is the training-specific group
(`inline_local_functions`, `avgpool_ceil_to_floor`, `flatten_to_reshape`,
`global_pool_to_reduce`, `gemm_to_matmul`, `act_weight_conv_to_matmul`,
`TRAINING_RULES`) and `filename_safe_io_names`, both deliberately scoped to
stay in `scripts/axera` per the sections above -- there is no further
"pick this up" item pending on the promotable side of this migration.
