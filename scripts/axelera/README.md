# Axelera Voyager SDK / Metis AIPU compatibility check

A docs-derived, no-hardware/no-Docker estimate of whether a graph is
friendly to Axelera AI's [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk)
(`deploy.py`, which compiles ONNX models to `.axmodel` for Metis AIPU
devices), and of whether `onnxsim.simplify()` stays safe to run ahead of it.

Also see the ["Deploying to Axelera Metis devices" section](../../README.md#deploying-to-axelera-metis-devices-voyager-sdk)
of the top-level README, and `tests/test_voyager_sdk_patterns.py` (onnxsim's
own compatibility tests against Voyager SDK's Focus/Reshape+Gemm graph
rewriter, in the main `tests/` directory) for the structural-simplification
side of this. This directory is about the AIPU op-support side instead.

## What this actually is, and isn't

Voyager SDK publishes a real per-operator AIPU support reference
(`docs/reference/compiler/onnx-support.md` and its per-opset detail pages,
auto-generated from Axelera's own operator acceleration definitions) --
including formal `rule`/`allow_config` predicates for most "Constrained"
operators, not just an op-type list. `scrape_onnx_support_docs.py` scrapes
that into `voyager_op_support_data.py`; `voyager_ops.py` and
`voyager_simulator.py` turn it into a queryable partition + a best-effort
per-node constraint evaluator -- all of that **without** any real compiler.

There is also a real backend now: **`voyager_backend.py`** wraps Voyager
SDK's actual `axelera.compiler.quantize()`. An earlier version of this
README claimed the real compiler was unreachable here (proprietary,
credentials-only) -- that was wrong. `axelera-rt`/`axelera-devkit` install
from a genuinely public Artifactory PyPI mirror with no login, exactly as
voyager-sdk's own `docs/user-guides/sdk-install.md` documents:

```bash
pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt axelera-devkit[all]
```

(This is a large, optional install -- it pulls torch, several CUDA toolkit
packages, TVM, and more. Nothing in onnxsim's own test suite or CI requires
it.) Running it for real turned up two concrete things:

- Quantizing a `Conv -> BatchNormalization -> Relu` graph and its
  onnxsim-simplified `Conv -> Relu` form (BN fused into Conv) through the
  real quantizer produces **bit-identical** output.
- A `Conv` with `auto_pad="SAME_UPPER"` (violating the scraped rule
  `auto_pad == "NOTSET"`) makes `quantize()` fail outright, and the real
  compiler's own warning quotes that *exact* constraint string --
  confirming `voyager_op_support_data.py` matches the compiler's actual
  internal check, and correcting `onnx-support.md`'s own "falls back to
  host CPU" framing: that's what happens for an *undocumented* op type, not
  for a documented-but-violated "Constrained" one.

See `voyager_backend.py`'s module docstring for the full account (including
what still doesn't work here -- `axelera.compiler.compile()`, i.e. real
deployable `.axmodel` artifacts, needs a "device support directory" this
environment doesn't have), and `tests/test_axelera_voyager_real_compiler.py`
for these pinned as regression tests (skipped automatically if
`axelera.compiler` isn't installed).

`voyager_ops.py`/`voyager_simulator.py` stay docs-only by design even
though the real compiler turned out to be reachable -- they're useful
exactly because they don't need the heavy optional install. Read their
output as an estimate corroborated by, but not equivalent to, what
`voyager_backend.py` (better) or the real `deploy.py` (authoritative) would
say.

## Usage

```python
import sys

sys.path.insert(0, "scripts/axelera")  # or run from this directory

import onnx
import voyager_simulator as sim

model = onnx.load("model.onnx")

print(sim.coverage(model))  # 'full' | 'partial' | 'none', by op-type membership alone
p = sim.partition(model)
print(p.npu_node_fraction, p.cpu_fallback_op_types)

for result in sim.evaluate_all_constraints(model):
    if result.verdict == "violated":
        # observed, for at least one op: quantize() fails outright here,
        # not a graceful CPU fallback -- see voyager_backend.py's docstring
        print(result.op_type, "violates a documented constraint:", result.detail)
    elif result.verdict == "unknown":
        print(result.op_type, "constraint not statically checkable here")
```

With the real compiler installed (see above), cross-check against it directly:

```python
import voyager_backend as backend

if backend.has_axelera_compiler():
    result = backend.compare_before_after_simplify(
        model, simplified_model, calibration_dataset_fn, test_input
    )
    print(result["bit_identical"], result["max_abs_diff"])
```

## Legalizing: fixing what `evaluate_constraints()` finds

`legalize.py` holds semantics-preserving rewrites for three of the scraped
constraints -- the ones with a fix that's an exact rewrite, not just
"avoid this op": `Conv`/`AveragePool`/`MaxPool`'s `auto_pad == "NOTSET"`
requirement (this is the same rule the real compiler's own warning quoted
back, above), `Gemm`'s `transA == 0`, and `MaxPool`'s `storage_order == 0`
when the `Indices` output goes unused. See `legalize.py`'s module docstring
for exactly what backs each rule and its real caveats (in particular:
`gemm_transA_to_transpose`'s inserted `Transpose` is itself outside what
the scraped `Transpose` rule covers for a rank-2 tensor).

```python
import legalize

model = onnx.load("model.onnx")
applied = legalize.legalize(model)  # {rule_name: sites changed}
onnx.save(model, "model.legalized.onnx")
```

Or from the command line: `legalize.py in.onnx out.onnx`. `tests/
test_axelera_legalize.py` checks each rule both structurally (against the
ONNX operator spec's own `auto_pad` formula) and numerically (`onnx.
reference.ReferenceEvaluator`, before vs. after), and cross-checks the
`evaluate_constraints()` verdict flips from `"violated"` to `"ok"` where the
rewrite is supposed to fully resolve a node -- none of that needs Docker,
a device, or the optional `axelera-rt`/`axelera-devkit` install.

## Regenerating `voyager_op_support_data.py`

The scraped data snapshot tracks whatever Voyager SDK checkout it was last
run against, not upstream automatically. To refresh it:

```bash
git clone https://github.com/axelera-ai-hub/voyager-sdk /tmp/voyager-sdk
python3 scrape_onnx_support_docs.py /tmp/voyager-sdk > voyager_op_support_data.py
```

The scraper cross-checks opsets 14-17 against each other and prints a
warning to stderr if a future Voyager SDK release makes them diverge (as of
this writing, support levels and constraints are identical across all four,
so only opset 17 -- the compiler's own recommended default -- is captured
in detail).
