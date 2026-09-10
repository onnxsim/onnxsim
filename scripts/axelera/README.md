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
per-node constraint evaluator.

**Unlike `scripts/axera/`, nothing here was ever checked against a real
compiler or real hardware.** Axera's Pulsar2 tooling got to download an
already-compiled `.axmodel` from a public model repo and hand-decode it
without needing Axera's own toolchain at all. Voyager SDK's compiler has no
such public artifact to inspect, and running `deploy.py` directly --
including its no-hardware `--pipe=quantized` calibration/quantization path
-- needs `axelera-types` and `axelera-runtime`, proprietary packages served
only from Axelera's own private `axelera_runtime` package index (see
`installer_support.py` in a voyager-sdk checkout). That index is not public
and this environment has no credentials for it, so there was no way to
actually run any part of Voyager SDK's compiler here. Everything in this
directory is "what the docs say", never "what the compiler was observed to
do" -- see `voyager_simulator.py`'s module docstring for the fuller version
of this caveat, and for exactly how conservatively `evaluate_constraints()`
tries to compensate (fail closed to "unknown" on anything it can't
statically resolve, rather than guess).

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
        print(result.op_type, "would fall back to CPU:", result.detail)
    elif result.verdict == "unknown":
        print(result.op_type, "constraint not statically checkable here")
```

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
