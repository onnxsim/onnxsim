# Radxa Dragon Q8B tinygrad CPU tuning

Measured 2026-11-17 on `radxa-dragon-q8b`, SC8280XP,
Linux 7.0.11-8-qcom aarch64, Python 3.12.16, tinygrad 0.14.0,
ONNX Runtime 1.32.0, LLVM/Clang 21. Environment: `requirements.lock`.

Median of 30 synchronized TinyJit replays, milliseconds:

| Model | BEAM=0 | BEAM=2 | Speedup | Max absolute error, tuned |
|---|---:|---:|---:|---:|
| focus_preprocess_graph | 0.683 | 0.438 | 1.56x | 0 |
| focus_conv_test | 104.418 | 36.488 | 2.86x | 2.86e-6 |
| focus_conv_nested_test | 113.288 | 46.874 | 2.42x | 2.86e-6 |
| sbs_s50_postprocess_graph, repaired/inlined copy | 5.979 | 2.511 | 2.38x | 5.72e-6 |

Every setting passed three independent input checks, including JIT replay,
at `rtol=atol=1e-4`. Inputs are resident on CPU; compilation/search and
input/output transfers are excluded. Single-thread ORT medians are also
recorded in JSON (about 0.87, 14.28, 14.15, and 2.91 ms respectively), using host
arrays. ORT remains substantially faster for the convolution. These are
individual existing graph fixtures, not full detector/classifier benchmarks.
CPU affinity and frequency were not pinned.

The remote environment, script, logs, JSON and tinygrad's tuning cache remain
available at `/home/radxa/tinygrad-opt` and `/home/radxa/.cache/tinygrad`.
The input fixtures are under `/home/radxa/voyager-sdk/tests/assets`.
For example:

```sh
ssh radxa@radxa-dragon-q8b
cd /home/radxa/tinygrad-opt
.venv/bin/python tune_onnx.py \
  /home/radxa/voyager-sdk/tests/assets/focus_conv_test.onnx \
  --beam 0,2 --json focus-conv-results.json
```

The original SBS fixture cannot load in ORT because its local function has an
empty formal input name. A separate copy was repaired by removing the unused
empty formal parameter and matching empty call argument, then inlining the
function and removing the unused custom opset import. The original fixture
was preserved. Reproduce the copy with:

```python
import onnx
from onnx import inliner

model = onnx.load('/home/radxa/voyager-sdk/tests/assets/sbs_s50_postprocess_graph.onnx')
for function in model.functions:
    empty = [i for i, name in enumerate(function.input) if name == '']
    for node in model.graph.node:
        if (node.domain, node.op_type) == (function.domain, function.name):
            for i in reversed(empty):
                assert node.input[i] == ''
                del node.input[i]
    for i in reversed(empty):
        del function.input[i]
model = inliner.inline_local_functions(model)
used = {node.domain for node in model.graph.node}
imports = [opset for opset in model.opset_import if opset.domain in used]
del model.opset_import[:]
model.opset_import.extend(imports)
onnx.checker.check_model(model)
onnx.save(model, '/home/radxa/tinygrad-opt/sbs_s50.inlined.onnx')
```

This pass uses CPU code generation and preceded Hexagon SDK provisioning.
See the [FastRPC setup notes](../../../hexagon/fastrpc_linux/README.md) for
the separate v68 SDK installation and physical cDSP validation.
