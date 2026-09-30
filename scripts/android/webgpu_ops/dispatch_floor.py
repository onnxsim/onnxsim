"""Write chains of N trivial WebGPU dispatches (X -> Add c0 -> Add c1 ... ) to measure the per-dispatch wall-clock floor.

    python dispatch_floor.py OUT_DIR [SIZE] [N ...]

Each Add uses its own constant so nothing folds or fuses. Time them with bench.cc (`bench chain_N.onnx webgpu 5 30`, with
`iobind=1 enableGraphCapture=1 sync=tiny.onnx` for graph capture) and take the slope over N.
"""

import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper, parser

out = sys.argv[1]
size = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
ns = [int(a) for a in sys.argv[3:]] or [1, 50, 100, 200]
os.makedirs(out, exist_ok=True)
for n in ns:
    body = "\n".join(f"  t{i + 1} = Add(t{i}, c{i})" for i in range(n))
    model = parser.parse_model(
        f'<ir_version: 8, opset_import: ["" : 17]> g (float[1,{size}] t0) => (float[1,{size}] t{n}) {{\n{body}\n}}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(
            np.full((1, size), 0.001 * (i + 1), np.float32), f"c{i}"
        )
        for i in range(n)
    )
    onnx.checker.check_model(model)
    onnx.save(model, f"{out}/chain_{n}.onnx")
