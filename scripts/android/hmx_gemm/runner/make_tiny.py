#!/usr/bin/env python3
"""A tiny ResNet-shaped float graph (every op kind the runner supports: 7x7 s2 stem on 3 channels, 3x3 MaxPool s2,
3x3 s1 convs, residual Adds, a 3x3 s2 + 1x1 s2 downsample) with random weights -> onnxsim full_qdq (uint8 NHWC
I/O) + a random input + ORT CPU's output, for the runner's hexagon-sim test.  usage: make_tiny.py <outdir> [seed]"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, parser

BODY = """
<ir_version: 8, opset_import: ["" : 17]>
tiny (float[1,3,32,32] x) => (float[1,128,4,4] y) {
  c1 = Conv <kernel_shape = [7, 7], strides = [2, 2], pads = [3, 3, 3, 3]> (x, w1, b1)
  r1 = Relu (c1)
  p1 = MaxPool <kernel_shape = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1]> (r1)
  c2 = Conv <kernel_shape = [3, 3], pads = [1, 1, 1, 1]> (p1, w2, b2)
  r2 = Relu (c2)
  c3 = Conv <kernel_shape = [3, 3], pads = [1, 1, 1, 1]> (r2, w3, b3)
  a3 = Add (c3, p1)
  r3 = Relu (a3)
  c4 = Conv <kernel_shape = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1]> (r3, w4, b4)
  r4 = Relu (c4)
  c5 = Conv <kernel_shape = [3, 3], pads = [1, 1, 1, 1]> (r4, w5, b5)
  d5 = Conv <kernel_shape = [1, 1], strides = [2, 2]> (r3, wd, bd)
  a5 = Add (c5, d5)
  y = Relu (a5)
}
"""


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    m = parser.parse_model(BODY)
    shapes = {
        "w1": (64, 3, 7, 7),
        "w2": (64, 64, 3, 3),
        "w3": (64, 64, 3, 3),
        "w4": (128, 64, 3, 3),
        "w5": (128, 128, 3, 3),
        "wd": (128, 64, 1, 1),
    }
    for name, s in shapes.items():
        fan = s[1] * s[2] * s[3]
        m.graph.initializer.append(
            numpy_helper.from_array(
                (rng.standard_normal(s) * np.sqrt(2 / fan)).astype(np.float32), name
            )
        )
        m.graph.initializer.append(
            numpy_helper.from_array(
                (rng.standard_normal(s[0]) * 0.1).astype(np.float32), "b" + name[1:]
            )
        )
    onnx.checker.check_model(m)
    from onnxsim.full_qdq import quantize_full_qdq, quantized_io

    calib = [
        {"x": rng.standard_normal((1, 3, 32, 32)).astype(np.float32)} for _ in range(8)
    ]
    q, info = quantized_io(quantize_full_qdq(m, calib), nhwc_inputs=["x"])
    onnx.save(q, out / "model.onnx")
    xq = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8)
    xq.tofile(out / "input.bin")
    y = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    ).run(None, {"x": xq})[0]
    y.tofile(out / "ref.bin")
    print(out, info, y.shape, y.dtype)


if __name__ == "__main__":
    main()
