#!/usr/bin/env python3
"""Build the real (not isolated-`MatMul`-probe) training-step graph used for
the multi-phase calibration-swap demonstration
(`docs/axera-on-device-training-handoff.md`'s "Multi-phase calibration swap"
section): the same small `x -> Conv -> Relu -> Flatten -> Gemm -> logits`
forward model `tests/test_build_resident_train_step.py` already uses,
through the real, unmodified pipeline (`add_mse_loss`/`build_resident_step`).

`gb` (the Gemm bias) is deliberately **not** trained here -- it hits a real,
separate Pulsar2 compiler crash (`TileFailException("AxQuantizedSub, tuple
index out of range")` on the SGD subtract for that specific tiny 10-element
tensor), unrelated to calibration and not investigated further; only `cw`/
`gw` are trainable.

Usage: build once per calibration phase, e.g.::

    python3 build_multiphase_calib_swap_probe.py step.onnx
    python3 -c "import sys; sys.path.insert(0,'.'); from _local_import import \
        ensure_repo_onnxsim; ensure_repo_onnxsim(); import make_training_calib \
        as m; m.make_work_dir('step.onnx', 'phase1', weight_scale=0.05); \
        m.make_work_dir('step.onnx', 'phase2', weight_scale=0.0005)"

then `pulsar2_docker.build()` each `phaseN/` work dir with
`config_path='config/step.json'`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _local_import import ensure_repo_onnxsim  # noqa: E402

ensure_repo_onnxsim()

import build_resident_train_step as brts  # noqa: E402


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def forward_model() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    cw = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.3
    gw = rng.standard_normal((10, 32)).astype(np.float32) * 0.2
    gb = rng.standard_normal((10,)).astype(np.float32) * 0.1
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,1,4,4] x) => (float[1,10] logits)
        {
          h = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, cw)
          r = Relu(h)
          f = Flatten<axis=1>(r)
          logits = Gemm<transB=1>(f, gw, gb)
        }
        """
    )
    model.graph.initializer.extend([_f32(cw, "cw"), _f32(gw, "gw"), _f32(gb, "gb")])
    return model


def main(argv=None) -> int:
    out_path = (argv or sys.argv[1:])[0]
    forward = forward_model()
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)
    step_model, state = brts.build_resident_step(with_loss, ["cw", "gw"])
    print(f"step graph: {len(step_model.graph.node)} nodes")
    for p, out_name in state.items():
        print(f"  state: {p} -> {out_name}")
    onnx.save(step_model, out_path)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
