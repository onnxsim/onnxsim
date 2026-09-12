#!/usr/bin/env python3
"""Build resident-dataset-`Gather` training-step graphs for the real
resnet18 probe `docs/axera-on-device-training-handoff.md`'s "Trading free
memory for throughput" section uses (`Conv_268/271/274` + `fc.weight`,
5,361,664 trainable params) -- both the flattened-dataset workaround
(`add_resident_dataset(..., flatten=True)`, the default, and now confirmed
on real hardware) and the original native-shape variant that section found
failing on any real AX650N.

There is no committed exporter anywhere in this repo for this project's
64x64-input resnet18d probe (every earlier real-hardware resnet18 result in
the handoff doc built it from an ad-hoc, uncommitted session scratchpad
copy -- a pre-existing gap this module does not attempt to close). Pass
`--forward-onnx` pointing at one: a resnet18d-shaped ONNX graph whose first
input is `[1, 3, 64, 64]` and that carries `onnx::Conv_268`/`_271`/`_274`
and `fc.weight` as its last-block/head initializers (the exact scope every
resnet18 section in the handoff doc uses).

Usage::

    build_resident_dataset_gather_probe.py FORWARD.onnx OUT_DIR --rows 190

writes `OUT_DIR/gather_step_n<rows>.onnx` (flattened, the fix) and, with
`--also-native`, `OUT_DIR/native_step_n<rows>.onnx` (the original failing
shape, for an A/B rebuild) -- feed either to `make_training_calib.py` with
`index_inputs={"batch_index": rows}` and then `pulsar2_docker.build()`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_resident_train_step as brts  # noqa: E402

PARAMS = ["onnx::Conv_268", "onnx::Conv_271", "onnx::Conv_274", "fc.weight"]


def build(forward_onnx: str, n_rows: int, out_dir: str, also_native: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fwd = onnx.shape_inference.infer_shapes(onnx.load(forward_onnx))
    with_loss = brts.add_mse_loss(fwd, fwd.graph.output[0].name, num_classes=1000)

    rng = np.random.default_rng(0)
    input_shape = tuple(
        d.dim_value for d in with_loss.graph.input[0].type.tensor_type.shape.dim
    )
    x_data = rng.standard_normal((n_rows, *input_shape[1:])).astype(np.float32)
    y_data = rng.standard_normal((n_rows, 1000)).astype(np.float32)

    variants = [("gather", True)]
    if also_native:
        variants.append(("native", False))

    for label, flatten in variants:
        resident = brts.add_resident_dataset(
            with_loss, {"x": x_data, "y": y_data}, flatten=flatten
        )
        onnx.checker.check_model(resident)
        step_model, state = brts.build_resident_step(resident, params=PARAMS)
        onnx.checker.check_model(step_model)
        print(
            f"{label} N={n_rows}: {len(step_model.graph.node)} nodes, "
            f"state={list(state)}"
        )
        out_path = os.path.join(out_dir, f"{label}_step_n{n_rows}.onnx")
        onnx.save(step_model, out_path)
        print("wrote", out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward_onnx")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--rows",
        type=int,
        default=190,
        help="resident dataset row count (default 190, the real, measured "
        "compile-time OCM ceiling for this scope on AX650 -- see the "
        "handoff doc's 'Trading free memory for throughput' section)",
    )
    parser.add_argument("--also-native", action="store_true")
    args = parser.parse_args(argv)
    build(args.forward_onnx, args.rows, args.out_dir, args.also_native)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
