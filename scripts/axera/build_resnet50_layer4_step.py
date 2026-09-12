#!/usr/bin/env python3
"""Extends `build_resnet50_batch_step.py`'s single-bottleneck-block scope
(`layer4.2` + `fc.weight`, 6.5M params) to the remaining two `layer4`
bottleneck blocks -- the concrete next step
`docs/axera-on-device-training-handoff.md`'s "A different architecture:
resnet50, first compile" section names explicitly: "the remaining two
bottleneck blocks of `layer4` (watch compile time; extrapolate from this
block's 57.8s before jumping straight to all three)".

Reuses every export/fixup helper from `build_resnet50_batch_step.py`
unchanged (the `dynamo=True` opset-18 `ReduceMean` downgrade and the
batch-1-literal flatten `Reshape` fix apply identically regardless of
trainable scope) -- only `TRAIN_PARAMS` differs, so this module imports that
one rather than duplicating it.

Two scopes, added incrementally per the doc's own "watch compile time"
caution rather than jumping straight to all three blocks:

- ``layer4_1_2``: `layer4.1` + `layer4.2`, 6 convs + `fc.weight`, 7 trainable
  tensors (roughly double the original scope's tap count).
- ``layer4_all``: all three bottleneck blocks + `fc.weight`, 9 convs +
  `fc.weight`, 10 trainable tensors (the doc's "all three" case).

Usage::

    build_resnet50_layer4_step.py OUT_DIR SCOPE BATCH

where `SCOPE` is `layer4_1_2` or `layer4_all`.
"""

from __future__ import annotations

import argparse
import os
import sys

import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_resident_train_step as brts  # noqa: E402
import build_resnet50_batch_step as b50  # noqa: E402

SCOPES = {
    "layer4_1_2": [
        "layer4.1.conv1.weight",
        "layer4.1.conv2.weight",
        "layer4.1.conv3.weight",
        "layer4.2.conv1.weight",
        "layer4.2.conv2.weight",
        "layer4.2.conv3.weight",
        "fc.weight",
    ],
    "layer4_all": [
        "layer4.0.conv1.weight",
        "layer4.0.conv2.weight",
        "layer4.0.conv3.weight",
        "layer4.1.conv1.weight",
        "layer4.1.conv2.weight",
        "layer4.1.conv3.weight",
        "layer4.2.conv1.weight",
        "layer4.2.conv2.weight",
        "layer4.2.conv3.weight",
        "fc.weight",
    ],
}


def build_step(out_dir: str, scope: str, batch: int):
    """Same pipeline as `build_resnet50_batch_step.build_step`, generalized
    over `SCOPES[scope]` instead of that module's fixed `TRAIN_PARAMS`.

    Note `layer4.0` has its own `downsample` conv (the residual shortcut,
    stride-2 1x1) that is deliberately *not* included in `layer4_all` --
    only the three blocks' own main-path convs plus `fc.weight`, matching
    `layer4.2`'s original scope choice (main path only, no shortcut convs).
    """
    os.makedirs(out_dir, exist_ok=True)
    fwd_path = os.path.join(out_dir, "resnet50d_fwd.onnx")
    if not os.path.exists(fwd_path):
        b50.export_forward(fwd_path)

    fwd = onnx.load(fwd_path)
    fwd = b50._downgrade_reduce_axes_to_attr(fwd)
    fwd = b50._fix_flatten_reshape(fwd)
    onnx.checker.check_model(fwd)
    fwd = onnx.shape_inference.infer_shapes(fwd)
    fwd = brts._fold_constants(fwd)

    params = SCOPES[scope]
    init_names = {t.name for t in fwd.graph.initializer}
    missing = [p for p in params if p not in init_names]
    if missing:
        raise RuntimeError(f"trainable params not found after folding: {missing}")

    fwd_b = brts.set_batch(fwd, batch)
    fwd_b = onnx.shape_inference.infer_shapes(fwd_b)

    with_loss = brts.add_mse_loss(fwd_b, "logits", num_classes=1000)
    step_model, state = brts.build_resident_step(with_loss, params=params)
    onnx.checker.check_model(step_model)

    step_path = os.path.join(out_dir, f"resnet50_step_{scope}_b{batch}.onnx")
    onnx.save(step_model, step_path)
    return step_path, state, params


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("scope", choices=sorted(SCOPES))
    parser.add_argument("batch", type=int)
    args = parser.parse_args(argv)

    step_path, state, params = build_step(args.out_dir, args.scope, args.batch)
    model = onnx.load(step_path)
    print(
        f"scope={args.scope} batch={args.batch}: {len(params)} trainable tensors, "
        f"{len(model.graph.node)} nodes, wrote {step_path}"
    )
    for p, out_name in state.items():
        print(f"  state: {p} -> {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
