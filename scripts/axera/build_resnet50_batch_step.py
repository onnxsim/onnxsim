#!/usr/bin/env python3
"""Build a batch-parametric resnet50d `layer4.2` + `fc.weight` resident
training step -- the trainable scope `build_resident_train_step.py`'s own
docstring and this project's handoff doc call "resnet50, first compile"
(6.5M trainable params, 64x64 input, `zero_init_last=False` to avoid the
degenerate all-zero-`conv3` CSE collision that default init produces).

Every earlier compile of this shape (the first-compile session, and the
batching/vNPU sweeps on resnet18) grew its own uncommitted scratchpad copy
of this export -- this is the first committed version, written while closing
the "resnet50 never had batch>1 tested" gap
(`docs/axera-on-device-training-handoff.md`'s "resnet50 batch scaling"
section has the real batch 1/4/8 sweep this produced).

Two export gotchas fixed here, neither specific to batch scaling and both
likely to recur on any future `timm`-exported model this pipeline touches:

- `torch.onnx.export(..., dynamo=False)` (the legacy TorchScript exporter)
  loses real per-layer names for every non-stem tensor of a plain
  `nn.Sequential`-of-blocks model like `resnet50d` -- confirmed directly
  here, the same naming loss `build_whisper_train_step.py`'s docstring
  documents for Whisper's encoder layers. `dynamo=True` (torch.export-based)
  keeps real names, at the cost of landing on **opset 18** regardless of the
  requested `opset_version` -- `_downgrade_reduce_axes_to_attr` below
  reverses the one opset-18-only construct (`ReduceMean` with axes as an
  *input* rather than an attribute) this pipeline can't carry through
  `onnxsim.qat_graph.make_step_graph`, which always declares its own output
  model at a fixed opset 17.
- `timm`'s `SelectAdaptivePool2d.flatten` traces to a `Reshape` whose target
  shape is a **batch-1-literal** constant (`[1, 2048]`) -- `set_batch`, by
  its own docstring, only rewrites the declared *input* shape, not every
  batch-shaped constant an exporter baked in downstream.
  `_fix_flatten_reshape` rewrites that one Reshape's target to `[-1, 2048]`
  (the standard ONNX "infer this dim" sentinel) so it works at any batch --
  the same *class* of fix `build_w2v2_feature_extractor_step.py` needed for
  wav2vec2's own batch-dependent flatten shape (PR #1372), on a different
  architecture.

Usage::

    build_resnet50_batch_step.py OUT_DIR BATCH

writes `OUT_DIR/resnet50d_fwd.onnx` (the raw dynamo export, batch-1, shared
across every batch size) and `OUT_DIR/resnet50_step_b<BATCH>.onnx` (the step
graph) -- pass the latter to `make_training_calib.make_work_dir` and then
`pulsar2_docker.build()` to compile.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_resident_train_step as brts  # noqa: E402
from _local_import import ensure_repo_onnxsim, fresh  # noqa: E402

ensure_repo_onnxsim()

# scripts/axera/legalize.py and scripts/axelera/legalize.py are two
# different, same-named modules sharing one `sys.modules["legalize"]` entry
# -- a plain `import legalize` risks silently getting axelera's copy if
# something upstream already claimed that bare name. `fresh` reloads
# directly from this file's own directory regardless of what's cached.
legalize = fresh("legalize", HERE)

TRAIN_PARAMS = [
    "layer4.2.conv1.weight",
    "layer4.2.conv2.weight",
    "layer4.2.conv3.weight",
    "fc.weight",
]


def export_forward(onnx_path: str) -> None:
    """Writes a real, unmodified `resnet50d` (random init, 64x64 input) to
    `onnx_path` via the dynamo exporter -- see this module's docstring for
    why `dynamo=True` is required here (real per-layer names) despite its
    opset-18 side effect."""
    import timm
    import torch

    torch.manual_seed(0)
    model = timm.create_model("resnet50d", pretrained=False, zero_init_last=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        (x,),
        onnx_path,
        opset_version=18,
        dynamo=True,
        input_names=["x"],
        output_names=["logits"],
    )


def _downgrade_reduce_axes_to_attr(model: onnx.ModelProto) -> onnx.ModelProto:
    """Converts every axes-as-*input* `ReduceMean` (opset >= 18's form) back
    to axes-as-*attribute* (pre-18), and drops the declared opset to 17.

    See this module's docstring for why: `onnxsim.qat_graph.make_step_graph`
    always declares its own output model at a fixed opset 17
    (`qat_graph._OPSET`) while copying node protos through verbatim, so an
    axes-as-input `ReduceMean` surviving from an opset-18 forward export
    into the step graph fails the opset-17 schema (confirmed directly:
    `Node with schema ReduceMean:13 has input size 2 not in range [min=1,
    max=1]`). `legalize._set_axes` exists for exactly this choice but keys
    off the *current* model's declared opset (still 18 here), so it does
    the opposite of what this needs -- hence the hand-rolled downgrade,
    which must run before any pipeline code that assumes attribute-form
    axes (i.e. before `set_batch`/`add_mse_loss`/`build_resident_step`).
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializers = {t.name: t for t in out.graph.initializer}
    for node in out.graph.node:
        if node.op_type != "ReduceMean" or len(node.input) < 2:
            continue
        axes_name = node.input[1]
        if not axes_name:
            continue
        axes = numpy_helper.to_array(initializers[axes_name]).tolist()
        del node.input[1:]
        keep = [a for a in node.attribute if a.name != "noop_with_empty_axes"]
        del node.attribute[:]
        node.attribute.extend(keep)
        node.attribute.append(onnx.helper.make_attribute("axes", list(axes)))
    for opset in out.opset_import:
        if not opset.domain:
            opset.version = 17
    return out


def _fix_flatten_reshape(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrites the one `Reshape` whose target shape is a rank-2, batch-1-
    literal constant (`timm`'s global-pool flatten, `[1, C]`) to `[-1, C]`
    -- see this module's docstring. Must run before `set_batch`, since it
    patches a specific node by the batch-1 shape it currently has baked in.
    Raises if the graph doesn't have exactly one such node, since a silent
    no-op here would leave a batch>1 build failing later with a confusing
    onnxruntime/Pulsar2 shape-mismatch error instead of this one, clearer.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializers = {t.name: t for t in out.graph.initializer}
    fixed = 0
    for node in out.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_init = initializers.get(node.input[1])
        if shape_init is None:
            continue
        dims = numpy_helper.to_array(shape_init).tolist()
        if len(dims) == 2 and dims[0] == 1:
            shape_init.CopyFrom(
                numpy_helper.from_array(np.array([-1, dims[1]], dtype=np.int64), shape_init.name)
            )
            fixed += 1
    if fixed != 1:
        raise RuntimeError(f"expected exactly one flatten reshape to fix, patched {fixed}")
    return out


def build_step(out_dir: str, batch: int):
    """Returns `(step_path, state)` -- `state` maps each of `TRAIN_PARAMS`
    to its step graph's updated-value output name, same convention as
    `build_resident_train_step.build_resident_step`."""
    os.makedirs(out_dir, exist_ok=True)
    fwd_path = os.path.join(out_dir, "resnet50d_fwd.onnx")
    if not os.path.exists(fwd_path):
        export_forward(fwd_path)

    fwd = onnx.load(fwd_path)
    fwd = _downgrade_reduce_axes_to_attr(fwd)
    fwd = _fix_flatten_reshape(fwd)
    onnx.checker.check_model(fwd)
    fwd = onnx.shape_inference.infer_shapes(fwd)
    fwd = brts._fold_constants(fwd)

    init_names = {t.name for t in fwd.graph.initializer}
    missing = [p for p in TRAIN_PARAMS if p not in init_names]
    if missing:
        raise RuntimeError(f"trainable params not found after folding: {missing}")

    fwd_b = brts.set_batch(fwd, batch)
    fwd_b = onnx.shape_inference.infer_shapes(fwd_b)

    with_loss = brts.add_mse_loss(fwd_b, "logits", num_classes=1000)
    step_model, state = brts.build_resident_step(with_loss, params=TRAIN_PARAMS)
    onnx.checker.check_model(step_model)

    step_path = os.path.join(out_dir, f"resnet50_step_b{batch}.onnx")
    onnx.save(step_model, step_path)
    return step_path, state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("batch", type=int)
    args = parser.parse_args(argv)

    step_path, state = build_step(args.out_dir, args.batch)
    model = onnx.load(step_path)
    print(f"batch={args.batch}: {len(model.graph.node)} nodes, wrote {step_path}")
    for p, out_name in state.items():
        print(f"  state: {p} -> {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
