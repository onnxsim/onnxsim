#!/usr/bin/env python3
"""Build the wav2vec2 CNN feature extractor's training-step graph -- the
smaller, shallower audio target `docs/axera-audio-speech-op-coverage.md`'s
"A smaller target than Whisper" section surveyed and this script actually
compiles.

`Wav2Vec2Model(Wav2Vec2Config()).feature_extractor` (7 Conv1D layers, 4.2M
params, no attention/LayerNorm chain) exports to `{Add, Constant, Conv, Div,
Erf, InstanceNormalization, Mul, Reshape, Shape, Unsqueeze}` -- every op
except `Unsqueeze` already has a `graph_grad` gradient rule. `Unsqueeze`
only appears on the non-trainable raw-waveform input's own path (adding the
channel axis), so it is rewritten to an equivalent `Reshape` before
`build_resident_step` ever sees it -- the same trick `legalize.py`'s
`flatten_to_reshape` already uses for the analogous case, not new gradient
machinery.

`fe.conv_layers.0.conv.weight` (not a later layer) is the trainable weight:
`build_resident_step`'s own `onnxsim.simplify()` pass renames later
layers' weight initializers via CSE (e.g. `fe.conv_layers.6.conv.weight`
-> `_v_100`) while layer 0's name survives, so picking layer 0 sidesteps
that renaming churn rather than fighting it.

Usage::

    build_w2v2_feature_extractor_step.py out/step.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_resident_train_step as brts  # noqa: E402


def _export_feature_extractor(onnx_path: str, batch: int = 1) -> None:
    """Exports with a *static* leading dim of `batch` -- not a dynamic axis.

    Every trainable-weight training-step graph in this pipeline (resnet18,
    Whisper) is built from a static-batch export and gets its batch scaled
    later via `build_resident_train_step.set_batch()`. That trick doesn't
    apply cleanly here: `build()` below bakes a `flatten_shape` initializer
    from the *exported* batch dim (the CNN feature extractor's per-sample
    output length depends on the input's own static shape through several
    strided Conv1D layers, unlike resnet18's pooling-then-Gemm tail, which
    is batch-shape-agnostic downstream of `x`). So batch is a real *export*
    parameter here, not a post-hoc graph edit -- see `build`'s own docstring.
    """
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    fe = Wav2Vec2Model(_default_config()).feature_extractor.eval()

    class Wrapped(nn.Module):
        def __init__(self, fe):
            super().__init__()
            self.fe = fe

        def forward(self, x):
            return self.fe(x)

    x = torch.randn(batch, 4000)
    torch.onnx.export(
        Wrapped(fe),
        (x,),
        onnx_path,
        input_names=["x"],
        output_names=["feat"],
        opset_version=17,
        dynamo=False,
    )


def _default_config():
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config()


def _unsqueeze_to_reshape(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replaces the one `Unsqueeze` on `x`'s own path with an equivalent
    `Reshape` -- `graph_grad` has no gradient rule for `Unsqueeze`, but
    `build_backward` demands one for every node it walks regardless of
    whether a gradient actually flows through it (`onnxsim.qat_graph`'s own
    docstring), so it must go before `build_resident_step` runs.
    """
    g = model.graph
    (idx, node) = next((i, n) for i, n in enumerate(g.node) if n.op_type == "Unsqueeze")
    x_info = next(inp for inp in g.input if inp.name == node.input[0])
    x_shape = [d.dim_value for d in x_info.type.tensor_type.shape.dim]
    new_shape = [x_shape[0], 1, x_shape[1]]
    shape_name = "unsqueeze_to_reshape_shape"
    g.initializer.append(
        numpy_helper.from_array(np.array(new_shape, dtype=np.int64), shape_name)
    )
    new_node = helper.make_node(
        "Reshape", [node.input[0], shape_name], list(node.output), name="x_reshape"
    )
    del g.node[idx]
    g.node.insert(idx, new_node)
    onnx.checker.check_model(model)
    return model


def build(out_path: str, param: str = "fe.conv_layers.0.conv.weight", batch: int = 1):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        _export_feature_extractor(f.name, batch=batch)
        model = onnx.load(f.name)

    model = _unsqueeze_to_reshape(model)
    model = onnx.shape_inference.infer_shapes(model)

    feat_shape = next(
        [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in list(model.graph.value_info) + list(model.graph.output)
        if vi.name == "feat"
    )
    batch, ch, seq = feat_shape
    flat_dim = ch * seq
    flat_shape_name = "flatten_shape"
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([batch, flat_dim], dtype=np.int64), flat_shape_name
        )
    )
    model.graph.node.append(
        helper.make_node(
            "Reshape", ["feat", flat_shape_name], ["logits"], name="flatten_feat"
        )
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, flat_dim])
    )
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)

    with_loss = brts.add_mse_loss(model, "logits", num_classes=flat_dim)
    step_model, state = brts.build_resident_step(with_loss, params=[param])
    onnx.checker.check_model(step_model)
    onnx.save(step_model, out_path)
    return step_model, state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_path")
    p.add_argument("--param", default="fe.conv_layers.0.conv.weight")
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args(argv)
    step_model, state = build(args.out_path, args.param, batch=args.batch)
    print(f"wrote {args.out_path}: {len(step_model.graph.node)} nodes, state={state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
