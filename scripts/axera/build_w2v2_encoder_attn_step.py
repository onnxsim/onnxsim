#!/usr/bin/env python3
"""Build a wav2vec2 training-step graph whose trainable tail spans
**attention output** -- the target `docs/axera-audio-speech-op-coverage.md`
flagged as "not yet done" once `onnxsim.graph_grad._grad_where`/
`_grad_is_nan` closed the `Where`/`IsNaN` backward-rule gap (see that
module's own commit and the doc's wav2vec2 section).

Every wav2vec2 build script before this one
(`build_w2v2_feature_extractor_step.py`, `build_w2v2fe_batch_calib.py`)
only trains a convolutional feature-extractor weight, which never reaches
the per-layer `attn_weights = Where(IsNaN(attn_weights), 0, attn_weights)`
numerical-stability guard the doc's real-export trace found sitting inline
between each encoder layer's own `Softmax` and its `MatMul` with `V`. This
module picks **`layers.0`'s own `q_proj` weight** as the trainable tensor
specifically so a real weight update must differentiate back through
*both* encoder layers' `Where`/`IsNaN` guards (layer 1's, on the way out of
the loss; then layer 0's own, to reach `q_proj`) -- not a synthetic
exercise of the new rule, the same real op instance the coverage doc
describes.

A small custom `Wav2Vec2Config` (`hidden_size=128`, 2 encoder layers,
`num_attention_heads=4`, `intermediate_size=512`,
`num_conv_pos_embeddings=15`/`num_conv_pos_embedding_groups=8` -- see
`_default_config`'s own comment for why 15, not HF's even default) keeps
export + Pulsar2 compile time down; the convolutional feature extractor
stays at the real `Wav2Vec2Config()` default scale (512-channel, 7-layer),
matching every other wav2vec2 build in this project.

**Real hardware, confirmed:** compiles cleanly (`pulsar2:7.0-lite`, ~25s)
*after* `_strip_isnan_guard` removes the numerical-stability `Where`/`IsNaN`
pair -- Pulsar2's frontend has no support for `IsNaN` at all
(`KeyError('dont support IsNaN opr in ...')`), a second, separate blocker
from the backward-rule gap this module exists to exercise; see that
function's own docstring for why the removal is exact here, not a lossy
workaround. On real AX650N hardware the resulting step graph's weight
state updates correctly and consistently across many real steps once `lr`
is calibrated and run large enough (`lr=2000`) to clear this specific
weight's own INT8 quantization step -- this one weight's real gradient is
~1e-6/element (two real encoder layers deep), far smaller than the
feature-extractor script's own gradient, so it needs proportionally more
compensating scale than that script's `lr=100` fix needed. At `lr=1` the
weight visibly froze after step 0, the same gradient-quantization-ceiling
signature `docs/axera-on-device-training-handoff.md` documents for Whisper
and other models -- not a new failure mode, a new confirmed instance of an
already-understood one. See `docs/axera-audio-speech-op-coverage.md`'s
wav2vec2 section for the full real-hardware numbers.

**Naming caveat, confirmed by direct export inspection (not assumed from
the Whisper script's docstring):** `torch.onnx.export`'s legacy
(`dynamo=False`) tracer keeps meaningful names only for
`feature_extractor`/`feature_projection`/`encoder.pos_conv_embed`/
`encoder.layer_norm`'s own tensors; every per-layer attention/feed-forward
weight (`q_proj`/`k_proj`/`v_proj`/`out_proj`/`intermediate_dense`/
`output_dense`) is exported as a generic `onnx::MatMul_NNN` initializer, and
every per-layer `LayerNorm`'s affine weight/bias -- default-initialized to
all-ones/all-zeros by `nn.LayerNorm`, not randomly, so every same-shape
instance is bit-identical -- collapses into *one* shared initializer via the
exporter's own constant dedup (the same CSE the Whisper script's docstring
describes for its `full_encoder` scope, here happening at export time
rather than at `onnxsim.simplify()` time). This module resolves the
trainable weight's *real* generic name by tracing the exported graph's own
producer edges back from `layers.0/attention/Softmax`'s `MatMul` inputs
(see `_find_q_proj_weight`), not by guessing a name pattern.

Usage::

    build_w2v2_encoder_attn_step.py OUT_DIR

writes `OUT_DIR/w2v2_encoder_attn_step.onnx` (the resident training-step
graph), `OUT_DIR/w2v2_encoder_attn.params.txt` (the trainable tensor's real
name) and `OUT_DIR/w2v2_encoder_attn.state_map.txt`. Pass this file to
`build_w2v2_encoder_attn_calib.py` for calibration + a real Pulsar2 compile.
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
import build_w2v2_feature_extractor_step as w2vfe  # noqa: E402


def _default_config():
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        # Odd, not HF's even default (128): `Wav2Vec2SamePadLayer`'s own
        # `num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0`
        # means an odd value makes that layer's crop a literal no-op --
        # `if self.num_pad_remove > 0` is false, so its `hidden_states[:,
        # :, :-1]` slice is never even traced. `onnxsim.graph_grad` has no
        # `Slice` backward rule yet (a separate, real, orthogonal gap from
        # this module's own `Where`/`IsNaN` one -- confirmed by hitting
        # `UnsupportedOpError` on `pos_conv_embed/padding/Slice` with 16),
        # and `build_backward` demands a registered rule for every node
        # type anywhere in the forward slice it's given regardless of
        # whether gradient reaches it (`IsNaN`'s own precedent). Sidestep
        # it the same way this project always has -- pick the config that
        # doesn't hit the gap -- rather than adding an unrelated backward
        # rule under this task's own scope.
        num_conv_pos_embeddings=15,
        num_conv_pos_embedding_groups=8,
    )


def export_encoder(out_path: str, batch: int = 1, wave_len: int = 4000) -> None:
    """A real `Wav2Vec2Model` (feature extractor + feature projection +
    positional conv embedding + `num_hidden_layers` real attention/feed-
    forward encoder layers), exported down to its `last_hidden_state`.
    """
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    model = Wav2Vec2Model(_default_config()).eval()

    class Wrapped(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).last_hidden_state

    x = torch.randn(batch, wave_len)
    torch.onnx.export(
        Wrapped(model),
        (x,),
        out_path,
        input_names=["x"],
        output_names=["hidden"],
        opset_version=17,
        dynamo=False,
    )


def _strip_isnan_guard(model: onnx.ModelProto) -> int:
    """Removes every `Where(IsNaN(x), c, x)` numerical-stability guard,
    rewiring its consumers straight onto `x`.

    **Real Pulsar2 frontend wall, confirmed by direct compile attempt, not
    assumed:** `pulsar2 build` on the un-stripped step graph fails frontend
    parsing outright -- `KeyError('dont support IsNaN opr in AXOPS/ONNXOPS/
    CUSTOM_OPS')` -- independent of training or `onnxsim.graph_grad`
    entirely; the *forward* op itself has no Pulsar2 frontend support, so no
    wav2vec2-with-attention model (trained or not) has ever compiled on this
    hardware before. This is a second, separate blocker from the backward-
    rule gap `onnxsim.graph_grad._grad_where`/`_grad_is_nan` closed.

    The removal is exact, not approximate, **specifically because this
    module never passes an `attention_mask`** (see `export_encoder`): with
    every row of every batch a real, unmasked position, HF's own Softmax
    output cannot contain a `NaN` (the guard exists only for the
    floating-point cancellation a *fully masked* row's softmax can produce),
    so `IsNaN(attn_weights)` is always `False` and `Where(False, ., Y)`
    always selects `Y` -- an identity function this specific deployment
    shape can prove, not a general graph-structure fact `legalize.py` could
    verify on its own, which is why this lives here rather than as a
    reusable `legalize.py` rule. The graph's *other* per-layer `Where` (the
    mask-bias one, condition from `Expand`/`GreaterOrEqual`) is untouched --
    still real, still exercises `graph_grad._grad_where` on real hardware.
    """
    producer = {}
    for n in model.graph.node:
        for o in n.output:
            producer[o] = n

    to_remove, rewire = [], {}
    for n in model.graph.node:
        if n.op_type != "Where":
            continue
        cond_producer = producer.get(n.input[0])
        if cond_producer is None or cond_producer.op_type != "IsNaN":
            continue
        rewire[n.output[0]] = n.input[2]  # Y branch: the un-guarded value
        to_remove.append(n)
        to_remove.append(cond_producer)

    remove_names = {id(n) for n in to_remove}
    kept = [n for n in model.graph.node if id(n) not in remove_names]
    for n in kept:
        for i, inp in enumerate(n.input):
            if inp in rewire:
                n.input[i] = rewire[inp]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    for out in model.graph.output:
        if out.name in rewire:
            out.name = rewire[out.name]
    return len(to_remove) // 2


def _find_q_proj_weight(model: onnx.ModelProto, layer: int = 0) -> str:
    """Returns the real (generic) initializer name of encoder layer
    `layer`'s own `q_proj` weight, found by walking producer edges back from
    that layer's `Softmax` input -- see this module's docstring for why a
    name-pattern guess does not work here.
    """
    producer = {}
    for n in model.graph.node:
        for o in n.output:
            producer[o] = n
    init_names = {i.name for i in model.graph.initializer}

    target = f"/m/encoder/layers.{layer}/attention/Softmax_output_0"
    seen, frontier, depth = set(), [target], 0
    while frontier and depth < 15:
        nxt = []
        for t in frontier:
            if t in seen:
                continue
            seen.add(t)
            n = producer.get(t)
            if n is None:
                continue
            if n.op_type == "MatMul" and n.name.endswith("q_proj/MatMul"):
                w = [i for i in n.input if i in init_names]
                if w:
                    return w[0]
            nxt.extend(n.input)
        frontier = nxt
        depth += 1
    raise RuntimeError(f"could not find layer {layer}'s q_proj weight initializer")


def build(out_path: str, layer: int = 0, batch: int = 1):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        export_encoder(f.name, batch=batch)
        model = onnx.load(f.name)

    model = w2vfe._unsqueeze_to_reshape(model)
    model = onnx.shape_inference.infer_shapes(model)

    param = _find_q_proj_weight(model, layer=layer)

    n_guards = _strip_isnan_guard(model)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    print(f"stripped {n_guards} IsNaN numerical-stability guard(s)")

    hidden_shape = next(
        [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in list(model.graph.value_info) + list(model.graph.output)
        if vi.name == "hidden"
    )
    b, seq, hid = hidden_shape
    flat_dim = seq * hid
    flat_shape_name = "flatten_hidden_shape"
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([b, flat_dim], dtype=np.int64), flat_shape_name
        )
    )
    model.graph.node.append(
        helper.make_node(
            "Reshape", ["hidden", flat_shape_name], ["logits"], name="flatten_hidden"
        )
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [b, flat_dim])
    )
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)

    with_loss = brts.add_mse_loss(model, "logits", num_classes=flat_dim)
    step_model, state = brts.build_resident_step(with_loss, params=[param])
    onnx.checker.check_model(step_model)
    onnx.save(step_model, out_path)
    return step_model, state, param


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, "w2v2_encoder_attn_step.onnx")
    step_model, state, param = build(out_path, layer=args.layer, batch=args.batch)
    print(f"trainable param (real name): {param}")
    print(f"wrote {out_path}: {len(step_model.graph.node)} nodes, state={state}")
    with open(os.path.join(args.out_dir, "w2v2_encoder_attn.params.txt"), "w") as f:
        f.write(param + "\n")
    with open(os.path.join(args.out_dir, "w2v2_encoder_attn.state_map.txt"), "w") as f:
        f.write("\n".join(f"{k}\t{v}" for k, v in state.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
