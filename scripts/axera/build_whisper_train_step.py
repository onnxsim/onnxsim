#!/usr/bin/env python3
"""Build resident training-step graphs for a real Whisper-base encoder --
the memory-heavy case in `docs/axera-on-device-training-handoff.md`'s "A
memory-heavy case" section, chosen specifically to put real pressure on
device memory (every earlier case in that document -- resnet18d, resnet50d,
both 64x64 with a handful of trainable convs -- was sized to stay clear of
Pulsar2's compile-time wall, and consequently never used more than 6.1% of
the card's 7040 MiB CMM either).

Requires `transformers`/`torch` (export only, CPU, random init -- no
pretrained weights, no GPU). Everything downstream of the export is plain
`onnx`/`onnxsim`, no torch needed.

Usage::

    build_whisper_train_step.py OUT_DIR

writes `OUT_DIR/whisper_base_enc.onnx` (the raw export) and, for each of
three trainable scopes, `OUT_DIR/whisper_step_<scope>.onnx` plus a
`<scope>.params.txt` (the trainable tensor names) and a
`<scope>.state_map.txt` (`name\\tnext_name` per line, `qat_graph.StepGraph
.state` written out) -- pass `whisper_step_<scope>.onnx` to
`scripts/axera/make_training_calib.py` and then `pulsar2_docker.build()` to
compile.

Trainable scopes, all a *suffix* of the encoder's own float32 initializers in
first-use (= layer) order -- not a name-based layer selector, since the
exporter only keeps meaningful names for `layers.0`'s own tensors and every
other layer's weights get generic `onnx::MatMul_NNN` names:

- `last_half`: roughly the last 3 of 6 encoder layers, 11,534,336 params.
  Compiles and runs; see the handoff doc for the real 142.4 MiB CMM number.
- `full_encoder`: every non-stem tensor, 20,431,360 params (99.2% of the
  encoder). Builds and verifies correctly on host, but **does not compile**:
  Pulsar2's frontend hard-errors on the one LayerNorm-affine tensor CSE
  shares across all 13 `LayerNormalization` nodes (PyTorch's default init
  makes them bit-identical) once it becomes a live/state input -- see the
  handoff doc for the exact `KeyError` and why it's a real legalization gap,
  not a workaround-able quirk of this script.
- `full_encoder_no_ln`: `full_encoder` minus that one shared LayerNorm-affine
  tensor (20,430,848 params, 99.998% of the same trainable weight). Gets
  past the frontend error but hits a *different* wall inside Pulsar2's NPU
  backend tiler (`TileFailException`, internal to the closed-source
  scheduler) -- also documented, not chased further.
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
from _local_import import fresh  # noqa: E402

# scripts/axera/legalize.py and scripts/axelera/legalize.py are two
# different, same-named modules sharing one `sys.modules["legalize"]` entry
# -- a plain `import legalize` risks silently getting axelera's copy if
# something upstream already claimed that bare name. `fresh` reloads
# directly from this file's own directory regardless of what's cached.
legalize = fresh("legalize", HERE)

_STEM = {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"}


def export_encoder(out_path: str) -> None:
    """Writes a real, unmodified `openai/whisper-base`-sized encoder to
    `out_path`: `d_model=512`, 6 layers, 8 heads, `ffn_dim=2048`, and its
    real 3000-mel-frame / 1500-position input size -- not the shrunk
    `max_source_positions` `docs/axera-audio-speech-op-coverage.md`'s
    reproduction snippet uses to keep its coverage-survey export small.
    """
    import torch
    from transformers import WhisperConfig, WhisperModel

    cfg = WhisperConfig(
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=6,
        encoder_attention_heads=8,
        d_model=512,
        encoder_ffn_dim=2048,
        decoder_layers=1,
        decoder_attention_heads=1,
    )
    enc = WhisperModel(cfg).eval().get_encoder()
    x = torch.randn(1, 80, 3000)
    torch.onnx.export(enc, (x,), out_path, opset_version=17, dynamo=False)


def add_loss_3d(model: onnx.ModelProto, logits: str) -> onnx.ModelProto:
    """MSE loss for a rank-3 `[batch, seq, dim]` logits tensor, reduced to a
    true scalar with **explicit** axes (never omit them -- the AX650's bare
    `ReduceMean` silently reduces only the last axis, `docs/axera-on-device-
    training-handoff.md`'s "Two vendor bugs" section), mirroring
    `build_resident_train_step.add_mse_loss`'s 2-D version.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    logits_shape = [int(d) for d in legalize._value_shapes(model)[logits]]
    g.input.append(helper.make_tensor_value_info("y", TensorProto.FLOAT, logits_shape))
    g.node.extend(
        [
            helper.make_node("Sub", [logits, "y"], ["loss_diff"], name="loss_diff"),
            helper.make_node(
                "Mul", ["loss_diff", "loss_diff"], ["loss_sq"], name="loss_sq"
            ),
            helper.make_node(
                "ReduceMean",
                ["loss_sq"],
                ["loss"],
                name="loss_mean",
                axes=[0, 1, 2],
                keepdims=0,
            ),
        ]
    )
    g.output.append(helper.make_tensor_value_info("loss", TensorProto.FLOAT, []))
    return out


def trainable_scopes(fwd: onnx.ModelProto) -> dict:
    """`{scope_name: [trainable tensor names]}` for `fwd` (already folded --
    see `build_resident_train_step._fold_constants`), the three scopes this
    module's docstring describes.
    """
    # float32, rank >= 1 only: build_resident_step trains initializers, and
    # a few (the fused-QKV Split's int64 sizes input; the GELU/attention-
    # scale scalars _fold_constants turns into rank-0 initializers) are not
    # weights -- a real weight always has rank >= 1.
    float_inits = {
        i.name
        for i in fwd.graph.initializer
        if i.data_type == TensorProto.FLOAT and len(i.dims) >= 1
    }
    seen, seenset = [], set()
    for n in fwd.graph.node:
        for inp in n.input:
            if inp in float_inits and inp not in seenset:
                seenset.add(inp)
                seen.append(inp)
    non_stem = [n for n in seen if n not in _STEM]

    ln_affine = set()
    for n in fwd.graph.node:
        if n.op_type == "LayerNormalization":
            ln_affine.add(n.input[1])
            if len(n.input) > 2:
                ln_affine.add(n.input[2])

    return {
        "last_half": non_stem[len(non_stem) // 2 :],
        "full_encoder": non_stem,
        "full_encoder_no_ln": [p for p in non_stem if p not in ln_affine],
    }


def main(argv=None) -> int:
    parser_ = argparse.ArgumentParser(description=__doc__)
    parser_.add_argument("out_dir")
    args = parser_.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    enc_path = os.path.join(args.out_dir, "whisper_base_enc.onnx")
    export_encoder(enc_path)

    model = onnx.shape_inference.infer_shapes(onnx.load(enc_path))
    fwd = add_loss_3d(model, model.graph.output[0].name)
    fwd = brts._fold_constants(fwd)

    scopes = trainable_scopes(fwd)
    initializers = {t.name: t for t in fwd.graph.initializer}

    for name, params in scopes.items():
        n_params = sum(
            int(np.prod(numpy_helper.to_array(initializers[p]).shape)) for p in params
        )
        print(f"=== {name}: {len(params)} tensors, {n_params:,} trainable params ===")
        step_model, state = brts.build_resident_step(fwd, params, loss_output="loss")
        onnx.checker.check_model(step_model)
        print(
            f"  {len(step_model.graph.node)} nodes, "
            f"{len(step_model.graph.initializer)} initializers"
        )

        onnx.save(step_model, os.path.join(args.out_dir, f"whisper_step_{name}.onnx"))
        with open(os.path.join(args.out_dir, f"{name}.params.txt"), "w") as f:
            f.write("\n".join(params))
        with open(os.path.join(args.out_dir, f"{name}.state_map.txt"), "w") as f:
            f.write("\n".join(f"{k}\t{v}" for k, v in state.items()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
