#!/usr/bin/env python3
"""Build a resident training-step graph for a real super-resolution model --
EDSR ("Enhanced Deep Residual Networks for Single Image Super-Resolution"),
the first architecture surveyed in this domain
(`docs/axera-super-resolution-op-coverage.md`).

Requires `super-image` (`pip install super-image`; pulls in `torch`,
`torchvision`, `opencv-python`, `h5py`, `huggingface-hub` -- export only, CPU,
random init, no pretrained weights or dataset needed). Everything downstream
of the export is plain `onnx`/`onnxsim`, no `super-image` needed.

EDSR's real architecture (`super_image.models.edsr.modeling_edsr.EdsrModel`):
a head `Conv`, `n_resblocks` residual blocks (each two `Conv`s + `ReLU`,
`res = body(x); res += x`), a tail `Upsampler` (`Conv` widening channels by
`scale**2`, then `nn.PixelShuffle(scale)` -- ONNX's `DepthToSpace`, `mode=
"CRD"`) and a final `Conv` back to 3 color channels. Every op except
`DepthToSpace` was already covered on both axes (a `graph_grad` backward
rule, NPU support) before this investigation; `DepthToSpace` is now covered
too (`onnxsim.graph_grad._grad_depth_to_space`), closing the one real gap
this domain's own coverage survey found.

Usage::

    build_edsr_train_step.py OUT_DIR

writes `OUT_DIR/edsr_tiny.onnx` (the raw export) and
`OUT_DIR/edsr_step.onnx` (the resident training step) plus
`OUT_DIR/edsr_step.params.txt` (the trainable tensor names) and
`OUT_DIR/edsr_step.state_map.txt` (`name\\tnext_name` per line) -- pass
`edsr_step.onnx` to `scripts/axera/make_training_calib.py` and then
`pulsar2_docker.build()` to compile.
"""

from __future__ import annotations

import argparse
import os
import sys

import onnx
from onnx import TensorProto, helper

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

# Tiny relative to a real deployed EDSR (n_resblocks=16-32, n_feats=64-256
# depending on scale) -- big enough that the residual-block/upsampler
# structure is real and inspectable, small enough to export/compile in well
# under a minute. scale=2 keeps the tail to a single Conv+PixelShuffle pair
# (a power-of-2 scale >2 repeats that pair log2(scale) times, per
# `Upsampler`'s own code).
SCALE = 2
N_RESBLOCKS = 2
N_FEATS = 8
LR_SIZE = 16


def export_edsr(out_path: str) -> None:
    """Writes a real, unmodified (tiny-config) EDSR to `out_path`.

    Seeded (`torch.manual_seed(0)`): a real, found-the-hard-way gap in an
    earlier version of this function -- an unseeded export gives every
    *frozen* (non-trainable) tensor a fresh random value each call, which
    changes the actual gradient magnitude a trainable tensor downstream of
    it sees. Two otherwise-identical real-hardware runs, differing only in
    which unseeded export produced the compiled `.axmodel`, showed real
    training at one and a fully frozen loss at the other -- not a
    calibration or backend bug, just an unseeded random network genuinely
    having different real gradients call to call. Seeding here is what
    makes the real-hardware result in `docs/axera-super-resolution-op-
    coverage.md` reproducible rather than a one-off.
    """
    import torch
    from super_image.models.edsr.configuration_edsr import EdsrConfig
    from super_image.models.edsr.modeling_edsr import EdsrModel

    torch.manual_seed(0)
    cfg = EdsrConfig(scale=SCALE, n_resblocks=N_RESBLOCKS, n_feats=N_FEATS, n_colors=3)
    model = EdsrModel(cfg).eval()
    lowres = torch.randn(1, 3, LR_SIZE, LR_SIZE)
    # Named "lowres", not the natural "lr" -- that name is reserved
    # throughout this pipeline for the learning-rate scalar
    # `build_resident_step()` itself declares as a graph input, and the two
    # collide silently at export time (onnxsim.simplify's own SSA check is
    # what actually catches it, well downstream of this export call).
    torch.onnx.export(
        model,
        (lowres,),
        out_path,
        opset_version=17,
        dynamo=False,
        input_names=["lowres"],
    )


def add_mse_loss_nchw(model: onnx.ModelProto, sr: str) -> onnx.ModelProto:
    """Returns a copy of `model` with an `hr` (high-resolution target) input
    and a scalar MSE `loss` output appended: `loss = mean((sr - hr) ** 2)`.

    `ReduceMean` with **explicit** axes over every axis (`[0, 1, 2, 3]`),
    never a bare reduce-all -- the AX650's own `ReduceMean` silently reduces
    only the last axis (`docs/axera-on-device-training-handoff.md`'s "Two
    vendor bugs" section), the same guard `build_resident_train_step.
    add_mse_loss`'s rank-2 version and `build_whisper_train_step.
    add_loss_3d`'s rank-3 version both already carry, extended here to
    super-resolution's rank-4 `[N, C, H, W]` output.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    sr_shape = [int(d) for d in legalize._value_shapes(model)[sr]]
    g.input.append(helper.make_tensor_value_info("hr", TensorProto.FLOAT, sr_shape))
    g.node.extend(
        [
            helper.make_node("Sub", [sr, "hr"], ["loss_diff"], name="loss_diff"),
            helper.make_node(
                "Mul", ["loss_diff", "loss_diff"], ["loss_sq"], name="loss_sq"
            ),
            helper.make_node(
                "ReduceMean",
                ["loss_sq"],
                ["loss"],
                name="loss_mean",
                axes=[0, 1, 2, 3],
                keepdims=0,
            ),
        ]
    )
    g.output.append(helper.make_tensor_value_info("loss", TensorProto.FLOAT, []))
    return out


def trainable_scope(
    fwd: onnx.ModelProto, scope: str = "tail", weights_only: bool = False
) -> list:
    """`{scope}`'s float32 weight/bias initializers, in first-use order --
    `"tail"` (default): the upsampler's widening `Conv` plus the final
    color-channel `Conv`, the tensors closest to the newly-covered
    `DepthToSpace` (a gradient must pass through it to reach the widening
    `Conv`'s weight, the direct exercise of `_grad_depth_to_space` this
    scope is chosen for); `"head"`: the first `Conv`, requiring a gradient
    through every residual block *and* `DepthToSpace` to reach it -- the
    fuller, harder-to-reach scope; `"all"`: every trainable tensor.

    `weights_only` (default `False`) drops every rank-1 (bias) tensor from
    the result. **Originally a real hardware finding, now fixed
    generically, not a style preference to keep**: a resident training
    step's in-graph SGD update (`w_next = w - lr * grad`, an ordinary
    `Sub`) used to crash Pulsar2's own NPU backend tiler on a rank-1
    operand -- confirmed on two different real compiles here (a 3-element
    and a 32-element bias, both `TileFailException("AxQuantizedSub, tuple
    index out of range")`), so size was not the trigger, rank was; every
    earlier domain in this project happened to only train rank>=2 weight
    tensors, which is almost certainly why this was never found before
    EDSR's own survey. **Fixed in
    `build_resident_train_step.build_resident_step()` itself**, which now
    reshapes any rank-1 state tensor's update to rank-2 around the `Sub`
    transparently -- `weights_only=True` is no longer required for a
    real compile, kept only as an option for a smaller/faster scope. See
    `docs/axera-super-resolution-op-coverage.md`'s real-hardware section
    for the confirmed compile/train result with biases included.
    """
    float_inits = {
        i.name
        for i in fwd.graph.initializer
        if i.data_type == TensorProto.FLOAT and len(i.dims) >= 1
    }
    ranks = {i.name: len(i.dims) for i in fwd.graph.initializer}
    seen, seenset = [], set()
    for n in fwd.graph.node:
        for inp in n.input:
            if inp in float_inits and inp not in seenset:
                seenset.add(inp)
                seen.append(inp)

    if scope == "all":
        result = seen
    elif scope == "head":
        result = [p for p in seen if p.startswith("head.")]
    elif scope == "tail":
        result = [p for p in seen if p.startswith("tail.")]
    else:
        raise ValueError(f"unknown scope {scope!r}")
    if weights_only:
        result = [p for p in result if ranks[p] != 1]
    return result


def main(argv=None) -> int:
    parser_ = argparse.ArgumentParser(description=__doc__)
    parser_.add_argument("out_dir")
    parser_.add_argument("--scope", choices=["head", "tail", "all"], default="tail")
    parser_.add_argument(
        "--weights-only",
        action="store_true",
        help=(
            "drop rank-1 (bias) tensors -- required for a real Pulsar2 "
            "compile today, see trainable_scope()'s own docstring"
        ),
    )
    args = parser_.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    enc_path = os.path.join(args.out_dir, "edsr_tiny.onnx")
    export_edsr(enc_path)

    model = onnx.shape_inference.infer_shapes(onnx.load(enc_path))
    sr_name = model.graph.output[0].name
    fwd = add_mse_loss_nchw(model, sr_name)

    params = trainable_scope(fwd, args.scope, weights_only=args.weights_only)
    print(f"=== {args.scope}: {len(params)} tensors ===")
    for p in params:
        print(" ", p)

    step_model, state = brts.build_resident_step(fwd, params, loss_output="loss")
    onnx.checker.check_model(step_model)
    print(
        f"  {len(step_model.graph.node)} nodes, "
        f"{len(step_model.graph.initializer)} initializers"
    )

    onnx.save(step_model, os.path.join(args.out_dir, "edsr_step.onnx"))
    with open(os.path.join(args.out_dir, "edsr_step.params.txt"), "w") as f:
        f.write("\n".join(params))
    with open(os.path.join(args.out_dir, "edsr_step.state_map.txt"), "w") as f:
        f.write("\n".join(f"{k}\t{v}" for k, v in state.items()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
