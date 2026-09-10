#!/usr/bin/env python3
"""Replay what the AX650N will compute, from the build's own quantisation table.

`pulsar2 build` writes `<output>/quant/quant_axmodel.json`, and it contains
every number needed to reproduce the card exactly: one entry per tensor giving
its bit width, quantisation policy, and -- through a hash into a shared
`values` table -- its scale and zero point.

Feeding those scales back into the float graph as `QuantizeLinear` /
`DequantizeLinear` pairs and running it under onnxruntime reproduces real
AX650N output to a fraction of a decibel. On four builds of the same graph
(INT8 and 16-bit activations, with and without `precision.py`'s weight split)
the replay landed within **0.19 dB** of the card, two of them exactly:

| build | replayed | device |
| --- | --- | --- |
| INT8 | 21.64 dB | 21.64 dB |
| INT8, split weights | 21.15 dB | 21.34 dB |
| U16 | 25.77 dB | 25.77 dB |
| U16, split weights | 33.03 dB | 33.00 dB |

Which settles something this repository had left open: there is no unexplained
numerical floor in the hardware. The card computes exactly the affine
quantisation the compiler wrote down, so **accuracy on this NPU is set by the
scales, and the scales are set by calibration** -- both of which can now be
searched offline, without a card and without a rebuild.

The scales come from the build, so this needs a `pulsar2 build` output
directory. It does not need Docker or a device.

Usage::

    replay.py model.onnx output_dir/ --input x=input.npy --ref
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper

#: activation widths the table uses; 32 means the tensor was left in float.
_ACTIVATION_BITS = (8, 16)


#: Ax op prefixes/names whose output the hardware actually rounds. Everything
#: else in `quant_axmodel.onnx` -- `AxReshape`, `AxSlice`, `AxTranspose`,
#: `AxTile`, `AxPad` -- moves codes around without re-quantising them.
_QUANTISING = ("AxQuantized", "AxQuantizeLinear", "AxRequantizeLinear")


def surviving_edges(build_dir, quantising_only=False):
    """Tensor names the card actually rounds, from the compiler's own graph.

    `quant/quant_axmodel.onnx` is what Pulsar2 lowers -- its `AxQuantized*`
    ops, fusion applied. Two kinds of tensor in `quant_axmodel.json` are not
    rounded on the card and must not be rounded in a replay:

    * **fused away.** 385 of the Audio8 decoder's 1002 tensors are simply not
      in that graph, folded inside an `AxQuantizedRMSNorm` or
      `AxQuantizedRoPE`. Rounding them read 1.12 dB against a measured 7.37.
    * **moved, not computed** (`quantising_only`, off by default). `AxReshape`,
      `AxSlice`, `AxTranspose`, `AxTile` and `AxPad` carry codes without
      recomputing them, so arguably nothing rounds their output either. This
      is *not* the default because on the decoder it overshoots as badly as
      the first rule undershoots: 23.40 dB against the same measured 7.37,
      and the two rules bracket it rather than settling it. Ten `Reshape`
      outputs alone are worth 4.6 dB (3.65 -> 8.23), so a movement op's output
      is clearly rounded *sometimes*. Which is unresolved; the default stays
      on the pessimistic side, where the answer is a lower bound rather than a
      flattering guess.

    Returns `None` if the file is absent, meaning "no filter".
    """
    path = os.path.join(build_dir, "quant", "quant_axmodel.onnx")
    if not os.path.exists(path):
        return None
    graph = onnx.load(path, load_external_data=False).graph
    return {
        o
        for node in graph.node
        for o in node.output
        if not quantising_only or node.op_type.startswith(_QUANTISING)
    } | {i.name for i in graph.input}


def load_scales(build_dir, fused_aware=True, quantising_only=False):
    """`{tensor: (bit_width, scale, zero_point, per_channel, axis)}` for a build.

    `quant_axmodel.json` stores one config per (op, tensor) pair and hashes
    into a shared `values` table, so a tensor consumed by several ops appears
    several times; the entries agree, and the first one carrying a value wins.

    `fused_aware` drops the tensors `surviving_edges` says the compiler fused
    away. Weights are always kept -- fusion does not stop a weight being
    quantised.
    """
    path = os.path.join(build_dir, "quant", "quant_axmodel.json")
    with open(path) as fh:
        doc = json.load(fh)
    keep = surviving_edges(build_dir, quantising_only) if fused_aware else None
    values = doc["values"]
    out = {}
    for cfg in doc["tensor_configs"].values():
        for tensor, entry in cfg.items():
            if tensor in out:
                continue
            value = values.get(str(entry.get("hash")))
            if not value or not value.get("scale"):
                continue
            policy = entry["policy"]
            if (
                keep is not None
                and tensor not in keep
                and not policy.get("PER_CHANNEL")
            ):
                continue
            out[tensor] = (
                entry["bit_width"],
                np.asarray(value["scale"], dtype=np.float32),
                np.asarray(value["zero_point"], dtype=np.float64),
                bool(policy.get("PER_CHANNEL")),
                None,
            )
    return out


def _weight_axis(node, ndim):
    from precision import channel_axis

    return channel_axis(node, ndim)


def insert_qdq(model, scales):
    """Return a copy of `model` with the build's quantisation spelled out.

    Every tensor the table knows about gets a `QuantizeLinear` /
    `DequantizeLinear` pair at its producer: activations per tensor and
    asymmetric, weights per output channel and symmetric. Tensors the table
    does not mention (shape operands, anything the compiler kept in float)
    are left alone.

    Returns `(model, n_activations, n_weights)`.
    """
    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    used = {n.name for n in graph.node if n.name}
    n_act = n_w = 0

    def const(name, arr):
        graph.initializer.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def unique(stem):
        name, k = stem, 0
        while name in used:
            k += 1
            name = f"{stem}_{k}"
        used.add(name)
        return name

    # weights first: they are initializers, so quantise the stored values
    # directly rather than adding nodes the compiler would not have.
    for node in graph.node:
        if len(node.input) < 2:
            continue
        init = inits.get(node.input[1])
        entry = scales.get(node.input[1])
        if init is None or entry is None:
            continue
        bits, scale, zero, per_channel, _ = entry
        w = numpy_helper.to_array(init).astype(np.float32)
        axis = _weight_axis(node, w.ndim) if per_channel else None
        s = (
            scale.reshape([-1] + [1] * (w.ndim - 1 - (axis or 0)))
            if per_channel
            else scale.reshape(())
        )
        if per_channel and axis:
            s = scale.reshape([1] * axis + [-1] + [1] * (w.ndim - axis - 1))
        z = zero.reshape(s.shape) if per_channel else zero.reshape(())
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        q = np.clip(np.rint(w / s) + z, lo, hi)
        init.CopyFrom(
            numpy_helper.from_array(((q - z) * s).astype(np.float32), init.name)
        )
        n_w += 1

    # then activations. Spelled as arithmetic (`Div`, `Round`, `Clip`, `Mul`)
    # rather than `QuantizeLinear`, because a UINT16 zero point needs opset 21
    # and bumping a real graph's opset that far breaks ops whose signature
    # moved on the way (`ReduceMean`'s `axes` became an input at 18). The
    # arithmetic form runs at whatever opset the model already declares, and
    # ONNX `Round` is round-half-to-even, the same rule `QuantizeLinear` uses.
    def fake_quant(raw, out, bits, scale, zero):
        s_name = const(unique(f"{out}_s"), np.float32(scale))
        z_name = const(unique(f"{out}_z"), np.float32(zero))
        lo_name = const(unique(f"{out}_lo"), np.float32(0.0))
        hi_name = const(unique(f"{out}_hi"), np.float32(2**bits - 1))
        t = [unique(f"{out}_t{i}") for i in range(4)]
        return [
            helper.make_node("Div", [raw, s_name], [t[0]], name=unique(f"{out}_Q0")),
            helper.make_node("Round", [t[0]], [t[1]], name=unique(f"{out}_Q1")),
            helper.make_node("Add", [t[1], z_name], [t[2]], name=unique(f"{out}_Q2")),
            helper.make_node(
                "Clip", [t[2], lo_name, hi_name], [t[3]], name=unique(f"{out}_Q3")
            ),
            helper.make_node(
                "Sub", [t[3], z_name], [t[0] + "_d"], name=unique(f"{out}_Q4")
            ),
            helper.make_node(
                "Mul", [t[0] + "_d", s_name], [out], name=unique(f"{out}_Q5")
            ),
        ]

    new_nodes = []
    for node in graph.node:
        new_nodes.append(node)
        for i, out in enumerate(node.output):
            entry = scales.get(out)
            if entry is None or out in inits:
                continue
            bits, scale, zero, per_channel, _ = entry
            if per_channel or bits not in _ACTIVATION_BITS:
                continue
            raw = unique(f"{out}_pre_q")
            node.output[i] = raw
            new_nodes.extend(
                fake_quant(
                    raw, out, bits, float(scale.reshape(())), float(zero.reshape(()))
                )
            )
            n_act += 1
    del graph.node[:]
    graph.node.extend(new_nodes)

    # the graph's own inputs are quantised too, and have no producer to hang a
    # pair off, so give them one at the front.
    front = []
    for inp in graph.input:
        entry = scales.get(inp.name)
        if entry is None:
            continue
        bits, scale, zero, per_channel, _ = entry
        if per_channel or bits not in _ACTIVATION_BITS:
            continue
        renamed = unique(f"{inp.name}_in")
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name == inp.name:
                    node.input[i] = renamed
        front.extend(
            fake_quant(
                inp.name,
                renamed,
                bits,
                float(scale.reshape(())),
                float(zero.reshape(())),
            )
        )
        n_act += 1
    if front:
        nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(front + nodes)
    model.ir_version = max(model.ir_version, 7)
    return model, n_act, n_w


def replay(model, build_dir, feeds, fused_aware=True, quantising_only=False):
    """Run `model` as the card will run it. Returns the graph's outputs."""
    import onnxruntime as ort

    scales = load_scales(
        build_dir, fused_aware=fused_aware, quantising_only=quantising_only
    )
    quantised, _, _ = insert_qdq(model, scales)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantised.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def snr_db(ref, got):
    ref = np.asarray(ref, np.float64).ravel()
    got = np.asarray(got, np.float64).ravel()
    n = min(ref.size, got.size)
    err = ref[:n] - got[:n]
    return float(10 * np.log10((ref[:n] ** 2).sum() / max((err**2).sum(), 1e-30)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model")
    p.add_argument("build_dir")
    p.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=FILE.npy",
        help="repeatable",
    )
    p.add_argument(
        "--ref", action="store_true", help="also run the float model and report the SNR"
    )
    args = p.parse_args(argv)

    model = onnx.load(args.model)
    feeds = {}
    for spec in args.input:
        name, _, path = spec.partition("=")
        feeds[name] = np.load(path)
    if not feeds:
        for inp in model.graph.input:
            shape = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]
            feeds[inp.name] = (
                np.random.default_rng(0).standard_normal(shape).astype(np.float32)
            )

    got = replay(model, args.build_dir, feeds)
    if args.ref:
        import onnxruntime as ort

        ref = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"]).run(
            None, feeds
        )
        for i, (r, g) in enumerate(zip(ref, got)):
            print(f"output {i}: {snr_db(r, g):.2f} dB")
    else:
        for i, g in enumerate(got):
            print(
                f"output {i}: shape {np.shape(g)} "
                f"rms {float(np.sqrt((np.asarray(g, np.float64) ** 2).mean())):.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
