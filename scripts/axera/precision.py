#!/usr/bin/env python3
"""Weight-residual splitting: two INT8 passes that carry ~16 bits of weight.

A Pulsar2-compiled graph loses accuracy in three separable places:

* ``e_W`` -- the weight is quantised to INT8, **per output channel**,
  symmetric, at ``max|w_c| / 127.5`` (confirmed against a build's own
  ``quant/quant_axmodel.json``, which agrees to the last digit),
* ``e_X`` -- the input activation is quantised per tensor, **asymmetric**
  uint8, at ``(max - min) / 255`` with ``zp = round(-min/scale)``,
* ``e_Y`` -- the INT32 accumulator is requantised on the way out, same rule.

This module attacks ``e_W`` with the Ozaki scheme's shape
(`docs/ozaki-scheme-axera-handoff.md`) and none of the machinery that note
found broken. Rather than pinning scales through QDQ -- which crashes
Pulsar2's PPQ pass -- split the weight in two::

    W_hi = the value the compiler's own quantiser will land on
    W_lo = W - W_hi                    # peak(W_lo) <= peak(W)/255, per channel

and emit ``conv(x, W_hi) + conv(x, W_lo)``. Nothing asks the compiler for a
scale: ``W_lo``'s peak is 255x smaller, so ordinary per-tensor PTQ *derives* a
255x finer scale for the second convolution on its own. On a real conv weight
that lifts the weight's own SNR from 37.7 dB to 79.5 dB.

**The residual must be taken against what the compiler will materialise, not
against what you asked for.** `quantise_dequantise` is not idempotent -- the
peak of a quantised channel is ``127`` steps, not ``127.5``, so re-quantising
shifts the scale by 0.4% and moves values by up to half a step. Subtracting
the once-quantised weight leaves that half-step behind and the split recovers
48 dB instead of 79 dB; a hardware probe built that way gained nothing at all
(see `compiler_view`).

What it costs: one extra convolution and one extra ``Add``, so the output is
requantised twice instead of once -- ``e_Y`` is paid twice, 3 dB worse on that
term. **The split therefore only pays when ``e_Y`` is not the binding
constraint**, which in practice means running the split ops' activations at
16 bits (``layer_configs: [{"op_types": [...], "data_type": "U16"}]``). At
INT8 activations throughout, per-channel weight quantisation is already good
enough that the extra requantisation costs more than the residual returns.

Usage::

    precision.py in.onnx out.onnx                  # split every eligible op
    precision.py --dry-run in.onnx                 # report per-layer e_W
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

#: Ops whose second input is a weight this module knows how to split.
SPLITTABLE = ("Conv", "ConvTranspose", "MatMul", "Gemm")


def channel_axis(node, ndim):
    """Which axis of `node`'s weight carries the per-channel scale.

    Pulsar2 gives every weight one scale per *output* channel, and where that
    axis sits depends on the op's own layout: `Conv` is ``(Cout, Cin/g, k...)``
    but `ConvTranspose` is ``(Cin, Cout/g, k...)``, and `Gemm`'s depends on
    `transB`.
    """
    if node.op_type == "ConvTranspose":
        return 1
    if node.op_type == "Conv":
        return 0
    if node.op_type == "Gemm":
        trans_b = any(a.name == "transB" and a.i for a in node.attribute)
        return 0 if trans_b else ndim - 1
    return ndim - 1  # MatMul: x @ W, output channel last


def quantise_dequantise(w, bits=8, axis=0):
    """`w` through Pulsar2's weight quantiser and back.

    Symmetric, one scale per `axis` slice, step ``max|w_c| / (2**(bits-1) -
    0.5)``, round-half-even -- the quantiser `README.md` recovered byte-exactly
    from real weight tables, minus the ``+128`` storage offset, which cancels
    here.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 0:
        axis = None
    red = None if axis is None else tuple(i for i in range(w.ndim) if i != axis)
    peak = np.abs(w).max(axis=red, keepdims=red is not None)
    peak = np.asarray(peak, dtype=np.float32)
    step = np.where(peak == 0, 1.0, peak / (2.0 ** (bits - 1) - 0.5))
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return (np.clip(np.rint(w / step), lo, hi) * step).astype(np.float32)


def compiler_view(w, bits=8, axis=0):
    """What Pulsar2 will actually materialise for an initializer holding `w`.

    For a weight that is already quantised this is *not* `w`: a quantised
    channel's peak is 127 steps where the scale assumed 127.5, so the compiler
    derives a scale 0.4% smaller and re-rounds. The residual has to be taken
    against this, or half a step of the original error survives the split.
    """
    return quantise_dequantise(w, bits, axis)


def peak_to_rms(w, axis=None):
    """How much of the quantiser's range the outliers are wasting.

    With `axis`, the median across per-channel slices -- which is the number
    that matters, since the scale is per channel. Per-tensor peak/rms
    overstates the damage badly: 14.7 against 5.6 on the same weights.
    """
    w = np.asarray(w, dtype=np.float64)
    if axis is None:
        rms = float(np.sqrt((w**2).mean()))
        return float(np.abs(w).max()) / rms if rms else 0.0
    red = tuple(i for i in range(w.ndim) if i != axis)
    rms = np.sqrt((w**2).mean(axis=red))
    peak = np.abs(w).max(axis=red)
    ok = rms > 0
    return float(np.median(peak[ok] / rms[ok])) if ok.any() else 0.0


def weight_error_db(w, bits=8, axis=0):
    """SNR, in dB, that quantising `w` alone costs the layer's output.

    Rounding error is uniform over one step, so its energy is ``step**2/12``
    against the weight's own variance; both ride the same activations, so the
    activations cancel and only `peak_to_rms` survives::

        e_W = 10 log10(12 * (2**(bits-1) - 0.5)**2) - 20 log10(peak/rms)
    """
    p2r = peak_to_rms(w, axis)
    if p2r == 0.0:
        return float("inf")
    full = 2.0 ** (bits - 1) - 0.5
    return float(10 * np.log10(12 * full**2) - 20 * np.log10(p2r))


def _initializers(model):
    return {i.name: i for i in model.graph.initializer}


def splittable_nodes(model, ops=SPLITTABLE):
    """Nodes whose weight is a float initializer this module can split."""
    inits = _initializers(model)
    out = []
    for node in model.graph.node:
        if node.op_type not in ops or len(node.input) < 2:
            continue
        w = inits.get(node.input[1])
        if w is None or w.data_type != TensorProto.FLOAT or not w.dims:
            continue
        out.append(node)
    return out


def _unique(taken, stem):
    name, n = stem, 0
    while name in taken:
        n += 1
        name = f"{stem}_{n}"
    taken.add(name)
    return name


def split_weight(w, bits=8, axis=0):
    """`(W_hi, W_lo)` such that the compiler's view of the pair reconstructs `w`."""
    w = np.asarray(w, dtype=np.float32)
    hi = quantise_dequantise(w, bits, axis)
    lo = (w - compiler_view(hi, bits, axis)).astype(np.float32)
    return hi, lo


def weight_residual_split(
    model, bits=8, ops=SPLITTABLE, layers=None, min_peak_to_rms=0.0
):
    """Rewrite eligible weighted ops as ``op(x, W_hi) + op(x, W_lo)``.

    `layers` restricts the rewrite to a set of node names; `min_peak_to_rms`
    is the calibration-free filter, applied to the per-channel peak/rms.

    Returns the number of nodes rewritten. The rewrite is *not* exact in
    float -- ``W_hi + W_lo`` differs from ``W`` by the compiler's own
    re-rounding of ``W_hi``, about 0.4% of one INT8 step -- but a float run of
    the result is far closer to the original than either half alone.
    """
    inits = _initializers(model)
    taken = (
        {i.name for i in model.graph.initializer}
        | {n.name for n in model.graph.node if n.name}
        | {o for n in model.graph.node for o in n.output}
    )
    targets = []
    for node in splittable_nodes(model, ops):
        if layers is not None and node.name not in layers:
            continue
        w = numpy_helper.to_array(inits[node.input[1]])
        if peak_to_rms(w, channel_axis(node, w.ndim)) < min_peak_to_rms:
            continue
        targets.append(node)
    if not targets:
        return 0

    replacement = {}
    for node in targets:
        w = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32)
        hi, lo = split_weight(w, bits, channel_axis(node, w.ndim))
        stem = node.name or node.output[0]
        hi_w = _unique(taken, f"{stem}_hi_w")
        lo_w = _unique(taken, f"{stem}_lo_w")
        model.graph.initializer.extend(
            [numpy_helper.from_array(hi, hi_w), numpy_helper.from_array(lo, lo_w)]
        )
        hi_out = _unique(taken, f"{stem}_hi")
        lo_out = _unique(taken, f"{stem}_lo")

        hi_node = onnx.NodeProto()
        hi_node.CopyFrom(node)
        hi_node.name = _unique(taken, f"{stem}_hi_op")
        hi_node.input[1] = hi_w
        del hi_node.output[:]
        hi_node.output.append(hi_out)

        lo_node = onnx.NodeProto()
        lo_node.CopyFrom(node)
        lo_node.name = _unique(taken, f"{stem}_lo_op")
        lo_node.input[1] = lo_w
        # The bias / Gemm `C` rides on the high half only: it is not what is
        # being split, and adding it to both would double it.
        del lo_node.input[2:]
        del lo_node.output[:]
        lo_node.output.append(lo_out)

        add = helper.make_node(
            "Add",
            [hi_out, lo_out],
            [node.output[0]],
            name=_unique(taken, f"{stem}_join"),
        )
        replacement[id(node)] = [hi_node, lo_node, add]

    rebuilt = []
    for node in model.graph.node:
        rebuilt.extend(replacement.get(id(node), [node]))
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    return len(targets)


def split_op_types(model, ops=SPLITTABLE):
    """Op types a split graph introduces, for a `layer_configs` entry.

    The whole point of the split is undone if the ``Add`` that joins the two
    halves requantises to INT8, so the caller almost always wants to promote
    these to ``U16``.
    """
    return sorted({n.op_type for n in splittable_nodes(model, ops)} | {"Add"})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input")
    p.add_argument("output", nargs="?")
    p.add_argument(
        "--bits",
        type=int,
        default=8,
        help="width the compiler will quantise the weight to (8)",
    )
    p.add_argument(
        "--min-peak-to-rms",
        type=float,
        default=0.0,
        help="only split weights at least this outlier-heavy",
    )
    p.add_argument("--ops", default=",".join(SPLITTABLE))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be split and why, change nothing",
    )
    args = p.parse_args(argv)

    model = onnx.load(args.input)
    ops = tuple(o for o in args.ops.split(",") if o)
    if args.dry_run or not args.output:
        inits = _initializers(model)
        print(f"{'node':<44}{'op':<15}{'peak/rms':>9}{'e_W dB':>9}{'split dB':>10}")
        for node in splittable_nodes(model, ops):
            w = numpy_helper.to_array(inits[node.input[1]])
            ax = channel_axis(node, w.ndim)
            hi, lo = split_weight(w, args.bits, ax)
            got = compiler_view(hi, args.bits, ax) + quantise_dequantise(
                lo, args.bits, ax
            )
            err = float(((w - got).astype(np.float64) ** 2).sum())
            sig = float((w.astype(np.float64) ** 2).sum())
            print(
                f"{(node.name or node.output[0])[:43]:<44}{node.op_type:<15}"
                f"{peak_to_rms(w, ax):>9.2f}{weight_error_db(w, args.bits, ax):>9.2f}"
                f"{10 * np.log10(sig / max(err, 1e-30)):>10.2f}"
            )
        return 0
    n = weight_residual_split(
        model, bits=args.bits, ops=ops, min_peak_to_rms=args.min_peak_to_rms
    )
    onnx.save(model, args.output, save_as_external_data=model.ByteSize() > 2**30)
    print(f"split {n} op(s); promote {split_op_types(model, ops)} to U16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
