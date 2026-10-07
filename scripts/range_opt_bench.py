#!/usr/bin/env python3
"""Measure what ``onnxsim.range_opt`` finds on a real model, and what it is worth.

Usage::

    python scripts/range_opt_bench.py resnet18.onnx --preset imagenet
    python scripts/range_opt_bench.py distilbert.onnx --preset tokens --vocab 30522 --allow-interface-change
    python scripts/range_opt_bench.py model.onnx --preset imagenet --diagnose   # how close do operands come?

Presets give the *declared box* the rewrites are allowed to rely on:

* ``imagenet``: ImageNet-normalised pixels, ``(x - mean) / std`` for ``x in [0, 1]`` per channel,
  input ``x`` pinned to ``[1, 3, 224, 224]`` (a dynamic batch dim otherwise leaves only value hulls).
* ``tokens``: integer ``input_ids`` in ``[0, vocab)`` and a 0/1 ``attention_mask``.

For every run it prints the rewrites found and applied, the maximum output difference between the
original and the rewritten model on sampled inputs *inside* the box, and ONNX Runtime latency of
both, interleaved (original / rewritten / original-again, one thread) so that machine load hits all
three alike -- the original-again column is the measurement-noise floor.
"""

import argparse
import collections
import json
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

from onnxsim import interval, range_opt

_MEAN, _STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def _preset(name, vocab):
    if name == "imagenet":
        lo = ((0 - _MEAN) / _STD).reshape(1, 3, 1, 1)
        hi = ((1 - _MEAN) / _STD).reshape(1, 3, 1, 1)
        return {"x": (lo, hi)}, {"x": [1, 3, 224, 224]}
    return {"input_ids": (0, vocab - 1), "attention_mask": (0, 1)}, None


def _feeds(model, preset, ranges, rng, vocab):
    feeds = {}
    for vi in model.graph.input:
        tt = vi.type.tensor_type
        shape = [
            d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1
            for d in tt.shape.dim
        ]
        int32 = tt.elem_type == onnx.TensorProto.INT32
        if preset == "imagenet":
            lo, hi = ranges["x"]
            feeds[vi.name] = (lo + (hi - lo) * rng.random(shape)).astype(np.float32)
        elif vi.name == "input_ids":
            feeds[vi.name] = rng.integers(0, vocab, size=shape).astype(
                np.int32 if int32 else np.int64
            )
        else:
            mask = (rng.random(shape) < 0.8).astype(np.int64)
            mask[..., 0] = 1
            feeds[vi.name] = mask.astype(np.int32 if int32 else np.int64)
    return feeds


def _session(model):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), opt, providers=["CPUExecutionProvider"]
    )


def _diagnose(model, ranges, shapes):
    """Operand hulls of the ops the dead-code rules look at: how far from firing are they?"""
    res = interval.propagate(model, ranges, shapes)
    out = {}
    for op in ("Relu", "Clip"):
        hulls = [
            res.hull(n.input[0])
            for n in model.graph.node
            if n.op_type == op and n.input[0] in res.intervals
        ]
        if hulls:
            out[op] = {
                "count": len(hulls),
                "operand_lo_nonneg": sum(1 for a, _ in hulls if a >= 0),
                "operand_hi_le_6": sum(1 for _, b in hulls if b <= 6),
                "first_hulls": [[round(a, 2), round(b, 2)] for a, b in hulls[:3]],
                "widest": [min(a for a, _ in hulls), max(b for _, b in hulls)],
            }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--preset", choices=("imagenet", "tokens"), required=True)
    ap.add_argument("--vocab", type=int, default=30522)
    ap.add_argument("--allow-interface-change", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument(
        "--no-unconditional",
        action="store_true",
        help="skip the second (unbounded) analysis",
    )
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("-o", "--output", help="save the rewritten model here")
    args = ap.parse_args(argv)

    model = onnx.load(args.model)
    ranges, shapes = _preset(args.preset, args.vocab)
    if args.diagnose:
        print(json.dumps(_diagnose(model, ranges, shapes), indent=1))
        return 0
    kw = dict(
        allow_interface_change=args.allow_interface_change,
        input_shapes=shapes,
        check_unconditional=not args.no_unconditional,
    )
    t0 = time.perf_counter()
    props = range_opt.analyze(model, ranges, **kw)
    analyze_s = time.perf_counter() - t0
    by = collections.Counter(
        (p.rule, "apply" if p.applies else "report") for p in props
    )
    new, log = range_opt.apply(model, ranges, **kw)
    if args.output:
        onnx.save(new, args.output)
    rng = np.random.default_rng(0)
    so, sn, so2 = _session(model), _session(new), _session(model)
    worst = scale = 0.0
    for _ in range(8):
        fo = _feeds(model, args.preset, ranges, rng, args.vocab)
        fn = _feeds_like(new, fo)
        a, b = so.run(None, fo)[0], sn.run(None, fn)[0]
        worst, scale = (
            max(worst, float(np.max(np.abs(a - b)))),
            max(scale, float(np.max(np.abs(a)))),
        )
    fo = _feeds(model, args.preset, ranges, rng, args.vocab)
    fn = _feeds_like(new, fo)
    for s, f in ((so, fo), (sn, fn), (so2, fo)):
        for _ in range(5):
            s.run(None, f)
    times = ([], [], [])
    for _ in range(args.runs):
        for s, f, t in zip((so, sn, so2), (fo, fn, fo), times):
            t0 = time.perf_counter()
            s.run(None, f)
            t.append((time.perf_counter() - t0) * 1000)
    med = [float(np.median(t)) for t in times]
    print(
        json.dumps(
            {
                "model": args.model,
                "analyze_s": round(analyze_s, 1),
                "proposals": {f"{r}:{k}": v for (r, k), v in sorted(by.items())},
                "applied": [r["rule"] for r in log if r["applied"]],
                "nodes": [len(model.graph.node), len(new.graph.node)],
                "max_abs_diff_inside_box": worst,
                "output_scale": scale,
                "ort_median_ms": {
                    "original": round(med[0], 2),
                    "rewritten": round(med[1], 2),
                    "original_again": round(med[2], 2),
                },
                "ratio_rewritten": round(med[1] / med[0], 3),
                "noise_floor_ratio": round(med[2] / med[0], 3),
            },
            indent=1,
        )
    )
    return 0


def _feeds_like(new, feeds):
    """Cast the sampled feeds to the (possibly narrowed) input dtypes of ``new``."""
    out = {}
    for vi in new.graph.input:
        v = feeds[vi.name]
        out[vi.name] = (
            v.astype(np.int32)
            if vi.type.tensor_type.elem_type == onnx.TensorProto.INT32
            else v
        )
    return out


if __name__ == "__main__":
    sys.exit(main())
