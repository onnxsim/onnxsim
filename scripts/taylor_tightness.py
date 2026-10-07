#!/usr/bin/env python3
"""Reproduce the tightness table in docs/taylor-models.md.

For each case this runs ``onnxsim.taylor.compare_methods`` (interval, zonotope, CROWN, Taylor
model, mean-value form) and prints the mean output width of every method next to the width
*sampled* through onnxruntime (random, random-vertex and corner inputs). ``sampled`` is an inner
approximation of the true range, so a sound method's ratio is at or above 1.00x.

Usage: ``python scripts/taylor_tightness.py`` (needs onnxruntime; a few minutes on a CPU).
Weights are seeded, so the table is reproducible up to onnxruntime's float32 rounding.
"""

import math
import sys

import numpy as np
import onnx
from onnx import numpy_helper, parser

from onnxsim import taylor


def build(body, inits=None, opset=17):
    m = parser.parse_model(f'<ir_version: 9, opset_import: ["" : {opset}]> ' + body)
    m.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (inits or {}).items()
    )
    onnx.checker.check_model(m)
    return m


def cases():
    rng = np.random.default_rng(0)

    def f32(*s):
        return rng.standard_normal(s).astype(np.float32)

    centres = np.linspace(-1.5, 1.5, 6).astype(np.float32)
    gamma = np.linspace(0.8, 1.2, 6).astype(np.float32)
    beta = np.linspace(-0.1, 0.1, 6).astype(np.float32)
    softmax = build("m (float[2,4] x) => (float[2,4] y) { y = Softmax<axis=-1>(x) }")
    gelu = build("m (float[1,6] x) => (float[1,6] y) { y = Gelu(x) }", opset=20)
    gelu_dec = build(
        """
        m (float[1,6] x) => (float[1,6] y) {
          s = Constant<value=float {1.4142135}>()
          one = Constant<value=float {1.0}>()
          half = Constant<value=float {0.5}>()
          t = Div(x, s)
          e = Erf(t)
          u = Add(e, one)
          v = Mul(x, u)
          y = Mul(v, half)
        }"""
    )
    ln = build(
        "m (float[1,6] x) => (float[1,6] y) "
        "{ y = LayerNormalization<axis=-1, epsilon=1e-5>(x, g, b) }",
        {"g": gamma, "b": beta},
    )
    ln_dec = build(
        """
        m (float[1,6] x) => (float[1,6] y) {
          mu = ReduceMean<axes=[-1], keepdims=1>(x)
          d = Sub(x, mu)
          sq = Mul(d, d)
          var = ReduceMean<axes=[-1], keepdims=1>(sq)
          eps = Constant<value=float {1e-5}>()
          ve = Add(var, eps)
          sd = Sqrt(ve)
          n = Div(d, sd)
          gn = Mul(n, gamma)
          y = Add(gn, beta)
        }""",
        {"gamma": gamma, "beta": beta},
        opset=13,
    )
    tokens, d, h = 4, 8, 8
    attn = build(
        f"""
        m (float[{tokens},{d}] x) => (float[{tokens},{h}] y) {{
          q = MatMul(x, Wq)
          k = MatMul(x, Wk)
          v = MatMul(x, Wv)
          kt = Transpose<perm=[1,0]>(k)
          sc = MatMul(q, kt)
          ss = Mul(sc, scale)
          a = Softmax<axis=-1>(ss)
          y = MatMul(a, v)
        }}""",
        {
            "Wq": 0.4 * f32(d, h),
            "Wk": 0.4 * f32(d, h),
            "Wv": 0.4 * f32(d, h),
            "scale": np.array(1.0 / math.sqrt(h), dtype=np.float32),
        },
    )
    convsig = build(
        """
        m (float[1,2,6,6] x) => (float[1,3,6,6] y) {
          c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)
          s1 = Sigmoid(c1)
          c2 = Conv<pads=[1,1,1,1]>(s1, W2, B2)
          y = Sigmoid(c2)
        }""",
        {
            "W1": 0.8 * f32(4, 2, 3, 3),
            "B1": 0.2 * f32(4),
            "W2": 0.8 * f32(3, 4, 3, 3),
            "B2": 0.2 * f32(3),
        },
    )
    return [
        ("Softmax [2,4]", softmax, lambda w: {"x": (-w, w)}, (0.1, 0.25)),
        (
            "GELU fused [1,6], x in 0.3 +- w",
            gelu,
            lambda w: {"x": (0.3 - w, 0.3 + w)},
            (0.25, 1.0),
        ),
        (
            "GELU decomposed (Div/Erf/Add/Mul), x in +-w",
            gelu_dec,
            lambda w: {"x": (-w, w)},
            (0.25, 0.5),
        ),
        (
            "LayerNorm fused [1,6], spread centres +- w",
            ln,
            lambda w: {"x": (centres - w, centres + w)},
            (0.05, 0.1),
        ),
        (
            "LayerNorm decomposed, spread centres +- w",
            ln_dec,
            lambda w: {"x": (centres - w, centres + w)},
            (0.05, 0.1),
        ),
        (
            "1-head attention [4,8], 32 inputs",
            attn,
            lambda w: {"x": (-w, w)},
            (0.05, 0.15),
        ),
        (
            "Conv-Sigmoid-Conv-Sigmoid [1,2,6,6]",
            convsig,
            lambda w: {"x": (-w, w)},
            (0.1, 0.3),
        ),
    ]


def main() -> int:
    methods = ("interval", "zonotope", "crown", "taylor", "mean_value")
    print("| case | box w | sampled | " + " | ".join(methods) + " |")
    print("|---|---|---|" + "---|" * len(methods))
    for title, model, make_ranges, widths in cases():
        for w in widths:
            cmp = taylor.compare_methods(
                model, make_ranges(w), samples=1500, methods=methods
            )
            out = cmp.outputs[0]
            ref = cmp.width("sampled", out)
            cells = []
            for method in methods:
                if method not in cmp.bounds:
                    cells.append("n/a")
                    continue
                x = cmp.width(method, out)
                cells.append(
                    "unbounded" if not np.isfinite(x) else f"{x:.4g} ({x / ref:.2f}x)"
                )
            print(f"| {title} | {w} | {ref:.4g} | " + " | ".join(cells) + " |")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
