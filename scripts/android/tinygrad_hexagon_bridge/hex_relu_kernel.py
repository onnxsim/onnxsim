"""V65 HVX int32 ReLU custom kernel for tinygrad.

ReLU is common in quantized residual paths. The generic DSP renderer can
otherwise scalarize the comparison, so this keeps the operation in one
128-byte vector per iteration.
"""

from __future__ import annotations

import argparse
import functools
import os


def build_kernel(n: int, x, kernel_name: str = "hex_relu"):
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert n % 32 == 0, "V65 int32 vector kernel requires 32-lane padding"
    i32x32 = "int __attribute__((vector_size(128)))"

    def kernel_fn(C: UOp, A: UOp) -> UOp:
        v_rng = UOp.range(n // 32, 0, AxisType.WEAK)
        x_idx, y_idx = A[v_rng * 32], C[v_rng * 32]
        step = UOp(
            Ops.CUSTOM,
            dtypes.void,
            (y_idx, x_idx),
            arg=(
                f"*({i32x32}*){{0}} = __builtin_elementwise_max("
                f"*({i32x32}*){{1}}, ({i32x32})0);"
            ),
        )
        return step.end(v_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(n, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, x, fxn=functools.partial(kernel_fn))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1 << 16)
    parser.add_argument("--out", default="relu_kernel.c")
    args = parser.parse_args()
    os.environ.setdefault("DEV", "DSP")
    os.environ.setdefault("MOCKDSP", "1")

    import numpy as np
    from tinygrad import Tensor
    from tinygrad.renderer.cstyle import ClangRenderer

    rng = np.random.default_rng(7)
    x_np = rng.integers(-(1 << 20), 1 << 20, args.n).astype(np.int32)
    captured = {}
    original = ClangRenderer.render

    def capture(self, uops):
        source = original(self, uops)
        captured["source"] = source
        return source

    ClangRenderer.render = capture
    try:
        got = build_kernel(args.n, Tensor(x_np)).realize().numpy()
    finally:
        ClangRenderer.render = original
    if not np.array_equal(got, np.maximum(x_np, 0)):
        raise SystemExit("ReLU kernel is incorrect")
    source = captured["source"]
    marker = source.find("/* DSP boilerplate */")
    with open(args.out, "w") as f:
        f.write((source[:marker] if marker >= 0 else source).rstrip() + "\n")
    print(f"correctness: True; wrote {args.out}")


if __name__ == "__main__":
    main()
