#!/usr/bin/env python3
"""Build (and cache) the device min/max models for every tensor size the 16-bit
MatMul/Conv chains of the training step read or write. Pulsar2 only, no device."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step_runner as sr  # noqa: E402
import tinygrad_ax_backend as axb  # noqa: E402
import u16_chain  # noqa: E402

CACHE = os.environ.get("U16_CACHE", "/mnt/data/cache/claude-work/u16-seg-cache")


def main() -> None:
    model = sr.load_step()
    segs, _ = sr.build_plan(model, sr.load_records(), axb.load_calibration(sr.STEP_CALIB), None, False, precision_overrides=None)
    shapes = {
        v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    sizes: set[int] = set()
    for s in segs:
        if s.kind == "matmul_chain":
            for t in (*s.inputs, *s.outputs):
                if t in shapes:
                    sizes.add(int(np.prod(shapes[t])))
    print(len(sizes), "distinct sizes", flush=True)
    for k, n in enumerate(sorted(sizes)):
        try:
            u16_chain.amax_blob(CACHE, n)
            print(f"  amax {n} ({k + 1}/{len(sizes)})", flush=True)
        except Exception as exc:
            print(f"  amax {n}: {type(exc).__name__}: {exc}"[:160], flush=True)


if __name__ == "__main__":
    main()
