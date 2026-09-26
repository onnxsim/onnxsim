#!/usr/bin/env python3
"""qdq_layer.py output dir -> a flat "case" for the DSP clients / hexagon-sim programs:
meta.txt "M K N zx zy relu sx sy H W k stride", x.bin uint8 [H*W, C] (NHWC rows), w.bin int8 [N, C, k, k] (ONNX
order), bq.bin int32 [N], sw.bin fp32 [N], ref.bin uint8 [Ho*Wo, N] (ORT CPU levels). M = rows of x.
usage: export_case.py <layer dir> <out> [rows (1x1 only)]"""

import sys
from pathlib import Path

import numpy as np


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    p = np.load(src / "params.npz")
    k, stride = int(p["k"]), int(p["stride"])
    assert not (len(sys.argv) > 3 and k != 1), "row subsets only for 1x1 layers"
    xq = p["xq"][0]  # [C, H, W]
    c, h, w = xq.shape
    n = p["wq"].shape[0]
    x = xq.reshape(c, -1).T.copy()  # [M, K]
    ref = np.fromfile(src / "ref_q.bin", np.uint8).reshape(n, -1).T.copy()  # [M, N]
    if rows:
        x, ref = x[:rows], ref[:rows]
    m = x.shape[0]
    x.tofile(out / "x.bin")
    p["wq"].astype(np.int8).tofile(out / "w.bin")
    p["bq"].astype(np.int32).tofile(out / "bq.bin")
    p["sw"].astype(np.float32).tofile(out / "sw.bin")
    ref.tofile(out / "ref.bin")
    (out / "meta.txt").write_text(
        f"{m} {c} {n} {int(p['zx'])} {int(p['zy'])} {int(p['relu'])} {float(p['sx']).hex()} {float(p['sy']).hex()}"
        f" {h} {w} {k} {stride}\n"
    )
    print(out, m, c, n)


if __name__ == "__main__":
    main()
