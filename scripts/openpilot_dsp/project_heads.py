"""Project the heads (everything outside the conv backbone) as simulated int8 GEMVs.

Every Gemm/MatMul with a constant weight outside `quantize.backbone_nodes` is run through `hvx65`'s
`gemv` kernel on hexagon-sim at its real (M, K, N): int8 weights prepacked for vrmpy and evicted from L2
before the call, so they stream from DDR with l2fetch prefetch as they would every frame; K padded to 4
and N to 32. Activation-by-activation MatMuls (the attention's QK^T / AV over 9 tokens) and the float
epilogues (LayerNorm, Softmax, Sigmoid, GELU, the heads' requant) are not simulated and are listed apart.

  python project_heads.py driving_fp32.onnx [--a16] [--clock-mhz 1000] [--json out.json]
"""

import argparse
import json
import os

import numpy as np
import onnx
import quantize
from onnx import shape_inference
from project_latency import HERE, rup, sim_cycles


def head_gemms(path):
    m = shape_inference.infer_shapes(onnx.load(path))
    g = m.graph
    shp = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(g.value_info) + list(g.input) + list(g.output)
    }
    init = {i.name: list(i.dims) for i in g.initializer}
    bb = {n.name for n in quantize.backbone_nodes(m)}
    out, other = [], []
    for n in g.node:
        if n.op_type not in ("Gemm", "MatMul") or n.name in bb:
            continue
        a = shp.get(n.input[0])
        M = int(np.prod(a[:-1])) if a else 1
        if n.input[1] not in init:
            other.append((n.name, a, shp.get(n.input[1])))
            continue
        w = init[n.input[1]]
        tb = any(x.name == "transB" and x.i for x in n.attribute)
        K, N = (w[1], w[0]) if tb else (w[0], w[1])
        out.append((n.name, M, K, N, n.input[1]))
    return out, other


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument(
        "--a16",
        action="store_true",
        help="uint16 activations (two vrmpy per weight vector)",
    )
    ap.add_argument("--clock-mhz", type=float, default=1000.0)
    ap.add_argument(
        "--w8-groups",
        help="file: comma list of head weight groups (head_bits.py) kept int8; every other "
        "head GEMV gets int16 weights, estimated at 2x its simulated int8 cycles (twice the bytes and "
        "vrmpy passes)",
    )
    ap.add_argument("--cache", default=os.path.join(HERE, "sim_cycles_cache.json"))
    ap.add_argument("--json")
    args = ap.parse_args()
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    gemms, other = head_gemms(args.model)
    w8 = None
    if args.w8_groups:
        from head_bits import group

        w8 = set(open(args.w8_groups).read().strip().split(","))
    rows, tot, wbytes = [], 0, 0
    for name, M, K, N, wname in gemms:
        key = ["gemv", M, rup(K, 4), rup(N, 32), int(args.a16), 1]
        c = sim_cycles(key, cache)
        json.dump(cache, open(args.cache, "w"), indent=0)
        wide = w8 is not None and group(wname) not in w8
        if wide:
            c *= 2
        rows.append((name, M, K, N, c, "w16 est" if wide else "w8 sim"))
        tot += c
        wbytes += K * N * (2 if wide else 1)
    ms = tot / (args.clock_mhz * 1e3)
    print(
        f"{args.model}: {len(gemms)} weight GEMV/GEMMs, {wbytes / 1e6:.1f} MB weights, "
        f"{tot / 1e6:.2f} M cycles single thread -> {ms:.2f} ms at {args.clock_mhz:.0f} MHz "
        f"({wbytes / tot:.1f} weight bytes/cycle); a16={args.a16}"
    )
    big = sorted(rows, key=lambda r: -r[4])[:6]
    for r in big:
        print(f"  {r[0]:40s} M={r[1]} K={r[2]} N={r[3]}: {r[4]} cycles")
    print(
        f"  not simulated: {len(other)} activation x activation MatMuls {[o[1:] for o in other]}"
    )
    if args.json:
        json.dump(
            {
                "rows": rows,
                "total_cycles": tot,
                "ms": ms,
                "weight_bytes": wbytes,
                "other": other,
                "a16": args.a16,
            },
            open(args.json, "w"),
            indent=1,
            default=str,
        )


if __name__ == "__main__":
    main()
