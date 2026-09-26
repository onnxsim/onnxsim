"""Project openpilot model latency on the 845's V65 cDSP from hexagon-sim cycles of the hvx65 kernels.

Every Conv of the model's vision backbone (everything up to the last Conv; the Gemm/MatMul heads are
listed separately) is mapped to one measured kernel run on its real shape:

- 1x1 conv            -> `pw`   P=H*W (padded to 32), K=Cin, N=Cout   (uint8 input)  / `pw16` (uint16 input)
- dense kxk conv      -> `pw`   on the im2col'd input, K=Cin*k*k padded to 4 (im2col itself: estimated)
- depthwise kxk       -> `dw3` (3x3) / `dwc3` (5x5, 7x7), or round 1's `dwc` with --dw-kernel dwc; channels-last,
                         C padded to 128 (uint16 input: 2x, estimated)
- elementwise (Gelu, Add, Mul, ...) on backbone activations: the measured `gelu` LUT rate per byte
  (uint16: 2x bytes, estimated)
- 1x1 conv at 1x1 spatial (a GEMV): weight bytes / an assumed DDR bandwidth (estimated)

Activation dtype per conv input comes from the quantization policy: `--act uint8` / `uint16` for the
whole backbone, or `--policy` (quantize.py JSON, `tensor_dtypes` overriding a uint16 base).

Clock / parallelism are assumptions, printed with the result: `--clock-mhz` (default 1000) and
`--hvx-threads` (default 2; perfect scaling assumed).
"""

import argparse
import json
import os
import re
import subprocess

import onnx
from onnx import shape_inference

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "hvx65", "run_sim.sh")


def rup(x, m):
    return (x + m - 1) // m * m


def sim_cycles(args, cache):
    key = " ".join(map(str, args))
    if key not in cache:
        out = subprocess.run(
            [SIM, *map(str, args)], capture_output=True, text=True, check=True
        ).stdout
        cache[key] = int(re.search(r"cycles=(\d+)", out).group(1))
        print(f"  sim {key}: {cache[key]} cycles", flush=True)
    return cache[key]


def layers(model_path, policy, act):
    m = shape_inference.infer_shapes(onnx.load(model_path))
    g = m.graph
    shp = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(g.value_info) + list(g.input) + list(g.output)
    }
    for i in g.initializer:
        shp[i.name] = list(i.dims)
    last = max(i for i, n in enumerate(g.node) if n.op_type == "Conv")
    td = json.load(open(policy)).get("tensor_dtypes", {}) if policy else {}

    def dt(t):
        return td.get(t, act)

    out = []
    for n in g.node[: last + 1]:
        if n.op_type == "Conv":
            x, w, y = shp[n.input[0]], shp[n.input[1]], shp[n.output[0]]
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            grp, s, k = a.get("group", 1), (a.get("strides") or [1])[0], w[2]
            out.append(
                dict(
                    kind="conv",
                    name=n.output[0],
                    node=n.name,
                    x=x,
                    w=w,
                    y=y,
                    group=grp,
                    stride=s,
                    k=k,
                    dtype=dt(n.input[0]),
                )
            )
        elif (
            n.op_type in ("Gelu", "Add", "Mul", "Sub", "Div", "Relu", "Sigmoid")
            and n.output[0] in shp
        ):
            y = shp[n.output[0]]
            if len(y) == 4:
                out.append(
                    dict(
                        kind="eltwise",
                        op=n.op_type,
                        name=n.output[0],
                        elems=y[0] * y[1] * y[2] * y[3],
                        dtype=dt(n.output[0]),
                    )
                )
    heads = [n for n in g.node[last + 1 :] if n.op_type in ("Gemm", "MatMul")]
    head_params = sum(
        int(__import__("numpy").prod(shp[n.input[1]]))
        for n in heads
        if n.input[1] in shp
    )
    return out, len(heads), head_params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--act", default="uint8", choices=["uint8", "uint16"])
    ap.add_argument("--policy")
    ap.add_argument(
        "--dw-kernel",
        default="best",
        choices=["dwc", "best"],
        help="depthwise kernel: dwc (round 1) or best (dw3 for 3x3, dwc3 otherwise; §7)",
    )
    ap.add_argument(
        "--w16",
        default="",
        help="comma list of conv node names with int16 weights, or 'all' "
        "(pw/dense: 2 vrmpy weight-byte passes, estimated as 2x the int8-weight cycles; depthwise "
        "already multiplies 16-bit weights)",
    )
    ap.add_argument("--clock-mhz", type=float, default=1000.0)
    ap.add_argument("--hvx-threads", type=int, default=2)
    ap.add_argument(
        "--ddr-gbps",
        type=float,
        default=8.0,
        help="assumed DSP DDR read bandwidth for GEMV layers",
    )
    ap.add_argument("--cache", default=os.path.join(HERE, "sim_cycles_cache.json"))
    ap.add_argument("--json")
    args = ap.parse_args()
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    lut_rate = None
    rows = []
    w16 = set(args.w16.split(","))
    for L in layers(args.model, args.policy, args.act)[0]:
        u16 = L["dtype"] == "uint16"
        if L["kind"] == "eltwise":
            if lut_rate is None:
                lut_rate = 262144 / sim_cycles(["gelu", 262144], cache)
            rows.append(
                (
                    L["name"],
                    "eltwise",
                    L["elems"] * (2 if u16 else 1) / lut_rate,
                    "est" if u16 else "rate",
                    0,
                )
            )
            continue
        x, w, y, k, s = L["x"], L["w"], L["y"], L["k"], L["stride"]
        P = y[2] * y[3]
        macs = P * w[0] * w[1] * k * k
        if L["group"] > 1:
            if args.dw_kernel == "dwc":
                dw_args = ["dwc", x[1], x[2], x[3], k, s]
            elif k == 3:
                dw_args = ["dw3", x[1], x[2], x[3], s]
            else:
                dw_args = ["dwc3", x[1], x[2], x[3], k, s, 2]
            cyc = sim_cycles(dw_args, cache) * (2 if u16 else 1)
            rows.append(
                (
                    L["name"],
                    f"dw{k}x{k}/s{s} C={x[1]}",
                    cyc,
                    "sim" + ("x2est" if u16 else ""),
                    macs,
                )
            )
        elif P == 1:
            byts = w[0] * w[1]
            cyc = (
                byts / (args.ddr_gbps * 1e9) * args.clock_mhz * 1e6 * args.hvx_threads
            )  # bandwidth-bound, not thread-scaled
            rows.append((L["name"], f"gemv {w[1]}->{w[0]}", cyc, "est", macs))
        else:
            K = rup(w[1] * k * k, 4)
            N = rup(w[0], 8)
            cyc = sim_cycles(["pw16" if u16 else "pw", rup(P, 32), K, N], cache)
            note = "sim"
            if args.w16 == "all" or L["node"] in w16:
                cyc *= 2
                note += " x2 W16 est"
            if k > 1:
                cyc += (
                    rup(P, 32) * K * (2 if u16 else 1) / 64.0
                )  # im2col copy at 64 B/cycle: estimate
                note = "sim+im2col est"
            rows.append(
                (
                    L["name"],
                    f"{'dense' if k > 1 else 'pw'} {w[1]}->{w[0]} k{k} s{s} P={P}{' u16' if u16 else ''}",
                    cyc,
                    note,
                    macs,
                )
            )
        json.dump(cache, open(args.cache, "w"), indent=0)
    total = sum(r[2] for r in rows)
    by = {}
    for r in rows:
        kk = r[1].split()[0].split("/")[0]
        kk = "dw" if kk.startswith("dw") else kk
        by[kk] = by.get(kk, 0) + r[2]
    ms = total / args.hvx_threads / (args.clock_mhz * 1e3)
    print(
        f"{args.model}: {len(rows)} backbone ops, {sum(r[4] for r in rows) / 1e6:.0f} MMAC"
    )
    print(
        f"  single-thread cycles {total / 1e6:.2f} M; by kind (M): "
        + ", ".join(
            f"{k} {v / 1e6:.2f}" for k, v in sorted(by.items(), key=lambda t: -t[1])
        )
    )
    print(
        f"  projected backbone latency: {ms:.2f} ms at {args.clock_mhz:.0f} MHz x {args.hvx_threads} HVX threads (assumptions)"
    )
    if args.json:
        json.dump(
            {
                "rows": rows,
                "total_cycles": total,
                "ms": ms,
                "clock_mhz": args.clock_mhz,
                "hvx_threads": args.hvx_threads,
            },
            open(args.json, "w"),
            indent=1,
        )


if __name__ == "__main__":
    main()
