"""Upper-bound wall-clock saving per fusion/elimination category: 13 us per removed dispatch + moved bytes / 50 GB/s.

Reads an2.json from profile_breakdown.py. Constants calibrated on the phone (see WEBGPU_SURVEY.md, dispatch floor section).
"""

import collections
import json

res = json.load(open("an2.json"))
FLOOR = 13e-6
BW = 50e9
ACT = {"Relu", "Gelu", "QuickGelu", "Sigmoid", "Clip", "Tanh"}
out = {}


def cost(r, mult):
    return FLOOR + mult * r["outb"] / BW


for name, rows in res.items():
    cat = collections.defaultdict(lambda: [0, 0.0])
    for i, r in enumerate(rows):
        op = r["op"]
        prev = rows[i - 1]["op"] if i else ""
        if op == "Add" and prev in ("Conv", "Gemm", "MatMul"):
            k = "Conv+Add residual epilogue"
            c = cost(r, 2)
        elif op in ACT and prev in ("Conv", "Add", "Gemm"):
            k = "activation (" + op + ") into producer epilogue"
            c = cost(r, 2)
        elif op == "Transpose":
            k = "Transpose elimination"
            c = cost(r, 2)
        elif op == "Concat":
            k = "Concat elimination (producers write slices)"
            c = cost(r, 2)
        elif op == "Split":
            k = "Split elimination (strided views)"
            c = cost(r, 2)
        elif op in ("Slice", "Pad", "Resize", "DepthToSpace"):
            k = op + " fusion/elimination"
            c = cost(r, 2)
        elif op in ("Add", "Mul", "Sub", "Div"):
            k = "other elementwise (%s) fusion" % op
            c = cost(r, 3)
        else:
            continue
        cat[k][0] += 1
        cat[k][1] += c * 1e3
    tot = sum(v[1] for v in cat.values())
    n = sum(v[0] for v in cat.values())
    out[name] = (n, tot, sorted(cat.items(), key=lambda kv: -kv[1][1]))
    print(
        "==",
        name,
        "total nodes %d; removable %d nodes, upper-bound saving %.1f ms"
        % (len(rows), n, tot),
    )
    for k, (cn, ct) in out[name][2][:5]:
        print("    %-52s n=%-3d %.2f ms" % (k, cn, ct))
