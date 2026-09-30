"""Per-model op/provider breakdown from bench.cc PROFILE json files (prof/<model>_2*.json, 1 warmup + 3 runs -> 4 runs).

Writes an2.json (one run of nodes with median-of-runs durations) for fusion_estimate.py. Profiler durations are approximate;
wall-clock medians are authoritative.
"""

import collections
import glob
import json

NOD = {
    "Reshape",
    "Unsqueeze",
    "Squeeze",
    "Flatten",
    "Identity",
}  # zero-copy metadata ops in the WebGPU EP
ELEM = {
    "Add",
    "Mul",
    "Sub",
    "Div",
    "Relu",
    "Sigmoid",
    "Clip",
    "Gelu",
    "QuickGelu",
    "Tanh",
    "Log",
    "Exp",
    "Sqrt",
    "Pow",
    "Erf",
    "Neg",
    "Abs",
    "Where",
    "Cast",
    "Equal",
    "Less",
    "Greater",
}
MOVE = {
    "Transpose",
    "Concat",
    "Split",
    "Slice",
    "Pad",
    "Gather",
    "Resize",
    "DepthToSpace",
    "Expand",
    "Tile",
}
res = {}
for name in [
    "resnet50",
    "yolo11n",
    "yolo26n",
    "rtdetr_pre",
    "rtdetr_mid0",
    "rtdetr_mid1",
    "rtdetr_post",
    "sam_l0_enc",
]:
    ev = json.load(open(glob.glob("prof/%s_2*" % name)[0]))
    nodes = sorted(
        [
            e
            for e in ev
            if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")
        ],
        key=lambda e: e["ts"],
    )
    R = 4
    n = len(nodes) // R
    # events of one run are contiguous in ts order after sorting? group by node_index
    byidx = collections.defaultdict(list)
    for e in nodes:
        byidx[e["args"]["node_index"] + "|" + e["name"]].append(e)
    seq = []  # one run in order: take the last run
    last = nodes[-n:]
    med = {k: sorted(x["dur"] for x in v)[len(v) // 2] for k, v in byidx.items()}
    rows = []
    for e in last:
        a = e["args"]
        k = a["node_index"] + "|" + e["name"]
        rows.append(
            dict(
                op=a["op_name"],
                dur=med[k],
                prov=a["provider"],
                out=a.get("output_type_shape"),
                outb=int(a.get("output_size", 0)),
                inp=a.get("input_type_shape"),
                name=e["name"],
            )
        )
    res[name] = rows
    print("==", name, "nodes", len(rows))
    cnt = collections.Counter(r["op"] for r in rows)
    tm = collections.Counter()
    for r in rows:
        tm[r["op"]] += r["dur"]
    cpu = [r["op"] + ":" + r["name"][:50] for r in rows if "CPU" in r["prov"]]
    print("   CPU-EP nodes:", cpu)
    nod = sum(v for k, v in cnt.items() if k in NOD)
    el = sum(v for k, v in cnt.items() if k in ELEM)
    mv = sum(v for k, v in cnt.items() if k in MOVE)
    print(
        "   zero-copy(no dispatch)=%d  elementwise=%d  data-movement=%d  conv=%d  other=%d"
        % (nod, el, mv, cnt["Conv"], len(rows) - nod - el - mv - cnt["Conv"])
    )
    print(
        "   median-of-runs time (approx, ms):",
        ", ".join(
            "%s %d x%.2f" % (o, cnt[o], tm[o] / 1000)
            for o, _ in sorted(tm.items(), key=lambda x: -x[1])[:10]
        ),
    )
json.dump(res, open("an2.json", "w"))
