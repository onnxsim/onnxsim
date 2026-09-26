"""Rank the driving model's backbone windows by the damage of dropping only them from uint16 to uint8 activations
(head kept float). Produced policy_driving_mixed.json: windows 4-11 uint8, the rest uint16. Run from the work
directory holding driving_fp32.onnx, driving_supercombo.onnx and inputs_seg{3,8,12}.npz. Arg: window size (8)."""

import json
import sys

import onnx
import precision_search
import quantize
import run_models

from onnxsim.calibration_pick import run_outputs
from onnxsim.full_qdq import quantize_full_qdq

m = onnx.load("driving_fp32.onnx")
nodes = list(m.graph.node)
last = max(i for i, n in enumerate(nodes) if n.op_type == "Conv")
head = [n.name or n.output[0] for n in nodes[last + 1 :]]
cal = quantize.driving_samples(
    "driving_fp32.onnx", "inputs_seg3.npz,inputs_seg12.npz", 48, 24
)
ev = quantize.driving_samples("driving_fp32.onnx", "inputs_seg12.npz", 32, 18)
metric = precision_search.driving_metric(
    run_models.output_slices("driving_supercombo.onnx")
)
fo = run_outputs(m, ev)
# calibrate once
probe = quantize_full_qdq(m, cal, activation_dtype="uint16", exclude_nodes=head)
rng = None
bb = [
    i
    for i, n in enumerate(nodes[: last + 1])
    if n.op_type in ("Conv", "Gelu", "Add", "Mul", "Sub", "Div", "Concat")
]
W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
groups = [bb[i : i + W] for i in range(0, len(bb), W)]


def score(u8_idx):
    td = {o: "uint8" for i in u8_idx for o in nodes[i].output}
    q = quantize_full_qdq(
        m, cal, activation_dtype="uint16", exclude_nodes=head, tensor_dtypes=td
    )
    return metric(fo, run_outputs(q, ev))


base = score([])
print("all-u16 backbone", base, flush=True)
print("all-u8 backbone", score(bb), flush=True)
res = []
for gi, gr in enumerate(groups):
    s = score(gr)
    res.append((s, gi, [nodes[i].output[0] for i in gr]))
    print(
        f"group {gi} ({nodes[gr[0]].output[0]} .. {nodes[gr[-1]].output[0]}): {s:.2f}",
        flush=True,
    )
json.dump(res, open("drv_group_sweep.json", "w"))
