#!/usr/bin/env python3
"""Run one real image through the full MaskRCNN-12-qdq model in ONNX Runtime and capture every
RoiAlign node's real inputs (feature map, RoIs, batch indices) and output, plus the node attributes,
into roi_real.npz / roi_meta.json -- the ground truth gen_roialign_test_data.py and
make_roialign_single_node_models.py consume.

    python dump_real_roialign_io.py --model MaskRCNN-12-qdq.onnx --image 000000000139.jpg --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maskrcnn_e2e"))
from eval_common import canvas  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--image", required=True)
p.add_argument("--out", type=Path, default=Path("."))
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def key(name):
    return name.replace("/", "_").replace(":", "_")


def attr(a):
    v = onnx.helper.get_attribute_value(a)
    return v.decode() if isinstance(v, bytes) else v


m = onnx.load(args.model)
ras = [n for n in m.graph.node if n.op_type == "RoiAlign"]
names = []
for n in ras:
    for i in [*n.input, n.output[0]]:
        if i not in names:
            names.append(i)
existing = {o.name for o in m.graph.output}
for nm in names:
    if nm not in existing:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
img = canvas(Path(args.image), 800, 1088)
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
outs = s.run(names, {s.get_inputs()[0].name: img})
np.savez(args.out / "roi_real.npz", **{key(nm): v for nm, v in zip(names, outs)})
meta = {"nodes": [{"name": n.name, "inputs": [key(i) for i in n.input], "output": key(n.output[0]),
                   "attrs": {a.name: attr(a) for a in n.attribute}} for n in ras]}
json.dump(meta, open(args.out / "roi_meta.json", "w"), indent=1)
for n in meta["nodes"]:
    print(n["name"], n["attrs"], [outs[names.index(i)].shape for i in ras[meta["nodes"].index(n)].input])
