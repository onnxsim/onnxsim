#!/usr/bin/env python3
"""Build one single-node RoiAlign ONNX model per real call (same opset/attributes as the real
rest.onnx node) plus .npz inputs, and time each in ONNX Runtime CPU -- the op's current baseline,
since the whole rest.onnx remainder runs on ORT CPU today. Models/inputs are also what the phone-side
ORT timing harness (ort_roialign_bench.c) loads."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
d = np.load(src / "roi_real.npz")
meta = json.load(open(src / "roi_meta.json"))
rows = []
for i, n in enumerate(meta["nodes"]):
    feat, rois, bidx = (d[k] for k in n["inputs"])
    a = n["attrs"]
    node = helper.make_node("RoiAlign", ["X", "rois", "batch_indices"], ["Y"], **a)
    g = helper.make_graph([node], "roialign", [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, list(feat.shape)),
        helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(rois.shape)),
        helper.make_tensor_value_info("batch_indices", TensorProto.INT64, list(bidx.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12)])
    m.ir_version = 7
    onnx.save(m, out / f"call{i}.onnx")
    feat.astype(np.float32).tofile(out / f"call{i}_X.bin")
    rois.astype(np.float32).tofile(out / f"call{i}_rois_nchw.bin")
    bidx.astype(np.int64).tofile(out / f"call{i}_bidx.bin")
    res = {}
    for th in (1, 0):
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        s = ort.InferenceSession(str(out / f"call{i}.onnx"), so, providers=["CPUExecutionProvider"])
        feeds = {"X": feat, "rois": rois, "batch_indices": bidx}
        y = s.run(None, feeds)[0]
        assert np.array_equal(y, d[n["output"]])
        ts = []
        for _ in range(7):
            t = time.perf_counter(); s.run(None, feeds); ts.append(time.perf_counter() - t)
        res[th] = sorted(ts)[3] * 1e3
    rows.append((i, feat.shape, rois.shape[0], a["output_height"], res[1], res[0]))
    print(f"call{i} {feat.shape[2]}x{feat.shape[3]} R={rois.shape[0]} out={a['output_height']} ort_1thr={res[1]:.2f}ms ort_default={res[0]:.2f}ms")
print(f"TOTAL ort_1thr={sum(r[4] for r in rows):.2f}ms ort_default={sum(r[5] for r in rows):.2f}ms")
