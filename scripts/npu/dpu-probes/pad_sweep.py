"""Pad sweep: single convs cin 16, cout 16, hin 16. (k=3, pad 0/1/2) and (k=5, pad 0/1/2). Prints literal (non-relocated)
descriptor words that change, per configuration. Compile only. Run on H8 after reloc.py and litdiff.py."""
import os
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R
import litdiff as L

CFG = [(3, 0), (3, 1), (3, 2), (5, 0), (5, 1), (5, 2)]
vals = {}
for k, p in CFG:
    hout = 16 + 2 * p - k + 1
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 16, 16, 16])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16, hout, hout])
    w = numpy_helper.from_array((np.arange(16 * 16 * k * k).reshape(16, 16, k, k) * 0.001).astype(np.float16), "W")
    path = R.compile_model("k%dp%d" % (k, p), [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[k, k], pads=[p] * 4)], [w], inp, out)
    vals[(k, p)] = (L.sec_words(path), L.reloc_words(path))
    print("compiled k%d pad%d" % (k, p), flush=True)

for name in ("task.dpu.invariant.0.0", "task.dpu.variant.0.0"):
    n = min(len(vals[c][0][name]) for c in CFG)
    for j in range(n):
        if any(j in vals[c][1].get(name, set()) for c in CFG):
            continue
        row = [int(vals[c][0][name][j]) for c in CFG]
        if len(set(row)) == 1:
            continue
        print("%-24s word %4d  " % (name, j) + "  ".join("k%dp%d=0x%08x" % (k, p, v) for (k, p), v in zip(CFG, row)), flush=True)
