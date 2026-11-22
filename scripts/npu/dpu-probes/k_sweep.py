"""Kernel-size sweep: single convs with cin 16, cout 16, hin 16, kernel k in {1, 3, 5}, pad k//2. Prints literal (non-relocated)
descriptor words that change with k, with values per k. Compile only. Run on H8 after reloc.py and litdiff.py."""
import os
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R
import litdiff as L

KS = [1, 3, 5]
vals = {}
for k in KS:
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 16, 16, 16])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16, 16, 16])
    w = numpy_helper.from_array((np.arange(16 * 16 * k * k).reshape(16, 16, k, k) * 0.001).astype(np.float16), "W")
    path = R.compile_model("k%d" % k, [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[k, k], pads=[k // 2] * 4)], [w], inp, out)
    vals[k] = (L.sec_words(path), L.reloc_words(path))
    print("compiled k", k, flush=True)

for name in ("task.dpu.invariant.0.0", "task.dpu.variant.0.0"):
    n = min(len(vals[k][0][name]) for k in KS)
    for j in range(n):
        if any(j in vals[k][1].get(name, set()) for k in KS):
            continue
        row = [int(vals[k][0][name][j]) for k in KS]
        if len(set(row)) == 1:
            continue
        print("%-24s word %4d  " % (name, j) + "  ".join("k%d=0x%08x" % (k, v) for k, v in zip(KS, row)), flush=True)
