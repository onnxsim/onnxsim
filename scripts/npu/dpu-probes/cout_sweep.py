"""cout-only sweep: single 1x1 convs with cin 16, hin 16, cout in {4, 8, 16, 32, 64}. Prints every literal (non-relocated)
descriptor word that changes with cout, with its value per cout. Compile only. Run on H8 after reloc.py and litdiff.py."""
import os
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R
import litdiff as L

COUTS = [4, 8, 16, 32, 64]
vals = {}
for cout in COUTS:
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 16, 16, 16])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, 16, 16])
    w = numpy_helper.from_array((np.arange(cout * 16).reshape(cout, 16, 1, 1) * 0.01).astype(np.float16), "W")
    path = R.compile_model("cout%d" % cout, [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])], [w], inp, out)
    vals[cout] = (L.sec_words(path), L.reloc_words(path))
    print("compiled cout", cout, flush=True)

for name in ("task.dpu.invariant.0.0", "task.dpu.variant.0.0"):
    n = min(len(vals[c][0][name]) for c in COUTS)
    for j in range(n):
        if any(j in vals[c][1].get(name, set()) for c in COUTS):
            continue
        row = [int(vals[c][0][name][j]) for c in COUTS]
        if len(set(row)) == 1:
            continue
        print("%-24s word %4d  " % (name, j) + "  ".join("co%d=0x%08x" % (c, v) for c, v in zip(COUTS, row)), flush=True)
