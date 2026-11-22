"""cin-only sweep: single 1x1 convs with cout 16, hin 16, cin in {4, 8, 32, 64}. For each literal (non-relocated) descriptor
word, report its value per cin and whether it follows a simple law (constant, proportional to cin, or cin/4 in some bits).
Compile only. Run on H8 after reloc.py and litdiff.py."""
import os
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R
import litdiff as L

CINS = [4, 8, 32, 64]
vals = {}
for cin in CINS:
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, 16, 16])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16, 16, 16])
    w = numpy_helper.from_array((np.arange(16 * cin).reshape(16, cin, 1, 1) * 0.01).astype(np.float16), "W")
    path = R.compile_model("cin%d" % cin, [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])], [w], inp, out)
    vals[cin] = (L.sec_words(path), L.reloc_words(path))
    print("compiled cin", cin, flush=True)

for name in ("task.dpu.invariant.0.0", "task.dpu.variant.0.0"):
    n = min(len(vals[c][0][name]) for c in CINS)
    for j in range(n):
        if any(j in vals[c][1].get(name, set()) for c in CINS):
            continue
        row = [int(vals[c][0][name][j]) for c in CINS]
        if len(set(row)) == 1:
            continue
        print("%-24s word %4d  " % (name, j) + "  ".join("cin%d=0x%08x" % (c, v) for c, v in zip(CINS, row)), flush=True)
