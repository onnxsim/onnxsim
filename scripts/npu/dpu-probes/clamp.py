"""Clamp encoding test: Clip(0,6) and Clip(-1,2) between the two 1x1 convs, compared with the Relu-between variant.
Compile only. Run on H8 after reloc.py, litdiff.py, placement.py (uses the saved relu_mid blob)."""
import os
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R
import litdiff as L

inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8, 16, 16])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16, 16, 16])
w1 = numpy_helper.from_array((np.arange(16 * 8).reshape(16, 8, 1, 1) * 0.01).astype(np.float16), "W1")
w2 = numpy_helper.from_array((np.arange(16 * 16).reshape(16, 16, 1, 1) * 0.01).astype(np.float16), "W2")


def clip_model(name, lo, hi):
    inits = [w1, w2, numpy_helper.from_array(np.array(lo, np.float16), "lo"), numpy_helper.from_array(np.array(hi, np.float16), "hi")]
    nodes = [helper.make_node("Conv", ["x", "W1"], ["m"], kernel_shape=[1, 1]),
             helper.make_node("Clip", ["m", "lo", "hi"], ["r"]),
             helper.make_node("Conv", ["r", "W2"], ["y"], kernel_shape=[1, 1])]
    return R.compile_model(name, nodes, inits, inp, y)


relu = os.path.expanduser("~/npu-dpu/roadmap/reloc/relu_mid.blob")
c06 = clip_model("clip06", 0.0, 6.0)
cm12 = clip_model("clip_m1_2", -1.0, 2.0)
base = L.sec_words(relu)
rb = L.reloc_words(relu)
for label, path in (("Clip(0,6)", c06), ("Clip(-1,2)", cm12)):
    mod = L.sec_words(path)
    rm = L.reloc_words(path)
    print("== %s vs relu_mid" % label, flush=True)
    for name in ("task.dpu.invariant.0.0", "task.dpu.variant.0.0"):
        a, b = base[name], mod[name]
        n = min(len(a), len(b))
        diff = [j for j in range(n) if a[j] != b[j] and j not in rb.get(name, set()) and j not in rm.get(name, set())]
        for j in diff:
            print("   %-24s word %4d  relu 0x%08x  %s 0x%08x" % (name, j, a[j], label, b[j]), flush=True)
