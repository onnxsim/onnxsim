"""Placement test: the same 8->16->16 chain with (a) a Relu on the input and (b) a Relu between the two convs. Sizes of the
intermediate and the output stay fixed; only the op sequence changes. Diff literal, non-relocated descriptor words against
the plain chain. Compile only. Run on H8 after reloc.py and litdiff.py."""
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
var_in = R.compile_model("relu_in", [helper.make_node("Relu", ["x"], ["r"]),
                                     helper.make_node("Conv", ["r", "W1"], ["m"], kernel_shape=[1, 1]),
                                     helper.make_node("Conv", ["m", "W2"], ["y"], kernel_shape=[1, 1])], [w1, w2], inp, y)
var_mid = R.compile_model("relu_mid", [helper.make_node("Conv", ["x", "W1"], ["m"], kernel_shape=[1, 1]),
                                       helper.make_node("Relu", ["m"], ["r"]),
                                       helper.make_node("Conv", ["r", "W2"], ["y"], kernel_shape=[1, 1])], [w1, w2], inp, y)
base = L.sec_words(L.chain)
rb = L.reloc_words(L.chain)
for label, path in (("relu on input", var_in), ("relu between convs", var_mid)):
    mod = L.sec_words(path)
    rm = L.reloc_words(path)
    print("== %s" % label, flush=True)
    for name in sorted(set(base) & set(mod)):
        a, b = base[name], mod[name]
        n = min(len(a), len(b))
        diff = [j for j in range(n) if a[j] != b[j] and j not in rb.get(name, set()) and j not in rm.get(name, set())]
        print("  %-26s words %d/%d literal diffs %d" % (name, len(a), len(b), len(diff)), flush=True)
        for j in diff[:12]:
            print("     word %4d  plain 0x%08x  variant 0x%08x" % (j, a[j], b[j]), flush=True)
