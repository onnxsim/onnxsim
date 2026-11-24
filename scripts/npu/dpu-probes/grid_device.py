"""Device verification of the placement grid. Every config of placement_grid.py (1x1 convs, cin and cout in {8,16,32,64}, hw in
{16,8}) is compiled, run on the NPU with infer_tool, and compared with a numpy reference. Reports pass or fail for each config,
and marks the configs where the compile-only placement rule did not hold. Run on H8 after reloc.py."""
import os
import subprocess
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R

OV = os.path.expanduser("~/openvino_git/bin/intel64/Release")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/ze-loader/lib") + ":" + OV)
D = os.path.expanduser("~/npu-dpu/roadmap/grid_dev"); os.makedirs(D, exist_ok=True)
RULE_FAILS = {(8, 8, 32), (8, 8, 64), (8, 16, 64), (8, 32, 64), (8, 64, 64)}  # (hw, cin, cout): the five exceptions in placement_grid.py


def run_one(hw, cin, cout):
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, hw, hw])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, hw, hw])
    rng = np.random.default_rng(2)
    W = rng.standard_normal((cout, cin, 1, 1)).astype(np.float16)
    x = rng.standard_normal((1, cin, hw, hw)).astype(np.float16)
    blob = R.compile_model("gdev_%d_%d_%d" % (hw, cin, cout), [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])],
                           [numpy_helper.from_array(W, "W")], inp, out)
    x.tofile(os.path.join(D, "in.bin"))
    od = os.path.join(D, "out"); os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    r = subprocess.run([os.path.join(OV, "infer_tool"), "-m", blob, "-d", "NPU", "-i", os.path.join(D, "in.bin"), "-o", od],
                       env=ENV, cwd=OV, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.listdir(od):
        return None
    got = np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16).astype(np.float64).reshape(cout, hw * hw)
    ref = np.einsum("oc,cp->op", W[:, :, 0, 0].astype(np.float64), x.astype(np.float64).reshape(cin, hw * hw))
    return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-6))


passed = 0
total = 0
for hw in (16, 8):
    for cin in (8, 16, 32, 64):
        for cout in (8, 16, 32, 64):
            total += 1
            rel = run_one(hw, cin, cout)
            tag = " (rule exception)" if (hw, cin, cout) in RULE_FAILS else ""
            if rel is None:
                print("hw %2d cin %2d cout %2d: device run failed%s" % (hw, cin, cout, tag), flush=True)
                continue
            ok = rel < 2e-3
            passed += ok
            print("hw %2d cin %2d cout %2d: rel err %.2e %s%s" % (hw, cin, cout, rel, "PASS" if ok else "FAIL", tag), flush=True)
print("device pass: %d of %d" % (passed, total), flush=True)
