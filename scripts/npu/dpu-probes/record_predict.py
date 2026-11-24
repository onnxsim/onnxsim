"""Prediction test for the per-channel record word. Each channel's record word 0 is 0x1a100 + 32*c (32 bytes = one padded weight
row of 16 fp16). Hypothesis: adding 0x40 bytes (two rows) to channel c's word makes channel c compute with weight row c+2, same
bias: predicted output[c] = W[c+2] . x + bias[c]. Checked for channels 0 and 5 against the device. Run on H8 after reloc.py."""
import os
import struct
import subprocess
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R

OV = os.path.expanduser("~/openvino_git/bin/intel64/Release")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/ze-loader/lib") + ":" + OV)
D = os.path.expanduser("~/npu-dpu/roadmap/record"); os.makedirs(D, exist_ok=True)
C, K, HW = 8, 16, 16
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, C, HW, HW])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, K, HW, HW])
W = (np.random.default_rng(5).standard_normal((K, C, 1, 1))).astype(np.float16)
B = (np.arange(K) * 0.5 + 1.0).astype(np.float16)
x = np.random.default_rng(1).standard_normal((1, C, HW, HW)).astype(np.float16)
x.tofile(os.path.join(D, "in.bin"))
blob = R.compile_model("record_pred", [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[1, 1])],
                       [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")], inp, out)
data = bytearray(open(blob, "rb").read())
cons = {s["name"]: s for s in R.elf_sections(bytes(data))}["buffer.Constant.0.constant"]
xf = x.astype(np.float64).reshape(C, HW * HW)
Wf = W[:, :, 0, 0].astype(np.float64)
Bf = B.astype(np.float64)


def infer(buf):
    p = os.path.join(D, "p.blob"); open(p, "wb").write(bytes(buf))
    od = os.path.join(D, "out"); os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    try:
        r = subprocess.run([os.path.join(OV, "infer_tool"), "-m", p, "-d", "NPU", "-i", os.path.join(D, "in.bin"), "-o", od],
                           env=ENV, cwd=OV, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.listdir(od):
        return None
    return np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16).astype(np.float64).reshape(K, HW * HW)


for c in (0, 5):
    p = bytearray(data)
    at = cons["off"] + 320 + 16 * c
    struct.pack_into("<I", p, at, (struct.unpack_from("<I", p, at)[0] + 0x40) & 0xFFFFFFFF)
    got = infer(p)
    if got is None:
        print("channel %d: device error" % c, flush=True)
        continue
    pred = Wf[c + 2] @ xf + Bf[c]
    alt = Wf[c] @ xf + Bf[c]
    print("channel %d (+0x40): device vs predicted row c+2: max err %.4f (pred max %.3f) | vs unchanged row c: max err %.4f" % (
        c, np.abs(got[c] - pred).max(), np.abs(pred).max(), np.abs(got[c] - alt).max()), flush=True)
