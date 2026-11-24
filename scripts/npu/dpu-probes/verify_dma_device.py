"""Device check of the DMA size fields on a 1x1 conv (cin 8, cout 16, 16x16). Hypotheses:
  A. input transfer size (task.dma.0.0 word 286) halved: the device loads only the first 4 input channels. Prediction: the output
     equals W[:, :4] . x[:4] (the other channels contribute zero if CMX is zero there). Reported as match fraction.
  B. output transfer size (task.dma.0.1 word 6) halved: the device writes only the first 8 output channels. Prediction: the first 8
     output channels equal the reference exactly.
Run on H8 after reloc.py."""
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
D = os.path.expanduser("~/npu-dpu/roadmap/verify_dma"); os.makedirs(D, exist_ok=True)
C, K, HW = 8, 16, 16
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, C, HW, HW])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, K, HW, HW])
W = (np.random.default_rng(5).standard_normal((K, C, 1, 1))).astype(np.float16)
x = np.random.default_rng(1).standard_normal((1, C, HW, HW)).astype(np.float16)
x.tofile(os.path.join(D, "in.bin"))
blob = R.compile_model("verify_dma", [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])],
                       [numpy_helper.from_array(W, "W")], inp, out)
data = bytearray(open(blob, "rb").read())
secs = {s["name"]: s for s in R.elf_sections(bytes(data))}
ref_full = np.einsum("oc,cp->op", W[:, :, 0, 0].astype(np.float64), x.astype(np.float64).reshape(C, HW * HW))


def at(name, j):
    return secs[name]["off"] + 4 * j


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


ref = infer(data)
assert ref is not None, "base run failed"
print("base ok; input size field = %d, output size field = %d" % (
    struct.unpack_from("<I", data, at("task.dma.0.0", 286))[0], struct.unpack_from("<I", data, at("task.dma.0.1", 6))[0]), flush=True)

# A. input transfer halved
p = bytearray(data)
struct.pack_into("<I", p, at("task.dma.0.0", 286), (C // 2) * HW * HW * 2)
got = infer(p)
if got is None:
    print("A. input size halved: device error", flush=True)
else:
    pred = np.einsum("oc,cp->op", W[:, :C // 2, 0, 0].astype(np.float64), x[0, :C // 2].astype(np.float64).reshape(C // 2, HW * HW))
    match = float(np.isclose(got, pred, atol=2e-2).mean())
    print("A. input size halved: match with first-4-channel prediction %.4f | match with full reference %.4f" % (
        match, float(np.isclose(got, ref_full, atol=2e-2).mean())), flush=True)

# B. output transfer halved
p = bytearray(data)
struct.pack_into("<I", p, at("task.dma.0.1", 6), (K // 2) * HW * HW * 2)
got = infer(p)
if got is None:
    print("B. output size halved: device error", flush=True)
else:
    first = float(np.isclose(got[:K // 2], ref[:K // 2], atol=1e-2).mean())
    second = float(np.isclose(got[K // 2:], ref[K // 2:], atol=1e-2).mean())
    print("B. output size halved: first 8 channels match reference %.4f | last 8 channels match %.4f" % (first, second), flush=True)
