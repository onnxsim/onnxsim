"""End-to-end test of our descriptor rules on the device. A compiled template (3x3, pad 1, cin 16, cout 32, 16x16) is patched with
NEW weights and biases using only the rules we verified:
  - weights: constant-buffer slot for (o, i, tap) = 160 + 128*G + (o//16)*16*R + (o%16)*R + tap*cin + i, minus addend/2 (stream),
    with G = ceil(cout/16), R = roundup(cin*9, 16), addend = 320 bytes (weight transfer source)
  - bias: fp32 per output channel at byte 332 + 256*(o//16) + 16*(o%16) of the constant buffer
  - post-scale = 1.0 (invariant word 132, keep) ; clamp upper max (word 128 keep), clamp lower = -max (word 129 keep)
Run 1: no activation. Run 2: lower clamp = 0 (Relu). Compared with numpy. Run on H8 after reloc.py."""
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
D = os.path.expanduser("~/npu-dpu/roadmap/gen_conv"); os.makedirs(D, exist_ok=True)
CIN, COUT, HW = 16, 32, 16

inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, CIN, HW, HW])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, COUT, HW, HW])
W0 = (np.random.default_rng(1).standard_normal((COUT, CIN, 3, 3))).astype(np.float16)
B0 = (np.random.default_rng(2).standard_normal(COUT)).astype(np.float32)
template = R.compile_model("gen_template", [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[3, 3], pads=[1] * 4)],
                           [numpy_helper.from_array(W0, "W"), numpy_helper.from_array(B0.astype(np.float16), "B")], inp, out)
data = bytearray(open(template, "rb").read())
secs = {s["name"]: s for s in R.elf_sections(bytes(data))}
cons = secs["buffer.Constant.0.constant"]
addend = [a for t, off, sym, a in R.relocs(template) if t.startswith("task.dma.1.0") and "Constant" in sym][0]
G = -(-COUT // 16)
Rr = -(-(CIN * 9) // 16) * 16

rng = np.random.default_rng(7)
W = (rng.standard_normal((COUT, CIN, 3, 3)) * 0.5).astype(np.float16)
B = (rng.standard_normal(COUT) * 2).astype(np.float32)
X = rng.standard_normal((1, CIN, HW, HW)).astype(np.float16)
X.tofile(os.path.join(D, "in.bin"))


def build(relu):
    p = bytearray(data)
    # weights
    for o in range(COUT):
        for i in range(CIN):
            for kh in range(3):
                for kw in range(3):
                    slot = 160 + 128 * G + (o // 16) * 16 * Rr + (o % 16) * Rr + (kh * 3 + kw) * CIN + i
                    stream = slot - addend // 2
                    at = cons["off"] + addend + 2 * stream
                    p[at:at + 2] = W[o, i, kh, kw].tobytes()
    # bias records (fp32) and clamp / scale words
    for o in range(COUT):
        at = cons["off"] + 332 + 256 * (o // 16) + 16 * (o % 16)
        p[at:at + 4] = struct.pack("<f", float(B[o]))
    inv = secs["task.dpu.invariant.0.0"]
    if relu:
        struct.pack_into("<f", p, inv["off"] + 4 * 129, 0.0)
    return p


def infer(buf):
    path = os.path.join(D, "p.blob"); open(path, "wb").write(bytes(buf))
    od = os.path.join(D, "out"); os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    try:
        r = subprocess.run([os.path.join(OV, "infer_tool"), "-m", path, "-d", "NPU", "-i", os.path.join(D, "in.bin"), "-o", od],
                           env=ENV, cwd=OV, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.listdir(od):
        return None
    return np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16).astype(np.float64).reshape(COUT, HW * HW)


xp = np.pad(X.astype(np.float64)[0], ((0, 0), (1, 1), (1, 1)))
ref = np.zeros((COUT, HW, HW))
for i in range(3):
    for j in range(3):
        ref += np.einsum("oc,chw->ohw", W[:, :, i, j].astype(np.float64), xp[:, i:i + HW, j:j + HW])
ref += B.astype(np.float64)[:, None, None]
for relu in (False, True):
    got = infer(build(relu))
    if got is None:
        print("relu=%s: device error" % relu, flush=True)
        continue
    expect = np.maximum(ref, 0) if relu else ref
    expect = expect.reshape(COUT, -1)
    err = np.abs(got - expect).max()
    print("relu=%s: max err vs numpy %.4f (ref max %.2f) -> %s" % (relu, err, np.abs(expect).max(), "PASS" if err < 0.05 else "FAIL"), flush=True)
