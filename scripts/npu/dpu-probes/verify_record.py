"""Verification of the per-channel weight record (device). Cases:
  A. 1x1, cin 16, cout 32 (two groups): channel 20 (group 1, c=4). Record base = 320 + 256*(o//16) + 16*(o%16) = 640.
     Edit its word 0 by +0x40 (two padded rows of 16 fp16 = 32 B each). Prediction: channel 20 = W[22] . x + bias[20].
  B. 3x3 pad 1, cin 8, cout 16: channel 5. Row stride = R*2 with R = roundup(cin*9, 16) = 80, so 160 B. Edit word 0 by +160 B.
     Prediction: 3x3 conv of x with W[6] + bias[5] on channel 5 (zero padding).
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
D = os.path.expanduser("~/npu-dpu/roadmap/verify_record"); os.makedirs(D, exist_ok=True)


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
    return np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16).astype(np.float64)


def case_a():
    C, K, HW = 16, 32, 16
    W = np.random.default_rng(3).standard_normal((K, C, 1, 1)).astype(np.float16)
    B = (np.arange(K) * 0.25 + 1.0).astype(np.float16)
    x = np.random.default_rng(4).standard_normal((1, C, HW, HW)).astype(np.float16)
    x.tofile(os.path.join(D, "in.bin"))
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, C, HW, HW])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, K, HW, HW])
    blob = R.compile_model("vr_a", [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[1, 1])],
                           [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")], inp, out)
    data = bytearray(open(blob, "rb").read())
    cons = {s["name"]: s for s in R.elf_sections(bytes(data))}["buffer.Constant.0.constant"]
    ref = infer(data)
    o = 20
    base = 320 + 256 * (o // 16) + 16 * (o % 16)
    p = bytearray(data)
    at = cons["off"] + base
    struct.pack_into("<I", p, at, (struct.unpack_from("<I", p, at)[0] + 0x40) & 0xFFFFFFFF)
    got = infer(p)
    xf = x.astype(np.float64).reshape(C, HW * HW)
    pred = W[22, :, 0, 0].astype(np.float64) @ xf + float(B[o])
    alt = W[o, :, 0, 0].astype(np.float64) @ xf + float(B[o])
    changed = [int(k) for k in np.nonzero(np.abs(got.reshape(K, -1) - ref.reshape(K, -1)).max(axis=1) > 1e-2)[0]]
    print("A. channel 20 record @%d: changed channels %s | device vs row 22: %.4f (pred max %.2f) | vs row 20: %.4f" % (
        base, changed, np.abs(got.reshape(K, -1)[o] - pred).max(), np.abs(pred).max(), np.abs(got.reshape(K, -1)[o] - alt).max()), flush=True)


def case_b():
    C, K, HW = 8, 16, 16
    W = np.random.default_rng(5).standard_normal((K, C, 3, 3)).astype(np.float16)
    B = (np.arange(K) * 0.5 + 1.0).astype(np.float16)
    x = np.random.default_rng(6).standard_normal((1, C, HW, HW)).astype(np.float16)
    x.tofile(os.path.join(D, "in.bin"))
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, C, HW, HW])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, K, HW, HW])
    blob = R.compile_model("vr_b", [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
                           [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")], inp, out)
    data = bytearray(open(blob, "rb").read())
    cons = {s["name"]: s for s in R.elf_sections(bytes(data))}["buffer.Constant.0.constant"]
    ref = infer(data)
    c = 5
    p = bytearray(data)
    at = cons["off"] + 320 + 16 * c
    struct.pack_into("<I", p, at, (struct.unpack_from("<I", p, at)[0] + 160) & 0xFFFFFFFF)
    got = infer(p)
    xp = np.pad(x.astype(np.float64)[0], ((0, 0), (1, 1), (1, 1)))
    Wr = W[c + 1].astype(np.float64)
    pred = np.zeros((HW, HW))
    for i in range(3):
        for j in range(3):
            pred += np.einsum("c,chw->hw", Wr[:, i, j], xp[:, i:i + HW, j:j + HW])
    pred += float(B[c])
    alt = np.zeros((HW, HW))
    Wa = W[c].astype(np.float64)
    for i in range(3):
        for j in range(3):
            alt += np.einsum("c,chw->hw", Wa[:, i, j], xp[:, i:i + HW, j:j + HW])
    alt += float(B[c])
    changed = [int(k) for k in np.nonzero(np.abs(got.reshape(K, -1) - ref.reshape(K, -1)).max(axis=1) > 1e-2)[0]]
    g = got.reshape(K, HW, HW)[c]
    print("B. 3x3 channel 5 record +160 B: changed channels %s | device vs row 6: %.4f (pred max %.2f) | vs row 5: %.4f" % (
        changed, np.abs(g - pred).max(), np.abs(pred).max(), np.abs(g - alt).max()), flush=True)


case_a()
case_b()
