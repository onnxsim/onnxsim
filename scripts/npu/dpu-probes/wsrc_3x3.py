"""Weight source model, 3x3 (device). Conv 3x3 pad 1, 8 -> 32 channels, 16x16. The padded row is R = roundup(cin*9, 16) = 80 fp16 =
160 bytes. Shifting the weight source by S bytes (multiple of 160): record shift S/16, row shift S/160. Predicted
output[c] = W[c + S/16 + S/160] (3x3, zero padding) + bias[c + S/16]. Tested for S = 160 and S = 320, plus a non-multiple
S = 32 as a falsification (the row shift is fractional there, so the prediction should not hold). Run on H8."""
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
D = os.path.expanduser("~/npu-dpu/roadmap/wsrc_3x3"); os.makedirs(D, exist_ok=True)
C, K, HW = 8, 32, 16
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, C, HW, HW])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, K, HW, HW])
W = (np.random.default_rng(5).standard_normal((K, C, 3, 3))).astype(np.float16)
B = (np.arange(K) * 0.25 + 1.0).astype(np.float16)
x = np.random.default_rng(1).standard_normal((1, C, HW, HW)).astype(np.float16)
x.tofile(os.path.join(D, "in.bin"))
blob = R.compile_model("wsrc_3x3", [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
                       [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")], inp, out)
data = bytearray(open(blob, "rb").read())
secs = {s["name"]: s for s in R.elf_sections(bytes(data))}
rela_sec = secs["rela.task.dma.1.0.symtab"]
entry = None
for k in range(rela_sec["size"] // 24):
    e = rela_sec["off"] + 24 * k
    if struct.unpack_from("<Q", data, e)[0] == 40 and struct.unpack_from("<q", data, e + 16)[0] == 320:
        entry = e + 16
assert entry is not None, "weight relocation not found"
xp = np.pad(x.astype(np.float64)[0], ((0, 0), (1, 1), (1, 1)))
Wf = W.astype(np.float64)
Bf = B.astype(np.float64)


def ref_row(r, b):
    out_ = np.zeros((HW, HW))
    for i in range(3):
        for j in range(3):
            out_ += np.einsum("c,chw->hw", Wf[r, :, i, j], xp[:, i:i + HW, j:j + HW])
    return (out_ + Bf[b]).ravel()


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


for S in (160, 320, 32):
    p = bytearray(data)
    struct.pack_into("<q", p, entry, 320 + S)
    got = infer(p)
    if got is None:
        print("S=%d: device error" % S, flush=True)
        continue
    rec = S // 16
    rows = S / 160
    ok, bad = [], []
    for c in range(K):
        b = c + rec
        if b >= K or rows != int(rows):
            continue
        r = c + rec + int(rows)
        if r >= K:
            continue
        err = np.abs(got[c] - ref_row(r, b)).max()
        (ok if err < 0.05 else bad).append((c, round(float(err), 3)))
    if rows != int(rows):
        frac_ok = float(np.mean([np.abs(got[c] - ref_row(c + rec, c + rec)).max() < 0.05 for c in range(K - rec)]))
        print("S=%d (fractional row shift %.2f): channels matching the integer-row prediction %.2f" % (S, rows, frac_ok), flush=True)
    else:
        print("S=%d: predicted channels matching %d, mismatching %d %s" % (S, len(ok), len(bad), bad[:4]), flush=True)
