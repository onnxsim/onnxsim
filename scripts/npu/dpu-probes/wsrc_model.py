"""Weight source model (device). Shifting the weight DMA source by S bytes (the relocation addend, 320 + S) moves the constant bytes
that land in CMX. The model: the device reads each channel's record at its absolute CMX address, which now holds the record S/16
channels ahead, and each weight row at its absolute address, which now holds the row S/32 rows ahead (1x1, 32-byte padded row).
Predicted: output[c] = W[c + S/16 + S/32] . x + bias[c + S/16] for the channels whose rows exist. Tested for S = 64 and S = 96
on a 1x1 conv, 8 -> 32 channels, 16x16. Channels whose rows fall outside the transfer are reported separately. Run on H8."""
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
D = os.path.expanduser("~/npu-dpu/roadmap/wsrc_model"); os.makedirs(D, exist_ok=True)
C, K, HW = 8, 32, 16
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, C, HW, HW])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, K, HW, HW])
W = (np.random.default_rng(5).standard_normal((K, C, 1, 1))).astype(np.float16)
B = (np.arange(K) * 0.25 + 1.0).astype(np.float16)
x = np.random.default_rng(1).standard_normal((1, C, HW, HW)).astype(np.float16)
x.tofile(os.path.join(D, "in.bin"))
blob = R.compile_model("wsrc_model", [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[1, 1])],
                       [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")], inp, out)
data = bytearray(open(blob, "rb").read())
rela = [(off, sym, a, t) for t, off, sym, a in R.relocs(blob) if t.startswith("task.dma.1.0") and "Constant" in sym]
secs = {s["name"]: s for s in R.elf_sections(bytes(data))}
rela_sec = secs["rela.task.dma.1.0.symtab"]
entry = None
for k in range(rela_sec["size"] // 24):
    e = rela_sec["off"] + 24 * k
    if struct.unpack_from("<Q", data, e)[0] == 40 and struct.unpack_from("<q", data, e + 16)[0] == 320:
        entry = e + 16
assert entry is not None, "weight relocation not found"
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


for S in (64, 96):
    p = bytearray(data)
    struct.pack_into("<q", p, entry, 320 + S)
    got = infer(p)
    if got is None:
        print("S=%d: device error" % S, flush=True)
        continue
    rec, row = S // 16, S // 32
    ok_ch, bad_ch = [], []
    for c in range(K):
        r = c + rec + row
        b = c + rec
        if r >= K or b >= K:
            continue
        pred = Wf[r] @ xf + Bf[b]
        err = np.abs(got[c] - pred).max()
        (ok_ch if err < 0.05 else bad_ch).append((c, round(float(err), 3)))
    print("S=%d: predicted channels matching %d, mismatching %d %s" % (S, len(ok_ch), len(bad_ch), bad_ch[:4]), flush=True)
