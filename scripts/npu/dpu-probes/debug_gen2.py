"""Debug the all-shapes generator on one shape (64 -> 64, 1x1, 56x56): (a) template patched with its own weights must reproduce the
template output; (b) template patched with new weights must match numpy. Run on H8 after reloc.py."""
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
D = os.path.expanduser("~/npu-dpu/roadmap/debug_gen"); os.makedirs(D, exist_ok=True)
cout, cin, k, hin, pad, stride = 64, 64, 1, 56, 0, 1
hout = hin
rng = np.random.default_rng(3)
tw = (rng.standard_normal((cout, cin, k, k)) * 0.05).astype(np.float16)
nw = (rng.standard_normal((cout, cin, k, k)) * 0.05).astype(np.float16)
nb = (rng.standard_normal(cout) * 0.5).astype(np.float32)
x = rng.standard_normal((cin, hin, hin)).astype(np.float16)
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, hin, hin])
o = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, hout, hout])
tpath = R.compile_model("dbg_t2", [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[1, 1])],
                        [numpy_helper.from_array(tw, "W"), numpy_helper.from_array((np.arange(cout) * 0.25 + 1.0).astype(np.float16), "B")], inp, o)
data = bytearray(open(tpath, "rb").read())
secs = {s["name"]: s for s in R.elf_sections(bytes(data))}
cons = secs["buffer.Constant.0.constant"]
addend = [a for t, off, sym, a in R.relocs(tpath) if t.startswith("task.dma.1.0") and "Constant" in sym][0]
G = -(-cout // 16)
Rr = -(-(cin * k * k) // 16) * 16
print("addend", addend, "G", G, "R", Rr, "cons size", cons["size"], flush=True)


def write_weights(buf, W):
    o_idx, i_idx, t_idx = np.meshgrid(np.arange(cout), np.arange(cin), np.arange(k * k), indexing="ij")
    e = 160 + 128 * G + (o_idx // 16) * 16 * Rr + (o_idx % 16) * Rr + t_idx * cin + i_idx - addend // 2
    pos = cons["off"] + addend + 2 * e
    vals = W.reshape(cout, cin, k * k).view(np.uint16)
    for p0, v in zip(pos.ravel(), vals.ravel()):
        buf[int(p0):int(p0) + 2] = int(v).to_bytes(2, "little")


def write_bias(buf, B):
    for c in range(cout):
        p0 = cons["off"] + 332 + 256 * (c // 16) + 16 * (c % 16)
        buf[p0:p0 + 4] = struct.pack("<f", float(B[c]))


def infer(buf):
    p = os.path.join(D, "p.blob"); open(p, "wb").write(bytes(buf))
    x.tofile(os.path.join(D, "in.bin"))
    od = os.path.join(D, "out"); os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    r = subprocess.run([os.path.join(OV, "infer_tool"), "-m", p, "-d", "NPU", "-i", os.path.join(D, "in.bin"), "-o", od],
                       env=ENV, cwd=OV, capture_output=True, text=True, timeout=600)
    return np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16).astype(np.float64).reshape(cout, -1)


# (a) template with its own weights, bias zero
a = bytearray(data)
write_weights(a, tw)
got_a = infer(a)
ref_a = np.einsum("oc,cp->op", tw[:, :, 0, 0].astype(np.float64), x.astype(np.float64).reshape(cin, -1))
print("(a) template self-patch vs numpy: max err %.4f (ref max %.3f)" % (np.abs(got_a - ref_a).max(), np.abs(ref_a).max()), flush=True)
# (b) new weights and bias
b = bytearray(data)
write_weights(b, nw)
write_bias(b, nb)
got_b = infer(b)
ref_b = np.einsum("oc,cp->op", nw[:, :, 0, 0].astype(np.float64), x.astype(np.float64).reshape(cin, -1)) + nb.astype(np.float64)[:, None]
print("(b) new weights+bias vs numpy: max err %.4f (ref max %.3f)" % (np.abs(got_b - ref_b).max(), np.abs(ref_b).max()), flush=True)

# (c) new weights, zero bias  |  (d) old weights, new bias
c = bytearray(data)
write_weights(c, nw)
got_c = infer(c)
ref_c = np.einsum("oc,cp->op", nw[:, :, 0, 0].astype(np.float64), x.astype(np.float64).reshape(cin, -1))
print("(c) new weights, zero bias: max err %.4f (ref max %.3f)" % (np.abs(got_c - ref_c).max(), np.abs(ref_c).max()), flush=True)
d = bytearray(data)
write_bias(d, nb)
got_d = infer(d)
ref_d = np.einsum("oc,cp->op", tw[:, :, 0, 0].astype(np.float64), x.astype(np.float64).reshape(cin, -1)) + nb.astype(np.float64)[:, None]
err_per_group = [float(np.abs(got_d[g * 16:(g + 1) * 16] - ref_d[g * 16:(g + 1) * 16]).max()) for g in range(4)]
print("(d) old weights, new bias: max err %.4f | per 16-channel group %s" % (np.abs(got_d - ref_d).max(), [round(e, 3) for e in err_per_group]), flush=True)
