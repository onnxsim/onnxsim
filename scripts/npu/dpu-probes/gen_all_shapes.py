"""Generator validity over ResNet50's conv shapes (device). For each of the 23 unique conv shapes (cout, cin, k, hin, pad, stride):
compile a template with OpenVINO, write NEW weights and biases into it with our layout rules (vectorised), run on the NPU, and
compare with numpy. Layout rules used:
  weight slot stream index e(o, i, tap) = 160 + 128*G + (o//16)*16*R + (o%16)*R + tap*cin + i - addend/2   (G=ceil(cout/16), R=roundup(cin*k*k,16))
  bias fp32 at constant byte 332 + 256*(o//16) + 16*(o%16)
  addend = relocation addend of the weight transfer (read from the blob, not assumed).
Run on H8 after reloc.py. Prints one line per shape: PASS / FAIL with the error, or the failure."""
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
D = os.path.expanduser("~/npu-dpu/roadmap/gen_all"); os.makedirs(D, exist_ok=True)
SHAPES = [(64, 3, 7, 224, 3, 2), (64, 64, 1, 56, 0, 1), (64, 64, 3, 56, 1, 1), (64, 256, 1, 56, 0, 1), (128, 128, 3, 28, 1, 1),
          (128, 128, 3, 56, 1, 2), (128, 256, 1, 56, 0, 1), (128, 512, 1, 28, 0, 1), (256, 64, 1, 56, 0, 1), (256, 256, 3, 14, 1, 1),
          (256, 256, 3, 28, 1, 2), (256, 512, 1, 28, 0, 1), (256, 1024, 1, 14, 0, 1), (512, 128, 1, 28, 0, 1), (512, 256, 1, 56, 0, 2),
          (512, 512, 3, 7, 1, 1), (512, 512, 3, 14, 1, 2), (512, 1024, 1, 14, 0, 1), (512, 2048, 1, 7, 0, 1), (1024, 256, 1, 14, 0, 1),
          (1024, 512, 1, 28, 0, 2), (2048, 512, 1, 7, 0, 1), (2048, 1024, 1, 14, 0, 2)]


def infer(path, x):
    x.tofile(os.path.join(D, "in.bin"))
    od = os.path.join(D, "out"); os.makedirs(od, exist_ok=True)
    for f in os.listdir(od):
        os.remove(os.path.join(od, f))
    try:
        r = subprocess.run([os.path.join(OV, "infer_tool"), "-m", path, "-d", "NPU", "-i", os.path.join(D, "in.bin"), "-o", od],
                           env=ENV, cwd=OV, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.listdir(od):
        return None
    return np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16)


def conv_ref(x, W, B, k, pad, stride):
    xp = np.pad(x.astype(np.float64), ((0, 0), (pad, pad), (pad, pad)))
    cout, cin = W.shape[0], W.shape[1]
    hout = (x.shape[1] + 2 * pad - k) // stride + 1
    out = np.zeros((cout, hout, hout))
    for i in range(k):
        for j in range(k):
            patch = xp[:, i:i + stride * hout:stride, j:j + stride * hout:stride]
            out += np.einsum("oc,chw->ohw", W[:, :, i, j].astype(np.float64), patch)
    return out + B.astype(np.float64)[:, None, None]


rng = np.random.default_rng(11)
for cout, cin, k, hin, pad, stride in SHAPES:
    tag = "%d_%d_%d_%d_%d_%d" % (cout, cin, k, hin, pad, stride)
    try:
        hout = (hin + 2 * pad - k) // stride + 1
        W = (rng.standard_normal((cout, cin, k, k)) * 0.05).astype(np.float16)
        B = (rng.standard_normal(cout) * 0.5).astype(np.float32)
        x = rng.standard_normal((cin, hin, hin)).astype(np.float16)
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, hin, hin])
        o = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, hout, hout])
        tw = (rng.standard_normal((cout, cin, k, k)) * 0.05).astype(np.float16)
        tb = (np.arange(cout) * 0.25 + 1.0).astype(np.float16)  # nonzero: a zero-bias template has no bias records to patch
        tpath = R.compile_model("gen_t_" + tag, [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[k, k], pads=[pad] * 4,
                                                                  strides=[stride, stride])],
                                [numpy_helper.from_array(tw, "W"), numpy_helper.from_array(tb, "B")], inp, o)
        data = bytearray(open(tpath, "rb").read())
        secs = {s["name"]: s for s in R.elf_sections(bytes(data))}
        cons = secs["buffer.Constant.0.constant"]
        addend = [a for t, off, sym, a in R.relocs(tpath) if t.startswith("task.dma.1.0") and "Constant" in sym][0]
        G = -(-cout // 16)
        Rr = -(-(cin * k * k) // 16) * 16
        o_idx, i_idx, t_idx = np.meshgrid(np.arange(cout), np.arange(cin), np.arange(k * k), indexing="ij")
        e = 160 + 128 * G + (o_idx // 16) * 16 * Rr + (o_idx % 16) * Rr + t_idx * cin + i_idx - addend // 2
        byte_pos = cons["off"] + addend + 2 * e
        vals = W.reshape(cout, cin, k * k)
        flat_pos = byte_pos.ravel()
        flat_vals = vals.view(np.uint16).ravel()
        for n in range(0, flat_pos.size):
            p0 = int(flat_pos[n])
            data[p0:p0 + 2] = int(flat_vals[n]).to_bytes(2, "little")
        bias_pos = cons["off"] + 332 + 256 * (np.arange(cout) // 16) + 16 * (np.arange(cout) % 16)
        for c in range(cout):
            p0 = int(bias_pos[c])
            data[p0:p0 + 4] = struct.pack("<f", float(B[c]))
        path = os.path.join(D, "p_%s.blob" % tag); open(path, "wb").write(bytes(data))
        got = infer(path, x)
        if got is None:
            print("%-28s device error" % tag, flush=True)
            continue
        ref = conv_ref(x.astype(np.float64), W, B, k, pad, stride)
        g = got.astype(np.float64).reshape(cout, hout, hout)
        err = np.abs(g - ref).max() / max(np.abs(ref).max(), 1e-6)
        print("%-28s rel err %.2e -> %s" % (tag, err, "PASS" if err < 2e-2 else "FAIL"), flush=True)
    except Exception as ex:
        print("%-28s ERROR %s" % (tag, str(ex).splitlines()[-1][:90]), flush=True)
