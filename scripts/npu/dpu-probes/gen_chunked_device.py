"""Generated chunked conv (device). For each ResNet50 conv shape, the conv is split into sub-launches with chunked2.pick (sizes that
pass the single-launch limits). Each sub-launch is a template of its own shape, patched with the rules (weights in the record
layout, bias records zeroed), run on the NPU, and the partial sums are added on the host; the bias is added once on the host.
Compared with numpy. Shapes that failed as single launches are rerun chunked. Run on H8 (needs ~/chunked2.py and reloc.py)."""
import os
import struct
import subprocess
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R
import chunked2 as ch

OV = os.path.expanduser("~/openvino_git/bin/intel64/Release")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/ze-loader/lib") + ":" + OV)
D = os.path.expanduser("~/npu-dpu/roadmap/gen_chunked"); os.makedirs(D, exist_ok=True)
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
    r = subprocess.run([os.path.join(OV, "infer_tool"), "-m", path, "-d", "NPU", "-i", os.path.join(D, "in.bin"), "-o", od],
                       env=ENV, cwd=OV, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.listdir(od):
        return None
    return np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16).astype(np.float64)


def sub_blob(ki, ko, k, hin, pad, stride, Wsub, tag):
    """Generated sub-launch: template of this sub-shape with nonzero bias, weights patched by the rules, bias records zeroed."""
    hout = (hin + 2 * pad - k) // stride + 1
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, ki, hin, hin])
    o = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, ko, hout, hout])
    tw = np.ones((ko, ki, k, k), np.float16) * 0.01
    tb = (np.arange(ko) * 0.25 + 1.0).astype(np.float16)
    path = R.compile_model("gc_" + tag, [helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[k, k], pads=[pad] * 4,
                                                           strides=[stride, stride])],
                           [numpy_helper.from_array(tw, "W"), numpy_helper.from_array(tb, "B")], inp, o)
    data = bytearray(open(path, "rb").read())
    secs = {s["name"]: s for s in R.elf_sections(bytes(data))}
    cons = secs["buffer.Constant.0.constant"]
    addend = [a for t, off, sym, a in R.relocs(path) if t.startswith("task.dma.1.0") and "Constant" in sym][0]
    G = -(-ko // 16)
    Rr = -(-(ki * k * k) // 16) * 16
    o_i, i_i, t_i = np.meshgrid(np.arange(ko), np.arange(ki), np.arange(k * k), indexing="ij")
    e = 160 + 128 * G + (o_i // 16) * 16 * Rr + (o_i % 16) * Rr + t_i * ki + i_i - addend // 2
    pos = (cons["off"] + addend + 2 * e).ravel()
    vals = Wsub.reshape(ko, ki, k * k).view(np.uint16).ravel()
    for p0, v in zip(pos, vals):
        data[int(p0):int(p0) + 2] = int(v).to_bytes(2, "little")
    for c in range(ko):
        p0 = cons["off"] + 332 + 256 * (c // 16) + 16 * (c % 16)
        data[p0:p0 + 4] = struct.pack("<f", 0.0)
    out_path = os.path.join(D, "p_%s.blob" % tag)
    open(out_path, "wb").write(bytes(data))
    return out_path, hout


rng = np.random.default_rng(21)
for cout, cin, k, hin, pad, stride in SHAPES:
    tag = "%d_%d_%d_%d_%d_%d" % (cout, cin, k, hin, pad, stride)
    try:
        hout = (hin + 2 * pad - k) // stride + 1
        W = (rng.standard_normal((cout, cin, k, k)) * 0.05).astype(np.float16)
        B = (rng.standard_normal(cout) * 0.5).astype(np.float32)
        x = rng.standard_normal((cin, hin, hin)).astype(np.float16)
        out = np.zeros((cout, hout, hout))
        launches = 0
        CI, CO = ch.pick(cout, cin, k, hin, pad, stride)
        for co in range(0, cout, CO):
            for ci in range(0, cin, CI):
                ko, ki = min(CO, cout - co), min(CI, cin - ci)
                sub_w = W[co:co + ko, ci:ci + ki]
                path, _ = sub_blob(ki, ko, k, hin, pad, stride, sub_w, "%s_%d_%d" % (tag, co, ci))
                got = infer(path, x[ci:ci + ki].copy())
                launches += 1
                if got is None:
                    raise RuntimeError("device error")
                out[co:co + ko] += got.reshape(ko, hout, hout)
        out += B.astype(np.float64)[:, None, None]
        xp = np.pad(x.astype(np.float64), ((0, 0), (pad, pad), (pad, pad)))
        ref = np.zeros((cout, hout, hout))
        for i in range(k):
            for j in range(k):
                ref += np.einsum("oc,chw->ohw", W[:, :, i, j].astype(np.float64), xp[:, i:i + stride * hout:stride, j:j + stride * hout:stride])
        ref += B.astype(np.float64)[:, None, None]
        err = np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-6)
        print("%-28s launches %3d (chunk %dx%d) rel err %.2e -> %s" % (tag, launches, CO, CI, err, "PASS" if err < 2e-2 else "FAIL"), flush=True)
    except Exception as ex:
        print("%-28s ERROR %s" % (tag, str(ex).splitlines()[-1][:90]), flush=True)
