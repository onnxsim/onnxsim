"""Confirm the input read base candidate: invariant word 17 (base_offset_a). Base: 1x1 conv 8->16, 16x16. Checks:
  control: word 17 alone (+d) -> should change the output if it is on the read path
  co-edit: word 17 + input DMA destination (task.dma.0.0 word 292) by the same +d -> unchanged if it is the read partner
  for d in (+0x1000, -0x800, +0x4000). Run on H8 after reloc.py."""
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
D = os.path.expanduser("~/npu-dpu/roadmap/confirm_w17"); os.makedirs(D, exist_ok=True)
C, K, HW = 8, 16, 16
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, C, HW, HW])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, K, HW, HW])
W = (np.random.default_rng(5).standard_normal((K, C, 1, 1))).astype(np.float16)
x = np.random.default_rng(1).standard_normal((1, C, HW, HW)).astype(np.float16)
x.tofile(os.path.join(D, "in.bin"))
blob = R.compile_model("confirm_w17", [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])],
                       [numpy_helper.from_array(W, "W")], inp, out)
data = bytearray(open(blob, "rb").read())
secs = {s["name"]: s for s in R.elf_sections(bytes(data))}


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
    return np.fromfile(os.path.join(od, os.listdir(od)[0]), dtype=np.float16).astype(np.float64)


def bump(buf, name, j, d):
    v = struct.unpack_from("<I", buf, at(name, j))[0]
    struct.pack_into("<I", buf, at(name, j), (v + d) & 0xFFFFFFFF)


ref = infer(data)
assert ref is not None, "base run failed"
W17 = ("task.dpu.invariant.0.0", 17)
W292 = ("task.dma.0.0", 292)
for d in (0x1000, -0x800, 0x4000):
    p = bytearray(data); bump(p, *W17, d); got = infer(p)
    c = "error" if got is None else ("unchanged" if np.abs(got - ref).max() < 1e-2 else "changed (max %.3f)" % np.abs(got - ref).max())
    p = bytearray(data); bump(p, *W17, d); bump(p, *W292, d); got2 = infer(p)
    co = "error" if got2 is None else ("unchanged" if np.abs(got2 - ref).max() < 1e-2 else "changed (max %.3f)" % np.abs(got2 - ref).max())
    print("d %+6d: control word17 alone %-22s | co-edit word17 + input dest %s" % (d, c, co), flush=True)
