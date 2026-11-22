"""List the ELF relocations in a compiled NPU blob: for each rela.* section, the target section, the byte offset of the
patched word, the referenced symbol, and the addend. Compiles a single 1x1 conv and a chained 1x1 conv (8->16->16)
with OpenVINO compile_tool first. Run on H8."""
import os
import struct
import subprocess
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

OV = os.path.expanduser("~/openvino_git/bin/intel64/Release")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/ze-loader/lib") + ":" + OV)
D = os.path.expanduser("~/npu-dpu/roadmap/reloc"); os.makedirs(D, exist_ok=True)


def compile_model(name, nodes, inits, inp, out):
    g = helper.make_graph(nodes, "g", [inp], [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 9
    p = os.path.join(D, name + ".onnx"); onnx.save(m, p)
    blob = p[:-5] + ".blob"
    r = subprocess.run([os.path.join(OV, "compile_tool"), "-m", p, "-d", "NPU", "-o", blob, "-ip", "FP16", "-op", "FP16"],
                       env=ENV, cwd=OV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return blob


def elf_sections(data):
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shnum = struct.unpack_from("<H", data, 0x3C)[0]
    shstr = struct.unpack_from("<H", data, 0x3E)[0]
    S = []
    for i in range(shnum):
        nm, ty, fl, addr, off, size, link, info, align, entsize = struct.unpack_from("<IIQQQQIIQQ", data, shoff + 64 * i)
        S.append(dict(nm=nm, ty=ty, off=off, size=size, link=link, entsize=entsize))
    strs = data[S[shstr]["off"]:]
    for s in S:
        s["name"] = strs[s["nm"]:].split(b"\0")[0].decode(errors="replace")
    return S


def relocs(path):
    data = open(path, "rb").read()
    S = elf_sections(data)
    out = []
    for s in S:
        if not s["name"].startswith("rela."):
            continue
        target = s["name"][len("rela."):]
        symtab = S[s["link"]]
        strtab = S[symtab["link"]]
        for k in range(s["size"] // 24):
            r_off, r_info, addend = struct.unpack_from("<QQq", data, s["off"] + 24 * k)
            sym = r_info >> 32
            st_name = struct.unpack_from("<I", data, symtab["off"] + 24 * sym)[0]
            symname = data[strtab["off"] + st_name:].split(b"\0")[0].decode(errors="replace")
            out.append((target, r_off, symname, addend))
    return out


if __name__ == "__main__":
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8, 16, 16])
    w1 = numpy_helper.from_array((np.arange(16 * 8).reshape(16, 8, 1, 1) * 0.01).astype(np.float16), "W1")
    w2 = numpy_helper.from_array((np.arange(16 * 16).reshape(16, 16, 1, 1) * 0.01).astype(np.float16), "W2")
    y1 = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16, 16, 16])
    single = compile_model("single", [helper.make_node("Conv", ["x", "W1"], ["y"], kernel_shape=[1, 1])], [w1], inp, y1)
    chain_mid = helper.make_tensor_value_info("m", TensorProto.FLOAT16, [1, 16, 16, 16])
    chain = compile_model("chain", [helper.make_node("Conv", ["x", "W1"], ["m"], kernel_shape=[1, 1]),
                                    helper.make_node("Conv", ["m", "W2"], ["y"], kernel_shape=[1, 1])], [w1, w2], inp, y1)
    for label, path in (("single", single), ("chain", chain)):
        rs = relocs(path)
        print("==", label, "relocations:", len(rs), flush=True)
        for target, off, symname, addend in rs:
            print("  %-28s off %6d  sym %-40s addend %d" % (target, off, symname, addend), flush=True)
