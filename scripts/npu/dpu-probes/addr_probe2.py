"""Address probe P2 (v2 design: internal side branch).
Main: x -> conv(8->16) -> m -> conv(16->16) -> p.
Side: x -> conv(8->C) -> s -> conv(C->16) -> q  (s is an INTERNAL tensor, so it lives in CMX).
Output: y = Add(p, q). Sweep C in {8, 16, 32, 64}; hold out C = 40.
Control P0: main chain alone (no side branch). Fit each literal, non-relocated word as a + b*C; check the held-out prediction.
Compile only. Run on H8 after reloc.py and litdiff.py."""
import os
import subprocess
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R  # noqa: F401  (ELF helpers)
import litdiff as L

OV = os.path.expanduser("~/openvino_git/bin/intel64/Release")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/ze-loader/lib") + ":" + OV)
D = os.path.expanduser("~/npu-dpu/roadmap/addr2"); os.makedirs(D, exist_ok=True)
SWEEP = [8, 16, 32, 64]
HELD = 40


def init(name, shape, scale=0.01):
    n = int(np.prod(shape))
    return numpy_helper.from_array((np.arange(n).reshape(shape) * scale).astype(np.float16), name)


def compile_graph(name, C):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8, 16, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16, 16, 16])
    inits = [init("W1", (16, 8, 1, 1)), init("W2", (16, 16, 1, 1))]
    nodes = [helper.make_node("Conv", ["x", "W1"], ["m"], kernel_shape=[1, 1]),
             helper.make_node("Conv", ["m", "W2"], ["p"], kernel_shape=[1, 1])]
    if C is None:
        nodes.append(helper.make_node("Identity", ["p"], ["y"]))
    else:
        inits += [init("Ws", (C, 8, 1, 1)), init("Wt", (16, C, 1, 1))]
        nodes += [helper.make_node("Conv", ["x", "Ws"], ["s"], kernel_shape=[1, 1]),
                  helper.make_node("Conv", ["s", "Wt"], ["q"], kernel_shape=[1, 1]),
                  helper.make_node("Add", ["p", "q"], ["y"])]
    g = helper.make_graph(nodes, "p2", [x], [y], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 9
    p = os.path.join(D, name + ".onnx"); onnx.save(m, p)
    blob = p[:-5] + ".blob"
    r = subprocess.run([os.path.join(OV, "compile_tool"), "-m", p, "-d", "NPU", "-o", blob, "-ip", "FP16", "-op", "FP16"],
                       env=ENV, cwd=OV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return blob


def words_of(blob):
    return L.sec_words(blob), L.reloc_words(blob)


sweep = {C: words_of(compile_graph("p2_C%d" % C, C)) for C in SWEEP}
print("compiled sweep", SWEEP, flush=True)
held = words_of(compile_graph("p2_C%d" % HELD, HELD))
print("compiled held-out C", HELD, flush=True)
control = words_of(compile_graph("p0", None))
print("compiled control P0", flush=True)

ctl_w, ctl_r = control
c8_w, c8_r = sweep[8]
print("== control: literal words that differ between P0 and P2 (C=8)", flush=True)
for name in sorted(set(ctl_w) & set(c8_w)):
    a, b = ctl_w[name], c8_w[name]
    n = min(len(a), len(b))
    d = [j for j in range(n) if a[j] != b[j] and j not in ctl_r.get(name, set()) and j not in c8_r.get(name, set())]
    print("  %-24s %d" % (name, len(d)), flush=True)

print("== address candidates: literal words, linear fit on sweep, held-out check", flush=True)
hits = 0
for name in ("task.dpu.invariant.0.0", "task.dpu.variant.0.0", "task.dma.0.0", "task.dma.0.1", "task.dma.1.0"):
    if name not in sweep[SWEEP[0]][0]:
        continue
    n = min(len(sweep[C][0][name]) for C in SWEEP)
    for j in range(n):
        if any(j in sweep[C][1].get(name, set()) for C in SWEEP):
            continue
        row = np.array([int(sweep[C][0][name][j]) for C in SWEEP], dtype=np.int64)
        if len(set(row.tolist())) == 1:
            continue
        xs = np.array(SWEEP, np.float64)
        b_, a_ = np.polyfit(xs, row.astype(np.float64), 1)
        fit_ok = np.allclose(a_ + b_ * xs, row, atol=1.0)
        pred = int(round(a_ + b_ * HELD))
        h = int(held[0][name][j]) if j < len(held[0].get(name, [])) else None
        held_ok = h is not None and abs(h - pred) <= 1 and j not in held[1].get(name, set())
        hits += int(fit_ok and held_ok)
        print("%-24s word %4d  %s  slope %.1f  held-out pred %s actual %s %s" % (
            name, j, "LINEAR" if fit_ok else "nonlinear", b_, pred, h, "PASS" if held_ok else "fail"), flush=True)
print("accepted linear+held-out words: %d" % hits, flush=True)
