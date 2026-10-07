"""One scaling measurement: the certified quantization bound of a 3-layer conv net.
usage: zonotope_box_noise_bench.py <mode> <image_size> <channels>
  mode: dense | box_noise | box_noise+fresh | box_all (experiment: also boxes the FIRST model)

Run each cell in its own process (peak RSS is per process) and under a memory cap, e.g.
`systemd-run --user --scope -p MemoryMax=16G python scripts/zonotope_box_noise_bench.py box_noise+fresh 32 8`.
See docs/zonotope-box-noise.md for the measured table.
Measures the FIRST zonotope.bound_difference call that quant_verify makes (the total bound);
quant_verify makes further calls for its weights-only floor, which this script does not time."""
import json
import resource
import sys
import time

import numpy as np
from onnx import numpy_helper as nh
from onnx import parser

mode, size, ch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
from onnxsim import interval, quant_verify
from onnxsim import zonotope as Z

rng = np.random.default_rng(0)


def f32(x):
    return np.asarray(x, np.float32)


def wq(W):  # int8 symmetric per-output-channel fake quantisation
    s = np.maximum(np.abs(W).max(axis=(1, 2, 3), keepdims=True), 1e-12) / 127
    return f32(np.round(W / s).clip(-128, 127) * s)


def build(Ws, scales=None):
    lines, prev, inits = [], "x", {}
    for i, (W, B, stride) in enumerate(Ws):
        inits[f"W{i}"], inits[f"B{i}"] = W, B
        lines.append(f"a{i} = Conv<strides=[{stride},{stride}], pads=[1,1,1,1]>({prev}, W{i}, B{i})")
        out = f"a{i}"
        if i < len(Ws) - 1:
            lines.append(f"r{i} = Relu(a{i})")
            out = f"r{i}"
        if scales is not None:
            inits[f"S{i}"], inits[f"Z{i}"] = f32(scales[i]), np.int8(0)
            lines += [f"q{i} = QuantizeLinear({out}, S{i}, Z{i})", f"d{i} = DequantizeLinear(q{i}, S{i}, Z{i})"]
            out = f"d{i}"
        prev = out
    lines.append(f"y = Identity({prev})")
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["" : 13]> g (float[1,3,%d,%d] x) => (float[1,%d,%d,%d] y) {\n%s\n}'
        % (size, size, ch, (size + 1) // 2, (size + 1) // 2, "\n".join(lines))
    )
    m.graph.initializer.extend(nh.from_array(np.asarray(v), k) for k, v in inits.items())
    return m


chs = [3, ch, ch, ch]
Ws = []
for i in range(3):
    Ws.append((f32(rng.standard_normal((chs[i + 1], chs[i], 3, 3)) / np.sqrt(9 * chs[i])), f32(rng.standard_normal(chs[i + 1]) * 0.1), 2 if i == 1 else 1))
base = build(Ws)
box = {"x": (-1.0, 1.0)}
res = interval.propagate(base, box)
scales = []
for i in range(3):
    lo, hi = res.hull(f"r{i}" if i < 2 else f"a{i}")
    scales.append(max(abs(lo), abs(hi), 1e-6) / 127)
mq = build([(wq(W), B, s) for (W, B, s) in Ws], scales)

orig = Z.bound_difference
calls = []

if mode == "box_all":  # experiment only: ALSO box the first model's relaxation symbols (not an API)
    _E = Z._Evaluator

    class _AllBoxed(_E):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            k["box_fresh"] = True
            super().__init__(*a, **k)

    Z._Evaluator = _AllBoxed


def wrapped(a, b, r=None, *x, **k):
    if mode != "dense":
        k.update(box_inputs=["__qnoise_*"], box_fresh=(mode in ("box_noise+fresh", "box_all")))
    t = time.perf_counter()
    out = orig(a, b, r, *x, **k)
    calls.append(dict(seconds=time.perf_counter() - t, worst=float(out.worst), stats=getattr(out, "stats", {}), notes=[n for n in out.notes if "symbol cap" in n]))
    return out


Z.bound_difference = wrapped
t0 = time.perf_counter()
rep = quant_verify.verify(base, mq, box, breakdown=False)
total = time.perf_counter() - t0
first = calls[0]
stats = first["stats"]
sym = max((v.get("max_symbols", 0) for v in stats.values()), default=0)
gen = max((v.get("max_generator_elements", 0) for v in stats.values()), default=0)
boxel = max((v.get("max_box_elements", 0) for v in stats.values()), default=0)
print("RESULT " + json.dumps(dict(
    mode=mode, size=size, channels=ch, bound=first["worst"], bound_s=round(first["seconds"], 2),
    verify_total_s=round(total, 2), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    max_symbols=sym, largest_generator_GB=round(gen * 8 / 1e9, 3), max_box_elements=boxel,
    symbol_cap_hit=bool(first["notes"]),
)))
