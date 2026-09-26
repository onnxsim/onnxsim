#!/usr/bin/env python3
"""QNN side of the HMX-vs-QNN comparison: chains of L identical 1x1 convs / MatMuls (L=2 and L=6),
timed by ../../htp_exploration/ceiling/run_ceiling.sh and differenced by summarize_ceiling.py, so the
per-layer time excludes graph launch and I/O.

Variants: u8 (uint8 act / int8 weight QDQ, the ceiling scripts' own builder) and f16 (a plain fp32
graph, which the QNN EP runs in fp16 on the HTP).

usage: gen_models.py <outdir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "htp_exploration" / "ceiling")
)
from gen_ceiling_models import HI, LO, build  # noqa: E402


def build_f16(kind, cin, cout, hw, layers):
    rng = np.random.default_rng(0)
    inits, nodes, cur = [], [], "x"
    for i in range(layers):
        if kind == "matmul":
            w = (rng.standard_normal((cin, cout)) / np.sqrt(cin)).astype(np.float32)
            inits.append(numpy_helper.from_array(w, f"w{i}"))
            nodes.append(helper.make_node("MatMul", [cur, f"w{i}"], [f"y{i}"]))
        else:
            w = (rng.standard_normal((cout, cin, 1, 1)) / np.sqrt(cin)).astype(
                np.float32
            )
            inits.append(numpy_helper.from_array(w, f"w{i}"))
            nodes.append(
                helper.make_node("Conv", [cur, f"w{i}"], [f"y{i}"], kernel_shape=[1, 1])
            )
        cur = f"y{i}"
    nodes.append(helper.make_node("Identity", [cur], ["y"]))
    ins = [hw, cin] if kind == "matmul" else [1, cin, hw, hw]
    outs = [hw, cout] if kind == "matmul" else [1, cout, hw, hw]
    g = helper.make_graph(
        nodes,
        "f16",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ins)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, outs)],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m)
    return m, (hw * cin * cout) if kind == "matmul" else (hw * hw * cin * cout)


def widen(m, w):
    """a square-built conv model -> spatial width w (1x1 convs keep no internal shapes)"""
    for vi in (m.graph.input[0], m.graph.output[0]):
        vi.type.tensor_type.shape.dim[3].dim_value = w
    return m


# (kind, cin, cout, hw, variant): conv1x1 at hw=64 is a GEMM with M = 4096 rows (pixels), K = cin, N = cout;
# hw "HxW" = a non-square conv input (M = H*W)
SWEEP = [
    ("conv3x3", 128, 128, "32x64", "u8"),
    ("conv3x3", 256, 256, "32x64", "u8"),
    ("conv1x1", 1024, 1024, "32x64", "u8"),
    ("conv1x1", 512, 512, "32x64", "u8"),
    ("conv1x1", 256, 256, 64, "u8"),
    ("conv1x1", 512, 512, 64, "u8"),
    ("conv1x1", 1024, 1024, 64, "u8"),
    ("matmul", 1024, 1024, 4096, "u8"),
    ("conv1x1", 256, 256, 64, "f16"),
    ("conv1x1", 512, 512, 64, "f16"),
    ("conv1x1", 1024, 1024, 64, "f16"),
    ("matmul", 1024, 1024, 4096, "f16"),
]


def main():
    """usage: gen_models.py <outdir> [lo hi [tag-filter]]: chain lengths (default the ceiling study's 2 and 6) and a
    substring filter on the tags"""
    out = Path(sys.argv[1])
    lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (LO, HI)
    filt = sys.argv[4] if len(sys.argv) > 4 else ""
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for kind, cin, cout, hw, variant in SWEEP:
        tag = f"{kind}_{cin}x{cout}_{hw}_{variant}"
        if filt not in tag:
            continue
        h, w = (int(x) for x in hw.split("x")) if isinstance(hw, str) else (hw, hw)
        e = {
            "tag": tag,
            "kind": kind,
            "cin": cin,
            "cout": cout,
            "hw": hw,
            "variant": variant,
            "lo": lo,
            "hi": hi,
        }
        for L in (lo, hi):
            m, macs = (
                build_f16(kind, cin, cout, h, L)
                if variant == "f16"
                else build(kind, cin, cout, h, L, variant)
            )
            if w != h:
                m, macs = widen(m, w), macs // h * w
            onnx.save(m, out / f"{tag}_L{L}.onnx")
            e["macs_per_layer"] = macs
        manifest.append(e)
        print(tag, f"{e['macs_per_layer'] / 1e9:.2f} GMAC/layer")
    m, _ = build("conv1x1", 32, 32, 8, 0, "u8")
    onnx.save(m, out / "tiny_L0.onnx")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
