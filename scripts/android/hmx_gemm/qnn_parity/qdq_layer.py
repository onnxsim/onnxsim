#!/usr/bin/env python3
"""One QDQ conv layer in the form onnxsim's full_qdq / QNN expect (uint8 activations with zero point, int8
per-channel symmetric weights, int32 bias), plus a random input and ORT CPU's output as the reference.

    x(fp32) -> Q(sx, zx) -> DQ -> Conv(DQ(w int8, sw[c]), DQ(b int32, sx*sw[c])) [-> Relu] -> Q(sy, zy) -> DQ -> y

usage: qdq_layer.py <outdir> <cin> <cout> <h> <w> [k=1] [stride=1] [relu=0] [seed=0]
writes model.onnx, input.bin (fp32), ref_q.bin (uint8 ORT CPU output levels), params.npz (everything a kernel
needs: xq, wq, bq, sx, zx, sw, sy, zy)
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def build(cin, cout, h, w, k, stride, relu, seed):
    rng = np.random.default_rng(seed)
    sx, zx = np.float32(0.0213), np.uint8(128)
    sy, zy = (
        np.float32(0.0371 if not relu else 0.0187),
        np.uint8(128 if not relu else 0),
    )
    wf = rng.standard_normal((cout, cin, k, k)).astype(np.float32) / np.sqrt(
        cin * k * k
    )
    sw = (np.abs(wf).reshape(cout, -1).max(1) / 127).astype(np.float32)
    wq = np.clip(np.round(wf / sw[:, None, None, None]), -127, 127).astype(np.int8)
    bq = rng.integers(-3000, 3000, cout).astype(np.int32)
    inits = [
        numpy_helper.from_array(np.array(sx), "sx"),
        numpy_helper.from_array(np.array(zx), "zx"),
        numpy_helper.from_array(np.array(sy), "sy"),
        numpy_helper.from_array(np.array(zy), "zy"),
        numpy_helper.from_array(wq, "wq"),
        numpy_helper.from_array(sw, "sw"),
        numpy_helper.from_array(np.zeros(cout, np.int8), "zw"),
        numpy_helper.from_array(bq, "bq"),
        numpy_helper.from_array((sx * sw).astype(np.float32), "sb"),
        numpy_helper.from_array(np.zeros(cout, np.int32), "zb"),
    ]
    p = k // 2
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "sx", "zx"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "sx", "zx"], ["xd"]),
        helper.make_node("DequantizeLinear", ["wq", "sw", "zw"], ["wd"], axis=0),
        helper.make_node("DequantizeLinear", ["bq", "sb", "zb"], ["bd"], axis=0),
        helper.make_node(
            "Conv",
            ["xd", "wd", "bd"],
            ["c"],
            kernel_shape=[k, k],
            pads=[p] * 4,
            strides=[stride] * 2,
        ),
    ]
    cur = "c"
    if relu:
        nodes.append(helper.make_node("Relu", [cur], ["r"]))
        cur = "r"
    nodes += [
        helper.make_node("QuantizeLinear", [cur, "sy", "zy"], ["yq"]),
        helper.make_node("DequantizeLinear", ["yq", "sy", "zy"], ["y"]),
    ]
    ho, wo = (h + 2 * p - k) // stride + 1, (w + 2 * p - k) // stride + 1
    g = helper.make_graph(
        nodes,
        "qdq_layer",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, ho, wo])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m)
    x = (rng.standard_normal((1, cin, h, w)) * 1.2).astype(np.float32)
    return (
        m,
        x,
        dict(
            wq=wq,
            bq=bq,
            sx=sx,
            zx=zx,
            sw=sw,
            sy=sy,
            zy=zy,
            k=k,
            stride=stride,
            relu=relu,
        ),
    )


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cin, cout, h, w = (int(a) for a in sys.argv[2:6])
    k, stride, relu, seed = (
        int(a) for a in (sys.argv[6:10] + ["1", "1", "0", "0"][len(sys.argv[6:10]) :])
    )
    m, x, prm = build(cin, cout, h, w, k, stride, relu, seed)
    onnx.save(m, out / "model.onnx")
    x.tofile(out / "input.bin")
    y = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    ).run(None, {"x": x})[0]
    yq = np.clip(np.round(y / prm["sy"]) + int(prm["zy"]), 0, 255).astype(np.uint8)
    yq.tofile(out / "ref_q.bin")
    xq = np.clip(np.round(x / prm["sx"]) + int(prm["zx"]), 0, 255).astype(np.uint8)
    np.savez(out / "params.npz", xq=xq, **prm)
    print(f"{out}: y {y.shape}, levels {yq.min()}..{yq.max()}")


if __name__ == "__main__":
    main()
