"""chains.py OUTDIR : chains of N identical (or alternating pair) standard NCHW Convs at the real shapes of ResNet-50 / YOLO11n classes.
Files chain_<tag>_N{2,10}.onnx and chains.json {tag: {kind, shape, convs:[(cin,cout,k,s)], flops_per_conv}}.
Per-conv wall time = (median N=10 - median N=2) / 8 on the phone (ORT fuses/transposes only at the ends of the chain)."""
import sys, os, json
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
# tag: (H, [(cin, cout, k, relu)])  stride 1, pad k//2, alternating list repeated
CFG = {
    # ResNet-50 classes
    "r_w64_56": (56, [(64, 64, 3)]), "r_w128_28": (28, [(128, 128, 3)]), "r_w256_14": (14, [(256, 256, 3)]), "r_w512_7": (7, [(512, 512, 3)]),
    "r_m64_256_56": (56, [(64, 256, 1), (256, 64, 1)]), "r_m128_512_28": (28, [(128, 512, 1), (512, 128, 1)]),
    "r_m256_1024_14": (14, [(256, 1024, 1), (1024, 256, 1)]), "r_m512_2048_7": (7, [(512, 2048, 1), (2048, 512, 1)]),
    # YOLO11n classes
    "y_w64_80": (80, [(64, 64, 3)]), "y_w64_20": (20, [(64, 64, 3)]),
    "y_t32_40": (40, [(32, 32, 3)]), "y_t16_32_80": (80, [(16, 32, 3), (32, 16, 3)]), "y_t32_64_40": (40, [(32, 64, 3), (64, 32, 3)]),
    "y_t8_16_160": (160, [(8, 16, 3), (16, 8, 3)]),
    "y_m192_128_40": (40, [(192, 128, 1), (128, 192, 1)]), "y_m80_80_80": (80, [(80, 80, 1)]), "y_m384_256_20": (20, [(384, 256, 1), (256, 384, 1)]),
    "y_m256_256_20": (20, [(256, 256, 1)]), "y_m64_64_80": (80, [(64, 64, 1)]), "y_m32_32_160": (160, [(32, 32, 1)]),
}
meta = {}
for tag, (hw, convs) in CFG.items():
    cin0 = convs[0][0]
    for n in (2, 10):
        nodes, inits = [], []
        cur = "X"
        for i in range(n):
            cin, cout, k = convs[i % len(convs)]
            w = (rng.standard_normal((cout, cin, k, k)) * (1.0 / np.sqrt(cin * k * k))).astype("f")
            b = (rng.standard_normal(cout) * 0.1).astype("f")
            inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
            o = f"t{i}"
            nodes.append(helper.make_node("Conv", [cur, f"w{i}", f"b{i}"], [o], kernel_shape=[k, k], pads=[k // 2] * 4, strides=[1, 1], name=f"c{i}"))
            cur = o
        cout_last = convs[(n - 1) % len(convs)][1]
        g = helper.make_graph(nodes, tag, [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, cin0, hw, hw])],
                              [helper.make_tensor_value_info(cur, TensorProto.FLOAT, [1, cout_last, hw, hw])], inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.save(m, os.path.join(out, f"chain_{tag}_N{n}.onnx"))
    fl = [2 * hw * hw * c[0] * c[1] * c[2] * c[2] for c in convs]
    meta[tag] = dict(hw=hw, convs=convs, gflop_per_conv=sum(fl) / len(fl) / 1e9,
                     mb_per_conv=sum(4 * hw * hw * (c[0] + c[1]) for c in convs) / len(convs) / 1e6)
json.dump(meta, open(os.path.join(out, "chains.json"), "w"), indent=1)
print(len(CFG), "chains")
