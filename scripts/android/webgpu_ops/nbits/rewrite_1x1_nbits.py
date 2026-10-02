"""Rewrite stride-1 1x1 Convs into com.microsoft MatMulNBits (weight-only, symmetric, block-wise).

    python rewrite_1x1_nbits.py in.onnx out.onnx --bits 4|8 --block 32|64|128 [--min-k 0] [--fuse-bias] [--drop-bias]

Conv(X[N,C,H,W], W[O,C,1,1], B)  ->  Transpose(X, 0,2,3,1) -> MatMulNBits(K=C, N=O) -> Add(B) -> Transpose(0,3,1,2)

The model stays NCHW; on the WebGPU EP the layout transformer turns the 3x3 convs into NHWC and the transpose optimizer cancels
back-to-back transposes. Convs with C % block != 0 are skipped (MatMulNBits needs K to be a multiple of the block size).
Quantization: symmetric, scale = max|w| / (2^(bits-1) - 1) per (output channel, K-block), implicit zero point 2^(bits-1).
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def quantize_cols(w_kn: np.ndarray, bits: int, block: int):
    """w_kn: [K, N] float. Returns packed B [N, K/block, block*bits/8] uint8, scales [N, K/block] float32, dequantized [K, N]."""
    k, n = w_kn.shape
    nb = k // block
    w = w_kn.T.reshape(n, nb, block).astype(np.float32)  # [N, nb, block]
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=2)
    scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, :, None]), -qmax - 1, qmax).astype(np.int32) + (1 << (bits - 1))  # unsigned, zp = 2^(bits-1)
    deq = ((q - (1 << (bits - 1))).astype(np.float32) * scale[:, :, None]).reshape(n, k).T
    if bits == 8:
        packed = q.astype(np.uint8)
    else:
        lo = q[:, :, 0::2]
        hi = q[:, :, 1::2]
        packed = (lo | (hi << 4)).astype(np.uint8)
    return packed, scale, deq


def rewrite(model, bits, block, min_k=0, fuse_bias=False, drop_bias=False, accuracy_level=0):
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes = []
    new_inits = []
    n_done = n_skip = 0
    skipped = []
    quant_err = []
    for idx, node in enumerate(g.node):
        ok = False
        if node.op_type == "Conv" and len(node.input) >= 2 and node.input[1] in inits:
            attrs = {a.name: a for a in node.attribute}
            ks = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else None
            st = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
            pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
            dil = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1]
            grp = attrs["group"].i if "group" in attrs else 1
            w = numpy_helper.to_array(inits[node.input[1]])
            ap = attrs["auto_pad"].s.decode() if "auto_pad" in attrs else "NOTSET"
            if w.ndim == 4 and w.shape[2:] == (1, 1) and st == [1, 1] and not any(pads) and grp == 1 and dil == [1, 1] and ap in ("NOTSET", "VALID"):
                ok = True
                o, c = w.shape[0], w.shape[1]
                if c % block != 0 or c < min_k:
                    ok = False
                    n_skip += 1
                    skipped.append((node.name, c, o))
        if not ok:
            new_nodes.append(node)
            continue
        wkn = w.reshape(o, c).T  # [K, N]
        packed, scale, deq = quantize_cols(wkn, bits, block)
        quant_err.append(float(np.abs(deq - wkn).max() / (np.abs(wkn).max() + 1e-12)))
        base = f"nb{idx}"
        new_inits.append(numpy_helper.from_array(packed, base + "_B"))
        new_inits.append(numpy_helper.from_array(scale.reshape(-1), base + "_scales"))
        x = node.input[0]
        tin, mm, tout_pre = base + "_nhwc", base + "_mm", base + "_mm_b"
        new_nodes.append(helper.make_node("Transpose", [x], [tin], perm=[0, 2, 3, 1], name=base + "_tin"))
        has_bias = len(node.input) > 2 and node.input[2]
        mm_inputs = [tin, base + "_B", base + "_scales"]
        if has_bias and fuse_bias:
            mm_inputs += ["", "", node.input[2]]  # zero_points, g_idx empty; bias is input 5
        new_nodes.append(
            helper.make_node("MatMulNBits", mm_inputs, [mm], domain="com.microsoft", K=c, N=o, bits=bits, block_size=block, accuracy_level=accuracy_level, name=base)
        )
        cur = mm
        if has_bias and not fuse_bias and not drop_bias:
            new_nodes.append(helper.make_node("Add", [mm, node.input[2]], [tout_pre], name=base + "_bias"))
            cur = tout_pre
        new_nodes.append(helper.make_node("Transpose", [cur], [node.output[0]], perm=[0, 3, 1, 2], name=base + "_tout"))
        n_done += 1
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    # drop weights no longer referenced
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    del g.value_info[:]
    return n_done, n_skip, skipped, quant_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--min-k", type=int, default=0)
    ap.add_argument("--fuse-bias", action="store_true", help="pass the conv bias as MatMulNBits input 5 instead of a separate Add")
    ap.add_argument("--accuracy-level", type=int, default=0, help="MatMulNBits accuracy_level attr; 4 enables the DP4A int8-activation path of the WebGPU kernel")
    ap.add_argument("--drop-bias", action="store_true", help="timing experiment only: drop the bias (wrong outputs)")
    a = ap.parse_args()
    m = onnx.load(a.inp)
    n_done, n_skip, skipped, qerr = rewrite(m, a.bits, a.block, a.min_k, a.fuse_bias, a.drop_bias, a.accuracy_level)
    onnx.save(m, a.out)
    print(f"{a.inp} -> {a.out}: bits={a.bits} block={a.block}: rewrote {n_done} 1x1 convs, skipped {n_skip} (K % block != 0 or < min-k); "
          f"max rel weight quantization error {max(qerr) if qerr else 0:.3e}")


if __name__ == "__main__":
    main()
