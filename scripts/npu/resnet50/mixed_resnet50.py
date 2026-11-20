"""ResNet50 through our path: walk the ONNX graph; Relu and Add run as SHAVE blobs on the NPU (fp16, tiled),
every other op runs through tinygrad's ONNX op table on the CPU backend. Compared with onnxruntime.

Usage: mixed_resnet50.py [npu_ops]   e.g. "Relu,Add" (default). Ops not listed run on tinygrad."""
import glob
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
sys.path.insert(0, os.environ.get("CONV_NPU_DIR", os.path.expanduser("~/conv-npu-dir")))  # folder with conv_npu.py
import conv_npu  # noqa: E402
from onnx import numpy_helper
import onnxruntime as ort

os.environ.setdefault("DEV", "CPU")
from tinygrad import Tensor
from tinygrad.nn.onnx import get_onnx_ops

OV = os.path.expanduser("~/openvino_git/bin/intel64/Release")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/ze-loader/lib") + ":" + OV)
MODEL = os.path.expanduser("~/models/resnet50_opset17.onnx")
# single launches above ~0.4M elements fail on the device (see NOTES), so only sizes up to 401408 are used
RELU_BLOBS = {int(os.path.basename(p)[5:-5]): p for p in glob.glob(os.path.expanduser("~/npu-tinygrad/blobs/relu_*.blob"))
              if int(os.path.basename(p)[5:-5]) <= 401408}
ADD_BLOBS = {2048: os.path.expanduser("~/add-build-2048/add_npu.blob"),
             100352: os.path.expanduser("~/add-build-100352/add_npu.blob"),
             200704: os.path.expanduser("~/add-build-200704/add_npu.blob")}
NPU_OPS = set((sys.argv[1] if len(sys.argv) > 1 else "Relu,Add").split(","))


def npu_launch(blob, arrays):
    """Run a SHAVE blob on the NPU in-process (cached request per blob). arrays: fp16 1-D arrays, one per input."""
    flat = np.concatenate([a.astype(np.float16).ravel() for a in arrays])
    req = conv_npu._request(blob)
    sizes = [a.size for a in arrays]
    out = None
    if len(arrays) == 1:
        res = req.infer({0: arrays[0].astype(np.float16).reshape(1, -1)})
    else:
        res = req.infer({i: arrays[i].astype(np.float16).reshape(1, -1) for i in range(len(arrays))})
    out = np.asarray(list(res.values())[0], dtype=np.float16).reshape(-1)
    return out


def chunks(n, sizes):
    """Split n elements into launches of the largest available blob size that divides the remaining work."""
    out, rem = [], n
    best = max(s for s in sizes if s <= n and n % s == 0) if any(n % s == 0 and s <= n for s in sizes) else None
    if best is None:
        return None
    return [best] * (n // best)


def npu_relu(x):
    flat = x.numpy().astype(np.float16).ravel()
    n = flat.size
    sizes = sorted(RELU_BLOBS)
    ch = chunks(n, sizes)
    if ch is None:
        raise RuntimeError("no Relu blob for %d elements" % n)
    out = np.empty(n, dtype=np.float16)
    pos = 0
    for c in ch:
        out[pos:pos + c] = npu_launch(RELU_BLOBS[c], [flat[pos:pos + c]])
        pos += c
    return Tensor(out.astype(np.float32).reshape(x.shape))


def npu_add(a, b):
    fa, fb = a.numpy().astype(np.float16).ravel(), b.numpy().astype(np.float16).ravel()
    n = fa.size
    ch = chunks(n, sorted(ADD_BLOBS))
    if ch is None:
        raise RuntimeError("no Add blob for %d elements" % n)
    out = np.empty(n, dtype=np.float16)
    pos = 0
    for c in ch:
        out[pos:pos + c] = npu_launch(ADD_BLOBS[c], [fa[pos:pos + c], fb[pos:pos + c]])
        pos += c
    return Tensor(out.astype(np.float32).reshape(a.shape))


MAX_BLOBS = {200704: os.path.expanduser("~/max-build-200704/max_npu.blob")}


def npu_max2(a, b):
    """Elementwise max of two fp16 1-D arrays on the NPU (tiled over the available max blob sizes)."""
    n = a.size
    ch = chunks(n, sorted(MAX_BLOBS))
    if ch is None:
        raise RuntimeError("no Max blob for %d elements" % n)
    out = np.empty(n, dtype=np.float16)
    pos = 0
    for c in ch:
        out[pos:pos + c] = npu_launch(MAX_BLOBS[c], [a[pos:pos + c], b[pos:pos + c]])
        pos += c
    return out


def npu_maxpool(x, attrs):
    """MaxPool as a max over the kernel's shifted, strided windows: each window pair is one NPU max launch."""
    k, s, p = attrs["kernel_shape"], attrs["strides"], attrs["pads"]  # pads = [top, left, bottom, right]
    arr = x.numpy().astype(np.float16)
    N, C, H, W = arr.shape
    padded = np.pad(arr, ((0, 0), (0, 0), (p[0], p[2]), (p[1], p[3])), constant_values=np.float16(-65504))
    Ho = (H + p[0] + p[2] - k[0]) // s[0] + 1
    Wo = (W + p[1] + p[3] - k[1]) // s[1] + 1
    wins = [np.ascontiguousarray(padded[:, :, i:i + s[0] * Ho:s[0], j:j + s[1] * Wo:s[1]]).ravel()
            for i in range(k[0]) for j in range(k[1])]
    m = wins[0]
    for w in wins[1:]:
        m = npu_max2(m, w)
    return Tensor(m.astype(np.float32).reshape(N, C, Ho, Wo))


def npu_gap(x):
    """GlobalAveragePool: spatial sum as pairwise NPU adds over each (y, x) slice, then the divide by H*W on the CPU."""
    arr = x.numpy().astype(np.float16)
    N, C, H, W = arr.shape
    slices = [np.ascontiguousarray(arr[:, :, i, j]).ravel() for i in range(H) for j in range(W)]
    s = slices[0]
    for sl in slices[1:]:
        ch = chunks(s.size, sorted(ADD_BLOBS))
        out = np.empty(s.size, dtype=np.float16)
        pos = 0
        for c in ch:
            out[pos:pos + c] = npu_launch(ADD_BLOBS[c], [s[pos:pos + c], sl[pos:pos + c]])
            pos += c
        s = out
    return Tensor((s.astype(np.float32) / (H * W)).reshape(N, C, 1, 1))


MUL_BLOB = os.path.expanduser("~/mul-build-200704/mul2_npu.blob")
TILE = 200704


def npu_binop(blob, a, b):
    """Elementwise a OP b on the NPU: flat fp16 arrays, padded with zeros to whole 200704-element tiles."""
    n = a.size
    m = -(-n // TILE) * TILE
    pa, pb = np.zeros(m, np.float16), np.zeros(m, np.float16)
    pa[:n], pb[:n] = a, b
    out = np.empty(m, np.float16)
    for pos in range(0, m, TILE):
        out[pos:pos + TILE] = npu_launch(blob, [pa[pos:pos + TILE], pb[pos:pos + TILE]])
    return out[:n]


def npu_gemm(ins, attrs):
    """Gemm y = A @ B^T + C (transB=1): products on the NPU (mul), a pairwise add tree over K (add), then bias (add)."""
    assert attrs.get("transB", 0) == 1 and attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
    xv = ins[0].numpy().astype(np.float16).reshape(-1)
    W = ins[1].numpy().astype(np.float16)            # [N, K]
    bias = ins[2].numpy().astype(np.float16).reshape(-1) if len(ins) > 2 and ins[2] is not None else np.zeros(W.shape[0], np.float16)
    N, K = W.shape
    P = npu_binop(MUL_BLOB, W.ravel(), np.broadcast_to(xv, (N, K)).ravel()).reshape(N, K)
    cols = K
    while cols > 1:
        h = cols // 2
        P = npu_binop(ADD_TILE, np.ascontiguousarray(P[:, :h]).ravel(),
                      np.ascontiguousarray(P[:, h:cols]).ravel()).reshape(N, h)
        cols = h
    y = npu_binop(ADD_TILE, P[:, 0].copy(), bias)
    return Tensor(y.astype(np.float32).reshape(1, N))


ADD_TILE = ADD_BLOBS[200704]

def npu_conv(ins, attrs):
    """Conv on the NPU through the MLIR SHAVE route (conv_npu.run_conv): tiled launches of our own kernels."""
    assert attrs.get("group", 1) == 1 and all(d == 1 for d in attrs.get("dilations", [1, 1]))
    x = ins[0].numpy()[0].astype(np.float16)
    W = ins[1].numpy().astype(np.float16)
    B = ins[2].numpy().astype(np.float16) if len(ins) > 2 and ins[2] is not None else np.zeros(W.shape[0], np.float16)
    s = attrs.get("strides", [1, 1])[0]
    p = attrs.get("pads", [0, 0, 0, 0])[0]
    assert attrs.get("pads", [0, 0, 0, 0]) == [p] * 4 and attrs.get("strides", [1, 1]) == [s, s]
    return Tensor(conv_npu.run_conv(x, W, B, s, p)[None].astype(np.float32))


def npu_flatten(x, attrs):
    """Flatten is a view: the same data, a new shape. No computation, no copy."""
    axis = attrs.get("axis", 1)
    shape = x.shape
    outer = int(np.prod(shape[:axis])) if axis > 0 else 1
    return x.reshape(outer, -1)


NPU_IMPL = {"Flatten": lambda ins, attrs: npu_flatten(ins[0], attrs), "Conv": lambda ins, attrs: npu_conv(ins, attrs),
            "Relu": lambda ins, attrs: npu_relu(ins[0]), "Add": lambda ins, attrs: npu_add(ins[0], ins[1]),
            "MaxPool": lambda ins, attrs: npu_maxpool(ins[0], attrs),
            "GlobalAveragePool": lambda ins, attrs: npu_gap(ins[0]),
            "Gemm": lambda ins, attrs: npu_gemm(ins, attrs)}


def attrs_of(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def main():
    model = onnx.load(MODEL)
    tg_ops = get_onnx_ops()
    env = {init.name: Tensor(numpy_helper.to_array(init).astype(np.float32)) for init in model.graph.initializer}
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)
    env[model.graph.input[0].name] = Tensor(x)
    counts = {"npu": 0, "tinygrad": 0}
    npu_by_op, tg_by_op = {}, {}
    t0 = time.time()
    for node in model.graph.node:
        ins = [env[n] if n else None for n in node.input]
        a = attrs_of(node)
        if node.op_type in NPU_OPS and node.op_type in NPU_IMPL:
            outs = [NPU_IMPL[node.op_type](ins, a)]
            counts["npu"] += 1
            npu_by_op[node.op_type] = npu_by_op.get(node.op_type, 0) + 1
        else:
            impl = tg_ops[node.op_type]
            impl = impl.get(max(impl)) if isinstance(impl, dict) else impl
            res = impl(*ins, **a)
            outs = list(res) if isinstance(res, tuple) else [res]
            counts["tinygrad"] += 1
            tg_by_op[node.op_type] = tg_by_op.get(node.op_type, 0) + 1
        for name, t in zip(node.output, outs):
            env[name] = t
    got = env[model.graph.output[0].name].numpy()
    print("walk time %.1fs" % (time.time() - t0))
    print("nodes on NPU: %d %s | nodes on tinygrad CPU: %d %s" % (counts["npu"], npu_by_op, counts["tinygrad"], tg_by_op))
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {model.graph.input[0].name: x})[0]
    d = np.abs(got - ref)
    print("max abs diff vs onnxruntime %.4g (ref max %.4g) | top-1 %d vs %d match %s"
          % (d.max(), np.abs(ref).max(), int(got.argmax()), int(ref.argmax()), int(got.argmax()) == int(ref.argmax())))


if __name__ == "__main__":
    main()
