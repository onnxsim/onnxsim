"""Layer-wise AdaRound (Nagel et al. 2020) for the int8 per-channel weights of every Conv, in torch.

onnxsim's own rounding passes (`apply_adaround`, `apply_gptq`, `apply_tesseraq`) target weight-only INT4
MatMul/Gemm, and `apply_adaquant` targets QDQ MatMul/Gemm; none handles Conv. openpilot's backbones are
all Conv, so this is a small torch port of the same objective for Conv:

  W_q = s * clip(floor(W / s) + h(V), -127, 127),  h(V) = clip(sigmoid(V) * 1.2 - 0.1, 0, 1)
  min_V  || conv(X, W_q) - conv(X, W) ||^2 + lam * sum(1 - |2 h(V) - 1|^beta)     (beta annealed 20 -> 2)

with s = max|W| / 127 per output channel (the grid `quantize_full_qdq` uses), X the layer's input captured
from the fp32 model on real route frames. The result is written back into the fp32 model as float
weights lying exactly on that grid (each channel's max-magnitude element is pinned to its +-127 code so
the scale `quantize_full_qdq` recomputes is unchanged), so any later quantization reproduces the codes.

  python adaround_conv.py driving_fp32.onnx out.onnx --kind driving --inputs inputs_seg3.npz,inputs_seg12.npz
"""

import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
import quantize
import torch
import torch.nn.functional as F
from onnx import numpy_helper


def capture_inputs(model, convs, samples):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    del m.graph.output[:]
    names = sorted({c.input[0] for c in convs})
    for n in names:
        m.graph.output.append(
            onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None)
        )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(
        m.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    outs = [s.run(names, feed) for feed in samples]
    return {n: np.concatenate([o[i] for o in outs]) for i, n in enumerate(names)}


def attrs(node):
    a = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
    pads = a.get("pads", [0, 0, 0, 0])
    assert pads[0] == pads[2] and pads[1] == pads[3], "asymmetric pads"
    return dict(
        stride=tuple(a.get("strides", [1, 1])),
        padding=(pads[0], pads[1]),
        dilation=tuple(a.get("dilations", [1, 1])),
        groups=a.get("group", 1),
    )


def adaround_layer(x, w, b, kw, iters, lr, lam, batch, seed):
    torch.manual_seed(seed)
    x = torch.from_numpy(x)
    w = torch.from_numpy(w)
    b = torch.from_numpy(b) if b is not None else None
    with torch.no_grad():
        y = F.conv2d(x, w, b, **kw)
    s = w.abs().reshape(w.shape[0], -1).amax(1).clamp_min(1e-12) / 127
    s = s.reshape(-1, 1, 1, 1)
    wf = torch.floor(w / s)
    rest = w / s - wf
    # init V so that h(V) = the fractional part (round-to-nearest starting point, as in the paper)
    v = torch.log(
        (rest.clamp(1e-3, 1 - 1e-3) + 0.1)
        / 1.2
        / (1 - (rest.clamp(1e-3, 1 - 1e-3) + 0.1) / 1.2)
    )
    v = v.clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    n = x.shape[0]

    def h(v):
        return torch.clamp(torch.sigmoid(v) * 1.2 - 0.1, 0, 1)

    def wq(hard):
        r = (v >= 0).float() if hard else h(v)
        return s * torch.clamp(wf + r, -127, 127)

    with torch.no_grad():
        rtn = s * torch.clamp(torch.round(w / s), -127, 127)
        base = F.mse_loss(F.conv2d(x, rtn, b, **kw), y).item()
    warm = int(0.2 * iters)
    for it in range(iters):
        idx = torch.randint(0, n, (min(batch, n),))
        rec = F.mse_loss(F.conv2d(x[idx], wq(False), b, **kw), y[idx])
        if it < warm:
            loss = rec
        else:
            beta = 20 + (2 - 20) * (it - warm) / max(1, iters - warm)
            loss = rec + lam * (1 - (2 * h(v) - 1).abs().pow(beta)).sum() / v.numel()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        # hard decision: h(V) >= 0.5 rounds up (the regularizer has pushed h to 0/1 by now)
        q = torch.clamp(wf + (h(v) >= 0.5).float(), -127, 127)
        # pin every channel's max-|w| element to its exact +-127 code so the recomputed scale is unchanged
        flat = w.reshape(w.shape[0], -1)
        am = flat.abs().argmax(1)
        qf = q.reshape(q.shape[0], -1)
        qf[torch.arange(qf.shape[0]), am] = torch.round(
            flat[torch.arange(flat.shape[0]), am] / s.reshape(-1)
        )
        wout = (s * qf.reshape(w.shape)).numpy()
        after = F.mse_loss(F.conv2d(x, torch.from_numpy(wout), b, **kw), y).item()
    return wout, base, after


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--kind", choices=["driving", "dm"], required=True)
    ap.add_argument(
        "--inputs", required=True, help="calibration frames .npz, comma list"
    )
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--stride", type=int, default=36)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--calib", default="0,0.164,0.005")
    ap.add_argument(
        "--only", help="comma list of conv output names (default: every Conv)"
    )
    args = ap.parse_args()
    torch.set_num_threads(16)
    m = onnx.load(args.model)
    g = m.graph
    convs = [n for n in g.node if n.op_type == "Conv"]
    if args.only:
        keep = set(args.only.split(","))
        convs = [c for c in convs if c.output[0] in keep]
    samples = (
        quantize.driving_samples(args.model, args.inputs, args.samples, args.stride)
        if args.kind == "driving"
        else quantize.dm_samples(
            args.inputs,
            args.samples,
            args.stride,
            [float(v) for v in args.calib.split(",")],
        )
    )
    t = time.time()
    xs = capture_inputs(m, convs, samples)
    print(
        f"captured {len(xs)} conv inputs from {len(samples)} samples in {time.time() - t:.0f}s",
        flush=True,
    )
    init = {i.name: k for k, i in enumerate(g.initializer)}
    tot_b = tot_a = 0.0
    for c in convs:
        w = numpy_helper.to_array(g.initializer[init[c.input[1]]]).astype(np.float32)
        b = (
            numpy_helper.to_array(g.initializer[init[c.input[2]]]).astype(np.float32)
            if len(c.input) > 2 and c.input[2]
            else None
        )
        t = time.time()
        wout, base, after = adaround_layer(
            xs[c.input[0]], w, b, attrs(c), args.iters, args.lr, args.lam, args.batch, 0
        )
        g.initializer[init[c.input[1]]].CopyFrom(
            numpy_helper.from_array(wout, c.input[1])
        )
        tot_b += base
        tot_a += after
        print(
            f"  {c.output[0]:40s} {str(list(w.shape)):18s} rtn mse {base:.3e} -> adaround {after:.3e} "
            f"({10 * np.log10(base / max(after, 1e-30)):+.1f} dB) {time.time() - t:.0f}s",
            flush=True,
        )
    onnx.save(m, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
