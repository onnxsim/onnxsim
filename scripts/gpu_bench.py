#!/usr/bin/env python3
"""End-to-end benchmark of the opt-in GPU backend of onnxsim.crown / backward_diff / quant_verify.

Not run in CI. Every case runs in its own subprocess (so a case that exhausts memory, hangs or
is killed loses nothing else) and appends one JSON line to ``--out`` as soon as it finishes.

    python scripts/gpu_bench.py --list
    python scripts/gpu_bench.py --device cuda --precision float32 --suite quick --out r.jsonl
    python scripts/gpu_bench.py --case head_crown_32_8 --device cuda --precision float32   # one case

Backends compared by running the same suite per (device, precision):

    cpu / float64        numpy (today's default)
    torch-cpu / float64  the torch backend on the CPU
    cuda / float32|float64   the torch backend on an accelerator (CUDA or ROCm)

Models: the 3-layer conv classifier of ``cost_head`` (the one quant_verify's backward engine was
benchmarked on), MLPs for branch and bound, and the real ResNet18 export when
``/mnt/data/cache/claude-work/qs-work/resnet18.onnx`` exists.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

QS = "/mnt/data/cache/claude-work/qs-work"
BD = "/mnt/data/cache/claude-work/bd-bench"
BOX = {"x": (-1.0, 1.0)}


def _f32(x: Any) -> np.ndarray:
    return np.asarray(x, np.float32)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def head_models(size: int, ch: int, ncls: int = 10, seed: int = 0):
    """3 x (Conv, Relu) -> GlobalAveragePool -> Flatten -> MatMul+Add(ncls); float and int8 fake-quantized."""
    from onnx import numpy_helper as nh
    from onnx import parser

    rng = np.random.default_rng(seed)
    chs = [3, ch, ch, ch]
    convs = [
        (
            _f32(rng.standard_normal((chs[i + 1], chs[i], 3, 3)) / np.sqrt(9 * chs[i])),
            _f32(rng.standard_normal(chs[i + 1]) * 0.1),
            2 if i == 1 else 1,
        )
        for i in range(3)
    ]
    head = (
        _f32(rng.standard_normal((ch, ncls)) / np.sqrt(ch)),
        _f32(rng.standard_normal(ncls) * 0.1),
    )

    def wq(w: np.ndarray, axis: int) -> np.ndarray:
        red = tuple(i for i in range(w.ndim) if i != axis)
        s = np.maximum(np.abs(w).max(axis=red, keepdims=True), 1e-12) / 127
        return _f32(np.round(w / s).clip(-128, 127) * s)

    def build(cv, hd, scales=None):
        lines, prev, inits = [], "x", {}
        for i, (w, b, st) in enumerate(cv):
            inits[f"W{i}"], inits[f"B{i}"] = w, b
            lines += [
                f"a{i} = Conv<strides=[{st},{st}], pads=[1,1,1,1]>({prev}, W{i}, B{i})",
                f"r{i} = Relu(a{i})",
            ]
            out = f"r{i}"
            if scales is not None:
                inits[f"S{i}"], inits[f"Z{i}"] = _f32(scales[i]), np.int8(0)
                lines += [
                    f"q{i} = QuantizeLinear({out}, S{i}, Z{i})",
                    f"d{i} = DequantizeLinear(q{i}, S{i}, Z{i})",
                ]
                out = f"d{i}"
            prev = out
        inits["V"], inits["C"] = hd
        lines += [
            f"p = GlobalAveragePool({prev})",
            "f = Flatten(p)",
            "m = MatMul(f, V)",
            "y = Add(m, C)",
        ]
        m = parser.parse_model(
            f'<ir_version: 9, opset_import: ["" : 13]> g (float[1,3,{size},{size}] x) => (float[1,{ncls}] y) {{\n'
            + "\n".join(lines)
            + "\n}"
        )
        m.graph.initializer.extend(
            nh.from_array(np.asarray(v), k) for k, v in inits.items()
        )
        return m

    base = build(convs, head)
    from onnxsim import interval

    res = interval.propagate(base, BOX)
    scales = [
        max(abs(min(res.hull(f"r{i}")[0], 0)), abs(res.hull(f"r{i}")[1]), 1e-6) / 127
        for i in range(3)
    ]
    q = build(
        [(wq(w, 0), b, st) for (w, b, st) in convs], (wq(head[0], 1), head[1]), scales
    )
    return base, q


def mlp_model(n_in: int, widths: Tuple[int, ...], n_out: int, seed: int = 1):
    from onnx import numpy_helper as nh
    from onnx import parser

    rng = np.random.default_rng(seed)
    dims = [n_in, *widths, n_out]
    lines, prev, inits = [], "x", {}
    for i in range(len(dims) - 1):
        inits[f"W{i}"] = _f32(rng.standard_normal((dims[i], dims[i + 1])) * 0.8)
        inits[f"B{i}"] = _f32(rng.standard_normal(dims[i + 1]) * 0.2)
        lines.append(f"m{i} = MatMul({prev}, W{i})\n a{i} = Add(m{i}, B{i})")
        prev = f"a{i}"
        if i < len(dims) - 2:
            lines.append(f"r{i} = Relu({prev})")
            prev = f"r{i}"
    lines.append(f"y = Identity({prev})")
    m = parser.parse_model(
        f'<ir_version: 8, opset_import: ["" : 13]> g (float[1,{n_in}] x) => (float[1,{n_out}] y) {{\n'
        + "\n".join(lines)
        + "\n}"
    )
    m.graph.initializer.extend(nh.from_array(v, k) for k, v in inits.items())
    return m


def resnet18(size: int, k: int = 10):
    """The real float ResNet18, pinned to ``size`` x ``size``, restricted to its first ``k`` logits."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(f"{QS}/resnet18.onnx")
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[0].ClearField("dim_param")
    dims[0].dim_value = 1
    dims[2].dim_value = size
    dims[3].dim_value = size
    od = m.graph.output[0].type.tensor_type.shape.dim
    od[0].ClearField("dim_param")
    od[0].dim_value = 1
    m = onnx.shape_inference.infer_shapes(m)
    out = m.graph.output[0].name
    for n in m.graph.node:
        for i, o in enumerate(n.output):
            if o == out:
                n.output[i] = out + "_full"
    for name, arr in {out + "_s": [0], out + "_e": [k], out + "_a": [1]}.items():
        m.graph.initializer.append(
            numpy_helper.from_array(np.array(arr, np.int64), name)
        )
    m.graph.node.append(
        helper.make_node(
            "Slice", [out + "_full", out + "_s", out + "_e", out + "_a"], [out]
        )
    )
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, k]))
    return m


def resnet_box(size: int):
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    lo = ((0 - mean) / std).reshape(1, 3, 1, 1) * np.ones((1, 3, size, size))
    hi = ((1 - mean) / std).reshape(1, 3, 1, 1) * np.ones((1, 3, size, size))
    return {"x": (lo, hi)}


# --------------------------------------------------------------------------
# cases: each returns a dict of results; the harness adds time and memory
# --------------------------------------------------------------------------


def _width(tb: Any) -> Dict[str, float]:
    return {
        "width_mean": float(np.mean(tb.hi - tb.lo)),
        "lo_min": float(tb.lo.min()),
        "hi_max": float(tb.hi.max()),
    }


def case_head_crown(
    size: int, ch: int, method: str = "crown", refine: bool = False, iters: int = 20
):
    def run(device, precision):
        from onnxsim import crown

        base, _ = head_models(size, ch)
        t = crown.bounds(
            base,
            BOX,
            method=method,
            refine=refine,
            alpha_iters=iters,
            device=device,
            precision=precision,
        )["y"]
        return {"method_used": t.method, **_width(t)}

    return run


def case_head_verify(size: int, ch: int):
    def run(device, precision):
        from onnxsim import quant_verify

        base, q = head_models(size, ch)
        rep = quant_verify.verify(
            base,
            q,
            BOX,
            engine="backward",
            breakdown=False,
            device=device,
            precision=precision,
        )
        return {"worst": float(rep.worst), "sites": len(rep.sites)}

    return run


def case_mlp_bab(budget: int, split: str, leaf: str, widths=(32, 32, 32)):
    def run(device, precision):
        from onnxsim import crown

        m = mlp_model(6, widths, 3)
        r = crown.bab_bounds(
            m,
            BOX,
            budget=budget,
            split=split,
            leaf_method=leaf,
            alpha_iters=15,
            device=device,
            precision=precision,
        )
        tb = r.bounds["y"]
        return {
            "evals": r.evaluations,
            "regions": r.regions,
            "exhausted": r.exhausted,
            **_width(tb),
        }

    return run


def case_conv_bab(size: int, ch: int, budget: int):
    def run(device, precision):
        from onnxsim import crown

        base, _ = head_models(size, ch)
        r = crown.bab_bounds(
            base, BOX, budget=budget, split="input", device=device, precision=precision
        )
        return {
            "evals": r.evaluations,
            "regions": r.regions,
            "exhausted": r.exhausted,
            **_width(r.bounds["y"]),
        }

    return run


def case_resnet_crown(size: int, k: int = 10):
    def run(device, precision):
        from onnxsim import crown

        m = resnet18(size, k)
        t = crown.bounds(
            m,
            resnet_box(size),
            method="crown",
            refine=False,
            device=device,
            precision=precision,
        )[m.graph.output[0].name]
        return {"method_used": t.method, **_width(t)}

    return run


def case_resnet_verify(size: int, scheme: str = "weight_only", k: int = 10):
    def run(device, precision):
        import onnx

        from onnxsim import quant_verify

        path = f"{BD}/resnet18_{scheme}_{size}.onnx"
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing cached quantized export {path}")
        fa = resnet18(size, k)
        qa = onnx.load(path)
        # same restriction as the reference model: keep the first k logits
        from onnx import TensorProto, helper, numpy_helper

        out = qa.graph.output[0].name
        for n in qa.graph.node:
            for i, o in enumerate(n.output):
                if o == out:
                    n.output[i] = out + "_full"
        for name, arr in {out + "_s": [0], out + "_e": [k], out + "_a": [1]}.items():
            qa.graph.initializer.append(
                numpy_helper.from_array(np.array(arr, np.int64), name)
            )
        qa.graph.node.append(
            helper.make_node(
                "Slice", [out + "_full", out + "_s", out + "_e", out + "_a"], [out]
            )
        )
        del qa.graph.output[:]
        qa.graph.output.append(
            helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, k])
        )
        rep = quant_verify.verify(
            fa,
            qa,
            resnet_box(size),
            engine="backward",
            breakdown=False,
            device=device,
            precision=precision,
        )
        return {"worst": float(rep.worst), "sites": len(rep.sites)}

    return run


def case_transfer(size: int = 224):
    """Time to move every ResNet18 weight to the device once (the per-call upload cost)."""

    def run(device, precision):
        import onnx
        from onnx import numpy_helper

        if device in (None, "cpu"):
            return {"skipped": "no device"}
        import torch

        m = onnx.load(f"{QS}/resnet18.onnx")
        arrs = [numpy_helper.to_array(t) for t in m.graph.initializer]
        dev = "cpu" if device == "torch-cpu" else device
        dt = torch.float32 if precision == "float32" else torch.float64
        t0 = time.perf_counter()
        ts = [
            torch.as_tensor(a.astype(np.float64)).to(device=dev, dtype=dt) for a in arrs
        ]
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        dt_s = time.perf_counter() - t0
        return {
            "upload_s": dt_s,
            "mb": sum(a.size for a in arrs)
            * (4 if precision == "float32" else 8)
            / 1e6,
            "n": len(ts),
        }

    return run


def build_cases() -> Dict[str, Callable[..., Dict[str, Any]]]:
    c: Dict[str, Callable[..., Dict[str, Any]]] = {}
    for ch in (8, 16):
        for s in (16, 32, 64, 112, 224):
            c[f"head_crown_{s}_{ch}"] = case_head_crown(s, ch)
            c[f"head_verify_{s}_{ch}"] = case_head_verify(s, ch)
    for s in (8, 16):
        c[f"head_refine_{s}_8"] = case_head_crown(s, 8, refine=True)
    for s in (8, 16, 32):
        c[f"head_alpha_{s}_8"] = case_head_crown(s, 8, method="alpha", iters=20)
    c["mlp_bab_64_input"] = case_mlp_bab(64, "input", "crown")
    c["mlp_bab_256_input"] = case_mlp_bab(256, "input", "crown")
    c["mlp_bab_32_relu_beta"] = case_mlp_bab(32, "relu", "beta")
    c["conv_bab_8_8_16"] = case_conv_bab(8, 8, 16)
    c["conv_bab_16_8_16"] = case_conv_bab(16, 8, 16)
    for s in (64, 224):
        c[f"resnet_crown_{s}"] = case_resnet_crown(s)
        c[f"resnet_verify_{s}"] = case_resnet_verify(s)
    c["transfer_resnet18"] = case_transfer()
    return c


SUITES = {
    "quick": [
        "head_crown_16_8",
        "head_crown_32_8",
        "head_crown_64_8",
        "head_verify_16_8",
        "head_verify_32_8",
        "head_alpha_16_8",
        "mlp_bab_64_input",
        "mlp_bab_32_relu_beta",
        "conv_bab_8_8_16",
    ],  # fmt: skip
    "full": [],  # filled below: everything
}


def run_one(name: str, device: str, precision: str, reps: int) -> Dict[str, Any]:
    cases = build_cases()
    fn = cases[name]
    dev = None if device == "cpu" else device
    prec = (
        None if precision == "float64" and device in ("cpu", "torch-cpu") else precision
    )
    out: Dict[str, Any] = {"case": name, "device": device, "precision": precision}
    cuda = False
    try:
        import torch

        cuda = bool(device.startswith("cuda")) and torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            out["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    times: List[float] = []
    res: Dict[str, Any] = {}
    for r in range(reps):
        t0 = time.perf_counter()
        res = fn(dev, prec)
        times.append(time.perf_counter() - t0)
        if times[0] > 20.0:  # a slow case is run once
            break
    out.update(res)
    out["time_first_s"] = round(times[0], 4)
    out["time_s"] = round(float(np.median(times)), 4)
    out["reps"] = len(times)
    out["host_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    if cuda:
        out["device_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--suite", choices=["quick", "full"], default="quick")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="float64")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--timeout", type=int, default=900, help="seconds per case (suite mode)"
    )
    ap.add_argument("--out", help="append JSON lines here (suite mode)")
    ap.add_argument(
        "--tag", default="", help="free-form label stored with every result"
    )
    args = ap.parse_args()
    cases = build_cases()
    if args.list:
        print("\n".join(cases))
        return 0
    if args.case:
        print(json.dumps(run_one(args.case, args.device, args.precision, args.reps)))
        return 0
    names = list(cases) if args.suite == "full" else SUITES["quick"]
    for name in names:
        cmd = [
            sys.executable,
            __file__,
            "--case",
            name,
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--reps",
            str(args.reps),
        ]
        t0 = time.time()
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.timeout
            )
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            if p.returncode == 0 and lines:
                rec = json.loads(lines[-1])
            else:
                rec = {"case": name, "device": args.device, "precision": args.precision,
                       "error": (p.stderr.strip().splitlines() or ["failed"])[-1][:300], "rc": p.returncode}  # fmt: skip
        except subprocess.TimeoutExpired:
            rec = {
                "case": name,
                "device": args.device,
                "precision": args.precision,
                "timeout_s": args.timeout,
            }
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["tag"] = args.tag
        line = json.dumps(rec)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
