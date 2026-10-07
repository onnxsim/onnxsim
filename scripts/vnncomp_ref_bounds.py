#!/usr/bin/env python3
"""Reference bounds from auto_LiRPA (an EXTERNAL oracle; none of its code is used by onnxsim).

Runs in a separate virtualenv that has torch + auto_LiRPA (BSD-licensed, installed from
https://github.com/Verified-Intelligence/auto_LiRPA); it imports nothing from onnxsim. Input is
the JSON written by ``vnncomp_bench.py export-specs``; for every group (one input box, many spec
rows ``a . Y``) it computes sound lower bounds of the specs with auto_LiRPA's CROWN and
alpha-CROWN on the same network and box that onnxsim's engines are given.

  python vnncomp_ref_bounds.py specs.json --out ref.json [--alpha-iters 20]
"""

import argparse
import json
import time
from typing import Any, Dict, List

import numpy as np
import onnx
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from onnx import numpy_helper


class OnnxNet(torch.nn.Module):
    """A tiny ONNX-graph interpreter (MLP / CNN ops): enough to hand these networks to auto_LiRPA."""

    def __init__(self, model: onnx.ModelProto) -> None:
        super().__init__()
        self.nodes = [n for n in model.graph.node]
        inits = {t.name for t in model.graph.initializer}
        self.inp = [i.name for i in model.graph.input if i.name not in inits][0]
        self.out = model.graph.output[0].name
        self.names: Dict[str, str] = {}
        for t in model.graph.initializer:
            self._register(t.name, torch.tensor(numpy_helper.to_array(t)))
        for n in self.nodes:
            if n.op_type == "Constant":
                v = next(a for a in n.attribute if a.name == "value")
                self._register(n.output[0], torch.tensor(numpy_helper.to_array(v.t)))
        self.nodes = [n for n in self.nodes if n.op_type != "Constant"]

    def _register(self, name: str, value: torch.Tensor) -> None:
        safe = "c_" + "".join(ch if ch.isalnum() else "_" for ch in name)
        self.register_buffer(
            safe, value.float() if value.dtype == torch.float64 else value
        )
        self.names[name] = safe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F = torch.nn.functional
        env: Dict[str, Any] = {self.inp: x}
        for name, safe in self.names.items():
            env[name] = getattr(self, safe)
        for n in self.nodes:
            a = [env[i] for i in n.input if i]
            at = {k.name: onnx.helper.get_attribute_value(k) for k in n.attribute}
            op = n.op_type
            if op == "Sub":
                r = a[0] - a[1]
            elif op == "Add":
                r = a[0] + a[1]
            elif op == "Div":
                r = a[0] / a[1]
            elif op == "Mul":
                r = a[0] * a[1]
            elif op == "MatMul":
                r = a[0] @ a[1]
            elif op == "Gemm":
                u, v = a[0], a[1]
                v = v.t() if at.get("transB", 0) else v
                u = u.t() if at.get("transA", 0) else u
                r = u @ v
                if at.get("alpha", 1.0) != 1.0:
                    r = r * at["alpha"]
                if len(a) > 2:
                    r = r + (a[2] if at.get("beta", 1.0) == 1.0 else a[2] * at["beta"])
            elif op == "Relu":
                r = torch.relu(a[0])
            elif op == "Sigmoid":
                r = torch.sigmoid(a[0])
            elif op == "Tanh":
                r = torch.tanh(a[0])
            elif op == "Flatten":
                r = a[0].reshape(a[0].shape[0], -1)
            elif op == "Conv":
                pads = at.get("pads", [0, 0, 0, 0])
                r = F.conv2d(
                    a[0],
                    a[1],
                    a[2] if len(a) > 2 else None,
                    stride=tuple(at.get("strides", [1, 1])),
                    padding=(pads[0], pads[1]),
                    dilation=tuple(at.get("dilations", [1, 1])),
                    groups=at.get("group", 1),
                )
            elif op == "Identity":
                r = a[0]
            else:
                raise NotImplementedError(f"reference converter has no op {op}")
            env[n.output[0]] = r
        return env[self.out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha-iters", type=int, default=20)
    ap.add_argument("--no-alpha", action="store_true")
    a = ap.parse_args()
    specs = json.load(open(a.specs))
    out: List[Dict[str, Any]] = []
    for rec in specs:
        model = onnx.load(rec["onnx"])
        in_shape = [
            int(d.dim_value) if d.dim_value > 0 else 1
            for d in [
                i
                for i in model.graph.input
                if i.name not in {t.name for t in model.graph.initializer}
            ][0].type.tensor_type.shape.dim
        ]
        net = OnnxNet(model).eval()
        res: Dict[str, Any] = {"key": rec["key"], "group": rec["group"]}
        lo = torch.tensor(np.array(rec["lo"], dtype=np.float32)).reshape(in_shape)
        hi = torch.tensor(np.array(rec["hi"], dtype=np.float32)).reshape(in_shape)
        C = torch.tensor(np.array(rec["A"], dtype=np.float32)).unsqueeze(
            0
        )  # (1, rows, outputs)
        for method in ("CROWN",) + (() if a.no_alpha else ("alpha-CROWN",)):
            try:
                bm = BoundedModule(
                    net,
                    torch.zeros(in_shape),
                    device="cpu",
                    bound_opts={"conv_mode": "matrix"},
                )
                if method == "alpha-CROWN":
                    bm.set_bound_opts(
                        {
                            "optimize_bound_args": {
                                "iteration": a.alpha_iters,
                                "lr_alpha": 0.1,
                            }
                        }
                    )
                x = BoundedTensor(
                    (lo + hi) / 2, PerturbationLpNorm(norm=float("inf"), x_L=lo, x_U=hi)
                )
                t0 = time.time()
                lb, _ = bm.compute_bounds(x=(x,), method=method, C=C)
                res[method] = {
                    "lb": lb.detach().reshape(-1).tolist(),
                    "seconds": round(time.time() - t0, 3),
                }
            except Exception as e:  # noqa: BLE001 - recorded, never hidden
                res[method] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        out.append(res)
        print(
            rec["key"],
            rec["group"],
            {
                m: (r.get("seconds"), r.get("error"))
                for m, r in res.items()
                if m in ("CROWN", "alpha-CROWN")
            },
            flush=True,
        )
        json.dump(out, open(a.out, "w"))
    print("wrote", a.out, len(out), "groups")


if __name__ == "__main__":
    main()
