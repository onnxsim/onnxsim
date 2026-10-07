#!/usr/bin/env python3
"""auto_LiRPA reference bounds in float64 (an EXTERNAL oracle; no auto_LiRPA code is used by onnxsim).

Run inside the virtualenv that has torch + auto_LiRPA (see vnncomp_ref_bounds.py). It reuses that
script's tiny ONNX interpreter but keeps every tensor in float64, because the float32 reference of
the first conformance run can by itself produce differences above a 1e-6 relative threshold on deep
networks. Methods: IBP, CROWN (auto_LiRPA computes intermediate bounds with CROWN) and
CROWN with interval intermediate bounds (``init_bounds``-free: ``CROWN`` after forcing IBP
intermediates is available as method ``backward`` with ``bound_opts={'forward_refinement': False}``
is NOT assumed; only what the installed version supports is run and anything else is recorded).

  python vnncomp_tightness_ref64.py specs.json --out ref64.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import onnx
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vnncomp_ref_bounds as R  # noqa: E402  (the float32 reference's ONNX interpreter)
from auto_LiRPA import BoundedModule, BoundedTensor  # noqa: E402
from auto_LiRPA.perturbations import PerturbationLpNorm  # noqa: E402

torch.set_default_dtype(torch.float64)


def _register64(self: Any, name: str, value: torch.Tensor) -> None:
    safe = "c_" + "".join(ch if ch.isalnum() else "_" for ch in name)
    self.register_buffer(safe, value.double() if value.is_floating_point() else value)
    self.names[name] = safe


R.OnnxNet._register = _register64  # type: ignore[method-assign]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="IBP,CROWN")
    a = ap.parse_args()
    out: List[Dict[str, Any]] = []
    for rec in json.load(open(a.specs)):
        model = onnx.load(rec["onnx"])
        inits = {t.name for t in model.graph.initializer}
        in_shape = [
            int(d.dim_value) if d.dim_value > 0 else 1
            for d in [i for i in model.graph.input if i.name not in inits][
                0
            ].type.tensor_type.shape.dim
        ]
        net = R.OnnxNet(model).eval()
        lo = torch.tensor(np.array(rec["lo"], dtype=np.float64)).reshape(in_shape)
        hi = torch.tensor(np.array(rec["hi"], dtype=np.float64)).reshape(in_shape)
        C = torch.tensor(np.array(rec["A"], dtype=np.float64)).unsqueeze(0)
        res: Dict[str, Any] = {"key": rec["key"], "group": rec["group"]}
        for method in a.methods.split(","):
            try:
                # "CROWN+ibpcmp" = CROWN with auto_LiRPA's own option that intersects every intermediate
                # backward bound with the interval bound (compare_crown_with_ibp, default False).
                # "CROWN+dense" = also disable sparse intermediate bounds (default True).
                opts = {"conv_mode": "matrix"}
                if "ibpcmp" in method:
                    opts["compare_crown_with_ibp"] = True
                if "dense" in method:
                    opts["sparse_intermediate_bounds"] = False
                bm = BoundedModule(
                    net,
                    torch.zeros(in_shape, dtype=torch.float64),
                    device="cpu",
                    bound_opts=opts,
                )
                x = BoundedTensor(
                    (lo + hi) / 2, PerturbationLpNorm(norm=float("inf"), x_L=lo, x_U=hi)
                )
                t0 = time.time()
                lb, ub = bm.compute_bounds(x=(x,), method=method.split("+")[0], C=C)
                res[method] = {
                    "lb": lb.detach().reshape(-1).tolist(),
                    "ub": ub.detach().reshape(-1).tolist(),
                    "dtype": str(lb.dtype),
                    "seconds": round(time.time() - t0, 3),
                }
            except Exception as e:  # noqa: BLE001 - recorded, never hidden
                res[method] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        out.append(res)
        print(
            rec["key"],
            rec["group"],
            {
                m: res[m].get("dtype", res[m].get("error"))
                for m in res
                if m not in ("key", "group")
            },
            flush=True,
        )
        json.dump(out, open(a.out, "w"))
    print("wrote", a.out, len(out), "groups")


if __name__ == "__main__":
    main()
