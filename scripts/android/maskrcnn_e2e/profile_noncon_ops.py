#!/usr/bin/env python3
"""Time the backbone's non-conv operators (Resize, MaxPool, Add, Sigmoid) at their real shapes.

Companion to `profile_conv_shapes.py` -- see its docstring and `../README.md`'s "Full backbone
profile" section for the methodology and results (isolated per-shape timings overshoot the real
fused-graph total; use the percentage breakdown, not the absolute milliseconds).
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnx.shape_inference
import tvm
import tvm.contrib.hexagon  # noqa: F401
from tvm import relay
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.rpc.tracker import Tracker


def _attr(node, key, default=None):
    for a in node.attribute:
        if a.name == key:
            return list(a.ints) if a.ints else a.i
    return default


def extract_shapes(backbone_path: Path):
    model = onnx.shape_inference.infer_shapes(onnx.load(str(backbone_path)))
    shapes = {
        v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    }
    resize, maxpool, add, sigmoid = (collections.Counter() for _ in range(4))
    for node in model.graph.node:
        ishape = shapes.get(node.input[0])
        if not ishape:
            continue
        if node.op_type == "Resize":
            resize[ishape] += 1
        elif node.op_type == "MaxPool":
            maxpool[
                (
                    ishape,
                    tuple(_attr(node, "kernel_shape")),
                    tuple(_attr(node, "strides")),
                )
            ] += 1
        elif node.op_type == "Add":
            add[ishape] += 1
        elif node.op_type == "Sigmoid":
            sigmoid[ishape] += 1
    return resize, maxpool, add, sigmoid


def build_specs(resize, maxpool, add, sigmoid):
    specs = []
    for ishape, count in resize.items():
        out_shape = (ishape[0], ishape[1], ishape[2] * 2, ishape[3] * 2)
        specs.append(
            (
                "resize", ishape, count,
                lambda i=ishape, o=out_shape: relay.image.resize2d(
                    relay.var("d", shape=i, dtype="int8"),
                    size=o[2:], method="nearest_neighbor", rounding_method="round_prefer_floor",
                ),
                "int8",
            )
        )  # fmt: skip
    for (ishape, kernel, stride), count in maxpool.items():
        specs.append(
            (
                "maxpool", ishape, count,
                lambda i=ishape, k=kernel, s=stride: relay.nn.max_pool2d(
                    relay.var("d", shape=i, dtype="uint8"), pool_size=k, strides=s
                ),
                "uint8",
            )
        )  # fmt: skip
    for ishape, count in add.items():
        specs.append(
            (
                "add", ishape, count,
                lambda i=ishape: relay.add(
                    relay.var("a", shape=i, dtype="int32"), relay.var("b", shape=i, dtype="int32")
                ),
                "int32",
            )
        )  # fmt: skip
    for ishape, count in sigmoid.items():
        specs.append(
            (
                "sigmoid", ishape, count,
                lambda i=ishape: relay.sigmoid(relay.var("d", shape=i, dtype="float32")),
                "float32",
            )
        )  # fmt: skip
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("maskrcnn_work"))
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("noncon_profile.json"))
    args = parser.parse_args()

    specs = build_specs(*extract_shapes(args.workdir / "backbone.onnx"))
    rng = np.random.default_rng(9)
    arch = tvm.target.hexagon("v73")
    target = tvm.target.Target(arch, host=arch)
    tracker = Tracker(host="127.0.0.1", port=9198)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9198,
            "rpc_server_port": 7078,
            "workspace_base": "/data/local/tmp/tvm_noncon_profile",
            "adb_server_socket": None,
        },
    )
    results = []
    try:
        launcher.start_server()
        for name, ishape, count, build, dtype in specs:
            out = build()
            variables = relay.analysis.free_vars(out)
            mod = tvm.IRModule.from_expr(relay.Function(variables, out))
            inputs = {}
            for v in variables:
                shape = tuple(int(d) for d in v.type_annotation.shape)
                inputs[v.name_hint] = (
                    rng.normal(0, 3, shape).astype(dtype)
                    if dtype == "float32"
                    else rng.integers(0, 100, shape).astype(dtype)
                )
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target)
            with launcher.create_session() as session:
                executor = session.get_executor_from_factory(lib)
                executor.load_params(tvm.runtime.save_param_dict(lib.get_params()))
                for key, value in inputs.items():
                    executor.set_input(key, value)
                executor.run()
                times = []
                for _ in range(args.repeat):
                    start = time.time()
                    executor.run()
                    times.append(time.time() - start)
            median = float(np.median(times))
            results.append(
                {
                    "op": name,
                    "ishape": list(ishape),
                    "count": count,
                    "median_s": median,
                    "total_s": median * count,
                }
            )
            print(
                f"{name:8} in={ishape} x{count:<2} {median * 1e3:9.3f} ms  x{count} = {median * count * 1e3:9.3f} ms",
                flush=True,
            )
    finally:
        launcher.stop_server()
        tracker.terminate()
    args.output.write_text(json.dumps(results, indent=1))
    total = sum(r["total_s"] for r in results)
    print(f"\nsum of all non-conv ops (weighted, isolated): {total * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
