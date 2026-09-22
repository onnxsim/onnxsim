#!/usr/bin/env python3
"""Time every unique int8 Conv shape in the Mask R-CNN backbone once, on the real phone.

Weighted by occurrence count, this reconstructs the backbone's per-layer-type time breakdown
without needing `tvm.contrib.hexagon`'s `get_graph_debug_executor().run_individual()`, which on
this device fails even on a one-op graph at the `GetFunction` stage (most likely this Hexagon RPC
skeleton build lacks `USE_PROFILER` support -- not something fixable from the Python side).

Run `prepare.py` first. Shapes are extracted from `workdir/backbone.onnx` via
`onnx.shape_inference` and deduplicated (ResNet-50 repeats bottleneck shapes across its 3/4/6/3
blocks: 76 Conv nodes typically collapse to ~37 unique shapes). Each unique shape becomes a
standalone `qnn.conv2d` + `qnn.requantize` module, built and timed on the phone once.

Isolated per-shape timings necessarily *overshoot* the real fused-graph total (no cross-op
layout/weight-prepacking sharing, no pipelining): expect roughly 2x the measured full-backbone
time. Use the percentage breakdown, not the absolute milliseconds, to prioritize what to look at;
see `../README.md`'s "Full backbone profile" section for the results this produced and what they
mean. Companion script: `profile_noncon_ops.py` (Resize/MaxPool/Add/Sigmoid).
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


def extract_unique_conv_shapes(backbone_path: Path):
    """[(ishape, wshape, stride, pad, count), ...], sorted by count descending."""
    model = onnx.shape_inference.infer_shapes(onnx.load(str(backbone_path)))
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    }
    initializers = {i.name: list(i.dims) for i in model.graph.initializer}
    producer = {output: node for node in model.graph.node for output in node.output}

    def weight_shape(name):
        node = producer.get(name)
        while node is not None and node.op_type == "DequantizeLinear":
            name = node.input[0]
            node = producer.get(name)
        return initializers.get(name)

    convs = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        ishape, wshape = shapes.get(node.input[0]), weight_shape(node.input[1])
        if not ishape or not wshape:
            continue
        stride = _attr(node, "strides", [1, 1])
        pads = _attr(node, "pads", [0, 0, 0, 0])
        convs.append((tuple(ishape), tuple(wshape), tuple(stride), tuple(pads[:2])))

    counts = collections.Counter(convs)
    return [
        (list(k[0]), list(k[1]), list(k[2]), list(k[3]), v)
        for k, v in counts.most_common()
    ]  # fmt: skip


def conv_module(rng, ishape, wshape, stride, pad):
    cout, cin, kh, kw = wshape
    weight = rng.integers(-40, 40, wshape).astype("int8")
    data = relay.var("data", shape=ishape, dtype="uint8")
    conv = relay.qnn.op.conv2d(
        data, relay.const(weight), relay.const(114, "int32"), relay.const(0, "int32"),
        relay.const(0.02, "float32"), relay.const(0.005, "float32"), kernel_size=(kh, kw),
        channels=cout, strides=tuple(stride), padding=tuple(pad), out_dtype="int32",
    )  # fmt: skip
    out = relay.qnn.op.requantize(
        conv, relay.const(0.0001, "float32"), relay.const(0, "int32"),
        relay.const(0.05, "float32"), relay.const(100, "int32"), out_dtype="uint8",
    )  # fmt: skip
    mod = tvm.IRModule.from_expr(relay.Function([data], out))
    image = rng.integers(0, 255, ishape).astype("uint8")
    out_h = (ishape[2] + 2 * pad[0] - kh) // stride[0] + 1
    out_w = (ishape[3] + 2 * pad[1] - kw) // stride[1] + 1
    macs = cout * cin * kh * kw * out_h * out_w
    return mod, image, macs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("maskrcnn_work"))
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("conv_profile.json"))
    args = parser.parse_args()

    shapes = extract_unique_conv_shapes(args.workdir / "backbone.onnx")
    print(
        f"{sum(c for *_, c in shapes)} Conv nodes, {len(shapes)} unique shapes",
        flush=True,
    )
    rng = np.random.default_rng(7)
    arch = tvm.target.hexagon("v73")
    target = tvm.target.Target(arch, host=arch)
    tracker = Tracker(host="127.0.0.1", port=9197)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9197,
            "rpc_server_port": 7077,
            "workspace_base": "/data/local/tmp/tvm_conv_profile",
            "adb_server_socket": None,
        },
    )
    results = []
    try:
        launcher.start_server()
        for ishape, wshape, stride, pad, count in shapes:
            mod, image, macs = conv_module(rng, ishape, wshape, stride, pad)
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target)
            with launcher.create_session() as session:
                executor = session.get_executor_from_factory(lib)
                executor.load_params(tvm.runtime.save_param_dict(lib.get_params()))
                executor.set_input("data", image)
                executor.run()
                times = []
                for _ in range(args.repeat):
                    start = time.time()
                    executor.run()
                    times.append(time.time() - start)
            median = float(np.median(times))
            results.append(
                {
                    "ishape": ishape, "wshape": wshape, "stride": stride, "pad": pad,
                    "count": count, "median_s": median, "macs": macs, "total_s": median * count,
                }
            )  # fmt: skip
            print(
                f"in={ishape} w={wshape} stride={stride} x{count:<2} {median * 1e3:8.3f} ms  "
                f"x{count} = {median * count * 1e3:9.3f} ms  {macs / median / 1e9:6.2f} GMAC/s",
                flush=True,
            )
    finally:
        launcher.stop_server()
        tracker.terminate()
    args.output.write_text(json.dumps(results, indent=1))
    total = sum(r["total_s"] for r in results)
    print(f"\nsum of all conv layers (weighted, isolated): {total * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
