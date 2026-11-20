#!/usr/bin/env python3
"""Per-layer device timing for a static-shape ONNX model on Metis.

The card has no per-layer profile API. So each node of the model is split into its own
ONNX model, compiled with the Voyager SDK, and run on the card in isolation. The device
kernel time per run comes from the runtime's host trace (the ``krn`` field, in
nanoseconds). The wall time comes from the host clock.

Each layer is measured alone, so its numbers include per-call overhead and do not add up
to the fused model's latency. Compare layers with each other, not with the full model.

Run with the Voyager SDK environment active:

    python layer_profile.py model.onnx --frames 200
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="layerprof-"))
TRACE = WORK / "host_trace.log"
# The runtime reads these when it starts, so they must be set before the context is created.
os.environ["AXE_LZ_PERF_LOG_PATH"] = str(TRACE)
os.environ["AXE_PROFILING_CONFIG"] = "HOST_TEMPLATE"

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, shape_inference  # noqa: E402

from axelera import compiler  # noqa: E402
from axelera.runtime import Context  # noqa: E402


def split_nodes(model_path: str):
    """Yield (name, op_type, sub_model) per node. Shapes come from shape inference."""
    model = shape_inference.infer_shapes(onnx.load(model_path))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    infos = {vi.name: vi for vi in list(g.input) + list(g.output) + list(g.value_info)}
    for idx, node in enumerate(g.node):
        ins = [infos[n] for n in node.input if n and n not in inits and n in infos]
        outs = [infos[n] for n in node.output]
        init_used = [inits[n] for n in node.input if n in inits]
        graph = helper.make_graph([node], f"layer{idx}", ins, outs, initializer=init_used)
        sub = helper.make_model(graph, opset_imports=model.opset_import)
        sub.ir_version = model.ir_version
        yield f"{idx:02d}_{node.name or node.op_type}", node.op_type, sub


def compile_layer(sub, out_dir: Path, samples: int) -> Path:
    path = out_dir / "layer.onnx"
    onnx.save(sub, path)
    rng = np.random.default_rng(0)
    shape = [d.dim_value for d in sub.graph.input[0].type.tensor_type.shape.dim]
    calib = (rng.uniform(-1, 1, shape).astype(np.float32) for _ in range(samples))
    cfg = compiler.CompilerConfig()
    q = compiler.quantize(model=str(path), calibration_dataset=calib, config=cfg)
    compiler.compile(model=q, config=cfg, output_dir=out_dir / "compiled")
    return out_dir / "compiled" / "model.json"


def read_trace(offset: int):
    """Parse the complete trace lines after ``offset``. Returns (rows, new_offset)."""
    if not TRACE.exists():
        return [], offset
    data = TRACE.read_bytes()[offset:]
    rows, consumed = [], 0
    for raw in data.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break
        consumed += len(raw)
        fields = dict(p.split(":", 1) for p in raw.decode().split()[1:] if ":" in p)
        rows.append({k: int(fields[k], 16) for k in ("krn", "tot") if k in fields})
    return rows, offset + consumed


def measure(ctx: Context, model_json: Path, frames: int) -> dict:
    m = ctx.load_model(str(model_json))
    conn = ctx.device_connect(None, num_sub_devices=1)
    inst = conn.load_model_instance(m, num_sub_devices=1, aipu_cores=1, host_profiling=True)
    x = [np.zeros(m.inputs()[0].shape, m.inputs()[0].dtype)]
    o = [np.zeros(t.shape, t.dtype) for t in m.outputs()]
    for _ in range(10):  # warm-up
        inst.run(x, o)
    _, offset = read_trace(TRACE.stat().st_size if TRACE.exists() else 0)
    walls = []
    for _ in range(frames):
        t0 = time.perf_counter()
        inst.run(x, o)
        walls.append((time.perf_counter() - t0) * 1e6)
    rows, _ = read_trace(offset)
    krn = [r["krn"] / 1000 for r in rows if "krn" in r]
    tot = [r["tot"] / 1000 for r in rows if "tot" in r]
    return {
        "kernel_us_median": statistics.median(krn) if krn else None,
        "host_total_us_median": statistics.median(tot) if tot else None,
        "wall_us_median": statistics.median(walls),
        "frames": len(walls),
        "trace_lines": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--calib", type=int, default=16)
    args = ap.parse_args()

    results = []
    for name, op, sub in split_nodes(args.model):
        out = WORK / name
        out.mkdir(parents=True, exist_ok=True)
        try:
            model_json = compile_layer(sub, out, args.calib)
        except Exception as exc:  # the SDK rejects some single-op graphs, e.g. a lone activation
            first = str(exc).strip().splitlines()[0][:120] if str(exc).strip() else type(exc).__name__
            results.append({"layer": name, "op": op, "error": first})
            print(json.dumps(results[-1]), flush=True)
            continue
        # A fresh context per layer: each measurement holds a device connection, and the card
        # has only 4 sub-devices, so connections must be released before the next layer.
        ctx = Context()
        stats = measure(ctx, model_json, args.frames)
        del ctx
        gc.collect()
        results.append({"layer": name, "op": op, **stats})
        print(json.dumps(results[-1]), flush=True)

    print("\nper-layer (each layer run alone; kernel time from the runtime host trace):")
    print(f"{'layer':32} {'op':10} {'kernel us':>10} {'host us':>10} {'wall us':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['layer']:32} {r['op']:10} not isolatable: {r['error']}")
            continue
        k = "n/a" if r["kernel_us_median"] is None else f"{r['kernel_us_median']:.1f}"
        h = "n/a" if r["host_total_us_median"] is None else f"{r['host_total_us_median']:.1f}"
        print(f"{r['layer']:32} {r['op']:10} {k:>10} {h:>10} {r['wall_us_median']:10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
