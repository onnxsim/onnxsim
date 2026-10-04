#!/usr/bin/env python3
"""Does a larger batch raise RK3588 NPU utilisation? Measured, not assumed.

Context: `scripts/rknn/nanopc_npu_utilization.py` measured the harness at
~26% of one core (~9% of the 3-core NPU) for a single serialized inference
stream. Batch size is the obvious lever, so this tests it directly rather than
assuming.

What it tries, in order:

1. **Explicit batch in the ONNX graph** -- builds the model with a batch
   dimension N > 1 and compiles it for `rk3588`. This is the only real route: an
   N-D batch is a property of the graph, not a converter knob.
2. **`batch_size` as a config kwarg** -- checked because it existed in
   `rknn-toolkit` (v1). `rknn-toolkit2`'s `config()` has no such parameter
   (verified by signature), so this documents that it is genuinely gone rather
   than silently ignored.
3. **Whether a batched model actually improves *utilisation*** -- the honest
   question. A batched model amortizes one submission over N inputs, which
   raises NPU work per call, but the same host-side per-call overhead still
   applies. If the NPU is not the bottleneck, batch will not help throughput.

Writes a JSON report; exits 0 if any batch > 1 compiled.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_HERE)
for _p in (_SCRIPTS_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _install_onnx_mapping_shim() -> None:
    """RKNN-Toolkit2 2.3.2 still reads the removed ``onnx.mapping``."""
    import types

    if hasattr(onnx, "mapping"):
        return
    mapping = types.ModuleType("onnx.mapping")
    table = {
        d: onnx.helper.tensor_dtype_to_np_dtype(d)
        for d in onnx.TensorProto.DataType.values()
        if d != onnx.TensorProto.UNDEFINED
    }
    mapping.TENSOR_TYPE_TO_NP_TYPE = table
    mapping.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in table.items()}
    onnx.mapping = mapping
    sys.modules["onnx.mapping"] = mapping


def build_batched_model(batch: int, channels: int = 3, size: int = 224,
                        width: int = 32) -> onnx.ModelProto:
    """A small Conv-BN-Relu-Conv-Relu-GAP block with a real batch dimension.

    Channel bookkeeping is the fiddly part, and two mistakes here both produce
    confusing downstream errors rather than a shape-inference complaint at the
    node:

    * BN parameter tensors are sized to the *preceding* conv's output channels
      (``channels`` for ``bn0``), not to the model's eventual width.
    * ``conv1`` takes ``r`` (which has ``channels`` channels) as input, so its
      weight must be ``(width, channels, 3, 3)`` -- not ``(width, width, 3, 3)``.
      With the wrong shape, onnx's own checker passes but ORT fails at *run*
      time inside RKNN's constant-folding session with
      ``Input channels C is not equal to kernel channels * group. C: 16 kernel
      channels: 32``, because RKNN folds through an NCHWc pass first.
    """
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal((channels, channels, 3, 3)).astype(np.float32) * 0.05
    w1 = rng.standard_normal((width, channels, 3, 3)).astype(np.float32) * 0.05
    initializers = [
        numpy_helper.from_array(w0, "w0"),
        # bn0 follows conv0, whose output has `channels` channels.
        numpy_helper.from_array(np.ones(channels, np.float32), "bn_s"),
        numpy_helper.from_array(np.zeros(channels, np.float32), "bn_b"),
        numpy_helper.from_array(np.zeros(channels, np.float32), "bn_m"),
        numpy_helper.from_array(np.ones(channels, np.float32), "bn_v"),
        numpy_helper.from_array(w1, "w1"),
    ]
    nodes = [
        helper.make_node("Conv", ["x", "w0"], ["c"], pads=[1, 1, 1, 1], name="conv0"),
        helper.make_node("BatchNormalization", ["c", "bn_s", "bn_b", "bn_m", "bn_v"],
                         ["bn"], name="bn0", epsilon=1e-5),
        helper.make_node("Relu", ["bn"], ["r"], name="relu0"),
        helper.make_node("Conv", ["r", "w1"], ["c2"], pads=[1, 1, 1, 1], name="conv1"),
        helper.make_node("Relu", ["c2"], ["r2"], name="relu1"),
        helper.make_node("GlobalAveragePool", ["r2"], ["g"], name="gap"),
        # Flatten so the output is [N, width] rather than [N, width, 1, 1];
        # keeps the batch dim visible in the RKNN model's output shape.
        helper.make_node("Flatten", ["g"], ["y"], name="flat", axis=1),
    ]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, channels, size, size])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, width])
    graph = helper.make_graph(nodes, f"batched{batch}", [x], [y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def compile_rknn(onnx_path: str, rknn_path: str, calib: list[np.ndarray],
                 optimization_level: int) -> None:
    _install_onnx_mapping_shim()
    from rknn.api import RKNN

    calib_dir = rknn_path + ".calib"
    os.makedirs(calib_dir, exist_ok=True)
    paths = []
    for i, arr in enumerate(calib):
        p = os.path.join(calib_dir, f"{i:03d}.npy")
        np.save(p, arr.astype(np.float32))
        paths.append(p)
    txt = calib_dir + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")

    r = RKNN(verbose=False)
    try:
        for name, ret in (
            ("config", r.config(target_platform="rk3588",
                                mean_values=[[0.0, 0.0, 0.0]],
                                std_values=[[1.0, 1.0, 1.0]],
                                optimization_level=optimization_level)),
            ("load_onnx", r.load_onnx(model=onnx_path)),
            ("build", r.build(do_quantization=True, dataset=txt)),
            ("export", r.export_rknn(rknn_path)),
        ):
            if ret != 0:
                raise RuntimeError(f"rknn {name} failed: {ret}")
    finally:
        r.release()


def _remote_base(args: argparse.Namespace, tool: str) -> list[str]:
    """``ssh``/``scp`` prefix wrapped in ``sshpass`` when a password is given."""
    options = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "LogLevel=ERROR"]
    base = ["sshpass", "-p", args.ssh_password] if args.ssh_password else []
    return [*base, tool, *options]


def measure_utilization(args: argparse.Namespace, work: str, model: str,
                        feeds: str, duration: float,
                        interval: float = 0.5) -> dict:
    """Run the model on the board and time-average /sys/kernel/debug/rknpu/load."""
    import re

    prefix = [*_remote_base(args, "ssh"), f"{args.user}@{args.host}"]
    marker = f"rknn-batch-{os.getpid()}"
    proc = subprocess.Popen(
        [*prefix, f"cd {work} && exec -a {marker} python3 nanopc_rknn_runner.py "
                  f"{model} --input-feeds {feeds} --warmup 5 --iterations 100000 "
                  f"--output-dir {work}/batch_out"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    samples: list[list[int]] = []
    try:
        time.sleep(2.0)
        deadline = time.time() + duration
        while time.time() < deadline:
            out = subprocess.run(
                [*prefix, "echo pi | sudo -S cat /sys/kernel/debug/rknpu/load"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            found = [int(m.group(2)) for m in re.finditer(r"Core(\d+):\s*(\d+)%", out)]
            if len(found) == 3:
                samples.append(found)
            time.sleep(interval)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(
            [*prefix, f"pkill -f {marker} || true; sleep 1; pkill -9 -f {marker} || true"],
            capture_output=True, text=True, timeout=30,
        )
    if not samples:
        return {}
    cores = [statistics.mean(s[i] for s in samples) for i in range(3)]
    return {
        "per_core_mean_pct": [round(c, 1) for c in cores],
        "capacity_used_pct": round(sum(cores) / 3.0, 1),
        "samples": len(samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--host", default="nanopc-t6.tailf0b7b1.ts.net")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--ssh-password", default=None)
    ap.add_argument("--work-dir", default="/tmp/rknn-batch-work")
    ap.add_argument("--remote-dir", default="/tmp/rknn-batch")
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--skip-board", action="store_true",
                    help="only report whether each batch compiles")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.ssh_password is None:
        args.ssh_password = "pi"
    os.makedirs(args.work_dir, exist_ok=True)

    _install_onnx_mapping_shim()
    import inspect

    from rknn.api import RKNN

    report: dict = {
        "config_has_batch_size_kwarg": "batch_size" in inspect.signature(RKNN.config).parameters,
        "config_parameters": sorted(inspect.signature(RKNN.config).parameters),
        "results": [],
    }

    ssh = [*_remote_base(args, "ssh"), f"{args.user}@{args.host}"]
    scp = [*_remote_base(args, "scp")]
    subprocess.run([*ssh, f"mkdir -p {args.remote_dir}"], check=True,
                   capture_output=True, text=True)
    subprocess.run(
        [*scp, os.path.join(_HERE, "nanopc_rknn_runner.py"),
         f"{args.user}@{args.host}:{args.remote_dir}/nanopc_rknn_runner.py"],
        check=True, capture_output=True, text=True,
    )

    for batch in args.batches:
        entry: dict = {"batch": batch}
        model = build_batched_model(batch)
        onnx_path = os.path.join(args.work_dir, f"batch{batch}.onnx")
        rknn_path = os.path.join(args.work_dir, f"batch{batch}.rknn")
        onnx.save(model, onnx_path)

        # Calibration must match the model's declared batch.
        rng = np.random.default_rng(batch)
        calib = [rng.uniform(0, 1, (batch, 3, 224, 224)).astype(np.float32) for _ in range(4)]

        try:
            compile_rknn(onnx_path, rknn_path, calib, args.optimization_level)
            entry["compiled"] = True
            entry["rknn_size_kib"] = round(os.path.getsize(rknn_path) / 1024, 1)
        except Exception as exc:
            entry["compiled"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
            report["results"].append(entry)
            print(f"batch={batch}: FAILED to compile -- {entry['error']}", flush=True)
            continue

        print(f"batch={batch}: compiled OK ({entry['rknn_size_kib']} KiB)", flush=True)

        if not args.skip_board:
            feeds_path = os.path.join(args.work_dir, f"batch{batch}.feeds.npz")
            np.savez(feeds_path, x=calib[0])
            remote = f"{args.remote_dir}/batch{batch}"
            subprocess.run([*ssh, f"mkdir -p {remote}"], check=True,
                           capture_output=True, text=True)
            for local, name in ((rknn_path, "model.rknn"), (feeds_path, "feeds.npz")):
                subprocess.run(
                    [*scp, local, f"{args.user}@{args.host}:{remote}/{name}"],
                    check=True, capture_output=True, text=True,
                )
            # The runner runs with cwd=remote, so it must live *there* too --
            # uploading it only to args.remote_dir leaves every batch run
            # failing with "can't open file .../nanopc_rknn_runner.py".
            subprocess.run(
                [*scp, os.path.join(_HERE, "nanopc_rknn_runner.py"),
                 f"{args.user}@{args.host}:{remote}/nanopc_rknn_runner.py"],
                check=True, capture_output=True, text=True,
            )
            entry["utilization"] = measure_utilization(
                args, remote, "model.rknn", "feeds.npz", args.duration,
            )
            u = entry["utilization"]
            if u:
                print(f"          cores={u['per_core_mean_pct']} "
                      f"-> {u['capacity_used_pct']}% of NPU capacity", flush=True)

        report["results"].append(entry)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.output}", flush=True)
    else:
        print("\n" + json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())