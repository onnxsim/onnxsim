#!/usr/bin/env python3
"""Measure per-call host overhead vs actual NPU compute on a RK3588 board.

Context: `nanopc_npu_utilization.py` measured a single inference stream at ~26%
of one NPU core, i.e. the NPU is idle ~74% of each call. The question this
answers is *what that idle time is*, and therefore what could reduce it.

Method: run a ladder of models that differ only in how much real NPU work they
do (same op mix, growing channel width / spatial size), and regress wall-clock
per-call latency against input size. For a fixed per-call overhead `F` plus a
per-unit-work cost `k`:

    latency(N) = F + k * N

so the **intercept is the fixed host-side cost** and the slope is the NPU work.
Two extra comparisons isolate specific candidate costs:

* **ctypes/Python call overhead** -- `rknn_run` is timed from Python, so a
  non-trivial part of "fixed" could be the language boundary rather than the
  driver. Timed against a null call through the same ctypes trampoline.
* **scheduling jitter** -- the board runs a GNOME desktop and is not isolated;
  `os.sched_setaffinity` + `SCHED_FIFO` are tested to see how much of the
  variance is preemption rather than real work.

Everything is reported as measured numbers; no claim here should be read as
"this fixes it" without the measured delta.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

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


def build_conv_stack(width: int, repeats: int, size: int = 32,
                     channels: int = 3) -> onnx.ModelProto:
    """`repeats` x (Conv3x3 + Relu) at fixed width, then GlobalAveragePool.

    Width and repeat count both scale the NPU's real work while leaving the
    op *mix* (and therefore the per-layer submission pattern) identical, which
    is what makes the regression meaningful.
    """
    rng = np.random.default_rng(0)
    initializers, nodes = [], []
    in_ch = channels
    prev = "x"
    for i in range(repeats):
        w = rng.standard_normal((width, in_ch, 3, 3)).astype(np.float32) * 0.02
        initializers.append(numpy_helper.from_array(w, f"w{i}"))
        nodes.append(helper.make_node("Conv", [prev, f"w{i}"], [f"c{i}"],
                                     pads=[1, 1, 1, 1], name=f"conv{i}"))
        nodes.append(helper.make_node("Relu", [f"c{i}"], [f"r{i}"], name=f"relu{i}"))
        prev = f"r{i}"
        in_ch = width
    nodes.append(helper.make_node("GlobalAveragePool", [prev], ["y"], name="gap"))
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels, size, size])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width])
    graph = helper.make_graph(nodes, f"stack_w{width}_r{repeats}", [x], [y],
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def compile_rknn(onnx_path: str, rknn_path: str, calib: list[np.ndarray]) -> None:
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
            ("config", r.config(target_platform="rk3588", optimization_level=3)),
            ("load_onnx", r.load_onnx(model=onnx_path)),
            ("build", r.build(do_quantization=True, dataset=txt)),
            ("export", r.export_rknn(rknn_path)),
        ):
            if ret != 0:
                raise RuntimeError(f"rknn {name} failed: {ret}")
    finally:
        r.release()


# ---------------------------------------------------------------- board side

BOARD_PROBE = r'''
import ctypes, json, os, statistics, sys, time
import numpy as np

RTN = ctypes.CDLL("/usr/lib/librknnrt.so")
RTN_TENSOR_FLOAT32, RTN_TENSOR_INT8 = 0, 2
RTN_TENSOR_NHWC = 1
RKNN_QUERY_IN_OUT_NUM, RKNN_QUERY_INPUT_ATTR, RKNN_QUERY_OUTPUT_ATTR = 0, 1, 2


class Num(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class Attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32), ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16), ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32), ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32), ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32), ("fl", ctypes.c_int8),
        ("_pad", ctypes.c_uint8 * 3), ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float), ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through_attr", ctypes.c_uint8), ("_pad2", ctypes.c_uint8 * 3),
        ("h_stride", ctypes.c_uint32),
    ]


class Mem(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("virt_addr", ctypes.c_void_p), ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32), ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


RTN.rknn_init.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
RTN.rknn_query.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                           ctypes.c_uint32]
RTN.rknn_create_mem.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
RTN.rknn_create_mem.restype = ctypes.POINTER(Mem)
RTN.rknn_set_io_mem.argtypes = [ctypes.c_uint32, ctypes.POINTER(Mem),
                                ctypes.POINTER(Attr)]
RTN.rknn_run.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
RTN.rknn_destroy.argtypes = [ctypes.c_uint32]
RTN.rknn_destroy_mem.argtypes = [ctypes.c_uint32, ctypes.POINTER(Mem)]

opts = json.loads(sys.argv[2])
model_path = sys.argv[1]

if opts.get("affinity"):
    # Pin to one CPU so preemption from the desktop does not dominate.
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[0]})
if opts.get("realtime"):
    # SCHED_FIFO needs CAP_SYS_NICE; fall back silently if not permitted.
    try:
        param = os.sched_param(1)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
    except Exception:
        pass

buf = ctypes.create_string_buffer(open(model_path, "rb").read())
ctx = ctypes.c_uint32(0)
assert RTN.rknn_init(ctypes.byref(ctx), buf, len(buf), 0, None) == 0

num = Num()
RTN.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(num), ctypes.sizeof(num))

in_type = RTN_TENSOR_INT8 if opts.get("pass_through") else RTN_TENSOR_FLOAT32


def _alloc_size(a):
    """Bytes to allocate for an IO buffer.

    Float view (pass_through=0): the runtime converts on write, needing the
    stride-padded extent at 4 bytes/element -- the same sizing rule as
    nanopc_rknn_runner._float_buffer_size.

    Pass-through view (pass_through=1): the buffer holds the model's native
    int8/fp16 data verbatim, so it is the stride-padded extent at the model's
    own element width.
    """
    padded = a.size_with_stride or a.size
    if opts.get("pass_through"):
        width = {1: 4, 2: 2, 3: 1}.get(a.type, 1)
        return padded * width
    return padded * 4


mems_in, mems_out = [], []
for i in range(num.n_input):
    a = Attr(index=i)
    RTN.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, ctypes.byref(a), ctypes.sizeof(a))
    size = _alloc_size(a)
    m = RTN.rknn_create_mem(ctx, size)
    mems_in.append(m)
    ctypes.memset(m.contents.virt_addr, 0, size)
    a.pass_through = 1 if opts.get("pass_through") else 0
    a.type = in_type
    if a.n_dims == 4:
        a.fmt = RTN_TENSOR_NHWC
    assert RTN.rknn_set_io_mem(ctx, m, ctypes.byref(a)) == 0
for i in range(num.n_output):
    a = Attr(index=i)
    RTN.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(a), ctypes.sizeof(a))
    size = _alloc_size(a)
    m = RTN.rknn_create_mem(ctx, size)
    mems_out.append(m)
    ctypes.memset(m.contents.virt_addr, 0, size)
    a.pass_through = 1 if opts.get("pass_through") else 0
    a.type = in_type
    assert RTN.rknn_set_io_mem(ctx, m, ctypes.byref(a)) == 0

warmup, iters = opts.get("warmup", 20), opts.get("iterations", 200)
for _ in range(warmup):
    RTN.rknn_run(ctx, None)

samples = []
for _ in range(iters):
    t0 = time.perf_counter_ns()
    RTN.rknn_run(ctx, None)
    samples.append((time.perf_counter_ns() - t0) / 1e6)

# Cost of the measurement boundary itself: `time.perf_counter_ns` plus an empty
# Python loop body, i.e. the floor any host-side timing here can resolve.
null = []
for _ in range(iters):
    t0 = time.perf_counter_ns()
    null.append((time.perf_counter_ns() - t0) / 1e6)

ordered = sorted(samples)
out = {
    "latency_ms": {
        "min": round(ordered[0], 4),
        "p50": round(statistics.median(ordered), 4),
        "mean": round(statistics.mean(ordered), 4),
        "p95": round(ordered[int(round((len(ordered) - 1) * 0.95))], 4),
        "max": round(ordered[-1], 4),
        "stdev": round(statistics.pstdev(ordered), 4),
    },
    "timer_floor_ms": round(statistics.median(null), 6),
    "affinity_pinned": bool(opts.get("affinity")),
    "realtime_sched": bool(opts.get("realtime")),
    "pass_through": bool(opts.get("pass_through")),
    "cpu": os.cpu_count(),
}
for m in mems_out + mems_in:
    RTN.rknn_destroy_mem(ctx, m)
RTN.rknn_destroy(ctx)
print(json.dumps(out))
'''


def _base(args: argparse.Namespace, tool: str) -> list[str]:
    options = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "LogLevel=ERROR"]
    base = ["sshpass", "-p", args.ssh_password] if args.ssh_password else []
    return [*base, tool, *options]


def run_probe(args: argparse.Namespace, model: str, **opts) -> dict:
    ssh = [*_base(args, "ssh"), f"{args.user}@{args.host}"]
    payload = json.dumps(opts)
    proc = subprocess.run(
        [*ssh, f"cd {args.remote_dir} && python3 overhead_probe.py {model} '{payload}'"],
        capture_output=True, text=True, timeout=300,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("{")), None)
    if not line:
        raise RuntimeError(f"probe failed: {proc.stdout[-500:]} {proc.stderr[-800:]}")
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="nanopc-t6.tailf0b7b1.ts.net")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--ssh-password", default=None)
    ap.add_argument("--work-dir", default="/tmp/rknn-overhead-work")
    ap.add_argument("--remote-dir", default="/tmp/rknn-overhead")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--repeats", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.ssh_password is None:
        args.ssh_password = "pi"
    os.makedirs(args.work_dir, exist_ok=True)

    ssh = [*_base(args, "ssh"), f"{args.user}@{args.host}"]
    scp = [*_base(args, "scp")]
    subprocess.run([*ssh, f"mkdir -p {args.remote_dir}"], check=True,
                   capture_output=True, text=True)
    probe_local = os.path.join(args.work_dir, "overhead_probe.py")
    with open(probe_local, "w", encoding="utf-8") as f:
        f.write(BOARD_PROBE)
    subprocess.run([*scp, probe_local, f"{args.user}@{args.host}:{args.remote_dir}/overhead_probe.py"],
                   check=True, capture_output=True, text=True)

    report: dict = {"ladder": [], "variants": []}
    rng = np.random.default_rng(0)

    # --- Ladder: latency vs real NPU work, to extract the fixed intercept.
    for repeats in args.repeats:
        model = build_conv_stack(args.width, repeats, args.size)
        onnx_path = os.path.join(args.work_dir, f"ladder_r{repeats}.onnx")
        rknn_path = os.path.join(args.work_dir, f"ladder_r{repeats}.rknn")
        onnx.save(model, onnx_path)
        calib = [rng.uniform(0, 1, (1, 3, args.size, args.size)).astype(np.float32)
                 for _ in range(4)]
        compile_rknn(onnx_path, rknn_path, calib)
        subprocess.run([*ssh, f"mkdir -p {args.remote_dir}/r{repeats}"], check=True,
                       capture_output=True, text=True)
        subprocess.run([*scp, rknn_path,
                        f"{args.user}@{args.host}:{args.remote_dir}/r{repeats}/model.rknn"],
                       check=True, capture_output=True, text=True)
        entry = {"repeats": repeats, "width": args.width, "size": args.size}
        entry.update(run_probe(args, f"r{repeats}/model.rknn",
                               iterations=args.iterations))
        entry["nodes"] = len(model.graph.node)
        report["ladder"].append(entry)
        print(f"repeats={repeats:2d} nodes={entry['nodes']:3d} "
              f"min={entry['latency_ms']['min']:.4f} "
              f"p50={entry['latency_ms']['p50']:.4f} ms", flush=True)

    # --- Variants on the largest model: isolate specific candidate costs.
    biggest = max(args.repeats)
    model_ref = f"r{biggest}/model.rknn"
    for name, opts in (
        ("baseline", {}),
        ("cpu_pinned", {"affinity": True}),
        ("cpu_pinned+SCHED_FIFO", {"affinity": True, "realtime": True}),
        ("int8_pass_through", {"pass_through": True}),
        ("int8_pass_through+pinned+RT",
         {"pass_through": True, "affinity": True, "realtime": True}),
    ):
        entry = {"variant": name}
        entry.update(run_probe(args, model_ref, iterations=args.iterations, **opts))
        report["variants"].append(entry)
        print(f"{name:30s} min={entry['latency_ms']['min']:.4f} "
              f"p50={entry['latency_ms']['p50']:.4f} "
              f"stdev={entry['latency_ms']['stdev']:.4f} ms", flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.output}", flush=True)
    else:
        print("\n" + json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())