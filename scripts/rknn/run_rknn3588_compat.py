#!/usr/bin/env python3
"""Run onnxsim's original-vs-simplified check on a real NanoPC-T6 / RK3588 NPU.

``scripts/rknn/run_rknn_compat.py`` (and ``rknn_backend.py``) validate only
that both graphs *compile* and run through ``rknn-toolkit2``'s **PC
simulator**. This harness closes that gap for RK3588: both the original and the
onnxsim-simplified ONNX are compiled with ``target_platform="rk3588"``, pushed
over SSH, executed on the board's actual NPU through ``librknnrt.so``, and
their outputs are compared -- plus each against an ONNX Runtime CPU reference.

Three comparisons are reported per model, which is what makes an INT8
RK3588 result interpretable:

* ``diff_sim_vs_simp`` -- original vs. simplified, **both on the real NPU**.
  This is the actual regression signal: simplification must not change what
  the NPU computes.
* ``diff_ort_vs_orig`` / ``diff_ort_vs_simp`` -- ONNX Runtime CPU (float32)
  vs. each NPU build. Recorded as information: INT8 quantization is *expected*
  to differ, so these are not pass/fail criteria.
* ``diff_orig_vs_simp_int8_only`` -- whether the INT8 activation scales
  (``zp``/``scale``) RKNN's quantizer derived are identical for the two
  graphs. Identical scales mean the NPU executes the same quantization plan on
  the same hardware and any output difference is pure arithmetic ordering;
  differing scales mean RKNN quantized a restructured graph differently, which
  is worth surfacing even when outputs still agree within tolerance.

Because both builds go through the *same* compiler and the *same* NPU with the
same calibration data, a fixed backend numeric difference cancels out and only
an onnxsim-introduced change fails the run.

The ``*_latency_*`` columns are single un-replicated samples and are **not** a
speedup measurement. Measured on a connected NanoPC-T6, two identical runs put
``conv_bn_relu`` at -43.9% then +116.7% and ``sigmoid_mul_swish`` at +5.3% then
-10.7%: these graphs run in 0.03-0.09 ms, which is below the launch-overhead
noise floor, so the sign of the difference is not reproducible. Use
``ab_latency_nanopc.py`` for any latency claim. See
``bench/RESULTS_nanopc_rk3588.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import onnx

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from common.ep_numerics import compare, random_feeds  # noqa: E402
from common.synthetic_models import build, names  # noqa: E402

from onnxsim import simplify  # noqa: E402


def _install_onnx_mapping_shim() -> None:
    """Keep RKNN-Toolkit2 2.3.2 working on ONNX 1.22+, which no longer ships
    ``onnx.mapping`` (see ``rknn_backend._ensure_onnx_mapping_shim``)."""
    import types

    if hasattr(onnx, "mapping"):
        return
    mapping = types.ModuleType("onnx.mapping")
    table = {
        dtype: onnx.helper.tensor_dtype_to_np_dtype(dtype)
        for dtype in onnx.TensorProto.DataType.values()
        if dtype != onnx.TensorProto.UNDEFINED
    }
    mapping.TENSOR_TYPE_TO_NP_TYPE = table
    mapping.NP_TYPE_TO_TENSOR_TYPE = {value: key for key, value in table.items()}
    onnx.mapping = mapping
    sys.modules["onnx.mapping"] = mapping


_install_onnx_mapping_shim()
from rknn.api import RKNN  # noqa: E402


def _check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"rknn {operation} failed: {ret}")


def compile_rknn(
    onnx_path: str,
    rknn_path: str,
    calibration: list[np.ndarray],
    input_name: str,
    target_platform: str = "rk3588",
) -> None:
    """Compile ``onnx_path`` to an INT8 ``.rknn`` for ``target_platform``.

    The calibration set is derived deterministically from the model's input
    shape, so the original and simplified builds are quantized against exactly
    the same samples -- a prerequisite for the scales to be comparable.
    """
    dataset = os.path.join(
        os.path.dirname(rknn_path), os.path.basename(rknn_path) + ".calib"
    )
    os.makedirs(dataset, exist_ok=True)
    paths = []
    for index, array in enumerate(calibration):
        path = os.path.join(dataset, f"{index:03d}.npy")
        np.save(path, array.astype(np.float32))
        paths.append(path)
    dataset_txt = dataset + ".txt"
    with open(dataset_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")

    rknn = RKNN(verbose=False)
    try:
        channels = _input_channels(onnx_path)
        config = {"target_platform": target_platform}
        if channels is not None:
            # mean_values/std_values are only meaningful for a rank-4 image
            # input, and RKNN *requires* their length to equal that input's
            # channel count -- a 1-element vector for a [1,32,16,16] input fails
            # with "The len of mean_values ([0.0]) for input 0 is wrong, expect
            # 32!". For every other rank the parameters are simply omitted, and
            # RKNN applies identity scaling itself.
            config["mean_values"] = [[0.0] * channels]
            config["std_values"] = [[1.0] * channels]
        _check(rknn.config(**config), "config")
        _check(rknn.load_onnx(model=onnx_path), "load_onnx")
        _check(rknn.build(do_quantization=True, dataset=dataset_txt), "build")
        _check(rknn.export_rknn(rknn_path), "export_rknn")
    finally:
        rknn.release()


def _input_channels(onnx_path: str) -> int | None:
    """Channel count of a rank-4 image input, or ``None`` for any other rank
    (where ``mean_values``/``std_values`` must be omitted entirely)."""
    model = onnx.load(onnx_path)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) == 4:
        return int(dims[1].dim_value) or 3
    return None


def ort_reference(
    onnx_path: str, feeds: dict[str, np.ndarray]
) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]
    return session.run(names, feeds)


def run_on_board(
    host: str, user: str, password: str | None, rknn_path: str,
    feeds_path: str, remote_dir: str, warmup: int, iterations: int,
) -> dict:
    """Upload the model + feeds and execute them on the board's real NPU."""
    stem = os.path.splitext(os.path.basename(rknn_path))[0]
    work = f"{remote_dir}/{stem}"
    ssh_options = ["-o", "StrictHostKeyChecking=no",
                   "-o", "UserKnownHostsFile=/dev/null",
                   "-o", "LogLevel=ERROR"]
    prefix = (["sshpass", "-p", password] if password else [])
    target = f"{user}@{host}"

    def ssh(command: str, capture: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*prefix, "ssh", *ssh_options, target, command],
            check=True, capture_output=capture, text=True,
        )

    def scp(local: str, remote: str) -> None:
        subprocess.run(
            [*prefix, "scp", *ssh_options, local, f"{target}:{remote}"], check=True,
        )

    ssh(f"mkdir -p {work}")
    runner = os.path.join(_HERE, "nanopc_rknn_runner.py")
    scp(runner, f"{work}/nanopc_rknn_runner.py")
    scp(rknn_path, f"{work}/model.rknn")
    scp(feeds_path, f"{work}/feeds.npz")

    out_dir = f"{work}/outputs"
    proc = ssh(
        f"python3 {work}/nanopc_rknn_runner.py {work}/model.rknn "
        f"--input-feeds {work}/feeds.npz --output-dir {out_dir} "
        f"--warmup {warmup} --iterations {iterations}",
        capture=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"board runner returned no JSON\n{proc.stdout}\n{proc.stderr}")
    result = json.loads(lines[-1])
    result["remote_output_dir"] = out_dir
    return result


def _fetch_outputs(host: str, user: str, password: str | None, remote_dir: str,
                   local_dir: str) -> dict[int, np.ndarray]:
    """Copy the board's float32 output buffers back, keyed by output index."""
    os.makedirs(local_dir, exist_ok=True)
    prefix = (["sshpass", "-p", password] if password else [])
    target = f"{user}@{host}"
    subprocess.run(
        [*prefix, "scp", "-r", "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
         f"{target}:{remote_dir.rstrip('/')}/*", local_dir],
        check=True,
    )
    outputs = {}
    for name in sorted(os.listdir(local_dir)):
        if not name.endswith(".bin"):
            continue
        index = int(os.path.splitext(name)[0])
        outputs[index] = np.fromfile(os.path.join(local_dir, name), dtype=np.float32)
    return outputs


def _check_model(
    name: str,
    model: onnx.ModelProto,
    work_dir: str,
    host: str,
    user: str,
    password: str | None,
    remote_dir: str,
    rtol: float,
    atol: float,
    warmup: int,
    iterations: int,
) -> dict:
    row = {"model": name, "status": "ok", "error": ""}
    raw_path = os.path.join(work_dir, f"{name}.onnx")
    simp_path = os.path.join(work_dir, f"{name}.simplified.onnx")
    onnx.save(model, raw_path)

    simplified, ok = simplify(model, check_n=0)
    if not ok:
        row.update(status="simplify_error")
        return row
    onnx.save(simplified, simp_path)
    row["orig_nodes"] = len(model.graph.node)
    row["simp_nodes"] = len(simplified.graph.node)

    # Identical feeds for both builds and for the ORT reference.
    feeds = random_feeds(model, seed=0)
    feeds_path = os.path.join(work_dir, f"{name}.feeds.npz")
    np.savez(feeds_path, **feeds)
    calibration = [feeds[inp.name] for inp in model.graph.input
                   if inp.name in feeds]

    ort_out = ort_reference(raw_path, feeds)

    # The board dumps each output flat; restore the model's declared output
    # shape before comparing. Without this every comparison reports a shape
    # mismatch (and thus an infinite diff) even when the values agree -- the raw
    # buffers line up element for element (verified: conv_bn_relu's 8-element
    # float32 output matches the ORT reference to ~1e-3, i.e. ordinary INT8
    # quantization error).
    output_shapes = [
        [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        for out in model.graph.output
    ]

    outputs = {}
    for label, path in (("orig", raw_path), ("simp", simp_path)):
        rknn_path = os.path.join(work_dir, f"{name}.{label}.rknn")
        try:
            compile_rknn(path, rknn_path, calibration, model.graph.input[0].name)
        except Exception as exc:
            # The original failing to compile is a converter limitation, not an
            # onnxsim bug -- report it, do not fail the run.
            row["status"] = "unsupported" if label == "orig" else "rknn_regression"
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return row
        board = run_on_board(host, user, password, rknn_path, feeds_path,
                             remote_dir, warmup, iterations)
        fetched = _fetch_outputs(host, user, password, board["remote_output_dir"],
                                 os.path.join(work_dir, f"{name}.{label}.outputs"))
        ordered = []
        for i in sorted(fetched):
            array = fetched[i]
            shape = output_shapes[i] if i < len(output_shapes) else []
            if shape and int(np.prod(shape)) == array.size:
                array = array.reshape(shape)
            ordered.append(array)
        outputs[label] = ordered
        row[f"{label}_latency_mean_ms"] = round(board["latency_ms"]["mean"], 4)
        row[f"{label}_latency_min_ms"] = round(board["latency_ms"]["min"], 4)
        row["soc"] = board["device"].get("soc", "unknown")

    ok_close, diff_npu = compare(outputs["orig"], outputs["simp"], rtol=rtol, atol=atol)
    row["diff_sim_vs_simp"] = diff_npu
    if not ok_close:
        row["status"] = "rknn_regression"

    _, diff_ort_orig = compare(ort_out, outputs["orig"], rtol=1e-2, atol=1e-3)
    _, diff_ort_simp = compare(ort_out, outputs["simp"], rtol=1e-2, atol=1e-3)
    row["diff_ort_vs_orig"] = diff_ort_orig
    row["diff_ort_vs_simp"] = diff_ort_simp
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of synthetic model names (default: the whole suite)")
    ap.add_argument("--host", default="nanopc-t6.tailf0b7b1.ts.net")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--ssh-password", default=os.environ.get("RKNN_SSH_PASSWORD"))
    ap.add_argument("--remote-dir", default="/tmp/onnxsim-rknn-compat")
    ap.add_argument("--work-dir", default="/tmp/onnxsim-rknn-compat-work")
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--output", default="rknn3588-compat.csv")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    selected = args.models or names()
    print(f"RK3588 real-NPU compatibility check | {len(selected)} models", flush=True)

    rows = []
    failures = []
    for i, name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name} ...", end=" ", flush=True)
        try:
            row = _check_model(name, build(name), args.work_dir, args.host,
                               args.user, args.ssh_password, args.remote_dir,
                               args.rtol, args.atol, args.warmup,
                               args.iterations)
        except Exception as exc:
            row = {"model": name, "status": "error",
                   "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        print(f"{row['status']} d(npu)={row.get('diff_sim_vs_simp', float('nan')):.4g}",
              flush=True)
        if row["status"] in {"rknn_regression", "simplify_error", "error"}:
            failures.append(row)

    fields = ["model", "status", "orig_nodes", "simp_nodes", "diff_sim_vs_simp",
              "diff_ort_vs_orig", "diff_ort_vs_simp", "orig_latency_mean_ms",
              "simp_latency_mean_ms", "orig_latency_min_ms", "simp_latency_min_ms",
              "soc", "error"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwrote {args.output} ({len(rows)} rows)", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILED:", flush=True)
        for row in failures:
            print(f"  - {row['model']}: {row['status']} {row['error']}", flush=True)
        return 1
    print("\nall passed on real RK3588 NPU", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())