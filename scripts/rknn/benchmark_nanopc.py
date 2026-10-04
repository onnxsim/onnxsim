#!/usr/bin/env python3
"""Upload RKNN models to a NanoPC-T6 (RK3588) board and benchmark them there.

Same host-compiles / board-runs split as ``benchmark_luckfox.py``: the RKNN
compiler only ships for x86-64 Linux, while ``librknnrt.so`` runs on the board.
The uploaded runner is ``nanopc_rknn_runner.py``, which drives the board's
``librknnrt.so`` through ctypes.

Unlike the Luckfox driver this one accepts SSH credentials through
``--ssh-password`` / ``RKNN_SSH_PASSWORD`` as well as the normal
``ssh``/``scp`` configuration, because the FriendlyElec image ships a single
password-authenticated ``pi`` account.

The device node needs no configuration on this board: its RKNPU driver
registers as a DRM device (``/dev/dri/card1`` + ``/dev/dri/renderD129``, no
``/dev/rknpu``), and ``librknnrt.so`` 2.3.0 discovers that itself. The
``render`` group owns the node and the stock ``pi`` user is already in it
(verified on the connected board).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "nanopc_rknn_runner.py")


def _base(args: argparse.Namespace, tool: str) -> list[str]:
    """Build the ``ssh``/``scp`` prefix, wrapped in ``sshpass`` when a password
    is given. The FriendlyElec image has a single password-authenticated
    ``pi`` account with no installed SSH key."""
    options = [
        opt
        for pair in
        [("-o", value) for value in args.ssh_option]
        + [("-o", "StrictHostKeyChecking=no"),
           ("-o", "UserKnownHostsFile=/dev/null"),
           ("-o", "LogLevel=ERROR")]
        for opt in pair
    ]
    if args.ssh_password:
        return ["sshpass", "-p", args.ssh_password, tool, *options]
    return [tool, *options]


def _run(args: argparse.Namespace, remote_command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*_base(args, "ssh"), args.target, remote_command], **kwargs
    )


def _copy(args: argparse.Namespace, local: str, remote: str, extra: list[str] | None = None) -> None:
    subprocess.run(
        [*_base(args, "scp"), *(extra or []), local, f"{args.target}:{remote}"],
        check=True,
    )


def benchmark(args: argparse.Namespace, model: str, feeds: str | None,
              remote_dir: str, iterations: int, warmup: int) -> dict:
    model = os.path.abspath(model)
    if not os.path.isfile(model):
        raise SystemExit(f"model does not exist: {model}")
    stem = os.path.splitext(os.path.basename(model))[0]
    work = f"{remote_dir}/{stem}"
    _run(args, f"mkdir -p {shlex.quote(work)}", check=True,
         capture_output=True, text=True)
    remote_model = f"{work}/model.rknn"
    remote_runner = f"{work}/nanopc_rknn_runner.py"
    _copy(args, model, remote_model)
    _copy(args, RUNNER, remote_runner)
    if feeds:
        _copy(args, os.path.abspath(feeds), f"{work}/feeds.npz")

    out_dir = f"{work}/outputs"
    command = [
        "python3", shlex.quote(remote_runner), shlex.quote(remote_model),
        "--warmup", str(warmup), "--iterations", str(iterations),
    ]
    if feeds:
        command += ["--input-feeds", shlex.quote(f"{work}/feeds.npz")]
    command += ["--output-dir", shlex.quote(out_dir)]

    proc = _run(args, " ".join(command), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"board runner failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"board runner returned no JSON\n{proc.stdout}\n{proc.stderr}")
    result = json.loads(lines[-1])
    result["remote_dir"] = out_dir
    return result


def fetch(args: argparse.Namespace, remote_dir: str, local_dir: str) -> list[str]:
    """Copy the board's dumped output buffers back, one subdirectory per model."""
    local = os.path.abspath(local_dir)
    os.makedirs(local, exist_ok=True)
    subprocess.run(
        [*_base(args, "scp"), "-r",
         f"{args.target}:{remote_dir.rstrip('/')}/*", local],
        check=True,
    )
    return sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(local)
        for name in names
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="host-side compiled .rknn model(s)")
    ap.add_argument("--host", default="nanopc-t6.tailf0b7b1.ts.net")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--remote-dir", default="/tmp/onnxsim-rknn")
    ap.add_argument("--ssh-password", default=os.environ.get("RKNN_SSH_PASSWORD"))
    ap.add_argument("--ssh-option", action="append", default=[])
    ap.add_argument("--feeds", default=None, help="npz of input arrays for every model")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--output", default=None, help="write the JSON summary here")
    ap.add_argument("--fetch-dir", default=None,
                    help="copy the board's output .bin files back to this directory")
    args = ap.parse_args()
    args.target = f"{args.user}@{args.host}"

    summary = []
    for model in args.models:
        print(f"== {os.path.basename(model)}", flush=True)
        result = benchmark(args, model, args.feeds, args.remote_dir,
                           args.iterations, args.warmup)
        latency = result["latency_ms"]
        print(f"   librknnrt={result['librknnrt_version']} "
              f"device={result['device']} n_in={result['n_input']} "
              f"n_out={result['n_output']}")
        print(f"   latency min={latency['min']:.3f} mean={latency['mean']:.3f} "
              f"p50={latency['p50']:.3f} p95={latency['p95']:.3f} "
              f"max={latency['max']:.3f} ms", flush=True)
        if args.fetch_dir:
            fetched = fetch(args, result["remote_dir"], args.fetch_dir)
            result["fetched"] = fetched
            print(f"   fetched {len(fetched)} files to {args.fetch_dir}", flush=True)
        summary.append(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.output}", flush=True)
    else:
        print("\n" + json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())