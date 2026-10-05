#!/usr/bin/env python3
"""Measure a precompiled TPU-MLIR CVI model on SG2002 over SSH.

Runs the board's model_runner timer and PMU counters. Optional driver usage
sampling reads /proc/tpu/usage_profiling during one sustained run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List


_TIMER_RE = re.compile(r"each run takes ([0-9.]+) ms")
_USAGE_RE = re.compile(r"usage=([0-9.]+)%")
_PMU_RE = re.compile(
    r"cv181x_tpu_clock:\s*([0-9.]+)Mhz,\s*inferece_data:\s*([0-9.]+)MB,\s*"
    r"inference_bw:\s*([0-9.]+)MB/s\s+"
    r"tdma_exe_tick:\s*([0-9]+)t,\s*tiu_exe_tick\s*:?\s*([0-9]+)t,\s*"
    r"inference_tick\s*:?\s*([0-9]+)t\s+"
    r"tdma_exe_percent:\s*([0-9.]+)%,\s*tiu_exe_percent:\s*([0-9.]+)%,\s*"
    r"paralellism_percent\s*([0-9.]+)%\s+(?:[^\n]*\n)*?"
    r"tdma_exe_ms:\s*([0-9.]+)ms,\s*tiu_exe_ms:\s*([0-9.]+)ms,\s*"
    r"inference_ms:\s*([0-9.]+)ms",
    re.MULTILINE,
)
_PMU_NAMES = (
    "clock_mhz",
    "data_mb",
    "bandwidth_mb_s",
    "tdma_ticks",
    "tiu_ticks",
    "inference_ticks",
    "tdma_percent",
    "tiu_percent",
    "parallelism_percent",
    "tdma_ms",
    "tiu_ms",
    "inference_ms",
)


def _exec(client, command: str) -> str:
    _stdin, stdout, stderr = client.exec_command(command, timeout=3600)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"remote command failed ({status}):\n{(output + error)[-8000:]}")
    return output + error


def _runner_command(
    runner: str,
    library_dir: str,
    model_path: str,
    input_path: str,
    output_path: str,
    count: int,
    pmu: bool = False,
) -> str:
    prefix = "TPU_ENABLE_PMU=1 " if pmu else ""
    return (
        f"{prefix}LD_LIBRARY_PATH={shlex.quote(library_dir)} "
        f"{shlex.quote(runner)} --model {shlex.quote(model_path)} "
        f"--input {shlex.quote(input_path)} --output {shlex.quote(output_path)} "
        f"--count {count} --enable-timer"
    )


def _driver_usage_command(runner_command: str, proc_path: str, log_path: str) -> str:
    qproc, qlog = shlex.quote(proc_path), shlex.quote(log_path)
    return f'''set -eu
proc={qproc}
state=$(cat "$proc" 2>/dev/null || true)
if [ "$state" != "profiling is disabled" ]; then
  echo "driver usage profiling is already enabled or unavailable: $state"
  exit 2
fi
echo 1 > "$proc"
trap 'echo 0 > "$proc"' EXIT HUP INT TERM
{runner_command} > {qlog} 2>&1 &
runpid=$!
while kill -0 "$runpid" 2>/dev/null; do
  sleep 1
  cat "$proc"
done
set +e
wait "$runpid"
status=$?
set -e
printf "__MODEL_RUN_OUTPUT__\\n"
cat {qlog}
exit "$status"
'''


def _median_pmu(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {name: float(statistics.median(row[name] for row in rows)) for name in _PMU_NAMES}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="precompiled .cvimodel")
    parser.add_argument("input_npz", type=Path, help="input tensors in model_runner NPZ format")
    parser.add_argument("--host", required=True, help="SG2002 hostname or address")
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--password-env", default="TPU_MLIR_SSH_PASSWORD")
    parser.add_argument("--runner", default="/usr/bin/model_runner")
    parser.add_argument("--library-dir", default="/usr/bin/lib")
    parser.add_argument("--remote-dir", default="/tmp/onnxsim-tpu-profile")
    parser.add_argument("--count", type=int, default=100, help="inferences per runner process")
    parser.add_argument("--repeat", type=int, default=3, help="normal and PMU process repeats")
    parser.add_argument(
        "--batch", type=int, default=1, help="images per inference, for throughput reporting"
    )
    parser.add_argument("--skip-pmu", action="store_true", help="run only normal device timing")
    parser.add_argument(
        "--driver-usage",
        action="store_true",
        help="temporarily enable /proc/tpu/usage_profiling for one sustained run",
    )
    parser.add_argument("--json-out", type=Path, help="write the report to this file")
    args = parser.parse_args()
    if args.count < 1 or args.repeat < 1 or args.batch < 1:
        parser.error("--count, --repeat, and --batch must be positive")
    for path in (args.model, args.input_npz):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    return args


def main() -> int:
    args = _parse_args()
    try:
        import paramiko
    except ImportError:
        print(
            "paramiko is required; install it with `python -m pip install paramiko`",
            file=sys.stderr,
        )
        return 2

    password = os.environ.get(args.password_env)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    remote_run = f"{args.remote_dir.rstrip('/')}/{uuid.uuid4().hex}"
    remote_model = f"{remote_run}/model.cvimodel"
    remote_input = f"{remote_run}/input.npz"
    remote_outputs = [f"{remote_run}/output-{i}.npz" for i in range(args.repeat + 1)]
    report: Dict[str, Any] = {
        "target": args.host,
        "model": str(args.model),
        "input_npz": str(args.input_npz),
        "count": args.count,
        "repeat": args.repeat,
        "batch": args.batch,
    }
    try:
        client.connect(
            args.host,
            port=args.port,
            username=args.user,
            password=password,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
            look_for_keys=True,
            allow_agent=True,
        )
        _exec(client, "mkdir -p " + shlex.quote(remote_run))
        sftp = client.open_sftp()
        try:
            sftp.put(str(args.model), remote_model)
            sftp.put(str(args.input_npz), remote_input)
        finally:
            sftp.close()

        normal_ms = []
        for i in range(args.repeat):
            output = _exec(
                client,
                _runner_command(
                    args.runner,
                    args.library_dir,
                    remote_model,
                    remote_input,
                    remote_outputs[i],
                    args.count,
                ),
            )
            matches = _TIMER_RE.findall(output)
            if not matches:
                raise RuntimeError("model_runner output did not contain its device timing")
            normal_ms.append(float(matches[-1]))
        report["normal"] = {
            "timing_source": "model_runner device timer",
            "each_run_ms": normal_ms,
            "median_ms": float(statistics.median(normal_ms)),
            "images_per_second": 1000.0 * args.batch / statistics.median(normal_ms),
        }

        if not args.skip_pmu:
            all_rows: List[Dict[str, float]] = []
            pmu_run_ms = []
            for i in range(args.repeat):
                output = _exec(
                    client,
                    _runner_command(
                        args.runner,
                        args.library_dir,
                        remote_model,
                        remote_input,
                        remote_outputs[args.repeat],
                        args.count + 1,
                        pmu=True,
                    ),
                )
                pmu_run_ms.extend(float(value) for value in _TIMER_RE.findall(output))
                rows = [
                    dict(zip(_PMU_NAMES, map(float, match)))
                    for match in _PMU_RE.findall(output)
                ]
                if len(rows) < 2:
                    raise RuntimeError("TPU_ENABLE_PMU produced fewer than two counter samples")
                all_rows.extend(rows[1:])
            report["pmu"] = {
                "median": _median_pmu(all_rows),
                "samples": all_rows,
                "warmup_samples_discarded": args.repeat,
                "measured_samples": len(all_rows),
                "runner_each_run_ms": pmu_run_ms,
            }

        if args.driver_usage:
            output = _exec(
                client,
                _driver_usage_command(
                    _runner_command(
                        args.runner,
                        args.library_dir,
                        remote_model,
                        remote_input,
                        remote_outputs[args.repeat],
                        args.count,
                    ),
                    "/proc/tpu/usage_profiling",
                    f"{remote_run}/driver-run.log",
                ),
            )
            samples = [float(value) for value in _USAGE_RE.findall(output)]
            valid = [value for value in samples if 0.0 <= value <= 100.0]
            report["driver_usage"] = {
                "source": "/proc/tpu/usage_profiling",
                "interval_ms": 1000,
                "samples_percent": samples,
                "valid_samples_percent": valid,
                "discarded_out_of_range": len(samples) - len(valid),
                "median_percent": float(statistics.median(valid)) if valid else None,
            }
    except Exception as error:  # noqa: BLE001 - present transport and device errors to the user
        print(f"SG2002 profiling failed: {error}", file=sys.stderr)
        return 1
    finally:
        if client.get_transport() is not None and client.get_transport().is_active():
            try:
                sftp = client.open_sftp()
                cleanup_paths = [
                    remote_model,
                    remote_input,
                    *remote_outputs,
                    f"{remote_run}/driver-run.log",
                ]
                for path in cleanup_paths:
                    try:
                        sftp.remove(path)
                    except OSError:
                        pass
                sftp.close()
                _exec(client, "rmdir " + shlex.quote(remote_run))
            except Exception:
                pass
            client.close()

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
