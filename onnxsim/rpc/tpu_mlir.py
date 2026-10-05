"""TPU-MLIR compilation and SG2002 inference for the onnxsim RPC server.

Compilation runs in the RPC server's TPU-MLIR environment. The resulting CVI model
and input tensors are sent to an SG2002 over SSH and executed with model_runner.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx

from . import _protocol as proto


class TpuMlirRunner:
    """A compiled TPU-MLIR model resident on an SG2002 board."""

    _board_lock = threading.Lock()
    _pmu_pattern = re.compile(
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

    def __init__(self, model_bytes: bytes, options: Dict[str, Any], work_dir: str):
        self.options = dict(options)
        self.ssh_host = str(self.options.get("ssh_host", ""))
        if not self.ssh_host:
            raise proto.RPCError(
                "TPU-MLIR runtime is disabled; start the server with --tpu-ssh-host"
            )
        self.ssh_user = str(self.options.get("ssh_user", "root"))
        self.ssh_port = int(self.options.get("ssh_port", 22))
        self.password_env = str(
            self.options.get("password_env", "TPU_MLIR_SSH_PASSWORD")
        )
        self.remote_dir = str(self.options.get("remote_dir", "/tmp/onnxsim-tpu-rpc"))
        self.runner = str(self.options.get("runner", "/usr/bin/model_runner"))
        self.library_dir = str(self.options.get("library_dir", "/usr/bin/lib"))
        self.chip = str(self.options.get("chip", "cv181x"))
        self.quantize = str(self.options.get("quantize", "BF16"))
        self.calibration_table = self.options.get("calibration_table")
        self.python = str(self.options.get("python", sys.executable))
        self.transform = shutil.which(
            str(self.options.get("model_transform", "model_transform.py"))
        ) or str(self.options.get("model_transform", "model_transform.py"))
        self.deploy = shutil.which(
            str(self.options.get("model_deploy", "model_deploy.py"))
        ) or str(self.options.get("model_deploy", "model_deploy.py"))
        self.model = onnx.ModelProto()
        self.model.ParseFromString(model_bytes)
        self.input_names, input_shapes = self._static_float_inputs(self.model)
        self.input_shapes = dict(zip(self.input_names, input_shapes))
        self.batch_size = input_shapes[0][0] if input_shapes else 1
        self.output_names = [value.name for value in self.model.graph.output]
        if not self.input_names or not self.output_names:
            raise proto.RPCError(
                "TPU-MLIR requires at least one model input and output"
            )

        self.model_id = uuid.uuid4().hex
        self.remote_model = f"{self.remote_dir}/{self.model_id}.cvimodel"
        root = Path(work_dir) / "tpu-mlir" / self.model_id
        root.mkdir(parents=True, exist_ok=False)
        onnx_path = root / "model.onnx"
        mlir_path = root / "model.mlir"
        cvimodel_path = root / "model.cvimodel"
        onnx_path.write_bytes(model_bytes)
        self._run_local(
            [
                self.python,
                self.transform,
                "--model_name",
                self.model_id,
                "--model_def",
                str(onnx_path),
                "--input_shapes",
                json.dumps(input_shapes, separators=(",", ":")),
                "--mlir",
                str(mlir_path),
            ]
        )
        deploy_command = [
            self.python,
            self.deploy,
            "--mlir",
            str(mlir_path),
            "--chip",
            self.chip,
            "--quantize",
            self.quantize,
            "--model",
            str(cvimodel_path),
        ]
        if self.quantize.upper() == "INT8":
            if not self.calibration_table:
                raise proto.RPCError(
                    "INT8 TPU-MLIR compilation requires a server-side calibration table; "
                    "provide runtime options['calibration_table']"
                )
            calibration_path = Path(str(self.calibration_table)).expanduser()
            if not calibration_path.is_file():
                raise proto.RPCError(
                    f"TPU-MLIR calibration table does not exist: {calibration_path}"
                )
            deploy_command.extend(["--calibration_table", str(calibration_path)])
        if "opt" in self.options:
            try:
                deploy_opt = int(self.options["opt"])
            except (TypeError, ValueError) as error:
                raise proto.RPCError(
                    "TPU-MLIR deploy option 'opt' must be 1, 2, or 3"
                ) from error
            if deploy_opt not in (1, 2, 3):
                raise proto.RPCError("TPU-MLIR deploy option 'opt' must be 1, 2, or 3")
            deploy_command.extend(["--opt", str(deploy_opt)])
        for name, flag in (
            ("do_winograd", "--do_winograd"),
            ("matmul_perchannel", "--matmul_perchannel"),
        ):
            if self.options.get(name, False):
                deploy_command.append(flag)
        self._run_local(deploy_command)
        if not cvimodel_path.is_file():
            raise proto.RPCError("model_deploy completed without producing a CVI model")
        self._connect()
        self._ssh("mkdir -p " + shlex.quote(self.remote_dir))
        self._scp(str(cvimodel_path), self.remote_model)

    @staticmethod
    def _static_float_inputs(
        model: onnx.ModelProto,
    ) -> Tuple[List[str], List[List[int]]]:
        initializers = {initializer.name for initializer in model.graph.initializer}
        names: List[str] = []
        shapes: List[List[int]] = []
        for value in model.graph.input:
            if value.name in initializers:
                continue
            tensor = value.type.tensor_type
            if tensor.elem_type != onnx.TensorProto.FLOAT:
                raise proto.RPCError(
                    f"TPU-MLIR RPC currently supports float32 model inputs; "
                    f"{value.name!r} has ONNX element type {tensor.elem_type}"
                )
            dims = []
            for dim in tensor.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    raise proto.RPCError(
                        f"TPU-MLIR RPC needs static positive input dimensions; "
                        f"{value.name!r} has a dynamic dimension"
                    )
                dims.append(int(dim.dim_value))
            names.append(value.name)
            shapes.append(dims)
        return names, shapes

    def _run_local(self, command: List[str]) -> str:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(self.options.get("compile_timeout", 1800)),
        )
        if result.returncode:
            raise proto.RPCError(
                f"TPU-MLIR command failed ({result.returncode}):\n{result.stdout[-12000:]}"
            )
        return result.stdout

    def _connect(self) -> None:
        try:
            import paramiko  # type: ignore[import-untyped]
        except ImportError as error:
            raise proto.RPCError(
                "paramiko is required for the TPU-MLIR board connection"
            ) from error
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.load_system_host_keys()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh_client.connect(
            self.ssh_host,
            port=self.ssh_port,
            username=self.ssh_user,
            password=os.environ.get(self.password_env),
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
            look_for_keys=True,
            allow_agent=True,
        )
        self._sftp = self._ssh_client.open_sftp()

    def _ssh(self, command: str) -> str:
        _stdin, stdout, stderr = self._ssh_client.exec_command(
            command, timeout=int(self.options.get("run_timeout", 300))
        )
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        if status:
            raise proto.RPCError(
                f"SG2002 command failed ({status}):\n{(output + error)[-8000:]}"
            )
        return output

    def _scp(self, local: str, remote: str) -> None:
        try:
            self._sftp.put(local, remote)
        except Exception as error:  # noqa: BLE001 - add transfer context
            raise proto.RPCError(
                f"failed to upload a file to the SG2002: {error}"
            ) from error

    def _download(self, remote: str, local: str) -> None:
        try:
            self._sftp.get(remote, local)
        except Exception as error:  # noqa: BLE001 - add transfer context
            raise proto.RPCError(
                f"failed to download a file from the SG2002: {error}"
            ) from error

    def _npz_command(
        self,
        arrays: Dict[str, np.ndarray],
        count: int = 1,
        enable_timer: bool = True,
        enable_pmu: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Optional[float], List[Dict[str, float]]]:
        missing = [name for name in self.input_names if name not in arrays]
        extra = [name for name in arrays if name not in self.input_names]
        if missing or extra:
            raise proto.RPCError(
                f"input names mismatch (missing={missing}, extra={extra})"
            )
        for name in self.input_names:
            array = np.asarray(arrays[name])
            if array.dtype != np.float32:
                raise proto.RPCError(
                    f"input {name!r} must be float32, got {array.dtype}"
                )
            if tuple(array.shape) != tuple(self.input_shapes[name]):
                raise proto.RPCError(
                    f"input {name!r} has shape {array.shape}; expected "
                    f"{tuple(self.input_shapes[name])}"
                )
            arrays[name] = np.ascontiguousarray(array)

        with (
            self._board_lock,
            tempfile.TemporaryDirectory(prefix="onnxsim-tpu-run-") as temp,
        ):
            input_path = Path(temp) / "input.npz"
            output_path = Path(temp) / "output.npz"
            np.savez(input_path, **arrays)
            nonce = uuid.uuid4().hex
            remote_input = f"{self.remote_dir}/{nonce}-input.npz"
            remote_output = f"{self.remote_dir}/{nonce}-output.npz"
            self._scp(str(input_path), remote_input)
            try:
                shell = (
                    ("TPU_ENABLE_PMU=1 " if enable_pmu else "")
                    + f"LD_LIBRARY_PATH={shlex.quote(self.library_dir)} "
                    f"{shlex.quote(self.runner)} --model "
                    f"{shlex.quote(self.remote_model)} "
                    f"--input {shlex.quote(remote_input)} "
                    f"--output {shlex.quote(remote_output)} "
                    f"--count {int(count)}"
                    + (" --enable-timer" if enable_timer else "")
                )
                output = self._ssh(shell)
                if enable_pmu:
                    self._last_pmu_output = output
                timing = re.search(r"each run takes ([0-9.]+) ms", output)
                pmu_rows = self._parse_pmu(output) if enable_pmu else []
                self._download(remote_output, str(output_path))
            finally:
                try:
                    self._ssh(
                        f"rm -f {shlex.quote(remote_input)} "
                        f"{shlex.quote(remote_output)}"
                    )
                except Exception:
                    pass
            with np.load(output_path, allow_pickle=False) as data:
                output_arrays = [np.array(data[key], copy=True) for key in data.files]
            if len(output_arrays) != len(self.output_names):
                raise proto.RPCError(
                    f"SG2002 returned {len(output_arrays)} outputs; model expects "
                    f"{len(self.output_names)}"
                )
            outputs = dict(zip(self.output_names, output_arrays))
            elapsed_s = float(timing.group(1)) / 1000.0 if timing else None
            return outputs, elapsed_s, pmu_rows

    @classmethod
    def _parse_pmu(cls, output: str) -> List[Dict[str, float]]:
        names = (
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
        return [
            dict(zip(names, map(float, match)))
            for match in cls._pmu_pattern.findall(output)
        ]

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        outputs, _, _ = self._npz_command(inputs)
        return outputs

    def time(self, inputs: Dict[str, np.ndarray], number: int, repeat: int):
        results = []
        for _ in range(repeat):
            _, device_s, _ = self._npz_command(inputs, number)
            if device_s is None:
                raise proto.RPCError("model_runner did not report a device timing")
            results.append(device_s)
        return results, {
            "timing_source": "model_runner device timer",
            "batch": self.batch_size,
        }

    def pmu_time(self, inputs: Dict[str, np.ndarray], number: int, repeat: int):
        if self.chip.lower() != "cv181x":
            raise proto.RPCError("SG2002 PMU counters require chip cv181x")
        all_rows: List[Dict[str, float]] = []
        results = []
        for _ in range(repeat):
            # Run one extra inference and discard its counters as a per-process warmup.
            _, _, rows = self._npz_command(
                inputs, number + 1, enable_timer=False, enable_pmu=True
            )
            if len(rows) < 2:
                raise proto.RPCError(
                    "SG2002 runtime did not emit enough PMU counters; verify "
                    "TPU_ENABLE_PMU support. Runtime output:\n"
                    + self._last_pmu_output[-3000:]
                )
            rows = rows[1:]
            all_rows.extend(rows)
            results.append(
                float(np.median([row["inference_ms"] for row in rows])) / 1000.0
            )
        keys = all_rows[0].keys()
        stats: dict[str, Any] = {
            key: float(np.median([row[key] for row in all_rows])) for key in keys
        }
        stats.update(
            timing_source="SG2002 TPU PMU",
            warmup_runs=repeat,
            measured_runs=len(all_rows),
            batch=self.batch_size,
            samples=all_rows,
        )
        return results, stats

    def close(self) -> None:
        try:
            self._ssh(f"rm -f {shlex.quote(self.remote_model)}")
        except Exception:
            pass
        try:
            self._sftp.close()
            self._ssh_client.close()
        except Exception:
            pass
