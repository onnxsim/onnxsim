"""onnxsim RPC server: run ONNX models on the machine it is started on.

The TVM-style workflow -- start a small server on the target device, forward its port (for
example ``adb forward tcp:9090 tcp:9090``), then upload, run and time models from the host --
without TVM's compiler stack. Models execute with onnxruntime when it is installed (one cached
session per loaded model, so timing excludes session creation) and fall back to onnxsim's
pure-Python reference evaluator otherwise.

Security: a server executes whatever ONNX model a client sends it. It binds to loopback by
default, the shared ``key`` is only an identifier (like TVM's device key), not authentication, and
onnxruntime custom-op libraries are never loaded. Expose it only on a trusted link.
"""

from __future__ import annotations

import math
import os
import platform
import re
import socket
import socketserver
import statistics
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx

from . import _protocol as proto


def _sanitize(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    if not base:
        raise proto.RPCError(f"invalid file name {name!r}")
    return base


def _random_tensors(specs, seed: Optional[int]) -> Dict[str, np.ndarray]:
    """Generate bounded random arrays from tensor metadata without receiving tensor blobs."""
    rng = np.random.default_rng(seed)
    tensors = {}
    seen = set()
    for spec in specs:
        name, dtype = spec["name"], spec["dtype"]
        if name in seen:
            raise proto.RPCError(f"duplicate tensor name {name!r}")
        seen.add(name)
        if dtype not in proto.DTYPES:
            raise proto.RPCError(f"unsupported tensor dtype {dtype!r}")
        try:
            shape = tuple(int(dim) for dim in spec["shape"])
        except (KeyError, TypeError, ValueError):
            raise proto.RPCError(f"invalid shape for tensor {name!r}") from None
        if any(dim < 0 for dim in shape):
            raise proto.RPCError(f"invalid shape for tensor {name!r}")
        np_dtype = np.dtype(dtype)
        byte_count = math.prod(shape) * np_dtype.itemsize
        if byte_count > proto.DEFAULT_MAX_BLOB_BYTES:
            raise proto.RPCError(
                f"random tensor {name!r} exceeds the tensor size limit"
            )
        has_low, has_high = "low" in spec, "high" in spec
        if has_low != has_high:
            raise proto.RPCError(f"random tensor {name!r} needs both low and high")
        if has_low:
            low, high = spec["low"], spec["high"]
            if dtype.startswith("float"):
                if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                    raise proto.RPCError(f"invalid random range for tensor {name!r}")
                value = rng.uniform(low, high, size=shape).astype(np_dtype)
            else:
                if int(low) != low or int(high) != high or low >= high:
                    raise proto.RPCError(
                        f"invalid integer random range for tensor {name!r}"
                    )
                minimum, maximum = (
                    (0, 2)
                    if dtype == "bool"
                    else (
                        np.iinfo(np_dtype).min,
                        np.iinfo(np_dtype).max + 1,
                    )
                )
                if low < minimum or high > maximum:
                    raise proto.RPCError(f"random range is outside dtype {dtype!r}")
                value = rng.integers(
                    low,
                    high - 1,
                    size=shape,
                    dtype=np.uint8 if dtype == "bool" else np_dtype,
                    endpoint=True,
                ).astype(np_dtype)
        elif dtype == "bool":
            value = rng.integers(0, 2, size=shape, dtype=np.uint8).astype(np.bool_)
        elif dtype.startswith("float"):
            value = rng.standard_normal(size=shape).astype(np_dtype)
        else:
            limits = np.iinfo(np_dtype)
            value = rng.integers(
                limits.min,
                limits.max,
                size=shape,
                dtype=np_dtype,
                endpoint=True,
            )
        tensors[name] = value
    if not tensors:
        raise proto.RPCError("random inputs require at least one tensor")
    return tensors


class _Runner:
    """One loaded model: an onnxruntime session when available, else the reference evaluator."""

    def __init__(
        self,
        model_bytes: bytes,
        providers: Optional[Sequence[str]],
        single_threaded: bool,
    ):
        self.model_bytes = model_bytes
        self.providers = list(providers) if providers else None
        self.single_threaded = single_threaded
        self.session = None
        try:
            import onnxruntime as ort
        except ImportError:
            return
        available = ort.get_available_providers()
        chosen = self.providers or ["CPUExecutionProvider"]
        unknown = [p for p in chosen if p not in available]
        if unknown:
            raise proto.RPCError(
                f"execution providers {unknown} are not available here: {available}"
            )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if single_threaded:
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_bytes, options, providers=chosen)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.session is not None:
            names = [o.name for o in self.session.get_outputs()]
            return dict(zip(names, self.session.run(None, inputs)))
        from onnxsim import backend

        return dict(backend.run_model(model_bytes_to_proto(self.model_bytes), inputs))


class _TinygradRunner:
    """Runs a model with tinygrad's ONNX frontend on a chosen tinygrad device.

    Exists so tinygrad's code generation can be benchmarked on the server's hardware through the
    same session API. ``options`` are tinygrad codegen knobs (an allow-list, e.g. ``{"BEAM": 2}``)
    applied around compilation and execution. The frontend caches Python constants between calls,
    so this suits static-shape models; a graph with data-dependent shapes needs a fresh load per
    input.
    """

    ALLOWED_OPTIONS = ("BEAM", "NOOPT")
    DEVICE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*(:[0-9]+)?$")

    def __init__(
        self,
        model_bytes: bytes,
        device: Optional[str],
        options: Optional[Dict[str, Any]],
        work_dir: str,
    ):
        try:
            from tinygrad import Device, Tensor
            from tinygrad.nn.onnx import OnnxRunner
        except ImportError as error:
            raise proto.RPCError(
                "the tinygrad runtime needs tinygrad installed on the server"
            ) from error
        if device is not None and not self.DEVICE_PATTERN.match(device):
            raise proto.RPCError(f"invalid tinygrad device {device!r}")
        options = dict(options or {})
        unknown = sorted(set(options) - set(self.ALLOWED_OPTIONS))
        if unknown:
            raise proto.RPCError(
                f"unsupported tinygrad options {unknown}; allowed: {list(self.ALLOWED_OPTIONS)}"
            )
        self.options = {k: int(v) for k, v in options.items()}
        self.device = device or Device.DEFAULT
        self._Tensor, self._Device = Tensor, Device
        model = model_bytes_to_proto(model_bytes)
        self.output_names = [o.name for o in model.graph.output]
        handle, path = tempfile.mkstemp(suffix=".onnx", dir=work_dir)
        try:
            with os.fdopen(handle, "wb") as f:
                f.write(model_bytes)
            # Weights are created on tinygrad's default device: scope it to the requested one.
            with self._scope():
                self.runner = OnnxRunner(path)
        finally:
            os.unlink(path)

    def _scope(self):
        from tinygrad.helpers import Context

        return Context(DEV=self.device, **self.options)

    def _forward(self, tensors):
        with self._scope():
            outputs = self.runner(tensors)
            outs = [outputs[name] for name in self.output_names]
            self._Tensor.realize(*outs)
        return outs

    def _to_device(self, inputs: Dict[str, np.ndarray]):
        return {
            k: self._Tensor(v, device=self.device).realize() for k, v in inputs.items()
        }

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        outs = self._forward(self._to_device(inputs))
        return {name: out.numpy() for name, out in zip(self.output_names, outs)}

    def time(self, inputs: Dict[str, np.ndarray], number: int, repeat: int):
        """(per-call seconds, statistics), measuring steady-state ``TinyJit`` replays.

        Replaying a captured graph removes the Python ONNX interpreter's per-call cost, so the
        numbers reflect the generated kernels. ``stats`` also carries the eager (interpreted) call
        time and, from one replay under ``DEBUG=2``, device kernel time, FLOPs and bytes moved.
        """
        import contextlib
        import io

        from tinygrad import TinyJit
        from tinygrad.helpers import Context, GlobalCounters

        tensors = self._to_device(inputs)
        names = list(tensors)
        eager_start = time.perf_counter()
        self._forward(tensors)  # first call also compiles the kernels
        self._Device[self.device].synchronize()
        eager_first = time.perf_counter() - eager_start
        eager_start = time.perf_counter()
        self._forward(tensors)
        self._Device[self.device].synchronize()
        eager_call = time.perf_counter() - eager_start

        def step(*args):
            return tuple(o.realize() for o in self._forward(dict(zip(names, args))))

        jit = TinyJit(step)
        # Call 1 warms up, call 2 captures, call 3 is the first replay. The capture compiles the
        # graph when the call returns, i.e. outside ``_forward``, so codegen options (BEAM) must
        # be in scope around the JIT calls themselves.
        with self._scope():
            for _ in range(3):
                jit(*tensors.values())
        self._Device[self.device].synchronize()
        results = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(number):
                jit(*tensors.values())
            self._Device[self.device].synchronize()
            results.append((time.perf_counter() - start) / number)
        GlobalCounters.reset()
        with Context(DEBUG=2), contextlib.redirect_stdout(io.StringIO()):
            jit(*tensors.values())
            self._Device[self.device].synchronize()
        stats = {
            "kernels": int(GlobalCounters.kernel_count),
            "gflops": GlobalCounters.global_ops / 1e9,
            "gbytes": GlobalCounters.global_mem / 1e9,
            "kernel_time_s": float(GlobalCounters.time_sum_s),
            "eager_call_s": eager_call,
            "first_call_s": eager_first,
            "device": self.device,
            "options": self.options,
        }
        return results, stats


def _tinygrad_worker(connection, model_bytes, device, options, work_dir) -> None:
    """Entry point of a tinygrad worker process: build the runner, then serve run/time requests."""
    try:
        runner = _TinygradRunner(model_bytes, device, options, work_dir)
        connection.send(("ok", None))
    except BaseException as error:  # noqa: BLE001 - relayed to the parent
        connection.send(("error", f"{type(error).__name__}: {error}"))
        return
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        if request[0] == "close":
            return
        try:
            result = getattr(runner, request[0])(*request[1:])
            connection.send(("ok", result))
        except BaseException as error:  # noqa: BLE001
            connection.send(("error", f"{type(error).__name__}: {error}"))


class _TinygradProxy:
    """A tinygrad model living in its own worker process.

    One process per loaded model means every (device, codegen options) pair starts from clean
    kernel and schedule caches -- so a ``BEAM`` setting really applies instead of silently reusing
    kernels compiled earlier under another setting -- tinygrad's thread-bound state (its SQLite
    disk cache, device contexts) stays on the worker's main thread, and a GPU fault cannot take
    the server down.
    """

    def __init__(
        self, model_bytes: bytes, device: Optional[str], options, work_dir: str
    ):
        import importlib.util
        import multiprocessing

        if importlib.util.find_spec("tinygrad") is None:
            raise proto.RPCError(
                "the tinygrad runtime needs tinygrad installed on the server"
            )
        context = multiprocessing.get_context("spawn")
        self._parent, child = context.Pipe()
        self._process = context.Process(
            target=_tinygrad_worker,
            args=(child, model_bytes, device, options, work_dir),
            daemon=True,
        )
        self._process.start()
        child.close()
        self._lock = threading.Lock()
        self._receive()  # raises if the runner could not be created

    def _receive(self):
        try:
            status, value = self._parent.recv()
        except EOFError:
            raise proto.RPCError("the tinygrad worker process died") from None
        if status != "ok":
            self.close()
            raise proto.RPCError(value)
        return value

    def _call(self, *request):
        with self._lock:
            self._parent.send(request)
            return self._receive()

    def run(self, inputs):
        return self._call("run", inputs)

    def time(self, inputs, number, repeat):
        return self._call("time", inputs, number, repeat)

    def close(self) -> None:
        try:
            self._parent.send(("close",))
        except (OSError, ValueError):
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()


def _close(runner) -> None:
    close = getattr(runner, "close", None)
    if close is not None:
        close()


def _make_runner(
    header: Dict[str, Any], model_bytes: bytes, work_dir: str, server=None
):
    runtime = header.get("runtime") or "onnxruntime"
    if runtime == "onnxruntime":
        return _Runner(
            model_bytes, header.get("providers"), bool(header.get("single_threaded"))
        )
    if runtime == "tinygrad":
        return _TinygradProxy(
            model_bytes, header.get("device"), header.get("options"), work_dir
        )
    if runtime == "tpu_mlir":
        from .tpu_mlir import TpuMlirRunner

        if server is None:
            raise proto.RPCError("TPU-MLIR runner requires an RPC server configuration")
        options = dict(server.tpu_mlir_options)
        model_options = header.get("options") or {}
        for name in (
            "quantize",
            "calibration_table",
            "opt",
            "do_winograd",
            "matmul_perchannel",
        ):
            if name in model_options:
                options[name] = model_options[name]
        return TpuMlirRunner(model_bytes, options, work_dir)
    raise proto.RPCError(
        f"unknown runtime {runtime!r} (expected 'onnxruntime', 'tinygrad' or 'tpu_mlir')"
    )


def model_bytes_to_proto(data: bytes) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.ParseFromString(data)
    return model


class _Handler(socketserver.BaseRequestHandler):
    server: "RPCServer"

    def handle(self) -> None:
        sock = self.request
        models: Dict[int, Any] = {}
        counter = 0
        try:
            first, _ = proto.recv_message(sock, self.server.max_blob_bytes)
            client_key = first.get("key", "")
            if first.get("op") != "hello" or (
                self.server.key and client_key != self.server.key
            ):
                proto.send_message(
                    sock,
                    {
                        "ok": False,
                        "error": f"handshake rejected (server key {self.server.key!r})",
                    },
                )
                return
            proto.send_message(sock, {"ok": True, "info": self.server.info()})
            while True:
                try:
                    header, blobs = proto.recv_message(sock, self.server.max_blob_bytes)
                except ConnectionError:
                    return
                op = header.get("op")
                if op == "close":
                    proto.send_message(sock, {"ok": True})
                    return
                try:
                    reply, out_blobs = self._dispatch(
                        op, header, blobs, models, counter
                    )
                    if op == "load_model":
                        counter = reply["handle"]
                    reply["ok"] = True
                except Exception as error:  # noqa: BLE001 - reported to the client
                    if self.server.verbose:
                        traceback.print_exc()
                    reply, out_blobs = (
                        {"ok": False, "error": f"{type(error).__name__}: {error}"},
                        [],
                    )
                proto.send_message(sock, reply, out_blobs)
        except (ConnectionError, proto.RPCError, OSError):
            return
        finally:
            for runner in models.values():
                _close(runner)
            models.clear()

    # ---- operations -----------------------------------------------------------------------

    def _dispatch(
        self, op, header, blobs, models, counter
    ) -> Tuple[Dict[str, Any], List[bytes]]:
        server = self.server
        server.stats[op] = server.stats.get(op, 0) + 1
        if op == "info":
            return {"info": server.info()}, []
        if op == "upload":
            if len(blobs) != 1:
                raise proto.RPCError("upload takes exactly one blob")
            name = _sanitize(header["name"])
            path = os.path.join(server.work_dir, name)
            with open(path, "wb") as f:
                f.write(blobs[0])
            return {"name": name, "size": len(blobs[0])}, []
        if op == "load_model":
            if blobs:
                data = blobs[0]
            else:
                with open(
                    os.path.join(server.work_dir, _sanitize(header["name"])), "rb"
                ) as f:
                    data = f.read()
            runner = _make_runner(header, data, server.work_dir, server)
            handle = counter + 1
            models[handle] = runner
            return {"handle": handle}, []
        if op == "unload":
            _close(models.pop(int(header["handle"]), None))
            return {}, []
        if op == "run":
            runner = self._model(models, header)
            outputs = runner.run(proto.decode_tensors(header["tensors"], blobs))
            specs, out_blobs = proto.encode_tensors(outputs)
            return {"tensors": specs}, out_blobs
        if op == "time":
            runner = self._model(models, header)
            if header.get("random_inputs"):
                if blobs:
                    raise proto.RPCError("random inputs must not include tensor blobs")
                inputs = _random_tensors(header["tensors"], header.get("seed"))
            else:
                inputs = proto.decode_tensors(header["tensors"], blobs)
            number, repeat = (
                max(int(header.get("number", 1)), 1),
                max(int(header.get("repeat", 1)), 1),
            )
            if header.get("pmu"):
                pmu_time = getattr(runner, "pmu_time", None)
                if pmu_time is None:
                    raise proto.RPCError(
                        "PMU timing is only supported by the TPU-MLIR SG2002 runtime"
                    )
                results, stats = pmu_time(inputs, number, repeat)
                return {
                    "results": results,
                    "median": statistics.median(results),
                    "stats": stats,
                }, []
            if hasattr(runner, "time"):
                results, stats = runner.time(inputs, number, repeat)
                return {
                    "results": results,
                    "median": statistics.median(results),
                    "stats": stats,
                }, []
            runner.run(inputs)  # warm-up, as TVM's time_evaluator does before measuring
            results = []
            for _ in range(repeat):
                start = time.perf_counter()
                for _ in range(number):
                    runner.run(inputs)
                results.append((time.perf_counter() - start) / number)
            return {"results": results, "median": statistics.median(results)}, []
        if op == "run_once":
            # One-shot execution without a handle: the model rides in blob 0, inputs after it.
            runner = _make_runner(header, blobs[0], server.work_dir, server)
            try:
                outputs = runner.run(proto.decode_tensors(header["tensors"], blobs[1:]))
            finally:
                _close(runner)
            specs, out_blobs = proto.encode_tensors(outputs)
            return {"tensors": specs}, out_blobs
        if op == "xdna_compile_resnet":
            from .xdna import compile_resnet

            header = dict(header, _xdna_python=server.xdna_python)
            with server.xdna_lock:
                result, out_blobs = compile_resnet(header, blobs, server.work_dir)
            return {"result": result}, out_blobs
        if op == "xdna_run_resnet":
            from .xdna import run_resnet

            header = dict(header, _xdna_python=server.xdna_python)
            with server.xdna_lock:
                result, out_blobs = run_resnet(header, blobs, server.work_dir)
            return {"result": result}, out_blobs
        if op == "xdna_compare_vitis_resnet":
            from .xdna import compare_resnet

            header = dict(
                header,
                _xdna_python=server.xdna_python,
                _vitis_python=server.vitis_python,
            )
            with server.xdna_lock:
                result, out_blobs = compare_resnet(header, blobs, server.work_dir)
            return {"result": result}, out_blobs
        raise proto.RPCError(f"unknown operation {op!r}")

    @staticmethod
    def _model(models: Dict[int, Any], header: Dict[str, Any]) -> Any:
        try:
            return models[int(header["handle"])]
        except KeyError:
            raise proto.RPCError(
                f"no such model handle {header.get('handle')!r}"
            ) from None


class RPCServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        key: str = "",
        work_dir: Optional[str] = None,
        max_blob_bytes: int = proto.DEFAULT_MAX_BLOB_BYTES,
        verbose: bool = False,
        xdna_python: Optional[str] = None,
        vitis_python: Optional[str] = None,
        tpu_mlir_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__((host, port), _Handler)
        self.key = key
        self.work_dir = work_dir or os.path.join(os.getcwd(), "onnxsim_rpc_workspace")
        os.makedirs(self.work_dir, exist_ok=True)
        self.max_blob_bytes = max_blob_bytes
        self.verbose = verbose
        self.xdna_python = xdna_python or sys.executable
        self.vitis_python = vitis_python or self.xdna_python
        self.tpu_mlir_options = dict(tpu_mlir_options or {})
        self.xdna_lock = threading.Lock()
        self.stats: Dict[str, int] = {}
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self.server_address[:2]
        return str(host), int(port)

    def info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "protocol": 1,
            "key": self.key,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "xdna_python": self.xdna_python,
            "vitis_python": self.vitis_python,
            "tpu_mlir": {
                "enabled": bool(self.tpu_mlir_options.get("ssh_host")),
                "chip": self.tpu_mlir_options.get("chip", "cv181x"),
                "quantize": self.tpu_mlir_options.get("quantize", "BF16"),
            },
            "onnx": onnx.__version__,
        }
        try:
            import onnxruntime as ort

            info["onnxruntime"] = ort.__version__
            info["providers"] = ort.get_available_providers()
        except ImportError:
            info["onnxruntime"] = None
            info["providers"] = []
        try:
            import importlib.metadata

            from tinygrad import Device

            info["tinygrad"] = {
                "version": importlib.metadata.version("tinygrad"),
                "default_device": Device.DEFAULT,
            }
        except Exception:  # noqa: BLE001 - tinygrad is optional
            info["tinygrad"] = None
        try:
            import onnxsim

            info["onnxsim"] = onnxsim.__version__
        except Exception:  # noqa: BLE001
            info["onnxsim"] = None
        return info

    def start(self) -> "RPCServer":
        """Serve in a background thread (for tests and embedding)."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()

    def register_with_tracker(
        self,
        tracker: Tuple[str, int],
        advertise_host: Optional[str] = None,
        retry: float = 2.0,
    ) -> threading.Thread:
        """Keep this server registered with a tracker (re-registers if the tracker restarts)."""
        host = advertise_host or _default_advertise_host(self.address[0])

        def loop():
            while True:
                try:
                    sock = socket.create_connection(tracker, timeout=10)
                    sock.settimeout(None)
                    proto.send_message(
                        sock,
                        {
                            "op": "register",
                            "key": self.key,
                            "addr": [host, self.address[1]],
                        },
                    )
                    proto.recv_message(sock)
                    while sock.recv(
                        1
                    ):  # the tracker never writes again; blocks until it closes
                        pass
                except OSError:
                    pass
                time.sleep(retry)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return thread


def _default_advertise_host(bound: str) -> str:
    if bound not in ("0.0.0.0", "::", ""):
        return bound
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
