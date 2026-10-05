"""onnxsim RPC client: a TVM-shaped session API for running ONNX models on a remote device.

::

    import onnxsim.rpc as rpc

    remote = rpc.connect("127.0.0.1", 9090, key="pixel")     # or rpc.connect_tracker(...).request(key)
    remote.upload("model.onnx")                               # like tvm.rpc session.upload
    model = remote.load_model("model.onnx")                   # like session.load_module
    outputs = model.run({"x": x})
    timing = model.time_evaluator({"x": x}, number=5, repeat=3)   # like module.time_evaluator
    print(timing.median * 1e3, "ms")

    with rpc.remote_executor(remote):                         # fold constants on the device
        simplified, ok = onnxsim.simplify(model_proto)
"""

from __future__ import annotations

import json
import os
import socket
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from . import _protocol as proto
from ._protocol import RPCError

ModelLike = Union[str, os.PathLike, bytes, onnx.ModelProto]


@dataclass
class ProfileResult:
    """Per-call times in seconds (each entry is the mean of ``number`` runs), like TVM's."""

    results: List[float]
    # Extra runtime statistics when the server provides them (the tinygrad runtime reports kernel
    # count, GFLOPs, GB moved and device kernel time for one steady-state call).
    stats: Optional[Dict[str, Any]] = None

    @property
    def mean(self) -> float:
        return statistics.fmean(self.results)

    @property
    def median(self) -> float:
        return statistics.median(self.results)

    @property
    def min(self) -> float:
        return min(self.results)

    @property
    def max(self) -> float:
        return max(self.results)

    @property
    def std(self) -> float:
        return statistics.pstdev(self.results) if len(self.results) > 1 else 0.0


@dataclass(frozen=True)
class RandomInput:
    """Shape and dtype metadata for server-generated benchmark input values.

    Set both ``low`` and ``high`` to sample uniformly in ``[low, high)``.
    Without bounds, floats use a normal distribution and integers span their
    dtype's full range.
    """

    shape: Tuple[int, ...]
    dtype: str = "float32"
    low: Optional[float] = None
    high: Optional[float] = None


def _random_input_specs(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = []
    for name, value in inputs.items():
        if isinstance(value, RandomInput):
            dtype = np.dtype(value.dtype).name
            if dtype not in proto.DTYPES:
                raise RPCError(f"tensor {name!r} has unsupported dtype {dtype}")
            spec = {"name": name, "dtype": dtype, "shape": list(value.shape)}
            if value.low is not None or value.high is not None:
                if value.low is None or value.high is None:
                    raise ValueError("random input low and high must be set together")
                spec.update(low=value.low, high=value.high)
            specs.append(spec)
        else:
            specs.extend(proto.encode_tensor_specs({name: value}))
    return specs


def _model_bytes(model: ModelLike) -> bytes:
    if isinstance(model, bytes):
        return model
    if isinstance(model, onnx.ModelProto):
        return model.SerializeToString()
    with open(model, "rb") as f:
        return f.read()


class RemoteModel:
    """A model loaded on the server; keeps its onnxruntime session alive between calls."""

    def __init__(self, session: "Session", handle: int):
        self._session, self.handle = session, handle

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        specs, blobs = proto.encode_tensors(inputs)
        reply, out = self._session._call(
            {"op": "run", "handle": self.handle, "tensors": specs}, blobs
        )
        return proto.decode_tensors(reply["tensors"], out)

    def time_evaluator(
        self,
        inputs: Dict[str, Union[np.ndarray, RandomInput]],
        number: int = 1,
        repeat: int = 3,
        random_inputs: bool = False,
        seed: Optional[int] = None,
        pmu: bool = False,
    ) -> ProfileResult:
        """Time server-side calls, optionally generating input values on the server.

        In random mode, ``inputs`` supplies only input names, shapes and dtypes. The
        server generates one seeded random set and reuses it for this timing request.
        Set ``pmu=True`` for TPU-MLIR on SG2002 to return TIU/GDMA hardware counters;
        those runs use the TPU PMU inference interval instead of normal wall timing.
        """
        specs: List[Dict[str, Any]]
        blobs: List[bytes]
        if random_inputs:
            specs, blobs = _random_input_specs(inputs), []
        else:
            if any(isinstance(value, RandomInput) for value in inputs.values()):
                raise ValueError(
                    "RandomInput specifications require random_inputs=True"
                )
            specs, blobs = proto.encode_tensors(inputs)
        header: Dict[str, Any] = {
            "op": "time",
            "handle": self.handle,
            "tensors": specs,
            "number": number,
            "repeat": repeat,
        }
        if random_inputs:
            header["random_inputs"] = True
            if seed is not None:
                header["seed"] = int(seed)
        if pmu:
            header["pmu"] = True
        reply, _ = self._session._call(header, blobs)
        return ProfileResult(list(reply["results"]), reply.get("stats"))

    def close(self) -> None:
        self._session._call({"op": "unload", "handle": self.handle})


class Session:
    def __init__(self, sock: socket.socket, info: Dict[str, Any]):
        self._sock, self.info = sock, info

    def _call(self, header: Dict[str, Any], blobs: Sequence[bytes] = ()):
        proto.send_message(self._sock, header, blobs)
        reply, out = proto.recv_message(self._sock)
        if not reply.get("ok"):
            raise RPCError(reply.get("error", "remote error"))
        return reply, out

    def upload(
        self, data: Union[str, os.PathLike, bytes], name: Optional[str] = None
    ) -> str:
        """Copy a file (or bytes) into the server's workspace; returns its remote name."""
        if isinstance(data, bytes):
            if name is None:
                raise ValueError("uploading raw bytes needs a name")
            payload = data
        else:
            with open(data, "rb") as f:
                payload = f.read()
            name = name or os.path.basename(os.fspath(data))
        reply, _ = self._call({"op": "upload", "name": name}, [payload])
        return reply["name"]

    def load_model(
        self,
        model: ModelLike,
        providers: Optional[Sequence[str]] = None,
        single_threaded: bool = False,
        runtime: str = "onnxruntime",
        device: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> RemoteModel:
        """Load a model on the server: an uploaded file name, a path, bytes or a ``ModelProto``.

        ``runtime="tinygrad"`` runs it through tinygrad's ONNX frontend on ``device`` (a tinygrad
        device name such as ``"NV"`` or ``"CPU"``) with codegen ``options`` such as ``{"BEAM": 2}``.
        ``runtime="tpu_mlir"`` compiles it on a configured TPU-MLIR server and runs the resulting
        CVI model on its configured SG2002 board.
        """
        header: Dict[str, Any] = {
            "op": "load_model",
            "single_threaded": single_threaded,
        }
        if providers:
            header["providers"] = list(providers)
        if runtime != "onnxruntime":
            header.update(runtime=runtime, device=device, options=options or {})
        if isinstance(model, str) and not os.path.exists(model):
            header["name"] = model  # a name previously passed to upload()
            reply, _ = self._call(header)
        else:
            reply, _ = self._call(header, [_model_bytes(model)])
        return RemoteModel(self, reply["handle"])

    def run(
        self,
        model: ModelLike,
        inputs: Dict[str, np.ndarray],
        providers: Optional[Sequence[str]] = None,
        runtime: str = "onnxruntime",
        device: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """One-shot: send the model with its inputs, get the outputs (no handle kept)."""
        specs, blobs = proto.encode_tensors(inputs)
        header: Dict[str, Any] = {"op": "run_once", "tensors": specs}
        if providers:
            header["providers"] = list(providers)
        if runtime != "onnxruntime":
            header.update(runtime=runtime, device=device, options=options or {})
        reply, out = self._call(header, [_model_bytes(model), *blobs])
        return proto.decode_tensors(reply["tensors"], out)

    def xdna_compile_resnet(
        self,
        model: ModelLike,
        kind: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compile a supported XDNA ResNet artifact on the RPC server.

        ``kind`` is ``"resnet"`` (needs a server-side IRON example path in options),
        ``"fused_bottleneck"`` (needs ``options["block"]``), ``"fused_stage"`` (1-3 blocks, or
        1-8 with ``options["blocked"]``), ``"resnet_body"`` (the whole bottleneck body as ONE
        xclbin; needs ``options["groups"]``, block-prefix groups with one core column each), or
        ``"resnet_network"`` (stem Conv + MaxPool + all four bottleneck stages on the device as ONE
        xclbin, one core column per stage; needs ``options["stages"]``: block-prefix lists, projection
        block first), ``"resnet_engine"`` (the layer-sequential engine: stem Conv + MaxPool + every conv
        layer as jobs over all 32 cores; needs no options, ``options["stem"] = False`` leaves the stem
        to the host), ``"graph_engine"`` (the same engine for an arbitrary QDQ CNN such as YOLO: the job structure
        is compiled from the model), or ``"maxpool_u8"``.
        Returned artifact paths are on the RPC server and can be passed to
        :meth:`xdna_run_resnet` on the same server.

        Compiles are cached on the server, content-addressed by the model bytes, the compiler
        command (device, columns, blocks/groups, ...), the ``scripts/xdna`` sources, the IRON
        toolchain identity and compile-relevant environment variables. The reply carries
        ``"cache": "hit" | "miss" | "bypass"`` and ``"cache_key"``; pass
        ``options["no_cache"] = True`` to force a fresh compile. See ``docs/rpc.md``.
        """
        if kind not in (
            "resnet",
            "fused_bottleneck",
            "fused_stage",
            "resnet_body",
            "resnet_network",
            "resnet_engine",
            "graph_engine",
            "maxpool_u8",
        ):
            raise ValueError(f"unsupported XDNA compile kind {kind!r}")
        reply, _ = self._call(
            {"op": "xdna_compile_resnet", "kind": kind, "options": options or {}},
            [_model_bytes(model)],
        )
        return reply["result"]

    def xdna_run_resnet(
        self,
        model: ModelLike,
        manifest: Union[str, os.PathLike, bytes, Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run and profile the XDNA ResNet graph on the RPC server.

        Manifest artifact paths and any fused-kernel paths must be valid on the
        server. The result contains the graph runner's profile and correctness report.
        """
        if isinstance(manifest, bytes):
            manifest_bytes = manifest
        elif isinstance(manifest, dict):
            manifest_bytes = json.dumps(manifest).encode("utf-8")
        else:
            with open(manifest, "rb") as stream:
                manifest_bytes = stream.read()
        reply, _ = self._call(
            {"op": "xdna_run_resnet", "options": options or {}},
            [_model_bytes(model), manifest_bytes],
        )
        return reply["result"]["report"]

    def xdna_compare_vitis_resnet(
        self,
        model: ModelLike,
        manifest: Union[str, os.PathLike, bytes, Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compare full-graph XDNA and Vitis AI runs on the RPC server.

        Both runs use the same model, input seed, warmup count, and iteration count.
        Vitis AI profiling runs separately from its timed loop. Use the same server
        for XDNA, Vitis AI, and all artifact paths referenced by the manifest.
        """
        if isinstance(manifest, bytes):
            manifest_bytes = manifest
        elif isinstance(manifest, dict):
            manifest_bytes = json.dumps(manifest).encode("utf-8")
        else:
            with open(manifest, "rb") as stream:
                manifest_bytes = stream.read()
        reply, _ = self._call(
            {"op": "xdna_compare_vitis_resnet", "options": options or {}},
            [_model_bytes(model), manifest_bytes],
        )
        return reply["result"]

    def executor(self, providers: Optional[Sequence[str]] = None):
        from .executor import RemoteModelExecutor

        return RemoteModelExecutor(self, providers)

    def close(self) -> None:
        try:
            self._call({"op": "close"})
        except (OSError, RPCError):
            pass
        self._sock.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def connect(host: str, port: int, key: str = "", timeout: float = 30.0) -> Session:
    """Connect to a server; ``key`` must match the server's key if it has one."""
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(None)
    proto.send_message(sock, {"op": "hello", "key": key, "client": "onnxsim"})
    reply, _ = proto.recv_message(sock)
    if not reply.get("ok"):
        sock.close()
        raise RPCError(reply.get("error", "handshake failed"))
    return Session(sock, reply["info"])


class TrackerClient:
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self._addr, self._timeout = (host, port), timeout

    def _ask(self, header: Dict[str, Any]) -> Dict[str, Any]:
        with socket.create_connection(self._addr, timeout=self._timeout) as sock:
            proto.send_message(sock, header)
            reply, _ = proto.recv_message(sock)
        if not reply.get("ok"):
            raise RPCError(reply.get("error", "tracker error"))
        return reply

    def summary(self) -> Dict[str, List[Tuple[str, int]]]:
        return self._ask({"op": "summary"})["servers"]

    def request(self, key: str, timeout: float = 30.0) -> Session:
        host, port = self._ask({"op": "request", "key": key})["addr"]
        return connect(host, port, key=key, timeout=timeout)


def connect_tracker(host: str, port: int, timeout: float = 30.0) -> TrackerClient:
    return TrackerClient(host, port, timeout)


@contextmanager
def remote_executor(
    session: Session, providers: Optional[Sequence[str]] = None
) -> Iterator[Any]:
    """Make ``onnxsim.simplify`` (and friends) evaluate constant-folding sub-models remotely."""
    from onnxsim import onnx_simplifier

    executor = session.executor(providers)
    token = onnx_simplifier._executor_override.set(executor)
    try:
        yield executor
    finally:
        onnx_simplifier._executor_override.reset(token)
