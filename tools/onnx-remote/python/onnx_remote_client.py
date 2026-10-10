"""Dependency-free Python client for the onnx-remote v5 wire protocol.

It speaks exactly what ``remote_transport.cpp`` does, with numpy as the only
dependency, so a host-side script can drive any onnx-remote worker (the
reference worker, the AXCL worker, ...) without the C++ client:

    client = Client("127.0.0.1", 39501)
    (y,) = client.run("add", [a, b])                       # float32 arrays
    (bits,) = client.run("identity", [Tensor(BFLOAT16, u16)])
    info = client.io_info("/models/layer0.axmodel")        # AXCL worker only

Wire format (all integers big-endian, tensor payloads little-endian):

    header    u32 magic 'ORTR'  u16 version 5  u16 kind (1 RUN, 2 OK, 3 ERROR)  u64 payload bytes
    request   u64 request_id  string op  string artifact_id  u64 n + model bytes
              u64 n + artifact bytes  u32 profiling (0 off, 1 summary, 2 detailed)  tensors
    OK        u64 request_id  tensors  profile events  string artifact_id  string manifest
              u64 n + artifact bytes
    ERROR     u64 request_id  string message
    tensors   u32 count, then per tensor: u32 ONNX dtype  u32 rank  u64 dims[rank]
              u64 element count  payload
    profile   u32 count, then per event: string name  string category  u64 start_us
              u64 duration_us  string detail
    string    u32 length + bytes

A worker serves one request per connection, so every call opens its own
socket. The C++ side keeps float32 in ``Tensor::data`` and every other dtype
in ``Tensor::raw_data``; on the wire both are the same little-endian payload.

numpy has no bfloat16: a BFLOAT16 tensor is a uint16 array of bit patterns.
Use ``Tensor(dtype, array)`` to send an array under an ONNX dtype other than
the one its numpy dtype implies (only the element size has to match), and
``Client.request(...).dtypes`` to see the ONNX dtype of each output.

``decode_request`` / ``encode_response`` are the worker's half of the
protocol; they exist so tests can stand up a fake worker in Python.

Resident inputs (AXCL worker, ``run_resident`` / ``reset_state``): a model's
inputs can stay in the worker's device buffers between requests, e.g. an LLM
layer's KV cache. The wire format is unchanged. ``run_resident`` names its
model like ``run`` does and carries a short text control in the ``artifact``
bytes (``encode_resident_control``); an input sent with no elements
(``keep(dtype)``) keeps the buffer's contents, and an output listed under
``omit`` comes back with no elements:

    state = client.reset_state(path)                 # zero-filled inputs, new state id
    r = client.run_resident(path, [keep(BFLOAT16), keep(BFLOAT16), idx, x, mask],
                            state=state, writes=[(0, 0, pos), (1, 1, pos)], omit=[0, 1])

A worker that no longer holds that state answers "resident state lost ...",
raised as ``ResidentStateLost``; send every input in full (no ``state``) to
establish a new one. See the "AXCL worker" section of the README.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

MAGIC = 0x4F525452  # 'ORTR'
VERSION = 5
KIND_RUN, KIND_OK, KIND_ERROR = 1, 2, 3

PROFILING_OFF, PROFILING_SUMMARY, PROFILING_DETAILED = 0, 1, 2

# ONNX TensorProto.DataType values the transport carries (remote_transport.cpp dtype_bytes).
FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64 = 1, 2, 3, 4, 5, 6, 7
BOOL, FLOAT16, DOUBLE, UINT32, UINT64, BFLOAT16 = 9, 10, 11, 12, 13, 16

# ONNX dtype -> little-endian numpy dtype of the payload. BFLOAT16 is carried as its bit pattern.
NUMPY_DTYPE = {
    FLOAT: np.dtype("<f4"),
    UINT8: np.dtype("u1"),
    INT8: np.dtype("i1"),
    UINT16: np.dtype("<u2"),
    INT16: np.dtype("<i2"),
    INT32: np.dtype("<i4"),
    INT64: np.dtype("<i8"),
    BOOL: np.dtype("?"),
    FLOAT16: np.dtype("<f2"),
    DOUBLE: np.dtype("<f8"),
    UINT32: np.dtype("<u4"),
    UINT64: np.dtype("<u8"),
    BFLOAT16: np.dtype("<u2"),
}
DTYPE_NAME = {
    FLOAT: "FLOAT",
    UINT8: "UINT8",
    INT8: "INT8",
    UINT16: "UINT16",
    INT16: "INT16",
    INT32: "INT32",
    INT64: "INT64",
    BOOL: "BOOL",
    FLOAT16: "FLOAT16",
    DOUBLE: "DOUBLE",
    UINT32: "UINT32",
    UINT64: "UINT64",
    BFLOAT16: "BFLOAT16",
}
# numpy dtype -> the ONNX dtype it is sent as when none is given (uint16 means UINT16, not BFLOAT16).
_ONNX_OF_NUMPY = {
    (v.kind, v.itemsize): k for k, v in NUMPY_DTYPE.items() if k != BFLOAT16
}

# Limits of remote_transport.h; a request that breaks one is refused by the peer.
MAX_OP_BYTES = 128
MAX_ARTIFACT_ID_BYTES = 256
MAX_MANIFEST_BYTES = 64 * 1024
MAX_TENSORS = 32
MAX_RANK = 8
MAX_TENSOR_BYTES = 256 * 1024 * 1024
MAX_MESSAGE_BYTES = 512 * 1024 * 1024
MAX_PROFILE_EVENTS = 256
MAX_ERROR_BYTES = 4 * 512


class RemoteError(RuntimeError):
    """The worker answered with an ERROR message (its text is the exception text)."""


class ResidentStateLost(RemoteError):
    """The worker does not hold the resident state a `run_resident` request referred to
    (model evicted, reloaded, unloaded, run in full, or reset by another client)."""


RESIDENT_STATE_LOST = "resident state lost"


class ProtocolError(RuntimeError):
    """The peer's bytes are not a valid v5 message, or a request breaks a transport limit."""


@dataclass(frozen=True)
class Tensor:
    """An array to send under an explicit ONNX dtype (same element size as the array's)."""

    dtype: int
    array: np.ndarray


@dataclass(frozen=True)
class ProfileEvent:
    name: str
    category: str
    start_us: int
    duration_us: int
    detail: str


@dataclass
class Request:
    op: str
    inputs: list = field(default_factory=list)  # np.ndarray, in wire order
    dtypes: list = field(default_factory=list)  # ONNX dtype of each input
    artifact_id: str = ""
    model: bytes = b""
    artifact: bytes = b""
    profiling: int = PROFILING_OFF
    request_id: int = 0


@dataclass
class Response:
    request_id: int = 0
    ok: bool = True
    error: str = ""
    outputs: list = field(default_factory=list)  # np.ndarray, in wire order
    dtypes: list = field(default_factory=list)  # ONNX dtype of each output
    profile: list = field(default_factory=list)  # ProfileEvent
    artifact_id: str = ""
    manifest: str = ""
    artifact: bytes = b""


TensorLike = Union[np.ndarray, Tensor, tuple]


def keep(dtype: int) -> Tensor:
    """A `run_resident` input that is not sent: a tensor of `dtype` with no elements,
    which tells the worker to keep that input's device buffer as it is."""
    return Tensor(dtype, np.zeros((0,), NUMPY_DTYPE[dtype]))


def encode_resident_control(
    state: Optional[int] = None, writes: Sequence = (), omit: Sequence[int] = ()
) -> bytes:
    """The text control of a `run_resident` request (its `artifact` bytes), one
    directive per line:

        state <id>               the resident state this request continues
        write <out> <in> <slot>  after execution, output <out> is copied into input <in>
                                 at byte offset <slot> * (byte size of output <out>)
        omit <out>               output <out> is not returned (it comes back empty)
    """
    lines = [] if state is None else [f"state {int(state)}"]
    lines += [f"write {int(o)} {int(i)} {int(slot)}" for o, i, slot in writes]
    lines += [f"omit {int(o)}" for o in omit]
    return "".join(line + "\n" for line in lines).encode()


def decode_resident_control(control: bytes) -> tuple:
    """The worker's side: (state id or None, [(out, in, slot), ...], [omitted outputs])."""
    state, writes, omit = None, [], []
    for line in control.decode().splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            numbers = [int(f) for f in fields[1:]]
            if fields[0] == "state" and len(numbers) == 1 and state is None:
                state = numbers[0]
            elif fields[0] == "write" and len(numbers) == 3:
                writes.append(tuple(numbers))
            elif fields[0] == "omit" and len(numbers) == 1:
                omit.append(numbers[0])
            else:
                raise ValueError(line)
        except ValueError:
            raise ProtocolError(f"bad resident control directive: {line[:64]}")
    return state, writes, omit


# ---- encoding ----------------------------------------------------------------
def _u32(v: int) -> bytes:
    return struct.pack(">I", v)


def _u64(v: int) -> bytes:
    return struct.pack(">Q", v)


def _string(s: Union[str, bytes]) -> bytes:
    s = s.encode() if isinstance(s, str) else bytes(s)
    return _u32(len(s)) + s


def _as_tensor(t: TensorLike) -> tuple[int, np.ndarray]:
    """(ONNX dtype, C-contiguous little-endian array) of one input."""
    if isinstance(t, Tensor):
        dtype, a = t.dtype, np.asarray(t.array)
    elif isinstance(t, tuple):
        dtype, a = int(t[0]), np.asarray(t[1])
    else:
        a = np.asarray(t)
        dtype = _ONNX_OF_NUMPY.get((a.dtype.kind, a.dtype.itemsize))
        if dtype is None:
            raise ProtocolError(f"numpy dtype {a.dtype} has no transport dtype")
    want = NUMPY_DTYPE.get(dtype)
    if want is None:
        raise ProtocolError(f"unsupported tensor dtype {dtype}")
    if a.dtype.itemsize != want.itemsize:
        raise ProtocolError(
            f"{a.dtype} array cannot be sent as {DTYPE_NAME[dtype]}: element size "
            f"{a.dtype.itemsize} != {want.itemsize}"
        )
    # Same element size: keep the bytes, only fix byte order and contiguity.
    # ascontiguousarray would turn a rank-0 array into rank 1, hence the reshape.
    shape = a.shape
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False))
    return dtype, a.reshape(shape)


def encode_tensors_parts(tensors: Sequence[TensorLike]) -> list:
    """The tensor list as a sequence of buffers: small `bytes` headers, and each payload
    as a memoryview of the array's own memory (no copy when the array is already
    C-contiguous and little-endian)."""
    if len(tensors) > MAX_TENSORS:
        raise ProtocolError("too many tensors")
    out: list = [_u32(len(tensors))]
    for t in tensors:
        dtype, a = _as_tensor(t)
        if a.ndim > MAX_RANK:
            raise ProtocolError("tensor rank exceeds limit")
        if a.nbytes > MAX_TENSOR_BYTES:
            raise ProtocolError("tensor too large")
        out.append(
            _u32(dtype)
            + _u32(a.ndim)
            + b"".join(_u64(d) for d in a.shape)
            + _u64(a.size)
        )
        if a.nbytes:
            out.append(memoryview(a.reshape(-1)).cast("B"))
    return out


def encode_tensors(tensors: Sequence[TensorLike]) -> bytes:
    return b"".join(encode_tensors_parts(tensors))


def _message(kind: int, body: bytes) -> bytes:
    if len(body) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message too large")
    return _u32(MAGIC) + struct.pack(">HH", VERSION, kind) + _u64(len(body)) + body


def encode_request_payload_parts(
    op: str,
    inputs: Sequence[TensorLike] = (),
    *,
    artifact_id: str = "",
    model: bytes = b"",
    artifact: bytes = b"",
    profiling: int = PROFILING_OFF,
    request_id: int = 0,
) -> list:
    """encode_request_payload() as a sequence of buffers (see encode_tensors_parts)."""
    op_bytes = op.encode()
    if not op_bytes or len(op_bytes) > MAX_OP_BYTES:
        raise ProtocolError(
            f"invalid operation: {len(op_bytes)} bytes, the op field holds 1..{MAX_OP_BYTES}"
        )
    if len(artifact_id.encode()) > MAX_ARTIFACT_ID_BYTES:
        raise ProtocolError("artifact request too large")
    if profiling not in (PROFILING_OFF, PROFILING_SUMMARY, PROFILING_DETAILED):
        raise ProtocolError("invalid profiling level")
    return [
        _u64(request_id),
        _string(op_bytes),
        _string(artifact_id),
        _u64(len(model)),
        bytes(model),
        _u64(len(artifact)),
        bytes(artifact),
        _u32(profiling),
        *encode_tensors_parts(inputs),
    ]


def encode_request_payload(
    op: str, inputs: Sequence[TensorLike] = (), **fields
) -> bytes:
    """The request without the 16-byte transport header (what a DORA message carries)."""
    return b"".join(encode_request_payload_parts(op, inputs, **fields))


def encode_request(op: str, inputs: Sequence[TensorLike] = (), **fields) -> bytes:
    """A complete RUN message (header + payload)."""
    return _message(KIND_RUN, encode_request_payload(op, inputs, **fields))


# Buffers at least this large are written to the socket from their own memory;
# smaller ones are gathered into one write.
_SEND_DIRECT_BYTES = 64 * 1024


def encode_request_parts(op: str, inputs: Sequence[TensorLike] = (), **fields) -> list:
    """encode_request() as a sequence of buffers whose concatenation is the same
    message: the header, then the payload with every tensor's data as a view of
    the array (see encode_tensors_parts)."""
    parts = encode_request_payload_parts(op, inputs, **fields)
    total = sum(len(p) for p in parts)
    if total > MAX_MESSAGE_BYTES:
        raise ProtocolError("message too large")
    return [_u32(MAGIC) + struct.pack(">HH", VERSION, KIND_RUN) + _u64(total), *parts]


def send_parts(sock: socket.socket, parts: Sequence) -> None:
    """Write buffers in order without joining the large ones: a tensor payload goes
    to the socket straight from the array. (Joining copies every payload three times
    on the way, and an llm_build layer call carries two full KV caches.)"""
    small: list = []
    for p in parts:
        if len(p) < _SEND_DIRECT_BYTES:
            small.append(p)
            continue
        if small:
            sock.sendall(b"".join(small))
            small = []
        sock.sendall(p)
    if small:
        sock.sendall(b"".join(small))


def encode_response(response: Response) -> bytes:
    """A complete OK or ERROR message; the worker's side of the protocol."""
    if not response.ok:
        return _message(
            KIND_ERROR,
            _u64(response.request_id)
            + _string(response.error.encode()[:MAX_ERROR_BYTES]),
        )
    tensors = [
        Tensor(d, a) if d is not None else a
        for a, d in zip(
            response.outputs,
            list(response.dtypes)
            + [None] * (len(response.outputs) - len(response.dtypes)),
        )
    ]
    parts = [_u64(response.request_id), encode_tensors(tensors)]
    parts.append(_u32(len(response.profile)))
    for e in response.profile:
        parts.append(
            _string(e.name)
            + _string(e.category)
            + _u64(e.start_us)
            + _u64(e.duration_us)
            + _string(e.detail)
        )
    parts.append(_string(response.artifact_id) + _string(response.manifest))
    parts.append(_u64(len(response.artifact)) + bytes(response.artifact))
    return _message(KIND_OK, b"".join(parts))


# ---- decoding ----------------------------------------------------------------
class _Reader:
    def __init__(self, b: bytes):
        self.b, self.at = memoryview(b), 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.at + n > len(self.b):
            raise ProtocolError("truncated message")
        out = bytes(self.b[self.at : self.at + n])
        self.at += n
        return out

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]

    def string(self, limit: int) -> str:
        n = self.u32()
        if n > limit:
            raise ProtocolError("string exceeds limit")
        return self.take(n).decode(errors="replace")

    def blob(self) -> bytes:
        return self.take(self.u64())

    def done(self) -> bool:
        return self.at == len(self.b)


def _decode_tensors(r: _Reader) -> tuple[list, list]:
    count = r.u32()
    if count > MAX_TENSORS:
        raise ProtocolError("invalid tensor count")
    arrays, dtypes = [], []
    for _ in range(count):
        dtype, rank = r.u32(), r.u32()
        np_dtype = NUMPY_DTYPE.get(dtype)
        if np_dtype is None:
            raise ProtocolError(f"invalid tensor dtype {dtype}")
        if rank > MAX_RANK:
            raise ProtocolError("invalid tensor rank")
        shape = tuple(r.u64() for _ in range(rank))
        n = r.u64()
        elements = 1
        for d in shape:
            elements *= d
        if n != elements or n * np_dtype.itemsize > MAX_TENSOR_BYTES:
            raise ProtocolError("invalid tensor payload")
        raw = r.take(n * np_dtype.itemsize)
        arrays.append(np.frombuffer(raw, np_dtype).reshape(shape).copy())
        dtypes.append(dtype)
    return arrays, dtypes


def decode_response_payload(body: bytes, ok: bool) -> Response:
    r = _Reader(body)
    response = Response(request_id=r.u64(), ok=ok)
    if not ok:
        response.error = r.string(MAX_ERROR_BYTES)
        if not r.done():
            raise ProtocolError("invalid error response")
        return response
    response.outputs, response.dtypes = _decode_tensors(r)
    count = r.u32()
    if count > MAX_PROFILE_EVENTS:
        raise ProtocolError("invalid profile event count")
    for _ in range(count):
        name, category = r.string(128), r.string(128)
        start_us, duration_us = r.u64(), r.u64()
        response.profile.append(
            ProfileEvent(name, category, start_us, duration_us, r.string(512))
        )
    response.artifact_id = r.string(MAX_ARTIFACT_ID_BYTES)
    response.manifest = r.string(MAX_MANIFEST_BYTES)
    response.artifact = r.blob()
    if not r.done():
        raise ProtocolError("trailing response bytes")
    return response


def decode_request_payload(body: bytes) -> Request:
    """The worker's side: parse a RUN payload."""
    r = _Reader(body)
    request_id = r.u64()
    op = r.string(MAX_OP_BYTES)
    if not op:
        raise ProtocolError("invalid operation")
    artifact_id = r.string(MAX_ARTIFACT_ID_BYTES)
    model, artifact = r.blob(), r.blob()
    profiling = r.u32()
    if profiling > PROFILING_DETAILED:
        raise ProtocolError("invalid profiling level")
    inputs, dtypes = _decode_tensors(r)
    if not r.done():
        raise ProtocolError("trailing request bytes")
    return Request(
        op, inputs, dtypes, artifact_id, model, artifact, profiling, request_id
    )


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 22))
        if not chunk:
            raise ProtocolError("connection closed")
        buf += chunk
    return bytes(buf)


def _recv_message(sock: socket.socket, kinds: tuple) -> tuple[int, bytes]:
    magic, version, kind, n = struct.unpack(">IHHQ", _recv_exact(sock, 16))
    if (
        magic != MAGIC
        or version != VERSION
        or kind not in kinds
        or n > MAX_MESSAGE_BYTES
    ):
        raise ProtocolError("invalid message header")
    return kind, _recv_exact(sock, n)


def receive_request(sock: socket.socket) -> Request:
    """The worker's side: read one RUN message from a connected socket."""
    return decode_request_payload(_recv_message(sock, (KIND_RUN,))[1])


def receive_response(sock: socket.socket) -> Response:
    kind, body = _recv_message(sock, (KIND_OK, KIND_ERROR))
    return decode_response_payload(body, kind == KIND_OK)


# ---- client ------------------------------------------------------------------
class Client:
    """One worker endpoint. Each call is one connection (the workers serve one request per socket)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 39501,
        *,
        connect_timeout: Optional[float] = 10.0,
        io_timeout: Optional[float] = 3600.0,
    ):
        self.host, self.port = host, int(port)
        self.connect_timeout, self.io_timeout = connect_timeout, io_timeout
        self._next_id = 1

    def connect(self) -> socket.socket:
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout
        )
        sock.settimeout(self.io_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def request(
        self,
        op: str,
        inputs: Sequence[TensorLike] = (),
        *,
        artifact_id: str = "",
        model: bytes = b"",
        artifact: bytes = b"",
        profiling: int = PROFILING_OFF,
        check: bool = True,
    ) -> Response:
        """Send one request and return the decoded Response. With `check`, an ERROR
        answer raises RemoteError; without it the Response has ok=False and `error` set."""
        request_id = self._next_id
        self._next_id += 1
        parts = encode_request_parts(
            op,
            inputs,
            artifact_id=artifact_id,
            model=model,
            artifact=artifact,
            profiling=profiling,
            request_id=request_id,
        )
        with self.connect() as sock:
            send_parts(sock, parts)
            response = receive_response(sock)
        if response.ok and response.request_id != request_id:
            raise ProtocolError(
                f"response is for request {response.request_id}, sent {request_id}"
            )
        if check and not response.ok:
            if response.error.startswith(RESIDENT_STATE_LOST):
                raise ResidentStateLost(response.error)
            raise RemoteError(response.error)
        return response

    def run(self, op: str, inputs: Sequence[TensorLike] = (), **fields) -> list:
        """Run `op` on `inputs`; returns the outputs as numpy arrays (BFLOAT16 as uint16 bits)."""
        return self.request(op, inputs, **fields).outputs

    def capabilities(self) -> dict:
        return json.loads(self.request("capabilities").manifest)

    def load_compiled(self, artifact_id: str, artifact: bytes) -> Response:
        return self.request("load_compiled", artifact_id=artifact_id, artifact=artifact)

    def run_compiled(
        self, artifact_id: str, inputs: Sequence[TensorLike] = (), **fields
    ) -> list:
        return self.request(
            "run_compiled", inputs, artifact_id=artifact_id, **fields
        ).outputs

    # AXCL worker: the model is named by artifact id, or by a path sent in the `model` bytes.
    def run_path(self, path: str, inputs: Sequence[TensorLike] = (), **fields) -> list:
        """Run the compiled model at `path` (a path on the worker). A path that fits the op
        field goes there, the worker's original request form; a longer one goes in the
        `model` bytes under op `run`."""
        if len(path.encode()) <= MAX_OP_BYTES:
            return self.request(path, inputs, **fields).outputs
        return self.request("run", inputs, model=path.encode(), **fields).outputs

    def io_info(self, path: Optional[str] = None, *, artifact_id: str = "") -> dict:
        """Input/output names, ONNX dtypes, shapes and byte sizes of a compiled model:
        {"inputs": [{"name", "dtype", "dtype_name", "axcl_dtype", "shape", "bytes"}, ...],
        "outputs": [...], "model": path, "cached": bool}. Loads the model into the worker's cache."""
        if (path is None) == (not artifact_id):
            raise ValueError("io_info needs exactly one of path / artifact_id")
        response = self.request(
            "io_info", artifact_id=artifact_id, model=(path or "").encode()
        )
        return json.loads(response.manifest)

    def unload(self, path: Optional[str] = None, *, artifact_id: str = "") -> dict:
        """Drop one model from the worker's cache, or all of them when neither is given."""
        response = self.request(
            "unload", artifact_id=artifact_id, model=(path or "").encode()
        )
        return json.loads(response.manifest)

    # AXCL worker: inputs that stay in the worker's device buffers between requests.
    def reset_state(self, path: Optional[str] = None, *, artifact_id: str = "") -> int:
        """Zero-fill every input buffer of one model (loading it if needed) and start a
        new resident state. Returns the state id to pass to `run_resident`."""
        if (path is None) == (not artifact_id):
            raise ValueError("reset_state needs exactly one of path / artifact_id")
        response = self.request(
            "reset_state", artifact_id=artifact_id, model=(path or "").encode()
        )
        return int(json.loads(response.manifest)["state_id"])

    def run_resident(
        self,
        path: Optional[str],
        inputs: Sequence[TensorLike],
        *,
        artifact_id: str = "",
        state: Optional[int] = None,
        writes: Sequence = (),
        omit: Sequence[int] = (),
        profiling: int = PROFILING_OFF,
    ) -> Response:
        """Run a model whose inputs may stay on the worker.

        `inputs` has one entry per model input, in engine order: an array to upload, or
        `keep(dtype)` to leave that input's device buffer as it is. `state` is the id a
        `reset_state` or an earlier `run_resident` returned; it must be given when any
        input is kept, and the worker answers `ResidentStateLost` when it no longer
        holds that state. Without `state`, every input has to be sent and a new state
        is established. `writes` is a list of (output, input, slot): after execution
        output `output` is copied, on the worker, into input `input` at byte offset
        slot * that output's byte size. Outputs listed in `omit` come back empty.

        Returns the Response; `resident_state_id(response)` is the state id to use next
        (unchanged when `state` was given)."""
        if (path is None) == (not artifact_id):
            raise ValueError("run_resident needs exactly one of path / artifact_id")
        return self.request(
            "run_resident",
            inputs,
            artifact_id=artifact_id,
            model=(path or "").encode(),
            artifact=encode_resident_control(state, writes, omit),
            profiling=profiling,
        )


def resident_state_id(response: Response) -> int:
    """The state id in a `run_resident` / `reset_state` response."""
    return int(json.loads(response.manifest)["state_id"])
