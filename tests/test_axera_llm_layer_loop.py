"""Offline tests of the host-side LLM decode loop and of the onnx-remote Python client.

No device, no Docker, no real checkpoint: a tiny random llama-style checkpoint
is written to a temp dir (config.json + a hand-written safetensors file) and
the loop (scripts/axera/llm_layer_loop.py) runs against `NumpyBackend`, a
float32 model of the compiled layers' I/O contract. The same is done for a
qwen3-style config (grouped-query attention, q_norm/k_norm, head_dim that is
not hidden / heads, tied embeddings, rope_theta 1e6) and for a llama-style
one with untied lm_head, float32 storage and one KV head for four heads.

The wire client (tools/onnx-remote/python/onnx_remote_client.py) is checked
two ways: against the C++ reference worker built from this tree
(`cmake -S tools/onnx-remote -B build/onnx-remote && cmake --build build/onnx-remote`,
skipped when the binary is absent; `ONNX_REMOTE_WORKER` overrides the path),
and against a small in-process worker that answers the AXCL worker's
`io_info` / run-by-path requests from the numpy layers, which drives
`RpcBackend` over a real socket.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts", "axera"))
sys.path.insert(0, os.path.join(ROOT, "tools", "onnx-remote", "python"))

import llm_layer_loop as loop  # noqa: E402
import llm_reference as ref  # noqa: E402
import onnx_remote_client as rc  # noqa: E402
import run_llm_rpc  # noqa: E402

HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, VOCAB, LAYERS, CACHE_LEN = (
    32,
    4,
    2,
    64,
    64,
    2,
    15,
)
KV_DIM = KV_HEADS * (HIDDEN // HEADS)
PROMPT = [3, 17, 42, 7, 29]
NEW_TOKENS = 8
# The names llm_build gives a SmolLM2 (model_type llama) build; discovery is tested below.
FILES = loop.ModelFiles("llama", 128, LAYERS)


def write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Minimal safetensors writer: u64 header length, JSON header, raw little-endian data.
    A uint16 array is written as BF16 (bit patterns), float32 as F32."""
    header, blobs, offset = {}, [], 0
    for name, a in tensors.items():
        kind = {np.dtype("<f4"): "F32", np.dtype("<u2"): "BF16"}[a.dtype]
        raw = np.ascontiguousarray(a).tobytes()
        header[name] = {
            "dtype": kind,
            "shape": list(a.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def write_checkpoint(
    directory,
    *,
    seed=0,
    head_dim=None,
    kv_heads=KV_HEADS,
    qk_norm=False,
    tied=True,
    bf16_embedding=True,
    stored_inv_freq=False,
    **config_extra,
) -> str:
    """A tiny random HF-layout checkpoint. `config_extra` goes into config.json as it is."""
    hd = head_dim or HIDDEN // HEADS
    config = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": kv_heads,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": tied,
        "eos_token_id": 0,
        **config_extra,
    }
    if head_dim:
        config["head_dim"] = head_dim
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f)
    rng = np.random.default_rng(seed)

    def w(*shape, scale):
        return (rng.standard_normal(shape) * scale).astype("<f4")

    # The embedding is stored as BF16, like a bf16 checkpoint: exact on the host side.
    embedding = w(VOCAB, HIDDEN, scale=1.0)
    tensors = {
        "model.embed_tokens.weight": loop.f32_to_bf16(embedding)
        if bf16_embedding
        else embedding
    }
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = 1 + w(HIDDEN, scale=0.1)
        # Large q/k so attention is sharp: the decode then depends on the mask and the cache rows.
        tensors[p + "self_attn.q_proj.weight"] = w(HEADS * hd, HIDDEN, scale=0.6)
        tensors[p + "self_attn.k_proj.weight"] = w(kv_heads * hd, HIDDEN, scale=0.6)
        tensors[p + "self_attn.v_proj.weight"] = w(kv_heads * hd, HIDDEN, scale=0.4)
        tensors[p + "self_attn.o_proj.weight"] = w(HIDDEN, HEADS * hd, scale=0.4)
        if qk_norm:
            # After the norm q and k have unit RMS: the gain is what keeps attention sharp.
            tensors[p + "self_attn.q_norm.weight"] = 2 + w(hd, scale=0.5)
            tensors[p + "self_attn.k_norm.weight"] = 2 + w(hd, scale=0.5)
        if (
            stored_inv_freq
        ):  # a buffer old llama checkpoints carry (JackFram/llama-160m)
            theta = config.get("rope_theta", 10000.0)
            tensors[p + "self_attn.rotary_emb.inv_freq"] = (
                1.0 / theta ** (np.arange(0, hd, 2) / hd)
            ).astype("<f4")
        tensors[p + "post_attention_layernorm.weight"] = 1 + w(HIDDEN, scale=0.1)
        tensors[p + "mlp.gate_proj.weight"] = w(INTERMEDIATE, HIDDEN, scale=0.2)
        tensors[p + "mlp.up_proj.weight"] = w(INTERMEDIATE, HIDDEN, scale=0.2)
        tensors[p + "mlp.down_proj.weight"] = w(HIDDEN, INTERMEDIATE, scale=0.2)
    tensors["model.norm.weight"] = 1 + w(HIDDEN, scale=0.1)
    if not tied:
        tensors["lm_head.weight"] = w(VOCAB, HIDDEN, scale=1.0)
    write_safetensors(os.path.join(directory, "model.safetensors"), tensors)
    return str(directory)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    return write_checkpoint(
        tmp_path_factory.mktemp("tiny-llama"), model_type="llama", rope_theta=10000.0
    )


# name -> write_checkpoint arguments. Every size that differs from the base config differs
# in a way the compiled I/O shows: qwen3's KV width (2 x 16) is not hidden / heads x kv heads.
ARCHITECTURES = {
    "qwen3": dict(
        model_type="qwen3",
        head_dim=16,
        kv_heads=2,
        qk_norm=True,
        tied=True,
        rope_theta=1000000,
        attention_bias=False,
        seed=1,
    ),
    # like JackFram/llama-160m (float32, untied lm_head, no rope_theta in the config, a stored
    # inv_freq buffer), but grouped: one KV head serves all four heads
    "llama-gqa": dict(
        model_type="llama",
        kv_heads=1,
        tied=False,
        bf16_embedding=False,
        stored_inv_freq=True,
        seed=2,
    ),
}


@pytest.fixture(scope="module", params=sorted(ARCHITECTURES))
def arch(request, tmp_path_factory):
    """(name, checkpoint dir, reference model) of one extra architecture."""
    directory = write_checkpoint(
        tmp_path_factory.mktemp("tiny-" + request.param), **ARCHITECTURES[request.param]
    )
    return request.param, directory, ref.Model.load(directory)


@pytest.fixture(scope="module")
def model(checkpoint) -> ref.Model:
    return ref.Model.load(checkpoint)


@pytest.fixture(scope="module")
def spec(model) -> loop.IOSpec:
    return loop.IOSpec.from_model(model, CACHE_LEN)


def test_bf16_conversions():
    cases = {
        0x3F800000: 0x3F80,  # 1.0
        0x3F808000: 0x3F80,  # tie, kept half even -> down
        0x3F818000: 0x3F82,  # tie, kept half odd -> up
        0x3F808001: 0x3F81,  # above tie -> up
        0x3F807FFF: 0x3F80,  # below tie -> down
        0x7F7FFFFF: 0x7F80,  # FLT_MAX -> +inf
        0xC7800000: 0xC780,  # -65536.0 (the mask value)
        0x80000000: 0x8000,  # -0.0
        0x00000000: 0x0000,
    }
    for bits, want in cases.items():
        got = int(loop.f32_to_bf16(np.array([bits], "<u4").view("<f4"))[0])
        assert got == want, (hex(bits), hex(got), hex(want))
    assert np.isnan(
        loop.bf16_to_f32(loop.f32_to_bf16(np.array([np.nan], np.float32)))[0]
    )
    # little-endian byte order of the buffer that goes to the engine
    assert loop.f32_to_bf16(np.array([-65536.0], np.float32)).tobytes() == b"\x80\xc7"
    assert loop.f32_to_bf16(np.array([1.0], np.float32)).tobytes() == b"\x80\x3f"
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [
            rng.standard_normal(20000).astype(np.float32) * 100,
            rng.integers(0, 2**32, 20000, dtype=np.uint32).view(np.float32),
        ]
    )
    x = x[np.isfinite(x)]
    bits = loop.f32_to_bf16(x)
    back = loop.bf16_to_f32(bits)
    assert (
        loop.f32_to_bf16(back) == bits
    ).all()  # bf16-representable values are fixed points
    assert (back.view(np.uint32) & 0xFFFF == 0).all()
    # nearest: no other bf16 neighbour is closer
    finite = np.isfinite(back)
    xd, bd = x[finite].astype(np.float64), back[finite].astype(np.float64)
    for step in (-1, 1):
        other = loop.bf16_to_f32(
            (bits[finite].astype(np.int32) + step).astype(np.uint16)
        ).astype(np.float64)
        ok = np.isfinite(other)
        assert (np.abs(xd - bd)[ok] <= np.abs(xd - other)[ok]).all()


def test_safetensors_reader_and_reference(checkpoint, model):
    assert (model.H, model.L, model.kvdim, model.vocab) == (
        HIDDEN,
        LAYERS,
        KV_DIM,
        VOCAB,
    )
    # BF16 is widened exactly: the embedding round-trips through bf16 bit patterns unchanged.
    assert (loop.bf16_to_f32(loop.f32_to_bf16(model.E)) == model.E).all()
    full_logits, full_hidden = model.forward_full(PROMPT)
    cached_logits, cached_hidden = model.forward_cached(PROMPT)
    np.testing.assert_allclose(cached_logits, full_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cached_hidden, full_hidden, rtol=1e-4, atol=1e-4)


def test_loop_reproduces_reference_greedy(model, spec):
    host = loop.Host.from_model(model, spec)
    backend = loop.NumpyBackend(model, spec)
    trace: dict = {}
    got = loop.decode(PROMPT, NEW_TOKENS, backend, host=host, trace=trace)
    assert len(got) == NEW_TOKENS
    assert got == model.greedy(
        PROMPT, NEW_TOKENS, rnd=loop.bf16_round
    )  # same bf16 boundaries
    assert got == model.greedy(PROMPT, NEW_TOKENS)  # plain float32
    assert len(set(got)) > 2  # not a degenerate constant continuation
    assert trace["hiddens"].shape == (LAYERS + 1, len(PROMPT) + NEW_TOKENS - 1, HIDDEN)
    # The host is agnostic to whether the cached K row is RoPE-rotated.
    unrotated = loop.NumpyBackend(model, spec, k_post_rope=False)
    assert loop.decode(PROMPT, NEW_TOKENS, unrotated, host=host) == got
    # The context limit is refused before any layer runs.
    with pytest.raises(ValueError, match="compiled context"):
        loop.decode(PROMPT, CACHE_LEN, backend, host=host)


def test_negative_controls_change_tokens(model, spec):
    backend = loop.NumpyBackend(model, spec)
    want = model.greedy(PROMPT, NEW_TOKENS)
    # nothing masked: the unused (zero) cache rows take part in attention
    assert (
        loop.decode(
            PROMPT,
            NEW_TOKENS,
            backend,
            host=loop.Host.from_model(model, spec),
            mask_value=0.0,
        )
        != want
    )

    class ShiftedSlot(loop.Host):  # writes each token's cache rows one slot too far
        def step(self, token, backend, mask_value=loop.MASK_NEG, want_logits=True):
            pos = self.pos
            result = super().step(token, backend, mask_value, want_logits)
            for cache in (self.K, self.V):
                cache[:, 0, pos + 1] = cache[:, 0, pos]
                cache[:, 0, pos] = 0
            return result

    assert (
        loop.decode(
            PROMPT, NEW_TOKENS, backend, host=ShiftedSlot(model.E, spec, model.eos)
        )
        != want
    )


class FakeEngine:
    """Stands in for DeviceBackend's load()/run(): raw bytes in engine order in, raw bytes out."""

    def __init__(self, backend: loop.NumpyBackend):
        self.backend, self.loads = backend, []

    def load(self, path):
        self.loads.append(path)
        base = os.path.basename(path)
        return (
            ("post", None)
            if "post" in base
            else ("layer", int(base.split("_l")[1].split("_")[0]))
        )

    def run(self, handle, arrays):
        kind, index = handle
        s = self.backend.spec
        raws = [a.tobytes() for a in arrays]
        if kind == "post":
            ((name, (dtype, shape)),) = s.post_inputs.items()
            out = self.backend.run_post(
                {name: np.frombuffer(raws[0], dtype).reshape(shape)}
            )
            return [out["output"].tobytes()]
        feeds = {
            name: np.frombuffer(raw, dtype).reshape(shape)
            for raw, (name, (dtype, shape)) in zip(raws, s.layer_inputs.items())
        }
        out = self.backend.run_layer(index, feeds)
        return [out[name].tobytes() for name in loop.LAYER_OUTPUT_ORDER]


def test_device_backend_and_diagnostics(model, spec):
    want = model.greedy(PROMPT, NEW_TOKENS)
    numpy_backend = loop.NumpyBackend(model, spec)
    engine = FakeEngine(numpy_backend)
    unloaded = []
    device = loop.DeviceBackend(
        engine.load,
        engine.run,
        layer_paths=FILES.layer_paths("/models"),
        post_path=FILES.post_path("/models"),
        spec=spec,
        unload=unloaded.append,
        max_loaded=2,
    )
    assert (
        loop.decode(PROMPT, NEW_TOKENS, device, host=loop.Host.from_model(model, spec))
        == want
    )
    assert (
        len(engine.loads) > LAYERS + 1 and unloaded
    )  # 3 models through an LRU of 2 handles
    device.close()

    layers = loop.compare_layers(PROMPT, numpy_backend, model, verbose=False)
    assert layers["output"].shape == (len(PROMPT), LAYERS)
    assert (
        layers["output"].max() < 0.02
        and layers["V"].max() < 0.02
        and layers["K_rot"].max() < 0.02
    )
    assert (
        layers["K_raw"][1:].min() > 0.1
    )  # the cached K row is the rotated one (position > 0)
    free = loop.compare_decode(PROMPT, NEW_TOKENS, numpy_backend, model, verbose=False)
    assert free["tokens"] == want and free["first_divergence"] is None
    assert (
        free["hidden_rel_per_layer"].shape == (LAYERS + 1,)
        and free["hidden_rel_per_layer"].max() < 0.05
    )


# ---- onnx-remote wire protocol ---------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reference_worker_binary():
    candidates = [
        os.environ.get("ONNX_REMOTE_WORKER"),
        os.path.join(ROOT, "build", "onnx-remote", "onnx-remote-worker"),
        os.path.join(ROOT, "tools", "onnx-remote", "build", "onnx-remote-worker"),
    ]
    return next((c for c in candidates if c and os.access(c, os.X_OK)), None)


@pytest.fixture(scope="module")
def reference_worker():
    binary = _reference_worker_binary()
    if binary is None:
        pytest.skip(
            "onnx-remote-worker is not built (cmake -S tools/onnx-remote -B build/onnx-remote)"
        )
    port = _free_port()
    process = subprocess.Popen([binary, "--port", str(port)], stderr=subprocess.DEVNULL)
    client = rc.Client("127.0.0.1", port, io_timeout=30)
    try:
        deadline = time.time() + 10
        while True:
            try:
                client.capabilities()
                break
            except OSError:
                if process.poll() is not None or time.time() > deadline:
                    pytest.fail("onnx-remote-worker did not start listening")
                time.sleep(0.05)
        yield client
    finally:
        process.kill()
        process.wait()


def test_client_against_reference_worker(reference_worker):
    client = reference_worker
    caps = client.capabilities()
    assert caps["protocol"] == "onnx-remote-v5" and "identity" in caps["supported_ops"]

    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b = rng.standard_normal((1, 3, 1)).astype(np.float32)
    (y,) = client.run("identity", [a])
    assert y.dtype == np.float32 and (y == a).all()
    (y,) = client.run("add", [a, b])  # broadcast, float32 in the C++ `data` vector
    assert y.shape == (2, 3, 4) and (y == a + b).all()
    (y,) = client.run("relu", [a])
    assert (y == np.maximum(a, 0)).all()

    # every non-float dtype travels as raw little-endian bytes and must round-trip bit for bit
    for dtype, np_dtype in rc.NUMPY_DTYPE.items():
        if dtype == rc.BOOL:
            x = rng.integers(0, 2, (3, 5)).astype(np.bool_)
        else:
            x = (
                rng.integers(0, 256, (3, 5, np_dtype.itemsize), dtype=np.uint8)
                .view(np_dtype)
                .reshape(3, 5)
            )
        response = client.request(
            "identity", [rc.Tensor(dtype, x)], profiling=rc.PROFILING_DETAILED
        )
        assert response.dtypes == [dtype], rc.DTYPE_NAME[dtype]
        assert response.outputs[0].shape == (3, 5)
        assert response.outputs[0].tobytes() == x.tobytes(), rc.DTYPE_NAME[dtype]
        assert response.profile and response.profile[0].name == "identity"
    # the numpy dtype picks the wire dtype when none is given; bf16 needs the explicit Tensor
    bits = loop.f32_to_bf16(a)
    assert client.request("identity", [bits]).dtypes == [rc.UINT16]
    response = client.request("identity", [rc.Tensor(rc.BFLOAT16, bits)])
    assert (
        response.dtypes == [rc.BFLOAT16]
        and (loop.bf16_to_f32(response.outputs[0]) == loop.bf16_round(a)).all()
    )
    # rank 0 and an empty tensor
    (y,) = client.run("identity", [np.float32(2.5)])
    assert y.shape == () and y == 2.5
    (y,) = client.run("identity", [np.zeros((0, 4), np.int64)])
    assert y.shape == (0, 4) and y.dtype == np.int64

    # ERROR messages
    with pytest.raises(rc.RemoteError, match="unsupported operation: nope"):
        client.run("nope", [a])
    with pytest.raises(rc.RemoteError, match="float32 tensors only"):
        client.run("add", [a.astype(np.int32), a.astype(np.int32)])
    response = client.request(
        "io_info", model=b"/x.axmodel", check=False
    )  # an AXCL-worker op
    assert not response.ok and "unsupported operation: io_info" in response.error
    with pytest.raises(rc.ProtocolError):
        client.run(
            "identity", [rc.Tensor(rc.BFLOAT16, a)]
        )  # float32 elements are not 2 bytes


class FakeAxclWorker:
    """In-process worker with the AXCL worker's request forms, backed by the numpy layers.

    It mimics a `--hidden_state_type fp16` build: the K/V/hidden tensors are real IEEE
    halves, which the engine reports with type NONE and the worker labels FLOAT16; `mask`
    is BFLOAT16 and `indices` UINT32. A tensor sent under any other dtype is refused, as
    remote_axcl_worker.cpp does. (On an AX8850 a default bf16 build reports every 16-bit
    tensor as BFLOAT16; sending bf16 bits under a FLOAT16 label gave infinities.)
    """

    def __init__(
        self,
        backend: loop.NumpyBackend,
        directory: str = "/models",
        files: loop.ModelFiles = FILES,
    ):
        self.backend, self.spec = backend, backend.spec
        self.requests: list = []
        self.models = {path: i for i, path in enumerate(files.layer_paths(directory))}
        self.models[files.post_path(directory)] = None
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def close(self):
        self.listener.close()

    @staticmethod
    def _dtype(name: str) -> int:
        return {"mask": rc.BFLOAT16, "indices": rc.UINT32}.get(name, rc.FLOAT16)

    def _specs(self, index):
        s = self.spec
        return (
            (s.post_inputs, s.post_outputs)
            if index is None
            else (s.layer_inputs, s.layer_outputs)
        )

    def _io_info(self, path: str) -> str:
        def entries(tensors):
            return [
                {
                    "name": name,
                    "dtype": self._dtype(name),
                    "dtype_name": rc.DTYPE_NAME[self._dtype(name)],
                    "axcl_dtype": {rc.BFLOAT16: 14, rc.UINT32: 8, rc.FLOAT16: 0}[
                        self._dtype(name)
                    ],
                    "shape": list(shape),
                    "bytes": int(np.prod(shape)) * np.dtype(dtype).itemsize,
                }
                for name, (dtype, shape) in tensors.items()
            ]

        ins, outs = self._specs(self.models[path])
        return json.dumps(
            {
                "schema_version": 1,
                "model": path,
                "cached": True,
                "inputs": entries(ins),
                "outputs": entries(outs),
            }
        )

    def _execute(self, request: rc.Request) -> rc.Response:
        response = rc.Response(request_id=request.request_id)
        if request.op == "capabilities":
            response.manifest = json.dumps({"protocol": "onnx-remote-v5"})
            return response
        path = (
            request.model.decode() if request.op in ("io_info", "run") else request.op
        )
        if path not in self.models:
            response.ok, response.error = False, "AXCL model setup failed: " + path
            return response
        if request.op == "io_info":
            response.manifest = self._io_info(path)
            return response
        index = self.models[path]
        ins, outs = self._specs(index)
        if len(request.inputs) != len(ins):
            response.ok, response.error = False, "AXCL input count mismatch"
            return response
        feeds = {}
        for i, (array, dtype, (name, (np_dtype, shape))) in enumerate(
            zip(request.inputs, request.dtypes, ins.items())
        ):
            if dtype != self._dtype(name):
                response.ok = False
                response.error = f"AXCL input {i} dtype mismatch: model wants ONNX dtype {self._dtype(name)}, request sent {dtype}"
                return response
            raw = np.frombuffer(array.tobytes(), np_dtype)
            if (
                dtype == rc.FLOAT16
            ):  # the engine holds IEEE halves; the numpy layers take bf16
                raw = loop._fp16_to_bf16(raw)
            feeds[name] = raw.reshape(shape)
        result = (
            self.backend.run_post(feeds)
            if index is None
            else self.backend.run_layer(index, feeds)
        )
        response.outputs = [
            loop._bf16_to_fp16(result[name]).reshape(result[name].shape)
            if self._dtype(name) == rc.FLOAT16
            else result[name]
            for name in outs
        ]
        response.dtypes = [self._dtype(name) for name in outs]
        return response

    def _serve(self):
        while True:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            with connection:
                request = rc.receive_request(connection)
                self.requests.append((request.op, list(request.dtypes)))
                connection.sendall(rc.encode_response(self._execute(request)))


def test_rpc_backend_packs_from_io_info(model, spec):
    worker = FakeAxclWorker(loop.NumpyBackend(model, spec))
    try:
        client = rc.Client("127.0.0.1", worker.port, io_timeout=30)
        info = client.io_info(FILES.layer_paths("/models")[0])
        assert [e["name"] for e in info["inputs"]] == list(loop.LAYER_INPUT_ORDER)
        assert info["inputs"][0]["shape"] == [1, CACHE_LEN, KV_DIM]
        assert info["inputs"][0]["bytes"] == CACHE_LEN * KV_DIM * 2

        # file names given (as from a local listing), sizes from the worker's io_info
        backend = loop.RpcBackend(client, "/models", files=FILES)
        assert backend.spec == spec
        backend.preload()
        got = loop.decode(
            PROMPT, NEW_TOKENS, backend, host=loop.Host.from_model(model, spec)
        )
        assert got == model.greedy(PROMPT, NEW_TOKENS)
        fed = len(PROMPT) + NEW_TOKENS - 1
        assert (
            backend.calls == fed * LAYERS + NEW_TOKENS
        )  # post only when logits are needed

        # one io_info per model, and every layer call used the dtypes io_info gave
        ops = [op for op, _ in worker.requests]
        assert ops.count("io_info") == LAYERS + 1 + 1  # +1: the direct call above
        layer_calls = [
            d for op, d in worker.requests if op.endswith("_together.axmodel")
        ]
        assert set(map(tuple, layer_calls)) == {
            (rc.FLOAT16, rc.FLOAT16, rc.UINT32, rc.FLOAT16, rc.BFLOAT16)
        }

        # the worker's own errors surface as RemoteError
        with pytest.raises(
            rc.RemoteError, match="AXCL model setup failed: /models/missing.axmodel"
        ):
            client.io_info("/models/missing.axmodel")
        with pytest.raises(
            rc.RemoteError,
            match="dtype mismatch: model wants ONNX dtype 10, request sent 16",
        ):
            client.run(
                FILES.post_path("/models"),
                [rc.Tensor(rc.BFLOAT16, np.zeros((1, 1, HIDDEN), "<u2"))],
            )
        # a path too long for the 128-byte op field travels in the model bytes under op `run`
        long_path = "/models/" + "d" * 150 + "/x.axmodel"
        with pytest.raises(
            rc.RemoteError, match="AXCL model setup failed: " + long_path
        ):
            client.run_path(long_path, [])
        assert worker.requests[-1][0] == "run"
        with pytest.raises(rc.ProtocolError, match="op field"):
            client.run(long_path, [])
    finally:
        worker.close()


# ---- more architectures, and nothing assumed about names or sizes ----------------------------


def test_architectures_reference_and_loop(arch):
    name, directory, model = arch
    want_kv = {"qwen3": 2 * 16, "llama-gqa": 1 * 8}[name]
    assert model.kvdim == want_kv and model.qk_norm == (name == "qwen3")
    assert model.rope_theta == {"qwen3": 1e6, "llama-gqa": 1e4}[name]
    # the reference agrees with itself: one token at a time with a cache == whole sequence
    full_logits, full_hidden = model.forward_full(PROMPT)
    cached_logits, cached_hidden = model.forward_cached(PROMPT)
    np.testing.assert_allclose(cached_logits, full_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cached_hidden, full_hidden, rtol=1e-4, atol=1e-4)

    spec = loop.IOSpec.from_model(model, CACHE_LEN)
    assert spec.kv_dim == want_kv and spec.cache_len == CACHE_LEN
    host = loop.Host.from_model(model, spec)
    backend = loop.NumpyBackend(model, spec)
    want = model.greedy(PROMPT, NEW_TOKENS)
    got = loop.decode(PROMPT, NEW_TOKENS, backend, host=host)
    assert got == want and got == model.greedy(PROMPT, NEW_TOKENS, rnd=loop.bf16_round)
    assert len(set(got)) > 2
    unrotated = loop.NumpyBackend(model, spec, k_post_rope=False)
    assert loop.decode(PROMPT, NEW_TOKENS, unrotated, host=host) == want
    # the mask and the cache rows still decide the result
    assert loop.decode(PROMPT, NEW_TOKENS, backend, host=host, mask_value=0.0) != want
    layers = loop.compare_layers(PROMPT, backend, model, verbose=False)
    assert max(layers[k].max() for k in ("output", "V", "K_rot")) < 0.02

    if name == "qwen3":
        # a BF16 embedding table is used as stored, straight from the file mapping
        assert host.embed_bf16.base is not None and not host.embed_bf16.flags.writeable
        # q_norm/k_norm take part: the same weights read as a plain llama give other tokens
        plain = dict(model.cfg, model_type="llama")
        weights = ref.read_safetensors(os.path.join(directory, "model.safetensors"))
        weights = {k: v for k, v in weights.items() if "_norm.weight" not in k}
        assert ref.Model(plain, weights).greedy(PROMPT, NEW_TOKENS) != want
        with pytest.raises(ValueError, match="q_norm"):
            ref.Model(model.cfg, weights)
    else:
        # lm_head is its own tensor, and the float32 table is rounded to bf16 for the device
        assert model.embedding_bits() is None
        assert (host.embed_bf16 == loop.f32_to_bf16(model.E)).all()
        tied = dict(model.cfg, tie_word_embeddings=True)
        weights = ref.read_safetensors(os.path.join(directory, "model.safetensors"))
        del weights["lm_head.weight"]
        assert ref.Model(tied, weights).greedy(PROMPT, NEW_TOKENS) != want
    # a layer tensor the reference does not implement is refused, not ignored
    weights = ref.read_safetensors(os.path.join(directory, "model.safetensors"))
    weights["model.layers.0.self_attn.layer_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="not implemented"):
        ref.Model(model.cfg, weights)
    # sizes of another architecture are refused by name, before anything runs
    other = loop.IOSpec(spec.hidden, spec.kv_dim + 8, CACHE_LEN, spec.vocab, LAYERS)
    with pytest.raises(ValueError, match="compiled models"):
        loop.NumpyBackend(model, other)


def test_weights_are_converted_on_use(checkpoint):
    # cache_bytes=0: nothing is kept; the results are the same
    kept = ref.Model.load(checkpoint)
    frugal = ref.Model.load(checkpoint, cache_bytes=0)
    assert frugal.greedy(PROMPT, NEW_TOKENS) == kept.greedy(PROMPT, NEW_TOKENS)
    assert not frugal._cached and all(size == 0 for _, size in frugal._cache.values())
    # the BF16 embedding is the only tensor here that needs widening: rows on lookup,
    # the whole table once it is wanted whole (it is also the tied lm_head)
    fresh = ref.Model.load(checkpoint)
    assert fresh.embed(PROMPT).shape == (len(PROMPT), HIDDEN) and fresh._cached == 0
    assert fresh.E.shape == (VOCAB, HIDDEN) and fresh._cached == VOCAB * HIDDEN * 4
    assert fresh.E is fresh.E and kept._cached == fresh._cached
    # F32 tensors are views of the file mapping, not copies
    q = kept.weight("model.layers.0.self_attn.q_proj.weight")
    assert not q.flags.owndata and not q.flags.writeable


def test_discover_files():
    smol = [f"llama_p128_l{i}_together.axmodel" for i in range(30)]
    files = loop.discover_files([*smol, "llama_post.axmodel", "build.log"])
    assert files == loop.ModelFiles("llama", 128, 30)
    assert files.layer_pattern == "llama_p128_l{i}_together.axmodel"
    assert files.layer_paths("/root/smol/out/")[29] == (
        "/root/smol/out/llama_p128_l29_together.axmodel"
    )
    assert files.post_path("/root/smol/out") == "/root/smol/out/llama_post.axmodel"
    qwen = [f"qwen3_p64_l{i}_together.axmodel" for i in range(28)]
    assert loop.discover_files([*qwen, "qwen3_post.axmodel"]) == loop.ModelFiles(
        "qwen3", 64, 28
    )
    # a build that is still running: layers missing, or no post model yet
    with pytest.raises(FileNotFoundError, match="still running"):
        loop.discover_files([*qwen[:5], *qwen[6:], "qwen3_post.axmodel"])
    with pytest.raises(FileNotFoundError, match="no qwen3_post.axmodel"):
        loop.discover_files(qwen)
    with pytest.raises(FileNotFoundError, match="no .*layer files"):
        loop.discover_files(["llama_post.axmodel"])
    both = [*smol, "llama_post.axmodel", *qwen, "qwen3_post.axmodel"]
    with pytest.raises(ValueError, match="more than one"):
        loop.discover_files(both)
    assert loop.discover_files(both, prefix="qwen3").num_layers == 28
    # a directory that cannot be listed is probed, first hit wins
    present = set(loop.ModelFiles("qwen3", 64, 28).layer_paths("/w"))
    asked = []

    def exists(path):
        asked.append(path)
        return path in present

    assert loop.probe_files(exists, "/w", ["qwen3", "llama"], 28) == loop.ModelFiles(
        "qwen3", 64, 28
    )
    assert asked == [
        "/w/qwen3_p128_l0_together.axmodel",
        "/w/qwen3_p64_l0_together.axmodel",
    ]
    with pytest.raises(FileNotFoundError, match="none of"):
        loop.probe_files(exists, "/elsewhere", ["qwen3"], 28)


def write_fake_axmodel(path: str, subgraphs) -> None:
    """A file with the outer structure of a compiled .axmodel: one `neu mode` node per
    subgraph over typed graph inputs/outputs. `subgraphs` is a list of (inputs, outputs),
    each a dict name -> (onnx dtype, shape).

    Built with onnx.helper: the text format cannot spell an op type with a space in it."""
    import onnx
    from onnx import helper

    nodes, graph_inputs, graph_outputs = [], [], []
    for i, (ins, outs) in enumerate(subgraphs):
        nodes.append(
            helper.make_node(
                "neu mode", list(ins), list(outs), name=f"subgraph_npu_{i}"
            )
        )
        graph_inputs += [
            helper.make_tensor_value_info(n, d, s) for n, (d, s) in ins.items()
        ]
        graph_outputs += [
            helper.make_tensor_value_info(n, d, s) for n, (d, s) in outs.items()
        ]
    graph = helper.make_graph(nodes, "axmodel", graph_inputs, graph_outputs)
    onnx.save(helper.make_model(graph), path)


def write_fake_build(
    directory: str, spec: loop.IOSpec, files: loop.ModelFiles, prefill=4
):
    """An llm_build output directory for `spec`: per layer the decode subgraph first and
    the prefill subgraph (suffix _1, other shapes) second, as pulsar2 writes them."""
    bf16, u32 = rc.BFLOAT16, rc.UINT32
    s, d, h = spec.cache_len, spec.kv_dim, spec.hidden
    decode = (
        {
            "K_cache": (bf16, [1, s, d]),
            "V_cache": (bf16, [1, s, d]),
            "indices": (u32, [1, 1]),
            "input": (bf16, [1, 1, h]),
            "mask": (bf16, [1, 1, s + 1]),
        },
        {
            "K_cache_out": (bf16, [1, 1, d]),
            "V_cache_out": (bf16, [1, 1, d]),
            "output": (bf16, [1, 1, h]),
        },
    )
    prefill_graph = (
        {
            "K_cache_1": (bf16, [1, 1, d]),
            "V_cache_1": (bf16, [1, 1, d]),
            "indices_1": (u32, [1, prefill]),
            "input_1": (bf16, [1, prefill, h]),
            "mask_1": (bf16, [1, prefill, prefill]),
        },
        {
            "K_cache_out_1": (bf16, [1, prefill, d]),
            "V_cache_out_1": (bf16, [1, prefill, d]),
            "output_1": (bf16, [1, prefill, h]),
        },
    )
    for path in files.layer_paths(directory, os.sep):
        write_fake_axmodel(path, [decode, prefill_graph])
    write_fake_axmodel(
        files.post_path(directory, os.sep),
        [({"input": (bf16, [1, 1, h])}, {"output": (bf16, [1, 1, spec.vocab])})],
    )


def test_sizes_and_names_come_from_the_compiled_files(arch, tmp_path):
    pytest.importorskip("onnx")
    name, _, model = arch
    spec = loop.IOSpec.from_model(model, CACHE_LEN)
    files = loop.ModelFiles(model.model_type, 4, LAYERS)
    write_fake_build(str(tmp_path), spec, files)

    assert loop.list_model_dir(str(tmp_path)) == files
    io = loop.axmodel_io(files.layer_paths(str(tmp_path), os.sep)[0])
    assert [e["name"] for e in io["inputs"]] == list(
        loop.LAYER_INPUT_ORDER
    )  # decode subgraph only
    assert io["inputs"][0] == {
        "name": "K_cache",
        "dtype": rc.BFLOAT16,
        "shape": [1, CACHE_LEN, spec.kv_dim],
        "bytes": CACHE_LEN * spec.kv_dim * 2,
    }
    assert loop.IOSpec.from_model_dir(str(tmp_path)) == spec

    # NumpyBackend and DeviceBackend with nothing but the directory
    numpy_backend = loop.NumpyBackend(model, loop.IOSpec.from_model_dir(str(tmp_path)))
    engine = FakeEngine(numpy_backend)
    device = loop.DeviceBackend(engine.load, engine.run, str(tmp_path))
    assert device.spec == spec and len(device.layer_paths) == LAYERS
    want = model.greedy(PROMPT, NEW_TOKENS)
    host = loop.Host.from_model(model, device.spec)
    assert loop.decode(PROMPT, NEW_TOKENS, device, host=host) == want
    assert os.path.basename(engine.loads[0]) == files.layer_names()[0]
    assert len(host.step_seconds) == len(PROMPT) + NEW_TOKENS - 1

    # the context limit is the compiled K_cache's row count
    with pytest.raises(ValueError, match=f"compiled context holds {CACHE_LEN}"):
        loop.decode(PROMPT, CACHE_LEN, device, host=host)
    # a checkpoint of another shape is refused against these files
    wrong = loop.IOSpec(spec.hidden, spec.kv_dim * 2, CACHE_LEN, spec.vocab, LAYERS)
    write_fake_build(str(tmp_path), wrong, files)
    with pytest.raises(ValueError, match="compiled models"):
        loop.Host.from_model(model, loop.IOSpec.from_model_dir(str(tmp_path)))


def test_rpc_backend_probes_names_and_reads_sizes(arch):
    _, _, model = arch
    spec = loop.IOSpec.from_model(model, CACHE_LEN)
    files = loop.ModelFiles(
        model.model_type, 64, LAYERS
    )  # not the first prefill length tried
    worker = FakeAxclWorker(loop.NumpyBackend(model, spec), "/root/m/out", files)
    try:
        client = rc.Client("127.0.0.1", worker.port, io_timeout=30)
        backend = loop.RpcBackend(
            client, "/root/m/out", num_layers=model.L, prefixes=[model.model_type]
        )
        assert backend.files == files and backend.spec == spec
        host = loop.Host.from_model(model, backend.spec)
        got = loop.decode(PROMPT, NEW_TOKENS, backend, host=host, stop_at_eos=True)
        assert got == model.greedy(PROMPT, NEW_TOKENS, stop_at_eos=True)
        assert backend.layer_seconds > 0 and backend.post_seconds > 0
        lines = run_llm_rpc.timing_summary(host.step_seconds, len(PROMPT))
        assert lines[0].startswith("prefill") and f"{len(PROMPT)} tokens" in lines[0]
        with pytest.raises(FileNotFoundError, match="none of"):
            loop.RpcBackend(client, "/root/m/out", num_layers=model.L, prefixes=["x"])
    finally:
        worker.close()


def test_cli_decodes_against_a_worker(arch, tmp_path, capsys):
    name, directory, model = arch
    spec = loop.IOSpec.from_model(model, CACHE_LEN)
    files = loop.ModelFiles(model.model_type, 128, LAYERS)
    worker = FakeAxclWorker(loop.NumpyBackend(model, spec), "/root/m/out", files)
    common = ["--port", str(worker.port), "--model-dir", "/root/m/out"]
    common += ["--checkpoint", directory, "--max-new-tokens", str(NEW_TOKENS)]
    ids = ",".join(map(str, PROMPT))
    try:
        # names probed on the worker (nothing local), sizes from io_info, ids instead of text
        assert (
            run_llm_rpc.main([*common, "--prompt-ids", ids, "--compare-reference"]) == 0
        )
        out = capsys.readouterr().out
        assert f"tokens {model.greedy(PROMPT, NEW_TOKENS)}" in out
        assert "probed on the worker" in out and files.post_name in out
        assert f"cache {CACHE_LEN} rows" in out and f"KV width {spec.kv_dim}" in out
        assert "identical to the float32 reference: True" in out
        assert "prefill (prompt, one token per step): %d tokens" % len(PROMPT) in out
        # names from a listing of a local copy of the directory
        for file_name in [*files.layer_names(), files.post_name, "build.log"]:
            (tmp_path / file_name).touch()
        local = ["--local-model-dir", str(tmp_path), "--prompt-ids", ids]
        assert run_llm_rpc.main([*common, *local, "--stop-at-eos"]) == 0
        assert f"listed in {tmp_path}" in capsys.readouterr().out
        # more tokens than the compiled cache holds, a checkpoint without a tokenizer
        with pytest.raises(SystemExit, match=f"compiled context holds {CACHE_LEN}"):
            run_llm_rpc.main([*common[:-1], str(CACHE_LEN), "--prompt-ids", ids])
        with pytest.raises(SystemExit, match="--prompt-ids"):
            run_llm_rpc.main([*common, "--prompt", "hi"])
    finally:
        worker.close()


def test_stop_at_eos_takes_every_eos_id(model, spec):
    backend = loop.NumpyBackend(model, spec)
    want = model.greedy(PROMPT, NEW_TOKENS)
    stop = want[3]
    host = loop.Host(model.E, spec, eos=[VOCAB - 1, stop])  # several ids, as qwen3 has
    got = loop.decode(PROMPT, NEW_TOKENS, backend, host=host, stop_at_eos=True)
    assert got == want[: want.index(stop) + 1]
    assert loop.decode(PROMPT, NEW_TOKENS, backend, host=host) == want


def test_request_is_sent_without_joining_the_payloads():
    # the parts written one by one are exactly encode_request's bytes, small and large tensors alike
    rng = np.random.default_rng(3)
    caches = rng.integers(0, 2**16, (2, 1, 255, 192), dtype=np.uint16)
    tensors = [
        rc.Tensor(rc.FLOAT16, caches[0]),
        rc.Tensor(rc.BFLOAT16, caches[1][:, ::2]),  # not contiguous
        np.array([[7]], "<u4"),
        np.float32(1.5),
        np.zeros((0, 3), np.int64),
    ]
    fields = dict(artifact_id="a", model=b"/m/x.axmodel", request_id=9)
    want = rc.encode_request("run", tensors, **fields)
    left, right = socket.socketpair()
    received = bytearray()

    def drain():
        while chunk := right.recv(1 << 16):
            received.extend(chunk)

    reader = threading.Thread(target=drain)
    reader.start()
    parts = rc.encode_request_parts("run", tensors, **fields)
    with left:
        rc.send_parts(left, parts)
    reader.join()
    right.close()
    assert bytes(received) == want
    # the large payload is handed over as a view of the array, not as a copy
    payload = rc.encode_tensors_parts(tensors[:1])[2]
    assert np.shares_memory(np.frombuffer(payload, np.uint8), caches[0])


QWEN3_LIKE_TEMPLATE = (
    "{%- if messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}{%- if message.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
)
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are tiny.<|im_end|>\\n' }}{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def test_chat_templates(tmp_path):
    by_hand = run_llm_rpc.render_chat_by_hand
    user = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    # qwen3: thinking is disabled by an empty think block after the generation prompt
    assert by_hand(QWEN3_LIKE_TEMPLATE, "hi") == user + "<think>\n\n</think>\n\n"
    assert by_hand(QWEN3_LIKE_TEMPLATE, "hi", enable_thinking=True) == user
    system = "<|im_start|>system\nS<|im_end|>\n"
    assert by_hand(QWEN3_LIKE_TEMPLATE, "hi", "S", True) == system + user
    # plain ChatML: the template's default system message unless one is given
    default = "<|im_start|>system\nYou are tiny.<|im_end|>\n"
    assert by_hand(CHATML_TEMPLATE, "hi") == default + user
    assert by_hand(CHATML_TEMPLATE, "hi", "S") == system + user
    with pytest.raises(ValueError, match="not ChatML"):
        by_hand("{{ bos_token }}[INST] {{ messages[0].content }} [/INST]", "hi")

    # a checkpoint directory: with a template, and without one
    with pytest.raises(ValueError, match="no chat template"):
        run_llm_rpc.render_chat(str(tmp_path), "hi")
    with open(tmp_path / "tokenizer_config.json", "w") as f:
        json.dump({"chat_template": QWEN3_LIKE_TEMPLATE, "eos_token": "<|im_end|>"}, f)
    assert run_llm_rpc.load_chat_template(str(tmp_path))[0] == QWEN3_LIKE_TEMPLATE
    assert (
        run_llm_rpc.render_chat(str(tmp_path), "hi") == user + "<think>\n\n</think>\n\n"
    )

    # where jinja2 is installed, the hand-written forms are what the templates render to
    pytest.importorskip("jinja2")
    jinja = run_llm_rpc.render_chat_jinja
    for template in (QWEN3_LIKE_TEMPLATE, CHATML_TEMPLATE):
        for system_text in (None, "S"):
            for thinking in (False, True):
                assert jinja(template, "hi", system_text, thinking) == by_hand(
                    template, "hi", system_text, thinking
                ), (template, system_text, thinking)
