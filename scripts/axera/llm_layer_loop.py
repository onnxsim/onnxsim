"""Host-side greedy decode loop for a llama-family LLM (llama, qwen3) compiled
layer by layer with `pulsar2 llm_build` (hidden state bf16, weights s8).

Each transformer layer is its own .axmodel and the final norm + lm_head is a
"post" .axmodel; the embedding lookup, the KV caches, the attention mask and
the argmax live here on the host. Only the decode subgraph of each layer file
is used (one token per call); the prompt is prefilled one token at a time
through it.

Raw byte layout of every tensor that crosses the backend interface (all
little-endian, C order, no padding). H = hidden size, D = kv heads x head
dim, S = kv_cache_len, V = vocabulary. None of them is assumed: `IOSpec` is
read from the compiled models (`IOSpec.from_io` on the worker's `io_info`,
`IOSpec.from_axmodels` on the files) or from a checkpoint's config
(`IOSpec.from_model`), and the file names come from listing the output
directory (`discover_files`). Built with `--kv_cache_len 255`:
SmolLM2-135M H 576, D 192, V 49152; Qwen3-0.6B H 1024, D 1024, V 151936.

  layer inputs, engine order
    K_cache   uint16[1, S, D]    bf16 bits   row s = K row of the token at position s
    V_cache   uint16[1, S, D]    bf16 bits   (head-major)
    indices   uint32[1, 1]                   absolute position of the current token (0..S-1)
    input     uint16[1, 1, H]    bf16 bits   hidden state entering the layer
    mask      uint16[1, 1, S+1]  bf16 bits   additive; [0..S-1] cache rows, [S] current token
  layer outputs, engine order
    K_cache_out uint16[1, 1, D]  bf16 bits   current token's K row (host stores it at row `indices`)
    V_cache_out uint16[1, 1, D]  bf16 bits   current token's V row
    output      uint16[1, 1, H]  bf16 bits   hidden state leaving the layer
  post model
    input     uint16[1, 1, H]    bf16 bits   last layer's output (final RMSNorm is inside post)
    output    uint16[1, 1, V]    bf16 bits   logits

numpy has no bfloat16, so bf16 tensors are carried as uint16 arrays holding
the top 16 bits of the IEEE float32 pattern (`f32_to_bf16`, round to nearest
even; `bf16_to_f32` is exact).

Backends (anything with `run_layer` / `run_post`):
  NumpyBackend   float32 model of the layers' I/O contract (offline tests)
  DeviceBackend  two user callables, load(path) / run(handle, arrays)
  RpcBackend     an onnx-remote AXCL worker; tensors are packed from its `io_info`

See docs/axera-llm-rpc-decode.md.
"""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, Sequence

import llm_reference as ref
import numpy as np

LAYER_INPUT_ORDER = ("K_cache", "V_cache", "indices", "input", "mask")
LAYER_OUTPUT_ORDER = ("K_cache_out", "V_cache_out", "output")
POST_INPUT_ORDER = ("input",)
POST_OUTPUT_ORDER = ("output",)

# ONNX TensorProto.DataType -> element size, for the dtypes an io_info can report.
_ONNX_ITEMSIZE = {
    1: 4,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 4,
    7: 8,
    9: 1,
    10: 2,
    11: 8,
    12: 4,
    13: 8,
    16: 2,
}

# Additive mask value for "do not attend". -65536.0 is exactly representable
# in bf16 (bits 0xC780). 0.0 (bits 0x0000) means "attend".
MASK_NEG = -65536.0


# --------------------------------------------------------------------------
# bf16 <-> float32
# --------------------------------------------------------------------------
def f32_to_bf16(a) -> np.ndarray:
    """float32 -> bf16 bit pattern (uint16), round to nearest, ties to even.

    bf16 is the top 16 bits of the float32 pattern (1 sign, 8 exponent, 7
    fraction bits). Rounding adds 0x7FFF plus the lsb of the kept half, so a
    carry propagates naturally into the exponent (and to +-inf on overflow).
    NaNs are mapped to a quiet NaN with the sign kept.
    """
    f = np.ascontiguousarray(a, dtype="<f4")
    u = f.view("<u4")
    hi = u >> 16
    # carry of (low half + 0x7FFF + lsb of the kept half) out of 16 bits, all within uint32
    hi += ((u & 0xFFFF) + 0x7FFF + (hi & 1)) >> 16
    r = hi.astype(np.uint16)
    nan = np.isnan(f)
    if nan.any():
        r = np.where(nan, ((u >> 16) | 0x7FC0).astype(np.uint16), r)
    return r.astype("<u2")


def bf16_to_f32(b) -> np.ndarray:
    """bf16 bit pattern (uint16) -> float32, exact (low 16 bits zero)."""
    u = np.ascontiguousarray(b, dtype="<u2").astype("<u4") << 16
    return u.view("<f4")


def bf16_round(a) -> np.ndarray:
    """float32 -> nearest bf16-representable float32."""
    return bf16_to_f32(f32_to_bf16(a))


_ONNX_FLOAT16 = 10
_FP16_MAX = 65504.0


def _bf16_to_fp16(b) -> np.ndarray:
    """bf16 bit pattern (uint16) -> IEEE half bit pattern (uint16). Values
    beyond fp16's range are clamped to its largest finite value, so the
    additive mask's -65536 stays finite."""
    f = np.clip(bf16_to_f32(b), -_FP16_MAX, _FP16_MAX)
    return f.astype("<f2").view("<u2")


def _fp16_to_bf16(h) -> np.ndarray:
    """IEEE half bit pattern (any 16-bit array or bytes) -> bf16 bit pattern (uint16)."""
    raw = np.frombuffer(np.ascontiguousarray(h).tobytes(), dtype="<f2")
    return f32_to_bf16(raw.astype("<f4"))


# --------------------------------------------------------------------------
# Files of an llm_build output directory
# --------------------------------------------------------------------------
_LAYER_FILE = re.compile(
    r"^(?P<prefix>.+)_p(?P<prefill>\d+)_l(?P<index>\d+)_together\.axmodel$"
)
_POST_FILE = re.compile(r"^(?P<prefix>.+)_post\.axmodel$")
# --prefill_len values tried when a directory can only be probed (see probe_files).
PROBE_PREFILL_LENS = (128, 64, 256, 32, 96, 192, 320, 512, 1024)


@dataclass(frozen=True)
class ModelFiles:
    """`pulsar2 llm_build` names its files after the checkpoint's `model_type`:
    `{prefix}_p{prefill_len}_l{i}_together.axmodel` per layer and `{prefix}_post.axmodel`
    (SmolLM2: `llama_p128_l0_together.axmodel`; Qwen3: `qwen3_p128_l0_together.axmodel`)."""

    prefix: str
    prefill_len: int
    num_layers: int

    @property
    def layer_pattern(self) -> str:
        return f"{self.prefix}_p{self.prefill_len}_l{{i}}_together.axmodel"

    @property
    def post_name(self) -> str:
        return f"{self.prefix}_post.axmodel"

    def layer_names(self) -> list:
        return [self.layer_pattern.format(i=i) for i in range(self.num_layers)]

    def layer_paths(self, directory: str, sep: str = "/") -> list:
        return [directory.rstrip(sep) + sep + n for n in self.layer_names()]

    def post_path(self, directory: str, sep: str = "/") -> str:
        return directory.rstrip(sep) + sep + self.post_name


def discover_files(names: Iterable[str], prefix: Optional[str] = None) -> ModelFiles:
    """The layer/post file set among `names` (a directory listing). `prefix` picks one
    when the directory holds more than one model; nothing about the names is assumed
    beyond the `_p{N}_l{i}_together.axmodel` / `_post.axmodel` suffixes."""
    names = list(names)
    groups: dict = {}
    for n in names:
        m = _LAYER_FILE.match(os.path.basename(n))
        if m and prefix in (None, m["prefix"]):
            groups.setdefault((m["prefix"], int(m["prefill"])), set()).add(
                int(m["index"])
            )
    if not groups:
        raise FileNotFoundError(
            "no *_p{N}_l{i}_together.axmodel layer files"
            + (f" with prefix {prefix!r}" if prefix else "")
            + f" among {len(names)} names"
        )
    if len(groups) > 1:
        raise ValueError(
            f"more than one layer file set {sorted(groups)}: pass prefix= or explicit paths"
        )
    (((found, prefill), indices),) = groups.items()
    if indices != set(range(len(indices))):
        raise FileNotFoundError(
            f"{found}_p{prefill}: layer files present for {sorted(indices)}, "
            f"not a complete 0..{max(indices)} (is the build still running?)"
        )
    files = ModelFiles(found, prefill, len(indices))
    posts = {
        os.path.basename(n) for n in names if _POST_FILE.match(os.path.basename(n))
    }
    if files.post_name not in posts:
        raise FileNotFoundError(
            f"{len(indices)} layer files but no {files.post_name} (is the build still running?)"
        )
    return files


def list_model_dir(directory: str, prefix: Optional[str] = None) -> ModelFiles:
    """discover_files() on a local directory."""
    return discover_files(os.listdir(directory), prefix)


def probe_files(
    exists: Callable[[str], bool],
    directory: str,
    prefixes: Sequence[str],
    num_layers: int,
    prefill_lens: Sequence[int] = PROBE_PREFILL_LENS,
) -> ModelFiles:
    """For a directory that cannot be listed (it is on the worker): try
    `{prefix}_p{N}_l0_together.axmodel` for each candidate and take the first that
    `exists(path)` accepts. `num_layers` has to come from the checkpoint's config."""
    tried = []
    for prefix in dict.fromkeys(prefixes):
        for prefill in prefill_lens:
            files = ModelFiles(prefix, prefill, num_layers)
            path = files.layer_paths(directory)[0]
            if exists(path):
                return files
            tried.append(os.path.basename(path))
    raise FileNotFoundError(f"none of {tried} could be opened in {directory}")


# --------------------------------------------------------------------------
# I/O spec
# --------------------------------------------------------------------------
def axmodel_io(path: str) -> dict:
    """Inputs and outputs of a compiled model file, in the form of the worker's
    `io_info`: {"inputs": [{"name", "dtype", "shape", "bytes"}, ...], "outputs": [...]}.

    An llm_build layer file holds two `neu mode` nodes, the decode subgraph
    first and the prefill one second; only the first node's tensors are
    reported, which is also what the engine exposes by default."""
    import onnx

    g = onnx.load(path, load_external_data=False).graph
    typed = {v.name: v for v in [*g.input, *g.output, *g.value_info]}
    node = next((n for n in g.node if n.op_type == "neu mode"), None)
    ins = list(node.input) if node else [v.name for v in g.input]
    outs = list(node.output) if node else [v.name for v in g.output]

    def entry(name):
        t = typed[name].type.tensor_type
        shape = [int(d.dim_value) for d in t.shape.dim]
        return {
            "name": name,
            "dtype": int(t.elem_type),
            "shape": shape,
            "bytes": int(np.prod(shape)) * _ONNX_ITEMSIZE[int(t.elem_type)],
        }

    return {"inputs": [entry(n) for n in ins], "outputs": [entry(n) for n in outs]}


def _by_name(entries: Sequence[dict], names: Sequence[str], what: str) -> dict:
    """Engine tensors keyed by our names: by name when the engine uses them, else by position."""
    if len(entries) != len(names):
        raise ValueError(
            f"{what}: engine has {len(entries)} tensors, expected {len(names)} {tuple(names)}"
        )
    if sorted(e["name"] for e in entries) == sorted(names):
        return {e["name"]: e for e in entries}
    return dict(zip(names, entries))


@dataclass(frozen=True)
class IOSpec:
    hidden: int
    kv_dim: int  # num_key_value_heads * head_dim
    cache_len: int  # rows of K_cache = tokens the compiled context holds
    vocab: int
    num_layers: int

    @classmethod
    def from_model(cls, model: "ref.Model", cache_len: int) -> "IOSpec":
        """Sizes of a checkpoint; `cache_len` is the build's --kv_cache_len."""
        return cls(model.H, model.kvdim, int(cache_len), model.vocab, model.L)

    @classmethod
    def from_io(cls, layer_io: dict, post_io: dict, num_layers: int) -> "IOSpec":
        """Sizes of the compiled models, from one layer's and the post model's I/O
        description (the worker's `io_info`, or `axmodel_io`)."""
        li = _by_name(layer_io["inputs"], LAYER_INPUT_ORDER, "layer inputs")
        lo = _by_name(layer_io["outputs"], LAYER_OUTPUT_ORDER, "layer outputs")
        po = _by_name(post_io["outputs"], POST_OUTPUT_ORDER, "post outputs")
        k_shape = list(li["K_cache"]["shape"])
        if len(k_shape) != 3 or k_shape[0] != 1:
            raise ValueError(
                f"K_cache shape {k_shape}, expected [1, cache_len, kv_dim]"
            )
        spec = cls(
            int(li["input"]["shape"][-1]),
            int(k_shape[2]),
            int(k_shape[1]),
            int(po["output"]["shape"][-1]),
            int(num_layers),
        )
        ios = {**li, **lo}
        for name, (dt, shape) in {**spec.layer_inputs, **spec.layer_outputs}.items():
            e = ios[name]
            if (
                tuple(e["shape"]) != shape
                or e["bytes"] != int(np.prod(shape)) * np.dtype(dt).itemsize
            ):
                raise ValueError(
                    f"compiled layer tensor {e['name']!r} is {e['shape']} ({e['bytes']} bytes), "
                    f"the loop's {name} would be {list(shape)} {np.dtype(dt)}"
                )
        return spec

    @classmethod
    def from_axmodels(
        cls, layer_path: str, post_path: str, num_layers: int
    ) -> "IOSpec":
        return cls.from_io(axmodel_io(layer_path), axmodel_io(post_path), num_layers)

    @classmethod
    def from_model_dir(cls, directory: str, prefix: Optional[str] = None) -> "IOSpec":
        """Sizes of the llm_build output in a local directory."""
        files = list_model_dir(directory, prefix)
        return cls.from_axmodels(
            files.layer_paths(directory, os.sep)[0],
            files.post_path(directory, os.sep),
            files.num_layers,
        )

    def check_model(self, model: "ref.Model") -> None:
        """Refuse a checkpoint whose sizes are not the compiled models'."""
        got = (model.H, model.kvdim, model.vocab, model.L)
        want = (self.hidden, self.kv_dim, self.vocab, self.num_layers)
        if got != want:
            raise ValueError(
                "checkpoint (hidden, kv heads x head dim, vocab, layers) = %s, compiled models = %s"
                % (got, want)
            )

    @property
    def layer_inputs(self) -> dict:
        """name -> (numpy dtype, shape), in engine order."""
        return OrderedDict(
            K_cache=("<u2", (1, self.cache_len, self.kv_dim)),
            V_cache=("<u2", (1, self.cache_len, self.kv_dim)),
            indices=("<u4", (1, 1)),
            input=("<u2", (1, 1, self.hidden)),
            mask=("<u2", (1, 1, self.cache_len + 1)),
        )

    @property
    def layer_outputs(self) -> dict:
        return OrderedDict(
            K_cache_out=("<u2", (1, 1, self.kv_dim)),
            V_cache_out=("<u2", (1, 1, self.kv_dim)),
            output=("<u2", (1, 1, self.hidden)),
        )

    @property
    def post_inputs(self) -> dict:
        return OrderedDict(input=("<u2", (1, 1, self.hidden)))

    @property
    def post_outputs(self) -> dict:
        return OrderedDict(output=("<u2", (1, 1, self.vocab)))


# --------------------------------------------------------------------------
# Backend interface
# --------------------------------------------------------------------------
class LayerBackend(Protocol):
    def run_layer(
        self, layer_index: int, feeds: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """feeds: K_cache, V_cache, indices, input, mask (dtypes/shapes of IOSpec.layer_inputs).
        returns: K_cache_out, V_cache_out, output (IOSpec.layer_outputs)."""
        ...

    def run_post(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """feeds: input. returns: output (bf16 logits)."""
        ...


def _check(arrs: dict, want: dict, what: str) -> None:
    if set(arrs) != set(want):
        raise ValueError(f"{what}: names {sorted(arrs)} != {sorted(want)}")
    for n, (dt, shape) in want.items():
        a = arrs[n]
        if a.dtype != np.dtype(dt) or a.shape != shape:
            raise ValueError(
                f"{what}: {n} is {a.dtype}{a.shape}, expected {np.dtype(dt)}{shape}"
            )


class NumpyBackend:
    """Float32 implementation of the compiled layers' believed I/O contract.

    It decodes the same raw tensors a device would get (bf16 bits in, bf16
    bits out), attends over all cache_len + 1 slots and relies on `mask` alone
    to hide unused rows -- so a loop that builds the mask or the cache wrongly
    produces wrong tokens here too. `k_post_rope=True` puts the rotated K row
    in K_cache_out; the host only copies that row, so either choice decodes the same.
    """

    def __init__(self, model: ref.Model, spec: IOSpec, k_post_rope: bool = True):
        self.m = model
        self.spec = spec
        self.k_post_rope = k_post_rope
        spec.check_model(model)

    def run_layer(self, layer_index, feeds):
        s = self.spec
        _check(feeds, s.layer_inputs, f"layer {layer_index} feeds")
        pos = int(feeds["indices"][0, 0])
        if not 0 <= pos < s.cache_len:
            raise ValueError(
                f"indices {pos} out of the RoPE table (0..{s.cache_len - 1})"
            )
        K = bf16_to_f32(feeds["K_cache"][0])
        V = bf16_to_f32(feeds["V_cache"][0])
        x = bf16_to_f32(feeds["input"][0, 0])
        mask = bf16_to_f32(feeds["mask"][0, 0])
        if not self.k_post_rope:  # cache holds unrotated K: rotate row j by position j
            cos, sin = self.m.rope_cos_sin(np.arange(s.cache_len))
            K = self.m._rot(
                K.reshape(s.cache_len, self.m.nkv, self.m.hd),
                cos[:, None],
                sin[:, None],
            ).reshape(K.shape)
        y, k_row, v_row = self.m.layer_step(layer_index, x, pos, K, V, mask)
        if not self.k_post_rope:
            cos, sin = self.m.rope_cos_sin(pos)
            k_row = self.m._rot(
                k_row.reshape(self.m.nkv, self.m.hd), cos, -sin
            ).reshape(-1)  # undo
        return {
            "K_cache_out": f32_to_bf16(k_row).reshape(1, 1, s.kv_dim),
            "V_cache_out": f32_to_bf16(v_row).reshape(1, 1, s.kv_dim),
            "output": f32_to_bf16(y).reshape(1, 1, s.hidden),
        }

    def run_post(self, feeds):
        _check(feeds, self.spec.post_inputs, "post feeds")
        logits = self.m.post(bf16_to_f32(feeds["input"][0, 0]), apply_norm=True)
        return {"output": f32_to_bf16(logits).reshape(1, 1, self.spec.vocab)}


class DeviceBackend:
    """Backend over two user-supplied callables; this class touches no device API.

        load(path) -> handle
        run(handle, list_of_arrays_in_engine_order) -> list_of_arrays

    Inputs handed to `run` are C-contiguous little-endian numpy arrays with
    exactly the engine's byte sizes (uint16 for bf16 tensors, uint32 for
    `indices`); `arr.tobytes()` is the buffer to copy to the device. Each
    returned item may be a numpy array of any dtype, bytes, bytearray or
    memoryview, as long as its byte size is the tensor's; it is reinterpreted
    as little-endian uint16 (bf16 bits). A float32 array with the right
    element count is also accepted and rounded to bf16.

    `model_dir` is a local directory: without `layer_paths` / `post_path` it
    is listed for the layer and post files (`discover_files`; `prefix` picks
    a model when it holds several), and without `spec` the sizes are read
    from the first layer file and the post file (`IOSpec.from_axmodels`).

    Handles are loaded on first use and kept. If the device cannot hold all
    the models, pass `unload` and `max_loaded` for an LRU of handles.

    `output_order` / `input_order` are the engine's tensor orders; change
    them here if a device run shows a different order.
    """

    def __init__(
        self,
        load: Callable[[str], object],
        run: Callable[[object, list], list],
        model_dir: str = ".",
        *,
        layer_paths: Optional[Sequence[str]] = None,
        post_path: Optional[str] = None,
        spec: Optional[IOSpec] = None,
        prefix: Optional[str] = None,
        input_order: Sequence[str] = LAYER_INPUT_ORDER,
        output_order: Sequence[str] = LAYER_OUTPUT_ORDER,
        unload: Optional[Callable[[object], None]] = None,
        max_loaded: Optional[int] = None,
    ):
        self._load, self._run, self._unload = load, run, unload
        if layer_paths is None or post_path is None:
            files = list_model_dir(model_dir, prefix)
            if layer_paths is None:
                layer_paths = files.layer_paths(model_dir, os.sep)
            post_path = post_path or files.post_path(model_dir, os.sep)
        self.layer_paths, self.post_path = list(layer_paths), post_path
        if spec is None:
            spec = IOSpec.from_axmodels(
                self.layer_paths[0], self.post_path, len(self.layer_paths)
            )
        self.spec = spec
        if len(self.layer_paths) != spec.num_layers:
            raise ValueError(
                f"{len(self.layer_paths)} layer paths for {spec.num_layers} layers"
            )
        if sorted(input_order) != sorted(LAYER_INPUT_ORDER) or sorted(
            output_order
        ) != sorted(LAYER_OUTPUT_ORDER):
            raise ValueError(
                "input_order/output_order must be permutations of the layer tensor names"
            )
        self.input_order, self.output_order = tuple(input_order), tuple(output_order)
        self.max_loaded = max_loaded
        self._handles: OrderedDict = OrderedDict()

    def _handle(self, path: str):
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        if self.max_loaded is not None:
            while len(self._handles) >= self.max_loaded:
                _, old = self._handles.popitem(last=False)
                if self._unload is not None:
                    self._unload(old)
        h = self._handles[path] = self._load(path)
        return h

    def close(self) -> None:
        while self._handles:
            _, h = self._handles.popitem()
            if self._unload is not None:
                self._unload(h)

    @staticmethod
    def _pack(feeds: dict, want: dict, order: Sequence[str], what: str) -> list:
        _check(feeds, want, what)
        return [np.ascontiguousarray(feeds[n], dtype=want[n][0]) for n in order]

    @staticmethod
    def _unpack(outs, want: dict, order: Sequence[str], what: str) -> dict:
        outs = list(outs)
        if len(outs) != len(order):
            raise ValueError(
                f"{what}: engine returned {len(outs)} tensors, expected {len(order)} {tuple(order)}"
            )
        res = {}
        for n, o in zip(order, outs):
            dt, shape = want[n]
            count = int(np.prod(shape))
            if isinstance(o, np.ndarray) and o.dtype == np.float32 and o.size == count:
                res[n] = f32_to_bf16(o).reshape(shape)
                continue
            raw = o.tobytes() if isinstance(o, np.ndarray) else bytes(o)
            if len(raw) != count * np.dtype(dt).itemsize:
                raise ValueError(
                    f"{what}: {n} has {len(raw)} bytes, expected {count * np.dtype(dt).itemsize}"
                )
            res[n] = np.frombuffer(raw, dtype=dt).reshape(shape).copy()
        return res

    def run_layer(self, layer_index, feeds):
        s, what = self.spec, f"layer {layer_index}"
        ins = self._pack(feeds, s.layer_inputs, self.input_order, what + " feeds")
        outs = self._run(self._handle(self.layer_paths[layer_index]), ins)
        return self._unpack(outs, s.layer_outputs, self.output_order, what + " outputs")

    def run_post(self, feeds):
        s = self.spec
        ins = self._pack(feeds, s.post_inputs, POST_INPUT_ORDER, "post feeds")
        outs = self._run(self._handle(self.post_path), ins)
        return self._unpack(outs, s.post_outputs, POST_OUTPUT_ORDER, "post outputs")


class RpcBackend:
    """Backend over an onnx-remote AXCL worker (tools/onnx-remote/remote_axcl_worker.cpp).

    `client` is an `onnx_remote_client.Client` (tools/onnx-remote/python), or
    anything with its two calls:

        client.io_info(path) -> {"inputs": [{"name", "dtype", "shape", "bytes"}, ...], "outputs": [...]}
        client.run_path(path, [(onnx_dtype, array), ...]) -> list of arrays

    Paths are the worker's: `model_dir` is a directory on the machine the
    worker runs on. Nothing about the wire dtype is hard-coded. The first use
    of a model asks the worker for its `io_info` (which also loads it into
    the worker's model cache), and each call then sends every tensor in the
    engine's order under the ONNX dtype the worker reported for it. That
    matters because the label decides the payload: a default build
    (`--hidden_state_type bf16`) reports every 16-bit tensor as BFLOAT16 and
    takes the host's bf16 patterns as they are, while an fp16 build reports
    its K/V/hidden tensors with an unknown type code that the worker reads
    as FLOAT16. Those are real IEEE halves, so the host's bf16 patterns are
    converted on the way in and out (`_bf16_to_fp16`, `_fp16_to_bf16`); its
    `mask` stays BFLOAT16. Sending bf16 bits under the FLOAT16 label gave
    infinities on the device.

    Tensors are matched to the engine's by name; if the engine's names are
    not the graph's, by position in `LAYER_INPUT_ORDER` / `LAYER_OUTPUT_ORDER`.

    The worker's directory cannot be listed over the protocol, so the file
    names come from one of: explicit `layer_paths` / `post_path`; `files` (a
    `ModelFiles`, e.g. `list_model_dir()` on a local copy of the directory);
    or `num_layers` plus `prefixes` (the checkpoint's `model_type`), which
    probes the worker with `io_info` for `{prefix}_p{N}_l0_together.axmodel`.
    Without `spec`, the sizes are taken from the worker's `io_info` of the
    first layer and of the post model.

    The K/V caches are the bulk of every layer call (2 x S x D bf16, re-sent
    in full for each layer of each token: the protocol has no persistent
    tensors). This class adds no copy of them: the host's cache array is
    handed to the client as it is, and the client writes it to the socket
    from its own memory.

    `layer_seconds` / `post_seconds` accumulate the wall time of the worker calls.
    """

    def __init__(
        self,
        client,
        model_dir: str = ".",
        *,
        layer_paths: Optional[Sequence[str]] = None,
        post_path: Optional[str] = None,
        spec: Optional[IOSpec] = None,
        files: Optional[ModelFiles] = None,
        num_layers: Optional[int] = None,
        prefixes: Sequence[str] = (),
    ):
        self.client = client
        self._info: dict = {}
        self.calls = 0
        self.layer_seconds = self.post_seconds = 0.0
        if layer_paths is None or post_path is None:
            if files is None:
                if num_layers is None and spec is not None:
                    num_layers = spec.num_layers
                if num_layers is None or not prefixes:
                    raise ValueError(
                        "RpcBackend needs layer_paths + post_path, files=, or num_layers= and prefixes= to probe"
                    )
                files = probe_files(self._exists, model_dir, prefixes, num_layers)
            # Joined with "/": these are paths on the worker (a Linux process), not on this host.
            if layer_paths is None:
                layer_paths = files.layer_paths(model_dir)
            post_path = post_path or files.post_path(model_dir)
        self.files = files
        self.layer_paths, self.post_path = list(layer_paths), post_path
        if spec is None:
            spec = IOSpec.from_io(
                self.info(self.layer_paths[0]),
                self.info(self.post_path),
                len(self.layer_paths),
            )
        self.spec = spec
        if len(self.layer_paths) != spec.num_layers:
            raise ValueError(
                f"{len(self.layer_paths)} layer paths for {spec.num_layers} layers"
            )

    def _exists(self, path: str) -> bool:
        """True when the worker can open `path` (it stays loaded in the worker's cache)."""
        try:
            self.info(path)
            return True
        except Exception:  # RemoteError: the worker could not set the model up
            return False

    def info(self, path: str) -> dict:
        """The worker's io_info for `path`, asked once."""
        if path not in self._info:
            self._info[path] = self.client.io_info(path)
        return self._info[path]

    def preload(self) -> None:
        """Ask for every model's io_info now, so all loads happen before the first token."""
        for path in [*self.layer_paths, self.post_path]:
            self.info(path)

    @staticmethod
    def _order(engine: Sequence[dict], names: Sequence[str], what: str) -> list:
        """Our tensor name for each engine tensor, in engine order."""
        named = _by_name(engine, names, what)
        return [next(n for n, x in named.items() if x is e) for e in engine]

    def _call(
        self,
        path: str,
        feeds: dict,
        want_in: dict,
        in_names,
        want_out: dict,
        out_names,
        what: str,
    ) -> dict:
        _check(feeds, want_in, what + " feeds")
        info = self.info(path)
        tensors = []
        for e, name in zip(
            info["inputs"], self._order(info["inputs"], in_names, what + " inputs")
        ):
            a = np.ascontiguousarray(feeds[name], dtype=want_in[name][0])
            if (
                _ONNX_ITEMSIZE.get(e["dtype"]) != a.dtype.itemsize
                or e["bytes"] != a.nbytes
            ):
                raise ValueError(
                    f"{what}: engine input {e['name']!r} is ONNX dtype {e['dtype']} {e['shape']} "
                    f"({e['bytes']} bytes), the loop has {name} {a.dtype}{a.shape} ({a.nbytes} bytes)"
                )
            if e["dtype"] == _ONNX_FLOAT16:
                # A real fp16 tensor (--hidden_state_type fp16): the host's
                # bf16 pattern has to become an IEEE half, not just be relabelled.
                a = _bf16_to_fp16(a)
            tensors.append((e["dtype"], a.reshape(e["shape"])))
        outs = self.client.run_path(path, tensors)
        self.calls += 1
        order = self._order(info["outputs"], out_names, what + " outputs")
        # float32 would be widened by _unpack; here every output is a raw 16-bit pattern.
        raws = [
            _fp16_to_bf16(o).tobytes()
            if e["dtype"] == _ONNX_FLOAT16
            else np.ascontiguousarray(o).tobytes()
            for o, e in zip(outs, info["outputs"])
        ]
        return DeviceBackend._unpack(raws, want_out, order, what + " outputs")

    def run_layer(self, layer_index, feeds):
        s = self.spec
        t0 = time.perf_counter()
        out = self._call(
            self.layer_paths[layer_index],
            feeds,
            s.layer_inputs,
            LAYER_INPUT_ORDER,
            s.layer_outputs,
            LAYER_OUTPUT_ORDER,
            f"layer {layer_index}",
        )
        self.layer_seconds += time.perf_counter() - t0
        return out

    def run_post(self, feeds):
        s = self.spec
        t0 = time.perf_counter()
        out = self._call(
            self.post_path,
            feeds,
            s.post_inputs,
            POST_INPUT_ORDER,
            s.post_outputs,
            POST_OUTPUT_ORDER,
            "post",
        )
        self.post_seconds += time.perf_counter() - t0
        return out


# --------------------------------------------------------------------------
# Host state and the loop
# --------------------------------------------------------------------------
def _eos_ids(eos) -> frozenset:
    if eos is None:
        return frozenset()
    return frozenset(int(e) for e in (eos if isinstance(eos, Iterable) else [eos]))


def embedding_bits(model: "ref.Model", chunk_rows: int = 8192) -> np.ndarray:
    """A checkpoint's embedding table as bf16 bit patterns, uint16 [vocab, hidden].
    A BF16 checkpoint's table is used as stored, straight from the file mapping (no
    copy, and exact); any other is rounded to bf16 a block of rows at a time."""
    bits = model.embedding_bits()
    if bits is not None:
        return bits
    out = np.empty((model.vocab, model.H), "<u2")
    for beg in range(0, model.vocab, chunk_rows):
        end = min(beg + chunk_rows, model.vocab)
        out[beg:end] = f32_to_bf16(model.embed(range(beg, end)))
    return out


class Host:
    """What the host owns: the embedding table (bf16 bits) and the KV caches.

    `embed` is the table as float32 (rounded to bf16 here) or as uint16 bf16
    bit patterns (used as it is; it may be a read-only file mapping). `eos` is
    one token id or several.

    `step_seconds` gets one (total, post) pair of wall times per fed token.
    """

    def __init__(self, embed: np.ndarray, spec: IOSpec, eos=None):
        self.spec = spec
        self.eos_ids = _eos_ids(eos)
        self.eos = eos if isinstance(eos, int) else min(self.eos_ids, default=None)
        # [vocab, hidden] uint16; exact for a bf16 checkpoint
        self.embed_bf16 = (
            embed if embed.dtype == np.dtype("<u2") else f32_to_bf16(embed)
        )
        if self.embed_bf16.shape != (spec.vocab, spec.hidden):
            raise ValueError(
                f"embedding table {self.embed_bf16.shape}, the compiled models want {(spec.vocab, spec.hidden)}"
            )
        self.reset()

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, spec: IOSpec) -> "Host":
        return cls.from_model(ref.Model.load(checkpoint_dir), spec)

    @classmethod
    def from_model(cls, m: ref.Model, spec: IOSpec) -> "Host":
        spec.check_model(m)
        return cls(embedding_bits(m), spec, m.eos_ids or m.eos)

    def reset(self) -> None:
        s = self.spec
        # bf16 bits 0x0000 = +0.0: unused rows are zero and masked out anyway.
        self.K = np.zeros((s.num_layers, 1, s.cache_len, s.kv_dim), "<u2")
        self.V = np.zeros((s.num_layers, 1, s.cache_len, s.kv_dim), "<u2")
        self.pos = 0
        self.step_seconds: list = []

    def build_mask(self, pos: int, mask_value: float = MASK_NEG) -> np.ndarray:
        """Additive mask for the token at position `pos`: cache rows 0..pos-1
        (the tokens already written) and the last entry (the current token)
        are 0.0, everything else is `mask_value`."""
        s = self.spec
        mf = np.full(s.cache_len + 1, mask_value, np.float32)
        mf[:pos] = 0.0
        mf[s.cache_len] = 0.0
        return f32_to_bf16(mf).reshape(1, 1, s.cache_len + 1)

    def layer_feeds(
        self, layer_index: int, x_bf16: np.ndarray, pos: int, mask: np.ndarray
    ) -> dict:
        return {
            "K_cache": self.K[layer_index],
            "V_cache": self.V[layer_index],
            "indices": np.array([[pos]], dtype="<u4"),
            "input": np.ascontiguousarray(x_bf16, "<u2").reshape(
                1, 1, self.spec.hidden
            ),
            "mask": mask,
        }

    def step(
        self,
        token: int,
        backend: LayerBackend,
        mask_value: float = MASK_NEG,
        want_logits: bool = True,
    ):
        """Feed one token through all layers (+ post). Returns (logits_bf16 [vocab] or None,
        hiddens_bf16 [L+1, hidden]); hiddens[0] is the embedding, hiddens[l+1] layer l's output."""
        s = self.spec
        pos = self.pos
        t0 = time.perf_counter()
        if pos >= s.cache_len:
            raise ValueError(
                f"position {pos} exceeds the compiled context ({s.cache_len} tokens, positions 0..{s.cache_len - 1})"
            )
        # one row, copied out of the table (which may be a file mapping)
        x = np.array(self.embed_bf16[int(token)], dtype="<u2").reshape(1, 1, s.hidden)
        mask = self.build_mask(pos, mask_value)
        hid = [x.reshape(s.hidden).copy()]
        for li in range(s.num_layers):
            out = backend.run_layer(li, self.layer_feeds(li, x, pos, mask))
            _check(out, s.layer_outputs, f"layer {li} outputs")
            # The layer returns only this token's rows; the host owns the cache.
            self.K[li, 0, pos, :] = out["K_cache_out"][0, 0]
            self.V[li, 0, pos, :] = out["V_cache_out"][0, 0]
            x = out["output"]
            hid.append(x.reshape(s.hidden).copy())
        self.pos = pos + 1
        logits = None
        t1 = time.perf_counter()
        if want_logits:
            po = backend.run_post({"input": x})
            _check(po, s.post_outputs, "post outputs")
            logits = po["output"].reshape(s.vocab)
        t2 = time.perf_counter()
        self.step_seconds.append((t2 - t0, t2 - t1))
        return logits, np.stack(hid)


def argmax_bf16(logits_bf16: np.ndarray) -> int:
    """Greedy pick on bf16 logits; ties (frequent at bf16 resolution) go to the lowest id."""
    return int(np.argmax(bf16_to_f32(logits_bf16)))


def decode(
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    backend: LayerBackend,
    *,
    host: Optional[Host],
    stop_at_eos: bool = False,
    mask_value: float = MASK_NEG,
    trace: Optional[dict] = None,
) -> list[int]:
    """Greedy decode. Returns the new token ids.

    The prompt is prefilled one token at a time through the decode subgraph;
    the post model is only called when logits are needed (last prompt token
    and every generated token). `trace`, if given, receives
    'hiddens' float32 [L+1, T_fed, H] and 'logits' float32 [n_post_calls, V].
    """
    if host is None:
        raise ValueError("decode needs a Host (Host.from_checkpoint / Host.from_model)")
    host.reset()
    if not prompt_ids:
        raise ValueError("empty prompt")
    n_fed = len(prompt_ids) + max(max_new_tokens - 1, 0)
    if n_fed > host.spec.cache_len:
        raise ValueError(
            f"{n_fed} tokens to feed, the compiled context holds {host.spec.cache_len}"
        )
    hs, lgs = [], []
    logits = None
    for i, t in enumerate(prompt_ids):
        logits, h = host.step(
            t,
            backend,
            mask_value,
            want_logits=(i == len(prompt_ids) - 1 or trace is not None),
        )
        hs.append(h)
        if trace is not None:
            lgs.append(logits)
    out: list[int] = []
    for i in range(max_new_tokens):
        nxt = argmax_bf16(logits)
        out.append(nxt)
        if (stop_at_eos and nxt in host.eos_ids) or i == max_new_tokens - 1:
            break
        logits, h = host.step(nxt, backend, mask_value)
        hs.append(h)
        lgs.append(logits)
    if trace is not None:
        trace["hiddens"] = bf16_to_f32(np.stack(hs, axis=1))
        trace["logits"] = bf16_to_f32(np.stack(lgs)) if lgs else None
    return out


# --------------------------------------------------------------------------
# Diagnostics against llm_reference.py (docs/axera-llm-rpc-decode.md says how to read them)
# --------------------------------------------------------------------------
def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    )


def compare_layers(
    token_ids: Sequence[int],
    backend: LayerBackend,
    model: ref.Model,
    *,
    mask_value: float = MASK_NEG,
    layers: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> dict:
    """Teacher-forced per-layer check: every layer call gets the *reference*
    hidden state and the *reference* cache (both rounded to bf16), so an
    error at (token t, layer l) is that single call's own error and does not
    come from upstream layers.

    Returns rel-L2 errors: 'output' [T, L] vs ref hidden, 'V' [T, L] vs ref
    v row, 'K_rot' [T, L] vs the RoPE-rotated ref k row and 'K_raw' [T, L] vs
    the unrotated one, plus 'KV_swapped' [T, L] = cosine(K_cache_out, ref v).
    """
    host = Host.from_model(model, backend.spec)
    layers = list(range(model.L)) if layers is None else list(layers)
    T = len(token_ids)
    res = {
        k: np.full((T, model.L), np.nan)
        for k in ("output", "V", "K_rot", "K_raw", "KV_swapped")
    }
    st = model.new_state()
    for t, tok in enumerate(token_ids):
        pos = st["pos"]
        mask = host.build_mask(pos, mask_value)
        K_before = [k.copy() for k in st["K"]]
        V_before = [v.copy() for v in st["V"]]
        _, hid = model.step(st, tok, bf16_round)  # reference with bf16 boundaries
        for li in layers:
            host.K[li, 0, :pos] = f32_to_bf16(K_before[li])
            host.V[li, 0, :pos] = f32_to_bf16(V_before[li])
            out = backend.run_layer(
                li, host.layer_feeds(li, f32_to_bf16(hid[li]), pos, mask)
            )
            y = bf16_to_f32(out["output"]).ravel()
            k = bf16_to_f32(out["K_cache_out"]).ravel()
            v = bf16_to_f32(out["V_cache_out"]).ravel()
            k_rot, v_ref = st["K"][li][pos], st["V"][li][pos]
            cos, sin = model.rope_cos_sin(pos)
            k_raw = model._rot(k_rot.reshape(model.nkv, model.hd), cos, -sin).ravel()
            res["output"][t, li] = _rel(y, hid[li + 1])
            res["V"][t, li] = _rel(v, v_ref)
            res["K_rot"][t, li] = _rel(k, k_rot)
            res["K_raw"][t, li] = _rel(k, k_raw)
            res["KV_swapped"][t, li] = _cos(k, v_ref)
        if verbose:
            r = {k: res[k][t, layers] for k in res}
            print(
                "tok %3d pos %3d  output rel max %.3e (layer %d)  V %.3e  K_rot %.3e  K_raw %.3e  cos(Kout, Vref) max %.3f"
                % (
                    t,
                    pos,
                    r["output"].max(),
                    layers[int(r["output"].argmax())],
                    r["V"].max(),
                    r["K_rot"].max(),
                    r["K_raw"].max(),
                    np.abs(r["KV_swapped"]).max(),
                )
            )
    return res


def compare_decode(
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    backend: LayerBackend,
    model: ref.Model,
    *,
    mask_value: float = MASK_NEG,
    verbose: bool = True,
    host: Optional[Host] = None,
    stop_at_eos: bool = False,
) -> dict:
    """Free-running check: run decode() on `backend` and ref.greedy (float32),
    report tokens, the first diverging step with the reference's top-2 logit
    gap there, and the chained per-layer rel error of the hidden states over
    the common token prefix. The post model runs on every fed token here (the
    trace wants all logits), so timings include one post call per prompt token."""
    tr_d: dict = {}
    tr_r: dict = {}
    toks = decode(
        prompt_ids,
        max_new_tokens,
        backend,
        host=host or Host.from_model(model, backend.spec),
        stop_at_eos=stop_at_eos,
        mask_value=mask_value,
        trace=tr_d,
    )
    rtoks = model.greedy(
        prompt_ids, max_new_tokens, stop_at_eos=stop_at_eos, trace=tr_r
    )
    # The same float32 weights with every layer boundary rounded to bf16, as the compiled
    # pipeline's I/O is: what is left of a difference to it is the layers' own arithmetic.
    btoks = model.greedy(
        prompt_ids, max_new_tokens, rnd=bf16_round, stop_at_eos=stop_at_eos
    )
    first = next((i for i, (a, b) in enumerate(zip(toks, rtoks)) if a != b), None)
    if first is None and len(toks) != len(rtoks):
        first = min(len(toks), len(rtoks))
    n_common = len(prompt_ids) + (len(toks) - 1 if first is None else first)
    n_common = min(n_common, tr_d["hiddens"].shape[1], tr_r["hiddens"].shape[1])
    hd, hr = tr_d["hiddens"][:, :n_common], tr_r["hiddens"][:, :n_common]
    per_layer = np.array([_rel(hd[i], hr[i]) for i in range(hd.shape[0])])  # [L+1]
    # top-2 logit gap of the reference at each of its greedy picks: how much room each had
    picks = np.sort(tr_r["logits"][len(prompt_ids) - 1 :], axis=-1)
    gaps = picks[:, -1] - picks[:, -2]
    res = {
        "tokens": toks,
        "ref_tokens": rtoks,
        "ref_tokens_bf16": btoks,
        "first_divergence": first,
        "hidden_rel_per_layer": per_layer,
        "ref_top2_gaps": gaps,
    }
    if first is not None and first < len(gaps):
        res["ref_top2_gap"] = float(gaps[first])
    if verbose:
        print("backend tokens", toks)
        print("ref    tokens", rtoks)
        print(
            "identical"
            if first is None
            else "first divergence at new token %d, ref top-2 logit gap %s"
            % (first, "%.4f" % res["ref_top2_gap"] if "ref_top2_gap" in res else "n/a")
        )
        print(
            "ref with bf16 layer boundaries:",
            "same tokens as the float32 ref" if btoks == rtoks else btoks,
            "(the backend matches it)" if btoks == toks and btoks != rtoks else "",
        )
        print(
            "ref top-2 logit gap along its greedy path: min %.4f at new token %d "
            "(bf16 logits near 16 are 0.125 apart)"
            % (float(gaps.min()), int(gaps.argmin()))
        )
        print(
            "chained hidden rel-L2 per layer (0 = embedding):",
            np.array2string(per_layer, precision=4, max_line_width=160),
        )
    return res
