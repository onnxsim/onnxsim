"""Builds a runnable ONNX graph *and* its weights directly from a
HuggingFace Transformers checkpoint directory (``config.json`` +
``*.safetensors``), for the same Llama-family block shape
:mod:`onnxsim.gguf_reconstruct` supports -- RMSNorm, rotary position
embeddings, grouped-query attention, a SwiGLU FFN -- plus Qwen3's per-head
QK-RMSNorm, confirmed against the real ``transformers`` source
(``modeling_qwen3.py``, fetched from the ``huggingface/transformers`` repo
at the time this was written): ``query_states = self.q_norm(self.q_proj(
hidden_states).view(hidden_shape)).transpose(1, 2)`` -- i.e. project, reshape
to per-head vectors, RMSNorm *each head's own* ``head_dim`` slice (not the
whole hidden state), *then* transpose and apply RoPE. Same for
``key_states``/``k_norm``. Qwen3 also decouples ``head_dim`` from
``hidden_size // num_attention_heads`` -- confirmed for real against a
`Qwen/Qwen3-0.6B` checkpoint (``hidden_size=1024``,
``num_attention_heads=16`` would imply 64, but the checkpoint's own
``config.json`` sets ``head_dim=128``) -- so ``head_dim`` is read from the
config with that fallback, never computed unconditionally.

Complements :mod:`onnxsim.gguf_reconstruct`: same graph-building approach
(construct the topology from declared hyperparameters, then attach the
checkpoint's own tensors as initializers), applied to the source format
most LLM checkpoints actually ship as -- a raw HuggingFace directory --
rather than requiring a GGUF conversion first. Reuses that module's
op-building helpers (``_Builder``, ``_linear``, ``_rmsnorm``,
``_apply_rope``, ...) unchanged, so both builders produce structurally
identical graphs for the same architecture family.

Supported ``config.json`` ``"model_type"``: ``"llama"``, ``"mistral"``,
``"qwen2"``, ``"qwen3"``. Same "known architecture template, fail clearly
otherwise" philosophy as :mod:`onnxsim.gguf_reconstruct` -- see that
module's docstring for why. No MoE support (unlike that module's Mixtral
path): HF's per-expert tensor layout (``mlp.experts.{i}.gate_proj.weight``,
one initializer per expert) differs from GGUF's fused-expert tensors
``_moe_ffn`` expects (one tensor stacking every expert), and adapting that
function to the HF layout hasn't been done here -- ``num_local_experts``/
``num_experts`` present in the config raises :class:`UnsupportedArchitectureError`.

Reads ``*.safetensors`` with a small self-contained parser (the format: an
8-byte little-endian header-length prefix, a JSON header describing each
tensor's dtype/shape/byte-offset, then the raw tensor data) -- no
``safetensors`` package dependency, matching how
:mod:`onnxsim.gguf_reconstruct` parses GGUF's own binary format directly
rather than depending on a ``gguf`` package. Handles both a single
``model.safetensors`` and a sharded checkpoint (``model.safetensors.index.
json`` + ``model-NNNNN-of-MMMMM.safetensors``), reading only the bytes for
tensors this graph actually declares, not the whole checkpoint into memory
at once.

Scope note on shapes, same as :mod:`onnxsim.gguf_reconstruct`:
``batch_size``/``seq_len`` are concrete, caller-chosen static dimensions,
not dynamic axes, and there is no KV-cache-aware incremental-decode graph
here -- a single-shape prefill-style forward pass only.

Precision note, confirmed for real against `Qwen/Qwen3-0.6B` (stored as
BF16 end to end, ~1.5GB): every non-FLOAT32-dtype weight gets an explicit
``Cast`` to FLOAT32 in the graph, right after being declared (see
``declare()``). This was tried two ways, in order:

1. Upcast the *stored initializer bytes* themselves (BF16 -> FLOAT32) on
   read, keeping every op's input already FLOAT32 with no extra node. This
   is what GGUF's own raw-dtype path does *not* need, but was tried here
   first for symmetry with it -- and confirmed necessary regardless: the
   op-building helpers reused from ``gguf_reconstruct`` (``_rmsnorm``,
   ``_apply_rope``, the causal mask/``inv_sqrt_d`` constants, ...) build
   every constant as FLOAT32, so a *preserved*-BF16 weight reaching one of
   those ops is a mixed-dtype node onnxruntime rejects outright (``Type
   Error: ... (Add) bound to different types (tensor(bfloat16) and
   tensor(float))``).
2. But upcasting the stored bytes roughly doubles the model's total size --
   confirmed to matter, not just in principle: `Qwen/Qwen3-0.6B`'s real
   ~1.5GB BF16 checkpoint upcasts to ~3.0GB, over protobuf's ~2.1GB
   (``2**31 - 1``) single-message serialization limit, and
   ``onnx.checker.check_model``/``model.SerializeToString()`` failed
   outright (``EncodeError: Failed to serialize proto``) on the resulting
   model. So the *initializer* stays the checkpoint's own compact dtype
   (BF16 stays BF16-sized), and a graph-level ``Cast`` node upcasts the
   *value* to FLOAT32 at first use instead -- same fix for the mixed-dtype
   problem, none of the size blowup.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx.helper

from onnxsim.gguf_reconstruct import (
    _IR_VERSION,
    _OPSET,
    UnsupportedArchitectureError,
    _apply_rope,
    _Builder,
    _linear,
    _rmsnorm,
    _unsqueeze,
)

_SAFETENSORS_DTYPE_TO_ONNX: Dict[str, Tuple[int, np.dtype]] = {
    "F64": (onnx.TensorProto.DOUBLE, np.dtype("<f8")),
    "F32": (onnx.TensorProto.FLOAT, np.dtype("<f4")),
    "F16": (onnx.TensorProto.FLOAT16, np.dtype("<f2")),
    "I64": (onnx.TensorProto.INT64, np.dtype("<i8")),
    "I32": (onnx.TensorProto.INT32, np.dtype("<i4")),
    "I16": (onnx.TensorProto.INT16, np.dtype("<i2")),
    "I8": (onnx.TensorProto.INT8, np.dtype("i1")),
    "U8": (onnx.TensorProto.UINT8, np.dtype("u1")),
    "BOOL": (onnx.TensorProto.BOOL, np.dtype("?")),
    # BF16 has no native numpy dtype and is upcast to FLOAT32 on read
    # instead (a confirmed-necessary special case, not an oversight) --
    # see _read_tensor.
}

_SUPPORTED_MODEL_TYPES = frozenset({"llama", "mistral", "qwen2", "qwen3"})


class _SafetensorsEntry:
    __slots__ = ("file_path", "dtype", "shape", "start", "end")

    def __init__(
        self, file_path: str, dtype: str, shape: List[int], start: int, end: int
    ):
        self.file_path = file_path
        self.dtype = dtype
        self.shape = shape
        self.start = start
        self.end = end


def _read_safetensors_header(path: str) -> Dict[str, dict]:
    """The JSON header of one ``.safetensors`` file: ``{tensor_name: {"dtype":
    str, "shape": [int, ...], "data_offsets": [start, end]}}`` (plus a
    ``"__metadata__"`` entry this ignores) -- no tensor byte data read."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _index_safetensors_checkpoint(hf_dir: str) -> Dict[str, _SafetensorsEntry]:
    """Map every tensor name in `hf_dir` to the file/byte-range holding it,
    across either a single ``model.safetensors`` or a sharded checkpoint
    (``model.safetensors.index.json`` + ``model-NNNNN-of-MMMMM.safetensors``).
    Data offsets in a safetensors header are relative to the *end of the
    header itself* within that file, so each entry's ``start``/``end`` here
    are already adjusted to be absolute file offsets.
    """
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        file_names = sorted(set(weight_map.values()))
    else:
        single = os.path.join(hf_dir, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(
                f"no model.safetensors or model.safetensors.index.json under {hf_dir!r}"
            )
        file_names = ["model.safetensors"]
        weight_map = None

    entries: Dict[str, _SafetensorsEntry] = {}
    for file_name in file_names:
        file_path = os.path.join(hf_dir, file_name)
        with open(file_path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
        header_end = 8 + header_len
        header = _read_safetensors_header(file_path)
        for name, info in header.items():
            if weight_map is not None and weight_map.get(name) != file_name:
                continue
            start, end = info["data_offsets"]
            entries[name] = _SafetensorsEntry(
                file_path,
                info["dtype"],
                info["shape"],
                header_end + start,
                header_end + end,
            )
    return entries


def _read_tensor(entry: _SafetensorsEntry, name: str) -> onnx.TensorProto:
    """Read one tensor's raw bytes and wrap them as an ONNX initializer,
    preserving the checkpoint's own dtype (e.g. real BF16 weights stay
    BF16-sized in the initializer -- see ``declare()``'s own docstring for
    why the *graph*, not the stored bytes, is where this gets upcast to
    FLOAT32 for actual computation).
    """
    with open(entry.file_path, "rb") as f:
        f.seek(entry.start)
        raw = f.read(entry.end - entry.start)
    if entry.dtype == "BF16":
        return onnx.helper.make_tensor(
            name, onnx.TensorProto.BFLOAT16, entry.shape, vals=raw, raw=True
        )
    if entry.dtype not in _SAFETENSORS_DTYPE_TO_ONNX:
        raise UnsupportedArchitectureError(
            f"tensor {name!r} has safetensors dtype {entry.dtype!r}, which "
            "this reconstructor cannot decode (supported: F64/F32/F16/BF16/"
            "I64/I32/I16/I8/U8/BOOL)"
        )
    onnx_dtype, np_dtype = _SAFETENSORS_DTYPE_TO_ONNX[entry.dtype]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(entry.shape)
    # safetensors is always little-endian on disk, so np_dtype above is
    # explicitly "<..." regardless of host order -- correct for *decoding*
    # `raw` (frombuffer reinterprets those bytes as little-endian
    # regardless of host order), but on a big-endian host the resulting
    # array's own dtype object stays tagged "<..." too, and
    # onnx.numpy_helper.from_array's dtype->TensorProto lookup only
    # recognizes a *native*-order dtype (np.dtype("f4") is "<f4" on a
    # little-endian host, so this was never visible there) -- confirmed via
    # this module's own big-endian CI. astype() to the platform's native
    # byte order both fixes that lookup and -- unlike a bare view/reinterpret
    # -- correctly byte-swaps the underlying bytes so the decoded values
    # themselves are unchanged (a single-byte dtype like I8/U8/BOOL has no
    # byte order to normalize, so this is a harmless no-op copy for those).
    arr = arr.astype(arr.dtype.newbyteorder("="))
    return onnx.numpy_helper.from_array(arr, name=name)


def read_hf_config(hf_dir: str) -> dict:
    """``config.json`` from a HuggingFace checkpoint directory, as a plain
    dict -- no ``transformers`` dependency, this is just JSON."""
    with open(os.path.join(hf_dir, "config.json")) as f:
        return json.load(f)


def _reconstruct_llama_family_hf(
    config: dict, entries: Dict[str, _SafetensorsEntry], batch_size: int, seq_len: int
) -> onnx.GraphProto:
    model_type = config["model_type"]

    n_embd = int(config["hidden_size"])
    n_layer = int(config["num_hidden_layers"])
    n_head = int(config["num_attention_heads"])
    n_head_kv = int(config.get("num_key_value_heads", n_head))
    eps = float(config.get("rms_norm_eps", 1e-5))
    freq_base = float(config.get("rope_theta", 10000.0))
    attention_bias = bool(config.get("attention_bias", False))
    vocab_size = int(config["vocab_size"])
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    for key in ("num_local_experts", "num_experts"):
        if config.get(key):
            raise UnsupportedArchitectureError(
                f"config has {key}={config[key]!r} -- MoE HF checkpoints are "
                "not supported (see this module's docstring for why)"
            )

    if n_embd % n_head != 0 and "head_dim" not in config:
        raise UnsupportedArchitectureError(
            f"hidden_size={n_embd} is not divisible by "
            f"num_attention_heads={n_head}, and config has no explicit "
            "head_dim to fall back to"
        )
    head_dim = int(config.get("head_dim") or (n_embd // n_head))
    if n_head % n_head_kv != 0:
        raise UnsupportedArchitectureError(
            f"num_attention_heads={n_head} is not a multiple of "
            f"num_key_value_heads={n_head_kv} (grouped-query attention "
            "requires an integer head-repeat factor)"
        )
    n_rep = n_head // n_head_kv
    use_qk_norm = model_type == "qwen3"

    b = _Builder()

    def declare(name: str) -> str:
        """Declare weight `name` as an initializer and return the name to
        actually *use* as a graph value -- which is a `Cast`-to-FLOAT32
        node's output, not the initializer itself, whenever the checkpoint's
        own dtype isn't already FLOAT32. See this module's docstring for
        why the cast lives in the graph rather than in the stored bytes.
        """
        entry = entries.get(name)
        if entry is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor {name!r} for "
                f"model_type={model_type!r}"
            )
        b.initializers.append(_read_tensor(entry, name))
        if entry.dtype == "F32":
            return name
        return b.op("Cast", [name], f"{name}.f32", to=onnx.TensorProto.FLOAT)

    def declare_optional(name: str) -> Optional[str]:
        return declare(name) if name in entries else None

    token_embd = declare("model.embed_tokens.weight")

    input_ids = "input_ids"
    position_ids = "position_ids"
    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            input_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
        onnx.helper.make_tensor_value_info(
            position_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
    ]

    x = b.op("Gather", [token_embd, input_ids], "embed", axis=0)

    # RoPE cos/sin: identical across every layer -- computed once, same as
    # gguf_reconstruct._reconstruct_llama_family.
    inv_freq = 1.0 / (
        freq_base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    inv_freq_c = b.const(
        inv_freq.reshape(1, 1, -1).astype(np.float32), prefix="inv_freq"
    )
    pos_f = b.op("Cast", [position_ids], "pos_f", to=onnx.TensorProto.FLOAT)
    pos_unsq = _unsqueeze(b, pos_f, [-1], "pos_unsq")
    freqs = b.op("Mul", [pos_unsq, inv_freq_c], "freqs")
    emb = b.op("Concat", [freqs, freqs], "rope_emb", axis=-1)
    cos = b.op("Cos", [emb], "rope_cos")
    sin = b.op("Sin", [emb], "rope_sin")
    cos_b = _unsqueeze(b, cos, [1], "rope_cos_b")
    sin_b = _unsqueeze(b, sin, [1], "rope_sin_b")

    causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    mask_c = b.const(causal_mask, prefix="causal_mask")
    inv_sqrt_d = b.const(
        np.array(1.0 / np.sqrt(head_dim), dtype=np.float32), prefix="inv_sqrt_d"
    )

    def reshape(t: str, dims: List[int], prefix: str) -> str:
        return b.op("Reshape", [t, b.shape_const(dims)], prefix)

    for i in range(n_layer):
        p = f"model.layers.{i}"
        resid = x
        h = _rmsnorm(
            b, x, declare(f"{p}.input_layernorm.weight"), eps, f"{p}.attn_norm"
        )

        q = _linear(
            b,
            h,
            declare(f"{p}.self_attn.q_proj.weight"),
            declare_optional(f"{p}.self_attn.q_proj.bias") if attention_bias else None,
            f"{p}.q_proj",
        )
        k = _linear(
            b,
            h,
            declare(f"{p}.self_attn.k_proj.weight"),
            declare_optional(f"{p}.self_attn.k_proj.bias") if attention_bias else None,
            f"{p}.k_proj",
        )
        v = _linear(
            b,
            h,
            declare(f"{p}.self_attn.v_proj.weight"),
            declare_optional(f"{p}.self_attn.v_proj.bias") if attention_bias else None,
            f"{p}.v_proj",
        )

        q = reshape(q, [batch_size, seq_len, n_head, head_dim], f"{p}.q_r")
        k = reshape(k, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.k_r")
        v = reshape(v, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.v_r")

        if use_qk_norm:
            # Confirmed against real transformers/models/qwen3/modeling_qwen3.py:
            # RMSNorm is applied to each head's own head_dim slice *before*
            # the transpose to [B, H, S, D] -- i.e. on the still-[B, S, H,
            # D]-shaped tensor, exactly where this module docstring says.
            q = _rmsnorm(
                b, q, declare(f"{p}.self_attn.q_norm.weight"), eps, f"{p}.q_norm"
            )
            k = _rmsnorm(
                b, k, declare(f"{p}.self_attn.k_norm.weight"), eps, f"{p}.k_norm"
            )

        q = b.op("Transpose", [q], f"{p}.q_t", perm=[0, 2, 1, 3])
        k = b.op("Transpose", [k], f"{p}.k_t", perm=[0, 2, 1, 3])
        v = b.op("Transpose", [v], f"{p}.v_t", perm=[0, 2, 1, 3])

        q = _apply_rope(b, q, cos_b, sin_b, head_dim, f"{p}.q_rope")
        k = _apply_rope(b, k, cos_b, sin_b, head_dim, f"{p}.k_rope")

        # Grouped-query attention via broadcasting -- same design as
        # gguf_reconstruct._reconstruct_llama_family, see its own comment.
        q5 = reshape(q, [batch_size, n_head_kv, n_rep, seq_len, head_dim], f"{p}.q5")
        k5 = _unsqueeze(b, k, [2], f"{p}.k5")
        v5 = _unsqueeze(b, v, [2], f"{p}.v5")

        k5t = b.op("Transpose", [k5], f"{p}.k5t", perm=[0, 1, 2, 4, 3])
        scores = b.op("MatMul", [q5, k5t], f"{p}.scores")
        scores = b.op("Mul", [scores, inv_sqrt_d], f"{p}.scores_scaled")
        scores = b.op("Add", [scores, mask_c], f"{p}.scores_masked")
        attn = b.op("Softmax", [scores], f"{p}.softmax", axis=-1)
        out5 = b.op("MatMul", [attn, v5], f"{p}.attn_out5")

        out = reshape(out5, [batch_size, n_head, seq_len, head_dim], f"{p}.out_r")
        out = b.op("Transpose", [out], f"{p}.out_t", perm=[0, 2, 1, 3])
        out = reshape(out, [batch_size, seq_len, n_head * head_dim], f"{p}.out_flat")
        out = _linear(
            b,
            out,
            declare(f"{p}.self_attn.o_proj.weight"),
            declare_optional(f"{p}.self_attn.o_proj.bias") if attention_bias else None,
            f"{p}.o_proj",
        )
        x = b.op("Add", [resid, out], f"{p}.attn_resid")

        resid = x
        h = _rmsnorm(
            b, x, declare(f"{p}.post_attention_layernorm.weight"), eps, f"{p}.ffn_norm"
        )
        gate = _linear(
            b, h, declare(f"{p}.mlp.gate_proj.weight"), None, f"{p}.gate_proj"
        )
        up = _linear(b, h, declare(f"{p}.mlp.up_proj.weight"), None, f"{p}.up_proj")
        silu = b.op("Sigmoid", [gate], f"{p}.silu_sig")
        silu = b.op("Mul", [gate, silu], f"{p}.silu")
        act = b.op("Mul", [silu, up], f"{p}.act")
        ffn_out = _linear(
            b, act, declare(f"{p}.mlp.down_proj.weight"), None, f"{p}.down_proj"
        )
        x = b.op("Add", [resid, ffn_out], f"{p}.ffn_resid")

    x = _rmsnorm(b, x, declare("model.norm.weight"), eps, "output_norm")

    if not tie_word_embeddings and "lm_head.weight" in entries:
        lm_head = declare("lm_head.weight")
    else:
        # Tied embeddings: reuse token_embd (already declared above), same
        # fallback gguf_reconstruct._reconstruct_llama_family uses.
        lm_head = token_embd
    logits = _linear(b, x, lm_head, None, "lm_head")

    return onnx.helper.make_graph(
        b.nodes,
        f"hf_{model_type}",
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info(
                logits, onnx.TensorProto.FLOAT, [batch_size, seq_len, vocab_size]
            )
        ],
        initializer=b.initializers,
    )


def reconstruct_hf_graph(
    hf_dir: str, batch_size: int = 1, seq_len: int = 8
) -> onnx.ModelProto:
    """Build a runnable ONNX graph -- structure *and* weights -- directly
    from a HuggingFace checkpoint directory, for a recognized architecture
    (``model_type`` "llama", "mistral", "qwen2", or "qwen3" -- see this
    module's docstring).

    :param hf_dir: path to the checkpoint directory (containing
            ``config.json`` and either ``model.safetensors`` or
            ``model.safetensors.index.json`` + shards)
    :param batch_size: static batch dimension baked into the returned
            graph's input/output shapes (not a dynamic axis -- see this
            module's docstring's scope note)
    :param seq_len: static sequence-length dimension, likewise baked in
    :returns: the constructed, hydrated model (inputs ``input_ids``/
            ``position_ids``, both ``int64[batch_size, seq_len]``; output
            ``logits``, ``float32[batch_size, seq_len, vocab_size]``)
    :raises UnsupportedArchitectureError: if ``model_type`` is not one this
            builder has a template for, a required tensor is missing, or
            the checkpoint is MoE
    """
    config = read_hf_config(hf_dir)
    model_type = config.get("model_type")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise UnsupportedArchitectureError(
            f"model_type={model_type!r} has no graph template here -- "
            f"supported: {sorted(_SUPPORTED_MODEL_TYPES)}"
        )
    entries = _index_safetensors_checkpoint(hf_dir)
    graph = _reconstruct_llama_family_hf(config, entries, batch_size, seq_len)
    return onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )
