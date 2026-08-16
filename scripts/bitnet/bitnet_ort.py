#!/usr/bin/env python3
"""Rewrite the ternary (BitNet b1.58) linear layers of an ONNX model into
onnxruntime's native low-bit kernels.

BitNet b1.58 (https://github.com/microsoft/BitNet) replaces every projection in
a transformer decoder with a ``BitLinear`` whose weight is *ternary*: each
element is one of ``{-s, 0, +s}`` for a single per-tensor scale ``s`` (the
absmean quantizer). An ONNX export of such a model still stores those weights as
dense float32 initializers, so the graph is 16x larger than it needs to be and
runs on the generic float ``MatMul`` kernel.

This module finds those weights -- structurally, by testing each ``MatMul``
initializer for ternary values, so it works on any BitNet export regardless of
how it was produced -- and rewrites them into one of two onnxruntime-native
forms:

``nbits`` (default)
    ``com.microsoft::MatMulNBits`` with ``bits=2``. Ternary maps exactly onto
    2-bit codes (``{0,1,2}`` with zero-point ``1``), so the weight
    *representation* is lossless while shrinking 16x. Accuracy then depends only
    on ``accuracy_level``: level 1 reproduces the float model, level 4 (the
    default) computes in int8, which is BitNet's own W1.58A8 arithmetic and the
    only 2-bit path onnxruntime has a fast kernel for.

``int8``
    The same W1.58A8 arithmetic written out explicitly: per-token absmax int8
    activation quantization feeding ``MatMulInteger`` (int8 x int8 -> int32),
    rescaled by ``a_scale * w_scale``. Uses **only standard ONNX operators**, so
    it targets runtimes without contrib ops -- but on onnxruntime's CPU provider
    it is slower than float, so prefer ``nbits`` there.

Neither form needs a custom op or a BitNet runtime. See ``convert_bitnet.py``
for the command-line driver and ``README.md`` for the measured results.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnx.shape_inference
from onnx import TensorProto, helper, numpy_helper

# onnxruntime's MatMulNBits kernel only accepts these block sizes.
_NBITS_BLOCK_SIZES = (16, 32, 64, 128, 256)
# ...but only some of them have an int8-compute kernel for 2-bit weights. With
# block_size 16 or 256 onnxruntime silently falls back to the unvectorized float
# path, which is ~300x slower than the int8 one (and ~40x slower than a plain
# float MatMul). Auto-selection therefore prefers 128 and never picks those two;
# they are still honoured when requested explicitly. Measured on 2560x6912,
# single-threaded, one token: 128 -> 8.2x vs float, 64 -> 5.7x, 32 -> 2.7x.
_PREFERRED_BLOCK_SIZES = (128, 64, 32)
# Ternary codes are stored as {0, 1, 2} with a zero-point of 1, so
# ``(code - zero_point)`` recovers {-1, 0, +1}. Note onnxruntime's *default*
# zero-point for 2-bit is 2, so the zero_points input must always be supplied.
_TERNARY_ZERO_POINT = 1
# BitNet's activation quantizer clamps the per-token absmax before dividing
# (see ``activation_quant`` in the reference implementation).
_ACT_ABSMAX_FLOOR = 1e-5

# MatMulNBits' ``accuracy_level`` selects the compute type: 0 leaves it at the
# input type (float32), 4 quantizes activations to int8. For 2-bit weights the
# float path has no vectorized kernel and runs ~35x slower than a plain float
# MatMul, while the int8 path hits MLAS' blocked kernel and runs ~8x *faster*.
# Level 4 is also exactly BitNet's own W1.58A8 arithmetic, so it is the default.
ACCURACY_LEVEL_FP32 = 1
ACCURACY_LEVEL_INT8 = 4

MODES = ("nbits", "int8")


@dataclasses.dataclass
class TernaryWeight:
    """A ``[K, N]`` weight matrix decomposed into ternary codes and scales."""

    codes: np.ndarray  # int8 [K, N], values in {-1, 0, 1}
    scales: np.ndarray  # float32 [N], per output channel

    @property
    def shape(self) -> Tuple[int, int]:
        return self.codes.shape  # type: ignore[return-value]

    @property
    def per_tensor(self) -> bool:
        return bool(np.all(self.scales == self.scales[0]))


@dataclasses.dataclass
class ConversionStats:
    converted: int = 0
    skipped: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    weight_bytes_before: int = 0
    weight_bytes_after: int = 0

    @property
    def compression(self) -> float:
        if not self.weight_bytes_after:
            return 0.0
        return self.weight_bytes_before / self.weight_bytes_after


def as_ternary(weight: np.ndarray, rtol: float = 1e-5) -> Optional[TernaryWeight]:
    """Decompose ``weight`` into ternary codes x per-channel scales, or ``None``.

    A column is ternary when every element is within ``rtol`` (relative to the
    column's largest magnitude) of ``{-s, 0, +s}``. An all-zero column is
    ternary with an arbitrary scale, but a matrix that is *entirely* zero is
    rejected: it carries no BitNet signal and rewriting it only adds nodes.
    """
    if weight.ndim != 2 or weight.dtype != np.float32:
        return None
    scales = np.abs(weight).max(axis=0)
    if not np.any(scales > 0):
        return None
    safe = np.where(scales > 0, scales, 1.0).astype(np.float32)
    normalized = weight / safe
    codes = np.rint(normalized)
    if np.abs(codes).max() > 1:
        return None
    if not np.all(np.abs(normalized - codes) <= rtol):
        return None
    return TernaryWeight(codes.astype(np.int8), safe)


def _pack_2bit(values: np.ndarray) -> np.ndarray:
    """Pack uint8 codes along the last axis, 4 per byte, low bits first."""
    count = values.shape[-1]
    padded_len = (count + 3) // 4 * 4
    padded = np.zeros(values.shape[:-1] + (padded_len,), np.uint8)
    padded[..., :count] = values
    packed = np.zeros(values.shape[:-1] + (padded_len // 4,), np.uint8)
    for i in range(4):
        packed |= (padded[..., i::4] << (2 * i)).astype(np.uint8)
    return packed


def _choose_block_size(k: int) -> Optional[int]:
    for block in _PREFERRED_BLOCK_SIZES:
        if block <= k:
            return block
    return None


def pack_matmul_nbits(
    ternary: TernaryWeight, block_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(B, scales, zero_points)`` initializers for MatMulNBits.

    ``B`` is ``[N, n_blocks, block_size/4]`` uint8, ``scales`` is a flat
    ``[N * n_blocks]`` float32 laid out N-major, and ``zero_points`` is the
    2-bit packing of a constant 1 per block.
    """
    k, n = ternary.shape
    n_blocks = (k + block_size - 1) // block_size
    # Pad the tail block with the zero-point so padded lanes dequantize to 0.
    padded = np.full((n_blocks * block_size, n), _TERNARY_ZERO_POINT, np.uint8)
    padded[:k] = (ternary.codes + _TERNARY_ZERO_POINT).astype(np.uint8)
    # [K, N] -> [N, n_blocks, block_size] -> packed
    blocks = padded.T.reshape(n, n_blocks, block_size)
    packed = _pack_2bit(blocks)
    scales = np.repeat(ternary.scales.astype(np.float32), n_blocks)
    zero_points = _pack_2bit(np.full((n, n_blocks), _TERNARY_ZERO_POINT, np.uint8))
    return packed, scales, zero_points


class _GraphEditor:
    """Name allocation and initializer bookkeeping for one graph rewrite."""

    def __init__(self, graph: onnx.GraphProto) -> None:
        self.graph = graph
        self.initializers = {init.name: init for init in graph.initializer}
        self._used = {init.name for init in graph.initializer}
        self._used |= {out for node in graph.node for out in node.output}
        self._used |= {inp.name for inp in graph.input}

    def unique(self, base: str) -> str:
        name = base
        i = 0
        while name in self._used:
            i += 1
            name = f"{base}_{i}"
        self._used.add(name)
        return name

    def add_initializer(self, array: np.ndarray, base: str) -> str:
        name = self.unique(base)
        self.graph.initializer.append(numpy_helper.from_array(array, name))
        return name

    def drop_unused_initializers(self) -> None:
        used = {inp for node in self.graph.node for inp in node.input}
        used |= {out.name for out in self.graph.output}
        dropped = {init.name for init in self.graph.initializer} - used
        keep = [init for init in self.graph.initializer if init.name in used]
        del self.graph.initializer[:]
        self.graph.initializer.extend(keep)
        keep_vi = [vi for vi in self.graph.value_info if vi.name not in dropped]
        del self.graph.value_info[:]
        self.graph.value_info.extend(keep_vi)


def _nbits_nodes(
    editor: _GraphEditor,
    node: onnx.NodeProto,
    ternary: TernaryWeight,
    block_size: int,
    accuracy_level: int,
) -> List[onnx.NodeProto]:
    packed, scales, zero_points = pack_matmul_nbits(ternary, block_size)
    k, n = ternary.shape
    prefix = node.name or f"{node.op_type}_{node.output[0]}"
    return [
        helper.make_node(
            "MatMulNBits",
            [
                node.input[0],
                editor.add_initializer(packed, f"{prefix}_q2"),
                editor.add_initializer(scales, f"{prefix}_scales"),
                editor.add_initializer(zero_points, f"{prefix}_zp"),
            ],
            [node.output[0]],
            name=editor.unique(f"{prefix}_MatMulNBits"),
            domain="com.microsoft",
            K=k,
            N=n,
            bits=2,
            block_size=block_size,
            accuracy_level=accuracy_level,
        )
    ]


def _int8_nodes(
    editor: _GraphEditor, node: onnx.NodeProto, ternary: TernaryWeight, opset: int
) -> List[onnx.NodeProto]:
    x = node.input[0]
    prefix = node.name or f"{node.op_type}_{node.output[0]}"
    v = lambda suffix: editor.unique(f"{prefix}_{suffix}")  # noqa: E731

    abs_x, amax, amax_c = v("abs"), v("amax"), v("amax_clip")
    a_scale, scaled, rounded, clipped, q = (
        v("a_scale"),
        v("scaled"),
        v("round"),
        v("clip"),
        v("q"),
    )
    acc, acc_f, out_scale = v("acc"), v("acc_f"), v("out_scale")

    axes = editor.add_initializer(np.array([-1], np.int64), f"{prefix}_axes")
    nodes = [helper.make_node("Abs", [x], [abs_x])]
    if opset >= 18:
        nodes.append(helper.make_node("ReduceMax", [abs_x, axes], [amax], keepdims=1))
    else:
        nodes.append(
            helper.make_node("ReduceMax", [abs_x], [amax], axes=[-1], keepdims=1)
        )
    nodes += [
        helper.make_node(
            "Clip",
            [
                amax,
                editor.add_initializer(
                    np.float32(_ACT_ABSMAX_FLOOR), f"{prefix}_floor"
                ),
            ],
            [amax_c],
        ),
        helper.make_node(
            "Div",
            [amax_c, editor.add_initializer(np.float32(127.0), f"{prefix}_127")],
            [a_scale],
        ),
        helper.make_node("Div", [x, a_scale], [scaled]),
        helper.make_node("Round", [scaled], [rounded]),
        helper.make_node(
            "Clip",
            [
                rounded,
                editor.add_initializer(np.float32(-128.0), f"{prefix}_qmin"),
                editor.add_initializer(np.float32(127.0), f"{prefix}_qmax"),
            ],
            [clipped],
        ),
        helper.make_node("Cast", [clipped], [q], to=TensorProto.INT8),
        helper.make_node(
            "MatMulInteger",
            [q, editor.add_initializer(ternary.codes, f"{prefix}_ternary")],
            [acc],
            name=editor.unique(f"{prefix}_MatMulInteger"),
        ),
        helper.make_node("Cast", [acc], [acc_f], to=TensorProto.FLOAT),
        helper.make_node(
            "Mul",
            [a_scale, editor.add_initializer(ternary.scales, f"{prefix}_w_scale")],
            [out_scale],
        ),
        helper.make_node("Mul", [acc_f, out_scale], [node.output[0]]),
    ]
    return nodes


def _default_opset(model: onnx.ModelProto) -> int:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    return 0


def _ensure_ms_domain(model: onnx.ModelProto) -> None:
    if any(imp.domain == "com.microsoft" for imp in model.opset_import):
        return
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def convert(
    model: onnx.ModelProto,
    mode: str = "nbits",
    block_size: Optional[int] = None,
    accuracy_level: int = ACCURACY_LEVEL_INT8,
    rtol: float = 1e-5,
    only: Optional[Sequence[str]] = None,
) -> Tuple[onnx.ModelProto, ConversionStats]:
    """Rewrite every ternary ``MatMul`` in ``model`` into an onnxruntime kernel.

    ``accuracy_level`` (``nbits`` only) selects MatMulNBits' compute type. The
    default 4 (int8) is both BitNet's own W1.58A8 arithmetic and the only path
    with an optimized 2-bit kernel -- see ``ACCURACY_LEVEL_INT8``.

    ``only`` restricts the rewrite to the named nodes (useful for bisecting a
    conversion). Nodes that cannot be converted are left untouched and recorded
    in ``ConversionStats.skipped``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    converted = onnx.ModelProto()
    converted.CopyFrom(model)
    graph = converted.graph
    editor = _GraphEditor(graph)
    opset = _default_opset(converted)
    stats = ConversionStats()
    ternary_cache: Dict[str, Optional[TernaryWeight]] = {}

    new_nodes: List[onnx.NodeProto] = []
    for node in graph.node:
        replacement = None
        skip_reason = _unconvertible_reason(node, editor, only)
        if skip_reason is None:
            weight_name = node.input[1]
            if weight_name not in ternary_cache:
                array = numpy_helper.to_array(editor.initializers[weight_name])
                ternary_cache[weight_name] = as_ternary(array, rtol=rtol)
            ternary = ternary_cache[weight_name]
            if ternary is None:
                skip_reason = "weight is not ternary"
            elif mode == "nbits":
                block = block_size or _choose_block_size(ternary.shape[0])
                if block is None:
                    skip_reason = (
                        f"K={ternary.shape[0]} is below the minimum block size 32"
                    )
                elif block not in _NBITS_BLOCK_SIZES:
                    raise ValueError(
                        f"block_size must be one of {_NBITS_BLOCK_SIZES}, got {block}"
                    )
                else:
                    replacement = _nbits_nodes(
                        editor, node, ternary, block, accuracy_level
                    )
            else:
                replacement = _int8_nodes(editor, node, ternary, opset)

        if replacement is None:
            new_nodes.append(node)
            if skip_reason is not None and _is_initializer_matmul(node, editor):
                stats.skipped.append((node.name or node.output[0], skip_reason))
            continue

        stats.weight_bytes_before += editor.initializers[node.input[1]].ByteSize()
        stats.weight_bytes_after += _replacement_weight_bytes(graph, replacement)
        stats.converted += 1
        new_nodes.extend(replacement)

    del graph.node[:]
    graph.node.extend(new_nodes)
    editor.drop_unused_initializers()
    if stats.converted:
        if mode == "nbits":
            _ensure_ms_domain(converted)
        # The rewrite replaces float MatMul outputs with the same names but
        # different producers; stale shape annotations would now be unchecked.
        converted = onnx.shape_inference.infer_shapes(converted, strict_mode=False)
    return converted, stats


def _is_initializer_matmul(node: onnx.NodeProto, editor: _GraphEditor) -> bool:
    return (
        node.op_type == "MatMul"
        and len(node.input) == 2
        and node.input[1] in editor.initializers
    )


def _unconvertible_reason(
    node: onnx.NodeProto, editor: _GraphEditor, only: Optional[Sequence[str]]
) -> Optional[str]:
    """``None`` when ``node`` is a rewrite candidate, else why it is not."""
    if node.op_type != "MatMul":
        return "not a MatMul"
    if len(node.input) != 2 or node.input[1] not in editor.initializers:
        return "weight is not a graph initializer"
    if only is not None and node.name not in only and node.output[0] not in only:
        return "excluded by --only"
    return None


def _replacement_weight_bytes(
    graph: onnx.GraphProto, replacement: Sequence[onnx.NodeProto]
) -> int:
    """Serialized size of the initializers the replacement nodes introduced."""
    produced = {inp for node in replacement for inp in node.input}
    return sum(init.ByteSize() for init in graph.initializer if init.name in produced)


def infer_vocab_size(model: onnx.ModelProto) -> Optional[int]:
    """Vocabulary size read off the token-embedding ``Gather``, if there is one."""
    initializers = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Gather" or node.input[0] not in initializers:
            continue
        table = initializers[node.input[0]]
        if len(table.dims) == 2 and table.data_type == TensorProto.FLOAT:
            return int(table.dims[0])
    return None


def random_feeds(
    model: onnx.ModelProto,
    batch: int = 1,
    seq: int = 32,
    vocab: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Random inputs for ``model``, resolving dynamic dims to ``batch``/``seq``.

    Integer inputs are treated as token ids and kept inside the vocabulary, so
    the embedding ``Gather`` stays in range.
    """
    rng = np.random.RandomState(seed)
    if vocab is None:
        vocab = infer_vocab_size(model) or 1024
    initializers = {init.name for init in model.graph.initializer}
    feeds: Dict[str, np.ndarray] = {}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        tensor = value.type.tensor_type
        shape = [
            dim.dim_value if dim.HasField("dim_value") else (batch if i == 0 else seq)
            for i, dim in enumerate(tensor.shape.dim)
        ]
        dtype = helper.tensor_dtype_to_np_dtype(tensor.elem_type)
        if np.issubdtype(dtype, np.integer):
            feeds[value.name] = rng.randint(0, vocab, size=shape).astype(dtype)
        elif dtype == np.bool_:
            feeds[value.name] = np.ones(shape, np.bool_)
        else:
            feeds[value.name] = rng.standard_normal(shape).astype(dtype)
    return feeds


def run(model: onnx.ModelProto, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Run ``model`` on onnxruntime's CPU provider."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, feeds)


def benchmark(
    model: onnx.ModelProto,
    feeds: Dict[str, np.ndarray],
    repeats: int = 20,
    threads: int = 1,
) -> float:
    """Mean onnxruntime wall-clock milliseconds per run, after warm-up.

    Single-threaded by default: low-bit matmul kernels are memory bound, and
    thread-count noise otherwise swamps the difference being measured.
    """
    import time

    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    for _ in range(max(3, repeats // 10)):
        session.run(None, feeds)
    start = time.perf_counter()
    for _ in range(repeats):
        session.run(None, feeds)
    return (time.perf_counter() - start) / repeats * 1000


def max_relative_error(
    reference: Sequence[np.ndarray], actual: Sequence[np.ndarray]
) -> float:
    """Largest ``|a - r| / max(|r|)`` across a set of output tensors."""
    worst = 0.0
    for ref, act in zip(reference, actual):
        scale = float(np.abs(ref).max())
        if scale == 0.0:
            scale = 1.0
        worst = max(worst, float(np.abs(act - ref).max()) / scale)
    return worst
