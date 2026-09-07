"""OCP Microscaling (MX) formats (Rouhani et al., 2023, "Microscaling Data
Formats for Deep Learning", https://arxiv.org/abs/2310.10537; standardized
by the Open Compute Project, "OCP Microscaling Formats (MX) Specification
v1.0"). onnxsim ports the format's own definition, not any framework's
code -- MX formats are a data representation, not a fitting algorithm, so
there is no reference "implementation" to diverge from the way there is
for e.g. GPTQ's column-update order.

Every INT4 scheme already in onnxsim (`quantize_weight_only_int4` and
everything built on it) and :mod:`onnxsim.nf4` share one property: the
per-block **scale** is an ordinary float32 value, fit to that block's own
data (an absmax ratio). MX formats make a different, narrower choice for
the scale specifically: it must be a **pure power of two** -- stored as
an 8-bit field with no mantissa at all (called "E8M0": an 8-bit exponent,
zero mantissa bits), rather than an arbitrary float32. This is the actual
definition of "microscaling": a block's dynamic range is captured by a
single exponent, so applying the scale is an exponent add, not a real
multiply -- the reason MX formats are landing in hardware natively
(NVIDIA Blackwell, AMD CDNA3/MI300, Intel Gaudi3, per the OCP consortium
spec this module implements), unlike a per-block float32 scale, which
still needs an actual multiplier.

MXFP4 pairs that power-of-two block scale with **E2M1** elements: 1 sign
bit, 2 exponent bits, 1 mantissa bit, evaluating to a small, fixed set of
magnitudes -- ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` (the two smallest,
``0``/``0.5``, are E2M1's own subnormals; the rest follow the usual
``(1 + mantissa/2) * 2^(exponent - 1)`` formula with exponent bias 1).
Unlike :mod:`onnxsim.nf4`'s codebook (16 values fit to a standard normal
distribution's own quantile points), MXFP4's codebook is not
data-derived at all -- it is exactly what a 4-bit IEEE-754-style
floating-point format's own bit patterns evaluate to, fixed by the format
definition, identical for every tensor.

ONNX has no native MX tensor type (as of this writing), so -- following
the exact same approach :mod:`onnxsim.nf4` already uses for its own
non-uniform codebook, since neither has a standard affine
``DequantizeLinear`` representation -- this module builds the
dequantization out of ordinary ONNX ops any opset-11+ runtime already
supports: ``Gather`` the (per-element) 4-bit code out of a 16-entry
constant codebook tensor, then ``Mul`` by the per-block power-of-two
scale (computed in this module as an ordinary float32 value equal to
``2^shared_exponent`` -- the E8M0 *value* the format specifies, not its
raw 8-bit exponent-field encoding, since ONNX has no E8M0 tensor type to
store that encoding in either). Reconstruction is therefore numerically
exact to what a real MXFP4 dequantize produces; only the on-disk *bit*
representation (E8M0's packed exponent byte, E2M1's packed nibble) isn't
reproduced -- the same simplification :mod:`onnxsim.nf4` already makes for
its own codes (stored one byte per element, unpacked) and
:mod:`onnxsim.quip_sharp` makes for its E8-lattice codes (stored as plain
INT4, not the paper's own curated 2-bit/weight codebook indices).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name

# E2M1's own 16 bit patterns, evaluated per the format definition (bias 1,
# subnormal exponent field 0): magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6},
# signed. Fixed by the OCP MX spec -- not fit to any data, unlike
# onnxsim.nf4's own codebook.
MXFP4_CODEBOOK: List[float] = [
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]

# The largest magnitude E2M1 can represent (6.0 == 1.5 * 2^2) -- used to
# pick each block's own power-of-two shared scale so the block's own
# largest-magnitude element lands within E2M1's representable range.
_MXFP4_MAX_MAGNITUDE = 6.0

# OCP MX spec's own canonical block size for every MX format.
MX_BLOCK_SIZE = 32


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(w_name, weight_transposed)`` or ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[1], False
    if node.op_type == "Gemm":
        num_inputs = len(node.input)
        if num_inputs not in (2, 3):
            return None
        trans_a = attrs.get("transA")
        if trans_a is not None and trans_a.i != 0:
            return None
        alpha = attrs.get("alpha")
        if alpha is not None and alpha.f != 1.0:
            return None
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
        trans_b = attrs.get("transB")
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[1], weight_transposed
    return None


def _nearest_codebook_index(normalized: np.ndarray) -> np.ndarray:
    codebook = np.asarray(MXFP4_CODEBOOK, dtype=np.float64)
    diffs = np.abs(normalized[..., np.newaxis] - codebook)
    return np.argmin(diffs, axis=-1).astype(np.uint8)


def _quantize_mxfp4_blockwise(
    w_nk: np.ndarray, block_size: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Returns ``(codes_nk, scale_blocks)`` for ``w_nk`` ([N, K], output
    channel first): MXFP4 codebook indices in ``[0, 15]`` and one
    power-of-two scale per ``(output channel, block-of-K)`` group, shape
    ``[N, K // block_size]``. Assumes ``K % block_size == 0``.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    max_abs = np.maximum(np.abs(blocks).max(axis=2), 1e-30)  # [N, num_blocks]
    # The smallest power-of-two scale that keeps the block's own largest
    # magnitude within E2M1's representable range (max 6.0): using
    # floor(log2(max_abs)) - 2 instead (a block's own binade) can let
    # max_abs / scale land up to 8.0, past the codebook's actual max, so
    # the top of every block would silently clip to 6.0 with a much
    # larger error than the codebook's own gaps -- this ceil-based choice
    # is exact: 2^shared_exponent >= max_abs / 6.0, so max_abs / scale is
    # always <= 6.0.
    shared_exponent = np.ceil(np.log2(max_abs / _MXFP4_MAX_MAGNITUDE))
    scale = np.exp2(shared_exponent)  # a pure power of two -- E8M0's own value
    normalized = blocks / scale[:, :, np.newaxis]
    codes = _nearest_codebook_index(normalized)  # [N, num_blocks, block_size]
    return codes.reshape(n, k), scale


def _quantize_weight_only_mxfp4_impl(
    model: Union[str, onnx.ModelProto],
    block_size: int,
    skip_names: Optional[Iterable[str]],
    quantize_block: Callable[[np.ndarray, int], "tuple[np.ndarray, np.ndarray]"],
) -> onnx.ModelProto:
    """Shared driver behind :func:`quantize_weight_only_mxfp4` and
    :func:`onnxsim.focus_fp4.quantize_weight_only_mxfp4_focus`.

    Matches every eligible layer, calls ``quantize_block(w_nk, block_size)``
    on that layer's ``[N, K]`` (output-channel-first) weight -- which must
    return ``(codes_nk, scale_blocks)`` with exactly the shapes and dtypes
    :func:`_quantize_mxfp4_blockwise` returns -- and emits the MXFP4
    dequantization subgraph and initializers for the result.

    Every byte of the emitted graph (initializer *names*, shapes, dtypes,
    node types, node order) is fixed here rather than by the caller's
    fitting choice, so any ``quantize_block`` that keeps the same
    power-of-two scale produces a model differing from plain MXFP4's only
    in the ``*_mxfp4_q`` code values -- the structural guarantee
    :mod:`onnxsim.focus_fp4` relies on.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    codebook_name = None  # created lazily on first match

    nodes = list(graph.node)
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        w_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue

        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape
        if k % block_size != 0:
            continue

        if codebook_name is None:
            codebook_name = _unique_name("mxfp4_codebook", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(MXFP4_CODEBOOK, dtype=np.float32), name=codebook_name
                )
            )
        num_blocks = k // block_size

        codes_nk, scale_blocks = quantize_block(w_nk, block_size)
        codes_orig = codes_nk if weight_transposed else codes_nk.T
        scale_orig = scale_blocks if weight_transposed else scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)

        wq = onnx.numpy_helper.from_array(
            codes_orig.astype(np.uint8),
            name=_unique_name(f"{w_name}_mxfp4_q", taken_names),
        )
        graph.initializer.append(wq)
        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_mxfp4_scale", taken_names),
        )
        graph.initializer.append(ws)

        if weight_transposed:
            blocked_shape = [n, num_blocks, block_size]
            scale_shape = [n, num_blocks, 1]
        else:
            blocked_shape = [num_blocks, block_size, n]
            scale_shape = [num_blocks, 1, n]

        cast_out = _unique_name(f"{w_name}_mxfp4_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [wq.name], [cast_out], to=onnx.TensorProto.INT64
        )

        gather_out = _unique_name(f"{w_name}_mxfp4_gathered", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather", [codebook_name, cast_out], [gather_out], axis=0
        )

        blocked_shape_name = _unique_name(f"{w_name}_mxfp4_blocked_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(blocked_shape, dtype=np.int64), name=blocked_shape_name
            )
        )
        reshaped_out = _unique_name(f"{w_name}_mxfp4_reshaped", taken_names)
        reshape1_node = onnx.helper.make_node(
            "Reshape", [gather_out, blocked_shape_name], [reshaped_out]
        )

        scale_shape_name = _unique_name(f"{w_name}_mxfp4_scale_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(scale_shape, dtype=np.int64), name=scale_shape_name
            )
        )
        scale_reshaped_out = _unique_name(f"{w_name}_mxfp4_scale_reshaped", taken_names)
        reshape2_node = onnx.helper.make_node(
            "Reshape", [ws.name, scale_shape_name], [scale_reshaped_out]
        )

        scaled_out = _unique_name(f"{w_name}_mxfp4_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul", [reshaped_out, scale_reshaped_out], [scaled_out]
        )

        orig_shape_name = _unique_name(f"{w_name}_mxfp4_orig_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray([dim0, dim1], dtype=np.int64), name=orig_shape_name
            )
        )
        dq_out = _unique_name(f"{w_name}_mxfp4_dq", taken_names)
        reshape3_node = onnx.helper.make_node(
            "Reshape",
            [scaled_out, orig_shape_name],
            [dq_out],
            name=_unique_name(f"{w_name}_mxfp4_dequant", taken_names),
        )

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in (
            cast_node,
            gather_node,
            reshape1_node,
            reshape2_node,
            mul_node,
            reshape3_node,
        ):
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out


def quantize_weight_only_mxfp4(
    model: Union[str, onnx.ModelProto],
    block_size: int = MX_BLOCK_SIZE,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into the OCP MXFP4 format -- see this module's own
    docstring for the technique. Needs no calibration data: both the
    codebook and the per-block power-of-two scale come from the weight's
    own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale group
            along the reduction dimension; the OCP MX spec's own canonical
            choice is 32 for every MX format
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Reshape(Gather(codebook, Cast(Wq, INT64)), ...), Ws) ->
            Reshape(..., original shape)`` feeding the original MatMul/Gemm
            node -- ordinary ONNX ops only, no contrib op and no minimum
            opset beyond what ``Gather``/``Cast``/``Reshape``/``Mul``
            themselves need (opset 11+; unlike onnxsim's affine INT4
            schemes, this needs no opset-21 ``DequantizeLinear``
            ``block_size`` attribute since the codebook lookup is built
            from ordinary ops directly). Layers with a non-constant,
            non-2-D, or non-block-divisible weight are left untouched.
    """
    return _quantize_weight_only_mxfp4_impl(
        model, block_size, skip_names, _quantize_mxfp4_blockwise
    )
