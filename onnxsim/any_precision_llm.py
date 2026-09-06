"""Any-Precision LLM (Park, Hyun, Cho, Sung, Choi, 2024, ICML 2024,
"Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized
LLMs", https://arxiv.org/abs/2402.10517). onnxsim ports the algorithm, not
any framework's code, per the same rationale as :mod:`onnxsim.awq`/
:mod:`onnxsim.gptq` (the paper's own reference implementation quantizes
live PyTorch weights with a custom CUDA kernel serving the packed
multi-precision format, with no ONNX export path).

Every other weight-only quantizer in onnxsim picks **one bit-width** and
quantizes independently for it: asking for both a 4-bit and a 6-bit
version of the same model means running the quantizer twice, from
scratch, with no relationship between the two results beyond sharing the
same float source. Any-Precision LLM's own idea: quantize **once**, to the
*highest* bit-width ever needed, using a scheme where every lower
bit-width's own quantization is recoverable *exactly* by discarding the
highest-order... no, by discarding the *lowest*-order refinement bits of
that single quantization -- a **nested / incremental bit-plane**
construction, not independent per-bit-width quantization. Concretely, for
one value in one channel: start from a 1-bit code (which half of the
value's own local range it falls in), then repeatedly **bisect** -- at each
step, split every *existing* bin into two halves using that bin's own
member values only, appending one more bit to every code -- until reaching
the target maximum bit-width. Because a bin is only ever refined, never
merged or reassigned across an old bin boundary, the code built this way
has an exact algebraic nesting property: the ``b``-bit code, for *any*
``b`` up to the maximum, is recoverable from the max-bit-width code by a
plain integer right-shift (``code_at_max_bits >> (max_bits - b)``) -- one
quantization pass serves every precision level up to its own maximum, the
paper's own "any precision" deployment story (a single stored artifact,
truncate bit-planes for a cheaper/smaller version, keep them all for the
most accurate one).

This module's own version of the per-bin split rule (a good-faith
reproduction of the paper's own described "incremental upgrade" mechanism,
not a transcription of its own exact seed/split procedure, which this
module does not claim to reproduce exactly): each bisection step splits a
bin at the **midpoint of that bin's own current member values**
(``0.5 * (bin.min() + bin.max())``) -- simple, deterministic, and
sufficient to demonstrate the nesting property exactly (verified directly
in this module's own test file: the algebraic right-shift relationship
holds bit-for-bit, not just approximately). After the code tree is built
once (to ``max_bits``), materializing any single precision level ``bits``
is: right-shift the max-depth code down to a ``bits``-wide code, then
reconstruct each element as **the mean of its own bin's own original
values** -- the maximum-likelihood constant reconstruction for a fixed
partition. This module's first implementation instead fit one global
affine map (``value ~= code * scale + zero``) per block, by analogy with
every *uniform* quantizer elsewhere in this repo -- but this scheme's own
bins are not laid out on any fixed linear grid (each bisection's own split
point comes from that specific bin's own local data), so a straight-line
fit across all codes is not guaranteed to improve as bins get refined, and
empirically did not (caught by this module's own
reconstruction-improves-with-more-bits test, which failed under the
affine version and passes under the per-bin-mean version below). Per-bin
mean reconstruction, by contrast, *is* guaranteed monotonically
non-increasing in squared error as bins are refined further -- splitting a
bin can only reduce or preserve the variance it contributes (the law of
total variance), never increase it.

**What this module does not claim.** The paper's own reported gains are
about *deployment infrastructure* (one download serves many precisions,
with a real packed-bit-plane storage format and a custom serving kernel);
this module verifies the *nesting property* holds exactly, and that
reconstruction error at any single materialized precision is *competitive
with* (not necessarily better than) an ordinary independent quantizer at
that same precision -- being constrained to reuse a lower level's own bin
boundaries is a real constraint an independent quantizer doesn't have, so
this module does not claim nested quantization always wins on raw
accuracy at a fixed bit-width, only that it wins on not needing to
re-quantize from scratch for every precision level. As with several other
modules in this repo (:mod:`onnxsim.ibert_gelu`, :mod:`onnxsim.ibert_softmax`),
onnxsim has no lower-than-float32, arbitrary-bit-width ONNX tensor type to
express genuine packed sub-8-bit storage in (only INT4 and INT8 are
native) -- this module represents every materialized precision level as an
ordinary float32 quantize-dequantize round trip (the same simplification
:mod:`onnxsim.easyquant` already makes for its own W8A8 weights), not a
literal packed multi-bit-plane binary format.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.llm_int8 import _match_matmul_like


def _nested_bitplane_codes(values: np.ndarray, max_bits: int) -> np.ndarray:
    """Returns integer codes in ``[0, 2**max_bits - 1]`` for the 1-D array
    ``values``, built by repeated within-bin bisection (see this module's
    own docstring) -- with the exact property that, for every
    ``1 <= bits <= max_bits``, ``codes >> (max_bits - bits)`` is that
    ``bits``-wide nested quantization's own code.
    """
    codes = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(max_bits):
        new_codes = codes.copy()
        for bin_id in np.unique(codes):
            mask = codes == bin_id
            bin_values = values[mask]
            if bin_values.size <= 1:
                new_codes[mask] = codes[mask] * 2
                continue
            split = 0.5 * (float(bin_values.min()) + float(bin_values.max()))
            new_codes[mask] = codes[mask] * 2 + (bin_values >= split).astype(np.int64)
        codes = new_codes
    return codes


def _dequantize_by_bin_mean(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    # Reconstructs each element as the mean of its own bin's own original
    # values -- the maximum-likelihood constant reconstruction for a fixed
    # partition, and (by the law of total variance -- splitting a bin can
    # only ever reduce or preserve its own contributed squared error,
    # never increase it) the choice that makes reconstruction error
    # monotonically non-increasing as bins get refined. A single global
    # affine fit (`value ~= code * scale + zero`) does NOT have this
    # property here: this module's own bisection splits are chosen from
    # each bin's own local data, not laid out on any fixed linear grid, so
    # a straight-line fit across all codes can fit *worse* after a split
    # than before one (verified empirically -- an earlier version of this
    # module used exactly that affine fit and failed its own reconstruction-
    # improves-with-more-bits test).
    out = np.empty_like(values)
    for code in np.unique(codes):
        mask = codes == code
        out[mask] = values[mask].mean()
    return out


def _quantize_channel_nested(
    row: np.ndarray, bits: int, max_bits: int, block_size: int
) -> np.ndarray:
    k = row.shape[0]
    out = np.empty_like(row)
    for start in range(0, k, block_size):
        end = min(start + block_size, k)
        block = row[start:end].astype(np.float64)
        max_codes = _nested_bitplane_codes(block, max_bits)
        codes_b = max_codes >> (max_bits - bits)
        out[start:end] = _dequantize_by_bin_mean(block, codes_b)
    return out


def apply_any_precision_llm(
    float_model: Union[str, onnx.ModelProto],
    bits: int = 4,
    max_bits: int = 8,
    block_size: int = 32,
) -> onnx.ModelProto:
    """Weight-only quantizes every matched MatMul/"vanilla" Gemm layer to
    ``bits`` bits per element, using Any-Precision LLM's own nested
    bit-plane code construction (built once, to ``max_bits``, then
    right-shifted down to ``bits``) -- see this module's own docstring.
    Calling this again with a *different* ``bits`` (holding ``max_bits``
    and ``block_size`` fixed) reuses the exact same underlying code tree
    for every shared block, the paper's own "quantize once, deploy at any
    precision up to ``max_bits``" property.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param bits: the precision level to materialize, ``1 <= bits <=
            max_bits``
    :param max_bits: the highest bit-width the nested code tree is built
            to -- the ceiling every ``bits`` value shares a common code
            tree with
    :param block_size: elements per (output-channel, K-block) affine fit,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default block size convention
    :returns: ``float_model`` with every matched layer's weight replaced
            by its ``bits``-bit nested-quantization float32
            quantize-dequantize round trip. Layers with a non-constant,
            non-2-D weight are left untouched.
    """
    if not (1 <= bits <= max_bits):
        raise ValueError(f"bits ({bits}) must be in [1, max_bits={max_bits}]")
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)

    initializer_map = {t.name: t for t in float_model.graph.initializer}
    candidates = []  # (w_init, weight_transposed)
    for node in float_model.graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        _x_name, w_name, _bias_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((w_init, weight_transposed))
    if not candidates:
        return float_model

    out = onnx.ModelProto()
    out.CopyFrom(float_model)
    out_initializer_map = {t.name: t for t in out.graph.initializer}

    for w_init, weight_transposed in candidates:
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output-channel first

        quant_nk = np.empty_like(w_nk)
        for i in range(w_nk.shape[0]):
            quant_nk[i, :] = _quantize_channel_nested(
                w_nk[i, :], bits, max_bits, block_size
            )
        quant = quant_nk if weight_transposed else quant_nk.T

        out_init = out_initializer_map[w_init.name]
        out_init.CopyFrom(
            onnx.numpy_helper.from_array(quant.astype(np.float32), name=w_init.name)
        )

    return out
