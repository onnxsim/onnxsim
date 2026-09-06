"""llama.cpp's IQ4_NL -- the simplest of GGUF's "I-quant" family: a fixed,
non-uniform 16-entry lookup table ("codebook") used for 4-bit weight-only
quantization instead of a uniform, evenly-spaced 4-bit grid. Code ``c in
[0, 15]`` decodes to ``codebook[c] * block_scale``, with one scale per
32-element block -- quantizing means, per block: pick ``scale =
max(|block|) / max(|codebook|)``, then snap every element to whichever of
the 16 ``codebook[i] * scale`` values is numerically closest. Neural
network weights are typically close to normally distributed, and (the same
motivation as :mod:`onnxsim.nf4`'s own NF4 codebook) a *uniform* 4-bit grid
wastes resolution on the tails while under-resolving the dense region near
zero where most of the weight mass sits -- a fixed, distribution-matched,
non-uniform codebook fixes that without needing any per-tensor calibration.

**Codebook provenance -- an honesty note, per this repo's own established
norm (see e.g. :mod:`onnxsim.gguf_kquant`'s own docstring) against shipping
recalled/assumed numeric constants without verifying them
computationally.** Before writing this module, this repository's own GGUF
decoding code was searched for an existing, already-verified IQ4_NL
codebook to reuse: ``onnxsim/gguf_reconstruct.py`` and
``onnxsim/ggml_kquant.h`` (which decodes Q4_K and the rest of the K-quant
family Q4_K_M/Q4_K_S already import) were checked, along with a
whole-repository, case-insensitive grep for ``"IQ4_NL"`` and
``"kvalues_iq4nl"`` (llama.cpp's own name for its I-quant codebook
constant). **Neither turned up anything** -- this repository decodes
K-quants, not any I-quant format, and has no ``kvalues_iq4nl`` table
anywhere to transcribe from or cross-check against, unlike Q4_K's decoder
(which *was* available to verify against). llama.cpp's real IQ4_NL
codebook could only otherwise come from this module's own recalled/trained
knowledge of that constant -- exactly the kind of unverified numeric claim
this repo's own precedent (and the task this module was written under)
says not to ship.

So: **the 16 values below are NOT llama.cpp's own ``kvalues_iq4nl``
table.** They are this module's own, independently and *computationally*
derived 16-level non-uniform codebook: a generalized Lloyd-Max
(a.k.a. Lloyd's algorithm / 1-D k-means) scalar quantizer, run to
convergence against a standard normal distribution via numerical
quadrature over a fixed, dense grid (deterministic -- no random sampling
involved, so it is exactly reproducible, not just "a plausible-looking
number"). :func:`_lloyd_max_gaussian_codebook` below *is* that derivation
-- the algorithm, not a hardcoded magic array -- and
``IQ4_NL_CODEBOOK``'s 16 literal values are simply that function's own
output, computed once (Lloyd-Max convergence takes a few hundred
iterations over a 200001-point grid, a few seconds -- too slow to redo on
every import) and pasted in below; ``tests/test_iq4_nl.py`` re-invokes
:func:`_lloyd_max_gaussian_codebook` directly and checks it reproduces
these exact literals, so nothing here is an unverifiable claim. A
by-construction Lloyd-Max quantizer against a Gaussian source is *provably*
a locally-optimal (minimum mean squared error) 16-level non-uniform
codebook for that distribution, and empirically -- see this module's own
test file -- it comfortably beats a naive uniform 4-bit quantizer on
Gaussian-ish data, which is the entire point of a "non-linear" codebook in
the first place. It is very likely qualitatively similar in shape to
llama.cpp's real table (both are 16-level, symmetric-ish, denser near zero
codebooks fit to a roughly-Gaussian weight distribution) but is not claimed
to be, and should not be assumed to be, numerically identical to it.

Like :mod:`onnxsim.nf4`, :mod:`onnxsim.gguf_kquant`, and every other
weight-only PTQ module in this repo with a fixed or self-derived codebook,
no ONNX tensor type below INT4 exists to store a genuine 4-bit code
natively next to a non-uniform-codebook dequantization op (ONNX has no such
op at all), so -- following the exact pattern
:func:`onnxsim.apply_gguf_q4_k_quantization` already established for Q4_K
-- this module represents the result as a plain float32
quantize-dequantize round trip folded directly into a new initializer (no
new graph nodes), rather than :mod:`onnxsim.nf4`'s alternative of building
the codebook lookup out of ordinary ``Gather``/``Reshape``/``Mul`` ops.

**Scope**: matches ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`, shared with several other
onnxsim modules) and, optionally, ``Conv`` -- any constant float32 weight,
of any rank, is flattened, zero-padded up to a whole number of 32-element
blocks, quantized, and reshaped back; padding elements are quantized along
with the rest but discarded before reshaping, so they cannot affect the
tensor's real values (a padding element is always exactly 0, which can
never be a block's own largest-magnitude element unless the whole block is
already all-zero, so padding cannot even change a boundary block's own
chosen scale). No calibration data is needed -- like NF4 and Q4_K, every
quantization decision comes from the weight tensor's own values.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.llm_int8 import _match_matmul_like

_BLOCK_SIZE = 32
_NUM_LEVELS = 16


def _lloyd_max_gaussian_codebook(
    num_levels: int = _NUM_LEVELS,
    iterations: int = 500,
    grid_half_width: float = 8.0,
    grid_points: int = 200_001,
) -> List[float]:
    """Generalized Lloyd-Max (1-D k-means) scalar quantizer for a standard
    normal source, run to convergence over a fixed, dense grid -- see this
    module's own docstring for why this (not a recalled/transcribed
    constant) is this module's actual codebook derivation. Deterministic:
    no random sampling, so calling this twice always gives the same
    answer, up to grid/iteration-count settings.

    Each iteration alternates the two classic Lloyd-Max steps: (1) assign
    every grid point to its nearest current codebook level (the
    minimum-distortion partition for the current levels), then (2)
    recompute each level as the *Gaussian-density-weighted mean* of the
    grid points assigned to it (the minimum-MSE reconstruction value for
    the current partition -- for a fixed partition, the mean is the exact
    least-squares-optimal representative). Alternating these two provably
    monotonic-improvement steps converges to a local optimum; a Gaussian
    source's own symmetry means any reasonable symmetric initialization
    converges to the (essentially unique) global optimum in practice, which
    is also enforced explicitly below by symmetrizing the converged result.

    :returns: ``num_levels`` sorted float levels, exactly symmetric about
            0 (no level at exactly 0 for an even ``num_levels``), scaled so
            the largest-magnitude level is exactly ``1.0`` -- matching this
            module's own ``scale = max(|block|) / max(|codebook|)``
            convention.
    """
    grid = np.linspace(-grid_half_width, grid_half_width, grid_points)
    density = np.exp(-0.5 * grid**2)  # unnormalized N(0,1) density
    # A reasonable symmetric starting guess -- Lloyd-Max iteration corrects
    # this regardless of the exact starting spread.
    levels = np.linspace(-2.5, 2.5, num_levels)
    for _ in range(iterations):
        nearest = np.argmin(np.abs(grid[:, np.newaxis] - levels[np.newaxis, :]), axis=1)
        new_levels = levels.copy()
        for i in range(num_levels):
            mask = nearest == i
            if np.any(mask):
                new_levels[i] = np.average(grid[mask], weights=density[mask])
        new_levels.sort()
        converged = np.allclose(new_levels, levels, atol=1e-13)
        levels = new_levels
        if converged:
            break
    levels.sort()
    # Cancel residual grid-discretization asymmetry: a standard normal is
    # exactly symmetric, so the true optimum is too.
    symmetrized = 0.5 * (levels - levels[::-1])
    symmetrized = symmetrized / np.max(np.abs(symmetrized))
    return [float(v) for v in symmetrized]


# Computed once by calling _lloyd_max_gaussian_codebook() above (a few
# hundred Lloyd-Max iterations over a 200001-point grid, a few seconds --
# too slow to redo on every import of this module) -- see this module's own
# top-of-file docstring for the full provenance/honesty note.
# tests/test_iq4_nl.py re-derives this via _lloyd_max_gaussian_codebook()
# directly and checks it reproduces these exact literals.
IQ4_NL_CODEBOOK: List[float] = [
    -1.0,
    -0.757290980404998,
    -0.5923241681050287,
    -0.45994900549399614,
    -0.34508882475608954,
    -0.24055736084608642,
    -0.1421631738251121,
    -0.04704566832784618,
    0.04704566832784618,
    0.1421631738251121,
    0.24055736084608642,
    0.34508882475608954,
    0.45994900549399614,
    0.5923241681050287,
    0.757290980404998,
    1.0,
]


def _quantize_dequantize_blocks(blocks: np.ndarray) -> np.ndarray:
    """``blocks``: float64 ``[num_blocks, 32]``. Returns the same shape,
    each element replaced by its IQ4_NL quantize-dequantize round trip:
    ``codebook[nearest_code] * scale``, ``scale = max(|block|) /
    max(|codebook|)`` (floored to avoid a divide-by-zero on an all-zero
    block).
    """
    codebook = np.asarray(IQ4_NL_CODEBOOK, dtype=np.float64)
    max_abs_codebook = np.max(np.abs(codebook))
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / max_abs_codebook  # [B]
    normalized = blocks / scale[:, np.newaxis]  # bring into the codebook's own range
    # [B, 32, 16] absolute differences -> nearest codebook index per
    # element. Blocks are small (tens of thousands of elements at most for
    # any single layer), so a dense (B, 32, 16) distance tensor is cheap.
    diffs = np.abs(normalized[..., np.newaxis] - codebook)
    nearest = codebook[np.argmin(diffs, axis=-1)]  # [B, 32]
    return nearest * scale[:, np.newaxis]


def quantize_dequantize_iq4_nl(values: np.ndarray) -> np.ndarray:
    """IQ4_NL quantize-dequantize round trip over a flattened float array of
    any length -- padded with zeros up to a whole number of 32-element
    blocks before quantizing (the padding is quantized along with the real
    data but discarded before returning, so it cannot change any real
    value -- see this module's own docstring on why).

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, each element replaced by its IQ4_NL
            round trip, as float64
    """
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    padded_n = -(-n // _BLOCK_SIZE) * _BLOCK_SIZE
    if padded_n != n:
        flat = np.concatenate([flat, np.zeros(padded_n - n, dtype=np.float64)])
    blocks = flat.reshape(-1, _BLOCK_SIZE)
    dequantized = _quantize_dequantize_blocks(blocks).reshape(-1)
    return dequantized[:n].reshape(original_shape)


def apply_iq4_nl_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched ``MatMul``/"vanilla" ``Gemm``
    (and, optionally, ``Conv``) float32 weight into llama.cpp's IQ4_NL
    format -- see this module's own docstring for the technique and its
    codebook's own honestly-scoped provenance. Needs no calibration data:
    every quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input (any rank),
            not just ``MatMul``/``Gemm``'s 2-D one
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by its
            IQ4_NL quantize-dequantize round-tripped float32 version, same
            name and shape as the node's original weight input, stored
            under a *new* initializer name (the original initializer is
            left in the graph, unused). Layers with a non-constant or
            non-float32 weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    # MatMul/vanilla-Gemm's weight (via _match_matmul_like) and Conv's
    # weight are both always node.input[1] -- no separate index bookkeeping
    # needed.
    candidates: list[tuple[onnx.NodeProto, str]] = []
    for node in graph.node:
        w_name: Optional[str] = None
        match = _match_matmul_like(node)
        if match is not None:
            _x_name, matched_w_name, _bias_name, _weight_transposed = match
            w_name = matched_w_name
        elif include_conv and node.op_type == "Conv" and len(node.input) >= 2:
            w_name = node.input[1]
        if w_name is None or w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if w_init is None or w_init.data_type != onnx.TensorProto.FLOAT:
            continue
        candidates.append((node, w_name))

    quantized_names: dict[str, str] = {}
    for node, w_name in candidates:
        new_name = quantized_names.get(w_name)
        if new_name is None:
            w_init = initializer_map[w_name]
            w = onnx.numpy_helper.to_array(w_init)
            w_quant = quantize_dequantize_iq4_nl(w).astype(np.float32)

            new_name = _unique_name(f"{w_name}_iq4_nl", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
