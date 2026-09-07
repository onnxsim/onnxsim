"""FOCUS's **CRS** (Coupled-Relaxation Scaling) and **DGS** (Dual-Granularity
Scaling) for MXFP4, as a data-free per-block search.

**Where the idea comes from.** FOCUS is Tencent AngelSlim's FP4
scale-optimization method; the description this module is written against is
AngelSlim's own documentation for it (``docs/source/features/quantization/
focus_fp4.md``, "FOCUS: end-to-end scale optimization for FP4"). Its
observation is about an asymmetry in how a block-scaled FP4 format is
actually deployed: ordinarily one scale does both jobs,

    W_bar_i = Q_E2M1(W_i / S_i)          (quantize, offline)
    W_hat_i = W_bar_i * S_i              (dequantize, on the accelerator)

but the *quantization* scale only ever exists offline, while generating the
FP4 codes -- it is never stored in the deployed model, so nothing about it
has to satisfy the hardware's scale format. Only the *dequantization* scale
is constrained (E8M0, a pure power of two, for MXFP4; E4M3 for NVFP4). FOCUS
widens the optimization space along two axes that exploit that:

- **CRS**: give each block a full-precision coefficient ``c_i`` that
  decouples the two scales -- ``S_i^q = S_i^dq * c_i``, codes
  ``W_bar_i = Q_E2M1(W_i / S_i^q)``, reconstruction
  ``W_hat_i = W_bar_i * S_i^dq``. ``S_i^dq`` still satisfies the hardware
  constraint; ``c_i`` participates only in the offline fit and **is
  discarded at export**.
- **DGS**: split each hardware block into smaller sub-blocks (AngelSlim's
  own choice: 8 elements) and give each sub-block its own coefficient
  ``c_i^k``. The dequantization scale stays at the original hardware
  granularity, so the deployed format is again unchanged.

**What is and is not ported here -- read this before comparing against the
paper's numbers.** FOCUS optimizes ``c_i`` with an *end-to-end trained*
procedure (AngelSlim's documentation ships training configs and a multi-GPU
setup for it). onnxsim is stateless PTQ graph rewriting with no training
loop, so **that part is deliberately not ported**. What is ported is the
*mechanism* -- the decoupled quantization scale and its two granularities --
fitted instead by a **data-free per-block grid search** over ``c_i``
against that block's own local reconstruction error, the same shape
:func:`onnxsim.llm_fp4._search_llm_fp4_blockwise` already uses for its
``(FP4 format, clip ratio)`` search and :mod:`onnxsim.adaround`/
:mod:`onnxsim.omniquant` use for their own per-layer auxiliary quantities.
**FOCUS's reported accuracy is therefore neither claimed nor reproduced
here**: a grid search against a local objective is a strictly weaker fitting
procedure than the paper's own trained one, and this module has not been
evaluated on any of the paper's benchmarks. What *is* guaranteed is the
structural property CRS/DGS rest on, and it is guaranteed by construction
rather than by measurement -- see below.

**The structural guarantee (this is what is actually claimed).** The stored
dequantization scale is not merely "also a power of two": it is the
*identical* array :func:`onnxsim.mx_quantization._quantize_mxfp4_blockwise`
itself computes, taken from that function directly, and the graph is emitted
by the very same driver
(:func:`onnxsim.mx_quantization._quantize_weight_only_mxfp4_impl`) that
:func:`onnxsim.mx_quantization.quantize_weight_only_mxfp4` uses. Initializer
names, shapes, dtypes, node types and node order are therefore identical to
the plain MXFP4 path by construction; a FOCUS-quantized model differs from a
plain-MXFP4-quantized one *only* in the ``*_mxfp4_q`` code bytes. Nothing
downstream -- a runtime, an exporter, a checker -- can tell the two apart by
format.

**Why the objective is not plain element-wise MSE (an honesty note, in the
style of :mod:`onnxsim.iq4_nl`'s own "Codebook provenance" section).** With
the dequantization scale ``S^dq`` held fixed, the reachable reconstruction
values for an element are exactly ``{codebook[j] * S^dq}``, and the choice
that minimizes that element's *squared error* is, by definition, the nearest
one -- which is precisely what plain
:func:`onnxsim.mx_quantization.quantize_weight_only_mxfp4` already does
(``argmin`` over the codebook of ``|W / S^dq - codebook|``). So a search
over ``c_i`` judged by plain, unweighted, element-wise reconstruction MSE
**provably cannot beat plain MXFP4** -- ``c_i = 1`` is already its optimum,
and every other coefficient can only tie or lose. That is a theorem about
the objective, not a shortcoming of the search, and it is worth stating
plainly because it is easy to assume otherwise: at a *fixed* dequantization
scale, CRS/DGS buy nothing under an i.i.d.-input assumption. This module
does not paper over that -- :func:`quantize_weight_only_mxfp4_focus` with
``aggregate_error_weight=0.0`` reproduces plain MXFP4's codes *exactly*, and
``tests/test_focus_fp4.py`` asserts it.

Where the decoupled scale does buy something is where FOCUS's own end-to-end
loss lives: the error that reaches the *layer's output*, not the weight. For
``Y = X W`` the weight error ``D`` costs ``D^T E[X^T X] D``, and the
data-free ``E[X^T X] = I`` simplification is exactly what collapses the
objective back to element-wise MSE. Real transformer activations are not
that -- post-GELU/post-ReLU activations are non-negative and share a mean
component, so ``E[X^T X]`` carries a rank-one term. Taking
``E[X^T X] = sigma^2 (I + lambda * 11^T)`` (the standard "mean-shifted,
positively correlated inputs" model, the same phenomenon
:mod:`onnxsim.bias_correction` exists to correct) gives, per block,

    objective = sum_j D_j^2  +  lambda * (sum_j D_j)^2

-- element-wise MSE plus a penalty on the block's *aggregate signed* error.
That second term is not minimized by per-element nearest rounding, so it is
a genuine, non-degenerate use of the extra freedom CRS/DGS provide: a
slightly different ``c_i`` flips a handful of near-boundary elements the
other way, paying a small MSE cost to cancel a much larger systematic bias.
``lambda`` is this module's ``aggregate_error_weight`` and models the ratio
of the squared mean to the variance of the layer's input activations.
**This objective is onnxsim's own choice, not something AngelSlim's
documentation specifies** -- FOCUS trains against the network loss itself,
which needs data; this is the data-free surrogate that keeps the mechanism
non-trivial. The search is monotone by construction (the ``c_i = 1``
candidate is always evaluated, and DGS refinement starts from CRS's own
solution and only accepts strict improvements), so the objective it reports
is never worse than plain MXFP4's, at any setting.

**MXFP4 only.** The searcher below (:func:`_search_focus_codes`) takes an
arbitrary codebook and an arbitrary already-fixed dequantization scale, so
it is not specific to E8M0 and would apply unchanged to
:mod:`onnxsim.nvfp4_quantization`'s E4M3-scaled NVFP4. Only the MXFP4 entry
point is shipped: MXFP4 is the case where the coupling is most constrained
(E8M0 has no mantissa bits at all), and adding an NVFP4 entry point would
mean duplicating that module's own graph emission -- which, unlike
:mod:`onnxsim.mx_quantization`'s, has not been factored into a reusable
driver -- for a mechanism whose benefit here comes from the aggregate-error
term rather than from the scale format. That is a mechanical follow-up, not
a design question.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.mx_quantization import (
    MX_BLOCK_SIZE,
    MXFP4_CODEBOOK,
    _quantize_mxfp4_blockwise,
    _quantize_weight_only_mxfp4_impl,
)

# AngelSlim's own DGS sub-block size: 8 elements inside the (32-element)
# hardware block.
FOCUS_DGS_SUB_BLOCK_SIZE = 8


def _nearest_index(normalized: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """``argmin`` over ``codebook`` for every element of ``normalized`` --
    the same rule :func:`onnxsim.mx_quantization._nearest_codebook_index`
    applies, generalized to an arbitrary codebook so this module's search is
    not tied to E2M1.
    """
    diffs = np.abs(normalized[..., np.newaxis] - codebook)
    return np.argmin(diffs, axis=-1)


def _coefficient_grid(num_candidates: int, max_coefficient: float) -> np.ndarray:
    """Candidate CRS/DGS coefficients: values spaced evenly in log space over
    ``[1 / max_coefficient, max_coefficient]``, **always including exactly
    1.0** (the plain-MXFP4 choice) and ordered closest-to-1.0 first, so that
    ties in the search keep plain MXFP4's own codes rather than an arbitrary
    neighbour.
    """
    if num_candidates <= 1 or max_coefficient <= 1.0:
        return np.asarray([1.0])
    half = max(num_candidates // 2, 1)
    log_max = float(np.log(max_coefficient))
    grid = np.exp(np.linspace(-log_max, log_max, 2 * half + 1))
    grid[half] = 1.0  # exact, not just within float error
    order = np.argsort(np.abs(np.log(grid)), kind="stable")
    return grid[order]


def _search_focus_codes(
    blocks: np.ndarray,
    dequant_scale: np.ndarray,
    codebook: np.ndarray,
    coefficients: np.ndarray,
    sub_block_size: Optional[int],
    aggregate_error_weight: float,
    num_refinement_passes: int,
) -> np.ndarray:
    """Picks each block's codes by searching FOCUS's decoupled quantization
    scale, with ``dequant_scale`` held fixed at whatever the hardware format
    already committed to.

    :param blocks: ``[N, num_blocks, block_size]`` float64 weight values
    :param dequant_scale: ``[N, num_blocks]`` float64, the *stored* scale --
            never modified here, only divided into ``blocks`` after being
            multiplied by a trial coefficient
    :param codebook: the format's own values, ``[num_codes]`` float64
    :param coefficients: CRS/DGS candidates, as built by
            :func:`_coefficient_grid` (must contain 1.0 for the monotonicity
            guarantee to hold)
    :param sub_block_size: ``None`` for CRS (one coefficient per hardware
            block); a divisor of ``block_size`` for DGS
    :param aggregate_error_weight: ``lambda`` in the module docstring's
            objective ``sum(D^2) + lambda * (sum D)^2``
    :param num_refinement_passes: DGS coordinate-descent passes over the
            sub-blocks (ignored for CRS)
    :returns: ``[N, num_blocks, block_size]`` uint8 codebook indices
    """
    n, num_blocks, block_size = blocks.shape
    scale3 = dequant_scale[:, :, np.newaxis]
    lam = float(aggregate_error_weight)

    # --- CRS: one coefficient per hardware block, exhaustive over the grid.
    best_objective = np.full((n, num_blocks), np.inf)
    best_codes = np.zeros((n, num_blocks, block_size), dtype=np.int64)
    for coefficient in coefficients:
        codes = _nearest_index(blocks / (scale3 * coefficient), codebook)
        delta = codebook[codes] * scale3 - blocks
        objective = np.sum(delta * delta, axis=-1) + lam * np.sum(delta, axis=-1) ** 2
        improved = objective < best_objective  # strict: 1.0 first, ties keep it
        best_objective = np.where(improved, objective, best_objective)
        best_codes = np.where(improved[:, :, np.newaxis], codes, best_codes)

    if sub_block_size is None or sub_block_size >= block_size:
        return best_codes.astype(np.uint8)

    # --- DGS: refine per sub-block by coordinate descent, starting from the
    # CRS solution and accepting only strict improvements, so the block
    # objective is monotonically non-increasing and DGS can never come out
    # behind CRS (which in turn can never come out behind plain MXFP4).
    num_sub = block_size // sub_block_size
    delta = codebook[best_codes] * scale3 - blocks
    sub_delta = delta.reshape(n, num_blocks, num_sub, sub_block_size)
    sub_square = np.sum(sub_delta * sub_delta, axis=-1)  # [N, num_blocks, num_sub]
    sub_sum = np.sum(sub_delta, axis=-1)  # [N, num_blocks, num_sub]
    total_square = np.sum(sub_square, axis=-1)  # [N, num_blocks]
    total_sum = np.sum(sub_sum, axis=-1)  # [N, num_blocks]

    for _ in range(max(num_refinement_passes, 0)):
        for sub in range(num_sub):
            lo = sub * sub_block_size
            hi = lo + sub_block_size
            sub_blocks = blocks[:, :, lo:hi]
            rest_square = total_square - sub_square[:, :, sub]
            rest_sum = total_sum - sub_sum[:, :, sub]

            local_square = sub_square[:, :, sub]
            local_sum = sub_sum[:, :, sub]
            local_objective = (
                rest_square + local_square + lam * (rest_sum + local_sum) ** 2
            )
            local_codes = best_codes[:, :, lo:hi]

            for coefficient in coefficients:
                codes = _nearest_index(sub_blocks / (scale3 * coefficient), codebook)
                delta = codebook[codes] * scale3 - sub_blocks
                square = np.sum(delta * delta, axis=-1)
                summed = np.sum(delta, axis=-1)
                objective = rest_square + square + lam * (rest_sum + summed) ** 2
                improved = objective < local_objective
                local_objective = np.where(improved, objective, local_objective)
                local_square = np.where(improved, square, local_square)
                local_sum = np.where(improved, summed, local_sum)
                local_codes = np.where(improved[:, :, np.newaxis], codes, local_codes)

            best_codes[:, :, lo:hi] = local_codes
            sub_square[:, :, sub] = local_square
            sub_sum[:, :, sub] = local_sum
            total_square = rest_square + local_square
            total_sum = rest_sum + local_sum

    return best_codes.astype(np.uint8)


def _quantize_mxfp4_focus_blockwise(
    w_nk: np.ndarray,
    block_size: int,
    coefficients: np.ndarray,
    sub_block_size: Optional[int],
    aggregate_error_weight: float,
    num_refinement_passes: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """The FOCUS counterpart of
    :func:`onnxsim.mx_quantization._quantize_mxfp4_blockwise`, with the
    identical ``(codes_nk, scale_blocks)`` contract.

    The scale is not recomputed here -- it is taken *from*
    :func:`onnxsim.mx_quantization._quantize_mxfp4_blockwise` itself, so
    "the stored dequantization scale is exactly the one plain MXFP4 would
    store" holds by construction rather than by a duplicated formula that
    could drift.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    _plain_codes, scale = _quantize_mxfp4_blockwise(w_nk, block_size)
    blocks = w_nk.reshape(n, num_blocks, block_size)
    codes = _search_focus_codes(
        blocks,
        scale,
        np.asarray(MXFP4_CODEBOOK, dtype=np.float64),
        coefficients,
        sub_block_size,
        aggregate_error_weight,
        num_refinement_passes,
    )
    return codes.reshape(n, k), scale


def quantize_weight_only_mxfp4_focus(
    model: Union[str, onnx.ModelProto],
    block_size: int = MX_BLOCK_SIZE,
    sub_block_size: Optional[int] = FOCUS_DGS_SUB_BLOCK_SIZE,
    aggregate_error_weight: float = 1.0,
    num_coefficient_candidates: int = 17,
    max_coefficient: float = 1.25,
    num_refinement_passes: int = 2,
    coefficients: Optional[Sequence[float]] = None,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Quantizes exactly what
    :func:`onnxsim.mx_quantization.quantize_weight_only_mxfp4` quantizes,
    into exactly the format it emits, but picks each block's E2M1 codes with
    FOCUS's decoupled quantization scale (CRS, and optionally DGS) fitted by
    a data-free per-block grid search. Read this module's own docstring
    first -- in particular for what is *not* ported (FOCUS's end-to-end
    trained fit of the coefficients) and why the search objective is not
    plain element-wise MSE.

    The emitted model is structurally indistinguishable from
    :func:`~onnxsim.mx_quantization.quantize_weight_only_mxfp4`'s: same
    initializer names, shapes and dtypes, same nodes in the same order, and
    a bit-identical ``*_mxfp4_scale`` power-of-two scale. Only the
    ``*_mxfp4_q`` code values differ. Needs no calibration data.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) *hardware* scale
            group, i.e. the granularity of the stored power-of-two scale;
            the OCP MX spec's own canonical choice is 32
    :param sub_block_size: DGS granularity -- elements per coefficient
            *inside* a hardware block. Must divide ``block_size``.
            AngelSlim's own choice is 8; pass ``None`` (or ``block_size``)
            for plain CRS, one coefficient per hardware block. The stored
            dequantization scale stays at ``block_size`` granularity either
            way -- this only changes how finely the *discarded* offline
            coefficient may vary
    :param aggregate_error_weight: ``lambda`` in the per-block objective
            ``sum(D^2) + lambda * (sum D)^2``, modelling the ratio of the
            squared mean to the variance of the layer's input activations
            (see this module's docstring). ``0.0`` reduces the objective to
            plain element-wise MSE, for which per-element nearest rounding
            is already optimal -- so ``0.0`` reproduces
            :func:`~onnxsim.mx_quantization.quantize_weight_only_mxfp4`'s
            codes exactly, and is the degenerate setting, not a neutral one
    :param num_coefficient_candidates: size of the coefficient grid; forced
            odd internally so that ``1.0`` is always a candidate
    :param max_coefficient: grid endpoint -- candidates are log-spaced over
            ``[1 / max_coefficient, max_coefficient]``. E2M1's widest
            relative codebook gap is ``6 / 5 = 1.2``, so ``1.25`` already
            covers every reachable boundary flip
    :param num_refinement_passes: DGS coordinate-descent passes over the
            sub-blocks of each hardware block; ignored when
            ``sub_block_size`` is ``None``. Each pass can only lower the
            block objective
    :param coefficients: an explicit candidate list, overriding
            ``num_coefficient_candidates``/``max_coefficient``. Mostly for
            tests -- ``(1.0,)`` pins the search to the plain-MXFP4 choice
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by the
            same MXFP4 dequantization subgraph
            :func:`~onnxsim.mx_quantization.quantize_weight_only_mxfp4`
            emits. Layers with a non-constant, non-2-D, or
            non-block-divisible weight are left untouched.
    """
    if sub_block_size is not None and (
        sub_block_size <= 0 or block_size % sub_block_size != 0
    ):
        raise ValueError(
            f"sub_block_size ({sub_block_size}) must be a positive divisor of "
            f"block_size ({block_size})"
        )
    if aggregate_error_weight < 0.0:
        raise ValueError("aggregate_error_weight must be non-negative")
    if coefficients is not None:
        grid = np.asarray(list(coefficients), dtype=np.float64)
        if grid.size == 0 or np.any(grid <= 0.0):
            raise ValueError("coefficients must be a non-empty list of positive floats")
    else:
        grid = _coefficient_grid(num_coefficient_candidates, max_coefficient)

    def quantize_block(
        w_nk: np.ndarray, actual_block_size: int
    ) -> "tuple[np.ndarray, np.ndarray]":
        return _quantize_mxfp4_focus_blockwise(
            w_nk,
            actual_block_size,
            grid,
            sub_block_size,
            aggregate_error_weight,
            num_refinement_passes,
        )

    return _quantize_weight_only_mxfp4_impl(
        model, block_size, skip_names, quantize_block
    )
