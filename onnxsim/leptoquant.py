"""LeptoQuant (Tencent Hunyuan AI Infra Team, 2026, "AngelSlim: A more
accessible, comprehensive, and efficient toolkit for large model
compression", https://arxiv.org/abs/2602.21233) -- the outlier-aware
block-FP8 weight scale search AngelSlim introduces there. onnxsim ports
the *algorithm*, not that toolkit's code, per the same rationale
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` already give (AngelSlim quantizes
live PyTorch modules, with no ONNX export path).

Read :mod:`onnxsim.deepseek_fp8` first -- this module is a drop-in
refinement of exactly the weight side of that scheme, and reuses its own
FP8 machinery. DeepSeek-V3-style block FP8 picks each 128x128 block's
scale from that block's own true maximum magnitude
(``scale = max(abs(block)) / 448``). Weight distributions are typically
Laplacian-peaked with a few outliers, so a handful of large-magnitude
elements force that scale up, and the resulting coarse step size is then
paid for by every other (much more numerous, near-zero) element in the
block -- E4M3's 3-bit mantissa is scarce enough that this is a real loss.

LeptoQuant's own fix is to treat a chosen large-but-not-quite-maximal
magnitude as the block's effective "outlier boundary": elements beyond it
simply saturate to FP8's largest finite value during the round trip,
while the resulting *smaller* scale buys a finer effective step size for
the bulk of the block's values. How aggressively to clip is decided per
block by a small grid search over ``alpha`` (the fraction of elements
allowed to exceed the boundary; AngelSlim's own paper describes searching
``alpha`` in ``[0, 0.001]``), scoring each candidate by that block's own
exact reconstruction mean squared error:

- ``alpha = 0`` makes ``clip_value`` the block's true absmax -- i.e.
  literally :mod:`onnxsim.deepseek_fp8`'s own choice. Because it is always
  one of the grid's candidates, the search can never do measurably worse
  than plain max-based scaling: it only departs from it when clipping some
  outliers demonstrably lowers that block's own MSE.
- ``alpha > 0`` makes ``clip_value`` the ``1 - alpha`` quantile of
  ``abs(block)``, so an alpha-fraction of that block's elements saturate.

This is **data-free** (no calibration activations -- every decision comes
from the weight's own values, like every other onnxsim weight-only pass)
and **closed-form/grid-search only**: each candidate is evaluated by an
exact block reconstruction, with no gradient descent or training loop
anywhere, matching this repo's convention of porting the portable part of
a technique.

Two deliberate scope notes:

- FP8 rounding itself is *not* reimplemented here. This module calls
  :mod:`onnxsim.deepseek_fp8`'s own ``_fp8_round_trip``/``_FP8_MAX``,
  which reproduce the real ONNX ``FLOAT8E4M3FN`` cast semantics: how FP8
  values are represented is a data-representation concern already solved
  there, and LeptoQuant changes only *which scale* that round trip runs
  at.
- This port is **weight-only** (hence
  :func:`apply_leptoquant`'s weight-side-only behavior). AngelSlim's own
  LeptoQuant may additionally touch activations; reproducing that would
  need calibration data and runtime graph surgery, so it is out of scope
  here -- the same "port the closed-form part and name the scope
  honestly" convention :mod:`onnxsim.gptq` and friends already follow.

Because the round-tripped weight is written back as an ordinary float32
initializer (an FP8-round-tripped value cast back to float32 is just a
float32 number), this pass emits **no new graph nodes at all** -- no
``Cast``, no opset gate. Unlike :mod:`onnxsim.deepseek_fp8`, which needs
opset 19+ for the FP8 ``Cast`` it inserts on the *activation* side, this
module works on any opset :func:`onnxsim.llm_int8._match_matmul_like`
itself supports.

**Scope**: only ``MatMul`` and "vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) with a constant 2-D float32
weight are matched; ``Conv`` is left untouched, a scope decision several
other onnxsim modules already make.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.deepseek_fp8 import _FP8_MAX, _fp8_round_trip
from onnxsim.llm_int8 import _match_matmul_like

# AngelSlim's own paper describes searching the outlier fraction alpha in
# [0, 0.001]. alpha = 0 is the plain-absmax (deepseek_fp8) candidate and
# is always kept in the grid as the search's own safety net.
DEFAULT_ALPHA_GRID = (0.0, 1e-4, 2e-4, 5e-4, 1e-3)


def quantize_dequantize_block_fp8_leptoquant(
    values: np.ndarray,
    block_size: int = 128,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> np.ndarray:
    """LeptoQuant's outlier-aware FP8 E4M3 quantize-dequantize round trip
    over a 2-D array, one scale per ``block_size`` x ``block_size`` tile.
    Same shape/padding contract as
    :func:`onnxsim.deepseek_fp8.quantize_dequantize_block_fp8` (zero-padded
    up to whole tiles, the padding discarded before returning), but each
    block's scale is grid-searched over ``alpha_grid`` and scored by that
    block's own reconstruction MSE instead of always coming from the
    block's true absmax -- see this module's own docstring.

    With ``alpha_grid=(0.0,)`` this reduces exactly to
    :func:`onnxsim.deepseek_fp8.quantize_dequantize_block_fp8`.

    :param values: 2-D float array
    :param block_size: tile size along both axes
    :param alpha_grid: outlier fractions to try per block; ``0.0`` means
            "clip at the block's true absmax", ``a > 0`` means "clip at the
            ``1 - a`` quantile of ``abs(block)``"
    :returns: same shape as ``values``, float64
    """
    values = np.asarray(values, dtype=np.float64)
    rows, cols = values.shape
    padded_rows = -(-rows // block_size) * block_size
    padded_cols = -(-cols // block_size) * block_size
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float64)
    padded[:rows, :cols] = values

    out = np.empty_like(padded)
    for r in range(0, padded_rows, block_size):
        for c in range(0, padded_cols, block_size):
            block = padded[r : r + block_size, c : c + block_size]
            abs_block = np.abs(block)
            best: Optional[np.ndarray] = None
            best_mse = np.inf
            for alpha in alpha_grid:
                if alpha == 0.0:
                    # deepseek_fp8's own choice -- the safety-net candidate.
                    clip_value = float(np.max(abs_block))
                else:
                    clip_value = float(np.quantile(abs_block, 1.0 - alpha))
                scale = max(clip_value, 1e-12) / _FP8_MAX
                reconstructed = _fp8_round_trip(block / scale) * scale
                mse = float(np.mean(np.square(block - reconstructed)))
                if mse < best_mse:
                    best_mse = mse
                    best = reconstructed
            assert best is not None  # alpha_grid is never empty in practice
            out[r : r + block_size, c : c + block_size] = best
    return out[:rows, :cols]


def apply_leptoquant(
    float_model: Union[str, onnx.ModelProto],
    block_size: int = 128,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched MatMul/"vanilla" Gemm layer to
    LeptoQuant's outlier-aware block FP8 E4M3 scheme -- see this module's
    own docstring for the technique. Needs no calibration data: every
    quantization decision comes from the weight tensor's own values.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param block_size: weight tile size along both axes (K and N)
    :param alpha_grid: per-block outlier fractions to grid-search over
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``float_model`` with every matched layer's weight replaced by
            a new float32 initializer holding its LeptoQuant block-FP8
            round trip. No graph nodes are added or removed at all; layers
            with a non-constant or non-2-D weight are left untouched.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(float_model)
    graph = out.graph

    initializer_map = {t.name: t for t in graph.initializer}
    candidates = []  # (node, weight_initializer, weight_transposed)
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        _x_name, w_name, _bias_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, w_init, weight_transposed))
    if not candidates:
        return out

    taken_names = _all_names(graph)

    for node, w_init, weight_transposed in candidates:
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output-channel first
        w_quant_nk = quantize_dequantize_block_fp8_leptoquant(
            w_nk, block_size, alpha_grid
        )
        w_quant = w_quant_nk if weight_transposed else w_quant_nk.T

        new_w_name = _unique_name(f"{w_init.name}_leptoquant", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(w_quant.astype(np.float32), name=new_w_name)
        )
        node.input[1] = new_w_name

    return out
