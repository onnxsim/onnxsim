"""Sherry (Huang, Wu, Hu, Yu, Yang, Zhu, Liu, Wu, "Sherry:
Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained
Sparsification", https://arxiv.org/abs/2601.07892, code at
https://github.com/Tencent/AngelSlim tree/sherry/Sherry) -- Tencent
AngelSlim's **3:4 sparse ternary** weight format, at 1.25 bits/weight the
lowest-bit-width member of this repo's data-free weight-only family
alongside :mod:`onnxsim.gguf_ternary_quant` (BitNet b1.58's plain ternary
absmean rule) and :mod:`onnxsim.billm`/:mod:`onnxsim.bwa_ptq` (binary).

**The rule.** Every *contiguous block of 4 weights* is constrained to hold
exactly 3 non-zeros (each ``+-1``) and exactly 1 zero. The paper solves
``min ||W - T * alpha||^2`` under that 3:4 constraint with its own greedy
closed-form strategy, **"Sparse-AbsMean"**:

1. **Prune** -- in each 4-weight block, the element with the *smallest
   absolute magnitude* is set to zero (the one whose removal costs the
   least reconstruction error).
2. **Ternarize** -- the surviving three become ``sign(w)``.
3. **Scale** -- the per-output-channel factor ``alpha`` is the **mean
   absolute value of that channel's non-pruned weights** (the closed-form
   least-squares scale for a ``+-1`` code, hence "AbsMean"). It is a mean
   over the *kept* weights only, **not** over all of the channel's weights
   -- averaging in the pruned zeros would shrink ``alpha`` by a factor of
   ``3/4`` and systematically under-reconstruct every kept weight.

Like :mod:`onnxsim.gguf_ternary_quant` (and unlike
:mod:`onnxsim.gptq`/:mod:`onnxsim.awq` and the rest of the
calibration-driven family), this is entirely **data-free and closed-form**:
no calibration data, no search, no training -- ``W`` alone determines the
result.

**Why 3:4.** A block of 4 is power-of-two SIMD-aligned; 25% sparsity stays
in a safe margin rather than the aggressive unstructured sparsity that
usually costs accuracy; and the arithmetic works out exactly:
``C(4, 3) * 2**(3 - 1) = 16`` distinct magnitude/sparsity patterns, which
plus one sign bit exactly saturate a 5-bit index -- 5 bits per 4 weights =
**1.25 bits/weight**.

**Honesty note**, per this repo's established norm (see
:mod:`onnxsim.gguf_ternary_quant`'s own honesty note and
:mod:`onnxsim.iq4_nl`'s "codebook provenance" section):

- The quantization rule implemented here is **the paper's own published
  Sparse-AbsMean strategy**, not something onnxsim derived or tuned. It is
  verified in ``tests/test_sherry.py`` with real numpy arithmetic against a
  hand-computed example (which element each 4-block prunes, the resulting
  signs, and the exact ``alpha``), not against a recalled constant.
- **Sherry's other half, "Arenas" (Annealing Residual Synapse), is _not_
  implemented and is out of scope.** Arenas is a quantization-aware
  *training* mechanism: it adds a decaying full-precision residual term
  (``Y = X(T * alpha) + lambda_t * (X W)``, with ``lambda_t`` annealed to
  zero by the end of training) to counteract gradient homogenization and
  weight trapping while the network trains. onnxsim is stateless PTQ graph
  rewriting with no training loop at all (see
  ``docs/nncf-comparison-future-work.md``'s "Explicitly out of scope"
  section on QAT). **The accuracy numbers reported in the paper come from
  Sherry _with_ Arenas under QAT**; this module implements only the
  quantization scheme, so it should not be expected to reproduce the
  paper's benchmark accuracy.
- ONNX has no sub-INT4 tensor type and no 5-bit packed format, so --
  exactly like :mod:`onnxsim.gguf_ternary_quant`, :mod:`onnxsim.iq4_nl` and
  :mod:`onnxsim.gguf_kquant` -- the result here is a plain **float32
  quantize-dequantize round trip folded into a new initializer**, not the
  literal 5-bit packed layout. The "1.25 bits/weight" is a property of the
  *format*, not of what this module writes: the model this module produces
  is the same size as the float32 model it came from, and its value is
  simulating the format's numerics (and feeding downstream tooling), not
  compressing the file.

**``alpha`` granularity and layout.** The paper's scale is per-channel, and
"channel" here means the same thing every other weight-only quantizer in
this repo means by it -- the **output channel**, i.e. weights are
normalized to ``[N, K]`` (output channel first) before quantizing, exactly
as :mod:`onnxsim.awq`/:mod:`onnxsim.pb_llm`/:mod:`onnxsim.bwa_ptq` do:

- ``MatMul(X, W)`` / ``Gemm(..., transB=0)``: ``W`` is ``[K, N]``, so the
  output channel is its *last* axis and ``W.T`` is the ``[N, K]`` view.
- ``Gemm(..., transB=1)``: ``W`` is already ``[N, K]``.
- ``Conv``: ``W`` is ``[N, C_in, kH, kW, ...]``, output channel first
  already, flattened to ``[N, C_in * kH * kW * ...]``.

The 4-element blocks are therefore contiguous *along the reduction axis
within one output channel row* -- the same "a group runs along K inside one
row" convention :func:`onnxsim.quantize_weight_only_int4` and
:mod:`onnxsim.bwa_ptq` already use, and the reason a ``transB=1`` ``Gemm``
and the equivalent untransposed ``MatMul`` quantize to exactly transposed
results rather than to two different weights (``tests/test_sherry.py``
checks precisely that).

**Ragged tails and padding.** A channel row whose length is not a multiple
of 4 is zero-padded up to one, and the padding is made to have *no* effect
on the real weights -- which takes care, because a min-magnitude prune and
a mean-based scale are each sensitive to it in a different way:

- A padded zero has the smallest possible magnitude, so a naive
  ``argmin(|w|)`` would spend the block's one zero slot on it. That is
  exactly right here, and deliberate: in the real packed format a ragged
  tail *is* padded out to a whole 4-block, and the pad is what occupies
  that block's zero slot -- so every *real* weight in a ragged final block
  survives. The alternative reading (prune the smallest of the tail's real
  elements anyway) would destroy a real weight purely because the tensor's
  length is not a multiple of 4, and would zero a 1-element tail outright.
  To keep this exact under ties (a real weight that is itself exactly
  ``0.0`` would otherwise tie with the pads and could be the slot
  selected), the prune search below scores padded slots at ``-1``, strictly
  below every real ``|w|``.
- The padded zeros must **never** enter the ``alpha`` mean: they are not
  weights, and (unlike a max-based scale, which is invariant to them --
  cf. :mod:`onnxsim.gguf_legacy_quant`'s Q4_0) averaging them in would
  silently dilute ``alpha``. The mean below is taken over kept *real*
  elements only, divided by their real count -- the same care
  :mod:`onnxsim.gguf_ternary_quant` takes with its own mean-based scale.

Padding is therefore inert: the result is exactly what a by-hand
Sparse-AbsMean pass over the real elements alone produces
(``tests/test_sherry.py`` checks that against an independent, loop-written
reference for every row length from 1 to 17). Note this is the one place a
pad and a real ``0.0`` weight behave differently: a *real* trailing zero is
a weight, so it is either its block's pruned element or a kept element that
contributes ``|0|`` to the mean, whereas a pad is neither.

**Sign convention.** ``np.sign`` is used as the paper writes it, so a
weight that is exactly ``0.0`` and is not its block's pruned element stays
``0.0`` rather than being forced to ``+alpha``: zero is its own best
ternary code, and a tie-broken ``+-1`` would only add error. Such a block
then holds two zeros -- the only way a block departs from "exactly one
zero", and only for weights that were already exactly zero.

**Scope.** Matches ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) with a constant 2-D float32
weight and, optionally, ``Conv`` with a constant float32 weight of any
rank >= 2. Anything else -- a non-constant weight, a non-float32 weight, a
non-2-D MatMul/Gemm weight (whose output channel axis is not unambiguous)
-- is left untouched.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.llm_int8 import _match_matmul_like

_BLOCK_SIZE = 4  # the paper's own N:M block size -- the "4" of 3:4
_NONZERO_PER_BLOCK = 3  # the "3" of 3:4 -- exactly one element per block is pruned


def _sherry_rows(rows: np.ndarray) -> np.ndarray:
    """Sparse-AbsMean over a 2-D ``[channels, elements]`` array: one
    ``alpha`` per row (output channel), with the 4-element blocks running
    along the row. See this module's own docstring for the rule itself and
    for how a ragged final block is padded.

    :param rows: 2-D float array, output channel first
    :returns: same shape as ``rows``, float64
    """
    rows = np.asarray(rows, dtype=np.float64)
    n_rows, k = rows.shape
    padded_k = -(-k // _BLOCK_SIZE) * _BLOCK_SIZE
    if padded_k != k:
        rows = np.concatenate(
            [rows, np.zeros((n_rows, padded_k - k), dtype=np.float64)], axis=1
        )

    blocks = rows.reshape(n_rows, -1, _BLOCK_SIZE)
    real = np.zeros(padded_k, dtype=bool)
    real[:k] = True
    real_blocks = np.broadcast_to(real.reshape(1, -1, _BLOCK_SIZE), blocks.shape)
    abs_blocks = np.abs(blocks)

    # Step 1, prune: each block's smallest-|w| element is zeroed. Padded
    # slots score -1, strictly below every real |w| (>= 0), so a ragged
    # final block spends its one zero on a pad and keeps all of its real
    # weights -- see this module's docstring on padding. argmin takes the
    # first minimum, so ties resolve deterministically (leftmost).
    prune_key = np.where(real_blocks, abs_blocks, -1.0)
    prune_index = np.argmin(prune_key, axis=-1)
    pruned = prune_index[..., np.newaxis] == np.arange(_BLOCK_SIZE)
    kept = real_blocks & ~pruned

    # Step 3, scale: alpha = mean(|w|) over the channel's KEPT elements --
    # not over all of them (that would shrink alpha by ~3/4), and not over
    # the padded slots (which are not weights at all).
    kept_count = np.maximum(np.count_nonzero(kept, axis=(1, 2)), 1)
    alpha = np.sum(np.where(kept, abs_blocks, 0.0), axis=(1, 2)) / kept_count

    # Step 2, ternarize: every kept element becomes sign(w).
    code = np.where(kept, np.sign(blocks), 0.0)
    dequant = code * alpha[:, np.newaxis, np.newaxis]
    return dequant.reshape(n_rows, padded_k)[:, :k]


def quantize_dequantize_sherry(values: np.ndarray) -> np.ndarray:
    """Sherry 3:4 sparse-ternary quantize-dequantize round trip over a
    flattened float array of any length -- one zero per contiguous
    4-element block (that block's smallest ``|w|``), ``sign(w)`` for the
    other three, and one shared ``alpha = mean(|w|)`` over the non-pruned
    elements.

    ``values`` is treated as **one** channel: it is flattened and a single
    ``alpha`` covers all of it, mirroring
    :func:`onnxsim.quantize_dequantize_ternary`'s own flattened entry
    point. :func:`apply_sherry_quantization` is the per-output-channel
    entry point -- it normalizes each matched layer's weight to ``[N, K]``
    and gives every output channel its own ``alpha``, which is what the
    paper specifies.

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, float64
    """
    values = np.asarray(values, dtype=np.float64)
    return _sherry_rows(values.reshape(1, -1)).reshape(values.shape)


def apply_sherry_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched layer's float32 weight into
    Sherry's 3:4 sparse ternary format, one ``alpha`` per output channel --
    see this module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input (any
            rank >= 2; the output channel is axis 0 and the rest of the
            weight is flattened into the row)
    :param skip_names: weight initializer names to leave unquantized
    :returns: ``model`` with every matched layer's weight replaced by its
            3:4 sparse ternary round trip; layers with a non-constant,
            non-float32, or non-2-D MatMul/Gemm weight are left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    # `channel_first` records whether the weight's own axis 0 already is
    # the output channel (a transB=1 Gemm, or a Conv) or whether the weight
    # is [K, N] and has to be transposed into [N, K] first.
    candidates: "list[tuple[onnx.NodeProto, str, bool]]" = []
    for node in graph.node:
        w_name: Optional[str] = None
        channel_first = False
        is_conv = False
        match = _match_matmul_like(node)
        if match is not None:
            _x_name, matched_w_name, _bias_name, weight_transposed = match
            w_name = matched_w_name
            channel_first = weight_transposed
        elif include_conv and node.op_type == "Conv" and len(node.input) >= 2:
            w_name = node.input[1]
            channel_first = True
            is_conv = True
        if w_name is None or w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if w_init is None or w_init.data_type != onnx.TensorProto.FLOAT:
            continue
        if len(w_init.dims) < 2 or (not is_conv and len(w_init.dims) != 2):
            continue  # no unambiguous output channel axis -- leave it alone
        candidates.append((node, w_name, channel_first))

    # Keyed by layout too: one shared initializer consumed both as [K, N]
    # and as an already-transposed [N, K] weight needs both quantizations.
    quantized_names: "dict[tuple[str, bool], str]" = {}
    for node, w_name, channel_first in candidates:
        key = (w_name, channel_first)
        new_name = quantized_names.get(key)
        if new_name is None:
            w_init = initializer_map[w_name]
            w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
            shape = w.shape
            # [N, K], output channel first -- the same normalization
            # onnxsim.awq/onnxsim.pb_llm/onnxsim.bwa_ptq use.
            w_nk = w.reshape(shape[0], -1) if channel_first else w.T
            q_nk = _sherry_rows(w_nk)
            q = q_nk.reshape(shape) if channel_first else q_nk.T
            w_quant = np.ascontiguousarray(q, dtype=np.float32)

            new_name = _unique_name(f"{w_name}_sherry", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[key] = new_name
        node.input[1] = new_name

    return out
