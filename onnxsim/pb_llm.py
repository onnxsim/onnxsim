"""PB-LLM (Shang, Yuan, Wu, Dong, 2024, ICLR 2024, "PB-LLM: Partially
Binarized Large Language Models", https://arxiv.org/abs/2310.00034) --
a **structured mixed-precision binarizer**: per matched layer, a small
salience-selected fraction of input channels (columns) stays at INT8,
every other column is pushed to ~1 bit/element, and both live in the same
``Code``/``Scale`` initializer pair so the graph rewrite needs only a
single ``Cast``/``Mul``.

Relationship to :mod:`onnxsim.billm` (the repo's other weight binarizer --
read that module's own docstring first). BiLLM binarizes **the entire
weight matrix**: every column ends up represented by one or two binary
(``{-1,+1}``) levels, and the salient/non-salient split there only decides
*how many* binary levels a column gets (two, via a residual, for salient
columns; one for everything else) -- the whole layer stays in the
"BiLLM"-shaped, close-to-1-bit-average family regardless of the split.
PB-LLM is a different point in the design space: its salient fraction is
not binarized *at all* -- it is quantized to genuine INT8, the same
precision :func:`onnxsim.quantize_weight_only_int8_block` already produces
for a whole layer -- and only the *remaining*, non-salient fraction is
binarized. So where BiLLM's split is "1-bit vs. 2-bit", PB-LLM's split is
"8-bit vs. ~1-bit": a real mixed high/low bit-width layer, exactly the
"Partially-Binarized" property the paper's own name describes, not a
finer-grained flavor of full binarization. The two are complementary
points on a spectrum rather than one subsuming the other:
``salient_ratio=0.0`` degenerates PB-LLM to a plain per-column binarizer
(no INT8 columns at all, the same *shape* of output BiLLM's own
non-salient path produces, though still via a single flat per-column scale
rather than BiLLM's own block-searched, residual-compensated scheme), and
``salient_ratio=1.0`` degenerates it to plain per-column INT8 (no
binarization at all) -- BiLLM has no comparable "all INT8" limit, since
every one of its output columns is always binary.

Relationship to :mod:`onnxsim.mixed_precision` (the repo's other
salience-driven bit-width dispatcher -- read that module's own docstring
too). That module picks a bit-width **per whole layer**
(:func:`onnxsim.apply_mixed_precision_quantization` sends the top
``high_bits_fraction`` of *layers*, ranked by an INT4-reconstruction-MSE-
times-activation-energy score, to block-wise INT8, and every other whole
layer to block-wise INT4) -- it never splits precision *within* one
layer's own weight matrix. PB-LLM's split is per **column**, inside a
single layer, and its salience score is a different quantity entirely:
:mod:`onnxsim.mixed_precision` asks "how much does this whole layer's
*output* degrade if INT4-quantized" (an MSE-of-reconstruction proxy);
PB-LLM asks "how much does *this column* already matter" via a direct
Hessian-diagonal-weighted-magnitude score,

    salience_j = mean_n(|W[n, j]|) * diag(H)_j,   H = X^T X

the same style of score GPTQ/SparseGPT/OWQ (:mod:`onnxsim.owq`) build on,
but -- unlike OWQ's own Optimal-Brain-Surgeon score
(``mean_n[(W[n,j] - RTN(W[n,j]))^2] / [H^-1]_jj``, read
:mod:`onnxsim.owq`'s own docstring) -- computed directly from ``diag(H)``
with **no Hessian inversion at all**: ``diag(H)_j`` is just each input
channel's own calibration activation energy (``sum_samples(X[:, j]^2)``),
a cheap, always-well-defined proxy for "how strongly this channel actually
gets excited", multiplied by the weight column's own average magnitude
("how much does the weight there matter if it does get excited"). OWQ's
own score additionally needs an *existing* quantization (its own RTN
residual) to rank against and a damped Cholesky solve of ``H^-1``; PB-LLM's
runs directly off the original float weight and a plain sum of squares,
and -- unlike OWQ, which restores its selected columns to *exact* float32
via a correction term added on top of someone else's already-INT4-
quantized model -- PB-LLM quantizes the whole layer itself, in one pass,
straight from the float model.

**The technique**, one matched layer at a time:

1. **Salience** (as above): ``salience_j = mean_n(|W[n, j]|) * diag(H)_j``
   for each input channel ``j``, ``H = X^T X`` from real calibration
   activations (the same Hessian construction :mod:`onnxsim.gptq`/
   :mod:`onnxsim.billm`/:mod:`onnxsim.owq` all use).

2. **Split**: the top ``salient_ratio`` fraction of columns by salience
   (rounded to the nearest column count, paper's own experiments sweep
   roughly 10-30%) form the salient set; every other column is
   non-salient.

3. **Salient columns -> INT8**: per-column symmetric round-to-nearest,
   ``scale_j = max_n(|W[n, j]|) / 127``, ``code_{n,j} = round(W[n,j] /
   scale_j)`` clipped to ``[-127, 127]`` -- the same granularity
   :func:`onnxsim.quantize_weight_only_int8_block` uses for a whole block,
   specialized here to a single column (the natural per-column limit of a
   block-wise scale).

4. **Non-salient columns -> ~1 bit**: ``scale_j = mean_n(|W[n, j]|)``,
   ``code_{n,j} = sign(W[n, j])`` (reusing :func:`onnxsim.billm._sign`, the
   paper's own ``sign(x) = 1 if x >= 0 else -1`` convention) -- a plain
   per-column flavor of :mod:`onnxsim.billm`'s own single-level
   ``_binary`` primitive (mean-abs scale, elementwise sign), computed
   per-column here rather than over :mod:`onnxsim.billm`'s own
   block-of-columns group, since PB-LLM's salient/non-salient split is
   already per-column and needs no further block-search step of its own.
   This is a deliberate simplification of the paper's own "Optimal"-mode
   residual refinement (Section 4.2), which adds a second binarization
   level on top of the non-salient columns' own residual for extra
   accuracy, the same way :mod:`onnxsim.billm` applies a residual to *its
   own* salient columns -- a reasonable future addition (it would reuse
   :func:`onnxsim.billm._binary`'s residual pattern directly), not
   attempted here since the paper reports its simpler "Naive" binarization
   mode (what this module implements) already recovers most of PB-LLM's
   accuracy gain over full binarization.

Because an INT8 code and a ~1-bit ``{-1, +1}`` code both reconstruct the
same way (``value = code * scale``, just over a different code range), one
matched layer needs only a **single** code tensor and a **single**
per-column scale tensor -- no separate branches to merge in-graph:

    Before:
      Y = MatMul(X, W) [+ bias]        -- W constant, [K, N], float32

    After:
      Code: initializer, int8, [K, N]  -- per-element code: a full INT8
            value (``[-127, 127]``) for a salient column, ``{-1, +1}`` for
            a non-salient one
      Scale: initializer, float32, [K, 1]  -- per-column scale
      What_hat = Mul(Cast(Code, float), Scale)
      Y = MatMul(X, What_hat) [+ bias]

No ``Gather``/codebook lookup (the code cast to float already *is* the
reconstructed value up to the per-column scale, the same reasoning
:mod:`onnxsim.billm`'s own docstring gives for its own encoding) and no
``Add`` of a second term (unlike :mod:`onnxsim.billm`'s two-level salient
path -- PB-LLM's salient columns are single-level INT8, not a binary
residual pair). Ordinary ONNX ops only (``Cast``/``Mul``), opset 11+.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import (
    _activation_rows,
    _add_probe_outputs,
    _all_names,
    _unique_name,
)
from onnxsim.billm import _sign
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.quip_sharp import _match_matmul_like


def _pb_llm_quantize_columns(
    w_nk: np.ndarray, diag_h: np.ndarray, salient_ratio: float
) -> "tuple[np.ndarray, np.ndarray]":
    """Splits ``w_nk``'s ([N, K], output channel first) columns into a
    salient (INT8) and non-salient (~1-bit) set by
    ``mean_n(|w|) * diag_h`` per column (see this module's own docstring),
    and quantizes each column accordingly. Returns ``(code_nk, scale_k)``:
    an ``[N, K]`` int8 code array and a length-``K`` per-column float64
    scale array.
    """
    n, k = w_nk.shape
    code = np.empty((n, k), dtype=np.int8)
    scale = np.empty(k, dtype=np.float64)
    if k == 0:
        return code, scale

    col_mag = np.mean(np.abs(w_nk), axis=0)
    salience = col_mag * diag_h
    order = np.argsort(-salience)
    num_salient = int(round(salient_ratio * k))
    num_salient = max(0, min(k, num_salient))
    salient_cols = order[:num_salient]
    nonsalient_cols = order[num_salient:]

    if salient_cols.size > 0:
        w_sal = w_nk[:, salient_cols]
        scale_sal = np.maximum(np.abs(w_sal).max(axis=0), 1e-12) / 127.0
        code_sal = np.clip(np.round(w_sal / scale_sal), -127, 127)
        code[:, salient_cols] = code_sal.astype(np.int8)
        scale[salient_cols] = scale_sal

    if nonsalient_cols.size > 0:
        w_ns = w_nk[:, nonsalient_cols]
        scale_ns = np.mean(np.abs(w_ns), axis=0)
        code_ns = _sign(w_ns)
        code[:, nonsalient_cols] = code_ns.astype(np.int8)
        scale[nonsalient_cols] = scale_ns

    return code, scale


def quantize_weight_only_pb_llm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    salient_ratio: float = 0.15,
    skip_names: Optional[Iterable[str]] = None,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Partially binarizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight: the ``salient_ratio`` fraction of input channels
    with the highest Hessian-diagonal-weighted magnitude stay INT8, every
    other channel is binarized to ~1 bit/element -- see this module's own
    docstring for the technique and how it differs from
    :mod:`onnxsim.billm` (full binarization) and
    :mod:`onnxsim.mixed_precision` (whole-layer, not per-column,
    bit-width dispatch).

    Needs real calibration activations to compute each layer's
    Hessian-diagonal salience, the same as :mod:`onnxsim.gptq`/
    :mod:`onnxsim.billm`/:mod:`onnxsim.owq`.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian diagonal from -- see
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same
            name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param salient_ratio: fraction of each layer's input columns (by
            count, most salient first) kept at INT8; the rest are
            binarized. ``0.0`` binarizes every column, ``1.0`` quantizes
            every column to INT8 -- see this module's own docstring for
            both limits
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Cast(Code), Scale)`` feeding the original MatMul/Gemm
            node -- ordinary ONNX ops only, opset 11+. Layers with a
            non-constant, non-2-D, or non-float32 weight, or whose
            activation input has no feature axis at all (rank < 2), are
            left untouched; a higher-rank ``[batch, seq, K]`` activation
            is flattened to ``[batch * seq, K]``, which is exact.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    candidates = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, _bias_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_name, weight_transposed))

    if not candidates:
        return out

    probe_names = sorted({c[1] for c in candidates})
    probe_model = _add_probe_outputs(model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(result[name], dtype=np.float64))

    for node, x_name, w_name, weight_transposed in candidates:
        acts = _activation_rows(activations[x_name])
        if not acts:
            continue  # no usable activation (no feature axis); skip
        x = np.concatenate(acts, axis=0)

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape
        if x.shape[1] != k:
            continue  # activation's feature dim doesn't match K; skip

        diag_h = np.sum(x**2, axis=0)  # [K]
        code_nk, scale_k = _pb_llm_quantize_columns(w_nk, diag_h, salient_ratio)

        code_orig = code_nk if weight_transposed else code_nk.T
        assert code_orig.shape == (dim0, dim1)

        # scale_k is indexed along K (the reduction dim). When
        # weight_transposed (W is [N, K], K last), it broadcasts against W
        # as-is; otherwise (W is [K, N], K first) it needs a trailing
        # size-1 axis to broadcast against axis 0 instead of axis -1 --
        # the same reasoning onnxsim.billm's own quantize_weight_only_billm
        # uses for its own per-column scale.
        scale_orig = scale_k if weight_transposed else scale_k[:, np.newaxis]

        prefix = f"{w_name}_pb_llm"
        code_name = _unique_name(f"{prefix}_code", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(code_orig.astype(np.int8), name=code_name)
        )
        scale_name = _unique_name(f"{prefix}_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_orig.astype(np.float32), name=scale_name)
        )

        cast_out = _unique_name(f"{prefix}_code_f", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [code_name], [cast_out], to=onnx.TensorProto.FLOAT
        )
        dq_out = _unique_name(f"{prefix}_dq", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [cast_out, scale_name],
            [dq_out],
            name=_unique_name(f"{prefix}_dequant", taken_names),
        )

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in (cast_node, mul_node):
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out
