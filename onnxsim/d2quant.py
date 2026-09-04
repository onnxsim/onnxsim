r"""D2Quant (Yan, Bao, Li, Zhang, Zhang, Xie, Sun and Zhang, 2026, "D2Quant:
Accurate Low-bit Post-Training Weight Quantization for LLMs",
https://arxiv.org/abs/2602.02546, code at
https://github.com/XIANGLONGYAN/D2Quant). A weight-only PTQ framework built
around two independent techniques, both ported here:

* **Dual-Scale Quantizer (DSQ)** (:func:`apply_dsq`) -- a weight-side fix
  targeted specifically at down-projection matrices (the second Linear in a
  SwiGLU/GLU-style MLP block, i.e. the one whose *input* is an elementwise-
  gated activation). That gated activation has an unusually heavy-tailed
  distribution, which is well documented as a quantization bottleneck for
  the down-projection weight that consumes it -- a single group-wide scale
  ends up stretched to cover a handful of outlier-aligned columns, blurring
  every other column's precision.
* **Deviation-Aware Correction (DAC)** (:func:`apply_dac`) -- an
  activation-side fix: weight quantization shifts a layer's own output
  *mean*, and this shift is measurably more pronounced and more consistently
  directional (in the paper's own terms, higher "SNR") right after the
  attention block than elsewhere. DAC estimates that per-channel mean shift
  from calibration data and folds it directly into the bias of the
  LayerNormalization that comes right after the shift was introduced.

Both techniques share the same "absorbable"/zero-extra-node philosophy this
repo's own :mod:`onnxsim.bias_correction` and :mod:`onnxsim.outlier_suppression`
already use -- see each function's own docstring for exactly how.

**Scope.** This ports the paper's two per-technique *mechanisms* faithfully,
not its Algorithm 1 end-to-end block-wise pipeline (which interleaves DSQ,
DAC, and attention/FFN quantization block-by-block, re-deriving calibration
activations after each block is quantized so later blocks see the same drift
inference will). Both :func:`apply_dsq` and :func:`apply_dac` are meant to
compose with any of onnxsim's other own weight quantizers instead:

1. Run :func:`apply_dsq` on the float model -- it quantizes every matched
   down-projection directly (to the same INT4 block format
   :func:`onnxsim.quantize_weight_only_int4` produces) and rescales its
   paired up-projection's raw float weight in place, absorbing DSQ's own
   auxiliary scale with zero new nodes.
2. Quantize everything else (gate/up-projections, attention projections,
   ...) with any onnxsim weight-only quantizer, e.g.
   :func:`onnxsim.quantize_weight_only_int4` -- since step 1 already
   rescaled the up-projection's raw values, that quantizer's own ordinary
   per-channel scale absorbs DSQ's contribution automatically.
3. Run :func:`apply_dac` (comparing the original float model against the
   now-fully-quantized model from steps 1-2) to fold each LayerNormalization's
   measured mean-shift deviation into its own bias.

Also deliberately not ported: the paper's own equivalent up/down scaling
derivation is presented as an exact closed-form alternating optimization
against real (GPTQ-style) block-wise quantization error over 15 iterations
against a specific baseline quantizer; :func:`apply_dsq` solves the same
per-column-scale objective (``min_s ||W - Q(W / s) * s||`` over a plain
per-channel-block symmetric quantizer, alternating between re-quantizing and
a closed-form least-squares scale update) rather than reproducing that
baseline's own exact solver line-for-line -- the same "faithful to the
objective, not a specific reference implementation" stance
:mod:`onnxsim.hqq` already takes for its own IRLS solver.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.smoothquant import _match_matmul_like

_INT4 = onnx.TensorProto.INT4


# ---------------------------------------------------------------------------
# Dual-Scale Quantizer (DSQ)
# ---------------------------------------------------------------------------


def _quantize_int4_blockwise_symmetric(
    w_nk: np.ndarray, block_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Ordinary symmetric INT4 block quantization -- one absmax-derived
    scale per (output row, ``block_size``-wide slice of the reduction
    dimension), matching :func:`onnxsim.quantize_weight_only_int4`'s own
    scheme (codes in ``[-7, 7]``). Returns ``(codes_nk, scale_nb)``.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    amax = np.max(np.abs(blocks), axis=2)
    scale = np.maximum(amax / 7.0, 1e-12)
    scale3 = scale[:, :, np.newaxis]
    codes = np.clip(np.round(blocks / scale3), -7, 7)
    return codes.reshape(n, k), scale


def _dequantize_int4_blockwise_symmetric(
    codes_nk: np.ndarray, scale_nb: np.ndarray, block_size: int
) -> np.ndarray:
    n, k = codes_nk.shape
    num_blocks = k // block_size
    codes3 = codes_nk.reshape(n, num_blocks, block_size)
    scale3 = scale_nb[:, :, np.newaxis]
    return (codes3 * scale3).reshape(n, k)


def _dsq_optimize(
    w_nk: np.ndarray, block_size: int, num_iterations: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solves ``min_s ||W - Q(W / s) * s||_F^2`` for a per-column scale
    ``s`` (one scalar per reduction-dimension column, shared by every output
    row/block) by alternating: (a) freeze ``s``, re-quantize ``W / s`` with
    the ordinary block quantizer above; (b) freeze the resulting integer
    codes' dequantized values, re-solve ``s`` in closed form (per-column
    weighted least squares: ``s[h] = sum_row(W[:,h] * dequant[:,h]) /
    sum_row(dequant[:,h] ** 2)``). Returns ``(s, codes_nk, scale_nb)`` for
    the ``W / s`` quantization from the final iteration.
    """
    k = w_nk.shape[1]
    s_c = np.ones(k, dtype=np.float64)
    codes = np.zeros_like(w_nk)
    scale_blocks = np.zeros((w_nk.shape[0], k // block_size), dtype=np.float64)
    for _ in range(max(num_iterations, 0)):
        w_norm = w_nk / s_c[np.newaxis, :]
        codes, scale_blocks = _quantize_int4_blockwise_symmetric(w_norm, block_size)
        dequant_norm = _dequantize_int4_blockwise_symmetric(
            codes, scale_blocks, block_size
        )
        num = np.sum(w_nk * dequant_norm, axis=0)
        den = np.sum(dequant_norm * dequant_norm, axis=0)
        s_c = np.where(den > 1e-20, num / den, s_c)
    w_norm = w_nk / s_c[np.newaxis, :]
    codes, scale_blocks = _quantize_int4_blockwise_symmetric(w_norm, block_size)
    return s_c, codes, scale_blocks


def _pack_int4_signed(codes: np.ndarray) -> bytes:
    # Two's-complement nibble packing, low-nibble-first (ONNX's documented
    # INT4 raw_data layout) -- same masking trick as
    # onnxsim.hqq._pack_uint4, but the values here are already signed
    # (numpy's bitwise AND on a signed dtype operates on its two's-complement
    # bit pattern, so `& 0xF` recovers the correct nibble either way).
    flat = codes.astype(np.int64).ravel()
    nibbles = (flat & 0xF).astype(np.uint8)
    lo = nibbles[0::2]
    hi = nibbles[1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed.tobytes()


def apply_dsq(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    num_iterations: int = 15,
) -> onnx.ModelProto:
    """Applies the Dual-Scale Quantizer to every matched down-projection in
    a SwiGLU/GLU-style MLP block -- see this module's own docstring for the
    technique and :func:`onnxsim.apply_dsq`'s companion :func:`apply_dac`.

    A "matched" block is a plain MatMul/vanilla-Gemm node (the down-proj)
    whose activation input is produced by an elementwise ``Mul`` with
    exactly two operands, one of which is *directly* (with no intervening
    op) the output of another plain MatMul/vanilla-Gemm node (the up-proj)
    -- the shape every SwiGLU MLP (``down(silu(gate(x)) * up(x))``) and
    plain bilinear GLU takes, regardless of which operand carries the
    nonlinearity (only the *unactivated* operand's producer can be safely
    rescaled, since scaling before a nonlinearity does not commute with it).
    Both the up-proj's own output and the gated (down-proj input) tensor
    must have exactly one consumer and must not themselves be graph outputs
    -- exactly the conservative "only fully-owned consumers get touched"
    stance :mod:`onnxsim.outlier_suppression`'s own Gamma Migration takes,
    for the same reason (an external/other consumer would silently observe
    a rescaled value it never asked for).

    The auxiliary per-column scale DSQ derives for the down-projection is
    "absorbable": instead of inserting a node to multiply the down-proj's
    dequantized output by it, the *reciprocal* is folded into the up-proj's
    raw weight (its own output channels are exactly the down-proj's
    reduction/input channels), so the gated activation is already correctly
    pre-scaled by the time it reaches the down-proj -- no new runtime op
    beyond the down-proj's own ``DequantizeLinear``.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) quantization
            group along the down-projection's reduction dimension, matching
            :func:`onnxsim.quantize_weight_only_int4`'s own default
    :param num_iterations: alternating scale/re-quantization steps (the
            paper's own default, ``15``, empirically saturates)
    :returns: ``model`` with every matched down-projection weight replaced
            by ``DequantizeLinear(Wq, Ws, axis=<reduction axis>,
            block_size=block_size)`` (INT4 codes in ``[-7, 7]``, matching
            :func:`onnxsim.quantize_weight_only_int4`'s own storage shape)
            and every matched up-projection's raw float weight rescaled in
            place -- feed the result to a weight-only quantizer (e.g.
            :func:`onnxsim.quantize_weight_only_int4`) to quantize the rest
            of the model; that quantizer's own per-channel scale absorbs the
            rescaling for free. An opset below 21 (INT4 tensors and
            ``DequantizeLinear``'s ``block_size`` both need it), or a model
            with no matched block, is returned unchanged. Consider calling
            :func:`onnxsim.simplify` afterward to drop the now-orphaned
            float down-projection initializers.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    opset_ge_21 = any(
        o.domain in ("", "ai.onnx") and o.version >= 21 for o in model.opset_import
    )
    if not opset_ge_21:
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    graph_output_names = {o.name for o in graph.output}

    producer_of: Dict[str, onnx.NodeProto] = {}
    consumer_count: Dict[str, int] = {}
    for n in graph.node:
        for out_name in n.output:
            producer_of[out_name] = n
        for in_name in n.input:
            consumer_count[in_name] = consumer_count.get(in_name, 0) + 1

    targets = []  # (down_node, down_w_init, down_transposed, up_w_init, up_transposed)
    for down_node in graph.node:
        match = _match_matmul_like(down_node)
        if match is None:
            continue
        down_x, down_w, down_transposed = match
        gate_mul = producer_of.get(down_x)
        if (
            gate_mul is None
            or gate_mul.op_type != "Mul"
            or len(gate_mul.input) != 2
            or consumer_count.get(down_x, 0) != 1
            or down_x in graph_output_names
        ):
            continue

        up_info = None
        for operand in gate_mul.input:
            up_node = producer_of.get(operand)
            if up_node is None:
                continue
            up_match = _match_matmul_like(up_node)
            if up_match is None:
                continue
            _, up_w, up_transposed = up_match
            if (
                consumer_count.get(operand, 0) != 1
                or operand in graph_output_names
                or consumer_count.get(up_w, 0) != 1
            ):
                continue
            up_info = (up_w, up_transposed)
            break
        if up_info is None:
            continue
        up_w, up_transposed = up_info

        down_w_init = initializer_map.get(down_w)
        up_w_init = initializer_map.get(up_w)
        if (
            down_w_init is None
            or up_w_init is None
            or down_w_init.data_type != onnx.TensorProto.FLOAT
            or up_w_init.data_type != onnx.TensorProto.FLOAT
            or len(down_w_init.dims) != 2
            or len(up_w_init.dims) != 2
        ):
            continue

        down_k = down_w_init.dims[1] if down_transposed else down_w_init.dims[0]
        up_n = up_w_init.dims[0] if up_transposed else up_w_init.dims[1]
        if down_k != up_n or down_k % block_size != 0:
            continue

        targets.append(
            (down_node, down_w_init, down_transposed, up_w_init, up_transposed)
        )

    for down_node, down_w_init, down_transposed, up_w_init, up_transposed in targets:
        w = onnx.numpy_helper.to_array(down_w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if down_transposed else w.T  # [N=Dout, K=H]

        s_c, codes_nk, scale_nb = _dsq_optimize(w_nk, block_size, num_iterations)

        up_w = onnx.numpy_helper.to_array(up_w_init).astype(np.float64)
        up_dim0, up_dim1 = up_w.shape
        up_nk = up_w if up_transposed else up_w.T  # [N=H, K]
        up_new_nk = up_nk * s_c[:, np.newaxis]
        up_new = up_new_nk if up_transposed else up_new_nk.T
        up_new = up_new.reshape(up_dim0, up_dim1).astype(np.float32)
        up_w_init.CopyFrom(onnx.numpy_helper.from_array(up_new, name=up_w_init.name))

        codes_orig = codes_nk if down_transposed else codes_nk.T
        scale_orig = scale_nb if down_transposed else scale_nb.T
        assert codes_orig.shape == (dim0, dim1)

        wq = onnx.TensorProto()
        wq.name = _unique_name(f"{down_w_init.name}_dsq_q", taken_names)
        wq.data_type = _INT4
        wq.dims.extend(codes_orig.shape)
        wq.raw_data = _pack_int4_signed(codes_orig)
        graph.initializer.append(wq)

        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{down_w_init.name}_dsq_scale", taken_names),
        )
        graph.initializer.append(ws)

        reduction_axis = 1 if down_transposed else 0
        dq_out = _unique_name(f"{down_w_init.name}_dsq_dq", taken_names)
        dq_node = onnx.helper.make_node(
            "DequantizeLinear",
            [wq.name, ws.name],
            [dq_out],
            name=_unique_name(f"{down_w_init.name}_dsq_dequant", taken_names),
            axis=reduction_axis,
            block_size=block_size,
        )
        graph.node.insert(
            next(i for i, n in enumerate(graph.node) if n is down_node), dq_node
        )
        for i, inp in enumerate(down_node.input):
            if inp == down_w_init.name:
                down_node.input[i] = dq_out

    return out


# ---------------------------------------------------------------------------
# Deviation-Aware Correction (DAC)
# ---------------------------------------------------------------------------


def _apply_ln_bias_correction(
    graph: onnx.GraphProto,
    ln_node: onnx.NodeProto,
    correction: np.ndarray,
    initializer_map: Dict[str, onnx.TensorProto],
    taken_names: "set[str]",
) -> None:
    if len(ln_node.input) >= 3 and ln_node.input[2]:
        bias_init = initializer_map.get(ln_node.input[2])
        if (
            bias_init is None
            or bias_init.data_type != onnx.TensorProto.FLOAT
            or list(bias_init.dims) != [correction.shape[0]]
        ):
            return
        bias = onnx.numpy_helper.to_array(bias_init).astype(np.float64)
        bias_init.CopyFrom(
            onnx.numpy_helper.from_array(
                (bias + correction).astype(np.float32), name=bias_init.name
            )
        )
        return

    new_bias = onnx.numpy_helper.from_array(
        correction.astype(np.float32),
        name=_unique_name(f"{ln_node.output[0]}_dac_bias", taken_names),
    )
    graph.initializer.append(new_bias)
    initializer_map[new_bias.name] = new_bias
    if len(ln_node.input) >= 3:
        ln_node.input[2] = new_bias.name
    else:
        ln_node.input.append(new_bias.name)


def apply_dac(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    min_expected_error_reduction: float = 0.5,
    correction_threshold: float = 1e-12,
) -> onnx.ModelProto:
    """Empirically measures, per channel and per ``LayerNormalization``
    node, how much of that channel's own quantization-induced output
    deviation is a consistent, directional *mean* shift (as opposed to
    unstructured noise), and folds the shift directly into that same
    LayerNormalization's own bias for every channel where the shift
    dominates -- see this module's own docstring for the technique.

    This mirrors :func:`onnxsim.correct_bias`'s own measurement
    machinery (run both models on the same calibration data, measure a
    per-channel mean deviation at every matched node, present in both
    models under the same output tensor name) but differs in exactly the
    way the paper's own Deviation-Aware Correction differs from plain bias
    correction: instead of adding a new correction term right after the
    layer whose output was measured, it is folded into the *following*
    LayerNormalization's own bias -- ``LayerNormalization`` already computes
    ``normalize(x) * scale + bias``, and adding a per-channel constant
    ``mu`` to its output is exactly equivalent to using ``bias + mu``, with
    no new node needed (the same zero-new-node philosophy
    :mod:`onnxsim.outlier_suppression`'s own Gamma Migration uses for its
    own per-channel scale). A ``LayerNormalization`` with no bias input at
    all gets one added (a plain new initializer, still no new node).

    Unlike the paper -- which selects entire LayerNorm layers to correct by
    hand, based on an empirical observation that post-attention layers see
    a much more consistent shift than pre-attention ones -- this applies the
    same underlying criterion the paper uses to justify that choice
    (its own theoretical result that a channel's expected squared-error
    reduction from correcting a mean shift is ``mu^2 / (mu^2 + sigma^2)``,
    the deviation's own noise-to-signal ratio) directly, per channel, on
    every ``LayerNormalization`` node present in both models: a channel is
    corrected only when that ratio reaches ``min_expected_error_reduction``.
    This generalizes to any transformer topology without needing to first
    identify which LayerNorm is structurally "post-attention" -- a
    pre-attention (or any other) LayerNorm's channels simply see a low
    ratio in practice and are left uncorrected, the same outcome the
    paper's own hand-picked selection produces.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path). Assumes ``quantized_model`` was
            produced without renaming any ``LayerNormalization`` node's own
            output tensor -- true of every onnxsim ``quantize_*``/``apply_*``
            function, including :func:`apply_dsq`.
    :param calibration_data: representative input batches to measure each
            LayerNormalization's own deviation on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative correction than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :param min_expected_error_reduction: only correct a channel whose
            measured deviation's own ``mu^2 / (mu^2 + sigma^2)`` reaches
            this fraction (in ``[0, 1)``) -- the paper's own closed-form
            expected squared-error reduction from applying the correction.
            The default, ``0.5``, corrects a channel only when the mean
            shift itself, not run-to-run noise, is the dominant source of
            that channel's own deviation.
    :param correction_threshold: skip a LayerNormalization whose largest
            per-channel correction (after the ratio gate above) never
            exceeds this in absolute value -- avoids a numerically-pointless
            edit, not an accuracy knob.
    :returns: ``quantized_model`` with a per-channel mean-shift correction
            folded into every measurably-shifted, gated-in
            ``LayerNormalization``'s own bias
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    quantized_ln_outputs = {
        n.output[0]
        for n in quantized_model.graph.node
        if n.op_type == "LayerNormalization" and n.output
    }
    candidates = [
        n.output[0]
        for n in float_model.graph.node
        if n.op_type == "LayerNormalization"
        and n.output
        and n.output[0] in quantized_ln_outputs
    ]
    if not candidates:
        return quantized_model

    float_probe = _add_probe_outputs(float_model, candidates)
    quantized_probe = _add_probe_outputs(quantized_model, candidates)

    sums: Dict[str, np.ndarray] = {}
    sumsqs: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for batch in calibration_data:
        float_out = backend.run_model(float_probe, batch, providers=providers)
        quantized_out = backend.run_model(quantized_probe, batch, providers=providers)
        for name in candidates:
            f = np.asarray(float_out[name], dtype=np.float64)
            q = np.asarray(quantized_out[name], dtype=np.float64)
            if f.shape != q.shape or f.ndim == 0:
                continue
            diff = f - q
            reduce_axes = tuple(range(diff.ndim - 1))
            s = diff.sum(axis=reduce_axes) if reduce_axes else diff
            ssq = (diff * diff).sum(axis=reduce_axes) if reduce_axes else diff * diff
            cnt = diff.size // diff.shape[-1]
            if name in sums:
                sums[name] = sums[name] + s
                sumsqs[name] = sumsqs[name] + ssq
                counts[name] += cnt
            else:
                sums[name] = s
                sumsqs[name] = ssq
                counts[name] = cnt

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    graph = corrected.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    node_by_output = {
        n.output[0]: n
        for n in graph.node
        if n.op_type == "LayerNormalization" and n.output
    }

    for name, total in sums.items():
        cnt = counts[name]
        mu = total / cnt
        var = np.maximum(sumsqs[name] / cnt - mu * mu, 0.0)
        expected_reduction = (mu * mu) / (mu * mu + var + 1e-12)
        correction = np.where(
            expected_reduction >= min_expected_error_reduction, mu, 0.0
        )
        if np.max(np.abs(correction)) <= correction_threshold:
            continue
        ln_node = node_by_output.get(name)
        if ln_node is None or len(ln_node.input) < 2:
            continue
        gamma_init = initializer_map.get(ln_node.input[1])
        if (
            gamma_init is None
            or gamma_init.data_type != onnx.TensorProto.FLOAT
            or list(gamma_init.dims) != [correction.shape[0]]
        ):
            continue
        _apply_ln_bias_correction(
            graph, ln_node, correction.astype(np.float32), initializer_map, taken_names
        )

    return corrected
