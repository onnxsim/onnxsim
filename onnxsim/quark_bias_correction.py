"""Bias correction the way Quark's ONNX flow does it
(``quark.onnx.algorithm.bc.bias_correction``), for the Q/DQ models
:mod:`onnxsim.quark_compat` emits.

:func:`onnxsim.bias_correction.correct_bias` (the general tool) adds a constant
after every layer using the end-to-end error of the whole model. Quark's
version is local and rewrites the *quantized bias constant* instead:

1. for each quantized ``Conv`` / ``Gemm``, the quantized model is run on the
   calibration data and the node's input (the dequantized activation) and its
   output after any following ``Relu`` / ``Clip`` / ``QuantizeLinear`` /
   ``DequantizeLinear`` chain are recorded;
2. the *float* layer (plus the float ``Relu`` / ``Clip`` that follows it) is run
   on those same quantized inputs, so the difference ``float - quantized``
   isolates this layer's own weight, bias and output rounding error;
3. its mean per output channel (4-D ``NCHW`` outputs over batch and space,
   2-D ``Gemm`` outputs over the batch; other ranks are skipped) is added to
   the dequantized bias, damped when large: with ``m = max|mean|`` and
   ``b = max|bias| / 256``, a correction with ``m > b`` and ``m > 0.1`` is
   scaled by ``b / m``;
4. the bias is re-quantized with its *existing* scale (``round(bias / scale)``
   as int32). Layers without a bias are left alone, as there is nothing to
   rewrite.

With power-of-two calibration (``XINT8``; ``pof2=True``) Quark does *not*
reuse the bias scale in step 4. It hands the corrected float bias to its
power-of-two quantizer, which derives a fresh scale (a MinMSE search over five
power-of-two candidates, from the bias *itself*) and a zero point, and writes the
resulting integer codes into the bias tensor -- without updating the bias's
``DequantizeLinear`` scale / zero point. Whenever the fresh scale differs from
the stored one the integer codes and the stored scale no longer describe the
same number: the corrected bias evaluates to ``code * stored_scale`` instead of
``code * fresh_scale`` (typically off by a power of two), and an asymmetric
fresh zero point shifts every code. That is a Quark quirk, but this module
reproduces it by default for parity (``quark_scale=True``) and warns when it
changes a bias's meaning; ``quark_scale=False`` keeps the stored scale and
writes ``round(bias / scale)`` instead, which is what the float intent says.
:func:`quark_pof2_quantize` is the re-derivation. (With ``Int32Bias=True`` the
same happens to the int32 biases, whose fresh power-of-two scale is about
2**-24 and whose codes are around 1e8: they match Quark's to float32 noise in
the measured mean error, 1e-5 relative, not bit for bit.)

Quark's other calibration methods: ``MinMax`` / ``Percentile`` write the
``round(bias / stored scale)`` int32 codes, and every other method (Entropy,
Distribution, LayerwisePercentile, ...) leaves the biases untouched
(``method=`` below).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

_CHAIN_Q = ("Relu", "Clip", "QuantizeLinear", "DequantizeLinear")
_CHAIN_F = ("Relu", "Clip")
_POF2_CANDIDATES = 5
# Quark's quantize_data ranges: the type's own, or the symmetric one
_RANGES = {
    np.dtype(np.int8): ((-128, 127), (-127, 127)),
    np.dtype(np.uint8): ((0, 255), (0, 255)),
    np.dtype(np.int16): ((-32768, 32767), (-32767, 32767)),
    np.dtype(np.uint16): ((0, 65535), (0, 65535)),
    np.dtype(np.int32): ((-(2**31), 2**31 - 1), (-(2**31 - 1), 2**31 - 1)),
}


def _pos(scale: Any) -> int:
    """Fixed-point position of a scale: ``round(-log2(scale))``, clamped."""
    s = min(max(float(scale), 2.0**-127), 2.0**127)
    return int(np.rint(-np.log2(s)))


def quark_pof2_quantize(
    data: np.ndarray, dtype: Any, symmetric: bool
) -> Tuple[np.ndarray, np.float32, int]:
    """Quark's power-of-two re-quantization of a corrected float bias:
    ``(codes, scale, zero_point)`` the way its ``quantize_data`` (``MinMSE``
    method) derives them from ``data`` alone.

    The scale of the (optionally symmetrized, always zero-including) range is
    rounded to a power of two ``2**-p``; the zero point follows from that;
    then the five candidates ``2**-(p-1) ... 2**-(p+3)`` are tried on the data
    and the one with the least squared error wins (the first on ties). The
    codes include the zero point and are clipped to the symmetric code range.
    """
    dt = np.dtype(dtype)
    (qmin, qmax), (smin, smax) = _RANGES[dt]
    if symmetric:
        qmin, qmax = smin, smax
    d = np.asarray(data, dtype=np.float32)
    lo = np.minimum(d.min() if d.size else np.float32(0), np.float32(0))
    hi = np.maximum(d.max() if d.size else np.float32(0), np.float32(0))
    big = np.float32(np.finfo(np.float32).max / 2)
    lo, hi = np.float32(max(lo, -big)), np.float32(min(hi, big))
    if symmetric:
        a = np.maximum(np.abs(lo), np.abs(hi))
        lo, hi = -a, a
    scale = np.float64(hi - lo) / (np.float64(qmax) - np.float64(qmin))
    if scale < np.finfo(np.float32).tiny:
        scale32, zp = np.float32(1.0), 0
    else:
        zp = int(np.round(np.float64(qmin) - np.float64(lo) / scale))
        scale32 = np.float32(scale)
    pos = _pos(scale32)
    pof2 = np.float32(2.0**-pos)
    new_lo = np.minimum((np.float32(qmin) - np.float32(zp)) * pof2, np.float32(0))
    # (an int32 range minus the float32 ratio is float64 in NumPy, so the
    # int32 zero point comes out as 1 when float32 rounds ``qmin``)
    ratio = new_lo / pof2
    zp = int(
        np.round(qmin - np.float64(ratio))
        if dt.itemsize >= 4
        else np.round(np.float32(qmin) - ratio)
    )
    if symmetric and dt == np.dtype(np.uint8) and zp == 127:
        zp = 128
    clip_lo, clip_hi = smin, smax
    best, best_scale, best_codes = np.inf, pof2, None
    for i in range(_POF2_CANDIDATES):
        s = np.float32(2.0 ** -(_pos(pof2) + i - 1))
        wide = np.float64 if dt == np.dtype(np.int32) else np.float32
        x = np.round(d.astype(np.float32) / s).astype(wide) + wide(zp)
        codes = np.clip(x, clip_lo, clip_hi)
        deq = (codes.astype(np.float32) - np.float32(zp)) * s
        diff = np.sum((deq - d) ** 2)
        if diff < best:
            best, best_scale, best_codes = diff, s, codes
    assert best_codes is not None
    return best_codes.astype(dt), best_scale, zp


def _consumers(graph: onnx.GraphProto, node: onnx.NodeProto) -> List[onnx.NodeProto]:
    outs = {o for o in node.output if o}
    return [n for n in graph.node if outs & set(n.input)]


def _chain_end(graph: onnx.GraphProto, node: onnx.NodeProto, ops: Sequence[str]):
    """The node after ``node`` along the first-consumer ``ops`` chain, and the
    chain's nodes."""
    chain = [node]
    nxt = _consumers(graph, node)
    while nxt and nxt[0].op_type in ops:
        chain.append(nxt[0])
        nxt = _consumers(graph, nxt[0])
    return chain


def _float_submodel(
    float_model: onnx.ModelProto, chain: Sequence[onnx.NodeProto]
) -> onnx.ModelProto:
    g = float_model.graph
    inits = {t.name: t for t in g.initializer}
    sub = onnx.ModelProto()
    sub.ir_version = float_model.ir_version
    sub.opset_import.extend(float_model.opset_import)
    sub.graph.name = "bc_sub"
    start = chain[0].input[0]
    sub.graph.input.append(
        onnx.helper.make_tensor_value_info(start, onnx.TensorProto.FLOAT, None)
    )
    sub.graph.output.append(onnx.ValueInfoProto(name=chain[-1].output[0]))
    seen = set()
    for n in chain:
        sub.graph.node.append(n)
        for x in n.input:
            if x in inits and x not in seen:
                seen.add(x)
                sub.graph.initializer.append(inits[x])
    return sub


def _session(model: onnx.ModelProto, providers: Optional[Sequence[str]]):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=so,
        providers=list(providers or ["CPUExecutionProvider"]),
    )


def _float_nodes(float_model: onnx.ModelProto) -> List[onnx.NodeProto]:
    return [n for n in float_model.graph.node if n.op_type in ("Conv", "Gemm")]


def correct_bias_quark(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]] = None,
    activation_symmetric: bool = False,
    method: str = "minmax",
    quark_scale: bool = True,
) -> onnx.ModelProto:
    """Return ``quant_model`` with Quark's bias correction applied (see the
    module docstring). Float and quantized ``Conv`` / ``Gemm`` nodes are paired
    by name, or by position when unnamed.

    :param method: the calibration method family, which decides how the
        corrected bias is written: ``"minmax"`` (also Quark's Percentile) --
        ``round(bias / stored scale)`` as int32; ``"pof2"`` (``XINT8``'s
        MinMSE / NonOverflow) -- Quark's power-of-two re-quantization
        (:func:`quark_pof2_quantize`) written without updating the scale;
        anything else -- biases left alone, as Quark does.
    :param activation_symmetric: Quark's ``ActivationSymmetric`` option, which
        picks the symmetric code range of the ``"pof2"`` re-quantization.
    :param quark_scale: with ``method="pof2"``, keep Quark's inconsistency (the
        codes belong to a freshly derived scale that is not stored, see the
        module docstring; warns when it changes a bias). ``False`` writes
        ``round(bias / stored scale)`` instead.
    """
    if method not in ("minmax", "pof2"):
        return quant_model
    qm = onnx.ModelProto()
    qm.CopyFrom(quant_model)
    qg = qm.graph
    if not calibration_data:
        return qm

    q_nodes = [n for n in qg.node if n.op_type in ("Conv", "Gemm")]
    f_nodes = _float_nodes(float_model)
    f_by_name = {n.name: n for n in f_nodes if n.name}
    producer = {o: n for n in qg.node for o in n.output}
    inits = {t.name: t for t in qg.initializer}

    plan: List[Tuple[onnx.NodeProto, onnx.NodeProto, str, str, str]] = []
    for i, qn in enumerate(q_nodes):
        fn = f_by_name.get(qn.name) if qn.name else None
        if fn is None and not qn.name and len(f_nodes) == len(q_nodes):
            fn = f_nodes[i]
        if fn is None or len(qn.input) != 3 or not qn.input[2]:
            continue
        end = _chain_end(qg, qn, _CHAIN_Q)[-1]
        plan.append((qn, fn, qn.input[0], end.output[0], qn.input[2]))
    if not plan:
        return qm

    probe = onnx.ModelProto()
    probe.CopyFrom(qm)
    have = {o.name for o in probe.graph.output}
    for _, _, tin, tout, _ in plan:
        for t in (tin, tout):
            if t not in have:
                have.add(t)
                probe.graph.output.append(onnx.ValueInfoProto(name=t))
    sess = _session(probe, providers)
    names = [o.name for o in sess.get_outputs()]
    seen: Dict[str, List[np.ndarray]] = {}
    for batch in calibration_data:
        for k, v in zip(names, sess.run(None, batch)):
            seen.setdefault(k, []).append(v)

    for qn, fn, tin, tout, bias_t in plan:
        dq = producer.get(bias_t)
        if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 3:
            continue
        chain = _chain_end(float_model.graph, fn, _CHAIN_F)
        fsess = _session(_float_submodel(float_model, chain), providers)
        f_in = fsess.get_inputs()[0].name
        f_out = [fsess.run(None, {f_in: x})[0] for x in seen[tin]]
        q_out = seen[tout]
        try:
            fo, qo = np.array(f_out), np.array(q_out)
        except ValueError:
            continue
        if qo.ndim == 5:
            diff = np.mean(fo - qo, axis=(0, 1, 3, 4))
        elif qo.ndim == 3:
            diff = np.mean(fo - qo, axis=(0, 1))
        else:
            continue
        b_name, s_name, z_name = dq.input[0], dq.input[1], dq.input[2]
        if not all(x in inits for x in (b_name, s_name, z_name)):
            continue
        b_init, s_init, z_init = inits[b_name], inits[s_name], inits[z_name]
        scale = numpy_helper.to_array(s_init)
        zp = numpy_helper.to_array(z_init)
        bias_q = numpy_helper.to_array(b_init)
        bias_f = ((bias_q.astype(np.float32) - zp.astype(np.float32)) * scale).astype(
            np.float32
        )
        max_diff = np.max(np.abs(diff))
        max_bias = np.max(np.abs(bias_f))
        damp = 1
        plus = max_bias / 256
        if max_diff > plus and max_diff > 0.1:
            damp = damp * plus / max_diff
        new_f = bias_f + diff * damp
        if method == "pof2" and quark_scale:
            q, new_scale, new_zp = quark_pof2_quantize(
                new_f, bias_q.dtype, activation_symmetric
            )
            if not np.array_equal(
                np.float32(new_scale), scale.reshape(-1)[:1]
            ) or new_zp != int(zp.reshape(-1)[0]):
                warnings.warn(
                    f"BiasCorrection of {qn.name or qn.output[0]!r} follows Quark's "
                    f"power-of-two flow: the corrected bias codes belong to scale "
                    f"{float(new_scale):g} / zero point {new_zp}, but the stored "
                    f"DequantizeLinear keeps {float(scale.reshape(-1)[0]):g} / "
                    f"{int(zp.reshape(-1)[0])}, so the bias evaluates differently "
                    "from the corrected float value (a Quark quirk, reproduced "
                    "for parity; extra_options['BiasCorrectionStoredScale']=True "
                    "keeps the stored scale)",
                    stacklevel=2,
                )
        else:
            q = (np.asarray(new_f) / scale).round()
            if bias_q.dtype == np.int32:
                q = q.astype(np.int32)
            else:  # int8 bias: keep the scale, clip to range
                info = np.iinfo(bias_q.dtype)
                q = np.clip(q + zp, info.min, info.max).astype(bias_q.dtype)
        b_init.CopyFrom(numpy_helper.from_array(q.reshape(b_init.dims), b_name))
    return qm


__all__: Any = ["correct_bias_quark", "quark_pof2_quantize"]
