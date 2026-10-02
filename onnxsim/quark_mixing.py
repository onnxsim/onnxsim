"""In-place precision mixing of a quantized model -- a port of the graph surgery
of AMD Quark's ``MixingStrategy`` (``quark.onnx.algorithm.mprecision``).
Independent implementation: Quark's source was read for the contract.

Quark's AutoMixprecision does not re-quantize anything: it edits the *quantized*
baseline model in place, one node at a time, and :class:`QuarkMixer` does the
same on any model whose quantizers have Quark's shape -- ``QuantizeLinear`` /
``DequantizeLinear`` pairs (``ExtendedQuantizeLinear`` /
``ExtendedDequantizeLinear`` for the half types), the ``com.amd.quark``
``BFPQuantizeDequantize`` / ``MXQuantizeDequantize`` nodes, and constants held
as a ``DequantizeLinear`` over their codes (the quantizer folded the ``Q``).
So a mix can go between any two of integer, ``float16`` / ``bfloat16``, BFP and
MX precisions, whichever way the baseline was built.

For a node, Quark handles its slots in this order: activation input 0, the
weight (a constant second input) or second activation, the bias (and then
refreshes the int32 bias scale) or third activation, further activations, the
outputs. A slot's current quantizer and its target decide the edit:

=============== =========================== ================================
current         target                      edit
=============== =========================== ================================
Q/DQ pair       integer / half              new scale and zero point (and, for a
                                            folded constant, new codes); the
                                            op type / domain follow the dtype
Q/DQ pair       BFP / MX                    the pair becomes a ``BFP`` / ``MX``
                                            node (a folded constant is
                                            dequantized to float first)
BFP / MX node   BFP / MX                    a fresh node (``<name>_Mixed``,
                                            default block axis)
BFP / MX node   integer / half              a Q/DQ pair (``<name>_Mixed_Q`` /
                                            ``_DQ``)
=============== =========================== ================================

Scales and zero points of an integer or half target come from the tensor's
calibrated range (activations), the float constant, or the dequantized
constant, with Quark's ``compute_scale_zp`` / ``compute_scale_zp_fp``
(including its power-of-two rounding for ``pof2`` specs). ``shared_param_mode``
is Quark's: scale / zero point initializers shared with nodes outside the
promoted pair are either kept (``"propagate"``: every sharing Q / DQ takes the
new op type / domain, and sees the new values) or copied for the pair
(``"unshare"``).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

COP_DOMAIN = "com.amd.quark"
Q_OPS = ("QuantizeLinear", "ExtendedQuantizeLinear")
DQ_OPS = ("DequantizeLinear", "ExtendedDequantizeLinear")
FN_OPS = ("BFPQuantizeDequantize", "MXQuantizeDequantize")

INT_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32")
HALF_DTYPES = ("float16", "bfloat16")

_TENSOR_TYPE = {
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
    "int16": TensorProto.INT16,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "uint32": TensorProto.UINT32,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
}
_NP = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
}
#: Quark's ``ONNX_INT_TYPE_RANGE`` / ``ONNX_INT_TYPE_SYMMETRIC_RANGE``
_RANGE = {
    "uint8": (0, 255),
    "int8": (-128, 127),
    "uint16": (0, 65535),
    "int16": (-32768, 32767),
    "uint32": (0, 4294967295),
    "int32": (-2147483648, 2147483647),
}
_SYM_RANGE = {
    "int8": (-127, 127),
    "int16": (-32767, 32767),
    "int32": (-2147483647, 2147483647),
}

#: ``(Precision, ...)``: ``(dtype, symmetric)`` or ``(dtype, symmetric, pof2)``
Precision = Tuple[Any, ...]


def kind_of(dtype: str) -> str:
    """``"int"`` / ``"half"`` / ``"block"`` (BFP and MX formats)."""
    if dtype in INT_DTYPES:
        return "int"
    if dtype in HALF_DTYPES:
        return "half"
    from onnxsim.quark_fakequant_graph import node_spec

    node_spec(dtype)  # raises for an unknown dtype
    return "block"


def unpack(prec: Precision) -> Tuple[str, bool, bool]:
    """``(dtype, symmetric, power of two)`` of a precision tuple."""
    dtype = prec[0]
    sym = bool(prec[1]) if len(prec) > 1 and prec[1] is not None else False
    pof2 = bool(prec[2]) if len(prec) > 2 else False
    return dtype, sym, pof2


def _np_half(dtype: str) -> Any:
    return onnx.helper.tensor_dtype_to_np_dtype(_TENSOR_TYPE[dtype])


def _qmin_qmax(dtype: str, symmetric: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Quark's ``get_qmin_qmax_for_qType`` (``reduce_range=False``)."""
    if dtype == "float16":
        return np.array(-65504.0, np.float32), np.array(65504.0, np.float32)
    if dtype == "bfloat16":
        return np.array(-3.38953139e38, np.float32), np.array(3.38953139e38, np.float32)
    table = _SYM_RANGE if symmetric and dtype in _SYM_RANGE else _RANGE
    lo, hi = table[dtype]
    return np.array(lo, _NP[dtype]), np.array(hi, _NP[dtype])


def _scale2pos(scale: float) -> int:
    scale = min(max(scale, float(2**-127)), float(2**127))
    return int(np.rint(-np.log2(scale)))


def _pos2scale(pos: int) -> float:
    return float(np.power(2.0, -pos))


def compute_scale_zp(
    rmin: np.ndarray,
    rmax: np.ndarray,
    dtype: str,
    symmetric: bool,
    pof2: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(zero_point, scale)`` of Quark's ``compute_scale_zp`` for an integer
    ``dtype`` over the float range ``[rmin, rmax]`` (``pof2``: Quark's
    ``PowerOfTwoMethod`` rounding of the scale to the nearest power of two)."""
    qmin, qmax = _qmin_qmax(dtype, symmetric)
    rmin = np.minimum(rmin, np.array(0, dtype=rmin.dtype))
    rmax = np.maximum(rmax, np.array(0, dtype=rmax.dtype))
    f32max = np.finfo(np.float32).max
    if rmin == -np.inf or rmin < -f32max / 2:
        rmin = np.full_like(rmin, -f32max / 2)
    if rmax == np.inf or rmax > f32max / 2:
        rmax = np.full_like(rmax, f32max / 2)
    if symmetric:
        absmax = np.maximum(np.abs(rmin), np.abs(rmax))
        rmin = -absmax
        rmax = +absmax
    dr = np.array(rmax - rmin, dtype=np.float64)
    dq = np.array(qmax, dtype=np.float64) - np.array(qmin, dtype=np.float64)
    scale = np.array(dr / dq)
    if np.isnan(scale):
        raise ValueError("NaN detected, please check the correctness of the model")
    if scale < np.finfo(rmax.dtype).tiny:
        scale = np.array(1.0, dtype=rmax.dtype)
        zero_point = np.array(0, dtype=qmin.dtype)
    else:
        zero_point = np.array(np.round(qmin - rmin / scale), dtype=qmin.dtype)
        scale = scale.astype(rmax.dtype)
    if not pof2:
        if symmetric and dtype == "uint8" and zero_point == 127:
            zero_point = np.array(128, dtype=qmin.dtype)
        return zero_point, scale
    pos = _scale2pos(scale.item())
    pof2_scale = np.array(_pos2scale(pos), dtype=scale.dtype)
    new_rmin = np.minimum(
        (qmin.astype(np.float32) - zero_point.astype(np.float32)) * pof2_scale,
        np.array(0, dtype=rmin.dtype),
    )
    new_zp = np.array(np.round(qmin - new_rmin / pof2_scale), dtype=qmin.dtype)
    if symmetric and dtype == "uint8" and new_zp == 127:
        new_zp = np.array(128, dtype=qmin.dtype)
    return new_zp, pof2_scale


def compute_scale_zp_fp(
    rmin: np.ndarray, rmax: np.ndarray, dtype: str, symmetric: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """``(zero_point, scale)`` of Quark's ``compute_scale_zp_fp`` for
    ``float16`` / ``bfloat16``: scale 1.0, the zero point 0 (symmetric) or the
    type's lowest value."""
    qmin, qmax = _qmin_qmax(dtype, symmetric)
    rmin = np.minimum(rmin, np.array(0, dtype=rmin.dtype))
    rmax = np.maximum(rmax, np.array(0, dtype=rmax.dtype))
    f32max = np.finfo(np.float32).max
    if rmin == -np.inf or rmin < -f32max / 2:
        rmin = np.full_like(rmin, -f32max / 2)
    if rmax == np.inf or rmax > f32max / 2:
        rmax = np.full_like(rmax, f32max / 2)
    if symmetric:
        absmax = np.maximum(np.abs(rmin), np.abs(rmax))
        rmin = -absmax
        rmax = +absmax
    scale = np.array(1.0, dtype=np.float32)
    if scale < np.finfo(rmax.dtype).tiny:
        return np.array(0, dtype=scale.dtype), np.array(1.0, dtype=rmax.dtype)
    scale = scale.astype(rmax.dtype)
    if symmetric:
        zero_point = np.array(0, dtype=scale.dtype)
    else:
        zero_point = np.array(np.round(qmin - rmin / scale), dtype=scale.dtype)
    return zero_point, scale


def fake_ranges(model: onnx.ModelProto) -> Dict[str, Tuple[float, float]]:
    """Quark's ``fake_calibration``: the range ``[0, 1]`` for every node input /
    output that is not an initializer -- what its AutoMixprecision is given
    when the baseline uses a float / block format (no calibration is run)."""
    inits = {t.name for t in model.graph.initializer}
    out: Dict[str, Tuple[float, float]] = {}
    for n in model.graph.node:
        for x in list(n.input) + list(n.output):
            if x not in inits and x not in out:
                out[x] = (0.0, 1.0)
    return out


def _from_array(value: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(np.asarray(value), name)


class QuarkMixer:
    """Promotes nodes of a quantized model to a target precision in place.

    :param model: the quantized baseline (copied; not modified)
    :param ranges: ``f(float tensor name) -> (lo, hi) or None``, a mapping, or
            ``None`` -- the calibrated range of an activation
    :param shared_param_mode: ``"propagate"`` or ``"unshare"`` (see the module
            docstring)
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        ranges: Any = None,
        shared_param_mode: str = "propagate",
    ) -> None:
        if shared_param_mode not in ("propagate", "unshare"):
            raise ValueError("shared_param_mode must be 'propagate' or 'unshare'")
        self.mode = shared_param_mode
        self._meta = onnx.ModelProto()
        self._meta.CopyFrom(model)
        del self._meta.graph.node[:]
        del self._meta.graph.initializer[:]
        self.nodes: List[onnx.NodeProto] = []
        for n in model.graph.node:
            c = onnx.NodeProto()
            c.CopyFrom(n)
            self.nodes.append(c)
        self.inits: Dict[str, TensorProto] = {}
        for t in model.graph.initializer:
            c = TensorProto()
            c.CopyFrom(t)
            self.inits[t.name] = c
        self.graph_outputs = {o.name for o in model.graph.output}
        self._range: Callable[[str], Any]
        if ranges is None:

            def no_range(name: str) -> Any:
                return None

            self._range = no_range
        elif callable(ranges):
            self._range = ranges
        else:
            table = ranges

            def table_range(name: str) -> Any:
                return table.get(name)

            self._range = table_range
        self.promoted_tensors: Set[str] = set()
        self.promoted_nodes: Set[str] = set()

    # -- graph access ------------------------------------------------------------

    def _maps(self):
        out_to_node: Dict[str, onnx.NodeProto] = {}
        in_to_nodes: Dict[str, List[onnx.NodeProto]] = {}
        for n in self.nodes:
            for o in n.output:
                out_to_node[o] = n
            for x in n.input:
                in_to_nodes.setdefault(x, []).append(n)
        return out_to_node, in_to_nodes

    def _remove(self, node: onnx.NodeProto) -> None:
        for i, n in enumerate(self.nodes):
            if n is node:
                del self.nodes[i]
                return

    def _set_init(self, name: str, value: np.ndarray) -> None:
        self.inits.pop(name, None)
        self.inits[name] = _from_array(value, name)

    def _array(self, name: str) -> Optional[np.ndarray]:
        t = self.inits.get(name)
        return None if t is None else numpy_helper.to_array(t)

    # -- structure discovery (Quark's ``ONNXQuantizedModel``) ----------------------

    def _input_slot(self, tensor: str, out_to_node: Dict[str, onnx.NodeProto]):
        prod = out_to_node.get(tensor)
        if prod is not None and prod.op_type in FN_OPS:
            return (prod,)
        return self._find_input_qdq(tensor, out_to_node)

    @staticmethod
    def _find_input_qdq(tensor: str, out_to_node: Dict[str, onnx.NodeProto]):
        if tensor not in out_to_node:
            return (None, None)
        dq = out_to_node[tensor]
        if dq.op_type not in DQ_OPS:
            return (None, None)
        if dq.input[0] not in out_to_node:
            return (dq, None)  # a folded Q
        q = out_to_node[dq.input[0]]
        if q.op_type not in Q_OPS:
            return (dq, None)
        return (dq, q)

    def _output_slot(self, tensor: str, in_to_nodes: Dict[str, List[onnx.NodeProto]]):
        first = in_to_nodes.get(tensor)
        if first and first[0].op_type in FN_OPS:
            return (first[0],)
        if tensor not in in_to_nodes:
            return (None, None)
        assert len(in_to_nodes[tensor]) == 1, (
            f"output {tensor!r} is read by several nodes (Quark asserts this too)"
        )
        q = in_to_nodes[tensor][0]
        if q.op_type not in Q_OPS:
            return (None, None)
        if q.output[0] not in in_to_nodes:
            return (None, q)
        dq = in_to_nodes[q.output[0]][0]
        if dq.op_type not in DQ_OPS:
            return (None, q)
        return (dq, q)

    # -- scale / zero point ---------------------------------------------------------

    def _dequantize_weight(self, dq: onnx.NodeProto) -> Optional[np.ndarray]:
        """Float data of a DQ-only constant, ``None`` for a runtime tensor."""
        codes = self._array(dq.input[0])
        if codes is None or len(dq.input) < 3:
            return None
        scale = self._array(dq.input[1])
        zp = self._array(dq.input[2])
        if scale is None or zp is None:
            return None
        return (codes.astype(np.float32) - zp.astype(np.float32)) * scale.astype(
            np.float32
        )

    def _scale_zp_from_range(
        self,
        rmin: np.ndarray,
        rmax: np.ndarray,
        prec: Tuple[str, bool, bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        dtype, symmetric, pof2 = prec
        if kind_of(dtype) == "half":
            zero_point, scale = compute_scale_zp_fp(rmin, rmax, dtype, symmetric)
            zp_dtype = _np_half(dtype)
        else:
            zero_point, scale = compute_scale_zp(rmin, rmax, dtype, symmetric, pof2)
            zp_dtype = _NP[dtype]
        scale_np = np.asarray(scale, dtype=rmin.dtype).reshape(())
        zp_np = np.asarray(zero_point, dtype=zp_dtype).reshape(())
        return scale_np, zp_np

    def _compute_scale_zp(
        self,
        float_tensor: Optional[str],
        dq: Optional[onnx.NodeProto],
        prec: Tuple[str, bool, bool],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if float_tensor:
            r = self._range(float_tensor)
            if r is not None:
                rmin = np.asarray(r[0])
                rmax = np.asarray(r[1])
                if rmin.dtype != np.float16:  # Quark's calibrators report float32
                    rmin, rmax = rmin.astype(np.float32), rmax.astype(np.float32)
                if dq is not None and len(dq.input) > 1:
                    existing = self._array(dq.input[1])
                    if existing is not None and existing.dtype != rmin.dtype:
                        rmin = rmin.astype(existing.dtype)
                        rmax = rmax.astype(existing.dtype)
                return self._scale_zp_from_range(rmin, rmax, prec)
            init = self._array(float_tensor)
            if init is not None:
                return self._scale_zp_from_range(
                    np.asarray(init.min(), dtype=init.dtype),
                    np.asarray(init.max(), dtype=init.dtype),
                    prec,
                )
        if dq is not None:
            data = self._dequantize_weight(dq)
            if data is not None:
                return self._scale_zp_from_range(
                    np.asarray(data.min(), dtype=data.dtype),
                    np.asarray(data.max(), dtype=data.dtype),
                    prec,
                )
        return None

    # -- the edits ------------------------------------------------------------------

    @staticmethod
    def _qdq_name_domain(dtype: str) -> Tuple[str, str, str]:
        if dtype in ("int8", "uint8"):
            return "QuantizeLinear", "DequantizeLinear", ""
        if dtype in ("int16", "uint16", "int32"):
            return "QuantizeLinear", "DequantizeLinear", "com.microsoft"
        return "ExtendedQuantizeLinear", "ExtendedDequantizeLinear", COP_DOMAIN

    def _make_fn(self, dtype: str, x: str, y: str, name: str) -> onnx.NodeProto:
        from onnxsim.quark_fakequant_graph import node_spec

        op, attrs = node_spec(dtype, 1)
        return helper.make_node(
            op,
            [x],
            [y],
            name=name,
            domain=COP_DOMAIN,
            **attrs,  # type: ignore[arg-type]
        )

    def _apply_scale_zp(
        self,
        dq: onnx.NodeProto,
        q: Optional[onnx.NodeProto],
        scale_np: np.ndarray,
        zp_np: np.ndarray,
        dtype: str,
        symmetric: bool,
    ) -> None:
        float_data = self._dequantize_weight(dq) if q is None else None
        q_name, dq_name, domain = self._qdq_name_domain(dtype)
        dq.op_type = dq_name
        dq.domain = domain
        if q is not None:
            q.op_type = q_name
            q.domain = domain
        promoted = [n for n in (dq, q) if n is not None]
        promoted_ids = {id(n) for n in promoted}
        if self.mode == "unshare":
            for idx in (1, 2):
                name = dq.input[idx]
                users = [
                    n
                    for n in self.nodes
                    if len(n.input) > idx
                    and n.input[idx] == name
                    and id(n) not in promoted_ids
                ]
                if not users or name not in self.inits:
                    continue
                new_name, counter = name + "_mp", 0
                while new_name in self.inits:
                    counter += 1
                    new_name = f"{name}_mp{counter}"
                copy = TensorProto()
                copy.CopyFrom(self.inits[name])
                copy.name = new_name
                self.inits[new_name] = copy
                for n in promoted:
                    if len(n.input) > idx and n.input[idx] == name:
                        n.input[idx] = new_name
        else:  # propagate
            for idx in (1, 2):
                name = dq.input[idx]
                for other in self.nodes:
                    if id(other) in promoted_ids:
                        continue
                    if other.op_type not in Q_OPS and other.op_type not in DQ_OPS:
                        continue
                    if len(other.input) <= idx or other.input[idx] != name:
                        continue
                    other.op_type = q_name if other.op_type in Q_OPS else dq_name
                    other.domain = domain
        self._set_init(dq.input[1], scale_np)
        self._set_init(dq.input[2], zp_np)
        if q is not None:
            if q.input[1] != dq.input[1]:
                self._set_init(q.input[1], scale_np)
            if q.input[2] != dq.input[2]:
                self._set_init(q.input[2], zp_np)
        if float_data is not None:
            if kind_of(dtype) == "half":
                new_quant = float_data.astype(_np_half(dtype))
            else:
                qmin, qmax = _qmin_qmax(dtype, symmetric)
                new_quant = np.clip(
                    np.round(float_data / scale_np + zp_np), qmin, qmax
                ).astype(_NP[dtype])
            if dq.input[0] in self.inits:
                self.inits.pop(dq.input[0])
                self.inits[dq.input[0]] = _from_array(new_quant, dq.input[0])

    def _refine_bias_scale(self, node: onnx.NodeProto) -> None:
        """Quark's ``_refine_bias_scale``: the bias scale becomes
        ``input_scale * weight_scale`` (the codes of a folded bias are rescaled
        and truncated)."""
        if len(node.input) != 3:
            return
        out_to_node, _ = self._maps()
        dq, q = self._find_input_qdq(node.input[2], out_to_node)
        if dq is None or len(dq.input) < 3:
            return
        zp = self.inits.get(dq.input[2])
        if zp is None or zp.data_type != TensorProto.INT32:
            return
        input_dq, _ = self._find_input_qdq(node.input[0], out_to_node)
        if input_dq is None:
            return
        input_scale = self._array(input_dq.input[1])
        if input_scale is None:
            return
        weight_dq, _ = self._find_input_qdq(node.input[1], out_to_node)
        if weight_dq is None:
            return
        weight_scale = self._array(weight_dq.input[1])
        if weight_scale is None:
            return
        new_scale = (input_scale * weight_scale).astype(input_scale.dtype)
        if q is None:
            old_scale = self._array(dq.input[1])
            bias = self._array(dq.input[0])
            assert old_scale is not None and bias is not None
            bias = bias.astype(np.float32)
            bias = bias * old_scale / new_scale
            with np.errstate(invalid="ignore", over="ignore"):
                bias = bias.astype(np.int32)
            self.inits[dq.input[0]].CopyFrom(_from_array(bias, dq.input[0]))
        elif q.input[1] != dq.input[1]:
            self.inits[q.input[1]].CopyFrom(_from_array(new_scale, q.input[1]))
        self.inits[dq.input[1]].CopyFrom(_from_array(new_scale, dq.input[1]))
        zp_shape = tuple(self.inits[dq.input[2]].dims)
        if zp_shape != new_scale.shape:
            # per-channel weights moved to one scale: Quark leaves the old
            # zero-point vector behind, which ONNX Runtime refuses to run
            self.inits[dq.input[2]].CopyFrom(
                _from_array(np.zeros(new_scale.shape, np.int32), dq.input[2])
            )

    def _promote_qdq_with_qdq(
        self,
        dq: onnx.NodeProto,
        q: Optional[onnx.NodeProto],
        prec: Tuple[str, bool, bool],
    ) -> None:
        if dq is not None and dq.output[0] in self.graph_outputs:
            float_tensor: Optional[str] = dq.output[0]
        else:
            float_tensor = q.input[0] if q is not None else None
        res = self._compute_scale_zp(float_tensor, dq, prec)
        if res is None:
            warnings.warn(
                f"cannot promote the quant node {dq.name!r}: no calibration "
                "range or initializer found",
                UserWarning,
                stacklevel=3,
            )
            return
        scale_np, zp_np = res
        self._apply_scale_zp(dq, q, scale_np, zp_np, prec[0], prec[1])

    def _promote_qdq_with_fn(
        self,
        dq: onnx.NodeProto,
        q: Optional[onnx.NodeProto],
        dtype: str,
        is_output: bool,
    ) -> None:
        name = dq.name + "_Mixed_fn"
        if is_output:
            original = dq.output[0]
            if original in self.graph_outputs:
                fn_out = original
            else:
                fn_out = name + "_output"
                for n in self.nodes:
                    for i, x in enumerate(n.input):
                        if x == original:
                            n.input[i] = fn_out
            assert q is not None
            self.nodes.append(self._make_fn(dtype, q.input[0], fn_out, name))
            self._remove(q)
            self._remove(dq)
            return
        if q is not None:
            upstream = q.input[0]
        else:
            data = self._dequantize_weight(dq)
            if data is not None:
                upstream = dq.input[0] + "_float_Mixed"
                if upstream not in self.inits:
                    self.inits[upstream] = _from_array(data, upstream)
            else:
                upstream = dq.input[0]
        self.nodes.append(self._make_fn(dtype, upstream, dq.output[0], name))
        if q is not None:
            self._remove(q)
        self._remove(dq)

    def _promote_fn_slot(
        self, fn: onnx.NodeProto, prec: Tuple[str, bool, bool]
    ) -> None:
        dtype = prec[0]
        if kind_of(dtype) == "block":
            self.nodes.append(
                self._make_fn(dtype, fn.input[0], fn.output[0], fn.name + "_Mixed")
            )
            self._remove(fn)
            return
        float_tensor = (
            fn.output[0] if fn.output[0] in self.graph_outputs else fn.input[0]
        )
        res = self._compute_scale_zp(float_tensor, fn, prec)
        if res is None:
            warnings.warn(
                f"cannot promote the quant node {fn.name!r} to QDQ nodes: no "
                "calibration range or initializer found",
                UserWarning,
                stacklevel=3,
            )
            return
        scale_np, zp_np = res
        q_name, dq_name, domain = self._qdq_name_domain(dtype)
        base = fn.name + "_Mixed"
        scale_name, zp_name, q_out = (
            base + "_scale",
            base + "_zero_point",
            base + "_quantized",
        )
        for nm, val in ((scale_name, scale_np), (zp_name, zp_np)):
            if nm not in self.inits:
                self.inits[nm] = _from_array(val, nm)
        self.nodes.append(
            helper.make_node(
                q_name,
                [fn.input[0], scale_name, zp_name],
                [q_out],
                base + "_Q",
                domain=domain,
            )
        )
        self.nodes.append(
            helper.make_node(
                dq_name,
                [q_out, scale_name, zp_name],
                [fn.output[0]],
                base + "_DQ",
                domain=domain,
            )
        )
        self._remove(fn)

    def _handle_slot(
        self,
        slot: tuple,
        prec: Optional[Precision],
        is_output: bool = False,
    ) -> None:
        if prec is None:
            return
        p = unpack(prec)
        dtype = p[0]
        if len(slot) == 1 and slot[0] is not None:
            fn = slot[0]
            self._promote_fn_slot(fn, p)
            self.promoted_tensors.add(fn.input[0])
        elif len(slot) == 2 and slot[0] is not None:
            dq, q = slot
            if kind_of(dtype) == "block":
                self._promote_qdq_with_fn(dq, q, dtype, is_output)
            else:
                self._promote_qdq_with_qdq(dq, q, p)
            if q is not None:
                self.promoted_tensors.add(q.input[0])

    def promote_node(self, node_name: str, spec: Any) -> None:
        """Promote one node to ``spec`` (a ``TargetSpec``: ``inputs`` /
        ``outputs`` / ``weight`` / ``bias`` precisions, ``None`` = untouched)."""
        if not (spec.inputs or spec.weight or spec.bias or spec.outputs):
            return
        target = next(
            (
                n
                for n in self.nodes
                if (n.name or (n.output[0] if n.output else "")) == node_name
            ),
            None,
        )
        if target is None:
            return
        out_to_node, in_to_nodes = self._maps()
        input_slots = [self._input_slot(t, out_to_node) for t in target.input]
        output_slots = [self._output_slot(t, in_to_nodes) for t in target.output]
        self.promoted_nodes.add(node_name)

        def is_const(slot: tuple) -> bool:
            if len(slot) == 1 and slot[0] is not None:
                return slot[0].input[0] in self.inits
            if len(slot) == 2:
                dq, q = slot
                ref = q if q is not None else dq
                return (
                    ref is not None and bool(ref.input) and ref.input[0] in self.inits
                )
            return False

        if len(input_slots) >= 1:
            self._handle_slot(input_slots[0], spec.inputs)
        if len(input_slots) >= 2:
            self._handle_slot(
                input_slots[1], spec.weight if is_const(input_slots[1]) else spec.inputs
            )
        if len(input_slots) >= 3:
            if is_const(input_slots[2]):
                self._handle_slot(input_slots[2], spec.bias)
                self._refine_bias_scale(target)
            else:
                self._handle_slot(input_slots[2], spec.inputs)
        for slot in input_slots[3:]:
            self._handle_slot(slot, spec.inputs)
        for slot in output_slots:
            self._handle_slot(slot, spec.outputs, is_output=True)

    # -- result -----------------------------------------------------------------------

    def result(self) -> onnx.ModelProto:
        """The edited model: nodes topologically sorted, unused initializers
        dropped (Quark runs ``clean_initializers`` and ``topological_sort``)."""
        used = {x for n in self.nodes for x in n.input if x}
        used |= self.graph_outputs
        m = onnx.ModelProto()
        m.CopyFrom(self._meta)
        for n in _toposort(
            self.nodes, set(self.inits) | {v.name for v in m.graph.input}
        ):
            m.graph.node.append(n)
        for name, t in self.inits.items():
            if name in used:
                m.graph.initializer.append(t)
        # value_info of a tensor a quantizer now writes in another dtype is stale
        retyped = {
            o
            for n in self.nodes
            if n.op_type in Q_OPS + DQ_OPS + FN_OPS
            for o in n.output
        }
        keep = [v for v in m.graph.value_info if v.name not in retyped]
        del m.graph.value_info[:]
        m.graph.value_info.extend(keep)
        domains = {o.domain for o in m.opset_import}
        for n in self.nodes:
            if n.domain and n.domain not in domains:
                m.opset_import.append(helper.make_opsetid(n.domain, 1))
                domains.add(n.domain)
        return m


def _toposort(nodes: List[onnx.NodeProto], known: Set[str]) -> List[onnx.NodeProto]:
    """Stable topological order (nodes keep their relative order when they are
    already ready)."""
    produced: Set[str] = set()
    for n in nodes:
        produced.update(o for o in n.output if o)
    done = set(known) | {x for n in nodes for x in n.input if x and x not in produced}
    todo = list(nodes)
    out: List[onnx.NodeProto] = []
    while todo:
        rest = []
        progressed = False
        for n in todo:
            if all((not x) or x in done for x in n.input):
                out.append(n)
                done.update(o for o in n.output if o)
                progressed = True
            else:
                rest.append(n)
        if not progressed:
            raise ValueError("the graph is not a DAG")
        todo = rest
    return out


__all__ = [
    "QuarkMixer",
    "compute_scale_zp",
    "compute_scale_zp_fp",
    "fake_ranges",
    "kind_of",
]
