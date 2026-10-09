"""Pulsar2's MinMax quantization parameters without running Pulsar2.

``calibrate(model, samples)`` returns, for every tensor of a small float ONNX
graph, the scale, zero point and signedness that Pulsar2 7.0-lite (AX650,
``calibration_method: MinMax``) writes to ``quant_axmodel.json`` for the same
graph and calibration samples. ``stitch_calibration`` turns the result into
the ``scales`` / ``zero_points`` / ``signed`` arguments of
``graph_stitch.stitch_model``, so a fused graph can be built from standalone
programs with no Pulsar2 run on the fused graph at all
(``docs/axera-pulsar-free-calibration.md``).

Rules
-----

Each rule was read off ``quant_axmodel.json`` of native builds and is named in
the result (``TensorQuant.rule``):

``UNSIGNED``
    An activation (graph input or node output). u8. The range over all
    calibration samples is widened to include 0; ``lo`` and ``hi`` are float32
    values, ``s = float32((float64(hi) - float64(lo)) / 255)`` and
    ``zp = rint(-lo / s)`` (float64, on the float32 ``s``).
``SIGNED_ACT``
    An activation that is an operand of a MatMul whose two operands are both
    activations. s8, ``s = float32(max|x|) / float32(127.5)``, zero point 0.
    A Softmax output read by such a MatMul follows this rule too.
``CONST_U8``
    An initializer read by an element-wise op. u8, the ``UNSIGNED`` formula
    over the constant's own values (so a positive constant such as an epsilon
    of 1e-5 gets ``s = 1e-5 / 255`` and code 255).
``WEIGHT_S8_PC``
    An initializer that is operand 1 of a MatMul (compiled to a fully
    connected layer; Pulsar2 names it ``w_trans``). s8 per output channel:
    ``s_c = float32(max|w[:, c]|) / float32(127.5)``, zero point 0. The codes
    are ``clip(rint(w / s_c), -128, 127)`` with the division in float32 (the
    quantizer of ``emitter.codes_of``, byte-exact against the weight tables);
    a float64 division rounds about 0.03% of the codes the other way.
``BAKED_FORWARD`` (not a tensor rule)
    The calibration forward pass runs on the *fake-quantized* constants and
    weights, so every activation downstream of a constant gets its range from
    the quantized values, not the float ones.

Neg needs no rule of its own: its output's own range gives ``s_y == s_x`` and
``zp_y == 255 - zp_x`` on the measured builds.

What is derived, what is a fit
------------------------------

The formulas above are a fit to native builds, not read from Pulsar2's code.
The comparison covered 238 tensors of 73 builds (standalone Sigmoid, Mul, Add,
Div, ReduceMean, Sqrt, Neg, MatMul and Softmax at ``[1,64]`` and ``[1,576]``
and three calibrations; fused SiLU, RMSNorm and attention; the graph_stitch
components): every zero point and every signedness matched, 225 scales matched
bit for bit and 13 were one float32 ulp off. The 13 are all outputs of MatMul
(two activations), Softmax, Sqrt, Div or Mul: the forward pass here is numpy
float32 (``_evaluate``), which does not round every operation the way
Pulsar2's own evaluator does. A scale that is one ulp off changes float lanes
by one ulp; on the measured graphs the device's output codes were unchanged
(``docs/axera-pulsar-free-calibration.md``).

A MatMul with a constant weight was compared on 13 more builds (six 64x64 and
five 576x576 weight sets, the 576x576 layer at two calibrations). Its product
is evaluated in float64 and rounded to float32, because a float32 BLAS sum
depends on the machine: the output scale is bit-exact on 7 builds, one ulp off
on 5 and two ulps off on 1 (float32 BLAS here: up to 5). No summation order
tried reproduces all of them. The 25 single-op builds at other widths (100 to
2048, and the three RMSNorm components at 576) add 58 tensors: 52 bit-exact,
5 one ulp off (three ReduceMean outputs, two Softmax outputs) and one Softmax
output (width 384) two ulps off.

The forward pass uses numpy's float32 ``exp`` and ``sqrt``; another numpy
build may round ``exp`` differently and move a Sigmoid or Softmax scale by one
ulp. A float32 MatMul of two activations goes through BLAS and has the same
caveat.

Refused
-------

``NotImplementedError``: an op outside ``SUPPORTED_OPS`` (Concat is left out
on purpose: Pulsar2 leaves a standalone Concat float); a MatMul whose first
operand is an initializer; an initializer read by an op other than Add, Mul,
Div or MatMul. ``ValueError``: a graph input without samples, inputs with
different sample counts, a sample that does not have the input's size.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper

F32, F64 = np.float32, np.float64
SUPPORTED_OPS = (
    "Sigmoid",
    "Mul",
    "Add",
    "Div",
    "Sqrt",
    "Neg",
    "MatMul",
    "ReduceMean",
    "Softmax",
)
CONST_OPS = ("Add", "Mul", "Div")  # element-wise ops that may read an initializer


@dataclass(frozen=True)
class TensorQuant:
    """One tensor's quantization: ``scale`` is a float32 scalar, or a float32
    array with one entry per output channel (``WEIGHT_S8_PC``)."""

    scale: np.ndarray
    zero_point: int
    signed: bool
    rule: str
    lo: float | None = None
    hi: float | None = None

    @property
    def per_channel(self) -> bool:
        return np.ndim(self.scale) > 0


# ---- the rules ---------------------------------------------------------------------
def unsigned_minmax(lo: float, hi: float) -> tuple[np.float32, int]:
    """``UNSIGNED`` / ``CONST_U8``: ``(scale, zero point)`` of a u8 tensor whose
    values span ``lo..hi``."""
    lo, hi = F32(min(lo, 0)), F32(max(hi, 0))
    s = F32((float(hi) - float(lo)) / 255.0)
    zp = 0 if s == 0 else int(np.clip(np.rint(-float(lo) / float(s)), 0, 255))
    return s, zp


def signed_symmetric(absmax: float) -> tuple[np.float32, int]:
    """``SIGNED_ACT``: ``(scale, 0)`` of an s8 tensor with ``max|x| = absmax``."""
    return F32(F32(absmax) / F32(127.5)), 0


def weight_scales(w: np.ndarray) -> np.ndarray:
    """``WEIGHT_S8_PC``: one float32 scale per output channel of ``w [in, out]``."""
    return (np.abs(np.asarray(w, F32)).max(axis=0).astype(F32) / F32(127.5)).astype(F32)


def weight_codes(w: np.ndarray) -> np.ndarray:
    """``WEIGHT_S8_PC`` codes of ``w [in, out]`` as int64 in -128..127. The
    division is float32 (see the module docstring). A channel's largest
    magnitude maps to +-127.5, which rounds to +-128: -128 is kept and +128
    clips to 127."""
    w = np.asarray(w, F32)
    return np.clip(np.rint(w / weight_scales(w)), -128, 127).astype(np.int64)


def fake_quant_u8(w: np.ndarray) -> np.ndarray:
    """A ``CONST_U8`` constant as the calibration forward pass sees it."""
    s, zp = unsigned_minmax(w.min(), w.max())
    if s == 0:
        return w.astype(F32)
    q = np.clip(np.rint(w.astype(F64) / float(s)) + zp, 0, 255)
    return ((q - zp) * float(s)).astype(F32)


def fake_quant_weight(w: np.ndarray) -> np.ndarray:
    """A ``WEIGHT_S8_PC`` weight as the calibration forward pass sees it:
    ``weight_codes`` times the channel scale."""
    return (weight_codes(w) * weight_scales(w).astype(F64)).astype(F32)


# ---- the forward pass --------------------------------------------------------------
def _attrs(node: onnx.NodeProto) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _evaluate(
    model: onnx.ModelProto, feed: Mapping, consts: Mapping, weights: Iterable[str] = ()
) -> dict:
    """Every tensor of ``model`` for one sample, in numpy float32. A MatMul
    by one of ``weights`` is summed in float64 and rounded to float32."""
    v = {k: np.asarray(a, F32) for k, a in feed.items()}
    v.update(consts)
    for n in model.graph.node:
        i = [v[x] for x in n.input]
        a, op = _attrs(n), n.op_type
        if op == "Sigmoid":
            o = 1 / (1 + np.exp(-i[0]))
        elif op == "Mul":
            o = i[0] * i[1]
        elif op == "Add":
            o = i[0] + i[1]
        elif op == "Div":
            o = i[0] / i[1]
        elif op == "Sqrt":
            o = np.sqrt(i[0])
        elif op == "Neg":
            o = -i[0]
        elif op == "MatMul" and n.input[1] in weights:
            o = i[0].astype(F64) @ i[1].astype(F64)
        elif op == "MatMul":
            o = i[0] @ i[1]
        elif op == "ReduceMean":
            axes = a["axes"] if "axes" in a else [int(x) for x in i[1].ravel()]
            o = i[0].mean(axis=tuple(axes), keepdims=bool(a.get("keepdims", 1)))
        elif op == "Softmax":
            ax = a.get("axis", -1)
            e = np.exp(i[0] - i[0].max(axis=ax, keepdims=True))
            o = e / e.sum(axis=ax, keepdims=True)
        else:
            raise NotImplementedError(f"op {op!r}; supported: {SUPPORTED_OPS}")
        v[n.output[0]] = np.asarray(o).astype(F32)
    return v


def _roles(model: onnx.ModelProto) -> tuple[set, set, set, set]:
    """``(weights, element-wise constants, axes initializers, signed
    activations)`` of the graph, refusing what no build measured."""
    g = model.graph
    inits = {t.name for t in g.initializer}
    weights, consts, axes, signed = set(), set(), set(), set()
    for n in g.node:
        if n.op_type not in SUPPORTED_OPS:
            raise NotImplementedError(f"op {n.op_type!r}; supported: {SUPPORTED_OPS}")
        used = [x for x in n.input if x in inits]
        if n.op_type == "MatMul":
            if n.input[0] in inits:
                raise NotImplementedError(
                    "MatMul with an initializer as its first operand was not measured"
                )
            if n.input[1] in inits:
                weights.add(n.input[1])
            else:
                signed.update(n.input)
        elif n.op_type == "ReduceMean" and used == list(n.input[1:]):
            axes.update(used)
        elif used and n.op_type not in CONST_OPS:
            raise NotImplementedError(
                f"{n.op_type} reading initializer {used[0]!r} was not measured"
            )
        else:
            consts.update(used)
    if weights & consts:
        raise NotImplementedError(
            f"initializers {sorted(weights & consts)} are both a MatMul weight "
            "and an element-wise constant"
        )
    return weights, consts, axes, signed


def calibrate(
    model: onnx.ModelProto,
    samples: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, TensorQuant]:
    """Quantization of every tensor of ``model``: graph inputs, node outputs
    and float initializers, keyed by the tensor's name in ``model`` (Pulsar2
    renames a MatMul weight to ``w_trans``; here it keeps its name).

    ``samples`` maps each graph input to its calibration samples (any shape
    with the input's element count; Pulsar2's ``Numpy`` calibration format)."""
    g = model.graph
    weights, consts, axes, signed = _roles(model)
    arrays = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    shapes = {
        i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
        for i in g.input
        if i.name not in arrays
    }
    missing = sorted(set(shapes) - set(samples))
    if missing:
        raise ValueError(f"no calibration samples for graph inputs {missing}")
    counts = {len(samples[k]) for k in shapes}
    if len(counts) != 1 or not next(iter(counts)):
        raise ValueError("every graph input needs the same, nonzero sample count")

    out: dict[str, TensorQuant] = {}
    baked = {}  # BAKED_FORWARD: what the forward pass reads for each initializer
    for name, w in arrays.items():
        if name in weights:
            out[name] = TensorQuant(weight_scales(w), 0, True, "WEIGHT_S8_PC")
            baked[name] = fake_quant_weight(w)
        elif name in consts:
            s, zp = unsigned_minmax(w.min(), w.max())
            out[name] = TensorQuant(
                s, zp, False, "CONST_U8", float(w.min()), float(w.max())
            )
            baked[name] = fake_quant_u8(w)
        else:
            baked[name] = w  # ReduceMean axes or unused: not quantized
    runs = []
    for k in range(next(iter(counts))):
        feed = {}
        for name, shape in shapes.items():
            x = np.asarray(samples[name][k], F32)
            if x.size != int(np.prod(shape)):
                raise ValueError(
                    f"sample {k} of {name!r} has {x.size} elements, the input {shape}"
                )
            feed[name] = x.reshape(shape)
        runs.append(_evaluate(model, feed, baked, weights))
    for t in [*shapes, *(o for n in g.node for o in n.output)]:
        lo = min(float(r[t].min()) for r in runs)
        hi = max(float(r[t].max()) for r in runs)
        if t in signed:
            s, zp = signed_symmetric(max(abs(lo), abs(hi)))
            out[t] = TensorQuant(s, zp, True, "SIGNED_ACT", lo, hi)
        else:
            s, zp = unsigned_minmax(lo, hi)
            out[t] = TensorQuant(s, zp, False, "UNSIGNED", lo, hi)
    return out


def stitch_calibration(
    quants: Mapping[str, TensorQuant],
    tensors: Iterable[str] | None = None,
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    """``(scales, zero_points, signed)`` as ``graph_stitch.stitch`` and
    ``stitch_model`` take them, for ``tensors`` (default: every tensor with one
    scale; per-channel weights are left out, a stitched graph has none)."""
    names = [k for k, q in quants.items() if not q.per_channel]
    if tensors is not None:
        names = list(tensors)
        bad = sorted(k for k in names if k not in quants or quants[k].per_channel)
        if bad:
            raise ValueError(f"no single-scale calibration for tensors {bad}")
    scales = {k: float(quants[k].scale) for k in names}
    zero_points = {k: int(quants[k].zero_point) for k in names}
    return scales, zero_points, sorted(k for k in names if quants[k].signed)


def calibrate_for_stitch(
    model: onnx.ModelProto,
    samples: Mapping[str, Sequence[np.ndarray]],
    tensors: Iterable[str] | None = None,
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    """``stitch_calibration(calibrate(model, samples), tensors)``."""
    return stitch_calibration(calibrate(model, samples), tensors)
