"""The NPU graph rewrites Quark's ``XINT8`` (power-of-two scales, ``EnableNPUCnn``)
quantizer applies to the Q/DQ graph it has just built, for the Q/DQ models
:mod:`onnxsim.quark_compat` emits. Two stages, in Quark's order:

**DPU simulation** (``SimulateDPU``, each switchable like Quark's option):

- ``ConvertLeakyReluToDPUVersion``: ``LeakyRelu`` ``alpha`` -> ``round(alpha *
  256) / 256``;
- ``ConvertSigmoidToHardSigmoid``: ``Sigmoid`` -> ``HardSigmoid(alpha=1/6)``;
- ``ConvertHardSigmoidToDPUVersion``: a ``Mul`` by ``(2731 / 16384) / (1 / 6)``
  after every ``HardSigmoid(alpha=1/6)``;
- ``ConvertAvgPoolToDPUVersion``: a ``Mul`` after every ``AveragePool`` /
  ``GlobalAveragePool`` with a square kernel that makes ``sum / n`` behave like
  the NPU's fixed-point reciprocal (``9 * 7 / 64`` for 3x3, ... , the best
  ``k / 2**n`` approximation of ``1 / (kh * kw)`` otherwise, ``1.0`` beyond
  255);
- ``ConvertReduceMeanToDPUVersion``: the same for a ``ReduceMean``;
- ``ConvertClipToDPUVersion`` (opt-in): ``Clip`` bounds rounded into the int8
  range.

**Quantization-position refinement** (``adjust_quantize_info``): the scales of
the Q/DQ pairs are powers of two ``2**-pos``; the NPU limits the shifts between
them, and ``MaxLoopNum`` (5) rounds of the following passes move the positions
until none changes -- ``AlignConcat`` (output and inputs of a ``Concat`` take
the smallest position), ``AlignPool`` / ``AlignPad`` / ``AlignSlice`` (input
and output of a pool / pad / slice take the smaller position),
``AdjustShiftRead`` (``Add`` / ``Sub`` inputs at most 7 positions apart),
``AdjustShiftWrite`` (``Add``: ``min(ipos) - opos`` in [-7, 25]; ``Mul``:
``sum(ipos) - opos`` in [0, 32]), ``AdjustShiftCut`` (``wpos + ipos - opos`` of
a ``Conv`` / ``Gemm`` in [0, 16], moving the *weight* position),
``AdjustShiftBias`` (``wpos + ipos - bpos`` in ``[min(0, shift_cut - 16), 15]``,
moving the *bias* position), ``AdjustHardSigmoid`` and ``AdjustShiftSwish``.
Only the scales change: the stored integer codes stay, so moving a weight or
bias position rescales that constant by a power of two -- as Quark does.

The lookups (which Q/DQ node is "the input position of this node": the one
producing its first input, through a ``Pad`` for a pool; "the output position":
the first Q/DQ reading its output, through the DPU ``Mul`` of a pool /
``HardSigmoid`` or a following Relu-like node whose Q/DQ was removed) follow
Quark's, including its first-match-in-graph-order rule. Not reproduced:
``ConvertSoftmaxToDPUVersion`` / ``ConvertInstanceNormToDPUVersion`` (off by
default; Quark's integer softmax emulation and a custom operator), and the
order-dependence on Quark's node order (positions are visited in this graph's
node order, which can differ from Quark's topologically re-sorted one).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_QDQ = ("QuantizeLinear", "DequantizeLinear")
_ANNOTATE_OPS = (
    "Conv",
    "Add",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "MatMul",
    "Gemm",
    "ConvTranspose",
)
_AVG_POOLS = ("AveragePool", "GlobalAveragePool")
HARD_SIGMOID_SCALE = (2731.0 / 16384.0) / (1.0 / 6.0)
# Quark's option names for the refinement passes -> defaults
_ADJUST_OPTIONS = (
    "AdjustShiftCut",
    "AdjustShiftBias",
    "AdjustShiftRead",
    "AdjustShiftWrite",
    "AdjustHardSigmoid",
    "AdjustShiftSwish",
    "AlignConcat",
    "AlignPool",
    "AlignPad",
    "AlignSlice",
)
_SIMULATE_DEFAULTS = {
    "ConvertLeakyReluToDPUVersion": True,
    "ConvertSigmoidToHardSigmoid": True,
    "ConvertHardSigmoidToDPUVersion": True,
    "ConvertAvgPoolToDPUVersion": True,
    "ConvertReduceMeanToDPUVersion": True,
    "ConvertClipToDPUVersion": False,
}


def scale2pos(scale: float) -> int:
    """Fixed-point position of a power-of-two scale (``round(-log2(scale))``,
    the scale clamped to ``[2**-127, 2**127]``)."""
    scale = min(max(float(scale), 2.0**-127), 2.0**127)
    return int(np.rint(-np.log2(scale)))


def pos2scale(pos: int) -> float:
    return float(np.power(2.0, -pos))


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def dpu_leaky_relu_alpha(alpha: float) -> float:
    """The NPU's LeakyRelu slope: ``alpha`` as a multiple of 1/256."""
    return round(alpha * 256) / 256.0


def avg_pool_dpu_scale(kh: int, kw: int) -> float:
    """The factor the NPU's fixed-point average pooling is off by for a
    ``kh x kw`` window (``sum * k / 2**n`` instead of ``sum / (kh * kw)``,
    rescaled to ``1`` for the exact reciprocal)."""
    if kh > 255 or kw > 255:
        return 1.0
    special = {
        (3, 3): 9.0 * 7.0 / 64.0,
        (5, 5): 25.0 * 10.0 / 256.0,
        (6, 6): 36.0 * 7.0 / 256.0,
        (7, 7): 49.0 * 21.0 / 1024.0,
        (14, 14): 196.0 * 21.0 / 4096.0,
    }
    if (kh, kw) in special:
        return special[(kh, kw)]
    return reciprocal_dpu_scale(kh * kw)


def reciprocal_dpu_scale(rec: int) -> float:
    """``k / 2**n * rec`` for the ``n`` (below ``7 + ceil(log2(rec))``) at which
    ``k / 2**n`` is closest to ``1 / rec``, ``k = round(2**n / rec)``."""
    n_max = 7 + math.ceil(math.log2(rec))
    pows = [2**n for n in range(n_max)]
    ks = [round(p / rec) for p in pows]
    diffs = [abs(k / p - 1 / rec) for k, p in zip(ks, pows)]
    n = diffs.index(min(diffs))
    return (ks[n] / 2**n) * rec


class _Graph:
    """Lookups over a Q/DQ graph, keyed by tensor (never by node name)."""

    def __init__(
        self,
        model: onnx.ModelProto,
        relu_like: Callable[[onnx.NodeProto], bool],
    ) -> None:
        self.model = model
        self.g = model.graph
        self.relu_like = relu_like
        self._inits = {t.name: t for t in self.g.initializer}
        # the graph's structure does not change while positions move: index it
        self._qdq_out: Dict[str, onnx.NodeProto] = {}
        self._qdq_in: Dict[str, onnx.NodeProto] = {}
        self._by_out: Dict[str, onnx.NodeProto] = {}
        self._by_in0: Dict[str, List[onnx.NodeProto]] = {}
        for n in self.g.node:
            if n.output:
                self._by_out.setdefault(n.output[0], n)
            if n.input:
                self._by_in0.setdefault(n.input[0], []).append(n)
            if n.op_type in _QDQ:
                if n.output:
                    self._qdq_out.setdefault(n.output[0], n)
                if n.input:
                    self._qdq_in.setdefault(n.input[0], n)
        self.leaky_inputs = {
            n.input[0] for n in self.g.node if n.op_type == "LeakyRelu" and n.input
        }
        self.hard_sigmoid_inputs = {
            n.input[0] for n in self.g.node if n.input and _is_hard_sigmoid(n)
        }

    # -- scales -----------------------------------------------------------
    def scale_init(self, node: onnx.NodeProto) -> Optional[onnx.TensorProto]:
        return self._inits.get(node.input[1]) if len(node.input) > 1 else None

    def pos(self, node: Optional[onnx.NodeProto]) -> Optional[int]:
        if node is None or node.op_type not in _QDQ:
            return None
        init = self.scale_init(node)
        if init is None:
            return None
        return scale2pos(float(numpy_helper.to_array(init).ravel()[0]))

    def set_pos(self, node: onnx.NodeProto, new_pos: int) -> None:
        init = self.scale_init(node)
        assert init is not None
        old = numpy_helper.to_array(init)
        init.CopyFrom(
            numpy_helper.from_array(
                np.full(old.shape, pos2scale(new_pos), dtype=old.dtype), init.name
            )
        )

    # -- topology ---------------------------------------------------------
    def qdq_producing(self, tensor: str) -> Optional[onnx.NodeProto]:
        """First Q/DQ node (graph order) whose output is ``tensor``."""
        return self._qdq_out.get(tensor)

    def qdq_reading(self, tensor: str) -> Optional[onnx.NodeProto]:
        return self._qdq_in.get(tensor)

    def ipos_node(self, node: onnx.NodeProto) -> Optional[onnx.NodeProto]:
        if not node.input:
            return None
        found = self.qdq_producing(node.input[0])
        if found is not None:
            return found
        if node.op_type in _AVG_POOLS:
            # a pool fed by a node without its own Q/DQ (a Pad): that node's input
            n = self._by_out.get(node.input[0])
            if n is not None and n.input:
                return self.qdq_producing(n.input[0])
        return None

    def ipos_node_by_id(self, node: onnx.NodeProto, i: int) -> Optional[onnx.NodeProto]:
        return self.qdq_producing(node.input[i]) if len(node.input) > i else None

    def opos_node(self, node: onnx.NodeProto) -> Optional[onnx.NodeProto]:
        found = self.qdq_reading(node.output[0])
        if found is not None:
            return found
        for n in self._by_in0.get(node.output[0], ()):
            connected = (
                node.op_type in _AVG_POOLS + ("HardSigmoid",) and n.op_type == "Mul"
            ) or (node.op_type in _ANNOTATE_OPS and self.relu_like(n))
            if connected:
                found = self.qdq_reading(n.output[0])
                if found is not None:
                    return found
        return None

    def pos_of(self, node: Optional[onnx.NodeProto]) -> Optional[int]:
        return self.pos(node)


def _is_hard_sigmoid(node: onnx.NodeProto) -> bool:
    """Quark's ``check_hard_sigmoid_condition``: ``alpha`` ~ 1/6 and ``beta``
    absent or ~ 0.5 (whatever the op type)."""
    attrs = {a.name: a for a in node.attribute}
    alpha = "alpha" in attrs and _approx(attrs["alpha"].f, 1.0 / 6.0)
    beta = "beta" not in attrs or _approx(attrs["beta"].f, 0.5)
    return bool(alpha and beta)


class _Refiner:
    def __init__(self, graph: _Graph) -> None:
        self.q = graph
        self.changed = True

    def _poses(
        self, node: onnx.NodeProto
    ) -> Optional[Tuple[List[onnx.NodeProto], List[int]]]:
        nodes: List[onnx.NodeProto] = []
        poses: List[int] = []
        for i in range(len(node.input)):
            n = self.q.ipos_node_by_id(node, i)
            if n is None:
                return None
            nodes.append(n)
        for n in nodes:
            p = self.q.pos(n)
            if p is None:
                return None
            poses.append(p)
        return nodes, poses

    def shift_cut(self) -> None:
        for node in list(self.q.g.node):
            if node.op_type not in ("Conv", "Gemm"):
                continue
            ipos = self.q.pos(self.q.ipos_node(node))
            opos = self.q.pos(self.q.opos_node(node))
            wnode = self.q.ipos_node_by_id(node, 1)
            wpos = self.q.pos(wnode)
            if wpos is None or ipos is None or opos is None:
                continue
            sc = wpos + ipos - opos
            new_sc = 0 if sc < 0 else 16 if sc > 16 else None
            if new_sc is not None:
                self.changed = True
                assert wnode is not None
                self.q.set_pos(wnode, new_sc + opos - ipos)

    def shift_bias(self) -> None:
        for node in list(self.q.g.node):
            if node.op_type not in ("Conv", "Gemm") or len(node.input) < 3:
                continue
            ipos = self.q.pos(self.q.ipos_node(node))
            opos = self.q.pos(self.q.opos_node(node))
            wpos = self.q.pos(self.q.ipos_node_by_id(node, 1))
            bnode = self.q.ipos_node_by_id(node, 2)
            bpos = self.q.pos(bnode)
            if wpos is None or ipos is None or opos is None or bpos is None:
                continue
            shift_cut = wpos + ipos - opos
            min_sb = min(0, -(24 - (8 + shift_cut)))
            if node.output[0] in self.q.leaky_inputs:
                min_sb = 0
            shift_bias = wpos + ipos - bpos
            new_sb = min_sb if shift_bias < min_sb else 15 if shift_bias > 15 else None
            if new_sb is not None:
                self.changed = True
                assert bnode is not None
                self.q.set_pos(bnode, wpos + ipos - new_sb)

    def shift_swish(self) -> None:
        def sigmoid_input(tensor: str) -> bool:
            return tensor in self.q.hard_sigmoid_inputs

        for node in list(self.q.g.node):
            if node.op_type != "Mul" or len(node.input) != 2:
                continue
            if not (sigmoid_input(node.input[0]) or sigmoid_input(node.input[1])):
                continue
            onode = self.q.opos_node(node)
            opos = self.q.pos(onode)
            if opos is None:
                continue
            p0 = self.q.pos(self.q.ipos_node_by_id(node, 0))
            p1 = self.q.pos(self.q.ipos_node_by_id(node, 1))
            if p0 is None or p1 is None:
                continue
            shift = p0 + p1 - opos
            new_opos = opos
            if shift < 0:
                new_opos = p0 + p1
            elif shift > 15:
                new_opos = p0 + p1 - 15
            if new_opos != opos:
                self.changed = True
                assert onode is not None
                self.q.set_pos(onode, new_opos)

    def hard_sigmoid(self) -> None:
        for node in list(self.q.g.node):
            if node.op_type != "HardSigmoid" or not _is_hard_sigmoid(node):
                continue
            inode, onode = self.q.ipos_node(node), self.q.opos_node(node)
            ipos, opos = self.q.pos(inode), self.q.pos(onode)
            if ipos is None or opos is None:
                continue
            new_ipos = min(ipos if ipos > 0 else 0, 15)
            new_opos = opos if opos > 7 else 7
            shift = 14 + new_ipos - new_opos
            new_opos = new_opos if shift > 0 else 14 + new_ipos
            if new_ipos != ipos:
                self.changed = True
                assert inode is not None
                self.q.set_pos(inode, new_ipos)
            if new_opos != opos:
                self.changed = True
                assert onode is not None
                self.q.set_pos(onode, new_opos)

    def shift_read(self) -> None:
        for node in list(self.q.g.node):
            if node.op_type not in ("Add", "Sub"):
                continue
            got = self._poses(node)
            if got is None:
                continue
            nodes, poses = got
            hi, lo = int(np.argmax(poses)), int(np.argmin(poses))
            if poses[hi] - poses[lo] > 7:
                self.changed = True
                self.q.set_pos(nodes[hi], poses[lo] + 7)

    def shift_write(self) -> None:
        for node in list(self.q.g.node):
            if node.op_type not in ("Add", "Mul"):
                continue
            got = self._poses(node)
            if got is None:
                continue
            _, poses = got
            onode = self.q.opos_node(node)
            opos = self.q.pos(onode)
            if opos is None:
                continue
            assert onode is not None
            if node.op_type == "Add":
                sw = min(poses) - opos
                new_sw = -7 if sw < -7 else 25 if sw > 25 else None
                if new_sw is not None:
                    self.changed = True
                    self.q.set_pos(onode, min(poses) - new_sw)
            else:  # Mul: no change flag in Quark (its loop can end early)
                sw = sum(poses) - opos
                new_sw = 0 if sw < 0 else 32 if sw > 32 else None
                if new_sw is not None:
                    self.q.set_pos(onode, sum(poses) - new_sw)

    def align_concat(self) -> None:
        for node in list(self.q.g.node):
            if node.op_type != "Concat":
                continue
            onode = self.q.opos_node(node)
            opos = self.q.pos(onode)
            if opos is None:
                continue
            assert onode is not None
            inodes = [self.q.ipos_node_by_id(node, i) for i in range(len(node.input))]
            poses = [self.q.pos(n) for n in inodes]
            min_pos = min([opos] + [p for p in poses if p is not None])
            if opos != min_pos:
                self.changed = True
                self.q.set_pos(onode, min_pos)
            for n in inodes:
                p = self.q.pos(n)
                if n is not None and p is not None and p != min_pos:
                    self.changed = True
                    self.q.set_pos(n, min_pos)

    def align_unary(self, ops: Sequence[str]) -> None:
        """Input and output of a pool / pad / slice take the smaller position."""
        for node in list(self.q.g.node):
            if node.op_type not in ops:
                continue
            inode, onode = self.q.ipos_node(node), self.q.opos_node(node)
            ipos, opos = self.q.pos(inode), self.q.pos(onode)
            if ipos is None or opos is None:
                continue
            assert inode is not None and onode is not None
            if opos > ipos:
                self.changed = True
                self.q.set_pos(onode, ipos)
            elif opos < ipos:
                self.changed = True
                self.q.set_pos(inode, opos)


def adjust_quantize_info(
    model: onnx.ModelProto,
    relu_like: Callable[[onnx.NodeProto], bool],
    max_loop_num: int = 5,
    **flags: bool,
) -> onnx.ModelProto:
    """Quark's ``adjust_quantize_info`` on the Q/DQ ``model`` (in place): up to
    ``max_loop_num`` rounds of the alignment passes, then the shift passes,
    until a round changes nothing. ``flags`` are the ``Adjust*`` / ``Align*``
    option names (all default on)."""
    refiner = _Refiner(_Graph(model, relu_like))

    def on(name: str) -> bool:
        return bool(flags.get(name, True))

    loops = 0
    while refiner.changed and loops < max_loop_num:
        loops += 1
        refiner.changed = False
        if on("AlignConcat"):
            refiner.align_concat()
        if on("AlignPool"):
            refiner.align_unary(("MaxPool", "AveragePool", "GlobalAveragePool"))
        if on("AlignPad"):
            refiner.align_unary(("Pad",))
        if on("AlignSlice"):
            refiner.align_unary(("Slice",))
        if on("AdjustShiftRead"):
            refiner.shift_read()
        if on("AdjustShiftWrite"):
            refiner.shift_write()
        if on("AdjustShiftCut"):
            refiner.shift_cut()
        if on("AdjustShiftBias"):
            refiner.shift_bias()
        if on("AdjustHardSigmoid"):
            refiner.hard_sigmoid()
        if on("AdjustShiftSwish"):
            refiner.shift_swish()
    return model


# -- DPU simulation ----------------------------------------------------------------


def _insert_mul(graph: onnx.GraphProto, node: onnx.NodeProto, scale: float) -> None:
    """Quark's ``insert_mul``: ``Mul(node output, scale)`` under the original
    output name, its ``Constant`` and the ``Mul`` appended at the end of the
    graph (where the refinement passes, visiting nodes in list order, find
    them)."""
    out = node.output[0]
    pre = out + "_Mul"
    const = out + "_Scale"
    if not node.name:
        node.name = out
    node.output[0] = pre
    graph.node.append(
        helper.make_node(
            "Constant",
            [],
            [const],
            value=helper.make_tensor("scale", TensorProto.FLOAT, [], [scale]),
        )
    )
    graph.node.append(helper.make_node("Mul", [pre, const], [out], name=pre))


def _float_source(producers: Dict[str, onnx.NodeProto], tensor: str) -> str:
    """The float tensor behind ``tensor``: through its DQ and the Q before it."""
    p = producers.get(tensor)
    if p is not None and p.op_type == "DequantizeLinear":
        tensor = p.input[0]
        q = producers.get(tensor)
        if q is not None:
            tensor = q.input[0]
    return tensor


def simulate_dpu(
    model: onnx.ModelProto,
    should_simulate: Callable[[onnx.NodeProto], bool],
    options: Dict[str, Any],
    nodes_to_skip: Optional[Set[str]] = None,
) -> onnx.ModelProto:
    """Quark's ``simulate_transforms`` (see the module docstring), in place."""

    def on(key: str) -> bool:
        return bool(options.get(key, _SIMULATE_DEFAULTS[key]))

    g = model.graph
    skip = nodes_to_skip or set()
    # (shapes are those of the graph before any rewrite -- Quark's ``value_info``
    # -- as the appended nodes leave the graph out of order)
    shapes: Optional[Dict[str, List[int]]] = None
    if on("ConvertAvgPoolToDPUVersion") or on("ConvertReduceMeanToDPUVersion"):
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            shapes = {
                vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
                # (Quark looks shapes up in ``value_info`` alone: a graph input
                # or output has none, and its pool / mean is left unconverted)
                for vi in inferred.graph.value_info
                if vi.type.HasField("tensor_type")
            }
        except Exception:  # pragma: no cover - shape inference failure
            shapes = {}
    if on("ConvertLeakyReluToDPUVersion"):
        for n in g.node:
            if n.op_type == "LeakyRelu" and should_simulate(n):
                for a in n.attribute:
                    if a.name == "alpha":
                        a.f = dpu_leaky_relu_alpha(a.f)
    if on("ConvertSigmoidToHardSigmoid"):
        # (the replacement is appended at the end of the graph, as in Quark)
        for n in list(g.node):
            if n.op_type == "Sigmoid" and should_simulate(n):
                new = helper.make_node(
                    "HardSigmoid", list(n.input), list(n.output), name=n.name
                )
                new.attribute.append(helper.make_attribute("alpha", 1.0 / 6.0))
                g.node.append(new)
                g.node.remove(n)
    if on("ConvertHardSigmoidToDPUVersion"):
        for n in list(g.node):
            if (
                n.op_type == "HardSigmoid"
                and _is_hard_sigmoid(n)
                and n.name not in skip
            ):
                _insert_mul(g, n, HARD_SIGMOID_SCALE)
    producers = {o: n for n in g.node for o in n.output}
    if on("ConvertAvgPoolToDPUVersion"):
        for n in list(g.node):
            if n.op_type not in _AVG_POOLS or not should_simulate(n):
                continue
            kh = kw = 0
            ok = False
            if n.op_type == "GlobalAveragePool":
                src = _float_source(producers, n.input[0])
                shape = (shapes or {}).get(src)
                if not shape:
                    continue
                if len(shape) == 4 and shape[2] == shape[3]:
                    ok, kh, kw = True, shape[2], shape[3]
            else:
                ks = next(
                    (list(a.ints) for a in n.attribute if a.name == "kernel_shape"),
                    None,
                )
                if ks and len(ks) == 2 and ks[0] == ks[1]:
                    ok, kh, kw = True, ks[0], ks[1]
            if ok and kh * kw > 0:
                _insert_mul(g, n, avg_pool_dpu_scale(kh, kw))
    if on("ConvertReduceMeanToDPUVersion"):
        for n in list(g.node):
            if n.op_type != "ReduceMean" or not should_simulate(n):
                continue
            src = _float_source(producers, n.input[0])
            shape = (shapes or {}).get(src)
            axes = None
            if len(n.input) == 1:
                axes = next(
                    (list(a.ints) for a in n.attribute if a.name == "axes"), None
                )
            elif len(n.input) == 2:
                inits = {t.name: t for t in g.initializer}
                if n.input[1] in inits:
                    axes = numpy_helper.to_array(inits[n.input[1]]).tolist()
            if axes is not None and shape:
                rec = 1
                for a in axes:
                    rec *= shape[a]
                if rec > 0:
                    _insert_mul(g, n, reciprocal_dpu_scale(rec))
    if on("ConvertClipToDPUVersion"):
        inits = {t.name: t for t in g.initializer}
        for n in g.node:
            if n.op_type != "Clip" or not should_simulate(n):
                continue
            for k, default in ((1, -128), (2, 127)):
                name = n.input[k] if len(n.input) > k else ""
                if name in inits:
                    v = float(numpy_helper.to_array(inits[name]).item())
                    v = max(-128, min(127, round(v)))
                    inits[name].CopyFrom(
                        numpy_helper.from_array(np.array(v, np.float32), name)
                    )
                elif not name:
                    new = (n.name or n.output[0]) + (
                        "_dpu_min" if k == 1 else "_dpu_max"
                    )
                    while len(n.input) <= k:
                        n.input.append("")
                    n.input[k] = new
                    g.initializer.append(
                        numpy_helper.from_array(np.array(default, np.float32), new)
                    )
    return model


def apply_npu_cnn_rewrites(
    model: onnx.ModelProto,
    options: Dict[str, Any],
    relu_like_ops: Sequence[str],
    should_simulate: Optional[Callable[[onnx.NodeProto], bool]] = None,
) -> onnx.ModelProto:
    """Run Quark's NPU CNN post-quantization stages on a copy of ``model``:
    DPU simulation unless ``SimulateDPU`` is False, then position refinement
    unless ``NPULimitationCheck`` is False. ``relu_like_ops`` are the op types
    whose Q/DQ pair behind a Conv / Add / pool Quark removed (``Relu``,
    ``Clip`` with (0, 6) / (0, 1) bounds, ...), through which an output
    position is looked up."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    inits = {t.name: t for t in out.graph.initializer}

    def relu_like(n: onnx.NodeProto) -> bool:
        if n.op_type not in relu_like_ops:
            return False
        if n.op_type == "Clip":
            if len(n.input) != 3 or n.input[1] not in inits or n.input[2] not in inits:
                return False
            try:
                lo = float(numpy_helper.to_array(inits[n.input[1]]).item())
                hi = float(numpy_helper.to_array(inits[n.input[2]]).item())
            except ValueError:
                return False
            return _approx(lo, 0.0) and (_approx(hi, 6.0) or _approx(hi, 1.0))
        return True

    if options.get("SimulateDPU", True) is not False:
        simulate_dpu(out, should_simulate or (lambda n: True), options)
    if options.get("NPULimitationCheck", True) is not False:
        adjust_quantize_info(
            out,
            relu_like,
            max_loop_num=int(options.get("MaxLoopNum", 5)),
            **{k: bool(options[k]) for k in _ADJUST_OPTIONS if k in options},
        )
    return out


__all__: Any = [
    "adjust_quantize_info",
    "apply_npu_cnn_rewrites",
    "avg_pool_dpu_scale",
    "dpu_leaky_relu_alpha",
    "pos2scale",
    "reciprocal_dpu_scale",
    "scale2pos",
    "simulate_dpu",
]
