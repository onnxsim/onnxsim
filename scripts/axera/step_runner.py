"""Run a whole training step with its covered nodes on the AX650 NPU.

``coverage_report`` says which nodes of a step graph our emitters can produce
at the step's calibration; this module actually executes the step that way:

* Every node ``plan_at_calibration`` covers becomes an **NPU segment**: its
  template retargeted to the predicted calibration (``step_calibration.py``)
  and run on the card. A covered live-operand MatMul runs its whole chain
  template (``matmul_record_emit``: the Gather/Mul/Reshape/Transpose feeding
  it) as one segment; Greater/Less -> Cast runs as its pair template.
* Everything else runs on the host, one node at a time in onnxruntime (a
  whole-graph session of the step passes 16 GiB; ``collect_ranges`` does the
  same).
* Every segment's input and output is float32: the templates quantize and
  dequantize inside, so segments pass plain float tensors.

Each NPU segment is also evaluated on the host as a **simulated** segment on
the same inputs: inputs fake-quantized at the segment's predicted input
parameters, the float ops, the output fake-quantized at its output
parameters. The device-vs-simulation difference, in output LSBs, is the
per-segment check that the emitted model computes what the template claims.

    python step_runner.py --mode npu --out report.json            # validation pass
    python step_runner.py --mode npu --validated report.json \
        --exclude '^(Softmax|Log|Neg)_' --host-optimizer --out final.json

``docs/axera-step-runner.md`` has the measured numbers.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import pickle
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import elementwise_scale_emit as ew  # noqa: E402
import matmul_record_emit as mre  # noqa: E402
import misc_op_record_emit as misc  # noqa: E402
import reshape_record_emit as rre  # noqa: E402
import step_recalibrate  # noqa: E402
import tinygrad_ax_backend as axb  # noqa: E402

STEP_ONNX = "/home/takecheeze/npu-scratch/t6-r18fold/step.onnx"
STEP_REF = "/home/takecheeze/npu-scratch/t6-r18fold/step1_ref.pkl"
STEP_OPS = os.path.join(
    _HERE, "fixtures", "tinygrad_ax_backend", "resnet18_step_ops.json.gz"
)
STEP_CALIB = os.path.join(
    _HERE, "fixtures", "step_calibration", "resnet18_step_calibration.json.gz"
)


# --------------------------------------------------------------------------
# quantization helpers


def qparams_of(calib: Mapping, name: str) -> tuple[float, int, bool]:
    q = calib["tensors"][name]
    return float(q["scale"]), int(q["zero_point"]), bool(q.get("signed"))


def fake_quant(x: np.ndarray, scale: float, zp: int, signed: bool) -> np.ndarray:
    lo, hi = (-128, 127) if signed else (0, 255)
    s = np.float32(scale)
    q = np.clip(np.rint(x.astype(np.float32) / s) + zp, lo, hi)
    return ((q - zp) * s).astype(np.float32)


# --------------------------------------------------------------------------
# host execution: one onnxruntime session per distinct node signature


class HostOps:
    def __init__(self, model: onnx.ModelProto):
        import onnxruntime as ort

        self._ort = ort
        self.model = model
        self.elem = {
            v.name: v.type.tensor_type.elem_type
            for v in list(model.graph.value_info)
            + list(model.graph.output)
            + list(model.graph.input)
        }
        self.inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        self.so = ort.SessionOptions()
        self.so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.so.log_severity_level = 3
        self.so.intra_op_num_threads = 8
        self.sessions: dict[bytes, Any] = {}

    def _session(self, n: onnx.NodeProto, args: Sequence[np.ndarray]):
        node = onnx.NodeProto()
        node.CopyFrom(n)
        live = [t for t in n.input if t]
        del node.input[:]
        node.input.extend(f"i{live.index(t)}" if t else "" for t in n.input)
        del node.output[:]
        node.output.extend(f"o{j}" if t else "" for j, t in enumerate(n.output))
        node.name = "n"
        sig = (
            node.SerializeToString()
            + repr([(a.dtype.str, a.shape) for a in args]).encode()
        )
        if sig not in self.sessions:
            ins = [
                helper.make_tensor_value_info(
                    f"i{j}", helper.np_dtype_to_tensor_dtype(a.dtype), a.shape
                )
                for j, a in enumerate(args)
            ]
            outs = [
                helper.make_tensor_value_info(f"o{j}", self.elem.get(t, 0), None)
                for j, t in enumerate(n.output)
                if t
            ]
            one = helper.make_model(
                helper.make_graph([node], "one", ins, outs),
                opset_imports=self.model.opset_import,
            )
            one.ir_version = self.model.ir_version
            self.sessions[sig] = self._ort.InferenceSession(
                one.SerializeToString(), self.so, providers=["CPUExecutionProvider"]
            )
        return self.sessions[sig]

    def run(self, n: onnx.NodeProto, env: Mapping[str, np.ndarray]) -> list:
        args = [env[t] if t in env else self.inits[t] for t in n.input if t]
        return self._session(n, args).run(
            None, {f"i{j}": a for j, a in enumerate(args)}
        )


# --------------------------------------------------------------------------
# the plan: which nodes form which NPU segment


@dataclasses.dataclass
class Segment:
    name: str
    kind: str  # emitter family
    nodes: list[str]  # step node names the template computes
    inputs: list[str]  # step tensors, in the template's input order
    outputs: list[str]  # step tensors, in the template's output order
    detail: str
    emit: Callable[[], onnx.ModelProto] = dataclasses.field(repr=False)
    in_q: list[tuple[float, int, bool]] = dataclasses.field(default_factory=list)
    out_q: list[tuple[float, int, bool]] = dataclasses.field(default_factory=list)
    unsafe: str = ""  # why the template's semantics differ from the node's
    # A template built at batch N/k runs k times on batch slices; ``split``
    # marks the inputs that carry the batch axis (step_template batch_split).
    batch_split: int = 1
    split: list[bool] = dataclasses.field(default_factory=list)
    input_shapes: list[tuple[int, ...]] = dataclasses.field(default_factory=list)
    output_shape: tuple[int, ...] = ()


_RETARGET_KEY = re.compile(r"retarget of (\S+) \(")
_FUSED_KEY = re.compile(r"retarget of the fused chain (\S+) \(")
_CLASS = re.compile(r"\((x\d+,y\d+(?:,z\d+)?)\)")


def _scale_dict(calib, names: Mapping[str, str]) -> tuple[dict, dict]:
    sc, zp = {}, {}
    for role, t in names.items():
        s, z, _ = qparams_of(calib, t)
        sc[role], zp[role] = s, z
    return sc, zp


def _dim0(model: onnx.ModelProto, name: str) -> int | None:
    for vi in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if vi.name == name and vi.type.tensor_type.shape.dim:
            return vi.type.tensor_type.shape.dim[0].dim_value
    return None


def _segment_for(
    rec: Mapping,
    detail: str,
    calib: Mapping,
    model: onnx.ModelProto,
    consumers: Mapping[str, list[onnx.NodeProto]],
    inits: Mapping[str, np.ndarray],
) -> Segment | None:
    op, name = rec["op"], rec["name"]
    ins, outs = list(rec["inputs"]), list(rec["outputs"])

    def q(ts):
        return [qparams_of(calib, t) for t in ts]

    if detail.startswith("matmul_record_emit.recalibrate"):
        entry = mre.step_template(name)
        tmpl = mre.load_model(entry["axmodel"])
        inv = {v: k for k, v in entry["names"].items()}
        # Template-only tensors (docs/axera-matmul-step-templates.md): a
        # ``__pre`` input feeds a Relu whose output is the step tensor (Relu
        # is idempotent, so the step's Relu output goes in); ``__side``
        # outputs only keep an input uint8 and are not step outputs.
        aliases = entry.get("aliases", {})
        for a, src in aliases.items():
            if a.endswith("__pre") and src in inv:
                inv[a] = inv[src]
        t_in = [inv[i.name] for i in tmpl.graph.input]
        t_out = [inv[o.name] for o in tmpl.graph.output if o.name not in aliases]
        # A ``__pre`` input stands for its step tensor (the template's Relu
        # is template-only), so that tensor is an input, not computed here.
        t_inputs = {i.name for i in tmpl.graph.input}
        t_inputs |= {src for a, src in aliases.items() if a in t_inputs}
        internal = [s for s, t in entry["names"].items() if t not in t_inputs]
        producers = {o: n.name for n in model.graph.node for o in n.output}
        nodes = sorted({producers[t] for t in internal if t in producers})

        def emit_mm():
            old = mre.load_scales(entry["quant"])
            real = {}
            for step_name, tname in entry["names"].items():
                qq = calib["tensors"][step_name]
                if "consumer_int8_scale" in qq and tname in old and old[tname][1] == 0:
                    real[step_name] = (qq["consumer_int8_scale"], 0.0)
                else:
                    real[step_name] = (qq["scale"], float(qq["zero_point"]))
                if "consumer_int8_scale" in qq and tname + mre.I8 in old:
                    real[step_name + mre.I8] = (qq["consumer_int8_scale"], 0.0)
            new = mre.step_node_scales(entry, old, real)
            out, _ = mre.recalibrate(tmpl, old, new)
            return out

        in_q = []
        old = mre.load_scales(entry["quant"])
        for t in t_in:
            qq = calib["tensors"][t]
            tn = entry["names"].get(t, "")
            if "consumer_int8_scale" in qq and tn in old and old[tn][1] == 0:
                in_q.append((float(qq["consumer_int8_scale"]), 0, True))
            else:
                in_q.append(qparams_of(calib, t))
        k = entry.get("batch_split", 1)
        split = [
            k > 1 and _dim0(model, t) == _dim0(tmpl, i.name) * k
            for t, i in zip(t_in, tmpl.graph.input)
        ]
        return Segment(
            name, "matmul_chain", nodes, t_in, t_out, detail, emit_mm, in_q, q(t_out),
            batch_split=k, split=split,
        )  # fmt: skip

    if op in ("Greater", "Less"):
        key = rec["attrs"]["misc_key"]
        cast = [c for c in consumers.get(outs[0], []) if c.op_type == "Cast"]
        if len(cast) != 1:
            return None
        tmpl, _ = misc.load_template(key)
        live = [t for t in ins if t not in inits][: len(tmpl.graph.input)]
        # not quantized: the device compares the float input directly
        return Segment(
            name, "compare_cast", [name, cast[0].name], live, [cast[0].output[0]],
            detail, lambda: misc.emit_model(key), [], [],
        )  # fmt: skip
    if op == "Cast":
        return None  # served by its Greater/Less pair segment

    fused = _FUSED_KEY.search(detail)
    if fused and op == "Reshape":
        # bias flatten: one program with its producing ReduceSum (#1908)
        key = fused.group(1)
        src = rec["attrs"]["fused_input"]
        producer = [
            c
            for c in consumers.get(src, [])
            if c.op_type == "ReduceSum" and ins[0] in c.output
        ]
        if len(producer) != 1:
            return None
        sc, zp = _scale_dict(calib, {"x": src, "y": outs[0]})

        def emit_fused():
            return misc.emit_model(key, sc, zp)

        return Segment(
            name, "reducesum_flatten", [producer[0].name, name], [src], outs, detail,
            emit_fused, q([src]), q(outs),
        )  # fmt: skip

    m = _RETARGET_KEY.search(detail)
    if m and detail.startswith("misc_op_record_emit"):
        key = m.group(1)
        sc, zp = _scale_dict(calib, {"x": ins[0], "y": outs[0]})

        def emit_misc():
            return misc.emit_model(key, sc, zp)

        return Segment(
            name, "misc", [name], [ins[0]], outs, detail, emit_misc, q(ins[:1]), q(outs)
        )

    if op == "Relu" and detail.startswith("Relu record retarget"):
        s, z, _ = qparams_of(calib, ins[0])
        shape = rec["shapes"][0]

        def emit_relu():
            tm, _ = ew.load_template("Relu", shape, {"x": 128, "y": 128})
            mc = ew.retarget_relu_records(
                bytes(ew._mcode_initializer(tm).raw_data), s, z
            )
            return step_recalibrate.with_mcode(tm, mc)

        return Segment(
            name, "relu", [name], ins, outs, detail, emit_relu, q(ins), q(outs)
        )

    if detail.startswith("ElementwiseScaleEdit"):
        cls = _CLASS.search(detail).group(1)
        key = axb.key_for_record(rec, cls)
        if op in ew.OPS:
            sc, _ = _scale_dict(calib, {"x": ins[0], "y": outs[0]})
            live = ins[:1]
        else:
            sc, _ = _scale_dict(calib, {"x": ins[0], "z": ins[1], "y": outs[0]})
            live = ins[:2]

        def emit_ew():
            return axb.EditSet([axb.ElementwiseScaleEdit(sc)]).build(key)

        input_shapes = [
            tuple(int(d) for d in s)
            for s in rec.get("attrs", {}).get("input_shapes", [])
        ]
        output_shape = tuple(
            int(d) for d in rec.get("attrs", {}).get("output_shape", ())
        )
        return Segment(
            name,
            "elementwise",
            [name],
            live,
            outs,
            detail,
            emit_ew,
            q(live),
            q(outs),
            input_shapes=input_shapes,
            output_shape=output_shape,
        )

    if op in ("Reshape", "Squeeze") and detail.startswith("reshape_record_emit"):
        s, z, _ = qparams_of(calib, ins[0])
        in_shape = rec["shapes"][0]
        out_shape = rec["attrs"].get("out", [])

        def emit_reshape():
            return rre.emit_step_reshape(in_shape, out_shape, s, z)

        seg = Segment(
            name,
            "reshape",
            [name],
            ins[:1],
            outs,
            detail,
            emit_reshape,
            q(ins[:1]),
            q(outs),
        )
        # a signed input (nonzero zero point) is emitted from the Reshape ->
        # Identity template; the Reshape -> Relu ones (#1891) would clip it
        return seg

    if detail == "GatherIndexEdit":
        key = axb.key_for_record(rec)
        idx = [int(i) for i in np.asarray(inits[ins[1]]).ravel()]
        s, z, _ = qparams_of(calib, ins[0])

        def emit_gather():
            # GatherIndexEdit keeps the template's own calibration; a Gather is
            # passive (its output shares the input's quantization), so move the
            # template's one (1/s, s, zero point) to the step's like a Reshape.
            gm = axb.EditSet([axb.GatherIndexEdit(idx)]).build(key)
            mc = rre.retarget_scale(
                bytes(axb._mcode_initializer(gm).raw_data),
                s,
                z,
                zp_regs=rre.GATHER_ZP_REGS,
            )
            return step_recalibrate.with_mcode(gm, mc)

        seg = Segment(
            name,
            "gather",
            [name],
            ins[:1],
            outs,
            detail,
            emit_gather,
            q(ins[:1]),
            q(outs),
        )
        if z == 0:
            seg.unsafe = "a zero point of 0 is a different Gather program (no template)"
        return seg

    if detail == "TemplateOnly" and op == "Transpose":
        key = axb.key_for_record(rec)

        def emit_transpose():
            return axb.EditSet([axb.TemplateOnly()]).build(key)

        return Segment(
            name, "transpose", [name], ins[:1], outs, detail, emit_transpose, [], []
        )
    return None


def build_plan(
    model: onnx.ModelProto,
    records: Sequence[Mapping],
    calib: Mapping,
    kinds: set[str] | None = None,
    include_unsafe: bool = False,
) -> tuple[list[Segment], dict[str, str]]:
    """NPU segments (only of ``kinds`` if given) and a per-node reason for
    every node left on the host."""
    cache = axb.TemplateCache()
    # Live MatMul/Conv validation scans the compiled MCode.  The same scan is
    # required by the segment emitter below, so defer it to
    # ``drop_unemittable`` instead of doing it once during planning and again
    # while materializing the models.
    plans = [
        axb.plan_at_calibration(r, calib, cache, validate_live=False) for r in records
    ]
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in model.graph.node:
        for t in n.input:
            consumers.setdefault(t, []).append(n)
    segs: list[Segment] = []
    host: dict[str, str] = {}
    taken: set[str] = set()
    candidates = []
    value_shapes = {
        v.name: tuple(int(d.dim_value) for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    for rec, (status, detail) in zip(records, plans):
        if (
            status in ("refused", "covered")
            and rec["op"] in axb.bse.OPS
            and rec.get("attrs", {}).get("form") == "broadcast"
            and detail.startswith("ElementwiseScaleEdit")
        ):
            input_shapes = [value_shapes.get(t, ()) for t in rec["inputs"][:2]]
            output_shape = value_shapes.get(rec["outputs"][0], ())
            if (
                output_shape
                and all(input_shapes)
                and all(t in calib.get("tensors", {}) for t in rec["inputs"][:2])
            ):
                # A full-shape binary template is semantically identical when
                # a live operand is broadcast at the segment boundary. Keep
                # constants and uncalibrated operands conservative.
                expanded = dict(rec)
                expanded["shapes"] = [list(output_shape)]
                expanded["attrs"] = dict(rec.get("attrs", {}))
                expanded["attrs"].update(
                    {
                        "form": "same_shape",
                        "output_shape": list(output_shape),
                        "input_shapes": [list(s) for s in input_shapes],
                    }
                )
                status, detail = axb.plan_at_calibration(
                    expanded, calib, cache, validate_live=False
                )
                rec = expanded
        if status != "covered":
            host[rec["name"]] = f"{status}: {detail}"
            continue
        seg = _segment_for(rec, detail, calib, model, consumers, inits)
        if seg is None or (kinds and seg.kind not in kinds):
            host[rec["name"]] = (
                f"covered ({detail}) but no runner segment"
                if seg is None
                else f"covered, kind {seg.kind} not selected"
            )
            continue
        if seg.unsafe and not include_unsafe:
            host[rec["name"]] = f"covered, but unsafe: {seg.unsafe}"
            continue
        candidates.append(seg)
    # multi-node segments (chains, fused pairs) claim their nodes first
    # A node inside two chains (the fc Squeeze feeds both the forward Gemm
    # chain and the fc dW chain) is recomputed by each; only a clash on a
    # node whose output a segment exports keeps the smaller segment out.
    exported: dict[str, set[str]] = {}
    for seg in candidates:
        exported[seg.name] = {
            n.name for n in model.graph.node if set(n.output) & set(seg.outputs)
        }
    claimed: dict[str, str] = {}
    for seg in sorted(candidates, key=lambda s: -len(s.nodes)):
        clash = [
            n
            for n in taken & set(seg.nodes)
            if n in exported[seg.name] or n in exported[claimed[n]]
        ]
        if clash:
            for n in seg.nodes:
                host.setdefault(n, f"covered, but inside another segment ({seg.name})")
            continue
        segs.append(seg)
        for n in seg.nodes:
            claimed.setdefault(n, seg.name)
        taken.update(seg.nodes)
    order = {r["name"]: k for k, r in enumerate(records)}
    segs.sort(key=lambda s: max(order.get(n, 0) for n in s.nodes))
    for s in segs:
        for n in s.nodes:
            host.pop(n, None)
    return segs, host


def drop_unemittable(
    segs: Sequence[Segment], host: dict[str, str]
) -> tuple[list[Segment], dict[str, bytes]]:
    """Emit every segment up front; one whose emitter refuses goes back to
    the host with the emitter's reason."""
    keep, blobs = [], {}
    for seg in segs:
        try:
            blobs[seg.name] = seg.emit().SerializeToString()
        except Exception as exc:
            for n in seg.nodes:
                host[n] = f"covered, but the emitter refused: {exc}"
            continue
        keep.append(seg)
    return keep, blobs


# --------------------------------------------------------------------------
# execution


@dataclasses.dataclass
class SegStat:
    segment: str
    kind: str
    nodes: int
    max_lsb: float = 0.0
    frac_gt1: float = 0.0
    max_abs: float = 0.0
    device_s: float = 0.0
    emit_s: float = 0.0
    sim_s: float = 0.0
    float_rel: float = 0.0  # device vs the float ops on the same inputs
    sim_float_rel: float = 0.0  # simulation vs float: the quantization alone
    float_mismatch: float = 0.0  # unquantized outputs: fraction != float
    error: str = ""


class StepRunner:
    """Execute ``model`` node by node, NPU segments on the device (``mode``
    ``"npu"``), simulated on the host (``"sim"``), or everything in float
    (``"float"``)."""

    def __init__(
        self,
        model: onnx.ModelProto,
        segments: Sequence[Segment],
        session=None,
        emit_dir: str | None = None,
        health_every: int = 0,
    ):
        self.model = model
        self.host = HostOps(model)
        self.session = session
        self.emit_dir = emit_dir
        self.health_every = health_every
        self.device_runs = 0
        self.stalled: list[str] = []
        self.nodes = list(model.graph.node)
        self.index = {n.name: k for k, n in enumerate(self.nodes)}
        self.by_name = {n.name: n for n in self.nodes}
        self.segments = list(segments)
        self.seg_of: dict[str, Segment] = {}
        for s in self.segments:
            for n in s.nodes:
                self.seg_of[n] = s
        self.fire_at = {
            s.name: max(self.index[n] for n in s.nodes) for s in self.segments
        }
        self._host_dup = self._host_duplicates()
        last: dict[str, int] = {}
        for k, n in enumerate(self.nodes):
            for t in n.input:
                last[t] = k
        for s in self.segments:
            for t in s.inputs:
                last[t] = max(last.get(t, 0), self.fire_at[s.name])
        self.last = last
        self._emitted: dict[str, bytes] = {}

    def _host_duplicates(self) -> set[str]:
        """Segment-internal nodes the host must compute as well: their output
        is consumed outside the segment (or before the segment fires) but is
        not a segment output."""
        producers = {o: n for n in self.nodes for o in n.output}
        dup: set[str] = set()
        for s in self.segments:
            members = set(s.nodes)
            fire = self.fire_at[s.name]

            def need(t: str) -> None:
                p = producers.get(t)
                if p is None or p.name not in members or p.name in dup:
                    return
                dup.add(p.name)
                for i in p.input:
                    need(i)

            for n in s.nodes:
                for t in self.by_name[n].output:
                    outside = [
                        c for c in self.nodes if t in c.input and c.name not in members
                    ]
                    early = any(self.index[c.name] < fire for c in outside)
                    if outside and (t not in s.outputs or early):
                        need(t)
            for o in self.model.graph.output:
                if (
                    o.name in {t for n in s.nodes for t in self.by_name[n].output}
                    and o.name not in s.outputs
                ):
                    need(o.name)
        return dup

    def emitted(self, seg: Segment) -> bytes:
        if seg.name not in self._emitted:
            self._emitted[seg.name] = seg.emit().SerializeToString()
            if self.emit_dir:
                with open(
                    os.path.join(self.emit_dir, f"{seg.name}.axmodel"), "wb"
                ) as f:
                    f.write(self._emitted[seg.name])
        return self._emitted[seg.name]

    def _float(self, seg: Segment, env: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        local = dict(env)
        for n in sorted(seg.nodes, key=self.index.get):
            node = self.by_name[n]
            for t, v in zip([t for t in node.output if t], self.host.run(node, local)):
                local[t] = v
        return [local[t] for t in seg.outputs]

    def _sim(self, seg: Segment, env: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        local = dict(env)
        values = []
        for j, (t, qq) in enumerate(zip(seg.inputs, seg.in_q)):
            value = fake_quant(local[t], *qq)
            values.append(value)
        if seg.output_shape:
            target = np.broadcast_shapes(*(value.shape for value in values))
            for t, value in zip(seg.inputs, values):
                local[t] = np.broadcast_to(value, target)
        else:
            for t, value in zip(seg.inputs, values):
                local[t] = value
        for n in sorted(seg.nodes, key=self.index.get):
            node = self.by_name[n]
            for t, v in zip([t for t in node.output if t], self.host.run(node, local)):
                local[t] = v
        outs = [local[t] for t in seg.outputs]
        return [
            fake_quant(o, *qq) if qq and o.dtype.kind == "f" else o
            for o, qq in zip(outs, seg.out_q or [None] * len(outs))
        ]

    def _device(self, seg: Segment, env: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        m = self.session.load(self.emitted(seg))
        try:
            ins = [np.asarray(env[t], dtype=np.float32) for t in seg.inputs]
            if seg.output_shape:
                target = np.broadcast_shapes(*(x.shape for x in ins))
                ins = [np.broadcast_to(x, target) for x in ins]
            if seg.batch_split > 1:
                parts = [
                    self.session.run(
                        m,
                        [
                            np.array_split(x, seg.batch_split)[j] if s else x
                            for x, s in zip(ins, seg.split)
                        ],
                    )
                    for j in range(seg.batch_split)
                ]
                ys = [np.concatenate(p) for p in zip(*parts)]
            else:
                ys = self.session.run(m, ins)
        finally:
            self.session.unload(m)
        want = {o.name: o for o in self.model.graph.value_info}
        out = []
        for t, y in zip(seg.outputs, ys):
            vi = want.get(t)
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim] if vi else None
            y = y.astype(np.float32)
            out.append(
                y.reshape(shape) if shape and int(np.prod(shape)) == y.size else y
            )
        return out

    def run(
        self,
        feeds: Mapping[str, np.ndarray],
        mode: str = "npu",
        check: bool = True,
        keep: Sequence[str] = (),
        progress: bool = False,
        stats_out: list | None = None,
    ) -> tuple[dict[str, np.ndarray], list[SegStat]]:
        env: dict[str, np.ndarray] = dict(feeds)
        keep_set = set(keep) | {o.name for o in self.model.graph.output}
        stats: list[SegStat] = stats_out if stats_out is not None else []
        t0 = time.time()
        for k, node in enumerate(self.nodes):
            seg = self.seg_of.get(node.name)
            if seg is None or mode == "float" or node.name in self._host_dup:
                for t, v in zip(
                    [t for t in node.output if t], self.host.run(node, env)
                ):
                    env[t] = v
            if seg is not None and mode != "float" and self.fire_at[seg.name] == k:
                st = SegStat(seg.name, seg.kind, len(seg.nodes))
                t1 = time.time()
                sim = self._sim(seg, env) if (mode == "sim" or check) else None
                st.sim_s = time.time() - t1
                if mode == "npu":
                    t1 = time.time()
                    try:
                        self.emitted(seg)
                    except Exception as exc:
                        st.error = f"emit: {type(exc).__name__}: {exc}"
                    st.emit_s = time.time() - t1
                    t1 = time.time()
                    if not st.error:
                        try:
                            dev = self._device(seg, env)
                            self.device_runs += 1
                            if (
                                self.health_every
                                and self.device_runs % self.health_every == 0
                            ):
                                import axcl_session

                                axcl_session.health_check(self.session)
                        except Exception as exc:  # recorded, then fall back to sim
                            st.error = f"{type(exc).__name__}: {exc}"
                            if isinstance(exc, _device_errors()):
                                # the card is suspect: void everything from here on
                                self.stalled = [
                                    s.segment for s in stats[-self.health_every :]
                                ] + [seg.name]
                                stats.append(st)
                                raise
                    if st.error:
                        dev = sim if sim is not None else self._sim(seg, env)
                    st.device_s = time.time() - t1
                    if sim is not None and not st.error:
                        _compare(st, seg, dev, sim)
                        flt = self._float(seg, env)
                        st.float_rel = max(_rel(d, f) for d, f in zip(dev, flt))
                        st.sim_float_rel = max(_rel(s_, f) for s_, f in zip(sim, flt))
                        if not seg.out_q:
                            st.float_mismatch = max(
                                float(
                                    (
                                        np.abs(
                                            np.asarray(d, np.float32).ravel()
                                            - np.asarray(f, np.float32).ravel()
                                        )
                                        > 1e-6
                                    ).mean()
                                )
                                for d, f in zip(dev, flt)
                            )
                    res = dev
                else:
                    res = sim
                for t, v in zip(seg.outputs, res):
                    env[t] = v
                stats.append(st)
            for t in set(node.input):
                if self.last.get(t) == k and t not in keep_set and t not in feeds:
                    env.pop(t, None)
            if progress and k % 100 == 0:
                print(
                    f"  node {k}/{len(self.nodes)} {time.time() - t0:.1f}s", flush=True
                )
        return {t: env[t] for t in keep_set if t in env}, stats


def _rel(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _device_errors() -> tuple:
    import axcl_session

    return (axcl_session.DeviceStall,)


def _compare(st: SegStat, seg: Segment, dev, sim) -> None:
    for k, (d, s) in enumerate(zip(dev, sim)):
        d = np.asarray(d, np.float32).ravel()
        s = np.asarray(s, np.float32).ravel()
        if d.size != s.size:
            st.error = f"output {k}: {d.size} elements, expected {s.size}"
            return
        diff = np.abs(d - s)
        st.max_abs = max(st.max_abs, float(diff.max(initial=0)))
        if k < len(seg.out_q) and seg.out_q[k]:
            lsb = diff / np.float32(seg.out_q[k][0])
            st.max_lsb = max(st.max_lsb, float(lsb.max(initial=0)))
            st.frac_gt1 = max(
                st.frac_gt1, float((lsb > 1.01).mean()) if lsb.size else 0.0
            )
        else:  # unquantized output (compare/cast, transpose, gather): exact values
            st.frac_gt1 = max(
                st.frac_gt1, float((diff > 1e-6).mean()) if diff.size else 0.0
            )


# --------------------------------------------------------------------------
# step feeds / references


def _float_gradients(model, feeds, grad_names) -> dict[str, np.ndarray]:
    """The float gradients, from a host-only run (cached next to the step)."""
    cache = STEP_REF + ".grads.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        if set(z.files) == set(grad_names):
            return {k: z[k] for k in z.files}
    outs, _ = StepRunner(model, []).run(feeds, "float", keep=list(grad_names.values()))
    grads = {w: np.asarray(outs[t]) for w, t in grad_names.items()}
    np.savez(cache, **grads)
    return grads


def load_step(path: str = STEP_ONNX) -> onnx.ModelProto:
    return shape_inference.infer_shapes(onnx.load(path))


def load_records(path: str = STEP_OPS) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def load_reference(path: str = STEP_REF) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def gradient_tensors(model: onnx.ModelProto, state_map: Mapping) -> dict[str, str]:
    """``{weight: gradient tensor}``: each weight's Adam first-moment update
    is ``m' = beta1 * m + (1 - beta1) * g``; ``g`` is the non-constant input of
    the second Mul, i.e. the backward pass's output for that weight."""
    prod = {o: n for n in model.graph.node for o in n.output}
    consts = {i.name for i in model.graph.initializer}
    out = {}
    for state, new in state_map.items():
        if not state.endswith("__m"):
            continue
        add = prod[new]
        for t in add.input:
            mul = prod.get(t)
            if mul is None or mul.op_type != "Mul" or state in mul.input:
                continue
            live = [i for i in mul.input if i not in consts]
            if len(live) == 1:
                out[state[:-3]] = live[0]
    return out


def optimizer_nodes(model: onnx.ModelProto, state_map: Mapping) -> set[str]:
    """Nodes of the optimizer update: ancestors of the new state (weights,
    moments) that are not ancestors of any gradient, i.e. the Adam math."""
    prod = {o: n for n in model.graph.node for o in n.output}

    def ancestors(tensors) -> set[str]:
        seen: set[str] = set()
        stack = list(tensors)
        while stack:
            n = prod.get(stack.pop())
            if n is None or n.name in seen:
                continue
            seen.add(n.name)
            stack.extend(n.input)
        return seen

    grads = gradient_tensors(model, state_map)
    return ancestors(state_map.values()) - ancestors(grads.values())


def gradients_from_moments(outputs: Mapping, state_map: Mapping) -> dict:
    """At step 1 (m0 = 0), Adam's new first moment is 0.1 * grad: each weight's
    gradient is 10 * its ``__m`` output."""
    grads = {}
    for w, out in state_map.items():
        if w.endswith("__m"):
            grads[w[:-3]] = 10.0 * np.asarray(outputs[out], np.float32)
    return grads


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def segment_passed(st: Mapping) -> bool:
    """A device run agrees with the simulation: no error, at most 2 LSB off
    (rounding at a half-LSB tie on both sides), and 1 LSB or less on all but
    0.1% of the elements."""
    return not st["error"] and st["max_lsb"] <= 2.01 and st["frac_gt1"] <= 0.001


def _reason_counts(host: Mapping[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in host.values():
        k = re.sub(r"\d+", "#", v)[:90]
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def summarize(stats: Sequence[SegStat]) -> dict:
    by: dict[str, dict] = {}
    for s in stats:
        d = by.setdefault(
            s.kind,
            {
                "segments": 0,
                "nodes": 0,
                "errors": 0,
                "max_lsb": 0.0,
                "worst": "",
                "bad": 0,
                "device_s": 0.0,
            },
        )
        d["segments"] += 1
        d["nodes"] += s.nodes
        d["device_s"] += s.device_s
        if s.error:
            d["errors"] += 1
        if s.max_lsb > d["max_lsb"]:
            d["max_lsb"], d["worst"] = s.max_lsb, s.segment
        if s.frac_gt1 > 0.001:
            d["bad"] += 1
    return by


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--kinds", help="comma-separated segment kinds to put on the NPU")
    p.add_argument("--mode", default="npu", choices=["npu", "sim", "float"])
    p.add_argument("--out", required=True)
    p.add_argument("--emit-dir")
    p.add_argument(
        "--limit", type=int, default=0, help="only the first N segments on the NPU"
    )
    p.add_argument("--only", help="comma-separated segment names to put on the NPU")
    p.add_argument(
        "--exclude", help="regex: segments whose name matches stay on the host"
    )
    p.add_argument(
        "--host-optimizer",
        action="store_true",
        help="keep the Adam update (every node after the gradients) on the host in float",
    )
    p.add_argument(
        "--validated",
        help="a previous npu report: segments whose device output was more than "
        "2 LSB from the simulation there (or failed) stay on the host",
    )
    p.add_argument(
        "--include-unsafe",
        action="store_true",
        help="also run segments whose template semantics differ (Reshape->Relu on signed data)",
    )
    p.add_argument(
        "--health-every",
        type=int,
        default=1,
        help="native health check every N device runs",
    )
    args = p.parse_args(argv)

    model = load_step()
    records = load_records()
    calib = axb.load_calibration(STEP_CALIB)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    segs, host = build_plan(model, records, calib, kinds, args.include_unsafe)
    if args.only:
        only = set(args.only.split(","))
        segs = [s for s in segs if s.name in only]
    if args.exclude:
        for sg in [sg for sg in segs if re.search(args.exclude, sg.name)]:
            for n in sg.nodes:
                host[n] = "excluded on the command line"
        segs = [sg for sg in segs if not re.search(args.exclude, sg.name)]
    if args.host_optimizer:
        opt = optimizer_nodes(model, load_reference()["state_map"])
        for sg in [sg for sg in segs if set(sg.nodes) & opt]:
            for n in sg.nodes:
                host[n] = "optimizer update kept in float on the host"
        segs = [sg for sg in segs if not set(sg.nodes) & opt]
    if args.validated:
        with open(args.validated) as f:
            prev = {st["segment"]: st for st in json.load(f)["segment_stats"]}
        failed = {n for n, st in prev.items() if not segment_passed(st)}
        for sg in [sg for sg in segs if sg.name in failed]:
            for n in sg.nodes:
                host[n] = (
                    f"device output != simulation in {os.path.basename(args.validated)}"
                )
        segs = [sg for sg in segs if sg.name not in failed]
    if args.limit:
        segs = segs[: args.limit]
    segs, blobs = drop_unemittable(segs, host)
    ref = load_reference()
    feeds = ref["feeds"]
    grad_names = gradient_tensors(model, ref["state_map"])
    float_grads = _float_gradients(model, feeds, grad_names)
    npu_nodes = sum(len(s.nodes) for s in segs)
    print(
        f"{len(segs)} NPU segments covering {npu_nodes}/{len(model.graph.node)} nodes",
        flush=True,
    )
    if args.emit_dir:
        os.makedirs(args.emit_dir, exist_ok=True)

    npu_nodes = sum(len(s.nodes) for s in segs)
    report: dict[str, Any] = {
        "segments": len(segs),
        "npu_nodes": npu_nodes,
        "nodes": len(model.graph.node),
        "host_reasons": _reason_counts(host),
    }
    t0 = time.time()
    if args.mode == "npu":
        import axcl_session

        with axcl_session.AXSession() as sess:
            report["health_before_lsb"] = axcl_session.health_check(sess)
            runner = StepRunner(model, segs, sess, args.emit_dir, args.health_every)
            runner._emitted.update(blobs)
            try:
                outs, stats = runner.run(
                    feeds,
                    "npu",
                    keep=list(grad_names.values()),
                    progress=True,
                    stats_out=(stats := []),
                )
                report["health_after_lsb"] = axcl_session.health_check(sess)
            except axcl_session.DeviceStall as exc:
                report["stall"] = {"error": str(exc), "suspects": runner.stalled}
                report["segment_stats"] = [dataclasses.asdict(s) for s in stats]
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=1)
                print(json.dumps(report["stall"], indent=1))
                return 2
            report["device_exec_ms"] = sess.exec_us / 1000
            report["device_runs"] = sess.runs
    else:
        runner = StepRunner(model, segs, None, args.emit_dir)
        outs, stats = runner.run(
            feeds, args.mode, check=False, keep=list(grad_names.values()), progress=True
        )
    report["wall_s"] = time.time() - t0
    report["per_kind"] = summarize(stats)
    report["segment_stats"] = [dataclasses.asdict(s) for s in stats]

    ref_out = ref["ref"]
    loss_name = "distill__add_27"
    if loss_name in outs and loss_name in ref_out:
        report["loss"] = {
            "run": float(np.ravel(outs[loss_name])[0]),
            "float": float(np.ravel(ref_out[loss_name])[0]),
        }
    # backward-pass gradients (the tensors the Adam update consumes)
    report["grads"] = {
        w: {
            "rel_err": rel_err(outs[t], float_grads[w]),
            "cos": cos(outs[t], float_grads[w]),
        }
        for w, t in grad_names.items()
    }
    # the step's actual result: each weight's update w' - w
    report["updates"] = {}
    for w, out in ref["state_map"].items():
        if w.endswith("__m") or w.endswith("__v") or out not in outs:
            continue
        d_run = np.asarray(outs[out], np.float64) - feeds[w]
        d_ref = np.asarray(ref_out[out], np.float64) - feeds[w]
        report["updates"][w] = {
            "rel_err": rel_err(d_run, d_ref),
            "cos": cos(d_run, d_ref),
        }
    for what in ("grads", "updates"):
        vals = report[what].values()
        report[f"{what}_summary"] = {
            "cos_median": float(np.nanmedian([v["cos"] for v in vals])),
            "cos_min": float(np.nanmin([v["cos"] for v in vals])),
            "rel_err_median": float(np.nanmedian([v["rel_err"] for v in vals])),
            "nan": int(sum(not np.isfinite(v["cos"]) for v in vals)),
        }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("segment_stats", "grads", "updates")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
