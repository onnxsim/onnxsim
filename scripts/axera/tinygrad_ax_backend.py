"""Template-plus-patch AX650 backend skeleton for tinygrad (architecture B).

First code for ``docs/axera-tinygrad-emitter-plan.md`` (#1834). It wires the
emitters this repository has already validated into the interfaces the plan
names, and exposes tinygrad-facing classes built on the user's tinygrad fork
``onnxsim/tinygrad`` (pinned below). It compiles nothing and touches no device:

* ``TemplateCache`` resolves a ``TemplateKey`` to a committed, Pulsar2-built and
  checked fixture. A key with no fixture raises ``ValueError``; building a new
  template (a Pulsar2 run) is deliberately out of scope here.
* ``Edit`` implementations change values inside a template without changing its
  structure, each by delegating to an existing emitter:
  ``GatherIndexEdit`` (``memory_emit.py``), ``TemplateOnly`` (Transpose,
  ``transpose_real_shapes.py``), ``ElementwiseScaleEdit`` (Relu/Sqrt,
  ``elementwise_scale_emit.py``; same-shape Add/Sub/Mul/Div,
  ``binary_op_scale_emit.py``), ``ConvWeightEdit`` (``conv_weight_learn.py``,
  ``conv_bias_requant.py``). ``predicted_npu_params`` exposes the tile-table
  predictors (``dma_tile_predict.py``, ``add_tile_predict.py``,
  ``elementwise_two_input_tile_predict.py``) as a cross-check.
* ``plan_node`` / ``coverage_report`` map a real graph's nodes onto what the
  backend can produce today and say why the rest is refused.
* ``tinygrad_classes()`` returns ``AXCompiler`` (a tinygrad ``Compiler`` whose
  "source" is a JSON template request and whose output is patched ``.axmodel``
  bytes) plus ``AXAllocator``/``AXProgram`` transport stubs that raise
  ``NotImplementedError`` -- the AXCL transport is roadmap milestone 1 and needs
  the device.

Every unvalidated op, shape, calibration class or edit raises ``ValueError``;
nothing here extrapolates.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import os
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import binary_op_scale_emit as bse  # noqa: E402
import elementwise_scale_emit as ew  # noqa: E402
import memory_emit  # noqa: E402
import transpose_real_shapes  # noqa: E402

TOOLCHAIN = "pulsar2:7.0-lite"
TINYGRAD_FORK = "onnxsim/tinygrad"
TINYGRAD_FORK_SHA = "ba00cbb3c6b8a6bcbee04c22cb4a0b0b9cee216e"
"""The ``onnxsim/tinygrad`` master commit this skeleton was written against.
Install with ``uv run --with "tinygrad@https://github.com/onnxsim/tinygrad/
archive/<sha>.tar.gz"``."""

_FIXTURES = os.path.join(_HERE, "fixtures")

# (x shape, w shape, strides, pads) -> template for the Conv weight edit.
# Only the two shapes whose full npu_params pipeline was validated against a
# native held-out build are listed (docs/axera-conv-weight-learn-stem.md, #1769;
# docs/axera-conv-weight-learn-downsample.md, #1771).
_CONV_TEMPLATES: dict[tuple, dict[str, Any]] = {
    ((16, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1, 1, 1)): {
        "dir": "conv_weight_learn",
        "model": "reference.axmodel.gz",
        "map": "stage1_map.npz",
        "kind": "biased",
        "block_at": 36864,
    },
    ((16, 64, 56, 56), (128, 64, 1, 1), (2, 2), (0, 0, 0, 0)): {
        "dir": "conv_learn_downsample",
        "model": "s1_reference.axmodel.gz",
        "map": "s1_map.npz",
        "kind": "contiguous",
        "block_at": 9216,
        "block_len": 2 * 4 * 128,
    },
}


def _tup(x) -> tuple:
    return tuple(int(v) for v in x)


@dataclasses.dataclass(frozen=True)
class TemplateKey:
    """What a cached compiled template is keyed by (plan section 6).

    ``shapes`` are the data inputs' shapes. ``attrs`` carries what changes the
    compiled program: ``perm`` (Transpose), ``indices`` = index count (Gather),
    ``w``/``strides``/``pads`` (Conv). ``calibration_class`` names the part of
    the calibration a value edit cannot change -- for Relu/Sqrt the zero points,
    e.g. ``"x0,y0"``.
    """

    op: str
    shapes: tuple[tuple[int, ...], ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    dtypes: tuple[str, ...] = ("float32",)
    calibration_class: str = ""
    toolchain: str = TOOLCHAIN

    def attr(self, name: str, default=None):
        return dict(self.attrs).get(name, default)

    def to_json(self) -> dict:
        return {
            "op": self.op,
            "shapes": [list(s) for s in self.shapes],
            "attrs": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in self.attrs
            },
            "dtypes": list(self.dtypes),
            "calibration_class": self.calibration_class,
            "toolchain": self.toolchain,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> TemplateKey:
        attrs = tuple(
            sorted(
                (k, tuple(v) if isinstance(v, list) else v)
                for k, v in d.get("attrs", {}).items()
            )
        )
        return cls(
            op=d["op"],
            shapes=tuple(_tup(s) for s in d["shapes"]),
            attrs=attrs,
            dtypes=tuple(d.get("dtypes", ("float32",))),
            calibration_class=d.get("calibration_class", ""),
            toolchain=d.get("toolchain", TOOLCHAIN),
        )


def _zero_points_from_class(cls: str) -> dict[str, int]:
    out = {}
    for part in filter(None, cls.split(",")):
        out[part[0]] = int(part[1:])
    return out


def _load_gz_model(path: str) -> onnx.ModelProto:
    with gzip.open(path, "rb") as f:
        return onnx.load_model_from_string(f.read())


@dataclasses.dataclass
class TemplateEntry:
    kind: str
    path: str
    meta: dict


class TemplateCache:
    """Resolves keys to committed Pulsar2-built fixtures. No builds."""

    def lookup(self, key: TemplateKey) -> TemplateEntry:
        if key.toolchain != TOOLCHAIN:
            raise ValueError(f"no templates for toolchain {key.toolchain!r}")
        if key.dtypes != ("float32",):
            raise ValueError(f"only float32 templates exist, got {key.dtypes}")
        op = key.op
        if op == "Gather":
            (shape,) = key.shapes
            pair = (shape, key.attr("indices"))
            name = memory_emit._GATHER_LAST_AXIS_TEMPLATES.get(pair)
            if name is None or key.attr("axis", len(shape) - 1) != len(shape) - 1:
                raise ValueError(f"no last-axis Gather template for {pair}")
            return TemplateEntry(
                "gather", os.path.join(_FIXTURES, name), {"pair": pair}
            )
        if op == "Transpose":
            (shape,) = key.shapes
            path = transpose_real_shapes.template_path(shape, key.attr("perm"))
            return TemplateEntry("transpose", path, {})
        if op in ew.OPS:
            (shape,) = key.shapes
            zps = _zero_points_from_class(key.calibration_class)
            _, meta = ew.load_template(op, shape, zps)
            return TemplateEntry(
                "elementwise", os.path.join(ew.TEMPLATE_DIR, meta["file"]), meta
            )
        if op in bse.OPS:
            (shape,) = key.shapes
            zps = _zero_points_from_class(key.calibration_class)
            _, meta = bse.load_template(op, shape, zps)
            return TemplateEntry(
                "binary", os.path.join(bse.TEMPLATE_DIR, meta["file"]), meta
            )
        if op == "Conv":
            (shape,) = key.shapes
            ck = (
                shape,
                _tup(key.attr("w", ())),
                _tup(key.attr("strides", (1, 1))),
                _tup(key.attr("pads", (0, 0, 0, 0))),
            )
            meta = _CONV_TEMPLATES.get(ck)
            if meta is None:
                raise ValueError(f"no validated Conv template for {ck}")
            return TemplateEntry(
                "conv", os.path.join(_FIXTURES, meta["dir"], meta["model"]), meta
            )
        raise ValueError(f"no templates for op {op!r}")

    def load(self, key: TemplateKey) -> onnx.ModelProto:
        return _load_gz_model(self.lookup(key).path)

    def get_or_build(self, key: TemplateKey, onnx_bytes: bytes | None = None):
        """Plan section 6: build on a miss. Builds are Pulsar2 runs, out of scope
        for this skeleton, so a miss is reported instead of built."""
        try:
            return self.lookup(key).path
        except ValueError as exc:
            raise NotImplementedError(
                f"template miss needs a Pulsar2 build (not in this skeleton): {exc}"
            ) from exc


def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    for init in model.graph.initializer:
        if init.name == name:
            return init
    raise ValueError(f"model has no initializer {name!r}")


def _mcode_initializer(model: onnx.ModelProto) -> onnx.TensorProto:
    found = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(found) != 1:
        raise ValueError(f"expected one *_neu MCode initializer, found {len(found)}")
    return found[0]


class Edit(Protocol):
    """A value change inside a template (plan section 6). ``validate`` refuses
    anything not decoded for that exact template; ``apply`` returns the edited
    model."""

    def validate(self, key: TemplateKey, entry: TemplateEntry) -> None: ...

    def apply(
        self, key: TemplateKey, entry: TemplateEntry, model: onnx.ModelProto
    ) -> onnx.ModelProto: ...

    def to_json(self) -> dict: ...


@dataclasses.dataclass
class TemplateOnly:
    """The template is already the complete artifact (Transpose: no data)."""

    def validate(self, key, entry):
        if entry.kind != "transpose":
            raise ValueError(f"{key.op} template needs an edit, not TemplateOnly")

    def apply(self, key, entry, model):
        return model

    def to_json(self):
        return {"type": "template_only"}


@dataclasses.dataclass
class GatherIndexEdit:
    """Retarget a last-axis Gather's constant indices (``memory_emit.py``)."""

    indices: Sequence[int]

    def validate(self, key, entry):
        if entry.kind != "gather":
            raise ValueError("GatherIndexEdit applies only to Gather templates")
        width = key.shapes[0][-1]
        if len(self.indices) != key.attr("indices"):
            raise ValueError("index count must match the template")
        if any(not 0 <= int(i) < width for i in self.indices):
            raise ValueError(f"indices must lie in [0, {width})")

    def apply(self, key, entry, model):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.axmodel")
            out = os.path.join(tmp, "out.axmodel")
            onnx.save(model, ref)
            memory_emit.emit_gather_last_axis_axmodel(
                ref, out, indices=[int(i) for i in self.indices]
            )
            return onnx.load(out, load_external_data=False)

    def to_json(self):
        return {"type": "gather_indices", "indices": [int(i) for i in self.indices]}


@dataclasses.dataclass
class ElementwiseScaleEdit:
    """New per-tensor scales at the template's zero points: Relu/Sqrt
    (``elementwise_scale_emit.py``, #1840) or same-shape Add/Sub/Mul/Div
    (``binary_op_scale_emit.py``)."""

    scales: Mapping[str, float]

    def validate(self, key, entry):
        if entry.kind == "binary":
            bse.op_values(key.op, self.scales)  # raises on missing/invalid scales
            return
        if entry.kind != "elementwise":
            raise ValueError(
                "ElementwiseScaleEdit applies only to Relu/Sqrt/Add/Sub/Mul/Div"
            )
        ew.op_floats(key.op, self.scales)  # raises on missing/invalid scales

    def apply(self, key, entry, model):
        if entry.kind == "binary":
            return bse.emit_model(
                model,
                key.op,
                entry.meta["scales"],
                dict(self.scales),
                entry.meta["zero_points"],
            )
        init = _mcode_initializer(model)
        init.raw_data = ew.retarget(
            bytes(init.raw_data), key.op, entry.meta["scales"], dict(self.scales)
        )
        return model

    def to_json(self):
        return {"type": "elementwise_scales", "scales": dict(self.scales)}


@dataclasses.dataclass
class ConvWeightEdit:
    """New frozen weights for a Conv template (``npu_params`` weight codes plus
    the bias-aware requant block). The MCode is the template's, unedited, so the
    activation calibration must be the template's own; this is the frozen-weight
    refresh path, not the training path (trainable weights are graph inputs)."""

    w: np.ndarray
    b: np.ndarray
    x_scale: float
    x_zero: float
    y_scale: float
    y_zero: float

    def validate(self, key, entry):
        if entry.kind != "conv":
            raise ValueError("ConvWeightEdit applies only to Conv templates")
        if tuple(self.w.shape) != _tup(key.attr("w", ())):
            raise ValueError(f"weight shape {self.w.shape} != template {key.attr('w')}")
        if tuple(self.b.shape) != (self.w.shape[0],):
            raise ValueError("bias must have Cout elements")

    def apply(self, key, entry, model):
        import conv_bias_requant
        import conv_weight_learn

        meta = entry.meta
        origin = np.load(os.path.join(_FIXTURES, meta["dir"], meta["map"]))["origin"]
        table_init = _initializer(model, "npu_params")
        ref = np.frombuffer(bytes(table_init.raw_data), np.uint8)
        args = (self.x_scale, self.x_zero, self.y_scale, self.y_zero)
        if meta["kind"] == "biased":
            table = conv_weight_learn.emit_biased(
                ref, origin, self.w, self.b, *args, meta["block_at"]
            )
        else:
            table = conv_bias_requant.emit_conv_table(
                ref,
                origin,
                self.w,
                self.b,
                *args,
                block_at=meta["block_at"],
                block_len=meta["block_len"],
            )
        table_init.raw_data = np.asarray(table, np.uint8).tobytes()
        return model

    def to_json(self):
        raise ValueError("ConvWeightEdit carries arrays; pass it in-process")


def edit_from_json(d: Mapping) -> Edit:
    kind = d.get("type")
    if kind == "template_only":
        return TemplateOnly()
    if kind == "gather_indices":
        return GatherIndexEdit(list(d["indices"]))
    if kind == "elementwise_scales":
        return ElementwiseScaleEdit(dict(d["scales"]))
    raise ValueError(f"unknown or non-serializable edit {kind!r}")


@dataclasses.dataclass
class EditSet:
    edits: list

    def build(self, key: TemplateKey, cache: TemplateCache | None = None):
        """Load the template for ``key``, validate every edit against it, then
        apply them in order. Nothing is applied if any edit is refused."""
        cache = cache or TemplateCache()
        entry = cache.lookup(key)
        if not self.edits:
            raise ValueError("an EditSet needs at least one edit (TemplateOnly ok)")
        for edit in self.edits:
            edit.validate(key, entry)
        model = _load_gz_model(entry.path)
        for edit in self.edits:
            model = edit.apply(key, entry, model)
        return model


def predicted_npu_params(
    op: str, shape: Sequence[int], scales: Mapping[str, float] | None = None
) -> bytes:
    """The tile-table predictors behind one call. Raises outside their domains."""
    import dma_tile_predict

    shape = [int(d) for d in shape]
    if op in ("Relu", "Sqrt"):
        return dma_tile_predict.predict_params(shape)
    if scales is None:
        raise ValueError(f"{op} tile prediction needs x/z/y scales")
    if op == "Add":
        import add_tile_predict

        return add_tile_predict.predict_params(
            shape, scales["x"], scales["z"], scales["y"]
        )
    if op in ("Sub", "Mul", "Div"):
        import elementwise_two_input_tile_predict as two

        return two.predict_params(op, shape, scales["x"], scales["z"], scales["y"])
    raise ValueError(f"no tile predictor for {op!r}")


# --------------------------------------------------------------------------
# Graph side: real graph nodes -> keys -> which edit, or why not.
# --------------------------------------------------------------------------

_ELEMENTWISE_ZP_CLASSES = ("x0,y0", "x128,y128")
# Div's denominators in the step are positive (sqrt(v) + eps), hence z0.
_BINARY_ZP_CLASSES = {
    "Add": ("x0,y0,z0", "x128,y128,z128"),
    "Sub": ("x0,y0,z0", "x128,y128,z128"),
    "Mul": ("x0,y0,z0", "x128,y128,z128"),
    "Div": ("x0,y0,z0", "x128,y128,z0"),
}


def extract_step_ops(onnx_path: str) -> list[dict]:
    """Per-node records of a real graph for ``coverage_report``: op type, data
    input shapes, and the attributes that pick a template."""
    from onnx import shape_inference

    model = shape_inference.infer_shapes(onnx.load(onnx_path, load_external_data=False))
    shapes: dict[str, list[int]] = {}
    for v in (
        list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    ):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    inits = {i.name: i for i in model.graph.initializer}
    consts = {n.output[0] for n in model.graph.node if n.op_type == "Constant"}
    graph_inputs = {v.name for v in model.graph.input}
    records = []
    for node in model.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        rec: dict[str, Any] = {"op": node.op_type, "attrs": {}}
        data_inputs = [i for i in node.input if i and i not in inits]
        rec["shapes"] = [shapes.get(i, []) for i in data_inputs[:1]]
        if node.op_type == "Transpose":
            rec["attrs"]["perm"] = list(attrs.get("perm", []))
        elif node.op_type == "Gather":
            rank = len(rec["shapes"][0]) if rec["shapes"] else 0
            rec["attrs"]["axis"] = int(attrs.get("axis", 0)) % max(rank, 1)
            idx = inits.get(node.input[1])
            rec["attrs"]["indices"] = int(np.prod(idx.dims)) if idx is not None else -1
        elif node.op_type == "Conv":
            rec["attrs"]["w"] = shapes.get(node.input[1]) or list(
                inits[node.input[1]].dims
            )
            rec["attrs"]["strides"] = list(attrs.get("strides", [1, 1]))
            rec["attrs"]["pads"] = list(attrs.get("pads", [0, 0, 0, 0]))
            rec["attrs"]["weight_is_graph_input"] = node.input[1] in graph_inputs
        elif node.op_type == "Reshape":
            rec["attrs"]["out"] = shapes.get(node.output[0], [])
        elif node.op_type in bse.OPS:
            a, b = node.input[:2]
            if {a, b} & (set(inits) | consts):
                rec["attrs"]["form"] = "const"
            elif shapes.get(a) != shapes.get(b):
                rec["attrs"]["form"] = "broadcast"
            else:
                rec["attrs"]["form"] = "same_shape"
        records.append(rec)
    return records


def key_for_record(rec: Mapping, calibration_class: str = "") -> TemplateKey:
    attrs = rec.get("attrs", {})
    keep: dict[str, Any] = {}
    if rec["op"] == "Transpose":
        keep["perm"] = tuple(attrs["perm"])
    elif rec["op"] == "Gather":
        keep["axis"] = attrs["axis"]
        keep["indices"] = attrs["indices"]
    elif rec["op"] == "Conv":
        keep["w"] = tuple(attrs["w"])
        keep["strides"] = tuple(attrs["strides"])
        keep["pads"] = tuple(attrs["pads"])
    return TemplateKey(
        op=rec["op"],
        shapes=tuple(_tup(s) for s in rec["shapes"]),
        attrs=tuple(sorted(keep.items())),
        calibration_class=calibration_class,
    )


def plan_node(rec: Mapping, cache: TemplateCache | None = None) -> tuple[str, str]:
    """``(status, detail)`` for one graph node. ``status`` is ``"covered"`` (a
    template plus an edit reproduce it), ``"conditional"`` (covered only if a
    calibration property holds that the graph alone does not show), or
    ``"refused"``."""
    cache = cache or TemplateCache()
    op = rec["op"]
    attrs = rec.get("attrs", {})
    try:
        if op == "Conv":
            key = key_for_record(rec)
            cache.lookup(key)
            if attrs.get("weight_is_graph_input"):
                return (
                    "refused",
                    "trainable weight is a graph input (runtime state); a template "
                    "exists for frozen-weight deployment only",
                )
            return ("covered", "ConvWeightEdit")
        if op in ew.OPS:
            hits = []
            for zp in _ELEMENTWISE_ZP_CLASSES:
                try:
                    cache.lookup(key_for_record(rec, zp))
                    hits.append(zp)
                except ValueError:
                    pass
            if not hits:
                raise ValueError("no template at this shape")
            return (
                "conditional",
                f"ElementwiseScaleEdit if zero points are one of {hits}",
            )
        if op in bse.OPS:
            form = attrs.get("form")
            if form != "same_shape":
                return (
                    "refused",
                    f"{op} with a {form} operand compiles to a different program; "
                    "templates exist for two live same-shape inputs only",
                )
            hits = []
            for zp in _BINARY_ZP_CLASSES[op]:
                try:
                    cache.lookup(key_for_record(rec, zp))
                    hits.append(zp)
                except ValueError:
                    pass
            if not hits:
                raise ValueError("no template at this shape")
            return (
                "conditional",
                f"ElementwiseScaleEdit if zero points are one of {hits}",
            )
        if op == "Gather":
            cache.lookup(key_for_record(rec))
            if attrs.get("indices", -1) < 0:
                raise ValueError("indices are not a constant initializer")
            return ("covered", "GatherIndexEdit")
        if op == "Transpose":
            cache.lookup(key_for_record(rec))
            return ("covered", "TemplateOnly")
        if op == "Reshape":
            shape = rec["shapes"][0] if rec["shapes"] else []
            out = attrs.get("out", [])
            if len(shape) == 2 and shape[0] == 1 and out == shape[1:]:
                return (
                    "conditional",
                    "bias flatten fuses into its neighbour (reshape_emit.py, "
                    "Relu-neighbour pairs only)",
                )
            return ("refused", "non-fused Reshape emits real DMA MCode (undecoded)")
        return ("refused", f"no template or edit for {op}")
    except ValueError as exc:
        return ("refused", str(exc))


def coverage_report(records: Sequence[Mapping]) -> dict:
    """Per-op counts of covered / conditional / refused nodes, with reasons."""
    cache = TemplateCache()
    per_op: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        status, detail = plan_node(rec, cache)
        per_op[rec["op"]][status] += 1
        reasons[rec["op"]][f"{status}: {detail}"] += 1
    totals = Counter()
    for c in per_op.values():
        totals.update(c)
    return {
        "nodes": len(records),
        "totals": dict(totals),
        "per_op": {op: dict(c) for op, c in sorted(per_op.items())},
        "reasons": {op: dict(c) for op, c in sorted(reasons.items())},
    }


# --------------------------------------------------------------------------
# tinygrad seam (fork pinned above). Imported lazily so the rest of this
# module works without tinygrad installed.
# --------------------------------------------------------------------------


def build_request(key: TemplateKey, edits: Sequence[Edit]) -> str:
    """The "source" ``AXCompiler`` compiles: a JSON template request. This is
    what a graph/JIT-level hook would emit for one fused subgraph (plan section
    5: the compile unit is the fused subgraph, not a per-kernel renderer)."""
    return json.dumps(
        {"key": key.to_json(), "edits": [e.to_json() for e in edits]}, sort_keys=True
    )


def compile_request(src: str, cache: TemplateCache | None = None) -> bytes:
    req = json.loads(src)
    key = TemplateKey.from_json(req["key"])
    model = EditSet([edit_from_json(e) for e in req["edits"]]).build(key, cache)
    return model.SerializeToString()


def tinygrad_classes() -> dict[str, type]:
    """``AXCompiler`` / ``AXAllocator`` / ``AXProgram`` subclassing the fork's
    ``tinygrad.device`` bases. Only the compiler does anything."""
    from tinygrad.device import Allocator, Compiler, Program

    class AXCompiler(Compiler):
        """Architecture B's compiler: template lookup + validated edits. The
        cache key disables tinygrad's disk cache; templates are already cached
        as fixtures."""

        def __init__(self, cache: TemplateCache | None = None):
            super().__init__(cachekey=None)
            self.cache = cache or TemplateCache()

        def compile(self, src: str) -> bytes:
            return compile_request(src, self.cache)

    class AXAllocator(Allocator):
        def _alloc(self, size, options):
            raise NotImplementedError("AXCL buffers: roadmap milestone 1 (device)")

        def _free(self, opaque, options):
            raise NotImplementedError("AXCL buffers: roadmap milestone 1 (device)")

    class AXProgram(Program):
        def __init__(self, dev, obj):
            raise NotImplementedError("AXCL load/run: roadmap milestone 1 (device)")

    return {
        "AXCompiler": AXCompiler,
        "AXAllocator": AXAllocator,
        "AXProgram": AXProgram,
    }


def npu_params_of(model_bytes: bytes) -> bytes:
    model = onnx.load_model_from_string(model_bytes)
    return bytes(_initializer(model, "npu_params").raw_data)


def mcode_of(model_bytes: bytes) -> bytes:
    return bytes(_mcode_initializer(onnx.load_model_from_string(model_bytes)).raw_data)


def gather_indices_of(model_bytes: bytes, count: int) -> list[int]:
    table = npu_params_of(model_bytes)
    return list(struct.unpack(f"<{count}I", table[: 4 * count]))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="write a graph's op records as gzipped JSON")
    ex.add_argument("onnx")
    ex.add_argument("out")
    cov = sub.add_parser("coverage", help="coverage report from op records")
    cov.add_argument("records")
    args = p.parse_args(argv)
    if args.cmd == "extract":
        with gzip.open(args.out, "wt") as f:
            json.dump(extract_step_ops(args.onnx), f)
        return 0
    with gzip.open(args.records, "rt") as f:
        print(json.dumps(coverage_report(json.load(f)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
