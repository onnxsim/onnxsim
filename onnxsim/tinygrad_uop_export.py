"""Exports a tinygrad ``UOp`` graph as ONNX, one ``NodeProto`` per ``UOp``,
under a custom, non-standard domain (``"tinygrad.uop"``) -- and imports it
back into a real ``UOp`` graph. Answers "could we represent a UOp graph in
ONNX, and round-trip it back" for :mod:`onnxsim.webgpu_tinygrad_codegen`'s
own UOp graphs (the same per-kernel ``Ops.SINK``-rooted ASTs that module's
own ``_lower_tensor_program`` walks and ``WGSLRenderer`` turns into WGSL).

``tinygrad`` is an **optional** dependency, matching the precedent set by
:mod:`onnxsim.webgpu_tinygrad_codegen` (see that module's own docstring):
nothing here runs, or is even imported, unless one of this module's public
functions is actually called.

## Two export shapes

- :func:`uop_to_onnx_model` -- a standalone ``ModelProto``: one flat graph,
  every ``UOp`` a node, synthetic ``u<i>`` names throughout (including the
  graph's own single output).
- :func:`uop_to_onnx_function` -- the same nodes, but wrapped as an ONNX
  **local function** (``FunctionProto``, ``model.functions``) with a real
  I/O boundary (caller-supplied leaf names, e.g. a node's actual tensor
  names) instead of synthetic ones throughout, referenced from the
  top-level graph by a single ``NodeProto``. This is the more useful shape
  in practice: a generic viewer (Netron has a dedicated "expand function"
  feature) shows the ordinary-looking op first and lets a reader drill into
  its UOp-level decomposition on demand, rather than only ever showing one
  giant flat graph of raw UOps.

Neither is runnable by any ONNX runtime -- no runtime implements
``ALU``/``RANGE``/``BUFFER``/``REDUCE``/... as tensor ops.

## Why bother, given it can't run

tinygrad already has its own UOp graph export/inspection story -- ``VIZ=1``
records every graph-rewrite step as a ``RewriteTrace`` dataclass and pickles
it to a temp file (``tinygrad/viz/serve.py``), then serves a bundled
d3.js/dagre web UI that converts a ``UOp`` to a JSON node/edge structure
(``uop_to_json``) on the fly for rendering. This module trades that for:

- **A stable, safe container.** A pickle is a live Python object graph
  frozen to disk -- unpickling runs arbitrary code, and the format silently
  breaks across tinygrad versions whenever a pickled class's shape changes
  (``UOp``, ``ShapeTracker``, ``KernelInfo``, ...). A ``.onnx`` file is
  plain protobuf: safe to open from anywhere, forward/backward tolerant of
  unknown fields the way protobuf always is, and -- unlike a pickle --
  actually **importable back** into a real ``UOp`` graph by this same
  module (see below), with no arbitrary code execution involved at any
  point: decoding is a small whitelisted dispatch over JSON, never
  ``eval``/``pickle.loads``.
- **Free generic tooling.** Netron (and any other protobuf/ONNX-aware
  viewer) renders a custom, unrecognized domain's nodes and edges
  generically -- readable graph structure with no bespoke UI to build or
  maintain.

## Encoding ``UOp.arg``, faithfully, for a whitelisted set of shapes

``UOp.arg``'s type varies wildly by op: a plain ``int``/``float``/``bool``
for ``CONST``, a ``ConstFloat`` (a tagged ``float`` subclass) also for
``CONST``, an ``(int, AxisType)`` pair for ``RANGE``, an ``(Ops, int)`` pair
for ``REDUCE``, a ``ParamArg`` dataclass (itself holding a ``DType`` and an
``AddrSpace`` enum member) for ``PARAM``, a ``KernelInfo`` dataclass for
``SINK``, or plain ``None`` for most ALU/``INDEX``/``STORE``/``END`` nodes.
:func:`_encode_value`/:func:`_decode_value` handle exactly this whitelisted
set -- recursively, as a single JSON-valued ``arg_json`` string attribute
(tagged by Python type, e.g. ``{"t": "AxisType", "name": "WEAK"}``) -- and
raise a clear ``TypeError``/``ValueError`` for anything outside it, rather
than silently guessing or falling back to an unparseable ``repr()``.

Two known, deliberate gaps, both raising clearly rather than mis-encoding:

- A pre-schedule ``UOp`` graph (movement ops with a ``ShapeTracker``/``View``
  arg) is out of scope -- this module targets the *post-schedule* per-kernel
  AST ``_lower_tensor_program`` itself works with, where movement ops have
  already been lowered into index arithmetic.
- ``KernelInfo.applied_opts``/``opts_to_apply``/``estimates`` are only
  supported in their default (empty/``None``) state -- a BEAM-search-tuned
  kernel's exact tuning is not reconstructed. ``name``, ``axis_types``,
  ``dont_use_locals``, and ``beam`` always round-trip.

## Verified round trip

``tests/test_tinygrad_uop_export.py`` doesn't just check structural
equality after import -- it re-renders the *reconstructed* graph through
the exact same ``to_program``/``WGSLRenderer`` pipeline
``onnxsim.webgpu_tinygrad_codegen`` uses, for a real ``Conv2D`` kernel AST,
and asserts the WGSL text is **byte-identical** to rendering the original.
That is the actual bar for "faithful": everything the real codegen path
reads off a ``UOp`` survives the round trip, not just what a human
eyeballing the graph would notice missing.

## Scope

Exports/imports exactly the ``UOp`` DAG reachable from one root (typically
an ``Ops.SINK``-rooted per-kernel AST, the same one
:func:`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program` passes to
``to_program``) -- not a whole multi-kernel schedule (``schedule_linear()``'s
own result, which threads several such ASTs together via ``Ops.CALL``
nodes). Handling one of those is just calling this once per kernel AST;
stitching multiple exported graphs/functions into one file isn't done here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import onnx
from onnx import helper

if TYPE_CHECKING:
    from tinygrad.uop.ops import UOp

__all__ = [
    "DOMAIN",
    "uop_to_onnx_model",
    "uop_to_onnx_function",
    "onnx_model_to_uop",
    "onnx_function_to_uop",
]

#: A deliberately unregistered, private domain string -- not
#: ``ai.onnx``/``com.microsoft``/anything a real ONNX runtime would ever
#: recognize, matching the "custom domain" convention ONNX itself documents
#: (see ``AttributeProto``/``NodeProto.domain``'s own comments in
#: ``onnx.proto``) for exactly this kind of private, non-interoperable
#: extension.
DOMAIN = "tinygrad.uop"
_OPSET_VERSION = 1


def _require_tinygrad():
    try:
        import tinygrad  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "onnxsim.tinygrad_uop_export needs the optional 'tinygrad' "
            "package: pip install tinygrad"
        ) from e


# --------------------------------------------------------------------------
# Value <-> JSON, for a whitelisted set of UOp.arg shapes (see module
# docstring). Never eval()/pickle.loads() -- decoding is a plain dispatch
# over an explicit "t" tag this module itself writes, so an untrusted file
# can make this raise, never execute anything.
# --------------------------------------------------------------------------


def _encode_float(v: float):
    """Standard JSON has no ``Infinity``/``NaN`` literal -- same convention
    (and same reason) as ``onnxsim.webgpu_kernel_metadata._encode_float``.
    """
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "Infinity"
    if v == float("-inf"):
        return "-Infinity"
    return v


def _decode_float(v) -> float:
    return float(v)


def _encode_value(v: Any) -> Any:
    from tinygrad.dtype import ConstFloat, DType
    from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, ParamArg

    if v is None:
        return {"t": "none"}
    if isinstance(v, bool):
        return {"t": "bool", "v": v}
    if isinstance(v, ConstFloat):
        return {"t": "ConstFloat", "v": _encode_float(float(v))}
    if isinstance(v, int):
        return {"t": "int", "v": v}
    if isinstance(v, float):
        return {"t": "float", "v": _encode_float(v)}
    if isinstance(v, str):
        return {"t": "str", "v": v}
    if isinstance(v, tuple):
        return {"t": "tuple", "v": [_encode_value(x) for x in v]}
    if isinstance(v, DType):
        return {"t": "DType", "name": str(v).removeprefix("dtypes.")}
    if isinstance(v, Ops):
        return {"t": "Ops", "name": v.name}
    if isinstance(v, AxisType):
        return {"t": "AxisType", "name": v.name}
    if isinstance(v, AddrSpace):
        return {"t": "AddrSpace", "name": v.name}
    if isinstance(v, ParamArg):
        return {
            "t": "ParamArg",
            "fields": {
                "slot": _encode_value(v.slot),
                "dtype": _encode_value(v.dtype),
                "vmin_vmax": _encode_value(v.vmin_vmax),
                "multiple_of": _encode_value(v.multiple_of),
                "name": _encode_value(v.name),
                "addrspace": _encode_value(v.addrspace),
                "axis": _encode_value(v.axis),
                "device": _encode_value(v.device),
                "volatile": _encode_value(v.volatile),
            },
        }
    if isinstance(v, KernelInfo):
        if v.applied_opts or v.opts_to_apply is not None or v.estimates is not None:
            raise ValueError(
                "KernelInfo.applied_opts/opts_to_apply/estimates are only "
                "supported empty/None -- a BEAM-search-tuned kernel's exact "
                "tuning isn't representable here (see this module's own "
                "docstring on scope)"
            )
        return {
            "t": "KernelInfo",
            "fields": {
                "name": _encode_value(v.name),
                "axis_types": _encode_value(v.axis_types),
                "dont_use_locals": _encode_value(v.dont_use_locals),
                "beam": _encode_value(v.beam),
            },
        }
    raise TypeError(
        f"onnxsim.tinygrad_uop_export doesn't know how to encode a {type(v).__name__!r} "
        "UOp.arg value -- this is a whitelisted encoder, not a general one (see this "
        "module's own docstring on scope)"
    )


def _decode_value(d: Any) -> Any:
    from tinygrad import dtypes
    from tinygrad.dtype import ConstFloat
    from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, ParamArg

    t = d["t"]
    if t == "none":
        return None
    if t == "bool":
        return bool(d["v"])
    if t == "int":
        return int(d["v"])
    if t == "float":
        return _decode_float(d["v"])
    if t == "ConstFloat":
        return ConstFloat(_decode_float(d["v"]))
    if t == "str":
        return str(d["v"])
    if t == "tuple":
        return tuple(_decode_value(x) for x in d["v"])
    if t == "DType":
        return getattr(dtypes, d["name"])
    if t == "Ops":
        return Ops[d["name"]]
    if t == "AxisType":
        return AxisType[d["name"]]
    if t == "AddrSpace":
        return AddrSpace[d["name"]]
    if t == "ParamArg":
        f = d["fields"]
        return ParamArg(
            slot=_decode_value(f["slot"]),
            dtype=_decode_value(f["dtype"]),
            vmin_vmax=_decode_value(f["vmin_vmax"]),
            multiple_of=_decode_value(f["multiple_of"]),
            name=_decode_value(f["name"]),
            addrspace=_decode_value(f["addrspace"]),
            axis=_decode_value(f["axis"]),
            device=_decode_value(f["device"]),
            volatile=_decode_value(f["volatile"]),
        )
    if t == "KernelInfo":
        f = d["fields"]
        return KernelInfo(
            name=_decode_value(f["name"]),
            axis_types=_decode_value(f["axis_types"]),
            dont_use_locals=_decode_value(f["dont_use_locals"]),
            beam=_decode_value(f["beam"]),
        )
    raise ValueError(f"unrecognized UOp.arg encoding tag {t!r}")


# --------------------------------------------------------------------------
# UOp -> ONNX nodes
# --------------------------------------------------------------------------


def _uop_nodes(
    root: "UOp", leaf_names: Optional[Dict["UOp", str]] = None
) -> Tuple[List[onnx.NodeProto], str]:
    """Builds the ``NodeProto`` list for the ``UOp`` DAG reachable from
    ``root``, one node per ``UOp`` in toposort order -- except a ``UOp`` in
    ``leaf_names`` (default: none), which gets *no* defining node at all,
    only its caller-supplied name: :func:`uop_to_onnx_function` uses this to
    turn specific leaves into the function's own formal parameters, and a
    formal parameter must not also be a node's output within the same
    function body (the same rule an ordinary ``GraphProto`` follows for its
    own declared inputs) -- emitting both would make that name ambiguously
    both "supplied by the caller" and "produced internally". Every other
    ``UOp`` gets its deterministic synthetic ``u<i>`` name.

    :returns: ``(nodes, root_name)``.
    """
    leaf_names = leaf_names or {}
    order = list(root.toposort())
    index: Dict["UOp", int] = {u: i for i, u in enumerate(order)}

    def name_for(u: "UOp") -> str:
        override = leaf_names.get(u)
        return override if override is not None else f"u{index[u]}"

    nodes = []
    for u in order:
        if u in leaf_names:
            continue
        kwargs: Dict[str, object] = {"dtype": str(u.dtype)}
        if u.arg is not None:
            kwargs["arg_json"] = json.dumps(_encode_value(u.arg))
        nodes.append(
            helper.make_node(
                op_type=u.op.name,
                inputs=[name_for(s) for s in u.src],
                outputs=[name_for(u)],
                name=name_for(u),
                domain=DOMAIN,
                **kwargs,  # type: ignore[arg-type]  # make_node's **kwargs really is Any at runtime
            )
        )
    return nodes, name_for(root)


def uop_to_onnx_model(root: "UOp") -> onnx.ModelProto:
    """Exports the ``UOp`` DAG reachable from ``root`` as a standalone ONNX
    ``ModelProto`` -- one flat graph, synthetic ``u<i>`` names throughout.
    See :func:`uop_to_onnx_function` for the local-function shape, and this
    module's own docstring for what round-trips and what doesn't.

    :param root: typically an ``Ops.SINK``-rooted per-kernel AST (a single
            ``Ops.CALL``'s own ``src[0]`` from a scheduled program -- see
            this module's own docstring on scope).
    :returns: a standalone ``ModelProto`` -- one graph, one custom-domain
            opset import, no standard-domain nodes at all. Never
            executable by any ONNX runtime; see this module's own
            docstring for why that's fine for what this is for.
    """
    _require_tinygrad()

    nodes, root_name = _uop_nodes(root)

    # ONNX's checker requires every graph output to carry a fully-formed
    # `type` (elem_type *and* a shape, even an empty/scalar one) -- there is
    # no "unknown/opaque" placeholder it accepts, so this uses a scalar
    # FLOAT as a formality; it is not a claim about the root UOp's real
    # dtype (already recorded faithfully as that node's own "dtype"
    # attribute above) or shape.
    graph = helper.make_graph(
        nodes,
        "tinygrad_uop_graph",
        inputs=[],
        outputs=[helper.make_tensor_value_info(root_name, onnx.TensorProto.FLOAT, [])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 1),
            helper.make_opsetid(DOMAIN, _OPSET_VERSION),
        ],
        producer_name="onnxsim.tinygrad_uop_export",
    )
    model.ir_version = onnx.IR_VERSION
    return model


def uop_to_onnx_function(
    root: "UOp",
    name: str,
    inputs: Dict["UOp", str],
) -> onnx.FunctionProto:
    """Exports the ``UOp`` DAG reachable from ``root`` as an ONNX **local
    function** (``FunctionProto``) -- the same nodes :func:`uop_to_onnx_model`
    would produce, but with a real I/O boundary instead of synthetic names
    throughout: ``inputs`` names the leaf ``UOp``s (typically each node's
    own ``PARAM`` uops) with the real tensor names a caller cares about
    (e.g. an ONNX node's own input names), and the function's own single
    output is the root ``UOp``.

    A model embedding this (in ``model.functions``) can reference it from
    an ordinary top-level graph via one ``NodeProto`` with
    ``domain=DOMAIN, op_type=name`` and the same real input/output names --
    see this module's own docstring for why that shape is more useful in
    practice (a generic viewer like Netron can expand the function on
    demand instead of only ever showing one flat graph of raw UOps).

    :param root: same as :func:`uop_to_onnx_model`.
    :param name: the function's own name (paired with :data:`DOMAIN` to key
            it) -- typically the real ONNX node's own name this UOp graph
            is the decomposition of.
    :param inputs: leaf ``UOp``\\ s (each must actually appear in ``root``'s
            own DAG) mapped to the real names the function should expose
            them under.
    """
    _require_tinygrad()

    nodes, root_name = _uop_nodes(root, leaf_names=inputs)
    return helper.make_function(
        domain=DOMAIN,
        fname=name,
        inputs=list(inputs.values()),
        outputs=[root_name],
        nodes=nodes,
        opset_imports=[helper.make_opsetid(DOMAIN, _OPSET_VERSION)],
    )


# --------------------------------------------------------------------------
# ONNX nodes -> UOp
# --------------------------------------------------------------------------


def _nodes_to_uop(
    nodes: List[onnx.NodeProto],
    leaf_uops: Optional[Dict[str, "UOp"]] = None,
) -> Dict[str, "UOp"]:
    """Reconstructs a ``UOp`` DAG from a ``NodeProto`` list produced by
    :func:`_uop_nodes` (or :func:`uop_to_onnx_model`/:func:`uop_to_onnx_function`),
    which is always already in a valid topological (dependency) order --
    each node is processed in file order, building ``src`` by looking up
    each input name in what's already been reconstructed.

    :param leaf_uops: real ``UOp``\\ s to bind specific input names to
            instead of reconstructing a node for them -- used when
            importing a :func:`uop_to_onnx_function` export whose inputs
            are meant to be spliced onto a caller's own leaf tensors rather
            than rebuilt from scratch. Defaults to none (every name gets
            reconstructed from its own node).
    :returns: every produced value name (including ``leaf_uops``' own),
            mapped to its ``UOp`` -- the caller picks out whichever name it
            actually wants (the proto's own declared output(s)), rather
            than this function guessing at "the last one processed" (which
            happens to work for a well-formed toposort-ordered list, but
            there is no reason to rely on that when the proto already says
            exactly which name is the output).
    :raises KeyError: a node's ``op_type`` isn't a real ``Ops`` member, or
            an input name isn't found (not yet reconstructed and not in
            ``leaf_uops``).
    """
    from tinygrad import dtypes
    from tinygrad.uop.ops import Ops, UOp

    by_name: Dict[str, "UOp"] = dict(leaf_uops or {})
    for node in nodes:
        if node.domain != DOMAIN:
            raise ValueError(
                f"node {node.name!r} has domain {node.domain!r}, expected {DOMAIN!r}"
            )
        op = Ops[node.op_type]
        attrs = {a.name: a for a in node.attribute}
        dtype = getattr(dtypes, attrs["dtype"].s.decode().removeprefix("dtypes."))
        arg = (
            _decode_value(json.loads(attrs["arg_json"].s.decode()))
            if "arg_json" in attrs
            else None
        )
        src = tuple(by_name[n] for n in node.input)
        u = UOp(op, dtype, src=src, arg=arg)
        for output_name in node.output:
            by_name[output_name] = u
    return by_name


def onnx_model_to_uop(model: onnx.ModelProto) -> "UOp":
    """Reconstructs the ``UOp`` graph a :func:`uop_to_onnx_model` export
    represents -- the inverse of that function. See this module's own
    docstring for exactly what round-trips (verified by re-rendering the
    result through ``WGSLRenderer`` and comparing byte-for-byte against the
    original in ``tests/test_tinygrad_uop_export.py``) and the whitelisted
    ``UOp.arg`` shapes this understands.
    """
    _require_tinygrad()
    if len(model.graph.output) != 1:
        raise ValueError(
            f"expected exactly one graph output, got {len(model.graph.output)}"
        )
    by_name = _nodes_to_uop(list(model.graph.node))
    return by_name[model.graph.output[0].name]


def onnx_function_to_uop(
    function: onnx.FunctionProto,
    inputs: Dict[str, "UOp"],
) -> "UOp":
    """Reconstructs the ``UOp`` graph a :func:`uop_to_onnx_function` export
    represents -- the inverse of that function.

    ``inputs`` is **required**, not optional: a function's own formal
    parameters (``function.input``) have no node defining them in
    ``function.node`` at all (see :func:`_uop_nodes`'s own docstring on
    why) -- exactly like a Python function's parameters, they only become
    real values once bound at a call site, so there is nothing to
    reconstruct them *from* without a caller supplying one real ``UOp`` per
    formal parameter name.

    :param inputs: real ``UOp``\\ s to splice in for the function's own
            named inputs (keyed by those same names, i.e.
            ``set(inputs) == set(function.input)``).
    :raises ValueError: ``inputs`` doesn't supply exactly the function's
            own declared formal parameters.
    """
    _require_tinygrad()
    if set(inputs) != set(function.input):
        raise ValueError(
            f"inputs {sorted(inputs)} doesn't match function {function.name!r}'s "
            f"own formal parameters {sorted(function.input)}"
        )
    if len(function.output) != 1:
        raise ValueError(
            f"expected exactly one function output, got {len(function.output)}"
        )
    by_name = _nodes_to_uop(list(function.node), leaf_uops=inputs)
    return by_name[function.output[0]]
