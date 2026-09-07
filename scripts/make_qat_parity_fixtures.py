"""Emit the reference step-graph fixtures the Python<->C++ parity test compares.

``onnxsim/qat_graph_builder.{h,cpp}`` is a C++ port of the emitter half of
``onnxsim/qat_graph.py``. Two implementations of one emitter that quietly
disagree would give the browser a training loop that behaves differently from
the Python for the same model, and nothing would say so -- the graphs would
both be valid, both run, and produce different weights. That is the whole
hazard of having ported it, and a comment cannot hold the line.

So this writes down what the *Python* emitter produces for a set of canonical
sequences, and both sides are then checked against that one artifact:

- ``tests/test_qat_parity.py`` asserts the fixture still matches what
  ``qat_graph.py`` emits today, which is what catches the Python side drifting
  away from a stale fixture;
- ``onnxsim/qat_graph_parity_test.cpp`` asserts the C++ emitter reproduces the
  same fixture.

Fixture == Python and fixture == C++ together give Python == C++, which is the
property actually wanted and which neither test could establish alone. It is
the same shape as ``scripts/convertmodel/test/step_graphs.json``: a generated
artifact committed to the tree, with a test on each side that fails loudly
when it goes stale rather than silently accepting it.

**What is compared, and what deliberately is not.** Node op types, input and
output *names*, attributes, and initializer names/dtypes/shapes/values, all in
emission order. Tensor *names* are included on purpose rather than normalized
away: ``GraphBuilder``'s counter is what makes them, so identical names mean
the two emitters ran the same operations in the same order, which is a much
stronger statement than "the graphs are isomorphic" and is the property that
actually breaks first when someone reorders a rule. What is *not* compared is
the byte encoding of a tensor -- ``onnx.numpy_helper.from_array`` and a
hand-built C++ ``TensorProto`` may legitimately choose ``float_data`` versus
``raw_data`` for the same values, and a runtime cannot tell the difference. So
values are compared as numbers.

Usage:
    python3 scripts/make_qat_parity_fixtures.py

Writes ``onnxsim/qat_parity_fixtures.json``. Re-run it whenever the emitter
changes on purpose, and commit the result alongside.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import numpy as np
import onnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from onnxsim import qat_graph  # noqa: E402

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "onnxsim", "qat_parity_fixtures.json"
)


def _attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    """A node's attributes as plain JSON, by name.

    Only the attribute types the emitter actually produces are handled -- ints
    (``to``, ``axis``, ``keepdims``) and int lists (``perm``). Anything else
    means the emitter grew a construct this fixture does not describe, so it
    raises rather than dropping it silently and letting the parity test pass on
    an incomplete comparison.
    """
    out: Dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            out[attr.name] = int(attr.i)
        elif attr.type == onnx.AttributeProto.INTS:
            out[attr.name] = [int(v) for v in attr.ints]
        else:
            raise NotImplementedError(
                f"attribute {attr.name!r} of node {node.op_type} has type "
                f"{onnx.AttributeProto.AttributeType.Name(attr.type)}, which this "
                "fixture format does not describe -- teach it that type rather "
                "than letting the parity comparison quietly skip the attribute"
            )
    return out


def _describe(b: qat_graph.GraphBuilder) -> Dict[str, Any]:
    """Everything a builder accumulated, in emission order."""
    return {
        "initializers": [
            {
                "name": t.name,
                "dims": [int(d) for d in t.dims],
                "dtype": int(t.data_type),
                # Values as numbers, not bytes -- see this module's docstring
                # for why the encoding is deliberately not part of the
                # comparison.
                "values": [
                    float(v) if t.data_type != onnx.TensorProto.INT64 else int(v)
                    for v in onnx.numpy_helper.to_array(t).reshape(-1).tolist()
                ],
            }
            for t in b.initializer
        ],
        "nodes": [
            {
                "op_type": n.op_type,
                "inputs": list(n.input),
                "outputs": list(n.output),
                "attributes": _attributes(n),
            }
            for n in b.nodes
        ],
    }


def _case_arithmetic() -> Dict[str, Any]:
    """Every plain wrapper, in one sequence.

    Kept as one case rather than one per operator because the counter is
    shared: running them together checks that each wrapper consumes exactly
    the number of names it should, which per-operator cases would not.
    """
    b = qat_graph.GraphBuilder()
    s = b.add("x", "y")
    s = b.sub(s, "y")
    s = b.mul(s, "y")
    s = b.div(s, "y")
    s = b.matmul(s, "w")
    s = b.transpose(s)
    s = b.transpose(s, [1, 0])
    s = b.sqrt(s)
    s = b.sigmoid(s)
    result = b.mean_square(s)
    out = _describe(b)
    out["result"] = result
    return out


def _case_masks_and_clip() -> Dict[str, Any]:
    """``clip``/``greater_mask``/``less_mask``.

    These are the ones where Python's argument evaluation order fixes the
    counter sequence -- ``clip`` emits both of its bound constants before the
    ``Clip`` node itself -- so a C++ port that nests the calls differently
    numbers the tensors differently and is caught here.
    """
    b = qat_graph.GraphBuilder()
    clipped = b.clip("x", -7.0, 7.0)
    gt = b.greater_mask(clipped, -7.0)
    lt = b.less_mask(clipped, 7.0)
    result = b.mul(gt, lt)
    out = _describe(b)
    out["result"] = result
    return out


def _case_round_to_nearest() -> Dict[str, Any]:
    """The composed rounding.

    Six names in a fixed order, and the single most likely thing to be
    reordered by someone porting it from the expression form the Python
    writes it in. It must also contain no ``Round`` node at all -- WebNN has
    no rounding operator, which is the reason the composition exists.
    """
    b = qat_graph.GraphBuilder()
    result = b.round_to_nearest("x")
    out = _describe(b)
    out["result"] = result
    return out


def _case_gather_rows() -> Dict[str, Any]:
    """Both forms of the minibatching primitive: fresh name, and written into
    a caller-chosen output name."""
    b = qat_graph.GraphBuilder()
    fresh = b.gather_rows("table", "idx")
    b.gather_rows("table", "idx", out="block_input")
    out = _describe(b)
    out["result"] = fresh
    return out


def _case_consts() -> Dict[str, Any]:
    """Scalar and array constants, and the prefix a builder was constructed
    with -- the prefix is part of every name it makes, so a port that dropped
    it would produce a graph that still ran and still disagreed."""
    b = qat_graph.GraphBuilder("pre_")
    b.const(0.5)
    b.const(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    result = b.add("x", b.const(-1.25))
    out = _describe(b)
    out["result"] = result
    return out


def _case_adam_update() -> Dict[str, Any]:
    """One Adam step: nineteen names, and the one place a nested call
    (``mul(grad, grad)`` inside the second moment) makes the evaluation order
    non-obvious.

    Also pins the four constants' *values*. ``1 - beta`` is computed in double
    precision and then narrowed; doing the subtraction in float32 instead
    lands on a different number (0.100000024 rather than 0.1), which would
    desynchronise every optimizer step between the two emitters while leaving
    both graphs perfectly valid.
    """
    b = qat_graph.GraphBuilder()
    param_next, m_next, v_next = qat_graph.adam_update(
        b, "p", "g", "m", "v", "lr", "mc", "vc"
    )
    out = _describe(b)
    out["result"] = [param_next, m_next, v_next]
    return out


def _case_step_graph() -> Dict[str, Any]:
    """A complete ``make_step_graph``: input and output declaration order,
    opset, IR version, and the state map that closes the loop.

    The declaration *order* is the part worth pinning. A runner binds by name,
    so a reordered input list still runs -- and a C++ port that emitted
    scalars before state, say, would produce a model that differs from the
    Python's in a way no numerical test would ever notice.
    """
    b = qat_graph.GraphBuilder()
    diff = b.sub("student", "teacher")
    loss = b.mean_square(diff)
    param_next, m_next, v_next = qat_graph.adam_update(
        b, "w", diff, "m", "v", "lr", "mc", "vc"
    )
    step = qat_graph.make_step_graph(
        b,
        constants={"teacher": [4, 3]},
        state={
            "w": ([4, 3], param_next),
            "m": ([4, 3], m_next),
            "v": ([4, 3], v_next),
        },
        scalars=["lr", "mc", "vc"],
        loss=loss,
        per_step={"rows": ([2], int(onnx.TensorProto.INT64))},
    )
    graph = step.model.graph
    out = _describe(b)
    out["model"] = {
        "opset": [
            {"domain": o.domain, "version": int(o.version)}
            for o in step.model.opset_import
        ],
        "ir_version": int(step.model.ir_version),
        "graph_name": graph.name,
        "inputs": [
            {
                "name": i.name,
                "elem_type": int(i.type.tensor_type.elem_type),
                "dims": [int(d.dim_value) for d in i.type.tensor_type.shape.dim],
            }
            for i in graph.input
        ],
        "outputs": [
            {
                "name": o.name,
                "elem_type": int(o.type.tensor_type.elem_type),
                "dims": [int(d.dim_value) for d in o.type.tensor_type.shape.dim],
            }
            for o in graph.output
        ],
        "state": dict(step.state),
        "loss_name": step.loss_name,
    }
    return out


CASES = {
    "arithmetic": _case_arithmetic,
    "masks_and_clip": _case_masks_and_clip,
    "round_to_nearest": _case_round_to_nearest,
    "gather_rows": _case_gather_rows,
    "consts": _case_consts,
    "adam_update": _case_adam_update,
    "step_graph": _case_step_graph,
}


def build() -> Dict[str, Any]:
    """Every case, plus the operator allowlist both emitters must agree on.

    ``ep_friendly_ops`` is in here for the same reason the cases are: it is a
    claim the C++ restates, and a member added on one side only would let one
    emitter produce a graph the other's tests reject.
    """
    cases = {name: fn() for name, fn in CASES.items()}

    # A case that emitted an operator outside the allowlist would make the
    # fixture itself the thing asserting something false, so refuse to write
    # one. This is the generator holding itself to what the tests downstream
    # will claim.
    emitted = {
        node["op_type"] for case in cases.values() for node in case.get("nodes", [])
    }
    outside = sorted(emitted - set(qat_graph.EP_FRIENDLY_OPS))
    if outside:
        raise SystemExit(
            f"cases emit {outside}, which are not in EP_FRIENDLY_OPS -- either the "
            "operator belongs in the allowlist (check its coverage on the WebGPU "
            "and WebNN backends first) or the case should not emit it"
        )

    return {
        "_comment": (
            "Generated by scripts/make_qat_parity_fixtures.py. Do not edit by "
            "hand. Both onnxsim/qat_graph.py and onnxsim/qat_graph_builder.cpp "
            "are asserted against this file; see that script for why the "
            "comparison is shaped this way."
        ),
        "ep_friendly_ops": sorted(qat_graph.EP_FRIENDLY_OPS),
        "cases": cases,
    }


def main() -> None:
    fixtures = build()
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixtures, f, indent=2, sort_keys=True)
        f.write("\n")
    total = sum(len(c.get("nodes", [])) for c in fixtures["cases"].values())
    print(
        f"wrote {os.path.relpath(FIXTURE_PATH)}: "
        f"{len(fixtures['cases'])} cases, {total} nodes, "
        f"{len(fixtures['ep_friendly_ops'])} allowlisted ops"
    )


if __name__ == "__main__":
    main()
