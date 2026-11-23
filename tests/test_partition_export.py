# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Export of device partitions as model-local ONNX functions."""

import numpy as np
import onnx
from onnx import parser
from onnx.reference import ReferenceEvaluator

from onnxsim.partition_export import (
    EXPORT_DOMAIN,
    _order,
    export_partitioned_model,
)


def _model(body, inputs, outputs, opset=20):
    return parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : {opset}]>
        g ({inputs}) => ({outputs}) {{
          {body}
        }}
        """
    )


CONV_IN = "float[1, 8, 16, 16] x, float[8, 8, 3, 3] w, float[1, 8, 16, 16] r"
CONV_OUT = "float[1, 8, 16, 16] y"


def _feeds(model):
    rng = np.random.default_rng(0)
    feeds = {}
    for v in model.graph.input:
        if v.name in {i.name for i in model.graph.initializer}:
            continue
        shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
        feeds[v.name] = rng.standard_normal(shape).astype(np.float32) + 2.0
    return feeds


def _run(model, feeds):
    return ReferenceEvaluator(model).run(None, feeds)


def test_each_device_region_becomes_one_function_call():
    # Conv+Relu run on WebNN, Mod+Cast fall back to CPU: two regions, two functions.
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  a = Relu(c)\n  m = Mod<fmod=1>(a, two)\n  y = Cast<to=1>(m)",
        CONV_IN + ", float[1] two = {2.0}",
        CONV_OUT,
    )
    result = export_partitioned_model(model, "webnn")

    assert [(f.device, f.node_count) for f in result.functions] == [
        ("webnn", 2),
        ("cpu", 2),
    ]
    assert result.inline_nodes == 0
    assert result.skipped == ()
    out = result.model
    assert [n.op_type for n in out.graph.node] == [f.name for f in result.functions]
    assert all(n.domain == EXPORT_DOMAIN for n in out.graph.node)
    assert len(out.functions) == 2
    assert {f.name for f in out.functions} == {f.name for f in result.functions}
    onnx.checker.check_model(out, full_check=True)


def test_exported_model_matches_the_original_numerically():
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  a = Relu(c)\n  m = Mod<fmod=1>(a, two)\n  y = Cast<to=1>(m)",
        CONV_IN + ", float[1] two = {2.0}",
        CONV_OUT,
    )
    feeds = _feeds(model)
    expected = _run(model, feeds)
    result = export_partitioned_model(model, "webnn")
    actual = _run(result.model, feeds)
    for e, a in zip(expected, actual):
        np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6)


def test_input_model_is_not_modified():
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  a = Relu(c)\n  m = Mod<fmod=1>(a, two)\n  y = Cast<to=1>(m)",
        CONV_IN + ", float[1] two = {2.0}",
        CONV_OUT,
    )
    before = model.SerializeToString()
    export_partitioned_model(model, "webnn")
    assert model.SerializeToString() == before


def test_small_regions_stay_inline():
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  a = Relu(c)\n  m = Mod<fmod=1>(a, two)\n  y = Cast<to=1>(m)",
        CONV_IN + ", float[1] two = {2.0}",
        CONV_OUT,
    )
    result = export_partitioned_model(model, "webnn", min_nodes=3)
    assert result.functions == ()
    assert result.inline_nodes == 4
    assert [r[2] for r in result.skipped] == ["fewer than 3 nodes"] * 2
    assert len(result.model.functions) == 0


def test_non_convex_region_is_not_outlined():
    # c and y are on WebNN, but the path c -> Mod (CPU) -> y leaves the region
    # and re-enters it, so it must stay inline.
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  m = Mod<fmod=1>(c, two)\n  y = Mul(c, m)",
        CONV_IN + ", float[1] two = {2.0}",
        CONV_OUT,
    )
    feeds = _feeds(model)
    expected = _run(model, feeds)
    result = export_partitioned_model(model, "webnn", min_nodes=1)
    assert ("webnn", (0, 2), "non-convex region") in result.skipped
    assert [f.device for f in result.functions] == ["cpu"]
    actual = _run(result.model, feeds)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)


def test_cycle_detection_orders_or_rejects():
    # Two regions that each depend on the other cannot be scheduled.
    cyclic = [
        (0, ["b"], ["a"], "A"),
        (1, ["a"], ["b"], "B"),
    ]
    assert _order(cyclic) is None

    # A chain is ordered by dependency, not by key.
    chain = [
        (0, ["t"], ["u"], "second"),
        (1, ["x"], ["t"], "first"),
    ]
    assert _order(chain) == ["first", "second"]
