"""Dedicated Q/DQ receiver numbering follows the original marking order."""

import pytest
import test_quark_int_presets_parity as P
from onnx import parser

pytestmark = P.pytestmark
_run_in_tmp_dir = P._run_in_tmp_dir


@pytest.mark.parametrize("repeated", ["first", "second", "both"])
@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_receiver_slots(repeated, reverse, tmp_path):
    a = "a = Add(x, x)" if repeated in ("first", "both") else "a = Add(x, z)"
    b = "b = Mul(x, x)" if repeated in ("second", "both") else "b = Mul(x, z)"
    lines = [a, b]
    if reverse:
        lines.reverse()
    model = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> graph (float[1,8] x) => (float[1,8] y) <float[1,8] z = {1,2,3,4,5,6,7,8}> { '
        + "\n".join(lines)
        + "\ny = Add(a, b) }"
    )
    for n in model.graph.node:
        n.name = n.output[0]
    data = P.P._data((1, 8), seed=13)
    P._check(model, data, tmp_path, "VINT8")


@pytest.mark.parametrize("seed", range(20))
def test_random_repeated_receivers(seed, tmp_path):
    model, shape = P._random_graph_ext(seed, big=bool(seed % 2))
    # Deliberately exercise a repeated slot without changing tensor dimensions.
    for n in model.graph.node:
        if n.op_type in ("Add", "Mul"):
            n.input[1] = n.input[0]
            break
    P._check(model, P.P._data(shape, seed=seed + 7), tmp_path, "VINT8")
