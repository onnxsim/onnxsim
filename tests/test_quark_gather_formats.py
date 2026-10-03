"""Constant-table format sharing across Gather and Transpose."""

import numpy as np
import pytest
import test_quark_block_preproc_parity as P
from onnx import numpy_helper, parser

pytestmark = P.pytestmark
_run_in_tmp_dir = P._run_in_tmp_dir


@pytest.mark.parametrize(
    "preset", ["BF16_BFP16", "BF16_MXINT8", "MX9_INT8", "BF16", "BFP16"]
)
@pytest.mark.parametrize("chain", [False, True])
def test_gather_inherits_table_format(preset, chain, tmp_path):
    body = (
        "g = Gather <axis: int = 0> (table, x)\ny = Transpose <perm: ints = [1, 0]> (g)"
        if chain
        else "y = Gather <axis: int = 0> (table, x)"
    )
    shape = "16, 2" if chain else "2, 16"
    model = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> gather (int64[2] x) => (float[{shape}] y) {{ {body} }}'
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.random.default_rng(5).normal(size=(8, 16)).astype(np.float32), "table"
        )
    )
    for i, node in enumerate(model.graph.node):
        node.name = f"n{i}"
    data = [{"x": np.array([i, 7 - i], np.int64)} for i in range(4)]
    extra = {"SkipPreprocess": True}
    q = P._quark(model, data, tmp_path, preset, extra)
    m = P._mine(model, data, preset, extra)
    P._assert_same(q, m, preset)
    P._assert_same_outputs(q, m, data, preset)
