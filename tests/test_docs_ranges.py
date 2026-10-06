"""docs/ranges.md must stay true.

Every Python block in the page marked ``<!-- doctest -->`` is executed top to bottom in
one shared namespace (each block carries its own asserts), the output the page quotes is
checked to really be printed, and the CLI example is replayed. The last two tests pin the
interval-analysis behaviour the page's worked example depends on.
"""

import os
import re

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim import interval, ranges

_DOC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "ranges.md")
_BLOCK = re.compile(r"<!-- doctest -->\n```python\n(.*?)```", re.DOTALL)


def _doc_blocks():
    with open(_DOC) as f:
        return _BLOCK.findall(f.read())


def _model(body, initializer=None, opset=15, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def test_doc_has_runnable_blocks():
    assert len(_doc_blocks()) >= 6


def test_doc_snippets_run_and_quoted_output_is_real(capsys):
    pytest.importorskip("z3", reason="the certify examples need z3-solver")
    namespace: dict = {}
    for i, block in enumerate(_doc_blocks()):
        try:
            exec(compile(block, f"docs/ranges.md block {i}", "exec"), namespace)
        except Exception as e:  # say which block of the page broke
            raise AssertionError(
                f"docs/ranges.md doctest block {i} failed: {e!r}"
            ) from e
    out = capsys.readouterr().out
    # lines the page shows as output
    assert "Certify: proved (proved-congruence)" in out
    assert "Certify: unproven-no-ranges (differs only for very large inputs" in out
    assert "leaves the annotated range" in out
    assert "MatMul_0" in out and "1625600" in out
    assert '{"min": [[[[0.0]], [[-1.0]], [[0.5]]]], "max": 1.0}' in out
    assert (
        "['p: observed [0.5, 1.5] leaves the annotated range (1 of 2 elements)']" in out
    )


def test_doc_cli_example(tmp_path, capsys):
    path = str(tmp_path / "model.onnx")
    onnx.save(
        _model(
            "m (float[1,3,4,4] image) => (float[1,3,4,4] prob) { prob = Sigmoid(image) }"
        ),
        path,
    )
    assert ranges.main([path, "--set", "image=0,1", "--set", "prob=0,1"]) == 0
    capsys.readouterr()
    ranges.main([path, "--show"])
    assert capsys.readouterr().out == "image: [0, 1]\nprob: [0, 1]\n"
    ranges.main([path, "--set", "prob=0,none", "--show"])
    assert capsys.readouterr().out == "image: [0, 1]\nprob: [0, inf]\n"
    ranges.main([path, "--clear", "--show"])
    assert capsys.readouterr().out == ""


def test_matmul_with_float32_weight_is_bounded_not_unsupported():
    # Interval arrays are float64 and initializers float32; the reference evaluator
    # rejects that mix for MatMul, which used to turn the op into "unsupported".
    layer = _model(
        "m (float[1,100] x) => (float[1,2] y) { y = MatMul(x, W) }",
        {"W": np.full((100, 2), 0.5, np.float32)},
    )
    res = interval.propagate(layer, {"x": (-1.0, 1.0)})
    assert res.unsupported == []
    lo, hi = res.hull("y")
    assert lo == pytest.approx(-50.0) and hi == pytest.approx(50.0)


def test_relative_error_is_unknown_not_tiny_when_the_output_is_unbounded():
    def bound(width):
        return interval.LayerQuantBound(
            node="n", op_type="MatMul", reduction_depth=4, act_range=(0.0, 1.0),
            act_scale=1.0 / 255, act_zero_point=0, acc_bound=1, acc_bound_full_range=1,
            int32_safe=True, fp32_cast_exact=True, max_abs_error=0.5, output_width=width,
        )  # fmt: skip

    assert np.isnan(bound(float("inf")).relative_error)
    assert bound(2.0).relative_error == pytest.approx(0.25)
    assert bound(0.0).relative_error == float("inf")
