import numpy as np
from onnx import numpy_helper, parser

from onnxsim.exact_fold_check import check_constant_fold


def _model(body, initializer=(), opset=17):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["": {opset}]> {body}')
    model.graph.initializer.extend(initializer)
    return model


def _init(name, array):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)


def _product_pair(rng, m=2, k=3, n=2):
    return rng.standard_normal((m, k)).astype(np.float32), rng.standard_normal(
        (k, n)
    ).astype(np.float32)


_ORIGINAL_MATMUL = """
agraph (float[2,2] x) => (float[2,2] y) {
    C = MatMul(A, B)
    y = Add(x, C)
}
"""

_FOLDED_MATMUL = """
agraph (float[2,2] x) => (float[2,2] y) {
    y = Add(x, C)
}
"""


def _pair(a, b, folded_c):
    original = _model(_ORIGINAL_MATMUL, [_init("A", a), _init("B", b)])
    simplified = _model(_FOLDED_MATMUL, [_init("C", folded_c)])
    return original, simplified


def test_correct_fold_matches():
    rng = np.random.default_rng(0)
    a, b = _product_pair(rng)
    original, simplified = _pair(a, b, a @ b)
    report = check_constant_fold(original, simplified)
    assert report.complete
    assert report.matches
    assert [(f.tensor, f.method) for f in report.findings] == [("C", "evaluation")]


def test_corrupted_folded_constant_is_flagged():
    rng = np.random.default_rng(1)
    a, b = _product_pair(rng)
    original, simplified = _pair(a, b, a @ b + 0.01)
    report = check_constant_fold(original, simplified)
    assert not report.matches
    assert [f.tensor for f in report.findings if not f.ok] == ["C"]


def test_corrupted_initializer_with_original_name_is_flagged():
    rng = np.random.default_rng(2)
    a, b = _product_pair(rng)
    original = _model(_ORIGINAL_MATMUL, [_init("A", a), _init("B", b)])
    simplified = _model(_FOLDED_MATMUL, [_init("A", a + 1.0), _init("C", a @ b)])
    report = check_constant_fold(original, simplified)
    assert not report.matches
    bad = [f for f in report.findings if not f.ok]
    assert [(f.tensor, f.method) for f in bad] == [("A", "initializer")]


def test_freivalds_accepts_correct_large_matmul():
    rng = np.random.default_rng(3)
    a, b = _product_pair(rng, m=6, k=8, n=5)
    original, simplified = _pair(a, b, a @ b)
    report = check_constant_fold(original, simplified, freivalds_min_flops=0)
    assert report.matches
    assert [f.method for f in report.findings] == ["freivalds"]


def test_freivalds_flags_one_corrupted_element():
    rng = np.random.default_rng(4)
    a, b = _product_pair(rng, m=6, k=8, n=5)
    folded = a @ b
    folded[2, 3] += 1.0
    original, simplified = _pair(a, b, folded)
    report = check_constant_fold(original, simplified, freivalds_min_flops=0)
    assert not report.matches
    bad = [f for f in report.findings if not f.ok]
    assert [(f.tensor, f.method) for f in bad] == [("C", "freivalds")]


def test_gemm_with_bias_is_checked_by_freivalds():
    rng = np.random.default_rng(5)
    a, b = _product_pair(rng, m=4, k=6, n=3)
    bias = rng.standard_normal(3).astype(np.float32)
    original = _model(
        """
        agraph (float[4,3] x) => (float[4,3] y) {
            C = Gemm(A, B, D)
            y = Add(x, C)
        }
        """,
        [_init("A", a), _init("B", b), _init("D", bias)],
    )
    simplified = _model(
        _FOLDED_MATMUL.replace("float[2,2]", "float[4,3]"), [_init("C", a @ b + bias)]
    )
    report = check_constant_fold(original, simplified, freivalds_min_flops=0)
    assert report.complete and report.matches
    assert [f.method for f in report.findings] == ["freivalds"]


def test_non_constant_subgraph_is_unanalysed():
    rng = np.random.default_rng(6)
    _, b = _product_pair(rng)
    original = _model(
        """
        agraph (float[2,3] x) => (float[2,2] y) {
            C = MatMul(x, B)
            y = Add(C, C)
        }
        """,
        [_init("B", b)],
    )
    simplified = _model(
        """
        agraph (float[2,3] x) => (float[2,2] y) {
            y = Add(x, C)
        }
        """,
        [_init("C", np.zeros((2, 2)))],
    )
    report = check_constant_fold(original, simplified)
    assert report.unanalysed == ["C"]
    assert report.findings == []
    assert not report.complete


def test_pinned_input_counts_as_constant():
    rng = np.random.default_rng(7)
    a, b = _product_pair(rng)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    original = _model(
        """
        agraph (float[2,3] x) => (float[2,2] y) {
            C = MatMul(x, B)
            y = Add(C, C)
        }
        """,
        [_init("B", b)],
    )
    simplified = _model(
        """
        agraph (float[2,3] x) => (float[2,2] y) {
            y = Add(x, C)
        }
        """,
        [_init("C", x @ b)],
    )
    report = check_constant_fold(original, simplified, input_ranges={"x": (x, x)})
    assert report.complete and report.matches
    assert [f.tensor for f in report.findings] == ["C"]
