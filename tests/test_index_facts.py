import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, parser

from onnxsim import index_facts as IF


def _model(body, initializer=None, opset=15):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    onnx.checker.check_model(model)
    return model


def _verdicts(report):
    return [f.verdict for f in report.facts]


def test_gather_with_indices_inside_the_axis_is_in_range():
    model = _model(
        """
        agraph (float[4,3] x, int64[2] i) => (float[2,3] y) {
            y = Gather(x, i)
        }
        """
    )
    report = IF.index_facts(model, {"i": (0, 3)})
    assert _verdicts(report) == [IF.IN_RANGE]
    assert report.proved


def test_gather_with_indices_past_the_axis_is_out_of_range():
    model = _model(
        """
        agraph (float[4,3] x, int64[2] i) => (float[2,3] y) {
            y = Gather(x, i)
        }
        """
    )
    report = IF.index_facts(model, {"i": (4, 6)})
    assert _verdicts(report) == [IF.OUT_OF_RANGE]
    assert not report.proved


def test_gather_index_that_straddles_the_axis_is_unknown():
    model = _model(
        """
        agraph (float[4,3] x, int64[2] i) => (float[2,3] y) {
            y = Gather(x, i)
        }
        """
    )
    report = IF.index_facts(model, {"i": (2, 5)})
    assert _verdicts(report) == [IF.UNKNOWN]


def test_negative_indices_are_checked_against_the_negative_window():
    model = _model(
        """
        agraph (float[4,3] x, int64[2] i) => (float[2,3] y) {
            y = Gather(x, i)
        }
        """
    )
    assert _verdicts(IF.index_facts(model, {"i": (-4, -1)})) == [IF.IN_RANGE]
    assert _verdicts(IF.index_facts(model, {"i": (-5, -1)})) == [IF.UNKNOWN]


def test_gather_elements_is_checked_on_its_axis():
    model = _model(
        """
        agraph (float[4,3] x, int64[4,3] i) => (float[4,3] y) {
            y = GatherElements(x, i)
        }
        """
    )
    assert _verdicts(IF.index_facts(model, {"i": (0, 3)})) == [IF.IN_RANGE]
    assert _verdicts(IF.index_facts(model, {"i": (0, 4)})) == [IF.UNKNOWN]


def test_slice_with_bounded_start_is_in_range():
    model = _model(
        """
        agraph (float[8] x, int64[1] s) => (float y) {
            y = Slice(x, s, ends, axes)
        }
        """,
        initializer={
            "ends": np.array([100], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
        },
    )
    report = IF.index_facts(model, {"s": (0, 3)})
    assert _verdicts(report) == [IF.IN_RANGE]
    assert report.facts[0].axis == 0


def test_slice_start_past_the_axis_is_out_of_range_and_says_it_is_clamped():
    model = _model(
        """
        agraph (float[8] x, int64[1] s) => (float y) {
            y = Slice(x, s, ends, axes)
        }
        """,
        initializer={
            "ends": np.array([100], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
        },
    )
    report = IF.index_facts(model, {"s": (9, 12)})
    assert _verdicts(report) == [IF.OUT_OF_RANGE]
    assert "outside" in report.facts[0].detail


def test_dynamic_axis_returns_unknown():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 15]>
        agraph (float[N,3] x, int64[2] i) => (float[2,3] y) {
            y = Gather(x, i)
        }
        """
    )
    report = IF.index_facts(model, {"i": (0, 1)})
    assert _verdicts(report) == [IF.UNKNOWN]
    assert not report.proved


def test_unbounded_index_returns_unknown():
    model = _model(
        """
        agraph (float[4,3] x, int64[2] i) => (float[2,3] y) {
            y = Gather(x, i)
        }
        """
    )
    report = IF.index_facts(model, {})
    assert _verdicts(report) == [IF.UNKNOWN]


def test_in_range_verdict_holds_under_onnxruntime():
    model = _model(
        """
        agraph (float[4,3] x, int64[2] i) => (float[2,3] y) {
            y = Gather(x, i)
        }
        """
    )
    assert _verdicts(IF.index_facts(model, {"i": (-4, 3)})) == [IF.IN_RANGE]
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    for _ in range(200):
        idx = rng.integers(-4, 4, size=2).astype(np.int64)
        sess.run(None, {"x": x, "i": idx})
