"""``simplify(range_opt=True)``: the range-driven rewrites of ``onnxsim.range_opt`` as an opt-in
step, with certification over the same box and fallback to the plain result on refutation."""

import importlib.util

import onnx
from onnx import parser

import onnxsim
from onnxsim import onnx_simplifier as _simp
from onnxsim import ranges

_HEADER = '<ir_version: 8, opset_import: ["": 17]>'


def _model(body: str) -> onnx.ModelProto:
    return parser.parse_model(f"{_HEADER}\n{body}")


def _relu_model(lo, hi):
    model = _model(
        """
        agraph (float[4] x) => (float[4] y)
        {
            y = Relu(x)
        }
        """
    )
    if lo is not None:
        ranges.set_range(model, "x", lo, hi)
    return model


def _ops(model):
    return [n.op_type for n in model.graph.node]


def _meta(model, key):
    return {p.key: p.value for p in model.metadata_props}.get(key)


def test_default_leaves_relu_and_records_nothing():
    sim, _ = onnxsim.simplify(_relu_model(0.5, 2.0), certify=False)
    assert _ops(sim) == ["Relu"]
    assert _meta(sim, "onnxsim.range_opt") is None


def test_relu_removed_when_input_range_is_nonnegative():
    sim, _ = onnxsim.simplify(_relu_model(0.5, 2.0), certify=False, range_opt=True)
    # The Relu feeds the graph output, so it becomes an Identity copy rather than vanishing.
    assert _ops(sim) == ["Identity"]
    assert _meta(sim, "onnxsim.range_opt") == "applied 1 rewrite(s)"
    assert _meta(sim, "onnxsim.precondition.range.x") is not None


def test_relu_kept_when_input_range_spans_zero():
    sim, _ = onnxsim.simplify(_relu_model(-1.0, 1.0), certify=False, range_opt=True)
    assert _ops(sim) == ["Relu"]
    assert _meta(sim, "onnxsim.range_opt") == "applied 0 rewrite(s)"


def test_no_ranges_records_and_changes_nothing():
    sim, _ = onnxsim.simplify(_relu_model(None, None), certify=False, range_opt=True)
    assert _ops(sim) == ["Relu"]
    assert _meta(sim, "onnxsim.range_opt") == "no ranges"


def test_refuted_rewrite_falls_back_to_plain_simplification(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object()
        if name == "z3"
        else real_find_spec(name, *a, **k),
    )
    monkeypatch.setattr(
        _simp, "_certify_snapshot", lambda model, path, explicit: (object(), "")
    )

    def fake_result(snap, simplified, kwargs, explicit):
        # Refute any result that dropped the Relu; prove the plain one.
        return (
            ("refuted", "synthetic")
            if "Relu" not in _ops(simplified)
            else ("proved", "")
        )

    monkeypatch.setattr(_simp, "_certify_result", fake_result)
    sim, _ = onnxsim.simplify(_relu_model(0.5, 2.0), certify=True, range_opt=True)
    assert _ops(sim) == ["Relu"]
    assert _meta(sim, "onnxsim.range_opt") == (
        "reverted: certify refuted the range-driven rewrites"
    )
    assert _meta(sim, "onnxsim.certify") == "proved"
