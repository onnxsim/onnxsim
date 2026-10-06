"""simplify()'s default-on certification (onnxsim.certify) and range-annotation hooks.

Covers the user-visible contract: quiet by default, the verdict recorded in the
returned model's metadata, ``certify=False`` opting out, size/availability guards,
and the annotated ranges driving ``check_n``'s random inputs and output warnings.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import model_checking
from onnxsim import onnx_simplifier as simp
from onnxsim import ranges as R

pytest.importorskip("z3", reason="needs the 'verify' extra (z3-solver)")


def _model(body, initializer=None, opset=15, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def _conv_bn_relu(seed=0, k=4):
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    return _model(
        f"""
        m (float[1,3,6,6] x) => (float[1,{k},4,4] y) {{
          c = Conv(x, W, B)
          b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
          y = Relu(b)
        }}""",
        dict(W=f(k, 3, 3, 3), B=f(k), g=rng.uniform(0.5, 1.5, k).astype(np.float32),
             be=f(k), mu=f(k), var=rng.uniform(0.5, 2, k).astype(np.float32)),
    )  # fmt: skip


def _meta(model, key):
    return next((p.value for p in model.metadata_props if p.key == key), None)


def test_default_is_silent_and_records_unproven_without_ranges(capsys):
    sim, _ = onnxsim.simplify(_conv_bn_relu())
    out = capsys.readouterr().out
    assert "WARNING" not in out and "Certify" not in out
    # No range annotated: a folded BN differs by float rounding for huge inputs, which is
    # not a bug -- so it is "unproven", not "refuted".
    assert _meta(sim, "onnxsim.certify") == "unproven-no-ranges"
    assert "annotate input ranges" in _meta(sim, "onnxsim.certify.detail")


def test_annotated_range_makes_it_proved():
    m = _conv_bn_relu()
    R.set_range(m, "x", -1.0, 1.0)
    sim, _ = onnxsim.simplify(m)
    assert _meta(sim, "onnxsim.certify") == "proved"


def test_certify_false_opts_out():
    sim, _ = onnxsim.simplify(_conv_bn_relu(), certify=False)
    assert _meta(sim, "onnxsim.certify") is None


def test_certify_true_reports_outcome(capsys):
    m = _conv_bn_relu()
    R.set_range(m, "x", -1.0, 1.0)
    onnxsim.simplify(m, certify=True)
    assert "Certify: proved" in capsys.readouterr().out


def test_size_guard_skips_and_explains(monkeypatch, capsys):
    monkeypatch.setattr(simp, "_CERTIFY_MAX_BYTES", 10)
    sim, _ = onnxsim.simplify(_conv_bn_relu(), certify=True)
    assert "larger than the certify size limit" in capsys.readouterr().out
    assert _meta(sim, "onnxsim.certify") is None
    sim, _ = onnxsim.simplify(_conv_bn_relu())  # default: silent
    assert capsys.readouterr().out.count("Certify") == 0


def test_simplify_signature_still_lists_certify():
    import inspect

    sig = inspect.signature(onnxsim.simplify)
    assert "certify" in sig.parameters and list(sig.parameters)[:2] == [
        "model",
        "check_n",
    ]


def test_refuted_verdict_is_classified_and_warned():
    good = _conv_bn_relu()
    R.set_range(good, "x", -1.0, 1.0)
    bad = onnx.ModelProto()
    bad.CopyFrom(good)
    for t in bad.graph.initializer:
        if t.name == "be":
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t) + 0.5, "be"))
    status, detail = simp._certify_result(good, bad, {}, False)
    assert status == "refuted" and "differ" in detail


def test_failures_inside_certify_never_break_simplify(monkeypatch, capsys):
    import onnxsim.certify as C

    def boom(*a, **k):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(C, "certify", boom)
    sim, ok = onnxsim.simplify(_conv_bn_relu())
    assert isinstance(sim, onnx.ModelProto) and ok
    assert (
        "certify failed internally: RuntimeError: solver exploded"
        in capsys.readouterr().out
    )
    assert _meta(sim, "onnxsim.certify") == "error"


def test_check_n_draws_random_inputs_from_annotated_range(monkeypatch):
    m = _model("m (float[1,8] x) => (float[1,8] y) { y = Relu(x) }")
    R.set_range(m, "x", 10.0, 11.0)
    seen = []
    real = model_checking.backend.run_model

    def spy(model, inputs, **kw):
        seen.append(inputs["x"].copy())
        return real(model, inputs, **kw)

    monkeypatch.setattr(model_checking.backend, "run_model", spy)
    sim, ok = onnxsim.simplify(m, check_n=3, certify=False)
    assert ok and seen
    assert all(a.min() >= 10.0 and a.max() <= 11.0 for a in seen)


def test_unannotated_inputs_keep_the_old_default_fill(monkeypatch):
    m = _model("m (float[1,8] x) => (float[1,8] y) { y = Relu(x) }")
    seen = []
    real = model_checking.backend.run_model
    monkeypatch.setattr(
        model_checking.backend,
        "run_model",
        lambda model, inputs, **kw: (
            seen.append(inputs["x"].copy()),
            real(model, inputs, **kw),
        )[1],
    )
    onnxsim.simplify(m, check_n=2, certify=False)
    assert all(a.min() >= 0.0 and a.max() < 1.0 for a in seen)


def test_output_leaving_annotated_range_warns_once(capsys):
    m = _model("m (float[1,8] x) => (float[1,8] y) { y = Relu(x) }")
    R.set_range(m, "x", 10.0, 11.0)
    R.set_range(m, "y", 0.0, 1.0)  # wrong on purpose: outputs are in [10, 11]
    onnxsim.simplify(m, check_n=3, certify=False)
    out = capsys.readouterr().out
    assert out.count("WARNING: original model output y") == 1
    assert "leaves the annotated range" in out
