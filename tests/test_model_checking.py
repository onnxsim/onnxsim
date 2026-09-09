"""Tests for ``onnxsim.model_checking``: the Random* op seed pinning added for
models like VOICEVOX's predict_sing_f0.onnx (see below), and the NaN handling of
the output comparison itself (GitHub issue #1285).
"""

import copy

import numpy as np
import onnx
from onnx import parser

from onnxsim import model_checking
from onnxsim.model_checking import compare


def _model(body, opset=17, ir_version=10):
    # Pin ir_version: the installed onnx package's default IR version can be
    # newer than the onnxruntime build available in some CI jobs supports
    # (matches test_backend.py's _make_foldable_model).
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _random_normal_like_model(with_seed: bool = False) -> onnx.ModelProto:
    node = (
        "y = RandomNormalLike<seed = 1.0>(x)"
        if with_seed
        else "y = RandomNormalLike(x)"
    )
    return _model(
        f"""
        g (float[1] x) => (float[1] y)
        {{
          {node}
        }}
        """
    )


def _plain_model() -> onnx.ModelProto:
    return _model(
        """
        g (float[1] x) => (float[1] y)
        {
          y = Identity(x)
        }
        """
    )


# --------------------------------------------------------------------------- #
# _has_unseeded_random_ops / _with_fixed_random_seeds
# --------------------------------------------------------------------------- #
def test_has_unseeded_random_ops_detects_missing_seed():
    assert model_checking._has_unseeded_random_ops(_random_normal_like_model())


def test_has_unseeded_random_ops_respects_existing_seed():
    # A seed the model author deliberately set is left alone: not "unseeded".
    assert not model_checking._has_unseeded_random_ops(
        _random_normal_like_model(with_seed=True)
    )


def test_has_unseeded_random_ops_false_for_plain_model():
    assert not model_checking._has_unseeded_random_ops(_plain_model())


def test_with_fixed_random_seeds_sets_seed_without_mutating_input():
    model = _random_normal_like_model()
    seeded = model_checking._with_fixed_random_seeds(model, 7.0)

    assert not any(a.name == "seed" for a in model.graph.node[0].attribute)
    seed_attrs = [a for a in seeded.graph.node[0].attribute if a.name == "seed"]
    assert len(seed_attrs) == 1
    assert seed_attrs[0].f == 7.0


def test_with_fixed_random_seeds_does_not_override_existing_seed():
    model = _random_normal_like_model(with_seed=True)
    seeded = model_checking._with_fixed_random_seeds(model, 7.0)
    seed_attrs = [a for a in seeded.graph.node[0].attribute if a.name == "seed"]
    assert len(seed_attrs) == 1
    assert seed_attrs[0].f == 1.0  # the author's seed, not the check's


# --------------------------------------------------------------------------- #
# compare(): the actual bug this fixes
# --------------------------------------------------------------------------- #
# onnxruntime.InferenceSession draws fresh noise from an unseeded Random* op on
# every session, and ``compare`` creates a brand-new session per model per
# check_n trial -- so directly exercising a real backend here would be
# inherently flaky (the exact draw pattern is an onnxruntime/reference-
# evaluator implementation detail, confirmed empirically to differ between
# "same graph, two fresh sessions" and "structurally different graphs" in
# ways that vary by onnxruntime version). Faking the backend isolates the
# actual bug -- an unseeded op's output legitimately differs call-to-call --
# from that implementation detail, and keeps the test fast and deterministic.
def _fake_run_model(call_count):
    def run_model(model, inputs, **kwargs):
        seed_attrs = [
            a.f for n in model.graph.node for a in n.attribute if a.name == "seed"
        ]
        if seed_attrs:
            value = seed_attrs[0]  # deterministic given the seed
        else:
            call_count[0] += 1
            value = float(call_count[0])  # a fresh "draw" every call
        return {"y": np.array([value], dtype=np.float32)}

    return run_model


def test_compare_pins_seed_so_random_op_model_passes(monkeypatch):
    monkeypatch.setattr(model_checking.backend, "run_model", _fake_run_model([0]))
    model = _random_normal_like_model()
    # Comparing the (structurally identical) model to itself must pass: within
    # each trial, model_ori's and model_opt's copies get the same pinned seed.
    assert compare(model, model, n_times=3) is True


def test_compare_without_seed_pinning_would_report_false_failure(monkeypatch):
    # Demonstrates the bug being fixed: with seed-pinning disabled, the same
    # fake non-deterministic backend makes compare() report a failure even
    # though model_ori and model_opt are the exact same model.
    monkeypatch.setattr(model_checking.backend, "run_model", _fake_run_model([0]))
    monkeypatch.setattr(model_checking, "_has_unseeded_random_ops", lambda model: False)
    model = _random_normal_like_model()
    assert compare(model, model, n_times=1) is False


def test_compare_skips_seeding_scan_when_check_n_is_zero():
    # model_ori is None when the caller has already freed it (check_n == 0);
    # this must not be dereferenced.
    assert compare(_plain_model(), None, n_times=0) is True


def test_compare_unaffected_for_plain_models(monkeypatch):
    calls = []
    real_run_model = model_checking.backend.run_model

    def spy_run_model(model, inputs, **kwargs):
        calls.append(model)
        return real_run_model(model, inputs, **kwargs)

    monkeypatch.setattr(model_checking.backend, "run_model", spy_run_model)
    model = _plain_model()
    assert compare(model, model, n_times=1) is True
    # No Random* ops -> the exact model objects are forwarded, no seeded copy.
    assert all(m is model for m in calls)


# --------------------------------------------------------------------------- #
# compare(): NaN on both sides is not a change (GitHub issue #1285)
# --------------------------------------------------------------------------- #
def _div_model() -> onnx.ModelProto:
    # Two separate inputs (rather than Div(x, x)) so that no optimizer pass can
    # rewrite the division away and change what the check actually compares.
    return _model(
        """
        g (float[3] x, float[3] z) => (float[3] y)
        {
          y = Div(x, z)
        }
        """
    )


# 0 / 0 -> NaN in the first position, ordinary numbers in the other two.
_DIV_BY_ZERO_INPUT = {
    "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
    "z": np.array([0.0, 1.0, 4.0], dtype=np.float32),
}


def _named(model: onnx.ModelProto, name: str) -> onnx.ModelProto:
    model = copy.deepcopy(model)
    model.graph.name = name
    return model


def _fake_backend(outputs):
    """A backend returning a per-model canned output, keyed by graph name.

    Lets a test pin exactly what "the original model" and "the simplified
    model" each returned, which is the only thing ``compare`` looks at.
    """

    def run_model(model, inputs, **kwargs):
        return {"y": outputs[model.graph.name]}

    return run_model


def _compare_fake(monkeypatch, ori, opt, verbose=True):
    monkeypatch.setattr(
        model_checking.backend,
        "run_model",
        _fake_backend(
            {
                "ori": np.array(ori, dtype=np.float32),
                "opt": np.array(opt, dtype=np.float32),
            }
        ),
    )
    model = _div_model()
    return compare(
        _named(model, "opt"),
        _named(model, "ori"),
        n_times=1,
        input_data=_DIV_BY_ZERO_INPUT,
        verbose=verbose,
    )


def test_div_by_zero_model_really_produces_nan():
    # Guards the premise of the tests below: were this model to stop producing
    # a NaN, they would start passing vacuously.
    res = model_checking.backend.run_model(_div_model(), _DIV_BY_ZERO_INPUT)
    assert np.isnan(res["y"][0])
    assert not np.isnan(res["y"][1:]).any()


def test_compare_tolerates_nan_produced_by_both_models():
    # The original model is already undefined (0 / 0) at this input, so an
    # equally-NaN output from the simplified model says nothing about
    # simplification and must not be reported as a change. Before the fix,
    # NaN != NaN made this print "The max diff is nan." and return False.
    model = _div_model()
    assert compare(model, model, n_times=1, input_data=_DIV_BY_ZERO_INPUT) is True


def test_simplify_check_passes_for_model_whose_output_is_nan():
    # End-to-end: simplify(..., check_n=1) reports success rather than the
    # false "Check failed" of issue #1285.
    import onnxsim

    _, check_ok = onnxsim.simplify(
        _div_model(), check_n=1, input_data=_DIV_BY_ZERO_INPUT
    )
    assert check_ok is True


def test_compare_still_catches_nan_on_one_side_only(monkeypatch):
    # What the fix must stay sensitive to: a NaN where the original model
    # returned a number is a genuine regression.
    assert (
        _compare_fake(
            monkeypatch, ori=[1.0, 2.0, 3.0], opt=[1.0, np.nan, 3.0], verbose=False
        )
        is False
    )


def test_compare_still_catches_number_where_original_was_nan(monkeypatch):
    # ... and the mirror image: the simplified model inventing a number where
    # the original was undefined is just as much a change.
    assert (
        _compare_fake(
            monkeypatch, ori=[1.0, np.nan, 3.0], opt=[1.0, 2.0, 3.0], verbose=False
        )
        is False
    )


def test_compare_still_catches_plain_numeric_difference(monkeypatch):
    # equal_nan must not have loosened the ordinary finite comparison.
    assert (
        _compare_fake(
            monkeypatch, ori=[1.0, 2.0, 3.0], opt=[1.0, 2.5, 3.0], verbose=False
        )
        is False
    )


def test_compare_tolerates_matching_infinities(monkeypatch):
    # np.allclose compares infinities exactly on its own; 1 / 0 -> +Inf on both
    # sides is as much a property of the model as a matching NaN is.
    assert (
        _compare_fake(
            monkeypatch,
            ori=[np.inf, -np.inf, 3.0],
            opt=[np.inf, -np.inf, 3.0],
        )
        is True
    )


def test_compare_still_catches_flipped_infinity_sign(monkeypatch):
    assert (
        _compare_fake(
            monkeypatch, ori=[np.inf, 2.0, 3.0], opt=[-np.inf, 2.0, 3.0], verbose=False
        )
        is False
    )


def test_compare_explains_a_one_sided_nan_in_its_report(monkeypatch, capsys):
    # "The max diff is nan." on its own reads exactly like the tolerated
    # both-sides-NaN case, which is what made issue #1285 confusing to read.
    _compare_fake(monkeypatch, ori=[1.0, 2.0, 3.0], opt=[1.0, np.nan, 3.0])
    assert "NaN appears in one model's output" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# _has_one_sided_nan
# --------------------------------------------------------------------------- #
def test_has_one_sided_nan():
    nan = np.array([1.0, np.nan], dtype=np.float32)
    num = np.array([1.0, 2.0], dtype=np.float32)
    assert model_checking._has_one_sided_nan(nan, num)
    assert model_checking._has_one_sided_nan(num, nan)
    assert not model_checking._has_one_sided_nan(nan, nan)
    assert not model_checking._has_one_sided_nan(num, num)


def test_has_one_sided_nan_false_for_non_float_outputs():
    # np.isnan raises on integer/bool arrays; they cannot hold a NaN anyway.
    ints = np.array([1, 2], dtype=np.int64)
    bools = np.array([True, False])
    assert not model_checking._has_one_sided_nan(ints, ints)
    assert not model_checking._has_one_sided_nan(bools, bools)
