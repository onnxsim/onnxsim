"""certify()'s zonotope fallback (verdict ``proved-affine``) and the ``unproven-no-ranges`` fix.

The fallback only runs for an output the Z3 steps left ``skipped``. Several tests force that
(by making the SMT step give up) so they exercise the fallback whatever Z3's speed on the
machine; others run the natural path on real ``onnxsim.simplify`` output.

The properties that matter most are the negative ones: the fallback never turns a wrong
rewrite into ``proved``, never reports ``refuted`` (a bound over the tolerance is an
over-approximation, not a witness), and never raises.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

import onnxsim

pytest.importorskip("z3", reason="onnxsim.certify needs the 'verify' extra (z3-solver)")
from onnxsim import certify as C  # noqa: E402
from onnxsim import ranges as R  # noqa: E402


def _model(body, initializer=None, opset=15, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    onnx.checker.check_model(model)
    return model


def _conv_bn_relu(hw=8, k=2, layers=2, seed=0, relu=True):
    """``layers`` x (Conv -> BatchNorm -> Relu): onnxsim folds each BatchNorm into its Conv,
    leaving a Relu between two rewritten layers, which is what Z3 case-splits."""
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    inits, lines, cin, prev = {}, [], 3, "x"
    for i in range(layers):
        inits.update(
            {
                f"W{i}": f(k, cin, 3, 3) * 0.3,
                f"B{i}": f(k) * 0.1,
                f"g{i}": rng.uniform(0.5, 1.5, k).astype(np.float32),
                f"be{i}": f(k) * 0.1,
                f"mu{i}": f(k) * 0.1,
                f"var{i}": rng.uniform(0.5, 2, k).astype(np.float32),
            }
        )
        act = f"r{i} = Relu(b{i})" if relu else f"r{i} = Identity(b{i})"
        lines += [
            f"c{i} = Conv<pads=[1,1,1,1]>({prev}, W{i}, B{i})",
            f"b{i} = BatchNormalization<epsilon=1e-5>(c{i}, g{i}, be{i}, mu{i}, var{i})",
            act,
        ]
        prev, cin = f"r{i}", k
    body = (
        f"m (float[1,3,{hw},{hw}] x) => (float[1,{k},{hw},{hw}] y) {{ "
        + " ".join(lines)
        + f" y = Identity({prev}) }}"
    )
    return _model(body, inits)


def _with(model, name, fn):
    out = onnx.ModelProto()
    out.CopyFrom(model)
    for t in out.graph.initializer:
        if t.name == name:
            t.CopyFrom(numpy_helper.from_array(fn(numpy_helper.to_array(t)), name))
    return out


_BOX = {"x": (-1.0, 1.0)}


@pytest.fixture
def smt_gives_up(monkeypatch):
    """Make every SMT window end ``skipped``, as when Z3 times out, so the fallback runs."""
    monkeypatch.setattr(
        C._Prover,
        "_smt",
        lambda self, a, b, mk: mk(C.SKIPPED, "forced: solver gave up"),
    )


# ---- the fallback proves what Z3 gave up on ---------------------------------------------


def test_fallback_proves_a_two_layer_fold_when_smt_gives_up(smt_gives_up):
    model = _conv_bn_relu()
    sim, _ = onnxsim.simplify(model, certify=False)
    assert "BatchNormalization" not in {n.op_type for n in sim.graph.node}
    report = C.certify(model, sim, input_ranges=_BOX)
    assert report.ok, str(report)
    assert report.outputs["y"] == C.PROVED_AFFINE
    (w,) = report.windows
    assert "zonotope: max|orig-simplified| <=" in w.detail
    assert "forced: solver gave up" in w.detail  # the SMT reason stays visible


def test_proved_affine_counts_as_proved():
    rep = C.CertifyReport({"y": C.PROVED_AFFINE, "z": C.PROVED_SMT}, [])
    assert rep.ok
    assert not C.CertifyReport({"y": C.PROVED_AFFINE, "z": C.SKIPPED}, []).ok


def test_natural_path_certifies_the_two_layer_fold():
    # No forcing: whichever step proves it, it must be proved (Z3 usually gives up here and the
    # fallback takes over; on a faster or slower machine another step may win).
    model = _conv_bn_relu(hw=8, k=2)
    sim, _ = onnxsim.simplify(model, certify=False)
    report = C.certify(
        model, sim, input_ranges=_BOX, max_work=100_000, timeout_ms=5_000
    )
    assert report.ok, str(report)


def test_simplify_records_a_proved_verdict_for_the_two_layer_fold():
    model = _conv_bn_relu(hw=8, k=2)
    R.set_range(model, "x", -1.0, 1.0)
    sim, _ = onnxsim.simplify(model, certify=True)
    meta = {p.key: p.value for p in sim.metadata_props}
    assert meta["onnxsim.certify"] == "proved", meta.get("onnxsim.certify.detail")


# ---- never proves a wrong rewrite -------------------------------------------------------


def test_wrong_bias_behind_a_relu_is_not_proved_and_not_refuted_by_the_fallback(
    smt_gives_up,
):
    good = _conv_bn_relu(layers=1, relu=True)
    bad = _with(good, "be0", lambda a: a + 0.25)
    report = C.certify(good, bad, input_ranges=_BOX)
    assert not report.ok
    # The fallback gave a bound but cannot claim a difference: skipped, with the number.
    assert report.outputs["y"] == C.SKIPPED
    assert "zonotope: max|orig-simplified| <=" in report.windows[0].detail
    bound = float(
        report.windows[0].detail.split("max|orig-simplified| <= ")[1].split(" ")[0]
    )
    assert bound > 0.2  # the real difference is 0.25 behind the Relu


def test_wrong_bias_is_refuted_by_z3_on_the_natural_path():
    good = _conv_bn_relu(layers=1, relu=True)
    bad = _with(good, "be0", lambda a: a + 0.25)
    report = C.certify(good, bad, input_ranges=_BOX)
    assert not report.ok and report.outputs["y"] == C.REFUTED


def _conv_relu(w):
    return _model(
        "m (float[1,3,6,6] x) => (float[1,4,4,4] y) { c = Conv(x, W, B)  y = Relu(c) }",
        dict(W=w, B=np.zeros(4, np.float32)),
    )


def test_rgb_bgr_swap_is_never_proved(smt_gives_up):
    w = np.random.default_rng(5).standard_normal((4, 3, 3, 3)).astype(np.float32)
    report = C.certify(
        _conv_relu(w),
        _conv_relu(w[:, ::-1].copy()),
        input_ranges={"x": (0.0, 1.0)},
    )
    assert not report.ok and report.outputs["y"] == C.SKIPPED
    assert "zonotope: max|orig-simplified| <=" in report.windows[0].detail


def test_rgb_bgr_swap_is_refuted_by_z3_on_the_natural_path():
    w = np.random.default_rng(5).standard_normal((4, 3, 3, 3)).astype(np.float32)
    report = C.certify(
        _conv_relu(w), _conv_relu(w[:, ::-1].copy()), input_ranges={"x": (0.0, 1.0)}
    )
    assert report.outputs["y"] == C.REFUTED


_PAIR_NAMES = [
    "good fold",
    "bias +0.25",
    "bias +1e-3",
    "weights x1.5",
    "weights perturbed 1e-3",
    "mean shifted",
]


def _pair(name):
    good = _conv_bn_relu(hw=6, k=2, layers=1)
    if name == "good fold":
        return good, onnxsim.simplify(good, certify=False)[0]
    edits = {
        "bias +0.25": ("be0", lambda a: a + 0.25),
        "bias +1e-3": ("be0", lambda a: a + 1e-3),
        "weights x1.5": ("W0", lambda a: a * 1.5),
        "weights perturbed 1e-3": ("W0", lambda a: a + 1e-3),
        "mean shifted": ("mu0", lambda a: a + 0.2),
    }
    tensor, fn = edits[name]
    return good, _with(good, tensor, fn)


@pytest.mark.parametrize("name", _PAIR_NAMES)
def test_fallback_never_proves_what_z3_refutes(name, monkeypatch):
    """Agreement: whenever Z3 finds a real difference, the zonotope step must not prove equality."""
    orig, other = _pair(name)
    with monkeypatch.context() as m:
        m.setattr(C, "_ZONOTOPE_MAX_ELEMENTS", 0)  # Z3 steps only
        z3_only = C.certify(orig, other, input_ranges=_BOX)
    proved, detail = C._zonotope_check(orig, other, _BOX, 1e-5, 1e-4, ["y"])["y"]
    if z3_only.outputs["y"] == C.REFUTED:
        assert not proved, f"{name}: zonotope proved a pair Z3 refuted: {detail}"
    if proved:
        assert z3_only.outputs["y"] != C.REFUTED


# ---- when it is not attempted -----------------------------------------------------------


def test_not_attempted_without_ranges(smt_gives_up):
    model = _conv_bn_relu()
    sim, _ = onnxsim.simplify(model, certify=False)
    report = C.certify(model, sim)  # no input_ranges
    assert report.outputs["y"] == C.SKIPPED
    assert "zonotope not used: input 'x' has no range" in report.windows[0].detail


def test_model_annotations_do_not_leak_into_the_fallback(smt_gives_up):
    # certify()'s rule is "inputs not listed in input_ranges are unbounded"; a model's own
    # annotations are passed in by simplify(), not read behind the caller's back.
    model = _conv_bn_relu()
    R.set_range(model, "x", -1.0, 1.0)
    sim, _ = onnxsim.simplify(model, certify=False)
    assert "x" in R.get_ranges(sim)
    report = C.certify(model, sim)
    assert report.outputs["y"] == C.SKIPPED
    assert "has no range" in report.windows[0].detail


def test_unbounded_range_is_not_attempted():
    model = _conv_bn_relu()
    sim, _ = onnxsim.simplify(model, certify=False)
    (proved, detail) = C._zonotope_check(
        model, sim, {"x": (-1.0, np.inf)}, 1e-5, 1e-4, ["y"]
    )["y"]
    assert not proved and "has an unbounded range" in detail


def test_respects_the_overall_time_budget(smt_gives_up):
    model = _conv_bn_relu()
    sim, _ = onnxsim.simplify(model, certify=False)
    report = C.certify(model, sim, input_ranges=_BOX, total_timeout_ms=0)
    assert report.outputs["y"] == C.SKIPPED
    assert "time budget exhausted" in report.windows[0].detail


def test_respects_the_cost_cap(smt_gives_up, monkeypatch):
    model = _conv_bn_relu()
    sim, _ = onnxsim.simplify(model, certify=False)
    monkeypatch.setattr(C, "_ZONOTOPE_MAX_ELEMENTS", 1)
    report = C.certify(model, sim, input_ranges=_BOX)
    assert report.outputs["y"] == C.SKIPPED
    assert "zonotope not used: too large" in report.windows[0].detail


def test_a_failure_inside_the_fallback_never_raises(smt_gives_up, monkeypatch):
    from onnxsim import zonotope

    def boom(*a, **k):
        raise RuntimeError("zonotope exploded")

    monkeypatch.setattr(zonotope, "bound_difference", boom)
    model = _conv_bn_relu()
    sim, _ = onnxsim.simplify(model, certify=False)
    report = C.certify(model, sim, input_ranges=_BOX)
    assert report.outputs["y"] == C.SKIPPED
    assert "failed (RuntimeError: zonotope exploded)" in report.windows[0].detail


def test_refuted_outputs_are_not_sent_to_the_fallback(monkeypatch):
    calls = []
    from onnxsim import zonotope

    real = zonotope.bound_difference
    monkeypatch.setattr(
        zonotope,
        "bound_difference",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    good = _conv_bn_relu(layers=1)
    bad = _with(good, "be0", lambda a: a + 0.25)
    assert C.certify(good, bad, input_ranges=_BOX).outputs["y"] == C.REFUTED
    assert calls == []


def test_range_annotations_are_stripped_from_the_models_given_to_zonotope():
    model = _conv_bn_relu(layers=1)
    R.set_range(model, "x", -1.0, 1.0)
    stripped = C._without_range_annotations(model)
    assert R.get_ranges(stripped) == {}
    assert "x" in R.get_ranges(model)  # the caller's model is untouched
    plain = _conv_bn_relu(layers=1)
    assert (
        C._without_range_annotations(plain) is plain
    )  # no copy when there is nothing to strip


# ---- the unproven-no-ranges fix ---------------------------------------------------------


def _verdict(model, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        sim, _ = onnxsim.simplify(model, **kw)
    meta = {p.key: p.value for p in sim.metadata_props}
    return meta.get("onnxsim.certify"), buf.getvalue()


def test_output_only_annotation_still_downgrades_to_unproven():
    model = _conv_bn_relu(layers=1, relu=False)  # a folded BatchNorm, input unbounded
    R.set_range(model, "y", -100.0, 100.0)  # annotated, but only on the OUTPUT
    status, out = _verdict(model)
    assert status == "unproven-no-ranges"
    assert "WARNING" not in out  # not printed as a refutation


def test_no_annotation_at_all_is_unchanged():
    status, out = _verdict(_conv_bn_relu(layers=1, relu=False))
    assert status == "unproven-no-ranges" and "WARNING" not in out


def test_input_annotation_still_proves_it_even_with_an_output_annotation():
    model = _conv_bn_relu(layers=1, relu=False)
    R.set_range(model, "x", -1.0, 1.0)
    R.set_range(model, "y", -100.0, 100.0)
    assert _verdict(model)[0] == "proved"


def test_output_annotation_does_not_count_as_an_input_range():
    model = _conv_bn_relu(layers=1, relu=False)
    R.set_range(model, "y", -100.0, 100.0)
    sim, _ = onnxsim.simplify(model, certify=False)
    status, detail = onnxsim.onnx_simplifier._certify_result(model, sim, {}, False)
    assert status == "unproven-no-ranges", detail
