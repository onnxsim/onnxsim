"""No-device checks for ``scripts/axera/width_retarget.py`` on committed
compiler-built fixtures: a ``[1, 64]`` program moved to another width, and
ReduceMean emitted from two templates, must equal the native build at that
width outside segment 0's slot table."""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import graph_stitch as gs  # noqa: E402
import width_retarget as wr  # noqa: E402

FIXTURES = os.path.join(AXERA, "fixtures", "width_retarget")
with open(os.path.join(FIXTURES, "index.json")) as _f:
    INDEX = json.load(_f)
BUILDS, GRAPHS = INDEX["builds"], INDEX["graphs"]


def _model(name):
    return gs.load_model(os.path.join(FIXTURES, BUILDS[name]["file"]))


def _segments(model):
    mc = bytes(gs.mre.mcode_initializer(model).raw_data)
    return [gs.records(r) for r in gs.suc.decode_segments(mc)]


# every native build at another width than 64, per op (ReduceMean and Neg below)
NATIVE = sorted(
    (b["op"], b["width"])
    for name, b in BUILDS.items()
    if b["op"] in wr.OPS
    and b["op"] != "Neg"
    and b["width"] != 64
    and not name.startswith("c")
    and (b["op"], b["width"]) != ("Softmax", 100)
)


def test_the_native_widths_are_the_ones_the_docstring_names():
    widths = {}
    for op, n in NATIVE:
        widths.setdefault(op, []).append(n)
    assert widths == {
        "Sigmoid": [100, 384, 512, 576, 2048],
        "Mul": [100, 384, 512, 576, 2048],
        "Softmax": [384, 512, 576, 2048],
        "Add": [100, 576],
        "Div": [100, 576],
        "Sqrt": [100, 576],
    }


@pytest.mark.parametrize("op,n", NATIVE)
def test_width_64_template_gives_the_native_build(op, n):
    native = BUILDS[f"{op.lower()}_{n}_pm1"]
    want = _model(f"{op.lower()}_{n}_pm1")
    moved = wr.retarget_width(_model(f"{op.lower()}_64_pm1"), n, op)
    # the width rewrite keeps the template's calibration; the native build's
    # calibration, slot order and PARAM placement are its own
    got = wr.calibrate_program(
        moved,
        op,
        native["scales"],
        native["zero_points"],
        slot_order=list(want.graph.node[0].input),
        output_param=native["output_param"],
    )
    assert gs.compare_models(got, want) == []


@pytest.mark.parametrize("cal", ["pm1", "pm4"])
def test_neg_width_576_from_the_width_64_template(cal):
    # pm1 is Neg's small program, pm4 the large one (with npu_params words)
    old, new = BUILDS[f"neg_64_{cal}"], BUILDS[f"neg_576_{cal}"]
    moved = wr.retarget_width(_model(f"neg_64_{cal}"), 576, "Neg")
    got = wr.calibrate_program(
        moved,
        "Neg",
        new["scales"],
        new["zero_points"],
        old_scales=old["scales"],
        old_zero_points=old["zero_points"],
    )
    assert gs.compare_models(got, _model(f"neg_576_{cal}")) == []


@pytest.mark.parametrize("op", ["Sigmoid", "Mul", "Add", "Div", "Sqrt", "Softmax"])
def test_width_64_is_the_template_itself(op):
    template = _model(f"{op.lower()}_64_pm1")
    same = wr.retarget_width(template, 64, op)
    assert same.SerializeToString() == template.SerializeToString()
    assert same is not template


def test_only_width_dependent_records_change():
    template = _model("mul_64_pm1")
    a, b = _segments(template), _segments(wr.retarget_width(template, 576, "Mul"))
    assert [len(s) for s in a] == [len(s) for s in b]
    changed = [(x, y) for sa, sb in zip(a, b) for x, y in zip(sa, sb) if x != y]
    assert len(changed) == 15
    assert all(x[:4] == y[:4] for x, y in changed)  # same verb and register


def _graph(name, programs):
    """The wiring of a committed ``[1,576]`` graph with ``programs`` replacing
    native components (by op index)."""
    g = GRAPHS[name]
    ops = [dict(o, program=_model(o["component"])) for o in g["ops"]]
    for k, program in programs.items():
        ops[k]["program"] = program
    wiring = {k: g[k] for k in ("inputs", "graph_inputs", "output")}
    wiring["ops"] = ops
    if "output_param" in g:
        wiring["output_param"] = g["output_param"]
    return wiring


def _native_graph(name, cal):
    q = GRAPHS[name]["calibrations"][cal]
    oracle = gs.load_model(os.path.join(FIXTURES, q["oracle"]))
    return q["scales"], q["zero_points"], oracle


@pytest.mark.parametrize("cal", ["pm1", "pm4"])
def test_silu_576_stitched_from_retargeted_templates(cal):
    # the two steps compose: width-64 programs moved to 576, then stitched
    sig = wr.retarget_width(_model("sigmoid_64_pm1"), 576, "Sigmoid")
    mul = wr.retarget_width(_model("mul_64_pm1"), 576, "Mul")
    scales, zps, oracle = _native_graph("silu576", cal)
    got = gs.stitch_model(_graph("silu576", {0: sig, 1: mul}), scales, zps)
    assert gs.compare_models(got, oracle) == []


@pytest.mark.parametrize("cal", ["pm1", "pm4"])
def test_rmsnorm_576_with_an_emitted_reducemean(cal):
    # the ReduceMean component emitted at 576 from the 256 and 384 templates,
    # in the other job order than the native 576 component: a component's own
    # PARAM placement does not matter, the graph's is given in the wiring
    b = BUILDS["reducemean_384_pm1"]
    rm = wr.emit_reducemean(
        576,
        b["scales"],
        b["zero_points"],
        output_param="late",
        small=_model("reducemean_256_pm1"),
        large=_model("reducemean_384_pm1"),
    )
    scales, zps, oracle = _native_graph("rmsnorm576", cal)
    got = gs.stitch_model(_graph("rmsnorm576", {1: rm}), scales, zps)
    assert gs.compare_models(got, oracle) == []


def test_sigmoid_512_runs_its_output_param_job_early():
    # the one retarget_width op whose native build has the other job order;
    # nothing predicts it, so calibrate_program takes it as an argument
    assert [n for n, b in BUILDS.items() if b.get("output_param") == "early"] == [
        "sigmoid_512_pm1",
        "reducemean_512_pm1",
        "reducemean_576_pm1",
        "reducemean_576_pm4",
    ]
    native, want = BUILDS["sigmoid_512_pm1"], _model("sigmoid_512_pm1")
    moved = wr.retarget_width(_model("sigmoid_64_pm1"), 512, "Sigmoid")
    late = wr.calibrate_program(
        moved, "Sigmoid", native["scales"], native["zero_points"], output_param="late"
    )
    early = wr.calibrate_program(
        moved, "Sigmoid", native["scales"], native["zero_points"], output_param="early"
    )
    assert gs.compare_models(early, want) == []
    assert gs.compare_models(late, want) != []
    # the same jobs in another order: the register deltas differ, not the count
    assert len(_segments(late)[2]) == len(_segments(want)[2])
    roles = [[j.role for j in gs.Program(m, "sigmoid").jobs] for m in (early, late)]
    assert roles[0] == ["PARAM", "QUANT", "PARAM", "CORE", "DEQUANT"]
    assert roles[1] == ["PARAM", "QUANT", "CORE", "PARAM", "DEQUANT"]


# ---- refused -------------------------------------------------------------------
@pytest.mark.parametrize(
    "op,n,why",
    [
        ("Sigmoid", 32, "only 64..2048"),
        ("Sigmoid", 4096, "only 64..2048"),
        ("Add", 2048, "only 64..576"),
        ("Neg", 608, "only 64..576"),
        ("Sigmoid", 96 + 1, "not a multiple of 32"),
        ("Mul", 200, "not a multiple of 32"),
        ("Neg", 100, "not a multiple of 32"),
        ("Softmax", 100, "different, padded program"),
        ("Softmax", 72, "different, padded program"),
        # 1088-byte buffers leave the scratch window; only multiples of 128 were seen
        ("Mul", 1088, "alignment"),
        ("Softmax", 1120, "alignment"),
    ],
)
def test_unmeasured_widths_are_refused(op, n, why):
    with pytest.raises(ValueError, match=why):
        wr.retarget_width(_model(f"{op.lower()}_64_pm1"), n, op)


def test_softmax_100_is_another_program():
    # why Softmax is refused there: 20 more main-engine records than at any
    # multiple of 32
    counts = {n: len(_segments(_model(f"softmax_{n}_pm1"))[2]) for n in (64, 100, 384)}
    assert counts == {64: 476, 100: 496, 384: 476}


def test_other_templates_and_ops_are_refused():
    with pytest.raises(ValueError, match=r"fitted from \[1, 64\] templates"):
        wr.retarget_width(_model("sigmoid_576_pm1"), 384, "Sigmoid")
    with pytest.raises(ValueError, match="emit_reducemean"):
        wr.retarget_width(_model("reducemean_64_pm1"), 384, "ReduceMean")
    with pytest.raises(ValueError, match="retarget_width serves"):
        wr.retarget_width(_model("sigmoid_64_pm1"), 384, "Tanh")
    # a template that already holds a relocated buffer cannot be moved
    with pytest.raises(ValueError, match="outside the scratch window"):
        wr._retarget(_model("sigmoid_2048_pm1"), 2048, 576)


def test_calibrate_program_arguments():
    model, b = _model("sigmoid_64_pm1"), BUILDS["sigmoid_64_pm1"]
    with pytest.raises(ValueError, match="output_param"):
        wr.calibrate_program(
            model, "Sigmoid", b["scales"], b["zero_points"], output_param="first"
        )
    same = wr.calibrate_program(model, "Sigmoid", b["scales"], b["zero_points"])
    assert gs.compare_models(same, model) == []
    neg, n = _model("neg_64_pm1"), BUILDS["neg_64_pm1"]
    with pytest.raises(ValueError, match="own calibration"):
        wr.calibrate_program(neg, "Neg", n["scales"], n["zero_points"])
    with pytest.raises(ValueError, match="slot and job order"):
        wr.calibrate_program(
            neg, "Neg", n["scales"], n["zero_points"], output_param="late"
        )


# ---- ReduceMean ------------------------------------------------------------------
REDUCEMEAN = sorted(
    (b["width"], b["calibration"]) for b in BUILDS.values() if b["op"] == "ReduceMean"
)
TEMPLATE_PAIRS = [(64, 576), (128, 288), (256, 384), (256, 576)]


def test_reducemean_native_builds():
    assert [n for n, c in REDUCEMEAN if c == "pm1"] == [
        64,
        100,
        128,
        256,
        288,
        384,
        512,
        576,
        2048,
    ]
    assert [n for n, c in REDUCEMEAN if c == "pm4"] == [64, 576]


@pytest.mark.parametrize("n,cal", REDUCEMEAN)
@pytest.mark.parametrize("small,large", TEMPLATE_PAIRS)
def test_reducemean_from_two_templates(small, large, n, cal):
    native = BUILDS[f"reducemean_{n}_{cal}"]
    got = wr.emit_reducemean(
        n,
        native["scales"],
        native["zero_points"],
        output_param=native["output_param"],
        small=_model(f"reducemean_{small}_pm1"),
        large=_model(f"reducemean_{large}_pm1"),
    )
    assert gs.compare_models(got, _model(f"reducemean_{n}_{cal}")) == []


def _core(model):
    prog = gs.Program(model, "reducemean")
    (job,) = [j for j in prog.jobs if j.role == "CORE"]
    return {gs._reg(w): gs._val(w) for w in job.records if w[0] == gs.A1}


def test_reducemean_has_three_core_forms():
    small, large = _model("reducemean_64_pm1"), _model("reducemean_576_pm1")
    pad = set(wr.REDUCE_PAD_GROUP)
    direct = _core(wr.reducemean_template(100, small=small))
    assert wr.REG_REDUCE_SMALL in direct and not pad & set(direct)
    assert direct[wr.REG_REDUCE_LAST] == 99
    padded = _core(wr.reducemean_template(384, large=large))
    assert pad <= set(padded)
    assert padded[wr.REG_REDUCE_PAD] == 128 and padded[wr.REG_REDUCE_LEN] == 384
    assert padded[wr.EIGHTH_REG] == 1
    whole = _core(wr.reducemean_template(2048, small=small, large=large))
    assert not pad & set(whole) and whole[wr.EIGHTH_REG] == 7
    for reg in wr.REDUCE_FROM_SMALL:
        assert whole[reg] == _core(small)[reg] != _core(large)[reg]


def test_reducemean_output_param_is_required():
    b = BUILDS["reducemean_576_pm1"]
    args = (576, b["scales"], b["zero_points"])
    large = _model("reducemean_576_pm1")
    with pytest.raises(TypeError, match="output_param"):
        wr.emit_reducemean(*args, large=large)
    with pytest.raises(ValueError, match="'early' or 'late'"):
        wr.emit_reducemean(*args, output_param=None, large=large)
    # both orders are programs; only "early" is this native build
    early = wr.emit_reducemean(*args, output_param="early", large=large)
    late = wr.emit_reducemean(*args, output_param="late", large=large)
    assert gs.compare_models(early, large) == []
    assert gs.compare_models(late, large) != []
    roles = [[j.role for j in gs.Program(m, "rm").jobs] for m in (early, late)]
    assert roles[0] == ["PARAM", "QUANT", "PARAM", "CORE", "DEQUANT"]
    assert roles[1] == ["PARAM", "QUANT", "CORE", "PARAM", "DEQUANT"]


def test_reducemean_templates_and_widths_are_checked():
    small, large = _model("reducemean_64_pm1"), _model("reducemean_576_pm1")
    with pytest.raises(ValueError, match="needs the large template"):
        wr.reducemean_template(384, small=small)
    with pytest.raises(ValueError, match="needs the small template"):
        wr.reducemean_template(100, large=large)
    with pytest.raises(ValueError, match="needs the small and the large template"):
        wr.reducemean_template(512)
    with pytest.raises(ValueError, match="small ReduceMean template"):
        wr.reducemean_template(128, small=large)
    with pytest.raises(ValueError, match="large ReduceMean template"):
        wr.reducemean_template(384, large=_model("reducemean_512_pm1"))
    with pytest.raises(ValueError, match="large ReduceMean template"):
        wr.reducemean_template(384, large=small)
    for n, why in ((200, "not a multiple of 32"), (4096, "only 64..2048")):
        with pytest.raises(ValueError, match=why):
            wr.reducemean_template(n, small=small, large=large)
