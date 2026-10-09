"""No-device checks for ``scripts/axera/graph_stitch.py`` on committed
compiler-built fixtures: a model stitched from standalone per-op programs must
equal the native fused build outside segment 0's slot table."""

import copy
import gzip
import json
import os
import sys

import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import graph_stitch as gs  # noqa: E402

INDEX = gs.load_index()
CASES = [(g, c) for g in sorted(INDEX["graphs"]) for c in ("pm1", "pm4")]
# decompressed records per segment (matrix 0, matrix 1, main, copy 3, copy 4)
RECORDS = {
    "silu": [8, 4, 452, 4, 4],
    "sqrt_mul": [8, 4, 292, 4, 4],
    "sig_add": [8, 4, 492, 40, 44],
    "mul_sig": [8, 4, 492, 4, 4],
    "rmsnorm": [8, 4, 524, 52, 56],
    "attention": [100, 100, 608, 68, 44],
}
TWO_OP = ("silu", "sqrt_mul", "sig_add", "mul_sig")
# standalone sources built at another calibration than the fused graph's
CROSS = [
    (g, src, cal)
    for g in TWO_OP
    for cal in ("pm1", "pm4")
    for src in ("pm1", "pm4", "asym")
    if src != cal
] + [
    (g, src, cal)
    for g in ("rmsnorm", "attention")
    for src, cal in (("pm4", "pm1"), ("pm1", "pm4"))
]


def _mcode(model: onnx.ModelProto) -> bytes:
    return bytes(gs.mre.mcode_initializer(model).raw_data)


def _params(model: onnx.ModelProto) -> bytes:
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name == "npu_params"
    )


def _case(graph, cal="pm1", sources=None):
    """A committed case with a wiring the test may edit."""
    wiring, scales, zps, signed, oracle = gs.fixture_case(graph, cal, sources)
    wiring = dict(wiring, ops=[copy.copy(o) for o in wiring["ops"]])
    return wiring, dict(scales), dict(zps), signed, oracle


@pytest.mark.parametrize("graph,cal", CASES)
def test_stitched_model_equals_native_fused_build(graph, cal):
    wiring, scales, zps, signed, oracle = gs.fixture_case(graph, cal)
    st = gs.stitch(wiring, scales, zps, signed)
    got = [gs.records(s) for s in st.segments]
    want = [gs.records(s) for s in gs.suc.decode_segments(_mcode(oracle))]
    assert [len(s) for s in got] == RECORDS[graph]
    assert got[1:] == want[1:]
    # segment 0's slot table (its waits for the other engines' final signals)
    # is a per-build permutation: the same records in any order
    lo, hi = gs.slot_table(want[0])
    assert hi - lo == 4
    assert got[0][:lo] == want[0][:lo] and got[0][hi:] == want[0][hi:]
    assert sorted(got[0][lo:hi]) == sorted(want[0][lo:hi])
    assert st.params == _params(oracle)
    # the blob's FlatBuffer fields, header and tail, and the model proto
    # outside the MCode bytes
    assert gs.compare_models(gs.stitched_model(st), oracle) == []


@pytest.mark.parametrize("graph,sources,cal", CROSS)
def test_sources_of_another_calibration_give_the_same_model(graph, sources, cal):
    # the standalone programs contribute structure only: every calibration
    # value comes from the fused graph's scales and zero points
    wiring, scales, zps, signed, oracle = gs.fixture_case(graph, cal, sources)
    model = gs.stitch_model(wiring, scales, zps, signed)
    assert gs.compare_models(model, oracle) == []


def test_compare_models_reports_a_changed_record():
    wiring, scales, zps, signed, oracle = gs.fixture_case("silu", "pm1")
    scales = dict(scales, y=scales["y"] * 2)
    diffs = gs.compare_models(gs.stitch_model(wiring, scales, zps, signed), oracle)
    assert diffs and all(d.startswith("segment 2 record") for d in diffs)


def test_gate_corpus_of_every_component_gives_the_same_models():
    # more standalone programs can only add jobs to infer order and gates from
    corpus = [gs.fixture_program(c, "pm1") for c in sorted(INDEX["components"])]
    for graph, cal in CASES:
        wiring, scales, zps, signed, oracle = gs.fixture_case(graph, cal)
        model = gs.stitch_model(wiring, scales, zps, signed, gate_corpus=corpus)
        assert gs.compare_models(model, oracle) == []


def test_attention_transpose_job_is_thirty_records():
    # the one job no standalone program contains (MAIN_TRANSPOSE_JOB)
    wiring, scales, zps, signed, _ = gs.fixture_case("attention", "pm1")
    st = gs.stitch(wiring, scales, zps, signed)
    jobs = [e for e in st.log if e["label"].endswith(":transpose")]
    assert [e["n"] for e in jobs] == [30]
    recs = gs.records(st.segments[gs.MAIN])[jobs[0]["start"] :][:30]
    assert recs[-1][0] == gs.A9 and not any(w[0] == gs.A2 for w in recs)
    written = {gs._reg(w): gs._val(w) for w in recs if w[0] == gs.A1}
    for reg, value in gs.MAIN_TRANSPOSE_JOB["constants"].items():
        assert written[reg] == value


# ---- rules derived from the standalone programs --------------------------------
SELF = {  # what a one-op wiring needs beyond the program's own tensor names
    "reducemean": dict(attrs={"count": 64}),
    "c_mulxx": dict(onnx_inputs=["x", "x"]),
    "c_addc": dict(onnx_inputs=["x", "eps"], consts=[{"name": "eps", "size": 1}]),
    "c_mulc": dict(onnx_inputs=["x", "gain"], consts=[{"name": "gain", "size": 64}]),
}
COMPONENTS = [
    (name, cal)
    for name in sorted(INDEX["components"])
    for cal in sorted(INDEX["components"][name]["files"])
]


def _self_stitch(name, cal, zero_points=None):
    """Stitch a one-op graph from a standalone program, at the program's own
    calibration and on its own engines."""
    meta = INDEX["components"][name]
    prog = gs.fixture_program(name, cal)
    place = {}
    for t in prog.etasks:
        if t.kind == "raw":
            place["matrix" if t.seg in gs.MATRIX_ENGINES else "transpose"] = t.seg
    if place:
        place["loader"] = prog.loaders[0].seg
    op = dict(
        SELF.get(name, {}),
        op=meta["op"],
        program=prog,
        out=(prog.out_names[0], prog.out_names[0]),
        place=place,
    )
    op["in"] = {n: n for n in prog.in_names}
    wiring = {"inputs": prog.in_names, "output": prog.out_names[0], "ops": [op]}
    q = meta["calibrations"][cal]
    return prog, gs.stitch(
        wiring, q["scales"], zero_points or q["zero_points"], q["signed"]
    )


@pytest.mark.parametrize("name,cal", COMPONENTS)
def test_one_op_graph_reproduces_its_standalone_program(name, cal):
    # job roles, slot numbering, register order, calibration formulas, loader
    # and matrix-engine emission, the sync model and the blob layout all hold
    # on the standalone program itself
    prog, st = _self_stitch(name, cal)
    assert st.segments == [b"".join(s) for s in prog.segs]
    if (name, cal) == ("add", "asym"):
        # this build holds Add's two scale words in the other order (Pulsar2's
        # Add operand order varies per build); the fused Add graphs and the
        # other four Add builds hold them in ONNX input order
        assert st.params[:2] == prog.params[2:4] and st.params[2:4] == prog.params[:2]
        assert st.params[4:] == prog.params[4:]
        return
    assert st.params == prog.params
    assert gs.compare_models(gs.stitched_model(st), prog.model) == []


@pytest.mark.parametrize(
    "name", sorted(f for f in os.listdir(gs.FIXTURES) if f.endswith(".axmodel.gz"))
)
def test_blob_rebuild_is_byte_exact(name):
    with gzip.open(os.path.join(gs.FIXTURES, name), "rb") as f:
        mc = _mcode(onnx.load_model_from_string(f.read()))
    fields = gs.parse_blob(mc)
    assert gs.build_blob(fields) == mc
    # segments 0 and 2 are compressed, 4 never, 1 and 3 unless they are stubs
    segs = [gs.records(s) for s in gs.suc.decode_segments(mc)]
    assert fields["comp"] == gs.compress_shape(segs)


def test_constant_values_quantize_to_the_standalone_bytes():
    rms = INDEX["graphs"]["rmsnorm"]
    for cal in ("pm1", "pm4"):
        q = rms["calibrations"][cal]
        gain = gs.quantize_constant(
            rms["constants"]["gain"], q["scales"]["gain"], q["zero_points"]["gain"]
        )
        assert gain == gs.fixture_program("c_mulc", cal).params[:64]
        eps = gs.quantize_constant(
            rms["constants"]["eps"], q["scales"]["eps"], q["zero_points"]["eps"]
        )
        assert eps == gs.fixture_program("c_addc", cal).params[:1] == b"\xff"


# ---- refused cases -------------------------------------------------------------
def test_unknown_op_is_refused():
    wiring, scales, zps, signed, _ = _case("silu")
    wiring["ops"][0]["op"] = "Tanh"
    with pytest.raises(NotImplementedError, match="not modelled"):
        gs.stitch(wiring, scales, zps, signed)


@pytest.mark.parametrize("tensor", ["x", "r"])
def test_sqrt_nonzero_zero_point_is_refused(tensor):
    # no standalone Sqrt has one, so its zero-point registers are unknown
    wiring, scales, zps, signed, _ = _case("sqrt_mul")
    zps[tensor] = 3
    with pytest.raises(NotImplementedError, match="Sqrt"):
        gs.stitch(wiring, scales, zps, signed)


def test_div_nonzero_divisor_zero_point_is_refused():
    zps = dict(INDEX["components"]["c_divb"]["calibrations"]["pm1"]["zero_points"])
    assert zps["b"] == 0
    with pytest.raises(NotImplementedError, match="Div"):
        _self_stitch("c_divb", "pm1", dict(zps, b=5))


def _attention(place_qk, place_pv):
    wiring, scales, zps, signed, _ = _case("attention")
    for op, place in ((wiring["ops"][0], place_qk), (wiring["ops"][2], place_pv)):
        op["place"] = place
    return lambda: gs.stitch(wiring, scales, zps, signed)


QK = {"matrix": 0, "loader": 3, "transpose": 3}
PV = {"matrix": 1, "loader": 4, "transpose": 2}


@pytest.mark.parametrize(
    "qk,pv,why",
    [
        # the placement is learned from the native build, not derived
        (None, None, "explicit place"),
        # the second transpose on its standalone copy engine needs restores
        (QK, dict(PV, transpose=3), "needs a restore"),
        (QK, dict(PV, matrix=0), "two ops on one matrix engine"),
        (dict(QK, transpose=4), PV, "register layouts differ"),
        (dict(QK, matrix=2), PV, "only a copy-engine transpose"),
    ],
)
def test_unsupported_engine_placements_are_refused(qk, pv, why):
    with pytest.raises(NotImplementedError, match=why):
        _attention(qk, pv)()
    assert _attention(QK, PV)().segments  # the native build's placement


def test_missing_calibration_is_refused():
    wiring, scales, zps, signed, _ = _case("silu")
    del scales["s"]
    with pytest.raises(ValueError, match="no scale or zero point"):
        gs.stitch(wiring, scales, zps, signed)


def test_unwired_tensor_is_refused():
    wiring, scales, zps, signed, _ = _case("silu")
    wiring["ops"][1]["in"] = {"a": "x"}
    with pytest.raises(ValueError, match="no wiring"):
        gs.stitch(wiring, scales, zps, signed)


def test_constants_must_account_for_the_program():
    wiring, scales, zps, signed, _ = _case("rmsnorm")
    wiring["ops"][5]["consts"] = []
    with pytest.raises(ValueError, match="constant bytes"):
        gs.stitch(wiring, scales, zps, signed)


def test_constant_of_another_value_is_refused():
    # initializer bytes are copied from the standalone program, so a program
    # built with another constant is the wrong source
    wiring, scales, zps, signed, _ = _case("rmsnorm")
    wiring["ops"][5]["consts"] = [{"name": "gain", "size": 64, "values": [1.0] * 64}]
    with pytest.raises(ValueError, match="does not quantize"):
        gs.stitch(wiring, scales, zps, signed)


def test_reducemean_needs_its_element_count():
    wiring, scales, zps, signed, _ = _case("rmsnorm")
    wiring["ops"][1]["attrs"] = {}
    with pytest.raises(ValueError, match="count"):
        gs.stitch(wiring, scales, zps, signed)


def test_graph_inputs_must_name_the_graph_inputs():
    wiring, scales, zps, signed, _ = _case("silu")
    wiring["graph_inputs"] = ["q"]
    with pytest.raises(ValueError, match="graph_inputs"):
        gs.stitch_model(wiring, scales, zps, signed)


# ---- width 576: the output PARAM placement and the wide ReduceMean ---------------
WIDTH_FIXTURES = os.path.join(AXERA, "fixtures", "width_retarget")
with open(os.path.join(WIDTH_FIXTURES, "index.json")) as _f:
    WIDTH_INDEX = json.load(_f)


def _width_model(file):
    return gs.load_model(os.path.join(WIDTH_FIXTURES, file))


def _width_case(graph, cal):
    """``(wiring, scales, zero_points, native fused model)`` of a ``[1,576]``
    graph: native standalone components (all built at ``pm1``) and the fused
    graph's calibration."""
    g = WIDTH_INDEX["graphs"][graph]
    ops = [
        dict(o, program=_width_model(WIDTH_INDEX["builds"][o["component"]]["file"]))
        for o in g["ops"]
    ]
    wiring = {k: g[k] for k in ("inputs", "graph_inputs", "output")}
    wiring["ops"] = ops
    if "output_param" in g:
        wiring["output_param"] = g["output_param"]
    q = g["calibrations"][cal]
    return wiring, q["scales"], q["zero_points"], _width_model(q["oracle"])


@pytest.mark.parametrize("cal", ["pm1", "pm4"])
def test_rmsnorm_576_stitches_from_native_components(cal):
    wiring, scales, zps, oracle = _width_case("rmsnorm576", cal)
    st = gs.stitch(wiring, scales, zps)
    assert [len(s) // gs.REC for s in st.segments] == [8, 4, 532, 52, 56]
    assert gs.compare_models(gs.stitched_model(st), oracle) == []
    # the output's PARAM job runs before ReduceMean's core (op 1), unlike the
    # [1,64] graph where it runs before the DEQUANT job
    roles = [e["label"].split(":", 1)[1] for e in st.log]
    assert roles.index("Mul:job3:PARAM") + 1 == roles.index("ReduceMean:job3:CORE")
    assert roles[-1] == "Mul:job4:DEQUANT"


def test_rmsnorm_576_needs_the_output_param_placement():
    # with the [1,64] graph's order the program is 12 records longer
    wiring, scales, zps, oracle = _width_case("rmsnorm576", "pm1")
    del wiring["output_param"]
    st = gs.stitch(wiring, scales, zps)
    assert [len(s) // gs.REC for s in st.segments] == [8, 4, 544, 52, 56]
    assert gs.compare_models(gs.stitched_model(st), oracle) != []


@pytest.mark.parametrize("cal", ["pm1", "pm4"])
def test_silu_576_stitches_from_native_components(cal):
    wiring, scales, zps, oracle = _width_case("silu576", cal)
    assert gs.compare_models(gs.stitch_model(wiring, scales, zps), oracle) == []


def test_output_param_moves_only_the_job_order():
    wiring, scales, zps, signed, oracle = _case("rmsnorm")
    late = gs.stitch(dict(wiring, output_param="late"), scales, zps, signed)
    # "late" is where the [1,64] graph already has it
    assert gs.compare_models(gs.stitched_model(late), oracle) == []
    early = gs.stitch(dict(wiring, output_param=1), scales, zps, signed)
    labels = [e["label"] for e in early.log]
    assert labels.index("op5:Mul:job3:PARAM") + 1 == labels.index(
        "op1:ReduceMean:job2:CORE"
    )
    assert early.params == late.params and early.segments[:2] == late.segments[:2]
    assert early.segments[gs.MAIN] != late.segments[gs.MAIN]


@pytest.mark.parametrize(
    "where,why", [(7, "no CORE job of op 7"), ("soon", "expected")]
)
def test_bad_output_param_is_refused(where, why):
    wiring, scales, zps, signed, _ = _case("rmsnorm")
    with pytest.raises(ValueError, match=why):
        gs.stitch(dict(wiring, output_param=where), scales, zps, signed)


def test_reducemean_lane_is_single_precision():
    # float32(float32(s_x / s_y) / n); the float64 form s_x / (s_y * n) is one
    # ulp above it on the native [1,384] build
    b = WIDTH_INDEX["builds"]["reducemean_384_pm1"]
    sx, sy = b["scales"]["x"], b["scales"]["y"]
    lane = gs.reducemean_lane(sx, sy, 384)
    assert lane == 0x3D38ECB5 and gs._f32bits(sx / (sy * 384)) == lane + 1
    prog = gs.Program(_width_model(b["file"]), "reducemean_384")
    (core,) = [j for j in prog.jobs if j.role == "CORE"]
    written = {gs._reg(w): gs._val(w) for w in core.records if w[0] == gs.A1}
    assert {written[r] for r in gs.LANES_LO} == {lane}
    # one 256-wide pooling window of zero points, and the pad bytes
    assert written[gs.REG_ZP_A] == b["zero_points"]["x"] * 256
    assert {written[r] for r in gs.LANES_PACKED} == {b["zero_points"]["x"] * 0x01010101}
    # every committed [1,64] build has equal float32 and float64 forms
    for cal, q in INDEX["components"]["reducemean"]["calibrations"].items():
        s = q["scales"]
        assert gs.reducemean_lane(s["x"], s["y"], 64) == gs._f32bits(
            s["x"] / (s["y"] * 64)
        ), cal
